"""Read and write ImageD11 formats: parameters, TensorMaps, sparse pixel files and DataSet files.

A simulated scanning-3DXRD dataset is a sparse pixels file, a DataSet file and a peaks table. The sparse file has one
group per dty row ("1.1", "2.1", ...), laid out like ImageD11's own segmented files, so ImageD11's
S3DXRD pipeline can read it directly.

ImageD11 itself is only imported by the functions that need it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

import h5py
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

import anri.geom
from anri.fwd.render import make_row, render_phases

if TYPE_CHECKING:
    from ImageD11.sinograms.tensor_map import TensorMap

_GEOMETRY_KEYS = ("y_center", "y_size", "tilt_y", "z_center", "z_size", "tilt_z", "tilt_x", "distance")
_ORIENTATION_KEYS = ("o11", "o12", "o21", "o22")


def detector_from_pars(pars: dict) -> dict:
    """Detector geometry from ImageD11 parameters, for mapping between detector pixels and the lab frame.

    Parameters
    ----------
    pars
        ImageD11 parameters: ``y_center``, ``y_size``, ``tilt_y``, ``z_center``, ``z_size``, ``tilt_z``, ``tilt_x``,
        ``distance`` and ``o11`` ... ``o22``. Other keys are ignored.

    Returns
    -------
    detector: dict
        "det_trans", "beam_cen_shift", "x_distance_shift" from :func:`anri.geom.detector.detector_transforms` (pixels to lab,
        see :func:`anri.geom.detector.det_to_lab`), and "s_step_lab", "f_step_lab", "det_origin_lab" from
        :func:`anri.geom.detector.detector_basis_vectors_lab` (lab to pixels, see :func:`anri.geom.detector.raytrace_to_det`)
    """
    det_trans, beam_cen_shift, x_distance_shift = anri.geom.detector.detector_transforms(
        *(float(pars[k]) for k in _GEOMETRY_KEYS), *(float(pars[k]) for k in _ORIENTATION_KEYS)
    )
    s_step_lab, f_step_lab, det_origin_lab = anri.geom.detector.detector_basis_vectors_lab(
        det_trans, beam_cen_shift, x_distance_shift
    )
    return {
        "det_trans": det_trans,
        "beam_cen_shift": beam_cen_shift,
        "x_distance_shift": x_distance_shift,
        "s_step_lab": s_step_lab,
        "f_step_lab": f_step_lab,
        "det_origin_lab": det_origin_lab,
    }


def gonio_from_pars(pars: dict, y0: float) -> dict:
    """Goniometer geometry from ImageD11 parameters.

    Parameters
    ----------
    pars
        ImageD11 parameters: ``wedge`` and ``chi`` (degrees, default 0) and ``omegasign`` (default 1).
        ImageD11's wedge has the opposite sign to anri's, so the returned ``wedge`` is ``-pars["wedge"]``.
    y0
        dty at which the rotation axis is in the beam

    Returns
    -------
    gonio: dict
        "wedge", "chi" (degrees, anri convention) and "y0"
    """
    if float(pars.get("omegasign", 1.0)) != 1.0:
        msg = "omegasign != 1 is not supported"
        raise ValueError(msg)
    return {
        "wedge": -float(pars.get("wedge", 0.0)),  # ImageD11's wedge is a left-handed rotation; anri's is right-handed
        "chi": float(pars.get("chi", 0.0)),
        "y0": y0,
    }


def beam_from_pars(pars: dict, k_in_lab: ArrayLike | None = None) -> dict:
    """Beam from ImageD11 parameters: ``wavelength`` (angstrom), and a direction.

    Parameters
    ----------
    pars
        ImageD11 parameters with ``wavelength``. ImageD11 has no beam direction: its beam is along lab x.
    k_in_lab
        [3] Direction of the incoming beam in the lab frame (default lab x), normalised here. It must not be
        vertical: beam divergence and polarisation are defined across it, see :func:`anri.geom.beam.beam_basis`.

    Returns
    -------
    beam: dict
        "wavelength" and "k_in_lab" [3], the unit vector of the incoming beam
    """
    k = jnp.array([1.0, 0.0, 0.0]) if k_in_lab is None else jnp.asarray(k_in_lab, dtype=float)
    return {"wavelength": float(pars["wavelength"]), "k_in_lab": k / jnp.linalg.norm(k)}


def geom_from_pars(
    pars: dict,
    y0: float,
    sig_wavelength: float,
    sig_ky: float,
    sig_kz: float,
    sig_beam: float,
    voxel_size: float,
    pol_factor: float = 1.0,
    sig_psf: float = 0.0,
    k_in_lab: ArrayLike | None = None,
    width_beam: float = 0.0,
    sig_beam_v: float = 0.0,
    width_beam_v: float = 0.0,
    voxel_3d: bool = False,
    sig_omega: float = 0.0,
) -> dict:
    """Build the geometry dict for :func:`anri.fwd.render.render_row` from ImageD11 parameters.

    Combines :func:`beam_from_pars`, :func:`gonio_from_pars` and :func:`detector_from_pars` with the spreads
    the renderer needs.

    Parameters
    ----------
    pars
        ImageD11 parameters: detector geometry (``y_center`` ... ``o22``), ``wavelength``, ``wedge``, ``chi``.
        ImageD11's wedge has the opposite sign to anri's, so the returned geometry has ``-wedge``.
    y0
        dty at which the rotation axis is in the beam
    sig_wavelength, sig_ky, sig_kz
        Standard deviations of the wavelength (angstrom) and of the beam divergence (radians)
    sig_beam, width_beam
        The beam's profile across it, horizontally: a flat top of ``width_beam`` (default 0: a Gaussian) blurred by a
        Gaussian of standard deviation ``sig_beam`` (same units as dty; must be > 0), see
        :func:`anri.geom.beam.beam_weight`. A pencil beam is narrow, a box or horizontal line beam wider than the sample.
    voxel_size
        Side length of the voxels (same units as dty)
    pol_factor
        Degree of horizontal polarisation, see :func:`anri.diffract.factors.polarisation`
    sig_psf
        Standard deviation of the detector point spread, in pixels. Spots much narrower than a pixel have
        intensity-weighted centroids snapped towards pixel centres (by up to ~0.3 px at 0.1 px wide); a real
        detector's point spread prevents that.
    k_in_lab
        [3] Direction of the incoming beam (default lab x), see :func:`beam_from_pars`
    sig_beam_v, width_beam_v
        The beam's profile across it, vertically, as for ``sig_beam`` and ``width_beam``. Only used for cubes: for
        columns (2D maps) the whole vertical profile crosses the voxel.
    voxel_3d
        Whether the voxels are cubes (a 3D map) rather than columns (a 2D map, e.g. a TensorMap layer)
    sig_omega
        Extra standard deviation of every peak in omega, in degrees (default 0): a simple stand-in for mosaicity,
        and a way to smooth the loss at the start of a refinement.
    """
    if sig_beam <= 0 or (voxel_3d and sig_beam_v <= 0):
        msg = "the beam's Gaussian widths must be > 0 (a small one gives a sharp-edged flat top)"
        raise ValueError(msg)
    return {
        **beam_from_pars(pars, k_in_lab),
        **gonio_from_pars(pars, y0),
        **detector_from_pars(pars),
        "sig_wavelength": sig_wavelength,
        "sig_ky": sig_ky,
        "sig_kz": sig_kz,
        "sig_beam": sig_beam,
        "width_beam": width_beam,
        "sig_beam_v": sig_beam_v,
        "width_beam_v": width_beam_v,
        "voxel_3d": voxel_3d,
        "voxel_size": voxel_size,
        "pol_factor": pol_factor,
        "sig_psf": sig_psf,
        "sig_omega": sig_omega,
    }


def write_par(path: str, pars: dict) -> None:
    """Write ImageD11 parameters as a ``.par`` file of ``key value`` lines."""
    with open(path, "w") as f:
        f.writelines(f"{key} {value}\n" for key, value in pars.items())


def read_par(path: str) -> dict:
    """Read an ImageD11 ``.par`` file of ``key value`` lines (numbers as floats, other values as strings).

    Parameters
    ----------
    path
        The file

    Returns
    -------
    dict
        Parameters
    """
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    out[parts[0]] = parts[1]
    return out


def read_dataset(dsfile: str, remap: dict | None = None) -> dict:
    """Read the scan layout of an ImageD11 DataSet file, without ImageD11.

    Parameters
    ----------
    dsfile
        ``<sample>_<dset>_dataset.h5``
    remap
        ``{old prefix: new prefix}`` for the paths the DataSet records, e.g. when the data were moved or are mounted
        elsewhere on this machine

    Returns
    -------
    dict
        "y0" (None if absent), "ybincens", "ybinedges", "obinedges", "dtymotor", "omegamotor", "parfile",
        "sparsefile", "pksfile" (the peaks table, "" if absent), "masterfile", the detector distortion files "e2dxfile", "e2dyfile", "detectorh5" and
        "splinefile", and the segmenter's detector mask "maskfile" (made absolute: a relative path is relative to the DataSet's folder; "" if absent), "detector"; "dty" and "omega" [scans, frames] (None if absent): each frame's motor positions as ImageD11 has binned
        them, the sinogram's rows; and "scans" (None if absent): each row's sparse group, or slice of one
        ("1.1::[0:1440]")
    """
    with h5py.File(dsfile, "r") as h:
        attrs = dict(h.attrs)
        out = {k: h[k][()] for k in ("ybincens", "ybinedges", "obinedges")}
        out["dty"] = h["dty"][()] if "dty" in h else None
        out["omega"] = h["omega"][()] if "omega" in h else None
        out["scans"] = (
            [x.decode() if isinstance(x, bytes) else str(x) for x in h["scans"][()]] if "scans" in h else None
        )
        seg = dict(h["lima_segmenter"].attrs) if "lima_segmenter" in h else {}
    if "maskfile" not in attrs and "maskfile" in seg:  # the segmenter's detector mask
        attrs["maskfile"] = seg["maskfile"]

    def absolute(key: str) -> str:
        path = str(attrs.get(key, ""))
        if path == "None":  # ImageD11 may save an unset path as "None"
            return ""
        if path and not os.path.isabs(path):
            path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(dsfile)), path))
        for old, new in (remap or {}).items():
            if path.startswith(old):
                path = os.path.expanduser(new) + path[len(old) :]
        return path

    out.update(
        parfile=absolute("parfile"),
        sparsefile=absolute("sparsefile"),
        pksfile=absolute("pksfile"),
        masterfile=absolute("masterfile"),
        e2dxfile=absolute("e2dxfile"),
        e2dyfile=absolute("e2dyfile"),
        detectorh5=absolute("detectorh5"),
        splinefile=absolute("splinefile"),
        maskfile=absolute("maskfile"),
        detector=str(attrs.get("detector", "")),
        y0=float(attrs["y0"]) if "y0" in attrs else None,
        dtymotor=str(attrs["dtymotor"]),
        omegamotor=str(attrs["omegamotor"]),
    )
    return out


def read_mask(path: str) -> np.ndarray:
    """Read a detector mask as ImageD11's segmenter uses it: any image fabio reads, non-zero for a masked pixel.

    Parameters
    ----------
    path
        The mask file (e.g. a DataSet's "maskfile", :func:`read_dataset`)

    Returns
    -------
    np.ndarray
        [n_slow, n_fast] bool, True where the pixel is measured
    """
    import fabio

    return np.asarray(fabio.open(path).data) == 0


def read_pars_json(parfile: str, phase: str | None = None) -> tuple[dict, str, dict]:
    """Read the geometry and one phase from an ImageD11 ``pars.json``.

    A file it names is looked up relative to ``pars.json``, or, if missing there (e.g. an absolute path from another
    machine), by its name beside ``pars.json``.

    Parameters
    ----------
    parfile
        ``pars.json``
    phase
        Phase name; may be omitted if there is only one

    Returns
    -------
    geometry: dict
        Detector and beam parameters
    phase: str
        The phase's name
    cell: dict
        The phase's parameters (``cell__a`` ... ``cell_lattice_[P,A,B,C,I,F,R]``)
    """
    import json

    with open(parfile) as f:
        pj = json.load(f)
    pdir = os.path.dirname(os.path.abspath(parfile))

    def path(name: str) -> str:
        p = os.path.join(pdir, name)
        return p if os.path.exists(p) else os.path.join(pdir, os.path.basename(name))

    phases = pj["phases"]
    if phase is None:
        if len(phases) != 1:
            msg = f"several phases in {parfile}: {list(phases)}; choose one"
            raise ValueError(msg)
        phase = next(iter(phases))
    return read_par(path(pj["geometry"]["file"])), phase, read_par(path(phases[phase]["file"]))


def read_geometry(parfile: str) -> dict:
    """Read the detector and beam geometry from an ImageD11 ``.par`` file or ``pars.json`` (any number of phases)."""
    if not parfile.endswith(".json"):
        return read_par(parfile)
    import json

    with open(parfile) as f:
        name = json.load(f)["geometry"]["file"]
    pdir = os.path.dirname(os.path.abspath(parfile))
    path = os.path.join(pdir, name)
    return read_par(path if os.path.exists(path) else os.path.join(pdir, os.path.basename(name)))


def read_monitor(sparsefile: str, groups: list, name: str, masterfile: str | None = None) -> dict:
    """Read a counter (e.g. "fpico6") for each scan group, frame by frame, as ImageD11's ``DataSet.get_monitor``.

    It is read from the sparse file's ``<group>/measurement``, or, if absent there, from the raw master file's
    scan of the same name: a sparse group holds the frames of that scan, in order. Values are cut to the group's
    number of frames.

    Parameters
    ----------
    sparsefile
        ImageD11 sparse pixels file
    groups
        Scan groups, e.g. "1.1"
    name
        Counter name
    masterfile
        The raw data's master file (``read_dataset(...)["masterfile"]``), for counters the sparse file lacks

    Returns
    -------
    dict
        ``{group: [n_frames] array}``
    """
    out = {}
    with h5py.File(sparsefile, "r") as h:
        n_frames = {g: len(h[f"{g}/nnz"]) for g in groups}
        missing = [g for g in groups if name not in h[f"{g}/measurement"]]
        out.update({g: np.asarray(h[f"{g}/measurement/{name}"][()], float) for g in groups if g not in missing})
    if missing:
        if not masterfile or not os.path.exists(masterfile):
            msg = (
                f"{name} is not in {sparsefile} for scans {missing[:3]}, and no master file was found ({masterfile!r})"
            )
            raise KeyError(msg)
        with h5py.File(masterfile, "r") as h:
            for g in missing:
                out[g] = np.asarray(h[f"{g}/measurement/{name}"][()], float)
    for g in groups:
        if len(out[g]) < n_frames[g]:
            msg = f"{name} has {len(out[g])} values for scan {g}, which has {n_frames[g]} frames"
            raise ValueError(msg)
        out[g] = out[g][: n_frames[g]]
    return out


# the detectors ImageD11 knows in a pyFAI detector file, by the start of the DataSet's "detector"
_PYFAI_DETECTORS = {"pilatus": "Pilatus_CdTe_2M", "p3": "Pilatus_CdTe_2M", "eiger": "Eiger2_CdTe_4M"}


def read_spatial(ds: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a DataSet's detector distortion maps, as ImageD11 corrects its peaks with them.

    ImageD11's order: a pyFAI detector file ("detectorh5": dx, dy from the mean of each pixel's corners, as
    ``ImageD11.blobcorrector.get_e2dx_from_h5``), else EDF maps ("e2dxfile", "e2dyfile", read with fabio). A pixel at
    (row, col) is at (row + dy[row, col], col + dx[row, col]), in pixels: dx along the fast axis, dy along the slow.
    Spline files are not supported.

    Parameters
    ----------
    ds
        From :func:`read_dataset`

    Returns
    -------
    tuple or None
        (dx, dy) [n_slow, n_fast] float32, or None if the DataSet names no correction
    """
    if ds.get("detectorh5"):
        det = str(ds.get("detector", "")).lower()
        key = next((v for k, v in _PYFAI_DETECTORS.items() if k in det), None)
        if key is None:
            msg = f"{ds['detectorh5']}: unknown detector {ds.get('detector')!r} (known: {list(_PYFAI_DETECTORS)})"
            raise ValueError(msg)
        with h5py.File(ds["detectorh5"], "r") as h:
            ps = h[f"entry_0000/pyFAI/{key}/pixel_size"][()]
            d = h[f"entry_0000/pyFAI/{key}/pixel_corners"][()]  # [n_slow, n_fast, 4 corners, (z, y, x)]
        s, f = np.mgrid[0 : d.shape[0], 0 : d.shape[1]]
        dy = d[:, :, :, 1].mean(2) / np.mean(ps) - s
        dx = d[:, :, :, 2].mean(2) / np.mean(ps) - f
        return dx.astype(np.float32), dy.astype(np.float32)
    if ds.get("e2dxfile") and ds.get("e2dyfile"):
        import fabio

        return (
            np.asarray(fabio.open(ds["e2dxfile"]).data, np.float32),
            np.asarray(fabio.open(ds["e2dyfile"]).data, np.float32),
        )
    if ds.get("splinefile"):
        msg = f"{ds['splinefile']}: spline distortion files are not supported; make e2dx/e2dy maps from it"
        raise ValueError(msg)
    return None


def read_peaks(pks_file: str, omega: np.ndarray, dty: np.ndarray, spatial: tuple | None = None) -> dict:
    """Read ImageD11's 2D peaks and merge them into one peak per 4D label and dty row.

    ImageD11's ``properties`` step labels the sparse pixels of each frame into 2D peaks, stores their moments
    (pixels, intensity, intensity-weighted row and column, frame), and joins 2D peaks that overlap in neighbouring
    frames and dty rows under one 4D label. Merging a label's 2D peaks within each row averages over the energy band
    (each frame of a rocking peak diffracts a different part of it) and keeps each row's scattering origin, where the
    beam crosses the grain at that dty.

    Parameters
    ----------
    pks_file
        ImageD11 peaks table (``<sample>_<dataset>_peaks_table.h5``, group ``pks2d``)
    omega, dty
        [rows, frames] motor positions of each frame, as the DataSet stores them (:func:`anri.io.imaged11.read_dataset`)
    spatial
        (dx, dy) distortion maps from :func:`anri.io.imaged11.read_spatial`, or None. Each 2D peak's centroid is
        moved by the distortion of its nearest pixel before merging, as ImageD11 corrects peaks.

    Returns
    -------
    dict
        Per merged peak: "sc", "fc" (corrected slow and fast centroids, pixels), "omega", "dty" (intensity-weighted
        over the merged 2D peaks), "row" (dty row), "intensity", "npix" (pixels) and "n2d" (2D peaks merged)
    """
    with h5py.File(pks_file, "r") as h:
        glabel = h["pks2d/glabel"][()]
        s1, sI, srI, scI, frm = h["pks2d/pk_props"][()]
    sI = sI.astype(float)
    s, f = srI / sI, scI / sI
    if spatial is not None:
        dx, dy = spatial
        i = np.clip(np.round(s).astype(int), 0, dx.shape[0] - 1)
        j = np.clip(np.round(f).astype(int), 0, dx.shape[1] - 1)
        s, f = s + dy[i, j], f + dx[i, j]
    row = frm // omega.shape[1]
    key, inv = np.unique(glabel.astype(np.int64) * omega.shape[0] + row, return_inverse=True)
    n = len(key)
    w = np.bincount(inv, sI, n)

    def mean(v: np.ndarray) -> np.ndarray:
        return np.bincount(inv, sI * v, n) / w

    return {
        "sc": mean(s),
        "fc": mean(f),
        "omega": mean(omega.ravel()[frm]),
        "dty": mean(dty.ravel()[frm]),
        "row": key % omega.shape[0],
        "intensity": w,
        "npix": np.bincount(inv, s1, n),
        "n2d": np.bincount(inv, None, n),
    }


def _scan_slices(scans: list) -> dict:
    """``{group: [(scan index, first frame, last frame + 1 or None), ...]}`` from DataSet scan names.

    A name is a sparse group ("1.1": all its frames) or a slice of one ("1.1::[0:1440]").
    """
    out: dict = {}
    for i, sc in enumerate(scans):
        name, _, rest = str(sc).partition("::[")
        a, b = (int(x) for x in rest.rstrip("]").split(":")) if rest else (0, None)
        out.setdefault(name, []).append((i, a, b))
    return out


def _frame_positions(
    gr: h5py.Group, name: str, motor: str, values: np.ndarray | list | None, slices: dict, path: str
) -> np.ndarray:
    """[n_frames] positions of one motor for sparse group gr: the DataSet's, where it covers the frame, else the group's.

    ``values`` holds the DataSet's positions [scans, frames] (e.g. its ``dty`` or ``omega``) and ``slices`` which of
    its scans hold which frames of which group (:func:`_scan_slices`). Frames of a group the DataSet slices but does not
    cover get NaN.
    """
    n = len(gr["nnz"])
    if name in slices and values is not None:
        out = np.full(n, np.nan)
        for i, a, b in slices[name]:
            b = n if b is None else min(b, n)
            v = np.asarray(values[i], float)[: b - a]
            out[a : a + len(v)] = v
        return out
    if motor in gr["measurement"]:
        return np.broadcast_to(np.asarray(gr[f"measurement/{motor}"][()], float), (n,)).copy()
    msg = f"{path}:{name} has no {motor}, and the DataSet has no positions for it; pass the DataSet's and its scans"
    raise KeyError(msg)


def read_sparse_frames(sparsefile: str, scan: str, frames: range | None = None) -> dict:
    """Read the sparse pixels of frames of one DataSet scan: a sparse group ("1.1") or a slice of one ("1.1::[0:1440]").

    Parameters
    ----------
    sparsefile
        ImageD11 sparse pixels file
    scan
        The scan's name, as the DataSet's ``scans``
    frames
        Frames of the scan (default all), numbered from the scan's first

    Returns
    -------
    dict
        "frame" (within the scan), "row", "col" and "intensity" of every pixel, and "shape" (n_slow, n_fast)
    """
    name, _, rest = scan.partition("::[")
    first = int(rest.rstrip("]").split(":")[0]) if rest else 0
    with h5py.File(sparsefile, "r") as h:
        g = h[name]
        nnz = g["nnz"][()]
        end = len(nnz) if not rest or not rest.rstrip("]").split(":")[1] else int(rest.rstrip("]").split(":")[1])
        frames = range(end - first) if frames is None else frames
        a, b = first + frames.start, min(first + frames.stop, end)
        starts = np.concatenate([[0], np.cumsum(nnz, dtype=np.int64)])
        s0, s1 = int(starts[a]), int(starts[b])
        return {"frame": np.repeat(np.arange(a, b) - first, nnz[a:b]), "row": g["row"][s0:s1], "col": g["col"][s0:s1],
                "intensity": g["intensity"][s0:s1], "shape": (int(g.attrs["shape0"]), int(g.attrs["shape1"]))}  # fmt: skip


def read_raw_frames(masterfile: str, scan: str, frames: range, detector: str = "eiger") -> np.ndarray:
    """Read raw detector frames of one DataSet scan from the experiment's master file, summed.

    Parameters
    ----------
    masterfile
        The raw data's master file (the DataSet's ``masterfile``)
    scan
        The scan's name, as the DataSet's ``scans`` ("1.1" or "1.1::[0:1440]")
    frames
        Frames of the scan, numbered from the scan's first
    detector
        The detector's name in the scan's ``measurement`` group

    Returns
    -------
    np.ndarray
        [n_slow, n_fast] float64, the frames summed; pixels at the integer type's largest value (dead pixels, module
        gaps) count as 0
    """
    name, _, rest = scan.partition("::[")
    first = int(rest.rstrip("]").split(":")[0]) if rest else 0
    with h5py.File(masterfile, "r") as h:
        d = h[f"{name}/measurement/{detector}"]
        x = d[first + frames.start : first + frames.stop]
    if np.issubdtype(x.dtype, np.integer):  # dead pixels and module gaps hold the type's largest value
        x = np.where(x == np.iinfo(x.dtype).max, 0, x)
    return x.astype(np.float64).sum(0)


def read_frame_dty(ds: dict, sparsefile: str | None = None) -> np.ndarray:
    """Each frame's dty: the DataSet's (corrected) dty of its row, plus the motion of dty within the row.

    ImageD11 gives every frame of a row one dty (e.g. the mean over a rotation of a fly or helical scan, where dty
    moves while omega turns). The raw readings in the sparse file move within the row; a straight line fitted to
    them over the row's frames (time) gives that motion, smooth below the encoder's resolution. Whatever the DataSet
    corrects per row (an offset, or the mean) is kept. Where the sparse file has no reading per frame, the motion is 0.

    Parameters
    ----------
    ds
        From :func:`read_dataset`: "dty", "scans", "dtymotor" and "sparsefile"
    sparsefile
        The sparse pixels file (default: the DataSet's)

    Returns
    -------
    np.ndarray
        [scans, frames] dty, as the DataSet's ``dty``
    """
    dty = np.asarray(ds["dty"], float)
    out = dty.copy()
    slices = _scan_slices(ds["scans"])
    with h5py.File(sparsefile or ds["sparsefile"], "r") as h:
        for name, parts in slices.items():
            meas = h[name].get("measurement")
            if meas is None or ds["dtymotor"] not in meas or np.ndim(meas[ds["dtymotor"]]) == 0:
                continue
            raw = np.asarray(meas[ds["dtymotor"]][()], float)
            for i, a, b in parts:
                n = dty.shape[1]
                r = raw[a : (len(raw) if b is None else b)][:n]
                if len(r) < 2:
                    continue
                t = np.arange(len(r), dtype=float)
                slope = np.polyfit(t, r, 1)[0]
                out[i, : len(r)] = dty[i, : len(r)] + slope * (t - t.mean())
    return out


def stream_sparse(
    sparsefile: str,
    ybinedges: ArrayLike,
    omega_motor: str,
    dty_motor: str,
    chunk: int,
    groups: list | None = None,
    gridstep: int = 1,
    dataset_dty: np.ndarray | None = None,
    scans: list | None = None,
    monitor: str | None = None,
    monitor_ref: float | None = None,
    masterfile: str | None = None,
    dataset_omega: np.ndarray | None = None,
    spatial: tuple | None = None,
    saturation: float | None = None,
) -> Iterator[tuple]:
    """Read sparse pixels a chunk at a time, with each frame's dty row.

    A frame's omega and dty are the DataSet's, where the DataSet has the frame (``dataset_omega``, ``dataset_dty``:
    ImageD11 has binned the scan into rotations by then, e.g. replacing the dty of a fly scan by its mean over each
    rotation), else the sparse group's own readings of ``omega_motor`` and ``dty_motor``. Its row is the bin of
    ``ybinedges`` holding its dty, divided by ``gridstep`` (rows summed in groups); frames outside the bins, or that
    the DataSet slices out, get row -1. With a monitor, intensities are normalised to the flux as
    ImageD11's ``DataSet.set_monitor`` does: multiplied by ``monitor_ref / monitor`` frame by frame, so beam decay
    within a rotation and between the dty rows' scans does not show in the data.

    Parameters
    ----------
    sparsefile
        ImageD11 sparse pixels file, one group per scan
    ybinedges
        [n_rows + 1] dty bin edges
    omega_motor, dty_motor
        Motor names in each group's ``measurement``
    chunk
        Pixels per chunk at most
    groups
        Groups to read (default: all)
    gridstep
        Rows summed in groups of this
    dataset_dty, scans
        [scans, frames] dty and scan names from the DataSet (:func:`read_dataset`). A DataSet scan is a sparse group
        ("1.1") or a slice of one ("1.1::[0:1440]": a fly scan split into its rotations)
    monitor
        Name of a counter in each group's ``measurement``, e.g. "fpico6"; None (default) for no normalisation.
        Frames where it is not positive are dropped.
    monitor_ref
        Reference value; default the counter's mean over the groups read
    masterfile
        Where to find the monitor if the sparse file lacks it, see :func:`read_monitor`
    dataset_omega
        [scans, frames] omega from the DataSet, as ``dataset_dty``
    spatial
        (dx, dy) [n_slow, n_fast] detector distortion maps (:func:`read_spatial`): each pixel is yielded at
        (row + dy[row, col], col + dx[row, col]), as ImageD11 corrects its peaks. None: the raw pixel positions
    saturation
        Detector counts at which a pixel saturated (e.g. 65535): each chunk then also holds whether each pixel's
        raw value reached it, before any normalisation. None (default): 5-tuples, no flag

    Yields
    ------
    tuple
        (slow, fast, omega, row, value), and with ``saturation`` also ``saturated`` (bool): NumPy arrays of at most
        ``chunk`` pixels
    """
    ybinedges = np.asarray(ybinedges)
    if groups is None:
        with h5py.File(sparsefile, "r") as h:
            groups = list(h.keys())
    mons = read_monitor(sparsefile, groups, monitor, masterfile) if monitor is not None else {}
    if monitor is not None and monitor_ref is None:
        monitor_ref = float(np.mean(np.concatenate(list(mons.values()))))
    slices = _scan_slices(scans) if scans is not None else {}
    with h5py.File(sparsefile, "r") as h:
        for name in groups:
            gr = h[name]
            nnz = gr["nnz"][()]
            om_f = _frame_positions(gr, name, omega_motor, dataset_omega, slices, sparsefile)
            dty_f = _frame_positions(gr, name, dty_motor, dataset_dty, slices, sparsefile)
            ok = np.isfinite(om_f) & np.isfinite(dty_f)
            k_f = np.where(ok, anri.geom.scan.dty_to_row(dty_f, ybinedges, gridstep), -1).astype(np.int32)
            om_f = np.where(ok, om_f, 0.0).astype(np.float32)
            scale = np.ones(len(nnz), np.float32)
            if monitor is not None:
                mon = mons[name]
                scale = np.where(mon > 0, monitor_ref / np.where(mon > 0, mon, 1.0), 0.0).astype(np.float32)
                k_f = np.where(mon > 0, k_f, -1).astype(np.int32)
            frame = np.repeat(np.arange(len(nnz), dtype=np.int32), nnz)  # each pixel's frame
            n = len(frame)
            for s0 in range(0, n, chunk):
                m = min(chunk, n - s0)
                fr = frame[s0 : s0 + m]
                slow, fast = gr["row"][s0 : s0 + m], gr["col"][s0 : s0 + m]
                if spatial is not None:  # where the pixel really is: its distortion-corrected position
                    slow, fast = slow + spatial[1][slow, fast], fast + spatial[0][slow, fast]
                raw = gr["intensity"][s0 : s0 + m]
                item = (slow.astype(np.float32), fast.astype(np.float32), om_f[fr], k_f[fr],
                        raw.astype(np.float32) * scale[fr])  # fmt: skip
                yield item if saturation is None else (*item, raw >= saturation)


def prefetch(chunks: Iterable, depth: int = 2) -> Iterator:
    """Read ahead: produce the items of an iterable in a background thread, up to ``depth`` ahead of the consumer.

    With :func:`stream_sparse`, the next chunk is read and decompressed while the current one is processed, so the
    disk and the GPU (or CPU) work at the same time instead of in turn. An error in the reader is raised in the
    consumer.

    Parameters
    ----------
    chunks
        Any iterable
    depth
        Items read ahead at most

    Yields
    ------
    object
        The items of chunks, in order
    """
    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=depth)
    done = object()
    stop = threading.Event()

    def reader() -> None:
        try:
            for item in chunks:
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
            q.put(done)
        except BaseException as e:  # noqa: BLE001  (handed to the consumer)
            q.put(e)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


def tensormap_from_recon(
    maps: dict, lattice_parameters: ArrayLike, spacegroup: int, phase_name: str, step: float
) -> TensorMap:
    """Build a single-phase 2D ImageD11 TensorMap from maps in reconstruction order.

    Parameters
    ----------
    maps
        ``{name: [n, n, ...] array}`` in reconstruction order (e.g. on the grid of :func:`anri.geom.scan.recon_to_step`),
        including "UBI" [n, n, 3, 3] (NaN outside the sample) and "phase_ids" [n, n] (0 inside, -1 outside)
    lattice_parameters
        a, b, c, alpha, beta, gamma
    spacegroup
        Space group number
    phase_name
        Name of the phase
    step
        Voxel size

    Returns
    -------
    TensorMap
        ``ImageD11.sinograms.tensor_map.TensorMap`` with shape (1, n, n)
    """
    from ImageD11.sinograms.tensor_map import TensorMap
    from ImageD11.unitcell import unitcell

    tmap = TensorMap(
        maps={k: TensorMap.recon_order_to_map_order(np.asarray(v)) for k, v in maps.items()}, steps=[step] * 3
    )
    tmap.phases = {0: unitcell(list(np.asarray(lattice_parameters, float)), int(spacegroup), name=phase_name)}
    return tmap


def entries_from_tensormap(tmap: TensorMap, phase_id: int = 0, z_layer: int = 0) -> dict:
    """Map entries for one phase of a 2D ImageD11 TensorMap, at sample-frame positions.

    Uses ImageD11's map -> reconstruction -> sample conventions (:func:`anri.geom.scan.recon_to_sample`), so a
    simulation from this map lines up with what ImageD11 reconstructs. Density comes from an optional "density" map (e.g. for pores),
    and is 1 where the map has none. An optional "sig_rot" map (radians) gives each entry an intrinsic orientation
    spread, see :func:`anri.fwd.render.render_row`.

    Parameters
    ----------
    tmap
        ``ImageD11.sinograms.tensor_map.TensorMap`` with square voxels in the slice
    phase_id
        Which phase to take, from ``tmap.phase_ids``
    z_layer
        Which z layer to take

    Returns
    -------
    entries: dict
        "ubi" [N, 3, 3], "pos" [N, 3] and "density" [N]; "sig_rot" [N] if the map has it
    """
    from ImageD11.sinograms.tensor_map import TensorMap

    from anri.maths.utils import float64

    _, ny, nx = tmap.shape
    ystep = float(tmap.steps[1])
    if not np.isclose(ystep, tmap.steps[2]):
        msg = "voxels must be square in the slice"
        raise ValueError(msg)
    phase = TensorMap.map_order_to_recon_order(tmap.phase_ids, z_layer)  # (nx, ny)
    ubi = TensorMap.map_order_to_recon_order(tmap.UBI, z_layer)
    ri, rj = np.nonzero(phase == phase_id)
    with float64():  # positions set the parallax: keep them exact
        sx, sy = (np.asarray(v) for v in anri.geom.scan.recon_to_sample(ri, rj, (nx, ny), ystep))
    if "density" in tmap.maps:
        density = TensorMap.map_order_to_recon_order(tmap["density"], z_layer)[ri, rj].astype(float)
    else:
        density = np.ones(ri.size)
    out = {
        "ubi": ubi[ri, rj],
        "pos": np.stack([sx, sy, np.zeros_like(sx)], 1).astype(float),
        "density": density,
    }
    if "sig_rot" in tmap.maps:
        out["sig_rot"] = TensorMap.map_order_to_recon_order(tmap["sig_rot"], z_layer)[ri, rj].astype(float)
    return out


def motor_grid(omega_range: tuple[float, float], ostep: float, dty_range: tuple[float, float], ystep: float) -> tuple:
    """Regular (dty row, frame) grid of motor positions for a phantom scan.

    Frames are centred on ``omega_range[0] + (i + 0.5) * ostep``. dty rows run from ``dty_range[0]`` to
    ``dty_range[1]`` inclusive, in steps of ``ystep``.

    Returns
    -------
    omega, dty: np.ndarray
        [n_rows, n_frames] each
    """
    n_frames = round((omega_range[1] - omega_range[0]) / ostep)
    n_rows = round((dty_range[1] - dty_range[0]) / ystep) + 1
    omega = omega_range[0] + (np.arange(n_frames) + 0.5) * ostep
    dty = dty_range[0] + np.arange(n_rows) * ystep
    return np.broadcast_to(omega, (n_rows, n_frames)).copy(), np.repeat(dty[:, None], n_frames, 1)


def write_scan(
    hout: h5py.File,
    scan: str,
    frame: np.ndarray,
    pixel: np.ndarray,
    value: np.ndarray,
    omega: np.ndarray,
    dty: np.ndarray,
    det_shape: tuple[int, int],
    cut: float = 1,
    omega_motor: str = "rot_center",
    dty_motor: str = "dty",
) -> int:
    """Write one dty row of sparse pixels as an ImageD11 scan group.

    Values are rounded to integer counts and only counts ``> cut`` are kept, as ImageD11's segmenter does.
    Pixels must be sorted by (frame, pixel), as :func:`anri.fwd.render.render_row` returns them.

    Parameters
    ----------
    hout
        Open, writable HDF5 file
    scan
        Group name, e.g. "1.1"
    frame, pixel, value
        Sparse pixels: frame index within the row, pixel = slow * n_fast + fast, and intensity
    omega, dty
        [n_frames] motor positions of the row
    det_shape
        (n_slow, n_fast)
    cut
        Keep counts strictly above this
    omega_motor, dty_motor
        Motor names, which must match the DataSet's ``omegamotor`` and ``dtymotor``

    Returns
    -------
    n_pixels: int
        Number of pixels written
    """
    counts = np.round(value)
    keep = counts > cut
    frame, pixel, counts = frame[keep], pixel[keep], counts[keep]
    n_frames = len(omega)
    opts = {"chunks": (10000,), "maxshape": (None,), "compression": "lzf", "shuffle": True}
    g = hout.create_group(scan)
    g.attrs["itype"] = "uint32"
    g.attrs["nframes"] = n_frames
    g.attrs["shape0"] = det_shape[0]
    g.attrs["shape1"] = det_shape[1]
    g.create_dataset("nnz", data=np.bincount(frame, minlength=n_frames).astype(np.uint32))
    g.create_dataset("row", data=(pixel // det_shape[1]).astype(np.uint16), **opts)
    g.create_dataset("col", data=(pixel % det_shape[1]).astype(np.uint16), **opts)
    g.create_dataset("intensity", data=counts.astype(np.uint32), **opts)
    g.create_dataset(f"measurement/{omega_motor}", data=np.asarray(omega, dtype=float))
    g.create_dataset(f"measurement/{dty_motor}", data=np.asarray(dty, dtype=float))
    g.create_dataset(f"instrument/positioners/{dty_motor}", data=np.asarray(dty, dtype=float))
    return int(counts.size)


def write_dataset(
    sparse_path: str,
    analysisroot: str,
    sample: str,
    dset: str,
    y0: float,
    parfile: str | None = None,
    e2dxfile: str | None = None,
    e2dyfile: str | None = None,
    omega_motor: str = "rot_center",
    dty_motor: str = "dty",
) -> str:
    """Write an ImageD11 DataSet file for a sparse pixels file, ready for the S3DXRD pipeline.

    The DataSet uses ImageD11's standard layout: it is saved as
    ``<analysisroot>/<sample>/<sample>_<dset>/<sample>_<dset>_dataset.h5``, and the peaks table,
    columnfiles and grains that the pipeline writes later go in the same folder.

    Parameters
    ----------
    sparse_path
        Sparse pixels file, e.g. from :func:`simulate_sparse`. It can live anywhere.
    analysisroot, sample, dset
        Where the DataSet goes, and its name
    y0
        dty at which the rotation axis is in the beam
    parfile
        Parameter file for the pipeline, e.g. the ``pars.json`` from :func:`write_pars`
    e2dxfile, e2dyfile
        Spatial distortion files, e.g. from :func:`write_zero_distortion`. ImageD11 needs a spatial
        correction to give the 2D peaks their corrected ``sc`` and ``fc`` columns.
    omega_motor, dty_motor
        Motor names used in the sparse file

    Returns
    -------
    dsfile: str
        Path of the DataSet file
    """
    from ImageD11.sinograms.dataset import DataSet

    ds = DataSet(
        analysisroot=os.path.abspath(analysisroot), sample=sample, dset=dset, omegamotor=omega_motor, dtymotor=dty_motor
    )
    ds.import_from_sparse(os.path.abspath(sparse_path))
    # these attributes are not declared on DataSet, so set them with setattr
    for name, path in (("parfile", parfile), ("e2dxfile", e2dxfile), ("e2dyfile", e2dyfile)):
        if path is not None:
            setattr(ds, name, os.path.abspath(path))
    dsfile = ds.dsfile_default
    os.makedirs(os.path.dirname(dsfile), exist_ok=True)
    ds.save(dsfile)
    # older ImageD11 releases don't save y0 themselves
    with h5py.File(dsfile, "a") as f:
        f.attrs["y0"] = y0
    return dsfile


def write_peaks_table(dsfile: str, nproc: int | None = None, algorithm: str = "lmlabel", wtmax: int = 70000) -> str:
    """Label the sparse pixels into ImageD11's peaks table: the last step of a simulation.

    After this, the S3DXRD pipeline can start at indexing (e.g. ``tomo_1_index``); the DataSet
    makes the 2D and 4D peaks from the peaks table on demand.

    This runs ``ImageD11.sinograms.properties.main``, which uses a multiprocessing pool. Scripts
    that call it must do so under ``if __name__ == "__main__":``.

    Parameters
    ----------
    dsfile
        DataSet file, e.g. from :func:`write_dataset`
    nproc
        Number of worker processes. ``None`` lets ImageD11 use every available core.
    algorithm, wtmax
        Labelling options for ImageD11, as in its ``0_segment_and_label`` notebook

    Returns
    -------
    pksfile: str
        Path of the peaks table
    """
    import ImageD11.sinograms.properties
    from ImageD11.sinograms.dataset import load

    options = {"algorithm": algorithm, "wtmax": wtmax, "save_overlaps": False, "nproc": nproc}
    ImageD11.sinograms.properties.main(dsfile, options=options)
    return load(dsfile).pksfile


def write_pars(folder: str, geometry: dict, phases: dict) -> str:
    """Write ImageD11 parameter files: ``geometry.par``, one ``<phase>.par`` per phase, and ``pars.json``.

    Parameters
    ----------
    folder
        Where to write them
    geometry
        Detector and beam parameters, as in an ImageD11 ``.par`` file. ImageD11 needs ``t_x``, ``t_y``, ``t_z`` and
        ``omegasign`` to compute peak geometry (e.g. ``DataSet.update_colfile_pars``): those missing are written as
        0, 0, 0 and 1 (the only ``omegasign`` anri supports).
    phases
        ``{phase_name: {"cell__a": ..., ..., "cell_lattice_[P,A,B,C,I,F,R]": ...}}``

    Returns
    -------
    path: str
        Path of ``pars.json``, which ``ImageD11.unitcell.Phases`` (and ``DataSet.phases``) read
    """
    import json

    os.makedirs(folder, exist_ok=True)
    geometry = {"t_x": 0.0, "t_y": 0.0, "t_z": 0.0, "omegasign": 1.0, **geometry}
    write_par(os.path.join(folder, "geometry.par"), geometry)
    for name, cell in phases.items():
        write_par(os.path.join(folder, f"{name}.par"), cell)
    schema = {"geometry": {"file": "geometry.par"}, "phases": {name: {"file": f"{name}.par"} for name in phases}}
    path = os.path.join(folder, "pars.json")
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
    return path


def write_zero_distortion(folder: str, det_shape: tuple[int, int]) -> tuple[str, str]:
    """Write ``e2dx.edf`` and ``e2dy.edf`` spatial distortion files that are zero everywhere.

    Simulated detectors have no distortion, but ImageD11 only gives 2D peaks corrected ``sc``/``fc``
    columns when a spatial correction is set.

    Returns
    -------
    e2dxfile, e2dyfile: str
    """
    from fabio.edfimage import EdfImage

    os.makedirs(folder, exist_ok=True)
    paths = os.path.join(folder, "e2dx.edf"), os.path.join(folder, "e2dy.edf")
    for path in paths:
        EdfImage(data=np.zeros(det_shape, np.float32)).write(path)
    return paths


def simulate_sparse(
    path: str,
    entries: dict,
    hkls: np.ndarray,
    F2: np.ndarray,
    geom: dict,
    omega: np.ndarray,
    dty: np.ndarray,
    det_shape: tuple[int, int],
    cut: float = 1,
    transmission: np.ndarray | None = None,
    omega_motor: str = "rot_center",
    dty_motor: str = "dty",
    window: tuple[int, int, int] = (3, 7, 7),
    batch: int = 2**16,
    max_frames: int | None = None,
) -> dict:
    """Render every dty row of a scan and write the sparse pixels file, one row at a time.

    Parameters
    ----------
    path
        Output HDF5 file; must not exist
    entries, hkls, F2, geom
        See :func:`anri.fwd.render.render_row`, for a single phase
    omega, dty
        [n_rows, n_frames] motor positions, e.g. from :func:`motor_grid` or an ImageD11 DataSet
    det_shape
        (n_slow, n_fast)
    cut
        Keep counts strictly above this
    transmission
        Optional [n_rows, n_frames] transmission factor per frame
    omega_motor, dty_motor
        Motor names written to the file
    window, batch, max_frames
        Passed on to :func:`anri.fwd.render.render_row`

    Returns
    -------
    stats: dict
        Per row: as :func:`write_rendered_rows`
    """
    with h5py.File(path, "w-") as hout:
        return write_rendered_rows(hout, [(entries, hkls, F2)], geom, omega, dty, det_shape, cut=cut,
                                   transmission=transmission, omega_motor=omega_motor, dty_motor=dty_motor,
                                   window=window, batch=batch, max_frames=max_frames)  # fmt: skip


def write_rendered_rows(
    hout: h5py.File, phases: list, geom: dict, omega: np.ndarray, dty: np.ndarray, det_shape: tuple[int, int], *,
    rows: range | None = None, cut: float = 1, transmission: np.ndarray | None = None, omega_motor: str = "rot_center",
    dty_motor: str = "dty", log: Callable | None = None, **render_kwargs: Any,  # noqa: ANN401
) -> dict:  # fmt: skip
    """Render dty rows of a scan, every phase summed, and write each as an ImageD11 scan group ("<row + 1>.1").

    Parameters
    ----------
    hout
        Open, writable HDF5 file
    phases
        [(entries, hkls, F2), ...] one per phase, see :func:`anri.fwd.render.render_phases`
    geom, omega, dty, det_shape, cut, transmission, omega_motor, dty_motor
        See :func:`simulate_sparse`
    rows
        The rows to render (default all)
    log
        Called with a line per row rendered, if given
    **render_kwargs
        Passed on to :func:`anri.fwd.render.render_row` (window, batch, max_frames, ...)

    Returns
    -------
    stats: dict
        Per row of the scan: "n_peaks" rendered, "n_pixels" written, "captured_min", the smallest share of a peak
        within the scan's omega range that its window holds, and "n_outside", the peaks selected beyond the ends of
        the scan, centred outside its omega range (left out of captured_min). Rows not rendered: 0, 0, 1 and 0
    """
    import time

    n_rows = omega.shape[0]
    stats = {"n_peaks": np.zeros(n_rows, int), "n_pixels": np.zeros(n_rows, int), "captured_min": np.ones(n_rows),
             "n_outside": np.zeros(n_rows, int)}  # fmt: skip
    for i in range(n_rows) if rows is None else rows:
        t0 = time.perf_counter()
        row = make_row(omega[i], dty[i], None if transmission is None else transmission[i])
        frame, pixel, value, rstats = render_phases(phases, geom, row, det_shape, **render_kwargs)
        stats["n_peaks"][i] = rstats["n_peaks"]
        inside = np.asarray(rstats["in_range"]) >= 0.5
        stats["n_outside"][i] = int(np.sum(~inside))
        if inside.any():
            stats["captured_min"][i] = (np.asarray(rstats["captured"])[inside] / rstats["in_range"][inside]).min()
        stats["n_pixels"][i] = write_scan(
            hout, f"{i + 1}.1", frame, pixel, value, omega[i], dty[i], det_shape, cut, omega_motor, dty_motor
        )
        if log is not None:
            log(f"row {i} (dty {float(np.mean(dty[i])):.4g}): {stats['n_peaks'][i]} peaks ({stats['n_outside'][i]} "
                f"outside the omega range), {stats['n_pixels'][i]} pixels above {cut:g}, window capture >= "
                f"{stats['captured_min'][i]:.4f} ({time.perf_counter() - t0:.1f} s)")  # fmt: skip
    return stats
