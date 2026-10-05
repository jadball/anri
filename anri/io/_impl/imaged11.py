"""Read and write ImageD11 formats: parameters, TensorMaps, sparse pixel files and DataSet files.

A simulated scanning-3DXRD dataset is a sparse pixels file, a DataSet file and a peaks table. The sparse file has one
group per dty row ("1.1", "2.1", ...), laid out like ImageD11's own segmented files, so ImageD11's
S3DXRD pipeline can read it directly.

ImageD11 itself is only imported by the functions that need it.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import h5py
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

import anri.geom
from anri.fwd._impl.render import make_row, render_row

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
        "det_trans", "beam_cen_shift", "x_distance_shift" from :func:`anri.geom.detector_transforms` (pixels to lab,
        see :func:`anri.geom.det_to_lab`), and "s_step_lab", "f_step_lab", "det_origin_lab" from
        :func:`anri.geom.detector_basis_vectors_lab` (lab to pixels, see :func:`anri.geom.raytrace_to_det`)
    """
    det_trans, beam_cen_shift, x_distance_shift = anri.geom.detector_transforms(
        *(float(pars[k]) for k in _GEOMETRY_KEYS), *(float(pars[k]) for k in _ORIENTATION_KEYS)
    )
    s_step_lab, f_step_lab, det_origin_lab = anri.geom.detector_basis_vectors_lab(
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
        vertical: beam divergence and polarisation are defined across it, see :func:`anri.geom.beam_basis`.

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
    """Build the geometry dict for :func:`anri.fwd.render_row` from ImageD11 parameters.

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
        :func:`anri.fwd.beam_weight`. A pencil beam is narrow, a box or horizontal line beam wider than the sample.
    voxel_size
        Side length of the voxels (same units as dty)
    pol_factor
        Degree of horizontal polarisation, see :func:`anri.fwd.polarisation`
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


def read_dataset(dsfile: str) -> dict:
    """Read the scan layout of an ImageD11 DataSet file, without ImageD11.

    Parameters
    ----------
    dsfile
        ``<sample>_<dset>_dataset.h5``

    Returns
    -------
    dict
        "y0" (None if absent), "ybincens", "ybinedges", "obinedges", "dtymotor", "omegamotor", "parfile" and
        "sparsefile" (made absolute: a relative path is relative to the DataSet's folder; "" if absent), and "dty"
        [scans, frames] and
        "scans" (None if absent): each scan's dty, for sparse files without a dty column
    """
    with h5py.File(dsfile, "r") as h:
        attrs = dict(h.attrs)
        out = {k: h[k][()] for k in ("ybincens", "ybinedges", "obinedges")}
        out["dty"] = h["dty"][()] if "dty" in h else None
        out["scans"] = (
            [x.decode() if isinstance(x, bytes) else str(x) for x in h["scans"][()]] if "scans" in h else None
        )

    def absolute(key: str) -> str:
        path = str(attrs.get(key, ""))
        if path and not os.path.isabs(path):
            path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(dsfile)), path))
        return path

    out.update(
        parfile=absolute("parfile"),
        sparsefile=absolute("sparsefile"),
        y0=float(attrs["y0"]) if "y0" in attrs else None,
        dtymotor=str(attrs["dtymotor"]),
        omegamotor=str(attrs["omegamotor"]),
    )
    return out


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
) -> Iterator[tuple]:
    """Read sparse pixels a chunk at a time, with each frame's dty row.

    A frame's row is the bin of ``ybinedges`` holding its dty reading, divided by ``gridstep`` (rows summed in
    groups); frames outside the bins get row -1.

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
        [scans, frames] dty and scan names from the DataSet, for files without a dty column

    Yields
    ------
    tuple
        (slow, fast, omega, row, value): NumPy arrays of at most ``chunk`` pixels
    """
    ybinedges = np.asarray(ybinedges)
    n_rows = len(ybinedges) - 1
    with h5py.File(sparsefile, "r") as h:
        for name in groups if groups is not None else list(h.keys()):
            gr = h[name]
            nnz = gr["nnz"][()]
            om_f = gr[f"measurement/{omega_motor}"][()].astype(np.float32)
            if dty_motor in gr["measurement"]:
                dty_f = np.broadcast_to(gr[f"measurement/{dty_motor}"][()], nnz.shape)
            elif dataset_dty is not None and scans is not None:
                dty_f = np.asarray(dataset_dty)[scans.index(name)][: len(nnz)]
            else:
                msg = f"{sparsefile}:{name} has no {dty_motor}; pass the DataSet's dty and scans"
                raise KeyError(msg)
            k_f = np.searchsorted(ybinedges, dty_f) - 1
            k_f = np.where((k_f >= 0) & (k_f < n_rows), k_f // gridstep, -1).astype(np.int32)
            frame = np.repeat(np.arange(len(nnz), dtype=np.int32), nnz)  # each pixel's frame
            n = len(frame)
            for s0 in range(0, n, chunk):
                m = min(chunk, n - s0)
                fr = frame[s0 : s0 + m]
                yield (
                    gr["row"][s0 : s0 + m].astype(np.float32),
                    gr["col"][s0 : s0 + m].astype(np.float32),
                    om_f[fr],
                    k_f[fr],
                    gr["intensity"][s0 : s0 + m].astype(np.float32),
                )


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
        ``{name: [n, n, ...] array}`` in reconstruction order (e.g. on the grid of :func:`anri.geom.recon_to_step`),
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

    Uses ImageD11's own map -> reconstruction -> sample conventions, so a simulation from this map lines
    up with what ImageD11 reconstructs. Density comes from an optional "density" map (e.g. for pores),
    and is 1 where the map has none.

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
        "ubi" [N, 3, 3], "pos" [N, 3] and "density" [N]
    """
    from ImageD11.sinograms.geometry import recon_to_sample
    from ImageD11.sinograms.tensor_map import TensorMap

    _, ny, nx = tmap.shape
    ystep = float(tmap.steps[1])
    if not np.isclose(ystep, tmap.steps[2]):
        msg = "voxels must be square in the slice"
        raise ValueError(msg)
    phase = TensorMap.map_order_to_recon_order(tmap.phase_ids, z_layer)  # (nx, ny)
    ubi = TensorMap.map_order_to_recon_order(tmap.UBI, z_layer)
    ri, rj = np.nonzero(phase == phase_id)
    sx, sy = recon_to_sample(ri, rj, (nx, ny), ystep)
    if "density" in tmap.maps:
        density = TensorMap.map_order_to_recon_order(tmap["density"], z_layer)[ri, rj].astype(float)
    else:
        density = np.ones(ri.size)
    return {
        "ubi": ubi[ri, rj],
        "pos": np.stack([sx, sy, np.zeros_like(sx)], 1).astype(float),
        "density": density,
    }


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
    Pixels must be sorted by (frame, pixel), as :func:`anri.fwd.render_row` returns them.

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
        Detector and beam parameters, as in an ImageD11 ``.par`` file
    phases
        ``{phase_name: {"cell__a": ..., ..., "cell_lattice_[P,A,B,C,I,F,R]": ...}}``

    Returns
    -------
    path: str
        Path of ``pars.json``, which ``ImageD11.unitcell.Phases`` (and ``DataSet.phases``) read
    """
    import json

    os.makedirs(folder, exist_ok=True)
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
        See :func:`anri.fwd.render_row`, for a single phase
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
        Passed on to :func:`anri.fwd.render_row`

    Returns
    -------
    stats: dict
        Per row: "n_peaks" rendered, "n_pixels" written, and the minimum window "captured" fraction
    """
    n_rows = omega.shape[0]
    stats = {"n_peaks": np.zeros(n_rows, int), "n_pixels": np.zeros(n_rows, int), "captured_min": np.ones(n_rows)}
    with h5py.File(path, "w-") as hout:
        for i in range(n_rows):
            row = make_row(omega[i], dty[i], None if transmission is None else transmission[i])
            frame, pixel, value, rstats = render_row(
                entries, hkls, F2, geom, row, det_shape, window=window, batch=batch, max_frames=max_frames
            )
            stats["n_peaks"][i] = rstats["n_peaks"]
            if rstats["captured"].size:
                stats["captured_min"][i] = rstats["captured"].min()
            stats["n_pixels"][i] = write_scan(
                hout, f"{i + 1}.1", frame, pixel, value, omega[i], dty[i], det_shape, cut, omega_motor, dty_motor
            )
    return stats
