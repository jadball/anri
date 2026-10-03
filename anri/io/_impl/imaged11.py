"""Read and write ImageD11 formats: parameters, TensorMaps, sparse pixel files and DataSet files.

A simulated scanning-3DXRD dataset is a sparse pixels file plus a DataSet file. The sparse file has one
group per dty row ("1.1", "2.1", ...), laid out like ImageD11's own segmented files, so ImageD11's
S3DXRD pipeline can read it directly.

ImageD11 itself is only imported by the functions that need it.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import h5py
import jax.numpy as jnp
import numpy as np

import anri.geom
from anri.fwd._impl.render import make_row, render_row

if TYPE_CHECKING:
    from ImageD11.sinograms.tensor_map import TensorMap

_GEOMETRY_KEYS = ("y_center", "y_size", "tilt_y", "z_center", "z_size", "tilt_z", "tilt_x", "distance")
_ORIENTATION_KEYS = ("o11", "o12", "o21", "o22")


def geom_from_pars(
    pars: dict,
    y0: float,
    sig_wavelength: float,
    sig_ky: float,
    sig_kz: float,
    sig_beam: float,
    voxel_size: float,
    pol_factor: float = 1.0,
) -> dict:
    """Build the geometry dict for :func:`anri.fwd._impl.render.render_row` from ImageD11 parameters.

    Parameters
    ----------
    pars
        ImageD11 parameters: detector geometry (``y_center`` ... ``o22``), ``wavelength``, ``wedge``, ``chi``
    y0
        dty at which the rotation axis is in the beam
    sig_wavelength, sig_ky, sig_kz
        Standard deviations of the wavelength (angstrom) and of the beam divergence (radians)
    sig_beam
        Standard deviation of the beam profile across lab y (same units as dty)
    voxel_size
        Side length of the voxels (same units as dty)
    pol_factor
        Degree of horizontal polarisation, see :func:`anri.fwd._impl.render.polarisation`
    """
    if float(pars.get("omegasign", 1.0)) != 1.0:
        msg = "omegasign != 1 is not supported"
        raise ValueError(msg)
    det_trans, beam_cen_shift, x_distance_shift = anri.geom.detector_transforms(
        *(float(pars[k]) for k in _GEOMETRY_KEYS), *(float(pars[k]) for k in _ORIENTATION_KEYS)
    )
    sc_lab, fc_lab, norm_lab = anri.geom.detector_basis_vectors_lab(det_trans, beam_cen_shift, x_distance_shift)
    return {
        "wavelength": float(pars["wavelength"]),
        "k_in_lab": jnp.array([1.0, 0.0, 0.0]),
        "wedge": float(pars.get("wedge", 0.0)),
        "chi": float(pars.get("chi", 0.0)),
        "y0": y0,
        "sc_lab": sc_lab,
        "fc_lab": fc_lab,
        "norm_lab": norm_lab,
        "sig_wavelength": sig_wavelength,
        "sig_ky": sig_ky,
        "sig_kz": sig_kz,
        "sig_beam": sig_beam,
        "voxel_size": voxel_size,
        "pol_factor": pol_factor,
    }


def write_par(path: str, pars: dict) -> None:
    """Write ImageD11 parameters as a ``.par`` file of ``key value`` lines."""
    with open(path, "w") as f:
        f.writelines(f"{key} {value}\n" for key, value in pars.items())


def entries_from_tensormap(tmap: TensorMap, phase_id: int = 0, z_layer: int = 0) -> dict:
    """Map entries for one phase of a 2D ImageD11 TensorMap, at sample-frame positions.

    Uses ImageD11's own map -> reconstruction -> sample conventions, so a simulation from this map lines
    up with what ImageD11 reconstructs. Density is 1 for every entry.

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
    return {
        "ubi": ubi[ri, rj],
        "pos": np.stack([sx, sy, np.zeros_like(sx)], 1).astype(float),
        "density": np.ones(ri.size),
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
    Pixels must be sorted by (frame, pixel), as :func:`anri.fwd._impl.render.render_row` returns them.

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
    ds_path: str,
    y0: float,
    parfile: str | None = None,
    sample: str = "sample",
    dset: str = "dataset",
    omega_motor: str = "rot_center",
    dty_motor: str = "dty",
) -> None:
    """Write an ImageD11 DataSet file for a sparse pixels file, ready for the S3DXRD pipeline."""
    from ImageD11.sinograms.dataset import DataSet

    folder = os.path.dirname(os.path.abspath(ds_path))
    ds = DataSet(analysisroot=folder, sample=sample, dset=dset, omegamotor=omega_motor, dtymotor=dty_motor)
    ds.import_from_sparse(os.path.abspath(sparse_path))
    ds.analysispath = folder
    if parfile is not None:
        setattr(ds, "parfile", os.path.abspath(parfile))  # noqa: B010 (not declared on DataSet)
    ds.save(ds_path)
    # older ImageD11 releases don't save y0 themselves
    with h5py.File(ds_path, "a") as f:
        f.attrs["y0"] = y0


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
) -> dict:
    """Render every dty row of a scan and write the sparse pixels file, one row at a time.

    Parameters
    ----------
    path
        Output HDF5 file; must not exist
    entries, hkls, F2, geom
        See :func:`anri.fwd._impl.render.render_row`, for a single phase
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
    window, batch
        Passed on to :func:`anri.fwd._impl.render.render_row`

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
                entries, hkls, F2, geom, row, det_shape, window=window, batch=batch
            )
            stats["n_peaks"][i] = rstats["n_peaks"]
            if rstats["captured"].size:
                stats["captured_min"][i] = rstats["captured"].min()
            stats["n_pixels"][i] = write_scan(
                hout, f"{i + 1}.1", frame, pixel, value, omega[i], dty[i], det_shape, cut, omega_motor, dty_motor
            )
    return stats
