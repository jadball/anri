"""The calibration as a stage: an indexed map (``anri.index fit``) and the scan's peaks in, one HDF5 file out.

:func:`stage_fit` matches ImageD11's peaks to the map, fits the geometry (:func:`anri.calib.fit.calibrate`) and frees
each orientation's lattice with that geometry (:func:`anri.calib.fit.strain_left`). :func:`write_pars` writes the
result as ImageD11 parameter files, for anri or ImageD11 to use.
"""

from __future__ import annotations

from collections.abc import Callable

import h5py
import numpy as np

from anri.io.provenance import (
    chain,
    check_out,
    read_provenance,
    stage_path,
    upstream,
    write_provenance,
    write_results,
)

FREE = ("distance", "y_center", "z_center", "tilt_x", "tilt_y", "tilt_z", "wavelength")
_COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")
_IJ = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def stage_fit(
    fit_file: str, out: str | None = None, *, peaks: str | None = None, lattice: list | None = None,
    free: tuple = FREE, tols: tuple = (0.1, 0.05, 0.025, 0.025), n_iter: int = 10, clip: float = 4.0,
    min_occ: float = 0.01, min_peaks: int = 20, command: str | None = None, log: Callable = print,
) -> str:  # fmt: skip
    """Calibrate the geometry on a scan of an indexed sample.

    The starting geometry and the phase are those the map was indexed with (the histogram stage's ``pars.json``).
    Its orientations and occupancies place each peak's scattering origin; each orientation is refitted with the
    geometry. Wedge, chi and y0 stay fixed.

    Parameters
    ----------
    fit_file
        From :func:`anri.index.stages.stage_fit`
    out
        Output file (default: the input's name with ``_calib``)
    peaks
        ImageD11 peaks table (default: the DataSet's ``pksfile``)
    lattice
        Reference lattice (a, b, c, alpha, beta, gamma), e.g. a calibrant's certified one (default: the phase's)
    free
        Free parameters, from :data:`anri.calib.fit.GEOMETRY`. A calibrant (known lattice) frees "wavelength"; any
        other sample (known wavelength) frees "lattice", its scale, instead: they cannot both be free
    tols
        Matching tolerance on (h, k, l) of each round
    n_iter
        Gauss-Newton iterations per round
    clip
        Leave out peaks further than this many (robust) standard deviations from their prediction
    min_occ
        Leave out occupancies below this x the largest
    min_peaks
        Free the lattice of orientations with at least this many matched peaks (the strain left)
    command, log
        The command line, if run from one (recorded), and progress messages

    Returns
    -------
    str
        The file written: the fitted geometry ("geometry", attributes as in an ImageD11 ``.par``), the covariance of
        the free parameters, each orientation's UBI and the strain left with its lattice free, and the matched peaks
    """
    import anri.io.imaged11 as io
    from anri.crystal.lattice import lpars_to_B

    from . import fit

    options = {"peaks": peaks, "lattice": lattice, "free": list(free), "tols": list(tols), "n_iter": n_iter,
               "clip": clip, "min_occ": min_occ, "min_peaks": min_peaks}  # fmt: skip
    hr = read_provenance(upstream(fit_file, "histogram"))["resolved"]
    fr = read_provenance(fit_file)["resolved"]
    ds = io.read_dataset(hr["dataset_file"])
    pks_file = peaks or ds["pksfile"]
    if not pks_file:
        msg = f"{hr['dataset_file']} names no peaks table: give it with peaks"
        raise ValueError(msg)
    lpars = np.asarray(lattice if lattice is not None else hr["lattice"], float)
    B = np.asarray(lpars_to_B(lpars))
    y0, step = float(fr["y0"]), float(fr["voxel_size"])
    geometry = io.read_geometry(hr["parfile"])
    pars0 = fit.base_pars(geometry, y0)
    out = stage_path(fit_file, "calib", out)
    check_out(out, [*chain(fit_file), hr["dataset_file"], pks_file, hr["parfile"]])

    pk = io.read_peaks(pks_file, ds["omega"], ds["dty"], io.read_spatial(ds))
    with h5py.File(fit_file, "r") as h:
        U, f, cand = h["U"][()], h["f"][()], h["cand"][()]
    grid_o = np.where(f > min_occ * f.max(), cand, -1).astype(np.int32)
    grid_w = np.where(grid_o >= 0, f, 0.0)
    log(f"{len(pk['sc'])} peaks from {pks_file}; map: {len(U)} orientations on {np.any(grid_o >= 0, 1).sum()} "
        f"voxels of {step:g}, y0 {y0:g}; lattice {np.round(lpars, 6).tolist()}")  # fmt: skip
    log("start (" + hr["parfile"] + "): " + ", ".join(f"{k} {pars0[k]:.7g}" for k in free))
    res = fit.calibrate(pk, np.linalg.inv(U @ B), grid_o, grid_w, step, pars0, tuple(free), tuple(tols), n_iter,
                        clip, log=log)  # fmt: skip
    st = fit.strain_left(pk, res["matched"], res["ubi"], B, res["pars"], res["sig"], min_peaks, n_iter, log=log)
    E = st["strain"]
    hydro = np.trace(E, axis1=1, axis2=2) / 3
    dev = E - hydro[:, None, None] * np.eye(3)
    ok = np.isfinite(hydro)
    err = np.sqrt(np.diag(res["cov"]))
    log("fitted: " + ", ".join(f"{k} {res['pars'][k]:.8g} +- {e:.2g}" for k, e in zip(free, err)))
    if ok.any():
        w = st["n_peaks"][ok]
        log(f"strain left with free lattices ({ok.sum()} orientations, peak-weighted mean, 1e-5): hydrostatic "
            f"{np.average(hydro[ok], weights=w) * 1e5:+.1f}; deviatoric "
            + ", ".join(f"{c} {np.average(dev[ok, i, j], weights=w) * 1e5:+.1f}" for c, (i, j) in zip(_COMPONENTS, _IJ))
            + f"; chi2 x {st['chi2_ratio']:.3f}")  # fmt: skip

    resolved = {"dataset_file": hr["dataset_file"], "peaks_file": pks_file, "parfile": hr["parfile"],
                "phase": hr["phase"], "space_group": hr["space_group"], "lattice": lpars.tolist(), "y0": y0,
                "voxel_size": step, **{f"start_{k}": pars0[k] for k in fit.GEOMETRY}}  # fmt: skip
    m = res["matched"]
    with h5py.File(out, "w") as h:
        write_provenance(h, "calib", fit_file, options, resolved, command)
        g = h.create_group("geometry")
        for k, v in fit.imaged11_pars(res["pars"]).items():
            g.attrs[k] = v
        g.attrs["lattice_scale"] = res["pars"]["lattice"]
        h["free"] = np.array(free, dtype="S")
        h["cov"], h["sig"], h["ubi"] = res["cov"], res["sig"], res["ubi"]
        h["strain"], h["n_peaks"] = E, st["n_peaks"]
        for k, v in m.items():
            h.create_dataset(
                f"matched/{k}", data=np.asarray(v), compression="lzf" if np.asarray(v).size > 1000 else None
            )
        for k in ("sc", "fc", "omega", "dty", "row", "intensity"):
            h[f"matched/{k}"] = pk[k][m["index"]]
    results = {**{k: res["pars"][k] for k in free}, **{f"{k}_err": float(e) for k, e in zip(free, err)},
               "peaks_matched": len(m["index"]), "sigma_sc_px": float(res["sig"][0]),
               "sigma_fc_px": float(res["sig"][1]), "sigma_omega_deg": float(res["sig"][2])}  # fmt: skip
    if ok.any():
        w = st["n_peaks"][ok]
        results["strain_left_hydrostatic"] = float(np.average(hydro[ok], weights=w))
        results["strain_left_deviatoric_max"] = float(np.abs(np.average(dev[ok], axis=0, weights=w)).max())
    write_results(out, results)
    log(f"-> {out}")
    return out


def write_pars(calib_file: str, folder: str) -> str:
    """Write a calibration as ImageD11 parameter files (``geometry.par``, ``<phase>.par``, ``pars.json``) in folder.

    The phase's cell is the reference lattice, scaled by the fitted lattice scale if it was free. Returns the path
    of ``pars.json``.
    """
    import anri.io.imaged11 as io

    r = read_provenance(calib_file)["resolved"]
    with h5py.File(calib_file, "r") as h:
        geometry = {k: (v.item() if hasattr(v, "item") else v) for k, v in h["geometry"].attrs.items()}
    scale = float(geometry.pop("lattice_scale"))
    _, name, cell = io.read_pars_json(r["parfile"], r["phase"])
    a = np.asarray(r["lattice"], float)
    cell = {**cell, "cell__a": a[0] * scale, "cell__b": a[1] * scale, "cell__c": a[2] * scale,
            "cell_alpha": a[3], "cell_beta": a[4], "cell_gamma": a[5]}  # fmt: skip
    key = "cell_lattice_[P,A,B,C,I,F,R]"
    if isinstance(cell.get(key), float) and float(cell[key]).is_integer():
        cell[key] = int(cell[key])  # a space group number, not 227.0
    return io.write_pars(folder, geometry, {name: cell})
