"""The calibration as stages: powder rings first, then an indexed map (``anri.index fit``) and the scan's peaks.

:func:`stage_rings` fits the distance, beam centre and two tilts to a known phase's powder rings (a dense image, or
a scan's sparse pixels summed), to get a geometry good enough to index.
:func:`stage_fit` then matches ImageD11's peaks to the map, fits the geometry (:func:`anri.calib.fit.calibrate`) and frees
each orientation's lattice with that geometry (:func:`anri.calib.fit.strain_left`). :func:`write_pars` writes the
result as ImageD11 parameter files, for anri or ImageD11 to use. :func:`load_inputs` reads what the stage starts
from, to look at before running it (the calibration guide does), and :func:`run_command` gives the command line
that remakes a calibration file.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import h5py
import numpy as np

from anri.io.provenance import (
    chain,
    check_out,
    read_provenance,
    recorder,
    stage_path,
    upstream,
    write_log,
    write_provenance,
    write_results,
)

FREE = ("distance", "y_center", "z_center", "tilt_x", "tilt_y", "tilt_z", "wavelength")
COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")  # the strain's six components, as (i, j) in IJ
IJ = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def select_occupancy(f: np.ndarray, min_occ: float) -> np.ndarray:
    """Return which of a fit's occupancies [V, K] the calibration uses: those above ``min_occ`` x the largest."""
    return f > min_occ * f.max()


def load_inputs(fit_file: str, peaks: str | None = None, lattice: list | None = None, min_occ: float = 0.01) -> dict:
    """Read what :func:`stage_fit` starts from: the peaks, the indexed map and the starting geometry.

    The map must be an ``anri.index`` fit stage file. A local search's is refused: its orientations, a fraction of a
    degree apart, let each round's matching pick the one that absorbs part of the geometry's error, so the rounds
    converge very slowly (on the Ni phantom, distance and wavelength still 1e-4 off after four rounds, against 1e-6
    from the fit stage's map). :func:`stage_fit` refits each orientation's rotation itself.

    Parameters
    ----------
    fit_file, peaks, lattice, min_occ
        See :func:`stage_fit`

    Returns
    -------
    dict
        "peaks" (:func:`anri.io.imaged11.read_peaks`) and "peaks_file"; the map as :func:`anri.calib.fit.match` takes
        it: "ubi" [G, 3, 3], "grid_o", "grid_w" (occupancies below min_occ x the largest left out) and "step" (the
        voxel size); "lattice" (the reference, a, b, c, alpha, beta, gamma) and its "B"; "pars" (the starting
        parameters, :func:`anri.calib.fit.base_pars`); "histogram" and "fit" (the resolved provenance of those stages)
    """
    import anri.io.imaged11 as io
    from anri.crystal.lattice import lpars_to_B

    from . import fit

    prov = read_provenance(fit_file)
    if prov["stage"] != "fit":
        msg = (f"{fit_file} is a {prov['stage']} stage file: give the fit stage's ({prov.get('source')}). A local "
               "search's orientations, a fraction of a degree apart, let each round's matching keep part of the "
               "geometry's error: the calibration then converges very slowly")  # fmt: skip
        raise ValueError(msg)
    hr = read_provenance(upstream(fit_file, "histogram"))["resolved"]
    fr = prov["resolved"]
    ds = io.read_dataset(hr["dataset_file"])
    pks_file = peaks or ds["pksfile"]
    if not pks_file:
        msg = f"{hr['dataset_file']} names no peaks table: give it with peaks"
        raise ValueError(msg)
    lpars = np.asarray(lattice if lattice is not None else hr["lattice"], float)
    B = np.asarray(lpars_to_B(lpars))
    with h5py.File(fit_file, "r") as h:
        U, f, cand = h["U"][()], h["f"][()], h["cand"][()]
    grid_o = np.where(select_occupancy(f, min_occ), cand, -1).astype(np.int32)
    n_used = max(int(np.max((grid_o >= 0).sum(1))), 1)  # each voxel's kept entries first, the empty columns dropped
    order = np.argsort(grid_o < 0, axis=1, kind="stable")[:, :n_used]
    grid_o, f = np.take_along_axis(grid_o, order, 1), np.take_along_axis(f, order, 1)
    return {
        "peaks": io.read_peaks(pks_file, ds["omega"], ds["dty"], io.read_spatial(ds)),
        "peaks_file": pks_file,
        "ubi": np.linalg.inv(U @ B),
        "grid_o": grid_o,
        "grid_w": np.where(grid_o >= 0, f, 0.0),
        "step": float(fr["voxel_size"]),
        "lattice": lpars,
        "B": B,
        "pars": fit.base_pars(io.read_geometry(hr["parfile"]), float(fr["y0"])),
        "histogram": hr,
        "fit": fr,
    }


def stage_fit(
    fit_file: str, out: str | None = None, *, peaks: str | None = None, lattice: list | None = None,
    free: tuple = FREE, tols: tuple = (0.1, 0.05, 0.025, 0.025), n_iter: int = 10, clip: float = 4.0,
    extra_rounds: int = 6, row_step: int = 1,
    min_occ: float = 0.01, min_peaks: int = 20, command: str | None = None, log: Callable = print,
) -> str:  # fmt: skip
    """Calibrate the geometry on a scan of an indexed sample.

    The starting geometry and the phase are those the map was indexed with (the histogram stage's ``pars.json``).
    Its orientations and occupancies place each peak's scattering origin; each orientation is refitted with the
    geometry. Wedge, chi and y0 stay fixed.

    For a calibrant or a little-deformed sample. Where a voxel holds orientations closer to one another than the
    geometry's error (cells, subgrains), matching each peak to the one that predicts it best absorbs part of that
    error, and the rounds converge slowly (on the 316L phantom's cells, a few tenths of a degree, the distance was
    still 6e-4 out after ten rounds): the stage warns. A deformed sample's geometry is refined with its strains.

    Parameters
    ----------
    fit_file
        From :func:`anri.index.stages.stage_fit` (not a local search's: see :func:`load_inputs`)
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
    extra_rounds
        Repeat the last tolerance while a round still moves the geometry by more than its errors, at most this many
        times (:func:`anri.calib.fit.calibrate`)
    row_step
        Use the peaks of every row_step-th dty row only, e.g. 10 on a CPU: matching costs in proportion to the
        peaks (on the Ni phantom every 10th row gave the geometry of all rows, within 1e-6)
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
    from . import fit

    log, lines = recorder(log)
    options = {"peaks": peaks, "lattice": lattice, "free": list(free), "tols": list(tols), "n_iter": n_iter,
               "clip": clip, "extra_rounds": extra_rounds, "row_step": row_step, "min_occ": min_occ, "min_peaks": min_peaks}  # fmt: skip
    d = load_inputs(fit_file, peaks, lattice, min_occ)
    hr, pk, B, lpars, pars0 = d["histogram"], d["peaks"], d["B"], d["lattice"], d["pars"]
    if row_step > 1:
        pk = {k: v[pk["row"] % row_step == 0] for k, v in pk.items()}
    pks_file, y0, step = d["peaks_file"], pars0["y0"], d["step"]
    out = stage_path(fit_file, "calib", out)
    check_out(out, [*chain(fit_file), hr["dataset_file"], pks_file, hr["parfile"]])

    log(f"{len(pk['sc'])} peaks from {pks_file}; map: {len(d['ubi'])} orientations on "
        f"{np.any(d['grid_o'] >= 0, 1).sum()} voxels of {step:g}, y0 {y0:g}; lattice {np.round(lpars, 6).tolist()}")  # fmt: skip
    log("start (" + hr["parfile"] + "): " + ", ".join(f"{k} {pars0[k]:.7g}" for k in free))
    res = fit.calibrate(pk, d["ubi"], d["grid_o"], d["grid_w"], step, pars0, tuple(free), tuple(tols), n_iter,
                        clip, extra_rounds, log=log)  # fmt: skip
    st = fit.strain_left(pk, res["matched"], res["ubi"], B, res["pars"], res["sig"], min_peaks, n_iter, log=log)
    E = st["strain"]
    parts = fit.split_strain(E, st["n_peaks"])
    ok = parts["fitted"]
    err = np.sqrt(np.diag(res["cov"]))
    log("fitted: " + ", ".join(f"{k} {res['pars'][k]:.8g} +- {e:.2g}" for k, e in zip(free, err)))
    if res["moved"].max() > 1:
        log(f"WARNING: the last round still moved {free[int(np.argmax(res['moved']))]} by {res['moved'].max():.1f} "
            f"sigma: the rounds have not converged after {len(tols) + extra_rounds}, and the errors above are too "
            "small. A deformed sample (orientations within a voxel closer than the geometry's error) converges "
            "slowly: calibrate on a calibrant or a little-deformed sample, or raise --extra-rounds")  # fmt: skip
    if ok.any():
        mean_dev = parts["mean_deviatoric"]
        log(f"strain left with free lattices ({ok.sum()} orientations, peak-weighted mean, 1e-5): hydrostatic "
            f"{parts['mean_hydrostatic'] * 1e5:+.1f}; deviatoric "
            + ", ".join(f"{c} {mean_dev[i, j] * 1e5:+.1f}" for c, (i, j) in zip(COMPONENTS, IJ))
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
               "sigma_fc_px": float(res["sig"][1]), "sigma_omega_deg": float(res["sig"][2]),
               "last_round_moved_sigma": float(res["moved"].max())}  # fmt: skip
    if ok.any():
        results["strain_left_hydrostatic"] = parts["mean_hydrostatic"]
        results["strain_left_deviatoric_max"] = float(np.abs(parts["mean_deviatoric"]).max())
    write_results(out, results)
    log(f"-> {out}")
    write_log(out, lines)
    return out


def stage_rings(
    source: str, out: str | None = None, *, parfile: str | None = None, phase: str | None = None,
    detector_mask: str | None = None, spatial: tuple | None = None, scans: int | None = None, threshold: float = 0.0,
    free: tuple = ("distance", "y_center", "z_center", "tilt_y", "tilt_z"), tols: tuple = (0.2, 0.1),
    n_iter: int = 30, n_eta: int = 360, clip: float = 4.0, merge: float = 1e-4, search: bool = True,
    max_shift: float = 200.0, span: tuple = (0.7, 1.4), command: str | None = None,
    log: Callable = print,
) -> str:  # fmt: skip
    """Fit the detector geometry to the powder rings of a known phase, as ImageD11's powder calibration does.

    The first calibration step: from the motors' distance and detector position to a geometry good enough to index
    (``python -m anri.index``), which :func:`stage_fit` then refines with the wavelength and tilt_x
    (:mod:`anri.calib.rings`).

    Parameters
    ----------
    source
        An ImageD11 DataSet (``..._dataset.h5``), whose sparse pixels are summed over every frame; or a detector
        image fabio reads (a powder image, e.g. a calibrant's, background subtracted)
    out
        Output file (default: ``<sample>_<dataset>_rings.h5``, or the image's name with ``_rings.h5``, in the working
        directory)
    parfile
        ``pars.json`` with the starting geometry and the phase (default: the DataSet's; needed for an image)
    phase
        The phase whose rings are fitted (default: the only one in parfile)
    detector_mask
        Detector mask, non-zero for masked pixels (default: the DataSet's maskfile; none: every pixel measured)
    spatial
        (e2dx, e2dy) distortion files for an image (a DataSet's own are used for its pixels)
    scans
        Sum only this many of the DataSet's scans, spread evenly (default: all)
    threshold
        Keep pixels above this, with it subtracted
    free
        Free parameters (:data:`anri.calib.fit.GEOMETRY`, not "lattice"); tilt_x moves no ring
    tols, n_iter, n_eta, clip
        As :func:`anri.calib.rings.fit_rings`
    merge
        Reflections within this d* (1 / angstrom) are one ring
    search
        First search for the centre (:func:`anri.calib.rings.find_centre`, up to ``max_shift`` pixels from the
        start) and the distance (:func:`anri.calib.rings.find_distance`, over ``span`` x the start), so that a start
        from the motors far from the answer still converges; the start's tilts are kept until the fit
    max_shift, span
        The search's range
    command, log
        The command line, if run from one (recorded), and progress messages

    Returns
    -------
    str
        The file written: the fitted geometry ("geometry", as :func:`stage_fit`'s, for :func:`write_pars`), the
        covariance of the free parameters, the rings ("tth", "used"), each (ring, eta sector)'s offset from its ring
        at the start and at the end ("offset_start", "offset", with the summed "intensity"), the image, and the
        searches' curves ("search"; :func:`anri.calib.diagnostics.read_rings` reads them all)
    """
    import anri.io.imaged11 as io

    from . import fit, rings

    log, lines = recorder(log)
    options = {"source": source, "parfile": parfile, "phase": phase, "detector_mask": detector_mask,
               "spatial": list(spatial) if spatial else None, "scans": scans, "threshold": threshold,
               "free": list(free), "tols": list(tols), "n_iter": n_iter, "n_eta": n_eta, "clip": clip,
               "merge": merge, "search": search, "max_shift": max_shift, "span": list(span)}  # fmt: skip
    is_ds = source.endswith("_dataset.h5")
    inputs = [source]
    if is_ds:
        ds = io.read_dataset(source)
        parfile = parfile or ds["parfile"]
        mask_file = detector_mask or ds["maskfile"]
        with h5py.File(ds["sparsefile"], "r") as h:
            groups = list(h.keys())
            a = h[groups[0]].attrs
            shape = (int(a["shape0"]), int(a["shape1"])) if "shape0" in a and "shape1" in a else None
        if scans:
            groups = [groups[i] for i in np.unique(np.linspace(0, len(groups) - 1, scans).round().astype(int))]
        dmaps = io.read_spatial(ds)
        spatial_src = ds["detectorh5"] or (f"{ds['e2dxfile']}, {ds['e2dyfile']}" if dmaps is not None else "")
        inputs.append(ds["sparsefile"])
        default_out = os.path.basename(source)[: -len("_dataset.h5")] + "_rings.h5"
    else:
        import fabio

        if not parfile:
            msg = "an image needs parfile (pars.json with the starting geometry and the phase)"
            raise ValueError(msg)
        mask_file = detector_mask
        image = np.asarray(fabio.open(source).data, np.float64)
        shape = image.shape
        dmaps = None
        if spatial:
            dmaps = tuple(np.asarray(fabio.open(f).data, np.float32) for f in spatial)
        spatial_src = ", ".join(spatial) if spatial else ""
        default_out = os.path.splitext(os.path.basename(source))[0] + "_rings.h5"
    good = io.read_mask(mask_file) if mask_file else None
    shape = shape or (good.shape if good is not None else None)
    if shape is None:
        msg = "the detector's shape is unknown (no mask, and none in the sparse file): give detector_mask"
        raise ValueError(msg)
    out = out or default_out
    parfile = os.path.abspath(parfile)
    check_out(out, [*inputs, parfile, mask_file])
    geo, phase, cell = io.read_pars_json(parfile, phase)
    lpars = np.array([cell[k] for k in ("cell__a", "cell__b", "cell__c", "cell_alpha", "cell_beta", "cell_gamma")])
    sg = int(cell["cell_lattice_[P,A,B,C,I,F,R]"])
    pars0 = fit.base_pars(geo, 0.0)
    log(f"pars {parfile}: phase {phase}, lattice {', '.join(f'{v:g}' for v in lpars)}, space group {sg}")
    log("start: " + ", ".join(f"{k} {pars0[k]:.7g}" for k in (*free, "wavelength")))

    if is_ds:
        chunks = io.prefetch(io.stream_sparse(ds["sparsefile"], ds["ybinedges"], ds["omegamotor"], ds["dtymotor"],
                                              1 << 22, groups, dataset_dty=ds["dty"], scans=ds["scans"],
                                              dataset_omega=ds["omega"]))  # fmt: skip
        image = rings.sum_frames(chunks, shape)
        log(f"summed {len(groups)} scans of {ds['sparsefile']}")
    px = rings.lit_pixels(image, good, dmaps, threshold)
    log(f"{len(px['sc'])} lit pixels; detector mask {mask_file or 'none'}; spatial correction {spatial_src or 'none'}")
    corner_s = np.array([0.0, 0.0, shape[0] - 1, shape[0] - 1])
    corner_f = np.array([0.0, shape[1] - 1, 0.0, shape[1] - 1])

    def rings_on(p: dict, f: float = 1.0) -> np.ndarray:  # the rings up to the detector's largest 2theta / f
        t = float(np.max(np.asarray(rings.pixel_tth_eta(corner_s, corner_f, p)[0])))
        return rings.ring_ds(lpars, sg, p["wavelength"], float(np.degrees(np.arctan(np.tan(np.radians(t)) / f))),
                             merge)  # fmt: skip

    sep = 2 * max(tols)
    start = rings.ring_offsets(px, rings_on(pars0), pars0, tols[0], n_eta, sep)["offset"]
    p1, centre, dist = pars0, None, None
    if search:
        p1, centre = rings.find_centre(px, pars0, max_shift, log=log)
        p1, dist = rings.find_distance(px, rings_on(p1, span[0]), p1, tols[0], tuple(span), log=log)
    ds_r = rings_on(p1)
    tth_max = float(np.degrees(np.arcsin(ds_r[-1] * p1["wavelength"] / 2) * 2)) if len(ds_r) else 0.0
    log(f"{len(ds_r)} rings up to 2theta {tth_max:.3f} deg")
    res = rings.fit_rings(px, ds_r, p1, tuple(free), tuple(tols), n_iter, n_eta, clip, log=log)
    end = rings.ring_offsets(px, ds_r, res["pars"], tols[-1], n_eta, sep)
    start = np.pad(start, ((0, max(len(ds_r) - len(start), 0)), (0, 0)), constant_values=np.nan)[: len(ds_r)]
    err = np.sqrt(np.diag(res["cov"]))
    log("fitted: " + ", ".join(f"{k} {res['pars'][k]:.8g} +- {e:.2g}" for k, e in zip(free, err)))

    resolved = {"dataset_file": os.path.abspath(source) if is_ds else "",
                "sparsefile": ds["sparsefile"] if is_ds else "", "image": "" if is_ds else os.path.abspath(source),
                "parfile": parfile, "phase": phase, "space_group": sg, "lattice": lpars.tolist(), "detector_mask": mask_file or "", "spatial_correction": spatial_src,
                "n_scans": len(groups) if is_ds else 0, "tth_max": tth_max, "n_rings": len(ds_r),
                **{f"start_{k}": pars0[k] for k in fit.GEOMETRY if k != "lattice"}}  # fmt: skip
    with h5py.File(out, "w") as h:
        write_provenance(h, "rings", None, options, resolved, command)
        g = h.create_group("geometry")
        for k, v in fit.imaged11_pars(res["pars"]).items():
            g.attrs[k] = v
        g.attrs["lattice_scale"] = 1.0
        h["free"] = np.array(free, dtype="S")
        h["cov"] = res["cov"]
        h["tth"], h["used"] = end["tth"], end["used"]
        h["offset_start"], h["offset"], h["intensity"] = start, end["offset"], end["intensity"]
        h["offset"].attrs["eta_edges_deg"] = np.linspace(-180.0, 180.0, n_eta + 1)
        h.create_dataset("image", data=image.astype(np.float32), compression="lzf", chunks=True)
        if centre is not None and dist is not None:  # the searches' curves (anri.calib.plots)
            g = h.create_group("search")
            g["centre_start"] = centre["start"]
            for i, lv in enumerate(centre["levels"]):
                c = g.create_group(f"centre/{i}")
                c.attrs["step"] = lv["step"]
                c["centres"], c["sharpness"] = lv["centres"], lv["sharpness"]
            g["distance"], g["score"] = dist["distance"], dist["score"]
    write_results(out, {**{k: res["pars"][k] for k in free}, **{f"{k}_err": float(e) for k, e in zip(free, err)},
                        "rms_deg": res["rms"], "n_sectors": res["n_sectors"]})  # fmt: skip
    log(f"-> {out}")
    write_log(out, lines)
    return out


def write_pars(calib_file: str, folder: str) -> str:
    """Write a calibration (:func:`stage_fit` or :func:`stage_rings`) as ImageD11 parameter files (``geometry.par``, ``<phase>.par``, ``pars.json``) in folder.

    The phase's cell is the reference lattice, scaled by the fitted lattice scale if it was free. ``pars.json``
    names the calibration file, so that the indexer's histogram stage, given it, records the calibration as its
    source (:func:`anri.io.imaged11.write_pars`). Returns the path of ``pars.json``.
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
    return io.write_pars(folder, geometry, {name: cell}, source=calib_file)


def run_command(path: str) -> str:
    """Return the ``python -m anri.calib`` command that remakes a rings or calibration file, with every option it recorded.

    Parameters
    ----------
    path
        From :func:`stage_rings` or :func:`stage_fit`
    """
    import shlex

    p = read_provenance(path)
    if p["stage"] == "rings":
        r = p["resolved"]
        cmd = ["python", "-m", "anri.calib", "rings", r["dataset_file"] or r["image"], "--out", os.path.abspath(path)]
        flags = {"n_iter": "iter", "detector_mask": "mask"}
    else:
        cmd = ["python", "-m", "anri.calib", "fit", p["source"]]
        if os.path.abspath(path) != stage_path(p["source"], "calib"):
            cmd += ["--out", path]
        flags = {"n_iter": "iter"}
    for k, v in p["options"].items():
        flag = "--" + flags.get(k, k).replace("_", "-")
        if k == "source":  # positional, as resolved
            continue
        if k == "parfile":  # as resolved: an absolute path
            cmd += [flag, p["resolved"]["parfile"]]
        elif k == "search":  # on by default; the command line can only turn it off
            cmd += [] if v else ["--no-search"]
        elif k in ("lattice", "span", "spatial"):  # several values
            cmd += [flag, *(str(x) for x in v)]
        elif isinstance(v, list):
            cmd += [flag, ",".join(str(x) for x in v)]
        else:
            cmd += [flag, str(v)]
    return shlex.join(cmd)


def commands(path: str) -> list[str]:
    """Return the commands that remake a rings or calibration file and every file before it, from the dataset, in order.

    A calibration's chain runs back through the indexer to the parameters it indexed with: when those were written
    by ``python -m anri.calib pars`` (:func:`write_pars`), the indexer's histogram stage records the rings or
    calibration file they came from as its source. So a calibration made from powder rings gives four commands:
    ``rings``, ``pars``, the indexer's ``run`` (:func:`anri.index.stages.run_command`, with ``--outdir`` where its
    files are) and ``fit``.

    Parameters
    ----------
    path
        From :func:`stage_rings` or :func:`stage_fit`
    """
    import shlex

    from anri.index.stages import run_command as index_command

    files = list(reversed(chain(path)))  # the first stage first
    stages = [read_provenance(f)["stage"] for f in files]
    out = []
    for i, (f, stage) in enumerate(zip(files, stages)):
        after = stages[i + 1] if i + 1 < len(files) else None
        if stage in ("rings", "calib"):
            out.append(run_command(f))
            if after == "histogram":  # its parameters, which the indexer read
                parfile = read_provenance(files[i + 1])["resolved"]["parfile"]
                out.append(shlex.join(["python", "-m", "anri.calib", "pars", f, os.path.dirname(parfile)]))
        elif after is None or after in ("rings", "calib"):  # the last file of an indexer's run
            out.append(f"{index_command(f)} --outdir {shlex.quote(os.path.dirname(os.path.abspath(f)))}")
    return out
