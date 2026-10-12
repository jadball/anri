"""Simulating a scanning-3DXRD dataset as three stages, each writing one HDF5 file with its provenance.

1. :func:`stage_prepare`: a map (an ImageD11 TensorMap, any number of phases, or the indexer's entries) and the
   instrument become everything the renderer needs: each phase's map entries, reflections and structure factors, the
   beam's spreads and the motor positions of the scan.
2. :func:`stage_render`: rows of the scan rendered into sparse pixels above the segmentation cut, as ImageD11 scan
   groups. The rows are independent, so they can be split between jobs (e.g. a SLURM array), one file each.
3. :func:`stage_assemble`: the rendered rows gathered into an ImageD11 dataset (sparse file, ``pars/`` and DataSet),
   which ``python -m anri.index`` and ImageD11 read.

Each file's provenance (:mod:`anri.io`) records its options, the values worked out, and the file it was made from;
the ImageD11 files stay as ImageD11 writes them, and the assemble file records how they were made.
"""

from __future__ import annotations

import glob
import os
import re
import time
from collections.abc import Callable

import h5py
import numpy as np

from anri.io import read_attrs, read_provenance, stage_path, write_attrs, write_provenance, write_results

# ID11's focusing optics: the standard deviation of the beam's convergence at the sample (rad, horizontal and
# vertical), from effective aperture / focal length (Si planar nanolenses: ~38 um / 10 cm; Al CRLs: ~115 um / 50 cm)
OPTICS = {"al-crl": 8e-5, "si-nanolens": 1.5e-4}
BANDWIDTH = 2e-3  # FWHM of d(lambda) / lambda: ID11's double Laue Si(111) monochromator
EIGER_4M = (2162, 2068)
_LATTICE_KEYS = ("cell__a", "cell__b", "cell__c", "cell_alpha", "cell_beta", "cell_gamma")
_SCAN = re.compile(r"^\d+\.1$")


def read_map(map_file: str) -> tuple[list[dict], float]:
    """Read a map as one dict per phase: "name", "entries" (:func:`anri.fwd.render_row`), "lattice", "space_group".

    Parameters
    ----------
    map_file
        An ImageD11 TensorMap (``.h5``; every phase in it), or the entries the indexer writes (``_entries.npz``, one
        phase, with its voxel size, lattice, space group and name)

    Returns
    -------
    phases, voxel_size
    """
    if map_file.endswith(".npz"):
        z = np.load(map_file)
        missing = [k for k in ("ubi", "pos", "density", "voxel_size", "lattice", "space_group", "phase") if k not in z]
        if missing:
            msg = (
                f"{map_file} lacks {missing}: give a TensorMap, or entries written by python -m anri.index populations"
            )
            raise ValueError(msg)
        entries = {k: np.asarray(z[k], float) for k in ("ubi", "pos", "density", "sig_rot") if k in z}
        phase = {"name": str(z["phase"]), "entries": entries, "lattice": [float(v) for v in z["lattice"]],
                 "space_group": int(z["space_group"])}  # fmt: skip
        return [phase], float(z["voxel_size"])
    from ImageD11.sinograms.tensor_map import TensorMap

    from anri.io import entries_from_tensormap

    tmap = TensorMap.from_h5(map_file)
    phases = []
    for pid, cell in sorted(tmap.phases.items()):
        entries = entries_from_tensormap(tmap, pid)
        if len(entries["ubi"]) == 0:
            continue
        try:
            sg = int(cell.symmetry)
        except (TypeError, ValueError):
            msg = f"phase {cell.name} of {map_file} has symmetry {cell.symmetry!r}: a space-group number is needed"
            raise ValueError(msg) from None
        phases.append({"name": str(cell.name), "entries": entries,
                       "lattice": [float(v) for v in cell.lattice_parameters], "space_group": sg})  # fmt: skip
    return phases, float(tmap.steps[1])


def _cif_of(cif: str | dict | None, name: str, n_phases: int) -> str | None:
    """Return the CIF for a phase: from {name: path}, or the one path given for a single phase."""
    if cif is None:
        return None
    if isinstance(cif, str):
        if n_phases > 1:
            msg = "several phases: give one CIF per phase, as {name: path} (--cif NAME=PATH)"
            raise ValueError(msg)
        return cif
    return cif.get(name)


def _tth_max(pars: dict, det_shape: tuple[int, int], margin_px: float = 10.0) -> float:
    """Return the largest 2theta whose ring lies wholly on the detector (approximately: no tilts), less a margin."""
    ns, nf = det_shape
    r = min(min(pars["y_center"], nf - pars["y_center"]) * pars["y_size"],
            min(pars["z_center"], ns - pars["z_center"]) * pars["z_size"])  # fmt: skip
    return float(np.degrees(np.arctan((r - margin_px * max(pars["y_size"], pars["z_size"])) / pars["distance"])))


# ------------------------------------------------------------------------------------------------------- 1. prepare
def stage_prepare(
    map_file: str, pars_file: str, out: str, *, cif: str | dict | None = None, optics: str = "al-crl",
    divergence: float | None = None, bandwidth: float = BANDWIDTH, beam_fwhm: float | None = None, psf: float = 0.5,
    pol_factor: float = 1.0, y0: float = 0.0, omega_range: tuple = (0.0, 180.0), omega_step: float = 0.05,
    dty_step: float | None = None, margin: float = 3.0, density: float = 1.0, det_shape: tuple = EIGER_4M,
    tth_max: float | None = None, command: str | None = None, log: Callable = print,
) -> str:  # fmt: skip
    """Turn a map and an instrument into everything the renderer needs, without rendering.

    Parameters
    ----------
    map_file
        See :func:`read_map`
    pars_file
        ImageD11 geometry: a ``.par`` file, or a ``pars.json``
    out
        Output file
    cif
        CIF for the structure factors: one path for a single phase, or {phase name: path}; phases without one get
        ``|F|^2`` = 1
    optics
        ID11's focusing optics, which set the beam's convergence: "al-crl" or "si-nanolens" (see :data:`OPTICS`)
    divergence
        The convergence's standard deviation (rad, horizontal and vertical), instead of the optics'
    bandwidth
        FWHM of d(lambda) / lambda (default: ID11's Laue monochromator)
    beam_fwhm
        FWHM of the beam across dty (default: the dty step)
    psf
        Standard deviation of the detector's point spread (pixels)
    pol_factor
        Degree of horizontal polarisation
    y0
        dty where the rotation axis is in the beam
    omega_range, omega_step
        The rotation (degrees) and the frame width
    dty_step
        Between rows (default: the map's voxel size). The rows cover the sample and margin x beam_fwhm either side
    density
        Counts scale: each entry's density is multiplied by this
    det_shape
        (n_slow, n_fast) pixels (default: an Eiger 4M)
    tth_max
        Largest 2theta of the reflections (default: the largest ring wholly on the detector)
    command, log
        The command line, if run from one, and where progress goes

    Returns
    -------
    str
        The file written
    """
    import Dans_Diffraction

    from anri.crystal import reflections, structure_factors
    from anri.io import motor_grid, read_par, read_pars_json
    from anri.maths.utils import fwhm_to_sigma

    options = {"map": os.path.abspath(map_file), "pars": os.path.abspath(pars_file), "optics": optics,
               "divergence": divergence, "bandwidth": bandwidth, "beam_fwhm": beam_fwhm, "psf": psf,
               "pol_factor": pol_factor, "y0": y0, "omega_range": list(omega_range), "omega_step": omega_step,
               "dty_step": dty_step, "margin": margin, "density": density, "det_shape": list(det_shape),
               "tth_max": tth_max}  # fmt: skip
    if isinstance(cif, str):
        options["cif"] = os.path.abspath(cif)
    elif cif:
        options["cif"] = [f"{k}={os.path.abspath(v)}" for k, v in cif.items()]
    if divergence is None and optics not in OPTICS:
        msg = f"optics must be one of {sorted(OPTICS)}, not {optics!r} (or give divergence)"
        raise ValueError(msg)
    pars = read_pars_json(pars_file)[0] if pars_file.endswith(".json") else read_par(pars_file)
    wl = float(pars["wavelength"])
    phases, voxel = read_map(map_file)
    ystep = voxel if dty_step is None else dty_step
    fwhm = ystep if beam_fwhm is None else beam_fwhm
    div = OPTICS[optics] if divergence is None else divergence
    tth = _tth_max(pars, det_shape) if tth_max is None else tth_max
    dsmax = 2 * np.sin(np.radians(tth / 2)) / wl
    radius = max(float(np.linalg.norm(p["entries"]["pos"][:, :2], axis=1).max()) for p in phases)
    n_side = int(np.ceil((radius + margin * fwhm) / ystep))
    omega, dty = motor_grid(tuple(omega_range), omega_step, (y0 - n_side * ystep, y0 + n_side * ystep), ystep)
    log(f"map {map_file}: {len(phases)} phase(s), voxel {voxel:g}, sample radius {radius:.4g}; scan {omega.shape[0]} "
        f"rows of {ystep:g} x {omega.shape[1]} frames of {omega_step:g} deg; beam FWHM {fwhm:g}, convergence {div:g} "
        f"rad ({optics if divergence is None else 'given'}), bandwidth {bandwidth:g} FWHM; 2theta up to {tth:.2f} "
        f"deg")  # fmt: skip

    resolved = {"wavelength": wl, "voxel_size": voxel, "dty_step": ystep, "beam_fwhm": fwhm, "divergence": div,
                "sig_wavelength": wl * fwhm_to_sigma(bandwidth), "sig_beam": fwhm_to_sigma(fwhm), "tth_max": tth,
                "n_rows": omega.shape[0], "n_frames": omega.shape[1], "sample_radius": radius,
                "phases": [p["name"] for p in phases]}  # fmt: skip
    with h5py.File(out, "w") as h:
        write_provenance(h, "prepare", None, options, resolved, command)
        write_attrs(h.create_group("pars"), pars)
        h["scan/omega"], h["scan/dty"] = omega, dty
        for p in phases:
            path = _cif_of(cif, p["name"], len(phases))
            hkls = reflections(p["lattice"], p["space_group"], wl, dsmax)["hkl"].astype(float)
            F2 = np.ones(len(hkls)) if path is None else structure_factors(Dans_Diffraction.Crystal(path), hkls, wl)
            g = h.create_group(f"phases/{p['name']}")
            write_attrs(g, {"lattice": p["lattice"], "space_group": p["space_group"], "cif": path})
            for k, v in p["entries"].items():
                g.create_dataset(k, data=v * density if k == "density" else v, compression="lzf")
            g["hkls"], g["F2"] = hkls, F2
            log(f"phase {p['name']}: {len(p['entries']['ubi'])} entries, {len(hkls)} reflections, "
                + ("|F|^2 = 1" if path is None else f"|F|^2 from {path}"))  # fmt: skip
    log(f"-> {out}")
    return out


def load_prepare(path: str) -> dict:
    """Read a prepare stage file back, ready to render.

    Returns
    -------
    dict
        "phases" [(entries, hkls, F2), ...], "names", "cells" (ImageD11 lattice parameters per phase), "geom"
        (:func:`anri.io.geom_from_pars`), "omega", "dty", "det_shape", "pars", "prov"
    """
    from anri.io import geom_from_pars

    prov = read_provenance(path)
    o, r = prov["options"], prov["resolved"]
    with h5py.File(path, "r") as h:
        pars = read_attrs(h["pars"])
        omega, dty = h["scan/omega"][()], h["scan/dty"][()]
        phases, cells = [], {}
        for name in r["phases"]:
            g = h[f"phases/{name}"]
            a = read_attrs(g)
            entries = {k: g[k][()] for k in ("ubi", "pos", "density", "sig_rot") if k in g}
            phases.append((entries, g["hkls"][()], g["F2"][()]))
            cells[name] = {**dict(zip(_LATTICE_KEYS, a["lattice"])), "cell_lattice_[P,A,B,C,I,F,R]": a["space_group"]}
    geom = geom_from_pars(pars, o["y0"], r["sig_wavelength"], r["divergence"], r["divergence"], r["sig_beam"],
                          r["voxel_size"], pol_factor=o["pol_factor"], sig_psf=o["psf"])  # fmt: skip
    return {"phases": phases, "names": r["phases"], "cells": cells, "geom": geom, "omega": omega, "dty": dty,
            "det_shape": tuple(o["det_shape"]), "pars": pars, "prov": prov}  # fmt: skip


# -------------------------------------------------------------------------------------------------------- 2. render
def parse_rows(rows: str | None, n_rows: int) -> range:
    """Rows "a:b" (Python slice, either end optional) of a scan of n_rows, as a range; None: every row."""
    if rows is None:
        return range(n_rows)
    a, _, b = rows.partition(":")
    return range(n_rows)[slice(int(a) if a else None, int(b) if b else None)]


def stage_render(
    prepare_file: str, out: str | None = None, *, rows: str | None = None, cut: float = 1.0,
    window: tuple = (3, 7, 7), max_frames: int | None = 31, batch: int = 2**16, command: str | None = None,
    log: Callable = print,
) -> str:  # fmt: skip
    """Render rows of the scan into sparse pixels, every phase summed, as ImageD11 scan groups.

    Parameters
    ----------
    prepare_file
        From :func:`stage_prepare`
    out
        Output file (default: the prepare file's name with ``_render``, and ``_<first>-<last>`` for some rows only)
    rows
        "a:b", the rows to render (a Python slice; default every row)
    cut
        Keep counts strictly above this, as ImageD11's segmentation does
    window, max_frames, batch
        See :func:`anri.fwd.render_row`

    Returns
    -------
    str
        The file written: a scan group "<row + 1>.1" per row rendered, and per-row "stats"
    """
    from anri.io import write_rendered_rows

    options = {"rows": rows, "cut": cut, "window": list(window), "max_frames": max_frames, "batch": batch}
    p = load_prepare(prepare_file)
    todo = parse_rows(rows, p["omega"].shape[0])
    if len(todo) == 0:
        msg = f"rows {rows!r}: none of the {p['omega'].shape[0]} rows"
        raise ValueError(msg)
    if out is None:
        out = stage_path(prepare_file, "render")
        if rows is not None:
            out = f"{out[:-3]}_{todo[0]:04d}-{todo[-1]:04d}.h5"
    log(f"rendering rows {todo[0]}..{todo[-1]} of {p['omega'].shape[0]}, {len(p['phases'])} phase(s)")
    t0 = time.perf_counter()
    with h5py.File(out, "w") as h:
        write_provenance(h, "render", prepare_file, options, {"first_row": todo[0], "last_row": todo[-1]}, command)
        stats = write_rendered_rows(h, p["phases"], p["geom"], p["omega"], p["dty"], p["det_shape"], rows=todo,
                                    cut=cut, log=log, window=tuple(window), max_frames=max_frames, batch=batch)  # fmt: skip
        idx = np.asarray(todo)
        h["stats/rows"] = idx
        for k, v in stats.items():
            h[f"stats/{k}"] = v[idx]
    write_results(out, {"peaks": int(stats["n_peaks"][idx].sum()), "pixels": int(stats["n_pixels"][idx].sum()),
                        "window_capture_min": float(stats["captured_min"][idx].min()),
                        "seconds": time.perf_counter() - t0})  # fmt: skip
    log(f"-> {out} ({time.perf_counter() - t0:.0f} s)")
    return out


# ------------------------------------------------------------------------------------------------------ 3. assemble
def render_parts(prepare_file: str) -> list[str]:
    """Return the render stage files beside a prepare file that were made from it (by id)."""
    pid = read_provenance(prepare_file)["id"]
    base = stage_path(prepare_file, "render")[:-3]
    found = []
    for f in sorted(glob.glob(f"{base}*.h5")):
        prov = read_provenance(f)
        if prov["stage"] == "render" and prov.get("source_id") == pid:
            found.append(f)
    return found


def stage_assemble(
    prepare_file: str, analysisroot: str, sample: str, dataset: str, out: str | None = None, *,
    parts: list | None = None, peaks_table: bool = False, command: str | None = None, log: Callable = print,
) -> str:  # fmt: skip
    """Gather rendered rows into an ImageD11 dataset: the sparse file, ``pars/`` and the DataSet (needs ImageD11).

    Writes ``{analysisroot}/{sample}/{sample}_{dataset}/{sample}_{dataset}_sparse.h5`` and ``_dataset.h5`` (refusing
    to overwrite a sparse file), and ``{analysisroot}/pars/`` (geometry, one ``.par`` per phase, ``pars.json``).

    Parameters
    ----------
    prepare_file
        From :func:`stage_prepare`
    analysisroot, sample, dataset
        Where the ImageD11 dataset goes
    out
        This stage's own file, the provenance of the dataset (default: the prepare file's name with ``_assemble``)
    parts
        Render stage files (default: every one beside the prepare file made from it); together they must hold every
        row once
    peaks_table
        Also write zero distortion maps and ImageD11's peaks table (2D peaks segmented from the pixels)

    Returns
    -------
    str
        This stage's file
    """
    from anri.io import write_dataset, write_pars, write_peaks_table, write_zero_distortion

    analysisroot = os.path.abspath(analysisroot)
    options = {"analysisroot": analysisroot, "sample": sample, "dataset": dataset,
               "peaks_table": peaks_table}  # fmt: skip
    p = load_prepare(prepare_file)
    pid, n_rows = p["prov"]["id"], p["omega"].shape[0]
    parts = render_parts(prepare_file) if parts is None else list(parts)
    seen = np.zeros(n_rows, int)
    for f in parts:
        prov = read_provenance(f)
        if prov.get("source_id") != pid:
            msg = f"{f} was not rendered from {prepare_file}"
            raise ValueError(msg)
        with h5py.File(f, "r") as h:
            seen[h["stats/rows"][()]] += 1
    if (seen != 1).any():
        msg = (f"the parts {parts} hold rows {np.flatnonzero(seen == 0).tolist()} not at all and "
               f"{np.flatnonzero(seen > 1).tolist()} more than once")  # fmt: skip
        raise ValueError(msg)
    folder = os.path.join(analysisroot, sample, f"{sample}_{dataset}")
    os.makedirs(folder, exist_ok=True)
    sparse = os.path.join(folder, f"{sample}_{dataset}_sparse.h5")
    n_pixels = 0
    with h5py.File(sparse, "w-") as hout:
        for f in parts:
            with h5py.File(f, "r") as h:
                for name in h:
                    if _SCAN.match(name):
                        h.copy(name, hout)
                n_pixels += int(h["stats/n_pixels"][()].sum())
    pars_dir = os.path.join(analysisroot, "pars")
    parfile = write_pars(pars_dir, p["pars"], p["cells"])
    e2dx = e2dy = None
    if peaks_table:
        e2dx, e2dy = write_zero_distortion(pars_dir, p["det_shape"])
    dsfile = write_dataset(sparse, analysisroot, sample, dataset, y0=p["prov"]["options"]["y0"], parfile=parfile,
                           e2dxfile=e2dx, e2dyfile=e2dy)  # fmt: skip
    log(f"{len(parts)} part(s), {n_rows} rows, {n_pixels} pixels -> {sparse}, {dsfile}, {parfile}")
    resolved = {"sparse_file": sparse, "dataset_file": dsfile, "parfile": parfile,
                "parts": [os.path.abspath(f) for f in parts], "part_ids": [read_provenance(f)["id"] for f in parts]}  # fmt: skip
    if peaks_table:
        resolved["peaks_table"] = write_peaks_table(dsfile)
    out = stage_path(prepare_file, "assemble", out)
    with h5py.File(out, "w") as h:
        write_provenance(h, "assemble", prepare_file, options, resolved, command)
    write_results(out, {"rows": n_rows, "pixels": n_pixels})
    log(f"-> {out}")
    return out
