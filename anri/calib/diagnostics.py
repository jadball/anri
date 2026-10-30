"""The numbers behind the calibration's choices and results, from its stage files (no plotting).

For the rings: :func:`read_rings`. For the scan's fit, in the order of its choices: :func:`read_occupancy` and
:func:`occupancy_kept` (``--min-occ``), :func:`match_distances` (``--tols``, and again after the fit), then the
result, :func:`read_calib`, with :func:`residual_trends` (``--clip``) and :func:`strain_left` (``--min-peaks``).
:mod:`anri.calib.plots` draws them.
"""

from __future__ import annotations

import h5py
import numpy as np

from anri.io.provenance import read_provenance

from . import fit
from .stages import select_occupancy


def read_rings(rings_file: str) -> dict:
    """Read a powder-ring calibration (:func:`anri.calib.stages.stage_rings`) for its plots.

    Parameters
    ----------
    rings_file
        From :func:`anri.calib.stages.stage_rings`

    Returns
    -------
    dict
        "start" and "geometry" (the starting and fitted parameters), "free", "err" (each free parameter's
        standard deviation), "tols" (the rounds' tolerances), "tth" [Nr] and "used" [Nr] (the rings, and those
        fitted), "offset_start" and "offset" [Nr, n_eta] (each (ring, eta sector)'s mean 2theta offset from its ring,
        degrees, NaN where empty), "intensity" [Nr, n_eta] and "eta" [n_eta] (sector centres, degrees); with the
        searches, "centre" ("start" and "levels" as :func:`anri.calib.rings.find_centre`) and "distance", "score"
        (:func:`anri.calib.rings.find_distance`)
    """
    p = read_provenance(rings_file)
    out: dict = {"start": {k[len("start_") :]: v for k, v in p["resolved"].items() if k.startswith("start_")},
                 "tols": list(p["options"]["tols"])}  # fmt: skip
    with h5py.File(rings_file, "r") as h:
        out["geometry"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in h["geometry"].attrs.items()}
        out["free"] = [f.decode() for f in h["free"][()]]
        out["err"] = dict(zip(out["free"], np.sqrt(np.diag(h["cov"][()]))))
        for k in ("tth", "used", "offset_start", "offset", "intensity"):
            out[k] = h[k][()]
        edges = h["offset"].attrs["eta_edges_deg"]
        out["eta"] = 0.5 * (edges[1:] + edges[:-1])
        if "search" in h:
            g = h["search"]
            levels = [{"step": float(g[f"centre/{i}"].attrs["step"]), "centres": g[f"centre/{i}/centres"][()],
                       "sharpness": g[f"centre/{i}/sharpness"][()]} for i in range(len(g["centre"]))]  # fmt: skip
            out["centre"] = {"start": g["centre_start"][()], "levels": levels}
            out["distance"], out["score"] = g["distance"][()], g["score"][()]
    return out


def read_occupancy(fit_file: str) -> dict:
    """Read the occupancies of an ``anri.index`` fit stage file, for choosing ``--min-occ``.

    Returns
    -------
    dict
        "f" [NR * NR, K] (each voxel's candidates' occupancies), "nr" (the grid's side) and "log_ratio" (log10 of every
        non-zero occupancy over the largest: real orientations stand apart from a floor of near-zero candidates)
    """
    with h5py.File(fit_file, "r") as h:
        f = h["f"][()]
    return {"f": f, "nr": round(np.sqrt(len(f))), "log_ratio": np.log10(f[f > 0] / f.max())}


def occupancy_kept(occ: dict, min_occ: float) -> dict:
    """Return what ``--min-occ`` keeps of a fit's occupancies, by the stage's own rule (:func:`select_occupancy`).

    Parameters
    ----------
    occ
        From :func:`read_occupancy`
    min_occ
        As :func:`anri.calib.stages.stage_fit`

    Returns
    -------
    dict
        "share" [nr, nr] (the share of each voxel's occupancy kept; NaN where it has none), "pairs_kept" and "pairs"
        ((voxel, orientation) pairs kept, and non-zero), "voxels" (voxels keeping any) and "fraction" (of all occupancy)
    """
    f = occ["f"]
    kept = np.where(select_occupancy(f, min_occ), f, 0.0)
    tot = f.sum(1)
    share = np.where(tot > 0, kept.sum(1) / np.maximum(tot, 1e-30), np.nan).reshape(occ["nr"], occ["nr"])
    return {"share": share, "pairs_kept": int((kept > 0).sum()), "pairs": int((f > 0).sum()),
            "voxels": int(np.any(kept > 0, 1).sum()), "fraction": float(kept.sum() / f.sum())}  # fmt: skip


def match_distances(inputs: dict, ubi: np.ndarray | None = None, pars: dict | None = None) -> dict:
    """Match every peak to the map (:func:`anri.calib.fit.match`) and return how far its (h, k, l) is from integers.

    Real matches pile up near 0 above a broad floor of chance ones: the first of ``--tols`` should take in the pile.

    Parameters
    ----------
    inputs
        From :func:`anri.calib.stages.load_inputs`
    ubi, pars
        The orientations and parameters to match with (default: the map's and the starting geometry; after a fit,
        :func:`read_calib`'s "ubi" and "pars")

    Returns
    -------
    dict
        For the peaks whose beam path crosses a voxel of the map: "err" (distance of (h, k, l) from integers) and
        "intensity"; and "n_peaks" (all peaks)
    """
    pk = inputs["peaks"]
    m = fit.match(pk, inputs["ubi"] if ubi is None else ubi, inputs["grid_o"], inputs["grid_w"], inputs["step"],
                  inputs["pars"] if pars is None else pars)  # fmt: skip
    seen = m["weight"] > 0
    return {"err": m["err"][seen], "intensity": pk["intensity"][seen], "n_peaks": len(seen)}


def read_calib(calib_file: str) -> dict:
    """Read a calibration (:func:`anri.calib.stages.stage_fit`) for its diagnostics and plots.

    Returns
    -------
    dict
        "start", "fitted" and "err" (each free parameter's starting and fitted values and standard deviation),
        "free", "geometry" (ImageD11 parameters, with "lattice_scale"), "pars" (the fit's parameters, to match with,
        :func:`match_distances`), "sig" [3] (the last round's standard deviations in sc, fc and omega), "ubi" [G, 3,
        3], "strain" [G, 3, 3] and "n_peaks" [G] (:func:`anri.calib.fit.strain_left`), "matched" (each matched peak's
        "residual" [P, 3], "sc", "fc", "omega", "dty", ...), "tols", "clip" (the stage's options) and "results"
    """
    p = read_provenance(calib_file)
    with h5py.File(calib_file, "r") as h:
        geometry = {k: (v.item() if hasattr(v, "item") else v) for k, v in h["geometry"].attrs.items()}
        free = [f.decode() for f in h["free"][()]]
        out = {k: h[k][()] for k in ("sig", "ubi", "strain", "n_peaks")}
        out["err"] = dict(zip(free, np.sqrt(np.diag(h["cov"][()]))))
        out["matched"] = {k: h[f"matched/{k}"][()] for k in h["matched"]}
    scale = float(geometry["lattice_scale"])
    out.update(free=free, geometry=geometry, results=p["results"], tols=list(p["options"]["tols"]),
               clip=float(p["options"]["clip"]),
               start={k: float(p["resolved"][f"start_{k}"]) for k in free},
               fitted={k: scale if k == "lattice" else float(geometry[k]) for k in free},
               pars={**fit.base_pars(geometry, float(p["resolved"]["y0"])), "lattice": scale})  # fmt: skip
    return out


def residual_trends(calib: dict, n_bins: int = 40) -> dict:
    """Return each matched peak's residual in standard deviations, against omega, detector azimuth and dty.

    With the right geometry the medians are flat at zero. A sinusoid against azimuth is the centre or the tilts (or,
    with them free, detector distortion); a trend against omega something fixed that turns with the sample (wedge,
    chi); a trend against dty y0 or misplaced grains.

    Parameters
    ----------
    calib
        From :func:`read_calib`
    n_bins
        Bins of each axis for the medians

    Returns
    -------
    dict
        "r" [P, 3] (sc, fc, omega residual / sig), "rms" [3], "against" ({"omega", "azimuth", "dty"}: [P] each,
        degrees for the angles), "centres" and "medians" (per axis, [n_bins] and [n_bins, 3]; NaN for empty bins)
    """
    mt, g = calib["matched"], calib["geometry"]
    r = mt["residual"] / calib["sig"]
    azimuth = np.degrees(np.arctan2(mt["sc"] - g["z_center"], mt["fc"] - g["y_center"]))  # about the fitted centre
    against = {"omega": mt["omega"], "azimuth": azimuth, "dty": mt["dty"]}
    centres, medians = {}, {}
    for name, x in against.items():
        edges = np.linspace(x.min(), x.max(), n_bins + 1)
        b = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
        centres[name] = 0.5 * (edges[:-1] + edges[1:])
        medians[name] = np.array([np.median(r[b == k], 0) if np.any(b == k) else np.full(3, np.nan)
                                  for k in range(n_bins)])  # fmt: skip
    return {"r": r, "rms": np.sqrt(np.mean(r**2, 0)), "against": against, "centres": centres, "medians": medians}


def strain_left(calib: dict) -> dict:
    """Return the strain a calibration leaves, split as :func:`anri.calib.fit.split_strain`, with "n_peaks".

    On a strain-free calibrant a right geometry leaves only noise: a deviatoric mean away from zero points to the
    tilts or the centre, a hydrostatic one to the wavelength or distance against the reference lattice.
    """
    return {**fit.split_strain(calib["strain"], calib["n_peaks"]), "n_peaks": np.asarray(calib["n_peaks"])}
