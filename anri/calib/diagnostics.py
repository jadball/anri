"""The numbers behind the calibration's choices and results, read from its stage files (no plotting)."""

from __future__ import annotations

import h5py
import numpy as np

from anri.io.provenance import read_provenance


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
