"""The numbers behind each choice of the indexer's options, for notebooks, workflow widgets or logs.

Pure functions of a dataset (:func:`anri.index.open_dataset`) or of the stages' outputs; they draw nothing. The plots
of :mod:`anri.index.plots` draw them, and return functions that redraw for other values of an option.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from .data import pixel_cake, ring_table, ring_widths
from .predict import GRID_STEPS, grid_chance, prune
from .stages import B_E, B_O, R_E, R_O, sample_groups


def ring_profile_cake(dset: dict, n_rings: int = 10, cif: str | None = None, step: float = 0.002,
                      n_eta: int = 180) -> dict:  # fmt: skip
    """Measure the 2theta profile and the (2theta, eta) cake on the scans the histogram stage measures ring widths on.

    Parameters
    ----------
    dset
        From :func:`anri.index.open_dataset`
    n_rings
        Rings shown, to choose the number used from
    cif
        CIF of the phase, for the structure factors listed (default: ``|F|^2 = 1``)
    step
        2theta bin width (degrees), as :func:`anri.index.ring_profile`
    n_eta
        eta bins over 360 degrees

    Returns
    -------
    dict
        "rings" (:func:`anri.index.ring_table` of n_rings), "tth" [n_t] bin centres, "profile" [n_t], "cake" [n_t,
        n_eta], "lo", "step", "hkl", "multiplicity", "F2" (per ring: a representative hkl, the number of reflections,
        their mean ``|F|^2``), "gap" (to the next ring) and "n_scans"
    """
    structure = None
    if cif:
        import Dans_Diffraction

        structure = Dans_Diffraction.Crystal(cif)
    rings = ring_table(dset["lattice"], dset["space_group"], dset["geo"]["wavelength"], n_rings, structure)
    lo = float(rings["tth"][0]) - 0.5
    n_t = int(np.ceil((float(rings["tth"][-1]) + 0.5 - lo) / step))
    scans = sample_groups(dset["groups"])
    C = pixel_cake(dset["stream"](scans), dset["geom"], lo, step, n_t, dset["chunk"], n_eta)
    ring_of_h = rings["ring_j"][::2]
    n = len(rings["tth"])
    hkl = [tuple(int(v) for v in sorted(np.abs(rings["hkls"][ring_of_h == r][0]), reverse=True)) for r in range(n)]
    return {"rings": rings, "tth": lo + (np.arange(n_t) + 0.5) * step, "profile": C.sum(1), "cake": C, "lo": lo,
            "step": step, "hkl": hkl, "multiplicity": [int(np.sum(ring_of_h == r)) for r in range(n)],
            "F2": [float(rings["F2"][ring_of_h == r].mean()) for r in range(n)], "gap": np.diff(rings["tth"]),
            "n_scans": len(scans)}  # fmt: skip


def ring_windows(cake: dict, n_rings: int, tth_tol: float | None = None) -> dict:
    """Return the 2theta windows of the first n_rings rings, as the histogram stage sets them.

    Parameters
    ----------
    cake
        From :func:`ring_profile_cake`
    n_rings
        Rings used
    tth_tol
        Tolerance (degrees), or None: measured per ring, ``|offset| + half-width``

    Returns
    -------
    dict
        "offset", "half_width", "tolerance", "counts" (in each window), "overlap" (each window meets the next) and
        "fraction" (of the intensity up to the last ring used that falls in a window)
    """
    tth, prof, x = cake["rings"]["tth"][:n_rings], cake["profile"], cake["tth"]
    off, hw = ring_widths(prof, cake["lo"], cake["step"], tth)
    tol = np.abs(off) + hw if tth_tol is None else np.full(n_rings, tth_tol)
    inside = np.zeros(len(x), bool)
    counts = []
    for t, w in zip(tth, tol):
        win = np.abs(x - t) < w
        inside |= win
        counts.append(float(prof[win].sum()))
    gap = cake["gap"]
    overlap = [r < len(gap) and r + 1 < n_rings and tol[r] + tol[r + 1] > gap[r] for r in range(n_rings)]
    used = x < tth[-1] + 0.5
    return {"offset": off, "half_width": hw, "tolerance": tol, "counts": counts, "overlap": overlap,
            "fraction": float(prof[inside].sum() / max(prof[used].sum(), 1e-30))}  # fmt: skip


def _h4(H: np.ndarray, n_rings: int, scan: dict) -> np.ndarray:
    """Return the fit histogram as [ring, eta, omega, row] on the host."""
    return np.asarray(H).reshape(n_rings, scan["n_e"] // R_E, scan["n_o"] // R_O, scan["n_rows"])


def _centres(scan: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the dty row centres and the omega bin centres of the fit histogram."""
    dty_c = scan["dty0"] + np.arange(scan["n_rows"]) * scan["ystep"]
    om_c = scan["om0"] + (np.arange(scan["n_o"] // R_O) + 0.5) * B_O * R_O
    return dty_c, om_c


def row_profile(H: np.ndarray, n_rings: int, scan: dict) -> dict:
    """Return the intensity per dty row and per omega bin, and each row against the mean of its neighbours.

    Steps from row to row mean the flux changed between the rows' scans (normalise with a monitor); a slow drift along
    omega is beam decay; sharp spikes are single strong spots.

    Returns
    -------
    dict
        "dty" [n_rows] row centres, "row" [n_rows] intensity, "omega" [n_o] bin centres, "per_omega" [n_o],
        "jump" [n_rows] row / mean of its neighbours (NaN at the ends and in rows below 5% of the brightest)
    """
    H4 = _h4(H, n_rings, scan)
    row, om = H4.sum((0, 1, 2)), H4.sum((0, 1, 3))
    nb = np.convolve(row, [0.5, 0.0, 0.5], "same")
    lit = row > 0.05 * row.max()
    lit[[0, -1]] = False
    dty_c, om_c = _centres(scan)
    return {"dty": dty_c, "row": row, "omega": om_c, "per_omega": om,
            "jump": np.where(lit, row / np.maximum(nb, 1e-30), np.nan)}  # fmt: skip


def y0_from_edges(H: np.ndarray, n_rings: int, scan: dict, frac: float = 0.05) -> dict:
    """Fit y0 to the middle of the sample's projection, between its edges, at every omega.

    A point fixed in the sample traces y0 + b sin(omega) + c cos(omega) in the sinogram. The middle between the first
    and last rows above frac x the projection's maximum (interpolated between rows) is much steadier than the
    intensity-weighted mean dty, which jumps as grains light up and go dark. Both need the whole sample within the
    scan.

    Returns
    -------
    dict
        "sinogram" [n_o, n_rows] (all rings and eta), "omega", "dty", "middle" and "mean" [n_o] (NaN where the
        projection is below 5% of the brightest), "y0" (from the middles), "y0_mean" (from the means), "fit" [n_o]
        (the fitted curve) and "edge" (the brighter end row's intensity / the brightest row's; above ~0.1 the sample
        may extend past the scan)
    """
    sino = _h4(H, n_rings, scan).sum((0, 1))  # [omega, row]
    dty_c, om_c = _centres(scan)
    w = sino.sum(1)
    good = w > 0.05 * w.max()
    step, nk = scan["ystep"], scan["n_rows"]

    def middle(p: np.ndarray) -> float:
        thr = frac * p.max()
        k = np.flatnonzero(p > thr)
        i0, i1 = k[0], k[-1]
        lo = dty_c[i0] - step * (p[i0] - thr) / max(p[i0] - (p[i0 - 1] if i0 > 0 else 0.0), 1e-30)
        hi = dty_c[i1] + step * (p[i1] - thr) / max(p[i1] - (p[i1 + 1] if i1 < nk - 1 else 0.0), 1e-30)
        return 0.5 * (lo + hi)

    mids = np.array([middle(p) if g else np.nan for p, g in zip(sino, good)])
    mean = np.where(good, sino @ dty_c / np.maximum(w, 1e-30), np.nan)
    A = np.stack([np.ones(len(om_c)), np.sin(np.radians(om_c)), np.cos(np.radians(om_c))], 1)
    coef = np.linalg.lstsq(A[good], mids[good], rcond=None)[0]
    coef_mean = np.linalg.lstsq(A[good], mean[good], rcond=None)[0]
    row = sino.sum(0)
    return {"sinogram": sino, "omega": om_c, "dty": dty_c, "middle": mids, "mean": mean, "y0": float(coef[0]),
            "y0_mean": float(coef_mean[0]), "fit": A @ coef, "edge": float(max(row[0], row[-1]) / row.max())}  # fmt: skip


def lit_curve(H_lit: np.ndarray, n_rings: int, scan: dict, cuts: np.ndarray | None = None) -> dict:
    """Return how much of the lit histogram is lit, and how much intensity it holds, as the lit cut rises.

    A good cut is near the elbow, where the lit area falls fast and the intensity held slowly: the sparser the lit
    map, the fewer wrong orientations match by chance; but a high cut drops weak spots (small grains).

    Returns
    -------
    dict
        "median" (the median non-empty bin), "cuts" (x the median), "area" and "held" (fractions at each cut),
        "per_ring" [n_rings, n_e, n_o] the histogram
    """
    Hs = np.asarray(H_lit).reshape(n_rings, scan["n_e"], scan["n_o"])
    vals = np.sort(Hs.ravel())
    med = float(np.median(vals[vals > 0]))
    cum = np.concatenate([[0.0], np.cumsum(vals, dtype=np.float64)])
    cuts = np.logspace(-1, 2, 61) if cuts is None else np.asarray(cuts)
    idx = np.searchsorted(vals, cuts * med, side="right")
    return {"median": med, "cuts": cuts, "area": 1 - idx / vals.size, "held": 1 - cum[idx] / cum[-1], "per_ring": Hs}


def eta_profile(H_lit: np.ndarray, n_rings: int, scan: dict) -> dict:
    """Return the intensity per eta bin, all rings: what an eta cut removes.

    Reflections at ``|sin eta|`` <= etacut are not used: near eta = 0 and 180 degrees a small rotation moves a spot by
    up to 1 / ``|sin eta|`` times as far in omega.

    Returns
    -------
    dict
        "eta" [n_e] bin centres, "profile" [n_e]
    """
    Hs = np.asarray(H_lit).reshape(n_rings, scan["n_e"], scan["n_o"])
    return {"eta": -180.0 + (np.arange(scan["n_e"]) + 0.5) * B_E, "profile": Hs.sum((0, 2))}


def grid_chances(d: dict, lit: dict, steps: tuple = GRID_STEPS, max_orientations: float = 4e6, seed: int = 0,
                 log: Callable = print) -> dict:  # fmt: skip
    """Return the chance completeness of every grid step, drawn as :func:`anri.index.choose_grid` draws it.

    Parameters
    ----------
    d
        From :func:`anri.index.load_histogram`
    lit
        From :func:`anri.index.lit_map`
    steps
        Grid steps (degrees), coarsest first
    max_orientations
        Skip steps with more orientations than this (estimated from the coarsest)
    seed
        As choose_grid

    Returns
    -------
    dict
        step: (chance completeness, orientations, worst-case distance), for the steps not skipped
    """
    rng = np.random.default_rng(seed)
    out = {}
    for s in steps:
        if out and out[steps[0]][1] * (steps[0] / s) ** 3 > max_orientations:
            log(f"grid {s} deg: skipped, ~{out[steps[0]][1] * (steps[0] / s) ** 3 / 1e6:.0f}M orientations")
            continue
        out[s] = grid_chance(s, d["ops"], d["B"], d["rings"], d["geom"], lit, rng)
        log(f"grid {s} deg: {out[s][1]} orientations, up to {out[s][2]:.2f} deg from the truth; chance completeness "
            f"{out[s][0]:.2f}")  # fmt: skip
    return out


def grid_deviances(d: dict, lit: dict, steps: tuple, etacut: float, min_lr: float = 25.0, keep: int = 100000,
                   censor: float = 0.0, log: Callable = print) -> dict:  # fmt: skip
    """Fit the orientations above chance of each grid step to the row-summed data: how fine a grid the fit needs.

    The deviance falls as the grid gets finer while the grid limits the fit, and flattens where the data no longer
    tell finer grids apart. About one orientation fit (:func:`anri.index.orientation_mlem`) per step.

    Returns
    -------
    dict
        step: (orientations fitted, deviance, orientations with likelihood ratio above min_lr)
    """
    import anri.crystal

    from .occupancy import deviance
    from .orientations import orientation_mlem

    sc = d["scan"]
    data = d["H"].reshape(-1, sc["n_rows"]).sum(1)
    bins = (B_E * R_E, B_O * R_O, sc["n_e"] // R_E, sc["n_o"] // R_O, sc["om0"])
    out = {}
    for s in steps:
        U, delta = anri.crystal.orientation_grid(s, d["ops"])
        _, comp, info = prune(U, delta, d["B"], d["rings"], d["geom"], lit, None, keep)
        fitted = np.flatnonzero(comp > info["chance"])
        _, lr, mu = orientation_mlem(data, U[fitted], d["B"], d["rings"], d["geom"], bins, etacut,
                                     log=lambda m: None, censor=censor, return_model=True)  # fmt: skip
        out[s] = (len(fitted), float(deviance(data, jnp.asarray(mu), censor)), int(np.sum(lr > min_lr)))
        log(f"grid {s} deg: {out[s][0]} orientations above chance; deviance {out[s][1]:.5g}; {out[s][2]} with "
            f"likelihood ratio > {min_lr:g}")  # fmt: skip
    return out


def nearest_stronger(U: np.ndarray, lr: np.ndarray, ops: np.ndarray, top: int = 5000) -> dict:
    """Return each of the strongest orientations' misorientation to the nearest one with a higher likelihood ratio.

    A grain shows as one strong orientation far from anything stronger (a peak), with a halo of weaker ones a grid
    step or two from it.

    Parameters
    ----------
    U, lr
        Orientations and their likelihood ratios (a filter stage file's fitted ones)
    ops
        Laue-group rotations
    top
        How many of the strongest

    Returns
    -------
    dict
        "order" [top] indices into U, strongest first; "distance" [top] degrees (180 for the strongest)
    """
    from anri.maths.rotation import mat_to_quat

    order = np.argsort(lr)[::-1][: min(len(lr), top)]
    q = mat_to_quat(U[order]).astype(np.float32)  # [M, 4]
    qs = mat_to_quat(U[order][:, None] @ ops[None]).astype(np.float32)  # [M, S, 4]
    dist = np.full(len(order), 180.0)
    for s0 in range(0, len(order), 256):
        c = np.abs(np.einsum("ik,jsk->ijs", q[s0 : s0 + 256], qs)).max(2)  # cos(half the misorientation)
        c[np.arange(c.shape[0])[:, None] <= np.arange(len(order))[None] - s0] = -1.0  # only stronger ones
        dist[s0 : s0 + 256] = np.degrees(2 * np.arccos(np.clip(c.max(1), -1.0, 1.0)))
    dist[0] = 180.0
    return {"order": order, "distance": dist}


def convergence_recorder(n_iter: int, snapshots: tuple = (1, 2, 3, 5, 10, 20, 50, 100),
                         log: Callable = print) -> tuple[Callable, dict]:  # fmt: skip
    """Return a callback for :func:`anri.index.stage_fit` that records how the occupancy fit converges, and its record.

    Returns
    -------
    callback, record
        Pass callback to the fit. record then holds "deviance" (per iteration) and "moved" (iteration: the share of
        each fitted voxel's occupancy that moved between orientations since the previous snapshot, per iteration;
        padded past the voxels fitted, which :func:`anri.index.plots.plot_convergence` cuts off)
    """
    snap = sorted({i for i in snapshots if i < n_iter} | {n_iter})
    record: dict = {"deviance": [], "moved": {}}
    last: dict = {"f": None, "it": 0}

    def callback(it: int, f: object, dev: float) -> None:
        record["deviance"].append(float(dev))
        if it in snap:
            fj = jnp.asarray(f)
            pb = fj / jnp.maximum(fj.sum(1, keepdims=True), 1e-30)
            if last["f"] is not None:
                pa = last["f"] / jnp.maximum(last["f"].sum(1, keepdims=True), 1e-30)
                record["moved"][it] = np.asarray(0.5 * jnp.abs(pb - pa).sum(1)) / (it - last["it"])
            last.update(f=fj, it=it)
            log(f"  iteration {it}: deviance {dev:.5g}")

    return callback, record


def candidate_shares(f: np.ndarray, voxels: np.ndarray, max_voxels: int = 200000) -> np.ndarray:
    """Return each voxel's occupancy, sorted, as shares of its total, to see whether voxels wanted more candidates.

    Parameters
    ----------
    f
        [Nv, cand] occupancies (a fit stage file)
    voxels
        [Nv] the voxels to look at (e.g. the occupied ones); at most max_voxels of them, evenly sampled

    Returns
    -------
    np.ndarray
        [n, cand] shares, largest first
    """
    iv = np.flatnonzero(voxels)
    iv = iv[:: max(1, len(iv) // max_voxels)]
    return np.sort(f[iv], 1)[:, ::-1] / np.maximum(f[iv].sum(1, keepdims=True), 1e-30)


def ring_ratios(H: np.ndarray, model: np.ndarray, n_rings: int, scan: dict, etacut: float) -> np.ndarray:
    """Return the fitted / measured intensity of each ring, over the eta bins the fit models."""
    n_e = scan["n_e"] // R_E
    eta_c = -180.0 + (np.arange(n_e) + 0.5) * B_E * R_E
    use = np.abs(np.sin(np.radians(eta_c))) > etacut
    return _h4(model, n_rings, scan)[:, use].sum((1, 2, 3)) / np.maximum(_h4(H, n_rings, scan)[:, use].sum((1, 2, 3)),
                                                                         1e-30)  # fmt: skip


def back_projection(d: dict, y0: float, nr: int | None = None) -> np.ndarray:
    """Reconstruct the whole sample from a histogram stage file's data, for a y0 (as :func:`anri.index.reconstruct`).

    Parameters
    ----------
    d
        From :func:`anri.index.load_histogram`
    y0
        dty where the rotation axis is in the beam
    nr
        Voxels per side (default: the grid the fit uses for this y0, :func:`anri.index.voxel_grid`); fix it to compare
        y0 values on one grid

    Returns
    -------
    np.ndarray
        [nr, nr] image in reconstruction order
    """
    from .mask import reconstruct
    from .stages import voxel_grid

    sc = d["scan"]
    scan = {**sc, "y0": y0}
    nr = voxel_grid(sc, y0) if nr is None else nr
    return reconstruct(d["H"], d["n_rings"], sc["n_e"] // R_E, sc["n_o"] // R_O, scan, B_O * R_O, nr)
