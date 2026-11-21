"""The rotation axis and the sample's drift, from Friedel pairs in the histogram.

Over a full turn every ray is measured twice: at dty and omega, and at 2 y0 - dty and omega + 180, where each
reflection's Friedel mate lights (ring, 180 - eta). Row k's twin is the mirrored row k'. Row k really sat at its
nominal dty plus ``ddty[k, omega]`` (:func:`anri.index.occupancy.system`), so with a trial y0 the pair's projection,
mirrored, is the row's shifted by

    m(k, omega) = ddty[k, omega] + ddty[k', omega + 180] + 2 (y0 - y0_true).

A drift of the sample against the rotation axis (it creeps on its pin, the stack above the rotation warms) moves
row k by ``ddty[k, omega] = s_k sin(omega) + c_k cos(omega)``, and the rows are measured in turn, so the drift is a
function of the row. The pairs give y0 (from the constant of m) and s_k - s_k', c_k - c_k': the part of the drift
that is odd about the time row y0 was measured. The even part (and a constant, which is a shift of the whole map) is
not seen by the pairs; a fit of the map sees it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


def _pair_shift(H4: jax.Array, k: jax.Array, kk: jax.Array, m0: jax.Array, o1: jax.Array, o2: jax.Array,
                je: jax.Array, use_e: jax.Array, om: jax.Array, ystep: float) -> tuple:  # fmt: skip
    """One Gauss-Newton step for one row: the change of m = a + b sin + c cos that best maps the pair on the row.

    ``kk`` is the mirrored row (fractional) at m = 0 and ``m0`` [n_pairs] the current m; returns the change, its
    normal matrix and the per-omega estimates (numerator, denominator) for an error bar.
    """
    n_k = H4.shape[3]
    x = kk - m0 / ystep  # where the pair is read: P(2 y0 - y - m)
    i0 = jnp.clip(jnp.floor(x).astype(jnp.int32), 1, n_k - 3)
    w = x - i0

    def at(i: jax.Array) -> jax.Array:  # the Friedel mates (180 - eta, omega + 180) in row i[omega]
        return H4[:, je[:, None], o2[None, :], i[None, :]]

    pm, p0, p1, p2 = at(i0 - 1), at(i0), at(i0 + 1), at(i0 + 2)
    Pv = (1 - w) * p0 + w * p1
    dP = ((1 - w) * (p1 - pm) + w * (p2 - p0)) / (2 * ystep)  # dP / d(its dty)
    A = H4[:, :, o1, k]  # [ring, eta, n_pairs]
    # A(y) = P(2 y0 - y - m - dm) ~ Pv - dm dP: least squares for dm, per omega
    e = use_e[None, :, None]
    num = jnp.sum(jnp.where(e, (Pv - A) * dP, 0.0), (0, 1))
    den = jnp.sum(jnp.where(e, dP * dP, 0.0), (0, 1))
    B = jnp.stack([jnp.ones_like(om), jnp.sin(om), jnp.cos(om)], 1)
    N = (B * den[:, None]).T @ B
    beta = jnp.linalg.solve(N + 1e-30 * jnp.eye(3), B.T @ num)
    sig = jnp.sum(jnp.where(e, A, 0.0))
    return beta, N, num, den, sig


def friedel_shifts(H: ArrayLike, n_rings: int, scan: dict, y0: float, etacut: float, n_iter: int = 5,
                   min_frac: float = 0.05, max_shift: float = 10.0) -> dict:  # fmt: skip
    """Fit, row by row, the shift m(omega) = a + b sin(omega) + c cos(omega) between each row and its Friedel pairs.

    Gauss-Newton: the pair is read again at the current shift each iteration, so shifts of a few rows are not
    underestimated as by one linearised step.

    Parameters
    ----------
    H, n_rings, scan
        The fit histogram ``H[ring, eta, omega, row]`` (1 x 1 deg) and its scan (:func:`anri.index.stages.load_histogram`);
        offsets already known (``scan["ddty"]``: fly and helical scans) are taken into account, so the shift is what
        remains
    y0
        Trial y0 (mirrors the rows)
    etacut
        Only eta bins with |sin(eta)| above this
    n_iter
        Gauss-Newton iterations
    min_frac
        Rows whose paired bins hold less than this x the brightest row's counts are left out (NaN): past the
        sample's edges there is nothing to pair
    max_shift
        Rows whose shift reaches this many dty steps at any omega are left out (NaN): not converged

    Returns
    -------
    dict
        "dty" [n_rows], "coef" [n_rows, 3] (a, b, c, in the units of dty; NaN where the row has no pair or too few
        counts), "err" [n_rows, 3] (from the scatter over omega), "mirror" [n_rows] (the paired row, fractional)
    """
    from .data import B_E, B_O, R_E, R_O

    n_e, n_o, n_k = scan["n_e"] // R_E, scan["n_o"] // R_O, scan["n_rows"]
    b_o = B_O * R_O
    half = round(180.0 / b_o)
    if n_o <= half:
        msg = f"the scan covers {n_o * b_o:.1f} deg: Friedel pairs need more than half a turn"
        raise ValueError(msg)
    if n_e % 2:
        msg = f"{n_e} eta bins: Friedel pairs need an even number"
        raise ValueError(msg)
    H4 = jnp.asarray(H).reshape(n_rings, n_e, n_o, n_k)
    eta = -180.0 + (np.arange(n_e) + 0.5) * B_E * R_E
    use_e = jnp.asarray(np.abs(np.sin(np.radians(eta))) > etacut)
    je = jnp.asarray((n_e // 2 - 1 - np.arange(n_e)) % n_e)  # 180 - eta
    o1 = np.arange(n_o - half)
    o2 = o1 + half
    om = jnp.asarray(np.radians(scan["om0"] + (o1 + 0.5) * b_o))
    dty = scan["dty0"] + np.arange(n_k) * scan["ystep"]
    mirror = (2 * y0 - dty - scan["dty0"]) / scan["ystep"]
    ok = (mirror >= 1) & (mirror <= n_k - 3) & (np.arange(n_k) >= 1) & (np.arange(n_k) <= n_k - 2)
    rows = np.flatnonzero(ok)
    ystep = float(scan["ystep"])
    o1j, o2j = jnp.asarray(o1), jnp.asarray(o2)
    Bm = jnp.stack([jnp.ones_like(om), jnp.sin(om), jnp.cos(om)], 1)
    known = jnp.zeros((n_k, n_o), H4.dtype) if scan.get("ddty") is None else jnp.asarray(scan["ddty"], H4.dtype)

    @jax.jit
    def run(H4: jax.Array, known: jax.Array, rows: jax.Array, kks: jax.Array) -> tuple:
        def one(args: tuple) -> tuple:
            k, kk = args
            i0 = jnp.clip(jnp.floor(kk).astype(jnp.int32), 0, n_k - 2)
            w = kk - i0  # offsets already known, of the row and of its pair (read between two rows, as the pair is)
            m_known = known[k, o1j] + (1 - w) * known[i0, o2j] + w * known[i0 + 1, o2j]
            coef = jnp.zeros(3, H4.dtype)
            for _ in range(n_iter):
                beta, N, num, den, sig = _pair_shift(H4, k, kk, m_known + Bm @ coef, o1j, o2j, je, use_e, om, ystep)
                coef = coef + beta
            # error bar: the scatter of the per-omega estimates about the fit, at the last step
            good = den > 0
            r = jnp.where(good, num / jnp.where(good, den, 1.0) - Bm @ beta, 0.0)
            s2 = jnp.sum(den * r**2) / jnp.maximum(jnp.sum(good) - 3, 1)
            err = jnp.sqrt(jnp.diag(jnp.linalg.inv(N + 1e-30 * jnp.eye(3))) * s2)
            return coef, err, sig

        return jax.lax.map(one, (rows, kks))

    coef, err, sig = jax.device_get(run(H4, known, jnp.asarray(rows), jnp.asarray(mirror[rows], H4.dtype)))
    out_c = np.full((n_k, 3), np.nan)
    out_e = np.full((n_k, 3), np.nan)
    sig = np.asarray(sig)
    keep = (sig >= min_frac * sig.max()) & (np.abs(coef[:, 0]) + np.hypot(coef[:, 1], coef[:, 2]) < max_shift * ystep)
    out_c[rows[keep]] = coef[keep]
    out_e[rows[keep]] = err[keep]
    return {"dty": dty, "coef": out_c, "err": out_e, "mirror": mirror}


def from_pairs(shifts: dict, y0: float, max_err: float = 0.5) -> dict:
    """Return the rotation axis and each row's drift (its part odd about row y0's time) from :func:`friedel_shifts`.

    Row k and its mirror k' measure the same pairs from both ends: their (b, c) should be opposite, and the two are
    averaged. Rows with no estimate, or one less certain than ``max_err`` (units of dty), hold their nearest
    neighbour's drift.

    Returns
    -------
    dict
        "y0" (y0 - a / 2, a's error-weighted median over the measured rows), "y0_err", "s" and "c" [n_rows] (``ddty = s sin(omega) + c
        cos(omega)``), "err" [n_rows] (each one's error), "measured" [n_rows] (False where held from a neighbour),
        "a" [n_rows] (each row's constant: it varies with the row if the stage drifts across the beam, or with a
        wrong mirror)
    """
    coef, err, mirror = shifts["coef"], shifts["err"], shifts["mirror"]
    n_k = len(coef)
    a = coef[:, 0]
    good = np.isfinite(a) & (err[:, 0] > 0)
    if not good.any():
        msg = "no row has a Friedel pair with enough counts: is the scan a full turn, and is y0 within it?"
        raise ValueError(msg)
    order = np.argsort(a[good])
    wa = (1.0 / err[good, 0] ** 2)[order]
    a_med = float(a[good][order][np.searchsorted(np.cumsum(wa), 0.5 * wa.sum())])  # weighted median
    y0_new = y0 - 0.5 * a_med
    mad = 1.4826 * float(np.median(np.abs(a[good] - a_med)))
    y0_err = 0.5 * mad / np.sqrt(good.sum())  # rows as independent: a lower bound

    def mirrored(v: np.ndarray) -> np.ndarray:  # v at the mirrored row, linearly between rows
        x = np.clip(mirror, 0, n_k - 1)
        i0 = np.clip(np.floor(x).astype(int), 0, n_k - 2)
        w = x - i0
        return (1 - w) * v[i0] + w * v[i0 + 1]

    # m's (b, c) = ddty(k) - ddty(k') (sin, cos parts); from row k' it is the opposite: average, then halve (odd)
    out = {}
    for i, name in ((1, "s"), (2, "c")):
        d_k, e_k = coef[:, i], err[:, i]
        d_m, e_m = -mirrored(d_k), mirrored(e_k)
        both = np.isfinite(d_k) & np.isfinite(d_m)
        wk, wm = 1 / np.maximum(e_k, 1e-12) ** 2, 1 / np.maximum(e_m, 1e-12) ** 2
        d = np.where(both, (wk * np.nan_to_num(d_k) + wm * np.nan_to_num(d_m)) / (wk + wm),
                     np.where(np.isfinite(d_k), d_k, d_m))  # fmt: skip
        e = np.where(both, 1 / np.sqrt(wk + wm), np.where(np.isfinite(d_k), e_k, e_m))
        out[name], out[name + "_err"] = 0.5 * d, 0.5 * e
    err_r = np.hypot(out["s_err"], out["c_err"])
    measured = np.isfinite(out["s"]) & np.isfinite(out["c"]) & (err_r <= max_err)
    if not measured.any():
        msg = f"no row's drift is measured better than {max_err:g}"
        raise ValueError(msg)
    idx = np.flatnonzero(measured)
    near = idx[np.clip(np.searchsorted(idx, np.arange(n_k)), 0, len(idx) - 1)]  # hold the nearest measured row
    left = idx[np.clip(np.searchsorted(idx, np.arange(n_k)) - 1, 0, len(idx) - 1)]
    near = np.where(np.abs(left - np.arange(n_k)) < np.abs(near - np.arange(n_k)), left, near)
    s, c = out["s"][near], out["c"][near]
    return {"y0": float(y0_new), "y0_err": float(y0_err), "s": s, "c": c, "err": np.where(measured, err_r, np.nan),
            "measured": measured, "a": a}  # fmt: skip


def roughness(s: np.ndarray, c: np.ndarray, err: np.ndarray, measured: np.ndarray, gap: int = 3,
              n_sigma: float = 5.0) -> dict:  # fmt: skip
    """Check that a drift is sensible: that it changes from row to row only by its noise, and where it jumps.

    A drift is slow: between neighbouring measured rows it should change by about its noise, so the ratio of the rms
    change to the noise is ~1 for a drift smooth on the scale of a row. A sudden change (a slip, or a bad row) is a
    jump: the drift ``gap`` rows on differs by more than ``n_sigma`` times the noise. Not between neighbours: the
    Friedel pairs are read between two rows, which spreads a slip over about two rows.

    Returns
    -------
    dict
        "ratio" (rms row-to-row change / that of the noise), "jumps" (row indices, the later of each pair; a jump
        shows on up to ``gap`` + 1 consecutive rows), "max_jump" (the largest change over ``gap`` rows, units of dty),
        "total" (the drift's range over the measured rows, 2nd to 98th percentile: noisy rows at the sample's edges
        do not inflate it)
    """
    idx = np.flatnonzero(measured)
    if len(idx) <= gap:
        return {"ratio": float("nan"), "jumps": np.zeros(0, int), "max_jump": float("nan"), "total": float("nan")}
    v = np.stack([s[idx], c[idx]], 1)
    e = err[idx]
    step = np.hypot(*np.diff(v, axis=0).T)
    noise = np.sqrt(e[1:] ** 2 + e[:-1] ** 2)
    jump = np.hypot(*(v[gap:] - v[:-gap]).T)
    noise_g = np.sqrt(e[gap:] ** 2 + e[:-gap] ** 2)
    total = float(np.hypot(*(np.percentile(v, 98, axis=0) - np.percentile(v, 2, axis=0))))
    return {"ratio": float(np.sqrt(np.mean(step**2) / np.mean(noise**2))),
            "jumps": idx[gap:][jump > n_sigma * noise_g], "max_jump": float(jump.max()), "total": total}  # fmt: skip


def smooth(v: np.ndarray, err: np.ndarray, measured: np.ndarray, window: int) -> np.ndarray:
    """Smooth a drift: its running mean over ``window`` rows, each measured row weighted by 1 / err^2.

    The fit takes each row's offset from the row nearest each voxel (:func:`anri.index.occupancy.system`), which is
    exact only for offsets that change little from one row to the next: a row whose offset differs from its
    neighbours' by d (in rows) changes its sensitivity by ~d. A drift's per-row noise must be averaged away first.
    """
    w = np.where(measured & np.isfinite(v), 1.0 / np.maximum(err, 1e-12) ** 2, 0.0)
    k = np.ones(window)
    num = np.convolve(np.where(w > 0, v, 0.0) * w, k, "same")
    den = np.convolve(w, k, "same")
    out = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
    idx = np.flatnonzero(np.isfinite(out))
    return out[idx[np.clip(np.searchsorted(idx, np.arange(len(v))), 0, len(idx) - 1)]] if len(idx) else out


def measure(H: ArrayLike, n_rings: int, scan: dict, y0: float, etacut: float, window: int = 31,
            n_sigma: float = 5.0) -> dict:  # fmt: skip
    """Measure y0 and the drift from the Friedel pairs, check the drift, and smooth it for the fit.

    What the fit's ``y0="pairs"`` and ``drift="pairs"`` use. The check (:func:`roughness`) is on each row's own
    value; the fit gets the drift smoothed over ``window`` rows (:func:`smooth`): the per-row values are too noisy
    to apply as they are (on Renk_a of ihma936, 0.05 um per row at 0.2 um steps).

    Returns
    -------
    dict
        :func:`from_pairs`'s, with "s" and "c" smoothed ("s_row", "c_row": each row's own), "window",
        "y0_guess" (the trial y0), "shifts" (:func:`friedel_shifts`'s), "roughness" (:func:`roughness`'s, on the
        rows' own values) and "dty" [n_rows] the rows' nominal dty
    """
    shifts = friedel_shifts(H, n_rings, scan, y0, etacut)
    p = from_pairs(shifts, y0)
    rough = roughness(p["s"], p["c"], p["err"], p["measured"], n_sigma=n_sigma)
    s_sm, c_sm = (smooth(p[k], p["err"], p["measured"], window) for k in ("s", "c"))
    return {**p, "s": s_sm, "c": c_sm, "s_row": p["s"], "c_row": p["c"], "window": int(window), "y0_guess": float(y0),
            "shifts": shifts, "dty": shifts["dty"], "roughness": rough}  # fmt: skip


def offsets(s: np.ndarray, c: np.ndarray, scan: dict) -> np.ndarray:
    """Each (row, omega bin)'s dty offset from a drift: ``s sin(omega) + c cos(omega)``, [n_rows, n_o] (as ``ddty``)."""
    from .data import B_O, R_O

    n_o = scan["n_o"] // R_O
    om = np.radians(scan["om0"] + (np.arange(n_o) + 0.5) * B_O * R_O)
    return s[:, None] * np.sin(om)[None] + c[:, None] * np.cos(om)[None]
