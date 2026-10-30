"""Fit the detector and beam geometry to the peaks of a scanning-3DXRD scan.

Each peak (one 4D label in one dty row, :func:`anri.io.imaged11.read_peaks`) is matched to an orientation of an
indexed map and a reflection, and scatters from where the beam crosses that orientation's voxels at the peak's dty
and omega: the occupancy-weighted centre of the chord. That is the parallax a single-grain fit (ImageD11's makemap
and refine_em) cannot model on a scan, where the beam crosses a large grain at a different place in every row.

The geometry is then fitted by Gauss-Newton (Levenberg-Marquardt) to the peaks' (sc, fc, omega) predicted by
:func:`anri.fwd.peak.get_centroid`, with one small rotation per orientation eliminated by a Schur complement: each
orientation's rotation only moves its own peaks, so its 3 x 3 block of the normal equations is solved and
substituted, leaving a small system for the geometry (exactly as fitting everything together). Either
the wavelength or the lattice scale can be free, not both: Bragg's law and the Ewald condition depend on them only
through wavelength / d. A calibrant has a known lattice, so its wavelength is free; any other sample has a known
wavelength, so its lattice scale is free. The distance is then fixed by the peaks' omegas (the Ewald condition), not
only by the radii of the rings, which leave it degenerate with the wavelength.

:func:`strain_left` then frees each orientation's lattice with the geometry fixed: the strain left on a strain-free
calibrant measures the calibration (as ImageD11's silicon notebook estimates its precision).

It is meant for calibrants and little-deformed samples: each peak is matched to the orientation that predicts it
best at the current geometry, so orientations closer to one another than the geometry's error absorb part of it,
and the rounds of :func:`calibrate` converge slowly. A deformed sample's geometry is refined with its strains.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from anri.diffract.scatter import k_to_q_lab, peak_lab_to_k_out, scale_norm_k
from anri.fwd.peak import get_centroid
from anri.geom.detector import det_to_lab, detector_basis_vectors_lab, detector_transforms
from anri.geom.gonio import lab_to_sample, origin_lab
from anri.maths.rotation import rod_to_mat
from anri.maths.utils import float64

GEOMETRY = ("distance", "y_center", "z_center", "tilt_x", "tilt_y", "tilt_z", "wavelength", "lattice")
_K_IN = (1.0, 0.0, 0.0)  # ImageD11's beam, along lab x


def _bucket(n: int) -> int:
    """Return the padded length for n peaks: the next power of two, at least 4096.

    Every round's number of matched peaks changes by tens of thousands; padded, they run the same compiled functions.
    """
    return 1 << max(12, int(np.ceil(np.log2(max(n, 1)))))


def _pars(x: jax.Array, base: dict, free: tuple) -> dict:
    """Return the parameters, with the free ones taken from x."""
    return {**base, **{k: x[i] for i, k in enumerate(free)}}


def _detector(p: dict) -> tuple:
    """(det_trans, beam_cen_shift, x_distance_shift) of :func:`anri.geom.detector.detector_transforms`."""
    return detector_transforms(p["y_center"], p["y_size"], p["tilt_y"], p["z_center"], p["z_size"], p["tilt_z"],
                               p["tilt_x"], p["distance"], p["o11"], p["o12"], p["o21"], p["o22"])  # fmt: skip


def _g_sample(sc: jax.Array, fc: jax.Array, omega: jax.Array, dty: jax.Array, origin: jax.Array, p: dict,
              det: tuple) -> jax.Array:  # fmt: skip
    """Scattering vector (sample frame, 1 / angstrom) of a peak at (sc, fc, omega) scattered from origin (sample)."""
    k_in = jnp.asarray(_K_IN, det[0].dtype)
    org = origin_lab(origin, k_in, omega, p["wedge"], p["chi"], dty, p["y0"])
    k_out = peak_lab_to_k_out(det_to_lab(sc, fc, *det), org, p["wavelength"])
    q = k_to_q_lab(scale_norm_k(k_in, p["wavelength"]), k_out)
    return lab_to_sample(q, omega, p["wedge"], p["chi"], 0.0, 0.0)


@partial(jax.jit, static_argnames=("n_grid", "n_t"))
def _match_chunk(pk: jax.Array, ubis: jax.Array, grid_o: jax.Array, grid_w: jax.Array, p: dict, step: float,
                 n_grid: int, n_t: int) -> tuple:  # fmt: skip
    """Match a chunk of peaks [C, 4] (sc, fc, omega, dty): see :func:`match`."""
    det = _detector(p)
    t = (jnp.arange(n_t) - n_t // 2) * step  # sample points along the beam's centre line (lab x)

    def one(q: jax.Array) -> tuple:
        sc, fc, om, dy = q[0], q[1], q[2], q[3]
        lab = jnp.stack([t, jnp.zeros_like(t), jnp.zeros_like(t)], 1)
        v = jax.vmap(lambda a: lab_to_sample(a, om, p["wedge"], p["chi"], dy, p["y0"]))(lab)
        v = v.at[:, 2].set(0.0)  # columns: the map is one layer
        ri = jnp.round(v[:, 0] / step).astype(jnp.int32) + n_grid // 2  # anri.geom.scan.recon_to_sample, inverted
        rj = jnp.round(-v[:, 1] / step).astype(jnp.int32) + n_grid // 2
        inside = (ri >= 0) & (ri < n_grid) & (rj >= 0) & (rj < n_grid)
        vox = jnp.where(inside, ri * n_grid + rj, 0)
        o = jnp.where(inside[:, None], grid_o[vox], -1)  # [n_t, K]
        w = jnp.where(inside[:, None] & (o >= 0), grid_w[vox], 0.0)
        g = jax.vmap(lambda c: _g_sample(sc, fc, om, dy, c, p, det))(v)  # [n_t, 3]
        hkl = jnp.einsum("tkij,tj->tki", ubis[jnp.maximum(o, 0)], g)
        err = jnp.where(w > 0, jnp.linalg.norm(hkl - jnp.round(hkl), axis=-1), jnp.inf)
        best = jnp.argmin(err)
        ob = o.ravel()[best]
        wo = jnp.sum(jnp.where(o == ob, w, 0.0), 1)  # the matched orientation's occupancy along the ray
        origin = jnp.sum(wo[:, None] * v, 0) / jnp.maximum(wo.sum(), 1e-30)
        return ob, jnp.round(hkl.reshape(-1, 3)[best]), err.ravel()[best], origin, wo.sum()

    return jax.vmap(one)(pk)


def match(peaks: dict, ubi: np.ndarray, grid_o: np.ndarray, grid_w: np.ndarray, step: float, pars: dict,
          chunk: int = 2048) -> dict:  # fmt: skip
    """Match each peak to an orientation and a reflection, and find where it scatters from.

    The beam's centre line at the peak's dty and omega is sampled at every voxel it crosses. Each orientation a
    sampled voxel holds gives the peak's (h, k, l) when it scatters from there; the match is the orientation and
    point with the (h, k, l) nearest integers. The peak then scatters from that orientation's occupancy-weighted
    centre along the line.

    Parameters
    ----------
    peaks
        From :func:`anri.io.imaged11.read_peaks`
    ubi
        [G, 3, 3] (U B)^-1 of each orientation, at the reference lattice (scaled by ``pars["lattice"]``)
    grid_o, grid_w
        [NR * NR, K] orientations (index into ubi, -1 for none) and occupancies of each voxel of the n x n
        reconstruction grid (:func:`anri.geom.scan.recon_positions` order)
    step
        Voxel size (as dty)
    pars
        Parameters, see :func:`base_pars`
    chunk
        Peaks per call (one compile)

    Returns
    -------
    dict
        Per peak: "orient" (index into ubi), "hkl", "err" (distance of (h, k, l) from integers), "origin" [3] (sample
        frame) and "weight" (the orientation's summed occupancy along the line)
    """
    n_grid = round(np.sqrt(len(grid_o)))
    n_t = int(np.ceil(n_grid * np.sqrt(2))) + 2  # the grid's diagonal
    # float32: (h, k, l) to ~1e-6 and origins to ~1e-4 um are plenty to match, and 8x faster on a GPU than float64
    pk = np.stack([peaks[k] for k in ("sc", "fc", "omega", "dty")], 1).astype(np.float32)
    n = len(pk)
    pk = np.concatenate([pk, np.repeat(pk[:1], -n % chunk, 0)])
    args = (jnp.asarray(np.asarray(ubi) * pars["lattice"], jnp.float32), jnp.asarray(grid_o, jnp.int32),
            jnp.asarray(grid_w, jnp.float32), {k: jnp.float32(v) for k, v in pars.items()}, float(step))  # fmt: skip
    out = [_match_chunk(jnp.asarray(pk[i : i + chunk]), *args, n_grid, n_t) for i in range(0, len(pk), chunk)]
    out = [np.concatenate([np.asarray(o[j]) for o in out])[:n] for j in range(5)]
    out[1], out[3] = out[1].astype(np.float64), out[3].astype(np.float64)  # hkl and origin, for the fit (float64)
    return dict(zip(("orient", "hkl", "err", "origin", "weight"), out))


def _sym(e: jax.Array) -> jax.Array:
    """Symmetric 3 x 3 matrix from (xx, yy, zz, yz, xz, xy)."""
    return jnp.array([[e[0], e[5], e[4]], [e[5], e[1], e[3]], [e[4], e[3], e[2]]])


def _deform(ubi: jax.Array, a: jax.Array) -> jax.Array:
    """Return ubi turned by the Rodrigues vector a[:3] and, if a has 9 entries, strained by a[3:] (sample frame)."""
    out = ubi @ rod_to_mat(a[:3]).T
    return out @ (jnp.eye(3, dtype=a.dtype) + _sym(a[3:])) if a.shape[0] == 9 else out


def _residual(x: jax.Array, a: jax.Array, ubi: jax.Array, obs: jax.Array, hkl: jax.Array, branch: jax.Array,
              origin: jax.Array, sig: jax.Array, base: dict, free: tuple) -> jax.Array:  # fmt: skip
    """(observed - predicted) / sigma in (sc, fc, omega) for one peak, its orientation deformed by a."""
    p = _pars(x, base, free)
    s_step, f_step, det_origin = detector_basis_vectors_lab(*_detector(p))
    c, _ = get_centroid(p["lattice"] * _deform(ubi, a), origin, hkl, 1.0 - 2.0 * branch, p["wavelength"],
                        jnp.asarray(_K_IN, x.dtype), 0.0, 0.0, p["wedge"], p["chi"], obs[3], p["y0"], s_step, f_step,
                        det_origin)  # fmt: skip
    d = obs[:3] - c
    d = d.at[2].set((d[2] + 180.0) % 360.0 - 180.0)
    return d / sig


@partial(jax.jit, static_argnames=("free", "n_orient", "n_a", "jac"))
def _normal(x: jax.Array, ubi: jax.Array, data: dict, sig: jax.Array, base: dict, free: tuple, n_orient: int,
            n_a: int, jac: bool = True) -> tuple:  # fmt: skip
    """Gauss-Newton normal equations, each orientation's n_a parameters kept apart; chi^2 alone if not jac."""
    o, w = data["orient"], data["w"]
    args = (ubi[o], data["obs"], data["hkl"], data["branch"], data["origin"])
    zero = jnp.zeros((len(o), n_a), x.dtype)

    def f(xx: jax.Array, a: jax.Array, ubi_p: jax.Array, obs: jax.Array, hkl: jax.Array, branch: jax.Array,
          origin: jax.Array) -> jax.Array:  # fmt: skip
        return _residual(xx, a, ubi_p, obs, hkl, branch, origin, sig, base, free)

    r = jax.vmap(f, (None, 0, 0, 0, 0, 0, 0))(x, zero, *args) * w[:, None]
    chi2 = jnp.sum(r**2)
    if not jac:
        return (chi2,)
    Jx, Ja = jax.vmap(jax.jacfwd(f, (0, 1)), (None, 0, 0, 0, 0, 0, 0))(x, zero, *args)
    Jx, Ja = Jx * w[:, None, None], Ja * w[:, None, None]
    seg = partial(jax.ops.segment_sum, segment_ids=o, num_segments=n_orient)
    return (chi2, jnp.einsum("pin,pim->nm", Jx, Jx), jnp.einsum("pin,pi->n", Jx, r),
            seg(jnp.einsum("pin,pia->pna", Jx, Ja)), seg(jnp.einsum("pia,pib->pab", Ja, Ja)),
            seg(jnp.einsum("pia,pi->pa", Ja, r)))  # fmt: skip


def _solve(H: tuple, mu: float) -> tuple:
    """Levenberg-Marquardt step (dx, da) with the orientations eliminated (Schur complement), and S^-1.

    H is (Hxx, bx, Hxa, Haa, ba) from :func:`_normal`: the free parameters' block and gradient, their coupling to
    each orientation's parameters, and each orientation's own block and gradient.
    """
    Hxx, bx, Hxa, Haa, ba = H
    eye = np.eye(Haa.shape[-1])
    Haa = Haa + mu * Haa * eye + 1e-12 * np.trace(Haa, axis1=1, axis2=2)[:, None, None] * eye + 1e-30 * eye
    Hinv = np.linalg.inv(Haa)
    S = Hxx + mu * np.diag(np.diag(Hxx)) - np.einsum("gna,gab,gmb->nm", Hxa, Hinv, Hxa)
    rhs = -bx + np.einsum("gna,gab,gb->n", Hxa, Hinv, ba)
    d = np.sqrt(np.diag(S))  # Jacobi scaling: distance (um) and tilts (rad) differ by 1e8
    Sinv = np.linalg.inv(S / np.outer(d, d)) / np.outer(d, d) if len(d) else S
    dx = Sinv @ rhs
    da = -np.einsum("gab,gb->ga", Hinv, ba + np.einsum("gna,n->ga", Hxa, dx))
    return dx, da, Sinv


def base_pars(pars: dict, y0: float) -> dict:
    """Return the parameters the fit works with, from ImageD11 parameters (anri's sign of wedge, a lattice scale of 1)."""
    keys = ("y_size", "z_size", "o11", "o12", "o21", "o22") + GEOMETRY[:-1]
    out = {k: float(pars[k]) for k in keys}
    out.update(wedge=-float(pars.get("wedge", 0.0)), chi=float(pars.get("chi", 0.0)), y0=float(y0), lattice=1.0)
    return out


def imaged11_pars(pars: dict) -> dict:
    """Return ImageD11 detector and beam parameters from the fit's (ImageD11's sign of wedge; no lattice scale)."""
    out = {k: v for k, v in pars.items() if k not in ("lattice", "y0", "wedge")}
    out["wedge"] = -pars["wedge"]
    return out


def fit_geometry(peaks: dict, m: dict, ubi: np.ndarray, pars: dict, free: tuple, sig: np.ndarray,
                 n_iter: int = 10, strain: bool = False, log: Callable = print) -> tuple:  # fmt: skip
    """Fit the free parameters and every orientation to matched peaks by Levenberg-Marquardt.

    Parameters
    ----------
    peaks
        From :func:`anri.io.imaged11.read_peaks`, the matched ones
    m
        Their matches, from :func:`match`, with "branch" (0 for etasign +1, 1 for -1, :func:`anri.fwd.peak.get_centroid`)
    ubi
        [G, 3, 3] (U B)^-1 of each orientation, see :func:`match`
    pars
        Parameters, see :func:`base_pars`
    free
        Names of the free parameters, from :data:`GEOMETRY` (may be empty)
    sig
        [3] the peaks' standard deviations in (sc, fc, omega): pixels, pixels, degrees
    n_iter
        Iterations
    strain
        Also free each orientation's lattice (6 strains), not only its rotation
    log
        Progress messages

    Returns
    -------
    tuple
        (pars, ubi, cov): the fitted parameters and orientations, and the covariance of the free parameters (scaled
        by chi^2 per degree of freedom)
    """
    n = len(m["orient"])
    pad = _bucket(n) - n  # one shape for every round: no recompile
    obs = np.stack([peaks[k] for k in ("sc", "fc", "omega", "dty")], 1)
    n_a = 9 if strain else 3

    def padded(a: np.ndarray) -> np.ndarray:
        return np.concatenate([a, np.repeat(a[:1], pad, 0)])

    def deform(ubi: np.ndarray, da: np.ndarray) -> np.ndarray:
        return np.asarray(jax.vmap(_deform)(jnp.asarray(ubi), jnp.asarray(da)))

    with float64():
        data = {"obs": padded(obs), "hkl": padded(m["hkl"]), "branch": padded(m["branch"]).astype(np.int32),
                "origin": padded(m["origin"]), "orient": padded(m["orient"]).astype(np.int32),
                "w": np.r_[np.ones(n), np.zeros(pad)]}  # fmt: skip
        data = {k: jnp.asarray(v) for k, v in data.items()}
        sg = jnp.asarray(sig, jnp.float64)
        base = {k: jnp.float64(v) for k, v in pars.items() if k not in free}
        x = np.array([pars[k] for k in free], float)
        ubi = np.asarray(ubi, float)
        mu, G = 1e-3, len(ubi)

        def normal(x: np.ndarray, ubi: np.ndarray, jac: bool = True) -> list:
            return [np.asarray(a) for a in _normal(jnp.asarray(x), jnp.asarray(ubi), data, sg, base, free, G, n_a, jac)]

        chi2, *H = normal(x, ubi)
        for it in range(n_iter):
            dx, da, _ = _solve(tuple(H), mu)
            ubi_new = deform(ubi, da)
            chi2_new = float(normal(x + dx, ubi_new, jac=False)[0])
            if chi2_new < chi2:
                x, ubi, mu = x + dx, ubi_new, mu / 10
                log(f"  GN {it}: chi2/peak {chi2_new / n:.4g} (mu {mu:.0e}); "
                    + ", ".join(f"{k} {v:.7g}" for k, v in zip(free, x)))  # fmt: skip
                done = chi2 - chi2_new < 1e-6 * chi2
                chi2, *H = normal(x, ubi)
                if done:
                    break
            else:
                mu *= 10
        _, _, Sinv = _solve(tuple(H), 0.0)
    cov = Sinv * chi2 / max(3 * n - len(free) - n_a * G, 1)
    return {**pars, **dict(zip(free, (float(v) for v in x)))}, ubi, cov


@jax.jit
def _predict(ubi: jax.Array, origin: jax.Array, hkl: jax.Array, dty: jax.Array, p: dict) -> tuple:
    """Both branches' (sc, fc, omega) of peaks [P] of orientations ubi [P, 3, 3], and their validity."""
    s_step, f_step, det_origin = detector_basis_vectors_lab(*_detector(p))
    k_in = jnp.asarray(_K_IN, ubi.dtype)

    def one(u: jax.Array, o: jax.Array, h: jax.Array, y: jax.Array, etasign: jax.Array) -> tuple:
        return get_centroid(u, o, h, etasign, p["wavelength"], k_in, 0.0, 0.0, p["wedge"], p["chi"], y, p["y0"], s_step,
                            f_step, det_origin)  # fmt: skip

    both = jax.vmap(one, in_axes=(None, None, None, None, 0))
    c, valid = jax.vmap(both, in_axes=(0, 0, 0, 0, None))(ubi, origin, hkl, dty, jnp.array([1.0, -1.0], ubi.dtype))
    return c, valid[:, 0]


def residuals(peaks: dict, m: dict, ubi: np.ndarray, pars: dict) -> tuple:
    """Observed - predicted (sc, fc, omega) of matched peaks for both branches: [P, 2, 3], and validity [P]."""
    n = len(m["orient"])
    idx = np.r_[np.arange(n), np.zeros(_bucket(n) - n, int)]  # padded with copies of the first: one compile
    with float64():
        c, ok = _predict(jnp.asarray(pars["lattice"] * np.asarray(ubi)[m["orient"][idx]]),
                         jnp.asarray(m["origin"][idx]), jnp.asarray(m["hkl"][idx]), jnp.asarray(peaks["dty"][idx]),
                         {k: jnp.float64(v) for k, v in pars.items()})  # fmt: skip
        c, ok = np.asarray(c)[:n], np.asarray(ok)[:n]
    obs = np.stack([peaks[k] for k in ("sc", "fc", "omega")], 1)
    d = obs[:, None, :] - c
    d[..., 2] = (d[..., 2] + 180.0) % 360.0 - 180.0
    return d, ok


def lattice_strain(ubi: np.ndarray, B: np.ndarray) -> tuple:
    """Strain of each orientation's lattice from the reference, in the sample frame, and its orientation.

    With ubi = B^-1 U^T (I + E) (E symmetric), K = B ubi = U^T (I + E) gives I + E = sqrt(K^T K) and U^T = K (I + E)^-1.

    Returns
    -------
    tuple
        E [G, 3, 3] and U [G, 3, 3]
    """
    K = np.einsum("ij,gjk->gik", B, ubi)
    w, v = np.linalg.eigh(np.einsum("gji,gjk->gik", K, K))
    stretch = np.einsum("gij,gj,gkj->gik", v, np.sqrt(w), v)
    U = np.swapaxes(K @ np.linalg.inv(stretch), 1, 2)
    return stretch - np.eye(3), U


def calibrate(peaks: dict, ubi: np.ndarray, grid_o: np.ndarray, grid_w: np.ndarray, step: float, pars: dict,
              free: tuple, tols: tuple = (0.1, 0.05, 0.025, 0.025), n_iter: int = 10, clip: float = 4.0,
              extra_rounds: int = 6, log: Callable = print, callback: Callable | None = None) -> dict:  # fmt: skip
    """Calibrate: match the peaks, fit the geometry, and again with the tolerances in turn.

    Parameters
    ----------
    peaks
        From :func:`anri.io.imaged11.read_peaks`
    ubi, grid_o, grid_w, step
        The indexed map, see :func:`match`
    pars
        Starting parameters, see :func:`base_pars`
    free
        Names of the free parameters, from :data:`GEOMETRY` ("wavelength" or "lattice", not both)
    tols
        Matching tolerance on (h, k, l) of each round, see :func:`match`
    n_iter
        Gauss-Newton iterations per round
    clip
        Leave out peaks further than this many (robust) standard deviations from their prediction
    extra_rounds
        After the tolerances, repeat the last one while a round still moves a free parameter by more than its
        standard deviation, at most this many times: each round matches the peaks again with the new geometry
    log
        Progress messages
    callback
        Called after each round as ``callback(round, pars, ubi, matched)``

    Returns
    -------
    dict
        "pars" (fitted), "ubi" (each orientation, rotated), "cov" (of the free parameters), "free", "sig" [3] (the
        peaks' robust standard deviations in sc, fc, omega), "matched" (the matched peaks: "index" into peaks,
        "orient", "hkl", "origin", "branch" and "residual" [P, 3]) and "moved" (how far the last round moved each
        free parameter, in its standard deviations: large if the rounds had not converged)
    """
    if "wavelength" in free and "lattice" in free:
        msg = "the wavelength and the lattice scale are degenerate: free one of them"
        raise ValueError(msg)
    ubi = np.asarray(ubi, float)
    cov, sig, keep, d, sub, before = None, None, None, None, {}, dict(pars)
    rounds, moved, rnd = list(tols), np.zeros(len(free)), 0
    while rnd < len(rounds):
        tol, before = rounds[rnd], dict(pars)
        m = match(peaks, ubi, grid_o, grid_w, step, pars)
        ok = (m["err"] < tol) & (m["weight"] > 0)
        sel = np.flatnonzero(ok)
        d, valid = residuals({k: v[sel] for k, v in peaks.items()}, {k: v[sel] for k, v in m.items()}, ubi, pars)
        branch = np.argmin(np.abs(d[..., 2]), 1)  # each reflection diffracts at two omegas: the nearer one
        # one peak per (orientation, reflection, branch, row): the best match
        key = np.stack([m["orient"][sel], *m["hkl"][sel].T.astype(int), branch, peaks["row"][sel]], 1)
        order = np.lexsort((m["err"][sel], *key.T[::-1]))
        first = order[np.r_[True, np.any(np.diff(key[order], axis=0) != 0, 1)]]
        sel, d, valid = sel[first], d[first, branch[first]], valid[first]
        sub = {k: v[sel] for k, v in m.items()}
        sub["branch"] = branch[first]
        pk = {k: v[sel] for k, v in peaks.items()}
        sig = np.maximum(1.4826 * np.median(np.abs(d - np.median(d, 0)), 0), (1e-4, 1e-4, 1e-6))  # noise-free data
        good = valid & np.all(np.abs(d) < clip * sig, 1)
        log(f"round {rnd}: tolerance {tol:g}: {ok.sum()} of {len(ok)} peaks matched, {len(sel)} unique, {good.sum()} "
            f"within {clip:g} sigma; sigma (sc, fc, omega) {sig[0]:.3f} px, {sig[1]:.3f} px, {sig[2]:.4f} deg")  # fmt: skip
        sub = {k: v[good] for k, v in sub.items()}
        pk = {k: v[good] for k, v in pk.items()}
        keep = sel[good]
        pars, ubi, cov = fit_geometry(pk, sub, ubi, pars, free, sig, n_iter, log=log)
        d, _ = residuals(pk, sub, ubi, pars)
        d = d[np.arange(len(keep)), sub["branch"]]
        log(f"round {rnd}: rms residual (sc, fc, omega) {np.sqrt(np.mean(d**2, 0)).round(4)}; "
            + ", ".join(f"{k} {pars[k]:.7g} +- {e:.2g}" for k, e in zip(free, np.sqrt(np.diag(cov)))))  # fmt: skip
        if callback is not None:
            callback(rnd, pars, ubi, keep)
        # how far this round moved the geometry, in its standard deviations: re-matching converges slowly when the
        # map holds orientations closer than the geometry's error (e.g. a local search's children)
        moved = np.abs([pars[k] - before[k] for k in free]) / np.sqrt(np.diag(cov))
        log(f"round {rnd}: moved " + ", ".join(f"{k} {v:.2g}" for k, v in zip(free, moved)) + " sigma")
        if rnd == len(rounds) - 1 and moved.max() > 1 and extra_rounds > 0:
            rounds.append(rounds[-1])
            extra_rounds -= 1
        rnd += 1
    matched = {"index": keep, "orient": sub["orient"], "hkl": sub["hkl"], "origin": sub["origin"],
               "branch": sub["branch"], "residual": d}  # fmt: skip
    return {"pars": pars, "ubi": ubi, "cov": cov, "free": free, "sig": sig, "matched": matched, "moved": moved}


def strain_left(peaks: dict, matched: dict, ubi: np.ndarray, B: np.ndarray, pars: dict, sig: np.ndarray,
                min_peaks: int = 20, n_iter: int = 10, log: Callable = print) -> dict:  # fmt: skip
    """Free each orientation's lattice with the geometry fixed: the strain a calibration leaves.

    On a strain-free calibrant, a right geometry leaves no strain beyond the noise: a deviatoric strain shows
    detector tilts or centre, a hydrostatic one the wavelength (or distance) against the reference lattice. This is
    ImageD11's silicon notebook's precision estimate, for each orientation of the map.

    Parameters
    ----------
    peaks, matched, ubi, pars, sig
        From :func:`calibrate`: the peaks and the result's "matched", "ubi", "pars" and "sig"
    B
        [3, 3] B matrix of the reference lattice
    min_peaks
        Fit orientations with at least this many peaks; the others keep their lattice (strain reported as NaN)
    n_iter, log
        See :func:`fit_geometry`

    Returns
    -------
    dict
        "strain" [G, 3, 3] (sample frame, from the reference lattice scaled by ``pars["lattice"]``), "n_peaks" [G],
        "chi2_ratio" (chi^2 per peak with free lattices over that without: how much the strain explains)
    """
    n_pk = np.bincount(matched["orient"], minlength=len(ubi))
    use = n_pk[matched["orient"]] >= min_peaks
    m = {k: v[use] for k, v in matched.items()}
    pk = {k: v[matched["index"][use]] for k, v in peaks.items()}
    d0 = m["residual"] / sig
    _, ubi_s, _ = fit_geometry(pk, m, ubi, pars, (), sig, n_iter, strain=True, log=log)
    d, _ = residuals(pk, m, ubi_s, pars)
    d = d[np.arange(len(m["branch"])), m["branch"]] / sig
    E, _ = lattice_strain(ubi_s, B)
    E[n_pk < min_peaks] = np.nan
    return {"strain": E, "n_peaks": n_pk, "chi2_ratio": float(np.sum(d**2) / np.sum(d0**2))}


def split_strain(strain: np.ndarray, n_peaks: np.ndarray) -> dict:
    """Split the strain left (:func:`strain_left`) into hydrostatic and deviatoric parts, and their peak-weighted means.

    Parameters
    ----------
    strain, n_peaks
        From :func:`strain_left`: [G, 3, 3] (NaN for orientations not fitted) and [G]

    Returns
    -------
    dict
        "hydrostatic" [G] (trace / 3), "deviatoric" [G, 3, 3], "fitted" [G] bool; and over the fitted orientations,
        weighted by their peaks, "mean_hydrostatic" and "mean_deviatoric" [3, 3] (NaN if none was fitted)
    """
    strain = np.asarray(strain, float)
    hydro = np.trace(strain, axis1=1, axis2=2) / 3
    dev = strain - hydro[:, None, None] * np.eye(3)
    ok = np.isfinite(hydro)
    out = {"hydrostatic": hydro, "deviatoric": dev, "fitted": ok, "mean_hydrostatic": np.nan,
           "mean_deviatoric": np.full((3, 3), np.nan)}  # fmt: skip
    if ok.any():
        w = np.asarray(n_peaks)[ok]
        out["mean_hydrostatic"] = float(np.average(hydro[ok], weights=w))
        out["mean_deviatoric"] = np.average(dev[ok], axis=0, weights=w)
    return out
