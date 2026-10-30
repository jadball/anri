"""Fit the detector geometry to powder rings: ImageD11's powder calibration, the first step before indexing.

The starting geometry (distance and detector position from the motors, the wavelength from the energy) is rarely
good enough to index: a ring 0.05 degrees off its prediction falls outside the indexer's ring windows. As ImageD11's
``powder_calib`` notebook does with ``transformer.fit``, the distance, beam centre and the two detector tilts that
move rings (tilt_y, tilt_z) are fitted so that the rings of a known phase fall at their 2theta. tilt_x turns the
detector in its own plane: it moves no ring and is left for the single-crystal calibration (:mod:`anri.calib.fit`),
as is the wavelength, which rings alone cannot separate from the distance.

The data are a detector image: a dense powder image, or a scan's sparse pixels summed over every frame
(:func:`sum_frames`). Each lit pixel's 2theta is seen from the lab origin (no parallax), as ImageD11 computes it.
Each pixel is assigned to the nearest ring within a tolerance, and the pixels of each (ring, eta sector) give one
residual: their intensity-weighted mean 2theta minus the ring's. Every sector counts once, as every peak of
ImageD11's sector-by-sector peak search does; the mean is over the pixels' exact 2theta, so an arc's curvature
does not bias it. The fit is Gauss-Newton (Levenberg-Marquardt) in rounds of decreasing tolerance, the pixels
re-assigned each round.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from anri.crystal.structure import reflections, rings
from anri.diffract.scatter import k_out_to_tth_eta
from anri.geom.detector import det_to_lab
from anri.maths.utils import float64

from .fit import _K_IN, _detector, _pars

FREE = ("distance", "y_center", "z_center", "tilt_y", "tilt_z")


def sum_frames(chunks: Iterable, shape: tuple) -> np.ndarray:
    """Sum sparse pixels over every frame into one detector image.

    Parameters
    ----------
    chunks
        (slow, fast, omega, row, value) chunks of :func:`anri.io.imaged11.stream_sparse`, read without the
        distortion correction (integer pixel positions); pixels of frames with row -1 are left out
    shape
        (n_slow, n_fast)

    Returns
    -------
    np.ndarray
        [n_slow, n_fast] float64
    """
    image = np.zeros(shape[0] * shape[1])
    for slow, fast, _, row, val in chunks:
        ok = row >= 0
        idx = slow[ok].astype(np.int64) * shape[1] + fast[ok].astype(np.int64)
        image += np.bincount(idx, weights=val[ok], minlength=len(image))
    return image.reshape(shape)


def lit_pixels(image: np.ndarray, good: np.ndarray | None = None, spatial: tuple | None = None,
               threshold: float = 0.0) -> dict:  # fmt: skip
    """Return an image's pixels above a threshold, at their distortion-corrected positions.

    Parameters
    ----------
    image
        [n_slow, n_fast] detector image (e.g. from :func:`sum_frames`)
    good
        [n_slow, n_fast] True where a pixel is measured (:func:`anri.io.imaged11.read_mask`); None: all
    spatial
        (dx, dy) distortion maps (:func:`anri.io.imaged11.read_spatial`); None: no correction
    threshold
        Keep pixels above this, with this subtracted

    Returns
    -------
    dict
        "sc", "fc" (pixel positions) and "intensity", [N] each
    """
    lit = image > threshold
    if good is not None:
        lit &= good
    s, f = np.nonzero(lit)
    sc, fc = s.astype(np.float64), f.astype(np.float64)
    if spatial is not None:
        sc, fc = sc + spatial[1][s, f], fc + spatial[0][s, f]
    return {"sc": sc, "fc": fc, "intensity": image[s, f] - threshold}


def ring_ds(lattice: np.ndarray, space_group: int, wavelength: float, tth_max: float,
            merge: float = 1e-4) -> np.ndarray:  # fmt: skip
    """d* (1 / angstrom) of a phase's rings up to 2theta tth_max (degrees); reflections within ``merge`` are one ring."""
    dsmax = 2 * np.sin(np.radians(tth_max) / 2) / wavelength
    ds = reflections(lattice, space_group, wavelength, dsmax)["ds"]
    return rings(ds, merge)[1]


@jax.jit
def pixel_tth_eta(sc: jax.Array, fc: jax.Array, p: dict) -> tuple:
    """(2theta, eta) of pixels (degrees), seen from the lab origin, for the geometry p (:func:`anri.calib.fit.base_pars`)."""
    det = _detector(p)
    lab = jax.vmap(lambda s, f: det_to_lab(s, f, *det))(sc, fc)
    return k_out_to_tth_eta(lab, jnp.asarray(_K_IN, lab.dtype))


def _ring_tth(ds: jax.Array, wavelength: jax.Array) -> jax.Array:
    """2theta (degrees) of rings of d* ds."""
    return jnp.degrees(2 * jnp.arcsin(ds * wavelength / 2))


@partial(jax.jit, static_argnames=("free", "n_groups"))
def _residuals(x: jax.Array, base: dict, free: tuple, sc: jax.Array, fc: jax.Array, w: jax.Array, group: jax.Array,
               ring: jax.Array, ds: jax.Array, n_groups: int) -> jax.Array:  # fmt: skip
    """[n_groups] each (ring, sector)'s weighted mean 2theta minus its ring's (0 where a sector has no pixels)."""
    p = _pars(x, base, free)
    tth, _ = pixel_tth_eta(sc, fc, p)
    W = jax.ops.segment_sum(w, group, n_groups)
    mean = jax.ops.segment_sum(w * tth, group, n_groups) / jnp.where(W > 0, W, 1.0)
    return jnp.where(W > 0, mean - _ring_tth(ds, p["wavelength"])[ring], 0.0)


@partial(jax.jit, static_argnames=("free", "n_groups"))
def _normal(x: jax.Array, base: dict, free: tuple, sc: jax.Array, fc: jax.Array, w: jax.Array, group: jax.Array,
            ring: jax.Array, ds: jax.Array, n_groups: int) -> tuple:  # fmt: skip
    """(J^T J, J^T r, r^T r) of :func:`_residuals`."""

    def res(xx: jax.Array) -> jax.Array:
        return _residuals(xx, base, free, sc, fc, w, group, ring, ds, n_groups)

    r = res(x)
    J = jax.jacfwd(res)(x)
    return J.T @ J, J.T @ r, r @ r


def assign(px: dict, ds: np.ndarray, pars: dict, tol: float, n_eta: int = 360, sep: float | None = None) -> dict:
    """Assign each pixel to its nearest ring within tol (degrees 2theta), and to an eta sector.

    Rings closer than ``sep`` to a neighbour are left out: their pixels cannot be told apart, and each ring's
    profile pulls the other's mean.

    Parameters
    ----------
    px
        From :func:`lit_pixels`
    ds
        [Nr] d* of the rings, ascending (:func:`ring_ds`)
    pars
        Geometry (:func:`anri.calib.fit.base_pars`)
    tol
        Largest |2theta - ring| (degrees)
    n_eta
        Sectors over 360 degrees of eta
    sep
        Smallest gap (degrees 2theta) from a ring to its neighbours (default 2 tol)

    Returns
    -------
    dict
        "ring" [N] (-1: none), "group" [N] (ring x n_eta + sector, for every pixel), "tth", "eta" [N] (degrees) and
        "used" [Nr] (rings kept)
    """
    with float64():
        p = {k: jnp.float64(v) for k, v in pars.items()}
        tth, eta = (np.asarray(a) for a in pixel_tth_eta(jnp.asarray(px["sc"]), jnp.asarray(px["fc"]), p))
        t = np.asarray(_ring_tth(jnp.asarray(ds), p["wavelength"]))
    gap = np.diff(t)
    sep = 2 * tol if sep is None else sep
    used = np.ones(len(t), bool)
    used[:-1] &= gap > sep
    used[1:] &= gap > sep
    i = np.clip(np.searchsorted(t, tth), 1, len(t) - 1)
    near = np.where(np.abs(tth - t[i - 1]) < np.abs(tth - t[i]), i - 1, i)
    ring = np.where((np.abs(tth - t[near]) < tol) & used[near], near, -1)
    sector = np.clip(((eta + 180.0) * n_eta / 360.0).astype(np.int64), 0, n_eta - 1)
    return {"ring": ring, "group": np.maximum(ring, 0) * n_eta + sector, "tth": tth, "eta": eta, "used": used}


def ring_offsets(px: dict, ds: np.ndarray, pars: dict, tol: float, n_eta: int = 360,
                 sep: float | None = None) -> dict:  # fmt: skip
    """Each (ring, eta sector)'s mean 2theta offset from its ring: the numbers behind a "straightened rings" plot.

    Parameters
    ----------
    px, ds, pars, tol, n_eta, sep
        As :func:`assign`

    Returns
    -------
    dict
        "offset" [Nr, n_eta] (degrees, NaN where a sector has no pixels), "intensity" [Nr, n_eta] (summed),
        "tth" [Nr] (each ring's 2theta, degrees), "used" [Nr]
    """
    a = assign(px, ds, pars, tol, n_eta, sep)
    n = len(ds) * n_eta
    ok = a["ring"] >= 0
    W = np.bincount(a["group"][ok], weights=px["intensity"][ok], minlength=n)
    S = np.bincount(a["group"][ok], weights=(px["intensity"] * a["tth"])[ok], minlength=n)
    with float64():
        t = np.asarray(_ring_tth(jnp.asarray(ds), jnp.float64(pars["wavelength"])))
    off = np.where(W > 0, S / np.where(W > 0, W, 1.0) - np.repeat(t, n_eta), np.nan)
    return {"offset": off.reshape(-1, n_eta), "intensity": W.reshape(-1, n_eta), "tth": t, "used": a["used"]}


def fit_rings(px: dict, ds: np.ndarray, pars: dict, free: tuple = FREE, tols: tuple = (0.1, 0.05),
              n_iter: int = 30, n_eta: int = 360, clip: float = 4.0, log: Callable = print) -> dict:  # fmt: skip
    """Fit the geometry so that the rings fall at their 2theta: see the module's description.

    Parameters
    ----------
    px
        From :func:`lit_pixels`
    ds
        [Nr] d* of the rings, ascending (:func:`ring_ds`)
    pars
        Starting geometry (:func:`anri.calib.fit.base_pars`)
    free
        Free parameters, from :data:`anri.calib.fit.GEOMETRY` (not "lattice"); tilt_x moves no ring
    tols
        Ring assignment tolerance (degrees 2theta) of each round. Each sector's mean is over the pixels within it,
        so the last must still span a few pixels (one Eiger pixel at 150 mm is 0.03 degrees). The rings are those
        at least 2 max(tols) from their neighbours, in every round
    n_iter
        Gauss-Newton iterations per round at most (the pixels are re-assigned before each)
    n_eta
        Eta sectors over 360 degrees: one residual per (ring, sector)
    clip
        Leave out sectors further than this many (robust) standard deviations from their ring, from the second round
    log
        Progress messages

    Returns
    -------
    dict
        "pars" (fitted), "cov" (of the free parameters), "rms" (of the sector residuals, degrees), "n_sectors" (used
        in the last round), "rounds" (per round: tolerance, sectors, rings, rms, iterations)
    """
    n_groups = len(ds) * n_eta
    x = np.array([pars[k] for k in free], float)
    w_all = np.asarray(px["intensity"], float)
    rounds, H, cost, n_sec = [], np.zeros((len(free), len(free))), 0.0, 0
    with float64():
        base = {k: jnp.float64(v) for k, v in pars.items()}
        sc, fc = jnp.asarray(px["sc"]), jnp.asarray(px["fc"])
        dsj = jnp.asarray(ds)
        ring_of = jnp.asarray(np.repeat(np.arange(len(ds)), n_eta))

        def setup(x: np.ndarray, tol: float, clipped: bool) -> tuple:
            """Assign the pixels at x: the residuals' arguments, and the (ring, sector) groups used."""
            a = assign(px, ds, {**pars, **dict(zip(free, x))}, tol, n_eta, 2 * max(tols))
            ok = a["ring"] >= 0
            group = jnp.asarray(a["group"])
            if clipped:  # leave out sectors far from their ring (another phase, a spot on a ring's tail)
                r = np.asarray(_residuals(jnp.asarray(x), base, free, sc, fc, jnp.asarray(np.where(ok, w_all, 0.0)),
                                          group, ring_of, dsj, n_groups))  # fmt: skip
                nz = r != 0
                s = 1.4826 * np.median(np.abs(r[nz])) if nz.any() else 0.0
                ok &= (np.abs(r) <= clip * s)[a["group"]]
            args = (base, free, sc, fc, jnp.asarray(np.where(ok, w_all, 0.0)), group, ring_of, dsj, n_groups)
            return args, np.unique(a["group"][ok]), np.unique(a["ring"][ok]).size

        for ir, tol in enumerate(tols):
            mu, it = 1e-3, 0
            for it in range(1, n_iter + 1):  # re-assign at every step: the windows follow the rings
                args, used, _ = setup(x, tol, ir > 0)
                if len(used) <= len(free):
                    msg = f"round {ir + 1}: {len(used)} (ring, sector) groups within {tol} deg of a ring: too few"
                    raise ValueError(msg)
                H, g, cost = (np.asarray(v) for v in _normal(jnp.asarray(x), *args))
                D = np.diag(np.maximum(np.diag(H), 1e-30 * np.diag(H).max()))
                while True:
                    dx = np.linalg.solve(H + mu * D, -g)
                    r = np.asarray(_residuals(jnp.asarray(x + dx), *args))
                    if float(r @ r) < cost or mu > 1e10:
                        break
                    mu *= 10
                if float(r @ r) >= cost:
                    break  # no step lowers the cost
                x, mu = x + dx, max(mu / 10, 1e-12)
                err = np.sqrt(np.diag(np.linalg.pinv(H)) * cost / max(len(used) - len(free), 1))
                if np.all(np.abs(dx) < 0.1 * err):
                    break  # steps far below the errors: converged
            args, used, n_rings = setup(x, tol, ir > 0)
            H, _, cost = (np.asarray(v) for v in _normal(jnp.asarray(x), *args))
            cost, n_sec = float(cost), len(used)
            rms = float(np.sqrt(cost / n_sec))
            rounds.append({"tol": tol, "n_sectors": n_sec, "n_rings": int(n_rings), "rms": rms, "n_iter": it})
            log(f"round {ir + 1}: tol {tol:g} deg, {it} iterations, {n_rings} rings, {n_sec} sectors, rms "
                f"{rms:.2e} deg; " + ", ".join(f"{k} {v:.7g}" for k, v in zip(free, x)))  # fmt: skip
    cov = np.linalg.pinv(H) * cost / max(n_sec - len(free), 1)
    return {"pars": {**pars, **{k: float(v) for k, v in zip(free, x)}}, "cov": cov,
            "rms": rounds[-1]["rms"], "n_sectors": n_sec, "rounds": rounds}  # fmt: skip
