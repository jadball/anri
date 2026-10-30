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
re-assigned at every iteration.

A ring assignment only works from a start within about one ring gap; the motors' centre can be 100 pixels off. So
:func:`find_centre` and :func:`find_distance` first search without assigning rings: the centre where the radial
histogram is sharpest (every ring a circle about it), then the distance whose ring windows catch the most
intensity.
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


def _angles(px: dict, pars: dict) -> tuple:
    """(2theta, eta) of the pixels (degrees, NumPy float64) for the geometry pars."""
    with float64():
        p = {k: jnp.float64(v) for k, v in pars.items()}
        return tuple(np.asarray(a) for a in pixel_tth_eta(jnp.asarray(px["sc"]), jnp.asarray(px["fc"]), p))


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
        Largest ``|2theta - ring|`` (degrees)
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
    tth, eta = _angles(px, pars)
    with float64():
        t = np.asarray(_ring_tth(jnp.asarray(ds), jnp.float64(pars["wavelength"])))
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


@partial(jax.jit, static_argnames=("n_r",))
def _sharpness(centres: jax.Array, sc: jax.Array, fc: jax.Array, w: jax.Array, size: jax.Array, rbin: float,
               n_r: int) -> jax.Array:  # fmt: skip
    """[C] sum of squares of the radial histogram about each centre [C, 2] (z_center, y_center; pixels)."""

    def one(c: jax.Array) -> jax.Array:
        r = jnp.hypot((sc - c[0]) * size[0], (fc - c[1]) * size[1])
        H = jax.ops.segment_sum(w, jnp.clip((r / rbin).astype(jnp.int32), 0, n_r - 1), n_r)
        return jnp.sum(H * H)

    return jax.lax.map(one, centres)


def find_centre(px: dict, pars: dict, max_shift: float = 200.0, max_pixels: int = 200_000,
                log: Callable = print) -> tuple:  # fmt: skip
    """Find the beam centre without assigning rings: where the rings' radial histogram is sharpest.

    About the right centre every ring is a circle (to the detector tilts' small ellipticity), so the histogram of
    the pixels' radii, weighted by the square root of their intensity (so a few strong spots do not decide), is most
    peaked: its sum of squares is largest, whatever the distance and the rings. The centre is searched on grids
    of decreasing step (8, 2, 0.5 pixels; bins of half a step), each around the best of the last, the first
    reaching ``max_shift`` pixels.

    Parameters
    ----------
    px
        From :func:`lit_pixels`
    pars
        Starting geometry (:func:`anri.calib.fit.base_pars`); only y_center and z_center change. The beam is at
        pixel (slow, fast) = (z_center, y_center), as in ImageD11
    max_shift
        Largest distance (pixels) from the start searched
    max_pixels
        Use at most this many pixels, spread evenly (the brightest are not preferred)
    log
        Progress messages

    Returns
    -------
    pars: dict
        With y_center and z_center found
    diag: dict
        "start" (z_center, y_center) and "levels": per grid, "step" (pixels), "centres" [C, 2] (z_center,
        y_center) and "sharpness" [C]
    """
    take = np.unique(np.linspace(0, len(px["sc"]) - 1, min(max_pixels, len(px["sc"]))).astype(np.int64))
    sc, fc = jnp.asarray(px["sc"][take], jnp.float32), jnp.asarray(px["fc"][take], jnp.float32)
    w = jnp.asarray(np.sqrt(np.maximum(px["intensity"][take], 0.0)), jnp.float32)
    size = jnp.asarray([pars["z_size"], pars["y_size"]], jnp.float32)
    pix = float(min(pars["z_size"], pars["y_size"]))
    r_max = float(np.hypot(np.ptp(px["sc"]) + 2 * max_shift, np.ptp(px["fc"]) + 2 * max_shift)) * pix
    best = np.array([pars["z_center"], pars["y_center"]])
    diag: dict = {"start": best.copy(), "levels": []}
    for step, half in ((8.0, max_shift), (2.0, 8.0), (0.5, 2.0)):
        g = np.arange(-half, half + step / 2, step)
        grid = best + np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        rbin = step / 2 * pix
        h = np.asarray(_sharpness(jnp.asarray(grid, jnp.float32), sc, fc, w, size, rbin, int(r_max / rbin) + 1))
        best = grid[np.argmax(h)]
        diag["levels"].append({"step": step, "centres": grid, "sharpness": h})
        log(f"centre search: {len(grid)} centres {step:g} px apart; best z_center {best[0]:.2f}, y_center "
            f"{best[1]:.2f} (sharpness {h.max() / np.median(h):.3g} x the median)")  # fmt: skip
    return {**pars, "z_center": float(best[0]), "y_center": float(best[1])}, diag


def find_distance(px: dict, ds: np.ndarray, pars: dict, tol: float = 0.1, span: tuple = (0.7, 1.4),
                  log: Callable = print) -> tuple:  # fmt: skip
    """Find the distance without assigning rings: the one whose ring windows catch the most intensity.

    With the centre found (:func:`find_centre`), a pixel at 2theta seen from distance L0 is at
    tan(2theta') = tan(2theta) L0 / L from distance L. For each L of a grid over ``span`` x L0 (steps moving the
    outermost ring by a quarter of tol), the score is the intensity (square root, so a few strong spots do not
    decide) within tol of a ring, less what a flat background would put in the same windows.

    Parameters
    ----------
    px, ds, pars
        As :func:`assign`
    tol
        Ring window (degrees 2theta)
    span
        Range searched, as factors of the starting distance
    log
        Progress messages

    Returns
    -------
    pars: dict
        With the distance found
    diag: dict
        "distance" [L] (the distances tried) and "score" [L]
    """
    tth, _ = _angles(px, pars)
    w = np.sqrt(np.maximum(px["intensity"], 0.0))
    u = np.tan(np.radians(tth))  # radius / L0
    order = np.argsort(u)
    u, C = u[order], np.concatenate([[0.0], np.cumsum(w[order])])
    t_ring = np.radians(np.asarray(_ring_tth(jnp.asarray(ds), pars["wavelength"]), float))
    dstep = np.radians(tol) / 4 / max(np.radians(tth.max()), 1e-6)
    scale = np.exp(np.arange(np.log(span[0]), np.log(span[1]), dstep))  # L / L0
    density = C[-1] / (u[-1] - u[0])  # a flat background's intensity per unit u
    tol_r = np.radians(tol)
    score = np.zeros(len(scale))
    for i, f in enumerate(scale):  # a ring at 2theta_r falls at u = tan(2theta_r) L / L0
        lo, hi = np.tan(t_ring - tol_r) * f, np.tan(t_ring + tol_r) * f
        inside = (lo > u[0]) & (hi < u[-1])
        caught = C[np.searchsorted(u, hi[inside])] - C[np.searchsorted(u, lo[inside])]
        score[i] = np.sum(caught - density * (hi[inside] - lo[inside]))
    best = int(np.argmax(score))
    p = {**pars, "distance": pars["distance"] * scale[best]}
    second = np.max(np.where(np.abs(np.arange(len(scale)) - best) > 8, score, -np.inf))
    log(f"distance search: {len(scale)} distances {span[0]:g}-{span[1]:g} x {pars['distance']:.6g}; best "
        f"{p['distance']:.6g} (x {scale[best]:.4f}), score {score[best]:.4g}, next best elsewhere {second:.4g}")  # fmt: skip
    return p, {"distance": pars["distance"] * scale, "score": score}


def fit_rings(px: dict, ds: np.ndarray, pars: dict, free: tuple = FREE, tols: tuple = (0.2, 0.1),
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
