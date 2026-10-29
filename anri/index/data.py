"""Coarse views of the data for indexing: rings, pixel angles, histograms and the lit map.

The sparse pixels are reduced once to a histogram ``H[ring, eta, omega, row]`` at the scale of the orientation grid
(a degree or so), and to a row-summed "lit" map of where there is intensity. Everything after that works on these.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from Dans_Diffraction.classes_crystal import Crystal as DansCrystal
from jax.typing import ArrayLike

from anri.crystal.structure import reflections, rings, structure_factors
from anri.diffract.scatter import ds_to_tth, k_out_to_tth_eta
from anri.geom.gonio import wrap_omega
from anri.geom.scan import dty_to_row
from anri.maths.utils import float64


def ring_table(
    lattice_parameters: ArrayLike,
    space_group: int,
    wavelength: float,
    n_rings: int | Iterable[int],
    structure: DansCrystal | None = None,
    merge: float = 1e-4,
) -> dict:
    """List the reflections of a crystal's rings, with their structure factors.

    Reflections within ``merge`` in d* are one ring (:func:`anri.crystal.structure.rings`): rings closer than the data
    can separate should be merged, not split between two rings. Systematic absences are removed
    (:func:`anri.crystal.structure.reflections`). Each reflection has two predictions, one per omega solution, indexed
    ``j = 2 h + branch``. With a structure (``Dans_Diffraction.Crystal``, e.g. from a CIF), each reflection gets its
    ``|F|^2`` (:func:`anri.crystal.structure.structure_factors`), scaled to a mean of 1; without one, ``|F|^2 = 1``. The
    lattice parameters given set the reflections and 2theta, so a refined lattice can be used with a CIF's atoms.

    Parameters
    ----------
    lattice_parameters
        [6] a, b, c (angstrom), alpha, beta, gamma (degrees)
    space_group
        Space-group number
    wavelength
        In angstrom
    n_rings
        How many rings, from the lowest 2theta; or which rings, as indices from the lowest 2theta (0 first). For low
        symmetry, pick strong rings well apart from their neighbours
    structure
        Optional: the same phase with its atoms, for ``|F|^2``
    merge
        Largest d* difference within a ring (1 / angstrom)

    Returns
    -------
    dict
        "hkls" [Nh, 3], "F2" [Nh], "ring_j" [2 Nh] (ring of each prediction, 0 to Nr - 1 in the order of "index"),
        "tth" [Nr] (degrees), "index" [Nr] (each ring's index among all rings), "tth_all" (degrees) and "ds_all"
        (1 / angstrom) of every ring up to the last one used, so that widths can be measured between true neighbours
    """
    index = np.arange(n_rings) if isinstance(n_rings, (int, np.integer)) else np.unique(np.asarray(list(n_rings), int))
    if len(index) == 0 or index[0] < 0:
        raise ValueError(f"rings must be a positive number or indices >= 0, not {n_rings}")
    dsmax = 0.5
    while True:  # enough d* range for the last ring used
        refl = reflections(lattice_parameters, space_group, wavelength, dsmax)
        ring, ring_ds = rings(refl["ds"], merge)
        if len(ring_ds) > index[-1] + 1 or dsmax >= 2 / wavelength:
            break
        dsmax *= 1.5
    if index[-1] >= len(ring_ds):
        raise ValueError(f"ring {index[-1]} asked for, but there are only {len(ring_ds)} rings")
    sel = np.isin(ring, index)
    hkls = refl["hkl"][sel]
    F2 = np.ones(len(hkls))
    if structure is not None:
        F2 = structure_factors(structure, hkls, wavelength)
        F2 = F2 / max(F2.mean(), 1e-30)
    with float64():
        tth_all = np.asarray(ds_to_tth(ring_ds[: index[-1] + 1], wavelength))
    return {
        "hkls": hkls.astype(np.float32),
        "F2": F2.astype(np.float32),
        "ring_j": np.repeat(np.searchsorted(index, ring[sel]), 2).astype(np.int32),
        "tth": tth_all[index].astype(np.float32),
        "index": index.astype(np.int32),
        "tth_all": tth_all.astype(np.float32),
        "ds_all": ring_ds[: index[-1] + 1].astype(np.float32),
    }


@jax.jit
def pixel_angles(slow: jax.Array, fast: jax.Array, omega: jax.Array, geom: dict) -> jax.Array:
    """Compute (2theta, eta, omega) of pixel centres, seen from the lab origin (no parallax).

    Parameters
    ----------
    slow, fast
        [N] pixel coordinates
    omega
        [N] omega of each pixel's frame (degrees)
    geom
        Geometry dict with "det_origin_lab", "s_step_lab", "f_step_lab" and "k_in_lab", e.g. from
        :func:`anri.io.imaged11.geom_from_pars`

    Returns
    -------
    jax.Array
        [N, 3] (2theta, eta, omega) in degrees
    """
    p = geom["det_origin_lab"] + slow[:, None] * geom["s_step_lab"] + fast[:, None] * geom["f_step_lab"]
    tth, eta = k_out_to_tth_eta(p, geom["k_in_lab"])
    return jnp.stack([tth, eta, omega], 1)


@partial(jax.jit, static_argnames=("n_t", "n_eta"))
def cake(x: jax.Array, val: jax.Array, lo: float, step: float, n_t: int, n_eta: int = 1) -> jax.Array:
    """Sum pixel intensities in (2theta, eta) bins: a cake. With n_eta = 1, the 2theta profile.

    Parameters
    ----------
    x
        [N, 3] (2theta, eta, omega) from :func:`pixel_angles`
    val
        [N] intensities
    lo, step, n_t
        2theta bins: n_t of width step from lo (degrees)
    n_eta
        eta bins over 360 degrees from -180

    Returns
    -------
    jax.Array
        [n_t, n_eta] intensity per bin
    """
    it = jnp.floor((x[:, 0] - lo) / step).astype(jnp.int32)
    ie = jnp.floor((x[:, 1] + 180.0) * n_eta / 360.0).astype(jnp.int32) % n_eta
    ok = (it >= 0) & (it < n_t)
    idx = jnp.where(ok, it * n_eta + ie, n_t * n_eta)
    return jax.ops.segment_sum(jnp.where(ok, val, 0.0), idx, n_t * n_eta + 1)[:-1].reshape(n_t, n_eta)


def pixel_cake(
    chunks: Iterable, geom: dict, lo: float, step: float, n_t: int, chunk: int, n_eta: int = 1, counts: bool = False
) -> np.ndarray:
    """Sum a stream of pixels into a (2theta, eta) cake (:func:`cake`), a chunk at a time.

    Parameters
    ----------
    chunks
        Iterable of (slow, fast, omega, row, value) NumPy arrays, e.g. a few scans of :func:`anri.io.imaged11.stream_sparse`
    geom
        Geometry dict, see :func:`pixel_angles`
    lo, step, n_t, n_eta
        The bins, see :func:`cake`
    chunk
        Pixels per call (every call is padded to it, so the jitted functions compile once)
    counts
        Also count the pixels in each bin

    Returns
    -------
    np.ndarray
        [n_t, n_eta] intensity per bin; with counts, [2, n_t, n_eta]: intensity and pixels per bin
    """
    total = jnp.zeros((n_t, n_eta), jnp.float32)
    n_pix = jnp.zeros((n_t, n_eta), jnp.float32)
    for item in chunks:
        slow, fast, omega, _, value = item[:5]
        m = len(value)

        def pad(a: ArrayLike, m: int = m) -> jax.Array:
            return jnp.asarray(np.pad(np.asarray(a, np.float32), (0, chunk - m)))

        x = pixel_angles(pad(slow), pad(fast), pad(omega), geom)
        total = total + cake(x, pad(value), lo, step, n_t, n_eta)
        if counts:
            n_pix = n_pix + cake(x, pad(np.ones(m)), lo, step, n_t, n_eta)
    return np.stack([np.asarray(total), np.asarray(n_pix)]) if counts else np.asarray(total)


@jax.jit
def pixel_solid_angle(slow: jax.Array, fast: jax.Array, geom: dict) -> jax.Array:
    """Solid angle (sr) of pixels, seen from the lab origin: area x cos(incidence) / distance^2.

    Parameters
    ----------
    slow, fast
        [N] pixel coordinates (of the pixel centre)
    geom
        Geometry dict, see :func:`pixel_angles`

    Returns
    -------
    jax.Array
        [N] solid angle of each pixel
    """
    p = geom["det_origin_lab"] + slow[:, None] * geom["s_step_lab"] + fast[:, None] * geom["f_step_lab"]
    n = jnp.cross(geom["s_step_lab"], geom["f_step_lab"])  # normal, |n| = pixel area
    return jnp.abs(p @ n) / jnp.linalg.norm(p, axis=1) ** 3


def detector_cake(
    good: ArrayLike, geom: dict, lo: float, step: float, n_t: int, n_eta: int, spatial: tuple | None = None,
    rows: int = 256,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Solid angle and number of the measured pixels in each (2theta, eta) bin, binned like the data.

    Each pixel counts in the bin of its centre after detector distortion (as :func:`anri.io.imaged11.stream_sparse`
    yields the data), so a bin's intensity (:func:`pixel_cake`) divided by its solid angle is the intensity per
    steradian, whatever the bins' size.

    Parameters
    ----------
    good
        [n_slow, n_fast] True where a pixel is measured (:func:`anri.io.imaged11.read_mask`; all True for no mask)
    geom
        Geometry dict, see :func:`pixel_angles`
    lo, step, n_t, n_eta
        The bins, see :func:`cake`
    spatial
        Detector distortion (dx, dy) as :func:`anri.io.imaged11.read_spatial`, or None
    rows
        Detector rows per call (the last call is padded, so the jitted functions compile once)

    Returns
    -------
    solid_angle, n_pixels: np.ndarray
        [n_t, n_eta] each
    """
    good = np.asarray(good, bool)
    n_s, n_f = good.shape
    om = jnp.zeros((n_t, n_eta), jnp.float32)
    npx = jnp.zeros((n_t, n_eta), jnp.float32)
    zeros = jnp.zeros(rows * n_f, jnp.float32)
    for s0 in range(0, n_s, rows):
        s, f = np.mgrid[s0 : s0 + rows, 0:n_f]
        inside = s < n_s
        s = np.minimum(s, n_s - 1)
        g = (good[s, f] & inside).ravel().astype(np.float32)
        ps, pf = s.ravel().astype(np.float32), f.ravel().astype(np.float32)
        if spatial is not None:
            ps, pf = ps + spatial[1][s, f].ravel(), pf + spatial[0][s, f].ravel()
        ps, pf = jnp.asarray(ps), jnp.asarray(pf)
        x = pixel_angles(ps, pf, zeros, geom)
        om = om + cake(x, pixel_solid_angle(ps, pf, geom) * g, lo, step, n_t, n_eta)
        npx = npx + cake(x, jnp.asarray(g), lo, step, n_t, n_eta)
    return np.asarray(om, np.float64), np.asarray(npx, np.float64)


def powder_pattern(
    intensity: ArrayLike, solid_angle: ArrayLike, n_pixels: ArrayLike, hits: ArrayLike, n_frames: int, lo: float,
    step: float,
) -> dict:  # fmt: skip
    """Make a powder pattern from a (2theta, eta) cake of all frames: intensity per steradian per frame, and its esd.

    The eta average of the intensity per solid angle is the azimuthal average of a 2D powder pattern, so the usual
    Debye-Scherrer Lorentz factor applies to it (no Lorentz or polarisation correction is made). Summed over a
    whole scan the grains act as a powder, but a coarse one: each eta bin is an independent estimate of the ring's
    average, so the esd is the spread over eta of the solid-angle-weighted mean, sqrt(n / (n - 1) sum w^2 (v - y)^2)
    with w the bins' share of the solid angle, never below the counting esd sqrt(sum I) / (solid angle x frames).

    Parameters
    ----------
    intensity
        [n_t, n_eta] summed intensity in each bin (:func:`pixel_cake`)
    solid_angle, n_pixels
        [n_t, n_eta] solid angle (sr) and number of the measured detector pixels in each bin (:func:`detector_cake`)
    hits
        [n_t, n_eta] number of data pixels in each bin (:func:`pixel_cake` with counts)
    n_frames
        Frames summed
    lo, step
        2theta bins (deg)

    Returns
    -------
    dict
        [n_t] each: "tth" (bin centres, deg), "intensity" (per sr per frame; 0 where nothing is measured), "sigma"
        (over eta), "sigma_counting", "coverage" (measured share of the full ring's solid angle), "lit_fraction"
        (share of the measured pixel-frames holding a data pixel: sparse data keep only pixels above the
        segmentation cut, so a bin lit in very few of them is mostly censored)
    """
    I, Om = np.asarray(intensity, float), np.asarray(solid_angle, float)
    n_t = I.shape[0]
    t0 = np.radians(lo + step * np.arange(n_t))
    ring = 2.0 * np.pi * (np.cos(t0) - np.cos(t0 + np.radians(step)))  # the full ring's solid angle
    S = Om.sum(1)
    ok = S > 0
    norm = np.where(ok, S, 1.0) * n_frames
    y = np.where(ok, I.sum(1) / norm, 0.0)
    sig_c = np.where(ok, np.sqrt(np.maximum(I.sum(1), 1.0)) / norm, 0.0)
    live = Om > 0
    n = live.sum(1)
    v = np.where(live, I / np.where(live, Om, 1.0), 0.0) / n_frames
    w = Om / np.where(ok, S, 1.0)[:, None]
    var = n / np.maximum(n - 1, 1) * np.sum(w**2 * (v - y[:, None]) ** 2 * live, 1)
    sig = np.where(n > 1, np.maximum(np.sqrt(var), sig_c), sig_c)
    npx = np.asarray(n_pixels, float).sum(1)
    lit = np.where(npx > 0, np.asarray(hits, float).sum(1) / (np.maximum(npx, 1.0) * n_frames), 0.0)
    return {"tth": np.degrees(t0) + 0.5 * step, "intensity": y, "sigma": sig, "sigma_counting": sig_c,
            "coverage": S / ring, "lit_fraction": lit}  # fmt: skip


def ring_widths(
    prof: ArrayLike, lo: float, step: float, ring_tth: ArrayLike, frac: float = 0.95, max_hw: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Measure each ring's offset and half-width (degrees 2theta) from a 2theta profile.

    Each ring is looked at within half the gap to its neighbours (at most ``max_hw``). Background: the median of the
    outer fifth of that window. Offset: the centroid minus the ring's 2theta. Half-width: the distance from the
    centroid that holds ``frac`` of the ring's net intensity. It sums everything that spreads a ring: parallax
    (sample size / distance), strain, peak size and detector distortion. A ring without intensity gets offset 0 and
    the whole window.

    Parameters
    ----------
    prof
        Intensity per 2theta bin, e.g. :func:`pixel_cake` with one eta bin
    lo, step
        The bins
    ring_tth
        [Nr] 2theta of the rings
    frac
        Fraction of a ring's intensity within the half-width
    max_hw
        Largest half-width considered

    Returns
    -------
    offset, half_width: np.ndarray
        [Nr] each, in degrees
    """
    prof, ring_tth = np.asarray(prof), np.asarray(ring_tth, float)
    x = lo + (np.arange(len(prof)) + 0.5) * step
    gaps = np.diff(ring_tth)
    half = np.minimum(np.concatenate([[2 * max_hw], gaps]), np.concatenate([gaps, [2 * max_hw]])) / 2
    off, hw = np.zeros(len(ring_tth)), np.zeros(len(ring_tth))
    for r, t in enumerate(ring_tth):
        m = np.abs(x - t) < half[r]
        xr, pr = x[m], prof[m]
        outer = np.abs(xr - t) > 0.8 * half[r]
        net = np.maximum(pr - (np.median(pr[outer]) if outer.any() else 0.0), 0.0)
        if net.sum() <= 0:
            hw[r] = half[r]
            continue
        c = np.sum(xr * net) / net.sum()
        d = np.abs(xr - c)
        o = np.argsort(d)
        k = np.searchsorted(np.cumsum(net[o]), frac * net.sum())
        off[r], hw[r] = c - t, d[o][min(k, len(o) - 1)] + step / 2
    return off, hw


def detector_coverage(
    good: ArrayLike, geom: dict, ring_tth: ArrayLike, ring_tol: ArrayLike, b_e: float, n_e: int,
    spatial: tuple | None = None, sub: int = 4, rows: int = 64,
) -> np.ndarray:  # fmt: skip
    """Fraction of each ring's detector area that is measured, per eta bin: 0 off the detector and in masked pixels.

    Each pixel is sampled at sub x sub points (as the data, after detector distortion), and every point within a
    ring's 2theta tolerance counts towards that ring's eta bin; the coverage is the share of them on measured pixels.
    A reflection predicted there is seen with this fraction of its intensity on average.

    Parameters
    ----------
    good
        [n_slow, n_fast] True where a pixel is measured (:func:`anri.io.imaged11.read_mask`; all True for no mask)
    geom
        Geometry dict, see :func:`pixel_angles`
    ring_tth, ring_tol
        [Nr] each: the rings' 2theta and tolerance (degrees), as the histograms bin them
    b_e, n_e
        The eta bins
    spatial
        Detector distortion (dx, dy) as :func:`anri.io.imaged11.read_spatial`, or None
    sub
        Sample points per pixel along each axis
    rows
        Detector rows per pass

    Returns
    -------
    np.ndarray
        [Nr, n_e] coverage in [0, 1]
    """
    good = np.asarray(good, bool)
    ring_tth, ring_tol = np.asarray(ring_tth, float), np.asarray(ring_tol, float)
    n_r, (n_s, n_f) = len(ring_tth), good.shape
    num, den = np.zeros(n_r * n_e), np.zeros(n_r * n_e)
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    for s0 in range(0, n_s, rows):
        s, f = np.mgrid[s0 : min(s0 + rows, n_s), 0:n_f]
        g = good[s, f].ravel().astype(float)
        ps, pf = s.ravel().astype(np.float32), f.ravel().astype(np.float32)
        if spatial is not None:  # where the pixel really is, as the data (anri.io.imaged11.stream_sparse)
            ps, pf = ps + spatial[1][s, f].ravel(), pf + spatial[0][s, f].ravel()
        for a in offs:
            for b in offs:
                x = np.asarray(pixel_angles(jnp.asarray(ps + a), jnp.asarray(pf + b), jnp.zeros(len(ps)), geom))
                ie = np.floor((x[:, 1] + 180.0) / b_e).astype(np.int64) % n_e
                for r in range(n_r):
                    m = np.abs(x[:, 0] - ring_tth[r]) < ring_tol[r]
                    num += np.bincount(r * n_e + ie[m], g[m], n_r * n_e)
                    den += np.bincount(r * n_e + ie[m], None, n_r * n_e)
    return np.where(den > 0, num / np.maximum(den, 1), 0.0).reshape(n_r, n_e)


def coarsen_coverage(cov: ArrayLike, n_e: int) -> np.ndarray:
    """Average a [Nr, m n_e] coverage (:func:`detector_coverage`) into n_e eta bins."""
    cov = np.asarray(cov)
    return cov.reshape(cov.shape[0], n_e, -1).mean(2)


def eta_bin(eta: jax.Array, b_e: float, n_e: int) -> jax.Array:
    """Return the histogram's eta bin of each angle (degrees): bins of b_e, periodic over 360 degrees from -180."""
    return jnp.floor((eta + 180.0) / b_e).astype(jnp.int32) % n_e


def omega_bin(om: jax.Array, om0: ArrayLike, b_o: float) -> jax.Array:
    """Return the histogram's omega bin of each angle (degrees): bins of b_o from om0, int32, possibly outside [0, n_o).

    Not wrapped here: wrap with :func:`anri.geom.gonio.wrap_omega` first where that is wanted.
    """
    return jnp.floor((om - om0) / b_o).astype(jnp.int32)


def eta_omega_corners(
    eta: jax.Array, om: jax.Array, om0: ArrayLike, b_e: float, b_o: float, n_e: int, n_o: int
) -> list:
    """Return the 2 x 2 (eta, omega) bins around each angle and their bilinear weights (bin centres on the grid).

    Omega is wrapped into [om0, om0 + 360) (:func:`anri.geom.gonio.wrap_omega`) first.

    Returns
    -------
    list
        Four (ie, io, w_eta, w_omega, inside) tuples: bin indices, the weights along eta and omega (multiply them in),
        and whether io is in [0, n_o)
    """
    fe = (eta + 180.0) / b_e - 0.5
    fo = (wrap_omega(om, om0) - om0) / b_o - 0.5
    e0, o0 = jnp.floor(fe).astype(jnp.int32), jnp.floor(fo).astype(jnp.int32)
    te, to = fe - e0, fo - o0
    corners = []
    for de, we in ((0, 1.0 - te), (1, te)):
        for do, wo in ((0, 1.0 - to), (1, to)):
            io = o0 + do
            corners.append(((e0 + de) % n_e, io, we, wo, (io >= 0) & (io < n_o)))
    return corners


@partial(jax.jit, static_argnames=("n_ring", "n_e", "n_o", "n_k"))
def histogram(
    x: jax.Array,
    val: jax.Array,
    row: jax.Array,
    ring_tth: jax.Array,
    tth_tol: ArrayLike,
    om0: float,
    b_e: float,
    b_o: float,
    n_ring: int,
    n_e: int,
    n_o: int,
    n_k: int,
) -> jax.Array:
    """Bin pixels into ``H[ring, eta, omega, row]`` (flattened).

    A pixel goes to the nearest ring if within ``tth_tol`` of it; eta bins are periodic over 360 degrees from -180,
    omega bins start at ``om0``. Pixels with row -1 are dropped.

    Parameters
    ----------
    x
        [N, 3] (2theta, eta, omega) from :func:`pixel_angles`
    val, row
        [N] intensity and dty row of each pixel
    ring_tth
        [n_ring] 2theta of the rings
    tth_tol
        2theta tolerance (degrees), a scalar or one per ring
    om0, b_e, b_o
        First omega bin edge, and the bin widths in eta and omega (degrees)
    n_ring, n_e, n_o, n_k
        Numbers of rings, eta bins, omega bins and rows

    Returns
    -------
    jax.Array
        [n_ring * n_e * n_o * n_k] intensity per bin
    """
    i = jnp.clip(jnp.searchsorted(ring_tth, x[:, 0]), 1, max(n_ring - 1, 1))
    ring = jnp.where(jnp.abs(x[:, 0] - ring_tth[i - 1]) < jnp.abs(x[:, 0] - ring_tth[i % n_ring]), i - 1, i % n_ring)
    ok = jnp.abs(x[:, 0] - ring_tth[ring]) < jnp.broadcast_to(tth_tol, ring_tth.shape)[ring]
    ie, io = eta_bin(x[:, 1], b_e, n_e), omega_bin(x[:, 2], om0, b_o)
    ok = ok & (io >= 0) & (io < n_o) & (row >= 0) & (row < n_k)
    idx = jnp.where(ok, ((ring * n_e + ie) * n_o + io) * n_k + row, n_ring * n_e * n_o * n_k)
    return jax.ops.segment_sum(jnp.where(ok, val, 0.0), idx, n_ring * n_e * n_o * n_k + 1)[:-1]


def histogram_pixels(
    chunks: Iterable,
    geom: dict,
    ring_tth: ArrayLike,
    tth_tol: ArrayLike,
    om0: float,
    histograms: list,
    chunk: int,
    saturated_into: int | None = None,
) -> list:
    """Stream pixels into one or more histograms in a single pass over the data.

    Each chunk's pixel angles are computed once and added to every histogram, so the data are read once however
    many histograms (e.g. a fine row-summed lit map and a coarse per-row one) are wanted.

    Parameters
    ----------
    chunks
        Iterable of (slow, fast, omega, row, value) NumPy arrays of at most ``chunk`` pixels each, e.g.
        :func:`anri.io.imaged11.stream_sparse`
    geom
        Geometry dict, see :func:`pixel_angles`
    ring_tth, tth_tol
        See :func:`histogram`
    om0
        First omega bin edge
    histograms
        One ``((b_e, b_o, n_e, n_o), n_rows)`` per histogram: eta and omega bin widths (degrees) and counts, and the
        number of rows (1 sums them all)
    chunk
        Pixels per call (chunks are padded to it, so one compile)
    saturated_into
        With chunks that flag saturated pixels (``stream_sparse(saturation=...)``): also count them, in the layout
        of this entry of ``histograms``

    Returns
    -------
    list
        One flat histogram [n_rings * n_e * n_o * n_rows] per entry of ``histograms``, and with ``saturated_into``
        the saturated pixels per bin last
    """
    ring_tth = jnp.asarray(ring_tth, jnp.float32)
    tth_tol = jnp.asarray(tth_tol, jnp.float32)
    n_ring = ring_tth.shape[0]
    out = [jnp.zeros(n_ring * b[2] * b[3] * n, jnp.float32) for b, n in histograms]
    if saturated_into is not None:
        (b_s, n_s), sat = histograms[saturated_into], None
        sat = jnp.zeros(n_ring * b_s[2] * b_s[3] * n_s, jnp.float32)
    zeros = jnp.zeros(chunk, jnp.int32)
    for item in chunks:
        slow, fast, omega, row, value = item[:5]
        m = len(value)

        def pad(a: ArrayLike, dt: type = np.float32, fill: float = 0, m: int = m) -> jax.Array:
            # a fresh array per chunk: a reused buffer can be overwritten while JAX still copies it (asynchronously)
            b = np.full(chunk, fill, dt)
            b[:m] = a
            return jnp.asarray(b)

        x = pixel_angles(pad(slow), pad(fast), pad(omega), geom)
        v, rows = pad(value), pad(row, np.int32, -1)
        for i, ((b_e, b_o, n_e, n_o), n_rows) in enumerate(histograms):
            r = rows if n_rows > 1 else jnp.where(rows >= 0, zeros, -1)
            out[i] = out[i] + histogram(x, v, r, ring_tth, tth_tol, om0, b_e, b_o, n_ring, n_e, n_o, n_rows)
        if saturated_into is not None and len(item) > 5:
            r = rows if n_s > 1 else jnp.where(rows >= 0, zeros, -1)
            sat = sat + histogram(x, pad(item[5]), r, ring_tth, tth_tol, om0, *b_s[:2], n_ring, *b_s[2:], n_s)
    return out if saturated_into is None else [*out, sat]


def ring_profile(chunks: Iterable, geom: dict, ring_tth: ArrayLike, chunk: int, step: float = 0.002) -> tuple:
    """Measure the rings' offsets and half-widths from a sample of pixels (see :func:`ring_widths`).

    Parameters
    ----------
    chunks
        Iterable of (slow, fast, omega, row, value) NumPy arrays, e.g. a few rows of :func:`anri.io.imaged11.stream_sparse`
    geom
        Geometry dict, see :func:`pixel_angles`
    ring_tth
        [Nr] 2theta of the rings
    chunk
        Pixels per call
    step
        2theta bin width (degrees)

    Returns
    -------
    offset, half_width: np.ndarray
        [Nr] each, in degrees
    """
    ring_tth = np.asarray(ring_tth, float)
    lo = float(ring_tth[0]) - 0.5
    n = int(np.ceil((float(ring_tth[-1]) + 0.5 - lo) / step))
    return ring_widths(pixel_cake(chunks, geom, lo, step, n, chunk)[:, 0], lo, step, ring_tth)


@jax.jit
def lit_table(lit: jax.Array) -> jax.Array:
    """Summed-area table of a lit map, for counting lit bins in any box in constant time.

    Parameters
    ----------
    lit
        [Nr, n_e, n_o] bool, where there is intensity

    Returns
    -------
    jax.Array
        [Nr, 3 n_e + 1, n_o + 1] int32, with eta tiled three times (it is periodic)
    """
    t = jnp.concatenate([lit, lit, lit], 1).astype(jnp.int32)
    return jnp.pad(jnp.cumsum(jnp.cumsum(t, 1), 2), ((0, 0), (1, 0), (1, 0)))


def coarsen_rows(H: ArrayLike, n_rows: int, g: int) -> tuple[jax.Array, int]:
    """Sum a histogram's rows in groups of g: row k goes to k // g.

    Parameters
    ----------
    H
        Flat histogram whose last axis is the n_rows rows
    n_rows
        Rows in H
    g
        Group size

    Returns
    -------
    H: jax.Array
        Flat coarse histogram
    n_rows: int
        Its number of rows, ceil(n_rows / g)
    """
    d = jnp.asarray(H).reshape(-1, n_rows)
    d = jnp.pad(d, ((0, 0), (0, -n_rows % g)))
    return d.reshape(d.shape[0], -1, g).sum(-1).ravel(), -(-n_rows // g)


def scan_beam(scan: dict) -> tuple[jax.Array, ArrayLike, ArrayLike]:
    """Return the beam direction, wedge and chi of a scan dict: ``scan["k_in_lab"]``, ``["wedge"]``, ``["chi"]``.

    Default: a beam along lab x and an untilted rotation axis.
    """
    k_in = jnp.asarray(scan.get("k_in_lab", (1.0, 0.0, 0.0)))
    return k_in, scan.get("wedge", 0.0), scan.get("chi", 0.0)


def coarsen_scan(scan: dict, g: int) -> dict:
    """Return the scan of a histogram whose rows are summed in groups of g (:func:`coarsen_rows`).

    Coarse row k is rows k g ... (k + 1) g - 1, at their middle: dty0 moves by (g - 1) / 2 rows, ystep is g times
    larger and there are ceil(n_rows / g) rows. With a beam profile ("voxel", "width_beam") voxels are g times larger
    and the beam's flat top (g - 1) rows wider. Per-row "exposure" becomes each group's mean (a coarse row is the sum
    of g rows), and "ddty" each group's mean real dty (weighted by exposure) less the coarse row's nominal dty.

    Parameters
    ----------
    scan
        Scan dict, see :func:`~anri.index.occupancy.system`
    g
        Group size

    Returns
    -------
    dict
        The coarse scan
    """
    ystep, n = scan["ystep"], scan["n_rows"]
    n_c = -(-n // g)
    out = {**scan, "dty0": scan["dty0"] + 0.5 * (g - 1) * ystep, "ystep": g * ystep, "n_rows": n_c}
    if "voxel" in scan:  # summed rows: a flat top (g - 1) rows wider, over voxels g x larger
        out["voxel"] = g * scan["voxel"]
        out["width_beam"] = scan["width_beam"] + (g - 1) * ystep

    def groups(a: jax.Array) -> jax.Array:  # [n, n_o] -> [n_c, g, n_o], missing rows of the last group as 0
        return jnp.pad(a, ((0, n_c * g - n), (0, 0))).reshape(n_c, g, -1)

    if "exposure" in scan:
        out["exposure"] = groups(scan["exposure"]).sum(1) / g
    if "ddty" in scan:
        w = groups(scan.get("exposure", jnp.ones_like(scan["ddty"])))
        real = groups(scan["ddty"] + (jnp.arange(n) * ystep)[:, None])  # each row's real dty, less dty0
        mean = (w * real).sum(1) / jnp.maximum(w.sum(1), 1e-30)
        nominal = 0.5 * (g - 1) * ystep + jnp.arange(n_c) * g * ystep  # each coarse row's dty, less dty0
        out["ddty"] = jnp.where(w.sum(1) > 0, mean - nominal[:, None], 0.0)
    return out


def dty_offsets(
    frame_dty: ArrayLike, frame_omega: ArrayLike, ybinedges: ArrayLike, gridstep: int, om0: float, b_o: float, n_o: int,
    dty0: float, ystep: float, bin_dty: ArrayLike | None = None, return_exposure: bool = False,
) -> np.ndarray | tuple:  # fmt: skip
    """Each histogram row's dty, per omega bin, less its nominal dty: where the beam really was.

    A row is the frames whose dty, as :func:`anri.io.imaged11.stream_sparse` binned them (``bin_dty``; default each DataSet
    row's mean dty, as when it bins by the DataSet's dty), falls in one bin of ``ybinedges``. In a step scan every frame of a row is at the row's dty and the offsets are 0. In a fly or helical scan dty
    moves while omega turns (e.g. one dty step per turn): the frames of one omega bin were taken at a dty up to half
    a step away from the row's. :func:`anri.index.occupancy.system` uses the offsets (``scan["ddty"]``) to put each voxel in
    the rows that really saw it.

    Parameters
    ----------
    frame_dty, frame_omega
        [rows, frames] each frame's dty and omega, e.g. :func:`anri.io.imaged11.read_frame_dty` and the DataSet's ``omega``
    ybinedges
        [n + 1] dty bin edges of the rows (the DataSet's ``ybinedges``)
    gridstep
        Rows summed in groups of this, as in the histogram
    om0, b_o, n_o
        The histogram's first omega bin edge, bin width and number of bins
    dty0, ystep
        The histogram's rows: row k is nominally at dty0 + k ystep
    bin_dty
        [rows, frames] the dty each frame was binned by (e.g. ``frame_dty`` itself when frames are binned by their
        own dty); default each row's mean of ``frame_dty``
    return_exposure
        Also return each (row, omega bin)'s exposure: its frames over the usual number (the median over bins with
        frames). Binned by their own dty, a helical scan's frames give some bins none and some two rotations' worth;
        the model must know (``scan["exposure"]``)

    Returns
    -------
    offsets: np.ndarray
        [n_rows, n_o] mean dty of the frames in each (row, omega bin) less dty0 + k ystep; 0 where no frames
    exposure: np.ndarray
        [n_rows, n_o], if return_exposure
    """
    fd = np.asarray(frame_dty, float)
    om = np.asarray(frame_omega, float)
    edges = np.asarray(ybinedges, float)
    n_rows = -(-(len(edges) - 1) // gridstep)
    bd = np.broadcast_to(np.nanmean(fd, 1, keepdims=True), fd.shape) if bin_dty is None else np.asarray(bin_dty, float)
    kk = dty_to_row(bd, edges, gridstep)  # each frame's row
    with float64():  # the omega bins of system
        o = np.asarray(omega_bin(wrap_omega(om, om0), om0, b_o)).astype(int)
    ok = (kk >= 0) & (o >= 0) & (o < n_o) & np.isfinite(fd)
    idx = kk[ok] * n_o + o[ok]
    s = np.bincount(idx, weights=fd[ok] - (dty0 + kk[ok] * ystep), minlength=n_rows * n_o)
    c = np.bincount(idx, minlength=n_rows * n_o)
    off = np.where(c > 0, s / np.maximum(c, 1), 0.0).reshape(n_rows, n_o)
    if not return_exposure:
        return off
    return off, (c / max(float(np.median(c[c > 0])), 1.0)).reshape(n_rows, n_o) if np.any(c > 0) else c
