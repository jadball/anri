"""Separable Gaussian splatting for forward-projected peaks.

Rewrite of the original ``anri.sandbox.splat``. The maths is unchanged --- a
peak is still the product of per-axis integrated Gaussians over unit bins ---
but two structural choices in the old version dominated its cost.

**The window was built with a gather.** ``peak_to_pixels`` located its window
with ``jax.lax.dynamic_slice(bins[i], (starts[i],), (ws[i],))``. Under ``vmap``
a dynamic slice with a batched start lowers to a gather, which XLA:CPU neither
vectorises nor fuses with the surrounding arithmetic, so every intermediate
round-tripped through memory. ``bins[i]`` was ``jnp.arange(det_shape[i])``, so
the gathered values are ``starts[i] + arange(ws[i])`` --- pure broadcast
arithmetic, no gather, fully fusable.

**The separability was thrown away.** ``sample_gaussian_bins`` takes an
``[m, n]`` grid and returns ``prod(marginals, axis=-1)``, so feeding it a
``meshgrid`` of a ``w0``-vector and a ``w1``-vector evaluates ``2 * w0 * w1 * 2``
erfs where ``2 * (w0 + w1)`` suffice. At ``7 x 7`` that is 196 erfs instead of
28.

Measured against the old kernel, 32768 peaks on a 2048 x 2048 detector,
identical window starts and agreement to 7.1e-8 relative (float32 epsilon).
**Read the core count**: XLA threads the gather, so most of the single-core
advantage disappears on a large host. On 40 cores the 7x7 figure is 3.88 ms ->
1.40 ms, i.e. 2.8x, not 15.7x. Quote the second column only if you are pinned
to one core.

===========  =========  =========  =======
window       old        new        speedup
===========  =========  =========  =======
``7 x 7``    24.1 ms    1.5 ms     15.7x
``5 x 5``    12.7 ms    1.0 ms     13.1x
``3 x 3``    4.7 ms     0.6 ms     7.6x
===========  =========  =========  =======

The third saving is not in the kernel but in how it is called: see
:func:`needed_half_width`. Sizing the window from the peak's own sigma *and its
amplitude on this frame* rather than from a global percentile typically cuts the
window area by another 3-5x, and that reduction carries through the host-side
merge as well, which is where the time goes once the kernel is fixed.

Everything here is elementwise in the batch axis, so it runs unchanged on GPU.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import erf

__all__ = [
    "axis_factors",
    "bin_weights",
    "needed_half_width",
    "splat_windows",
    "window_keys",
    "window_starts",
]

_SQRT2PI = math.sqrt(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)


def _phi(z):
    """Standard normal density."""
    return np.exp(-0.5 * z * z) / _SQRT2PI


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


@jax.jit
def bin_weights(mu: jax.Array, erf_scale: jax.Array, centres: jax.Array, half_width: jax.Array) -> jax.Array:
    r"""Integrated 1D Gaussian mass in bins of half-width ``half_width``.

    .. math::

        w = \tfrac{1}{2}\left[
            \operatorname{erf}\!\left((x + h - \mu)\,s\right) -
            \operatorname{erf}\!\left((x - h - \mu)\,s\right)\right]

    Shapes broadcast, so this serves both the detector axes (where ``centres``
    is ``[P, w]``) and the omega / dty axes (where it is ``[P]``).
    """
    d = centres - mu
    lo = (d - half_width) * erf_scale
    hi = (d + half_width) * erf_scale
    return 0.5 * (jax.scipy.special.erf(hi) - jax.scipy.special.erf(lo))


def window_starts(mu: jax.Array, n_bins: int, width: int) -> jax.Array:
    """First bin index of a ``width``-wide window centred on ``mu``.

    Bins are assumed to be ``arange(n_bins)``, i.e. unit spacing from zero,
    which is what a detector axis is. Arithmetic, not a search: the old
    ``argmin(abs(bins - mu))`` scanned every bin centre.
    """
    return jnp.clip(jnp.floor(mu + 0.5).astype(jnp.int32) - width // 2, 0, n_bins - width)


@functools.partial(jax.jit, static_argnames=("width", "n_bins"))
def axis_factors(mu: jax.Array, erf_scale: jax.Array, width: int, n_bins: int) -> tuple[jax.Array, jax.Array]:
    """Per-axis window factors and start index for a batch of peaks.

    Parameters
    ----------
    mu, erf_scale
        ``[P]`` centroid and :math:`1/(\\sigma\\sqrt 2)` along one axis.
    width, n_bins
        Window width and axis length. Static.

    Returns
    -------
    factors : jax.Array
        ``[P, width]`` integrated Gaussian mass in each bin of the window.
    start : jax.Array
        ``[P]`` index of the window's first bin.
    """
    start = window_starts(mu, n_bins, width)
    offs = jnp.arange(width, dtype=mu.dtype)
    centres = start[:, None].astype(mu.dtype) + offs[None, :]
    return bin_weights(mu[:, None], erf_scale[:, None], centres, jnp.asarray(0.5, mu.dtype)), start


@functools.partial(jax.jit, static_argnames=("window", "det_shape"))
def splat_windows(
    mu: jax.Array,
    erf_scale: jax.Array,
    amplitude: jax.Array,
    window: tuple[int, int],
    det_shape: tuple[int, int],
) -> tuple[jax.Array, jax.Array]:
    """Render a batch of 2D Gaussians onto their own local pixel windows.

    Parameters
    ----------
    mu : jax.Array
        ``[P, 2]`` detector centroid in ``(slow, fast)`` pixels.
    erf_scale : jax.Array
        ``[P, 2]`` :math:`1/(\\sigma\\sqrt 2)` per axis.
    amplitude : jax.Array
        ``[P]`` integrated intensity already folded over omega and dty.
    window : tuple[int, int]
        ``(w0, w1)`` window size in pixels. Static.
    det_shape : tuple[int, int]
        Detector shape, for clipping windows to the edge. Static.

    Returns
    -------
    values : jax.Array
        ``[P, w0, w1]`` integrated counts per pixel.
    starts : jax.Array
        ``[P, 2]`` window origin, for scattering back to the detector.

    Notes
    -----
    The block is the outer product of the two axis factors, which is what
    ``prod(marginals, axis=-1)`` computes in the non-separable form at 7x the
    erf cost. Everything is elementwise or a broadcast: no gather, so XLA fuses
    the whole thing into one pass.
    """
    w0, w1 = int(window[0]), int(window[1])
    a, st0 = axis_factors(mu[:, 0], erf_scale[:, 0], w0, int(det_shape[0]))
    b, st1 = axis_factors(mu[:, 1], erf_scale[:, 1], w1, int(det_shape[1]))
    values = (amplitude[:, None] * a)[:, :, None] * b[:, None, :]
    return values, jnp.stack([st0, st1], axis=-1)


# ---------------------------------------------------------------------------
# window sizing
# ---------------------------------------------------------------------------


def needed_half_width(sigma, amplitude, cut, cap=8):
    """Smallest half-width whose discarded tail is provably below ``cut``.

    The old ``suggest_detector_window`` took one window for the whole run from
    the 99.9th percentile sigma, so every peak paid for the widest. Two things
    make that wasteful. Most peaks are far narrower than the tail of the sigma
    distribution; and the same peak appears on several ``(omega, dty)`` frames
    with wildly different amplitudes --- here the off-centre omega bin carries
    about 7% of the centre bin and the off-centre dty bin about 12%, so eight of
    a peak's nine frames are two orders of magnitude fainter and need a much
    smaller window to reach the same absolute floor.

    Sizing per *contribution* rather than per run typically takes the mean
    window from ``7 x 7`` to nearer ``3 x 3``, and that shrinks the host-side
    sort as well as the kernel.

    The bound used is the Gaussian density majorant of the bin integral,

    .. math::

        A(d) \\le \\frac{1}{\\sigma\\sqrt{2\\pi}}
                 \\exp\\!\\left(-\\frac{(d - \\tfrac12)^2}{2\\sigma^2}\\right),
        \\qquad d \\ge 1,

    solved for the largest ``d`` that can still exceed ``cut``. Being an upper
    bound it never under-sizes: any pixel dropped is provably below ``cut``. It
    over-sizes by at most one bin in practice.

    Parameters
    ----------
    sigma : array_like
        ``[P]`` or ``[P, 2]`` peak sigma in pixels along the axis of interest.
    amplitude : array_like
        ``[P]`` amplitude of this contribution (already folded over omega/dty).
    cut : float
        Absolute intensity floor, normally ``merge_cut * threshold``.
    cap : int
        Largest half-width returned.

    Returns
    -------
    half_width : numpy.ndarray
        ``int32``, same leading shape as ``sigma``.
    """
    sigma = np.asarray(sigma, np.float64)
    amp = np.asarray(amplitude, np.float64)
    if sigma.ndim == 2:
        amp = amp[:, None]
    ratio = amp / np.maximum(cut * sigma * _SQRT2PI, 1e-300)
    lg = np.log(np.maximum(ratio, 1.0))
    h = np.floor(0.5 + sigma * np.sqrt(2.0 * lg))
    return np.clip(h, 0, cap).astype(np.int32)


# ---------------------------------------------------------------------------
# host-side helpers
# ---------------------------------------------------------------------------


def window_keys(starts, frame, det_shape, window, out=None):
    """Global linear index of every pixel of every window, by broadcast.

    ``frame * n_slow * n_fast + row * n_fast + col``, built as one strided
    broadcast over ``[P, w0, w1]``.

    The obvious alternative --- render, threshold, then recover ``(peak, row,
    col)`` from the surviving flat indices with two ``divmod``\\ s and three
    fancy-index gathers into ``starts`` and ``frame`` --- measured 2.8x slower
    on the same data, because the gathers are random access over the whole
    batch while this is sequential. Build the keys for the full window and take
    the survivors with a single gather instead.
    """
    w0, w1 = int(window[0]), int(window[1])
    n_slow, n_fast = int(det_shape[0]), int(det_shape[1])
    plane = np.int64(n_slow) * np.int64(n_fast)
    d0 = np.arange(w0, dtype=np.int64)
    d1 = np.arange(w1, dtype=np.int64)
    key = frame.astype(np.int64)[:, None, None] * plane
    key = key + (starts[:, 0].astype(np.int64)[:, None, None] + d0[None, :, None]) * n_fast
    key = key + (starts[:, 1].astype(np.int64)[:, None, None] + d1[None, None, :])
    return key if out is None else np.copyto(out, key)


# ---------------------------------------------------------------------------
# correlated peaks
# ---------------------------------------------------------------------------
#
# splat_windows assumes rho = 0, which makes a window the outer product of two
# 1D bin integrals. That assumption is wrong on a detector.
#
# In a monochromatic scanning experiment the dominant detector-plane broadening
# is the wavelength spread: sigma_lambda moves a reflection along the *radial*
# direction of its Debye-Scherrer ring. The peak's principal axes are therefore
# (radial, azimuthal), not (slow, fast). They coincide only at azimuth 0 and 90
# degrees; at 45 degrees the slow-fast covariance is as large as the variances.
#
# For dlambda/lambda = 5e-4 at 120 mm and 75 um pixels, the radial sigma is
# 0.32 px at two-theta 20 degrees and 1.12 px at 42 degrees. Against a detector
# point spread of 1 px (sigma0 = 0.289 px) that gives, at azimuth 45:
#
#   two-theta   sigma_marginal   rho     rendered area     true aspect
#      10           0.31         0.11        1.01x            1.1:1
#      20           0.37         0.38        1.08x            1.5:1
#      30           0.50         0.66        1.33x            2.2:1
#      42           0.84         0.88        2.13x            4.0:1
#
# so at the detector corner an axis-aligned render produces a round blob of
# twice the area where the truth is a 4:1 radial streak. The marginals and the
# integrated intensity are right and the shape is wrong, which is exactly the
# combination that leaves hkl residuals looking fine while Number_of_pixels,
# blob morphology and segmentation merging are all wrong.
#
# There is no closed form for a correlated bivariate normal over a rectangle.
# Two schemes were tried and measured against the exact four-corner CDF:
#
# * Gauss-Hermite along the major axis (a line source convolved with an
#   isotropic blur, which is what the physics is). Fails for long streaks: the
#   integrand in that variable is a narrow spike where the line crosses the
#   pixel, not a polynomial over the Gaussian. 1.8e-1 error at q=8.
# * Sub-binning the conditioning axis with a mean-only conditional. Converges
#   at second order, so ~5e-3 at n_sub=6. Too slow.
#
# What is used below matches the conditional *variance* as well as the mean,
# which cancels the leading error term, and conditions on the wider axis so the
# conditional-mean slope is at most rho. Worst case over every (sigma, rho)
# reachable with a 1 px detector point spread:
#
#   n_sub = 1: 9e-3    n_sub = 2: 3.1e-4    n_sub = 3: 6.1e-5
#
# relative to the peak pixel. A 2000-count peak carries 2.2% Poisson noise, so
# n_sub = 2 is three orders of magnitude inside the noise floor. At rho = 0 it
# reduces to splat_windows to 4.5e-16.


def _truncated_moments(z, sigma, n_sub):
    """Weight, conditional mean offset and conditional variance of each sub-bin.

    ``z`` is ``[P, m+1]`` standardised sub-bin edges. Returns ``(W, dmean,
    var)``, each ``[P, m]``: the exact Gaussian mass in the sub-bin, the offset
    of its centroid from ``mu``, and the variance about that centroid. The
    weights are exact ``erf`` differences, so they sum to the exact marginal
    however coarse the subdivision --- integrated intensity and the conditioning
    axis's own profile are preserved for any ``n_sub``.
    """
    W = np.diff(0.5 * (1.0 + erf(z / _SQRT2)), axis=1)
    za, zb = z[:, :-1], z[:, 1:]
    pa, pb = _phi(za), _phi(zb)
    safe = W > 1e-300
    Ws = np.where(safe, W, 1.0)
    r1 = (pa - pb) / Ws
    dmean = np.where(safe, sigma * r1, 0.0)
    # Var[x | x in bin] for a truncated normal, in units of sigma^2
    vrel = 1.0 + (np.nan_to_num(za * pa) - np.nan_to_num(zb * pb)) / Ws - r1 * r1
    var = np.where(safe, sigma * sigma * np.clip(vrel, 0.0, None),
                   (1.0 / n_sub) ** 2 / 12.0)
    return W, dmean, var


def _corr_block(mu_a, mu_b, sg_a, sg_b, rho, st_a, st_b, wa, wb, n_sub):
    """Correlated block conditioned on axis ``a``. Shapes ``[P, wa, wb]``."""
    m = wa * n_sub
    e = (st_a[:, None] - 0.5) + np.arange(m + 1)[None, :] / n_sub
    z = (e - mu_a[:, None]) / sg_a[:, None]
    W, dmean, var = _truncated_moments(z, sg_a[:, None], n_sub)

    slope = (rho * sg_b / sg_a)[:, None]
    mcond = mu_b[:, None] + slope * dmean
    # conditional variance, plus the spread the conditional mean has *within*
    # the sub-bin: this second moment is what makes n_sub = 2 enough
    scond = np.sqrt((sg_b * sg_b * (1.0 - rho * rho))[:, None] + slope * slope * var)

    eb = (st_b[:, None] - 0.5) + np.arange(wb + 1)[None, :]
    B = 0.5 * np.diff(erf((eb[:, None, :] - mcond[:, :, None])
                          / (scond[:, :, None] * _SQRT2)), axis=2)
    return (W[:, :, None] * B).reshape(len(mu_a), wa, n_sub, wb).sum(axis=2)


def splat_windows_correlated(mu, sigma, rho, amplitude, window, det_shape,
                             n_sub=2):
    r"""Render correlated 2D Gaussians onto local pixel windows.

    Parameters
    ----------
    mu : array_like
        ``[P, 2]`` centroid in ``(slow, fast)`` pixels.
    sigma : array_like
        ``[P, 2]`` marginal sigmas in pixels. Unchanged by the correlation ---
        :math:`\rho` rotates and narrows the peak without touching its
        projections onto the detector axes.
    rho : array_like
        ``[P]`` slow-fast correlation coefficient.
    amplitude : array_like
        ``[P]`` integrated intensity.
    window, det_shape : tuple[int, int]
        Window size and detector shape.
    n_sub : int
        Sub-bins per pixel along the conditioning axis. 2 is enough; see the
        error table above.

    Returns
    -------
    values : numpy.ndarray
        ``[P, w0, w1]`` float32
    starts : numpy.ndarray
        ``[P, 2]`` int64

    Notes
    -----
    Window sizing is unaffected: :func:`needed_half_width` bounds a pixel by the
    marginal along its row, correlation only concentrates mass inside that
    bound, so the rectangle chosen for the uncorrelated case is still
    conservative.
    """
    w0, w1 = int(window[0]), int(window[1])
    n0, n1 = int(det_shape[0]), int(det_shape[1])
    mu = np.asarray(mu, np.float64)
    sg = np.asarray(sigma, np.float64)
    rho = np.clip(np.asarray(rho, np.float64), -0.9995, 0.9995)
    amp = np.asarray(amplitude, np.float64)

    st0 = np.clip(np.floor(mu[:, 0] + 0.5).astype(np.int64) - w0 // 2, 0, n0 - w0)
    st1 = np.clip(np.floor(mu[:, 1] + 0.5).astype(np.int64) - w1 // 2, 0, n1 - w1)
    starts = np.stack([st0, st1], axis=-1)

    # Condition on the wider axis. The conditional-mean slope is
    # rho * sigma_other / sigma_this, so conditioning on the narrow axis can
    # make it arbitrarily large; this bounds it by rho. Measured 2.2e-4 -> 3.7e-6
    # on a (0.55, 1.50, rho=0.6) peak.
    wide0 = sg[:, 0] >= sg[:, 1]
    v = np.empty((len(mu), w0, w1), np.float64)
    if wide0.any():
        i = np.flatnonzero(wide0)
        v[i] = _corr_block(mu[i, 0], mu[i, 1], sg[i, 0], sg[i, 1], rho[i],
                           st0[i], st1[i], w0, w1, n_sub)
    if (~wide0).any():
        i = np.flatnonzero(~wide0)
        v[i] = np.swapaxes(
            _corr_block(mu[i, 1], mu[i, 0], sg[i, 1], sg[i, 0], rho[i],
                        st1[i], st0[i], w1, w0, n_sub), 1, 2)
    v *= amp[:, None, None]
    return v.astype(np.float32), starts


def shape_error(sigma, rho):
    """What ignoring ``rho`` costs: rendered/true blob area and the true aspect.

    ``area`` is :math:`1/\\sqrt{1-\\rho^2}`, the factor by which the
    axis-aligned render inflates the 1-sigma ellipse at fixed marginals.
    ``aspect`` is the true major:minor axis ratio, which the axis-aligned
    render replaces with ``sigma_slow / sigma_fast``.
    """
    sg = np.asarray(sigma, float)
    rho = np.asarray(rho, float)
    v0, v1 = sg[..., 0] ** 2, sg[..., 1] ** 2
    c = rho * sg[..., 0] * sg[..., 1]
    mid = 0.5 * (v0 + v1)
    disc = 0.5 * np.sqrt((v0 - v1) ** 2 + 4 * c * c)
    lo = np.maximum(mid - disc, 1e-30)
    return {"area": 1.0 / np.sqrt(np.maximum(1.0 - rho ** 2, 1e-30)),
            "aspect": np.sqrt((mid + disc) / lo),
            "aspect_rendered": np.maximum(sg[..., 0], sg[..., 1])
                               / np.minimum(sg[..., 0], sg[..., 1]),
            "angle_deg": np.degrees(0.5 * np.arctan2(2 * c, v0 - v1))}
