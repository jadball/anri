"""Small numerical helpers: 3x3 inverses, Gaussians, truncated exponentials, and float64 evaluation of JAX code.

``inv3`` is written out explicitly: XLA lowers ``jnp.linalg.inv`` to a batched LAPACK/cuSOLVER call that handles
one tiny matrix at a time. For the 3x3 matrices that Anri uses per voxel and per peak, plain arithmetic is several
times faster.
"""

import math
from contextlib import AbstractContextManager
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr
from jax.scipy.stats import norm
from jax.typing import ArrayLike

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))  # 2.3548...


def fwhm_to_sigma(fwhm: Any) -> Any:  # noqa: ANN401 (float, NumPy or JAX)
    """Return the standard deviation of a Gaussian from its full width at half maximum (FWHM / 2.3548...).

    Works on Python floats, NumPy and JAX arrays alike.
    """
    return fwhm / _FWHM_PER_SIGMA


def sigma_to_fwhm(sigma: Any) -> Any:  # noqa: ANN401 (float, NumPy or JAX)
    """Return the full width at half maximum of a Gaussian from its standard deviation (sigma x 2.3548...)."""
    return sigma * _FWHM_PER_SIGMA


@jax.jit
def _interval_mass(a: jax.Array, b: jax.Array) -> jax.Array:
    """Phi(b) - Phi(a) for a <= b, accurate in both tails and never negative.

    In the upper tail Phi(b) and Phi(a) are both close to 1 and their difference cancels, in float32 to about 1e-7,
    either sign. Times a bright peak's amplitude, and summed over the many voxels that share a pixel, that made
    model pixels far from a peak wrong, even negative. There Phi(-a) - Phi(-b) is used instead.
    """
    return jnp.maximum(jnp.where(a > 0, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a)), 0.0)


def bin_fractions(lo: ArrayLike, hi: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> jax.Array:
    """Fraction of a 1D Gaussian N(mu, sigma^2) that lies in [lo, hi)."""
    lo, hi, mu, sigma = (jnp.asarray(x) for x in (lo, hi, mu, sigma))
    return _interval_mass((lo - mu) / sigma, (hi - mu) / sigma)


def truncated_moments(lo: ArrayLike, hi: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> tuple:
    """Mass, mean and variance of N(mu, sigma^2) restricted to [lo, hi).

    Where the mass is negligible, the mean falls back to mu clipped into the interval and the variance to 0
    (those cells carry no intensity, and this keeps gradients finite).
    """
    lo, hi, mu, sigma = (jnp.asarray(x) for x in (lo, hi, mu, sigma))
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    mass = _interval_mass(a, b)
    pa, pb = jnp.asarray(norm.pdf(a)), jnp.asarray(norm.pdf(b))
    ok = mass > 1e-12
    safe = jnp.where(ok, mass, 1.0)
    r = (pa - pb) / safe
    mean = jnp.where(ok, mu + sigma * r, jnp.clip(mu, lo, hi))
    var = jnp.where(ok, sigma**2 * (1.0 + (a * pa - b * pb) / safe - r**2), 0.0)
    return mass, mean, jnp.maximum(var, 0.0)


def exponential_slices(scale: ArrayLike, cut: ArrayLike, n: int) -> tuple:
    """Mean and variance of each of n equal-probability slices of an exponential distribution truncated at cut.

    x has density exp(-x / scale) / scale on [0, cut], renormalised. Slice k holds the x between the quantiles k / n
    and (k + 1) / n. A mixture of n Gaussians with these means and variances (weights 1 / n) has exactly the mean
    and variance of the truncated exponential (law of total variance), and approaches its shape as n grows: e.g. a
    thick sensor's depth kernel (:mod:`anri.fwd.render`).

    Parameters
    ----------
    scale, cut
        Scale (mean of the untruncated exponential) and truncation, same units, cut > 0
    n
        Number of slices (static)

    Returns
    -------
    tuple
        means, variances [..., n]
    """
    b = jnp.maximum(jnp.asarray(scale), 1e-9)[..., None]
    R = jnp.maximum(jnp.asarray(cut), 1e-9)[..., None]
    P = -jnp.expm1(-R / b)  # untruncated mass below cut
    edges = -b * jnp.log1p(-(jnp.arange(n) / n) * P)
    edges = jnp.concatenate([edges, jnp.broadcast_to(R, edges.shape[:-1] + (1,))], -1)  # the last is the cut itself
    lo, hi = edges[..., :-1], edges[..., 1:]  # (from the formula, -b log(1 - P) = inf when cut >> scale)
    elo, ehi = jnp.exp(-lo / b), jnp.exp(-hi / b)
    Z = jnp.maximum(elo - ehi, 1e-30)
    m = b + (lo * elo - hi * ehi) / Z
    m2 = (elo * (lo**2 + 2 * lo * b + 2 * b**2) - ehi * (hi**2 + 2 * hi * b + 2 * b**2)) / Z
    return m, jnp.maximum(m2 - m**2, 0.0)


def ramp_integral(x: ArrayLike, sigma: ArrayLike) -> jax.Array:
    """Antiderivative of ndtr(x / sigma): x * Phi(x / sigma) + sigma * phi(x / sigma)."""
    x, sigma = jnp.asarray(x), jnp.asarray(sigma)
    return x * ndtr(x / sigma) + sigma * jnp.asarray(norm.pdf(x / sigma))


def ramp_integral2(x: ArrayLike, sigma: ArrayLike) -> jax.Array:
    """Antiderivative of :func:`ramp_integral`: (x^2 + sigma^2) / 2 * Phi(x / sigma) + x sigma / 2 * phi(x / sigma)."""
    x, sigma = jnp.asarray(x), jnp.asarray(sigma)
    return 0.5 * (x**2 + sigma**2) * ndtr(x / sigma) + 0.5 * x * sigma * jnp.asarray(norm.pdf(x / sigma))


def inv3(m: jax.Array) -> jax.Array:
    """Invert 3x3 matrices via the adjugate: inv(M) = adj(M) / det(M).

    Parameters
    ----------
    m
        [..., 3, 3] matrices

    Returns
    -------
    m_inv: jax.Array
        [..., 3, 3] inverses. Singular matrices give inf or NaN, as there is no pivoting.
    """
    c0, c1, c2 = m[..., :, 0], m[..., :, 1], m[..., :, 2]
    # the rows of adj(M) are cross products of the columns of M
    adj = jnp.stack([jnp.cross(c1, c2), jnp.cross(c2, c0), jnp.cross(c0, c1)], axis=-2)
    det = jnp.sum(c0 * jnp.cross(c1, c2), axis=-1)
    return adj / det[..., None, None]


def float64() -> AbstractContextManager:
    """Evaluate JAX in float64 inside a ``with`` block, without changing the global setting.

    Returns
    -------
    AbstractContextManager
        ``jax.enable_x64(True)``, or ``jax.experimental.enable_x64()`` on older JAX
    """
    if hasattr(jax, "enable_x64"):
        return jax.enable_x64(True)
    import importlib

    # only in older JAX (e.g. 0.4.30); getattr keeps the type checker, run with a newer JAX, from flagging it
    return getattr(importlib.import_module("jax.experimental"), "enable_x64")()  # noqa: B009
