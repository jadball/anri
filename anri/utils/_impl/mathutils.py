"""Small maths helpers.

``inv3`` is written out explicitly: XLA lowers ``jnp.linalg.inv`` to a batched LAPACK/cuSOLVER call that handles
one tiny matrix at a time. For the 3x3 matrices that Anri uses per voxel and per peak, plain arithmetic is several
times faster.
"""

import math

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))  # 2.3548...


def fwhm_to_sigma(fwhm: ArrayLike) -> ArrayLike:
    """Return the standard deviation of a Gaussian from its full width at half maximum (FWHM / 2.3548...).

    Works on Python floats, NumPy and JAX arrays alike.
    """
    return fwhm / _FWHM_PER_SIGMA


def sigma_to_fwhm(sigma: ArrayLike) -> ArrayLike:
    """Return the full width at half maximum of a Gaussian from its standard deviation (sigma x 2.3548...)."""
    return sigma * _FWHM_PER_SIGMA


@jax.jit
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
