"""Base functions for forward projection code."""

from typing import Iterable

import jax
import jax.numpy as jnp

from anri.diffract import omega_solns, q_lab_to_k_out, scale_norm_k
from anri.geom import lab_to_sample, sample_to_lab


@jax.jit
def hkl_to_k_omega(
    ubi: jax.Array,  # grain stuff
    hkl: jax.Array,
    etasign: int,  # peak stuff
    wavelength: float,
    ky: float,
    kz: float,  # beam
    wedge: float,
    chi: float,
    y0: float,  # gonio
) -> tuple[jax.Array, jax.Array, float, float]:
    r"""Forward-project a reciprocal space vector (h,k,l) with basis vectors (a*, b*, c*) into k-vectors and omega angles.

    This just chains together various transforms from :func:`anri.diffract` and :func:`anri.geom`.

    Parameters
    ----------
    ubi:
        [3,3] (U.B)^(-1) matrix of the grain/voxel
    hkl:
        [3] (h,k,l) reciprocal space vector
    etasign:
        +1 (omega1 in ImageD11) or -1 (omega2 in ImageD11) to select which omega solution to return
    wavelength:
        Wavelength in angstroms
    ky:
        y-component of the beam in the lab frame. Represents horizontal beam divergence, usually zero.
    kz:
        z-component of the beam in the lab frame. Represents vertical beam divergence, usually zero.
    wedge:
        Wedge motor value (degrees)
    chi:
        Chi motor value (degrees)
    y0:
        The true value of dty when the rotation axis (untilted by wedge, chi) intersects the beam

    Returns
    -------
    k_in_lab: jax.Array
        [3] k-in vector in laboratory frame (incoming beam) - not scaled or normalised!
    k_out_lab: jax.Array
        [3] k_out vector in laboratory frame
    omega: float
        Omega angle where diffraction occurs in degrees
    valid: float
        Boolean indicating if a valid solution exists
    """
    q_sample = jnp.linalg.inv(ubi) @ hkl

    k_in_lab = jnp.array([1.0, ky, kz])
    k_in_lab_norm = scale_norm_k(k_in_lab, wavelength)
    k_in_sample_norm = lab_to_sample(k_in_lab_norm, 0.0, wedge, chi, 0.0, 0.0)

    omega, valid = omega_solns(q_sample, etasign, k_in_sample_norm)

    q_lab = sample_to_lab(q_sample, omega, wedge, chi, 0.0, 0.0)

    k_out_lab = q_lab_to_k_out(q_lab, k_in_lab_norm)

    return k_in_lab, k_out_lab, omega, valid


@jax.jit
def get_cov_in(sig_origin: jax.Array, sig_wavelength: float, sig_ky: float, sig_kz: float) -> jax.Array:
    r"""Generate the input variance-covariance matrix from your sigma values.

    Parameters
    ----------
    sig_origin: jax.Array
        [3] Array of standard deviations on diffraction origin position. This is often your position uncertainty
    sig_wavelength: float
        Standard deviation on beam wavelength
    sig_ky
        Standard deviation on beam horizontal divergence
    sig_kz
        Standard deviation on beam vertical divergence

    Returns
    -------
    cov_in: jax.Array
        [6,6] Diagonal input variance-covariance matrix.

    Notes
    -----
    Builds a 6x6 input variance-covariance matrix.
    For $\vec{\sigma_{\text{origin}}} = \left(\sigma_x, \sigma_y, \sigma_z\right)$:
    $\matr{\Sigma}^{\text{in}} = \begin{bmatrix} \sigma_x^2 & 0 & 0 & 0 & 0 & 0 \\ 0 & \sigma_y^2 & 0 & 0 & 0 & 0\\0 & 0 & \sigma_z^2 & 0 & 0 & 0\\ 0 & 0 & 0 &\sigma_\lambda^2 & 0 & 0\\ 0 & 0 & 0 & 0 & \sigma_{k_y}^2 & 0 \\ 0 & 0 & 0& 0& 0 & \sigma_{k_z}^2\end{bmatrix} $

    """
    cov_in = jnp.diag(jnp.array([sig_origin**2, sig_origin**2, sig_origin**2, sig_wavelength**2, sig_ky**2, sig_kz**2]))

    return cov_in


@jax.jit
def propagate_cov(J_func_out: Iterable[jax.Array], cov_in: jax.Array) -> jax.Array:
    r"""Propagate an input covariance matrix with a Jacobian to yield an output covariance matrix.

    Parameters
    ----------
    J_func_out
        The output of calling :func:`jax.jacfwd` on a JAX jitted function.
    cov_in
        [6,6] The input covariance matrix - build with :func:`get_cov_in`. Must have the same dimensionality as J_func_out

    Returns
    -------
    cov_out: jax.Array
        [3,3] Output covariance matrix - the covariance in the outputs of the JAX jitted function

    Notes
    -----
    Propagation goes as:
    $\mathbf{\Sigma}^{\text{out}} = \mathbf {J_{f}} \mathbf{\Sigma}^{\text{in}}  \mathbf {J_{f}}^T$

    This handles multi-dimensional outputs - e.g. if one of the function inputs is a 3-vector, we get a 3x3 Jacobian for it.
    """
    J = jnp.concatenate([j if j.ndim > 1 else j[..., jnp.newaxis] for j in J_func_out], axis=-1)
    cov_out = J @ cov_in @ J.T
    return cov_out
