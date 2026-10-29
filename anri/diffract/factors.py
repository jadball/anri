"""Intensity factors of a reflection from its incoming and outgoing wave-vectors: Lorentz and polarisation."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from anri.geom.beam import beam_basis
from anri.geom.gonio import sample_to_lab


def lorentz(k_in: jax.Array, k_out: jax.Array, rot_axis: jax.Array) -> jax.Array:
    """Lorentz factor for rotation about ``rot_axis``.

    It is ``1 / |rot_axis . (k_in x k_out)|`` (Kabsch, 1988, p. 922) [1]_, equal to
    1 / :func:`ImageD11.refinegrains.lf` = ``1 / (sin(2theta) |sin(eta)|)`` for a vertical rotation axis.

    References
    ----------
    .. [1] Kabsch, W., 1988. Evaluation of single-crystal X-ray diffraction data from a position-sensitive detector.
       Journal of Applied Crystallography 21, 916–924. https://doi.org/10.1107/S0021889888007903
    """
    k_in = k_in / jnp.linalg.norm(k_in)
    k_out = k_out / jnp.linalg.norm(k_out)
    return 1.0 / jnp.abs(rot_axis @ jnp.cross(k_in, k_out))


def polarisation(k_in: jax.Array, k_out: jax.Array, factor: ArrayLike) -> jax.Array:
    """Polarisation factor for a beam polarised horizontally, with degree of polarisation ``factor``.

    Horizontal and vertical are across the beam, see :func:`anri.geom.beam.beam_basis`. For a beam along lab x
    (horizontal = lab y) this equals :func:`ImageD11.refinegrains.polarization` with ``eta0 = 0``.
    ``factor = 1`` is fully horizontally polarised, ``factor = 0`` is unpolarised. It is Kabsch's vector form
    (Kabsch, 1988, p. 922) [1]_, with the field along the polarisation plane's normal ``e_h`` with probability
    ``(1 + factor) / 2``, and Kahn et al.'s ``P = (1 + cos^2 2theta) / 2 - J' cos(2 rho) sin^2(2theta) / 2`` with
    ``J' = factor`` and ``rho = eta + 90`` degrees (Kahn et al., 1982, equations 13-15) [2]_.

    References
    ----------
    .. [1] Kabsch, W., 1988. Evaluation of single-crystal X-ray diffraction data from a position-sensitive detector.
       Journal of Applied Crystallography 21, 916–924. https://doi.org/10.1107/S0021889888007903
    .. [2] Kahn, R., Fourme, R., Gadet, A., Janin, J., Dumas, C., André, D., 1982. Macromolecular crystallography with
       synchrotron radiation: photographic data collection and polarization correction. Journal of Applied
       Crystallography 15, 330–337. https://doi.org/10.1107/S0021889882012060
    """
    _, e_h, e_v = beam_basis(k_in)
    k_out = k_out / jnp.linalg.norm(k_out)
    return 0.5 * (1 + factor) * (1 - (k_out @ e_h) ** 2) + 0.5 * (1 - factor) * (1 - (k_out @ e_v) ** 2)


def lorentz_polarisation(
    k_in: jax.Array, k_out: jax.Array, wedge: ArrayLike, chi: ArrayLike, pol_factor: ArrayLike
) -> jax.Array:
    """Lorentz x polarisation factor of a reflection, for rotation about the omega axis tilted by wedge and chi.

    Parameters
    ----------
    k_in, k_out
        [3] Incoming and outgoing wave-vectors in the lab frame (any length), e.g. from
        :func:`anri.fwd.peak.hkl_to_k_omega`
    wedge, chi
        Wedge and chi motor values (degrees): they tilt the rotation axis
    pol_factor
        Degree of horizontal polarisation, see :func:`polarisation`
    """
    rot_axis = sample_to_lab(jnp.array([0.0, 0.0, 1.0], k_in.dtype), 0.0, wedge, chi, 0.0, 0.0)
    return lorentz(k_in, k_out, rot_axis) * polarisation(k_in, k_out, pol_factor)
