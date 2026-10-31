"""Intensity factors of a reflection from its incoming and outgoing wave-vectors: Lorentz, polarisation and extinction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
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


_R_E = 2.8179403262e-5  # classical electron radius e^2 / mc^2, angstrom


def extinction(x: ArrayLike, X: ArrayLike, tth: ArrayLike, gaussian: ArrayLike) -> jax.Array:
    """Extinction factor ``y = y_p y_s`` of one polarisation component (Becker and Coppens, 1974).

    The integrated intensity is the kinematical one times ``y`` (1: no extinction) [1]_. Primary extinction, inside a
    perfect spherical domain, is their equations 37 and 38; secondary extinction, in a spherical mosaic crystal of
    such domains, is their equation 37 with the coefficients of equation 43 (Gaussian mosaic distribution, with
    ``2.12 X`` for ``2 X``: footnote to their Table 5) or 43a (Lorentzian). The two multiply (their equation 24).
    The fits hold for ``0 < x, X < 30``.

    Parameters
    ----------
    x, X
        Primary and secondary extinction parameters of this component (:func:`extinction_parameters` times ``K^2``:
        1 for the field normal to the scattering plane, ``cos^2 2theta`` in it)
    tth
        Scattering angle 2theta (radians)
    gaussian
        True for a Gaussian mosaic distribution, False for a Lorentzian

    References
    ----------
    .. [1] Becker, P.J., Coppens, P., 1974. Extinction within the limit of validity of the Darwin transfer equations.
       I. General formalisms for primary and secondary extinction and their application to spherical crystals. Acta
       Crystallographica Section A 30, 129–147. https://doi.org/10.1107/S0567739474000337
    """
    x, X, c = jnp.asarray(x), jnp.asarray(X), jnp.cos(tth)
    y_p = (1 + 2 * x + (0.20 + 0.45 * c) * x**2 / (1 + (0.22 - 0.12 * (0.5 - c) ** 2) * x)) ** -0.5
    y_g = (1 + 2.12 * X + (0.58 + 0.48 * c + 0.24 * c**2) * X**2 / (1 + (0.02 - 0.025 * c) * X)) ** -0.5
    b_l = jnp.where(c > 0, 0.15 - 0.2 * (0.75 - c) ** 2, -0.45 * c)
    y_l = (1 + 2 * X + (0.025 + 0.285 * c) * X**2 / (1 + b_l * X)) ** -0.5
    return y_p * jnp.where(gaussian, y_g, y_l)


def _sphere_path_ratio(mu_r: ArrayLike, tth: ArrayLike, n: int = 12) -> jax.Array:
    """Absorption-weighted mean path through a sphere over its unweighted mean (3 R / 2), at scattering angle tth.

    Becker and Coppens (1974, equation 52): the paths ``T1`` back along the incident beam and ``T2`` on along the
    diffracted beam, averaged over the sphere with weight ``exp(-mu (T1 + T2))``. ``mu_r`` is mu times the radius.
    Gauss-Legendre quadrature in radius and polar angle, uniform in azimuth (n points each).
    """
    g, w = (jnp.asarray(a) for a in np.polynomial.legendre.leggauss(n))
    r = 0.5 * (g + 1)  # radius in [0, 1], weight r^2 dr
    phi = 2 * jnp.pi * (jnp.arange(n) + 0.5) / n
    rr, cc, pp = jnp.meshgrid(r, g, phi, indexing="ij")
    wt = (0.5 * w[:, None, None] * r[:, None, None] ** 2) * w[None, :, None] * jnp.ones_like(pp)
    s = jnp.sqrt(1 - cc**2)
    p = jnp.stack([rr * cc, rr * s * jnp.cos(pp), rr * s * jnp.sin(pp)], -1)
    u0 = jnp.array([1.0, 0.0, 0.0])
    u = jnp.stack([jnp.cos(tth), jnp.sin(tth), 0.0])
    a0, a1, p2 = p @ u0, p @ u, (p**2).sum(-1)
    t = (a0 + jnp.sqrt(jnp.maximum(a0**2 - p2 + 1, 0))) + (-a1 + jnp.sqrt(jnp.maximum(a1**2 - p2 + 1, 0)))
    e = wt * jnp.exp(-jnp.asarray(mu_r, t.dtype) * t)
    return (e * t).sum() / e.sum() / 1.5


def extinction_parameters(
    F2: ArrayLike, volume: float, wavelength: float, tth: ArrayLike, domain_radius: float, crystal_radius: float,
    mosaic_g: float = float("inf"), gaussian: bool = True, mu: float = 0.0,
) -> tuple:  # fmt: skip
    """Primary and secondary extinction parameters (x, X) of reflections, for the field normal to the scattering plane.

    A phase is modelled as Darwin's mosaic crystal: spherical perfect domains of radius ``domain_radius``, misaligned
    by a Gaussian or Lorentzian distribution of width parameter ``mosaic_g``, making up a spherical crystal of radius
    ``crystal_radius`` (Zachariasen, 1967) [1]_. With the diffracting power per unit length
    ``Q = (r_e |F| / V)^2 lambda^3 / sin 2theta`` (Becker and Coppens, 1974, equation 5) [2]_ and
    ``a = 3 r sin 2theta / (2 lambda)`` (their 35b):

    - primary, ``x = (2 / 3) Q a t``, ``t = 3 r / 2`` the mean path through a domain (their 35a);
    - secondary, ``X = (2 / 3) Q alpha T`` (their 42), ``T = 3 R / 2``, with ``alpha = a / sqrt(1 + a^2 / (2 g^2))``
      for a Gaussian distribution and ``a / (1 + 2 a / (3 g))`` for a Lorentzian (their 40b, 41b);
    - with absorption, ``T`` is the absorption-weighted mean path through the crystal (their 52 and 53b), a
      first-order correction they find good for ``mu R`` up to about 1 at small Bragg angles (their 54).

    The parallel component has ``K^2 = cos^2 2theta`` times these (their section III.1, :func:`polarisation_extinction`).

    Parameters
    ----------
    F2
        [N] ``|F|^2`` per unit cell, absolute (electrons^2), e.g. :func:`anri.crystal.structure.structure_factors`
    volume
        Unit cell volume (angstrom^3)
    wavelength
        Angstrom
    tth
        [N] Scattering angles 2theta (radians)
    domain_radius, crystal_radius
        Radii of a perfect domain and of the mosaic crystal (micrometres, as voxel positions)
    mosaic_g
        Width parameter g of the mosaic distribution (1 / radian): ``W = sqrt(2) g exp(-2 pi g^2 eps^2)`` (Gaussian;
        Zachariasen's equation 3, standard deviation ``1 / (2 sqrt(pi) g)``) or ``2 g / (1 + 4 pi^2 eps^2 g^2)``
        (Lorentzian, Becker and Coppens' 39b). Infinite (the default): domains aligned, ``alpha = a``
    gaussian
        Gaussian (True) or Lorentzian (False) mosaic distribution
    mu
        Linear attenuation coefficient of the phase (1 / micrometre, :func:`anri.crystal.structure.attenuation`)

    Returns
    -------
    tuple
        x [N], X [N]

    References
    ----------
    .. [1] Zachariasen, W.H., 1967. A general theory of X-ray diffraction in crystals. Acta Crystallographica 23,
       558–564. https://doi.org/10.1107/S0365110X67003202
    .. [2] Becker, P.J., Coppens, P., 1974. Extinction within the limit of validity of the Darwin transfer equations.
       I. General formalisms for primary and secondary extinction and their application to spherical crystals. Acta
       Crystallographica Section A 30, 129–147. https://doi.org/10.1107/S0567739474000337
    """
    tth = jnp.asarray(tth)
    s2 = jnp.sin(tth)
    r, R = 1e4 * domain_radius, 1e4 * crystal_radius  # angstrom
    Q = _R_E**2 * jnp.asarray(F2) / volume**2 * wavelength**3 / s2
    a = 1.5 * r * s2 / wavelength
    x = (2 / 3) * Q * a * 1.5 * r
    alpha = a / jnp.sqrt(1 + a**2 / (2 * mosaic_g**2)) if gaussian else a / (1 + 2 * a / (3 * mosaic_g))
    ratio = jax.vmap(lambda t: _sphere_path_ratio(mu * crystal_radius, t))(tth.ravel()).reshape(tth.shape)
    X = (2 / 3) * Q * alpha * 1.5 * R * ratio
    return x, X


def polarisation_extinction(
    k_in: jax.Array, k_out: jax.Array, factor: ArrayLike, x: ArrayLike, X: ArrayLike, gaussian: ArrayLike
) -> jax.Array:
    """:func:`polarisation` times the extinction factor, each field component extinguished separately.

    A field component normal to the scattering plane (``k_in x k_out``) scatters with ``K = 1``, one in it with
    ``K = cos 2theta``; extinction acts on each with its own ``K^2 x`` (Becker and Coppens, 1974, equations 45-46)
    [1]_. Splitting the beam's two fields (:func:`polarisation`) by the plane's normal ``n`` gives the weights
    ``w = sum p_e (e . n)^2`` and ``1 - w``, so that ``polarisation = w + (1 - w) cos^2 2theta`` and this is
    ``w y(x, X) + (1 - w) cos^2 2theta y(x cos^2 2theta, X cos^2 2theta)``, :func:`extinction`. With ``x = X = 0`` it is
    :func:`polarisation`.

    Parameters
    ----------
    k_in, k_out
        [3] Incoming and outgoing wave-vectors (any length)
    factor
        Degree of horizontal polarisation, see :func:`polarisation`
    x, X, gaussian
        This reflection's primary and secondary parameters for ``K = 1`` (:func:`extinction_parameters`) and its
        mosaic distribution

    References
    ----------
    .. [1] Becker, P.J., Coppens, P., 1974. Extinction within the limit of validity of the Darwin transfer equations.
       I. General formalisms for primary and secondary extinction and their application to spherical crystals. Acta
       Crystallographica Section A 30, 129–147. https://doi.org/10.1107/S0567739474000337
    """
    _, e_h, e_v = beam_basis(k_in)
    k_in = k_in / jnp.linalg.norm(k_in)
    k_out = k_out / jnp.linalg.norm(k_out)
    n = jnp.cross(k_in, k_out)
    n = n / jnp.maximum(jnp.linalg.norm(n), 1e-12)
    w = 0.5 * (1 + factor) * (e_h @ n) ** 2 + 0.5 * (1 - factor) * (e_v @ n) ** 2
    c2 = (k_in @ k_out) ** 2
    tth = jnp.arccos(jnp.clip(k_in @ k_out, -1.0, 1.0))
    return w * extinction(x, X, tth, gaussian) + (1 - w) * c2 * extinction(x * c2, X * c2, tth, gaussian)


def lorentz_polarisation(
    k_in: jax.Array, k_out: jax.Array, wedge: ArrayLike, chi: ArrayLike, pol_factor: ArrayLike,
    ext: tuple | None = None,
) -> jax.Array:  # fmt: skip
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
    ext
        Optional ``(x, X, gaussian)`` of this reflection (:func:`extinction_parameters`): the polarisation factor is
        then :func:`polarisation_extinction`
    """
    rot_axis = sample_to_lab(jnp.array([0.0, 0.0, 1.0], k_in.dtype), 0.0, wedge, chi, 0.0, 0.0)
    if ext is None:
        return lorentz(k_in, k_out, rot_axis) * polarisation(k_in, k_out, pol_factor)
    return lorentz(k_in, k_out, rot_axis) * polarisation_extinction(k_in, k_out, pol_factor, *ext)
