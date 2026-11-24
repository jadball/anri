"""The beam: directions along and across it, where a point sits relative to it, and how much of a voxel it lights.

Its profile across it, horizontally and vertically, is a flat top of some width blurred by a Gaussian (a pure
Gaussian with zero width), and integrates to 1.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr
from jax.typing import ArrayLike

from anri.maths.utils import ramp_integral, ramp_integral2


@jax.jit
def beam_basis(k_in_lab: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Return unit vectors along a beam and across it, horizontally and vertically, in the lab frame.

    Horizontal is perpendicular to the beam and to lab z (the untilted omega axis); vertical is perpendicular to
    the beam and to horizontal. For a beam along lab x they are lab y and lab z. Beam divergence (``ky``, ``kz``)
    and polarisation are defined along these directions, so they follow the beam wherever it points.

    Parameters
    ----------
    k_in_lab
        [3] Direction of the incoming beam (need not be normalised; must not be vertical)

    Returns
    -------
    k_hat: jax.Array
        [3] Unit vector along the beam
    e_h: jax.Array
        [3] Horizontal unit vector across the beam, :math:`\hat{z} \times \hat{k}` normalised
    e_v: jax.Array
        [3] Vertical unit vector across the beam, :math:`\hat{k} \times \hat{e}_h`
    """
    k_hat = k_in_lab / jnp.linalg.norm(k_in_lab)
    e_h = jnp.cross(jnp.array([0.0, 0.0, 1.0], dtype=k_hat.dtype), k_hat)
    e_h = e_h / jnp.linalg.norm(e_h)
    return k_hat, e_h, jnp.cross(k_hat, e_h)


@jax.jit
def beam_offsets(pos_lab: jax.Array, k_in_lab: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Return where a lab point sits relative to the beam's centre line, which passes through the lab origin.

    Parameters
    ----------
    pos_lab
        [3] Point in the lab frame
    k_in_lab
        [3] Direction of the incoming beam (need not be normalised; must not be vertical)

    Returns
    -------
    across: jax.Array
        Distance across the beam, horizontally (along :math:`\hat{e}_h` of :func:`beam_basis`)
    height: jax.Array
        Height of the point above the beam's centre line, measured vertically (along lab z)
    rise: jax.Array
        Height of the beam's centre line at the point's horizontal position: ``pos_lab[2] - height``. A beam
        tilted by :math:`\alpha` out of the horizontal plane rises by :math:`\tan\alpha` per unit of horizontal
        distance along it.
    cos_alpha: jax.Array
        Cosine of the beam's tilt out of the horizontal plane
    """
    k, e_h, _ = beam_basis(k_in_lab)
    cos_alpha = jnp.hypot(k[0], k[1])
    along = (pos_lab[0] * k[0] + pos_lab[1] * k[1]) / cos_alpha  # horizontal distance along the beam
    rise = k[2] / cos_alpha * along
    return pos_lab @ e_h, pos_lab[2] - rise, rise, cos_alpha


def _smoothed(order: int, x: ArrayLike, width: ArrayLike, sigma: ArrayLike) -> jax.Array:
    """Antiderivative ``order`` of the beam profile: a flat top of ``width`` blurred by a Gaussian of ``sigma``.

    Order 1 is the profile's cumulative distribution, order 2 its integral. The profile integrates to 1. For a
    width much smaller than sigma it is the Gaussian's (the difference quotient below would lose precision).

    Both ends of the flat top are evaluated as one array (callers stack their points too): each call of the ramp
    integrals inlines a copy of ndtr, and the beam weight is differentiated twice in the renderer.

    The ramp integrals are evaluated at -|x| only, where they are small (:eq:`int-even`: P1(x) = 1 - P1(-x),
    P2(x) = x + P2(-x)); at large x they grow like x and differences of them would cancel. Selected by a where,
    whose branches are both exact, so derivatives of every order are right at x = 0 (-|x| would lose the second).
    """
    x, width, sigma = jnp.asarray(x), jnp.asarray(width), jnp.asarray(sigma)
    gauss = width < 0.01 * sigma
    w = jnp.where(gauss, 1.0, width)  # double-where: no NaN gradients from either branch
    pos = x > 0
    xn = jnp.where(pos, -x, x)
    ends = jnp.stack([xn + w / 2, xn - w / 2])
    if order == 1:
        r = ramp_integral(ends, sigma)
        p = jnp.where(gauss, ndtr(xn / sigma), (r[0] - r[1]) / w)
        return jnp.where(pos, 1.0 - p, p)
    r = ramp_integral2(ends, sigma)
    p = jnp.where(gauss, ramp_integral(xn, sigma), (r[0] - r[1]) / w)
    return jnp.where(pos, x + p, p)


def chord_weight(
    u: ArrayLike, omega: ArrayLike, voxel_size: ArrayLike, sigma: ArrayLike, width: ArrayLike
) -> jax.Array:
    """Beam profile integrated over a square voxel, per unit height, for a beam along lab x.

    The voxel is a square of side ``voxel_size`` rotated by ``omega`` (degrees) and ``u`` across the beam's centre
    (lab y). Its chord along the beam, as a function of lab y, is a trapezoid of area ``voxel_size**2``.

    :eq:`int-chord`, at u <= 0 (the weight is even in u, :eq:`int-even`, and there every term is a tail). Near
    omega = 0, 90, ... the trapezoid's half-widths a and b meet and its four-term difference cancels, losing
    ~eps x P2 / (a - b): below (a - b) = 2.5 sigma (float32) or 0.5 sigma (float64) the weight is the mean of the box
    over [b, a] by 6-point Gauss-Legendre instead (:eq:`int-chord-mean`), analytic in a and b: there its error and
    that of its first two omega derivatives are below the four-term form's rounding (the beam fold takes two).
    Double-where so neither branch produces NaN gradients.
    """
    u = jnp.asarray(u)
    u = jnp.where(u > 0, -u, u)  # not -|u|: its second derivative at 0 would be 0
    c = jnp.abs(jnp.cos(jnp.radians(omega)))
    s = jnp.abs(jnp.sin(jnp.radians(omega)))
    a = 0.5 * voxel_size * (c + s)  # half-width of the trapezoid at its base
    b = 0.5 * voxel_size * jnp.abs(c - s)  # half-width at its top
    height = voxel_size**2 / (a + b)

    dtype = jnp.result_type(a)
    near = (a - b) < (0.5 if jnp.finfo(dtype).eps < 1e-10 else 2.5) * sigma
    ab = jnp.where(near, 1.0, a - b)
    s2 = _smoothed(2, jnp.stack([u + a, u + b, u - b, u - a]), width, sigma)
    trapezoid = height / ab * (s2[0] - s2[1] - s2[2] + s2[3])
    nodes, weights = (
        np.asarray(x, dtype).reshape((6,) + (1,) * jnp.ndim(a)) for x in np.polynomial.legendre.leggauss(6)
    )
    t = 0.5 * (a + b) + 0.5 * (a - b) * nodes  # [6, ...] the nodes in [b, a]
    s1 = _smoothed(1, jnp.stack([u + t, u - t]), width, sigma)
    box_mean = height * jnp.sum(0.5 * weights * (s1[0] - s1[1]), axis=0)
    return jnp.where(near, box_mean, trapezoid)


def beam_weight(pos_lab: jax.Array, omega: ArrayLike, geom: dict) -> jax.Array:
    """How much of a voxel the beam illuminates: its profile integrated over the voxel.

    The beam's profile across it, horizontally (``geom["width_beam"]``, ``geom["sig_beam"]``) and vertically
    (``geom["width_beam_v"]``, ``geom["sig_beam_v"]``), is a flat top of that width blurred by a Gaussian of that
    standard deviation, and integrates to 1: a wider beam spreads the same flux. A voxel is a square of side
    ``geom["voxel_size"]`` in the sample xy plane, rotated by ``omega``, centred at ``pos_lab``, and either:

    - a column (2D map, ``geom["voxel_3d"]`` false): the whole vertical profile crosses it, so it integrates out;
      a beam tilted by ``alpha`` out of the horizontal plane travels ``1 / cos(alpha)`` further through it.
    - a cube (3D map): the vertical profile is integrated over the cube's height, measured vertically. For a
      tilted beam this treats the horizontal and vertical directions separately, which is exact only when it is
      horizontal.

    A beam at ``psi`` from lab x in the horizontal plane sees the voxel rotated by ``omega - psi``.

    Parameters
    ----------
    pos_lab
        [3] Lab position of the voxel's centre (in that frame, with its dty)
    omega
        Omega angle (degrees)
    geom
        Geometry dict, see :func:`anri.fwd.render.render_row`
    """
    k = beam_basis(geom["k_in_lab"])[0]
    u, dz, _, cos_alpha = beam_offsets(pos_lab, geom["k_in_lab"])
    size = geom["voxel_size"]
    psi = jnp.degrees(jnp.arctan2(k[1], k[0]))
    w = chord_weight(u, omega - psi, size, geom["sig_beam"], geom["width_beam"]) / cos_alpha
    sig_v = jnp.where(geom["voxel_3d"], geom["sig_beam_v"], 1.0) / cos_alpha  # unused for columns
    width_v = geom["width_beam_v"] / cos_alpha
    dz = jnp.where(dz > 0, -dz, dz)  # the cube's share is even in dz (:eq:`int-even`): at dz <= 0, no 1 - 1
    v = _smoothed(1, jnp.stack([dz + size / 2, dz - size / 2]), width_v, sig_v)
    v = v[0] - v[1]
    return w * jnp.where(geom["voxel_3d"], v, 1.0)
