"""The beam: directions along and across it, where a point sits relative to it, and how much of a voxel it lights.

Its profile across it, horizontally and vertically, is a flat top of some width blurred by a Gaussian (a pure
Gaussian with zero width), and integrates to 1.
"""

import jax
import jax.numpy as jnp
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
    """
    x, width, sigma = jnp.asarray(x), jnp.asarray(width), jnp.asarray(sigma)
    gauss = width < 0.01 * sigma
    w = jnp.where(gauss, 1.0, width)  # double-where: no NaN gradients from either branch
    ends = jnp.stack([x + w / 2, x - w / 2])
    if order == 1:
        r = ramp_integral(ends, sigma)
        return jnp.where(gauss, ndtr(x / sigma), (r[0] - r[1]) / w)
    r = ramp_integral2(ends, sigma)
    return jnp.where(gauss, ramp_integral(x, sigma), (r[0] - r[1]) / w)


def chord_weight(
    u: ArrayLike, omega: ArrayLike, voxel_size: ArrayLike, sigma: ArrayLike, width: ArrayLike
) -> jax.Array:
    """Beam profile integrated over a square voxel, per unit height, for a beam along lab x.

    The voxel is a square of side ``voxel_size`` rotated by ``omega`` (degrees) and ``u`` across the beam's centre
    (lab y). Its chord along the beam, as a function of lab y, is a trapezoid of area ``voxel_size**2``.
    """
    c = jnp.abs(jnp.cos(jnp.radians(omega)))
    s = jnp.abs(jnp.sin(jnp.radians(omega)))
    a = 0.5 * voxel_size * (c + s)  # half-width of the trapezoid at its base
    b = 0.5 * voxel_size * jnp.abs(c - s)  # half-width at its top
    height = voxel_size**2 / (a + b)

    # Near omega = 0, 90, ... the trapezoid becomes a rectangle and (a - b) -> 0: its four-term difference loses
    # ~eps x s2 / (a - b), the box's error grows as (a - b)^2. Balanced at 1e-6 in float64, ~3e-3 in float32 (weight
    # errors <~ 1e-3 within ~0.2 deg of the kink). Double-where so neither branch produces NaN gradients.
    tol = 1e-6 if jnp.finfo(jnp.result_type(a)).eps < 1e-10 else 3e-3
    is_box = (a - b) < tol * voxel_size
    ab = jnp.where(is_box, 1.0, a - b)
    s2 = _smoothed(2, jnp.stack([u + a, u + b, u - b, u - a]), width, sigma)
    trapezoid = height / ab * (s2[0] - s2[1] - s2[2] + s2[3])
    s1 = _smoothed(1, jnp.stack([u + a, u - a]), width, sigma)
    box = height * (s1[0] - s1[1])
    return jnp.where(is_box, box, trapezoid)


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
    v = _smoothed(1, jnp.stack([dz + size / 2, dz - size / 2]), width_v, sig_v)
    v = v[0] - v[1]
    return w * jnp.where(geom["voxel_3d"], v, 1.0)
