"""Utility functions for geometry transformations."""

import jax
import jax.numpy as jnp


@jax.jit
def skew(v: jax.Array) -> jax.Array:
    r"""Return the cross-product (skew-symmetric) matrix of vectors.

    $$
    [\vec{v}]_\times = \begin{pmatrix} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0 \end{pmatrix},
    \qquad [\vec{v}]_\times \vec{u} = \vec{v} \times \vec{u}
    $$

    It is antisymmetric, $[\vec{v}]_\times^T = -[\vec{v}]_\times$, and its square is
    $[\vec{v}]_\times^2 = \vec{v}\vec{v}^T - \abs{\vec{v}}^2 \matr{I}$, which lets rotation formulas avoid a
    matrix product. For a small rotation vector $\vec{\delta}$ (radians),
    $\matr{R} = \matr{I} + [\vec{\delta}]_\times$ to first order.

    Parameters
    ----------
    v
        [..., 3] vectors

    Returns
    -------
    jax.Array
        [..., 3, 3] matrices

    Notes
    -----
    The zero entries are ``0 * v_x``, not a constant, so that under ``shard_map`` every entry varies with ``v`` in the
    same way (JAX refuses to stack values that vary across devices with ones that do not).
    """
    v = jnp.asarray(v)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = 0.0 * x
    return jnp.stack([jnp.stack([o, -z, y], -1), jnp.stack([z, o, -x], -1), jnp.stack([-y, x, o], -1)], -2)


@jax.jit
def rmat_from_axis_angle(axis: jax.Array, angle: jax.Array) -> jax.Array:
    r"""Return rotation matrices for positive right-handed rotations about axes by angles (degrees).

    Rodrigues' formula, with $\hat{n}$ the unit axis and $\theta$ the angle:

    $$
    \matr{R} = \cos\theta \, \matr{I} + \sin\theta \, [\hat{n}]_\times + (1 - \cos\theta) \, \hat{n}\hat{n}^T
    $$

    (:func:`skew`), equivalently $\matr{I} + \sin\theta [\hat{n}]_\times + (1 - \cos\theta) [\hat{n}]_\times^2$,
    written without a matrix product.

    Parameters
    ----------
    axis
        [..., 3] Rotation axis vectors (need not be unit; not zero)
    angle
        [...] Rotation angles in degrees, broadcast against axis[..., 0]

    Returns
    -------
    R: jax.Array
        [..., 3, 3] Rotation matrices. $\matr{R} \cdot \vec{v_{\text{sample}}} = \vec{v_{\text{lab}}}$
    """
    axis = jnp.asarray(axis)
    n = axis / jnp.linalg.norm(axis, axis=-1, keepdims=True)
    t = jnp.radians(jnp.asarray(angle, axis.dtype))[..., None, None]
    c, s = jnp.cos(t), jnp.sin(t)
    return c * jnp.eye(3, dtype=axis.dtype) + s * skew(n) + (1.0 - c) * (n[..., :, None] * n[..., None, :])


@jax.jit
def rot_x(angle: float) -> jax.Array:
    """Return rotation matrix for positive right-handed rotation about the X axis by angle (degrees).

    Parameters
    ----------
    angle
        Rotation angle in degrees

    Returns
    -------
    R: jax.Array
        [3,3] Rotation matrix
    """
    a = jnp.radians(angle)
    c, s = jnp.cos(a), jnp.sin(a)
    # zeros_like/ones_like rather than literals so the matrix still
    # broadcasts correctly when angle is an array.
    z, o = jnp.zeros_like(c), jnp.ones_like(c)
    return jnp.array([[o, z, z], [z, c, -s], [z, s, c]])


@jax.jit
def rot_y(angle: float) -> jax.Array:
    """Return rotation matrix for positive right-handed rotation about the Y axis by angle (degrees).

    Parameters
    ----------
    angle
        Rotation angle in degrees

    Returns
    -------
    R: jax.Array
        [3,3] Rotation matrix
    """
    a = jnp.radians(angle)
    c, s = jnp.cos(a), jnp.sin(a)
    # zeros_like/ones_like rather than literals so the matrix still
    # broadcasts correctly when angle is an array.
    z, o = jnp.zeros_like(c), jnp.ones_like(c)
    return jnp.array([[c, z, s], [z, o, z], [-s, z, c]])


@jax.jit
def rot_z(angle: float) -> jax.Array:
    """Return rotation matrix for positive right-handed rotation about the Z axis by angle (degrees).

    Parameters
    ----------
    angle
        Rotation angle in degrees

    Returns
    -------
    R: jax.Array
        [3,3] Rotation matrix

    Notes
    -----
    An example - with 90 degrees:
    Rz(90) @ (X,Y,Z)_sample = (Y,-X,Z)_lab

    Imagine a diffractometer. If we rotate it 90 degrees, via the right-hand rule, the sample X axis goes to lab Y.

    Therefore we get R @ v_sample = v_lab.
    """
    a = jnp.radians(angle)
    c, s = jnp.cos(a), jnp.sin(a)
    # zeros_like/ones_like rather than literals so the matrix still
    # broadcasts correctly when angle is an array.
    z, o = jnp.zeros_like(c), jnp.ones_like(c)
    return jnp.array([[c, -s, z], [s, c, z], [z, z, o]])


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
