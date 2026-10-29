r"""Rotations: matrices, axis-angle, Rodrigues vectors and quaternions, angles and means, random rotations.

Pure rotation algebra, with no crystal symmetry (that is :mod:`anri.crystal`, built on this). Rotations are
right-handed and active: :math:`\matr{R} \vec{v}` turns the vector :math:`\vec{v}`, and a matrix taking sample to lab
coordinates satisfies :math:`\matr{R} \cdot \vec{v}_{\text{sample}} = \vec{v}_{\text{lab}}`. Angles are in degrees
where they are motor or user angles, radians where they are rotation-vector lengths.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .utils import float64


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

    Rodrigues' formula (e.g. Rowenhorst et al., 2015, equation 15 [1]_), with $\hat{n}$ the unit axis and $\theta$
    the angle:

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

    References
    ----------
    .. [1] Rowenhorst, D., Rollett, A.D., Rohrer, G.S., Groeber, M., Jackson, M., Konijnenberg, P.J., De Graef, M.,
       2015. Consistent representations of and conversions between 3D rotations. Modelling and Simulation in Materials
       Science and Engineering 23, 083501. https://doi.org/10.1088/0965-0393/23/8/083501
    """
    axis = jnp.asarray(axis)
    n = axis / jnp.linalg.norm(axis, axis=-1, keepdims=True)
    t = jnp.radians(jnp.asarray(angle, axis.dtype))[..., None, None]
    c, s = jnp.cos(t), jnp.sin(t)
    return c * jnp.eye(3, dtype=axis.dtype) + s * skew(n) + (1.0 - c) * (n[..., :, None] * n[..., None, :])


def rod_to_mat(r: ArrayLike) -> jax.Array:
    r"""Convert Rodrigues vectors ``r = tan(t / 2) n`` to rotation matrices.

    $$
    \matr{R} = \matr{I} + \frac{2}{1 + \vec{r} \cdot \vec{r}}
    \left([\vec{r}]_\times + \vec{r}\vec{r}^T - (\vec{r} \cdot \vec{r}) \matr{I}\right)
    $$

    with :func:`skew`, the second and third terms being $[\vec{r}]_\times^2$.

    Parameters
    ----------
    r
        [..., 3] Rodrigues vectors

    Returns
    -------
    jax.Array
        [..., 3, 3] rotation matrices
    """
    r = jnp.asarray(r)
    rr = jnp.sum(r * r, -1)[..., None, None]
    KK = r[..., :, None] * r[..., None, :] - rr * jnp.eye(3, dtype=r.dtype)  # [r]x^2, without a matrix product
    return jnp.eye(3, dtype=r.dtype) + 2.0 / (1.0 + rr) * (skew(r) + KK)


def mat_to_rod(U: ArrayLike) -> np.ndarray:
    """Convert rotation matrices (rotation angle below 180 degrees) to Rodrigues vectors.

    Parameters
    ----------
    U
        [..., 3, 3] rotation matrices

    Returns
    -------
    np.ndarray
        [..., 3] Rodrigues vectors
    """
    U = np.asarray(U)
    w = np.stack([U[..., 2, 1] - U[..., 1, 2], U[..., 0, 2] - U[..., 2, 0], U[..., 1, 0] - U[..., 0, 1]], -1)
    return w / (1.0 + np.trace(U, axis1=-2, axis2=-1))[..., None]


def mat_to_quat(U: ArrayLike) -> np.ndarray:
    """Convert rotation matrices to unit quaternions ``(w, x, y, z)`` with ``w >= 0``.

    By Shepperd's method [1]_: of the four formulas for the quaternion, each dividing by one component, use the one
    whose component is largest (4 w^2 = 1 + trace, 4 x_i^2 = 1 + 2 U_ii - trace), the best conditioned. Comparing the
    trace with the diagonal elements picks it.

    Parameters
    ----------
    U
        [..., 3, 3] rotation matrices

    Returns
    -------
    np.ndarray
        [..., 4] quaternions

    References
    ----------
    .. [1] Shepperd, S.W., 1978. Quaternion from rotation matrix. Journal of Guidance and Control 1, 223–224.
       https://doi.org/10.2514/3.55767b
    """
    U = np.asarray(U, float)
    tr = np.trace(U, axis1=-2, axis2=-1)
    # the four candidates of Shepperd's method; take the best-conditioned one for each matrix
    d = np.stack([tr, U[..., 0, 0], U[..., 1, 1], U[..., 2, 2]], -1)
    k = np.argmax(d, -1)
    q = np.zeros(U.shape[:-2] + (4,))
    a = np.stack(
        [
            1 + tr,
            U[..., 2, 1] - U[..., 1, 2],
            U[..., 0, 2] - U[..., 2, 0],
            U[..., 1, 0] - U[..., 0, 1],
        ],
        -1,
    )
    b = np.stack(
        [
            U[..., 2, 1] - U[..., 1, 2],
            1 + 2 * U[..., 0, 0] - tr,
            U[..., 0, 1] + U[..., 1, 0],
            U[..., 0, 2] + U[..., 2, 0],
        ],
        -1,
    )
    c = np.stack(
        [
            U[..., 0, 2] - U[..., 2, 0],
            U[..., 0, 1] + U[..., 1, 0],
            1 + 2 * U[..., 1, 1] - tr,
            U[..., 1, 2] + U[..., 2, 1],
        ],
        -1,
    )
    e = np.stack(
        [
            U[..., 1, 0] - U[..., 0, 1],
            U[..., 0, 2] + U[..., 2, 0],
            U[..., 1, 2] + U[..., 2, 1],
            1 + 2 * U[..., 2, 2] - tr,
        ],
        -1,
    )
    for i, v in enumerate((a, b, c, e)):
        m = k == i
        q[m] = v[m]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q * np.where(q[..., :1] < 0, -1.0, 1.0)


def quat_to_mat(q: ArrayLike) -> np.ndarray:
    """Convert unit quaternions ``(w, x, y, z)`` to rotation matrices.

    Parameters
    ----------
    q
        [..., 4] quaternions

    Returns
    -------
    np.ndarray
        [..., 3, 3] rotation matrices
    """
    q = np.asarray(q, float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def quat_mul(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Multiply quaternions ``(w, x, y, z)``: the rotation ``b`` followed by ``a``.

    Parameters
    ----------
    a, b
        [..., 4] quaternions (broadcast together)

    Returns
    -------
    np.ndarray
        [..., 4] products
    """
    a, b = np.asarray(a), np.asarray(b)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def rotation_angle(R: ArrayLike) -> jax.Array:
    """Rotation angle (radians) of rotation matrices, accurate at every angle.

    From the axial part (sin) and the trace (cos) together, ``atan2(|axial|, (trace - 1) / 2)``: ``arccos`` of the
    trace alone loses small angles (in float32, any angle below ~0.03 degrees).

    Parameters
    ----------
    R
        [..., 3, 3] rotation matrices

    Returns
    -------
    jax.Array
        [...] angles in [0, pi]
    """
    R = jnp.asarray(R)
    axial = 0.5 * jnp.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    return jnp.arctan2(jnp.linalg.norm(axial, axis=-1), 0.5 * (jnp.trace(R, axis1=-2, axis2=-1) - 1.0))


def mean_rotation(U: ArrayLike, w: ArrayLike) -> jax.Array:
    """Weighted mean of nearby rotations: the rotation closest to their weighted sum (SVD projection).

    The weighted arithmetic mean projected back onto the rotations, in the Frobenius norm (Moakher, 2002) [1]_. Align
    symmetric equivalents first (:func:`~anri.crystal.orientation.closest_equivalent`). The projection keeps det = +1,
    so the result is always a proper rotation. With all weights zero it is the identity.

    Parameters
    ----------
    U
        [M, 3, 3] rotations
    w
        [M] weights

    Returns
    -------
    jax.Array
        [3, 3] the mean rotation

    References
    ----------
    .. [1] Moakher, M., 2002. Means and averaging in the group of rotations. SIAM Journal on Matrix Analysis and
       Applications 24, 1–16. https://doi.org/10.1137/S0895479801383877
    """
    U, w = jnp.asarray(U), jnp.asarray(w)
    total = jnp.einsum("m,mij->ij", w, U) + jnp.where(jnp.sum(w) > 0, 0.0, 1.0) * jnp.eye(3, dtype=U.dtype)
    u, _, vt = jnp.linalg.svd(total)
    return u @ jnp.diag(jnp.array([1.0, 1.0, jnp.sign(jnp.linalg.det(u @ vt))], U.dtype)) @ vt


def random_rotations(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw rotations uniformly over SO(3).

    Parameters
    ----------
    n
        How many
    rng
        Random generator

    Returns
    -------
    np.ndarray
        [n, 3, 3] rotation matrices
    """
    q = rng.normal(size=(n, 4))
    return quat_to_mat(q / np.linalg.norm(q, axis=1, keepdims=True))


def small_rotations(n: int, sigma_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Draw small rotations: rotation vectors with each component normal with standard deviation sigma_deg.

    Parameters
    ----------
    n
        How many
    sigma_deg
        Standard deviation of each component, degrees
    rng
        Random generator

    Returns
    -------
    np.ndarray
        [n, 3, 3] rotation matrices
    """
    v = rng.normal(scale=sigma_deg, size=(n, 3))
    with float64():
        return np.asarray(
            rmat_from_axis_angle(v + (np.linalg.norm(v, axis=1, keepdims=True) == 0), np.linalg.norm(v, axis=1))
        )
