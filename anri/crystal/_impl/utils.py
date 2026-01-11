"""JAX-based crystallographic utilities.

This file may look a bit stupid - these are all very fundamental conversions with simple mathematics.
However, I feel it is important to have a single authoritative source for crystallographic conversion functions.
These can be tested against Dans_Diffraction and ImageD11 etc too.
"""

import jax
import jax.numpy as jnp


@jax.jit
def lpars_to_mt(lpars: jax.Array) -> jax.Array:
    r"""Convert lattice parameters to metric tensor.

    Parameters
    ----------
    lpars
        [6] Lattice parameters (a,b,c,alpha,beta,gamma) with angles in degrees

    Returns
    -------
    mt: jax.Array
        [3,3] metric tensor

    Notes
    -----
    From IUCr dictionary [1]_
    $$
    \tens{G_{ij}} =
    \begin{bmatrix}
    \vec{a} \cdot \vec{a} & \vec{a} \cdot \vec{b} & \vec{a} \cdot \vec{c} \\
    \vec{b} \cdot \vec{a} & \vec{b} \cdot \vec{b} & \vec{b} \cdot \vec{c} \\
    \vec{c} \cdot \vec{a} & \vec{c} \cdot \vec{b} & \vec{c} \cdot \vec{c}
    \end{bmatrix}
    $$

    From the general dot product definition: $\vec{a} \cdot \vec{b} = \abs{\vec{a}}\abs{\vec{b}}\cos\theta$

    Handily, we always know the angles between the basis vectors, they are $\left(\alpha, \beta, \gamma \right)$.

    Also we know for a vector $\vec{a}$ that $\vec{a} \cdot \vec{a} = \abs{\vec{a}}^2$.

    Therefore, we can rewrite:

    $$
    \tens{G_{ij}} =
    \begin{bmatrix}
    a^2            & a b \cos\gamma & a c \cos\beta  \\
    a b \cos\gamma &            b^2 & b c \cos\alpha \\
    a c \cos\beta  & b c \cos\alpha &            c^2
    \end{bmatrix}
    $$

    References
    ----------
    .. [1] https://dictionary.iucr.org/Metric_tensor
    """
    a, b, c, alpha, beta, gamma = lpars
    ca = jnp.cos(jnp.radians(alpha))
    cb = jnp.cos(jnp.radians(beta))
    cg = jnp.cos(jnp.radians(gamma))
    # fmt: off
    mt = jnp.array([[     a * a, a * b * cg, a * c * cb],
                                  [a * b * cg,      b * b, b * c * ca],
                                  [a * c * cb, b * c * ca,      c * c]])
    # fmt: on
    return mt


@jax.jit
def mt_to_lpars(mt: jax.Array) -> jax.Array:
    """Convert metric tensor back to lattice parameters.

    Parameters
    ----------
    mt
        [3,3] metric tensor

    Returns
    -------
    jax.Array
        [6] Lattice parameters (a,b,c,alpha,beta,gamma) with angles in degrees
    """
    a, b, c = jnp.sqrt(jnp.diag(mt))
    al = jnp.degrees(jnp.arccos(mt[1, 2] / b / c))
    be = jnp.degrees(jnp.arccos(mt[0, 2] / a / c))
    ga = jnp.degrees(jnp.arccos(mt[0, 1] / a / b))
    return jnp.array([a, b, c, al, be, ga])


@jax.jit
def mt_to_rmt(mt: jax.Array) -> jax.Array:
    """Convert metric tensor to reciprocal metric tensor.

    Parameters
    ----------
    mt
        [3,3] Metric tensor

    Returns
    -------
    rmt: jax.Array
        [3,3] Reciprocal metric tensor
    """
    rmt = jnp.linalg.inv(mt)
    return rmt


@jax.jit
def rmt_to_mt(rmt: jax.Array) -> jax.Array:
    """Convert reciprocal metric tensor to metric tensor.

    Parameters
    ----------
    rmt
        [3,3] Reciprocal metric tensor

    Returns
    -------
    mt: jax.Array
        [3,3] Metric tensor
    """
    mt = jnp.linalg.inv(rmt)
    return mt


@jax.jit
def rmt_to_B(rmt: jax.Array) -> jax.Array:
    r"""Convert reciprocal metric tensor to B matrix.

    Parameters
    ----------
    rmt
        [3,3] Reciprocal metric tensor

    Returns
    -------
    B: jax.Array
        [3,3] Reciprocal space orthogonalization matrix

    Notes
    -----
    $\tens{G^{ij}} = \matr{B^\dagger}\cdot\matr{B}$
    For any upper-triangular matrix $\matr{R}, $\matr{A} = \matr{R^\dagger} \cdot \matr{R}$,
    you can do upper-triangular Cholesky decomposition to recover $\matr{R}$ from $\matr{A}$.
    """
    B = jnp.linalg.cholesky(rmt, upper=True)
    return B


@jax.jit
def B_to_rmt(B: jax.Array) -> jax.Array:
    r"""Convert the B matrix to the reciprocal metric tensor.

    Parameters
    ----------
    B
        [3,3] Reciprocal space orthogonalization matrix

    Returns
    -------
    rmt: jax.Array
        [3,3] Reciprocal metric tensor

    Notes
    -----
    Just $\tens{G^ij} = \matr{B^\dagger} \cdot \matr{B}$
    """
    rmt = B.T @ B
    return rmt


@jax.jit
def hkl_B_to_q_crystal(hkl: jax.Array, B: jax.Array) -> jax.Array:
    """Convert hkl to Q in the crystal frame.

    Parameters
    ----------
    hkl
        [3] (h, k, l) like a Miller index.
    B
        [3,3] Reciprocal space orthogonalization matrix

    Returns
    -------
    q_crystal: jax.Array
        Q in the crystal frame
    """
    q_crystal = B @ hkl
    return q_crystal


@jax.jit
def volume_direct(mt: jax.Array) -> jax.Array:
    """Get the volume of the direct space unit cell from the metric tensor.

    Parameters
    ----------
    mt
        [3,3] Metric tensor

    Returns
    -------
    V_direct: jax.Array
        [1] Volume of direct space unit cell
    """
    V_direct = jnp.sqrt(jnp.linalg.det(mt))
    return V_direct


@jax.jit
def volume_recip(rmt: jax.Array) -> jax.Array:
    """Get the volume of the reciprocal space unit cell from the reciprocal metric tensor.

    Parameters
    ----------
    rmt
        [3,3] Reciprocal metric tensor

    Returns
    -------
    V_recip: jax.Array
        [1] Volume of reciprocal space unit cell
    """
    V_recip = jnp.sqrt(jnp.linalg.det(rmt))
    return V_recip


@jax.jit
def B_to_F(B: jax.Array) -> jax.Array:
    """Convert B matrix to F matrix.

    Parameters
    ----------
    B
        [3,3] Reciprocal space orthogonalization matrix

    Returns
    -------
    F: jax.Array
        [3,3] Real space fractionalization matrix
    """
    F = B.T
    return F


@jax.jit
def F_to_B(F: jax.Array) -> jax.Array:
    """Convert F matrix to B matrix.

    Parameters
    ----------
    F
        [3,3] Real space orthogonalization matrix

    Returns
    -------
    B: jax.Array
        [3,3] Reciprocal space orthogonalization matrix
    """
    B = F.T
    return B


@jax.jit
def F_to_O(F: jax.Array) -> jax.Array:
    """Convert F matrix to O matrix.

    Parameters
    ----------
    F
        [3,3] Real space fractionalization matrix

    Returns
    -------
    O: jax.Array
        [3,3] Real space orthogonalization matrix
    """
    O = jnp.linalg.inv(F)
    return O


@jax.jit
def O_to_F(O: jax.Array) -> jax.Array:
    """Convert O matrix to F matrix.

    Parameters
    ----------
    O
        [3,3] Real space orthogonalization matrix

    Returns
    -------
    F: jax.Array
        [3,3] Real space fractionalization matrix
    """
    F = jnp.linalg.inv(O)
    return F


@jax.jit
def O_to_A(O: jax.Array) -> jax.Array:
    """Convert O matrix to A matrix.

    Parameters
    ----------
    O
        [3,3] Real space orthogonalization matrix

    Returns
    -------
    A: jax.Array
        [3,3] Reciprocal space fractionalization matrix
    """
    A = O.T
    return A


@jax.jit
def A_to_O(A: jax.Array) -> jax.Array:
    """Convert A matrix to O matrix.

    Parameters
    ----------
    A
        [3,3] Reciprocal space fractionalization matrix

    Returns
    -------
    O: jax.Array
        [3,3] Real space orthogonalization matrix
    """
    O = A.T
    return O


@jax.jit
def UB_to_UBI(UB: jax.Array) -> jax.Array:
    """Invert U.B to get (U.B)^-1.

    Parameters
    ----------
    UB
        [3,3] U.B matrix

    Returns
    -------
    UBI: jax.Array
        [3,3] (U.B)^-1 matrix
    """
    return jnp.linalg.inv(UB)


@jax.jit
def UBI_to_UB(UBI: jax.Array) -> jax.Array:
    """Invert (U.B)^-1 to get (U.B).

    Parameters
    ----------
    UBI
        [3,3] (U.B)^-1 matrix

    Returns
    -------
    UB: jax.Array
        [3,3] U.B matrix
    """
    return jnp.linalg.inv(UBI)


@jax.jit
def UB_and_B_to_U(UB: jax.Array, B: jax.Array) -> jax.Array:
    """Get U matrix from UB and B.

    Parameters
    ----------
    UB
        [3,3] U.B matrix
    B
        [3,3] Reciprocal space orthogonalization matrix

    Returns
    -------
    U: jax.Arrray
        [3,3] U matrix
    """
    return UB @ jnp.linalg.inv(B)


@jax.jit
def UBI_to_mt(UBI: jax.Array) -> jax.Array:
    """Convert from (U.B)^-1 matrix to metric tensor.

    Parameters
    ----------
    UBI
        [3,3] (U.B)^-1 matrix

    Returns
    -------
    jax.Array
        [3,3] Metric tensor
    """
    mt = UBI @ UBI.T
    return mt


### cross-transforms for convenience


@jax.jit
def lpars_to_B(lpars: jax.Array) -> jax.Array:
    """Convert lattice parameters to B matrix (cross-transform).

    Parameters
    ----------
    lpars
        [6] Lattice parameters (a,b,c,alpha,beta,gamma) with angles in degrees

    Returns
    -------
    B: jax.Array
        [3,3] Reciprocal space orthogonalization matrix
    """
    mt = lpars_to_mt(lpars)
    rmt = mt_to_rmt(mt)
    B = rmt_to_B(rmt)
    return B


@jax.jit
def B_to_A(B: jax.Array) -> jax.Array:
    """Convert B matrix to A matrix (cross-transform).

    Parameters
    ----------
    B
        [3,3] Reciprocal space orthogonalization matrix

    Returns
    -------
    A
        [3,3] Reciprocal space fractionalization matrix
    """
    F = B_to_F(B)
    O = F_to_O(F)
    A = O_to_A(O)
    return A


@jax.jit
def B_to_mt(B: jax.Array) -> jax.Array:
    """Convert B matrix to metric tensor.

    Parameters
    ----------
    B
        [3,3] Reciprocal space orthogonalization matrix

    Returns
    -------
    mt: jax.Array
        [3,3] Metric tensor
    """
    rmt = B_to_rmt(B)
    mt = rmt_to_mt(rmt)
    return mt
