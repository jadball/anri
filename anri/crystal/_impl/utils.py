r"""JAX-based crystallographic utilities.

This file may look a bit stupid - these are all very fundamental conversions with simple mathematics.
However, I feel it is important to have a single authoritative source for crystallographic conversion functions.
These can be tested against Dans_Diffraction and ImageD11 etc too.

I have deliberately avoided defining the direct matrix $\matr{A}$, because you have to make a choice about how it aligns to $\matr{B}$.
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
def mt_to_lpars(metric: jax.Array) -> jax.Array:
    r"""Convert (direct or reciprocal) metric tensor back to (direct or reciprocal) lattice parameters.

    Parameters
    ----------
    metric
        [3,3] direct or reciprocal metric tensor

    Returns
    -------
    jax.Array
        [6] Direct or reciprocal lattice parameters (a,b,c,alpha,beta,gamma) with angles in degrees
    
    Notes
    -----
    For a metric tensor $\tens{G}$ (direct or reciprocal), given:
    $$
    \tens{G} =
    \begin{bmatrix}
    a^2            & a b \cos\gamma & a c \cos\beta  \\
    a b \cos\gamma &            b^2 & b c \cos\alpha \\
    a c \cos\beta  & b c \cos\alpha &            c^2
    \end{bmatrix}
    $$
    then:
    $$
    \begin{aligned}
    a &= \sqrt{\tens{G}_{0,0}} \\
    b &= \sqrt{\tens{G}_{1,1}} \\
    c &= \sqrt{\tens{G}_{2,2}} \\
    \alpha &= \cos^{-1}{\frac{\tens{G}_{1,2}}{b c}}  \\
    \beta &= \cos^{-1}{\frac{\tens{G}_{0,2}}{a c}}   \\
    \gamma &= \cos^{-1}{\frac{\tens{G}_{0,1}}{a b}}  \\
    \end{aligned}
    $$
    """
    a, b, c = jnp.sqrt(jnp.diag(metric))
    al = jnp.degrees(jnp.arccos(metric[1, 2] / (b * c)))
    be = jnp.degrees(jnp.arccos(metric[0, 2] / (a * c)))
    ga = jnp.degrees(jnp.arccos(metric[0, 1] / (a * b)))
    return jnp.array([a, b, c, al, be, ga])


@jax.jit
def lpars_rlpars_to_B(lpars: jax.Array, rlpars: jax.Array) -> jax.Array:
    r"""Get the Busing-Levy B matrix from the direct and reciprocal lattice parameters.

    Parameters
    ----------
    lpars
        [6] Lattice parameters (a,b,c,alpha,beta,gamma) with angles in degrees
    rlpars
        [6] Reciprocal lattice parameters (a*,b*,c*,alpha*,beta*,gamma*) with angles in degrees

    Returns
    -------
    B: jax.Array
        [3,3] Reciprocal space orthogonalization matrix
    
    Notes
    -----
    For a direct space lattice:
    $$\left( a, b, c, \alpha, \beta, \gamma \right)$$
    and a reciprocal space lattice:
    $$\left( a^*, b^*, c^*, \alpha^*, \beta^*, \gamma^* \right)$$
    We can say (Busing & Levy 1966, Equation 3) [2]_:
    $$B = 
    \begin{bmatrix}
    a^* & b^* \cos{\gamma^*} &  c^* \cos{\beta^*} \\
    0   & b^* \sin{\gamma^*} & -c^* \sin{\beta^*}\cos{\alpha} \\
    0   &                  0 & \frac{1}{c}
    \end{bmatrix}
    $$

    References
    ----------
    .. [2] https://doi.org/10.2172/4457192
    """
    a, b, c, alpha, beta, gamma = lpars
    astar, bstar, cstar, alphastar, betastar, gammastar = rlpars
    betastar_rad = jnp.radians(betastar)
    gammastar_rad = jnp.radians(gammastar)
    ca = jnp.cos(jnp.radians(alpha))

    # fmt: off
    B = jnp.array([[astar, bstar * jnp.cos(gammastar_rad),       cstar * jnp.cos(betastar_rad)],
                   [    0, bstar * jnp.sin(gammastar_rad), -cstar * jnp.sin(betastar_rad) * ca],
                   [    0,                               0,                             1. / c]])
    # fmt: on
    return B


@jax.jit
def mt_to_rmt(mt: jax.Array) -> jax.Array:
    r"""Convert metric tensor to reciprocal metric tensor.

    Parameters
    ----------
    mt
        [3,3] Metric tensor

    Returns
    -------
    rmt: jax.Array
        [3,3] Reciprocal metric tensor

    Notes
    -----
    $\tens{G^{ij}} = \tens{G_{ij}^{-1}}$
    """
    rmt = jnp.linalg.inv(mt)
    return rmt


@jax.jit
def rmt_to_mt(rmt: jax.Array) -> jax.Array:
    r"""Convert reciprocal metric tensor to metric tensor.

    Parameters
    ----------
    rmt
        [3,3] Reciprocal metric tensor

    Returns
    -------
    mt: jax.Array
        [3,3] Metric tensor

    Notes
    -----
    $\tens{G_{ij}} = \tens{G^{ij}^{-1}}$
    """
    mt = jnp.linalg.inv(rmt)
    return mt


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
    $\tens{G^{ij}} = \matr{B^T} \cdot \matr{B}$
    """
    rmt = B.T @ B
    return rmt


@jax.jit
def metric_to_volume(metric: jax.Array) -> jax.Array:
    r"""Get the volume of the (direct or reciprocal space) unit cell from the (direct or reciprocal space) metric tensor.

    Parameters
    ----------
    mt
        [3,3] (direct or reciprocal space) metric tensor

    Returns
    -------
    volume: jax.Array
        [1] Volume of (direct or reciprocal space) unit cell

    Notes
    -----
    $$V = \sqrt{\abs{\tens{G}}}$$
    """
    volume = jnp.sqrt(jnp.linalg.det(metric))
    return volume


@jax.jit
def UBI_to_mt(UBI: jax.Array) -> jax.Array:
    r"""Convert from (U.B)^-1 matrix to metric tensor.

    Parameters
    ----------
    UBI
        [3,3] (U.B)^-1 matrix

    Returns
    -------
    jax.Array
        [3,3] Metric tensor

    Notes
    -----
    $$
    \begin{aligned}
    \tens{G_{ij}} &= \left(\matr{UB}\right)^{-1} \cdot \left(\left(\matr{UB}\right)^{-1}\right)^T           \\
    \tens{G_{ij}} &= \matr{B}^{-1}\matr{U}^{-1} \cdot \left(\matr{B}^{-1}\matr{U}^{-1}\right)^T            \\
    \tens{G_{ij}} &= \matr{B}^{-1}\matr{U}^{-1} \cdot \left(\matr{U}^{-1}\right)^T \left(\matr{B}^{-1}\right)^T   \\
    \tens{G_{ij}} &= \matr{B}^{-1}\matr{U}^{-1} \cdot \matr{U} \left(\matr{B}^{-1}\right)^T   \\
    \tens{G_{ij}} &= \matr{B}^{-1} \cdot \left(\matr{B}^{-1}\right)^T   \\
    \tens{G_{ij}} &= \matr{B}^{-1} \cdot \left(\matr{B}^T\right)^{-1}   \\
    \tens{G_{ij}} &= \left(\matr{B}^T \cdot \matr{B}\right)^{-1}   \\
    \tens{G_{ij}} &= \left(\tens{G^{ij}}\right)^{-1}   \\
    \end{aligned}
    $$
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
    rlpars = mt_to_lpars(rmt)
    B = lpars_rlpars_to_B(lpars, rlpars)
    return B
