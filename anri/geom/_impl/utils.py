"""Utility functions for geometry transformations."""

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as jR


@jax.jit
def rmat_from_axis_angle(axis: jax.Array, angle: float) -> jax.Array:
    r"""Return rotation matrix for positive right-handed rotation about axis by angle (degrees).

    Parameters
    ----------
    axis
        [3] Rotation axis vector
    angle
        Rotation angle in degrees

    Returns
    -------
    R: jax.Array
        [3,3] Rotation matrix. $\matr{R} \cdot \vec{v_{\text{sample}}} = \vec{v_{\text{lab}}}$
    """
    #  normalise axis
    axis = axis / jnp.linalg.norm(axis)
    R = jR.from_rotvec(angle * axis, degrees=True).as_matrix()
    return R


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
