"""Utility functions for geometry transformations."""

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as jR


@jax.jit
def rmat_from_axis_angle(axis: jax.Array, angle: jax.Array) -> jax.Array:
    """Return rotation matrix for positive right-handed rotation about axis by angle (degrees).

    Parameters
    ----------
    axis
        [3] Rotation axis vector
    angle
        Rotation angle in degrees

    Returns
    -------
    Q: jax.Array
        [3,3] Rotation matrix
    """
    #  normalise axis
    axis = axis / jnp.linalg.norm(axis)
    Q = jR.from_rotvec(angle * axis, degrees=True).as_matrix()
    return Q


def rot_x(angle: jax.Array) -> jax.Array:
    """Return rotation matrix for positive right-handed rotation about the X axis by angle (degrees).

    Parameters
    ----------
    angle
        Rotation angle in degrees

    Returns
    -------
    Q: jax.Array
        [3,3] Rotation matrix
    """
    axis = jnp.array([1.0, 0.0, 0.0])
    return rmat_from_axis_angle(axis=axis, angle=angle)


def rot_y(angle: jax.Array) -> jax.Array:
    """Return rotation matrix for positive right-handed rotation about the Y axis by angle (degrees).

    Parameters
    ----------
    angle
        Rotation angle in degrees

    Returns
    -------
    Q: jax.Array
        [3,3] Rotation matrix
    """
    axis = jnp.array([0.0, 1.0, 0.0])
    return rmat_from_axis_angle(axis=axis, angle=angle)


def rot_z(angle: jax.Array) -> jax.Array:
    """Return rotation matrix for positive right-handed rotation about the Z axis by angle (degrees).

    Parameters
    ----------
    angle
        Rotation angle in degrees

    Returns
    -------
    Q: jax.Array
        [3,3] Rotation matrix
    """
    axis = jnp.array([0.0, 0.0, 1.0])
    return rmat_from_axis_angle(axis=axis, angle=angle)
