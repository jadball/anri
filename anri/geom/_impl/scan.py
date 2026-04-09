"""Scanning 3DXRD specific geometry functions.

These functions conform mostly to the ImageD11 sinogram geometry found in :func:`ImageD11.sinograms.geometry`

They are partially re-defined below:

The lab frame and the sample frame have the same definitions as in :func:`anri.geom.sample_to_lab` for example.

Step space
----------
This is simply an integer discretisation of the sample frame.
The i axis is aligned to sample X.
The j axis is aligned to sample -Y.

Recon space
-----------
Needed for tomographic reconstructions.
This is aligned such that the origin is in the bottom left corner of the image.
Therefore the size of the reconstruction

Diagrams
--------

Sample frame:

         ^
         |
         | sx
         |
<------- S (0, 0) rotation axis
    sy

Step space:

                +ve
                ^
                |
             si |
                |
-ve <--------   S (0, 0)--------> +ve
                |           sj
                |
                |
                v
              -ve


Reconstruction space (iradon output when plotted with origin="lower"):

   ^
   |
 i |      S
   |
(0, 0) ------->
         j

"""

# allow union operator in type annotations |
from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def step_grid_from_ybincens(
    ybincens: jax.Array, step_size: float, gridstep: float, y0: float
) -> tuple[jax.Array, jax.Array]:
    r"""Create reconstruction grid in "step" space (integer units of the step size).

    Parameters
    ----------
    ybincens
        [N] Bin centres for dty translations - from the scan command
    step_size
        Y translation step size between bin centres
    gridstep
        Downsampling step size for grid
    y0
        The true value of dty when the rotation axis (untilted by wedge, chi) intersects the beam

    Returns
    -------
    si: jax.Array
        Step space - array of integer steps in the i direction
    sj: jax.Array
        Step space - array of integer steps in the j direction
    """
    # Center y range on y0
    y_relative = ybincens - y0
    y_largest = jnp.max(jnp.abs(y_relative))

    # Find the maximum integer step needed
    max_int = jnp.ceil(y_largest / step_size).astype(jnp.int32)

    # Create a symmetric range from -max_int to +max_int
    steps = jnp.arange(-max_int, max_int + 1, gridstep, dtype=jnp.int32)

    # Create 2D grid
    si, sj = jnp.meshgrid(steps, steps, indexing="ij")

    return si, sj


@jax.jit
def step_to_recon(
    si: float | jax.Array, sj: float | jax.Array, recon_shape: tuple[int, int]
) -> tuple[float | jax.Array, float | jax.Array]:
    r"""Convert step space (integer steps leading from rotation axis) to rotation space (origin in corner).

    Parameters
    ----------
    si
        Step space - i direction
    sj
        Step space - j direction
    recon_shape
        [2] Shape of the reconstruction array

    Returns
    -------
    ri: jax.Array
        Recon space - i direction
    rj: jax.Array
        Recon space - j direction
    """
    ri = si + (recon_shape[0] // 2)
    rj = sj + (recon_shape[1] // 2)
    return ri, rj


@jax.jit
def recon_to_step(
    ri: float | jax.Array, rj: float | jax.Array, recon_shape: tuple[int, int]
) -> tuple[float | jax.Array, float | jax.Array]:
    r"""Convert rotation space (origin in corner) to step space (integer steps leading from rotation axis).

    Parameters
    ----------
    ri
        Recon space - i direction
    rj
        Recon space - j direction
    recon_shape
        [2] Shape of the reconstruction array

    Returns
    -------
    si: jax.Array
        Step space - i direction
    sj: jax.Array
        Step space - j direction
    """
    si = ri - (recon_shape[0] // 2)
    sj = rj - (recon_shape[1] // 2)
    return si, sj


@jax.jit
def step_to_sample(
    si: float | jax.Array, sj: float | jax.Array, ystep: float
) -> tuple[float | jax.Array, float | jax.Array]:
    r"""Convert step space (integer steps leading from rotation axis) to sample space.

    Parameters
    ----------
    si
        Step space - i direction
    sj
        Step space - j direction
    ystep
        Y translation step size between bin centres

    Returns
    -------
    sx: jax.Array
        Sample space - x direction
    sy: jax.Array
        Sample space - y direction
    """
    sx = si * ystep
    sy = -sj * ystep
    return sx, sy
