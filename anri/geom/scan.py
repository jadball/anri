"""Scanning 3DXRD specific geometry functions.

These functions conform mostly to the ImageD11 sinogram geometry found in :func:`ImageD11.sinograms.geometry`

They are partially re-defined below:

The lab frame and the sample frame have the same definitions as in :func:`anri.geom.gonio.sample_to_lab` for example.

.. rubric:: Step space

This is simply an integer discretisation of the sample frame.
The i axis is aligned to sample X.
The j axis is aligned to sample -Y.

.. rubric:: Recon space

Needed for tomographic reconstructions.
This is aligned such that the origin is in the bottom left corner of the image.
Therefore the size of the reconstruction

.. rubric:: Diagrams

Sample frame::

             ^
             |
             | sx
             |
    <------- S (0, 0) rotation axis
        sy

Step space::

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


Reconstruction space (iradon output when plotted with origin="lower")::

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
import numpy as np


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


@jax.jit
def recon_to_sample(
    ri: float | jax.Array, rj: float | jax.Array, recon_shape: tuple[int, int], ystep: float
) -> tuple[float | jax.Array, float | jax.Array]:
    """Convert reconstruction space (origin in corner) to sample space: :func:`recon_to_step`, then :func:`step_to_sample`.

    Parameters
    ----------
    ri, rj
        Recon space - i and j directions
    recon_shape
        [2] Shape of the reconstruction array
    ystep
        Y translation step size between bin centres

    Returns
    -------
    sx, sy: jax.Array
        Sample space - x and y directions
    """
    si, sj = recon_to_step(ri, rj, recon_shape)
    return step_to_sample(si, sj, ystep)


def recon_positions(n: int, step: float) -> jax.Array:
    """Sample-frame positions of every voxel of an n x n reconstruction grid, in reconstruction order.

    The grid is centred on the rotation axis, as ImageD11 reconstructs (see :func:`recon_to_sample`).

    Parameters
    ----------
    n
        Voxels along each side
    step
        Voxel size

    Returns
    -------
    jax.Array
        [n * n, 3] positions (x, y, 0), voxel (ri, rj) at row ri * n + rj
    """
    ri, rj = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
    sx, sy = recon_to_sample(ri, rj, (n, n), step)
    return jnp.stack([sx.ravel(), sy.ravel(), jnp.zeros(n * n)], 1)


def regrid(a: np.ndarray, n: int, fill: float = 0) -> np.ndarray:
    """Move a [m, m, ...] reconstruction-order array to a grid of n x n voxels of the same size, both centred on the axis.

    Grids centred on the rotation axis (:func:`recon_positions`) that differ only in size share their middle voxels
    exactly (an odd size puts the axis on a voxel; the sizes then differ by an even number), so the array is cropped
    or padded by the same number of voxels on every side.

    Parameters
    ----------
    a
        [m, m, ...] array in reconstruction order
    n
        Voxels per side of the target grid
    fill
        Value of the voxels padded on

    Returns
    -------
    np.ndarray
        [n, n, ...]
    """
    a = np.asarray(a)
    m = a.shape[0]
    if a.shape[1] != m or (m - n) % 2:
        msg = f"cannot centre a {a.shape[0]} x {a.shape[1]} grid on a {n} x {n} one: both must be square, odd or even alike"
        raise ValueError(msg)
    k = (m - n) // 2
    if k >= 0:
        return a[k : m - k, k : m - k]
    return np.pad(a, [(-k, -k), (-k, -k)] + [(0, 0)] * (a.ndim - 2), constant_values=fill)


def sino_shift_and_pad(y0: float, ny: int, ymin: float, ystep: float) -> tuple[float, int]:
    """Shift and padding that centre a sinogram of ny dty rows on the rotation axis, as ImageD11 does.

    Follows ``ImageD11.sinograms.geometry.sino_shift_and_pad``: the reconstruction is ``ny + pad`` voxels wide, odd.

    Parameters
    ----------
    y0
        dty at which the rotation axis is in the beam
    ny
        Number of dty rows
    ymin
        dty of the first row
    ystep
        dty step

    Returns
    -------
    shift: float
        Shift in rows
    pad: int
        Padding in rows
    """
    shift = ny // 2 - (y0 - ymin) / ystep
    pad = int(np.ceil(abs(shift) * 2)) + 1
    if (ny + pad) % 2 == 0:  # keep the reconstruction odd-sized
        pad += 1
    return shift, pad


def dty_to_row(dty: np.ndarray, ybinedges: np.ndarray, gridstep: int = 1) -> np.ndarray:
    """Return the scan row each dty falls in: its bin of ``ybinedges``, with rows summed in groups of ``gridstep``.

    Parameters
    ----------
    dty
        dty values (any shape), e.g. each frame's
    ybinedges
        [n_rows + 1] dty bin edges of the rows (the DataSet's ``ybinedges``)
    gridstep
        Rows summed in groups of this: row k of the result is bins k gridstep ... (k + 1) gridstep - 1

    Returns
    -------
    row: np.ndarray
        Row index per dty (shape of ``dty``); -1 outside the edges or where dty is not finite. Bins are
        (lo, hi]: a dty exactly on an edge is in the row below it.
    """
    dty = np.asarray(dty, float)
    edges = np.asarray(ybinedges, float)
    k = np.searchsorted(edges, np.where(np.isfinite(dty), dty, np.inf)) - 1
    return np.where((k >= 0) & (k < len(edges) - 1), k // gridstep, -1)
