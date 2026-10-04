"""2D phantom microstructures on a reconstruction grid, for testing indexing and refinement.

The phantom follows the microstructures Anri targets (see ``AGENTS.md``): grains, each split into cells that are
misoriented a little from the grain (dislocation cells, solidification cells), and annealing twins as lamellae with
sharp boundaries. Orientations are piecewise constant on a fine grid, so a phantom voxel smaller than the scan step
puts several orientations in one scan voxel.

All maps are in reconstruction order (n x n, see :func:`anri.geom.recon_positions`).
"""

from __future__ import annotations

import numpy as np
from jax.typing import ArrayLike

from anri.geom import recon_positions


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
    from anri.crystal import quat_to_mat

    q = rng.normal(size=(n, 4))
    return quat_to_mat(q / np.linalg.norm(q, axis=1, keepdims=True))


def axis_angle(axis: ArrayLike, angle_deg: ArrayLike) -> np.ndarray:
    """Rotation matrices about axes by angles (right-handed).

    Parameters
    ----------
    axis
        [..., 3] axes (need not be unit)
    angle_deg
        [...] angles in degrees

    Returns
    -------
    np.ndarray
        [..., 3, 3] rotation matrices
    """
    n = np.asarray(axis, float)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    t = np.radians(np.asarray(angle_deg, float))[..., None, None]
    K = np.zeros(n.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -n[..., 2], n[..., 1], -n[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


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
    return axis_angle(v + (np.linalg.norm(v, axis=1, keepdims=True) == 0), np.linalg.norm(v, axis=1))


def voronoi(xy: ArrayLike, seeds: ArrayLike) -> np.ndarray:
    """Label points by their nearest seed.

    Parameters
    ----------
    xy
        [N, 2] points
    seeds
        [M, 2] seed positions

    Returns
    -------
    np.ndarray
        [N] index of the nearest seed
    """
    xy, seeds = np.asarray(xy), np.asarray(seeds)
    out = np.empty(len(xy), int)
    for s0 in range(0, len(xy), 4096):
        out[s0 : s0 + 4096] = np.argmin(np.linalg.norm(xy[s0 : s0 + 4096, None] - seeds[None], axis=2), 1)
    return out


def polycrystal(
    n: int,
    step: float,
    radius: float,
    n_grains: int,
    cell_size: float = 1.5,
    cell_spread_deg: float = 0.3,
    twin_grains: int = 1,
    twin_axis: tuple[float, float, float] = (1.0, 1.0, 1.0),
    twin_angle_deg: float = 60.0,
    twin_period: float = 5.0,
    twin_thickness: float = 2.0,
    seed: int = 0,
) -> dict:
    """Make a disk-shaped polycrystal of grains with misoriented cells and twin lamellae.

    The disk is split into Voronoi grains with random orientations, and into Voronoi cells of about ``cell_size``
    (across grain boundaries alike), each turned from its grain by a small random rotation. The ``twin_grains`` largest
    grains carry twin lamellae: the twin is the grain turned by ``twin_angle_deg`` about the crystal direction
    ``twin_axis`` (60 degrees about <111> is the Sigma3 twin of FCC metals), in lamellae ``twin_thickness`` thick
    every ``twin_period`` with the twin plane normal to that axis.

    Parameters
    ----------
    n
        Grid of n x n voxels (reconstruction order)
    step
        Voxel size
    radius
        Disk radius, same units as step
    n_grains
        Number of grains
    cell_size
        Typical cell size
    cell_spread_deg
        Standard deviation of each component of a cell's rotation from its grain, degrees
    twin_grains
        How many grains are twinned
    twin_axis, twin_angle_deg
        The twin rotation, about a crystal direction
    twin_period, twin_thickness
        Lamella spacing and thickness
    seed
        Random seed

    Returns
    -------
    dict
        Maps in reconstruction order: "U" [n, n, 3, 3] (crystal to sample; NaN outside), "grain" and "cell" [n, n]
        (-1 outside), "twin" and "inside" [n, n] bool; and "pos" [n * n, 3], the voxel positions
    """
    rng = np.random.default_rng(seed)
    pos = np.asarray(recon_positions(n, step), float)
    xy = pos[:, :2]
    inside = np.linalg.norm(xy, axis=1) <= radius
    grain = voronoi(xy, rng.uniform(-radius, radius, (n_grains, 2)))
    U_g = random_rotations(n_grains, rng)
    n_cells = max(1, int(np.pi * radius**2 / cell_size**2))
    cell = voronoi(xy, rng.uniform(-radius, radius, (n_cells, 2)))
    U = small_rotations(n_cells, cell_spread_deg, rng)[cell] @ U_g[grain]
    twin = np.zeros(len(xy), bool)
    axis_c = np.asarray(twin_axis, float) / np.linalg.norm(twin_axis)
    largest = np.argsort(np.bincount(grain[inside], minlength=n_grains))[::-1]
    for g in largest[:twin_grains]:
        normal = U_g[g] @ axis_c  # the twin plane's normal in the sample frame
        d = xy @ normal[:2] / max(np.linalg.norm(normal[:2]), 1e-6)
        t = (grain == g) & (np.mod(d, twin_period) < twin_thickness)
        U[t] = axis_angle(normal, twin_angle_deg) @ U[t]
        twin |= t
    U[~inside] = np.nan
    return {
        "U": U.reshape(n, n, 3, 3),
        "grain": np.where(inside, grain, -1).reshape(n, n),
        "cell": np.where(inside, cell, -1).reshape(n, n),
        "twin": (twin & inside).reshape(n, n),
        "inside": inside.reshape(n, n),
        "pos": pos,
    }
