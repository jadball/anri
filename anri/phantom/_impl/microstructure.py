"""2D phantom microstructures on a reconstruction grid, for testing indexing and refinement.

The phantom follows the microstructures Anri targets (see ``AGENTS.md``): grains, each split into cells that are
misoriented a little from the grain (dislocation cells, solidification cells), and annealing twins as lamellae with
sharp boundaries. Optionally, dislocation walls whose small rotations accumulate like a random walk, and grains
bent about one axis, so peaks smear into single-maximum arcs (bananas) as in deformed metals. Orientations are
piecewise constant on a fine grid, so a phantom voxel smaller than the scan step puts several orientations in one scan
voxel.

All maps are in reconstruction order (n x n, see :func:`anri.geom.recon_positions`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from ImageD11.sinograms.tensor_map import TensorMap

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
    wall_spacing: float | None = None,
    wall_spread_deg: float = 0.1,
    bend_grains: int = 0,
    bend_deg: float = 1.0,
    seed: int = 0,
) -> dict:
    """Make a disk-shaped polycrystal of grains with misoriented cells and twin lamellae.

    The disk is split into Voronoi grains with random orientations, and into Voronoi cells of about ``cell_size``
    (across grain boundaries alike), each turned from its grain by a small random rotation. The ``twin_grains`` largest
    grains carry twin lamellae: the twin is the grain turned by ``twin_angle_deg`` about the crystal direction
    ``twin_axis`` (60 degrees about <111> is the Sigma3 twin of FCC metals), in lamellae ``twin_thickness`` thick
    every ``twin_period`` with the twin plane normal to that axis.

    Two options make orientation vary along the rays, as in deformed metals (both off by default):

    - **Dislocation walls** (``wall_spacing``): straight walls in random directions across the disk, on average
      ``wall_spacing`` apart along any line. Each turns everything on one side by a small random rotation (sample frame,
      each rotation-vector component normal with standard deviation ``wall_spread_deg``). The misorientation between
      two points then accumulates like a random walk: its variance grows with the number of walls between them, i.e.
      with their distance, and orientation is constant between walls. They apply inside grains and twins alike.
    - **Bent grains** (``bend_grains``): that many grains (the largest after the twinned ones) turn steadily about one
      random axis along one random in-plane direction, by ``bend_deg`` per ``radius`` of distance, about the grain's
      centre: lattice curvature with one dominant axis.

    The walls and bends draw from their own random stream, so a phantom without them is the same as before they existed.

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
    wall_spacing
        Mean distance between dislocation walls along a line (same units as step); None (default) for no walls
    wall_spread_deg
        Standard deviation of each component of a wall's rotation, degrees
    bend_grains
        How many grains are bent
    bend_deg
        Bend of those grains: degrees of rotation per ``radius`` of distance along the bend direction
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
    rng2 = np.random.default_rng([seed, 1])  # walls and bends: their own stream, so the rest is unchanged
    if wall_spacing is not None:
        # Lines p = x cos(t) + y sin(t), offsets uniform over [-r, r]: a segment of length d crosses on average
        # d * n / (pi * r) of them, so n = pi * r / wall_spacing puts walls wall_spacing apart along any line.
        r = np.sqrt(2.0) * max(radius, np.abs(xy).max())  # cover the whole grid, not only the disk
        n_walls = max(1, round(float(np.pi * r / wall_spacing)))
        t = rng2.uniform(0.0, np.pi, n_walls)
        p = rng2.uniform(-r, r, n_walls)
        v = np.radians(rng2.normal(scale=wall_spread_deg, size=(n_walls, 3)))  # rotation vectors, radians
        rv = np.zeros((len(xy), 3))
        for w0 in range(0, n_walls, 256):  # each voxel sums the rotations of the walls it is beyond
            w = slice(w0, w0 + 256)
            side = (xy[:, :1] * np.cos(t[w]) + xy[:, 1:] * np.sin(t[w])) > p[w]
            rv += side.astype(float) @ v[w]
        angle = np.linalg.norm(rv, axis=1)
        U = axis_angle(rv + (angle == 0)[:, None], np.degrees(angle)) @ U
    if bend_grains > 0:
        order = [g for g in largest if g not in set(largest[:twin_grains].tolist())]
        for g in order[:bend_grains]:
            m = grain == g
            axis = rng2.normal(size=3)
            phi = rng2.uniform(0.0, 2 * np.pi)
            d = (xy[m] - xy[m & inside].mean(0)) @ np.array([np.cos(phi), np.sin(phi)])
            U[m] = axis_angle(np.broadcast_to(axis, (m.sum(), 3)), bend_deg * d / radius) @ U[m]
    U[~inside] = np.nan
    return {
        "U": U.reshape(n, n, 3, 3),
        "grain": np.where(inside, grain, -1).reshape(n, n),
        "cell": np.where(inside, cell, -1).reshape(n, n),
        "twin": (twin & inside).reshape(n, n),
        "inside": inside.reshape(n, n),
        "pos": pos,
    }


def tensormap(ph: dict, lattice_parameters: ArrayLike, spacegroup: int, phase_name: str, step: float) -> TensorMap:
    """Turn a phantom from :func:`polycrystal` into a single-phase ImageD11 TensorMap, with its truth maps.

    Parameters
    ----------
    ph
        Output of :func:`polycrystal`
    lattice_parameters
        a, b, c, alpha, beta, gamma
    spacegroup
        Space group number
    phase_name
        Name of the phase
    step
        Voxel size, as passed to :func:`polycrystal`

    Returns
    -------
    TensorMap
        ``ImageD11.sinograms.tensor_map.TensorMap`` of shape (1, n, n) with "UBI", "phase_ids" (0 inside, -1 outside),
        "labels" (grain), "cell", "twin" (int8) and "misorientation": each voxel's disorientation in degrees from its
        grain's mean orientation (taken over the grain's non-twin voxels)
    """
    from anri.crystal import B_matrix, disorientation, laue_rotations, symmetry_matrices
    from anri.io import tensormap_from_recon

    B = np.asarray(B_matrix(np.asarray(lattice_parameters, float)))
    ops = laue_rotations(symmetry_matrices(int(spacegroup)), B)
    inside, grain, twin = ph["inside"].ravel(), ph["grain"].ravel(), ph["twin"].ravel()
    U = ph["U"].reshape(-1, 3, 3)
    mean_U = np.full_like(U, np.nan)
    for g in np.unique(grain[inside]):
        m = (grain == g) & ~twin
        u, _, vt = np.linalg.svd(U[m if m.any() else grain == g].mean(0))  # grain spreads are small: a plain mean
        mean_U[grain == g] = u @ vt
    mis = np.full(len(U), np.nan)
    mis[inside] = np.asarray(disorientation(U[inside], mean_U[inside], ops))
    shape = ph["inside"].shape
    maps = {
        "UBI": np.where(ph["inside"][..., None, None], np.linalg.inv(ph["U"] @ B), np.nan),
        "phase_ids": np.where(ph["inside"], 0, -1),
        "labels": ph["grain"],
        "cell": ph["cell"],
        "twin": ph["twin"].astype(np.int8),
        "misorientation": mis.reshape(shape),
    }
    return tensormap_from_recon(maps, lattice_parameters, spacegroup, phase_name, step)
