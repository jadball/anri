"""2D phantom microstructures on a reconstruction grid, for testing indexing and refinement.

The phantom follows the microstructures Anri targets (see ``AGENTS.md``): grains, each split into cells that are
misoriented a little from the grain (dislocation cells, solidification cells), and annealing twins as lamellae with
sharp boundaries. Optionally, as in deformed metals: cell orientations that accumulate from cell to cell like a
random walk, an intrinsic orientation spread inside each cell, and grains bent about one axis, so that peaks smear
into arcs (bananas). Orientations are
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


def brownian_field(xy: ArrayLike, step: float, scale: float, rms_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a 2D Brownian random field of rotation vectors: increments grow like a random walk with distance.

    Each of the 3 components is an independent Gaussian field whose increments have variance proportional to distance
    (spectrum ~ 1 / k^3), made by FFT on a grid of spacing ``step`` (twice the points' extent, so it does not wrap) and
    scaled so that points ``scale`` apart differ by ``rms_deg`` rms per component.

    Parameters
    ----------
    xy
        [N, 2] points
    step
        Grid spacing
    scale
        Distance at which the rms difference is ``rms_deg``
    rms_deg
        Rms difference per component at ``scale``, degrees
    rng
        Random generator

    Returns
    -------
    np.ndarray
        [N, 3] rotation vectors at the points (nearest grid node), degrees
    """
    xy = np.asarray(xy, float)
    lo = xy.min(0)
    m = int(2 * np.ceil(np.ptp(xy, 0).max() / step + 2))
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(m, step), np.fft.fftfreq(m, step), indexing="ij"))
    amp = np.where(k > 0, k, np.inf) ** -1.5
    noise = rng.normal(size=(3, m, m)) + 1j * rng.normal(size=(3, m, m))
    f = np.real(np.fft.ifft2(noise * amp, axes=(1, 2)))  # [3, m, m]
    lag = max(1, round(scale / step))
    d2 = np.mean((f[:, lag:, :] - f[:, :-lag, :]) ** 2)  # mean square increment at the scale, per component
    f *= rms_deg / np.sqrt(d2)
    i, j = np.round((xy - lo) / step).astype(int).T
    return f[:, i, j].T


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
    cell_walk_deg: float = 0.0,
    cell_sig_deg: float = 0.0,
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

    Three options make orientation vary along the rays, as in deformed metals (all off by default):

    - **Accumulating cells** (``cell_walk_deg``): on top of its own random rotation, each cell is turned by a random
      field sampled at the cell's seed, whose increments grow like a random walk with distance (a 2D Brownian field,
      each rotation-vector component independent). Two cells ``cell_size`` apart differ by ``cell_walk_deg`` rms per
      component, cells ``d`` apart by ``cell_walk_deg * sqrt(d / cell_size)``. Cells stay constant inside, with sharp
      boundaries.
    - **Intrinsic spread** (``cell_sig_deg``): the standard deviation of each component of a small rotation spread
      inside every voxel (dislocations within a cell, and what the beam height averages), returned as a "sig_rot" map
      in radians, as the renderer takes it (:func:`anri.fwd.render_row`).
    - **Bent grains** (``bend_grains``): that many grains (the largest after the twinned ones) turn steadily about one
      random axis along one random in-plane direction, by ``bend_deg`` per ``radius`` of distance, about the grain's
      centre: lattice curvature with one dominant axis.

    These draw from their own random stream, so a phantom without them is the same as before they existed.

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
    cell_walk_deg
        Rms difference per rotation-vector component between cells ``cell_size`` apart, from the accumulating field,
        degrees
    cell_sig_deg
        Intrinsic spread inside each voxel, degrees (standard deviation per rotation-vector component)
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
        (-1 outside), "twin" and "inside" [n, n] bool; "pos" [n * n, 3], the voxel positions; and with
        ``cell_sig_deg``, "sig_rot" [n, n] (radians, NaN outside)
    """
    rng = np.random.default_rng(seed)
    pos = np.asarray(recon_positions(n, step), float)
    xy = pos[:, :2]
    inside = np.linalg.norm(xy, axis=1) <= radius
    grain = voronoi(xy, rng.uniform(-radius, radius, (n_grains, 2)))
    U_g = random_rotations(n_grains, rng)
    n_cells = max(1, int(np.pi * radius**2 / cell_size**2))
    cell_seeds = rng.uniform(-radius, radius, (n_cells, 2))
    cell = voronoi(xy, cell_seeds)
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
    rng2 = np.random.default_rng([seed, 1])  # their own stream, so the rest is unchanged
    if cell_walk_deg > 0:
        walk = brownian_field(cell_seeds, step, cell_size, cell_walk_deg, rng2)  # [n_cells, 3] degrees
        angle = np.linalg.norm(walk, axis=1)
        U = axis_angle(walk + (angle == 0)[:, None], angle)[cell] @ U
    if bend_grains > 0:
        order = [g for g in largest if g not in set(largest[:twin_grains].tolist())]
        for g in order[:bend_grains]:
            m = grain == g
            axis = rng2.normal(size=3)
            phi = rng2.uniform(0.0, 2 * np.pi)
            d = (xy[m] - xy[m & inside].mean(0)) @ np.array([np.cos(phi), np.sin(phi)])
            U[m] = axis_angle(np.broadcast_to(axis, (m.sum(), 3)), bend_deg * d / radius) @ U[m]
    U[~inside] = np.nan
    out_sig = {"sig_rot": np.where(inside, np.radians(cell_sig_deg), np.nan).reshape(n, n)} if cell_sig_deg > 0 else {}
    return {
        **out_sig,
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
        grain's mean orientation (taken over the grain's non-twin voxels); and "sig_rot" (radians) if the phantom has
        an intrinsic spread
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
    if "sig_rot" in ph:
        maps["sig_rot"] = ph["sig_rot"]
    return tensormap_from_recon(maps, lattice_parameters, spacegroup, phase_name, step)
