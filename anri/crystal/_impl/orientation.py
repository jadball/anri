"""Crystal orientations: Laue-group symmetry, disorientations, fundamental zones and orientation grids.

An orientation is a rotation matrix ``U`` taking crystal Cartesian vectors to the sample frame, so that a reflection
``h`` scatters along ``g = U B h`` (``B`` without 2 pi). Two orientations ``U`` and ``U S`` give the same reflections
when ``S`` is a proper rotation of the crystal's Laue group (:func:`laue_rotations`), so orientations are compared
over those (:func:`disorientation`) and grids cover one fundamental zone (:func:`orientation_grid`).

Built on the rotation algebra of :mod:`anri.maths.rotation`. Grid generation runs once per indexing in NumPy;
:func:`anri.maths.rotation.rod_to_mat` is in JAX for use inside jitted code.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from anri.maths.rotation import mat_to_quat, quat_mul, quat_to_mat, rod_to_mat, rotation_angle
from anri.maths.utils import float64


def laue_rotations(sym_ops: ArrayLike, B: ArrayLike) -> np.ndarray:
    """Proper rotations of a crystal's Laue group, as Cartesian matrices ``S`` in the crystal frame.

    ``U`` and ``U S`` give the same reflections. Inversion is added (Friedel's law), so the group is that of the
    point group's proper rotations together with those of its improper operations times -1: 24 for m-3m, 12 for m-3
    and 6/mmm, 8 for 4/mmm, 6 for 6/m and -3m, 4 for 4/m and mmm, 3 for -3, 2 for 2/m and 1 for -1.

    Parameters
    ----------
    sym_ops
        [M, 4, 4] (or [M, 3, 3]) space-group operations on fractional coordinates, e.g. from
        :func:`anri.crystal.symmetry_matrices`
    B
        [3, 3] reciprocal-space B matrix

    Returns
    -------
    np.ndarray
        [n, 3, 3] rotations, the identity first
    """
    R = np.asarray(sym_ops, float)[:, :3, :3]
    R = R * np.sign(np.linalg.det(R))[:, None, None]  # improper operations times inversion
    B = np.asarray(B, float)
    S = B @ np.swapaxes(R, 1, 2) @ np.linalg.inv(B)  # reflections transform as h -> R^T h
    S = np.unique(np.round(S, 8), axis=0)
    is_identity = np.all(np.isclose(S, np.eye(3), atol=1e-6), axis=(1, 2))
    return np.concatenate([S[is_identity], S[~is_identity]])


def closest_equivalent(U_ref: ArrayLike, U: ArrayLike, ops: ArrayLike) -> tuple[jax.Array, jax.Array]:
    """Return each orientation's symmetric equivalent ``U S`` closest to a reference, and the angle between them.

    Parameters
    ----------
    U_ref
        [..., 3, 3] reference orientations (broadcast against U)
    U
        [..., 3, 3] orientations
    ops
        [n, 3, 3] Laue-group rotations, from :func:`laue_rotations`

    Returns
    -------
    U_S: jax.Array
        [..., 3, 3] the equivalents
    angle: jax.Array
        [...] their misorientation from U_ref (radians), from :func:`rotation_angle`
    """
    U_ref, U, ops = jnp.asarray(U_ref), jnp.asarray(U), jnp.asarray(ops)
    sym = U[..., None, :, :] @ ops  # [..., n, 3, 3]
    tr = jnp.einsum("...ij,...sij->...s", U_ref, sym)  # trace(U_ref^T U S): largest for the smallest angle
    best = jnp.take_along_axis(sym, jnp.argmax(tr, -1)[..., None, None, None], -3)[..., 0, :, :]
    return best, rotation_angle(jnp.swapaxes(U_ref, -1, -2) @ best)


def to_fundamental_zone(U: ArrayLike, ops: ArrayLike) -> np.ndarray:
    """Replace each orientation by its equivalent ``U S`` with the smallest rotation angle.

    Parameters
    ----------
    U
        [N, 3, 3] orientations
    ops
        [n, 3, 3] Laue-group rotations, from :func:`laue_rotations`

    Returns
    -------
    np.ndarray
        [N, 3, 3] orientations in the fundamental zone around the identity
    """
    with float64():
        return np.asarray(closest_equivalent(np.eye(3), U, ops)[0])


def disorientation(Ua: ArrayLike, Ub: ArrayLike, ops: ArrayLike) -> np.ndarray:
    """Smallest misorientation angle between orientations over a Laue group.

    Parameters
    ----------
    Ua, Ub
        [N, 3, 3] orientations
    ops
        [n, 3, 3] Laue-group rotations, from :func:`laue_rotations`

    Returns
    -------
    np.ndarray
        [N] angles in degrees
    """
    with float64():
        return np.degrees(np.asarray(closest_equivalent(Ua, Ub, ops)[1]))


# ----------------------------------------------------------------------------------------------- grids
def cubic_grid_misorientation(step_deg: float) -> float:
    """Largest misorientation (degrees) from any orientation to its nearest point of the cubic Rodrigues grid.

    Half the diagonal of a grid cell, at the origin where the grid is coarsest.

    Parameters
    ----------
    step_deg
        Grid step of :func:`orientation_grid` for a cubic crystal

    Returns
    -------
    float
        Worst-case misorientation in degrees
    """
    return float(np.degrees(2 * np.arctan(np.sqrt(3) / 2 * np.tan(np.radians(step_deg) / 2))))


def _cubic_rodrigues_grid(step_deg: float) -> np.ndarray:
    """Rodrigues vectors on a regular grid inside the cubic fundamental zone (spacing ~step_deg near the origin)."""
    dr = np.tan(np.radians(step_deg) / 2)
    lim = np.tan(np.pi / 8)
    g = np.arange(-lim, lim + 1e-12, dr)
    r = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return r[np.abs(r).sum(1) <= 1.0]


def _cube_to_ball(xyz: np.ndarray) -> np.ndarray:
    """Map points of the cube of edge pi^(2/3) onto the homochoric ball, preserving volume (Rosca et al., 2014)."""
    sc = (np.pi / 6) ** (1 / 6)
    beta = np.pi ** (5 / 6) / (2 * 6 ** (1 / 6))
    prek = (3 * np.pi / 4) ** (1 / 3) * 2**0.25 / beta
    r2 = np.sqrt(2.0)
    # which pyramid: the largest coordinate becomes z
    ax = np.argmax(np.abs(xyz), 1)
    perm = np.array([[1, 2, 0], [2, 0, 1], [0, 1, 2]])[ax]  # x -> (y, z, x), y -> (z, x, y), z -> (x, y, z)
    p = sc * np.take_along_axis(xyz, perm, 1)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = np.abs(y) <= np.abs(x)
        u = np.where(xy, x, y)  # the larger of x, y
        v = np.where(xy, y, x)
        t = np.where(u != 0, np.pi / 12 * v / u, 0.0)
        c, s = np.cos(t), np.sin(t)
        q = prek * u / np.sqrt(r2 - c)
        a1, a2 = (r2 * c - 1) * q, r2 * s * q
        T1, T2 = np.where(xy, a1, a2), np.where(xy, a2, a1)
        cc = T1 * T1 + T2 * T2
        s2 = np.pi * cc / (24 * z * z)
        c2 = np.sqrt(np.pi) * cc / np.sqrt(24) / z
        qq = np.sqrt(1 - s2)
        out = np.stack([T1 * qq, T2 * qq, np.sqrt(6 / np.pi) * z - c2], 1)
    on_axis = (x == 0) & (y == 0)
    out[on_axis] = np.stack([np.zeros(on_axis.sum()), np.zeros(on_axis.sum()), np.sqrt(6 / np.pi) * z[on_axis]], 1)
    out[np.all(p == 0, 1)] = 0.0
    inv = np.argsort(perm, 1)
    return np.take_along_axis(out, inv, 1)


def _ball_to_quat(h: np.ndarray) -> np.ndarray:
    """Map homochoric vectors to quaternions: |h| = (3/4 (w - sin w))^(1/3) for rotation angle w along h."""
    n = np.linalg.norm(h, axis=1)
    target = n**3
    lo, hi = np.zeros_like(n), np.full_like(n, np.pi)
    for _ in range(60):  # bisection: w - sin w increases on [0, pi]
        mid = 0.5 * (lo + hi)
        low = 0.75 * (mid - np.sin(mid)) < target
        lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
    w = 0.5 * (lo + hi)
    axis = np.where(n[:, None] > 0, h / np.where(n > 0, n, 1)[:, None], 0.0)
    return np.concatenate([np.cos(w / 2)[:, None], np.sin(w / 2)[:, None] * axis], 1)


def cubochoric_quaternions(semi_edge_steps: int) -> np.ndarray:
    """Quaternions of the cubochoric grid of rotations, uniform in volume (Rosca et al., 2014; Singh & De Graef, 2016).

    The grid has ``(2 N)^3`` points on the cube of edge pi^(2/3), the identity among them: indices -N + 1 ... N
    along each edge, since opposite faces of the cube are the same rotations (as orix's ``cubochoric_sampling``).

    Parameters
    ----------
    semi_edge_steps
        N, the grid points along the cube's semi-edge

    Returns
    -------
    np.ndarray
        [(2 N)^3, 4] quaternions ``(w, x, y, z)``
    """
    a = np.pi ** (2 / 3) / 2
    g = np.arange(-semi_edge_steps + 1, semi_edge_steps + 1) * a / semi_edge_steps
    xyz = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return _ball_to_quat(_cube_to_ball(xyz))


_SHELL = 0.5  # grid steps kept outside the zone


def orientation_grid(step_deg: float, ops: ArrayLike) -> tuple[np.ndarray, float]:
    """Orientations covering one fundamental zone of a Laue group, and their worst-case spacing.

    Cubic groups (24 rotations) use a regular Rodrigues grid of the cubic fundamental zone: for cubic crystals it
    needs about half the orientations of a cubochoric grid for the same worst case. Other groups use the cubochoric
    grid of all rotations (:func:`cubochoric_quaternions`, with ``N = 131.97 / (step - 0.037)`` as in Singh & De
    Graef, so ``step_deg`` is about the mean spacing) reduced to the fundamental zone around the identity, plus a shell
    half a step wide outside it: without the shell, points just outside the zone leave gaps along its boundary (their
    symmetric images are not grid points).

    Parameters
    ----------
    step_deg
        Grid step in degrees
    ops
        [n, 3, 3] Laue-group rotations, from :func:`laue_rotations`

    Returns
    -------
    U: np.ndarray
        [N, 3, 3] orientations (float32)
    delta: float
        Largest misorientation (degrees) from any orientation to its nearest grid point: exact for the cubic grid,
        measured for the cubochoric one (at most 1.15 x step for every Laue group, 1.5 to 12 degrees; 1.2 x step)
    """
    ops = np.asarray(ops, float)
    if len(ops) == 24:
        return np.asarray(rod_to_mat(_cubic_rodrigues_grid(step_deg)), np.float32), cubic_grid_misorientation(step_deg)
    n = max(1, int(np.round(131.97049 / (step_deg - 0.03732))))
    qops = mat_to_quat(ops)
    margin = np.radians(_SHELL * step_deg)  # a thin shell outside the zone, so its boundary has no gaps
    keep = []
    a = np.pi ** (2 / 3) / 2
    g = np.arange(-n + 1, n + 1) * a / n
    for x in g:  # one slab of the cube at a time, to bound memory
        yz = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        q = _ball_to_quat(_cube_to_ball(np.column_stack([np.full(len(yz), x), yz])))
        w_eq = np.abs(quat_mul(q[:, None], qops[None])[..., 0])  # |w| of each equivalent
        angle = 2 * np.arccos(np.clip(w_eq, 0.0, 1.0))
        keep.append(q[angle[:, 0] <= angle.min(1) + margin])
    return quat_to_mat(np.concatenate(keep)).astype(np.float32), 1.2 * step_deg
