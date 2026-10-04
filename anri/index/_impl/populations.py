"""Orientation populations per voxel from fitted occupancies."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


@partial(jax.jit, static_argnames=("m", "p"))
def _populations(
    f: jax.Array, cand: jax.Array, U_list: jax.Array, ops: jax.Array, radius: float, eps: float, m: int, p: int
) -> tuple:
    def one(fv: jax.Array, cv: jax.Array) -> tuple:
        total = jnp.maximum(jnp.sum(fv), 1e-30)
        fm, i = jax.lax.top_k(fv, m)
        Um = U_list[cv[i]]  # [m, 3, 3]
        valid = fm > eps * total
        assigned = jnp.full(m, -1)
        sym = Um[:, None] @ ops[None]  # [m, S, 3, 3]: the same lattice
        out = []
        for q in range(p):
            free = valid & (assigned < 0)
            seed = jnp.argmax(free)  # the most occupied free candidate (they are sorted)
            tr = jnp.einsum("ij,msij->ms", Um[seed], sym)  # trace(U_seed^T U S)
            aligned = sym[jnp.arange(m), jnp.argmax(tr, 1)]  # each candidate's equivalent closest to the seed
            ang = jnp.arccos(jnp.clip((tr.max(1) - 1) / 2, -1.0, 1.0))
            member = free & free[seed] & (ang <= radius)
            assigned = jnp.where(member, q, assigned)
            w = jnp.where(member, fm, 0.0)
            sw = jnp.sum(w)
            u, _, vt = jnp.linalg.svd(jnp.einsum("m,mij->ij", w, aligned) + jnp.where(sw > 0, 0.0, 1.0) * jnp.eye(3))
            mean = u @ jnp.diag(jnp.array([1.0, 1.0, jnp.sign(jnp.linalg.det(u @ vt))])) @ vt
            a = jnp.arccos(jnp.clip((jnp.einsum("ij,mij->m", mean, aligned) - 1) / 2, -1.0, 1.0))
            out.append((sw / total, mean, jnp.sqrt(jnp.sum(w * a**2) / jnp.maximum(sw, 1e-30)), jnp.sum(member)))
        return tuple(jnp.stack(x) for x in zip(*out))

    return jax.vmap(one)(f, cand)


def populations(
    f: ArrayLike,
    cand: ArrayLike,
    U_list: ArrayLike,
    ops: ArrayLike,
    radius_deg: float,
    p: int = 4,
    m: int = 16,
    eps: float = 0.02,
    vb: int = 1 << 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group each voxel's occupied candidate orientations into up to p populations.

    A voxel's m most occupied candidates above ``eps`` x its total are grouped greedily: the most occupied free
    candidate seeds a population, and every free candidate within ``radius_deg`` of it (over the Laue group) joins.
    Each population gets its share of the voxel's occupancy, the occupancy-weighted mean of its members (aligned to
    the seed over the symmetry, then projected back to a rotation) and their weighted rms misorientation from that
    mean. Populations closer than the radius come out as one, with a larger spread: what the data cannot separate is
    reported as a mean and a spread. The spread includes the orientation grid's own spacing (one orientation between
    grid points shares itself among the nearby points), so it is an upper bound.

    Parameters
    ----------
    f, cand
        [Nv, K] occupancies and orientation indices, from :func:`anri.index.fit_occupancy`
    U_list
        [Nq, 3, 3] the orientations that cand indexes
    ops
        [n, 3, 3] Laue-group rotations, from :func:`anri.crystal.laue_rotations`
    radius_deg
        Largest misorientation from a population's seed, e.g. 1.8 grid steps
    p
        Populations per voxel at most
    m
        Candidates considered per voxel
    eps
        Smallest occupancy considered, as a fraction of the voxel's total
    vb
        Voxels per call

    Returns
    -------
    fraction: np.ndarray
        [Nv, p] share of each voxel's occupancy (0 where absent), largest first
    U: np.ndarray
        [Nv, p, 3, 3] mean orientations
    spread: np.ndarray
        [Nv, p] rms misorientation from the mean, degrees
    n: np.ndarray
        [Nv, p] candidates in each population
    """
    f, cand = np.asarray(f, np.float32), np.asarray(cand, np.int32)
    U_list, ops = jnp.asarray(U_list, jnp.float32), jnp.asarray(ops, jnp.float32)
    radius = float(np.radians(radius_deg))
    nv = f.shape[0]
    m = min(m, f.shape[1])
    out = []
    for s0 in range(0, nv, vb):
        fb, cb = f[s0 : s0 + vb], cand[s0 : s0 + vb]
        n = len(fb)
        pad = vb - n if nv > vb else 0
        res = _populations(
            jnp.pad(fb, ((0, pad), (0, 0))), jnp.pad(cb, ((0, pad), (0, 0))), U_list, ops, radius, eps, m, p
        )
        out.append([np.asarray(x)[:n] for x in res])
    frac, U, spread, n = (np.concatenate(x) for x in zip(*out))
    return frac, U, np.degrees(spread), n
