"""Occupancy of orientations per voxel, fitted to the histogrammed data by MLEM.

The data are linear in the sample's density over position and orientation: ``d = A f``, with ``f[v, q]`` the
occupancy of orientation q in voxel v. ``A`` puts each predicted reflection of q at its (eta, omega) and in the dty
row where voxel v sits at that omega. Fitting every voxel jointly handles overlap between voxels on the same rays.

Occupancies are sparse: each voxel keeps K candidate orientations (``f [Nv, K]`` and ``cand [Nv, K]``, indices into
the orientation list), and the work runs over blocks of voxels, so memory is set by the block size, not by the
number of voxels times orientations.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .data import coarsen_rows


def system(
    eta: jax.Array,
    om: jax.Array,
    use: jax.Array,
    w: jax.Array,
    ring_j: jax.Array,
    pos: jax.Array,
    scan: dict,
    b_e: float,
    b_o: float,
    n_e: int,
    n_o: int,
) -> tuple[jax.Array, jax.Array]:
    """Entries of the system matrix: (cell index, weight) for voxels and predictions.

    A prediction (eta, omega) is spread bilinearly over the 2 x 2 nearest (eta, omega) bins and linearly over the 2
    rows nearest to where the voxel sits at that omega (lab y = x sin(omega) + y cos(omega)). Parallax is ignored.

    Parameters
    ----------
    eta, om, use, w
        [1 or Nv, Q, Nj] predictions (degrees), whether to use them, and their weights. A leading 1 means the same
        orientations for every voxel; Nv, each voxel's own.
    ring_j
        [Nj] ring of each prediction
    pos
        [Nv, 3] voxel positions in the sample frame
    scan
        "y0", "dty0", "ystep", "n_rows" and "om0": the rows are at dty = dty0 + k ystep
    b_e, b_o, n_e, n_o
        Bin widths and counts of the histogram in eta and omega

    Returns
    -------
    index, weight: jax.Array
        [Nv, Q, Nj, 8] each; unused corners have index -1
    """
    n_k = scan["n_rows"]
    om_w = jnp.mod(om - scan["om0"], 360.0) + scan["om0"]
    fe = (eta + 180.0) / b_e - 0.5
    fo = (om_w - scan["om0"]) / b_o - 0.5
    e0, o0 = jnp.floor(fe).astype(jnp.int32), jnp.floor(fo).astype(jnp.int32)
    te, to = fe - e0, fo - o0
    c, s = jnp.cos(jnp.radians(om)), jnp.sin(jnp.radians(om))
    ylab = pos[:, 0, None, None] * s + pos[:, 1, None, None] * c
    fk = (scan["y0"] - ylab - scan["dty0"]) / scan["ystep"]
    k0 = jnp.floor(fk).astype(jnp.int32)
    tk = fk - k0
    idx, wt = [], []
    for de, we in ((0, 1.0 - te), (1, te)):
        for do, wo in ((0, 1.0 - to), (1, to)):
            for dk, wk in ((0, 1.0 - tk), (1, tk)):
                ie, io, kk = (e0 + de) % n_e, o0 + do, k0 + dk
                good = use & (io >= 0) & (io < n_o) & (kk >= 0) & (kk < n_k)
                cell = ((ring_j[None, None, :] * n_e + ie) * n_o + io) * n_k + kk
                idx.append(jnp.where(good, cell, -1))
                wt.append(jnp.where(good, w * we * wo * wk, 0.0))
    return jnp.stack(idx, -1), jnp.stack(wt, -1)


def block_voxels(q: int, n_j: int, budget_bytes: float) -> int:
    """Voxels per block (a power of 2) so that one block's system entries for q orientations take ~budget_bytes.

    Counts ~256 bytes per (voxel, orientation, prediction): 8 corners of index and weight, temporaries and gathers.

    Parameters
    ----------
    q
        Orientations per voxel handled at once
    n_j
        Predictions per orientation
    budget_bytes
        Memory for one block

    Returns
    -------
    int
        Voxels per block
    """
    return int(2 ** max(0, np.floor(np.log2(budget_bytes / (256 * q * n_j)))))


def pad_voxels(pos: ArrayLike, vb: int) -> jax.Array:
    """Pad voxel positions to a multiple of vb with voxels far outside the scan (they reach no rows).

    Parameters
    ----------
    pos
        [Nv, 3] positions
    vb
        Block size

    Returns
    -------
    jax.Array
        [ceil(Nv / vb) vb, 3] positions
    """
    pos = jnp.asarray(pos, jnp.float32)
    return jnp.concatenate([pos, jnp.full((-pos.shape[0] % vb, 3), 1e6, pos.dtype)])


def _chunks(a: jax.Array, qc: int) -> jax.Array:
    return a.reshape(-1, qc, *a.shape[1:])


@partial(jax.jit, static_argnames=("n_cells", "dims", "qc"))
def _ones_block(
    pos_b: jax.Array, pred: tuple, ring_j: jax.Array, scan: dict, dims: tuple, n_cells: int, qc: int
) -> jax.Array:
    """Compute A 1 from one block of voxels, every orientation at unit occupancy."""

    def step(acc: jax.Array, ch: tuple) -> tuple:
        e, o, u, w = ch
        idx, wt = system(e[None], o[None], u[None], w[None], ring_j, pos_b, scan, dims[0], dims[1], dims[2], dims[3])
        w_flat = wt.ravel().astype(acc.dtype)  # scan's floats may be float64 (jax_enable_x64)
        return acc + jax.ops.segment_sum(w_flat, jnp.where(idx >= 0, idx, n_cells).ravel(), n_cells + 1)[:-1], None

    acc, _ = jax.lax.scan(step, jnp.zeros(n_cells, jnp.float32), tuple(_chunks(p, qc) for p in pred))
    return acc


@partial(jax.jit, static_argnames=("dims", "qc", "k"))
def _top_block(
    r: jax.Array, pos_b: jax.Array, pred: tuple, ring_j: jax.Array, scan: dict, dims: tuple, qc: int, k: int
) -> tuple:
    """Compute A^T r / A^T 1 for one block of voxels and every orientation, and keep each voxel's top k."""

    def one(ch: tuple) -> jax.Array:
        e, o, u, w = ch
        idx, wt = system(e[None], o[None], u[None], w[None], ring_j, pos_b, scan, dims[0], dims[1], dims[2], dims[3])
        num = jnp.sum(wt * jnp.where(idx >= 0, r[jnp.clip(idx, 0)], 0.0), (2, 3))
        return num / jnp.maximum(jnp.sum(wt, (2, 3)), 1e-30)

    s = jax.lax.map(one, tuple(_chunks(p, qc) for p in pred))  # [Nq / qc, vb, qc]
    val, ind = jax.lax.top_k(jnp.moveaxis(s, 0, 1).reshape(pos_b.shape[0], -1), k)
    return val, ind.astype(jnp.int32)


def candidates(
    d: jax.Array, pred: tuple, ring_j: jax.Array, pos: jax.Array, scan: dict, dims: tuple, k: int, vb: int,
    qc: int = 16, log: Callable = print,
) -> tuple[jax.Array, jax.Array]:  # fmt: skip
    """Pick each voxel's k best orientations by the first MLEM update from unit occupancy.

    ``f1 = A^T(d / A 1) / A^T 1``: two passes over every (voxel, orientation), in blocks of vb voxels. Dividing by
    ``A 1`` down-weights crowded cells, as MLEM does.

    Parameters
    ----------
    d
        [n_cells] histogram
    pred
        (eta, om, use, w), each [Nq, Nj] with Nq a multiple of qc
    ring_j
        [Nj] ring of each prediction
    pos
        [Nv, 3] positions, Nv a multiple of vb (:func:`pad_voxels`)
    scan, dims
        Rows and (b_e, b_o, n_e, n_o), see :func:`system`
    k
        Candidates per voxel
    vb, qc
        Voxels per block and orientations per chunk
    log
        Progress messages

    Returns
    -------
    f1, cand: jax.Array
        [Nv, k] first-update occupancies and orientation indices
    """
    n_cells = d.shape[0]
    blocks = pos.reshape(-1, vb, 3)
    t0 = time.perf_counter()
    a1 = jnp.zeros(n_cells, jnp.float32)
    for b in blocks:
        a1 = a1 + _ones_block(b, pred, ring_j, scan, dims, n_cells, qc)
    jax.block_until_ready(a1)
    log(f"  candidates: A 1 over {blocks.shape[0]} blocks of {vb} voxels: {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    r = jnp.where(a1 > 0, d / jnp.maximum(a1, 1e-30), 0.0)
    out = [_top_block(r, b, pred, ring_j, scan, dims, qc, k) for b in blocks]
    f1, cand = jnp.concatenate([o[0] for o in out]), jnp.concatenate([o[1] for o in out])
    jax.block_until_ready(cand)
    log(f"  candidates: top {k} per voxel: {time.perf_counter() - t0:.1f} s")
    return f1, cand


@partial(jax.jit, static_argnames=("n_cells", "dims", "vb"))
def forward(
    f: jax.Array, cand: jax.Array, pred: tuple, ring_j: jax.Array, pos: jax.Array, scan: dict, dims: tuple,
    n_cells: int, vb: int,
) -> jax.Array:  # fmt: skip
    """Forward-project occupancies: ``A f``.

    Parameters
    ----------
    f, cand
        [Nv, K] occupancies and orientation indices, Nv a multiple of vb
    pred
        (eta, om, use, w), each [Nq, Nj]
    ring_j, pos, scan, dims
        See :func:`system`
    n_cells
        Size of the histogram
    vb
        Voxels per block

    Returns
    -------
    jax.Array
        [n_cells] predicted histogram
    """
    k = f.shape[1]

    def step(acc: jax.Array, ch: tuple) -> tuple:
        fb, cb, pb = ch
        e, o, u, w = (p[cb] for p in pred)
        idx, wt = system(e, o, u, w, ring_j, pb, scan, dims[0], dims[1], dims[2], dims[3])  # [vb, K, Nj, 8]
        contrib = (fb[:, :, None, None] * wt).astype(acc.dtype)
        return acc + jax.ops.segment_sum(contrib.ravel(), jnp.where(idx >= 0, idx, n_cells).ravel(), n_cells + 1)[
            :-1
        ], None

    xs = (f.reshape(-1, vb, k), cand.reshape(-1, vb, k), pos.reshape(-1, vb, 3))
    acc, _ = jax.lax.scan(step, jnp.zeros(n_cells, f.dtype), xs)
    return acc


@partial(jax.jit, static_argnames=("dims", "vb"))
def backward(
    r: jax.Array, cand: jax.Array, pred: tuple, ring_j: jax.Array, pos: jax.Array, scan: dict, dims: tuple, vb: int
) -> jax.Array:
    """Back-project a histogram onto each voxel's candidates: ``A^T r``.

    Parameters
    ----------
    r
        [n_cells] histogram (e.g. the ratio of data to prediction)
    cand
        [Nv, K] orientation indices, Nv a multiple of vb
    pred, ring_j, pos, scan, dims, vb
        See :func:`forward`

    Returns
    -------
    jax.Array
        [Nv, K]
    """
    k = cand.shape[1]

    def one(ch: tuple) -> jax.Array:
        cb, pb = ch
        e, o, u, w = (p[cb] for p in pred)
        idx, wt = system(e, o, u, w, ring_j, pb, scan, dims[0], dims[1], dims[2], dims[3])
        return jnp.sum(wt * jnp.where(idx >= 0, r[jnp.clip(idx, 0)], 0.0), (2, 3))

    return jax.lax.map(one, (cand.reshape(-1, vb, k), pos.reshape(-1, vb, 3))).reshape(-1, k)


def mlem(
    d: jax.Array, cand: jax.Array, pred: tuple, ring_j: jax.Array, pos: jax.Array, scan: dict, dims: tuple,
    f0: jax.Array, n_iter: int, vb: int, log: Callable = print,
) -> jax.Array:  # fmt: skip
    """Fit occupancies by MLEM: ``f <- f A^T(d / A f) / A^T 1``, n_iter times, logging the Poisson deviance.

    Parameters
    ----------
    d
        [n_cells] histogram
    cand, pred, ring_j, pos, scan, dims, vb
        See :func:`forward`
    f0
        [Nv, K] starting occupancies
    n_iter
        Iterations
    log
        Progress messages

    Returns
    -------
    jax.Array
        [Nv, K] occupancies
    """
    n_cells = d.shape[0]
    norm = jnp.maximum(backward(jnp.ones(n_cells, d.dtype), cand, pred, ring_j, pos, scan, dims, vb), 1e-30)
    f = f0
    for it in range(n_iter):
        Af = forward(f, cand, pred, ring_j, pos, scan, dims, n_cells, vb)
        ratio = jnp.where(Af > 0, d / jnp.maximum(Af, 1e-30), 0.0)
        f = f * backward(ratio, cand, pred, ring_j, pos, scan, dims, vb) / norm
        if it % 5 == 0 or it == n_iter - 1:
            dev = jnp.sum(jnp.where(d > 0, d * jnp.log(jnp.maximum(d, 1e-30) / jnp.maximum(Af, 1e-30)), 0.0) - d + Af)
            log(f"  MLEM {it}: deviance {2 * float(dev):.4g}")
    return f


@partial(jax.jit, static_argnames=("k2",))
def _inherit(near9: jax.Array, f_c: jax.Array, cand_c: jax.Array, k2: int) -> jax.Array:
    def one(nb: jax.Array) -> jax.Array:
        c, w = cand_c[nb].ravel(), f_c[nb].ravel()  # the neighbourhood's candidates and coarse occupancies
        o = jnp.argsort(c)
        c, w = c[o], w[o]
        dup = jnp.concatenate([jnp.array([False]), c[1:] == c[:-1]])
        _, i = jax.lax.top_k(jnp.where(dup, -1.0, w), k2)  # each orientation once
        return c[i]

    return jax.vmap(one)(near9)


def inherit_candidates(
    pos: ArrayLike, pos_c: ArrayLike, f_c: ArrayLike, cand_c: ArrayLike, k2: int, vb: int = 1 << 14
) -> jax.Array:
    """Candidates for fine voxels from a coarse fit: what the nearest coarse voxel and its 8 neighbours occupied.

    Parameters
    ----------
    pos
        [Nv, 3] fine voxel positions
    pos_c
        [Nc, 3] coarse voxel positions
    f_c, cand_c
        [Nc, K] coarse occupancies and orientation indices
    k2
        Candidates per fine voxel, the most occupied first
    vb
        Fine voxels per call

    Returns
    -------
    jax.Array
        [Nv, k2] orientation indices
    """
    from scipy.spatial import KDTree

    pos = np.asarray(pos)[:, :2]
    tree = KDTree(np.asarray(pos_c)[:, :2])  # the 9 nearest by a tree: sorting every distance was minutes on big maps
    f_c, cand_c = jnp.asarray(f_c), jnp.asarray(cand_c)
    out = []
    for s0 in range(0, len(pos), vb):
        _, near9 = tree.query(pos[s0 : s0 + vb], k=9)
        out.append(np.asarray(_inherit(jnp.asarray(near9, jnp.int32), f_c, cand_c, k2)))
    return jnp.asarray(np.concatenate(out))


def candidates_from(
    d: jax.Array, cand_in: jax.Array, pred: tuple, ring_j: jax.Array, pos: jax.Array, scan: dict, dims: tuple,
    k: int, vb: int, log: Callable = print,
) -> tuple[jax.Array, jax.Array]:  # fmt: skip
    """Pick each voxel's k best orientations as :func:`candidates` does, but from its own list only.

    Parameters
    ----------
    d
        [n_cells] histogram
    cand_in
        [Nv, K2] orientations to score per voxel, e.g. from :func:`inherit_candidates`
    pred, ring_j, pos, scan, dims, vb
        See :func:`forward`
    k
        Candidates kept per voxel
    log
        Progress messages

    Returns
    -------
    f1, cand: jax.Array
        [Nv, k] first-update occupancies and orientation indices
    """
    t0 = time.perf_counter()
    n_cells = d.shape[0]
    a1 = forward(jnp.ones(cand_in.shape, jnp.float32), cand_in, pred, ring_j, pos, scan, dims, n_cells, vb)
    r = jnp.where(a1 > 0, d / jnp.maximum(a1, 1e-30), 0.0)
    num = backward(r, cand_in, pred, ring_j, pos, scan, dims, vb)
    f1 = num / jnp.maximum(backward(jnp.ones(n_cells, jnp.float32), cand_in, pred, ring_j, pos, scan, dims, vb), 1e-30)
    val, i = jax.lax.top_k(f1, k)
    cand = jnp.take_along_axis(cand_in, i, 1)
    jax.block_until_ready(cand)
    log(f"  candidates: top {k} of {cand_in.shape[1]} inherited per voxel: {time.perf_counter() - t0:.1f} s")
    return val, cand


def fit_occupancy(
    H: ArrayLike,
    pred: tuple,
    ring_j: ArrayLike,
    pos: ArrayLike,
    scan: dict,
    dims: tuple,
    k: int = 64,
    n_iter: int = 10,
    coarse: int = 1,
    block_bytes: float = 1e9,
    qc: int = 16,
    log: Callable = print,
    return_model: bool = False,
) -> tuple:
    """Fit sparse occupancies: candidates per voxel, then MLEM.

    With ``coarse = G > 1`` the full candidate pass runs on voxels G times larger (the histogram's rows summed in
    groups of G), and each voxel then scores only the 2k orientations its 3 x 3 coarse neighbourhood occupied most:
    about G^2 times cheaper for large maps, at the risk of missing a grain the coarse fit missed.

    Parameters
    ----------
    H
        [n_cells] histogram, rows last
    pred
        (eta, om, use, w), each [Nq, Nj] with Nq a multiple of qc
    ring_j
        [Nj] ring of each prediction
    pos
        [Nv, 3] voxel positions
    scan, dims
        Rows and (b_e, b_o, n_e, n_o) of the histogram, see :func:`system`
    k
        Candidates per voxel
    n_iter
        MLEM iterations
    coarse
        Coarse-to-fine factor (1: off)
    block_bytes
        Memory for one block of voxels' system entries
    qc
        Orientations per chunk in passes over every orientation
    return_model
        Also return the fitted histogram ``A f``, e.g. to compare with the data row by row
    log
        Progress messages

    Returns
    -------
    f, cand: np.ndarray
        [Nv, k] occupancies and orientation indices
    model: np.ndarray
        [n_cells] the fitted histogram, if return_model
    """
    H, ring_j, pos = jnp.asarray(H), jnp.asarray(ring_j), np.asarray(pos, np.float32)
    nv = len(pos)
    n_j = pred[0].shape[1]
    vb_all, vb = block_voxels(qc, n_j, block_bytes), block_voxels(k, n_j, block_bytes)
    vbp = max(vb_all, vb)
    pos_p = pad_voxels(pos, vbp)
    t0 = time.perf_counter()
    if coarse == 1:
        log(f"candidates: {nv} voxels x {pred[0].shape[0]} orientations, the top {k} per voxel")
        f0, cand = candidates(H, pred, ring_j, pos_p, scan, dims, k, vb_all, qc=qc, log=log)
    else:
        H_c, n_rows_c = coarsen_rows(H, scan["n_rows"], coarse)
        ystep = scan["ystep"]
        scan_c = {
            **scan,
            "dty0": scan["dty0"] + 0.5 * (coarse - 1) * ystep,
            "ystep": coarse * ystep,
            "n_rows": n_rows_c,
        }
        extent = float(np.abs(np.asarray(pos)[:, :2]).max())
        n_c = 2 * int(np.ceil(extent / (coarse * ystep))) + 3
        gc = (np.arange(n_c) - (n_c - 1) / 2) * coarse * ystep
        xx, yy = np.meshgrid(gc, gc, indexing="ij")
        pos_c = np.stack([xx.ravel(), yy.ravel(), np.zeros(n_c * n_c)], 1).astype(np.float32)
        pos_cp = pad_voxels(pos_c, vbp)
        log(
            f"coarse: {n_c} x {n_c} voxels of {coarse * ystep:g} x {pred[0].shape[0]} orientations, the top {k} per voxel"
        )
        f_c, cand_c = candidates(H_c, pred, ring_j, pos_cp, scan_c, dims, k, vb_all, qc=qc, log=log)
        f_c = mlem(H_c, cand_c, pred, ring_j, pos_cp, scan_c, dims, f_c, n_iter, vb, log=log)
        cand_in = inherit_candidates(pos_p, pos_c, f_c[: n_c * n_c], cand_c[: n_c * n_c], 2 * k)
        log(f"fine: {nv} voxels x {2 * k} orientations inherited from the coarse neighbourhood")
        f0, cand = candidates_from(H, cand_in, pred, ring_j, pos_p, scan, dims, k, max(vb // 2, 1), log=log)
    log(f"candidates: {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    f = mlem(H, cand, pred, ring_j, pos_p, scan, dims, f0, n_iter, vb, log=log)
    log(f"MLEM {n_iter} iterations: {time.perf_counter() - t0:.1f} s")
    if return_model:
        mu = forward(f, cand, pred, ring_j, pos_p, scan, dims, H.shape[0], vb)
        return np.asarray(f)[:nv], np.asarray(cand)[:nv], np.asarray(mu)
    return np.asarray(f)[:nv], np.asarray(cand)[:nv]
