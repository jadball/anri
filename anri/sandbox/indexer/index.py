"""Global indexing by occupancy: a list of orientations, their occupancy per voxel, fitted to coarsened data.

Stage 1 (here): a regular Rodrigues grid over the cubic fundamental zone, the data histogrammed into
H[ring, eta, omega, row], and each orientation's completeness (fraction of its predicted reflections that land on
intensity) in the row-summed histogram.

All angles in degrees unless stated.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from anri.fwd._impl.base import hkl_to_k_omega
from anri.geom import beam_basis


def tth_eta(v: jax.Array, k_in: jax.Array) -> jax.Array:
    """[..., 2] (2theta, eta) of directions or lab points v [..., 3], seen from the lab origin."""
    k, e_h, e_v = beam_basis(k_in)
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    return jnp.stack([jnp.degrees(jnp.arccos(jnp.clip(v @ k, -1.0, 1.0))), jnp.degrees(jnp.arctan2(-(v @ e_h), v @ e_v))], -1)


@jax.jit
def pixels_to_x(slow: jax.Array, fast: jax.Array, omega: jax.Array, geom: dict) -> jax.Array:
    """[N, 3] (2theta, eta, omega) of sparse pixels (pixel centres; omega of their frame)."""
    p = geom["det_origin_lab"] + slow[:, None] * geom["s_step_lab"] + fast[:, None] * geom["f_step_lab"]
    return jnp.concatenate([tth_eta(p, geom["k_in_lab"]), omega[:, None]], 1)


# ----------------------------------------------------------------------------------------------- orientations
def cubic_fz_grid(step_deg: float) -> np.ndarray:
    """[N, 3] Rodrigues vectors on a regular grid inside the cubic fundamental zone (spacing ~step_deg near the origin).

    The zone: |r_i| <= tan(pi / 8) and |r_1| + |r_2| + |r_3| <= 1. Near the origin a rotation of angle t has |r| =
    tan(t / 2), so a grid step dr = tan(step / 2) is ~step degrees there (finer towards the zone's edge).
    """
    dr = np.tan(np.radians(step_deg) / 2)
    lim = np.tan(np.pi / 8)
    g = np.arange(-lim, lim + 1e-12, dr)
    r = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return r[np.abs(r).sum(1) <= 1.0].astype(np.float32)


def rod_to_mat(r: jax.Array) -> jax.Array:
    """[..., 3, 3] rotation matrices of Rodrigues vectors r = tan(t / 2) n."""
    rr = jnp.sum(r * r, -1)[..., None, None]
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    zero = jnp.zeros_like(x)
    K = jnp.stack([jnp.stack([zero, -z, y], -1), jnp.stack([z, zero, -x], -1), jnp.stack([-y, x, zero], -1)], -2)
    return jnp.eye(3) + 2.0 / (1.0 + rr) * (K + K @ K)


def mat_to_rod(U: np.ndarray) -> np.ndarray:
    """[..., 3] Rodrigues vectors of rotation matrices (rotation angle < 180 degrees)."""
    w = np.stack([U[..., 2, 1] - U[..., 1, 2], U[..., 0, 2] - U[..., 2, 0], U[..., 1, 0] - U[..., 0, 1]], -1)
    return w / (1.0 + np.trace(U, axis1=-2, axis2=-1))[..., None]


def cubic_ops() -> np.ndarray:
    """[24, 3, 3] proper rotations of the cubic point group."""
    ops = []
    for perm in ((0, 1, 2), (1, 2, 0), (2, 0, 1), (1, 0, 2), (0, 2, 1), (2, 1, 0)):
        for signs in np.ndindex(2, 2, 2):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = 1 - 2 * signs[i]
            if np.linalg.det(M) > 0:
                ops.append(M)
    return np.array(ops)


def to_fz(U: np.ndarray) -> np.ndarray:
    """[N, 3] Rodrigues vectors of orientations U [N, 3, 3] (crystal -> sample), reduced to the cubic zone."""
    cands = U[:, None] @ cubic_ops()[None]  # U S: the same lattice
    tr = np.trace(cands, axis1=-2, axis2=-1)
    best = np.argmax(tr, 1)  # smallest rotation angle
    return mat_to_rod(cands[np.arange(len(U)), best])


def disorientation(Ua: np.ndarray, Ub: np.ndarray) -> np.ndarray:
    """[N] smallest misorientation angle (degrees) between Ua and Ub [N, 3, 3] over the cubic symmetry."""
    d = np.swapaxes(Ua, -1, -2)[:, None] @ Ub[:, None] @ cubic_ops()[None]
    tr = np.clip((np.trace(d, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr.max(1)))


# ----------------------------------------------------------------------------------------------- predictions
@jax.jit
def predict(rod: jax.Array, B: jax.Array, hkls: jax.Array, geom: dict) -> tuple:
    """(eta, omega) [Nq, Nh, 2] of orientations rod [Nq, 3] for hkls [Nh, 3] (both branches: Nh -> 2 Nh), and valid."""
    U = rod_to_mat(rod)
    ubi = jnp.linalg.inv(U @ B)

    def one(u, h, e):  # noqa: ANN001, ANN202
        _, k_out, om, ok = hkl_to_k_omega(u, h, e, geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, geom["wedge"], geom["chi"])
        return tth_eta(k_out, geom["k_in_lab"])[1], om, ok

    f = jax.vmap(jax.vmap(jax.vmap(one, (None, None, 0)), (None, 0, None)), (0, None, None))
    eta, om, ok = f(ubi, hkls, jnp.array([1.0, -1.0]))  # [Nq, Nh, 2]
    nq = rod.shape[0]
    return eta.reshape(nq, -1), om.reshape(nq, -1), ok.reshape(nq, -1)


# ----------------------------------------------------------------------------------------------- data
@partial(jax.jit, static_argnames=("n_ring", "n_e", "n_o", "n_k"))
def histogram(x, val, row, ring_tth, tth_tol, om0, b_e, b_o, n_ring, n_e, n_o, n_k):  # noqa: ANN001, ANN201
    """Add pixels (x [N, 3] = (2theta, eta, omega), val, row) into H[ring, eta, omega, row] (flat, float32)."""
    i = jnp.clip(jnp.searchsorted(ring_tth, x[:, 0]), 1, n_ring - 1)
    ring = jnp.where(jnp.abs(x[:, 0] - ring_tth[i - 1]) < jnp.abs(x[:, 0] - ring_tth[i]), i - 1, i)
    ok = jnp.abs(x[:, 0] - ring_tth[ring]) < tth_tol
    ie = jnp.floor((x[:, 1] + 180.0) / b_e).astype(jnp.int32) % n_e
    io = jnp.floor((x[:, 2] - om0) / b_o).astype(jnp.int32)
    ok = ok & (io >= 0) & (io < n_o) & (row >= 0) & (row < n_k)
    idx = jnp.where(ok, ((ring * n_e + ie) * n_o + io) * n_k + row, n_ring * n_e * n_o * n_k)
    return jax.ops.segment_sum(jnp.where(ok, val, 0.0), idx, n_ring * n_e * n_o * n_k + 1)[:-1]


@partial(jax.jit, static_argnames=("de", "do"))
def dilate(lit, de, do):  # noqa: ANN001, ANN201
    """Lit map [Nr, n_e, n_o] dilated by +-de bins in eta (periodic) and +-do bins in omega: the matching tolerance."""
    x = jnp.concatenate([lit[:, -de:], lit, lit[:, :de]], 1).astype(jnp.float32) if de else lit.astype(jnp.float32)
    x = jax.lax.reduce_window(x, 0.0, jax.lax.max, (1, 2 * de + 1, 2 * do + 1), (1, 1, 1), "SAME")
    return (x[:, de:x.shape[1] - de] if de else x) > 0


@partial(jax.jit, static_argnames=("n_e", "n_o"))
def completeness(lit, eta, om, ok, ring_of_h, om0, b_e, b_o, n_e, n_o, etacut=0.0):  # noqa: ANN001, ANN201
    """Fraction of each orientation's valid predictions that land on a lit (ring, eta, omega) bin. lit [Nr, n_e, n_o]."""
    ok = ok & (jnp.abs(jnp.sin(jnp.radians(eta))) > etacut)
    om_w = jnp.mod(om - om0, 360.0) + om0
    ie = jnp.floor((eta + 180.0) / b_e).astype(jnp.int32) % n_e
    io = jnp.floor((om_w - om0) / b_o).astype(jnp.int32)
    inside = ok & (io >= 0) & (io < n_o)
    hit = lit[ring_of_h[None, :], ie, jnp.clip(io, 0, n_o - 1)] & inside
    return jnp.sum(hit, 1) / jnp.maximum(jnp.sum(inside, 1), 1), jnp.sum(inside, 1)


# ----------------------------------------------------------------------------------------------- occupancy (MLEM)
@jax.jit
def predict_lp(rod, B, hkls, geom):  # noqa: ANN001, ANN201
    """As predict, plus each prediction's Lorentz x polarisation factor."""
    from anri.fwd._impl.render import _peak_factors

    U = rod_to_mat(rod)
    ubi = jnp.linalg.inv(U @ B)

    def one(u, h, e):  # noqa: ANN001, ANN202
        _, k_out, om, ok = hkl_to_k_omega(u, h, e, geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, geom["wedge"], geom["chi"])
        return tth_eta(k_out, geom["k_in_lab"])[1], om, ok, _peak_factors(u, h, e, geom)

    f = jax.vmap(jax.vmap(jax.vmap(one, (None, None, 0)), (None, 0, None)), (0, None, None))
    out = f(ubi, hkls, jnp.array([1.0, -1.0]))
    return tuple(o.reshape(rod.shape[0], -1) for o in out)


def system(eta, om, use, w_qj, ring_j, pos, scan, b_e, b_o, n_e, n_o):  # noqa: ANN001, ANN201
    """Entries of the system matrix for a chunk of orientations: (cell index, weight) [Nv, Nq, Nj, 8].

    A prediction (eta, omega) of orientation q, reflection j is spread bilinearly over the 2 x 2 nearest (eta, omega) bins
    and linearly over the 2 rows nearest to where voxel v sits at that omega; weight w_qj (F^2 x Lorentz x polarisation).
    Parallax is ignored (coarse level). Unused corners have index -1.
    """
    n_k = scan["n_rows"]
    om_w = jnp.mod(om - scan["om0"], 360.0) + scan["om0"]
    fe = (eta + 180.0) / b_e - 0.5
    fo = (om_w - scan["om0"]) / b_o - 0.5
    e0, o0 = jnp.floor(fe).astype(jnp.int32), jnp.floor(fo).astype(jnp.int32)
    te, to = fe - e0, fo - o0
    # row of each voxel at each prediction's omega: lab y = x sin(om) + y cos(om) for a rotation about z
    c, s_ = jnp.cos(jnp.radians(om)), jnp.sin(jnp.radians(om))
    ylab = pos[:, 0, None, None] * s_[None] + pos[:, 1, None, None] * c[None]  # [Nv, Nq, Nj]
    fk = (scan["y0"] - ylab - scan["dty0"]) / scan["ystep"]
    k0 = jnp.floor(fk).astype(jnp.int32)
    tk = fk - k0
    idx, wt = [], []
    for de, we in ((0, 1.0 - te), (1, te)):
        for do, wo in ((0, 1.0 - to), (1, to)):
            for dk, wk in ((0, 1.0 - tk), (1, tk)):
                ie = (e0 + de) % n_e
                io = o0 + do
                kk = k0 + dk
                good = (use & (io >= 0) & (io < n_o))[None] & (kk >= 0) & (kk < n_k)
                cell = ((ring_j[None, None, :] * n_e + ie[None]) * n_o + io[None]) * n_k + kk
                idx.append(jnp.where(good, cell, -1))
                wt.append(jnp.where(good, (w_qj * we * wo)[None] * wk, 0.0))
    return jnp.stack(idx, -1), jnp.stack(wt, -1)


def _chunks(a, qc):  # noqa: ANN001, ANN202
    return a.reshape(-1, qc, *a.shape[1:])


@partial(jax.jit, static_argnames=("n_cells", "dims", "qc"))
def forward(f, pred, ring_j, pos, scan, dims, n_cells, qc=16):  # noqa: ANN001, ANN201
    """A f: occupancies f [Nv, Nq] -> [n_cells]. pred = (eta, om, use, w) each [Nq, Nj], Nq a multiple of qc."""
    b_e, b_o, n_e, n_o = dims

    def step(acc, ch):  # noqa: ANN001, ANN202
        fq, eta, om, use, w = ch
        idx, wt = system(eta, om, use, w, ring_j, pos, scan, b_e, b_o, n_e, n_o)
        contrib = fq.T[:, :, None, None] * wt
        return acc + jax.ops.segment_sum(contrib.ravel(), jnp.where(idx >= 0, idx, n_cells).ravel(), n_cells + 1)[:-1], None

    acc, _ = jax.lax.scan(step, jnp.zeros(n_cells, f.dtype), (_chunks(f.T, qc), *(_chunks(p, qc) for p in pred)))
    return acc


@partial(jax.jit, static_argnames=("dims", "qc"))
def backward(r, pred, ring_j, pos, scan, dims, qc=16):  # noqa: ANN001, ANN201
    """A^T r: r [n_cells] -> [Nv, Nq]."""
    b_e, b_o, n_e, n_o = dims

    def one(ch):  # noqa: ANN001, ANN202
        idx, wt = system(*ch, ring_j, pos, scan, b_e, b_o, n_e, n_o)
        return jnp.sum(wt * jnp.where(idx >= 0, r[jnp.clip(idx, 0)], 0.0), (2, 3))  # [Nv, qc]

    out = jax.lax.map(one, tuple(_chunks(p, qc) for p in pred))  # [Nq / qc, Nv, qc]
    return jnp.moveaxis(out, 0, 1).reshape(pos.shape[0], -1)


def mlem(d, pred, ring_j, pos, scan, dims, f0, n_iter, log=print, qc=16):  # noqa: ANN001, ANN201
    """MLEM: f <- f A^T(d / A f) / A^T 1, n_iter times. Logs the Poisson deviance."""
    n_cells = d.shape[0]
    norm = jnp.maximum(backward(jnp.ones(n_cells, d.dtype), pred, ring_j, pos, scan, dims, qc=qc), 1e-30)
    f = f0
    for it in range(n_iter):
        Af = forward(f, pred, ring_j, pos, scan, dims, n_cells, qc=qc)
        ratio = jnp.where(Af > 0, d / jnp.maximum(Af, 1e-30), 0.0)
        f = f * backward(ratio, pred, ring_j, pos, scan, dims, qc=qc) / norm
        if it % 5 == 0 or it == n_iter - 1:
            dev = 2 * float(jnp.sum(jnp.where(d > 0, d * jnp.log(jnp.maximum(d, 1e-30) / jnp.maximum(Af, 1e-30)), 0.0) - d + Af))
            log(f"  MLEM {it}: deviance {dev:.4g}")
    return f
