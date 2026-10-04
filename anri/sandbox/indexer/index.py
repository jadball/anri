"""Global indexing by occupancy: a list of orientations, their occupancy per voxel, fitted to coarsened data.

Stage 1 (here): a regular Rodrigues grid over the cubic fundamental zone, the data histogrammed into
H[ring, eta, omega, row], and each orientation's completeness (fraction of its predicted reflections that land on
intensity) in the row-summed histogram.

All angles in degrees unless stated.
"""

from __future__ import annotations

import time
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
    """Add pixels (x [N, 3] = (2theta, eta, omega), val, row) into H[ring, eta, omega, row] (flat, float32).

    A pixel goes to the nearest ring if within tth_tol of it (deg; a scalar or one per ring)."""
    i = jnp.clip(jnp.searchsorted(ring_tth, x[:, 0]), 1, n_ring - 1)
    ring = jnp.where(jnp.abs(x[:, 0] - ring_tth[i - 1]) < jnp.abs(x[:, 0] - ring_tth[i]), i - 1, i)
    ok = jnp.abs(x[:, 0] - ring_tth[ring]) < jnp.broadcast_to(tth_tol, ring_tth.shape)[ring]
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


# ----------------------------------------------------------------------------------------------- tolerances
def grid_misorientation(step_deg: float) -> float:
    """Largest misorientation (deg) from any orientation to its nearest point of cubic_fz_grid(step_deg).

    Half the diagonal of a grid cell, at the origin where the grid is coarsest.
    """
    return float(np.degrees(2 * np.arctan(np.sqrt(3) / 2 * np.tan(np.radians(step_deg) / 2))))


@partial(jax.jit, static_argnames=("n",))
def tth_profile(x, val, lo, step, n):  # noqa: ANN001, ANN201
    """[n] intensity of pixels (x [N, 3] = (2theta, eta, omega), val) in 2theta bins of width step from lo."""
    i = jnp.floor((x[:, 0] - lo) / step).astype(jnp.int32)
    ok = (i >= 0) & (i < n)
    return jax.ops.segment_sum(jnp.where(ok, val, 0.0), jnp.where(ok, i, n), n + 1)[:-1]


def ring_widths(prof: np.ndarray, lo: float, step: float, ring_tth: np.ndarray, frac: float = 0.95,
                max_hw: float = 0.5) -> tuple:  # fmt: skip
    """Measured (offset, half-width) [Nr] of each ring (deg 2theta) from a 2theta profile (tth_profile).

    Each ring is looked at within half the gap to its neighbours (at most max_hw). Background: the median of the outer
    fifth of that window on each side. Offset: the centroid minus ring_tth. Half-width: the distance from the centroid
    that holds frac of the ring's net intensity. It sums everything that spreads a ring: parallax (sample size /
    distance), strain, peak size, detector distortion. A ring without intensity gets offset 0 and the whole window.
    """
    x = lo + (np.arange(len(prof)) + 0.5) * step
    gaps = np.diff(ring_tth)
    half = np.minimum(np.concatenate([[2 * max_hw], gaps]), np.concatenate([gaps, [2 * max_hw]])) / 2
    off, hw = np.zeros(len(ring_tth)), np.zeros(len(ring_tth))
    for r, t in enumerate(ring_tth):
        m = np.abs(x - t) < half[r]
        xr, pr = x[m], prof[m]
        outer = np.abs(xr - t) > 0.8 * half[r]
        net = np.maximum(pr - (np.median(pr[outer]) if outer.any() else 0.0), 0.0)
        if net.sum() <= 0:
            hw[r] = half[r]
            continue
        c = np.sum(xr * net) / net.sum()
        d = np.abs(xr - c)
        o = np.argsort(d)
        k = np.searchsorted(np.cumsum(net[o]), frac * net.sum())
        off[r], hw[r] = c - t, d[o][min(k, len(o) - 1)] + step / 2
    return off, hw


def match_tolerances(eta, ring_j, ring_tth, ring_hw, delta, frame_step):  # noqa: ANN001, ANN201
    """Matching tolerances (tol_eta, tol_omega) [Nq, Nj] (deg) for predictions at eta [Nq, Nj] of grid orientations.

    The truth is up to delta (deg) from the nearest grid point (grid_misorientation). To first order a rotation by delta
    moves a reflection by up to delta / cos(theta) (1 + tan(theta) |cot(eta)|) in eta and delta / (cos(theta)
    |sin(eta)|) in omega (checked numerically up to 2 deg for |sin(eta)| > 0.3). Added: in eta, the ring's measured
    half-width ring_hw (deg 2theta, ring_widths) as the same displacement on the detector across the ring; in omega,
    half a frame. Peak widths are not added: a broad peak lights a broad region of the lit map.
    """
    th = jnp.radians(ring_tth[ring_j] / 2)[None]
    s = jnp.maximum(jnp.abs(jnp.sin(jnp.radians(eta))), 1e-3)
    c = jnp.abs(jnp.cos(jnp.radians(eta)))
    tol_e = delta / jnp.cos(th) * (1 + jnp.tan(th) * c / s) + ring_hw[ring_j][None] / (jnp.sin(2 * th) * jnp.cos(2 * th))
    tol_o = delta / (jnp.cos(th) * s) + frame_step / 2
    return tol_e, tol_o


@jax.jit
def lit_table(lit):  # noqa: ANN001, ANN201
    """Summed-area table [Nr, 3 n_e + 1, n_o + 1] (int32) of a lit map [Nr, n_e, n_o], eta tiled 3 times (periodic)."""
    t = jnp.concatenate([lit, lit, lit], 1).astype(jnp.int32)
    return jnp.pad(jnp.cumsum(jnp.cumsum(t, 1), 2), ((0, 0), (1, 0), (1, 0)))


@partial(jax.jit, static_argnames=("n_e", "n_o"))
def completeness_tol(table, eta, om, ok, ring_j, tol_e, tol_o, om0, b_e, b_o, n_e, n_o):  # noqa: ANN001, ANN201
    """Fraction of each orientation's valid predictions with a lit bin within +-tol_e in eta and +-tol_o in omega.

    table from lit_table; eta, om, ok, tol_e, tol_o [Nq, Nj]. Each prediction checks its own box (bins that may hold a
    point within its tolerance); tol_e must be below 360 deg. Returns (completeness [Nq], valid predictions [Nq]).
    """
    om_w = jnp.mod(om - om0, 360.0) + om0
    ie = jnp.floor((eta + 180.0) / b_e).astype(jnp.int32) % n_e
    io = jnp.floor((om_w - om0) / b_o).astype(jnp.int32)
    de = jnp.minimum(jnp.ceil(tol_e / b_e).astype(jnp.int32), n_e - 1)
    do = jnp.ceil(tol_o / b_o).astype(jnp.int32)
    inside = ok & (io >= 0) & (io < n_o)
    e0, e1 = ie + n_e - de, ie + n_e + de + 1
    o0, o1 = jnp.clip(io - do, 0, n_o), jnp.clip(io + do + 1, 0, n_o)
    r = ring_j[None, :]
    count = table[r, e1, o1] - table[r, e0, o1] - table[r, e1, o0] + table[r, e0, o0]
    hit = (count > 0) & inside
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
    """Entries of the system matrix for voxels pos [Nv, 3] and predictions [1 or Nv, Q, Nj]: (cell index, weight)
    [Nv, Q, Nj, 8]. Predictions with a leading 1 are the same orientations for every voxel (dense); with Nv, each
    voxel's own (sparse: its candidates).

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
    ylab = pos[:, 0, None, None] * s_ + pos[:, 1, None, None] * c  # [Nv, Q, Nj]
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
                good = use & (io >= 0) & (io < n_o) & (kk >= 0) & (kk < n_k)
                cell = ((ring_j[None, None, :] * n_e + ie) * n_o + io) * n_k + kk
                idx.append(jnp.where(good, cell, -1))
                wt.append(jnp.where(good, w_qj * we * wo * wk, 0.0))
    return jnp.stack(idx, -1), jnp.stack(wt, -1)


def _chunks(a, qc):  # noqa: ANN001, ANN202
    return a.reshape(-1, qc, *a.shape[1:])


@partial(jax.jit, static_argnames=("n_cells", "dims", "qc"))
def forward(f, pred, ring_j, pos, scan, dims, n_cells, qc=16):  # noqa: ANN001, ANN201
    """A f: occupancies f [Nv, Nq] -> [n_cells]. pred = (eta, om, use, w) each [Nq, Nj], Nq a multiple of qc."""
    b_e, b_o, n_e, n_o = dims

    def step(acc, ch):  # noqa: ANN001, ANN202
        fq, eta, om, use, w = ch
        idx, wt = system(eta[None], om[None], use[None], w[None], ring_j, pos, scan, b_e, b_o, n_e, n_o)
        contrib = fq.T[:, :, None, None] * wt
        return acc + jax.ops.segment_sum(contrib.ravel(), jnp.where(idx >= 0, idx, n_cells).ravel(), n_cells + 1)[:-1], None

    acc, _ = jax.lax.scan(step, jnp.zeros(n_cells, f.dtype), (_chunks(f.T, qc), *(_chunks(p, qc) for p in pred)))
    return acc


@partial(jax.jit, static_argnames=("dims", "qc"))
def backward(r, pred, ring_j, pos, scan, dims, qc=16):  # noqa: ANN001, ANN201
    """A^T r: r [n_cells] -> [Nv, Nq]."""
    b_e, b_o, n_e, n_o = dims

    def one(ch):  # noqa: ANN001, ANN202
        idx, wt = system(*(c[None] for c in ch), ring_j, pos, scan, b_e, b_o, n_e, n_o)
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


# ----------------------------------------------------------------------------------------------- crystallography
def allowed(hkls: np.ndarray, sym_matrices: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """[N] False for systematically absent reflections of a space group.

    hkls [N, 3]; sym_matrices [M, 4, 4] space-group operations (R, t) on fractional coordinates (Dans_Diffraction's
    Symmetry.symmetry_matrices). h is absent if some operation has h R = h and h . t not an integer: then F(h) =
    exp(2 pi i h . t) F(h), so F(h) = 0. Covers lattice centring, screw axes and glide planes.
    """
    R, t = sym_matrices[:, :3, :3], sym_matrices[:, :3, 3]
    fixed = np.all(np.abs(np.einsum("ni,mij->nmj", hkls, R) - hkls[:, None, :]) < tol, -1)  # [N, M]: h R = h
    phase = np.einsum("ni,mi->nm", hkls, t)
    shifted = np.abs(phase - np.round(phase)) > tol
    return ~np.any(fixed & shifted, 1)


# ----------------------------------------------------------------------------------------------- sparse occupancy
# Each voxel keeps K candidate orientations: occupancies f [Nv, K] of orientations cand [Nv, K] (indices into the
# orientation list). Work runs over blocks of vb voxels, so memory is set by vb, not by the map's size.
def block_voxels(q: int, n_j: int, budget_bytes: float) -> int:
    """Voxels per block (a power of 2) so that one block's system entries for q orientations take ~budget_bytes.

    ~256 bytes per (voxel, orientation, reflection): 8 corners of index and weight, their temporaries, the gathers.
    """
    return int(2 ** max(0, np.floor(np.log2(budget_bytes / (256 * q * n_j)))))


def pad_voxels(pos: jax.Array, vb: int) -> jax.Array:
    """pos [Nv, 3] padded to a multiple of vb with voxels far outside the scan (they reach no rows)."""
    return jnp.concatenate([pos, jnp.full((-pos.shape[0] % vb, 3), 1e6, pos.dtype)])


@partial(jax.jit, static_argnames=("n_cells", "dims", "qc"))
def _ones_block(pos_b, pred, ring_j, scan, dims, n_cells, qc):  # noqa: ANN001, ANN202
    """A 1 from one block of voxels: every orientation at unit occupancy -> [n_cells]."""

    def step(acc, ch):  # noqa: ANN001, ANN202
        idx, wt = system(*(c[None] for c in ch), ring_j, pos_b, scan, *dims)
        return acc + jax.ops.segment_sum(wt.ravel(), jnp.where(idx >= 0, idx, n_cells).ravel(), n_cells + 1)[:-1], None

    acc, _ = jax.lax.scan(step, jnp.zeros(n_cells, jnp.float32), tuple(_chunks(p, qc) for p in pred))
    return acc


@partial(jax.jit, static_argnames=("dims", "qc", "k"))
def _top_block(r, pos_b, pred, ring_j, scan, dims, qc, k):  # noqa: ANN001, ANN202
    """(A^T r / A^T 1) for one block of voxels and every orientation, reduced to its top k: (value, index) [vb, k]."""

    def one(ch):  # noqa: ANN001, ANN202
        idx, wt = system(*(c[None] for c in ch), ring_j, pos_b, scan, *dims)
        num = jnp.sum(wt * jnp.where(idx >= 0, r[jnp.clip(idx, 0)], 0.0), (2, 3))
        return num / jnp.maximum(jnp.sum(wt, (2, 3)), 1e-30)  # [vb, qc]

    s = jax.lax.map(one, tuple(_chunks(p, qc) for p in pred))  # [Nq / qc, vb, qc]
    val, ind = jax.lax.top_k(jnp.moveaxis(s, 0, 1).reshape(pos_b.shape[0], -1), k)
    return val, ind.astype(jnp.int32)


def candidates(d, pred, ring_j, pos, scan, dims, k, vb, qc=16, log=print):  # noqa: ANN001, ANN201
    """Each voxel's k best orientations by the first MLEM update from unit occupancy, f1 = A^T(d / A 1) / A^T 1.

    Two passes over every (voxel, orientation), in blocks of vb voxels (pos a multiple of vb, pad_voxels). The
    division by A 1 down-weights crowded cells, as MLEM does. Returns (f1 [Nv, k], cand [Nv, k] int32).
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
def forward_sparse(f, cand, pred, ring_j, pos, scan, dims, n_cells, vb):  # noqa: ANN001, ANN201
    """A f for occupancies f [Nv, K] of orientations cand [Nv, K] (Nv a multiple of vb) -> [n_cells]."""
    k = f.shape[1]

    def step(acc, ch):  # noqa: ANN001, ANN202
        fb, cb, pb = ch
        idx, wt = system(*(p[cb] for p in pred), ring_j, pb, scan, *dims)  # [vb, K, Nj, 8]
        contrib = fb[:, :, None, None] * wt
        return acc + jax.ops.segment_sum(contrib.ravel(), jnp.where(idx >= 0, idx, n_cells).ravel(), n_cells + 1)[:-1], None

    xs = (f.reshape(-1, vb, k), cand.reshape(-1, vb, k), pos.reshape(-1, vb, 3))
    acc, _ = jax.lax.scan(step, jnp.zeros(n_cells, f.dtype), xs)
    return acc


@partial(jax.jit, static_argnames=("dims", "vb"))
def backward_sparse(r, cand, pred, ring_j, pos, scan, dims, vb):  # noqa: ANN001, ANN201
    """A^T r at each voxel's candidates: r [n_cells] -> [Nv, K]."""
    k = cand.shape[1]

    def one(ch):  # noqa: ANN001, ANN202
        cb, pb = ch
        idx, wt = system(*(p[cb] for p in pred), ring_j, pb, scan, *dims)
        return jnp.sum(wt * jnp.where(idx >= 0, r[jnp.clip(idx, 0)], 0.0), (2, 3))  # [vb, K]

    return jax.lax.map(one, (cand.reshape(-1, vb, k), pos.reshape(-1, vb, 3))).reshape(-1, k)


def mlem_sparse(d, cand, pred, ring_j, pos, scan, dims, f0, n_iter, vb, log=print):  # noqa: ANN001, ANN201
    """MLEM on sparse occupancies: f <- f A^T(d / A f) / A^T 1, n_iter times. Logs the Poisson deviance."""
    n_cells = d.shape[0]
    norm = jnp.maximum(backward_sparse(jnp.ones(n_cells, d.dtype), cand, pred, ring_j, pos, scan, dims, vb), 1e-30)
    f = f0
    for it in range(n_iter):
        Af = forward_sparse(f, cand, pred, ring_j, pos, scan, dims, n_cells, vb)
        ratio = jnp.where(Af > 0, d / jnp.maximum(Af, 1e-30), 0.0)
        f = f * backward_sparse(ratio, cand, pred, ring_j, pos, scan, dims, vb) / norm
        if it % 5 == 0 or it == n_iter - 1:
            dev = 2 * float(jnp.sum(jnp.where(d > 0, d * jnp.log(jnp.maximum(d, 1e-30) / jnp.maximum(Af, 1e-30)), 0.0) - d + Af))
            log(f"  MLEM {it}: deviance {dev:.4g}")
    return f


# ----------------------------------------------------------------------------------------------- populations
@partial(jax.jit, static_argnames=("m", "p"))
def _populations(f, cand, U_list, ops, radius, eps, m, p):  # noqa: ANN001, ANN202
    def one(fv, cv):  # noqa: ANN001, ANN202
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


def populations(f, cand, U_list, step_deg, p=4, m=16, eps=0.02, link=1.8, vb=1 << 14):  # noqa: ANN001, ANN201
    """Group each voxel's occupied candidates into up to p populations: (fraction, U, spread, n) [Nv, p, ...].

    A voxel's m most occupied candidates above eps x its total are grouped greedily: the most occupied free candidate
    seeds a population, and every free candidate within link x step_deg of it (over the cubic symmetry) joins. Each
    population gets its share of the voxel's occupancy, the occupancy-weighted mean orientation U (crystal -> sample)
    of its members, and their weighted rms misorientation from that mean (deg). Populations closer than ~link grid steps
    are not resolved: they come out as one, with a larger spread. The spread includes the grid's own spacing (a single
    orientation between grid points shares itself among the cell's corners), so it is an upper bound. Absent
    populations have fraction 0.
    """
    ops = jnp.asarray(cubic_ops(), jnp.float32)
    radius = float(np.radians(link * step_deg))
    nv = f.shape[0]
    out = []
    for s0 in range(0, nv, vb):
        fb, cb = f[s0:s0 + vb], cand[s0:s0 + vb]
        pad = vb - fb.shape[0] if nv > vb else 0
        fb = jnp.pad(jnp.asarray(fb), ((0, pad), (0, 0)))
        cb = jnp.pad(jnp.asarray(cb), ((0, pad), (0, 0)))
        out.append([np.asarray(x)[: vb - pad] for x in _populations(fb, cb, U_list, ops, radius, eps, m=min(m, f.shape[1]), p=p)])
    frac, U, spread, n = (np.concatenate(x) for x in zip(*out))
    return frac, U, np.degrees(spread), n


# ----------------------------------------------------------------------------------------------- coarse to fine
def coarsen_rows(d, n_k: int, g: int):  # noqa: ANN001, ANN201
    """Data [..., n_k] (flat) with rows summed in groups of g -> (flat data, rows): row k goes to k // g."""
    d = jnp.asarray(d).reshape(-1, n_k)
    d = jnp.pad(d, ((0, 0), (0, -n_k % g)))
    return d.reshape(d.shape[0], -1, g).sum(-1).ravel(), -(-n_k // g)


@partial(jax.jit, static_argnames=("k2",))
def _inherit(near9, f_c, cand_c, k2):  # noqa: ANN001, ANN202
    def one(nb):  # noqa: ANN001, ANN202
        c, w = cand_c[nb].ravel(), f_c[nb].ravel()  # [9 K]: the neighbourhood's candidates and coarse occupancies
        o = jnp.argsort(c)
        c, w = c[o], w[o]
        dup = jnp.concatenate([jnp.array([False]), c[1:] == c[:-1]])
        _, i = jax.lax.top_k(jnp.where(dup, -1.0, w), k2)  # each orientation once, by its first copy's occupancy
        return c[i]

    return jax.vmap(one)(near9)


def inherit_candidates(pos, pos_c, f_c, cand_c, k2, vb=1 << 14):  # noqa: ANN001, ANN201
    """[Nv, k2] candidates of fine voxels pos [Nv, 3] from a coarse fit (f_c, cand_c [Nc, K] at pos_c [Nc, 3]): the
    orientations occupied in the nearest coarse voxel and its 8 neighbours (nearest 9 by distance), most occupied first.
    """
    pc = np.asarray(pos_c)[:, :2]
    out = []
    for s0 in range(0, pos.shape[0], vb):
        p = np.asarray(pos[s0:s0 + vb])[:, :2]
        near9 = np.argsort(np.linalg.norm(p[:, None] - pc[None], axis=2), 1)[:, :9]
        out.append(np.asarray(_inherit(jnp.asarray(near9), jnp.asarray(f_c), jnp.asarray(cand_c), k2=k2)))
    return jnp.asarray(np.concatenate(out))


def candidates_from(d, cand_in, pred, ring_j, pos, scan, dims, k, vb, log=print):  # noqa: ANN001, ANN201
    """As candidates, but each voxel scores only its own list cand_in [Nv, K2] (e.g. inherit_candidates): f1 =
    A^T(d / A 1) / A^T 1 over those, then the top k. Cost ~ Nv x K2, not Nv x all orientations."""
    t0 = time.perf_counter()
    n_cells = d.shape[0]
    a1 = forward_sparse(jnp.ones(cand_in.shape, jnp.float32), cand_in, pred, ring_j, pos, scan, dims, n_cells, vb)
    r = jnp.where(a1 > 0, d / jnp.maximum(a1, 1e-30), 0.0)
    f1 = backward_sparse(r, cand_in, pred, ring_j, pos, scan, dims, vb) / jnp.maximum(
        backward_sparse(jnp.ones(n_cells, jnp.float32), cand_in, pred, ring_j, pos, scan, dims, vb), 1e-30)
    val, i = jax.lax.top_k(f1, k)
    cand = jnp.take_along_axis(cand_in, i, 1)
    jax.block_until_ready(cand)
    log(f"  candidates: top {k} of {cand_in.shape[1]} inherited per voxel: {time.perf_counter() - t0:.1f} s")
    return val, cand
