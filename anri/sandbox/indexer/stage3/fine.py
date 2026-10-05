"""Prototype of stage 3: a sparse fine histogram, a fine system matrix (parallax, beam profile), MLEM on local grids.

The fine histogram is stored as CSR over (ring, eta, omega) cells: for each cell, the sorted dty rows with data.
Lookups are a fixed number of binary-search steps, all in int32 (the full key would overflow int32 on big scans).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from anri.fwd import beam_weight, hkl_to_k_omega, lorentz, polarisation
from anri.fwd._impl.render import _scattering_origin
from anri.geom import raytrace_to_det, sample_to_lab
from anri.index._impl.data import _tth_eta, pixel_angles


# ----------------------------------------------------------------------------------------------- the sparse histogram
@partial(jax.jit, static_argnames=("n_ring", "n_e", "n_o", "n_k"))
def _cells(x, row, ring_tth, tth_tol, om0, b_e, b_o, n_ring, n_e, n_o, n_k):
    """Cell (ring, eta, omega) and row of each pixel, -1 where it falls outside."""
    i = jnp.clip(jnp.searchsorted(ring_tth, x[:, 0]), 1, max(n_ring - 1, 1))
    ring = jnp.where(jnp.abs(x[:, 0] - ring_tth[i - 1]) < jnp.abs(x[:, 0] - ring_tth[i % n_ring]), i - 1, i % n_ring)
    ok = jnp.abs(x[:, 0] - ring_tth[ring]) < tth_tol[ring]
    ie = jnp.floor((x[:, 1] + 180.0) / b_e).astype(jnp.int32) % n_e
    io = jnp.floor((x[:, 2] - om0) / b_o).astype(jnp.int32)
    ok = ok & (io >= 0) & (io < n_o) & (row >= 0) & (row < n_k)
    return jnp.where(ok, (ring * n_e + ie) * n_o + io, -1)


def sparse_histogram(chunks, geom, ring_tth, tth_tol, om0, bins, n_rows, chunk):
    """Sum pixels into the non-empty (cell, row) bins. Returns CSR: start [n_cells + 1], row [nnz], value [nnz]."""
    b_e, b_o, n_e, n_o = bins
    n_ring = len(ring_tth)
    ring_tth, tth_tol = jnp.asarray(ring_tth, jnp.float32), jnp.asarray(tth_tol, jnp.float32)
    keys, vals = [], []
    for slow, fast, omega, row, value in chunks:
        m = len(value)
        pad = lambda a, dt=np.float32, m=m: jnp.asarray(np.pad(np.asarray(a, dt), (0, chunk - m)))  # noqa: E731
        x = pixel_angles(pad(slow), pad(fast), pad(omega), geom)
        cell = np.asarray(_cells(x, pad(row, np.int32), ring_tth, tth_tol, om0, b_e, b_o, n_ring, n_e, n_o, n_rows))[:m]
        ok = cell >= 0
        k = cell[ok].astype(np.int64) * n_rows + np.asarray(row)[ok]
        u, inv = np.unique(k, return_inverse=True)  # reduce each chunk on the host
        keys.append(u)
        vals.append(np.bincount(inv, weights=np.asarray(value, np.float64)[ok]))
    k, inv = np.unique(np.concatenate(keys), return_inverse=True)
    v = np.bincount(inv, weights=np.concatenate(vals)).astype(np.float32)
    cell, row = k // n_rows, (k % n_rows).astype(np.int32)
    n_cells = n_ring * n_e * n_o
    start = np.searchsorted(cell, np.arange(n_cells + 1)).astype(np.int32)
    return {"start": start, "row": row, "value": v, "n_rows": n_rows}


def _lookup(start, rows, cell, row, n_steps):
    """Index of (cell, row) in the CSR, or -1: a binary search of fixed length within the cell's rows."""
    lo, hi = start[cell], start[cell + 1]
    for _ in range(n_steps):
        mid = (lo + hi) // 2
        go = rows[jnp.clip(mid, 0, rows.shape[0] - 1)] < row
        lo, hi = jnp.where(go & (lo < hi), mid + 1, lo), jnp.where(go | (lo >= hi), hi, mid)
    found = (lo < start[cell + 1]) & (rows[jnp.clip(lo, 0, rows.shape[0] - 1)] == row)
    return jnp.where(found, lo, -1)


# ----------------------------------------------------------------------------------------------- the fine model
def _fine_system(U, pos, hkls, F2, ring_j, geom, scan, bins, etacut, n_side):
    """(cell, row, weight) [vb, K, Nj, 4, 2 n_side + 1] for voxels pos [vb, 3] and their candidates U [vb, K, 3, 3].

    A prediction's spot is traced from the voxel's own position (parallax), binned bilinearly in (apparent eta,
    omega), and spread over the rows the beam reaches with the beam profile integrated over the voxel.
    """
    b_e, b_o, n_e, n_o = bins
    ubi = jnp.linalg.inv(U @ scan["B"])
    axis = sample_to_lab(jnp.array([0.0, 0.0, 1.0]), 0.0, geom["wedge"], geom["chi"], 0.0, 0.0)
    dk = jnp.arange(-n_side, n_side + 1)

    def one(u, p, h, e):  # one voxel, orientation, reflection, branch
        k_in, k_out, om, ok = hkl_to_k_omega(u, h, e, geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, geom["wedge"], geom["chi"])
        lp = lorentz(k_in, k_out, axis) * polarisation(k_in, k_out, geom["pol_factor"])
        c, s = jnp.cos(jnp.radians(om)), jnp.sin(jnp.radians(om))
        ylab = p[0] * s + p[1] * c
        kc = jnp.round((scan["y0"] - ylab - scan["dty0"]) / scan["ystep"]).astype(jnp.int32)  # the row it is centred in
        kk = kc + dk
        dty = scan["dty0"] + kk * scan["ystep"]
        w_row = jax.vmap(lambda d: beam_weight(sample_to_lab(p, om, geom["wedge"], geom["chi"], d, scan["y0"]), om, geom))(dty)
        origin = _scattering_origin(p, om, scan["dty0"] + kc * scan["ystep"], geom)
        sc, fc = raytrace_to_det(k_out, origin, geom["s_step_lab"], geom["f_step_lab"], geom["det_origin_lab"])
        pt = geom["det_origin_lab"] + sc * geom["s_step_lab"] + fc * geom["f_step_lab"]
        eta = _tth_eta(pt, geom["k_in_lab"])[1]  # apparent eta, seen from the lab origin as the data are
        use = ok & (jnp.abs(jnp.sin(jnp.radians(eta))) > etacut)
        return eta, om, use, lp, kk, w_row

    f = jax.vmap(jax.vmap(jax.vmap(jax.vmap(one, (None, None, None, 0)), (None, None, 0, None)), (0, None, None, None)), (0, 0, None, None))
    eta, om, use, lp, kk, w_row = f(ubi, pos, hkls, jnp.array([1.0, -1.0]))  # [vb, K, Nh, 2, ...]
    sh = eta.shape[:2] + (-1,)
    eta, om, use, lp = eta.reshape(sh), om.reshape(sh), use.reshape(sh), lp.reshape(sh)
    kk, w_row = kk.reshape(sh + (2 * n_side + 1,)), w_row.reshape(sh + (2 * n_side + 1,))
    om_w = jnp.mod(om - scan["om0"], 360.0) + scan["om0"]
    fe, fo = (eta + 180.0) / b_e - 0.5, (om_w - scan["om0"]) / b_o - 0.5
    e0, o0 = jnp.floor(fe).astype(jnp.int32), jnp.floor(fo).astype(jnp.int32)
    te, to = fe - e0, fo - o0
    w = lp * F2[None, None]
    cells, wts = [], []
    for de, we in ((0, 1.0 - te), (1, te)):
        for do, wo in ((0, 1.0 - to), (1, to)):
            ie, io = (e0 + de) % n_e, o0 + do
            good = use & (io >= 0) & (io < n_o)
            cells.append(jnp.where(good, (ring_j[None, None] * n_e + ie) * n_o + io, -1))
            wts.append(jnp.where(good, w * we * wo, 0.0))
    cell = jnp.stack(cells, -1)[..., None]  # [vb, K, Nj, 4, 1]
    wt = jnp.stack(wts, -1)[..., None] * w_row[..., None, :]  # [vb, K, Nj, 4, R]
    row = jnp.broadcast_to(kk[..., None, :], wt.shape)
    inside = (row >= 0) & (row < scan["n_rows"]) & (cell >= 0)
    return jnp.where(inside, cell, -1), row, jnp.where(inside, wt, 0.0)


@partial(jax.jit, static_argnames=("bins", "n_side", "n_steps"))
def fine_forward(f, U, pos, hkls, F2, ring_j, geom, scan, bins, etacut, n_side, data, n_steps):
    """A f at the data's non-empty bins: f [nb, vb, K], U [nb, vb, K, 3, 3], pos [nb, vb, 3] -> [nnz]; and A^T 1 per
    candidate (all bins inside the histogram, with data or not)."""
    nnz = data["row"].shape[0]

    def step(acc, ch):
        fb, ub, pb = ch
        cell, row, wt = _fine_system(ub, pb, hkls, F2, ring_j, geom, scan, bins, etacut, n_side)
        idx = _lookup(data["start"], data["row"], jnp.maximum(cell, 0), row, n_steps)
        idx = jnp.where(cell >= 0, idx, -1)
        c = (fb[:, :, None, None, None] * wt).astype(acc.dtype)
        acc = acc + jax.ops.segment_sum(c.ravel(), jnp.where(idx >= 0, idx, nnz).ravel(), nnz + 1)[:-1]
        return acc, jnp.sum(wt, (2, 3, 4))

    acc, norm = jax.lax.scan(step, jnp.zeros(nnz, jnp.float32), (f, U, pos))
    return acc, norm


@partial(jax.jit, static_argnames=("bins", "n_side", "n_steps"))
def fine_backward(r, U, pos, hkls, F2, ring_j, geom, scan, bins, etacut, n_side, data, n_steps):
    """A^T r for r at the data's non-empty bins -> [nb, vb, K]."""

    def one(ch):
        ub, pb = ch
        cell, row, wt = _fine_system(ub, pb, hkls, F2, ring_j, geom, scan, bins, etacut, n_side)
        idx = _lookup(data["start"], data["row"], jnp.maximum(cell, 0), row, n_steps)
        idx = jnp.where(cell >= 0, idx, -1)
        return jnp.sum(wt * jnp.where(idx >= 0, r[jnp.clip(idx, 0)], 0.0), (2, 3, 4))

    return jax.lax.map(one, (U, pos))


def fine_mlem(data, U, pos, f0, rings, geom, scan, bins, etacut, n_side, n_iter, log=print):
    """MLEM on per-voxel candidates U [nb, vb, K, 3, 3] against the sparse fine histogram."""
    n_steps = int(np.ceil(np.log2(max(data["n_rows"], 2)))) + 1
    d = jnp.asarray(data["value"])
    dj = {"start": jnp.asarray(data["start"]), "row": jnp.asarray(data["row"])}
    hk = jnp.asarray(rings["hkls"])
    F2 = jnp.asarray(np.repeat(rings.get("F2", np.ones(len(rings["hkls"]))), 2), jnp.float32)
    rj = jnp.asarray(rings["ring_j"])
    import time

    f = f0
    for it in range(n_iter):
        t0 = time.time()
        mu, norm = fine_forward(f, U, pos, hk, F2, rj, geom, scan, bins, etacut, n_side, dj, n_steps)
        ratio = jnp.where(mu > 0, d / jnp.maximum(mu, 1e-30), 0.0)
        f = f * fine_backward(ratio, U, pos, hk, F2, rj, geom, scan, bins, etacut, n_side, dj, n_steps) / jnp.maximum(norm, 1e-30)
        jax.block_until_ready(f)
        log(f"  fine MLEM iteration {it}: {time.time() - t0:.1f} s")
        if it % 5 == 0 or it == n_iter - 1:
            dev = 2 * float(jnp.sum(jnp.where(d > 0, d * jnp.log(jnp.maximum(d, 1e-30) / jnp.maximum(mu, 1e-30)), 0.0) - d)
                            + jnp.sum(f * norm))  # sum of mu over every bin, empty ones included
            log(f"  fine MLEM {it}: deviance {dev:.5g}")
    return f
