"""Refine orientations against measured pixels, with the expensive geometry linearised and a cheap model fitted.

Rendering a peak has an expensive part (ray tracing its centroid, propagating its covariance, the beam's profile
and the intensity factors) and a cheap part (integrating a Gaussian over the cells of its window). For small
rotations only the centroid moves, and linearly. So:

- **linearise** (now and then): for every peak in every dty row ("instance"), its centroid mu0 = (slow, fast,
  omega) and d mu / d theta for a small rotation theta of its entry, its covariance, its window, its amplitude
  times the beam's weight per frame of the window, and which cells of the window land on measured pixels;
- **sweep** (many times): mu = mu0 + J theta, the Gaussians in the windows, the model at the measured pixels (peaks
  on the same pixel summed, so voxels on the same beam path are fitted jointly), the residual, the censored cells
  (model above the segmentation cut where nothing was measured), and for each entry its gradient and curvature.
  Each entry takes its own damped Gauss-Newton step; each pixel's term in an entry's curvature is weighted by the
  whole model there over the entry's share (a separable surrogate), so entries sharing pixels cannot overshoot
  together.

Densities and spreads are held fixed: the indexer decides which population owns each voxel.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from anri.fwd._impl.render import (
    _frame_weights,
    _peak_factors,
    _peak_shape,
    _select_margin,
    _window_cells,
    select_peaks,
)


def _skew(w: jax.Array) -> jax.Array:
    z = jnp.zeros((), w.dtype)
    return jnp.stack([jnp.stack([z, -w[2], w[1]]), jnp.stack([w[2], z, -w[0]]), jnp.stack([-w[1], w[0], z])])


def _rotation(w: jax.Array) -> jax.Array:
    """Rotation matrices of rotation vectors w [N, 3] (radians), Rodrigues."""
    t = jnp.linalg.norm(w, axis=1)[:, None, None]
    K = jax.vmap(_skew)(w)
    small = t < 1e-12
    ts = jnp.where(small, 1.0, t)
    a = jnp.where(small, 1.0, jnp.sin(ts) / ts)
    b = jnp.where(small, 0.5, (1.0 - jnp.cos(ts)) / ts**2)
    return jnp.eye(3, dtype=w.dtype) + a * K + b * K @ K


@partial(jax.jit, static_argnames=("window", "det_shape", "n_search"))
def _linearise_chunk(ubi, pos, sig, dens, hkls, F2, geom, rows, pix_all, fs, e, h, b, ri, window, det_shape, n_search):  # noqa: ANN001, ANN202
    """Linearise a fixed-size chunk of instances, each in its own row of the stacked ``rows``.

    Per instance: mu0, J, the covariance, the window (edges, pixel origins, frames inside the row), the amplitude
    times the frame weights, the row's omega range, and for each window cell the index of its measured pixel in
    ``pix_all`` (or -1), by a binary search within the frame (``fs``: each row's frame starts in ``pix_all``).
    """
    wo = window[0]
    nfr = rows["omega_sorted"].shape[1]
    last = pix_all.shape[0] - 1

    def one(e, h, b, ri):  # noqa: ANN001, ANN202
        row = jax.tree.map(lambda a: a[ri], rows)
        etasign = 1.0 - 2.0 * b
        sg = None if sig is None else sig[e]

        def centroid(w):  # noqa: ANN001, ANN202
            u = ubi[e] @ (jnp.eye(3, dtype=ubi.dtype) + _skew(w)).T
            return _peak_shape(u, pos[e], hkls[h], etasign, sg, row, geom)[0]

        w0 = jnp.zeros(3, ubi.dtype)
        mu0, J = centroid(w0), jax.jacfwd(centroid)(w0)
        _, shape, valid = _peak_shape(ubi[e], pos[e], hkls[h], etasign, sg, row, geom)
        _, inside, fclip, rows_, cols, om_fr, (jo, i0, j0) = _window_cells(mu0, shape, None, row, window, det_shape)
        amp = dens[e] * F2[h] * _peak_factors(ubi[e], hkls[h], etasign, geom)
        wfr = amp * _frame_weights(pos[e], om_fr, fclip, row, geom)
        frames = jo + jnp.arange(wo)
        fin = (frames >= 0) & (frames < nfr) & valid
        edges = row["omega_edges"][jnp.clip(jo + jnp.arange(wo + 1), 0, nfr)]
        # each cell's measured pixel: binary search among the frame's pixels (sorted)
        pixel = rows_[:, :, None] * det_shape[1] + cols[:, None, :]  # [wo, ws, wf]
        f = row["order"][fclip]  # file-order frame of each window frame
        lo, hi = fs[ri, f], fs[ri, f + 1]  # [wo]
        lo, hi = jnp.broadcast_to(lo[:, None, None], pixel.shape), jnp.broadcast_to(hi[:, None, None], pixel.shape)
        end = hi

        def step(_: int, lh: tuple) -> tuple:
            lo, hi = lh
            mid = (lo + hi) // 2
            right = pix_all[jnp.minimum(mid, last)] < pixel
            return jnp.where(right & (lo < hi), mid + 1, lo), jnp.where(right | (lo >= hi), hi, mid)

        lo, _ = jax.lax.fori_loop(0, n_search, step, (lo, hi))
        ok = inside & fin[:, None, None] & (lo < end) & (pix_all[jnp.minimum(lo, last)] == pixel)
        idx = jnp.where(ok, lo, -1)
        om = jnp.stack([row["omega_min"], row["omega_max"]])
        return mu0, J, jnp.stack(shape), edges, i0, j0, fin, wfr, om, idx

    return jax.vmap(one)(e, h, b, ri)


def _values(mu, shape, edges, i0, j0, fin, wfr, om_range, window, det_shape):  # noqa: ANN001, ANN202
    """One instance's model in the cells of its (fixed) window, [wo * ws * wf]."""
    wo = window[0]
    row = {"omega_edges": edges, "omega_sorted": edges[:wo], "omega_min": om_range[0], "omega_max": om_range[1]}
    frac, inside, _, _, _, _, _ = _window_cells(mu, tuple(shape), (0, i0, j0), row, window, det_shape)
    return jnp.where(inside & fin[:, None, None], wfr[:, None, None] * frac, 0.0).ravel()


def linearise(entries: dict, hkls: np.ndarray, F2: np.ndarray, geom: dict, rows: list, meas: list,
              det_shape: tuple[int, int], window: tuple[int, int, int], chunk: int = 1 << 15,
              log: Callable | None = None) -> dict:  # fmt: skip
    """Tables of every peak instance (a reflection of an entry in one dty row) at the entries' current orientations.

    Parameters
    ----------
    entries
        "ubi" [N, 3, 3], "pos" [N, 3], "density" [N] and optionally "sig_rot" [N] (radians)
    hkls, F2, geom, det_shape
        As for :func:`anri.fwd.render_row`
    rows, meas
        Each dty row: from :func:`anri.fwd.make_row` (all with the same number of frames) and
        :func:`anri.refine.measured`
    window
        (frames, slow, fast) of each instance's window, fixed until the next linearisation
    chunk
        Instances per batch in the sweeps

    Returns
    -------
    dict
        Instance tables padded to whole chunks ("e", "mu0", "J", "shape", "edges", "i0", "j0", "fin", "wfr", "om",
        "live"), the matched cells per chunk ("m_inst", "m_cell", "m_pix"), "data" (all measured values, plus a
        trailing 0 as a sentinel), and the sizes
    """
    ubi = jnp.asarray(entries["ubi"], jnp.float32)
    pos = jnp.asarray(entries["pos"], jnp.float32)
    dens = jnp.asarray(entries["density"], jnp.float32)
    sig = None if "sig_rot" not in entries else jnp.asarray(entries["sig_rot"], jnp.float32)
    hkls_j, F2_j = jnp.asarray(hkls, jnp.float32), jnp.asarray(F2, jnp.float32)
    geom = jax.tree.map(
        lambda x: jnp.asarray(x, jnp.float32) if np.asarray(x).dtype.kind == "f" else jnp.asarray(x), geom
    )
    t_start = time.perf_counter()
    rows_f = [{k: np.asarray(v, np.float32) if np.asarray(v).dtype.kind == "f" else np.asarray(v) for k, v in r.items()}
              for r in rows]  # fmt: skip
    if len({r["omega_sorted"].shape[0] for r in rows_f}) > 1:
        raise ValueError("linearise: every row needs the same number of frames")
    # 1. every instance (entry, hkl, branch, row): one fixed-shape selection per row
    sel = jax.jit(select_peaks, static_argnames="det_shape")
    parts = []
    for i, r in enumerate(rows_f):
        row = {k: jnp.asarray(v) for k, v in r.items()}
        margin = _select_margin(window, row, geom, jnp.float32)
        e, h, b = np.nonzero(np.asarray(sel(ubi, pos, hkls_j, geom, row, margin, det_shape=det_shape)))
        parts.append(np.stack([e, h, b, np.full(e.size, i)]).astype(np.int32))
    inst = np.concatenate(parts, axis=1)
    n_inst = inst.shape[1]
    t_sel = time.perf_counter() - t_start
    # 2. rows stacked; all measured pixels in one array, each row's frame starts offset into it (+ a sentinel that
    # matches nothing)
    rows_st = {k: jnp.asarray(np.stack([r[k] for r in rows_f])) for k in rows_f[0]}
    offsets = np.concatenate([[0], np.cumsum([m["pixel"].size for m in meas])])
    fs = jnp.asarray(np.stack([m["frame_start"] + o for m, o in zip(meas, offsets[:-1])]).astype(np.int32))
    pix_all = jnp.asarray(np.concatenate([m["pixel"] for m in meas] + [np.full(1, -1, np.int32)]).astype(np.int32))
    n_search = int(np.ceil(np.log2(max(int(np.diff(m["frame_start"]).max()) for m in meas) + 1))) + 1
    n_pix = int(offsets[-1])
    # 3. linearise in fixed-size chunks (one compile); the hits come out on the host (np.nonzero never compiles)
    lin = 1 << 13
    n_all = -(-n_inst // lin) * lin
    inst = np.pad(inst, ((0, 0), (0, n_all - n_inst)))
    names = ("mu0", "J", "shape", "edges", "i0", "j0", "fin", "wfr", "om")
    tabs: dict = {k: [] for k in names}
    m_inst, m_cell, m_pix = [], [], []
    for s0 in range(0, n_all, lin):
        e, h, b, ri = (jnp.asarray(x) for x in inst[:, s0 : s0 + lin])
        out = _linearise_chunk(ubi, pos, sig, dens, hkls_j, F2_j, geom, rows_st, pix_all, fs, e, h, b, ri, window,
                               det_shape, n_search)  # fmt: skip
        for k, v in zip(names, out[:-1]):
            tabs[k].append(v)
        idx = np.asarray(out[-1]).reshape(lin, -1)
        ii, cc = np.nonzero(idx[: max(0, min(lin, n_inst - s0))] >= 0)
        m_inst.append(ii + s0)
        m_cell.append(cc)
        m_pix.append(idx[ii, cc])
        if log and (s0 // lin) % 100 == 99:
            log(f"    linearise: {s0 + lin} of {n_inst} instances, {time.perf_counter() - t_start:.1f} s")
    # 4. tables in the sweeps' chunks (instances past n_inst are dead)
    n_chunks = -(-n_inst // chunk)
    out = {}
    for k in names:
        v = jnp.concatenate(tabs[k])
        need = n_chunks * chunk
        v = v[:need] if v.shape[0] >= need else jnp.concatenate([v, jnp.repeat(v[:1], need - v.shape[0], 0)])
        out[k] = v.reshape((n_chunks, chunk) + v.shape[1:])
    out["e"] = jnp.asarray(np.pad(inst[0, :n_inst], (0, n_chunks * chunk - n_inst)).reshape(n_chunks, chunk))
    out["live"] = jnp.asarray((np.arange(n_chunks * chunk) < n_inst).reshape(n_chunks, chunk))
    # matched cells per chunk, padded to the longest: padding points past the last cell (dropped by scatters) and at
    # the sentinel pixel
    m_inst, m_cell, m_pix = np.concatenate(m_inst), np.concatenate(m_cell), np.concatenate(m_pix)
    owner = m_inst // chunk
    counts = np.bincount(owner, minlength=n_chunks)
    # padded to a power of two: the length barely changes when the windows are re-centred, and a new length would
    # recompile the sweeps
    n_m = 1 << int(np.ceil(np.log2(max(int(counts.max()) if counts.size else 1, 1))))
    pos_in = np.arange(m_inst.size) - np.concatenate([[0], np.cumsum(counts)[:-1]])[owner]
    M_inst = np.zeros((n_chunks, n_m), np.int32)
    M_cell = np.full((n_chunks, n_m), window[0] * window[1] * window[2], np.int32)
    M_pix = np.full((n_chunks, n_m), n_pix, np.int32)
    M_inst[owner, pos_in], M_cell[owner, pos_in], M_pix[owner, pos_in] = m_inst - owner * chunk, m_cell, m_pix
    out.update({"m_inst": jnp.asarray(M_inst), "m_cell": jnp.asarray(M_cell), "m_pix": jnp.asarray(M_pix)})
    out["data"] = jnp.asarray(np.concatenate([m["value"] for m in meas] + [np.zeros(1, np.float32)]))
    out.update({"n_inst": n_inst, "n_pix": n_pix, "n_entries": int(ubi.shape[0])})
    if log:
        log(f"    linearise: {n_inst} instances (selection {t_sel:.1f} s), {m_inst.size} matched cells, "
            f"{time.perf_counter() - t_start:.1f} s")  # fmt: skip
    return out


def _chunk_values(c: dict, theta: jax.Array, window: tuple, det_shape: tuple) -> jax.Array:
    """Model in the window cells of one chunk of instances, [chunk, cells], at entry rotations theta [N, 3]."""
    mu = c["mu0"] + jnp.einsum("bij,bj->bi", c["J"], theta[c["e"]])
    v = jax.vmap(partial(_values, window=window, det_shape=det_shape))(
        mu, c["shape"], c["edges"], c["i0"], c["j0"], c["fin"], c["wfr"], c["om"]
    )
    return jnp.where(c["live"][:, None], v, 0.0)


@partial(jax.jit, static_argnames=("window", "det_shape"))
def _model_chunk(arrs: dict, i: int, theta: jax.Array, model: jax.Array, cut: float, window: tuple,
                 det_shape: tuple) -> tuple:  # fmt: skip
    """Add chunk i's model at the measured pixels, and return its censored loss.

    The censored loss counts the cells that land on no measured pixel: above the cut, the model is penalised. The
    chunk is sliced here, inside jit, with i traced: one compile for every chunk.
    """
    c = jax.tree.map(lambda a: a[i], arrs)
    v = _chunk_values(c, theta, window, det_shape)
    vm = v[c["m_inst"], c["m_cell"]]
    model = model.at[c["m_pix"]].add(vm)
    cens_all = jnp.sum(jnp.maximum(v - cut, 0.0) ** 2)
    cens_matched = jnp.sum(jnp.where(c["m_pix"] < model.shape[0] - 1, jnp.maximum(vm - cut, 0.0) ** 2, 0.0))
    return model, cens_all - cens_matched


@partial(jax.jit, static_argnames=("window", "det_shape"))
def _grad_chunk(arrs: dict, i: int, theta: jax.Array, r: jax.Array, model: jax.Array, g: jax.Array, H: jax.Array,
                cut: float, window: tuple, det_shape: tuple) -> tuple:  # fmt: skip
    """Add chunk i's gradient and surrogate curvature, per entry, with respect to the rotations theta."""
    c = jax.tree.map(lambda a: a[i], arrs)
    mu = c["mu0"] + jnp.einsum("bij,bj->bi", c["J"], theta[c["e"]])

    def vals(m, shape, edges, i0, j0, fin, wfr, om):  # noqa: ANN001, ANN202
        return _values(m, shape, edges, i0, j0, fin, wfr, om, window, det_shape)

    v, dv = jax.vmap(lambda *a: (vals(*a), jax.jacfwd(vals)(*a)))(
        mu, c["shape"], c["edges"], c["i0"], c["j0"], c["fin"], c["wfr"], c["om"]
    )  # [B, C], [B, C, 3]
    v = jnp.where(c["live"][:, None], v, 0.0)
    dv = jnp.where(c["live"][:, None, None], dv, 0.0)
    # d loss / d value per cell: 2 (v - cut) where censored and above the cut, 2 r at measured pixels
    active = v > cut
    dl = jnp.where(active, 2.0 * (v - cut), 0.0)
    wt = active.astype(v.dtype)
    real = c["m_pix"] < model.shape[0] - 1
    dl = dl.at[c["m_inst"], c["m_cell"]].set(jnp.where(real, 2.0 * r[c["m_pix"]], dl[c["m_inst"], c["m_cell"]]))
    vm = v[c["m_inst"], c["m_cell"]]
    share = jnp.clip(model[c["m_pix"]] / jnp.maximum(vm, 1e-30), 1.0, 1e3)  # the surrogate's weight
    wt = wt.at[c["m_inst"], c["m_cell"]].set(jnp.where(real, share, wt[c["m_inst"], c["m_cell"]]))
    g_mu = jnp.einsum("bck,bc->bk", dv, dl)
    H_mu = 2.0 * jnp.einsum("bck,bc,bcl->bkl", dv, wt, dv)
    g = g.at[c["e"]].add(jnp.einsum("bki,bk->bi", c["J"], g_mu))
    H = H.at[c["e"]].add(jnp.einsum("bki,bkl,blj->bij", c["J"], H_mu, c["J"]))
    return g, H


def _arrays(tab: dict) -> dict:
    """Return the tables a sweep reads, each stacked by chunk: [n_chunks, ...]."""
    keys = ("e", "mu0", "J", "shape", "edges", "i0", "j0", "fin", "wfr", "om", "live", "m_inst", "m_cell", "m_pix")
    return {k: tab[k] for k in keys}


def _loss_model(tab: dict, theta: jax.Array, cut: float, window: tuple, det_shape: tuple) -> tuple:
    model = jnp.zeros(tab["n_pix"] + 1, jnp.float32)
    cens = 0.0
    arrs = _arrays(tab)
    for i in range(tab["e"].shape[0]):
        model, c = _model_chunk(arrs, i, theta, model, cut, window, det_shape)
        cens = cens + c
    model = model.at[-1].set(0.0)
    r = model - tab["data"]
    return float(jnp.sum(r**2) + cens), r, model


def refine_orientations(
    entries: dict,
    hkls: np.ndarray,
    F2: np.ndarray,
    geom: dict,
    rows: list,
    meas: list,
    det_shape: tuple[int, int],
    window: tuple[int, int, int] = (7, 7, 7),
    n_sweeps: int = 20,
    relinearise: int = 5,
    cut: float = 1.0,
    max_step: float = 5e-3,
    chunk: int = 1 << 15,
    log: Callable | None = print,
) -> tuple[dict, list]:
    """Refine each entry's orientation against the measured pixels: linearised geometry, cheap sweeps.

    Parameters
    ----------
    entries
        "ubi" [N, 3, 3], "pos" [N, 3], "density" [N] and optionally "sig_rot" [N] (radians), held fixed but for
        the orientations
    hkls, F2, geom, det_shape
        As for :func:`anri.fwd.render_row`
    rows, meas
        Each dty row (all with the same number of frames), from :func:`anri.fwd.make_row` and :func:`measured`
    window
        (frames, slow, fast) of each peak's window: wide enough for the peaks (with their spread) to move in
    n_sweeps
        Sweeps (each one pass over the data for the gradient, and one for the trial step's loss)
    relinearise
        Accepted sweeps between linearisations (the windows re-centred, the geometry recomputed)
    cut
        The segmentation threshold of the measured pixels
    max_step
        Largest rotation of an entry in one step (radians)
    chunk
        Instances per batch
    log
        Called with a line of progress per sweep (None for silence)

    Returns
    -------
    entries: dict
        As given, with "ubi" refined
    history: list
        Per sweep: "loss", "time", "lam", "accepted" and "linearised"
    """
    ubi = np.asarray(entries["ubi"], np.float64)
    n = ubi.shape[0]
    t0 = time.perf_counter()

    def lin(u: np.ndarray) -> dict:
        t = time.perf_counter()
        tab = linearise({**entries, "ubi": u}, hkls, F2, geom, rows, meas, det_shape, window, chunk, log=log)
        if log:
            log(f"  linearised: {tab['n_inst']} instances, {int((tab['m_pix'] < tab['n_pix']).sum())} matched cells, "
                f"{time.perf_counter() - t:.1f} s")  # fmt: skip
        return tab

    def gradient(tab: dict, theta: jax.Array) -> tuple:
        loss, r, model = _loss_model(tab, theta, cut, window, det_shape)
        g, H = jnp.zeros((n, 3), jnp.float32), jnp.zeros((n, 3, 3), jnp.float32)
        arrs = _arrays(tab)
        for i in range(tab["e"].shape[0]):
            g, H = _grad_chunk(arrs, i, theta, r, model, g, H, cut, window, det_shape)
        return loss, g, H

    tab = lin(ubi)
    theta = jnp.zeros((n, 3), jnp.float32)
    loss, g, H = gradient(tab, theta)
    lam, since = 1.0, 0  # the surrogate steps of many entries at once want damping of order 1
    history = [{"loss": loss, "time": time.perf_counter() - t0, "lam": lam, "accepted": True, "linearised": True}]
    if log:
        log(f"  sweep 0: loss {loss:.5g}")
    for it in range(1, n_sweeps + 1):
        D = jnp.diagonal(H, axis1=1, axis2=2)
        has = D.sum(1) > 0
        # damping at least 1e-3 of the median entry's curvature: entries with little data take no wild steps
        D = jnp.maximum(D, 1e-3 * jnp.nanmedian(jnp.where(has[:, None], D, jnp.nan)) + 1e-30)  # fixed shape
        step = -jnp.linalg.solve(H + lam * jax.vmap(jnp.diag)(D), g[..., None])[..., 0]
        step = jnp.where(has[:, None], step, 0.0)
        step = step / jnp.maximum(jnp.linalg.norm(step, axis=1) / max_step, 1.0)[:, None]
        loss_t = _loss_model(tab, theta + step, cut, window, det_shape)[0]
        accepted = loss_t < loss
        relin = False
        if accepted:
            theta, lam, since = theta + step, max(lam / 3.0, 1e-7), since + 1
            if since >= relinearise and it < n_sweeps:  # fold the rotations in and linearise again
                ubi = ubi @ np.swapaxes(np.asarray(_rotation(theta), np.float64), -1, -2)
                tab, theta, since, relin = lin(ubi), jnp.zeros((n, 3), jnp.float32), 0, True
            loss, g, H = gradient(tab, theta)
        else:
            lam = lam * 10.0
        history.append({"loss": loss, "time": time.perf_counter() - t0, "lam": lam, "accepted": bool(accepted),
                        "linearised": relin})  # fmt: skip
        if log:
            log(f"  sweep {it}: loss {loss_t:.5g}{'' if accepted else ' (step undone)'}, lam {lam:.2g}, "
                f"{history[-1]['time']:.1f} s")  # fmt: skip
    ubi = ubi @ np.swapaxes(np.asarray(_rotation(theta), np.float64), -1, -2)
    return {**{k: np.asarray(v) for k, v in entries.items()}, "ubi": ubi}, history
