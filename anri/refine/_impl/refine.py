"""Refine a map against measured sparse pixels, using the renderer as a differentiable forward model.

Each map entry (a voxel, or one of several orientations in a voxel) has 10 parameters: the deformation gradient F,
applied to its starting UBI as UBI = UBI0 F^T (the real-space basis vectors a' = F a), and its log density.
The loss, summed over the dty rows of the scan, is

    sum over measured pixels of (model - measured)^2  +  sum over model cells off the measured pixels of relu(model - cut)^2

The model at a measured pixel sums every peak there, so voxels along the same beam path are fitted jointly.
The second term treats the segmentation cut as censoring: a model cell where nothing was measured may be up to
`cut`. Each Levenberg-Marquardt step solves the Gauss-Newton system by conjugate gradients, with matrix-free
J p and J^T u products and the 10 x 10 block of each entry as preconditioner, and a trust region per entry limits
how far it moves.

The loss is close to quadratic only within about a tenth of a peak width, so the start must be close: for sharp
peaks in 0.05 degree frames, within about 0.003 degrees.
"""

from __future__ import annotations

import inspect
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from anri.fwd._impl.render import _omega_sigma, _select_margin, render_peaks, select_peaks

N_PARAMS = 10  # F - I (9, row-major) and log density


def measured(frame: np.ndarray, pixel: np.ndarray, value: np.ndarray, n_frames: int) -> dict:
    """One row's measured sparse pixels, sorted for the refiner.

    Parameters
    ----------
    frame, pixel, value
        Measured pixels of the row: frame index (in the row's file order), pixel index slow * n_fast + fast, and
        counts, e.g. as read from an ImageD11 sparse file (only pixels above the segmentation cut)
    n_frames
        Number of frames in the row

    Returns
    -------
    measured: dict
        "pixel" and "value" sorted by (frame, pixel), and "frame_start" [n_frames + 1], where each frame's pixels start
    """
    frame, pixel = np.asarray(frame, np.int64), np.asarray(pixel, np.int64)
    order = np.lexsort((pixel, frame))
    return {
        "pixel": pixel[order].astype(np.int32),
        "value": np.asarray(value, np.float32)[order],
        "frame_start": np.searchsorted(frame[order], np.arange(n_frames + 1)).astype(np.int32),
    }


def _classes(window: tuple[int, int, int], max_frames: int) -> list[int]:
    """Frames per window size class: window[0], 2 window[0] + 1, ... up to max_frames."""
    out = [window[0]]
    while 2 * out[-1] + 1 <= max_frames:
        out.append(2 * out[-1] + 1)
    return out


def _select(ubi: jax.Array, pos: jax.Array, hkls: jax.Array, geom: dict, rows: list, det_shape: tuple[int, int],
            window: tuple[int, int, int], max_frames: int) -> list:  # fmt: skip
    """Per row: (entry, hkl, branch, window class) of the peaks that reach it."""
    classes = np.asarray(_classes(window, max_frames))
    sel = jax.jit(select_peaks, static_argnames="det_shape")
    out = []
    for row in rows:
        margin = _select_margin((int(classes[-1]), window[1], window[2]), row, geom, ubi.dtype)
        mask = np.asarray(sel(ubi, pos, hkls, geom, row, margin, det_shape=det_shape))
        e, h, b = (x.astype(np.int32) for x in np.nonzero(mask))
        k = np.zeros(e.size, np.int32)
        if e.size and len(classes) > 1:
            ostep = float(np.median(np.diff(np.asarray(row["omega_edges"]))))
            n = 1 << int(np.ceil(np.log2(e.size)))  # padded to a power of two: few compiled shapes
            ehb = [jnp.asarray(np.pad(x, (0, n - e.size))) for x in (e, h, b)]
            sig = np.asarray(_omega_sigma(ubi, pos, hkls, *ehb, geom, row))[: e.size] / ostep
            k = np.searchsorted(classes, 2 * np.ceil(3.5 * sig + 0.5) + 1)
            k = np.minimum(k, len(classes) - 1).astype(np.int32)
        out.append((e, h, b, k))
    return out


def _prepare(peaks: list, rows: list, meas: list, window: tuple[int, int, int], max_frames: int, batch: int) -> list:
    """Device arrays per row: one padded group of peaks per window class, and the measured pixels (padded)."""
    classes = _classes(window, max_frames)
    n_pad = 1 << int(np.ceil(np.log2(max(max(m["value"].size for m in meas), 1))))
    out = []
    for (e, h, b, k), row, m in zip(peaks, rows, meas):
        if e.size == 0:
            continue
        groups = []
        for c, wo in enumerate(classes):
            sel = k == c
            if not sel.any():
                continue
            size = max(1024, batch * window[0] // wo)  # bigger windows, smaller batches
            n = int(sel.sum())
            nb = 1 << int(np.ceil(np.log2(-(-n // size))))
            pad = nb * size - n
            arr = [jnp.asarray(np.pad(x[sel], (0, pad)).reshape(nb, size)) for x in (e, h, b)]
            live = jnp.asarray((np.arange(nb * size) < n).reshape(nb, size))
            groups.append(((wo, window[1], window[2]), {"e": arr[0], "h": arr[1], "b": arr[2], "live": live}))
        pix = np.full(n_pad, np.iinfo(np.int32).max, np.int32)
        pix[: m["pixel"].size] = m["pixel"]
        val = np.zeros(n_pad, np.float32)
        val[: m["value"].size] = m["value"]
        out.append({"groups": groups, "row": row, "pix": jnp.asarray(pix), "fs": jnp.asarray(m["frame_start"]),
                    "val": jnp.asarray(val)})  # fmt: skip
    return out


def _jit_window(f: Callable) -> Callable:
    """jax.jit with the last argument (the window) static."""
    return jax.jit(f, static_argnames=(list(inspect.signature(f).parameters)[-1],))


def _make_fns(hkls: jax.Array, F2: jax.Array, geom: dict, pos: jax.Array, det_shape: tuple[int, int], cut: float,
              n_search: int) -> dict:  # fmt: skip
    """Jitted per-class passes over a row: forward, gradient and blocks, J p, J^T u."""
    dtype = pos.dtype

    def peak_values(th, ubi0_e, pos_e, h, b, row, window):  # noqa: ANN001, ANN202
        F = jnp.eye(3, dtype=dtype) + th[:9].reshape(3, 3)
        one = {"ubi": (ubi0_e @ F.T)[None], "pos": pos_e[None], "density": jnp.exp(th[9])[None]}
        fr, px, val, cap = render_peaks(
            jnp.zeros(1, jnp.int32), h[None], b[None], one, hkls, F2, geom, row, window, det_shape
        )
        return fr[0], px[0], val[0], cap[0]

    def match(fr, px, pix, fs):  # noqa: ANN001, ANN202
        """Index of each (frame, pixel) among the measured pixels, -1 if absent: binary search in the frame."""
        f = jnp.clip(fr, 0, fs.shape[0] - 2)
        lo, hi = fs[f], fs[f + 1]

        def step(_, lh):  # noqa: ANN001, ANN202
            lo, hi = lh
            mid = (lo + hi) // 2
            right = pix[jnp.minimum(mid, pix.shape[0] - 1)] < px
            return jnp.where(right & (lo < hi), mid + 1, lo), jnp.where(right | (lo >= hi), hi, mid)

        lo, _ = jax.lax.fori_loop(0, n_search, step, (lo, hi))
        ok = (fr >= 0) & (lo < fs[f + 1]) & (pix[jnp.minimum(lo, pix.shape[0] - 1)] == px)
        return jnp.where(ok, lo, -1)

    def cells(theta, ubi0, e, h, b, live, row, pix, fs, window):  # noqa: ANN001, ANN202
        def values(th_b):  # noqa: ANN001, ANN202
            return jax.vmap(lambda t, u, p, hh, bb: peak_values(t, u, p, hh, bb, row, window)[2])(
                th_b, ubi0[e], pos[e], h, b
            )

        fr, px, val, cap = jax.vmap(lambda t, u, p, hh, bb: peak_values(t, u, p, hh, bb, row, window))(
            theta[e], ubi0[e], pos[e], h, b)  # fmt: skip
        idx = jnp.where(live[:, None], match(fr, px, pix, fs), -1)
        return idx, val, jnp.where(live, cap, 1.0), values

    @_jit_window
    def forward(theta, ubi0, g, row, pix, fs, model, window):  # noqa: ANN001, ANN202
        def body(model, xs):  # noqa: ANN001, ANN202
            e, h, b, live = xs
            idx, val, cap, _ = cells(theta, ubi0, e, h, b, live, row, pix, fs, window)
            model = model.at[jnp.where(idx >= 0, idx, model.shape[0])].add(val, mode="drop")
            cens = jnp.sum(jnp.where((idx < 0) & live[:, None], jnp.maximum(val - cut, 0.0), 0.0) ** 2)
            return model, (cens, cap)

        model, (cens, cap) = jax.lax.scan(body, model, (g["e"], g["h"], g["b"], g["live"]))
        return model, jnp.sum(cens), jnp.min(cap)

    @_jit_window
    def grad_block(theta, ubi0, g, row, pix, fs, r, grad, block, window):  # noqa: ANN001, ANN202
        def body(carry, xs):  # noqa: ANN001, ANN202
            grad, block = carry
            e, h, b, live = xs
            idx, val, _, _ = cells(theta, ubi0, e, h, b, live, row, pix, fs, window)
            J = jax.vmap(jax.jacfwd(lambda t, u, p, hh, bb: peak_values(t, u, p, hh, bb, row, window)[2]))(
                theta[e], ubi0[e], pos[e], h, b
            )  # fmt: skip  [B, W, 10]
            matched = idx >= 0
            res = jnp.where(matched, r[jnp.maximum(idx, 0)], jnp.maximum(val - cut, 0.0))
            J = jnp.where(((matched | (val > cut)) & live[:, None])[..., None], J, 0.0)
            grad = grad.at[e].add(jnp.einsum("bwk,bw->bk", J, res))
            block = block.at[e].add(jnp.einsum("bwk,bwl->bkl", J, J))
            return (grad, block), None

        (grad, block), _ = jax.lax.scan(body, (grad, block), (g["e"], g["h"], g["b"], g["live"]))
        return grad, block

    @_jit_window
    def jp(theta, ubi0, g, row, pix, fs, p, u, window):  # noqa: ANN001, ANN202
        def body(u, xs):  # noqa: ANN001, ANN202
            e, h, b, live = xs
            idx, _, _, values = cells(theta, ubi0, e, h, b, live, row, pix, fs, window)
            _, Jp = jax.jvp(values, (theta[e],), (p[e],))
            return u.at[jnp.where(idx >= 0, idx, u.shape[0])].add(Jp, mode="drop"), None

        u, _ = jax.lax.scan(body, u, (g["e"], g["h"], g["b"], g["live"]))
        return u

    @_jit_window
    def jtu(theta, ubi0, g, row, pix, fs, p, u, out, window):  # noqa: ANN001, ANN202
        def body(out, xs):  # noqa: ANN001, ANN202
            e, h, b, live = xs
            idx, _, _, values = cells(theta, ubi0, e, h, b, live, row, pix, fs, window)
            val, lin = jax.linearize(values, theta[e])
            Jp = lin(p[e])
            matched = idx >= 0
            c = jnp.where(matched, u[jnp.maximum(idx, 0)], jnp.where(val > cut, Jp, 0.0))
            (grad,) = jax.linear_transpose(lin, theta[e])(jnp.where(live[:, None], c, 0.0))
            return out.at[e].add(grad), None

        out, _ = jax.lax.scan(body, out, (g["e"], g["h"], g["b"], g["live"]))
        return out

    return {"forward": forward, "grad_block": grad_block, "jp": jp, "jtu": jtu}


def _residual(fns: dict, rd: dict, theta: jax.Array, ubi0: jax.Array) -> tuple:
    """Residual at the row's measured pixels, censored loss and the smallest window capture."""
    model = jnp.zeros_like(rd["val"])
    cens, cap = 0.0, 1.0
    for window, g in rd["groups"]:
        model, c, k = fns["forward"](theta, ubi0, g, rd["row"], rd["pix"], rd["fs"], model, window)
        cens, cap = cens + c, jnp.minimum(cap, k)
    return model - rd["val"], cens, cap


def _loss(fns: dict, rows: list, theta: jax.Array, ubi0: jax.Array) -> float:
    total = 0.0
    for rd in rows:
        r, cens, _ = _residual(fns, rd, theta, ubi0)
        total += float(np.sum(np.asarray(r, np.float64) ** 2)) + float(cens)
    return total


def _gradient(fns: dict, rows: list, theta: jax.Array, ubi0: jax.Array) -> tuple:
    """Loss, J^T r and the 10 x 10 blocks J^T J of each entry."""
    grad, block = jnp.zeros_like(theta), jnp.zeros(theta.shape + (N_PARAMS,), theta.dtype)
    total, cap = 0.0, 1.0
    for rd in rows:
        r, cens, k = _residual(fns, rd, theta, ubi0)
        total += float(np.sum(np.asarray(r, np.float64) ** 2)) + float(cens)
        cap = min(cap, float(k))
        for window, g in rd["groups"]:
            grad, block = fns["grad_block"](theta, ubi0, g, rd["row"], rd["pix"], rd["fs"], r, grad, block, window)
    return total, grad, block, cap


def _jtj(fns: dict, rows: list, theta: jax.Array, ubi0: jax.Array, p: jax.Array) -> jax.Array:
    """(J^T J) p: J p summed at each row's measured pixels, then J^T back."""
    out = jnp.zeros_like(p)
    for rd in rows:
        u = jnp.zeros_like(rd["val"])
        for window, g in rd["groups"]:
            u = fns["jp"](theta, ubi0, g, rd["row"], rd["pix"], rd["fs"], p, u, window)
        for window, g in rd["groups"]:
            out = fns["jtu"](theta, ubi0, g, rd["row"], rd["pix"], rd["fs"], p, u, out, window)
    return out


def _cg_step(fns: dict, rows: list, theta: jax.Array, ubi0: jax.Array, grad: jax.Array, block: jax.Array, lam: float,
             n_cg: int) -> jax.Array:  # fmt: skip
    """Solve (J^T J + lam D) d = -g by conjugate gradients, preconditioned by each entry's block."""
    D = jnp.diagonal(block, axis1=1, axis2=2)
    has = (D.sum(1) > 0)[:, None]
    D = D + 1e-12 + 1e-9 * jnp.max(D)
    Minv = jnp.linalg.inv(block + lam * jax.vmap(jnp.diag)(D))
    x = jnp.zeros_like(grad)
    r = -jnp.where(has, grad, 0.0)
    z = jnp.where(has, jnp.einsum("nkl,nl->nk", Minv, r), 0.0)
    p, rz = z, jnp.vdot(r, z)
    r0 = float(jnp.linalg.norm(r))
    for _ in range(n_cg):
        Ap = jnp.where(has, _jtj(fns, rows, theta, ubi0, p) + lam * D * p, 0.0)
        alpha = rz / jnp.vdot(p, Ap)
        x, r = x + alpha * p, r - alpha * Ap
        if float(jnp.linalg.norm(r)) < 1e-4 * r0:
            break
        z = jnp.where(has, jnp.einsum("nkl,nl->nk", Minv, r), 0.0)
        rz_new = jnp.vdot(r, z)
        p, rz = z + (rz_new / rz) * p, rz_new
    return x


def _clip(step: jax.Array, r_F: float, r_rho: float) -> jax.Array:
    """Trust region per entry: largest |dF| at most r_F, |d log density| at most r_rho."""
    sF = jnp.maximum(jnp.max(jnp.abs(step[:, :9]), axis=1) / r_F, 1.0)
    sr = jnp.maximum(jnp.abs(step[:, 9]) / r_rho, 1.0)
    return jnp.concatenate([step[:, :9] / sF[:, None], step[:, 9:] / sr[:, None]], axis=1)


def refine(
    entries: dict,
    hkls: np.ndarray,
    F2: np.ndarray,
    geom: dict,
    rows: list,
    meas: list,
    det_shape: tuple[int, int],
    n_iter: int = 10,
    window: tuple[int, int, int] = (3, 7, 7),
    max_frames: int = 31,
    cut: float = 1.0,
    n_cg: int = 15,
    batch: int = 16384,
    log: Callable | None = print,
) -> tuple[dict, list]:
    """Refine every entry's UBI and density against measured sparse pixels.

    Parameters
    ----------
    entries
        Starting map: "ubi" [N, 3, 3], "pos" [N, 3] and "density" [N], as for :func:`anri.fwd.render_row`, for one
        phase. The start must be close: the loss is near quadratic only within about a tenth of a peak width.
    hkls, F2, geom
        As for :func:`anri.fwd.render_row`. The precision of the UBIs (float32 or float64) sets the precision used.
    rows
        The scan's dty rows, from :func:`anri.fwd.make_row`
    meas
        The measured pixels of each row, from :func:`measured`
    det_shape
        (n_slow, n_fast)
    n_iter
        Levenberg-Marquardt iterations
    window, max_frames
        Window per peak, and the most frames a peak broad in omega may get, as for :func:`anri.fwd.render_row`
    cut
        The segmentation threshold of the measured pixels: a model cell where nothing was measured may be up to it
    n_cg
        Conjugate gradient iterations per step
    batch
        Peaks per batch (for the smallest windows)
    log
        Called with a line of progress per iteration (None for silence)

    Returns
    -------
    entries: dict
        The refined map: "ubi", "pos" and "density"
    history: list
        Per iteration: "loss", "time" (seconds) and "capture" (smallest fraction of a peak inside its window)
    """
    ubi0 = jnp.asarray(entries["ubi"])
    dtype = ubi0.dtype
    pos = jnp.asarray(entries["pos"], dtype)
    hkls_j, F2_j = jnp.asarray(hkls, dtype), jnp.asarray(F2, dtype)
    geom = jax.tree.map(lambda x: jnp.asarray(x, dtype) if np.asarray(x).dtype.kind == "f" else jnp.asarray(x), geom)
    rows = [{k: (float(v) if k in ("omega_min", "omega_max", "dty_min", "dty_max") else
                 jnp.asarray(v, dtype) if np.asarray(v).dtype.kind == "f" else jnp.asarray(v)) for k, v in r.items()}
            for r in rows]  # fmt: skip
    n_search = int(np.ceil(np.log2(max(int(np.diff(m["frame_start"]).max()) for m in meas) + 1))) + 1
    fns = _make_fns(hkls_j, F2_j, geom, pos, tuple(det_shape), float(cut), n_search)
    theta = jnp.zeros((ubi0.shape[0], N_PARAMS), dtype).at[:, 9].set(jnp.log(jnp.asarray(entries["density"], dtype)))
    peaks = _select(ubi0, pos, hkls_j, geom, rows, tuple(det_shape), window, max_frames)
    rd = _prepare(peaks, rows, meas, window, max_frames, batch)
    t0 = time.perf_counter()
    loss, grad, block, cap = _gradient(fns, rd, theta, ubi0)
    history = [{"loss": loss, "time": time.perf_counter() - t0, "capture": cap}]
    if log:
        log(f"iteration 0: loss {loss:.5g}")
    r_F, r_rho, lam = 1e-3, 0.1, 1e-3
    for it in range(1, n_iter + 1):
        step = _cg_step(fns, rd, theta, ubi0, grad, block, lam, n_cg)
        for _ in range(8):  # shrink the trust region until the loss drops
            trial = theta + _clip(step, r_F, r_rho)
            if _loss(fns, rd, trial, ubi0) < loss:
                break
            r_F, r_rho = r_F / 4.0, r_rho / 4.0
        else:
            if log:
                log(f"iteration {it}: no step lowers the loss; stopping")
            break
        theta = trial
        r_F, r_rho = min(r_F * 2.0, 1e-2), min(r_rho * 2.0, 1.0)
        loss, grad, block, cap = _gradient(fns, rd, theta, ubi0)
        history.append({"loss": loss, "time": time.perf_counter() - t0, "capture": cap})
        if log:
            log(f"iteration {it}: loss {loss:.5g}, {history[-1]['time']:.1f} s")
    F = np.eye(3) + np.asarray(theta[:, :9], np.float64).reshape(-1, 3, 3)
    out = {
        "ubi": np.asarray(entries["ubi"], np.float64) @ np.swapaxes(F, -1, -2),
        "pos": np.asarray(entries["pos"]),
        "density": np.exp(np.asarray(theta[:, 9], np.float64)),
    }
    return out, history
