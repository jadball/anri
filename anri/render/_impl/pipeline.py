"""Streaming renderer: rows -> contributions -> batches -> pixels.

For a block of rows (dty bins, or omega bins for box beam):

1. :meth:`PeakTable.candidates` gives the peaks that reach the block.
2. The source evaluates them once on device (grid units).
3. Each peak is expanded on the host into (peak, frame) *contributions* over the frames it reaches.
4. Contributions carrying less than ``weight_cut`` of their peak are dropped, using the exact frame
   weight from :func:`~anri.render._impl.kernel.truncate`.
5. Frames are packed into batches of at most ``batch`` contributions and ``slots`` frames, and
   batches are run ``n_devices`` at a time under ``shard_map``, one batch per device.

The batch kernel is shared by sparse extraction (:func:`sparse_frames`), dense views
(:func:`render_frames`) and the image loss (:func:`loss_and_grad`).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .frames import Frames
from .kernel import flat_index, splat_peaks, truncate
from .sparse import _extract
from .table import PeakTable

try:
    from jax import shard_map as _shard_map
except ImportError:  # older jax
    from jax.experimental.shard_map import shard_map as _shard_map

__all__ = ["Measured", "loss_and_grad", "render_frames", "render_sum", "sparse_frames", "write_sparse"]


def _pow2(n: int, floor: int = 1) -> int:
    return max(floor, 1 << max(0, math.ceil(math.log2(max(n, 1)))))


@functools.cache
def _mesh() -> Mesh:
    return Mesh(np.array(jax.devices()), ("d",))


def _smap(fn: Callable, in_specs: tuple, out_specs: object) -> Callable:
    try:
        return _shard_map(fn, mesh=_mesh(), in_specs=in_specs, out_specs=out_specs)
    except TypeError:
        return _shard_map(fn, mesh=_mesh(), in_specs=in_specs, out_specs=out_specs, check_rep=False)


# ------------------------------------------------------------------------------------------ planning

_REPLICATED = PartitionSpec()
_SHARDED = PartitionSpec("d")
_COUPLING_MAX = 0.7  # detector-motor correlation above which the separable kernel is not trusted


def _replicate(*xs: jax.Array) -> tuple[jax.Array, ...]:
    return jax.jit(lambda *a: a, out_shardings=NamedSharding(_mesh(), _REPLICATED))(*xs)


@dataclasses.dataclass(frozen=True)
class Plan:
    """Contributions of one block, sorted by (frame, class, detector row)."""

    peak: np.ndarray
    """[C] index into ``ids`` / the block arrays."""
    frame: np.ndarray
    """[C] index into the frame table."""
    cls: np.ndarray
    """[C] kernel class: ``2 * window_index + correlated``."""
    ids: np.ndarray
    """[E] peak ids of the block's candidates."""
    mu: jax.Array
    """[E, D] grid-unit centroids, replicated on every device."""
    cov: jax.Array
    amp: jax.Array


@functools.lru_cache(maxsize=64)
def _eval_fn(source, frames: Frames, n_sigma: float) -> Callable:  # noqa: ANN001
    @jax.jit
    def run(params, dev_ids):  # noqa: ANN001, ANN202
        mu, cov, amp = source(params, dev_ids)
        g, gc = frames.to_grid(mu, cov)
        sd = jnp.sqrt(jnp.maximum(gc[:, 2, 2], 0.0))
        olo = jnp.floor(g[:, 2] - n_sigma * sd + 0.5).astype(jnp.int32)
        ohi = jnp.floor(g[:, 2] + n_sigma * sd + 0.5).astype(jnp.int32)
        return g, gc, amp, olo, ohi

    return run


@functools.lru_cache(maxsize=64)
def _eval_step(source, frames: Frames, n_sigma: float) -> Callable:  # noqa: ANN001
    """Sharded :func:`_eval_fn`: params replicated, ids [n_dev, n]."""
    inner = _eval_fn(source, frames, n_sigma)

    def local(params, dev_ids):  # noqa: ANN001, ANN202
        return tuple(x[None] for x in inner(params, jax.tree.map(lambda x: x[0], dev_ids)))

    return jax.jit(_smap(local, (_REPLICATED, _SHARDED), (_SHARDED,) * 5))


@functools.lru_cache(maxsize=64)
def _classify_step(frames: Frames) -> Callable:
    """Sharded: exact frame weight, brightest-pixel bound, post-frame spot size and correlations."""
    axes = (2, 3) if frames.scanning else (2,)
    periods = frames.periods

    def local(mu, cov, amp, peak, lo, hi):  # noqa: ANN001, ANN202
        m, c, a = mu[peak[0]], cov[peak[0]], amp[peak[0]]
        w, mp, p = truncate(m, c, lo[0], hi[0], axes, periods)
        sr = jnp.sqrt(jnp.maximum(p[:, 0, 0], 1e-12))
        sc = jnp.sqrt(jnp.maximum(p[:, 1, 1], 1e-12))
        rho = jnp.clip(jnp.abs(p[:, 0, 1]) / (sr * sc), 0.0, 1.0)
        pix = lambda s: jax.scipy.special.erf(0.5 / (jnp.sqrt(2.0) * s))
        bright = a * w * jnp.minimum(1.0, pix(sr) * pix(sc) / jnp.sqrt(jnp.maximum(1.0 - rho * rho, 1e-6)))
        sd = jnp.sqrt(jnp.maximum(jnp.diagonal(c, axis1=-2, axis2=-1), 1e-12))
        coupling = jnp.zeros_like(w)
        for i in (0, 1):
            for k in axes:
                coupling = jnp.maximum(coupling, jnp.abs(c[:, i, k]) / (sd[:, i] * sd[:, k]))
        out = (w, bright, jnp.maximum(sr, sc), rho, coupling, mp[:, 0])
        return tuple(x[None] for x in out)

    r, d = _REPLICATED, _SHARDED
    return jax.jit(_smap(local, (r, r, r, d, d, d), (d,) * 6))


def _expand(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(owner, rank within owner) for ``counts[i]`` items per owner."""
    owner = np.repeat(np.arange(counts.size), counts)
    starts = np.cumsum(counts) - counts
    return owner, np.arange(owner.size) - starts[owner]


def _sharded_chunks(n: int, n_dev: int, chunk: int) -> Iterator[tuple[int, int, int]]:
    """(start, stop, per-device size) covering ``n`` items, ``n_dev * size`` at a time."""
    size = _pow2(min(-(-n // n_dev), chunk), 256)
    for s in range(0, n, size * n_dev):
        yield s, min(n, s + size * n_dev), size


def plan_block(
    source,  # noqa: ANN001
    params: object,
    table: PeakTable,
    frames: Frames,
    r0: int,
    r1: int,
    weight_cut: float = 1e-4,
    pixel_cut: float = 0.0,
    windows: tuple[int, ...] = (5, 9, 15),
    win_sigma: float = 3.5,
    tolerance: float = 1e-2,
    n_sub: int = 2,
    frame_mask: np.ndarray | None = None,
    chunk: int = 2**20,
    stats: dict | None = None,
) -> Plan:
    """Contributions (peak, frame) for rows ``[r0, r1)``, culled and classified. See the module docs."""
    import time

    t0 = time.perf_counter()
    n_dev = jax.device_count()
    ids, lo, hi = table.candidates(r0, r1)
    lo, hi = np.maximum(lo, r0), np.minimum(hi, r1 - 1)
    run = _eval_step(source, frames, table.n_sigma)
    parts = []
    for s, e, per in _sharded_chunks(max(ids.size, 1), n_dev, chunk):
        chunk_ids = np.zeros(per * n_dev, np.int64)
        chunk_ids[: e - s] = ids[s:e]
        dev = jax.tree.map(lambda x, per=per: x.reshape(n_dev, per, *x.shape[1:]), source.device_ids(chunk_ids))
        parts.append(tuple(x.reshape(-1, *x.shape[2:])[: e - s] for x in run(params, dev)))
    mu, cov, amp, olo, ohi = (jnp.concatenate(p) if len(parts) > 1 else p[0] for p in zip(*parts))
    size = _pow2(ids.size, 256)  # few distinct shapes for the replicated block arrays
    pad = size - mu.shape[0]
    mu = jnp.pad(mu, ((0, pad), (0, 0)))
    cov = jnp.pad(cov, ((0, pad), (0, 0), (0, 0)))
    amp = jnp.pad(amp, (0, pad))
    olo, ohi = np.asarray(olo, np.int64)[: ids.size], np.asarray(ohi, np.int64)[: ids.size]
    mu, cov, amp = _replicate(mu, cov, amp)
    t1 = time.perf_counter()

    nrow = hi - lo + 1
    if frames.scanning:
        n_om = frames.n_omega
        if frames.omega_periodic:
            ohi = np.minimum(ohi, olo + n_om - 1)
        else:
            olo, ohi = np.maximum(olo, 0), np.minimum(ohi, n_om - 1)
        nom = np.maximum(ohi - olo + 1, 0)
        peak, k = _expand(nrow * nom)
        row = lo[peak] + k // nom[peak]
        om = (olo[peak] + k % nom[peak]) % n_om
        b = row * n_om + om
        first, count = frames.bin_offsets[b], frames.bin_offsets[b + 1] - frames.bin_offsets[b]
    else:
        peak, k = _expand(nrow)
        row = lo[peak] + k
        first, count = frames.row_offsets[row], frames.row_offsets[row + 1] - frames.row_offsets[row]
    owner, k = _expand(count)
    peak, frame = peak[owner], first[owner] + k
    if frame_mask is not None:
        keep = frame_mask[frame]
        peak, frame = peak[keep], frame[keep]
    t2 = time.perf_counter()

    step = _classify_step(frames)
    n = peak.size
    keep = np.zeros(n, bool)
    cls = np.zeros(n, np.int8)
    srow = np.zeros(n, np.float32)
    whalf = np.array([w // 2 for w in windows], float)
    for s, e, per in _sharded_chunks(n, n_dev, chunk):
        p = np.zeros(per * n_dev, np.int64)
        f = np.zeros(per * n_dev, np.int64)
        p[: e - s], f[: e - s] = peak[s:e], frame[s:e]
        flo, fhi = frames.intervals(f)
        shp = (n_dev, per)
        dt = mu.dtype
        out = step(
            mu,
            cov,
            amp,
            jnp.asarray(p.reshape(shp)),
            jnp.asarray(flo.reshape(*shp, -1), dt),
            jnp.asarray(fhi.reshape(*shp, -1), dt),
        )
        w, bright, sd, rho, coup, mrow = (np.asarray(x).reshape(-1)[: e - s] for x in out)
        keep[s:e] = (w > weight_cut) & (bright > pixel_cut)
        widx = np.minimum(np.searchsorted(whalf, win_sigma * sd), len(windows) - 1)
        if n_sub == 0:
            corr = np.zeros(e - s, bool)
        elif tolerance <= 0:
            corr = np.ones(e - s, bool)
        else:
            corr = (rho >= 5 * tolerance) | (coup >= _COUPLING_MAX)
        cls[s:e] = 2 * widx + corr
        srow[s:e] = mrow
    peak, frame, cls, srow = peak[keep], frame[keep], cls[keep], srow[keep]
    order = _frame_order(frame, cls, srow, frames.det_shape[0])
    t3 = time.perf_counter()
    if stats is not None:
        _add(stats, "candidates", ids.size)
        _add(stats, "contributions tried", n)
        _add(stats, "contributions kept", peak.size)
        for c in range(2 * len(windows)):
            _add(stats, f"kept {windows[c // 2]}px {'correlated' if c % 2 else 'separable'}", int((cls == c).sum()))
        _add(stats, "time: evaluate peaks", t1 - t0)
        _add(stats, "time: expand (host)", t2 - t1)
        _add(stats, "time: cull + classify", t3 - t2)
    return Plan(peak=peak[order], frame=frame[order], cls=cls[order], ids=ids, mu=mu, cov=cov, amp=amp)


def _frame_order(frame: np.ndarray, cls: np.ndarray, row: np.ndarray, n_det_rows: int) -> np.ndarray:
    """Order by (frame, class, detector row), in O(n) where possible.

    NumPy's stable sort is a radix sort for 16-bit keys, so two stable passes (secondary key, then
    frame) replace a comparison lexsort, which is several times slower for tens of millions of items.
    """
    if frame.size == 0:
        return np.zeros(0, np.int64)
    f0 = int(frame.min())
    span = int(frame.max()) - f0
    rbits = max(1, int(np.ceil(np.log2(max(n_det_rows, 2)))))
    if span < 2**16 and int(cls.max()) < 2 ** (16 - rbits):
        key2 = (cls.astype(np.uint16) << rbits) | np.clip(np.floor(row + 0.5), 0, n_det_rows - 1).astype(np.uint16)
        o1 = np.argsort(key2, kind="stable")
        o2 = np.argsort((frame[o1] - f0).astype(np.uint16), kind="stable")
        return o1[o2]
    return np.lexsort((row, cls, frame))


def _add(stats: dict, key: str, value: float) -> None:
    stats[key] = stats.get(key, 0) + value


def _pack(
    frame: np.ndarray, batch: int, slots: int, extra: np.ndarray | None = None, extra_batch: int = 0
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Split sorted ``frame`` into runs of whole frames: yields (frames in batch, first, end contribution).

    ``extra`` [n_frames] optionally counts something else per frame (measured pixels) that must also
    stay within ``extra_batch``; frames with only extra items are included too.
    """
    uniq = np.unique(frame)
    if extra is not None:
        uniq = np.union1d(uniq, np.flatnonzero(extra))
    start = np.searchsorted(frame, uniq, side="left")
    end = np.searchsorted(frame, uniq, side="right")
    count = end - start
    ex = extra[uniq] if extra is not None else np.zeros(uniq.size, np.int64)
    cum = np.concatenate([[0], np.cumsum(count)])
    ecum = np.concatenate([[0], np.cumsum(ex)])
    i = 0
    while i < uniq.size:
        # largest j with sum(count[i:j]) <= batch, sum(ex[i:j]) <= extra_batch, j - i <= slots
        j = min(
            np.searchsorted(cum, cum[i] + batch, side="right") - 1,
            np.searchsorted(ecum, ecum[i] + max(extra_batch, 0), side="right") - 1,
            i + slots,
            uniq.size,
        )
        if j <= i:
            msg = (
                f"frame {uniq[i]} alone needs {count[i]} contributions and {ex[i]} measured pixels, "
                f"more than batch={batch} / {extra_batch}"
            )
            raise ValueError(msg)
        yield uniq[i:j], int(start[i]), int(end[j - 1])
        i = j


# ------------------------------------------------------------------------------------- batch kernels


def _batch_blocks(mu, cov, amp, lo, hi, slot, frames: Frames, win: int, n_sub: int, n_slots: int):  # noqa: ANN001, ANN202
    blocks, starts = splat_peaks(mu, cov, amp, lo, hi, frames.shape, (0, 1), (win, win), n_sub, frames.periods)
    npix = frames.det_shape[0] * frames.det_shape[1]
    keys = slot[:, None] * npix + flat_index(starts, (win, win), frames.det_shape, (False, False))
    keys = jnp.where(slot[:, None] >= n_slots, n_slots * npix, keys)  # padding -> the dump pixel
    return keys.reshape(-1), blocks.reshape(-1)


@functools.lru_cache(maxsize=256)
def _accumulate_step(frames: Frames, win: int, n_sub: int, n_slots: int) -> Callable:
    def local(canvas, mu, cov, amp, peak, lo, hi, slot):  # noqa: ANN001, ANN202
        p = peak[0]
        keys, vals = _batch_blocks(mu[p], cov[p], amp[p], lo[0], hi[0], slot[0], frames, win, n_sub, n_slots)
        return canvas[0].at[keys].add(vals, mode="promise_in_bounds")[None], keys[None]

    r, d = _REPLICATED, _SHARDED
    return jax.jit(_smap(local, (d, r, r, r, d, d, d, d), (d, d)), donate_argnums=(0,))


@functools.lru_cache(maxsize=64)
def _extract_step(n_slots: int, npix: int, max_nnz: int) -> Callable:
    def local(canvas, keys, threshold):  # noqa: ANN001, ANN202
        out = _extract(canvas[0], keys[0], threshold[0], n_slots, npix, max_nnz)
        return tuple(x[None] for x in out)

    d = _SHARDED
    return jax.jit(_smap(local, (d, d, d), (d,) * 5), donate_argnums=(0,))


def _groups(  # noqa: ANN202
    plan: Plan,
    frames: Frames,
    batch: int,
    slots: int,
    n_dev: int,
    frame_slot: Callable[[np.ndarray], np.ndarray] | None = None,
    pad_slot: int | None = None,
):
    """Batches ``n_dev`` at a time. Yields (frames per device, {class: host arrays [n_dev, size, ...]})."""
    pad_slot = slots if pad_slot is None else pad_slot
    group = []
    for item in _pack(plan.frame, batch, slots, None, batch):
        group.append(item)
        if len(group) == n_dev:
            yield _stack_classes(plan, frames, group, pad_slot, n_dev, frame_slot)
            group = []
    if group:
        yield _stack_classes(plan, frames, group, pad_slot, n_dev, frame_slot)


def _stack_classes(plan, frames, group, pad_slot, n_dev, frame_slot):  # noqa: ANN001, ANN202
    k = 2 if frames.scanning else 1
    frames_per_dev = [g[0] for g in group] + [np.zeros(0, np.int64)] * (n_dev - len(group))
    per_class = {}
    for c in np.unique(np.concatenate([plan.cls[a:b] for _, a, b in group])):
        sel = [np.flatnonzero(plan.cls[a:b] == c) + a for _, a, b in group]
        size = _pow2(max(x.size for x in sel), 256)
        peak = np.zeros((n_dev, size), np.int64)
        slot = np.full((n_dev, size), pad_slot, np.int32)
        lo = np.zeros((n_dev, size, k))
        hi = np.zeros((n_dev, size, k))
        for dev, (idx, (fr, _, _)) in enumerate(zip(sel, group)):
            f = plan.frame[idx]
            peak[dev, : idx.size] = plan.peak[idx]
            slot[dev, : idx.size] = np.searchsorted(fr, f) if frame_slot is None else frame_slot(f)
            lo[dev, : idx.size], hi[dev, : idx.size] = frames.intervals(f)
        per_class[int(c)] = (peak, slot, lo, hi)
    return frames_per_dev, per_class


def _accumulate_group(canvas, plan, frames, per_class, windows, n_sub, n_slots, stats):  # noqa: ANN001, ANN202
    dt = plan.mu.dtype
    keys = []
    for c, (peak, slot, lo, hi) in per_class.items():
        win = windows[c // 2]
        step = _accumulate_step(frames, win, n_sub if c % 2 else 0, n_slots)
        canvas, k = step(
            canvas,
            plan.mu,
            plan.cov,
            plan.amp,
            jnp.asarray(peak),
            jnp.asarray(lo, dt),
            jnp.asarray(hi, dt),
            jnp.asarray(slot),
        )
        keys.append(k)
        if stats is not None:
            _add(stats, "window pixels", peak.shape[0] * peak.shape[1] * win * win)
    return canvas, keys


# ---------------------------------------------------------------------------------------- sparse out


def sparse_frames(
    source,  # noqa: ANN001
    table: PeakTable,
    frames: Frames,
    threshold: float,
    params: object = None,
    windows: tuple[int, ...] = (5, 9, 15),
    n_sub: int = 2,
    tolerance: float = 1e-2,
    weight_cut: float = 1e-4,
    pixel_cut: float | None = None,
    batch: int | None = None,
    slots: int | None = None,
    blocks: Sequence[tuple[int, int]] | None = None,
    stats: dict | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Render every frame and keep the pixels above ``threshold``, block by block.

    Parameters
    ----------
    source, table, frames
        What to render and where.
    threshold
        Expected counts; pixels ``<= threshold`` are dropped.
    params
        Source parameters, default ``source.params``.
    windows
        Detector window sizes (odd, px). Each contribution gets the smallest that reaches
        ``3.5`` sigma of its spot in that frame.
    n_sub
        Fidelity of the correlated kernel (see :func:`~anri.render._impl.kernel.splat_peaks`).
        0 renders everything separably.
    tolerance
        Contributions whose spot is close enough to axis-aligned, and only weakly tied to the motors,
        that the separable kernel is within about this fraction of the brightest pixel use it; the
        rest use the correlated kernel. 0 uses the correlated kernel everywhere.
    weight_cut
        Skip contributions with less than this fraction of their peak in the frame.
    pixel_cut
        Skip contributions whose brightest pixel is below this many counts. Default ``threshold / 100``.
        Many skipped contributions landing on one pixel can add up, so keep this well below threshold.
    batch
        Contributions per device call. Default: sized from device memory.
    slots
        Frames per device call. Default 64 on GPU, 4 on CPU (a small canvas stays in cache).
    blocks
        Row blocks to render, default :meth:`PeakTable.blocks`.
    stats
        If given, counts and timings are added to it.
    progress
        Called as ``progress(rows_done, rows_total)`` after each block.

    Yields
    ------
    frame, row, col, value, done
        For each block: pixels sorted by (frame, row, col), ``frame`` indexing ``frames``;
        ``done`` lists every frame of the block (including frames with no pixels).
    """
    import time

    params = source.params if params is None else params
    n_dev = jax.device_count()
    gpu = jax.devices()[0].platform != "cpu"
    slots = slots or (64 if gpu else 4)
    npix = frames.det_shape[0] * frames.det_shape[1]
    if (slots + 1) * npix >= 2**31:
        msg = f"slots={slots} too many for a {frames.det_shape} detector with int32 keys"
        raise ValueError(msg)
    pixel_cut = threshold / 100 if pixel_cut is None else pixel_cut
    batch = batch or _default_batch(max(windows), n_sub)
    nnz_cap: dict[int, int] = {}
    dtype = jax.tree.leaves(params)[0].dtype
    zeros = jax.jit(lambda: jnp.zeros((n_dev, slots * npix + 1), dtype), out_shardings=NamedSharding(_mesh(), _SHARDED))
    canvas = zeros()
    thr = jnp.full(n_dev, threshold, dtype)

    blocks = list(blocks or table.blocks())
    rows_total = sum(r1 - r0 for r0, r1 in blocks)
    rows_done = 0
    for r0, r1 in blocks:
        plan = plan_block(
            source,
            params,
            table,
            frames,
            r0,
            r1,
            weight_cut,
            pixel_cut,
            tuple(windows),
            3.5,
            tolerance,
            n_sub,
            stats=stats,
        )
        t0 = time.perf_counter()
        out = []
        for frames_per_dev, per_class in _groups(plan, frames, batch, slots, n_dev):
            while True:
                canvas, keys = _accumulate_group(canvas, plan, frames, per_class, tuple(windows), n_sub, slots, stats)
                total = sum(k.shape[1] for k in keys)
                cap = nnz_cap.setdefault(total, _pow2(total // 64, 1024))
                allk = jnp.concatenate(keys, axis=1) if len(keys) > 1 else keys[0]
                canvas, okeys, ovals, nnz, hits = _extract_step(slots, npix, cap)(canvas, allk, thr)
                nnz, hits = np.asarray(nnz), np.asarray(hits)
                if hits.max() <= cap:
                    break
                nnz_cap[total] = _pow2(int(hits.max()))  # canvas was reset; redo with room
            okeys, ovals = np.asarray(okeys).astype(np.int64), np.asarray(ovals)
            for d, fr in enumerate(frames_per_dev):
                k = okeys[d, : nnz[d]]
                sl, pix = np.divmod(k, npix)
                rr, cc = np.divmod(pix, frames.det_shape[1])
                out.append((fr[sl], rr, cc, ovals[d, : nnz[d]]))
            if stats is not None:
                _add(stats, "device calls", 1)
                _add(stats, "pixels extracted", int(nnz.sum()))
        f, r, c, v = (np.concatenate([o[i] for o in out]) if out else np.zeros(0) for i in range(4))
        order = np.lexsort((c, r, f))
        done = np.arange(frames.row_offsets[r0], frames.row_offsets[r1])
        if stats is not None:
            _add(stats, "time: render + extract", time.perf_counter() - t0)
        yield f[order].astype(np.int64), r[order].astype(np.int64), c[order].astype(np.int64), v[order], done
        rows_done += r1 - r0
        if progress is not None:
            progress(rows_done, rows_total)


def _default_batch(win: int, n_sub: int) -> int:
    import anri.backend

    info = anri.backend.check(verbose=False)
    # rough bytes per contribution in the correlated kernel at the largest window
    per = 4 * win * max(n_sub, 1) * (win + 16 * max(n_sub, 1)) * 30 + 16 * win * win
    b = int(0.25 * info.bytes_per_device / per) if info.bytes_per_device else 2**14
    return int(min(max(_pow2(b) // 2, 2**10), 2**20 if info.kind == "gpu" else 2**15))


def _counts(vals: np.ndarray, dtype: np.dtype, max_counts: float | None) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        top = float(np.iinfo(dtype).max) if max_counts is None else min(float(max_counts), np.iinfo(dtype).max)
        return np.clip(np.rint(vals), 1, top).astype(dtype)
    return (vals if max_counts is None else np.minimum(vals, max_counts)).astype(dtype)


def write_sparse(
    path: str,
    frames: Frames,
    pixels: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    scan_names: Sequence[str] | None = None,
    omega_motor: str = "rot_center",
    dty_motor: str = "dty",
    frames_per_scan: Sequence[int] | None = None,
    intensity_dtype: np.dtype = np.uint32,
    max_counts: float | None = None,
) -> None:
    """Write :func:`sparse_frames` output as an ImageD11 sparse file, one group per scan.

    Scans are written when all their frames have been rendered, frames in scan order. Frames of a
    scan that are not in ``frames`` (outside the bins) are written empty with motor position NaN.

    Parameters
    ----------
    path
        Output HDF5 file (overwritten).
    frames
        The frame table the pixels index.
    pixels
        Iterator from :func:`sparse_frames`.
    scan_names
        Group name per scan index, default ``"1.1", "2.1", ...``.
    omega_motor, dty_motor
        Motor names as the dataset expects them (``ds.omegamotor``, ``ds.dtymotor``).
    frames_per_scan
        Default: largest frame index seen per scan, plus one.
    intensity_dtype
        Integer types (the default) store expected counts rounded to the nearest count and at
        least 1, which is what ImageD11 needs: its labelling casts intensities to integers and
        rejects a peak that sums to less than one count. A float type stores them unchanged.
    max_counts
        Detector saturation: counts above this are clipped. Default: the integer type's maximum.
    """
    import h5py

    n_scans = int(frames.scan.max()) + 1 if frames.n_frames else 0
    names = list(scan_names) if scan_names is not None else [f"{i + 1}.1" for i in range(n_scans)]
    if frames_per_scan is None:
        fps = np.zeros(n_scans, np.int64)
        np.maximum.at(fps, frames.scan, frames.frame + 1)
    else:
        fps = np.asarray(frames_per_scan, np.int64)
    remaining = np.bincount(frames.scan, minlength=n_scans)
    pending: dict[int, list] = {}
    opts = {"compression": "gzip", "compression_opts": 1}

    def flush(hout, s):  # noqa: ANN001, ANN202
        parts = pending.pop(s, [])
        fidx = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.int64)
        rows = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.int64)
        cols = np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, np.int64)
        vals = np.concatenate([p[3] for p in parts]) if parts else np.zeros(0)
        local = frames.frame[fidx]
        order = np.lexsort((cols, rows, local))
        local, rows, cols, vals = local[order], rows[order], cols[order], vals[order]
        nnz = np.bincount(local, minlength=fps[s])
        mine = frames.scan == s
        omega = np.full(fps[s], np.nan)
        omega[frames.frame[mine]] = frames.omega[mine]
        g = hout.create_group(names[s])
        g.attrs["itype"] = np.dtype(np.uint16).name
        g.attrs["nframes"] = int(fps[s])
        g.attrs["shape0"], g.attrs["shape1"] = frames.det_shape
        g.create_dataset("row", data=rows.astype(np.uint16), **opts)
        g.create_dataset("col", data=cols.astype(np.uint16), **opts)
        g.create_dataset("intensity", data=_counts(vals, intensity_dtype, max_counts), **opts)
        g.create_dataset("nnz", data=nnz.astype(np.uint32))
        g.create_dataset("frame", data=local.astype(np.uint32), **opts)
        gm = g.create_group("measurement")
        gm.create_dataset(omega_motor, data=omega)
        g[omega_motor] = gm[omega_motor]
        if frames.scanning:
            dty = np.full(fps[s], np.nan)
            dty[frames.frame[mine]] = frames.dty[mine]
            gm.create_dataset(dty_motor, data=dty)
            g[dty_motor] = gm[dty_motor]
            g.create_group("instrument/positioners").create_dataset(dty_motor, data=float(np.nanmedian(dty)))

    with h5py.File(path, "w") as hout:
        for f, r, c, v, done in pixels:
            scan = frames.scan[f]
            for s in np.unique(scan):
                m = scan == s
                pending.setdefault(int(s), []).append((f[m], r[m], c[m], v[m]))
            finished = np.bincount(frames.scan[done], minlength=n_scans)
            remaining -= finished
            for s in np.flatnonzero((remaining == 0) & (finished > 0)):
                flush(hout, int(s))
        for s in range(n_scans):  # scans never reached (no frames in any block)
            if names[s] not in hout:
                flush(hout, s)


# ---------------------------------------------------------------------------------------- dense out


def render_frames(
    source,  # noqa: ANN001
    table: PeakTable,
    frames: Frames,
    views: Sequence[Sequence[int]],
    params: object = None,
    windows: tuple[int, ...] = (5, 9, 15),
    n_sub: int = 2,
    tolerance: float = 1e-2,
    weight_cut: float = 1e-6,
    batch: int | None = None,
) -> np.ndarray:
    """Dense detector images, each the sum of a set of frames.

    ``views[i]`` lists frame-table indices; ``[[f]]`` is one frame, ``np.flatnonzero(mask)`` a range.
    Views must not share frames. Returns [len(views), slow, fast]. Cost grows with the number of
    contributions, so a whole-scan sum is better done by :func:`render_sum`. Other arguments as
    :func:`sparse_frames`; nothing is cut by brightness.
    """
    params = source.params if params is None else params
    n_dev = jax.device_count()
    view_of = np.full(frames.n_frames, -1, np.int64)
    for i, v in enumerate(views):
        v = np.asarray(v, np.int64)
        if (view_of[v] >= 0).any():
            msg = "views must not share frames"
            raise ValueError(msg)
        view_of[v] = i
    mask = view_of >= 0
    slots = len(views)
    npix = frames.det_shape[0] * frames.det_shape[1]
    out = np.zeros((slots, *frames.det_shape))
    batch = batch or _default_batch(max(windows), n_sub)
    dtype = jax.tree.leaves(params)[0].dtype
    zeros = jax.jit(lambda: jnp.zeros((n_dev, slots * npix + 1), dtype), out_shardings=NamedSharding(_mesh(), _SHARDED))
    for r0 in np.unique(frames.row[mask]):
        plan = plan_block(
            source,
            params,
            table,
            frames,
            int(r0),
            int(r0) + 1,
            weight_cut,
            0.0,
            tuple(windows),
            3.5,
            tolerance,
            n_sub,
            frame_mask=mask,
        )
        for _, per_class in _groups(plan, frames, batch, 2**30, n_dev, frame_slot=lambda f: view_of[f], pad_slot=slots):
            canvas, _ = _accumulate_group(zeros(), plan, frames, per_class, tuple(windows), n_sub, slots, None)
            imgs = np.asarray(canvas)[:, : slots * npix].reshape(n_dev, slots, *frames.det_shape)
            out += imgs.sum(axis=0)
    return out


def render_sum(
    source,  # noqa: ANN001
    table: PeakTable,
    frames: Frames,
    image: str = "detector",
    omega: tuple[float, float] | None = None,
    dty: tuple[float, float] | None = None,
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
    params: object = None,
    win: int = 9,
    n_sub: int = 2,
    chunk: int = 2**16,
) -> np.ndarray:
    """One image with whole axes (or ranges of them) integrated out, straight from the peaks.

    Parameters
    ----------
    image
        ``"detector"``: [slow, fast], integrated over ``omega`` and ``dty`` (motor ranges, default
        everything). ``"sinogram"``: [n_omega, n_dty] (scanning) integrated over the detector, or
        the ``roi`` ``((slow0, slow1), (fast0, fast1))``.
    win
        Window per peak along the image axes, in bins.

    Notes
    -----
    This integrates each peak once over the whole range, rather than frame by frame, so it is much
    cheaper than summing :func:`render_frames` over many frames. It follows the bins, not the
    recorded frames, and ignores gaps between scans.
    """
    params = source.params if params is None else params
    dtype = jax.tree.leaves(params)[0].dtype
    big = 1e9
    if image == "detector":
        axes = (0, 1)
        o = (
            (-big, big)
            if omega is None
            else ((omega[0] - frames.omega0) / frames.ostep, (omega[1] - frames.omega0) / frames.ostep)
        )
        lo, hi = [min(o)], [max(o)]
        if frames.scanning:
            y = (
                (-big, big)
                if dty is None
                else ((dty[0] - frames.y0) / frames.ystep, (dty[1] - frames.y0) / frames.ystep)
            )
            lo.append(min(y))
            hi.append(max(y))
        periods = frames.periods if omega is not None else (0.0,) * len(frames.shape)
        rows = (0, table.n_rows)
        if frames.scanning and dty is not None:
            rows = (max(0, int(np.floor(min(y)))), min(table.n_rows, int(np.ceil(max(y))) + 1))
    elif image == "sinogram":
        if not frames.scanning:
            msg = "a sinogram needs a scanning geometry"
            raise ValueError(msg)
        axes = (2, 3)
        (s0, s1), (f0, f1) = roi or ((0, frames.det_shape[0]), (0, frames.det_shape[1]))
        lo, hi = [s0 - 0.5, f0 - 0.5], [s1 - 0.5, f1 - 0.5]
        periods, rows = frames.periods, (0, table.n_rows)
    else:
        msg = f"image must be 'detector' or 'sinogram', got {image!r}"
        raise ValueError(msg)

    run = _eval_fn(source, frames, table.n_sigma)
    size = (frames.shape[axes[0]], frames.shape[axes[1]])
    n_dev = jax.device_count()
    per = max(1, chunk // n_dev)
    wrap = (bool(periods[axes[0]]), bool(periods[axes[1]]))

    def local(total, params, dev_ids, n_valid):  # noqa: ANN001, ANN202
        g, gc, amp, _, _ = run(params, jax.tree.map(lambda x: x[0], dev_ids))
        amp = amp * (jnp.arange(per) < n_valid[0])
        lo_ = jnp.broadcast_to(jnp.asarray(lo, dtype), (per, len(lo)))
        hi_ = jnp.broadcast_to(jnp.asarray(hi, dtype), (per, len(hi)))
        blocks, starts = splat_peaks(g, gc, amp, lo_, hi_, frames.shape, axes, (win, win), n_sub, periods)
        idx = flat_index(starts, (win, win), size, wrap)
        return total[0].at[idx.reshape(-1)].add(blocks.reshape(-1), mode="promise_in_bounds")[None]

    d, r = _SHARDED, _REPLICATED
    add = jax.jit(_smap(local, (d, r, d, d), d), donate_argnums=(0,))
    total = jax.jit(lambda: jnp.zeros((n_dev, size[0] * size[1]), dtype), out_shardings=NamedSharding(_mesh(), d))()
    buf = np.zeros(per * n_dev, np.int64)
    fill = 0

    def flush(total, fill):  # noqa: ANN001, ANN202
        counts = np.clip(fill - np.arange(n_dev) * per, 0, per).astype(np.int32)
        dev = jax.tree.map(lambda x: x.reshape(n_dev, per, *x.shape[1:]), source.device_ids(buf))
        return add(total, params, dev, jnp.asarray(counts))

    for ids in _ids_touching(table, rows[0], rows[1], chunk, unique=not frames.scanning):
        s = 0
        while s < ids.size:
            n = min(ids.size - s, buf.size - fill)
            buf[fill : fill + n] = ids[s : s + n]
            fill += n
            s += n
            if fill == buf.size:
                total = flush(total, fill)
                fill = 0
    if fill:
        buf[fill:] = 0
        total = flush(total, fill)
    return np.asarray(total).sum(axis=0).reshape(size)


def _ids_touching(table: PeakTable, r0: int, r1: int, chunk: int, unique: bool) -> Iterator[np.ndarray]:
    """Each peak reaching rows ``[r0, r1)`` once, in chunks of at most ``chunk``.

    ``unique`` also removes duplicate ids, which only periodic box-beam tables have (a range that
    wraps round is stored as two entries).
    """
    before_ids, before_lo, _ = table.candidates(r0, r0 + 1) if r0 > 0 else (np.zeros(0, np.int64),) * 3
    extra = before_ids[(before_lo < r0) & ~np.isin(before_ids, table.wide_ids)]
    wide = (table.wide_lo < r1) & (table.wide_hi >= r0)
    extra = np.concatenate([extra, table.wide_ids[wide]])
    a, b = table.offsets[r0], table.offsets[r1]
    if unique:
        ids = np.unique(np.concatenate([extra, table.ids[a:b]]))
        for s in range(0, ids.size, chunk):
            yield ids[s : s + chunk]
        return
    for s in range(0, extra.size, chunk):
        yield extra[s : s + chunk]
    for s in range(a, b, chunk):
        yield table.ids[s : min(s + chunk, b)]


# ------------------------------------------------------------------------------------------- loss


@dataclasses.dataclass(frozen=True, eq=False)
class Measured:
    """Measured sparse pixels per frame of a :class:`Frames` table (CSR)."""

    offsets: np.ndarray
    """[n_frames + 1]"""
    pixel: np.ndarray
    """[M] int64 ``row * fast + col``"""
    value: np.ndarray
    """[M] counts"""

    @classmethod
    def from_arrays(
        cls, frames: Frames, frame: np.ndarray, row: np.ndarray, col: np.ndarray, value: np.ndarray
    ) -> Measured:
        order = np.argsort(frame, kind="stable")
        frame = np.asarray(frame)[order]
        offsets = np.searchsorted(frame, np.arange(frames.n_frames + 1))
        pix = np.asarray(row)[order].astype(np.int64) * frames.det_shape[1] + np.asarray(col)[order]
        return cls(offsets=offsets, pixel=pix, value=np.asarray(value, np.float64)[order])

    @classmethod
    def from_sparse_file(cls, path: str, frames: Frames, scan_names: Sequence[str] | None = None) -> Measured:
        """Read an ImageD11 sparse file whose scans are, in order, the scan indices of ``frames``."""
        import h5py

        n_scans = int(frames.scan.max()) + 1
        names = list(scan_names) if scan_names is not None else [f"{i + 1}.1" for i in range(n_scans)]
        lookup = {}
        for i in range(frames.n_frames):
            lookup[(int(frames.scan[i]), int(frames.frame[i]))] = i
        fs, rs, cs, vs = [], [], [], []
        with h5py.File(path, "r") as hin:
            for s, name in enumerate(names):
                g = hin[name]
                nnz = g["nnz"][:]
                local = np.repeat(np.arange(nnz.size), nnz)
                table_idx = np.array([lookup.get((s, int(f)), -1) for f in range(nnz.size)])[local]
                ok = table_idx >= 0
                fs.append(table_idx[ok])
                rs.append(g["row"][:][ok])
                cs.append(g["col"][:][ok])
                vs.append(g["intensity"][:][ok])
        return cls.from_arrays(frames, np.concatenate(fs), np.concatenate(rs), np.concatenate(cs), np.concatenate(vs))

    def count(self) -> np.ndarray:
        return np.diff(self.offsets)


def _put(x: np.ndarray, dtype: object = None) -> jax.Array:
    return jnp.asarray(x, dtype)


def _batches(  # noqa: ANN202
    plan: Plan,
    frames: Frames,
    batch: int,
    slots: int,
    n_dev: int,
    extra: np.ndarray | None = None,
    extra_batch: int = 0,
):
    """Loss batches ``n_dev`` at a time, as padded host arrays [n_dev, size, ...] (one kernel class)."""
    group = []
    for item in _pack(plan.frame, batch, slots, extra, extra_batch if extra is not None else batch):
        group.append(item)
        if len(group) == n_dev:
            yield _stack_one(plan, frames, group, slots, n_dev)
            group = []
    if group:
        yield _stack_one(plan, frames, group, slots, n_dev)


def _stack_one(plan, frames, group, slots, n_dev):  # noqa: ANN001, ANN202
    k = 2 if frames.scanning else 1
    size = _pow2(max(b - a for _, a, b in group), 256)
    peak = np.zeros((n_dev, size), np.int64)
    slot = np.full((n_dev, size), slots, np.int32)
    lo = np.zeros((n_dev, size, k))
    hi = np.zeros((n_dev, size, k))
    out_frames = []
    for d, (fr, a, b) in enumerate(group):
        f = plan.frame[a:b]
        peak[d, : b - a] = plan.peak[a:b]
        slot[d, : b - a] = np.searchsorted(fr, f)
        lo[d, : b - a], hi[d, : b - a] = frames.intervals(f)
        out_frames.append(fr)
    out_frames += [np.zeros(0, np.int64)] * (n_dev - len(group))
    return peak, slot, lo, hi, out_frames


def default_pixel_loss(model: jax.Array, data: jax.Array, measured: jax.Array, threshold: jax.Array) -> jax.Array:
    """Squared residual where something was measured; squared excess over threshold where not."""
    return jnp.where(measured, (model - data) ** 2, jnp.maximum(model - threshold, 0.0) ** 2)


@functools.lru_cache(maxsize=64)
def _loss_step(source, frames: Frames, win: int, n_sub: int, n_slots: int, pixel_loss: Callable) -> Callable:  # noqa: ANN001
    npix = frames.det_shape[0] * frames.det_shape[1]
    ncanvas = n_slots * npix + 1

    def batch_loss(params, dev_ids, lo, hi, slot, mkey, mval, threshold):  # noqa: ANN001, ANN202
        mu, cov, amp = source(params, dev_ids)
        g, gc = frames.to_grid(mu, cov)
        keys, vals = _batch_blocks(g, gc, amp, lo, hi, slot, frames, win, n_sub, n_slots)
        model = jnp.zeros(ncanvas, vals.dtype).at[keys].add(vals, mode="promise_in_bounds")
        data = jnp.zeros(ncanvas, vals.dtype).at[mkey].set(mval, mode="promise_in_bounds")
        seen = jnp.zeros(ncanvas, bool).at[mkey].set(mval > 0, mode="promise_in_bounds")
        allk = jnp.concatenate([keys, mkey])
        pos = jnp.arange(allk.size, dtype=jnp.int32)
        owner = jnp.full(ncanvas, -1, jnp.int32).at[allk].max(pos, mode="promise_in_bounds")
        rep = (owner[allk] == pos) & (allk < n_slots * npix)
        per = pixel_loss(model[allk], data[allk], seen[allk], threshold)
        return jnp.sum(jnp.where(rep, per, 0.0))

    def local(params, dev_ids, lo, hi, slot, mkey, mval, threshold):  # noqa: ANN001, ANN202
        sq = lambda t: jax.tree.map(lambda x: x[0], t)
        val = batch_loss(params, sq(dev_ids), lo[0], hi[0], slot[0], mkey[0], mval[0], threshold[0])
        return jax.lax.psum(val, "d")

    d, r = PartitionSpec("d"), PartitionSpec()
    # differentiate outside shard_map: its transpose sums the cotangents of the replicated params
    return jax.jit(jax.value_and_grad(_smap(local, (r, d, d, d, d, d, d, d), r)))


def loss_and_grad(
    params: object,
    source,  # noqa: ANN001
    table: PeakTable,
    frames: Frames,
    measured: Measured,
    threshold: float,
    pixel_loss: Callable = default_pixel_loss,
    win: int = 9,
    n_sub: int = 2,
    weight_cut: float = 1e-4,
    batch: int | None = None,
    slots: int = 8,
    blocks: Sequence[tuple[int, int]] | None = None,
) -> tuple[float, object]:
    """Image loss between the model and measured sparse frames, and its gradient with respect to ``params``.

    For every frame, the loss runs over the union of measured pixels and pixels the model lights up:
    ``pixel_loss(model, data, measured, threshold)`` summed. The default penalises residuals where
    something was measured and model intensity above ``threshold`` where nothing was, so both
    missing and extra intensity cost.

    Which (peak, frame) pairs contribute is decided from ``params`` without gradient; the values
    and the loss are differentiable. ``table`` should be built with a margin (``n_sigma``) that
    covers how far peaks may move during refinement.

    Parameters
    ----------
    pixel_loss
        ``(model, data, measured, threshold) -> per-pixel loss``, elementwise and jittable.
    blocks
        Row blocks to include, default all. A subset gives a stochastic estimate.

    Returns
    -------
    loss: float
    grad: same structure as ``params``
    """
    n_dev = jax.device_count()
    npix = frames.det_shape[0] * frames.det_shape[1]
    batch = batch or _default_batch(win, n_sub)
    mcount = measured.count()
    mbatch = _pow2(max(int(mcount.max()) if mcount.size else 1, batch * win * win // 4))  # packing limit
    dtype = jax.tree.leaves(params)[0].dtype
    step = _loss_step(source, frames, win, n_sub, slots, pixel_loss)
    total, grad = 0.0, jax.tree.map(jnp.zeros_like, params)
    thr = jnp.full(n_dev, threshold, dtype)
    for r0, r1 in blocks or table.blocks():
        plan = plan_block(source, params, table, frames, r0, r1, weight_cut, 0.0, (win,), 3.5, 0.0, n_sub)
        in_block = np.zeros(frames.n_frames, np.int64)
        sl = slice(frames.row_offsets[r0], frames.row_offsets[r1])
        in_block[sl] = mcount[sl]
        for peak, slot, lo, hi, bframes in _batches(
            plan, frames, batch, slots, n_dev, extra=in_block, extra_batch=mbatch
        ):
            need = max(int(mcount[fr].sum()) if len(fr) else 0 for fr in bframes)
            msize = min(mbatch, _pow2(need, 256))
            mkey = np.full((n_dev, msize), slots * npix, np.int64)  # padding -> the dump pixel
            mval = np.zeros((n_dev, msize))
            for d, fr in enumerate(bframes):
                idx = (
                    np.concatenate([np.arange(measured.offsets[f], measured.offsets[f + 1]) for f in fr])
                    if len(fr)
                    else np.zeros(0, np.int64)
                )
                sl_of = np.repeat(np.arange(len(fr)), mcount[fr]) if len(fr) else np.zeros(0, np.int64)
                mkey[d, : idx.size] = sl_of * npix + measured.pixel[idx]
                mval[d, : idx.size] = measured.value[idx]
            dev_ids = jax.tree.map(
                lambda x, shape=peak.shape: x.reshape(*shape, *x.shape[1:]),
                source.device_ids(plan.ids[peak.reshape(-1)]),
            )
            val, g = step(
                params,
                dev_ids,
                _put(lo, dtype),
                _put(hi, dtype),
                _put(slot),
                _put(mkey, jnp.int32),
                _put(mval, dtype),
                thr,
            )
            total += float(val)
            grad = jax.tree.map(jnp.add, grad, g)
    return total, grad
