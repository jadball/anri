"""Where peaks come from, and which peaks touch which rows.

Nothing here holds peak centroids or covariances for the whole sample: at 5e9 peaks that is ~440 GB.
Peaks are named by an int64 id and recomputed from the forward model whenever they are needed.

A *source* is any object with

``n_peaks``
    ids are ``0 .. n_peaks - 1``
``ndim``
    4 for scanning ``(sc, fc, omega, dty)``, 3 for box beam ``(sc, fc, omega)``
``params``
    default parameters: a pytree, the thing an optimiser refines
``device_ids(ids)``
    host int64 ids -> a pytree of device arrays (int32 or float) of the same length
``__call__(params, dev_ids)``
    -> ``mu`` [N,ndim] (px, px, deg[, dty units]), ``cov`` [N,ndim,ndim], ``amp`` [N].
    Peaks that do not exist (no diffraction solution) must have ``amp == 0``. Jittable.

:class:`PeakSource` is that object for anri's forward model over voxels (or grains) and reflections.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .frames import Frames

__all__ = ["PeakSource", "PeakTable"]


class PeakSource:
    """anri forward model for every (voxel, reflection, omega solution).

    Ids are ``(voxel * n_hkl + hkl) * 2 + branch``, branch 0 for ``etasign = +1`` and 1 for ``-1``.

    Parameters
    ----------
    ubi
        [V,3,3] UBI of each voxel or grain. Part of ``params``.
    origin
        [V,3] position in the sample frame, same units as dty. Part of ``params``.
    weight
        [V] scattering volume (or map intensity) of each voxel. Part of ``params``.
    hkl
        [H,3] reflections.
    intensity
        [H] intensity of each reflection. Fold Lorentz and polarisation in here.
    geometry
        Keyword arguments of :func:`anri.fwd.get_centroid_scan` other than ``ubi``, ``origin_sample``,
        ``hkl`` and ``etasign``: ``wavelength, k_in_lab, ky, kz, wedge, chi, y0, sc_lab, fc_lab, norm_lab``.
        Leave out ``y0`` for box beam (:func:`anri.fwd.get_centroid_box`).
    cov_in
        [6,6] input covariance, from :func:`anri.fwd.get_cov_in`.
    extra_var
        [ndim] variance added to every peak, e.g. a detector point spread in px^2. Do *not* add the
        bin widths here: frames and pixels are already integrated over their extent.
    omegasign
        +1 or -1: sign relating anri's omega to the rotation motor (ImageD11 ``omegasign``).
    dtype
        Working precision. float32 is the fast choice.
    """

    def __init__(
        self,
        ubi: np.ndarray,
        origin: np.ndarray,
        weight: np.ndarray,
        hkl: np.ndarray,
        intensity: np.ndarray,
        geometry: dict,
        cov_in: np.ndarray,
        extra_var: np.ndarray | None = None,
        omegasign: float = 1.0,
        dtype: jax.typing.DTypeLike = jnp.float32,
    ) -> None:
        import anri.fwd

        self.scanning = "y0" in geometry
        self.ndim = 4 if self.scanning else 3
        self.n_voxels = int(np.shape(ubi)[0])
        self.n_hkl = int(np.shape(hkl)[0])
        self.n_peaks = self.n_voxels * self.n_hkl * 2
        cast = lambda x: jnp.asarray(x, dtype)
        self.params = {"ubi": cast(ubi), "origin": cast(origin), "weight": cast(weight)}
        self.hkl = cast(hkl)
        self.intensity = cast(intensity)
        self.cov_in = cast(cov_in)
        self.extra_var = cast(np.zeros(self.ndim) if extra_var is None else extra_var)
        if omegasign not in (1, -1):
            msg = f"omegasign must be +1 or -1, got {omegasign}"
            raise ValueError(msg)
        self.axis_sign = cast(np.array([1.0, 1.0, omegasign] + [1.0] * (self.ndim - 3)))
        names = ("wavelength", "k_in_lab", "ky", "kz", "wedge", "chi") + (("y0",) if self.scanning else ())
        names += ("sc_lab", "fc_lab", "norm_lab")
        missing = set(names) - set(geometry)
        if missing:
            msg = f"geometry is missing {sorted(missing)}"
            raise ValueError(msg)
        self.geometry = tuple(cast(geometry[k]) for k in names)
        if self.scanning:
            self._centroid, self._cov = anri.fwd.get_centroid_scan, anri.fwd.propagate_cov_scan
        else:
            self._centroid, self._cov = anri.fwd.get_centroid_box, anri.fwd.propagate_cov_box

    def device_ids(self, ids: np.ndarray) -> tuple[jax.Array, jax.Array, jax.Array]:
        """(voxel, hkl) int32 and etasign float, for host int64 ``ids``."""
        ids = np.asarray(ids, np.int64)
        branch = ids % 2
        rest = ids // 2
        return (
            jnp.asarray((rest // self.n_hkl).astype(np.int32)),
            jnp.asarray((rest % self.n_hkl).astype(np.int32)),
            jnp.asarray(np.where(branch == 0, 1.0, -1.0), self.hkl.dtype),
        )

    def __call__(self, params: dict, dev_ids: tuple) -> tuple[jax.Array, jax.Array, jax.Array]:
        vox, h, eta = dev_ids
        ubi, origin = params["ubi"][vox], params["origin"][vox]
        hkl = self.hkl[h]
        geo = self.geometry

        def one(u, o, k, e):  # noqa: ANN001, ANN202
            c, valid = self._centroid(u, o, k, e, *geo)
            s = self._cov(u, o, k, e, *geo, self.cov_in)
            return c, s, valid

        mu, cov, valid = jax.vmap(one)(ubi, origin, hkl, eta)
        mu = mu * self.axis_sign
        cov = cov * self.axis_sign[:, None] * self.axis_sign[None, :] + jnp.diag(self.extra_var)
        ok = valid & jnp.isfinite(mu).all(-1) & jnp.isfinite(cov).all((-1, -2))
        amp = jnp.where(ok, self.intensity[h] * params["weight"][vox], 0.0)
        mu = jnp.where(ok[:, None], mu, 0.0)
        cov = jnp.where(ok[:, None, None], cov, jnp.eye(self.ndim, dtype=cov.dtype))
        return mu, cov, amp


def _supports(source, frames: Frames, n_sigma: float, win: int) -> Callable:  # noqa: ANN001
    """Sharded: params (replicated), dev_ids [n_dev, n] -> (row_lo, row_hi, keep) [n_dev, n]."""
    from jax.sharding import Mesh, PartitionSpec

    try:
        from jax import shard_map
    except ImportError:  # older jax
        from jax.experimental.shard_map import shard_map

    row_axis = 3 if frames.scanning else 2
    s, f = frames.det_shape

    def local(params, dev_ids):  # noqa: ANN001, ANN202
        mu, cov, amp = source(params, jax.tree.map(lambda x: x[0], dev_ids))
        g, gc = frames.to_grid(mu, cov)
        sd = jnp.sqrt(jnp.maximum(jnp.diagonal(gc, axis1=-2, axis2=-1), 0.0))
        reach = n_sigma * sd
        centre = g[:, row_axis]
        lo = jnp.floor(centre - reach[:, row_axis] + 0.5).astype(jnp.int32)
        hi = jnp.floor(centre + reach[:, row_axis] + 0.5).astype(jnp.int32)
        half = win / 2
        on_det = (
            (g[:, 0] + reach[:, 0] > -half)
            & (g[:, 0] - reach[:, 0] < s - 1 + half)
            & (g[:, 1] + reach[:, 1] > -half)
            & (g[:, 1] - reach[:, 1] < f - 1 + half)
        )
        keep = (amp > 0) & on_det & jnp.isfinite(g).all(-1)
        return lo[None], hi[None], keep[None]

    mesh = Mesh(np.array(jax.devices()), ("d",))
    d, r = PartitionSpec("d"), PartitionSpec()
    try:
        fn = shard_map(local, mesh=mesh, in_specs=(r, d), out_specs=(d, d, d))
    except TypeError:
        fn = shard_map(local, mesh=mesh, in_specs=(r, d), out_specs=(d, d, d), check_rep=False)
    return jax.jit(fn)


@dataclasses.dataclass(frozen=True, eq=False)
class PeakTable:
    """Peak ids indexed by the rows (dty bins, or omega bins for box beam) they touch.

    Entries are sorted by first row. Entries spanning more than ``max_span`` rows are kept apart, so
    that finding the peaks for a range of rows is one contiguous slice plus a short list. Build with
    :meth:`build`.
    """

    ids: np.ndarray
    """[E] int64 peak ids, sorted by first row."""
    span: np.ndarray
    """[E] uint16 rows touched minus one."""
    offsets: np.ndarray
    """[n_rows + 1] entries whose first row is ``r`` are ``[offsets[r], offsets[r + 1])``."""
    max_span: int
    wide_ids: np.ndarray
    wide_lo: np.ndarray
    wide_hi: np.ndarray
    n_rows: int
    n_sigma: float

    @property
    def n_entries(self) -> int:
        return int(self.ids.size + self.wide_ids.size)

    @classmethod
    def build(
        cls,
        source,  # noqa: ANN001
        frames: Frames,
        params: object = None,
        n_sigma: float = 4.0,
        win: int = 9,
        chunk: int = 2**20,
        max_span: int = 64,
        progress: Callable[[int, int], None] | None = None,
    ) -> PeakTable:
        """Evaluate every peak once and record which rows it reaches.

        Parameters
        ----------
        source
            Peak source (see module docs).
        frames
            The measurement grid.
        params
            Source parameters, default ``source.params``.
        n_sigma
            A peak reaches ``n_sigma`` marginal standard deviations along the row axis.
        win
            Detector window size, so peaks just off the detector still count.
        chunk
            Peaks per call, split across all devices.
        max_span
            Entries reaching more rows than this are kept in a separate list.
        progress
            Called as ``progress(done, total)`` after each chunk.

        Notes
        -----
        Host memory peaks at about 20 bytes per kept peak (ids, first row and span held per chunk,
        then placed by counting sort). Device work is one centroid and covariance per peak.
        """
        params = source.params if params is None else params
        run = _supports(source, frames, n_sigma, win)
        n_rows = frames.n_rows
        periodic = (not frames.scanning) and frames.omega_periodic
        row_dtype = np.int16 if n_rows < 2**15 else np.int32
        max_span = min(int(max_span), 2**16 - 1)

        chunks = []
        counts = np.zeros(n_rows, np.int64)
        wide = []
        n_dev = jax.device_count()
        per = max(1, -(-chunk // n_dev))
        chunk = per * n_dev
        for start in range(0, source.n_peaks, chunk):
            stop = min(start + chunk, source.n_peaks)
            ids = np.arange(start, start + chunk, dtype=np.int64)
            ids[stop - start :] = 0  # pad the last chunk to keep one compiled shape
            dev = jax.tree.map(lambda x, per=per: x.reshape(n_dev, per, *x.shape[1:]), source.device_ids(ids))
            lo, hi, keep = (np.asarray(a).reshape(-1) for a in run(params, dev))
            keep = keep.copy()
            keep[stop - start :] = False
            ids, lo, hi = ids[keep], lo[keep].astype(np.int64), hi[keep].astype(np.int64)
            if periodic:
                full = hi - lo + 1 >= n_rows
                lo = np.where(full, 0, lo)
                hi = np.where(full, n_rows - 1, hi)
                shift = np.floor_divide(lo, n_rows) * n_rows
                lo, hi = lo - shift, hi - shift
                wrap = hi >= n_rows
                # split wrapped ranges into [lo, n-1] and [0, hi-n]
                ids = np.concatenate([ids, ids[wrap]])
                lo = np.concatenate([lo, np.zeros(int(wrap.sum()), np.int64)])
                hi = np.concatenate([np.where(wrap, n_rows - 1, hi), hi[wrap] - n_rows])
            else:
                inside = (hi >= 0) & (lo < n_rows)
                ids, lo, hi = ids[inside], np.clip(lo[inside], 0, n_rows - 1), np.clip(hi[inside], 0, n_rows - 1)
            is_wide = hi - lo > max_span
            if is_wide.any():
                wide.append((ids[is_wide], lo[is_wide], hi[is_wide]))
            ids, lo, hi = ids[~is_wide], lo[~is_wide], hi[~is_wide]
            chunks.append((ids, lo.astype(row_dtype), (hi - lo).astype(np.uint16)))
            counts += np.bincount(lo, minlength=n_rows)
            if progress is not None:
                progress(stop, source.n_peaks)

        offsets = np.zeros(n_rows + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        out_ids = np.empty(offsets[-1], np.int64)
        out_span = np.empty(offsets[-1], np.uint16)
        cursor = offsets[:-1].copy()
        while chunks:  # counting sort, freeing each chunk as it is placed
            ids, lo, span = chunks.pop(0)
            order = np.argsort(lo, kind="stable")
            lo = lo[order].astype(np.int64)
            c = np.bincount(lo, minlength=n_rows)
            first = np.concatenate([[0], np.cumsum(c)[:-1]])
            pos = cursor[lo] + (np.arange(lo.size) - first[lo])
            out_ids[pos] = ids[order]
            out_span[pos] = span[order]
            cursor += c

        cat = lambda k, dt: np.concatenate([w[k] for w in wide]) if wide else np.zeros(0, dt)
        return cls(
            ids=out_ids,
            span=out_span,
            offsets=offsets,
            max_span=max_span,
            wide_ids=cat(0, np.int64),
            wide_lo=cat(1, np.int64),
            wide_hi=cat(2, np.int64),
            n_rows=n_rows,
            n_sigma=float(n_sigma),
        )

    def candidates(self, r0: int, r1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Entries touching rows ``[r0, r1)``: (ids, first row, last row)."""
        first_row = max(0, r0 - self.max_span)
        a, b = self.offsets[first_row], self.offsets[min(r1, self.n_rows)]
        lo = np.repeat(
            np.arange(first_row, min(r1, self.n_rows)), np.diff(self.offsets[first_row : min(r1, self.n_rows) + 1])
        )
        hi = lo + self.span[a:b]
        m = hi >= r0
        wm = (self.wide_lo < r1) & (self.wide_hi >= r0)
        return (
            np.concatenate([self.ids[a:b][m], self.wide_ids[wm]]),
            np.concatenate([lo[m], self.wide_lo[wm]]),
            np.concatenate([hi[m].astype(np.int64), self.wide_hi[wm]]),
        )

    def blocks(self, rows_per_block: int | None = None, max_entries: int = 2**23) -> list[tuple[int, int]]:
        """Split the rows into consecutive blocks of at most ``max_entries`` candidates (or fixed size)."""
        if rows_per_block is not None:
            return [(r, min(r + rows_per_block, self.n_rows)) for r in range(0, self.n_rows, rows_per_block)]
        per_row = np.diff(self.offsets) * (1 + math.ceil(np.mean(self.span)) if self.span.size else 1)
        out, start, acc = [], 0, 0
        for r in range(self.n_rows):
            if acc and acc + per_row[r] > max_entries:
                out.append((start, r))
                start, acc = r, 0
            acc += per_row[r]
        out.append((start, self.n_rows))
        return out
