"""The measurement grid and the frames on it.

A :class:`Frames` knows three things:

1. the grid: detector shape, and the omega (and, for scanning, dty) bin centres, as in ImageD11's
   ``ds.obincens`` / ``ds.ybincens``;
2. every recorded frame: its motor positions (``ds.omega`` / ``ds.dty``, recorded at the middle of
   the exposure) and where it came from in the master file (scan, frame);
3. how to turn peak centroids and covariances into grid units, where bin ``i`` spans
   ``[i - 1/2, i + 1/2)`` on every axis.

Frames are integrated over their own recorded position, ``omega +- omega_width/2`` and
``dty +- dty_width/2``, not over the bin they fall in. Interlaced and forth-and-back scans need
nothing special. Several frames in one bin (multi-turn scans) and bins with no frame (padded
half-scans) are both fine.

Frames are sorted by *row*, the streaming axis: dty bin for scanning, omega bin for box beam.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["Frames"]


def _uniform_step(centres: np.ndarray, name: str) -> float:
    if centres.size < 2:
        return 1.0
    steps = np.diff(centres)
    step = float(steps.mean())
    if not np.allclose(steps, step, rtol=1e-4, atol=1e-9 * abs(step)):
        msg = f"{name} bin centres are not uniformly spaced (steps {steps.min()} to {steps.max()})"
        raise ValueError(msg)
    return step


@dataclasses.dataclass(frozen=True, eq=False)
class Frames:
    """Measurement grid plus the frames recorded on it. Build with :meth:`from_arrays` or :meth:`from_dataset`."""

    det_shape: tuple[int, int]
    """(slow, fast) detector pixels."""
    omega0: float
    """First omega bin centre, degrees."""
    ostep: float
    """Omega bin spacing, degrees."""
    n_omega: int
    y0: float | None
    """First dty bin centre, or ``None`` for box beam."""
    ystep: float | None
    n_dty: int
    """Number of dty bins, 0 for box beam."""
    omega_width: float
    """Rotation during one frame, degrees. Frames integrate ``omega +- omega_width / 2``."""
    dty_width: float | None
    """Frames integrate ``dty +- dty_width / 2``. One dty step means each voxel is one step wide."""

    omega: np.ndarray
    """[F] recorded omega of each frame, in the bin convention (wrapped like ``obincens``), sorted by row."""
    dty: np.ndarray | None
    """[F] recorded dty of each frame, sorted by row. ``None`` for box beam."""
    scan: np.ndarray
    """[F] index into the dataset's scans."""
    frame: np.ndarray
    """[F] frame index within its scan."""
    row: np.ndarray
    """[F] streaming-axis bin: dty bin (scanning) or omega bin (box beam)."""
    obin: np.ndarray
    """[F] omega bin."""
    row_offsets: np.ndarray
    """[n_rows + 1] frames of row ``r`` are ``[row_offsets[r], row_offsets[r + 1])``."""
    bin_offsets: np.ndarray
    """[n_rows * n_omega + 1] (scanning) frames in (row, omega bin); unused for box beam."""

    @property
    def scanning(self) -> bool:
        return self.n_dty > 0

    @property
    def n_frames(self) -> int:
        return int(self.omega.size)

    @property
    def n_rows(self) -> int:
        return self.n_dty if self.scanning else self.n_omega

    @property
    def omega_periodic(self) -> bool:
        return abs(self.n_omega * self.ostep - 360.0) < 1e-6 * 360.0

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid size along (sc, fc, omega[, dty])."""
        return (*self.det_shape, self.n_omega) + ((self.n_dty,) if self.scanning else ())

    @property
    def periods(self) -> tuple[float, ...]:
        """Grid-unit period of each axis, 0 where not periodic."""
        om = float(self.n_omega) if self.omega_periodic else 0.0
        return (0.0, 0.0, om) + ((0.0,) if self.scanning else ())

    @classmethod
    def from_arrays(
        cls,
        det_shape: tuple[int, int],
        omega: np.ndarray,
        obincens: np.ndarray,
        dty: np.ndarray | None = None,
        ybincens: np.ndarray | None = None,
        omega_width: float | None = None,
        dty_width: float | None = None,
    ) -> Frames:
        """Build from motor positions.

        Parameters
        ----------
        det_shape
            (slow, fast) detector pixels.
        omega
            [n_scans, n_frames] or [N] recorded omega per frame, in the same convention as
            ``obincens`` (for ImageD11 multi-turn scans that is ``ds.omega_for_bins``).
        obincens
            Uniform omega bin centres, degrees.
        dty, ybincens
            Same for dty. Leave both ``None`` for box beam.
        omega_width, dty_width
            Integration width of one frame. Default: the bin spacing. For an interlaced scan binned
            at half the rotation step, pass the rotation step.
        """
        omega = np.asarray(omega, dtype=np.float64)
        shape2d = omega.shape if omega.ndim == 2 else (1, omega.size)
        obincens = np.asarray(obincens, dtype=np.float64)
        ostep = _uniform_step(obincens, "omega")
        scanning = dty is not None
        if scanning != (ybincens is not None):
            msg = "give both dty and ybincens (scanning) or neither (box beam)"
            raise ValueError(msg)

        om = omega.ravel()
        obin = np.floor((om - obincens[0]) / ostep + 0.5).astype(np.int64)
        if abs(obincens.size * ostep - 360.0) < 1e-6 * 360.0:
            obin %= obincens.size
        ok = (obin >= 0) & (obin < obincens.size)
        if scanning:
            dty = np.asarray(dty, dtype=np.float64).ravel()
            ybincens = np.asarray(ybincens, dtype=np.float64)
            ystep = _uniform_step(ybincens, "dty")
            ybin = np.floor((dty - ybincens[0]) / ystep + 0.5).astype(np.int64)
            ok &= (ybin >= 0) & (ybin < ybincens.size)
            row = ybin
        else:
            ystep = None
            row = obin
        if not ok.all():
            import warnings

            warnings.warn(f"{int((~ok).sum())} of {ok.size} frames fall outside the bins and are ignored", stacklevel=2)

        flat = np.flatnonzero(ok)
        n_rows = ybincens.size if scanning else obincens.size
        key = row[flat] * obincens.size + obin[flat]
        order = np.argsort(key, kind="stable")
        flat = flat[order]
        scan, frame = np.unravel_index(flat, shape2d)
        row_s, obin_s = row[flat], obin[flat]
        row_offsets = np.searchsorted(row_s, np.arange(n_rows + 1))
        bin_offsets = (
            np.searchsorted(key[order], np.arange(n_rows * obincens.size + 1)) if scanning else np.zeros(1, np.int64)
        )
        return cls(
            det_shape=tuple(int(v) for v in det_shape),
            omega0=float(obincens[0]),
            ostep=ostep,
            n_omega=int(obincens.size),
            y0=float(ybincens[0]) if scanning else None,
            ystep=ystep,
            n_dty=int(ybincens.size) if scanning else 0,
            omega_width=float(abs(ostep) if omega_width is None else omega_width),
            dty_width=(float(abs(ystep)) if dty_width is None else float(dty_width)) if scanning else None,
            omega=om[flat],
            dty=dty[flat] if scanning else None,
            scan=scan.astype(np.int64),
            frame=frame.astype(np.int64),
            row=row_s,
            obin=obin_s,
            row_offsets=row_offsets.astype(np.int64),
            bin_offsets=bin_offsets.astype(np.int64),
        )

    @classmethod
    def from_dataset(cls, ds, det_shape: tuple[int, int], **kwargs) -> Frames:  # noqa: ANN001, ANN003
        """Build from an ImageD11 ``DataSet`` (after ``guessbins``). ``kwargs`` go to :meth:`from_arrays`."""
        omega = getattr(ds, "omega_for_bins", None)
        omega = ds.omega if omega is None else omega
        return cls.from_arrays(det_shape, omega, ds.obincens, ds.dty, ds.ybincens, **kwargs)

    def frame_image(self) -> np.ndarray:
        """[n_omega, n_rows] position in this table of the first frame in each (omega, row) bin, -1 if none.

        Rows are dty bins (scanning) or a single row (box beam). Use ``scan[i]``, ``frame[i]`` to find
        the frame in the master file.
        """
        img = np.full((self.n_omega, self.n_dty if self.scanning else 1), -1, np.int64)
        idx = np.arange(self.n_frames)[::-1]
        img[self.obin[idx], self.row[idx] if self.scanning else 0] = idx
        return img

    # ---------------------------------------------------------------- unit conversion

    def to_grid(self, mu: jax.Array, cov: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Peak centroids [..., D] and covariances [..., D, D] from (px, px, deg[, dty units]) to grid units.

        Omega is wrapped: onto the full turn if the scan is one, else to within 180 degrees of the
        middle of the scan.
        """
        scale = [1.0, 1.0, 1.0 / self.ostep] + ([1.0 / self.ystep] if self.scanning else [])
        origin = [0.0, 0.0, self.omega0] + ([self.y0] if self.scanning else [])
        scale = jnp.asarray(scale, mu.dtype)
        om = mu[..., 2]
        if not self.omega_periodic:
            mid = self.omega0 + 0.5 * self.ostep * (self.n_omega - 1)
            om = mid + jnp.mod(om - mid + 180.0, 360.0) - 180.0
        mu = mu.at[..., 2].set(om)
        g = (mu - jnp.asarray(origin, mu.dtype)) * scale
        if not self.omega_periodic:
            return g, cov * scale[:, None] * scale[None, :]
        g = g.at[..., 2].set(jnp.mod(g[..., 2], float(self.n_omega)))
        return g, cov * scale[:, None] * scale[None, :]

    def intervals(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[N, K] grid-unit integration intervals (omega[, dty]) of frames ``idx``."""
        lo = [(self.omega[idx] - self.omega0 - 0.5 * self.omega_width) / self.ostep]
        hi = [(self.omega[idx] - self.omega0 + 0.5 * self.omega_width) / self.ostep]
        if self.scanning:
            lo.append((self.dty[idx] - self.y0 - 0.5 * self.dty_width) / self.ystep)
            hi.append((self.dty[idx] - self.y0 + 0.5 * self.dty_width) / self.ystep)
        lo, hi = np.stack(lo, -1), np.stack(hi, -1)
        swap = lo > hi  # negative steps
        return np.where(swap, hi, lo), np.where(swap, lo, hi)
