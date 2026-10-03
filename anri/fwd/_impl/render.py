"""Render peaks from a voxel map into sparse detector pixels, one scanning-3DXRD dty row at a time.

A map is a flat list of entries (position, UBI, density), so several orientations per voxel are allowed.
Each peak is an (entry, hkl, branch) triple. Its 4D centroid and covariance come from
:func:`anri.fwd.get_centroid_scan` and :func:`anri.fwd._impl.base.make_propagator`.

Pixel values are differentiable with respect to the map. Which pixels a peak touches is not:
window placement is computed from rounded centroids.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr
from jax.scipy.stats import norm
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike

try:
    from jax import shard_map
except ImportError:  # JAX < 0.6, e.g. 0.4.30, the last release for Python 3.9
    from jax.experimental.shard_map import shard_map

import anri.utils
from anri.geom import sample_to_lab

from .base import get_cov_in, hkl_to_k_omega, make_propagator
from .scan import get_centroid_scan, get_centroid_scan_both

# Covariance of (sc, fc, omega) from wavelength, ky and kz spreads (dims 3, 4, 5 of argnums).
# Origin spread (dims 0-2) is not propagated: the voxel's extent and the beam profile enter through dty_weight.
_COV_ELEMS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
_propagate = make_propagator(
    get_centroid_scan, argnums=(1, 4, 6, 7), has_aux=True, active_dims=(3, 4, 5), out_elems=_COV_ELEMS
)

# Added to the variances so that a peak with no instrumental broadening still renders (pixels, pixels, degrees).
_VAR_FLOOR = (1e-4, 1e-4, 1e-8)


def bin_fractions(lo: ArrayLike, hi: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> jax.Array:
    """Fraction of a 1D Gaussian N(mu, sigma^2) that lies in [lo, hi)."""
    lo, hi, mu, sigma = (jnp.asarray(x) for x in (lo, hi, mu, sigma))
    return ndtr((hi - mu) / sigma) - ndtr((lo - mu) / sigma)


def _ramp_integral(x: ArrayLike, sigma: ArrayLike) -> jax.Array:
    """Antiderivative of ndtr(x / sigma): x * Phi(x / sigma) + sigma * phi(x / sigma)."""
    x, sigma = jnp.asarray(x), jnp.asarray(sigma)
    return x * ndtr(x / sigma) + sigma * jnp.asarray(norm.pdf(x / sigma))


def dty_weight(delta: ArrayLike, omega: ArrayLike, voxel_size: ArrayLike, sig_beam: ArrayLike) -> jax.Array:
    """Diffracting area of a square voxel when the beam centre is ``delta`` away from the voxel centre.

    The beam is Gaussian across lab y with standard deviation ``sig_beam``.
    The voxel is a uniform square of side ``voxel_size`` in the sample xy plane, rotated by ``omega`` (degrees).
    Its chord length along the beam, as a function of lab y, is a trapezoid of area ``voxel_size**2``;
    the weight is that trapezoid convolved with the beam profile, in closed form.
    Integrated over ``delta`` it gives ``voxel_size**2``.

    Parameters
    ----------
    delta
        dty of the frame minus the dty that centres the voxel in the beam (same units as voxel_size)
    omega
        Omega angle (degrees)
    voxel_size
        Side length of the voxel
    sig_beam
        Standard deviation of the beam profile across lab y
    """
    c = jnp.abs(jnp.cos(jnp.radians(omega)))
    s = jnp.abs(jnp.sin(jnp.radians(omega)))
    a = 0.5 * voxel_size * (c + s)  # half-width of the trapezoid at its base
    b = 0.5 * voxel_size * jnp.abs(c - s)  # half-width at its top
    height = voxel_size**2 / (a + b)

    # Near omega = 0, 90, ... the trapezoid becomes a rectangle and (a - b) -> 0.
    # Double-where so neither branch produces NaN gradients.
    is_box = (a - b) < 1e-6 * voxel_size
    ab = jnp.where(is_box, 1.0, a - b)
    trapezoid = (
        height
        / ab
        * (
            _ramp_integral(delta + a, sig_beam)
            - _ramp_integral(delta + b, sig_beam)
            - _ramp_integral(delta - b, sig_beam)
            + _ramp_integral(delta - a, sig_beam)
        )
    )
    box = height * bin_fractions(-a, a, delta, sig_beam)
    return jnp.where(is_box, box, trapezoid)


def lorentz(k_in: jax.Array, k_out: jax.Array, rot_axis: jax.Array) -> jax.Array:
    """Lorentz factor for rotation about ``rot_axis``: 1 / |rot_axis . (k_in x k_out)|.

    Equal to 1 / :func:`ImageD11.refinegrains.lf` = 1 / (sin(2theta) |sin(eta)|) for a vertical rotation axis.
    """
    k_in = k_in / jnp.linalg.norm(k_in)
    k_out = k_out / jnp.linalg.norm(k_out)
    return 1.0 / jnp.abs(rot_axis @ jnp.cross(k_in, k_out))


def polarisation(k_out: jax.Array, factor: ArrayLike) -> jax.Array:
    """Polarisation factor for a beam polarised along lab y with degree of polarisation ``factor``.

    Equal to :func:`ImageD11.refinegrains.polarization` with ``eta0 = 0``.
    ``factor = 1`` is fully horizontally polarised, ``factor = 0`` is unpolarised.
    """
    k_out = k_out / jnp.linalg.norm(k_out)
    return 0.5 * (1 + factor) * (1 - k_out[1] ** 2) + 0.5 * (1 - factor) * (1 - k_out[2] ** 2)


def _peak_centroid(
    ubi: jax.Array, pos: jax.Array, hkl: jax.Array, etasign: ArrayLike, geom: dict
) -> tuple[jax.Array, jax.Array]:
    return get_centroid_scan(
        ubi, pos, hkl, etasign,
        geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, geom["wedge"], geom["chi"], geom["y0"],
        geom["s_step_lab"], geom["f_step_lab"], geom["det_origin_lab"],
    )  # fmt: skip


def _peak_cov(ubi: jax.Array, pos: jax.Array, hkl: jax.Array, etasign: ArrayLike, geom: dict) -> jax.Array:
    """[6] elements of the (sc, fc, omega) covariance, in the order of _COV_ELEMS."""
    cov_in = get_cov_in(jnp.zeros(3), geom["sig_wavelength"], geom["sig_ky"], geom["sig_kz"])
    return _propagate(
        ubi, pos, hkl, etasign,
        geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, geom["wedge"], geom["chi"], geom["y0"],
        geom["s_step_lab"], geom["f_step_lab"], geom["det_origin_lab"], cov_in,
    )  # fmt: skip


def _peak_factors(ubi: jax.Array, hkl: jax.Array, etasign: ArrayLike, geom: dict) -> jax.Array:
    """Lorentz x polarisation for one peak."""
    k_in, k_out, _, _ = hkl_to_k_omega(
        ubi, hkl, etasign, geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, geom["wedge"], geom["chi"]
    )
    rot_axis = sample_to_lab(jnp.array([0.0, 0.0, 1.0]), 0.0, geom["wedge"], geom["chi"], 0.0, 0.0)
    return lorentz(k_in, k_out, rot_axis) * polarisation(k_out, geom["pol_factor"])


def _wrap_omega(omega: jax.Array, omega_mid: jax.Array) -> jax.Array:
    """Shift omega by a multiple of 360 degrees to be closest to omega_mid."""
    return omega + 360.0 * jnp.round((omega_mid - omega) / 360.0)


@partial(jax.jit, static_argnames=("det_shape",))
def select_peaks(
    ubi: jax.Array,
    pos: jax.Array,
    hkls: jax.Array,
    geom: dict,
    row: dict,
    margin: jax.Array,
    det_shape: tuple[int, int],
) -> jax.Array:
    """Mask of the (entry, hkl, branch) peaks that can put intensity into this dty row.

    Parameters
    ----------
    ubi, pos
        [Ne, 3, 3] and [Ne, 3] for a chunk of map entries
    hkls
        [Nh, 3] hkls of the entries' phase
    geom
        Geometry dict, see :func:`render_row`
    row
        Row dict, see :func:`render_row`
    margin
        [4] how far outside the row a centroid may be and still contribute: (sc, fc, omega, dty)
    det_shape
        (n_slow, n_fast) detector shape

    Returns
    -------
    mask: jax.Array
        [Ne, Nh, 2] bool, branch 0 is etasign +1
    """

    def one(u: jax.Array, p: jax.Array, h: jax.Array) -> tuple[jax.Array, jax.Array]:
        centroids, valid = get_centroid_scan_both(
            u, p, h, geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, geom["wedge"], geom["chi"], geom["y0"],
            geom["s_step_lab"], geom["f_step_lab"], geom["det_origin_lab"],
        )  # fmt: skip
        return centroids, valid

    centroids, valid = jax.vmap(jax.vmap(one, in_axes=(None, None, 0)), in_axes=(0, 0, None))(ubi, pos, hkls)
    sc, fc, om, dty = centroids[..., 0], centroids[..., 1], centroids[..., 2], centroids[..., 3]
    om = _wrap_omega(om, 0.5 * (row["omega_min"] + row["omega_max"]))
    keep = (
        valid[..., None]
        & (sc > -margin[0]) & (sc < det_shape[0] - 1 + margin[0])
        & (fc > -margin[1]) & (fc < det_shape[1] - 1 + margin[1])
        & (om > row["omega_min"] - margin[2]) & (om < row["omega_max"] + margin[2])
        & (dty > row["dty_min"] - margin[3]) & (dty < row["dty_max"] + margin[3])
    )  # fmt: skip
    return keep


@partial(jax.jit, static_argnames=("window", "det_shape"))
def render_peaks(
    entry: jax.Array,
    hkl_idx: jax.Array,
    branch: jax.Array,
    entries: dict,
    hkls: jax.Array,
    F2: jax.Array,
    geom: dict,
    row: dict,
    window: tuple[int, int, int],
    det_shape: tuple[int, int],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Render a fixed-size batch of peaks into sparse contributions for one dty row.

    The peak is a Gaussian in (sc, fc, omega) with the propagated covariance. Its mass in each
    (frame, slow, fast) cell of the window is approximated by conditioning: the omega marginal over
    the frame, then slow given omega at the frame centre, then fast given slow and omega at the cell
    centres. This keeps the slow-fast and detector-omega correlations.

    Parameters
    ----------
    entry, hkl_idx, branch
        [B] int indices of the peaks: map entry, row of ``hkls``, and 0 / 1 for etasign +1 / -1
    entries
        Dict with "ubi" [N, 3, 3], "pos" [N, 3] and "density" [N]
    hkls, F2
        [Nh, 3] hkls and [Nh] structure factors squared
    geom, row
        See :func:`render_row`
    window
        Static (n_frames, n_slow, n_fast) window size, each odd
    det_shape
        Static (n_slow, n_fast) detector shape

    Returns
    -------
    frame: jax.Array
        [B, W] int32 frame index within the row (in file order), -1 where unused
    pixel: jax.Array
        [B, W] int32 pixel index slow * n_fast + fast
    value: jax.Array
        [B, W] intensity, 0 where unused
    captured: jax.Array
        [B] fraction of each peak's Gaussian that fell inside its window (before the dty weight)
    """
    wo, ws, wf = window
    ns, nf = det_shape

    def one(e: jax.Array, h: jax.Array, br: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        ubi, pos, hkl = entries["ubi"][e], entries["pos"][e], hkls[h]
        etasign = 1.0 - 2.0 * br
        centroid, valid = _peak_centroid(ubi, pos, hkl, etasign, geom)
        cov = _peak_cov(ubi, pos, hkl, etasign, geom)
        ss, ff, oo = cov[0] + _VAR_FLOOR[0], cov[1] + _VAR_FLOOR[1], cov[2] + _VAR_FLOOR[2]
        sf, so, fo = cov[3], cov[4], cov[5]
        mu_s, mu_f, mu_o, dty_c = centroid
        mu_o = _wrap_omega(mu_o, 0.5 * (row["omega_min"] + row["omega_max"]))

        # Window origin: no gradient through which cells a peak touches
        jo = jnp.searchsorted(row["omega_edges"], jax.lax.stop_gradient(mu_o)) - 1 - wo // 2
        i0 = jnp.round(jax.lax.stop_gradient(mu_s)).astype(int) - ws // 2
        j0 = jnp.round(jax.lax.stop_gradient(mu_f)).astype(int) - wf // 2
        frames = jo + jnp.arange(wo)  # [wo] in sorted-omega order
        rows = i0 + jnp.arange(ws)  # [ws]
        cols = j0 + jnp.arange(wf)  # [wf]
        nfr = row["omega_sorted"].shape[0]
        fclip = jnp.clip(frames, 0, nfr - 1)

        # omega marginal over each frame
        p_o = bin_fractions(row["omega_edges"][fclip], row["omega_edges"][fclip + 1], mu_o, jnp.sqrt(oo))
        d_o = row["omega_sorted"][fclip] - mu_o  # [wo]

        # slow | omega
        mu_s_o = mu_s + so / oo * d_o  # [wo]
        sd_s_o = jnp.sqrt(ss - so**2 / oo)
        p_s = bin_fractions(rows[None, :] - 0.5, rows[None, :] + 0.5, mu_s_o[:, None], sd_s_o)  # [wo, ws]

        # fast | slow, omega
        det = ss * oo - so**2
        ainv = jnp.array([[oo, -so], [-so, ss]]) / det
        bvec = jnp.array([sf, fo])
        gain = bvec @ ainv  # [2]
        d_s = rows[None, :] - mu_s  # [1, ws]
        mu_f_so = mu_f + gain[0] * d_s + gain[1] * d_o[:, None]  # [wo, ws]
        sd_f_so = jnp.sqrt(ff - bvec @ ainv @ bvec)
        p_f = bin_fractions(cols - 0.5, cols + 0.5, mu_f_so[..., None], sd_f_so)  # [wo, ws, wf]

        frac = p_o[:, None, None] * p_s[:, :, None] * p_f  # [wo, ws, wf]
        inside = (
            (frames >= 0)[:, None, None] & (frames < nfr)[:, None, None]
            & (rows >= 0)[None, :, None] & (rows < ns)[None, :, None]
            & (cols >= 0)[None, None, :] & (cols < nf)[None, None, :]
        )  # fmt: skip
        captured = jnp.sum(frac)

        # per-frame factors: beam over voxel at that frame's dty, and transmission
        w_dty = dty_weight(row["dty_sorted"][fclip] - dty_c, mu_o, geom["voxel_size"], geom["sig_beam"])
        per_frame = w_dty * row["transmission_sorted"][fclip]  # [wo]

        amp = entries["density"][e] * F2[h] * _peak_factors(ubi, hkl, etasign, geom)
        value = amp * per_frame[:, None, None] * frac
        use = inside & valid
        value = jnp.where(use, value, 0.0)
        frame_out = jnp.where(use, row["order"][fclip][:, None, None], -1)
        pixel = rows[None, :, None] * nf + cols[None, None, :]
        pixel_out = jnp.where(use, pixel, 0)
        return frame_out.ravel(), pixel_out.ravel(), value.ravel(), captured

    return jax.vmap(one)(entry, hkl_idx, branch)


@jax.jit
def _compact(
    frame: jax.Array, pixel: jax.Array, value: jax.Array, min_value: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Move the contributions that are kept (value >= min_value) to the front, without sorting.

    Returns fixed-size arrays plus the number of kept leading entries. Duplicates are not merged here.
    """
    frame, pixel, value = frame.ravel(), pixel.ravel(), value.ravel()
    keep = (frame >= 0) & (value >= min_value)
    n = frame.shape[0]
    target = jnp.where(keep, jnp.cumsum(keep) - 1, n)  # n is out of bounds, so dropped
    frame = jnp.zeros_like(frame).at[target].set(frame, mode="drop")
    pixel = jnp.zeros_like(pixel).at[target].set(pixel, mode="drop")
    value = jnp.zeros_like(value).at[target].set(value, mode="drop")
    return frame, pixel, value, jnp.sum(keep)


def make_row(omega: np.ndarray, dty: np.ndarray, transmission: np.ndarray | None = None) -> dict:
    """Build the row dict for :func:`render_row` from one scan's per-frame motor positions.

    Parameters
    ----------
    omega, dty
        [nframes] omega (degrees, frame centres) and dty for each frame, in file order
    transmission
        [nframes] optional transmission factor per frame (default 1)
    """
    omega = np.asarray(omega, dtype=float)
    dty = np.asarray(dty, dtype=float)
    if transmission is None:
        transmission = np.ones_like(omega)
    order = np.argsort(omega, kind="stable")
    om = omega[order]
    mids = 0.5 * (om[1:] + om[:-1])
    edges = np.concatenate([[om[0] - (mids[0] - om[0])], mids, [om[-1] + (om[-1] - mids[-1])]])
    return {
        "omega_sorted": jnp.asarray(om),
        "omega_edges": jnp.asarray(edges),
        "dty_sorted": jnp.asarray(dty[order]),
        "transmission_sorted": jnp.asarray(np.asarray(transmission, dtype=float)[order]),
        "order": jnp.asarray(order.astype(np.int32)),
        "omega_min": float(edges[0]),
        "omega_max": float(edges[-1]),
        "dty_min": float(dty.min()),
        "dty_max": float(dty.max()),
    }


@partial(jax.jit, static_argnames=("det_shape", "mesh"))
def _select_sharded(
    ubi: jax.Array,
    pos: jax.Array,
    hkls: jax.Array,
    geom: dict,
    row: dict,
    margin: jax.Array,
    det_shape: tuple[int, int],
    mesh: Mesh,
) -> jax.Array:
    """:func:`select_peaks` with the entries split across the devices of ``mesh``."""

    def local(u: jax.Array, p: jax.Array, h: jax.Array, g: dict, r: dict, m: jax.Array) -> jax.Array:
        return select_peaks(u, p, h, g, r, m, det_shape)

    specs = (P("d"), P("d"), P(), P(), P(), P())
    return shard_map(local, mesh=mesh, in_specs=specs, out_specs=P("d"))(ubi, pos, hkls, geom, row, margin)


@partial(jax.jit, static_argnames=("window", "det_shape", "mesh"))
def _render_sharded(
    entry: jax.Array,
    hkl_idx: jax.Array,
    branch: jax.Array,
    live: jax.Array,
    entries: dict,
    hkls: jax.Array,
    F2: jax.Array,
    geom: dict,
    row: dict,
    min_value: jax.Array,
    window: tuple[int, int, int],
    det_shape: tuple[int, int],
    mesh: Mesh,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """:func:`render_peaks` then :func:`_compact`, with the batch split across the devices of ``mesh``.

    Duplicate pixels are merged on the host.
    """

    def local(
        e: jax.Array,
        h: jax.Array,
        b: jax.Array,
        lv: jax.Array,
        ent: dict,
        hk: jax.Array,
        f2: jax.Array,
        g: dict,
        r: dict,
        mv: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        frame, pixel, value, captured = render_peaks(e, h, b, ent, hk, f2, g, r, window, det_shape)
        value = jnp.where(lv[:, None], value, 0.0)
        frame, pixel, value, count = _compact(frame, pixel, value, mv)
        return frame, pixel, value, count[None], captured

    specs = (P("d"),) * 4 + (P(),) * 6
    return shard_map(local, mesh=mesh, in_specs=specs, out_specs=(P("d"),) * 5)(
        entry, hkl_idx, branch, live, entries, hkls, F2, geom, row, min_value
    )


def _valid_from_shards(x: jax.Array, count: np.ndarray, per_shard: int) -> np.ndarray:
    """Copy the first count[k] elements of each device's shard of x to the host."""
    out = []
    for shard in sorted(x.addressable_shards, key=lambda sh: sh.index[0].start or 0):
        k = (shard.index[0].start or 0) // per_shard
        out.append(np.asarray(shard.data[: count[k]]))
    return np.concatenate(out)


def render_row(
    entries: dict,
    hkls: np.ndarray,
    F2: np.ndarray,
    geom: dict,
    row: dict,
    det_shape: tuple[int, int],
    window: tuple[int, int, int] = (3, 7, 7),
    batch: int = 2**16,
    select_chunk: int = 2**20,
    min_value: float = 1e-3,
    mesh: Mesh | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Render all peaks of one phase that reach one dty row into sparse pixels.

    Work is split across the devices of ``mesh`` (default :func:`anri.utils.mesh`): all GPUs,
    or all XLA CPU devices set up by :func:`anri.utils.setup`.

    Parameters
    ----------
    entries
        Dict with "ubi" [N, 3, 3], "pos" [N, 3] (sample frame, same length units as dty)
        and "density" [N], for map entries of a single phase
    hkls, F2
        [Nh, 3] hkls of that phase and [Nh] their structure factors squared
    geom
        Dict with "wavelength", "k_in_lab" [3], "wedge", "chi" (degrees), "y0",
        "s_step_lab", "f_step_lab", "det_origin_lab" [3] (from :func:`anri.geom.detector_basis_vectors_lab`),
        "sig_wavelength", "sig_ky", "sig_kz", "sig_beam", "voxel_size" and "pol_factor"
    row
        From :func:`make_row`
    det_shape
        (n_slow, n_fast)
    window
        (n_frames, n_slow, n_fast) window per peak, each odd
    batch
        Maximum number of peaks rendered at once, over all devices
    select_chunk
        Approximate number of (entry, hkl) pairs per call to :func:`select_peaks`
    min_value
        Contributions below this are dropped
    mesh
        Devices to use, default :func:`anri.utils.mesh`

    Returns
    -------
    frame, pixel, value: np.ndarray
        Sparse pixels sorted by (frame, pixel), with duplicates summed
    stats: dict
        "n_peaks" rendered and their "captured" window fractions
    """
    mesh = anri.utils.mesh() if mesh is None else mesh
    nd = mesh.size
    ubi = jnp.asarray(entries["ubi"])
    dtype = ubi.dtype
    pos = jnp.asarray(entries["pos"], dtype=dtype)
    hkls_j = jnp.asarray(hkls, dtype=dtype)
    F2_j = jnp.asarray(F2, dtype=dtype)
    entries = {"ubi": ubi, "pos": pos, "density": jnp.asarray(entries["density"], dtype=dtype)}
    geom = jax.tree.map(jnp.asarray, geom)
    row_j = jax.tree.map(jnp.asarray, row)
    n_entries, n_hkls = ubi.shape[0], hkls_j.shape[0]

    # 1. Which peaks can reach this row? Chunks of entries, padded to a multiple of the device count.
    wo, ws, wf = window
    ostep = float(np.max(np.diff(np.asarray(row["omega_edges"]))))
    margin = jnp.array(
        [ws // 2 + 1, wf // 2 + 1, (wo // 2 + 1) * ostep, 4 * float(geom["sig_beam"]) + float(geom["voxel_size"])],
        dtype=dtype,
    )
    ce = max(nd, (select_chunk // n_hkls) // nd * nd)
    found = []
    for start in range(0, n_entries, ce):
        stop = min(start + ce, n_entries)
        idx = np.arange(start, start + ce) % n_entries  # padding repeats real entries, then is discarded
        mask = _select_sharded(ubi[idx], pos[idx], hkls_j, geom, row_j, margin, det_shape, mesh)
        e, h, b = np.nonzero(np.asarray(mask)[: stop - start])
        found.append((e + start, h, b))
    e = np.concatenate([f[0] for f in found]).astype(np.int32)
    h = np.concatenate([f[1] for f in found]).astype(np.int32)
    b = np.concatenate([f[2] for f in found]).astype(np.int32)
    n_peaks = e.size
    if n_peaks == 0:
        empty = np.zeros(0, np.int32)
        return empty, empty, np.zeros(0), {"n_peaks": 0, "captured": np.zeros(0)}

    # 2. Render in fixed-size batches split over the devices. Small problems use smaller batches;
    # powers of two keep the number of compiled shapes small. Padding peaks are marked not live.
    per_device = min(max(batch // nd, 1), max(64, 1 << (-(-n_peaks // nd) - 1).bit_length()))
    batch = per_device * nd
    per_shard = per_device * wo * ws * wf
    min_value_j = jnp.asarray(min_value, dtype=dtype)
    frames, pixels, values, captured = [], [], [], []
    for start in range(0, n_peaks, batch):
        stop = min(start + batch, n_peaks)
        pad = batch - (stop - start)
        idx = [np.pad(x[start:stop], (0, pad)) for x in (e, h, b)]
        live = np.arange(batch) < stop - start
        fr, px, val, count, cap = _render_sharded(
            *idx, live, entries, hkls_j, F2_j, geom, row_j, min_value_j, window, det_shape, mesh
        )
        count = np.asarray(count)
        frames.append(_valid_from_shards(fr, count, per_shard))
        pixels.append(_valid_from_shards(px, count, per_shard))
        values.append(_valid_from_shards(val, count, per_shard))
        captured.append(np.asarray(cap)[: stop - start])

    # 3. Merge batches and devices: sort by (frame, pixel) and sum duplicates
    npix = det_shape[0] * det_shape[1]
    key = np.concatenate(frames).astype(np.int64) * npix + np.concatenate(pixels)
    value = np.concatenate(values)
    order = np.argsort(key)
    key, value = key[order], value[order]
    new = np.ones(key.size, bool)
    new[1:] = key[1:] != key[:-1]
    starts = np.flatnonzero(new)
    value = np.add.reduceat(value, starts)
    frame, pixel = np.divmod(key[starts], npix)
    stats = {"n_peaks": n_peaks, "captured": np.concatenate(captured)}
    return frame.astype(np.int32), pixel.astype(np.int32), value, stats
