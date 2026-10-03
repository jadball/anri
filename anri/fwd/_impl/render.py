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
from anri.geom import beam_basis, sample_to_lab

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


def truncated_moments(lo: ArrayLike, hi: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> tuple:
    """Mass, mean and variance of N(mu, sigma^2) restricted to [lo, hi).

    Where the mass is negligible, the mean falls back to mu clipped into the interval and the variance to 0
    (those cells carry no intensity, and this keeps gradients finite).
    """
    lo, hi, mu, sigma = (jnp.asarray(x) for x in (lo, hi, mu, sigma))
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    mass = ndtr(b) - ndtr(a)
    pa, pb = jnp.asarray(norm.pdf(a)), jnp.asarray(norm.pdf(b))
    ok = mass > 1e-12
    safe = jnp.where(ok, mass, 1.0)
    r = (pa - pb) / safe
    mean = jnp.where(ok, mu + sigma * r, jnp.clip(mu, lo, hi))
    var = jnp.where(ok, sigma**2 * (1.0 + (a * pa - b * pb) / safe - r**2), 0.0)
    return mass, mean, jnp.maximum(var, 0.0)


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


def polarisation(k_in: jax.Array, k_out: jax.Array, factor: ArrayLike) -> jax.Array:
    """Polarisation factor for a beam polarised horizontally, with degree of polarisation ``factor``.

    Horizontal and vertical are across the beam, see :func:`anri.geom.beam_basis`. For a beam along lab x
    (horizontal = lab y) this equals :func:`ImageD11.refinegrains.polarization` with ``eta0 = 0``.
    ``factor = 1`` is fully horizontally polarised, ``factor = 0`` is unpolarised.
    """
    _, e_h, e_v = beam_basis(k_in)
    k_out = k_out / jnp.linalg.norm(k_out)
    return 0.5 * (1 + factor) * (1 - (k_out @ e_h) ** 2) + 0.5 * (1 - factor) * (1 - (k_out @ e_v) ** 2)


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
    return lorentz(k_in, k_out, rot_axis) * polarisation(k_in, k_out, geom["pol_factor"])


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
    the frame, then slow given omega within the frame, then fast given slow within the row, with omega
    integrated out within the frame. Each step conditions on the within-cell (truncated) mean and variance.
    This keeps the slow-fast and detector-omega correlations, and the centroid of peaks narrower than a
    frame or pixel.

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
        # detector point spread adds to the slow and fast variances (pixels^2)
        psf2 = geom["sig_psf"] ** 2
        ss, ff, oo = cov[0] + psf2 + _VAR_FLOOR[0], cov[1] + psf2 + _VAR_FLOOR[1], cov[2] + _VAR_FLOOR[2]
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

        # omega over each frame: mass, and the mean and variance of omega within the frame. Conditioning on the
        # within-frame mean (not the frame centre) keeps peaks much narrower than a frame at their true position.
        p_o, m_o, v_o = truncated_moments(
            row["omega_edges"][fclip], row["omega_edges"][fclip + 1], mu_o, jnp.sqrt(oo)
        )  # [wo]
        d_o = m_o - mu_o

        # slow | omega in the frame
        slope = so / oo
        mu_s_o = mu_s + slope * d_o  # [wo]
        sd_s_o = jnp.sqrt(ss - so**2 / oo + slope**2 * v_o)  # [wo]
        p_s, m_s, v_s = truncated_moments(
            rows[None, :] - 0.5, rows[None, :] + 0.5, mu_s_o[:, None], sd_s_o[:, None]
        )  # [wo, ws]

        # fast | slow in the row, omega in the frame. Within the frame, omega is taken as N(m_o, v_o) and slow | omega
        # is Gaussian, so (slow, omega) are jointly Gaussian there: integrate omega out given slow, then average over
        # the row. Holding omega at m_o instead ignores that the row tells us where omega is within the frame, which
        # matters for peaks narrower in omega than a frame (most of them).
        det = ss * oo - so**2
        g_s = (sf * oo - fo * so) / det  # regression of fast on slow and omega
        g_o = (fo * ss - sf * so) / det
        var_f_so = ff - (g_s * sf + g_o * fo)  # variance of fast given slow and omega
        k = slope * v_o / sd_s_o**2  # [wo] slope of E[omega | slow] within the frame
        var_o_s = v_o * (1.0 - k * slope)  # [wo] variance of omega given slow within the frame
        e_o = m_o[:, None] + k[:, None] * (m_s - mu_s_o[:, None])  # [wo, ws] E[omega | slow] at the row's mean slow
        mu_f_so = mu_f + g_s * (m_s - mu_s) + g_o * (e_o - mu_o)  # [wo, ws]
        dfds = g_s + g_o * k  # [wo] slope of E[fast | slow] within the frame
        sd_f_so = jnp.sqrt(var_f_so + g_o**2 * var_o_s[:, None] + dfds[:, None] ** 2 * v_s)  # [wo, ws]
        p_f = bin_fractions(cols - 0.5, cols + 0.5, mu_f_so[..., None], sd_f_so[..., None])  # [wo, ws, wf]

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


def _as_jax(entries: dict, hkls: ArrayLike, F2: ArrayLike, geom: dict, row: dict) -> tuple:
    """Convert the renderer's inputs to JAX arrays, in the dtype of the entries' UBIs."""
    ubi = jnp.asarray(entries["ubi"])
    dtype = ubi.dtype
    entries = {
        "ubi": ubi,
        "pos": jnp.asarray(entries["pos"], dtype=dtype),
        "density": jnp.asarray(entries["density"], dtype=dtype),
    }
    hkls, F2 = jnp.asarray(hkls, dtype=dtype), jnp.asarray(F2, dtype=dtype)
    return entries, hkls, F2, jax.tree.map(jnp.asarray, geom), jax.tree.map(jnp.asarray, row)


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
        "sig_wavelength", "sig_ky", "sig_kz", "sig_beam", "sig_psf" (detector point spread, pixels), "voxel_size"
        and "pol_factor"
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
    entries, hkls_j, F2_j, geom, row_j = _as_jax(entries, hkls, F2, geom, row)
    ubi, pos = entries["ubi"], entries["pos"]
    dtype = ubi.dtype
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


def check_render(
    entries: dict,
    hkls: np.ndarray,
    geom: dict,
    row: dict,
    det_shape: tuple[int, int],
    window: tuple[int, int, int] = (3, 7, 7),
    n_peaks: int = 100,
    n_samples: int = 100_000,
    n_entries: int = 1000,
    seed: int = 0,
) -> dict:
    """Check rendered peaks against a Monte Carlo simulation of the beam spreads.

    For a random sample of peaks of the row, samples wavelength, ky and kz from their Gaussian spreads, pushes
    every sample through the forward model (:func:`anri.fwd.get_centroid_scan`), adds the detector point spread
    and the renderer's variance floor, and histograms the (slow, fast, omega) positions into the peak's window
    cells. :func:`render_peaks` linearises the forward model and integrates the resulting Gaussian over the cells
    approximately, so this checks both approximations, for your own map and geometry.

    Only peaks whose whole window is on the detector and inside the scan are compared.

    Parameters
    ----------
    entries, hkls, geom, row, det_shape, window
        As for :func:`render_row`
    n_peaks
        Number of peaks to check
    n_samples
        Monte Carlo samples per peak. The noise in a cell's fraction is about sqrt(fraction / n_samples).
    n_entries
        Number of map entries to pick peaks from, at random
    seed
        Random seed

    Returns
    -------
    result: dict
        Per checked peak: "entry", "hkl" (row of ``hkls``) and "branch" (0 for etasign +1), "max_cell_error" (largest
        absolute difference between rendered and Monte Carlo fraction of the peak in any cell), "captured" (fraction
        of the rendered Gaussian inside the window) and "captured_mc" (fraction of the Monte Carlo samples inside it)
    """
    rng = np.random.default_rng(seed)
    n_hkls = np.shape(hkls)[0]
    ents, hkls_j, ones, geom, row_j = _as_jax(
        {**entries, "density": np.ones(len(entries["ubi"]))}, hkls, np.ones(n_hkls), geom, row
    )  # unit density and |F|^2: only the fractions in the cells matter
    ubi, pos = ents["ubi"], ents["pos"]
    dtype = ubi.dtype
    wo, ws, wf = window

    # 1. Peaks of a random subset of entries that reach this row
    sub = rng.choice(ubi.shape[0], size=min(n_entries, ubi.shape[0]), replace=False)
    ostep = float(np.max(np.diff(np.asarray(row["omega_edges"]))))
    margin = jnp.array(
        [ws // 2 + 1, wf // 2 + 1, (wo // 2 + 1) * ostep, 4 * float(geom["sig_beam"]) + float(geom["voxel_size"])],
        dtype=dtype,
    )
    e, h, b = np.nonzero(np.asarray(select_peaks(ubi[sub], pos[sub], hkls_j, geom, row_j, margin, det_shape)))
    pick = rng.choice(e.size, size=min(n_peaks, e.size), replace=False)
    e, h, b = sub[e[pick]].astype(np.int32), h[pick].astype(np.int32), b[pick].astype(np.int32)

    # 2. Rendered fraction of each peak in each cell (the per-peak factors are constant over a window)
    frame, pixel, value, captured = (
        np.asarray(x) for x in render_peaks(e, h, b, ents, hkls_j, ones, geom, row_j, window, det_shape)
    )
    frame, pixel, value = (x.reshape(-1, wo, ws, wf) for x in (frame, pixel, value))
    total = value.sum(axis=(1, 2, 3))
    whole = (frame >= 0).all(axis=(1, 2, 3)) & (total > 0)

    # 3. Monte Carlo: beam spreads through the forward model, then point spread and floor
    def centroid(
        u: jax.Array,
        p: jax.Array,
        q: jax.Array,
        etasign: jax.Array,
        wavelength: jax.Array,
        ky: jax.Array,
        kz: jax.Array,
    ) -> jax.Array:
        return get_centroid_scan(
            u, p, q, etasign, wavelength, geom["k_in_lab"], ky, kz,
            geom["wedge"], geom["chi"], geom["y0"], geom["s_step_lab"], geom["f_step_lab"], geom["det_origin_lab"],
        )[0][:3]  # fmt: skip

    sample = jax.jit(jax.vmap(centroid, in_axes=(None, None, None, None, 0, 0, 0)))
    sd_beam = np.array([float(geom["sig_wavelength"]), float(geom["sig_ky"]), float(geom["sig_kz"])])
    psf2 = float(geom["sig_psf"]) ** 2
    sd_extra = np.sqrt([psf2 + _VAR_FLOOR[0], psf2 + _VAR_FLOOR[1], _VAR_FLOOR[2]])
    sorted_index = np.argsort(np.asarray(row["order"]))  # file-order frame -> sorted-omega frame
    edges_all = np.asarray(row["omega_edges"])
    omega_mid = 0.5 * (row["omega_min"] + row["omega_max"])
    keep, err, cap_mc = [], [], []
    for i in np.flatnonzero(whole):
        spreads = rng.standard_normal((n_samples, 3)) * sd_beam
        wavelength = float(geom["wavelength"]) + spreads[:, 0]
        x = np.array(
            sample(ubi[e[i]], pos[e[i]], hkls_j[h[i]], 1.0 - 2.0 * b[i], wavelength, spreads[:, 1], spreads[:, 2])
        )
        x += rng.standard_normal((n_samples, 3)) * sd_extra
        x[:, 2] += 360.0 * np.round((omega_mid - x[:, 2]) / 360.0)
        s0, f0 = divmod(int(pixel[i, 0, 0, 0]), det_shape[1])
        jo = sorted_index[frame[i, 0, 0, 0]]
        cells = [s0 - 0.5 + np.arange(ws + 1), f0 - 0.5 + np.arange(wf + 1), edges_all[jo : jo + wo + 1]]
        mc = np.moveaxis(np.histogramdd(x, bins=cells)[0], 2, 0) / n_samples  # (frame, slow, fast)
        rendered = value[i] / total[i] * captured[i]
        keep.append(i)
        err.append(np.abs(rendered - mc).max())
        cap_mc.append(mc.sum())
    keep = np.asarray(keep, dtype=int)
    return {
        "entry": e[keep],
        "hkl": h[keep],
        "branch": b[keep],
        "max_cell_error": np.asarray(err),
        "captured": captured[keep],
        "captured_mc": np.asarray(cap_mc),
    }


def _free_memory(devices: list) -> tuple[float, bool]:
    """Free memory per device in bytes, and whether the devices share the host's memory (CPU devices)."""
    if devices[0].platform == "cpu":
        import psutil

        return psutil.virtual_memory().available / len(devices), True
    free = [d.memory_stats()["bytes_limit"] - d.memory_stats()["bytes_in_use"] for d in devices]
    return float(min(free)), False


def guess_batch_size(
    entries: dict,
    hkls: np.ndarray,
    F2: np.ndarray,
    geom: dict,
    row: dict,
    det_shape: tuple[int, int],
    window: tuple[int, int, int] = (3, 7, 7),
    mesh: Mesh | None = None,
    memory_fraction: float = 0.25,
) -> int:
    """Largest ``batch`` for :func:`render_row` whose render step fits in a fraction of the free memory.

    Compiles the render step for two small batches (nothing is rendered) and reads XLA's memory analysis to get
    the bytes per peak, then fits as many peaks as ``memory_fraction`` of the free memory allows: on GPUs, of the
    least free device; on CPU, of the host memory shared by the CPU devices, counting the host copy of each
    batch's output too. Precision matters: float64 needs about twice the memory of float32. Selecting the peaks
    (``select_chunk`` in :func:`render_row`) needs memory too, a few hundred MB by default, which is not counted.

    Parameters
    ----------
    entries, hkls, F2, geom, row, det_shape, window, mesh
        As for :func:`render_row`
    memory_fraction
        Fraction of the free memory to use. The default leaves room for other users of a shared machine.

    Returns
    -------
    batch: int
        A power of two times the number of devices, at least 64 peaks per device
    """
    mesh = anri.utils.mesh() if mesh is None else mesh
    nd = mesh.size
    entries, hkls_j, F2_j, geom, row_j = _as_jax(entries, hkls, F2, geom, row)
    min_value = jnp.asarray(0.0, dtype=entries["ubi"].dtype)

    def sizes(per_device: int) -> tuple[int, int]:
        idx = [jax.ShapeDtypeStruct((per_device * nd,), jnp.int32)] * 3
        live = jax.ShapeDtypeStruct((per_device * nd,), jnp.bool_)
        compiled = _render_sharded.lower(
            *idx, live, entries, hkls_j, F2_j, geom, row_j, min_value, window, det_shape, mesh
        ).compile()
        m = compiled.memory_analysis()
        if m is None:
            msg = "XLA gives no memory analysis on this backend: pass batch to render_row yourself"
            raise RuntimeError(msg)
        total = m.temp_size_in_bytes + m.argument_size_in_bytes + m.output_size_in_bytes - m.alias_size_in_bytes
        return total, m.output_size_in_bytes

    (total1, out1), (total2, out2) = sizes(1024), sizes(2048)
    per_peak = (total2 - total1) / 1024
    fixed = total1 - 1024 * per_peak
    free, host = _free_memory(list(mesh.devices.flat))
    if host:
        per_peak += (out2 - out1) / 1024  # render_row copies each batch's output to the host
    per_device = int((memory_fraction * free - fixed) // per_peak)
    per_device = 1 << max(6, per_device.bit_length() - 1)  # largest power of two that fits, at least 64
    return per_device * nd
