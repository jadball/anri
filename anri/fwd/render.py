"""Render peaks from a voxel map into sparse detector pixels, one dty row (one rotation scan) at a time.

A map is a flat list of entries (position, UBI, density), so several orientations per voxel are allowed.
Each peak is an (entry, hkl, branch) triple. Its (slow, fast, omega) centroid is :func:`anri.fwd.peak.get_centroid`,
with the voxel at its real position for the row's dty, and its covariance comes from propagating the beam's spreads
through it with :func:`anri.fwd.peak.make_propagator`.

The beam is described by its profile across it, horizontally and vertically: each a flat top of some width,
blurred by a Gaussian (a pure Gaussian with zero width). A pencil beam is narrow both ways, a line beam wide one
way, a box beam wide both ways. Voxels are either columns (2D maps: the beam's vertical profile integrates out)
or cubes (3D maps). See :func:`anri.geom.beam.beam_weight`.

Absorption along the incoming and diffracted paths through the sample is modelled when the entries carry their
optical depth to the sample's edge (:func:`add_absorption`), with the same paths as the indexer
(:mod:`anri.geom.paths`).

Pixel values are differentiable with respect to the map. Which pixels a peak touches is not:
window placement is computed from rounded centroids.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike

try:
    from jax import shard_map
except ImportError:  # JAX < 0.6, e.g. 0.4.30, the last release for Python 3.9
    from jax.experimental.shard_map import shard_map

import anri.utils
from anri.diffract.factors import lorentz_polarisation
from anri.geom.beam import beam_offsets, beam_weight
from anri.geom.gonio import nearest_omega, sample_to_lab
from anri.geom.paths import attenuation_map, edge_distances, ray_directions, transmission
from anri.maths.rotation import skew
from anri.maths.utils import bin_fractions, truncated_moments

from .peak import get_centroid, get_centroid_both, hkl_to_k_omega, make_propagator

# Smallest batch per device that render_row compiles (see there)
_MIN_BATCH = 1024

# Elements of the (sc, fc, omega) covariance that the renderer uses
_COV_ELEMS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

# Added to the variances so that a peak with no instrumental broadening still renders (pixels, pixels, degrees).
_VAR_FLOOR = (1e-4, 1e-4, 1e-8)


def _centroid(
    ubi: jax.Array, pos: jax.Array, hkl: jax.Array, etasign: ArrayLike, wavelength: ArrayLike, ky: ArrayLike,
    kz: ArrayLike, dty: ArrayLike, geom: dict,
) -> tuple[jax.Array, jax.Array]:  # fmt: skip
    """:func:`anri.fwd.peak.get_centroid` with the geometry dict: (sc, fc, omega) of a peak, and whether it diffracts."""
    return get_centroid(
        ubi, pos, hkl, etasign, wavelength, geom["k_in_lab"], ky, kz, geom["wedge"], geom["chi"], dty, geom["y0"],
        geom["s_step_lab"], geom["f_step_lab"], geom["det_origin_lab"], jnp.logical_not(geom["voxel_3d"]),
    )  # fmt: skip


# Covariance of (sc, fc, omega) from the wavelength, ky and kz spreads. The voxel's extent and the beam profile
# enter through beam_weight instead.
_propagate = make_propagator(_centroid, argnums=(4, 5, 6), has_aux=True, out_elems=_COV_ELEMS)


def _centroid_rotated(
    ubi: jax.Array, pos: jax.Array, hkl: jax.Array, etasign: ArrayLike, rotvec: jax.Array, dty: ArrayLike, geom: dict
) -> tuple[jax.Array, jax.Array]:
    """Return the centroid with the lattice turned by a small sample-frame rotation (rotation vector, radians).

    UB -> R UB, so UBI -> UBI R^T, with R = I + [rotvec]x: exact to first order, which is all the propagation uses.
    """
    return _centroid(
        ubi @ (jnp.eye(3, dtype=ubi.dtype) + skew(rotvec)).T, pos, hkl, etasign, geom["wavelength"], 0.0, 0.0, dty, geom
    )


# Covariance of (sc, fc, omega) from an entry's intrinsic orientation spread (isotropic, sample frame)
_propagate_rotation = make_propagator(_centroid_rotated, argnums=(4,), has_aux=True, out_elems=_COV_ELEMS)


def _peak_cov(
    ubi: jax.Array,
    pos: jax.Array,
    hkl: jax.Array,
    etasign: ArrayLike,
    dty: ArrayLike,
    geom: dict,
    sig_rot: ArrayLike | None = None,
) -> jax.Array:
    """[6] elements of the (sc, fc, omega) covariance, in the order of _COV_ELEMS.

    sig_rot: the entry's intrinsic orientation spread, the standard deviation of each component of a small
    sample-frame rotation vector (radians). None (the default) leaves it out entirely, at no cost.
    """
    cov_in = jnp.diag(jnp.array([geom["sig_wavelength"], geom["sig_ky"], geom["sig_kz"]]) ** 2)
    cov = _propagate(ubi, pos, hkl, etasign, geom["wavelength"], 0.0, 0.0, dty, geom, cov_in)
    if sig_rot is not None:
        rot_in = jnp.eye(3, dtype=ubi.dtype) * jnp.asarray(sig_rot, ubi.dtype) ** 2
        cov = cov + _propagate_rotation(ubi, pos, hkl, etasign, jnp.zeros(3, ubi.dtype), dty, geom, rot_in)
    return cov


def _peak_factors(ubi: jax.Array, hkl: jax.Array, etasign: ArrayLike, geom: dict) -> jax.Array:
    """Lorentz x polarisation for one peak."""
    k_in, k_out, _, _ = hkl_to_k_omega(
        ubi, hkl, etasign, geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, geom["wedge"], geom["chi"]
    )
    return lorentz_polarisation(k_in, k_out, geom["wedge"], geom["chi"], geom["pol_factor"])


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
        [5] how far outside the row a centroid may be and still contribute: (sc, fc, omega), and how far a voxel
        may be from the beam's centre line, across it horizontally and vertically (vertically only for cubes)
    det_shape
        (n_slow, n_fast) detector shape

    Returns
    -------
    mask: jax.Array
        [Ne, Nh, 2] bool, branch 0 is etasign +1
    """
    dty = row["dty_mid"]

    def one(u: jax.Array, p: jax.Array, h: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        cens, valid = get_centroid_both(
            u, p, h, geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, geom["wedge"], geom["chi"], dty, geom["y0"],
            geom["s_step_lab"], geom["f_step_lab"], geom["det_origin_lab"], jnp.logical_not(geom["voxel_3d"]),
        )  # fmt: skip
        near_h, near_v = [], []
        for i in range(2):
            across, height, _, _ = beam_offsets(
                sample_to_lab(p, cens[i, 2], geom["wedge"], geom["chi"], dty, geom["y0"]), geom["k_in_lab"]
            )
            # within the omega margin an off-axis voxel moves across the beam by up to |p_xy| x margin (radians)
            near_h.append(jnp.abs(across) < margin[3] + jnp.hypot(p[0], p[1]) * jnp.radians(margin[2]))
            near_v.append(jnp.abs(height) < margin[4])
        return cens, valid, jnp.stack(near_h), jnp.stack(near_v)

    centroids, valid, near_h, near_v = jax.vmap(jax.vmap(one, in_axes=(None, None, 0)), in_axes=(0, 0, None))(
        ubi, pos, hkls
    )
    sc, fc, om = centroids[..., 0], centroids[..., 1], centroids[..., 2]
    om = nearest_omega(om, row["omega_mid"])
    keep = (
        valid[..., None]
        & (sc > -margin[0]) & (sc < det_shape[0] - 1 + margin[0])
        & (fc > -margin[1]) & (fc < det_shape[1] - 1 + margin[1])
        & (om > row["omega_min"] - margin[2]) & (om < row["omega_max"] + margin[2])
        & near_h & (near_v | ~geom["voxel_3d"])
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
    origins: tuple | None = None,
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
        Dict with "ubi" [N, 3, 3], "pos" [N, 3] and "density" [N]; optionally "sig_rot" [N], see :func:`render_row`
    hkls, F2
        [Nh, 3] hkls and [Nh] structure factors squared
    geom, row
        See :func:`render_row`
    window
        Static (n_frames, n_slow, n_fast) window size, each odd
    det_shape
        Static (n_slow, n_fast) detector shape
    origins
        Optional window origins, as from :func:`window_origins`: the first frame [B] (sorted-omega order) and each
        frame's first slow and fast pixel [B, n_frames]. By default each peak's window is centred on it, so the
        cells it covers jump as it moves; fixed origins keep the values smooth in the parameters (for a refiner)

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
    fr, px, val, cap, _, _ = _render_peaks(
        entry, hkl_idx, branch, entries, hkls, F2, geom, row, window, det_shape, origins
    )
    return fr, px, val, cap


def window_origins(
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
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Window origins :func:`render_peaks` chooses for these peaks: centred on each peak at the given parameters.

    Returns
    -------
    tuple
        First frame [B] (sorted-omega order), and each frame's first slow and fast pixel [B, n_frames]
    """
    return _render_peaks(entry, hkl_idx, branch, entries, hkls, F2, geom, row, window, det_shape, None)[5]


def _peak_shape(
    ubi: jax.Array, pos: jax.Array, hkl: jax.Array, etasign: ArrayLike, sig_rot: jax.Array | None, row: dict, geom: dict
) -> tuple:
    """Return a peak's centroid, covariance and whether it diffracts: the expensive, geometric part of rendering.

    Returns
    -------
    mu: jax.Array
        [3] (slow, fast, omega), omega before wrapping into the row's range
    shape: tuple
        (ss, ff, oo, sf, so, fo): variances and covariances, with the detector's point spread and floors added
    valid: jax.Array
        Whether the reflection diffracts
    """
    dty_row = row["dty_mid"]
    centroid, valid = _centroid(ubi, pos, hkl, etasign, geom["wavelength"], 0.0, 0.0, dty_row, geom)
    cov = _peak_cov(ubi, pos, hkl, etasign, dty_row, geom, sig_rot)
    # detector point spread adds to the slow and fast variances (pixels^2)
    psf2 = geom["sig_psf"] ** 2
    om2 = geom.get("sig_omega", 0.0) ** 2  # optional extra omega spread (degrees), e.g. to smooth a refinement
    shape = (cov[0] + psf2 + _VAR_FLOOR[0], cov[1] + psf2 + _VAR_FLOOR[1], cov[2] + om2 + _VAR_FLOOR[2],
             cov[3], cov[4], cov[5])  # fmt: skip
    return centroid, shape, valid


def _window_cells(
    mu: jax.Array, shape: tuple, org: tuple | None, row: dict, window: tuple[int, int, int], det_shape: tuple[int, int]
) -> tuple:
    """Integrate a peak's Gaussian over the (frame, slow, fast) cells of its window: the cheap part of rendering.

    Parameters
    ----------
    mu, shape
        Centroid and covariance, from :func:`_peak_shape`
    org
        Window origin (first frame, and each frame's first slow and fast pixel), or None: centred on the peak
    row, window, det_shape
        As for :func:`render_peaks`

    Returns
    -------
    tuple
        frac [wo, ws, wf] (fraction of the Gaussian in each cell), inside [wo, ws, wf] (cells on the detector and in
        the row), fclip [wo] (sorted frame of each window frame, clipped), rows [wo, ws], cols [wo, wf], om_fr [wo]
        (mean omega of the mass in each frame, unwrapped), in_range (the share of the peak's omega profile within the
        scan's omega range) and the origin (jo, i0, j0)
    """
    wo, ws, wf = window
    ns, nf = det_shape
    mu_s, mu_f, mu_o = mu
    ss, ff, oo, sf, so, fo = shape
    omega_peak = mu_o  # before wrapping
    mu_o = nearest_omega(mu_o, row["omega_mid"])

    # Window origin: no gradient through which cells a peak touches
    if org is None:
        jo = jnp.searchsorted(row["omega_edges"], jax.lax.stop_gradient(mu_o)) - 1 - wo // 2
    else:
        jo = org[0]
    frames = jo + jnp.arange(wo)  # [wo] in sorted-omega order
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

    # Each frame's pixel window is centred on the peak's mean position in that frame, so a peak that moves
    # across the detector with omega (broad in omega, e.g. near eta = 0) stays inside its window
    if org is None:
        i0 = jnp.round(jax.lax.stop_gradient(mu_s_o)).astype(int) - ws // 2  # [wo]
        j0 = jnp.round(jax.lax.stop_gradient(mu_f + fo / oo * d_o)).astype(int) - wf // 2  # [wo]
    else:
        i0, j0 = org[1], org[2]
    rows = i0[:, None] + jnp.arange(ws)  # [wo, ws]
    cols = j0[:, None] + jnp.arange(wf)  # [wo, wf]
    p_s, m_s, v_s = truncated_moments(rows - 0.5, rows + 0.5, mu_s_o[:, None], sd_s_o[:, None])  # [wo, ws]

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
    p_f = bin_fractions(
        cols[:, None, :] - 0.5, cols[:, None, :] + 0.5, mu_f_so[..., None], sd_f_so[..., None]
    )  # [wo, ws, wf]

    frac = p_o[:, None, None] * p_s[:, :, None] * p_f  # [wo, ws, wf]
    inside = (
        (frames >= 0)[:, None, None] & (frames < nfr)[:, None, None]
        & (rows >= 0)[:, :, None] & (rows < ns)[:, :, None]
        & (cols >= 0)[:, None, :] & (cols < nf)[:, None, :]
    )  # fmt: skip

    sd_o = jnp.sqrt(oo)
    in_range = ndtr((row["omega_edges"][-1] - mu_o) / sd_o) - ndtr((row["omega_edges"][0] - mu_o) / sd_o)
    return frac, inside, fclip, rows, cols, omega_peak + d_o, in_range, (jo, i0, j0)


def _frame_weights(pos: jax.Array, om_fr: jax.Array, fclip: jax.Array, row: dict, geom: dict) -> jax.Array:
    """Per frame of a window: the beam's profile over the voxel at that frame's dty, and transmission.

    At the mean omega of the peak's mass within each frame: a peak spread in omega (e.g. by sig_rot) diffracts at
    different omegas, where an off-axis voxel sits at a different place across the beam.
    """
    lab = jax.vmap(lambda o, d: sample_to_lab(pos, o, geom["wedge"], geom["chi"], d, geom["y0"]))(
        om_fr, row["dty_sorted"][fclip]
    )  # [wo, 3]
    w_beam = jax.vmap(beam_weight, in_axes=(0, 0, None))(lab, om_fr, geom)
    per_frame = w_beam * row["transmission_sorted"][fclip]  # [wo]

    return per_frame


def _transmission(
    depth: jax.Array, ubi: jax.Array, hkl: jax.Array, etasign: ArrayLike, omega: jax.Array, geom: dict
) -> jax.Array:
    """Fraction of a peak's intensity that survives absorption, at each omega (e.g. per window frame).

    ``depth`` [n_phi] is the entry's optical depth to the sample's edge (:func:`add_absorption`); the paths go back
    along the incoming beam and along the diffracted ray, from the voxel at that omega.
    """
    _, k_out, _, _ = hkl_to_k_omega(
        ubi, hkl, etasign, geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, geom["wedge"], geom["chi"]
    )
    k_out = jnp.broadcast_to(k_out, omega.shape + (3,))
    d_in, d_out = ray_directions(geom["k_in_lab"], k_out, omega, geom["wedge"], geom["chi"])
    return transmission(depth[None], d_in[None], d_out[None], 1.0)[0].astype(omega.dtype)


def _render_peaks(
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
    origins: tuple | None,
) -> tuple:
    """:func:`render_peaks`, also returning each peak's share of its omega profile in the scan and the window origins."""
    nf = det_shape[1]

    def one(e: jax.Array, h: jax.Array, br: jax.Array, org: tuple | None) -> tuple:
        ubi, pos, hkl = entries["ubi"][e], entries["pos"][e], hkls[h]
        etasign = 1.0 - 2.0 * br
        sig = entries["sig_rot"][e] if "sig_rot" in entries else None
        mu, shape, valid = _peak_shape(ubi, pos, hkl, etasign, sig, row, geom)
        frac, inside, fclip, rows, cols, om_fr, in_range, origin = _window_cells(mu, shape, org, row, window, det_shape)
        per_frame = _frame_weights(pos, om_fr, fclip, row, geom)
        if "optical_depth" in entries:
            per_frame = per_frame * _transmission(entries["optical_depth"][e], ubi, hkl, etasign, om_fr, geom)
        amp = entries["density"][e] * F2[h] * _peak_factors(ubi, hkl, etasign, geom)
        value = amp * per_frame[:, None, None] * frac
        use = inside & valid
        value = jnp.where(use, value, 0.0)
        frame_out = jnp.where(use, row["order"][fclip][:, None, None], -1)
        pixel = rows[:, :, None] * nf + cols[:, None, :]
        pixel_out = jnp.where(use, pixel, 0)
        frames = origin[0] + jnp.arange(window[0])
        real = (frames >= 0) & (frames < row["omega_sorted"].shape[0])
        captured = jnp.sum(jnp.where(real[:, None, None], frac, 0.0))  # frames past the scan's ends are clipped copies
        return frame_out.ravel(), pixel_out.ravel(), value.ravel(), captured, in_range, origin

    return jax.vmap(one, in_axes=(0, 0, 0, None if origins is None else 0))(entry, hkl_idx, branch, origins)


@jax.jit
def _compact(
    frame: jax.Array, pixel: jax.Array, value: jax.Array, min_value: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Move the contributions that are kept (value >= min_value) to the front, without sorting.

    Returns fixed-size arrays plus the number of kept leading entries. Duplicates are not merged here. Used on CPUs,
    where XLA's sort is several times slower than NumPy's, so :func:`_merge` would cost more than it saves.
    """
    frame, pixel, value = frame.ravel(), pixel.ravel(), value.ravel()
    keep = (frame >= 0) & (value >= min_value)
    n = frame.shape[0]
    target = jnp.where(keep, jnp.cumsum(keep) - 1, n)  # n is out of bounds, so dropped
    frame = jnp.zeros_like(frame).at[target].set(frame, mode="drop")
    pixel = jnp.zeros_like(pixel).at[target].set(pixel, mode="drop")
    value = jnp.zeros_like(value).at[target].set(value, mode="drop")
    return frame, pixel, value, jnp.sum(keep)


@jax.jit
def _merge(
    frame: jax.Array, pixel: jax.Array, value: jax.Array, min_value: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Drop contributions below min_value, then sum duplicate (frame, pixel)s: sorted, unique ones first.

    The merge on the device (GPUs), per batch; :func:`_merge_host` is the same merge on the host, across batches and
    devices (and the only one on CPUs, where XLA's sort is ~3x slower than NumPy's). The two must agree (tested).
    Neighbouring voxels light up the same pixels, so a batch has many duplicates (~100x for a grain). Merging them
    here means only the unique pixels go to the host. Returns fixed-size arrays plus the number of leading entries
    that are used.
    """
    frame, pixel, value = frame.ravel(), pixel.ravel(), value.ravel()
    keep = (frame >= 0) & (value >= min_value)
    last = jnp.iinfo(jnp.int32).max  # dropped entries sort to the end
    frame, pixel, value = jax.lax.sort(
        (jnp.where(keep, frame, last), jnp.where(keep, pixel, last), jnp.where(keep, value, 0.0)), num_keys=2
    )
    n = frame.shape[0]
    new = jnp.ones(n, bool).at[1:].set((frame[1:] != frame[:-1]) | (pixel[1:] != pixel[:-1])) & (frame != last)
    unique = jnp.cumsum(new) - 1  # index of each entry's unique (frame, pixel)
    first = jnp.where(new, unique, n)  # n is out of bounds, so dropped
    value = jnp.zeros_like(value).at[jnp.where(frame != last, unique, n)].add(value, mode="drop")
    frame = jnp.zeros_like(frame).at[first].set(frame, mode="drop")
    pixel = jnp.zeros_like(pixel).at[first].set(pixel, mode="drop")
    return frame, pixel, value, jnp.sum(new)


def _merge_host(
    frame: np.ndarray, pixel: np.ndarray, value: np.ndarray, npix: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum duplicate (frame, pixel)s on the host: unique ones, sorted by (frame, pixel).

    The host side of :func:`_merge` (same result), with NumPy's sort on one int64 key, frame x npix + pixel.

    Parameters
    ----------
    frame, pixel, value
        [N] contributions (all kept: :func:`_compact` or :func:`_merge` has dropped the rest)
    npix
        Pixels per frame (n_slow x n_fast)

    Returns
    -------
    frame, pixel, value: np.ndarray
        [M] unique (frame, pixel)s (int32) and their summed values
    """
    key = np.asarray(frame).astype(np.int64) * npix + np.asarray(pixel)
    order = np.argsort(key)
    key, value = key[order], np.asarray(value)[order]
    new = np.ones(key.size, bool)
    new[1:] = key[1:] != key[:-1]
    starts = np.flatnonzero(new)
    frame, pixel = np.divmod(key[starts], npix)
    return frame.astype(np.int32), pixel.astype(np.int32), np.add.reduceat(value, starts)


def make_row(omega: np.ndarray, dty: np.ndarray, transmission: np.ndarray | None = None) -> dict:
    """Build the row dict for :func:`render_row` from one scan's per-frame motor positions.

    Parameters
    ----------
    omega, dty
        [nframes] omega (degrees, frame centres) and dty for each frame, in file order
    transmission
        [nframes] optional transmission factor per frame (default 1)

    Returns
    -------
    dict
        Frames sorted by omega ("omega_sorted", "omega_edges", "dty_sorted", "transmission_sorted", and "order", each
        sorted frame's index in file order), and the row's range and middle in omega and dty ("omega_min",
        "omega_max", "omega_mid", "dty_min", "dty_max", "dty_mid")
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
        "omega_mid": float(0.5 * (edges[0] + edges[-1])),
        "dty_min": float(dty.min()),
        "dty_max": float(dty.max()),
        "dty_mid": float(0.5 * (dty.min() + dty.max())),
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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """:func:`render_peaks` then :func:`_merge` (GPUs) or :func:`_compact` (CPUs), split across the devices of ``mesh``.

    Pixels shared between devices or batches (and on CPUs, all duplicates) are merged on the host.
    """
    reduce = _compact if mesh.devices.flat[0].platform == "cpu" else _merge

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
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        frame, pixel, value, captured, in_range, _ = _render_peaks(e, h, b, ent, hk, f2, g, r, window, det_shape, None)
        value = jnp.where(lv[:, None], value, 0.0)
        frame, pixel, value, count = reduce(frame, pixel, value, mv)
        return frame, pixel, value, count[None], captured, in_range

    specs = (P("d"),) * 4 + (P(),) * 6
    return shard_map(local, mesh=mesh, in_specs=specs, out_specs=(P("d"),) * 6)(
        entry, hkl_idx, branch, live, entries, hkls, F2, geom, row, min_value
    )


def _valid_from_shards(x: jax.Array, count: np.ndarray, per_shard: int) -> np.ndarray:
    """Copy the first count[k] elements of each device's shard of x to the host.

    The slice on the device is rounded up to a power of 2 (at most the shard) and cut to count[k] on the host: a slice
    of each new length would compile again, and the counts change from batch to batch.
    """
    out = []
    for shard in sorted(x.addressable_shards, key=lambda sh: sh.index[0].start or 0):
        k = (shard.index[0].start or 0) // per_shard
        c = int(count[k])
        n = min(1 << max(c - 1, 0).bit_length(), per_shard)  # at most 2 x the elements needed
        data = shard.data if n == per_shard else shard.data[:n]
        out.append(np.asarray(data)[:c])
    return np.concatenate(out)


@jax.jit
def _omega_sigma(
    ubi: jax.Array,
    pos: jax.Array,
    hkls: jax.Array,
    e: jax.Array,
    h: jax.Array,
    b: jax.Array,
    geom: dict,
    row: dict,
    sig_rot: jax.Array | None = None,
) -> jax.Array:
    """Return the standard deviation in omega (degrees) of each peak, as render_peaks spreads it."""
    dty_row = row["dty_mid"]

    def one(ei: jax.Array, hi: jax.Array, bi: jax.Array) -> jax.Array:
        cov = _peak_cov(
            ubi[ei], pos[ei], hkls[hi], 1.0 - 2.0 * bi, dty_row, geom, None if sig_rot is None else sig_rot[ei]
        )
        return jnp.sqrt(cov[2] + geom.get("sig_omega", 0.0) ** 2 + _VAR_FLOOR[2])

    return jax.vmap(one)(e, h, b)


def _select_margin(window: tuple[int, int, int], row: dict, geom: dict, dtype: jnp.dtype) -> jax.Array:
    """Return the [5] margins for select_peaks.

    How far outside the row a centroid may be and still contribute, (slow, fast, omega) in pixels and degrees, and
    how far a voxel's centre may be from the beam's centre line, across it horizontally and vertically.
    """
    wo, ws, wf = window
    ostep = float(np.max(np.diff(np.asarray(row["omega_edges"]))))
    cos_alpha = float(beam_offsets(jnp.zeros(3), jnp.asarray(geom["k_in_lab"]))[3])  # the beam's tilt
    size = float(geom["voxel_size"])
    dty_range = float(row["dty_max"]) - float(row["dty_min"])  # frames of a row may have different dty
    across = float(geom["width_beam"]) / 2 + 4 * float(geom["sig_beam"]) + size + dty_range
    vertical = (float(geom["width_beam_v"]) / 2 + 4 * float(geom["sig_beam_v"])) / cos_alpha + size
    return jnp.array([ws // 2 + 1, wf // 2 + 1, (wo // 2 + 1) * ostep, across, vertical], dtype=dtype)


def _as_jax(entries: dict, hkls: ArrayLike, F2: ArrayLike, geom: dict, row: dict) -> tuple:
    """Convert the renderer's inputs to JAX arrays, in the dtype of the entries' UBIs."""
    ubi = jnp.asarray(entries["ubi"])
    dtype = ubi.dtype
    out = {
        "ubi": ubi,
        "pos": jnp.asarray(entries["pos"], dtype=dtype),
        "density": jnp.asarray(entries["density"], dtype=dtype),
    }
    if "sig_rot" in entries:  # only when given, so renders without it compile and cost the same as before
        out["sig_rot"] = jnp.broadcast_to(jnp.asarray(entries["sig_rot"], dtype=dtype), ubi.shape[:1])
    if "optical_depth" in entries:  # likewise: absorption only when asked for (add_absorption)
        out["optical_depth"] = jnp.asarray(entries["optical_depth"], dtype=jnp.float32)
    entries = out
    hkls, F2 = jnp.asarray(hkls, dtype=dtype), jnp.asarray(F2, dtype=dtype)
    return entries, hkls, F2, jax.tree.map(jnp.asarray, geom), jax.tree.map(jnp.asarray, row)


def row_peaks(
    entries: dict, hkls: jax.Array, geom: dict, row: dict, det_shape: tuple[int, int], window: tuple[int, int, int],
    max_frames: int | None = None, mesh: Mesh | None = None, select_chunk: int = 2**20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:  # fmt: skip
    """Return the peaks of one phase that can reach a dty row, and the window size each gets, as :func:`render_row`.

    Parameters
    ----------
    entries, hkls, geom, row
        As :func:`render_row` (JAX arrays, in the entries' dtype)
    det_shape, window, max_frames, mesh, select_chunk
        See :func:`render_row`

    Returns
    -------
    entry, hkl, branch: np.ndarray
        [P] int32 indices of the peaks: map entry, row of ``hkls``, and 0 / 1 for etasign +1 / -1
    peak_class: np.ndarray
        [P] index into classes of each peak's window
    classes: list
        Frames per window, smallest first (window[0], 2 window[0] + 1, ... up to max_frames)
    """
    mesh = anri.utils.mesh() if mesh is None else mesh
    nd = mesh.size
    ubi, pos = entries["ubi"], entries["pos"]
    dtype = ubi.dtype
    n_entries, n_hkls = ubi.shape[0], hkls.shape[0]
    wo, ws, wf = window
    classes = [wo]  # frames per window, smallest first
    if max_frames is not None:
        while 2 * classes[-1] + 1 <= max_frames:
            classes.append(2 * classes[-1] + 1)
    margin = _select_margin((classes[-1], ws, wf), row, geom, dtype)
    ce = max(nd, (select_chunk // n_hkls) // nd * nd)
    found = []
    for start in range(0, n_entries, ce):
        stop = min(start + ce, n_entries)
        idx = np.arange(start, start + ce) % n_entries  # padding repeats real entries, then is discarded
        mask = _select_sharded(ubi[idx], pos[idx], hkls, geom, row, margin, det_shape, mesh)
        e, h, b = np.nonzero(np.asarray(mask)[: stop - start])
        found.append((e + start, h, b))
    e = np.concatenate([f[0] for f in found]).astype(np.int32)
    h = np.concatenate([f[1] for f in found]).astype(np.int32)
    b = np.concatenate([f[2] for f in found]).astype(np.int32)
    n_peaks = e.size

    # Window size class of each peak: the fewest frames that hold +-3.5 sigma of it in omega
    peak_class = np.zeros(n_peaks, np.int32)
    if len(classes) > 1 and n_peaks > 0:
        ostep = float(np.median(np.diff(np.asarray(row["omega_edges"]))))
        chunk = 2**16  # fixed size: one compile
        pad = -n_peaks % chunk
        ehb = [jnp.asarray(np.pad(x, (0, pad))) for x in (e, h, b)]
        sig_frames = (
            np.concatenate(
                [
                    np.asarray(
                        _omega_sigma(
                            ubi, pos, hkls, *(x[i : i + chunk] for x in ehb), geom, row, entries.get("sig_rot")
                        )
                    )
                    for i in range(0, n_peaks + pad, chunk)
                ]
            )[:n_peaks]
            / ostep
        )
        needed = 2 * np.ceil(3.5 * sig_frames + 0.5) + 1
        peak_class = np.minimum(np.searchsorted(np.asarray(classes), needed), len(classes) - 1).astype(np.int32)

    return e, h, b, peak_class, classes


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
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Render all peaks of one phase that reach one dty row into sparse pixels.

    Work is split across the devices of ``mesh`` (default :func:`anri.utils.mesh`): all GPUs,
    or all XLA CPU devices set up by :func:`anri.utils.setup`.

    Parameters
    ----------
    entries
        Dict with "ubi" [N, 3, 3], "pos" [N, 3] (sample frame, same length units as dty)
        and "density" [N], for map entries of a single phase. Optionally "sig_rot" [N] (or a scalar): each entry's
        intrinsic orientation spread, the standard deviation (radians) of each component of a small sample-frame
        rotation vector, isotropic. It widens the entry's peaks in omega and on the detector through the same
        linearised propagation as the beam's spreads, so it suits spreads up to a few degrees. Widen ``window``
        (pixels) for spreads that move spots by several pixels. Optionally "optical_depth" [N, n_phi]
        (:func:`add_absorption`): absorption along the incoming and diffracted paths, at each frame's omega.
    hkls, F2
        [Nh, 3] hkls of that phase and [Nh] their structure factors squared
    geom
        Dict with "wavelength", "k_in_lab" [3], "wedge", "chi" (degrees), "y0",
        "s_step_lab", "f_step_lab", "det_origin_lab" [3] (from :func:`anri.geom.detector.detector_basis_vectors_lab`),
        "sig_wavelength", "sig_ky", "sig_kz", "sig_beam", "sig_psf" (detector point spread, pixels), "voxel_size"
        and "pol_factor"; optionally "sig_omega", an extra spread of every peak in omega (degrees, default 0)
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
    max_frames
        If set, peaks broad in omega get more frames: each peak's window has the fewest frames out of
        window[0], 2 window[0] + 1, ... (up to max_frames) that hold +-3.5 sigma of it in omega, and each size is
        rendered in its own batches. Default None: every peak gets window[0] frames.

    Returns
    -------
    frame, pixel, value: np.ndarray
        Sparse pixels sorted by (frame, pixel), with duplicates summed
    stats: dict
        "n_peaks" rendered; per peak, "captured" (the fraction of its Gaussian in its window, over the scan's frames)
        and "in_range" (the fraction of its omega profile within the scan's omega range: below 0.5 for the peaks
        selected beyond the ends of the scan); "window_frames" with max_frames. ``captured / in_range`` is the share
        of what the scan sees that the window holds
    """
    mesh = anri.utils.mesh() if mesh is None else mesh
    nd = mesh.size
    entries, hkls_j, F2_j, geom, row_j = _as_jax(entries, hkls, F2, geom, row)
    dtype = entries["ubi"].dtype

    # 1. Which peaks can reach this row, and the window size of each
    _, ws, wf = window
    e, h, b, peak_class, classes = row_peaks(entries, hkls_j, geom, row_j, det_shape, window, max_frames, mesh,
                                             select_chunk)  # fmt: skip
    n_peaks = e.size
    if n_peaks == 0:
        empty = np.zeros(0, np.int32)
        return empty, empty, np.zeros(0), {"n_peaks": 0, "captured": np.zeros(0), "in_range": np.zeros(0)}

    # 2. Render in fixed-size batches split over the devices, one window size at a time. Small problems use smaller
    # batches: 1024 peaks per device times a power of 4, so a scan compiles few shapes. Each compile takes seconds,
    # and XLA:GPU takes up to a minute for very small batches. Padding peaks are marked not live.
    min_value_j = jnp.asarray(min_value, dtype=dtype)
    frames, pixels, values = [], [], []
    captured_all, in_range_all = np.zeros(n_peaks), np.zeros(n_peaks)
    for k, wo_k in enumerate(classes):
        sel = np.flatnonzero(peak_class == k)
        if sel.size == 0:
            continue
        window_k = (wo_k, ws, wf)
        per_device = _MIN_BATCH
        while per_device < -(-sel.size // nd):
            per_device *= 4
        per_device = min(max(batch // nd, 1), per_device)
        batch_k = per_device * nd
        per_shard = per_device * wo_k * ws * wf
        for start in range(0, sel.size, batch_k):
            stop = min(start + batch_k, sel.size)
            pad = batch_k - (stop - start)
            idx = [np.pad(x[sel[start:stop]], (0, pad)) for x in (e, h, b)]
            live = np.arange(batch_k) < stop - start
            fr, px, val, count, cap, inr = _render_sharded(
                *idx, live, entries, hkls_j, F2_j, geom, row_j, min_value_j, window_k, det_shape, mesh
            )
            count = np.asarray(count)
            frames.append(_valid_from_shards(fr, count, per_shard))
            pixels.append(_valid_from_shards(px, count, per_shard))
            values.append(_valid_from_shards(val, count, per_shard))
            captured_all[sel[start:stop]] = np.asarray(cap)[: stop - start]
            in_range_all[sel[start:stop]] = np.asarray(inr)[: stop - start]

    # 3. Merge batches and devices: sort by (frame, pixel) and sum duplicates
    frame, pixel, value = _merge_host(
        np.concatenate(frames), np.concatenate(pixels), np.concatenate(values), det_shape[0] * det_shape[1]
    )
    stats = {"n_peaks": n_peaks, "captured": captured_all, "in_range": in_range_all}
    if len(classes) > 1:
        stats["window_frames"] = np.asarray(classes)[peak_class]
    return frame, pixel, value, stats


def add_absorption(
    phases: list, step: float, mu: ArrayLike | None = None, mu_map: ArrayLike | None = None, n_phi: int = 360,
    chunk: int = 2**18,
) -> list:  # fmt: skip
    """Give every map entry its optical depth to the sample's edge, so that :func:`render_row` models absorption.

    The sample is a slice of attenuation coefficients, the same above and below it (:mod:`anri.geom.paths`). It is
    built from the map itself (:func:`anri.geom.paths.attenuation_map`: every voxel holding entries is full, of its
    phases in proportion to their densities), one slice per layer of a 3D map; or given, e.g. from an absorption
    tomography, and then used for every layer. Diffracted rays leave the slice at an angle, so a 3D map's layers are
    each treated as a prism: paths through the layers above and below are not followed.

    Parameters
    ----------
    phases
        [(entries, hkls, F2), ...] one per phase, as :func:`render_phases` takes them
    step
        Voxel size of the map (and of mu_map), in the units of the entries' positions
    mu
        [n_phases] attenuation coefficient of each phase (1 / the units of step, as
        :func:`anri.crystal.structure.attenuation` gives it), to build the sample from the map
    mu_map
        [n, n] attenuation coefficients instead (1 / the units of step), reconstruction order, centred on the rotation
        axis (:func:`anri.geom.scan.recon_positions`)
    n_phi
        Directions in the slice: each entry gets n_phi float32 (1.4 kB for 360), so lower it for very large maps
    chunk
        Entries per call to :func:`anri.geom.paths.edge_distances` (a fixed size: one compile). Each call sums the
        whole slice along every direction, so fewer, larger calls are cheaper; a call holds [n_phi, chunk] float32

    Returns
    -------
    list
        The phases, each entries dict with "optical_depth" [N, n_phi] added
    """
    if (mu is None) == (mu_map is None):
        msg = "give mu (one per phase) or mu_map, not both"
        raise ValueError(msg)
    pos = [np.asarray(e["pos"], float) for e, _, _ in phases]
    allpos = np.concatenate(pos)
    z = allpos[:, 2] if allpos.shape[1] > 2 else np.zeros(len(allpos))
    layer = np.round(z / step).astype(int)
    if mu_map is None:
        mu = np.broadcast_to(np.asarray(mu, float), (len(phases),))
        mus = np.concatenate([np.full(len(p), m) for p, m in zip(pos, mu)])
        dens = np.concatenate([np.asarray(e["density"], float) for e, _, _ in phases])
        n = attenuation_map(allpos, mus, dens, step).shape[0]  # one grid for every layer: one compile
    depth = np.zeros((len(allpos), n_phi), np.float32)
    largest = int(np.unique(layer, return_counts=True)[1].max())
    chunk = min(chunk, max(1024, 1 << (largest - 1).bit_length()))  # small maps: a small fixed size
    for k in np.unique(layer):
        sel = np.flatnonzero(layer == k)
        m = mu_map if mu_map is not None else attenuation_map(allpos[sel], mus[sel], dens[sel], step, n)
        m = jnp.asarray(m, jnp.float32)
        for a in range(0, sel.size, chunk):
            idx = sel[a : a + chunk]
            p = np.zeros((chunk, 2), np.float32)
            p[: idx.size] = allpos[idx, :2]
            depth[idx] = np.asarray(edge_distances(m, step, jnp.asarray(p), n_phi))[: idx.size]
    starts = np.cumsum([0] + [len(p) for p in pos])
    return [({**e, "optical_depth": depth[a:b]}, h, f) for (e, h, f), a, b in zip(phases, starts[:-1], starts[1:])]


def render_phases(
    phases: list,
    geom: dict,
    row: dict,
    det_shape: tuple[int, int],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Render the peaks of several phases that reach one dty row, summed into one set of sparse pixels.

    Each phase is rendered by :func:`render_row`; where phases light the same pixel, their intensities add.

    Parameters
    ----------
    phases
        [(entries, hkls, F2), ...] one per phase, as :func:`render_row` takes them
    geom, row, det_shape
        See :func:`render_row`
    **kwargs
        Passed on to :func:`render_row` (window, max_frames, batch, ...)

    Returns
    -------
    frame, pixel, value, stats
        As :func:`render_row`, over every phase: "n_peaks" summed, "captured", "in_range" (and "window_frames")
        concatenated, phase by phase
    """
    out = [render_row(entries, hkls, F2, geom, row, det_shape, **kwargs) for entries, hkls, F2 in phases]
    if len(out) == 1:
        return out[0]
    frame, pixel, value = _merge_host(np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]),
                                      np.concatenate([o[2] for o in out]), det_shape[0] * det_shape[1])  # fmt: skip
    stats = {"n_peaks": sum(o[3]["n_peaks"] for o in out)}
    for k in ("captured", "in_range"):
        stats[k] = np.concatenate([o[3][k] for o in out])
    if all("window_frames" in o[3] for o in out):
        stats["window_frames"] = np.concatenate([o[3]["window_frames"] for o in out])
    return frame, pixel, value, stats


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
    every sample through the forward model (with the voxel at its real position), adds the detector point spread
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
    margin = _select_margin(window, row, geom, dtype)
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
        return _centroid(u, p, q, etasign, wavelength, ky, kz, dty_row, geom)[0]

    dty_row = row["dty_mid"]
    sample = jax.jit(jax.vmap(centroid, in_axes=(None, None, None, None, 0, 0, 0)))
    sd_beam = np.array([float(geom["sig_wavelength"]), float(geom["sig_ky"]), float(geom["sig_kz"])])
    psf2 = float(geom["sig_psf"]) ** 2
    sd_extra = np.sqrt(
        [psf2 + _VAR_FLOOR[0], psf2 + _VAR_FLOOR[1], float(geom.get("sig_omega", 0.0)) ** 2 + _VAR_FLOOR[2]]
    )
    sorted_index = np.argsort(np.asarray(row["order"]))  # file-order frame -> sorted-omega frame
    edges_all = np.asarray(row["omega_edges"])
    omega_mid = row["omega_mid"]
    keep, err, cap_mc = [], [], []
    for i in np.flatnonzero(whole):
        spreads = rng.standard_normal((n_samples, 3)) * sd_beam
        wavelength = float(geom["wavelength"]) + spreads[:, 0]
        x = np.array(
            sample(ubi[e[i]], pos[e[i]], hkls_j[h[i]], 1.0 - 2.0 * b[i], wavelength, spreads[:, 1], spreads[:, 2])
        )
        x += rng.standard_normal((n_samples, 3)) * sd_extra
        x[:, 2] = nearest_omega(x[:, 2], omega_mid)
        jo = sorted_index[frame[i, 0, 0, 0]]
        in_frame = np.searchsorted(edges_all[jo : jo + wo + 1], x[:, 2]) - 1  # which window frame each sample is in
        mc = np.zeros((wo, ws, wf))
        for j in range(wo):  # each frame has its own pixel window
            s0, f0 = divmod(int(pixel[i, j, 0, 0]), det_shape[1])
            cells = [s0 - 0.5 + np.arange(ws + 1), f0 - 0.5 + np.arange(wf + 1)]
            mc[j] = np.histogram2d(x[in_frame == j, 0], x[in_frame == j, 1], bins=cells)[0] / n_samples
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
    if per_peak <= 0:  # e.g. jaxlib 0.4.28 on macOS reports the same sizes for any batch
        msg = "XLA's memory analysis doesn't scale with the batch on this backend: pass batch to render_row yourself"
        raise RuntimeError(msg)
    fixed = total1 - 1024 * per_peak
    free, host = _free_memory(list(mesh.devices.flat))
    if host:
        per_peak += (out2 - out1) / 1024  # render_row copies each batch's output to the host
    per_device = int((memory_fraction * free - fixed) // per_peak)
    per_device = 1 << max(6, per_device.bit_length() - 1)  # largest power of two that fits, at least 64
    return per_device * nd
