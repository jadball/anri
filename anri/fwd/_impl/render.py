"""Render peaks from a voxel map into sparse detector pixels, one dty row (one rotation scan) at a time.

A map is a flat list of entries (position, UBI, density), so several orientations per voxel are allowed.
Each peak is an (entry, hkl, branch) triple. Its (slow, fast, omega) centroid is found with the voxel at its
real position for the row's dty, and its covariance by propagating the beam's spreads with
:func:`anri.fwd._impl.base.make_propagator`.

The beam is described by its profile across it, horizontally and vertically: each a flat top of some width,
blurred by a Gaussian (a pure Gaussian with zero width). A pencil beam is narrow both ways, a line beam wide one
way, a box beam wide both ways. Voxels are either columns (2D maps: the beam's vertical profile integrates out)
or cubes (3D maps). See :func:`beam_weight`.

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
from anri.geom import beam_basis, raytrace_to_det, sample_to_lab

from .base import hkl_to_k_omega, hkl_to_k_omega_both, make_propagator

# Smallest batch per device that render_row compiles (see there)
_MIN_BATCH = 1024

# Elements of the (sc, fc, omega) covariance that the renderer uses
_COV_ELEMS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

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


def _ramp_integral2(x: ArrayLike, sigma: ArrayLike) -> jax.Array:
    """Antiderivative of :func:`_ramp_integral`: (x^2 + sigma^2) / 2 * Phi(x / sigma) + x sigma / 2 * phi(x / sigma)."""
    x, sigma = jnp.asarray(x), jnp.asarray(sigma)
    return 0.5 * (x**2 + sigma**2) * ndtr(x / sigma) + 0.5 * x * sigma * jnp.asarray(norm.pdf(x / sigma))


def _smoothed(order: int, x: ArrayLike, width: ArrayLike, sigma: ArrayLike) -> jax.Array:
    """Antiderivative ``order`` of the beam profile: a flat top of ``width`` blurred by a Gaussian of ``sigma``.

    Order 1 is the profile's cumulative distribution, order 2 its integral. The profile integrates to 1. For a
    width much smaller than sigma it is the Gaussian's (the difference quotient below would lose precision).
    """
    x, width, sigma = jnp.asarray(x), jnp.asarray(width), jnp.asarray(sigma)
    gauss = width < 0.01 * sigma
    w = jnp.where(gauss, 1.0, width)  # double-where: no NaN gradients from either branch
    if order == 1:
        flat_top = (_ramp_integral(x + w / 2, sigma) - _ramp_integral(x - w / 2, sigma)) / w
        return jnp.where(gauss, ndtr(x / sigma), flat_top)
    flat_top = (_ramp_integral2(x + w / 2, sigma) - _ramp_integral2(x - w / 2, sigma)) / w
    return jnp.where(gauss, _ramp_integral(x, sigma), flat_top)


def _chord_weight(
    u: ArrayLike, omega: ArrayLike, voxel_size: ArrayLike, sigma: ArrayLike, width: ArrayLike
) -> jax.Array:
    """Beam profile integrated over a square voxel, per unit height, for a beam along lab x.

    The voxel is a square of side ``voxel_size`` rotated by ``omega`` (degrees) and ``u`` across the beam's centre
    (lab y). Its chord along the beam, as a function of lab y, is a trapezoid of area ``voxel_size**2``.
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
    s2 = partial(_smoothed, 2, width=width, sigma=sigma)
    trapezoid = height / ab * (s2(u + a) - s2(u + b) - s2(u - b) + s2(u - a))
    box = height * (_smoothed(1, u + a, width, sigma) - _smoothed(1, u - a, width, sigma))
    return jnp.where(is_box, box, trapezoid)


def _beam_offsets(pos_lab: jax.Array, geom: dict) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return a point's horizontal distance across the beam, its height above the beam, and the beam's tilt cosine.

    The beam's centre line passes through the lab origin along ``geom["k_in_lab"]``.
    """
    k, e_h, _ = beam_basis(geom["k_in_lab"])
    cos_alpha = jnp.hypot(k[0], k[1])
    along = (pos_lab[0] * k[0] + pos_lab[1] * k[1]) / cos_alpha  # horizontal distance along the beam
    return pos_lab @ e_h, pos_lab[2] - k[2] / cos_alpha * along, cos_alpha


def beam_weight(pos_lab: jax.Array, omega: ArrayLike, geom: dict) -> jax.Array:
    """How much of a voxel the beam illuminates: its profile integrated over the voxel.

    The beam's profile across it, horizontally (``geom["width_beam"]``, ``geom["sig_beam"]``) and vertically
    (``geom["width_beam_v"]``, ``geom["sig_beam_v"]``), is a flat top of that width blurred by a Gaussian of that
    standard deviation, and integrates to 1: a wider beam spreads the same flux. A voxel is a square of side
    ``geom["voxel_size"]`` in the sample xy plane, rotated by ``omega``, centred at ``pos_lab``, and either:

    - a column (2D map, ``geom["voxel_3d"]`` false): the whole vertical profile crosses it, so it integrates out;
      a beam tilted by ``alpha`` out of the horizontal plane travels ``1 / cos(alpha)`` further through it.
    - a cube (3D map): the vertical profile is integrated over the cube's height, measured vertically. For a
      tilted beam this treats the horizontal and vertical directions separately, which is exact only when it is
      horizontal.

    A beam at ``psi`` from lab x in the horizontal plane sees the voxel rotated by ``omega - psi``.

    Parameters
    ----------
    pos_lab
        [3] Lab position of the voxel's centre (in that frame, with its dty)
    omega
        Omega angle (degrees)
    geom
        Geometry dict, see :func:`render_row`
    """
    k = beam_basis(geom["k_in_lab"])[0]
    u, dz, cos_alpha = _beam_offsets(pos_lab, geom)
    size = geom["voxel_size"]
    psi = jnp.degrees(jnp.arctan2(k[1], k[0]))
    w = _chord_weight(u, omega - psi, size, geom["sig_beam"], geom["width_beam"]) / cos_alpha
    sig_v = jnp.where(geom["voxel_3d"], geom["sig_beam_v"], 1.0) / cos_alpha  # unused for columns
    width_v = geom["width_beam_v"] / cos_alpha
    v = _smoothed(1, dz + size / 2, width_v, sig_v) - _smoothed(1, dz - size / 2, width_v, sig_v)
    return w * jnp.where(geom["voxel_3d"], v, 1.0)


def lorentz(k_in: jax.Array, k_out: jax.Array, rot_axis: jax.Array) -> jax.Array:
    """Lorentz factor for rotation about ``rot_axis``.

    It is ``1 / |rot_axis . (k_in x k_out)|``, equal to 1 / :func:`ImageD11.refinegrains.lf` = ``1 / (sin(2theta) |sin(eta)|)`` for a vertical rotation axis.
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


def _scattering_origin(pos: jax.Array, omega: ArrayLike, dty: ArrayLike, geom: dict) -> jax.Array:
    """Where a voxel scatters, in the lab: its real position at this omega and dty.

    For a column (2D map), at the height of the beam where it crosses the column: the beam is taken to cross the
    rotation axis at the voxel's own height, so a voxel's height never depends on other layers.
    """
    lab = sample_to_lab(pos, omega, geom["wedge"], geom["chi"], dty, geom["y0"])
    k = beam_basis(geom["k_in_lab"])[0]
    cos_alpha = jnp.hypot(k[0], k[1])
    rise = k[2] / cos_alpha * (lab[0] * k[0] + lab[1] * k[1]) / cos_alpha
    return lab + jnp.where(geom["voxel_3d"], 0.0, 1.0) * jnp.array([0.0, 0.0, 1.0]) * rise


def _centroid(
    ubi: jax.Array, pos: jax.Array, hkl: jax.Array, etasign: ArrayLike, wavelength: ArrayLike, ky: ArrayLike,
    kz: ArrayLike, dty: ArrayLike, geom: dict,
) -> tuple[jax.Array, jax.Array]:  # fmt: skip
    """(sc, fc, omega) of a peak, with the voxel at its real position for this dty, and whether it diffracts."""
    _, k_out, omega, valid = hkl_to_k_omega(
        ubi, hkl, etasign, wavelength, geom["k_in_lab"], ky, kz, geom["wedge"], geom["chi"]
    )
    origin = _scattering_origin(pos, omega, dty, geom)
    sc, fc = raytrace_to_det(k_out, origin, geom["s_step_lab"], geom["f_step_lab"], geom["det_origin_lab"])
    return jnp.array([sc, fc, omega]), valid


# Covariance of (sc, fc, omega) from the wavelength, ky and kz spreads. The voxel's extent and the beam profile
# enter through beam_weight instead.
_propagate = make_propagator(_centroid, argnums=(4, 5, 6), has_aux=True, out_elems=_COV_ELEMS)


_CROSS_GENERATORS = np.array(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
)  # [r]x = sum_i r_i G_i


def _centroid_rotated(
    ubi: jax.Array, pos: jax.Array, hkl: jax.Array, etasign: ArrayLike, rotvec: jax.Array, dty: ArrayLike, geom: dict
) -> tuple[jax.Array, jax.Array]:
    """Return the centroid with the lattice turned by a small sample-frame rotation (rotation vector, radians).

    UB -> R UB, so UBI -> UBI R^T, with R = I + [rotvec]x: exact to first order, which is all the propagation uses.
    """
    # [rotvec]x as a sum over generators, not a stack: under shard_map the rotation's tangent varies across devices
    # while a zeros_like does not, and newer JAX refuses to stack the two
    cross = jnp.tensordot(rotvec, jnp.asarray(_CROSS_GENERATORS, rotvec.dtype), 1)
    return _centroid(
        ubi @ (jnp.eye(3, dtype=ubi.dtype) + cross).T, pos, hkl, etasign, geom["wavelength"], 0.0, 0.0, dty, geom
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
        [5] how far outside the row a centroid may be and still contribute: (sc, fc, omega), and how far a voxel
        may be from the beam's centre line, across it horizontally and vertically (vertically only for cubes)
    det_shape
        (n_slow, n_fast) detector shape

    Returns
    -------
    mask: jax.Array
        [Ne, Nh, 2] bool, branch 0 is etasign +1
    """
    dty = 0.5 * (row["dty_min"] + row["dty_max"])

    def one(u: jax.Array, p: jax.Array, h: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        _, k_outs, omegas, valid = hkl_to_k_omega_both(
            u, h, geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, geom["wedge"], geom["chi"]
        )
        cens, near_h, near_v = [], [], []
        for i in range(2):
            origin = _scattering_origin(p, omegas[i], dty, geom)
            sc, fc = raytrace_to_det(k_outs[i], origin, geom["s_step_lab"], geom["f_step_lab"], geom["det_origin_lab"])
            cens.append(jnp.array([sc, fc, omegas[i]]))
            across, height, _ = _beam_offsets(
                sample_to_lab(p, omegas[i], geom["wedge"], geom["chi"], dty, geom["y0"]), geom
            )
            # within the omega margin an off-axis voxel moves across the beam by up to |p_xy| x margin (radians)
            near_h.append(jnp.abs(across) < margin[3] + jnp.hypot(p[0], p[1]) * jnp.radians(margin[2]))
            near_v.append(jnp.abs(height) < margin[4])
        return jnp.stack(cens), valid, jnp.stack(near_h), jnp.stack(near_v)

    centroids, valid, near_h, near_v = jax.vmap(jax.vmap(one, in_axes=(None, None, 0)), in_axes=(0, 0, None))(
        ubi, pos, hkls
    )
    sc, fc, om = centroids[..., 0], centroids[..., 1], centroids[..., 2]
    om = _wrap_omega(om, 0.5 * (row["omega_min"] + row["omega_max"]))
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
    fr, px, val, cap, _ = _render_peaks(
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
    return _render_peaks(entry, hkl_idx, branch, entries, hkls, F2, geom, row, window, det_shape, None)[4]


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
    """:func:`render_peaks`, also returning the window origins used."""
    wo, ws, wf = window
    ns, nf = det_shape

    def one(e: jax.Array, h: jax.Array, br: jax.Array, org: tuple | None) -> tuple:
        ubi, pos, hkl = entries["ubi"][e], entries["pos"][e], hkls[h]
        etasign = 1.0 - 2.0 * br
        dty_row = 0.5 * (row["dty_min"] + row["dty_max"])
        centroid, valid = _centroid(ubi, pos, hkl, etasign, geom["wavelength"], 0.0, 0.0, dty_row, geom)
        cov = _peak_cov(ubi, pos, hkl, etasign, dty_row, geom, entries["sig_rot"][e] if "sig_rot" in entries else None)
        # detector point spread adds to the slow and fast variances (pixels^2)
        psf2 = geom["sig_psf"] ** 2
        om2 = geom.get("sig_omega", 0.0) ** 2  # optional extra omega spread (degrees), e.g. to smooth a refinement
        ss, ff, oo = cov[0] + psf2 + _VAR_FLOOR[0], cov[1] + psf2 + _VAR_FLOOR[1], cov[2] + om2 + _VAR_FLOOR[2]
        sf, so, fo = cov[3], cov[4], cov[5]
        mu_s, mu_f, mu_o = centroid
        omega_peak = mu_o  # before wrapping
        mu_o = _wrap_omega(mu_o, 0.5 * (row["omega_min"] + row["omega_max"]))

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
        captured = jnp.sum(frac)

        # per-frame factors: the beam's profile over the voxel at that frame's dty, and transmission
        # at the mean omega of the peak's mass within each frame: a peak spread in omega (e.g. by sig_rot) diffracts
        # at different omegas, where an off-axis voxel sits at a different place across the beam
        om_fr = omega_peak + d_o  # [wo]
        lab0 = jax.vmap(lambda o: sample_to_lab(pos, o, geom["wedge"], geom["chi"], geom["y0"], geom["y0"]))(om_fr)
        lab = lab0 + jnp.array([0.0, 1.0, 0.0]) * (row["dty_sorted"][fclip] - geom["y0"])[:, None]  # [wo, 3]
        w_beam = jax.vmap(beam_weight, in_axes=(0, 0, None))(lab, om_fr, geom)
        per_frame = w_beam * row["transmission_sorted"][fclip]  # [wo]

        amp = entries["density"][e] * F2[h] * _peak_factors(ubi, hkl, etasign, geom)
        value = amp * per_frame[:, None, None] * frac
        use = inside & valid
        value = jnp.where(use, value, 0.0)
        frame_out = jnp.where(use, row["order"][fclip][:, None, None], -1)
        pixel = rows[:, :, None] * nf + cols[:, None, :]
        pixel_out = jnp.where(use, pixel, 0)
        return frame_out.ravel(), pixel_out.ravel(), value.ravel(), captured, (jo, i0, j0)

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
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        frame, pixel, value, captured = render_peaks(e, h, b, ent, hk, f2, g, r, window, det_shape)
        value = jnp.where(lv[:, None], value, 0.0)
        frame, pixel, value, count = reduce(frame, pixel, value, mv)
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
    dty_row = 0.5 * (row["dty_min"] + row["dty_max"])

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
    k = np.asarray(geom["k_in_lab"], dtype=float)
    cos_alpha = np.hypot(k[0], k[1]) / np.linalg.norm(k)
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
    entries = out
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
        (pixels) for spreads that move spots by several pixels.
    hkls, F2
        [Nh, 3] hkls of that phase and [Nh] their structure factors squared
    geom
        Dict with "wavelength", "k_in_lab" [3], "wedge", "chi" (degrees), "y0",
        "s_step_lab", "f_step_lab", "det_origin_lab" [3] (from :func:`anri.geom.detector_basis_vectors_lab`),
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

    # Window size class of each peak: the fewest frames that hold +-3.5 sigma of it in omega
    peak_class = np.zeros(n_peaks, np.int32)
    if len(classes) > 1:
        ostep = float(np.median(np.diff(np.asarray(row["omega_edges"]))))
        chunk = 2**16  # fixed size: one compile
        pad = -n_peaks % chunk
        ehb = [jnp.asarray(np.pad(x, (0, pad))) for x in (e, h, b)]
        sig_frames = (
            np.concatenate(
                [
                    np.asarray(
                        _omega_sigma(
                            ubi, pos, hkls_j, *(x[i : i + chunk] for x in ehb), geom, row_j, entries.get("sig_rot")
                        )
                    )
                    for i in range(0, n_peaks + pad, chunk)
                ]
            )[:n_peaks]
            / ostep
        )
        needed = 2 * np.ceil(3.5 * sig_frames + 0.5) + 1
        peak_class = np.minimum(np.searchsorted(np.asarray(classes), needed), len(classes) - 1).astype(np.int32)

    # 2. Render in fixed-size batches split over the devices, one window size at a time. Small problems use smaller
    # batches: 1024 peaks per device times a power of 4, so a scan compiles few shapes. Each compile takes seconds,
    # and XLA:GPU takes up to a minute for very small batches. Padding peaks are marked not live.
    min_value_j = jnp.asarray(min_value, dtype=dtype)
    frames, pixels, values = [], [], []
    captured_all = np.zeros(n_peaks)
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
            fr, px, val, count, cap = _render_sharded(
                *idx, live, entries, hkls_j, F2_j, geom, row_j, min_value_j, window_k, det_shape, mesh
            )
            count = np.asarray(count)
            frames.append(_valid_from_shards(fr, count, per_shard))
            pixels.append(_valid_from_shards(px, count, per_shard))
            values.append(_valid_from_shards(val, count, per_shard))
            captured_all[sel[start:stop]] = np.asarray(cap)[: stop - start]

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
    stats = {"n_peaks": n_peaks, "captured": captured_all}
    if len(classes) > 1:
        stats["window_frames"] = np.asarray(classes)[peak_class]
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

    dty_row = 0.5 * (row["dty_min"] + row["dty_max"])
    sample = jax.jit(jax.vmap(centroid, in_axes=(None, None, None, None, 0, 0, 0)))
    sd_beam = np.array([float(geom["sig_wavelength"]), float(geom["sig_ky"]), float(geom["sig_kz"])])
    psf2 = float(geom["sig_psf"]) ** 2
    sd_extra = np.sqrt(
        [psf2 + _VAR_FLOOR[0], psf2 + _VAR_FLOOR[1], float(geom.get("sig_omega", 0.0)) ** 2 + _VAR_FLOOR[2]]
    )
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
