"""Occupancy of orientations per voxel, fitted to the histogrammed data by MLEM.

The data are linear in the sample's density over position and orientation: ``d = A f``, with ``f[v, q]`` the
occupancy of orientation q in voxel v. ``A`` puts each predicted reflection of q at its (eta, omega) and in the dty
row where voxel v sits at that omega. Fitting every voxel jointly handles overlap between voxels on the same rays.

Occupancies are sparse: each voxel keeps K candidate orientations (``f [Nv, K]`` and ``cand [Nv, K]``, indices into
the orientation list), and the work runs over blocks of voxels, so memory is set by the block size, not by the
number of voxels times orientations.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from anri.geom.beam import beam_basis, chord_weight
from anri.geom.gonio import find_dty_for_beam_xy, wrap_omega
from anri.geom.paths import edge_distances, ray_directions, transmission
from anri.geom.scan import recon_positions

from .data import coarsen_rows, coarsen_scan, eta_omega_corners, omega_bin, scan_beam


@jax.jit
def censored_ratio(d: jax.Array, mu: jax.Array, censor: float, saturated: jax.Array | None = None) -> jax.Array:
    """Compute the MLEM ratio d / mu, with empty bins taken as censored, and saturated bins as at least their data.

    A bin with no data had every pixel below the segmentation cut. Where the model predicts less than ``censor``
    there, the data cannot rule it out: the bin counts as agreeing (ratio 1). Where it predicts more, it counts as
    ``censor / 2``. ``censor = 0``: empty bins are zeros (ratio 0), plain MLEM. A bin holding a saturated pixel
    (``saturated``: their indices) recorded at most its true counts: where the model predicts at least the data
    there, it agrees (ratio 1); where it predicts less, the data pull it up as usual.
    """
    mus = jnp.maximum(mu, 1e-30)
    r = jnp.where(mu > 0, jnp.where(d > 0, d, jnp.where(mu < censor, mu, 0.5 * censor)) / mus, 0.0)
    if saturated is not None:
        r = r.at[saturated].set(jnp.where(mu[saturated] >= d[saturated], 1.0, r[saturated]))
    return r


@jax.jit
def deviance(d: jax.Array, mu: jax.Array, censor: float, saturated: jax.Array | None = None) -> jax.Array:
    """Poisson deviance of observed bins, plus 2 x (mu - censor) for empty bins the model predicts above ``censor``.

    ``censor = 0``: the plain Poisson deviance. Saturated bins (``saturated``: their indices) where the model predicts
    at least the data add nothing (see :func:`censored_ratio`).
    """
    mus = jnp.maximum(mu, 1e-30)
    obs = jnp.where(d > 0, d * jnp.log(jnp.maximum(d, 1e-30) / mus) - d + mu, 0.0)
    total = jnp.sum(obs + jnp.where(d > 0, 0.0, jnp.maximum(mu - censor, 0.0)))
    if saturated is not None:
        total = total - jnp.sum(jnp.where(mu[saturated] >= d[saturated], obs[saturated], 0.0))
    return 2.0 * total


def system(
    eta: jax.Array,
    om: jax.Array,
    use: jax.Array,
    w: jax.Array,
    ring_j: jax.Array,
    pos: jax.Array,
    scan: dict,
    b_e: float,
    b_o: float,
    n_e: int,
    n_o: int,
    n_beam: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Entries of the system matrix: (cell index, weight) for voxels and predictions.

    A prediction (eta, omega) is spread bilinearly over the 2 x 2 nearest (eta, omega) bins, and over the dty rows by
    where the voxel sits at that omega (lab y = x sin(omega) + y cos(omega)). Parallax is ignored.

    - ``n_beam = 0``: linearly over the 2 nearest rows.
    - ``n_beam > 0``: over the n_beam rows nearest to the voxel, each weighted by the beam's profile across dty
      integrated over the voxel's chord at that omega (as :func:`anri.geom.beam.beam_weight`, for a beam along lab x):
      a Gaussian of ``scan["sig_beam"]`` blurring a flat top of ``scan["width_beam"]``, over a square voxel of
      side ``scan["voxel"]``. The weights are scaled to sum to about 1 over the rows. Each weight is evaluated at
      the prediction's omega, so a voxel off the rotation axis moves across the beam with omega.

    Parameters
    ----------
    eta, om, use, w
        [1 or Nv, Q, Nj] predictions (degrees), whether to use them, and their weights. A leading 1 means the same
        orientations for every voxel; Nv, each voxel's own.
    ring_j
        [Nj] ring of each prediction
    pos
        [Nv, 3] voxel positions in the sample frame; or [Nv, 3 + n_phi], each followed by its distances to the
        sample's edge (:func:`anri.geom.paths.edge_distances`): every weight is then scaled by the transmission along
        the incoming and the diffracted paths (:func:`anri.geom.paths.transmission`), with ``scan["mu"]`` and
        ``scan["ring_tth"]``
    scan
        "y0", "dty0", "ystep", "n_rows" and "om0": the rows are at dty = dty0 + k ystep; optionally "ddty" [n_rows,
        n_o], each row's offset from that per omega bin (fly, helical scans: :func:`anri.index.data.dty_offsets`), and
        "exposure" [n_rows, n_o], each (row, omega bin)'s frames relative to a full one; "coverage" [n_rings, n_e],
        the measured share of each (ring, eta bin) (:func:`anri.index.data.detector_coverage`); with a beam profile also
        "sig_beam", "width_beam" and "voxel" (:func:`beam_rows`)
    b_e, b_o, n_e, n_o
        Bin widths and counts of the histogram in eta and omega
    n_beam
        Rows per prediction for the beam profile (static); 0 for the 2-row linear model

    Returns
    -------
    index, weight: jax.Array
        [Nv, Q, Nj, 8] each (4 n_beam with a beam profile); unused corners have index -1
    """
    n_k = scan["n_rows"]
    real = jnp.isfinite(pos[:, 2])  # padding (:func:`pad_voxels`) has no entries
    use = use & real[:, None, None]
    pos = jnp.where(real[:, None], pos, 0.0)
    k_in, wedge, chi = scan_beam(scan)
    if pos.shape[1] > 3:  # absorption: each voxel's edge distances follow its position
        k, e_h, e_v = beam_basis(k_in)
        tth = jnp.radians(jnp.asarray(scan["ring_tth"]))[ring_j][..., None]
        er = jnp.radians(eta)[..., None]
        k_out = jnp.cos(tth) * k + jnp.sin(tth) * (-jnp.sin(er) * e_h + jnp.cos(er) * e_v)  # [1 or Nv, Q, Nj, 3]
        d_in, d_out = ray_directions(k_in, k_out, om, wedge, chi)
        w = w * transmission(pos[:, 3:], d_in, d_out, scan["mu"])
        pos = pos[:, :3]
    dty = find_dty_for_beam_xy(pos[:, None, None, :], k_in, om, wedge, chi, scan["y0"])
    fk = (dty - scan["dty0"]) / scan["ystep"]  # the row whose beam is centred on the voxel
    if "ddty" in scan:  # rows not at their nominal dty (fly, helical scans): the beam's real dty in this omega bin
        io = jnp.clip(omega_bin(wrap_omega(om, scan["om0"]), scan["om0"], b_o), 0, n_o - 1)
        kh = jnp.clip(jnp.round(fk).astype(jnp.int32), 0, n_k - 1)
        fk = fk - scan["ddty"][kh, io] / scan["ystep"]
    if n_beam == 0:
        k0 = jnp.floor(fk).astype(jnp.int32)
        tk = fk - k0
        rows = ((0, 1.0 - tk), (1, tk))
    else:
        k0 = jnp.round(fk).astype(jnp.int32) - n_beam // 2
        norm = scan["ystep"] / scan["voxel"] ** 2  # the chord weight integrates to voxel^2 over dty
        psi = jnp.degrees(jnp.arctan2(k_in[1], k_in[0]))  # the beam sees the voxel turned by omega - psi
        rows = tuple(
            (dk, norm * chord_weight((k0 + dk - fk) * scan["ystep"], om - psi, scan["voxel"], scan["sig_beam"],
                                      scan["width_beam"]))
            for dk in range(n_beam)
        )  # fmt: skip
    idx, wt = [], []
    for ie, io, we, wo, inside in eta_omega_corners(eta, om, scan["om0"], b_e, b_o, n_e, n_o):
        if "coverage" in scan:  # the share of the bin's detector area that is measured
            we = we * scan["coverage"][ring_j[None, None, :], ie]
        for dk, wk in rows:
            kk = k0 + dk
            good = use & inside & (kk >= 0) & (kk < n_k)
            if "exposure" in scan:  # frames in this (row, omega bin), relative to a full one
                wk = wk * scan["exposure"][jnp.clip(kk, 0, n_k - 1), jnp.clip(io, 0, n_o - 1)]
            cell = ((ring_j[None, None, :] * n_e + ie) * n_o + io) * n_k + kk
            idx.append(jnp.where(good, cell, -1))
            wt.append(jnp.where(good, w * we * wo * wk, 0.0))
    return jnp.stack(idx, -1), jnp.stack(wt, -1)


def beam_rows(scan: dict) -> int:
    """Rows a voxel's beam profile reaches (odd): the voxel's half diagonal, half the flat top and 3 sigma each side.

    Parameters
    ----------
    scan
        "ystep", "sig_beam" (standard deviation of the beam across dty), "width_beam" (its flat top) and "voxel"

    Returns
    -------
    int
        n_beam for :func:`system`, e.g. as the fifth element of ``dims``
    """
    reach = 3.0 * scan["sig_beam"] + 0.5 * scan["width_beam"] + scan["voxel"] / np.sqrt(2.0)
    return 2 * int(np.ceil(reach / scan["ystep"])) + 1


def block_voxels(q: int, n_j: int, budget_bytes: float, n_corners: int = 8) -> int:
    """Voxels per block (a power of 2) so that one block's system entries for q orientations take ~budget_bytes.

    Counts ~32 bytes per corner of each (voxel, orientation, prediction): index and weight, temporaries and gathers.
    8 corners for the 2-row model, 4 n_beam with a beam profile (:func:`system`).

    Parameters
    ----------
    q
        Orientations per voxel handled at once
    n_j
        Predictions per orientation
    budget_bytes
        Memory for one block
    n_corners
        System entries per prediction

    Returns
    -------
    int
        Voxels per block
    """
    return int(2 ** max(0, np.floor(np.log2(budget_bytes / (32 * n_corners * q * n_j)))))


def pad_voxels(pos: ArrayLike, vb: int) -> jax.Array:
    """Pad voxel positions to a multiple of vb with voxels at z = NaN, which :func:`system` gives no entries.

    Not voxels far away: one at (1e6, 1e6) is on the axis's ray at omega = 135 and 315 degrees, and takes intensity.

    Parameters
    ----------
    pos
        [Nv, C] positions (C = 3, or 3 + n_phi with edge distances, see :func:`system`)
    vb
        Block size

    Returns
    -------
    jax.Array
        [ceil(Nv / vb) vb, C] positions
    """
    pos = jnp.asarray(pos, jnp.float32)
    pad = jnp.zeros((-pos.shape[0] % vb, pos.shape[1]), pos.dtype).at[:, 2].set(jnp.nan)
    return jnp.concatenate([pos, pad])


def add_cells(acc: jax.Array, idx: jax.Array, val: jax.Array) -> jax.Array:
    """Return acc [n_cells] with val added at the cells idx (any shape, the same as val's); idx -1 is no cell."""
    n = acc.shape[0]
    return acc + jax.ops.segment_sum(val.ravel().astype(acc.dtype), jnp.where(idx >= 0, idx, n).ravel(), n + 1)[:-1]


def take_cells(r: jax.Array, idx: jax.Array) -> jax.Array:
    """Return r [n_cells] at the cells idx (any shape); 0 where idx is -1 (no cell)."""
    return jnp.where(idx >= 0, r[jnp.clip(idx, 0)], 0.0)


def _chunks(a: jax.Array, qc: int) -> jax.Array:
    return a.reshape(-1, qc, *a.shape[1:])


@partial(jax.jit, static_argnames=("n_cells", "dims", "qc"))
def _ones_block(
    pos_b: jax.Array, pred: tuple, ring_j: jax.Array, scan: dict, dims: tuple, n_cells: int, qc: int
) -> jax.Array:
    """Compute A 1 from one block of voxels, every orientation at unit occupancy."""

    def step(acc: jax.Array, ch: tuple) -> tuple:
        e, o, u, w = ch
        idx, wt = system(e[None], o[None], u[None], w[None], ring_j, pos_b, scan, *dims)
        return add_cells(acc, idx, wt), None  # wt may be float64 (jax_enable_x64): added as acc's dtype

    acc, _ = jax.lax.scan(step, jnp.zeros(n_cells, jnp.float32), tuple(_chunks(p, qc) for p in pred))
    return acc


@partial(jax.jit, static_argnames=("dims", "qc", "k"))
def _top_block(
    r: jax.Array, pos_b: jax.Array, pred: tuple, ring_j: jax.Array, scan: dict, dims: tuple, qc: int, k: int
) -> tuple:
    """Compute A^T r / A^T 1 for one block of voxels and every orientation, and keep each voxel's top k."""

    def one(ch: tuple) -> jax.Array:
        e, o, u, w = ch
        idx, wt = system(e[None], o[None], u[None], w[None], ring_j, pos_b, scan, *dims)
        num = jnp.sum(wt * take_cells(r, idx), (2, 3))
        return num / jnp.maximum(jnp.sum(wt, (2, 3)), 1e-30)

    s = jax.lax.map(one, tuple(_chunks(p, qc) for p in pred))  # [Nq / qc, vb, qc]
    val, ind = jax.lax.top_k(jnp.moveaxis(s, 0, 1).reshape(pos_b.shape[0], -1), k)
    return val, ind.astype(jnp.int32)


def candidates(
    d: jax.Array, pred: tuple, ring_j: jax.Array, pos: jax.Array, scan: dict, dims: tuple, k: int, vb: int,
    qc: int = 16, log: Callable = print,
) -> tuple[jax.Array, jax.Array]:  # fmt: skip
    """Pick each voxel's k best orientations by the first MLEM update from unit occupancy.

    ``f1 = A^T(d / A 1) / A^T 1``: two passes over every (voxel, orientation), in blocks of vb voxels. Dividing by
    ``A 1`` down-weights crowded cells, as MLEM does.

    Parameters
    ----------
    d
        [n_cells] histogram
    pred
        (eta, om, use, w), each [Nq, Nj] with Nq a multiple of qc
    ring_j
        [Nj] ring of each prediction
    pos
        [Nv, 3] positions, Nv a multiple of vb (:func:`pad_voxels`)
    scan, dims
        Rows and (b_e, b_o, n_e, n_o), see :func:`system`
    k
        Candidates per voxel
    vb, qc
        Voxels per block and orientations per chunk
    log
        Progress messages

    Returns
    -------
    f1, cand: jax.Array
        [Nv, k] first-update occupancies and orientation indices
    """
    n_cells = d.shape[0]
    blocks = pos.reshape(-1, vb, pos.shape[-1])
    t0 = time.perf_counter()
    a1 = jnp.zeros(n_cells, jnp.float32)
    for b in blocks:
        a1 = a1 + _ones_block(b, pred, ring_j, scan, dims, n_cells, qc)
    jax.block_until_ready(a1)
    log(f"  candidates: A 1 over {blocks.shape[0]} blocks of {vb} voxels: {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    r = jnp.where(a1 > 0, d / jnp.maximum(a1, 1e-30), 0.0)
    out = [_top_block(r, b, pred, ring_j, scan, dims, qc, k) for b in blocks]
    f1, cand = jnp.concatenate([o[0] for o in out]), jnp.concatenate([o[1] for o in out])
    jax.block_until_ready(cand)
    log(f"  candidates: top {k} per voxel: {time.perf_counter() - t0:.1f} s")
    return f1, cand


@partial(jax.jit, static_argnames=("n_cells", "dims", "vb"))
def forward(
    f: jax.Array, cand: jax.Array, pred: tuple, ring_j: jax.Array, pos: jax.Array, scan: dict, dims: tuple,
    n_cells: int, vb: int,
) -> jax.Array:  # fmt: skip
    """Forward-project occupancies: ``A f``.

    Parameters
    ----------
    f, cand
        [Nv, K] occupancies and orientation indices, Nv a multiple of vb
    pred
        (eta, om, use, w), each [Nq, Nj]
    ring_j, pos, scan, dims
        See :func:`system`
    n_cells
        Size of the histogram
    vb
        Voxels per block

    Returns
    -------
    jax.Array
        [n_cells] predicted histogram
    """
    k = f.shape[1]

    def step(acc: jax.Array, ch: tuple) -> tuple:
        fb, cb, pb = ch
        e, o, u, w = (p[cb] for p in pred)
        idx, wt = system(e, o, u, w, ring_j, pb, scan, *dims)  # [vb, K, Nj, 8]
        return add_cells(acc, idx, fb[:, :, None, None] * wt), None

    xs = (f.reshape(-1, vb, k), cand.reshape(-1, vb, k), pos.reshape(-1, vb, pos.shape[-1]))
    acc, _ = jax.lax.scan(step, jnp.zeros(n_cells, f.dtype), xs)
    return acc


@partial(jax.jit, static_argnames=("dims", "vb"))
def backward(
    r: jax.Array, cand: jax.Array, pred: tuple, ring_j: jax.Array, pos: jax.Array, scan: dict, dims: tuple, vb: int
) -> jax.Array:
    """Back-project a histogram onto each voxel's candidates: ``A^T r``.

    Parameters
    ----------
    r
        [n_cells] histogram (e.g. the ratio of data to prediction)
    cand
        [Nv, K] orientation indices, Nv a multiple of vb
    pred, ring_j, pos, scan, dims, vb
        See :func:`forward`

    Returns
    -------
    jax.Array
        [Nv, K]
    """
    k = cand.shape[1]

    def one(ch: tuple) -> jax.Array:
        cb, pb = ch
        e, o, u, w = (p[cb] for p in pred)
        idx, wt = system(e, o, u, w, ring_j, pb, scan, *dims)
        return jnp.sum(wt * take_cells(r, idx), (2, 3))

    return jax.lax.map(one, (cand.reshape(-1, vb, k), pos.reshape(-1, vb, pos.shape[-1]))).reshape(-1, k)


@jax.jit
def _squarem_point(f: jax.Array, f1: jax.Array, f2: jax.Array) -> tuple[jax.Array, jax.Array]:
    """SQUAREM's extrapolation from f and two MLEM steps f1, f2 (r = f1 - f and v = f2 - 2 f1 + f fused, not stored)."""
    nr = jnp.sqrt(jnp.sum((f1 - f) ** 2))
    nv = jnp.sqrt(jnp.sum((f2 - 2.0 * f1 + f) ** 2))
    a = jnp.minimum(-nr / jnp.maximum(nv, 1e-30), -1.0)  # never shorter than plain MLEM
    # not below 1e-3 of f2: MLEM multiplies, so an occupancy at 0 could never come back
    return jnp.maximum(f - 2.0 * a * (f1 - f) + a * a * (f2 - 2.0 * f1 + f), 1e-3 * f2), a


def squarem(
    step: Callable, f: jax.Array, n_iter: int, log: Callable = print, label: str = "MLEM",
    callback: Callable | None = None,
) -> jax.Array:  # fmt: skip
    """Iterate an MLEM update ``step``, accelerated by SQUAREM, logging the deviance.

    Each iteration is a SQUAREM step (Varadhan and Roland, 2008) [1]_: two MLEM updates f1, f2
    from f; the point ``f - 2 a r + a^2 v`` with ``r = f1 - f``, ``v = f2 - 2 f1 + f`` and ``a = -norm(r) / norm(v)``
    (at most -1: never shorter than plain MLEM), kept above 1e-3 f2 (MLEM multiplies: an occupancy at 0 could never
    come back); and one more MLEM update from there. If the point fits worse than f1, f2 is kept instead, so the
    deviance never rises. An iteration costs 3 MLEM updates and goes about as far as 10 or more. It holds at most four
    copies of f at once. Used by the voxel fit (:func:`mlem`) and the global orientation fit
    (:func:`anri.index.orientations.fit`).

    Parameters
    ----------
    step
        One MLEM update: ``step(f) -> (f_new, deviance of f)``
    f
        Starting occupancies (any shape)
    n_iter
        SQUAREM iterations (3 MLEM updates each)
    log
        Progress messages, one per iteration: its SQUAREM step length, whether the step was taken, and the deviance
    label
        Name of the fit in the messages
    callback
        Called as ``callback(iteration, f, deviance)`` after each iteration (1, 2, ...), with the deviance before its
        last MLEM update, e.g. to record the fit's progress

    Returns
    -------
    jax.Array
        The occupancies

    References
    ----------
    .. [1] Varadhan, R., Roland, C., 2008. Simple and globally convergent methods for accelerating the convergence of
       any EM algorithm. Scandinavian Journal of Statistics 35, 335–353.
       https://doi.org/10.1111/j.1467-9469.2007.00585.x
    """
    for it in range(1, n_iter + 1):
        f1, _ = step(f)
        f2, dev1 = step(f1)
        fx, a = _squarem_point(f, f1, f2)
        del f1  # at most 4 copies of the occupancies at once
        fn, devx = step(fx)
        del fx
        taken = devx <= dev1
        f, dev = (fn, devx) if taken else (f2, dev1)  # never worse than plain MLEM
        del fn, f2
        if callback is not None:
            callback(it, f, dev)
        # every iteration: the deviances are on the host already, and a is a scalar computed before devx
        log(f"  {label} {it} (SQUAREM step {float(-a):.1f}{'' if taken else ', not taken'}): deviance {dev:.6g}")
    return f


def mlem(
    d: jax.Array, cand: jax.Array, pred: tuple, ring_j: jax.Array, pos: jax.Array, scan: dict, dims: tuple,
    f0: jax.Array, n_iter: int, vb: int, log: Callable = print, censor: float = 0.0, callback: Callable | None = None,
    saturated: jax.Array | None = None,
) -> jax.Array:  # fmt: skip
    """Fit occupancies by MLEM (``f <- f A^T(d / A f) / A^T 1``), accelerated by SQUAREM (:func:`squarem`).

    MLEM is the EM algorithm of emission tomography (Shepp and Vardi, 1982 [1]_; Lange and Carson, 1984 [2]_).

    An iteration costs 3 MLEM updates and goes about as far as 10 or more: thin features (twins one or two voxels
    thick) converge long after the deviance flattens. Empty bins are censored above ``censor`` (counts per bin,
    :func:`censored_ratio`); 0 for plain Poisson.

    Parameters
    ----------
    d
        [n_cells] histogram
    cand, pred, ring_j, pos, scan, dims, vb
        See :func:`forward`
    f0
        [Nv, K] starting occupancies
    n_iter
        SQUAREM iterations (3 MLEM updates each)
    log
        Progress messages
    censor
        Empty bins censored above this many counts per bin
    callback
        See :func:`squarem`
    saturated
        Indices of the bins holding a saturated pixel, censored from above (:func:`censored_ratio`)

    Returns
    -------
    jax.Array
        [Nv, K] occupancies

    References
    ----------
    .. [1] Shepp, L.A., Vardi, Y., 1982. Maximum likelihood reconstruction for emission tomography. IEEE Transactions on
       Medical Imaging 1, 113–122. https://doi.org/10.1109/TMI.1982.4307558
    .. [2] Lange, K., Carson, R., 1984. EM reconstruction algorithms for emission and transmission tomography. Journal
       of Computer Assisted Tomography 8, 306–316.
    """
    n_cells = d.shape[0]
    norm = jnp.maximum(backward(jnp.ones(n_cells, d.dtype), cand, pred, ring_j, pos, scan, dims, vb), 1e-30)

    def step(f: jax.Array) -> tuple:  # one MLEM update, and the deviance of the occupancies it started from
        Af = forward(f, cand, pred, ring_j, pos, scan, dims, n_cells, vb)
        f_new = f * backward(censored_ratio(d, Af, censor, saturated), cand, pred, ring_j, pos, scan, dims, vb) / norm
        return f_new, float(deviance(d, Af, censor, saturated))

    return squarem(step, f0, n_iter, log, callback=callback)


@partial(jax.jit, static_argnames=("k2",))
def _inherit(near9: jax.Array, f_c: jax.Array, cand_c: jax.Array, k2: int) -> jax.Array:
    def one(nb: jax.Array) -> jax.Array:
        c, w = cand_c[nb].ravel(), f_c[nb].ravel()  # the neighbourhood's candidates and coarse occupancies
        o = jnp.argsort(c)
        c, w = c[o], w[o]
        dup = jnp.concatenate([jnp.array([False]), c[1:] == c[:-1]])
        seg = jnp.cumsum(~dup) - 1  # which orientation each entry is, in sorted order
        w_max = jax.ops.segment_max(w, seg, num_segments=c.size)[seg]  # its largest occupancy in any neighbour
        v, i = jax.lax.top_k(jnp.where(dup, -jnp.inf, w_max), k2)  # each orientation once
        return jnp.where(v > -jnp.inf, c[i], -1)  # -1: fewer than k2 orientations in the neighbourhood

    return jax.vmap(one)(near9)


def inherit_candidates(
    pos: ArrayLike, pos_c: ArrayLike, f_c: ArrayLike, cand_c: ArrayLike, k2: int, vb: int = 1 << 14
) -> jax.Array:
    """Candidates for fine voxels from a coarse fit: what the nearest coarse voxel and its 8 neighbours occupied.

    Parameters
    ----------
    pos
        [Nv, 3] fine voxel positions
    pos_c
        [Nc, 3] coarse voxel positions
    f_c, cand_c
        [Nc, K] coarse occupancies and orientation indices
    k2
        Candidates per fine voxel, the most occupied first (an orientation in several neighbours by its largest
        occupancy)
    vb
        Fine voxels per call

    Returns
    -------
    jax.Array
        [Nv, k2] orientation indices, each once; -1 in the last slots where the neighbourhood holds fewer than k2
    """
    from scipy.spatial import KDTree

    pos = np.asarray(pos)[:, :2]
    tree = KDTree(np.asarray(pos_c)[:, :2])  # the 9 nearest by a tree: sorting every distance was minutes on big maps
    f_c, cand_c = jnp.asarray(f_c), jnp.asarray(cand_c)
    out = []
    for s0 in range(0, len(pos), vb):
        _, near9 = tree.query(pos[s0 : s0 + vb], k=9)
        out.append(np.asarray(_inherit(jnp.asarray(near9, jnp.int32), f_c, cand_c, k2)))
    return jnp.asarray(np.concatenate(out))


def candidates_from(
    d: jax.Array, cand_in: jax.Array, pred: tuple, ring_j: jax.Array, pos: jax.Array, scan: dict, dims: tuple,
    k: int, vb: int, log: Callable = print,
) -> tuple[jax.Array, jax.Array]:  # fmt: skip
    """Pick each voxel's k best orientations as :func:`candidates` does, but from its own list only.

    Parameters
    ----------
    d
        [n_cells] histogram
    cand_in
        [Nv, K2] orientations to score per voxel, e.g. from :func:`inherit_candidates`; -1 for none
    pred, ring_j, pos, scan, dims, vb
        See :func:`forward`
    k
        Candidates kept per voxel
    log
        Progress messages

    Returns
    -------
    f1, cand: jax.Array
        [Nv, k] first-update occupancies and orientation indices
    """
    t0 = time.perf_counter()
    n_cells = d.shape[0]
    ok = cand_in >= 0
    cand_in = jnp.maximum(cand_in, 0)  # the empty slots: any orientation, with no weight and never picked
    a1 = forward(ok.astype(jnp.float32), cand_in, pred, ring_j, pos, scan, dims, n_cells, vb)
    r = jnp.where(a1 > 0, d / jnp.maximum(a1, 1e-30), 0.0)
    num = backward(r, cand_in, pred, ring_j, pos, scan, dims, vb)
    f1 = num / jnp.maximum(backward(jnp.ones(n_cells, jnp.float32), cand_in, pred, ring_j, pos, scan, dims, vb), 1e-30)
    val, i = jax.lax.top_k(jnp.where(ok, f1, -jnp.inf), k)
    val = jnp.maximum(val, 0.0)  # an empty slot picked (fewer than k orientations): occupancy 0
    cand = jnp.take_along_axis(cand_in, i, 1)
    jax.block_until_ready(cand)
    log(f"  candidates: top {k} of {cand_in.shape[1]} inherited per voxel: {time.perf_counter() - t0:.1f} s")
    return val, cand


def fit_occupancy(
    H: ArrayLike,
    pred: tuple,
    ring_j: ArrayLike,
    pos: ArrayLike,
    scan: dict,
    dims: tuple,
    k: int = 64,
    n_iter: int = 10,
    coarse: int = 1,
    block_bytes: float = 1e9,
    qc: int = 16,
    log: Callable = print,
    return_model: bool = False,
    censor: float = 0.0,
    callback: Callable | None = None,
    absorb: tuple | None = None,
    saturated: ArrayLike | None = None,
    given: tuple | None = None,
) -> tuple:
    """Fit sparse occupancies: candidates per voxel, then MLEM.

    With ``coarse = G > 1`` the full candidate pass runs on voxels G times larger (the histogram's rows summed in
    groups of G), and each voxel then scores only the 2k orientations its 3 x 3 coarse neighbourhood occupied most:
    about G^2 times cheaper for large maps, at the risk of missing a grain the coarse fit missed.

    Parameters
    ----------
    H
        [n_cells] histogram, rows last
    pred
        (eta, om, use, w), each [Nq, Nj] with Nq a multiple of qc
    ring_j
        [Nj] ring of each prediction
    pos
        [Nv, 3] voxel positions
    scan, dims
        Rows and (b_e, b_o, n_e, n_o) of the histogram, plus n_beam for a beam profile (:func:`system`,
        :func:`beam_rows`)
    k
        Candidates per voxel (at most the number of orientations; fewer orientations, fewer candidates)
    n_iter
        MLEM iterations (SQUAREM, :func:`mlem`)
    coarse
        Coarse-to-fine factor (1: off)
    block_bytes
        Memory for one block of voxels' system entries
    qc
        Orientations per chunk in passes over every orientation
    return_model
        Also return the fitted histogram ``A f``, e.g. to compare with the data row by row
    censor
        Empty bins censored above this many counts per bin in the MLEM (:func:`censored_ratio`); 0: plain MLEM
    callback
        Passed to the final :func:`mlem`: ``callback(iteration, f, deviance)`` after each iteration, f over the fitted
        voxels padded to whole blocks (the first ``len(pos)`` are the voxels)
    absorb
        ``(sample_map, step)``: model absorption along the incoming and diffracted paths through the sample, a [n, n]
        map of the fraction of each voxel it fills (reconstruction order, voxels of ``step``), with ``scan["mu"]``
        (1 / the units of step) and ``scan["ring_tth"]`` (degrees). The occupancies are then free of absorption
    saturated
        Indices of the bins holding a saturated pixel, censored from above in the MLEM (:func:`censored_ratio`)
    given
        ``(f0, cand)``, each [Nv, k]: each voxel's own candidates (-1 for an empty slot) and starting occupancies,
        instead of picking them (k and coarse are then not used), e.g. from :func:`anri.index.local.inherit`
    log
        Progress messages

    Returns
    -------
    f, cand: np.ndarray
        [Nv, k] occupancies and orientation indices
    model: np.ndarray
        [n_cells] the fitted histogram, if return_model
    """
    H, ring_j, pos = jnp.asarray(H), jnp.asarray(ring_j), np.asarray(pos, np.float32)
    nv = len(pos)
    if pred[0].shape[0] == 0:
        msg = "no orientations to fit"
        raise ValueError(msg)
    k = given[1].shape[1] if given is not None else min(k, pred[0].shape[0])  # top_k needs k <= the orientations
    sat = None if saturated is None or np.size(saturated) == 0 else jnp.asarray(saturated, jnp.int32)

    def with_edges(p: np.ndarray) -> np.ndarray:  # positions followed by their edge distances (see system)
        if absorb is None:
            return p
        D = edge_distances(jnp.asarray(absorb[0], jnp.float32), absorb[1], jnp.asarray(p[:, :2]))
        return np.concatenate([p, np.asarray(D)], 1)

    t0 = time.perf_counter()
    pos = with_edges(pos)
    if absorb is not None:
        log(
            f"absorption: mu {float(scan['mu']):.3g}; paths to the sample's edge for {nv} voxels x {pos.shape[1] - 3} "
            f"directions: {time.perf_counter() - t0:.1f} s"
        )
    n_j = pred[0].shape[1]
    n_corners = 4 * dims[4] if len(dims) > 4 and dims[4] > 0 else 8
    vb_all, vb = block_voxels(qc, n_j, block_bytes, n_corners), block_voxels(k, n_j, block_bytes, n_corners)
    vbp = max(vb_all, vb)
    pos_p = pad_voxels(pos, vbp)
    t0 = time.perf_counter()
    if given is not None:
        cand_g = np.full((len(pos_p), k), -1, np.int32)
        f_g = np.zeros((len(pos_p), k), np.float32)
        cand_g[:nv], f_g[:nv] = given[1], given[0]
        log(f"candidates: given, {(cand_g[:nv] >= 0).sum(1).mean():.0f} of {k} per voxel on average")
        # an empty slot: orientation 0 at zero occupancy, which MLEM's multiplicative update keeps at zero
        f0, cand = jnp.asarray(np.where(cand_g >= 0, f_g, 0.0)), jnp.asarray(np.maximum(cand_g, 0))
    elif coarse == 1:
        log(f"candidates: {nv} voxels x {pred[0].shape[0]} orientations, the top {k} per voxel")
        f0, cand = candidates(H, pred, ring_j, pos_p, scan, dims, k, vb_all, qc=qc, log=log)
    else:
        H_c, _ = coarsen_rows(H, scan["n_rows"], coarse)
        scan_c = coarsen_scan(scan, coarse)
        step_c = scan_c["ystep"]
        extent = float(np.abs(np.asarray(pos)[:, :2]).max())
        n_c = 2 * int(np.ceil(extent / step_c)) + 3
        pos_c = np.asarray(recon_positions(n_c, step_c), np.float32)
        pos_cp = pad_voxels(with_edges(pos_c), vbp)
        log(f"coarse: {n_c} x {n_c} voxels of {step_c:g} x {pred[0].shape[0]} orientations, the top {k} per voxel")
        f_c, cand_c = candidates(H_c, pred, ring_j, pos_cp, scan_c, dims, k, vb_all, qc=qc, log=log)
        f_c = mlem(
            H_c, cand_c, pred, ring_j, pos_cp, scan_c, dims, f_c, n_iter, vb, log=log, censor=censor
        )  # rows summed: no saturation index
        cand_in = inherit_candidates(pos_p[:, :3], pos_c, f_c[: n_c * n_c], cand_c[: n_c * n_c], 2 * k)
        log(f"fine: {nv} voxels x {2 * k} orientations inherited from the coarse neighbourhood")
        f0, cand = candidates_from(H, cand_in, pred, ring_j, pos_p, scan, dims, k, max(vb // 2, 1), log=log)
    log(f"candidates: {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    f = mlem(H, cand, pred, ring_j, pos_p, scan, dims, f0, n_iter, vb, log=log, censor=censor, callback=callback,
             saturated=sat)  # fmt: skip
    log(f"MLEM {n_iter} iterations: {time.perf_counter() - t0:.1f} s")
    if return_model:
        mu = forward(f, cand, pred, ring_j, pos_p, scan, dims, H.shape[0], vb)
        return np.asarray(f)[:nv], np.asarray(cand)[:nv], np.asarray(mu)
    return np.asarray(f)[:nv], np.asarray(cand)[:nv]
