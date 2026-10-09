"""Coarse views of the data for indexing: rings, pixel angles, histograms and the lit map.

The sparse pixels are reduced once to a histogram ``H[ring, eta, omega, row]`` at the scale of the orientation grid
(a degree or so), and to a row-summed "lit" map of where there is intensity. Everything after that works on these.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from Dans_Diffraction.classes_crystal import Crystal as DansCrystal
from jax.typing import ArrayLike

from anri.crystal import reflections, rings, structure_factors
from anri.geom import beam_basis


def ring_table(
    lattice_parameters: ArrayLike,
    space_group: int,
    wavelength: float,
    n_rings: int,
    structure: DansCrystal | None = None,
) -> dict:
    """List the reflections of a crystal's first rings, with their structure factors.

    Systematic absences are removed (:func:`anri.crystal.reflections`). Each reflection has two predictions, one per
    omega solution, indexed ``j = 2 h + branch``. With a structure (``Dans_Diffraction.Crystal``, e.g. from a CIF),
    each reflection gets its ``|F|^2`` (:func:`anri.crystal.structure_factors`), scaled to a mean of 1; without one,
    ``|F|^2 = 1``. The lattice parameters given set the reflections and 2theta, so a refined lattice can be used with
    a CIF's atoms.

    Parameters
    ----------
    lattice_parameters
        [6] a, b, c (angstrom), alpha, beta, gamma (degrees)
    space_group
        Space-group number
    wavelength
        In angstrom
    n_rings
        How many rings, from the lowest 2theta
    structure
        Optional: the same phase with its atoms, for ``|F|^2``

    Returns
    -------
    dict
        "hkls" [Nh, 3], "F2" [Nh], "ring_j" [2 Nh] (ring of each prediction), "tth" [n_rings] (degrees)
    """
    dsmax = 0.5
    while True:  # enough d* range for the first n_rings rings
        refl = reflections(lattice_parameters, space_group, wavelength, dsmax)
        ring, ring_ds = rings(refl["ds"])
        if len(ring_ds) > n_rings or dsmax >= 2 / wavelength:
            break
        dsmax *= 1.5
    sel = ring < n_rings
    hkls = refl["hkl"][sel]
    F2 = np.ones(len(hkls))
    if structure is not None:
        F2 = structure_factors(structure, hkls, wavelength)
        F2 = F2 / max(F2.mean(), 1e-30)
    ring_tth = np.degrees(2 * np.arcsin(ring_ds[:n_rings] * wavelength / 2))
    return {
        "hkls": hkls.astype(np.float32),
        "F2": F2.astype(np.float32),
        "ring_j": np.repeat(ring[sel], 2).astype(np.int32),
        "tth": ring_tth.astype(np.float32),
    }


def _tth_eta(v: jax.Array, k_in: jax.Array) -> jax.Array:
    """[..., 2] (2theta, eta) in degrees of directions or lab points v [..., 3], seen from the lab origin."""
    k, e_h, e_v = beam_basis(k_in)
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    tth = jnp.degrees(jnp.arccos(jnp.clip(v @ k, -1.0, 1.0)))
    return jnp.stack([tth, jnp.degrees(jnp.arctan2(-(v @ e_h), v @ e_v))], -1)


@jax.jit
def pixel_angles(slow: jax.Array, fast: jax.Array, omega: jax.Array, geom: dict) -> jax.Array:
    """Compute (2theta, eta, omega) of pixel centres, seen from the lab origin (no parallax).

    Parameters
    ----------
    slow, fast
        [N] pixel coordinates
    omega
        [N] omega of each pixel's frame (degrees)
    geom
        Geometry dict with "det_origin_lab", "s_step_lab", "f_step_lab" and "k_in_lab", e.g. from
        :func:`anri.io.geom_from_pars`

    Returns
    -------
    jax.Array
        [N, 3] (2theta, eta, omega) in degrees
    """
    p = geom["det_origin_lab"] + slow[:, None] * geom["s_step_lab"] + fast[:, None] * geom["f_step_lab"]
    return jnp.concatenate([_tth_eta(p, geom["k_in_lab"]), omega[:, None]], 1)


@partial(jax.jit, static_argnames=("n",))
def tth_profile(x: jax.Array, val: jax.Array, lo: float, step: float, n: int) -> jax.Array:
    """Sum pixel intensities in 2theta bins.

    Parameters
    ----------
    x
        [N, 3] (2theta, eta, omega) from :func:`pixel_angles`
    val
        [N] intensities
    lo, step, n
        Bins: n of width step from lo (degrees)

    Returns
    -------
    jax.Array
        [n] intensity per bin
    """
    i = jnp.floor((x[:, 0] - lo) / step).astype(jnp.int32)
    ok = (i >= 0) & (i < n)
    return jax.ops.segment_sum(jnp.where(ok, val, 0.0), jnp.where(ok, i, n), n + 1)[:-1]


def ring_widths(
    prof: ArrayLike, lo: float, step: float, ring_tth: ArrayLike, frac: float = 0.95, max_hw: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Measure each ring's offset and half-width (degrees 2theta) from a 2theta profile.

    Each ring is looked at within half the gap to its neighbours (at most ``max_hw``). Background: the median of the
    outer fifth of that window. Offset: the centroid minus the ring's 2theta. Half-width: the distance from the
    centroid that holds ``frac`` of the ring's net intensity. It sums everything that spreads a ring: parallax
    (sample size / distance), strain, peak size and detector distortion. A ring without intensity gets offset 0 and
    the whole window.

    Parameters
    ----------
    prof
        Intensity per 2theta bin, from :func:`tth_profile`
    lo, step
        The bins
    ring_tth
        [Nr] 2theta of the rings
    frac
        Fraction of a ring's intensity within the half-width
    max_hw
        Largest half-width considered

    Returns
    -------
    offset, half_width: np.ndarray
        [Nr] each, in degrees
    """
    prof, ring_tth = np.asarray(prof), np.asarray(ring_tth, float)
    x = lo + (np.arange(len(prof)) + 0.5) * step
    gaps = np.diff(ring_tth)
    half = np.minimum(np.concatenate([[2 * max_hw], gaps]), np.concatenate([gaps, [2 * max_hw]])) / 2
    off, hw = np.zeros(len(ring_tth)), np.zeros(len(ring_tth))
    for r, t in enumerate(ring_tth):
        m = np.abs(x - t) < half[r]
        xr, pr = x[m], prof[m]
        outer = np.abs(xr - t) > 0.8 * half[r]
        net = np.maximum(pr - (np.median(pr[outer]) if outer.any() else 0.0), 0.0)
        if net.sum() <= 0:
            hw[r] = half[r]
            continue
        c = np.sum(xr * net) / net.sum()
        d = np.abs(xr - c)
        o = np.argsort(d)
        k = np.searchsorted(np.cumsum(net[o]), frac * net.sum())
        off[r], hw[r] = c - t, d[o][min(k, len(o) - 1)] + step / 2
    return off, hw


@partial(jax.jit, static_argnames=("n_ring", "n_e", "n_o", "n_k"))
def histogram(
    x: jax.Array,
    val: jax.Array,
    row: jax.Array,
    ring_tth: jax.Array,
    tth_tol: ArrayLike,
    om0: float,
    b_e: float,
    b_o: float,
    n_ring: int,
    n_e: int,
    n_o: int,
    n_k: int,
) -> jax.Array:
    """Bin pixels into ``H[ring, eta, omega, row]`` (flattened).

    A pixel goes to the nearest ring if within ``tth_tol`` of it; eta bins are periodic over 360 degrees from -180,
    omega bins start at ``om0``. Pixels with row -1 are dropped.

    Parameters
    ----------
    x
        [N, 3] (2theta, eta, omega) from :func:`pixel_angles`
    val, row
        [N] intensity and dty row of each pixel
    ring_tth
        [n_ring] 2theta of the rings
    tth_tol
        2theta tolerance (degrees), a scalar or one per ring
    om0, b_e, b_o
        First omega bin edge, and the bin widths in eta and omega (degrees)
    n_ring, n_e, n_o, n_k
        Numbers of rings, eta bins, omega bins and rows

    Returns
    -------
    jax.Array
        [n_ring * n_e * n_o * n_k] intensity per bin
    """
    i = jnp.clip(jnp.searchsorted(ring_tth, x[:, 0]), 1, max(n_ring - 1, 1))
    ring = jnp.where(jnp.abs(x[:, 0] - ring_tth[i - 1]) < jnp.abs(x[:, 0] - ring_tth[i % n_ring]), i - 1, i % n_ring)
    ok = jnp.abs(x[:, 0] - ring_tth[ring]) < jnp.broadcast_to(tth_tol, ring_tth.shape)[ring]
    ie = jnp.floor((x[:, 1] + 180.0) / b_e).astype(jnp.int32) % n_e
    io = jnp.floor((x[:, 2] - om0) / b_o).astype(jnp.int32)
    ok = ok & (io >= 0) & (io < n_o) & (row >= 0) & (row < n_k)
    idx = jnp.where(ok, ((ring * n_e + ie) * n_o + io) * n_k + row, n_ring * n_e * n_o * n_k)
    return jax.ops.segment_sum(jnp.where(ok, val, 0.0), idx, n_ring * n_e * n_o * n_k + 1)[:-1]


def histogram_pixels(
    chunks: Iterable,
    geom: dict,
    ring_tth: ArrayLike,
    tth_tol: ArrayLike,
    om0: float,
    histograms: list,
    chunk: int,
) -> list:
    """Stream pixels into one or more histograms in a single pass over the data.

    Each chunk's pixel angles are computed once and added to every histogram, so the data are read once however
    many histograms (e.g. a fine row-summed lit map and a coarse per-row one) are wanted.

    Parameters
    ----------
    chunks
        Iterable of (slow, fast, omega, row, value) NumPy arrays of at most ``chunk`` pixels each, e.g.
        :func:`anri.io.stream_sparse`
    geom
        Geometry dict, see :func:`pixel_angles`
    ring_tth, tth_tol
        See :func:`histogram`
    om0
        First omega bin edge
    histograms
        One ``((b_e, b_o, n_e, n_o), n_rows)`` per histogram: eta and omega bin widths (degrees) and counts, and the
        number of rows (1 sums them all)
    chunk
        Pixels per call (chunks are padded to it, so one compile)

    Returns
    -------
    list
        One flat histogram [n_rings * n_e * n_o * n_rows] per entry of ``histograms``
    """
    ring_tth = jnp.asarray(ring_tth, jnp.float32)
    tth_tol = jnp.asarray(tth_tol, jnp.float32)
    n_ring = ring_tth.shape[0]
    out = [jnp.zeros(n_ring * b[2] * b[3] * n, jnp.float32) for b, n in histograms]
    zeros = jnp.zeros(chunk, jnp.int32)
    for slow, fast, omega, row, value in chunks:
        m = len(value)

        def pad(a: ArrayLike, dt: type = np.float32, fill: float = 0, m: int = m) -> jax.Array:
            # a fresh array per chunk: a reused buffer can be overwritten while JAX still copies it (asynchronously)
            b = np.full(chunk, fill, dt)
            b[:m] = a
            return jnp.asarray(b)

        x = pixel_angles(pad(slow), pad(fast), pad(omega), geom)
        v, rows = pad(value), pad(row, np.int32, -1)
        for i, ((b_e, b_o, n_e, n_o), n_rows) in enumerate(histograms):
            r = rows if n_rows > 1 else jnp.where(rows >= 0, zeros, -1)
            out[i] = out[i] + histogram(x, v, r, ring_tth, tth_tol, om0, b_e, b_o, n_ring, n_e, n_o, n_rows)
    return out


def ring_profile(chunks: Iterable, geom: dict, ring_tth: ArrayLike, chunk: int, step: float = 0.002) -> tuple:
    """Measure the rings' offsets and half-widths from a sample of pixels (see :func:`ring_widths`).

    Parameters
    ----------
    chunks
        Iterable of (slow, fast, omega, row, value) NumPy arrays, e.g. a few rows of :func:`anri.io.stream_sparse`
    geom
        Geometry dict, see :func:`pixel_angles`
    ring_tth
        [Nr] 2theta of the rings
    chunk
        Pixels per call
    step
        2theta bin width (degrees)

    Returns
    -------
    offset, half_width: np.ndarray
        [Nr] each, in degrees
    """
    ring_tth = np.asarray(ring_tth, float)
    lo = float(ring_tth[0]) - 0.5
    n = int(np.ceil((float(ring_tth[-1]) + 0.5 - lo) / step))
    prof = jnp.zeros(n, jnp.float32)
    for slow, fast, omega, _, value in chunks:
        m = len(value)

        def pad(a: ArrayLike, m: int = m) -> jax.Array:
            return jnp.asarray(np.pad(np.asarray(a, np.float32), (0, chunk - m)))

        prof = prof + tth_profile(pixel_angles(pad(slow), pad(fast), pad(omega), geom), pad(value), lo, step, n)
    return ring_widths(np.asarray(prof), lo, step, ring_tth)


@jax.jit
def lit_table(lit: jax.Array) -> jax.Array:
    """Summed-area table of a lit map, for counting lit bins in any box in constant time.

    Parameters
    ----------
    lit
        [Nr, n_e, n_o] bool, where there is intensity

    Returns
    -------
    jax.Array
        [Nr, 3 n_e + 1, n_o + 1] int32, with eta tiled three times (it is periodic)
    """
    t = jnp.concatenate([lit, lit, lit], 1).astype(jnp.int32)
    return jnp.pad(jnp.cumsum(jnp.cumsum(t, 1), 2), ((0, 0), (1, 0), (1, 0)))


def coarsen_rows(H: ArrayLike, n_rows: int, g: int) -> tuple[jax.Array, int]:
    """Sum a histogram's rows in groups of g: row k goes to k // g.

    Parameters
    ----------
    H
        Flat histogram whose last axis is the n_rows rows
    n_rows
        Rows in H
    g
        Group size

    Returns
    -------
    H: jax.Array
        Flat coarse histogram
    n_rows: int
        Its number of rows, ceil(n_rows / g)
    """
    d = jnp.asarray(H).reshape(-1, n_rows)
    d = jnp.pad(d, ((0, 0), (0, -n_rows % g)))
    return d.reshape(d.shape[0], -1, g).sum(-1).ravel(), -(-n_rows // g)


def dty_offsets(
    frame_dty: ArrayLike, frame_omega: ArrayLike, ybinedges: ArrayLike, gridstep: int, om0: float, b_o: float, n_o: int,
    dty0: float, ystep: float, bin_dty: ArrayLike | None = None, return_exposure: bool = False,
) -> np.ndarray | tuple:  # fmt: skip
    """Each histogram row's dty, per omega bin, less its nominal dty: where the beam really was.

    A row is the frames whose dty, as :func:`anri.io.stream_sparse` binned them (``bin_dty``; default each DataSet
    row's mean dty, as when it bins by the DataSet's dty), falls in one bin of ``ybinedges``. In a step scan every frame of a row is at the row's dty and the offsets are 0. In a fly or helical scan dty
    moves while omega turns (e.g. one dty step per turn): the frames of one omega bin were taken at a dty up to half
    a step away from the row's. :func:`anri.index.system` uses the offsets (``scan["ddty"]``) to put each voxel in
    the rows that really saw it.

    Parameters
    ----------
    frame_dty, frame_omega
        [rows, frames] each frame's dty and omega, e.g. :func:`anri.io.read_frame_dty` and the DataSet's ``omega``
    ybinedges
        [n + 1] dty bin edges of the rows (the DataSet's ``ybinedges``)
    gridstep
        Rows summed in groups of this, as in the histogram
    om0, b_o, n_o
        The histogram's first omega bin edge, bin width and number of bins
    dty0, ystep
        The histogram's rows: row k is nominally at dty0 + k ystep
    bin_dty
        [rows, frames] the dty each frame was binned by (e.g. ``frame_dty`` itself when frames are binned by their
        own dty); default each row's mean of ``frame_dty``
    return_exposure
        Also return each (row, omega bin)'s exposure: its frames over the usual number (the median over bins with
        frames). Binned by their own dty, a helical scan's frames give some bins none and some two rotations' worth;
        the model must know (``scan["exposure"]``)

    Returns
    -------
    offsets: np.ndarray
        [n_rows, n_o] mean dty of the frames in each (row, omega bin) less dty0 + k ystep; 0 where no frames
    exposure: np.ndarray
        [n_rows, n_o], if return_exposure
    """
    fd = np.asarray(frame_dty, float)
    om = np.asarray(frame_omega, float)
    edges = np.asarray(ybinedges, float)
    n_rows = -(-(len(edges) - 1) // gridstep)
    bd = np.broadcast_to(np.nanmean(fd, 1, keepdims=True), fd.shape) if bin_dty is None else np.asarray(bin_dty, float)
    kk = np.searchsorted(edges, np.where(np.isfinite(bd), bd, np.inf)) - 1  # each frame's row
    kk = np.where((kk >= 0) & (kk < len(edges) - 1), kk // gridstep, -1)
    o = np.floor((np.mod(om - om0, 360.0)) / b_o).astype(int)
    ok = (kk >= 0) & (o >= 0) & (o < n_o) & np.isfinite(fd)
    idx = kk[ok] * n_o + o[ok]
    s = np.bincount(idx, weights=fd[ok] - (dty0 + kk[ok] * ystep), minlength=n_rows * n_o)
    c = np.bincount(idx, minlength=n_rows * n_o)
    off = np.where(c > 0, s / np.maximum(c, 1), 0.0).reshape(n_rows, n_o)
    if not return_exposure:
        return off
    return off, (c / max(float(np.median(c[c > 0])), 1.0)).reshape(n_rows, n_o) if np.any(c > 0) else c
