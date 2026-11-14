"""Coarse-to-fine in orientation and space: the fit stage's orientations refined off the search grid.

The search grid sets how close an orientation can be to the truth (its worst case, 0.87 x the step). A coarse
orientation puts its predicted spots in the wrong (eta, omega) bins of the fit's 1 degree histogram altogether, and
no voxel size makes a fit with it reliable: on half-sample scans it emptied the voxels near the rotation axis and made
streaks, and even on full scans it left a checkerboard in every grain (a synthetic phantom, 3 deg grid). So the fit
stage refines the filter's orientations first, on the row-summed data without any tomography (:func:`refine`: each
orientation replaced by its 27 :func:`children`, a third of the step apart, fitted by the filter's orientation fit
and kept by likelihood ratio), until the worst case is within a third of the bins.

The orientation error also enters the tomography: an error delta shifts a reflection's omega by about
-delta_z - delta_y cot(eta), which moves the voxel's predicted row by r sin(omega - phi) d_omega at a distance r from
the axis. A fit of voxels v with orientations off by up to e is consistent only if v >~ R e (R: the sample's radius),
so the first spatial level may need voxels a few rows wide (:func:`level_rows`). Each later level subdivides the
orientations by 3 and the voxels down to the rows' step: every voxel keeps the orientations holding at least a small
share of its occupancy (:func:`kept`: a thin twin lamella, a small share of a coarse voxel, keeps its own; a cover of
the largest would drop it), replaces them by their children (:func:`inherit`), each starting from a 27th of its
parent's occupancy, and each finer voxel fits the children its 3 x 3 coarser neighbourhood held
(:func:`anri.index.occupancy.inherit_candidates`).
"""

from __future__ import annotations

import itertools
import math
from typing import Callable

import numpy as np
from jax.typing import ArrayLike

from anri.maths.rotation import rod_to_mat
from anri.maths.utils import float64

N_CHILD = 27


def children(U: ArrayLike, step: float) -> np.ndarray:
    """Return the 27 rotations of a 3 x 3 x 3 grid around each orientation, step / 3 apart.

    Parameters
    ----------
    U
        [Nq, 3, 3] orientations (sample from crystal)
    step
        Spacing of the grid U came from (deg)

    Returns
    -------
    np.ndarray
        [Nq x 27, 3, 3]: orientation q's children are rows 27 q to 27 q + 26, rotated about sample axes by rotation
        vectors in {-step / 3, 0, step / 3}^3 (the 14th, the zero rotation, is q itself)
    """
    s = np.radians(step / 3)
    v = np.array(list(itertools.product((-s, 0.0, s), repeat=3)))
    t = np.linalg.norm(v, axis=1)
    rod = np.tan(t / 2)[:, None] * v / np.maximum(t, 1e-30)[:, None]  # Rodrigues vectors tan(t / 2) n
    with float64():  # exact rotations: the children are small steps from their parent
        R = np.asarray(rod_to_mat(rod))
    return np.einsum("cij,qjk->qcik", R, np.asarray(U, np.float64)).reshape(-1, 3, 3)


def refine(data: ArrayLike, U: np.ndarray, comp: np.ndarray, step: float, worst: float, refine_to: float, B: ArrayLike,
           rings: dict, geom: dict, bins: tuple, etacut: float, min_lr: float, keep: int, censor: float = 0.0,
           log: Callable = print) -> tuple:  # fmt: skip
    """Refine orientations on the row-summed data until their worst-case error is at most refine_to.

    Each round replaces every orientation by its 27 children, fits them all to the row-summed histogram
    (:func:`anri.index.orientations.fit`, as the filter stage) and keeps those above min_lr
    (:func:`anri.index.orientations.select_by_likelihood`). The worst case falls by 3 per round.

    Parameters
    ----------
    data
        [n_rings * n_e * n_o] histogram with the dty rows summed
    U, comp
        [Nq, 3, 3] orientations and [Nq] their completeness (carried to the children)
    step, worst
        Their grid's spacing and worst-case distance to the truth (deg)
    refine_to
        Stop when the worst case is at most this (deg)
    B, rings, geom, bins, etacut, censor
        As :func:`anri.index.orientations.fit`
    min_lr, keep
        As the filter stage: likelihood ratio needed, and at most this many orientations
    log
        Progress messages

    Returns
    -------
    U, comp, step, worst
        The refined orientations, their completeness, spacing and worst case
    """
    import time

    from .orientations import fit, select_by_likelihood

    while worst > refine_to:
        t0 = time.perf_counter()
        n_par = len(U)
        U, comp = children(U, step), np.repeat(comp, N_CHILD)
        step, worst = step / 3, worst / 3
        _, lr = fit(data, U, B, rings, geom, bins, etacut, log=lambda *m: None, censor=censor)
        kept_ = select_by_likelihood(np.arange(len(U)), lr, min_lr, keep)
        U, comp = U[kept_], comp[kept_]
        log(f"refine: {n_par} orientations -> {n_par * N_CHILD} children {step:.3g} deg apart (worst case "
            f"{worst:.3g} deg), row-summed fit: {len(U)} kept with likelihood ratio > {min_lr:g} "
            f"({time.perf_counter() - t0:.0f} s)")  # fmt: skip
        if len(U) == 0:
            msg = f"no orientation kept with likelihood ratio > {min_lr:g} at {step:.3g} deg"
            raise ValueError(msg)
    return U, comp, step, worst


def level_rows(radius: float, worst: float, ystep: float, levels: int) -> list[int]:
    """Return the voxel width, in dty rows, of each spatial level: about radius x the level's worst case.

    Level k fits orientations off by up to worst / 3^k: voxels narrower than radius x that (radians) cannot be fitted
    consistently (the error moves a voxel's predicted row by up to that much). The last level is always one row.

    Parameters
    ----------
    radius
        The fitted voxels' largest distance from the rotation axis
    worst
        The first level's worst-case orientation error (deg)
    ystep
        The dty step
    levels
        Levels after the first

    Returns
    -------
    list[int]
        levels + 1 widths, the last 1
    """
    rows = [max(1, math.ceil(radius * math.radians(worst / 3**k) / ystep)) for k in range(levels + 1)]
    return [*rows[:-1], 1]


def kept(f: np.ndarray, cand: np.ndarray, min_share: float, m: int) -> tuple:
    """Return each voxel's orientations holding at least min_share of its occupancy, largest first, at most m.

    Parameters
    ----------
    f, cand
        [Nv, K] occupancies and orientation indices of a fit (-1: none)
    min_share, m
        See above

    Returns
    -------
    cand, f, keep: np.ndarray
        [Nv, min(m, K)] each: the voxel's orientations, largest first, their occupancies, and which of them are kept
    """
    order = np.argsort(-f, 1)[:, :m]
    fs, cs = np.take_along_axis(f, order, 1), np.take_along_axis(cand, order, 1)
    tot = np.maximum(f.sum(1, keepdims=True), 1e-30)
    return cs, fs, (fs >= min_share * tot) & (fs > 0) & (cs >= 0)


def inherit(f: ArrayLike, cand: ArrayLike, n_orient: int, min_share: float = 0.01, m: int = 16) -> tuple:
    """Return the orientations to subdivide, and each voxel's children, each at a 27th of its parent's occupancy.

    Parameters
    ----------
    f, cand
        [Nv, K] occupancies and orientation indices of a fit (-1: none)
    n_orient
        Number of orientations cand indexes
    min_share, m
        A voxel keeps its orientations holding at least min_share of its occupancy, at most m (:func:`kept`)

    Returns
    -------
    use: np.ndarray
        [n_orient] the orientations some voxel keeps: the children of the j-th of them are rows 27 j to 27 j + 26 of
        :func:`children` of ``U[use]``
    f0, cand: np.ndarray
        [Nv, 27 min(m, K)] each voxel's children (indices into those, -1 for an empty slot) and their starting
        occupancies
    """
    cs, fs, keep = kept(np.asarray(f), np.asarray(cand), min_share, m)
    use = np.zeros(n_orient, bool)
    use[cs[keep]] = True
    rank = np.cumsum(use) - 1  # each subdivided orientation's block of children
    base = np.where(keep, rank[np.clip(cs, 0, n_orient - 1)] * N_CHILD, -1)  # (padding past n_orient: never kept)
    c = np.where(keep[..., None], base[..., None] + np.arange(N_CHILD), -1).reshape(len(cs), -1)
    w = np.where(keep[..., None], np.repeat(fs[..., None] / N_CHILD, N_CHILD, 2), 0.0).reshape(len(cs), -1)
    return use, w.astype(np.float32), c.astype(np.int32)
