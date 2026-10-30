"""Local search: sharpen a fit's orientations off the search grid, around those each voxel uses.

The search grid sets how close an orientation of the fit can be to the truth (up to 0.87 x the step). Where peaks are
sharper than that, the fit explains a grain's peaks with several grid neighbours, and with orientations that share
some of its reflections: a twin whose grid point lands nearer the exact twin than the grain's own lands to the grain
takes those reflections, a ghost twin. A finer grid over the whole fundamental zone costs the cube of the refinement;
instead, each level replaces every orientation a voxel uses by the 27 rotations of a 3 x 3 x 3 grid around it, a
third of the step apart (:func:`children`), and fits each voxel again over the children of its own orientations only
(:func:`inherit`), starting from its parents' occupancy. Occupancy only moves within each voxel's own orientations,
so a real twin in a thin lamella keeps its own children, which a ranking over every orientation would crowd out.
"""

from __future__ import annotations

import itertools

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


def covering(f: np.ndarray, cand: np.ndarray, cover: float, m: int) -> tuple:
    """Return each voxel's most occupied orientations that together hold cover of its occupancy, at most m.

    Parameters
    ----------
    f, cand
        [Nv, K] occupancies and orientation indices of a fit (-1: none)
    cover, m
        See above

    Returns
    -------
    cand, f, keep: np.ndarray
        [Nv, min(m, K)] each: the voxel's orientations, largest first, their occupancies, and which of them are used
    """
    order = np.argsort(-f, 1)[:, :m]
    fs, cs = np.take_along_axis(f, order, 1), np.take_along_axis(cand, order, 1)
    before = (np.cumsum(fs, 1) - fs) / np.maximum(f.sum(1, keepdims=True), 1e-30)  # held by the larger ones
    return cs, fs, (before < cover) & (fs > 0) & (cs >= 0)


def used(f: ArrayLike, cand: ArrayLike, n_orient: int, cover: float = 0.95, m: int = 8) -> np.ndarray:
    """Return which orientations some voxel uses: among its (at most m) most occupied, holding cover of its occupancy.

    Parameters
    ----------
    f, cand
        [Nv, K] occupancies and orientation indices of a fit (-1: none)
    n_orient
        Number of orientations cand indexes
    cover, m
        See above

    Returns
    -------
    np.ndarray
        [n_orient] booleans
    """
    cs, _, keep = covering(np.asarray(f), np.asarray(cand), cover, m)
    out = np.zeros(n_orient, bool)
    out[cs[keep]] = True
    return out


def inherit(f: ArrayLike, cand: ArrayLike, use: ArrayLike, cover: float = 0.95, m: int = 8) -> tuple:
    """Return each voxel's candidates for the next level: the children of its orientations, each at a 27th of them.

    Parameters
    ----------
    f, cand
        [Nv, K] occupancies and orientation indices of a fit (-1: none)
    use
        [n_orient] the orientations subdivided (:func:`used` with the same cover and m): the children of the j-th of
        them are rows 27 j to 27 j + 26 of :func:`children` of ``U[use]``
    cover, m
        As in :func:`used`

    Returns
    -------
    f0, cand: np.ndarray
        [Nv, 27 m] starting occupancies and indices into the children (-1 for an empty slot), for
        :func:`anri.index.occupancy.fit_occupancy`'s given
    """
    cs, fs, keep = covering(np.asarray(f), np.asarray(cand), cover, m)
    rank = np.cumsum(np.asarray(use)) - 1  # each subdivided orientation's block of children
    base = np.where(keep, rank[np.maximum(cs, 0)] * N_CHILD, -1)
    c = np.where(keep[..., None], base[..., None] + np.arange(N_CHILD), -1).reshape(len(cs), -1)
    w = np.where(keep[..., None], np.repeat(fs[..., None] / N_CHILD, N_CHILD, 2), 0.0).reshape(len(cs), -1)
    return w.astype(np.float32), c.astype(np.int32)
