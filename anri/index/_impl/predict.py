"""Predicted spots of trial orientations, matching tolerances, completeness, and pruning an orientation grid.

Completeness is the fraction of an orientation's predicted reflections that land on intensity in the row-summed
"lit" map. The tolerance of each prediction follows from how far the truth can be from its nearest grid orientation:
to first order a rotation by delta moves a reflection by up to ``delta / cos(theta) (1 + tan(theta) |cot(eta)|)`` in
eta and ``delta / (cos(theta) |sin(eta)|)`` in omega.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from anri.crystal import orientation_grid
from anri.fwd import hkl_to_k_omega, lorentz, polarisation
from anri.geom import sample_to_lab

from .data import _tth_eta

GRID_STEPS = (3.0, 2.5, 2.0, 1.5, 1.0)  # tried by choose_grid, coarsest first


@jax.jit
def predict(U: jax.Array, B: jax.Array, hkls: jax.Array, geom: dict) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Predict (eta, omega) of every reflection of some orientations, with both omega solutions.

    Parameters
    ----------
    U
        [Nq, 3, 3] orientations (crystal to sample)
    B
        [3, 3] reciprocal-space B matrix
    hkls
        [Nh, 3] reflections
    geom
        Geometry dict with "wavelength", "k_in_lab", "wedge" and "chi"

    Returns
    -------
    eta, omega, valid: jax.Array
        [Nq, 2 Nh] each, prediction ``j = 2 h + branch``; eta and omega in degrees
    """
    ubi = jnp.linalg.inv(U @ B)

    def one(u: jax.Array, h: jax.Array, e: jax.Array) -> tuple:
        _, k_out, om, ok = hkl_to_k_omega(
            u, h, e, geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, geom["wedge"], geom["chi"]
        )
        return _tth_eta(k_out, geom["k_in_lab"])[1], om, ok

    f = jax.vmap(jax.vmap(jax.vmap(one, (None, None, 0)), (None, 0, None)), (0, None, None))
    eta, om, ok = f(ubi, hkls, jnp.array([1.0, -1.0], U.dtype))
    nq = U.shape[0]
    return eta.reshape(nq, -1), om.reshape(nq, -1), ok.reshape(nq, -1)


@jax.jit
def lorentz_polarisation(U: jax.Array, B: jax.Array, hkls: jax.Array, geom: dict) -> jax.Array:
    """Lorentz x polarisation factor of every prediction of some orientations (see :func:`predict`).

    Parameters
    ----------
    U, B, hkls
        See :func:`predict`
    geom
        Geometry dict, as for :func:`predict` plus "pol_factor"

    Returns
    -------
    jax.Array
        [Nq, 2 Nh] factors
    """
    ubi = jnp.linalg.inv(U @ B)
    axis = sample_to_lab(jnp.array([0.0, 0.0, 1.0], U.dtype), 0.0, geom["wedge"], geom["chi"], 0.0, 0.0)

    def one(u: jax.Array, h: jax.Array, e: jax.Array) -> jax.Array:
        k_in, k_out, _, _ = hkl_to_k_omega(
            u, h, e, geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, geom["wedge"], geom["chi"]
        )
        return lorentz(k_in, k_out, axis) * polarisation(k_in, k_out, geom["pol_factor"])

    f = jax.vmap(jax.vmap(jax.vmap(one, (None, None, 0)), (None, 0, None)), (0, None, None))
    return f(ubi, hkls, jnp.array([1.0, -1.0], U.dtype)).reshape(U.shape[0], -1)


def match_tolerances(
    eta: jax.Array, ring_j: jax.Array, ring_tth: jax.Array, ring_hw: jax.Array, delta: float, frame_step: float
) -> tuple[jax.Array, jax.Array]:
    """Compute the matching tolerances in eta and omega of predictions of grid orientations.

    The truth is up to ``delta`` from the nearest grid point. To first order a rotation by delta moves a reflection
    by up to ``delta / cos(theta) (1 + tan(theta) |cot(eta)|)`` in eta and ``delta / (cos(theta) |sin(eta)|)`` in
    omega (checked numerically up to 2 degrees for ``|sin(eta)| > 0.3``). Added: in eta, the ring's measured
    half-width (:func:`anri.index.ring_widths`), seen as a displacement on the detector across the ring; in omega,
    half a frame. Peak widths are not added: a broad peak lights a broad region of the lit map.

    Parameters
    ----------
    eta
        [Nq, Nj] predicted eta (degrees)
    ring_j
        [Nj] ring of each prediction
    ring_tth, ring_hw
        [Nr] 2theta and measured half-width of each ring (degrees)
    delta
        Worst-case distance from the truth to the grid (degrees)
    frame_step
        Omega step of the frames (degrees)

    Returns
    -------
    tol_eta, tol_omega: jax.Array
        [Nq, Nj] each, in degrees
    """
    th = jnp.radians(ring_tth[ring_j] / 2)[None]
    s = jnp.maximum(jnp.abs(jnp.sin(jnp.radians(eta))), 1e-3)
    c = jnp.abs(jnp.cos(jnp.radians(eta)))
    tol_e = delta / jnp.cos(th) * (1 + jnp.tan(th) * c / s) + ring_hw[ring_j][None] / (
        jnp.sin(2 * th) * jnp.cos(2 * th)
    )
    tol_o = delta / (jnp.cos(th) * s) + frame_step / 2
    return tol_e, tol_o


@partial(jax.jit, static_argnames=("n_e", "n_o"))
def completeness(
    table: jax.Array,
    eta: jax.Array,
    om: jax.Array,
    ok: jax.Array,
    ring_j: jax.Array,
    tol_e: jax.Array,
    tol_o: jax.Array,
    om0: float,
    b_e: float,
    b_o: float,
    n_e: int,
    n_o: int,
) -> tuple[jax.Array, jax.Array]:
    """Fraction of each orientation's valid predictions with a lit bin within its own tolerance box.

    Parameters
    ----------
    table
        Summed-area table of the lit map, from :func:`anri.index.lit_table`
    eta, om, ok
        [Nq, Nj] predictions, from :func:`predict` (with any extra cuts applied to ok)
    ring_j
        [Nj] ring of each prediction
    tol_e, tol_o
        [Nq, Nj] tolerances (degrees), e.g. from :func:`match_tolerances`; tol_e below 360
    om0, b_e, b_o
        The lit map's first omega bin edge and bin widths
    n_e, n_o
        The lit map's numbers of eta and omega bins

    Returns
    -------
    completeness, n_valid: jax.Array
        [Nq] each
    """
    om_w = jnp.mod(om - om0, 360.0) + om0
    ie = jnp.floor((eta + 180.0) / b_e).astype(jnp.int32) % n_e
    io = jnp.floor((om_w - om0) / b_o).astype(jnp.int32)
    de = jnp.minimum(jnp.ceil(tol_e / b_e).astype(jnp.int32), n_e - 1)
    do = jnp.ceil(tol_o / b_o).astype(jnp.int32)
    inside = ok & (io >= 0) & (io < n_o)
    e0, e1 = ie + n_e - de, ie + n_e + de + 1
    o0, o1 = jnp.clip(io - do, 0, n_o), jnp.clip(io + do + 1, 0, n_o)
    r = ring_j[None, :]
    count = table[r, e1, o1] - table[r, e0, o1] - table[r, e1, o0] + table[r, e0, o0]
    hit = (count > 0) & inside
    return jnp.sum(hit, 1) / jnp.maximum(jnp.sum(inside, 1), 1), jnp.sum(inside, 1)


def completeness_of(
    U: ArrayLike, delta: float, B: ArrayLike, rings: dict, geom: dict, lit: dict, chunk: int = 1 << 15
) -> np.ndarray:
    """Completeness of many orientations, in chunks.

    Parameters
    ----------
    U
        [N, 3, 3] orientations
    delta
        Worst-case distance from the truth (degrees), which sets the tolerances
    B
        [3, 3] B matrix
    rings
        From :func:`anri.index.ring_table`, plus "hw" [Nr] measured half-widths (degrees)
    geom
        Geometry dict
    lit
        The lit map: "table" (:func:`anri.index.lit_table`), "om0", "bins" (b_e, b_o, n_e, n_o), "frame_step" and
        "etacut" (reflections with ``|sin eta|`` at or below it are not used)
    chunk
        Orientations per call

    Returns
    -------
    np.ndarray
        [N] completeness
    """
    b_e, b_o, n_e, n_o = lit["bins"]
    hkls, ring_j = jnp.asarray(rings["hkls"]), jnp.asarray(rings["ring_j"])
    tth, hw = jnp.asarray(rings["tth"], jnp.float32), jnp.asarray(rings["hw"], jnp.float32)
    U = np.asarray(U, np.float32)
    out = []
    for s0 in range(0, len(U), chunk):
        u = U[s0 : s0 + chunk]
        m = len(u)
        u = np.concatenate([u, np.repeat(np.eye(3, dtype=np.float32)[None], chunk - m, 0)]) if len(U) > chunk else u
        eta, om, ok = predict(jnp.asarray(u), jnp.asarray(B, jnp.float32), hkls, geom)
        te, to = match_tolerances(eta, ring_j, tth, hw, delta, lit["frame_step"])
        ok = ok & (jnp.abs(jnp.sin(jnp.radians(eta))) > lit["etacut"])
        c, _ = completeness(lit["table"], eta, om, ok, ring_j, te, to, lit["om0"], b_e, b_o, n_e, n_o)
        out.append(np.asarray(c)[:m])
    return np.concatenate(out)


def choose_grid(
    ops: ArrayLike,
    B: ArrayLike,
    rings: dict,
    geom: dict,
    lit: dict,
    max_chance: float = 0.5,
    steps: tuple = GRID_STEPS,
    n_sample: int = 1 << 14,
    seed: int = 0,
    log: Callable = print,
) -> float:
    """Choose the coarsest grid step whose chance completeness is at most max_chance.

    Chance completeness is the median completeness over a random sample of the grid: most grid orientations are
    wrong. A coarser grid has larger tolerances, so more chance matches. If no step qualifies, the finest is used.

    Parameters
    ----------
    ops
        [n, 3, 3] Laue-group rotations
    B, rings, geom, lit
        See :func:`completeness_of`
    max_chance
        Largest acceptable chance completeness
    steps
        Grid steps to try (degrees), coarsest first
    n_sample
        Orientations sampled per step
    seed
        Random seed for the sample
    log
        Progress messages

    Returns
    -------
    float
        Grid step in degrees
    """
    rng = np.random.default_rng(seed)
    for step in steps:
        U, delta = orientation_grid(step, ops)
        c = float(
            np.median(
                completeness_of(U[rng.choice(len(U), min(len(U), n_sample), replace=False)], delta, B, rings, geom, lit)
            )
        )
        log(f"  grid {step} deg ({len(U)} orientations): chance completeness {c:.2f}")
        if c <= max_chance:
            return step
    log(f"  no grid step reaches chance completeness <= {max_chance}: using the finest, {steps[-1]} deg. The lit map is "
        "crowded: raise the lit threshold, and keep everything above the completeness threshold")  # fmt: skip
    return steps[-1]


def prune(
    U: ArrayLike, delta: float, B: ArrayLike, rings: dict, geom: dict, lit: dict, min_comp: float | None = None,
    keep: int = 100000,
) -> tuple[np.ndarray, np.ndarray, dict]:  # fmt: skip
    """Keep the grid orientations whose completeness passes a threshold.

    The default threshold is halfway between chance (the grid's median completeness) and the grid's maximum.

    Parameters
    ----------
    U, delta
        The grid and its worst-case spacing, from :func:`anri.crystal.orientation_grid`
    B, rings, geom, lit
        See :func:`completeness_of`
    min_comp
        Threshold; default halfway between chance and the maximum
    keep
        At most this many are kept, the most complete first

    Returns
    -------
    kept: np.ndarray
        [Nk] indices into U, most complete first
    comp: np.ndarray
        [N] completeness of every grid orientation
    info: dict
        "chance", "min_comp", "n_above" (orientations above the threshold) and "capped" (whether keep bit)
    """
    comp = completeness_of(U, delta, B, rings, geom, lit)
    chance = float(np.median(comp))
    thr = min_comp if min_comp is not None else chance + 0.5 * (float(comp.max()) - chance)
    above = np.flatnonzero(comp >= thr)
    kept = above[np.argsort(comp[above])[::-1]][:keep]
    return kept, comp, {"chance": chance, "min_comp": thr, "n_above": len(above), "capped": len(above) > keep}


def predictions(U: ArrayLike, B: ArrayLike, rings: dict, geom: dict, etacut: float, qc: int = 16) -> tuple:
    """Predictions of an orientation list for :func:`fit_occupancy`, padded to a multiple of qc.

    Parameters
    ----------
    U
        [Nq, 3, 3] orientations
    B
        [3, 3] B matrix
    rings
        From :func:`anri.index.ring_table`
    geom
        Geometry dict
    etacut
        Reflections with ``|sin eta|`` at or below this are not used
    qc
        Pad to a multiple of this

    Returns
    -------
    tuple
        (eta, om, use, w), each [Nq padded, Nj]; w is Lorentz x polarisation x ``rings["F2"]`` (1 if absent)
    """
    U = np.asarray(U, np.float32)
    nq = len(U)
    Up = jnp.asarray(np.concatenate([U, np.repeat(U[:1], -nq % qc, 0)]))
    Bj, hkls = jnp.asarray(B, jnp.float32), jnp.asarray(rings["hkls"])
    eta, om, ok = predict(Up, Bj, hkls, geom)
    lp = lorentz_polarisation(Up, Bj, hkls, geom)
    use = ok & (jnp.abs(jnp.sin(jnp.radians(eta))) > etacut) & (jnp.arange(Up.shape[0]) < nq)[:, None]
    F2 = jnp.asarray(np.repeat(rings.get("F2", np.ones(len(rings["hkls"]))), 2), jnp.float32)
    return eta, om, use, jnp.where(use, lp * F2[None], 0.0)
