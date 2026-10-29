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

from anri.crystal.lattice import U_B_to_UBI
from anri.crystal.orientation import orientation_grid
from anri.diffract.factors import lorentz_polarisation
from anri.diffract.scatter import k_out_to_tth_eta
from anri.fwd.peak import hkl_to_k_omega
from anri.geom.gonio import wrap_omega

from .data import coarsen_coverage, eta_bin, omega_bin

GRID_STEPS = (3.0, 2.5, 2.0, 1.5, 1.0)  # tried by choose_grid, coarsest first


@jax.jit
def predict(
    U: jax.Array, B: jax.Array, hkls: jax.Array, geom: dict
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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
        Geometry dict with "wavelength", "k_in_lab", "wedge", "chi" and "pol_factor"

    Returns
    -------
    eta, omega, valid, lp: jax.Array
        [Nq, 2 Nh] each, prediction ``j = 2 h + branch``: eta and omega in degrees, whether it diffracts, and its
        Lorentz x polarisation factor (:func:`anri.diffract.factors.lorentz_polarisation`)
    """
    ubi = U_B_to_UBI(U, B)

    def one(u: jax.Array, h: jax.Array, e: jax.Array) -> tuple:
        k_in, k_out, om, ok = hkl_to_k_omega(
            u, h, e, geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, geom["wedge"], geom["chi"]
        )
        lp = lorentz_polarisation(k_in, k_out, geom["wedge"], geom["chi"], geom["pol_factor"])
        return k_out_to_tth_eta(k_out, geom["k_in_lab"])[1], om, ok, lp

    f = jax.vmap(jax.vmap(jax.vmap(one, (None, None, 0)), (None, 0, None)), (0, None, None))
    eta, om, ok, lp = f(ubi, hkls, jnp.array([1.0, -1.0], U.dtype))
    nq = U.shape[0]
    return eta.reshape(nq, -1), om.reshape(nq, -1), ok.reshape(nq, -1), lp.reshape(nq, -1)


def match_tolerances(
    eta: jax.Array, ring_j: jax.Array, ring_tth: jax.Array, ring_hw: jax.Array, delta: float, frame_step: float
) -> tuple[jax.Array, jax.Array]:
    """Compute the matching tolerances in eta and omega of predictions of grid orientations.

    The truth is up to ``delta`` from the nearest grid point. To first order a rotation by delta moves a reflection
    by up to ``delta / cos(theta) (1 + tan(theta) |cot(eta)|)`` in eta and ``delta / (cos(theta) |sin(eta)|)`` in
    omega (checked numerically up to 2 degrees for ``|sin(eta)| > 0.3``). Added: in eta, the ring's measured
    half-width (:func:`anri.index.data.ring_widths`), seen as a displacement on the detector across the ring; in omega,
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
    cov: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Fraction of each orientation's valid predictions with a lit bin within its own tolerance box.

    Predictions where less than half of the detector is measured (``cov``) are not counted: they could not be lit.

    Parameters
    ----------
    table
        Summed-area table of the lit map, from :func:`anri.index.data.lit_table`
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
    cov
        [n_rings, n_e] measured share of each (ring, eta bin) (:func:`anri.index.data.detector_coverage`), or None

    Returns
    -------
    completeness, n_valid: jax.Array
        [Nq] each
    """
    ie, io = eta_bin(eta, b_e, n_e), omega_bin(wrap_omega(om, om0), om0, b_o)
    de = jnp.minimum(jnp.ceil(tol_e / b_e).astype(jnp.int32), n_e - 1)
    do = jnp.ceil(tol_o / b_o).astype(jnp.int32)
    inside = ok & (io >= 0) & (io < n_o)
    if cov is not None:
        inside = inside & (cov[ring_j[None, :], ie] >= 0.5)
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
        From :func:`anri.index.data.ring_table`, plus "hw" [Nr] measured half-widths (degrees) and, optionally,
        "coverage" (the histogram stage's measured share of each (ring, eta bin))
    geom
        Geometry dict
    lit
        The lit map: "table" (:func:`anri.index.data.lit_table`), "om0", "bins" (b_e, b_o, n_e, n_o), "frame_step" and
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
    cov = rings.get("coverage")
    cov = None if cov is None else jnp.asarray(coarsen_coverage(cov, n_e), jnp.float32)
    U = np.asarray(U, np.float32)
    out = []
    for s0 in range(0, len(U), chunk):
        u = U[s0 : s0 + chunk]
        m = len(u)
        u = np.concatenate([u, np.repeat(np.eye(3, dtype=np.float32)[None], chunk - m, 0)]) if len(U) > chunk else u
        eta, om, ok, _ = predict(jnp.asarray(u), jnp.asarray(B, jnp.float32), hkls, geom)
        te, to = match_tolerances(eta, ring_j, tth, hw, delta, lit["frame_step"])
        ok = ok & (jnp.abs(jnp.sin(jnp.radians(eta))) > lit["etacut"])
        c, _ = completeness(lit["table"], eta, om, ok, ring_j, te, to, lit["om0"], b_e, b_o, n_e, n_o, cov)
        out.append(np.asarray(c)[:m])
    return np.concatenate(out)


def grid_chance(
    step: float, ops: ArrayLike, B: ArrayLike, rings: dict, geom: dict, lit: dict, rng: np.random.Generator,
    n_sample: int = 1 << 14,
) -> tuple[float, int, float]:  # fmt: skip
    """Return a grid step's chance completeness: the median completeness over a random sample of its orientations.

    Parameters
    ----------
    step
        Grid step (degrees), see :func:`anri.crystal.orientation.orientation_grid`
    ops, B, rings, geom, lit
        See :func:`choose_grid`
    rng
        Random generator for the sample (:func:`choose_grid` draws every step's from one, in order)
    n_sample
        Orientations in the sample

    Returns
    -------
    chance, n, delta
        The chance completeness, the grid's number of orientations and its worst-case distance from the truth
    """
    U, delta = orientation_grid(step, ops)
    sample = U[rng.choice(len(U), min(len(U), n_sample), replace=False)]
    return float(np.median(completeness_of(sample, delta, B, rings, geom, lit))), len(U), delta


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
        c, n, _ = grid_chance(step, ops, B, rings, geom, lit, rng, n_sample)
        log(f"  grid {step} deg ({n} orientations): chance completeness {c:.2f}")
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
        The grid and its worst-case spacing, from :func:`anri.crystal.orientation.orientation_grid`
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
    chance, halfway = completeness_levels(comp)
    thr = min_comp if min_comp is not None else halfway
    kept = select_by_completeness(comp, thr, keep)
    n_above = int(np.sum(comp >= thr))
    return kept, comp, {"chance": chance, "min_comp": thr, "n_above": n_above, "capped": n_above > keep}


def completeness_levels(comp: np.ndarray) -> tuple[float, float]:
    """Return the chance level of a grid's completeness (its median) and halfway from it to the maximum."""
    chance = float(np.median(comp))
    return chance, chance + 0.5 * (float(np.max(comp)) - chance)


def select_by_completeness(comp: np.ndarray, min_comp: float, keep: int) -> np.ndarray:
    """Return the indices of the orientations at or above min_comp, the most complete first, at most keep."""
    above = np.flatnonzero(comp >= min_comp)
    return above[np.argsort(comp[above])[::-1]][:keep]


def predictions(
    U: ArrayLike, B: ArrayLike, rings: dict, geom: dict, etacut: float, qc: int = 16, chunk: int = 1 << 16
) -> tuple:
    """Predictions of an orientation list for :func:`~anri.index.occupancy.fit_occupancy`, padded to a multiple of qc.

    Lists longer than chunk (e.g. a local search's millions) are predicted chunk by chunk, all of one shape (one
    compile), and padded to whole chunks.

    Parameters
    ----------
    U
        [Nq, 3, 3] orientations
    B
        [3, 3] B matrix
    rings
        From :func:`anri.index.data.ring_table`
    geom
        Geometry dict
    etacut
        Reflections with ``|sin eta|`` at or below this are not used
    qc
        Pad to a multiple of this
    chunk
        Orientations predicted in one call at most (a multiple of qc)

    Returns
    -------
    tuple
        (eta, om, use, w), each [Nq padded, Nj]; w is Lorentz x polarisation x ``rings["F2"]`` (1 if absent)
    """
    U = np.asarray(U, np.float32)
    nq = len(U)
    n = nq + (-nq % qc)
    if n > chunk:
        n += -n % chunk
    Up = np.concatenate([U, np.repeat(U[:1], n - nq, 0)])
    Bj, hkls = jnp.asarray(B, jnp.float32), jnp.asarray(rings["hkls"])
    step = min(n, chunk)
    parts = [predict(jnp.asarray(Up[i : i + step]), Bj, hkls, geom) for i in range(0, n, step)]
    eta, om, ok, lp = (parts[0][j] if len(parts) == 1 else jnp.concatenate([p[j] for p in parts]) for j in range(4))
    use = ok & (jnp.abs(jnp.sin(jnp.radians(eta))) > etacut) & (jnp.arange(n) < nq)[:, None]
    F2 = jnp.asarray(np.repeat(rings.get("F2", np.ones(len(rings["hkls"]))), 2), jnp.float32)
    return eta, om, use, jnp.where(use, lp * F2[None], 0.0)
