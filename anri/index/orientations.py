"""Which orientations are in the sample at all: one occupancy per orientation, fitted to the row-summed data.

Before fitting voxels, the dty rows of the histogram are summed and each grid orientation gets a single, global
occupancy by MLEM: like the voxel fit, it is intensity-aware and explains overlapping spots jointly. An
orientation's occupancy grows with its grain's volume, so orientations are kept by how much they matter to the fit,
not by how much there is of them: the likelihood ratio, the increase in deviance if that orientation alone were
removed. A small grain with clean spots scores well; a decoy that only borrows intensity from real grains does not.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .data import coarsen_coverage, eta_omega_corners
from .occupancy import censored_ratio, deviance, squarem
from .search import predict


def _system(
    U: jax.Array, B: jax.Array, hkls: jax.Array, F2: jax.Array, ring_j: jax.Array, geom: dict, etacut: float,
    cov: jax.Array | None, bins: tuple,
) -> tuple[jax.Array, jax.Array]:  # fmt: skip
    """(cell index, weight) [Q, Nj, 4] of orientations U [Q, 3, 3] in a row-summed (ring, eta, omega) histogram.

    cov: [n_rings, n_e] measured share of each (ring, eta bin) (:func:`anri.index.data.detector_coverage`), or None.
    """
    b_e, b_o, n_e, n_o, om0 = bins
    eta, om, ok, lp = predict(U, B, hkls, geom)
    w = lp * F2[None]
    use = ok & (jnp.abs(jnp.sin(jnp.radians(eta))) > etacut)
    idx, wt = [], []
    for ie, io, we, wo, inside in eta_omega_corners(eta, om, om0, b_e, b_o, n_e, n_o):
        if cov is not None:
            we = we * cov[ring_j[None], ie]
        good = use & inside
        idx.append(jnp.where(good, (ring_j[None] * n_e + ie) * n_o + io, -1))
        wt.append(jnp.where(good, w * we * wo, 0.0))
    return jnp.stack(idx, -1), jnp.stack(wt, -1)


def _coverage(rings: dict, n_e: int) -> jax.Array | None:
    """Return the rings' detector coverage in n_e eta bins, or None without one."""
    if rings.get("coverage") is None:
        return None
    return jnp.asarray(coarsen_coverage(rings["coverage"], n_e), jnp.float32)


@partial(jax.jit, static_argnames=("bins", "n_cells"))
def _forward(
    g: jax.Array, U: jax.Array, B: jax.Array, hkls: jax.Array, F2: jax.Array, ring_j: jax.Array, geom: dict,
    etacut: float, cov: jax.Array | None, bins: tuple, n_cells: int,
) -> jax.Array:  # fmt: skip
    def step(acc: jax.Array, ch: tuple) -> tuple:
        gc, uc = ch
        idx, wt = _system(uc, B, hkls, F2, ring_j, geom, etacut, cov, bins)
        c = (gc[:, None, None] * wt).ravel().astype(acc.dtype)
        return acc + jax.ops.segment_sum(c, jnp.where(idx >= 0, idx, n_cells).ravel(), n_cells + 1)[:-1], None

    acc, _ = jax.lax.scan(step, jnp.zeros(n_cells, jnp.float32), (g, U))
    return acc


@partial(jax.jit, static_argnames=("bins",))
def _backward(
    r: jax.Array, U: jax.Array, B: jax.Array, hkls: jax.Array, F2: jax.Array, ring_j: jax.Array, geom: dict,
    etacut: float, cov: jax.Array | None, bins: tuple,
) -> jax.Array:  # fmt: skip
    def one(uc: jax.Array) -> jax.Array:
        idx, wt = _system(uc, B, hkls, F2, ring_j, geom, etacut, cov, bins)
        return jnp.sum(wt * jnp.where(idx >= 0, r[jnp.clip(idx, 0)], 0.0), (1, 2))

    return jax.lax.map(one, U)


@partial(jax.jit, static_argnames=("bins",))
def _likelihood_ratio(
    d: jax.Array, mu: jax.Array, g: jax.Array, U: jax.Array, B: jax.Array, hkls: jax.Array, F2: jax.Array,
    ring_j: jax.Array, geom: dict, etacut: float, cov: jax.Array | None, bins: tuple, censor: float,
) -> jax.Array:  # fmt: skip
    def one(ch: tuple) -> jax.Array:
        gc, uc = ch
        idx, wt = _system(uc, B, hkls, F2, ring_j, geom, etacut, cov, bins)
        m = jnp.where(idx >= 0, mu[jnp.clip(idx, 0)], 0.0)
        dd = jnp.where(idx >= 0, d[jnp.clip(idx, 0)], 0.0)
        ga = gc[:, None, None] * wt
        rest = jnp.maximum(m - ga, 1e-6 * jnp.maximum(m, 1e-30))
        # log(m / rest) as log1p((m - rest) / rest): in float32, m / rest of a weak orientation rounds to 1 + a few
        # ulps, and times counts of 1e6 that is noise of order 1 in the ratio (which then changed from run to run)
        x = jnp.where(m - ga > 1e-6 * m, ga, m - rest) / jnp.maximum(rest, 1e-30)
        # empty bins: only the model above the censoring level costs likelihood (censor = 0: all of it, -ga)
        empty = jnp.where(censor > 0, jnp.maximum(rest - censor, 0.0) - jnp.maximum(m - censor, 0.0), -ga)
        term = jnp.where((idx >= 0) & (m > 0), jnp.where(dd > 0, dd * jnp.log1p(x) - ga, empty), 0.0)
        return 2 * jnp.sum(term, (1, 2))

    return jax.lax.map(one, (g, U))


def fit(
    d: ArrayLike,
    U: ArrayLike,
    B: ArrayLike,
    rings: dict,
    geom: dict,
    bins: tuple,
    etacut: float = 0.2,
    n_iter: int = 20,
    qc: int = 256,
    log: Callable = print,
    censor: float = 0.0,
    return_model: bool = False,
) -> tuple:
    """Fit one occupancy per orientation to a row-summed histogram by MLEM, and score each orientation.

    MLEM is accelerated by SQUAREM, as the voxel fit (:func:`anri.index.occupancy.squarem`): each iteration is two
    MLEM updates, an extrapolation and one more update, never fitting worse than plain MLEM. Decoys that borrow intensity
    from real grains fade slowly under plain MLEM; on AP1_1, 20 plain updates left them with likelihood ratios down to
    -5e5 (removing them would have improved the fit), 20 SQUAREM iterations -6e3, in 20 s against 12 s.

    The predictions of each orientation are computed afresh in every pass, so memory stays at a chunk of qc
    orientations whatever the number of orientations.

    Parameters
    ----------
    d
        [n_rings * n_e * n_o] histogram with the dty rows summed (e.g. the voxel fit's histogram, summed over rows)
    U
        [Nq, 3, 3] orientations, e.g. the grid orientations above chance completeness
    B
        [3, 3] B matrix
    rings
        From :func:`anri.index.data.ring_table` ("hkls", "ring_j" and, optionally, "F2" and the measured share of
        each (ring, eta bin), "coverage" from the histogram stage, averaged here into the histogram's eta bins)
    geom
        Geometry dict, see :func:`anri.index.search.predict`
    bins
        (b_e, b_o, n_e, n_o, om0): the histogram's eta and omega bin widths (degrees) and counts, and its first omega
        bin edge
    etacut
        Reflections with ``|sin eta|`` at or below this are not used
    n_iter
        SQUAREM iterations (3 MLEM updates each)
    qc
        Orientations per chunk
    log
        Progress messages
    censor
        Empty bins censored above this many counts per bin, in the fit and in the likelihood ratio
        (:func:`anri.index.occupancy.censored_ratio`); 0: plain Poisson
    return_model
        Also return the fitted histogram, e.g. for its deviance (:func:`anri.index.occupancy.deviance`)

    Returns
    -------
    occupancy: np.ndarray
        [Nq] global occupancy of each orientation (in the units of d over Lorentz x polarisation x F^2)
    likelihood_ratio: np.ndarray
        [Nq] increase in the Poisson deviance if that orientation alone were removed, the others fixed:
        ``2 sum_b [d_b log(mu_b / (mu_b - g a_b)) - g a_b]`` over observed bins (empty bins: the change in
        ``max(mu_b - censor, 0)``). About chi-square with one degree of freedom for an
        orientation that is not there (Wilks, 1938) [1]_, so ~25 is 5 sigma. Two caveats: an occupancy cannot be
        negative, so the null is half chi-square, half zero (Chernoff, 1954) [2]_; and holding the others fixed
        overstates the ratio where orientations share spots. Read it as a ranking.
    model: np.ndarray
        [n_cells] the fitted histogram, if return_model

    References
    ----------
    .. [1] Wilks, S.S., 1938. The large-sample distribution of the likelihood ratio for testing composite hypotheses.
       Annals of Mathematical Statistics 9, 60–62. https://doi.org/10.1214/aoms/1177732360
    .. [2] Chernoff, H., 1954. On the distribution of the likelihood ratio. Annals of Mathematical Statistics 25,
       573–578. https://doi.org/10.1214/aoms/1177728725
    """
    U = np.asarray(U, np.float32)
    n = len(U)
    pad = -n % qc
    Up = jnp.asarray(np.concatenate([U, np.repeat(U[:1], pad, 0)]).reshape(-1, qc, 3, 3))
    live = jnp.asarray((np.arange(n + pad) < n).reshape(-1, qc), jnp.float32)
    d = jnp.asarray(d, jnp.float32)
    F2 = np.repeat(rings.get("F2", np.ones(len(rings["hkls"]))), 2)
    args = (jnp.asarray(B, jnp.float32), jnp.asarray(rings["hkls"]), jnp.asarray(F2, jnp.float32),
            jnp.asarray(rings["ring_j"]), geom, etacut, _coverage(rings, bins[2]))  # fmt: skip
    norm = jnp.maximum(_backward(jnp.ones_like(d), Up, *args, bins), 1e-30)

    def step(g: jax.Array) -> tuple:  # one MLEM update, and the deviance of the occupancies it started from
        mu = _forward(g, Up, *args, bins, d.shape[0])
        return g * _backward(censored_ratio(d, mu, censor), Up, *args, bins) / norm, float(deviance(d, mu, censor))

    g = squarem(step, live, n_iter, log, label="orientation MLEM")
    mu = _forward(g, Up, *args, bins, d.shape[0])
    lr = _likelihood_ratio(d, mu, g, Up, *args, bins, censor)
    if return_model:
        return np.asarray(g).ravel()[:n], np.asarray(lr).ravel()[:n], np.asarray(mu)
    return np.asarray(g).ravel()[:n], np.asarray(lr).ravel()[:n]
