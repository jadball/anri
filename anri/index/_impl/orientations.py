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

from .predict import lorentz_polarisation, predict


def _system(
    U: jax.Array, B: jax.Array, hkls: jax.Array, F2: jax.Array, ring_j: jax.Array, geom: dict, etacut: float,
    bins: tuple,
) -> tuple[jax.Array, jax.Array]:  # fmt: skip
    """(cell index, weight) [Q, Nj, 4] of orientations U [Q, 3, 3] in a row-summed (ring, eta, omega) histogram."""
    b_e, b_o, n_e, n_o, om0 = bins
    eta, om, ok = predict(U, B, hkls, geom)
    w = lorentz_polarisation(U, B, hkls, geom) * F2[None]
    use = ok & (jnp.abs(jnp.sin(jnp.radians(eta))) > etacut)
    om_w = jnp.mod(om - om0, 360.0) + om0
    fe, fo = (eta + 180.0) / b_e - 0.5, (om_w - om0) / b_o - 0.5
    e0, o0 = jnp.floor(fe).astype(jnp.int32), jnp.floor(fo).astype(jnp.int32)
    te, to = fe - e0, fo - o0
    idx, wt = [], []
    for de, we in ((0, 1.0 - te), (1, te)):
        for do, wo in ((0, 1.0 - to), (1, to)):
            ie, io = (e0 + de) % n_e, o0 + do
            good = use & (io >= 0) & (io < n_o)
            idx.append(jnp.where(good, (ring_j[None] * n_e + ie) * n_o + io, -1))
            wt.append(jnp.where(good, w * we * wo, 0.0))
    return jnp.stack(idx, -1), jnp.stack(wt, -1)


@partial(jax.jit, static_argnames=("bins", "n_cells"))
def _forward(
    g: jax.Array, U: jax.Array, B: jax.Array, hkls: jax.Array, F2: jax.Array, ring_j: jax.Array, geom: dict,
    etacut: float, bins: tuple, n_cells: int,
) -> jax.Array:  # fmt: skip
    def step(acc: jax.Array, ch: tuple) -> tuple:
        gc, uc = ch
        idx, wt = _system(uc, B, hkls, F2, ring_j, geom, etacut, bins)
        c = (gc[:, None, None] * wt).ravel().astype(acc.dtype)
        return acc + jax.ops.segment_sum(c, jnp.where(idx >= 0, idx, n_cells).ravel(), n_cells + 1)[:-1], None

    acc, _ = jax.lax.scan(step, jnp.zeros(n_cells, jnp.float32), (g, U))
    return acc


@partial(jax.jit, static_argnames=("bins",))
def _backward(
    r: jax.Array, U: jax.Array, B: jax.Array, hkls: jax.Array, F2: jax.Array, ring_j: jax.Array, geom: dict,
    etacut: float, bins: tuple,
) -> jax.Array:  # fmt: skip
    def one(uc: jax.Array) -> jax.Array:
        idx, wt = _system(uc, B, hkls, F2, ring_j, geom, etacut, bins)
        return jnp.sum(wt * jnp.where(idx >= 0, r[jnp.clip(idx, 0)], 0.0), (1, 2))

    return jax.lax.map(one, U)


@partial(jax.jit, static_argnames=("bins",))
def _likelihood_ratio(
    d: jax.Array, mu: jax.Array, g: jax.Array, U: jax.Array, B: jax.Array, hkls: jax.Array, F2: jax.Array,
    ring_j: jax.Array, geom: dict, etacut: float, bins: tuple,
) -> jax.Array:  # fmt: skip
    def one(ch: tuple) -> jax.Array:
        gc, uc = ch
        idx, wt = _system(uc, B, hkls, F2, ring_j, geom, etacut, bins)
        m = jnp.where(idx >= 0, mu[jnp.clip(idx, 0)], 0.0)
        dd = jnp.where(idx >= 0, d[jnp.clip(idx, 0)], 0.0)
        ga = gc[:, None, None] * wt
        rest = jnp.maximum(m - ga, 1e-6 * jnp.maximum(m, 1e-30))
        term = jnp.where((idx >= 0) & (m > 0), jnp.where(dd > 0, dd * jnp.log(m / rest), 0.0) - ga, 0.0)
        return 2 * jnp.sum(term, (1, 2))

    return jax.lax.map(one, (g, U))


def orientation_mlem(
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
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one occupancy per orientation to a row-summed histogram by MLEM, and score each orientation.

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
        From :func:`anri.index.ring_table` ("hkls", "ring_j" and, optionally, "F2")
    geom
        Geometry dict, see :func:`anri.index.predict`
    bins
        (b_e, b_o, n_e, n_o, om0): the histogram's eta and omega bin widths (degrees) and counts, and its first omega
        bin edge
    etacut
        Reflections with ``|sin eta|`` at or below this are not used
    n_iter
        MLEM iterations
    qc
        Orientations per chunk
    log
        Progress messages

    Returns
    -------
    occupancy: np.ndarray
        [Nq] global occupancy of each orientation (in the units of d over Lorentz x polarisation x F^2)
    likelihood_ratio: np.ndarray
        [Nq] increase in the Poisson deviance if that orientation alone were removed, the others fixed:
        ``2 sum_b [d_b log(mu_b / (mu_b - g a_b)) - g a_b]``. About chi-square with one degree of freedom for an
        orientation that is not there, so ~25 is 5 sigma.
    """
    U = np.asarray(U, np.float32)
    n = len(U)
    pad = -n % qc
    Up = jnp.asarray(np.concatenate([U, np.repeat(U[:1], pad, 0)]).reshape(-1, qc, 3, 3))
    live = jnp.asarray((np.arange(n + pad) < n).reshape(-1, qc), jnp.float32)
    d = jnp.asarray(d, jnp.float32)
    F2 = np.repeat(rings.get("F2", np.ones(len(rings["hkls"]))), 2)
    args = (jnp.asarray(B, jnp.float32), jnp.asarray(rings["hkls"]), jnp.asarray(F2, jnp.float32),
            jnp.asarray(rings["ring_j"]), geom, etacut)  # fmt: skip
    norm = jnp.maximum(_backward(jnp.ones_like(d), Up, *args, bins), 1e-30)
    g = live
    for it in range(n_iter):
        mu = _forward(g, Up, *args, bins, d.shape[0])
        g = g * _backward(jnp.where(mu > 0, d / jnp.maximum(mu, 1e-30), 0.0), Up, *args, bins) / norm
        if it % 5 == 0 or it == n_iter - 1:
            dev = jnp.sum(jnp.where(d > 0, d * jnp.log(jnp.maximum(d, 1e-30) / jnp.maximum(mu, 1e-30)), 0.0) - d + mu)
            log(f"  orientation MLEM {it}: deviance {2 * float(dev):.4g}")
    mu = _forward(g, Up, *args, bins, d.shape[0])
    lr = _likelihood_ratio(d, mu, g, Up, *args, bins)
    return np.asarray(g).ravel()[:n], np.asarray(lr).ravel()[:n]
