"""Proofs for docs/source/theory/likelihood.rst: the Poisson deviance, MLEM, SQUAREM's step, censored bins, the
likelihood ratio and the voxel fit's candidates."""

import numpy as np
import sympy as sp

from tests.theory.common import (
    EPS,
    agree,
    evaluate,
    is_zero,
    rng,
)


def _sum_rtol(parts: list, total: float) -> float:
    """Relative error bound of a float64 sum of n terms, each a few roundings of its parts (|parts| summed): about
    (n + 4) ulps of the parts' sum over |total| (the sum's condition number)."""
    return (len(parts) + 4) * EPS * float(np.sum(np.abs(parts))) / abs(total)


# a small symbolic system: 3 bins, 2 occupancies
A = sp.Matrix(3, 2, sp.symbols("a0:6", positive=True))
f = sp.Matrix(sp.symbols("f0:2", positive=True))
d = sp.Matrix(sp.symbols("d0:3", positive=True))
mu = A * f


def _loglik(f_: sp.Matrix) -> sp.Expr:
    m = A * f_
    return sum(d[b] * sp.log(m[b]) - m[b] for b in range(3))


def _mlem(f_: sp.Matrix) -> sp.Matrix:
    m = A * f_
    back = A.T * sp.Matrix([d[b] / m[b] for b in range(3)])
    norm = A.T * sp.ones(3, 1)
    return sp.Matrix([f_[j] * back[j] / norm[j] for j in range(2)])


def _dev(d_: sp.Expr, m: sp.Expr) -> sp.Expr:
    return 2 * (d_ * sp.log(d_ / m) - d_ + m)


def test_deviance_gradient():
    """D's gradient in a bin is 2 (1 - d / mu): zero at mu = d, curvature 2 d / mu^2 > 0; the MLEM ratio is 1 - dD/2."""
    db, mb = sp.symbols("d mu", positive=True)
    D = _dev(db, mb)
    assert is_zero(sp.diff(D, mb) - 2 * (1 - db / mb))
    assert is_zero(D.subs(mb, db))
    assert is_zero(sp.diff(D, mb, 2) - 2 * db / mb**2)
    assert is_zero(db / mb - (1 - sp.diff(D, mb) / 2))


def test_deviance_code():
    """anri's deviance (censor 0) is lik-deviance."""
    from anri.index.occupancy import deviance

    g = rng(1)
    dv, mv = g.uniform(0.5, 9, 6), g.uniform(0.5, 9, 6)
    want = evaluate(sum(_dev(sp.Float(x, 30), sp.Float(y, 30)) for x, y in zip(dv, mv)), {})
    parts = [p for x, y in zip(dv, mv) for p in (x * np.log(x / y), x, y)]
    agree(want, deviance, dv, mv, 0.0, rtol=_sum_rtol(parts, float(want)), atol=0)


def test_mlem_fixed_point_is_kkt():
    """The MLEM update moves f by f * grad(loglik) / A^T 1: its fixed points are the KKT points f_j dl/df_j = 0."""
    grad = sp.Matrix([sp.diff(_loglik(f), f[j]) for j in range(2)])
    norm = A.T * sp.ones(3, 1)
    step = _mlem(f) - f
    assert is_zero(step - sp.Matrix([f[j] * grad[j] / norm[j] for j in range(2)]))


def test_mlem_keeps_total():
    """After one MLEM update the model's total equals the data's: sum_b (A f')_b = sum_b d_b."""
    assert is_zero(sum(A * _mlem(f)) - sum(d))


def test_candidates_are_first_update():
    """lik-candidates is the MLEM update from unit occupancy."""
    one = sp.ones(2, 1)
    back = A.T * sp.Matrix([d[b] / (A * one)[b] for b in range(3)])
    norm = A.T * sp.ones(3, 1)
    assert is_zero(_mlem(one) - sp.Matrix([back[j] / norm[j] for j in range(2)]))


def test_squarem_unit_step_is_two_updates():
    """SQUAREM's point at a = -1 is f2: the safeguard a <= -1 never steps short of two plain MLEM updates."""
    x, x1, x2, a = sp.symbols("f f1 f2 a")
    r, v = x1 - x, x2 - 2 * x1 + x
    assert is_zero((x - 2 * a * r + a**2 * v).subs(a, -1) - x2)


def test_squarem_code():
    """anri's SQUAREM point is lik-squarem with a clipped to [-step_max, -1] and the point kept above 1e-3 f2."""
    from anri.index.occupancy import _squarem_point

    g = rng(2)
    F, F1, F2 = (sp.Matrix(sp.symbols(f"{n}0:4")) for n in ("f", "g", "h"))
    r, v = F1 - F, F2 - 2 * F1 + F
    a_raw = -sp.sqrt((r.T * r)[0]) / sp.sqrt((v.T * v)[0])
    for step_max in (np.inf, 2.0):
        x, x1, x2 = g.uniform(0.1, 2, (3, 4))
        vals = {**dict(zip(F, x)), **dict(zip(F1, x1)), **dict(zip(F2, x2))}
        a = float(np.clip(evaluate(a_raw, vals), -step_max, -1.0))
        point = F - 2 * a * r + a**2 * v
        want = np.maximum(evaluate(point, vals).ravel(), 1e-3 * x2)
        # each entry: 4 roundings of terms of size |f| + 2 |a r| + a^2 |v|, and a (two norms, a sqrt, a division:
        # ~8 ulps) times its weight 2 |r| + 2 |a v| in the entry; both over the entry's size
        rv, vv = x1 - x, x2 - 2 * x1 + x
        terms = np.abs(x) + 2 * abs(a) * np.abs(rv) + a * a * np.abs(vv)
        from_a = abs(a) * (2 * np.abs(rv) + 2 * abs(a) * np.abs(vv))
        rtol = float(np.max(8 * EPS * (terms + from_a) / np.abs(want)))
        agree(want, lambda *z: _squarem_point(*z)[0], x, x1, x2, step_max, rtol=rtol, atol=0)


def test_censored_terms():
    """Empty bins: below c the deviance term and its gradient are 0 (ratio 1, consistent); at mu = c the term is
    continuous; above c its gradient asks for ratio 0, while the update uses c / (2 mu), the Poisson ratio of a datum
    c / 2. The update and the deviance are not one objective's there."""
    m, c = sp.symbols("mu c", positive=True)
    above = 2 * (m - c)
    assert is_zero(above.subs(m, c))
    assert is_zero((1 - sp.diff(above, m) / 2) - 0)
    assert is_zero(c / (2 * m) - (1 - sp.diff(_dev(c / 2, m), m) / 2))


def test_censored_code():
    """anri's censored_ratio and deviance are lik-censored and its deviance terms, saturated bins included."""
    from anri.index.occupancy import censored_ratio, deviance

    db, mb, c = sp.symbols("d mu c", positive=True)
    cv = 2.0
    dv = np.array([3.0, 0.0, 0.0, 5.0, 1.0, 4.0])
    mv = np.array([2.0, 1.5, 3.0, 6.0, 0.7, 2.5])
    sat = np.array([3, 5])  # bin 3: mu >= d (agrees), bin 5: mu < d (as usual)
    ratio, dev = [], 0.0
    for i, (x, y) in enumerate(zip(dv, mv)):
        if i in sat and y >= x:
            ratio.append(1.0)
        elif x > 0:
            ratio.append(float(evaluate(db / mb, {db: x, mb: y})))
            dev += float(evaluate(_dev(db, mb), {db: x, mb: y}))
        elif y < cv:
            ratio.append(1.0)
        else:
            ratio.append(float(evaluate(c / (2 * mb), {c: cv, mb: y})))
            dev += float(evaluate(2 * (mb - c), {c: cv, mb: y}))
    agree(np.array(ratio), censored_ratio, dv, mv, cv, sat, rtol=2 * EPS, atol=0)  # one division (and c / 2 exact)
    parts = [x * np.log(x / y) for x, y in zip(dv, mv) if x > 0] + list(dv) + list(mv) + [cv] * len(dv)
    agree(np.array(dev), deviance, dv, mv, cv, sat, rtol=_sum_rtol(parts, dev), atol=0)


def _lr(d_: sp.Expr, m: sp.Expr, ga: sp.Expr) -> sp.Expr:
    return 2 * (d_ * sp.log(m / (m - ga)) - ga)


def test_likelihood_ratio_is_deviance_change():
    """lik-lr is D(mu - g a) - D(mu) in an observed bin, and log(mu / (mu - g a)) = log1p(g a / (mu - g a))."""
    db, m, ga = sp.symbols("d mu ga", positive=True)
    assert is_zero(sp.expand_log(_dev(db, m - ga) - _dev(db, m) - _lr(db, m, ga), force=True))
    assert is_zero(m / (m - ga) - (1 + ga / (m - ga)))
    c = sp.symbols("c", positive=True)  # an empty bin above c on both sides: the censored deviance changes by -2 g a
    assert is_zero(2 * ((m - ga) - c) - 2 * (m - c) + 2 * ga)


def test_likelihood_ratio_code():
    """anri's likelihood ratio is lik-lr, with a_qb summed over an orientation's entries in each bin."""
    import jax.numpy as jnp

    import anri
    from anri.index.data import ring_table
    from anri.index.orientations import _likelihood_ratio, _system
    from anri.maths.utils import float64

    wl = 12.398419843320026 / 43.0
    pars = {"y_center": 1049.9, "y_size": 75.0, "tilt_y": 0.0, "z_center": 1116.5, "z_size": 75.0, "tilt_z": 0.0,
            "tilt_x": 0.0, "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1, "wavelength": wl,
            "wedge": 0.0, "chi": 0.0}  # fmt: skip
    geom = anri.io.imaged11.geom_from_pars(pars, 0.0, wl * 1e-4, 5e-5, 5e-5, sig_beam=0.5, voxel_size=1.0)
    lp = np.array([2.8665] * 3 + [90.0] * 3)
    B = anri.crystal.structure.B_matrix(lp)
    rings = ring_table(lp, 229, wl, 2)
    g = rng(3)
    q = g.normal(size=(2, 4))
    U = anri.maths.rotation.quat_to_mat(q / np.linalg.norm(q, axis=1, keepdims=True))
    bins, n_cells = (1.0, 1.0, 360, 180, 0.0), 2 * 360 * 180
    occ = np.array([3.0, 0.5])
    with float64():
        geom = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in geom.items()}
        args = (jnp.asarray(B), jnp.asarray(rings["hkls"]), jnp.ones(2 * len(rings["hkls"])),
                jnp.asarray(rings["ring_j"]), geom, 0.2, None)  # fmt: skip
        idx, wt = (np.asarray(a) for a in _system(jnp.asarray(U), *args, bins))
    a = np.zeros((2, n_cells))  # a_qb: each orientation's entries summed per bin
    for k in range(2):
        np.add.at(a[k], idx[k][idx[k] >= 0], wt[k][idx[k] >= 0])
    mu_v = occ @ a + g.uniform(1.0, 2.0, n_cells)
    d_v = np.where(g.uniform(size=n_cells) < 0.2, 0.0, g.uniform(0.5, 5.0, n_cells))
    want, rtol = [], 0.0
    for k in range(2):
        terms, parts = [], []
        for b in np.flatnonzero(a[k] > 0):
            gab = occ[k] * a[k, b]
            x, y, z = (sp.Float(v, 30) for v in (d_v[b], mu_v[b], gab))
            terms.append(-2 * z if d_v[b] == 0 else _lr(x, y, z))
            parts += [2 * d_v[b] * np.log1p(gab / (mu_v[b] - gab)), 2 * gab]
        want.append(float(evaluate(sum(terms), {})))
        rtol = max(rtol, _sum_rtol(parts, want[-1]))

    def code(*z: object) -> object:
        return np.asarray(_likelihood_ratio(*z, bins, 0.0)).ravel()

    with float64():
        agree(np.array(want), code, jnp.asarray(d_v), jnp.asarray(mu_v), jnp.asarray(occ[:, None]),
              jnp.asarray(U[:, None]), *args, rtol=rtol, atol=0)  # fmt: skip
