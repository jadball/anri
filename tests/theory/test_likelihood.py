"""Proofs for docs/source/theory/likelihood.rst: the Poisson deviance, MLEM, SQUAREM's step, censored bins, the
likelihood ratio and the voxel fit's candidates."""

import contextlib

import jax.numpy as jnp
import mpmath as mp
import numpy as np
import sympy as sp

from tests.theory.common import (
    EPS,
    agree,
    evaluate,
    is_zero,
    rng,
    rounds_to,
    tol,
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


def _pmf(k, m):
    return sp.exp(-m) * m**k / sp.factorial(k)


def test_censored_em():
    """lik-censored: for an empty bin (n <= c, c integer), E[n | n <= c] / mu = 1 - P(n = c) / P(n <= c), and that
    ratio is 1 - (1/2) dD/dmu for D = -2 log P(n <= c): the EM step and the deviance are one objective. c = 0 is the
    Poisson zero (ratio 0, D = 2 mu). Saturated bins (n >= d, d integer) likewise: E[n | n >= d] / mu
    = 1 + P(n = d - 1) / P(n >= d) = 1 - (1/2) dD/dmu for D = -2 log P(n >= d)."""
    m = sp.symbols("mu", positive=True)
    for c in range(5):
        F = sum(_pmf(j, m) for j in range(c + 1))
        cond_mean = sum(j * _pmf(j, m) for j in range(c + 1)) / F
        ratio = 1 - _pmf(c, m) / F
        assert is_zero(cond_mean / m - ratio)
        assert is_zero(ratio - (1 - sp.diff(-2 * sp.log(F), m) / 2))
    assert is_zero(-2 * sp.log(_pmf(0, m)) - 2 * m)
    k = sp.symbols("k", integer=True, nonnegative=True)
    for dd in range(1, 5):
        tail = 1 - sum(_pmf(j, m) for j in range(dd))  # P(n >= d)
        cond = (m - sum(j * _pmf(j, m) for j in range(dd))) / tail  # E[n 1(n >= d)] = mu - sum_{j < d} j P(j)
        ratio = 1 + _pmf(dd - 1, m) / tail
        assert is_zero(cond / m - ratio)
        assert is_zero(ratio - (1 - sp.diff(-2 * sp.log(tail), m) / 2))
    assert is_zero(sp.summation(k * _pmf(k, m), (k, 0, sp.oo)) - m)  # E[n] = mu, used above


def test_censored_gamma():
    """lik-censored-gamma: for integer c, P(n <= c) = Q(c + 1, mu) and P(n >= d) = P(d, mu) (regularised incomplete
    gammas), and dQ(a, x)/dx = -x^(a-1) e^-x / Gamma(a): so for a real c the hazard P(n = c) / P(n <= c) keeps its
    form, with P(n = c) = mu^c e^-mu / Gamma(c + 1)."""
    m = sp.symbols("mu", positive=True)
    a = sp.symbols("a", positive=True)
    for c in range(4):
        Q = sp.uppergamma(c + 1, m) / sp.gamma(c + 1)
        assert is_zero(sp.expand_func(Q) - sum(_pmf(j, m) for j in range(c + 1)))
        P = sp.lowergamma(c + 1, m) / sp.gamma(c + 1)
        assert is_zero(sp.expand_func(P) - (1 - sum(_pmf(j, m) for j in range(c + 1))))
    assert is_zero(sp.diff(sp.uppergamma(a, m) / sp.gamma(a), m) + m ** (a - 1) * sp.exp(-m) / sp.gamma(a))


def test_censored_tail_series():
    """lik-censored-tail: Gamma(a, x) = x^(a-1) e^-x int_0^oo e^-t (1 + t/x)^(a-1) dt (substituting u = x + t), and
    int_0^oo e^-t t^k dt = k!, so expanding (1 + t/x)^(a-1) by Taylor's theorem gives
    S = sum_k (a-1)(a-2)...(a-k) / x^k, whose remainder after n >= a - 1 terms is the next term times a factor in
    (0, 1]. The finite sum for integer c equals P(n <= c) / P(n = c) exactly."""
    x, t, a = sp.symbols("x t a", positive=True)
    lhs = (x + t) ** (a - 1) * sp.exp(-(x + t))
    rhs = x ** (a - 1) * sp.exp(-x) * (1 + t / x) ** (a - 1) * sp.exp(-t)
    assert is_zero(sp.powsimp(sp.expand_power_base(lhs / rhs, force=True), force=True) - 1)
    for k in range(6):
        assert sp.integrate(sp.exp(-t) * t**k, (t, 0, sp.oo)) == sp.factorial(k)
    m = sp.symbols("mu", positive=True)
    for c in range(5):
        S = sum(sp.ff(c, k) / m**k for k in range(c + 1))
        assert is_zero(sum(_pmf(j, m) for j in range(c + 1)) - _pmf(c, m) * S)


def test_saturated_series():
    """lik-saturated-series: P(a, x) = x^a e^-x / Gamma(a+1) sum_k x^k / ((a+1)...(a+k)): the partial sums F_N
    satisfy F_N' = x^(a-1) e^-x / Gamma(a) - (last term's) x^(a+N) e^-x / Gamma(a+N+1), telescoping, and F_N(0) = 0."""
    x, a = sp.symbols("x a", positive=True)
    for N in range(4):
        F = sum(x ** (a + k) * sp.exp(-x) / sp.gamma(a + k + 1) for k in range(N + 1))
        want = x ** (a - 1) * sp.exp(-x) / sp.gamma(a) - x ** (a + N) * sp.exp(-x) / sp.gamma(a + N + 1)
        assert is_zero(sp.gammasimp(sp.diff(F, x) - want))


def test_log_pmf_pieces():
    """lik-loader: log P(n = k) = -bd0 - log(2 pi k)/2 - stirlerr(k) with bd0 = k log(k/mu) + mu - k and stirlerr
    Stirling's series sum B_2m / (2m (2m-1) k^(2m-1)) [DLMF2026] (5.11.1); bd0 = (k - mu) v + 2k sum_j v^(2j+1)/(2j+1),
    v = (k - mu)/(k + mu), as log(k/mu) = log((1+v)/(1-v)) = 2 atanh(v); the code's coefficients are the Bernoulli
    ones, and the first omitted term, |B_12| / (132 k^11), is below 2.3e-16 at k = 15 [DLMF2026] (5.11(ii))."""
    m, v = sp.symbols("mu v", positive=True)
    vv = sp.symbols("vv", positive=True)  # log((1 + v) / (1 - v)) = 2 atanh(v), so with v = (k - mu)/(k + mu)
    assert is_zero(sp.diff(sp.log((1 + vv) / (1 - vv)) - 2 * sp.atanh(vv), vv))
    series = sp.series(2 * sp.atanh(v), v, 0, 19).removeO()
    assert is_zero(series - 2 * sum(v ** (2 * j + 1) / (2 * j + 1) for j in range(9)))
    km = m * (1 + v) / (1 - v)  # k in terms of v
    assert is_zero(sp.simplify(2 * km * v - (km - m) - (km - m) * v))  # 2 k v - (k - mu) = (k - mu) v
    coef = [sp.bernoulli(2 * j) / (2 * j * (2 * j - 1)) for j in range(1, 7)]
    assert coef[:5] == [sp.Rational(1, 12), -sp.Rational(1, 360), sp.Rational(1, 1260), -sp.Rational(1, 1680),
                        sp.Rational(1, 1188)]  # fmt: skip
    assert abs(coef[5]) / 15**11 < sp.Rational(23, 10**17)


def test_geometric_remainder():
    """lik-saturated-series: positive terms whose ratios r_j <= r < 1 keep falling leave, after t_k, at most
    sum_{j >= 1} t_k r^j = t_k r / (1 - r)."""
    t, r = sp.symbols("t r", positive=True)
    j = sp.symbols("j", integer=True, positive=True)
    total = sp.summation(t * r**j, (j, 1, sp.oo))
    assert is_zero(sp.piecewise_fold(total).args[0][0] - t * r / (1 - r))


def _exact_below(mu_, c_):
    """log P(n <= c) + mu and the EM ratio at 40 digits (sympy's incomplete gamma)."""
    mu_, c_ = sp.nsimplify(mu_), sp.nsimplify(c_)
    if mu_ == 0:
        return 0.0, (1.0 if c_ > 0 else 0.0)
    Q = sp.uppergamma(c_ + 1, mu_) / sp.gamma(c_ + 1)
    pmf = sp.exp(-mu_) * mu_**c_ / sp.gamma(c_ + 1)
    return float(sp.N(sp.log(Q) + mu_, 40)), float(sp.N(1 - pmf / Q, 40))


def _exact_above(mu_, d_):
    """log P(n >= d) and the EM ratio at 40 digits, by Kummer's form P(a, x) = x^a e^-x / Gamma(a + 1) 1F1(1; a + 1; x)
    (mpmath, which sums it to any precision; sympy's lowergamma does not converge at d = 1e5)."""
    mp.mp.dps = 40
    m, d = mp.mpf(mu_), mp.mpf(d_)
    log_p = d * mp.log(m) - m - mp.loggamma(d + 1) + mp.log(mp.hyp1f1(1, d + 1, m, maxterms=10**7))
    log_g = (d - 1) * mp.log(m) - m - mp.loggamma(d)
    return float(log_p), float(1 + mp.e ** (log_g - log_p))


def test_censored_code():
    """anri's censored_below and saturated_above against lik-censored at 40 digits, in float64, for integer and real
    c, across the regimes (mu = 0, below c, near, the deep tail where Q(c + 1, mu) < 1e-30)."""
    from anri.index.occupancy import censored_below, saturated_above
    from anri.maths.utils import float64

    mus = [0.0, 1e-6, 0.3, 1.0, 2.5, 6.0, 30.0, 80.0, 150.0, 1e3]
    for c in (0.0, 3.0, 2.7, 6.3):
        want = [_exact_below(x, c) for x in mus]
        with float64():
            got = [np.asarray(v) for v in censored_below(np.array(mus), c)]
        # the logarithm's terms are at most c |log mu| + mu + log Gamma(c + 1): 64 ulps of that bound each result
        scale = np.array([c * abs(np.log(max(x, 1e-300))) + x + float(sp.loggamma(c + 1)) + 1 for x in mus])
        assert np.all(np.abs(got[0] - [w[0] for w in want]) <= 64 * EPS * scale)
        assert np.all(np.abs(got[1] - [w[1] for w in want]) <= 64 * EPS * scale)
    for dd in (5.0, 30.0, 300.0, 1e5):
        ms = [1e-3, 1.0, 4.0, 0.8 * dd, dd, 1.03 * dd, 1.3 * dd, 3 * dd]
        want = [_exact_above(x, dd) for x in ms]
        for dtype, ctx in ((np.float64, float64), (np.float32, contextlib.nullcontext)):
            with ctx():
                got = [
                    np.asarray(v, np.float64) for v in saturated_above(np.array(ms, dtype), np.full(len(ms), dd, dtype))
                ]
            eps = float(np.finfo(dtype).eps)
            # the series' n terms each round once in the product recurrence and once in the sum: n eps relative in T,
            # with n <= (mu - d)+ + sqrt(2 d log(1/eps)) + 40 (its terms fall as exp(-k^2 / 2d) past mu); log P(n = d)
            # rounds in terms of size |log P|; the ratio, 1 + d / (mu T), in a few operations
            n_terms = np.array([max(x - dd, 0) + np.sqrt(2 * dd * np.log(1 / eps)) + 40 for x in ms])
            bound = (n_terms + 16) * eps * np.maximum(1, np.abs([w[0] for w in want]))
            assert np.all(np.abs(got[0] - [w[0] for w in want]) <= bound), (dd, dtype)
            assert np.all(np.abs(got[1] / np.array([w[1] for w in want]) - 1) <= 16 * eps), (dd, dtype)


def test_likelihood_ratio_floor():
    """lik-lr's floor: a bin only q explains, its model floored at 1e-6 mu, adds 2 (d log(mu / (1e-6 mu)) - (1 -
    1e-6) mu); one count (mu = d = 1) gives 2 (log 1e6 - 1) + 2e-6 = 25.6 > 25 = min_lr, and the term grows with d at
    a fixed d / mu."""
    d, mu = sp.symbols("d mu", positive=True)
    rest = mu / 10**6
    term = 2 * (d * sp.log(mu / rest) - (mu - rest))
    assert is_zero(term - 2 * (d * sp.log(10**6) - mu + mu / 10**6))
    one = term.subs({d: 1, mu: 1})
    assert rounds_to(one, "25.6") and one > 25
    assert sp.diff(term.subs(mu, d), d) > 0  # d log 1e6 - d (1 - 1e-6) grows with d


def test_censored_ratio_code():
    """censored_ratio, bin by bin: d / mu where recorded, lik-censored's empty ratio where d = 0 (n <= c), its
    saturated ratio where flagged (n >= d), 0 where mu = 0; at test_censored_code's bounds."""
    from anri.index.occupancy import censored_ratio
    from anri.maths.utils import float64

    mu = np.array([0.0, 0.5, 2.0, 7.5, 30.0, 4.0, 12.0])
    d = np.array([3.0, 0.0, 0.0, 6.0, 30.0, 0.0, 15.0])
    c = 2.7
    sat = np.array([6])  # the last bin holds a saturated pixel
    want = [0.0, _exact_below(0.5, c)[1], _exact_below(2.0, c)[1], 6.0 / 7.5, 1.0, _exact_below(4.0, c)[1],
            _exact_above(12.0, 15.0)[1]]  # fmt: skip
    with float64():
        got = np.asarray(censored_ratio(d, mu, c, sat))
    # recorded bins: one quotient; the censored ones within test_censored_code's 64 ulps of their scale, the
    # saturated one within 16 ulps relative
    scale = np.array([1.0, *(c * abs(np.log(x)) + x + float(sp.loggamma(c + 1)) + 1 for x in mu[1:])])
    np.testing.assert_array_less(np.abs(got - np.array(want, float)), 64 * EPS * scale * np.maximum(1, np.abs(want)))


def test_coarsen_rows_code():
    """coarsen_rows: row k of a histogram goes to k // g, the last group holding what is left; exactly."""
    from anri.index.data import coarsen_rows

    n_rows, g = 7, 3
    H = rng(32).integers(0, 9, (5, n_rows)).astype(float)
    want = np.zeros((5, 3))
    for k in range(n_rows):
        want[:, k // g] += H[:, k]
    Hc, n = coarsen_rows(H.ravel(), n_rows, g)
    assert n == 3
    np.testing.assert_array_equal(np.asarray(Hc).reshape(5, 3), want)


def test_coarsen_scan_code():
    """coarsen_scan: a coarse row is the sum of g rows, so its dty is their mean (dty0 + (g - 1) / 2 steps, step g
    steps), its voxels g times wider and the beam's flat top (g - 1) steps wider; its exposure the group's mean (a
    partial last group: its share of g rows), and its offset the exposure-weighted mean real dty less its own. Exactly,
    in units of a step (powers of two)."""
    from anri.index.data import coarsen_scan

    n, g, ystep, n_o = 7, 3, 0.5, 4
    r = rng(33)
    ex = r.integers(0, 3, (n, n_o)).astype(np.float32)
    dd = (r.integers(-4, 4, (n, n_o)) / 8).astype(np.float32)
    scan = {"dty0": -1.5, "ystep": ystep, "n_rows": n, "voxel": 0.5, "width_beam": 0.0}
    out = coarsen_scan({**scan, "exposure": jnp.asarray(ex), "ddty": jnp.asarray(dd)}, g)
    assert (out["dty0"], out["ystep"], out["n_rows"], out["voxel"], out["width_beam"]) == (-1.0, 1.5, 3, 1.5, 1.0)
    want_ex, want_dd = np.zeros((3, n_o)), np.zeros((3, n_o))
    for c in range(3):
        rows = range(c * g, min(c * g + g, n))
        want_ex[c] = sum(ex[k] for k in rows) / g
        w = sum(ex[k] for k in rows)
        real = sum(ex[k] * (dd[k] + k * ystep) for k in rows)
        want_dd[c] = np.where(w > 0, real / np.where(w > 0, w, 1) - ((g - 1) / 2 * ystep + c * g * ystep), 0.0)
    # float32 sums of a few multiples of 1/8 and their quotient by the weight: one rounding of the quotient
    e32 = float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(np.asarray(out["exposure"]), want_ex, rtol=e32, atol=0)
    np.testing.assert_allclose(np.asarray(out["ddty"]), want_dd, rtol=0, atol=4 * e32 * n * ystep)
    part = coarsen_scan(scan, g)  # no exposure: the partial last group's share
    np.testing.assert_allclose(np.asarray(part["exposure"])[:, 0], [1.0, 1.0, 1 / 3], rtol=e32, atol=0)  # float32


def _lr(d_: sp.Expr, m: sp.Expr, ga: sp.Expr) -> sp.Expr:
    return 2 * (d_ * sp.log(m / (m - ga)) - ga)


def test_likelihood_ratio_is_deviance_change():
    """lik-lr is D(mu - g a) - D(mu) in an observed bin, and log(mu / (mu - g a)) = log1p(g a / (mu - g a))."""
    db, m, ga = sp.symbols("d mu ga", positive=True)
    assert is_zero(sp.expand_log(_dev(db, m - ga) - _dev(db, m) - _lr(db, m, ga), force=True))
    assert is_zero(m / (m - ga) - (1 + ga / (m - ga)))
    # an empty bin: D = -2 log P(n <= c) = 2 mu - 2 log(P(n <= c) e^mu), so D(mu - g a) - D(mu) = -2 g a
    # - 2 [log(P e^mu)(mu - g a) - log(P e^mu)(mu)]: the code's two terms (its exact -g a, and censored_below's)
    for c in range(3):
        P = [sum(_pmf(j, x) for j in range(c + 1)) for x in (m - ga, m)]  # P(n <= c) at mu - g a and mu
        L = [sp.log(Pi) + x for Pi, x in zip(P, (m - ga, m))]
        assert is_zero(sp.expand_log(-2 * sp.log(P[0]) + 2 * sp.log(P[1]) - (-2 * ga - 2 * (L[0] - L[1])), force=True))


def test_likelihood_ratio_code():
    """anri's likelihood ratio is lik-lr, with a_qb summed over an orientation's entries in each bin (bins coarse enough
    that entries share them)."""
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
    rings = ring_table(lp, 229, wl, 6)
    g = rng(3)
    q = g.normal(size=(2, 4))
    U = anri.maths.rotation.quat_to_mat(q / np.linalg.norm(q, axis=1, keepdims=True))
    bins, n_cells = (30.0, 20.0, 12, 9, 0.0), 6 * 12 * 9  # coarse: an orientation's entries share bins
    occ = np.array([3.0, 0.5])
    with float64():
        geom = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in geom.items()}
        args = (jnp.asarray(B), jnp.asarray(rings["hkls"]), jnp.ones(2 * len(rings["hkls"])),
                jnp.asarray(rings["ring_j"]), geom, 0.2, None)  # fmt: skip
        idx, wt = (np.asarray(a) for a in _system(jnp.asarray(U), *args, bins))
    a = np.zeros((2, n_cells))  # a_qb: each orientation's entries summed per bin
    for k in range(2):
        np.add.at(a[k], idx[k][idx[k] >= 0], wt[k][idx[k] >= 0])
        kept = idx[k][idx[k] >= 0]
        assert len(np.unique(kept)) < len(kept)  # some bins hold several entries: the case the sum per bin is for
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


def test_mlem_is_metric_gradient():
    """lik-converged: for any bin terms D_b(mu_b) whose ratio is rho_b = 1 - D_b'/2 (lik-censored), the MLEM update
    f * A^T rho / A^T 1 - f is M^-1 grad(l), l = -D/2, M = diag(A^T 1 / f); and -grad(D) . df =
    2 sum (A^T 1) df^2 / f. Symbolic, 3 bins x 2 unknowns, bin terms with free coefficients."""
    A = sp.Matrix(3, 2, sp.symbols("a0:6", positive=True))
    f = sp.Matrix(sp.symbols("f0:2", positive=True))
    mu = A * f
    x = sp.Symbol("x", positive=True)
    c = sp.symbols("c0:9", real=True)  # generic bin terms (any coefficients): a recorded, an empty, a saturated bin
    Db = [c[3 * b] * x + c[3 * b + 1] * x**2 + c[3 * b + 2] * sp.log(x) for b in range(3)]
    D = sum(Db[b].subs(x, mu[b]) for b in range(3))
    rho = sp.Matrix([1 - sp.diff(Db[b], x).subs(x, mu[b]) / 2 for b in range(3)])  # each bin's EM ratio
    norm = A.T * sp.ones(3, 1)
    f_new = sp.Matrix([f[j] * (A.T * rho)[j] / norm[j] for j in range(2)])
    df = f_new - f
    grad_l = sp.Matrix([-sp.diff(D, f[j]) / 2 for j in range(2)])
    M = sp.diag(*[norm[j] / f[j] for j in range(2)])
    assert is_zero(sp.simplify(df - M.inv() * grad_l))
    grad_D = sp.Matrix([sp.diff(D, f[j]) for j in range(2)])
    delta = 2 * sum(norm[j] * df[j] ** 2 / f[j] for j in range(2))
    assert is_zero(sp.simplify(-(grad_D.T * df)[0] - delta))


def test_remaining_deviance():
    """lik-converged: near the optimum D - D* = e^T H e and MLEM is e -> (I - M^-1 H) e. On a mode with
    M^-1 H v = (1 - rho) v the step is -(1 - rho) e, delta = 2 (1 - rho)^2 e^T M e, D - D* = (1 - rho) e^T M e =
    delta / (2 (1 - rho)), and the next delta is rho^2 times this one. M-orthogonal modes add, and each
    delta_m / (2 (1 - rho_m)) <= delta_m / (2 (1 - rho_max))."""
    rho, m, e = sp.symbols("rho m e", positive=True)
    H = (1 - rho) * m  # one mode: M^-1 H = 1 - rho
    step = -(H / m) * e
    delta = 2 * m * step**2
    gap = e * H * e
    assert is_zero(sp.simplify(gap - delta / (2 * (1 - rho))))
    e2 = e + step
    assert is_zero(sp.simplify(2 * m * ((H / m) * e2) ** 2 - rho**2 * delta))
    # two M-orthogonal modes: the gaps add, and the slowest mode bounds them
    r1, r2, d1, d2 = sp.symbols("r1 r2 d1 d2", positive=True)
    gap2 = d1 / (2 * (1 - r1)) + d2 / (2 * (1 - r2))
    bound = (d1 + d2) / (2 * (1 - r2))  # r2 the slowest: r1 <= r2 < 1
    t = sp.Symbol("t", nonnegative=True)  # r1 = r2 - t
    # bound - gaps = d1 t / (2 (1 - r2) (1 - r2 + t)) >= 0 for t >= 0, r2 < 1
    assert is_zero(sp.simplify((bound - gap2).subs(r1, r2 - t) - d1 * t / (2 * (1 - r2) * (1 - r2 + t))))


def test_remaining_deviance_code():
    """remaining_deviance and step_decrease against lik-converged."""
    import jax.numpy as jnp

    from anri.index.occupancy import remaining_deviance, step_decrease

    r = rng(11)
    for rho in (0.1, 0.5, 0.9, 0.99):
        d0 = float(r.uniform(1, 100))
        d1 = rho**2 * d0
        want = d1 / (2 * (1 - rho))
        # rho = sqrt(d1 / d0) rounds by an ulp or two, amplified by 1 / (1 - rho) in d1 / (2 (1 - rho))
        np.testing.assert_allclose(remaining_deviance(d0, d1), want, rtol=16 * EPS * (1 + rho / (1 - rho)))
    assert remaining_deviance(1.0, 1.0) == float("inf") and remaining_deviance(1.0, 0.0) == 0.0
    f = r.uniform(0.1, 2, 20)
    f[3] = 0.0  # an occupancy at 0 stays there and adds nothing
    fn = f * r.uniform(0.5, 1.5, 20)
    norm = r.uniform(1, 5, 20)
    want = 2 * np.sum(np.where(f > 0, norm * (fn - f) ** 2 / np.where(f > 0, f, 1.0), 0.0))
    agree(np.array(want), lambda *a: step_decrease(*(jnp.asarray(x) for x in a)), f, fn, norm, **tol(want, 64))
