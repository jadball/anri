"""Proofs for docs/source/theory/refiner.rst: a refiner that needs no spot shape (deformation moments as a linear
tomography, and the bounds on what noise, the cut, the group rule and the pixels do to it)."""

from __future__ import annotations

import itertools

import numpy as np
import sympy as sp

from tests.theory.common import cayley, is_zero, rounds_to, rz, skew


def _mat(name: str) -> sp.Matrix:
    return sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"{name}{i}{j}", real=True))


def test_photon_moments_linear():
    """rf-moments: a population's sub-volumes (any number, any deformations F_i, volumes V_i) send photons with
    g = F_i g0; their volume-weighted moments of g, |g|^2 and g g^T are linear in V, sum V_i F_i, sum V_i F_i^T F_i
    and sum V_i F_i (x) F_i applied to g0, whatever the distribution (three sub-volumes, symbolic)."""
    g0 = sp.Matrix(sp.symbols("g0:3", real=True))
    F = [_mat(f"F{i}_") for i in range(3)]
    V = sp.symbols("V0:3", positive=True)
    Y1 = sum((V[i] * F[i] for i in range(3)), sp.zeros(3))
    C = sum((V[i] * F[i].T * F[i] for i in range(3)), sp.zeros(3))
    assert is_zero(sum((V[i] * F[i] * g0 for i in range(3)), sp.zeros(3, 1)) - Y1 * g0)
    assert is_zero(sum(V[i] * (F[i] * g0).dot(F[i] * g0) for i in range(3)) - (g0.T * C * g0)[0])
    for a, b in itertools.product(range(3), repeat=2):  # g g^T: the 4-index tensor sum V F_ak F_bl g0_k g0_l
        lhs = sum(V[i] * (F[i] * g0)[a] * (F[i] * g0)[b] for i in range(3))
        T = {(k, m): sum(V[i] * F[i][a, k] * F[i][b, m] for i in range(3)) for k in range(3) for m in range(3)}
        assert is_zero(lhs - sum(T[k, m] * g0[k] * g0[m] for k in range(3) for m in range(3)))


def test_rotation_invariant():
    """rf-invariant: |R U g0|^2 = g0^T U^T U g0 for every rotation R (a Cayley rotation, rational in its three
    parameters), so a population's sum of |g|^2 sees only U^T U; and the cell's scattering vector taken at its frame's
    centre is the photon's turned about the rotation axis, R_z(w_j - w) g, of the same length."""
    k = sp.Matrix(sp.symbols("k0:3", real=True))
    R = cayley(k)
    assert is_zero(R.T * R - sp.eye(3))
    U = _mat("U")
    g0 = sp.Matrix(sp.symbols("g0:3", real=True))
    assert is_zero((R * U * g0).dot(R * U * g0) - (g0.T * U.T * U * g0)[0])
    t = sp.Symbol("t", real=True)
    g = sp.Matrix(sp.symbols("h0:3", real=True))
    assert is_zero((rz(t) * g).dot(rz(t) * g) - g.dot(g))


def test_instrument_independent():
    """rf-instrument: a measured g_hat = s (g + D) with the scale s (wavelength) and the shift D (the incoming
    direction) independent of the photon's g, E[D] = 0: E[g_hat] = E[s] E[g] and E|g_hat|^2 = E[s^2] (E|g|^2 + E|D|^2),
    common to every population (discrete distributions, two values each, symbolic)."""
    g = [sp.Matrix(sp.symbols(f"g{i}_0:3", real=True)) for i in range(2)]
    pg = sp.Symbol("pg", positive=True)
    s = sp.symbols("s0:2", positive=True)
    ps = sp.Symbol("ps", positive=True)
    d = sp.Matrix(sp.symbols("d0:3", real=True))
    D = [d, -d]  # mean zero, equal weights
    Eg = pg * g[0] + (1 - pg) * g[1]
    Es, Es2 = ps * s[0] + (1 - ps) * s[1], ps * s[0] ** 2 + (1 - ps) * s[1] ** 2
    Eg2 = pg * g[0].dot(g[0]) + (1 - pg) * g[1].dot(g[1])
    mean = sp.zeros(3, 1)
    sq = 0
    for (wg, gi), (ws, si), Di in itertools.product(
        zip((pg, 1 - pg), g), zip((ps, 1 - ps), s), D
    ):  # every combination, independent
        w = wg * ws / 2
        gh = si * (gi + Di)
        mean += w * gh
        sq += w * gh.dot(gh)
    assert is_zero(mean - Es * Eg)
    assert is_zero(sq - Es2 * (Eg2 + d.dot(d)))


def test_weighted_mean():
    """rf-weight: photons weighted by a factor 1 + l^T d of the sub-volume's small rotation d (a Lorentz factor's
    variation) shift the population's mean F from E[I + [d]x] = I to I + [S l]x, S = Cov(d), to second order. With
    the Lorentz factor 1 / (sin 2 theta |sin eta|), l = -cot(eta) d eta / d d, so |S l| ~ sigma^2 |cot eta|: 1.49e-3
    rad for sigma = 1 deg at |sin eta| = 0.2."""
    d = sp.Matrix(sp.symbols("d0:3", real=True))
    l = sp.Matrix(sp.symbols("l0:3", real=True))
    S = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"S_{min(i, j)}{max(i, j)}"))
    expr = (1 + (l.T * d)[0]) * (sp.eye(3) + skew(d))
    E = expr.applyfunc(lambda e: _expect2(e, d, S))
    assert is_zero(E - (sp.eye(3) + skew(S * l)))
    sig2 = (sp.pi / 180) ** 2
    cot = sp.sqrt(1 - sp.Rational(1, 25)) / sp.Rational(1, 5)
    assert rounds_to(sig2 * cot * 10**3, "1.49")


def _expect2(expr: sp.Expr, d: sp.Matrix, S: sp.Matrix) -> sp.Expr:
    """E over zero-mean d with covariance S, of a polynomial in d of degree <= 2."""
    p = sp.Poly(sp.expand(expr), *d)
    out = 0
    for mono, coef in p.terms():
        idx = [i for i, n in enumerate(mono) for _ in range(n)]
        if not idx:
            out += coef
        elif len(idx) == 2:
            out += coef * S[idx[0], idx[1]]
        elif len(idx) == 1:
            pass  # zero mean
        else:
            raise ValueError("degree > 2")
    return out


def test_linear_estimator():
    """rf-linear: T = G Y + Delta + xi with E[xi] = 0; any left inverse A (A G = I) gives E[A T] - Y = A Delta,
    exactly, whatever the distribution of xi; and |A Delta|_i <= sum_j |A_ij| Delta_max_j. The residual
    e = (I - G A) T has E[e e^T] = (I - H)(Sigma + Delta Delta^T)(I - H)^T, H = G A."""
    G = sp.Matrix(5, 2, lambda i, j: sp.Rational((3 * i + 2 * j) % 7 - 3, (i + j) % 3 + 1))
    W = sp.diag(*[sp.Rational(k + 1, 2) for k in range(5)])
    A = (G.T * W * G).inv() * G.T * W
    assert is_zero(A * G - sp.eye(2))
    Y = sp.Matrix(sp.symbols("y0:2"))
    Delta = sp.Matrix(sp.symbols("D0:5"))
    assert is_zero(A * (G * Y + Delta) - Y - A * Delta)  # E[xi] = 0 drops out by linearity
    H = G * A
    xi = sp.Matrix(sp.symbols("x0:5"))
    e = (sp.eye(5) - H) * (G * Y + Delta + xi)
    assert is_zero(e - (sp.eye(5) - H) * (Delta + xi))  # (I - H) G = 0
    Sig = sp.Matrix(5, 5, lambda i, j: sp.Symbol(f"C_{min(i, j)}{max(i, j)}"))
    outer = (sp.eye(5) - H) * (Delta + xi) * ((Delta + xi).T) * (sp.eye(5) - H).T
    # E[xi xi^T] = Sigma, E[xi] = 0: replace xi_i xi_j by C_ij and drop terms linear in xi
    E = outer.applyfunc(lambda t: _expect_xi(t, xi, Sig))
    assert is_zero(E - (sp.eye(5) - H) * (Sig + Delta * Delta.T) * (sp.eye(5) - H).T)


def _expect_xi(expr: sp.Expr, xi: sp.Matrix, Sig: sp.Matrix) -> sp.Expr:
    p = sp.Poly(sp.expand(expr), *xi)
    out = 0
    for mono, coef in p.terms():
        idx = [i for i, n in enumerate(mono) for _ in range(n)]
        if not idx:
            out += coef
        elif len(idx) == 2:
            out += coef * Sig[idx[0], idx[1]]
    return out


def test_alias_bound_any_spot():
    """rf-alias-bound: a spot that is the point spread convolved with anything, nu (any measure, any asymmetry), has
    |chi_spot| = |chi_psf| |chi_nu| <= |chi_psf|, and |E[sin 2 pi k x]| <= |chi(2 pi k)|, so its binned mean is off its
    mean by at most min(1/2, sum_k |chi_psf(2 pi k)| / (pi k)): 0.0676 px for the measured point spread. Checked on
    asymmetric spots (the point spread convolved with exponentials of either sign and three scales)."""
    import mpmath as mp

    mp.mp.dps = 15
    w = sp.Symbol("w", real=True)
    mu = sp.Symbol("mu", real=True)
    # any characteristic function chi = r e^{i phi} about mu: E[sin(w x)] = Im(chi e^{i w mu})
    r, ph = sp.symbols("r phi", real=True)
    val = sp.im(r * sp.exp(sp.I * (ph + w * mu)))
    assert is_zero(sp.expand_complex(val) - r * sp.sin(ph + w * mu))  # |.| <= r = |chi|
    f, sc, sw = mp.mpf("0.28"), mp.mpf("0.25"), mp.mpf("1.3")
    bound = sum(
        ((1 - f) * mp.exp(-((2 * mp.pi * k * sc) ** 2) / 2) + f * mp.exp(-((2 * mp.pi * k * sw) ** 2) / 2))
        / (mp.pi * k)
        for k in range(1, 30)
    )
    assert rounds_to(sp.Float(str(bound), 20), "0.0676")
    # psf (x) exponential of scale b (sign: direction): cell masses by quadrature of the psf's cdf against it
    for b, m0 in (("0.2", "0.13"), ("-0.5", "0.31"), ("1.5", "0.42")):
        bb = mp.mpf(b)
        if True:
            m = mp.mpf(m0)

            def cdf(t, bb=bb, m=m):  # P(x <= t), x = m + psf + E, E ~ exponential of mean |b| (signed)
                def psf_cdf(u):
                    return (1 - f) * mp.ncdf(u / sc) + f * mp.ncdf(u / sw)

                s = abs(bb)
                return mp.quad(lambda e: psf_cdf(t - m - mp.sign(bb) * e) * mp.exp(-e / s) / s, [0, 3 * s, mp.inf])

            ks = range(-8, 10)
            mass = [cdf(k + mp.mpf(1) / 2) - cdf(k - mp.mpf(1) / 2) for k in ks]
            mean_cell = sum(k * q for k, q in zip(ks, mass)) / sum(mass)
            true_mean = m + bb
            assert abs(mean_cell - true_mean) <= bound + mp.mpf("1e-6")


def test_no_unbiased_estimator():
    """rf-no-unbiased: with a cut c >= 1 no function of the recorded values is an unbiased estimate of
    sum_p f_p lambda_p unless every f_p = 0. A cell's count below the cut is recorded as 0, so the coefficient of
    lambda_p alone in e^{sum lambda} E[h] is h(all zero) = the coefficient of 1, which must be 0 and f_p at once.
    Checked: E[h] e^lambda for one cell as a power series (c = 1, 2, 3), and two cells, c = 1."""
    lam = sp.Symbol("lambda", positive=True)
    for c in (1, 2, 3):
        hs = sp.symbols(f"h0:{c + 6}")  # h(recorded value), recorded 0 for n <= c
        series = sum((hs[0] if n <= c else hs[n]) * lam**n / sp.factorial(n) for n in range(c + 6))
        target = sp.series(lam * sp.exp(lam), lam, 0, c + 6).removeO()
        eqs = [sp.Eq(series.coeff(lam, n), target.coeff(lam, n)) for n in range(2)]
        assert sp.solve(eqs, hs[0]) == []  # h0 = 0 and h0 = 1
    l1, l2 = sp.symbols("l1 l2", positive=True)
    f1, f2 = sp.symbols("f1 f2", real=True)
    h = {}
    for n1, n2 in itertools.product(range(3), repeat=2):
        h[(n1 if n1 > 1 else 0, n2 if n2 > 1 else 0)] = sp.Symbol(f"h_{n1 if n1 > 1 else 0}_{n2 if n2 > 1 else 0}")
    series = sum(
        h[(n1 if n1 > 1 else 0, n2 if n2 > 1 else 0)] * l1**n1 * l2**n2 / (sp.factorial(n1) * sp.factorial(n2))
        for n1, n2 in itertools.product(range(3), repeat=2)
    )
    target = sp.expand((f1 * l1 + f2 * l2) * sp.series(sp.exp(l1), l1, 0, 3).removeO() *
                       sp.series(sp.exp(l2), l2, 0, 3).removeO())  # fmt: skip
    P = sp.Poly(series - target, l1, l2)
    sol = sp.solve([P.coeff_monomial(1), P.coeff_monomial(l1), P.coeff_monomial(l2)], [h[(0, 0)], f1, f2], dict=True)
    assert sol == [{h[(0, 0)]: 0, f1: 0, f2: 0}]


def test_cut_deficit_bound():
    """rf-cut-bound: a cell's light lost to the cut, lambda P(n <= c - 1), is at most kappa_c whatever lambda:
    kappa_1 = 1/e (at lambda = 1), kappa_2 = phi^3 e^-phi = 0.840 (at the golden ratio); and the box numbers: a 7 x 7
    pixel, 3 frame box has sum |x - m| = 252 cells (one axis), kappa_1 times that 92.7, so a first moment of N photons
    is off by at most 92.7 / N px: 0.0093 px at 1e4; below the noise sigma / sqrt(N) once N >= (92.7 / sigma)^2, 8600
    for sigma = 1 px."""
    lam = sp.Symbol("lambda", positive=True)
    d1 = lam * sp.exp(-lam)
    assert sp.solve(sp.diff(d1, lam), lam) == [1] and d1.subs(lam, 1) == sp.exp(-1)
    d2 = lam * (1 + lam) * sp.exp(-lam)
    crit = sp.solve(sp.diff(d2, lam), lam)
    phi = (1 + sp.sqrt(5)) / 2
    assert any(is_zero(r - phi) for r in crit)
    assert sp.N(sp.diff(d2, lam, 2).subs(lam, phi)) < 0
    assert is_zero(d2.subs(lam, phi) - phi**3 * sp.exp(-phi)) and rounds_to(phi**3 * sp.exp(-phi), "0.840")
    W = 3 * 7 * sum(abs(k) for k in range(-3, 4))
    assert W == 252
    assert rounds_to(W * sp.exp(-1), "92.7") and rounds_to(W * sp.exp(-1) / 10**4, "0.0093")
    assert float(f"{float((W * sp.exp(-1)) ** 2):.2g}") == 8600


def test_group_rule_bound():
    """rf-group: every kept neighbour of p is in p's group, so |G_p| >= 1 + (kept neighbours) on every pattern, and
    P(|G_p| < m | p kept) <= P(at most m - 2 of the 8 neighbours kept) (checked on every pattern of p's 5 x 5 window
    for m = 2, 3); that probability falls in each neighbour's a_q (its derivative is -P(exactly m - 2 of the others)),
    so neighbours each kept with at least a* give at most the binomial tail: for m = 3, (1 - a)^8 + 8 a (1 - a)^7,
    8.2e-5 at a* = P(n > 1) for 3 photons. Without such a bound none exists: a lone pair of kept pixels loses all of
    its light at m = 3 however bright."""
    from anri.io.imaged11 import segment

    nb8 = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    win = [(i, j) for i in range(5) for j in range(5)]
    rng = np.random.default_rng(0)
    pats = rng.integers(0, 2, (4000, 25))
    pats[:, 12] = 1  # p kept
    frame = np.repeat(np.arange(len(pats)), 25)
    pixel = np.tile(np.arange(25), len(pats))
    counts = 1 + pats.ravel()  # kept iff count 2 > cut 1
    for m in (2, 3):
        rec = segment(frame, pixel, counts, (5, 5), 1, m).reshape(len(pats), 25)[:, 12]
        kept_nb = sum(pats[:, (2 + a) * 5 + (2 + b)] for a, b in nb8)
        assert np.all(~rec <= (kept_nb <= m - 2))  # not recorded => at most m - 2 kept neighbours
    assert len(win) == 25
    a = sp.symbols("a0:8", positive=True)
    for j in (0, 1):
        P = sum(
            sp.Mul(*[a[i] if i in S else 1 - a[i] for i in range(8)])
            for k in range(j + 1)
            for S in itertools.combinations(range(8), k)
        )
        dP = sp.diff(P, a[0])
        others = sum(
            sp.Mul(*[a[i] if i in S else 1 - a[i] for i in range(1, 8)]) for S in itertools.combinations(range(1, 8), j)
        )
        assert is_zero(dP + others)
    lam = sp.Integer(3)
    astar = 1 - sp.exp(-lam) * (1 + lam)
    tail = (1 - astar) ** 8 + 8 * astar * (1 - astar) ** 7
    assert rounds_to(tail * 10**5, "8.2")
    # the lone pair: two adjacent kept pixels, nothing else, m = 3: neither recorded
    assert not segment(np.zeros(2, int), np.array([12, 13]), np.array([1000, 1000]), (5, 5), 1, 3).any()


def test_recorded_values_increasing():
    """rf-harris: every recorded value r_p is a non-decreasing function of every cell's count (a kept pixel stays kept
    with more counts, and more kept pixels only join groups), so by Harris' inequality Cov(r_p, r_q) >= 0 for
    independent counts. Checked: every pattern of counts 0..3 on a 1 x 5 strip, raising any one count by one, c = 1,
    m = 1, 2, 3 (segment); and the one-variable step 2 Cov(f, g) = E[(f(X) - f(Y))(g(X) - g(Y))] (X, Y independent
    copies), whose integrand is >= 0 for non-decreasing f and g."""
    from anri.io.imaged11 import segment

    n = 5
    pats = np.array(list(itertools.product(range(4), repeat=n)))
    frame = np.repeat(np.arange(len(pats)), n)
    pixel = np.tile(np.arange(n), len(pats))
    for m in (1, 2, 3):

        def recorded(counts, m=m):
            keep = segment(frame, pixel, counts.ravel(), (1, n), 1, m)
            return np.where(keep, counts.ravel(), 0).reshape(-1, n)

        base = recorded(pats)
        for i in range(n):
            up = pats.copy()
            up[:, i] += 1
            assert np.all(recorded(up) >= base)
    fx, fy, gx, gy = sp.symbols("fx fy gx gy")
    # E over independent copies: E[f(X)g(X)] = E[f(Y)g(Y)] = E[fg]; E[f(X)g(Y)] = E[f]E[g]
    Efg, Ef, Eg = sp.symbols("Efg Ef Eg")
    expanded = sp.expand((fx - fy) * (gx - gy))
    E = expanded.subs({fx * gx: Efg, fy * gy: Efg, fx * gy: Ef * Eg, fy * gx: Ef * Eg})
    assert is_zero(E - 2 * (Efg - Ef * Eg))
