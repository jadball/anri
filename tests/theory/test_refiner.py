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
    variation) shift the population's mean F from E[exp([d]x)] = I + (S - tr S I)/2 by a turn [S l]x, S = Cov(d), to second order. With
    the Lorentz factor 1 / (sin 2 theta |sin eta|), l = -cot(eta) d eta / d d, so |S l| ~ sigma^2 |cot eta|: 1.49e-3
    rad for sigma = 1 deg at |sin eta| = 0.2."""
    d = sp.Matrix(sp.symbols("d0:3", real=True))
    l = sp.Matrix(sp.symbols("l0:3", real=True))
    S = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"S_{min(i, j)}{max(i, j)}"))
    # Rodrigues' formula to second order: exp([d]x) = I + [d]x + [d]x^2 / 2 + O(d^3); third-order terms dropped
    R2 = sp.eye(3) + skew(d) + skew(d) * skew(d) / 2
    expr = ((1 + (l.T * d)[0]) * R2).applyfunc(lambda e: _truncate2(e, d))
    E = expr.applyfunc(lambda e: _expect2(e, d, S))
    assert is_zero(E - (sp.eye(3) + (S - S.trace() * sp.eye(3)) / 2 + skew(S * l)))
    sig2 = (sp.pi / 180) ** 2
    cot = sp.sqrt(1 - sp.Rational(1, 25)) / sp.Rational(1, 5)
    assert rounds_to(sig2 * cot * 10**3, "1.49")


def _truncate2(expr: sp.Expr, d: sp.Matrix) -> sp.Expr:
    """The terms of a polynomial in d of degree <= 2."""
    p = sp.Poly(sp.expand(expr), *d)
    return sum((coef * sp.prod([v**n for v, n in zip(d, mono)]) for mono, coef in p.terms() if sum(mono) <= 2), sp.S(0))


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


def test_weak_for_any_statistic():
    """rf-weak-any: a photon of a reflection whose g0 = a k + c z + s xi (k, z, xi orthonormal) has g = F g0, so any
    function of it, f(F g0), is unchanged by F -> F + v xi^T (any v) when s = 0: whatever statistic is taken (pixels,
    histograms, moments of any order), the data at frequency k see F's action on k and z, and on xi only through s
    (sin theta). For |g|^2 the six entries of F^T F enter with weights a^2, c^2, 2ac, 2as, 2cs, s^2."""
    a, c, s = sp.symbols("a c s", real=True)
    F = _mat("F")
    v = sp.Matrix(sp.symbols("v0:3", real=True))
    k, z, xi = sp.Matrix([1, 0, 0]), sp.Matrix([0, 0, 1]), sp.Matrix([0, 1, 0])
    g0 = a * k + c * z + s * xi
    f = sp.Function("f")
    g, gv = F * g0, (F + v * xi.T) * g0
    assert is_zero((f(*gv) - f(*g)).subs(s, 0))
    C = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"C_{min(i, j)}{max(i, j)}"))
    q = sp.expand((g0.T * C * g0)[0])
    kk, zz, xx, kz, kx, zx = C[0, 0], C[2, 2], C[1, 1], C[0, 2], C[0, 1], C[1, 2]
    assert is_zero(q - (a**2 * kk + c**2 * zz + 2 * a * c * kz + 2 * a * s * kx + 2 * c * s * zx + s**2 * xx))


def test_weak_amplification():
    """rf-amplify: G = U S V^T (U, V orthogonal, S diagonal): the least-squares bias is V S^-1 U^T Delta, so along v_i
    it is <u_i, Delta> / s_i (symbolic s_i, U and V rotations rational in a parameter). A nuisance column e fitted alongside a column g
    lowers its information: g^T P g = g^T g - (e^T g)^2 / e^T e <= g^T g, P = I - e e^T / e^T e."""
    s = sp.symbols("s0:2", positive=True)
    t, u = sp.symbols("t u", real=True)

    def rot(q):  # every rotation of the plane but by 180 degrees, rational in q
        return sp.Matrix([[1 - q**2, -2 * q], [2 * q, 1 - q**2]]) / (1 + q**2)

    U, V = rot(t), rot(u)
    G = U * sp.diag(*s) * V.T
    Delta = sp.Matrix(sp.symbols("D0:2"))
    b = (G.T * G).inv() * G.T * Delta
    for i in range(2):
        assert is_zero(sp.simplify(V[:, i].dot(b) - U[:, i].dot(Delta) / s[i]))
    g = sp.Matrix(sp.symbols("g0:3", real=True))
    e = sp.Matrix(sp.symbols("e0:3", real=True))
    P = sp.eye(3) - e * e.T / e.dot(e)
    assert is_zero(sp.simplify((g.T * P * g)[0] - (g.dot(g) - e.dot(g) ** 2 / e.dot(e))))


def test_discrete_weak_directions():
    """rf-discrete: a field constant on voxels, beta(x) = sum_v B_v chi(x - x_v) (lattice step h), has the transform
    chi_hat(k) B(k) with B(k) = sum_v B_v e^{-i k . x_v} periodic: B(k + 2 pi n / h) = B(k) for integer n (here in
    each axis, symbolic, voxels at integer multiples of h). The reflections carrying the alias k_n = k + 2 pi n / h see
    B(k) applied to k_n-hat and z; written in the frame of k, B k_n-hat = (k_n-hat . k-hat) B k-hat
    + (k_n-hat . xi-hat) B xi-hat, so the weak combination B(k) xi-hat(k) is seen with the coefficient
    (k_n-hat . xi-hat) chi_hat(k_n) T(k_n): through the voxel's shape and the beam's transfer at frequencies beyond k.
    At the first alias of a beam of FWHM h (Lambda = 1 in lim-transfer) T = exp(-pi^2 / (4 ln 2)) = 0.028."""
    h = sp.Symbol("h", positive=True)
    k = sp.Symbol("k", real=True)
    n, v = sp.symbols("n v", integer=True)
    phase = sp.exp(-sp.I * k * v * h)
    alias = sp.exp(-sp.I * (k + 2 * sp.pi * n / h) * v * h)
    assert sp.simplify(alias / phase) == 1  # e^{-2 pi i n v} = 1
    th, psi = sp.symbols("theta psi", real=True)  # directions of k and of the alias k_n in the slice
    kh = sp.Matrix([sp.cos(th), sp.sin(th), 0])
    xh = sp.Matrix([0, 0, 1]).cross(kh)
    knh = sp.Matrix([sp.cos(psi), sp.sin(psi), 0])
    B = _mat("B")
    decomp = knh.dot(kh) * (B * kh) + knh.dot(xh) * (B * xh)
    assert is_zero(B * knh - decomp)
    assert is_zero(sp.simplify(knh.dot(xh) - sp.sin(psi - th)))  # zero only when the alias is along k
    T = sp.exp(-(sp.pi**2) / (4 * sp.log(2)))
    assert rounds_to(T, "0.028")


def test_centred_features():
    """rf-centre: an error dG in the geometry (beam weights, runout) gives a misfit dG Y. Centred on a reference
    (features g - g_R, |g|^2 - |g_R|^2, with the counts' column), the misfit is dG (Y - Y_R): proportional to the
    deformation, not to the lattice. Here: one region, two populations of weights a_k (errors da_k), first moments."""
    a = sp.symbols("a0:2", positive=True)
    da = sp.symbols("da0:2", real=True)
    V = sp.symbols("V0:2", positive=True)
    gR = sp.Matrix(sp.symbols("gR0:3", real=True))
    m = [gR + sp.Matrix(sp.symbols(f"dm{k}_0:3", real=True)) for k in range(2)]  # each population's mean g
    true = sum(((a[k] + da[k]) * V[k] * (m[k] - gR) for k in range(2)), sp.zeros(3, 1))
    model = sum((a[k] * V[k] * (m[k] - gR) for k in range(2)), sp.zeros(3, 1))
    assert is_zero(true - model - sum((da[k] * V[k] * (m[k] - gR) for k in range(2)), sp.zeros(3, 1)))
    raw = sum(((a[k] + da[k]) * V[k] * m[k] for k in range(2)), sp.zeros(3, 1)) - sum(
        (a[k] * V[k] * m[k] for k in range(2)), sp.zeros(3, 1)
    )
    assert is_zero(raw - sum((da[k] * V[k] * m[k] for k in range(2)), sp.zeros(3, 1)))  # the whole lattice


def test_projected_estimator():
    """rf-project: keeping only the singular directions with s_i >= s_min, the bias is sum_kept <u_i, Delta> / s_i v_i,
    whose squared length sum_kept <u_i, Delta>^2 / s_i^2 <= |Delta|^2 / s_min^2 (orthonormal u_i, v_i; Bessel's
    inequality sum_kept <u_i, Delta>^2 <= |Delta|^2). Symbolic: three directions kept of four, U a product of rotations."""
    s = sp.symbols("s0:3", positive=True)
    smin = sp.Symbol("smin", positive=True)
    c = sp.symbols("c0:3", real=True)  # <u_i, Delta>, the kept ones; the fourth component is the rest
    rest = sp.Symbol("r", real=True)
    norm2 = sum(ci**2 for ci in c) + rest**2  # |Delta|^2 in the U basis
    bias2 = sum(c[i] ** 2 / s[i] ** 2 for i in range(3))
    # with s_i = smin + d_i, d_i >= 0: |Delta|^2 / smin^2 - bias2 is a sum of non-negative terms
    d = sp.symbols("d0:3", nonnegative=True)
    gap = sp.together((norm2 / smin**2 - bias2).subs({s[i]: smin + d[i] for i in range(3)}))
    num, den = sp.fraction(gap)
    # every monomial holds c_i^2 or r^2 (even), with d_i, smin >= 0, and a non-negative coefficient
    assert all(coef >= 0 for coef in sp.Poly(sp.expand(num), *c, rest, *d, smin).coeffs())
    assert den.is_positive


def test_sin_theta_numbers():
    """rf-amplify's numbers: Ni (a = 3.524 A) has sin(theta) = lambda / (2 d) of 0.0435 for 111 at 70 keV, 0.0709 for
    111 at 43 keV, 0.183 for 420 at 43 keV (lambda = 12.398 / E A): weak combinations amplified by 1 / sin theta =
    5.5-23, and C_xixi by 1 / sin^2 theta = 30-530."""
    hc, a0 = sp.Rational(12398, 1000), sp.Rational(3524, 1000)
    vals = []
    for E, hkl in ((70, (1, 1, 1)), (43, (1, 1, 1)), (43, (4, 2, 0))):
        d = a0 / sp.sqrt(sum(h * h for h in hkl))
        vals.append(hc / E / (2 * d))
    assert [rounds_to(v, q) for v, q in zip(vals, ("0.0435", "0.0709", "0.183"))] == [True] * 3
    assert rounds_to(1 / vals[2], "5.5") and rounds_to(1 / vals[0], "23")
    assert float(f"{float(1 / vals[2] ** 2):.1g}") == 30 and float(f"{float(1 / vals[0] ** 2):.2g}") == 530


def test_box_margin():
    """rf-box-margin: a rotation error delta (|delta| <= e) of the indexer's orientation moves a spot by
    d omega = j_w . delta, d eta = j_e . delta (dif-rotation-motion: j_w = (0, -cot eta, -1),
    j_e = (1, -tan theta / sin eta, 0)), so by Cauchy-Schwarz |d omega| <= e |j_w| = e / |sin eta| and
    |d eta| <= e sqrt(1 + tan^2 theta / sin^2 eta), attained for delta along j; a lattice strain eps moves 2 theta by
    -2 eps tan theta (Bragg's law). The strain's omega and eta terms are test_box_strain's."""
    eta, th = sp.symbols("eta theta", positive=True)
    jw = sp.Matrix([0, -sp.cot(eta), -1])
    je = sp.Matrix([1, -sp.tan(th) / sp.sin(eta), 0])
    assert is_zero(sp.simplify(jw.dot(jw) - 1 / sp.sin(eta) ** 2))
    assert is_zero(je.dot(je) - (1 + sp.tan(th) ** 2 / sp.sin(eta) ** 2))
    d = sp.Matrix(sp.symbols("d0:3", real=True))
    # Cauchy-Schwarz: |j|^2 |d|^2 - (j.d)^2 = |j x d|^2 >= 0, zero for d parallel to j
    for j in (jw, je):
        assert is_zero(sp.simplify(j.dot(j) * d.dot(d) - j.dot(d) ** 2 - j.cross(d).dot(j.cross(d))))
    # Bragg: lambda = 2 d sin(theta); d -> d (1 + eps): d theta / d eps = -tan theta at eps = 0
    eps, lam, d0 = sp.symbols("epsilon lambda d_0", positive=True)
    theta_eps = sp.asin(lam / (2 * d0 * (1 + eps)))
    dth = sp.diff(theta_eps, eps).subs(eps, 0)
    assert is_zero(sp.simplify(dth + sp.tan(sp.asin(lam / (2 * d0)))))


def test_box_strain():
    """rf-box-strain, rf-box-margin: g at its diffraction (beam along x, axis z, lab frame at the diffracting omega,
    g = k_out - k_in), turned by eps [d]x or stretched by (1 + eps). The Ewald condition k_in . g + |g|^2 / 2 = 0 and
    the implicit function theorem give the rotation's d omega = -d_z - d_y cot(eta) and d eta = d_x - d_y tan(theta) /
    sin(eta) (dif-rotation-motion: the convention is the same), and the stretch's d omega = -tan(theta) / sin(eta),
    d eta = -tan^2(theta) cot(eta), d 2theta = 2 tan(theta). For a unit u and symmetric E, |E u|^2 = (u . E u)^2 +
    |E u - (u . E u) u|^2 (stretch and turn). Lagrange's identity gives the combined bounds:
    (x^2 + y^2)(1 + T^2) - (x + T y)^2 = (T x - y)^2 >= 0, so eps_t + eps_g tan(theta) <= |eps| sec(theta), and
    likewise eps_t J + eps_g tan^2(theta) |cot(eta)| <= |eps| sqrt(J^2 + tan^4(theta) cot^2(eta))."""
    th, eta, lam, om, eps = sp.symbols("theta eta lambda omega epsilon", real=True)
    X = sp.Matrix([1, 0, 0])
    k_in = X / lam
    g0 = (sp.Matrix([sp.cos(2 * th), -sp.sin(2 * th) * sp.sin(eta), sp.sin(2 * th) * sp.cos(eta)]) - X) / lam
    at0 = {om: 0, eps: 0}

    def moves(g):
        ewald = k_in.dot(g) + g.dot(g) / 2
        d_om = -ewald.diff(eps).subs(at0) / ewald.diff(om).subs(at0)
        k_out = g + k_in
        dk = (k_out.diff(om) * d_om + k_out.diff(eps)).subs(at0)
        k0 = k_out.subs(at0)
        d_eta = (-dk[1] * k0[2] + k0[1] * dk[2]) / (k0[1] ** 2 + k0[2] ** 2)  # d atan2(-k_y, k_z)
        d_tth = -lam * dk[0] / sp.sin(2 * th)  # cos(2 theta) = lam k_out . x
        return [sp.simplify(sp.trigsimp(x)) for x in (d_om, d_eta, d_tth)]

    d = sp.Matrix(sp.symbols("d0:3", real=True))
    r_om, r_eta, r_tth = moves(rz(om) * (sp.eye(3) + eps * skew(d)) * g0)
    assert is_zero(sp.trigsimp(r_om - (-d[2] - d[1] * sp.cot(eta))))
    assert is_zero(sp.trigsimp(r_eta - (d[0] - d[1] * sp.tan(th) / sp.sin(eta))))
    assert is_zero(r_tth)
    s_om, s_eta, s_tth = moves(rz(om) * (1 + eps) * g0)
    assert is_zero(sp.simplify(sp.trigsimp(s_om + sp.tan(th) / sp.sin(eta))))
    assert is_zero(sp.simplify(sp.trigsimp(s_eta + sp.tan(th) ** 2 * sp.cot(eta))))
    assert is_zero(sp.simplify(sp.trigsimp(s_tth - 2 * sp.tan(th))))
    # stretch and turn of a unit vector
    u = sp.Matrix(sp.symbols("u0:3", real=True))
    E = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"E_{min(i, j)}{max(i, j)}", real=True))
    Eu = E * u
    stretch = u.dot(Eu)
    turn = Eu - stretch * u
    # |E u|^2 - stretch^2 - |turn|^2 = stretch^2 (1 - |u|^2): zero on the unit sphere
    assert is_zero(sp.expand(Eu.dot(Eu) - stretch**2 - turn.dot(turn) - stretch**2 * (1 - u.dot(u))))
    # Lagrange's identity, for both combined bounds
    x, y, T, J, c = sp.symbols("x y T J c", real=True)
    assert is_zero(sp.expand((x**2 + y**2) * (1 + T**2) - (x + T * y) ** 2 - (T * x - y) ** 2))
    assert is_zero(sp.expand((x**2 + y**2) * (J**2 + c**2) - (J * x + c * y) ** 2 - (c * x - J * y) ** 2))
    assert is_zero(sp.simplify(sp.sqrt(1 + sp.tan(th) ** 2) - 1 / sp.cos(th)).subs(th, sp.Rational(1, 7)))


def test_identify_populations():
    """rf-identify: with Z_k = int F U_k B d mu_k, population k adds Z_k h to every region it is in, so two
    populations in one region of every row have equal columns there and only Z_k + Z_k' is seen; their difference is
    seen in the rows where they are in different regions, through Z h for those h. Z is determined by Z h for
    reflections h whose matrix H = [h1 h2 h3] is invertible, and not when they are coplanar (Z -> Z + v n^T,
    n . h = 0, changes nothing); the metric's 6 numbers need 6 reflections with h h^T spanning the symmetric matrices
    (rank 6 for 100, 010, 001, 110, 101, 011; 5 without the last)."""
    Z = _mat("Z")
    h = [sp.Matrix(sp.symbols(f"h{i}_0:3", real=True)) for i in range(3)]
    H = h[0].row_join(h[1]).row_join(h[2])
    assert is_zero((Z * H) * H.inv() - Z)
    v = sp.Matrix(sp.symbols("v0:3", real=True))
    n = h[0].cross(h[1])
    for hh in (h[0], h[1], h[0] + 2 * h[1]):  # coplanar reflections
        assert is_zero((Z + v * n.T) * hh - Z * hh)
    hs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

    def vech(x):
        x = sp.Matrix(x)
        m = x * x.T
        return [m[0, 0], m[1, 1], m[2, 2], 2 * m[0, 1], 2 * m[0, 2], 2 * m[1, 2]]

    assert sp.Matrix([vech(x) for x in hs]).rank() == 6
    assert sp.Matrix([vech(x) for x in hs[:5]]).rank() == 5


def test_split_first_moments():
    """rf-split: at an in-plane frequency k (xi = z x k), the in-plane columns of M_hat (a sample-frame operator) split by
    k k^T + xi xi^T = I_2 into orthogonal idempotent parts; M_hat xi |k| = (k_x M_y - k_y M_x)_hat, the Fourier
    transform of i times (curl M)_z per row (not the Nye tensor: rf-curl-reciprocal): the weak part is each row's solenoidal (divergence-free) in-plane field, the seen part
    its gradient part, k k^T Z_hat = k (k . Z_hat) / |k|^2. On a domain Omega, div(phi w) = grad phi . w + phi div w,
    so by the divergence theorem every gradient is orthogonal to every w with div w = 0 in Omega and w . n = 0 on its
    boundary: L^2(Omega) = grad H^1 (+) those w, the seen part grad phi from the Neumann problem
    lap phi = div v, d phi / dn = v . n."""
    kx, ky = sp.symbols("k_x k_y", real=True)
    kk = sp.sqrt(kx**2 + ky**2)
    kh = sp.Matrix([kx, ky]) / kk
    xh = sp.Matrix([-ky, kx]) / kk
    Pk, Px = kh * kh.T, xh * xh.T
    assert is_zero(sp.simplify(Pk + Px - sp.eye(2)))
    assert is_zero(sp.simplify(Pk * Pk - Pk)) and is_zero(sp.simplify(Pk * Px))
    Zx, Zy = sp.symbols("Z_x Z_y")  # one row's in-plane columns at k
    w = sp.Matrix([Zx, Zy])
    assert is_zero(sp.simplify(xh.dot(w) * kk - (kx * Zy - ky * Zx)))
    x, y = sp.symbols("x y", real=True)
    phi = sp.Function("phi")(x, y)
    wx, wy = sp.Function("w_x")(x, y), sp.Function("w_y")(x, y)
    lhs = sp.diff(phi * wx, x) + sp.diff(phi * wy, y)
    rhs = sp.diff(phi, x) * wx + sp.diff(phi, y) * wy + phi * (sp.diff(wx, x) + sp.diff(wy, y))
    assert is_zero(sp.expand(lhs - rhs))


def test_split_metric():
    """rf-split-metric: the in-plane block of a symmetric field's transform C_hat splits into its k k, k xi and
    xi xi components (an orthonormal basis of 2 x 2 symmetric matrices: k k^T, (k xi^T + xi k^T) / sqrt 2,
    xi xi^T), seen through s^0, s^1, s^2 (rf-weak-any). |k|^2 xi^T C_hat xi = k_y^2 C_xx + k_x^2 C_yy - 2 k_x k_y C_xy,
    the transform of -(inc C)_zz, Saint-Venant's incompatibility; a compatible field sym grad u has no xi xi part, a
    Hessian grad grad phi only a k k part, an Airy field (psi_yy, psi_xx, -psi_xy) only a xi xi part."""
    kx, ky = sp.symbols("k_x k_y", real=True)
    kk = sp.sqrt(kx**2 + ky**2)
    kh = sp.Matrix([kx, ky]) / kk
    xh = sp.Matrix([-ky, kx]) / kk
    B = [kh * kh.T, (kh * xh.T + xh * kh.T) / sp.sqrt(2), xh * xh.T]
    for i in range(3):
        for j in range(3):
            ip = sp.simplify(sum(B[i][a, b] * B[j][a, b] for a in range(2) for b in range(2)))
            assert ip == (1 if i == j else 0)
    Cxx, Cyy, Cxy = sp.symbols("C_xx C_yy C_xy")
    C = sp.Matrix([[Cxx, Cxy], [Cxy, Cyy]])
    assert is_zero(sp.simplify((xh.T * C * xh)[0] * kk**2 - (ky**2 * Cxx + kx**2 * Cyy - 2 * kx * ky * Cxy)))
    ux, uy, p, q = sp.symbols("u_x u_y phi psi")
    k = sp.Matrix([kx, ky])
    u = sp.Matrix([ux, uy])
    compat = sp.I * (u * k.T + k * u.T) / 2  # sym grad u
    assert is_zero(sp.simplify((xh.T * compat * xh)[0]))
    hess = -k * k.T * p  # grad grad phi
    assert is_zero(sp.simplify((xh.T * hess * xh)[0])) and is_zero(sp.simplify((kh.T * hess * xh)[0]))
    airy = -sp.Matrix([[ky**2, -kx * ky], [-kx * ky, kx**2]]) * q  # (psi_yy, -psi_xy; -psi_xy, psi_xx)
    assert is_zero(sp.simplify((kh.T * airy * kh)[0])) and is_zero(sp.simplify((kh.T * airy * xh)[0]))


def test_beam_quarter_period():
    """rf-split's resolution: the beam's transfer exp(-pi^2 / (4 ln 2 Lambda^2)) (lim-transfer) is 1/4 at
    Lambda = pi / (2 sqrt(ln 2 ln 4)) = 1.6 beam FWHM, so below that period a seen component is amplified more than 4
    times."""
    L = sp.Symbol("Lambda", positive=True)
    T = sp.exp(-(sp.pi**2) / (4 * sp.log(2) * L**2))
    root = sp.pi / (2 * sp.sqrt(sp.log(2) * sp.log(4)))
    assert is_zero(sp.simplify(T.subs(L, root) - sp.Rational(1, 4)))
    assert rounds_to(root, "1.6")


def test_smin():
    """rf-smin: in whitened units (noise covariance I), a unit data pattern u has noise Var(u^T xi) = u^T u = 1 and,
    with each misfit component bounded |Delta_j| <= D_j, bias |u^T Delta| <= sum_j |u_j| D_j (the triangle
    inequality, not checked here). So the estimate along v_i errs by at most (z + sum_j |u_ij| D_j) / s_i at z standard deviations,
    and is kept iff that is <= epsilon: s_min = (z + sum_j |u_ij| D_j) / epsilon. A weak column whose coefficient in
    reflection h is s_h a_h (seen: a_h) has squared norm sum_h s_h^2 a_h^2: its gain relative to the seen column's is
    the a^2-weighted rms of s_h."""
    u = sp.Matrix(sp.symbols("u0:3", real=True))
    Cid = sp.eye(3)
    assert is_zero((u.T * Cid * u)[0] - u.dot(u))
    a = sp.symbols("a0:4", real=True)
    s = sp.symbols("s0:4", positive=True)
    weak = sum((s[h] * a[h]) ** 2 for h in range(4))
    seen = sum(a[h] ** 2 for h in range(4))
    rms2 = sum(a[h] ** 2 * s[h] ** 2 for h in range(4)) / seen
    assert is_zero(sp.simplify(weak / seen - rms2))


def test_histogram_linear():
    """rf-hist: the expected recorded sum over a bin B of cells (each divided by its L P) is sum_k a V_k times
    int K_B(F) d mu_k, K_B(F) the probability that a photon of a sub-volume F is counted in B: linear in each
    population's distribution mu_k, for any bins (three sub-volumes, two populations, symbolic). Each moment of
    rf-moments is a linear functional of the histogram over the cells themselves: sum_p f(g_p) n_p = sum_B f(g_B)
    n_B when each bin is one cell."""
    K = sp.symbols("K0:3", nonnegative=True)  # K_B of three sub-volumes
    w = sp.symbols("w0:3", nonnegative=True)  # their volumes in population 1
    w2 = sp.symbols("x0:3", nonnegative=True)  # and in population 2
    a1, a2, lam = sp.symbols("a1 a2 lambda", positive=True)
    E = a1 * sum(K[i] * w[i] for i in range(3)) + a2 * sum(K[i] * w2[i] for i in range(3))
    mixed = E.subs({w[i]: lam * w[i] for i in range(3)})
    assert is_zero(sp.diff(mixed, lam, 2))  # linear in population 1's distribution
    n = sp.symbols("n0:4")
    f = sp.Function("f")
    g = sp.symbols("g0:4")
    assert is_zero(sum(f(g[p]) * n[p] for p in range(4)) - sum(f(g[B]) * n[B] for B in range(4)))


def test_bin_edge():
    """rf-bin-edge: a photon is moved by the instrument (A4) by e, independent of it, with E|e|^2 = sigma^2; it leaves
    a bin only if |e| >= d, its distance to the bin's edge, and 1(|e| >= d) <= |e|^2 / d^2, so
    |K_B(F) - 1_B(F g0)| <= sigma^2 / d^2 (Chebyshev), whatever the instrument's shape."""
    e2, d = sp.symbols("e2 d", positive=True)  # |e|^2 and the distance
    # for |e| >= d: e2 / d^2 >= 1; for |e| < d: e2 / d^2 >= 0 = the indicator
    assert sp.simplify((e2 / d**2 - 1).subs(e2, d**2 + sp.Symbol("t", nonnegative=True))) >= 0


def test_discrete_projector():
    """rf-split-grid: on the voxel grid, with D the discrete gradient (or Hessian) restricted to a domain and its gauge
    (constants, or affine functions) removed so D has full column rank, P = D (D^T D)^-1 D^T is the orthogonal
    projector onto the seen fields: P = P^T, P^2 = P, P D = D, and (I - P) D = 0 (a 4 x 2 symbolic D). Its natural
    boundary conditions are those of the normal equations D^T D phi = D^T v; nothing else is imposed."""
    D = sp.Matrix(4, 2, sp.symbols("d0:8", real=True))
    P = D * (D.T * D).inv() * D.T
    assert is_zero(sp.simplify(P - P.T))
    assert is_zero(sp.simplify(P * P - P))
    assert is_zero(sp.simplify(P * D - D))
    # a 1D example: forward differences on 4 nodes (3 edges), gauge fixed by dropping node 0's value
    G = sp.Matrix([[1, 0, 0], [-1, 1, 0], [0, -1, 1]])  # phi_1 - phi_0, phi_2 - phi_1, phi_3 - phi_2 with phi_0 = 0
    assert G.rank() == 3
    PG = G * (G.T * G).inv() * G.T
    assert PG == sp.eye(3)  # in 1D every edge field is a gradient: nothing is weak


def test_free_plate_edge():
    """rf-plate: the seen part of a symmetric in-plane field C on a domain is grad grad phi minimising
    1/2 int |grad grad phi - C|^2. With M = grad grad phi - C, pointwise
    M : grad grad psi = d_i(M_ij d_j psi) - d_j((d_i M_ij) psi) + (d_i d_j M_ij) psi, so the first variation is
    int (d_i d_j M_ij) psi + the boundary integral of n_i M_ij d_j psi - (d_i M_ij) n_j psi. On an edge (here y = 0,
    n = (0, -1), t = (1, 0)), the tangential derivative's term differs from -(d_s M_nt) psi by a total derivative
    along the edge, -d_x(M_xy psi), which integrates to zero round a closed boundary: the boundary term is
    M_nn d_n psi - (Q_n + d_s M_nt) psi, Q_n = (d_i M_ij) n_j. psi and d_n psi are free on the boundary, so:
    biharmonic grad grad : M = 0 inside, and the free plate edge M_nn = 0, Q_n + d_s M_nt = 0 (with M_nt continuous
    at corners). The gradient's analogue (rf-split) is Neumann: d_n phi = v . n."""
    x, y = sp.symbols("x y", real=True)
    Mxx, Mxy, Myy = (sp.Function(f"M_{a}")(x, y) for a in ("xx", "xy", "yy"))
    M = sp.Matrix([[Mxx, Mxy], [Mxy, Myy]])
    psi = sp.Function("psi")(x, y)
    X = (x, y)
    lhs = sum(M[i, j] * sp.diff(psi, X[i], X[j]) for i in range(2) for j in range(2))
    divM = [sum(sp.diff(M[i, j], X[i]) for i in range(2)) for j in range(2)]  # (d_i M_ij)_j
    flux = [sum(M[i, j] * sp.diff(psi, X[j]) for j in range(2)) - divM[i] * psi for i in range(2)]
    rhs = (
        sum(sp.diff(flux[i], X[i]) for i in range(2))
        + sum(sp.diff(M[i, j], X[i], X[j]) for i in range(2) for j in range(2)) * psi
    )
    assert is_zero(sp.expand(lhs - rhs))
    n, t = sp.Matrix([0, -1]), sp.Matrix([1, 0])
    boundary = sum(n[i] * flux[i] for i in range(2))  # n . flux on y = 0
    Mnn, Mnt = (n.T * M * n)[0], (n.T * M * t)[0]
    Qn = sum(divM[j] * n[j] for j in range(2))
    dn_psi = sum(n[j] * sp.diff(psi, X[j]) for j in range(2))
    kirchhoff = Mnn * dn_psi - (Qn + sp.diff(Mnt, x)) * psi
    assert is_zero(sp.expand(boundary - kirchhoff + sp.diff(Mxy * psi, x)))


def test_polar_mean():
    """rf-polar: the polar factor Q of M = Q S moves, to first order, by Q Omega with S Omega + Omega S = W,
    W = Q^T dM - dM^T Q (skew); for S diagonal, Omega_ij = W_ij / (s_i + s_j). About Q = I with dM = E[R eps]
    (F = R (I + eps), eps symmetric, R = I + [d]x), the skew part of dM is that of E[[d]x eps]: the symmetric eps
    drops out, so the mean rotation read from int F differs from the rotations' own chordal mean only through the
    correlation of a sub-volume's rotation d with its strain eps."""
    s = sp.symbols("s0:3", positive=True)
    S = sp.diag(*s)
    w = sp.symbols("w0:3", real=True)
    W = skew(sp.Matrix(w))
    Om = sp.Matrix(3, 3, lambda i, j: W[i, j] / (s[i] + s[j]))
    assert is_zero(S * Om + Om * S - W)
    assert is_zero(Om + Om.T)  # skew: a rotation's tangent
    # M(t) = Q(t) S(t): with Q = I + t Omega, S = S0 + t Sd (Sd symmetric), dM = Omega S0 + Sd, and
    # Q^T dM - dM^T Q = Omega S0 + S0 Omega at first order
    Sd = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"q_{min(i, j)}{max(i, j)}"))
    dM = Om * S + Sd
    assert is_zero((dM - dM.T) - W)
    # about Q = I: dM = E[R eps] with R = I + [d]x: its skew part is that of E[[d]x eps] (eps symmetric drops out)
    d = sp.Matrix(sp.symbols("d0:3", real=True))
    eps = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"e_{min(i, j)}{max(i, j)}"))
    dMs = (sp.eye(3) + skew(d)) * eps
    cross = skew(d) * eps
    assert is_zero((dMs - dMs.T) - (cross - cross.T))


def test_box_shift():
    """rf-shift: for any light rho along an axis, the first moment over a box [-t + delta, t + delta] differs from that
    over [-t, t] by the light of two slabs of width delta at its faces: int_t^{t+delta} x rho - int_{-t}^{-t+delta}
    x rho (0 < delta < 2t, an antiderivative F of x rho, symbolic). On those slabs |x| <= t + delta (by inspection), so
    the change is at most (t + delta) times the light in the slabs, whatever rho is; for light symmetric about 0 the centred box's
    moment is 0 (ref-symmetric)."""
    t, dl = sp.symbols("t delta", positive=True)
    Fn = sp.Function("F")

    def I(a, b):
        return Fn(b) - Fn(a)

    assert is_zero((I(-t + dl, t + dl) - I(-t, t)) - (I(t, t + dl) - I(-t, -t + dl)))


def test_arc_box():
    """rf-arc-box: a rotation exp([d]x) turns g_hat to g_hat + d x g_hat + (1/2) d x (d x g_hat): to first order its
    image is -[g]x d, with covariance [g]x S [g]x^T for Cov(d) = S. The image of the ellipsoid d^T S^-1 d <= k^2
    under a linear map L is the ellipse y^T (L S L^T)^-1 y <= k^2: both have the support function
    k sqrt(w^T L S L^T w) (attained at d* = k S L^T w / sqrt(w^T L S L^T w), and no larger by Cauchy-Schwarz in the
    metric S). The second-order term's part across g is (1/2) d_perp (d . g), of length at most |d|^2 / 4."""
    d = sp.Matrix(sp.symbols("d0:3", real=True))
    gh = sp.Matrix(sp.symbols("g0:3", real=True))
    assert is_zero(d.cross(gh) + skew(gh) * d)
    S = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"S_{min(i, j)}{max(i, j)}"))
    assert is_zero(skew(gh) * S * skew(gh).T - (skew(gh) * S * skew(gh).T).T)  # symmetric
    # support function, with a concrete SPD S and L (rational) and symbolic w, k
    Sc = sp.Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 1]])
    L = sp.Matrix([[1, 0, 2], [0, 1, -1]])
    w = sp.Matrix(sp.symbols("w0:2", real=True))
    k = sp.Symbol("k", positive=True)
    q = (w.T * L * Sc * L.T * w)[0]
    ds = k * Sc * L.T * w / sp.sqrt(q)
    assert is_zero(sp.simplify((ds.T * Sc.inv() * ds)[0] - k**2))
    assert is_zero(sp.simplify((w.T * L * ds)[0] - k * sp.sqrt(q)))
    # Cauchy-Schwarz in the inner product <u, v> = u^T S v gives (w^T L d)^2 <= (w^T L S L^T w)(d^T S^-1 d): its terms
    dd = sp.Matrix(sp.symbols("e0:3", real=True))
    a, b = Sc.inv() * dd, L.T * w  # <u, v>_S := u^T S v; u = S^-1 d, v = L^T w
    assert is_zero(sp.expand((b.T * Sc * b)[0] - q))
    assert is_zero(sp.expand((a.T * Sc * b)[0] - (w.T * L * dd)[0]))
    assert is_zero(sp.expand((a.T * Sc * a)[0] - (dd.T * Sc.inv() * dd)[0]))
    # second order: (1/2) d x (d x g) = (1/2)(d (d.g) - g |d|^2); across g: (1/2) d_perp (d . g)
    gu = sp.Matrix([0, 0, 1])
    sec = d.cross(d.cross(gu)) / 2
    across = sec - sec.dot(gu) * gu
    dperp = d - d.dot(gu) * gu
    assert is_zero(across - dperp * d.dot(gu) / 2)
    x, y = sp.symbols("x y", real=True)  # |d_perp|, |d . g|: x y / 2 <= (x^2 + y^2) / 4
    assert is_zero(sp.expand((x**2 + y**2) / 4 - x * y / 2 - (x - y) ** 2 / 4))


def test_runout_harmonics():
    """rf-runout: a voxel at (x, y) is in the beam at dty = y0 - (x sin w + y cos w) + r(w) (dif-dty with a runout r
    across the beam). (1) A translation of the whole sample by (sx, sy) changes dty by -(sx sin w + sy cos w): exactly
    a runout's first harmonic, so that harmonic is the map's origin, a gauge, not a measurement. (2) Its Friedel pair is
    at w + pi, and dty(w + pi) - (2 y0 - dty(w)) = r(w) + r(w + pi): pairs see the runout's even harmonics (and its
    mean, y0) and none of its odd ones. Checked on harmonics 1-4."""
    w, x, y, y0, sx, sy = sp.symbols("omega x y y0 s_x s_y", real=True)
    c = sp.symbols("c1:5", real=True)
    s = sp.symbols("s1:5", real=True)
    r = sum(c[n - 1] * sp.cos(n * w) + s[n - 1] * sp.sin(n * w) for n in range(1, 5))

    def dty(om, xx, yy, rr):
        return y0 - (xx * sp.sin(om) + yy * sp.cos(om)) + rr

    shifted = dty(w, x + sx, y + sy, 0) - dty(w, x, y, 0)
    assert is_zero(shifted + sx * sp.sin(w) + sy * sp.cos(w))
    pair = dty(w + sp.pi, x, y, r.subs(w, w + sp.pi)) - (2 * y0 - dty(w, x, y, r))
    even = 2 * sum(c[n - 1] * sp.cos(n * w) + s[n - 1] * sp.sin(n * w) for n in (2, 4))
    assert is_zero(sp.simplify(sp.expand_trig(pair - even)))


def test_reflection_lemma():
    """rf-psf-leak: for a point spread radially non-increasing, a photon arriving at y beyond an edge of cell p (in the
    half-plane across it) puts at least as much light in the cell q across that edge as in p: reflecting across the
    edge's line maps p onto q, and for z in p and y across the line |sigma(z) - y| <= |z - y|. Here the line x = 1/2:
    |sigma z - y|^2 - |z - y|^2 = (1 - 2 z_x)(1 - 2 y_x) <= 0 for z_x <= 1/2 <= y_x."""
    zx, zy, yx, yy = sp.symbols("z_x z_y y_x y_y", real=True)
    sig = (1 - zx, zy)
    diff = (sig[0] - yx) ** 2 + (sig[1] - yy) ** 2 - ((zx - yx) ** 2 + (zy - yy) ** 2)
    assert is_zero(sp.expand(diff - (1 - 2 * zx) * (1 - 2 * yx)))


def test_psf_leak_number():
    """rf-psf-leak: for a photon arriving anywhere in cell p, an edge neighbour q of a point spread that is a mixture of
    centred isotropic Gaussians (shares f_c, sds s_c) receives at least
    l = sum_c f_c [Phi(2 / s_c) - Phi(1 / s_c)] [Phi(1 / s_c) - 1/2]: each Gaussian's mass factorises, the mass across
    the edge over [1/2 - y, 3/2 - y] grows with y (its derivative phi(1/2 - y) - phi(3/2 - y) >= 0 for y <= 1/2), the
    mass along it over [-1/2 - y, 1/2 - y] falls with |y| (derivative of opposite sign to y), so the least is at the
    far corner, y = (-1/2, +-1/2). For the measured point spread (0.25 px, 72 %; 1.3 px, 28 %), l = 0.0124."""
    y = sp.Symbol("y", real=True)
    s = sp.Symbol("s", positive=True)

    def Phi(x):
        return (1 + sp.erf(x / sp.sqrt(2))) / 2

    across = Phi((sp.Rational(3, 2) - y) / s) - Phi((sp.Rational(1, 2) - y) / s)
    d_across = sp.diff(across, y)
    phi = lambda x: sp.exp(-(x**2) / 2) / sp.sqrt(2 * sp.pi)
    assert is_zero(sp.simplify(d_across - (phi((sp.Rational(1, 2) - y) / s) - phi((sp.Rational(3, 2) - y) / s)) / s))
    # for y <= 1/2, |1/2 - y| <= |3/2 - y| and phi falls with |x|: the derivative is >= 0
    along = Phi((sp.Rational(1, 2) - y) / s) - Phi((-sp.Rational(1, 2) - y) / s)
    d_along = sp.diff(along, y)
    assert is_zero(sp.simplify(d_along - (phi((-sp.Rational(1, 2) - y) / s) - phi((sp.Rational(1, 2) - y) / s)) / s))
    corner = (Phi(2 / s) - Phi(1 / s)) * (Phi(1 / s) - sp.Rational(1, 2))
    assert is_zero(sp.simplify(corner - across.subs(y, -sp.Rational(1, 2)) * along.subs(y, sp.Rational(1, 2))))
    ell = sp.Rational(72, 100) * corner.subs(s, sp.Rational(1, 4)) + sp.Rational(28, 100) * corner.subs(
        s, sp.Rational(13, 10)
    )
    assert rounds_to(ell, "0.0124")


def test_group_rule_psf_bound():
    """rf-group-psf: with every edge neighbour receiving >= l of each photon arriving in p and, by the reflection
    lemma, some edge neighbour receiving >= its share in p of each photon arriving outside p, the edge neighbours of p
    hold >= l lambda_p together (l < 1); likewise the edge neighbours of a pair {p, q} (6 cells) hold >= min(l, 1/2)
    of the pair's light. The largest of n cells holds at least their mean, and P(Poisson(mu) <= c) falls in mu, so
    1 - w_p <= P(n <= c | l lambda_p / 4) + 8 P(n <= c | l lambda_p / 6) <= 9 P(n <= c | l lambda_p / 6), and the
    group rule's loss in a cell is at most 9 lambda P(n <= c | l lambda / 6) <= (54 / l) kappa, kappa = max mu P(n <= c)
    (0.840 at c = 1): 3.6e3 photons per cell for the measured point spread."""
    mu, lam, ell = sp.symbols("mu lambda ell", positive=True)
    c = 1
    Q = sum(sp.exp(-mu) * mu**j / sp.factorial(j) for j in range(c + 1))
    assert is_zero(sp.simplify(sp.diff(Q, mu) + sp.exp(-mu) * mu**c / sp.factorial(c)))  # falls in mu
    # lambda Q(l lambda / 6) = (6 / l) mu Q(mu) at mu = l lambda / 6: its maximum is (6 / l) kappa
    assert is_zero(sp.simplify(lam * Q.subs(mu, ell * lam / 6) - (6 / ell) * (mu * Q).subs(mu, ell * lam / 6)))
    phi = (1 + sp.sqrt(5)) / 2
    kappa = (mu * Q).subs(mu, phi)
    assert rounds_to(kappa, "0.840")
    lval = sp.Rational(72, 100) * (sp.erf(2 / sp.sqrt(2) * 4) - sp.erf(4 / sp.sqrt(2))) / 2 * sp.erf(
        4 / sp.sqrt(2)
    ) / 2 + sp.Rational(28, 100) * (sp.erf(2 / sp.sqrt(2) / sp.Rational(13, 10)) - sp.erf(
        1 / sp.sqrt(2) / sp.Rational(13, 10))) / 2 * sp.erf(1 / sp.sqrt(2) / sp.Rational(13, 10)) / 2  # fmt: skip
    assert rounds_to(54 * kappa / lval / 1000, "3.6")


def test_weight_lemma():
    """rf-weight-bound: a per-photon weight known only to a factor e^a, |a| <= G r (its log's gradient G times the
    distance r between where it is evaluated and where the photon is), changes each photon's weight by e^a - 1, and
    |e^a - 1| <= e^{|a|} - 1 (equal for a >= 0; for a < 0 the gap is 2 (cosh a - 1) >= 0). So a first moment about the
    population's mean is off by at most (e^{G r} - 1) times the light's mean absolute deviation."""
    a = sp.Symbol("a", negative=True)
    gap = (sp.exp(-a) - 1) - (1 - sp.exp(a))
    assert is_zero(sp.simplify(gap - 2 * (sp.cosh(a) - 1)))


def test_parallax_angle():
    """rf-parallax: a photon from x_v reaching the detector at X is seen from the origin turned by an angle alpha with
    sin(alpha) = |(X - x_v) x X| / (|X - x_v| |X|) = |x_v x X| / (|X - x_v| |X|) <= |x_v| / (|X| - |x_v|)."""
    X = sp.Matrix(sp.symbols("X0:3", real=True))
    v = sp.Matrix(sp.symbols("v0:3", real=True))
    assert is_zero((X - v).cross(X) + v.cross(X))


def test_box_margin_exact():
    """rf-box-margin-exact: along the rotation exp(t [d]x), t in [0, 1], the spot moves at the rates of
    dif-rotation-motion evaluated where it is, |d eta / dt| <= e J(|sin eta|), J(s) = sqrt(1 + tan^2 theta / s^2)
    < 1 + tan theta / s. With s* the larger root of s = s0 - e (1 + tan theta / s), |sin eta| stays above s* for all t
    (at a first touch, |sin eta| >= s0 - e J(s*) > s*), so |d omega| <= e / s* and |d eta| <= e J(s*), exactly; to
    second order e / s* = e / s0 + e^2 (s0 + tan theta) / s0^3. A strain eps (operator norm) turns g by an angle with
    sin <= |eps| / (1 - |eps|), since (I + eps) u x u = eps u x u, and changes |g| by a factor in [1 - |eps|, 1 + |eps|]."""
    s, s0, e, T = sp.symbols("s s_0 e T", positive=True)  # T = tan theta
    roots = sp.solve(sp.Eq(s, s0 - e * (1 + T / s)), s)
    big = (s0 - e + sp.sqrt((s0 - e) ** 2 - 4 * e * T)) / 2
    assert any(is_zero(sp.simplify(r - big)) for r in roots)
    assert is_zero(sp.simplify(big - (s0 - e * (1 + T / big))))
    J = sp.sqrt(1 + T**2 / s**2)
    assert is_zero(sp.simplify((1 + T / s) ** 2 - J**2 - 2 * T / s))  # J^2 < (1 + T/s)^2
    ser = sp.series(e / big, e, 0, 3).removeO()
    assert is_zero(sp.simplify(ser - (e / s0 + e**2 * (s0 + T) / s0**3)))
    u = sp.Matrix(sp.symbols("u0:3", real=True))
    E = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"E_{min(i, j)}{max(i, j)}"))
    assert is_zero(((sp.eye(3) + E) * u).cross(u) - (E * u).cross(u))


def test_group_rule_numbers():
    """rf-group-psf's numbers: the bound peaks at lambda = 6 phi / l = 781 (about 800) for l = 0.0124; a photon at a
    pixel's centre puts 0.0346 of itself in each edge neighbour (2.8 l, about three times), and parallax of 100 um at
    180 mm is sin(alpha) <= 100 / (180000 - 100) = 5.6e-4."""

    def Phi(x):
        return (1 + sp.erf(x / sp.sqrt(2))) / 2

    def corner(s):
        return (Phi(2 / s) - Phi(1 / s)) * (Phi(1 / s) - sp.Rational(1, 2))

    def centre(s):
        return (Phi(sp.Rational(3, 2) / s) - Phi(sp.Rational(1, 2) / s)) * (2 * Phi(sp.Rational(1, 2) / s) - 1)

    f = [(sp.Rational(72, 100), sp.Rational(1, 4)), (sp.Rational(28, 100), sp.Rational(13, 10))]
    ell = sum(w * corner(s) for w, s in f)
    mid = sum(w * centre(s) for w, s in f)
    phi = (1 + sp.sqrt(5)) / 2
    assert rounds_to(6 * phi / ell, "781")
    assert rounds_to(mid, "0.0346") and rounds_to(mid / ell, "2.8")
    assert rounds_to(sp.Rational(100, 180000 - 100) * 10**4, "5.6")


def test_omega_kernel_variance():
    """rf-omega-kernel: with no lattice rotation, ps-omega gives a reflection's omega shift as k_y - k_z cot(eta) -
    (d lambda / lambda) tan(theta) / sin(eta), three independent zero-mean terms, so its variance is
    s_h^2 + s_v^2 cot^2(eta) + s_l^2 tan^2(theta) / sin^2(eta): linear in the three variances, with coefficients known
    from each reflection's (eta, theta). A Gaussian of variance v is a Gaussian of variance v_lo convolved with one of
    v - v_lo (their characteristic functions multiply), so any v_lo <= v is a factor of the kernel, as (A5) needs."""
    ky, kz, dl = sp.symbols("k_y k_z d_l", real=True)
    sh, sv, sl = sp.symbols("s_h s_v s_l", positive=True)
    eta, th = sp.symbols("eta theta", positive=True)
    shift = ky - kz * sp.cot(eta) - dl * sp.tan(th) / sp.sin(eta)
    coeffs = [sp.diff(shift, q) for q in (ky, kz, dl)]
    var = sum(c**2 * s**2 for c, s in zip(coeffs, (sh, sv, sl)))  # independent, zero mean
    assert is_zero(var - (sh**2 + sv**2 * sp.cot(eta) ** 2 + sl**2 * sp.tan(th) ** 2 / sp.sin(eta) ** 2))
    w, v, vlo = sp.symbols("w v v_lo", positive=True)
    assert is_zero(sp.exp(-(w**2) * vlo / 2) * sp.exp(-(w**2) * (v - vlo) / 2) - sp.exp(-(w**2) * v / 2))


def test_omega_identifiable():
    """rf-omega-identify: (1) for a reflection whose predicted omega mu (in frames, |mu| < 1/2 from its frame's
    centre) is known, the share of its light in that frame, Phi((1/2 - mu)/sqrt v) - Phi((-1/2 - mu)/sqrt v), falls
    strictly with the kernel's variance v (its derivative is -(a phi(a/sqrt v) + |b| phi(b/sqrt v)) / (2 v^(3/2)) with
    a = 1/2 - mu > 0 > b = -1/2 - mu), so each reflection's frames determine its v. (2) v = s_h^2 + s_v^2 c + s_l^2 t (1 + c),
    c = cot^2(eta), t = tan^2(theta): three reflections (c1, t1), (c2, t1), (c1, t2) give a determinant
    (c2 - c1)(t2 - t1)(1 + c1) != 0, so the three variances are determined; on one ring (t fixed) the third column is
    t times (1, c) and the rank is 2, whatever the number of reflections."""
    mu = sp.Symbol("mu", real=True)
    v = sp.Symbol("v", positive=True)

    def Phi(x):
        return (1 + sp.erf(x / sp.sqrt(2))) / 2

    def phi(x):
        return sp.exp(-(x**2) / 2) / sp.sqrt(2 * sp.pi)

    a, b = sp.Rational(1, 2) - mu, -sp.Rational(1, 2) - mu
    share = Phi(a / sp.sqrt(v)) - Phi(b / sp.sqrt(v))
    expect = -(a * phi(a / sp.sqrt(v)) - b * phi(b / sp.sqrt(v))) / (2 * v ** sp.Rational(3, 2))
    assert is_zero(sp.simplify(sp.diff(share, v) - expect))  # -b = |b| > 0 and a > 0: strictly negative
    c1, c2, t1, t2 = sp.symbols("c1 c2 t1 t2", positive=True)

    def row(c, t):
        return [1, c, t * (1 + c)]

    D = sp.Matrix([row(c1, t1), row(c2, t1), row(c1, t2)]).det()
    assert is_zero(sp.factor(D) - (c2 - c1) * (t2 - t1) * (1 + c1))
    cs = sp.symbols("k0:5", positive=True)
    t = sp.Symbol("t", positive=True)
    assert sp.Matrix([row(c, t) for c in cs]).rank() == 2


def test_omega_floor_table():
    """rf-omega-bound's table: sum_k exp(-2 pi^2 k^2 r^2) / (pi k), capped at 1/2, for a floor of sd r frames:
    0.5 at 0.05, 0.355 at 0.1, 0.151 at 0.2, 0.054 at 0.3, 0.0023 at 0.5; and the ID11 terms in frames of 0.05 deg:
    bandwidth (3e-4 FWHM, tan theta = 0.08) 0.012 at |sin eta| = 1 and 0.058 at 0.2, horizontal convergence (8e-5 rad)
    0.092."""
    import mpmath as mp

    mp.mp.dps = 20
    quoted = {"0.05": "0.5", "0.1": "0.355", "0.2": "0.151", "0.3": "0.054", "0.5": "0.0023"}
    for r, q in quoted.items():
        rr = mp.mpf(r)
        b = mp.nsum(lambda k, rr=rr: mp.exp(-2 * mp.pi**2 * k**2 * rr**2) / (mp.pi * k), [1, mp.inf])
        assert rounds_to(sp.Float(str(min(b, mp.mpf("0.5"))), 20), q)
    h = sp.pi / 180 * sp.Rational(5, 100)
    sd_bw = sp.Rational(3, 10**4) / (2 * sp.sqrt(2 * sp.log(2))) * sp.Rational(8, 100)
    assert rounds_to(sd_bw / h, "0.012") and rounds_to(sd_bw / sp.Rational(1, 5) / h, "0.058")
    assert rounds_to(sp.Rational(8, 10**5) / h, "0.092")


def test_si_fisher():
    """rf-si-fisher: a Si reflection's counts in its frames are Poisson with means N P_j(mu, v). Its intensity N is a
    nuisance orthogonal to v (the cross information is sum_j d_v P_j = d_v sum_j P_j = 0), and the information on v is
    N sum_j (d_v P_j)^2 / P_j: for two frames N (d_v q)^2 / (q (1 - q)). The chain rule through v_r = x_r . theta,
    x_r = (1, cot^2 eta, tan^2 theta / sin^2 eta) / h^2, gives I(theta) = sum_r I_r x_r x_r^T. Checked for two cells."""
    N, q, dq = sp.symbols("N q dq", positive=True)  # q = P_out(v), dq = d q / d v
    P = [1 - q, q]
    dP = [-dq, dq]
    lam = [N * p for p in P]
    # Poisson information for (N, v): sum_j (d lam_j)(d lam_j)^T / lam_j
    I_NN = sum(pj**2 / l for pj, l in zip(P, lam))
    I_Nv = sum(pj * N * dpj / l for pj, dpj, l in zip(P, dP, lam))
    I_vv = sum((N * dpj) ** 2 / l for dpj, l in zip(dP, lam))
    assert is_zero(sp.simplify(I_Nv))
    assert is_zero(sp.simplify(I_vv - N * dq**2 / (q * (1 - q))))
    assert is_zero(sp.simplify(I_NN - 1 / N))
    th = sp.Matrix(sp.symbols("t0:3", positive=True))
    x = sp.Matrix(sp.symbols("x0:3", positive=True))
    Ir = sp.Symbol("I_r", positive=True)
    v = (x.T * th)[0]
    grad = sp.Matrix([sp.diff(v, ti) for ti in th])
    assert is_zero(Ir * grad * grad.T - Ir * x * x.T)


def test_si_fisher_narrow():
    """rf-si-narrow: a kernel of sd s frames much narrower than a frame spills only across the nearer edge, at
    distance d from the peak: q = Phi(-d/s), d_v q = d phi(d/s) / (2 s^3) (v = s^2). Over reflections spread evenly
    in d on [0, 1/2], the mean information 2 int_0^(1/2) N (d_v q)^2 / (q (1 - q)) dd is, with d = s z,
    (N / (2 s^3)) int_0^(1/(2s)) z^2 phi(z)^2 / (Phi(z) Phi(-z)) dz -> N K / (2 s^3), K = 1.10. So M reflections fix
    v to sd sqrt(2 s^3 / (M N K)) and s to a relative sd 1 / sqrt(2 M N K s): 0.27 % for M = 500, N = 1e4 photons,
    s = 0.012 frames (the bandwidth's), 0.87 % at N = 1e3, 0.099 % at s = 0.092 (the horizontal convergence's)."""
    d, s, z, N = sp.symbols("d s z N", positive=True)

    def Phi(x):
        return (1 + sp.erf(x / sp.sqrt(2))) / 2

    def phi(x):
        return sp.exp(-(x**2) / 2) / sp.sqrt(2 * sp.pi)

    v = sp.Symbol("v", positive=True)
    q = Phi(-d / sp.sqrt(v))
    dq = sp.diff(q, v).subs(v, s**2)
    assert is_zero(sp.simplify(dq - d * phi(d / s) / (2 * s**3)))
    integrand = 2 * N * dq**2 / (Phi(-d / s) * Phi(d / s))
    sub = (integrand.subs(d, s * z) * s).simplify()  # dd = s dz
    target = N / (2 * s**3) * z**2 * phi(z) ** 2 / (Phi(z) * Phi(-z))
    assert is_zero(sp.simplify(sub - target))
    import mpmath as mp

    mp.mp.dps = 20
    K = mp.quad(lambda t: t**2 * mp.npdf(t) ** 2 / (mp.ncdf(t) * mp.ncdf(-t)), [0, 2, 5, mp.inf])
    assert rounds_to(sp.Float(str(K), 20), "1.10")
    M = sp.Symbol("M", positive=True)
    var_v = 2 * s**3 / (M * N * sp.Symbol("K", positive=True))
    rel_s = sp.sqrt(var_v) / (2 * s**2)  # d s = d v / (2 s)
    assert is_zero(sp.simplify(rel_s - 1 / sp.sqrt(2 * M * N * sp.Symbol("K", positive=True) * s)))
    Kf = sp.Float(str(K), 20)
    for Mv, Nv, sv, q_ in ((500, 10**4, sp.Rational(12, 1000), "0.27"), (500, 10**3, sp.Rational(12, 1000), "0.87"),
                           (500, 10**4, sp.Rational(92, 1000), "0.099")):  # fmt: skip
        assert rounds_to(100 / sp.sqrt(2 * Mv * Nv * Kf * sv), q_)


def test_si_position_orthogonal():
    """rf-si-narrow: the frame masses are P_j = G(j - mu, v) with G(x, v) = Phi((x + 1/2)/sqrt v) - Phi((x - 1/2)/sqrt v)
    even in x, so G_x is odd and G_v even: the cross information between a reflection's position mu and v,
    -sum_j G_x G_v / G at j - mu, is odd in mu (j -> -j), and vanishes over reflections spread symmetrically in mu. The
    Si orientation, fitted jointly, costs the widths nothing on average; left unfitted, a position error of sd tau adds
    tau^2 to v (Gaussians convolve), which raises the floor: unsafe."""
    x = sp.Symbol("x", real=True)
    v = sp.Symbol("v", positive=True)

    def Phi(y):
        return (1 + sp.erf(y / sp.sqrt(2))) / 2

    G = Phi((x + sp.Rational(1, 2)) / sp.sqrt(v)) - Phi((x - sp.Rational(1, 2)) / sp.sqrt(v))
    assert is_zero(sp.simplify(G.subs(x, -x) - G))
    Gx, Gv = sp.diff(G, x), sp.diff(G, v)
    assert is_zero(sp.simplify(Gx.subs(x, -x) + Gx))
    assert is_zero(sp.simplify(Gv.subs(x, -x) - Gv))


def test_si_position_number():
    """rf-si-narrow's position scale: a 75 um pixel at 180 mm is 4.17e-4 rad; over sqrt(500) reflections 1.86e-5 rad,
    0.021 frames of 0.05 deg."""
    pix = sp.Rational(75, 180000)
    h = sp.pi / 180 * sp.Rational(5, 100)
    assert rounds_to(pix * 10**4, "4.17") and rounds_to(pix / sp.sqrt(500) * 10**5, "1.86")
    assert rounds_to(pix / sp.sqrt(500) / h, "0.021")


def test_background_confound():
    """rf-background: background light b(x) = b0 + b1 . (x - centre) over a region adds to its counts, first and
    second moments b0 (S0, S1, S2) + b1 . (S1, S2, S3), S_n the box's own moments (with 1/(LP) weights), linearly. In a
    box symmetric about its centre the odd ones vanish, so (1) a constant background leaves the centred first moment
    alone and adds b0 S2 to the second moments (confounded with the spread); (2) a population shifted by m,
    (N, N m, N (Sigma + m m^T)), equals an unshifted one with spread Sigma + m m^T plus a slope b1 = N S2^-1 m, in all
    three, exactly: a free slope per region removes that region's information on position. 2D, cells -2..2 by -1..1."""
    xs = [(i, j) for i in range(-2, 3) for j in range(-1, 2)]
    w = {c: sp.Symbol(f"w_{abs(c[0])}_{abs(c[1])}", positive=True) for c in xs}  # weights symmetric under x -> -x
    X = {c: sp.Matrix(c) for c in xs}
    S1 = sum((w[c] * X[c] for c in xs), sp.zeros(2, 1))
    S2 = sum((w[c] * X[c] * X[c].T for c in xs), sp.zeros(2, 2))
    S3 = [sum(w[c] * X[c][a] * X[c][b] * X[c][d] for c in xs) for a in range(2) for b in range(2) for d in range(2)]
    assert is_zero(S1) and all(is_zero(t) for t in S3)
    assert is_zero(S2[0, 1])
    N = sp.Symbol("N", positive=True)
    m = sp.Matrix(sp.symbols("m0:2", real=True))
    Sig = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"Sig{min(i, j)}{max(i, j)}"))
    b1 = N * S2.inv() * m
    # counts, first, second moments of: shifted population vs unshifted (spread Sig + m m^T) plus the slope
    shifted = (N, N * m, N * (Sig + m * m.T))
    slope_first = sum((w[c] * X[c] * (b1.T * X[c])[0] for c in xs), sp.zeros(2, 1))
    slope_count = sum(w[c] * (b1.T * X[c])[0] for c in xs)
    slope_second = sum((w[c] * X[c] * X[c].T * (b1.T * X[c])[0] for c in xs), sp.zeros(2, 2))
    unshifted = (N + slope_count, sp.zeros(2, 1) + slope_first, N * (Sig + m * m.T) + slope_second)
    assert is_zero(sp.simplify(shifted[0] - unshifted[0]))
    assert is_zero(sp.simplify(shifted[1] - unshifted[1]))
    assert is_zero(sp.simplify(shifted[2] - unshifted[2]))
    b0 = sp.Symbol("b0", real=True)
    assert is_zero(b0 * S1)  # a constant leaves the centred first moment alone


def test_richardson_bound():
    """rf-cost: for A symmetric positive definite with eigenvalues in [l, L], kappa = L / l, the step a = 2 / (l + L)
    gives |1 - a l| = |1 - a L| = (kappa - 1) / (kappa + 1), so the iteration x <- x + a (b - A x) contracts the error by
    that per step; ln((kappa - 1)/(kappa + 1)) = -2 atanh(1/kappa) <= -2 / kappa (atanh's series has positive
    coefficients), so k >= (kappa / 2) ln(1 / tol) steps reach tol. Its k-th iterate is x0 + p(A) r0 with deg p < k, in
    the Krylov space over which conjugate gradients minimises the A-norm error, so CG needs no more."""
    l, L, kap = sp.symbols("l L kappa", positive=True)
    a = 2 / (l + L)
    assert is_zero(sp.simplify((1 - a * l) - (L - l) / (L + l)))
    assert is_zero(sp.simplify((a * L - 1) - (L - l) / (L + l)))
    assert is_zero(sp.simplify(((L - l) / (L + l)).subs(L, kap * l) - (kap - 1) / (kap + 1)))
    y = sp.Symbol("y", positive=True)
    ser = sp.series(sp.atanh(y), y, 0, 12).removeO()
    assert all(c >= 0 for c in sp.Poly(ser, y).coeffs()) and sp.Poly(ser, y).coeff_monomial(y) == 1
    # atanh(y) = (1/2) log((1 + y)/(1 - y)): at y = 1/kappa, -2 atanh = log((kappa - 1)/(kappa + 1))
    k3 = sp.Symbol("k3", positive=True)
    at = sp.atanh(1 / (1 + k3)).rewrite(sp.log)  # kappa = 1 + k3 > 1
    assert is_zero(sp.simplify(sp.expand_log(2 * at + sp.log(k3 / (2 + k3)), force=True)))
    A = sp.Symbol("A")
    r0, al = sp.symbols("r0 alpha")
    x, r = 0, r0  # x0 = 0; residual r = b - A x
    for _ in range(3):
        x = x + al * r
        r = r - al * A * r
    assert sp.Poly(sp.expand(x / r0), A).degree() == 2  # x_3 = p(A) r0, deg p = 2 < 3


def test_ray_normal_symbol():
    """rf-cost: back-projecting ray sums along every direction, R^T R f(x) = int_0^pi int f(x + t theta) dt d theta,
    is f convolved with 1/|z| (polar coordinates d^2 z = |t| dt d theta), whose 2D transform is 2 pi / |k|
    (int_0^inf 2 pi J0(k r) dr = 2 pi / k): the ray sums' normal operator falls as 1/|k|, and with the beam's profile
    on both sides as T(k)^2 / |k|. A ramp |k| preconditioner leaves T(k)^2: kappa = 1 / T(k_max)^2, 5.9 when the beam's
    FWHM is the dty step (Nyquist period 2 FWHM, T = 0.41), 1.2e3 when it is two steps (T = 0.028); (kappa / 2) ln 1000
    is 20 steps for the first."""
    k, r = sp.symbols("k r", positive=True)
    assert is_zero(sp.integrate(2 * sp.pi * sp.besselj(0, k * r), (r, 0, sp.oo)) - 2 * sp.pi / k)

    def T(lam):  # lim-transfer
        return sp.exp(-(sp.pi**2) / (4 * sp.log(2) * lam**2))

    k1, k2 = 1 / T(2) ** 2, 1 / T(1) ** 2
    assert rounds_to(T(2), "0.41") and rounds_to(k1, "5.9")
    assert rounds_to(T(1), "0.028") and rounds_to(k2 / 1000, "1.2")
    assert rounds_to(k1 / 2 * sp.log(1000), "20")


def test_cost_numbers():
    """rf-cost's example: 3000 x 3000 voxels, 300 reflections each, 16 moment components, forward and adjoint:
    2 * 16 * 9e6 * 300 = 8.6e10 voxel-ray weights per iteration; 20 iterations 1.7e12."""
    per = 2 * 16 * 3000**2 * 300
    assert rounds_to(sp.Integer(per) / 10**10, "8.6") and rounds_to(sp.Integer(20 * per) / 10**12, "1.7")


def test_alternation_rate():
    """rf-runout-fit: alternating least squares on two blocks (the moments Y with the runout rho fixed, then rho with Y
    fixed) never raises the misfit, and near the solution, for one column each (a, b), it contracts the error by
    cos^2 of the angle between them per sweep: y <- a.(t - b r)/a.a, r <- b.(t - a y)/b.b leaves r - r* multiplied
    by (a.b)^2 / (a.a b.b). It stalls (cos = 1) only where the runout's data pattern is one a change of the map can
    make: the first harmonic, a translation (rf-runout)."""
    a = sp.Matrix(sp.symbols("a0:3", real=True))
    b = sp.Matrix(sp.symbols("b0:3", real=True))
    ys, rs = sp.symbols("y_s r_s", real=True)  # the exact solution, t = a ys + b rs
    t = a * ys + b * rs
    r = sp.Symbol("r", real=True)
    y_new = a.dot(t - b * r) / a.dot(a)
    r_new = b.dot(t - a * y_new) / b.dot(b)
    cos2 = a.dot(b) ** 2 / (a.dot(a) * b.dot(b))
    assert is_zero(sp.simplify((r_new - rs) - cos2 * (r - rs)))


def test_runout_identifiable():
    """rf-runout-fit: a population's counts against dty at the omega of each of its reflections are the beam-weighted
    projection of its voxels, whose centroid is y0 - (xbar sin w + ybar cos w) + r(w) (dif-dty; xbar, ybar its
    light's centre): a constant and a first harmonic from the population, plus the runout. So the runout's harmonics
    from the second up are the populations' centroid curves less their own first harmonics, shared by every
    population: determined once the reflections' omegas resolve them (2n + 1 distinct omegas per population for
    harmonics up to n, the harmonics being orthogonal over a full turn)."""
    w = sp.Symbol("omega", real=True)
    xs, ys = sp.symbols("x0:3", real=True), sp.symbols("y0_0:3", real=True)
    f = sp.symbols("f0:3", positive=True)
    y0 = sp.Symbol("y0", real=True)
    M = sum(f)
    centroid = sum(f[i] * (y0 - (xs[i] * sp.sin(w) + ys[i] * sp.cos(w))) for i in range(3)) / M
    xb = sum(f[i] * xs[i] for i in range(3)) / M
    yb = sum(f[i] * ys[i] for i in range(3)) / M
    assert is_zero(sp.simplify(centroid - (y0 - xb * sp.sin(w) - yb * sp.cos(w))))
    for m in range(4):
        for n in range(1, 4):
            if m != n:
                assert sp.integrate(sp.cos(m * w) * sp.cos(n * w), (w, 0, 2 * sp.pi)) == 0
                assert sp.integrate(sp.sin(m * w) * sp.sin(n * w), (w, 0, 2 * sp.pi)) == 0
            assert sp.integrate(sp.cos(m * w) * sp.sin(n * w), (w, 0, 2 * sp.pi)) == 0


def test_box_half_width():
    """rf-box-width: in a box of half-width t cells per axis over three axes, a first moment along one axis sums
    |x| over (2t + 1)^2 t (t + 1) cell-cells, about 4 t^4; with the cut's bound kappa W / N and the light beyond the
    faces' worst case sigma^2 / t (variance only, ref-tail), B(t) = 4 kappa t^4 / N + sigma^2 / t is least at
    t* = (sigma^2 N / (16 kappa))^(1/5), where B = (5/4) sigma^2 / t*: falling only as N^(-1/5). For sigma = 1 cell,
    N = 1e4, kappa = 1/e: t* = 4.43 cells, B = 0.282 cells."""
    t, N, kap, sig = sp.symbols("t N kappa sigma", positive=True)
    W = sum(sp.Abs(x) for x in range(-3, 4)) * 7 * 7
    assert W == 12 * 49 and (2 * 3 + 1) ** 2 * 3 * (3 + 1) == W  # (2t+1)^2 t (t+1) at t = 3
    B = 4 * kap * t**4 / N + sig**2 / t
    ts = (sig**2 * N / (16 * kap)) ** sp.Rational(1, 5)
    assert is_zero(sp.simplify(sp.diff(B, t).subs(t, ts)))
    assert is_zero(sp.simplify(B.subs(t, ts) - sp.Rational(5, 4) * sig**2 / ts))
    vals = {sig: 1, N: 10**4, kap: sp.exp(-1)}
    assert rounds_to(ts.subs(vals), "4.43") and rounds_to((sp.Rational(5, 4) * sig**2 / ts).subs(vals), "0.282")


def test_three_dimensional_weak():
    """rf-3d: in a stack of slices the rays are horizontal, so k = k_h + k_z z is carried by xi = z x k_h-hat for any
    k_z, and the weak part is the row xi of beta (lim-slice). With n = k-hat x xi, the Nye tensor of a plane wave,
    by differentiation, has alpha m = i beta (m x k) for any m; n x k = |k| xi and xi x k = -|k| n, so beta_xixi =
    alpha_xin / (i|k|), beta_xin = -alpha_xixi / (i|k|), and beta_xik enters no entry of alpha. n is z for an in-plane
    k. A frequency along z (k_h = 0) is carried by every horizontal ray, whose g0 span all directions."""
    kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)
    k = sp.Matrix([kx, ky, kz])
    kh = sp.Matrix([kx, ky, 0])
    z = sp.Matrix([0, 0, 1])
    xi = z.cross(kh) / sp.sqrt(kx**2 + ky**2)
    assert is_zero(sp.simplify(xi.dot(k)))  # xi carries every k_z
    kn = sp.sqrt(kx**2 + ky**2 + kz**2)
    khat = k / kn
    n = khat.cross(xi)
    assert is_zero(sp.simplify(n.dot(n) - 1)) and is_zero(sp.simplify(n.dot(k)))
    assert is_zero(sp.simplify(n.cross(k) - kn * xi)) and is_zero(sp.simplify(xi.cross(k) + kn * n))
    # the curl of a plane wave, by differentiation
    x = sp.Matrix(sp.symbols("x0:3", real=True))
    ph = sp.exp(sp.I * k.dot(x))
    B = sp.Matrix(3, 3, sp.symbols("b0:9"))

    def curl(Bm: sp.Matrix) -> sp.Matrix:
        return sp.Matrix(
            3,
            3,
            lambda i, j: sp.simplify(
                sum(sp.LeviCivita(j, a, b) * sp.diff(Bm[i, b] * ph, x[a]) for a in range(3) for b in range(3)) / ph
            ),
        )

    alpha = curl(B)
    m = sp.Matrix(sp.symbols("m0:3", real=True))
    assert is_zero(sp.simplify(alpha * m - sp.I * B * m.cross(k)))
    assert is_zero(sp.simplify((xi.T * B * xi)[0] - (xi.T * alpha * n)[0] / (sp.I * kn)))
    assert is_zero(sp.simplify((xi.T * B * n)[0] + (xi.T * alpha * xi)[0] / (sp.I * kn)))
    t = sp.Symbol("t")
    assert is_zero(sp.simplify(curl(B + t * xi * khat.T) - alpha))
    assert is_zero(sp.simplify(n.subs({kz: 0}) - z))  # horizontal k: n = z (lim-nye)


def test_curl_reciprocal():
    """rf-curl-reciprocal: for beta = eps + [w]x (fields, eps symmetric) and beta* = -beta^T, the curl of beta*'s rows,
    eps_jkl d_k beta*_il, is the Nye tensor alpha_ij = eps_jkl d_k beta_il less twice the curl of eps, by
    differentiation."""
    x = sp.symbols("x0:3", real=True)
    e = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i, 3):
            e[i][j] = e[j][i] = sp.Function(f"e{i}{j}")(*x)
    E = sp.Matrix(e)
    w = sp.Matrix([sp.Function(f"w{i}")(*x) for i in range(3)])
    beta = E + skew(w)
    bstar = -beta.T

    def curl(Bm: sp.Matrix) -> sp.Matrix:
        return sp.Matrix(
            3,
            3,
            lambda i, j: sum(sp.LeviCivita(j, a, b) * sp.diff(Bm[i, b], x[a]) for a in range(3) for b in range(3)),
        )

    assert is_zero(sp.expand(curl(bstar) - (curl(beta) - 2 * curl(E))))


def test_displacement_common_term():
    """rf-displacement: a radially symmetric feature f(rho) of the detector radius (here |g|^2 =
    (2 / lambda^2)(1 - L / sqrt(L^2 + rho^2)) on a detector normal to the beam at distance L) has Hessian eigenvalues
    f'' (radial) and f'/rho (tangential), so a zero-mean displacement of covariance (s_r^2, s_t^2) adds
    (f'' s_r^2 + f' s_t^2 / rho) / 2: relative to f, (s_r^2 + s_t^2) / rho^2 to leading order in rho / L. For a
    point spread of variance 0.5 px^2 per axis at rho = 420 px, 5.7e-6, the same for every population there."""
    rho, L, lam = sp.symbols("rho L lambda", positive=True)
    f = 2 / lam**2 * (1 - L / sp.sqrt(L**2 + rho**2))
    x, y = sp.symbols("x y", real=True)
    F = f.subs(rho, sp.sqrt(x**2 + y**2))
    Hxx = sp.diff(F, x, 2).subs({x: rho, y: 0})  # radial
    Hyy = sp.diff(F, y, 2).subs({x: rho, y: 0})  # tangential
    assert is_zero(sp.simplify(Hxx - sp.diff(f, rho, 2)))
    assert is_zero(sp.simplify(Hyy - sp.diff(f, rho) / rho))
    sr, st, e = sp.symbols("s_r s_t epsilon", positive=True)
    rel = ((sp.diff(f, rho, 2) * sr**2 + sp.diff(f, rho) / rho * st**2) / 2 / f).subs(L, rho / e)
    lead = sp.limit(rel, e, 0)
    assert is_zero(sp.simplify(lead - (sr**2 + st**2) / rho**2))
    assert rounds_to((sp.Rational(1, 2) + sp.Rational(1, 2)) / sp.Integer(420) ** 2 * 10**6, "5.7")


def test_parallax_exact():
    """rf-parallax-exact: a photon leaving a voxel at x_v along the unit u reaches the detector plane n . X = D at
    X = x_v + s u, s = (D - n . x_v) / (n . u), and X = (D / (n . u)) (u + n x (x_v x u) / D): seen from the origin its
    direction is exactly (I + eps A) u normalised, A = [n]x [e]x (e = x_v / |x_v|, eps = |x_v| / D, ||A|| <= 1). To
    first order in eps it is u + eps T(u), T(u) = (I - u u^T) A u, a cubic in u whose expansion about u0 ends at third
    order with the term -delta (delta^T A delta) (the coefficient of t^3 in T(u0 + t delta)). A first-moment feature
    l . g_hat then has, after its second-order expansion, the remainder -eps |k| (l . delta)(delta^T A delta) exactly,
    at most eps |k| |l| |delta|^3, and with mean a third moment of the spread (zero for a symmetric one); |g_hat|^2 the
    same with l = -2 |k| k_in-hat, since u . T(u) = (u^T A u)(1 - |u|^2) = 0 on the sphere. eps |delta|^3 is 3.0e-9 for
    100 um at 180 mm and a 1 deg spread."""
    n = sp.Matrix([0, 0, 1])  # detector normal (any unit n: the identities below hold for symbols too)
    x = sp.Matrix(sp.symbols("x0:3", real=True))
    u = sp.Matrix(sp.symbols("u0:3", real=True))
    D = sp.Symbol("D", positive=True)
    s = (D - n.dot(x)) / n.dot(u)
    X = x + s * u
    assert is_zero(sp.simplify(X - D / n.dot(u) * (u + n.cross(x.cross(u)) / D)))
    assert is_zero(n.cross(x.cross(u)) - skew(n) * skew(x) * u)
    # first order of the normalised (I + eps A) u at |u| = 1
    e = sp.Symbol("eps", positive=True)
    A = sp.Matrix(3, 3, sp.symbols("A0:9", real=True))
    y = (sp.eye(3) + e * A) * u
    Nn = y / sp.sqrt(y.dot(y))
    first = sp.Matrix([sp.diff(c, e).subs(e, 0) for c in Nn]).subs(u.dot(u), 1)
    T = A * u - u * (u.T * A * u)[0]
    norm1 = {u[2]: sp.sqrt(1 - u[0] ** 2 - u[1] ** 2)}
    assert is_zero(sp.simplify((first - T).subs(norm1)))
    # T's expansion about u0 ends at t^3 with -delta (delta^T A delta)
    t = sp.Symbol("t", real=True)
    u0 = sp.Matrix(sp.symbols("p0:3", real=True))
    dl = sp.Matrix(sp.symbols("d0:3", real=True))
    Tt = (A * (u0 + t * dl) - (u0 + t * dl) * ((u0 + t * dl).T * A * (u0 + t * dl))[0]).applyfunc(sp.expand)
    assert all(sp.Poly(c, t).degree() <= 3 for c in Tt)
    c3 = sp.Matrix([sp.Poly(c, t).coeff_monomial(t**3) for c in Tt])
    assert is_zero(c3 + dl * (dl.T * A * dl)[0])
    assert is_zero(sp.expand(u.dot(T) - (u.T * A * u)[0] * (1 - u.dot(u))))
    # ||A|| <= ||[n]x|| ||[e]x|| = 1: [n]x^T [n]x = I - n n^T, eigenvalues 1, 1, 0
    assert is_zero(skew(n).T * skew(n) - (sp.eye(3) - n * n.T))
    val = sp.Rational(100, 180000) * (sp.pi / 180) ** 3
    assert rounds_to(val * 10**9, "3.0")


def test_regions_forced():
    """rf-forced: a statistic over a fixed region is the integral of K_R over a population's distribution. At an edge
    of the instrument's kernel (K_R = Phi(x / w)), one point at the mean c and two points c +- e (same mass, same mean)
    give different light for c != 0: the light depends on the shape unless K_R is constant on the support, i.e. the
    region holds all the light (K_R = 1 on the support), where the integral is the mass for any distribution."""
    c, e, w = sp.Rational(1, 2), sp.Rational(3, 4), sp.Integer(1)

    def Phi(x: sp.Expr) -> sp.Expr:
        return (1 + sp.erf(x / sp.sqrt(2))) / 2

    diff = Phi(c / w) - (Phi((c + e) / w) + Phi((c - e) / w)) / 2
    assert abs(float(sp.N(diff, 30))) > 1e-3
    # c = 0: the two agree, so the difference is a matter of the shape relative to the edge
    assert sp.simplify(Phi(0) - (Phi(e / w) + Phi(-e / w)) / 2) == 0


def test_cell_linearisation():
    """rf-cell-linear: K(mu) = Phi((b - mu)/w) - Phi((a - mu)/w) has K'' = (phi'(z_b) - phi'(z_a)) / w^2 and
    |phi'(z)| = |z| phi(z) <= phi(1), so |K''| <= 2 phi(1) / w^2 and Taylor's remainder is at most phi(1) (delta/w)^2:
    2.4e-3 at delta = w / 10; cells of half-width w / 10 at w = 0.01 deg number 1.25e5 over 0.1 deg, 1.25e8 over 1."""
    mu, a, b = sp.symbols("mu a b", real=True)
    w = sp.Symbol("w", positive=True)
    z = sp.Symbol("z", real=True)
    phi = sp.exp(-(z**2) / 2) / sp.sqrt(2 * sp.pi)

    def Phi(x: sp.Expr) -> sp.Expr:
        return (1 + sp.erf(x / sp.sqrt(2))) / 2

    K = Phi((b - mu) / w) - Phi((a - mu) / w)
    dphi = sp.diff(phi, z)
    K2 = (dphi.subs(z, (b - mu) / w) - dphi.subs(z, (a - mu) / w)) / w**2
    assert sp.simplify(sp.diff(K, mu, 2) - K2) == 0
    assert sp.simplify(dphi + z * phi) == 0
    crit = sp.solve(sp.diff(z * phi, z), z)
    assert set(crit) == {-1, 1}  # |z phi| is largest at |z| = 1
    assert rounds_to(phi.subs(z, 1) * sp.Rational(1, 100), "0.0024")
    assert rounds_to((sp.Rational(1, 10) / (2 * sp.Rational(1, 1000))) ** 3 / 10**5, "1.25")
    assert rounds_to((sp.Integer(1) / (2 * sp.Rational(1, 1000))) ** 3 / 10**8, "1.25")


def test_sandwich():
    """rf-sandwich: for R0 within R^ within R1, sum over R^ minus sum over R0 is the sum over R^ \\ R0, a subset of the
    band R1 \\ R0, so it is at most the band's sum of |r f| (the triangle inequality): checked exactly for every R^ on
    a strip of 6 cells with a band of 4. The bound holds for every realisation, so it holds in expectation, however R^
    depends on the counts, and the band is a fixed region whose expected sum the data measure."""
    import itertools

    vals = [sp.Rational(v) for v in ("3/2", "-2", "1/3", "5", "-7/4", "2")]
    R0, band = {1, 2}, [0, 3, 4, 5]
    bound = sum(abs(vals[p]) for p in band)
    for n in range(len(band) + 1):
        for extra in itertools.combinations(band, n):
            Rh = R0 | set(extra)
            assert abs(sum(vals[p] for p in Rh) - sum(vals[p] for p in R0)) <= bound


# ---------------------------------------------------------------------------------------------------------------
# The beam's weight within a peak (rf-beam-motion, rf-beam-taylor, rf-beam-gauss, rf-beam-closure, rf-beam-first)


def test_beam_motion():
    """rf-beam-motion: a voxel at (x, y) in the sample sits at dty_v(w) = y0 - (x sin w + y cos w). Over a further turn
    d its dty moves by -zeta sin d - eta (cos d - 1), with (zeta, eta) its lab position along and across the beam at
    w; that move is the y-component of R_z(d) p - p, so its size is at most the chord 2 r |sin(d/2)| <= r |d|; and
    d(dty_v)/dw = -zeta, d^2(dty_v)/dw^2 = eta, both at most r in size."""
    x, y, y0, w, d = sp.symbols("x y y0 omega delta", real=True)
    dty = y0 - (x * sp.sin(w) + y * sp.cos(w))
    lab = rz(w) * sp.Matrix([x, y, 0])
    zeta, eta = lab[0], lab[1]
    move = dty.subs(w, w + d) - dty
    assert is_zero(sp.expand_trig(move - (-zeta * sp.sin(d) - eta * (sp.cos(d) - 1))))
    chord = rz(d) * lab - lab
    assert is_zero(sp.expand_trig(move + chord[1]))
    assert is_zero(sp.expand_trig(chord.dot(chord) - 4 * (x**2 + y**2) * sp.sin(d / 2) ** 2))
    # |sin(d/2)| <= |d|/2: d/2 - sin(d/2) vanishes at 0 and has the derivative (1 - cos(d/2))/2 >= 0
    assert is_zero(sp.diff(d / 2 - sp.sin(d / 2), d) - (1 - sp.cos(d / 2)) / 2)
    assert is_zero(sp.diff(dty, w) + zeta) and is_zero(sp.diff(dty, w, 2) - eta)
    assert is_zero(zeta**2 + eta**2 - (x**2 + y**2))


def test_beam_taylor():
    """rf-beam-taylor: the weight w(d) = P(u(d)) of a photon diffracting a turn d past the population's mean omega,
    with u = dty_b - dty_v the voxel's offset from the row along the scan's path, has w'' = P''(u) u'^2 + P'(u) u'',
    with |u'| <= r + |r_f| and |u''| <= r (rf-beam-motion; dty_b linear in omega within the path, int-dty-motion);
    Taylor's remainder is then at most (||P''|| (r + |r_f|)^2 + ||P'|| r) d^2 / 2. For a Gaussian profile of peak 1,
    ||P'|| = e^(-1/2)/sigma (at |u| = sigma) and ||P''|| = 1/sigma^2 (at u = 0; the other critical points, u^2 =
    3 sigma^2, give 2 e^(-3/2)/sigma^2, less)."""
    d, s = sp.symbols("delta sigma", positive=True)
    u = sp.Function("u")(d)
    P = sp.Function("P")
    w = P(u)
    q = sp.Symbol("q", real=True)
    P1, P2 = (sp.Subs(sp.Derivative(P(q), q, n), q, u) for n in (1, 2))
    assert is_zero((sp.diff(w, d, 2) - (P2 * sp.diff(u, d) ** 2 + P1 * sp.diff(u, d, 2))).doit())
    G = sp.exp(-(q**2) / (2 * s**2))
    G1, G2 = sp.diff(G, q), sp.diff(G, q, 2)
    assert set(sp.solve(sp.diff(G1, q), q)) == {-s, s}
    assert is_zero(sp.Abs(G1.subs(q, s)) - sp.exp(-sp.Rational(1, 2)) / s)
    assert set(sp.solve(sp.diff(G2, q), q)) == {0, -sp.sqrt(3) * s, sp.sqrt(3) * s}
    assert is_zero(sp.Abs(G2.subs(q, 0)) - 1 / s**2)
    assert is_zero(G2.subs(q, sp.sqrt(3) * s) - 2 * sp.exp(-sp.Rational(3, 2)) / s**2)
    assert 2 * sp.exp(-sp.Rational(3, 2)) < 1
    assert sp.limit(G2, q, sp.oo) == 0 and sp.limit(G1, q, sp.oo) == 0


def test_beam_gauss():
    """rf-beam-gauss: a Gaussian profile exp(-u^2 / 2 sigma^2), linear motion u = u0 + c d and a Gaussian omega offset
    d ~ N(0, s^2): the light kept is sigma / sqrt(sigma^2 + c^2 s^2) exp(-u0^2 / 2 (sigma^2 + c^2 s^2)) (the beam
    widened by c s), and its mean offset is -u0 c s^2 / (sigma^2 + c^2 s^2). This is int-fold with beta = -u0 c /
    sigma^2 and gamma = -c^2 / sigma^2. The page's numbers: FWHM 1 um, c = 200 um, s = 0.1 deg, u0 = sigma."""
    d = sp.Symbol("delta", real=True)
    u0, c = sp.symbols("u0 c", real=True)
    sg, s = sp.symbols("sigma s", positive=True)
    N = sp.exp(-(d**2) / (2 * s**2)) / sp.sqrt(2 * sp.pi * s**2)
    wgt = sp.exp(-((u0 + c * d) ** 2) / (2 * sg**2))
    kept = sp.integrate(N * wgt, (d, -sp.oo, sp.oo), conds="none").rewrite(sp.erf)  # erf + erfc = 1
    S2 = sg**2 + c**2 * s**2
    ratio = sp.simplify(kept / (sg / sp.sqrt(S2) * sp.exp(-(u0**2) / (2 * S2))))
    assert is_zero(sp.factor(sp.expand_log(sp.log(ratio), force=True)))
    first = sp.integrate(d * N * wgt, (d, -sp.oo, sp.oo), conds="none").rewrite(sp.erf)
    assert is_zero(sp.simplify(first / kept + u0 * c * s**2 / S2))
    beta, gamma = -u0 * c / sg**2, -(c**2) / sg**2  # int-fold's log-quadratic weight about d = 0
    assert is_zero(sp.expand(sp.log(wgt) - (-(u0**2) / (2 * sg**2) + beta * d + gamma * d**2 / 2)))
    Sp = 1 / (1 / s**2 - gamma)
    assert is_zero(sp.simplify(Sp * beta + u0 * c * s**2 / S2))
    assert is_zero(sp.simplify(sp.sqrt(Sp / s**2) - sg / sp.sqrt(S2)))
    sig = sp.Rational(1) / (2 * sp.sqrt(2 * sp.log(2)))  # um
    kap = 200 * sp.pi / 1800 / sig
    assert rounds_to(kap, "0.82")
    assert rounds_to(kap * (sp.pi / 1800) / (1 + kap**2) * 10**4, "8.6")  # rad, at u0 = sigma


def test_beam_closure():
    """rf-beam-closure: two omega distributions with the same mass, mean and variance, A (+-s, 1/2 each) and B (0 with
    1 - q, +-s/sqrt(q) with q/2 each), lit by a Gaussian beam centred on the voxel (u0 = 0, linear motion, kappa =
    c s / sigma): A keeps exp(-kappa^2 / 2), B keeps 1 - q + q exp(-kappa^2 / 2q). They agree to kappa^2 and differ
    by (1/q - 1) kappa^4 / 8 at the fourth order; at q = 1/4, kappa = 1, 0.607 against 0.784."""
    k, q = sp.symbols("kappa q", positive=True)
    A = sp.exp(-(k**2) / 2)
    B = 1 - q + q * sp.exp(-(k**2) / (2 * q))
    for m in range(3):  # equal moments of the omega offset, in units of s: 1, 0, 1
        mA = sp.Rational(1, 2) * (1 + (-1) ** m)
        mB = (1 - q) * (1 if m == 0 else 0) + q / 2 * (q ** -sp.Rational(m, 2)) * (1 + (-1) ** m)
        assert is_zero(mA - mB)
    ser = sp.series(B - A, k, 0, 6).removeO()
    assert is_zero(sp.expand(ser - (1 / q - 1) * k**4 / 8))
    assert rounds_to(A.subs(k, 1), "0.607") and rounds_to(B.subs({k: 1, q: sp.Rational(1, 4)}), "0.784")


def test_beam_first_order_linear():
    """rf-beam-first: with the omega offset to first order d = c . (F - I) g0 (c the gradient of the diffraction
    omega with g, known from the geometry), the first-order beam term of a centred first moment, sum_i V_i d_i
    (F_i g0 - g_R), is linear in V, Y1 and Y2 (three sub-volumes, symbolic); and it is bounded by V sd(d) sd(f) when
    the offsets have mean zero (Cauchy-Schwarz, as the covariance)."""
    g0 = sp.Matrix(sp.symbols("g0:3", real=True))
    gR = sp.Matrix(sp.symbols("gR0:3", real=True))
    cv = sp.Matrix(sp.symbols("c0:3", real=True))
    F = [_mat(f"F{i}_") for i in range(3)]
    V = sp.symbols("V0:3", positive=True)
    dl = [(cv.T * (F[i] - sp.eye(3)) * g0)[0] for i in range(3)]
    lhs = sum((V[i] * dl[i] * (F[i] * g0 - gR) for i in range(3)), sp.zeros(3, 1))
    Y0 = sum(V)
    Y1 = sum((V[i] * F[i] for i in range(3)), sp.zeros(3))
    Y2 = {(a, k, b, m): sum(V[i] * F[i][a, k] * F[i][b, m] for i in range(3)) for a, k, b, m in
          itertools.product(range(3), repeat=4)}  # fmt: skip
    for b in range(3):
        rhs = sum(cv[a] * Y2[a, k, b, m] * g0[k] * g0[m] for a in range(3) for k in range(3) for m in range(3))
        rhs -= sum(cv[a] * g0[a] * Y1[b, m] * g0[m] for a in range(3) for m in range(3))
        rhs -= gR[b] * (cv.T * (Y1 - Y0 * sp.eye(3)) * g0)[0]
        assert is_zero(lhs[b] - rhs)
    # the covariance bound: (E[d f])^2 <= E[d^2] E[f^2], here as the Gram determinant of two vectors
    a = sp.Matrix(sp.symbols("a0:3", real=True))
    b = sp.Matrix(sp.symbols("b0:3", real=True))
    assert is_zero(a.dot(a) * b.dot(b) - a.dot(b) ** 2 - a.cross(b).dot(a.cross(b)))


def test_beam_numbers():
    """rf-beam-taylor's numbers: kappa = (r + |r_f|) s / sigma and the first-order bound e^(-1/2) kappa s for the
    rotation about the axis, FWHM 1 um, r = 200 um, r_f = 0, s = 0.01, 0.03, 0.1 deg; and the largest s for which
    that bound stays below 1e-4 rad at r = 50, 200 and 500 um."""
    sig = sp.Rational(1) / (2 * sp.sqrt(2 * sp.log(2)))
    deg = sp.pi / 180
    for sd, kap, bias in (("0.01", "0.082", "8.7e-6"), ("0.03", "0.25", "7.8e-5"), ("0.1", "0.82", "8.7e-4")):
        s = sp.Rational(sd) * deg
        k = 200 * s / sig
        assert rounds_to(k, kap)
        b = sp.exp(-sp.Rational(1, 2)) * k * s
        mant, ex = bias.split("e")
        assert rounds_to(b * sp.Integer(10) ** (-int(ex)), mant)
    for r, smax in ((50, "0.068"), (200, "0.034"), (500, "0.021")):
        s = sp.sqrt(sp.Rational(1, 10**4) * sig / (sp.exp(-sp.Rational(1, 2)) * r))
        assert rounds_to(s / deg, smax)


def test_beam_misplace():
    """rf-beam-misplace: the weight at the photon's own omega, P(dty_b - dty_v(w + d)), is the profile at the mean
    omega with the voxel moved by m(d) = -zeta sin d - eta (cos d - 1) (rf-beam-motion). The rows' sum of a Gaussian
    profile, sum_n exp(-(n - y)^2 / 2 sigma^2), is by Poisson summation sqrt(2 pi) sigma (1 + 2 sum_k
    exp(-2 pi^2 sigma^2 k^2) cos 2 pi k y): its first harmonic's relative amplitude is 2 T(1) of lim-transfer, 0.057 at
    sigma = 1 / (2 sqrt(2 ln 2)) steps (FWHM one step)."""
    x, y, y0, w, d, b = sp.symbols("x y y0 omega delta dty_b", real=True)
    dty = y0 - (x * sp.sin(w) + y * sp.cos(w))
    lab = rz(w) * sp.Matrix([x, y, 0])
    m = -lab[0] * sp.sin(d) - lab[1] * (sp.cos(d) - 1)
    assert is_zero(sp.expand_trig((b - dty.subs(w, w + d)) - (b - dty - m)))
    sg = sp.Symbol("sigma", positive=True)
    t = sp.Symbol("t", real=True)
    k = sp.Symbol("k", integer=True, positive=True)
    # Fourier coefficient of the Gaussian at frequency k (per step): its transform at 2 pi k
    ft = sp.integrate(sp.exp(-(t**2) / (2 * sg**2)) * sp.exp(-2 * sp.pi * sp.I * k * t), (t, -sp.oo, sp.oo))
    assert is_zero(sp.simplify(ft - sp.sqrt(2 * sp.pi) * sg * sp.exp(-2 * sp.pi**2 * sg**2 * k**2)))
    s1 = 1 / (2 * sp.sqrt(2 * sp.log(2)))
    T1 = sp.exp(-(sp.pi**2) / (4 * sp.log(2)))  # lim-transfer at Lambda = 1
    assert is_zero(sp.simplify(sp.exp(-2 * sp.pi**2 * s1**2) - T1))
    assert rounds_to(2 * T1, "0.057")


def test_beam_edge():
    """rf-beam-edge: kappa = R s / sigma_b with R = N/2 steps and sigma_b = c / (2 sqrt(2 ln 2)) steps is
    N sqrt(2 ln 2) s / c = 1.18 N s / c; 3532 s at N = 3000, c = 1."""
    N, c, s = sp.symbols("N c s", positive=True)
    kap = (N / 2) * s / (c / (2 * sp.sqrt(2 * sp.log(2))))
    assert is_zero(kap - N * sp.sqrt(2 * sp.log(2)) * s / c)
    assert rounds_to(sp.sqrt(2 * sp.log(2)), "1.18")
    assert rounds_to(3000 * sp.sqrt(2 * sp.log(2)), "3532")


def test_beam_voxel():
    """rf-beam-voxel: r s <= step is kappa = r s / sigma_b <= step / sigma_b = 2 sqrt(2 ln 2) / c (2.35 at c = 1),
    and at r = rho N / 2 steps it is s <= 2 / (N rho); the numbers on a 3000-step map, and the +-6.5 rho steps of a
    voxel split equally between parts 0.5 deg apart in omega (s = 0.25 deg)."""
    r, s, sg, c, N, rho = sp.symbols("r s sigma_b c N rho", positive=True)
    assert is_zero((r * s / sg) / (1 / sg) - r * s)  # kappa <= 1/sigma_b (step = 1) is r s <= 1
    assert is_zero(1 / (c / (2 * sp.sqrt(2 * sp.log(2)))) - 2 * sp.sqrt(2 * sp.log(2)) / c)
    assert rounds_to(2 * sp.sqrt(2 * sp.log(2)), "2.35")
    assert is_zero(sp.solve(sp.Eq(rho * N / 2 * s, 1), s)[0] - 2 / (N * rho))
    deg = sp.pi / 180
    for rh, want in (
        (1, "0.038"),
        (sp.Rational(1, 2), "0.076"),
        (sp.Rational(1, 4), "0.15"),
        (sp.Rational(1, 10), "0.38"),
    ):
        assert rounds_to(2 / (3000 * rh) / deg, want)
    # two equal parts at +-0.25 deg: sd 0.25 deg; move r s in steps at r = rho N / 2
    a = sp.Rational(1, 4) * deg
    assert is_zero(sp.sqrt((a**2 + a**2) / 2) - a)
    assert rounds_to(1500 * a, "6.5")


def test_trace_residual():
    """rf-trace-residual: with dty_v(w) = y0 - (x sin w + y cos w), the photon's misplacement against the trace,
    e = dty_k(w) - n step - y~_k with y~_k = dty_k(wk) - (y_ref(wk) - y_ref(wR)), is exactly the three terms; the
    first is the dty move (rf-beam-motion, without y0) of the offset x_k - x_ref over the turn w - wk, so at most
    |x_k - x_ref| |w - wk| by the chord; the second has the trace's rate d y_ref / dw = -zeta_ref, at most r_ref in
    size, so it is at most r_ref |w - w_j| <= r_ref h / 2 (a fly scan adds r_f to the rate)."""
    x, y, xr, yr, y0, w, wk, wj, wR, n, st = sp.symbols(
        "x y x_r y_r y0 omega omega_k omega_j omega_R n step", real=True
    )

    def dty(px, py, t):
        return y0 - (px * sp.sin(t) + py * sp.cos(t))

    yt = dty(x, y, wk) - (dty(xr, yr, wk) - dty(xr, yr, wR))
    e = dty(x, y, w) - n * st - yt
    t1 = (dty(x, y, w) - dty(xr, yr, w)) - (dty(x, y, wk) - dty(xr, yr, wk))
    t2 = dty(xr, yr, w) - dty(xr, yr, wj)
    t3 = dty(xr, yr, wj) - dty(xr, yr, wR) - n * st
    assert is_zero(e - (t1 + t2 + t3))
    dx, dy = x - xr, y - yr
    assert is_zero(sp.expand(t1 - (dty(dx, dy, w) - dty(dx, dy, wk))))  # y0 cancels: the offset's own move
    d = sp.Symbol("delta", real=True)
    lab = rz(wk) * sp.Matrix([dx, dy, 0])
    chord = rz(d) * lab - lab
    assert is_zero(sp.expand_trig(t1.subs(w, wk + d) + chord[1]))
    assert is_zero(sp.expand_trig(chord.dot(chord) - 4 * (dx**2 + dy**2) * sp.sin(d / 2) ** 2))
    rate = sp.diff(dty(xr, yr, w), w)
    assert is_zero(rate + (rz(w) * sp.Matrix([xr, yr, 0]))[0])


def test_trace_numbers():
    """rf-trace's numbers at the edge of a 3000-step map (r = 1500 steps), 0.05 deg frames: the frame's term r h / 2 =
    0.65 steps, its rms r h / sqrt(12) = 0.38 (uniform within the frame); the rounding's rms step / sqrt(12) = 0.29;
    the first term's rms (D / 2) s at D = 9, s = 0.25 deg, 0.020 steps; and kappa = r h / sqrt(12) / sigma_b = 0.89
    with sigma_b = 1 / (2 sqrt(2 ln 2)) steps."""
    deg = sp.pi / 180
    h = sp.Rational(5, 100) * deg
    assert rounds_to(1500 * h / 2, "0.65") and rounds_to(1500 * h / sp.sqrt(12), "0.38")
    u = sp.Symbol("u", real=True)
    assert is_zero(sp.integrate(u**2, (u, -sp.Rational(1, 2), sp.Rational(1, 2))) - sp.Rational(1, 12))
    assert rounds_to(1 / sp.sqrt(12), "0.29")
    assert rounds_to(sp.Rational(9, 2) * sp.Rational(1, 4) * deg, "0.020")
    sig = 1 / (2 * sp.sqrt(2 * sp.log(2)))
    assert rounds_to(1500 * h / sp.sqrt(12) / sig, "0.89")
