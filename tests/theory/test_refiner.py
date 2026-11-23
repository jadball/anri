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
    |d eta| <= e sqrt(1 + tan^2 theta / sin^2 eta), attained for delta along j; a strain |eps| <= eps_max moves 2 theta
    by at most 2 eps_max tan theta and leaves omega and eta (Delta 2theta = -2 eps tan theta, from Bragg's law)."""
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
    """rf-split: at an in-plane frequency k (xi = z x k), the in-plane columns of Z_hat split by k k^T + xi xi^T = I_2
    into orthogonal idempotent parts; Z_hat xi |k| = (k_x Z_y - k_y Z_x)_hat, the Fourier transform of i times
    (curl Z)_z per row (lim-nye): the weak part is each row's solenoidal (divergence-free) in-plane field, the seen part
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
