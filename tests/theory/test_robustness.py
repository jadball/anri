"""Proofs for docs/source/theory/robustness.rst: what a fitted statistic is robust to (shape errors, the cut, the
pixels, a population's spread) and what splitting merged light costs."""

from __future__ import annotations

import numpy as np
import sympy as sp

from tests.theory.common import EPS, animals, boundary, is_zero, rounds_to, skew


def _rational(rows: int, cols: int, seed: int) -> sp.Matrix:
    return sp.Matrix(rows, cols, lambda i, j: sp.Rational((7 * i + 3 * j + seed) % 11 - 5, (i + 2 * j + seed) % 5 + 1))


def _spd(n: int, seed: int) -> sp.Matrix:
    L = sp.Matrix(n, n, lambda i, j: sp.Rational((3 * i + 5 * j + seed) % 7 + 1, 3) if j <= i else 0)
    return L * L.T


def test_fitted_statistic_bias():
    """ref-bias: a statistic t with E[t] = tau(theta0) + Delta, fitted by weighted least squares with any SPD weight W
    (linear tau, exact; a nonlinear one to first order, by Gauss-Newton about theta0): the estimate's mean is
    theta0 + (G^T W G)^-1 G^T W Delta and its covariance the sandwich, (G^T C^-1 G)^-1 at W = C^-1. With Delta = 0 every
    W gives the unbiased estimate."""
    G = _rational(6, 3, 1)
    W, C = _spd(6, 2), _spd(6, 5)
    theta0 = sp.Matrix(sp.symbols("th0:3"))
    tau0 = sp.Matrix(sp.symbols("tau0:6"))
    Delta = sp.Matrix(sp.symbols("D0:6"))
    A = (G.T * W * G).inv() * G.T * W  # theta_hat = theta0 + A (t - tau(theta0)), tau = tau0 + G (theta - theta0)
    mean = theta0 + A * ((tau0 + Delta) - tau0)  # E[t - tau(theta0)] = Delta
    assert is_zero(mean - theta0 - (G.T * W * G).inv() * G.T * W * Delta)
    assert is_zero((A * G) - sp.eye(3))  # the linear estimator reproduces any theta: unbiased when Delta = 0
    cov = A * C * A.T
    assert is_zero(cov - ((G.T * W * G).inv() * G.T * W * C * W * G * (G.T * W * G).inv()))
    Ci = C.inv()
    assert is_zero((G.T * Ci * G).inv() * G.T * Ci * C * Ci * G * (G.T * Ci * G).inv() - (G.T * Ci * G).inv())


def test_profiled_bias():
    """ref-profile: with nuisance columns G_eta fitted too, the bias on theta is (G_th^T W P G_th)^-1 G_th^T W P Delta,
    P = I - G_eta (G_eta^T W G_eta)^-1 G_eta^T W, so an error Delta = G_eta u (one that looks like a change of the
    nuisance) leaves theta unbiased, whatever u."""
    Gt, Ge = _rational(7, 2, 3), _rational(7, 2, 4)
    W = _spd(7, 1)
    G = Gt.row_join(Ge)
    u = sp.Matrix(sp.symbols("u0:2"))
    v = sp.Matrix(sp.symbols("v0:7"))
    Delta = Ge * u + v
    full = ((G.T * W * G).inv() * G.T * W * Delta)[:2, :]
    P = sp.eye(7) - Ge * (Ge.T * W * Ge).inv() * Ge.T * W
    prof = (Gt.T * W * P * Gt).inv() * Gt.T * W * P * Delta
    assert is_zero(full - prof)
    assert is_zero(prof.subs({vi: 0 for vi in v}))


def _reflect(q: tuple, shape: tuple) -> tuple:
    return (shape[0] - 1 - q[0], shape[1] - 1 - q[1])


def test_group_rule_is_symmetric():
    """ref-symmetric: the cut and the group rule commute with a point reflection sigma of the pixel lattice that maps
    the detector to itself. P(|G_p| = k | p kept) (rec-animal) is a sum over the k-animals S of p of
    prod_{S \\ p} a prod_{dS} (1 - a); sigma maps 8-neighbours to 8-neighbours, so it maps the animals of p one to one
    onto those of sigma(p), and each one's boundary onto its image's. Checked exactly on a 4 x 5 detector (reflected
    about its centre, a pixel's edge mid-point) and a 5 x 5 (about a pixel's centre), every pixel, k = 1 .. 4: then
    P_k(sigma(p); a) = P_k(p; a o sigma), and light symmetric about the centre gives symmetric expected recorded values
    (rec-expected) for any m."""
    for shape in ((4, 5), (5, 5)):
        on = {(i, j) for i in range(shape[0]) for j in range(shape[1])}
        for p in on:
            q = _reflect(p, shape)
            for k in (1, 2, 3, 4):
                mine = {S for S in animals(p, k) if S <= on}
                theirs = {S for S in animals(q, k) if S <= on}
                image = {frozenset(_reflect(r, shape) for r in S) for S in mine}
                assert image == theirs
                for S in mine:
                    assert {_reflect(r, shape) for r in boundary(S) & on} == boundary(
                        frozenset(_reflect(r, shape) for r in S)
                    ) & on


def test_symmetric_estimating_equation():
    """ref-symmetric: expected recorded values symmetric about a lattice point c make the expected estimating
    function of any odd score vanish at c: the centroid over a region symmetric about c, and the Poisson (or censored)
    score of any model even about c (a sympy function G of (x - c)^2), on 1D cells at c + k, k = -3 .. 3, and on cell
    edges, c + k + 1/2."""
    c = sp.Symbol("c", real=True)
    Gf = sp.Function("G")
    h = sp.Function("h")  # any function of the model's light per cell (the censored ratio, rec-likelihood's terms)
    mu = sp.Symbol("mu", real=True)
    for offs in ([sp.Integer(k) for k in range(-3, 4)], [sp.Rational(2 * k + 1, 2) for k in range(-3, 3)]):
        t = {o: sp.Symbol(f"t_{abs(o)}") for o in offs}  # true expected recorded value, symmetric
        xs = {o: c + o for o in offs}
        centroid = sum(t[o] * (xs[o] - c) for o in offs)
        assert is_zero(centroid)
        lam = {o: Gf((xs[o] - mu) ** 2) for o in offs}
        score = sum((t[o] / lam[o] - 1) * sp.diff(lam[o], mu) + h(lam[o]) * sp.diff(lam[o], mu) for o in offs)
        assert is_zero(sp.simplify(score.subs(mu, c).doit()))


def _binned_mean_bias(cdf, mu, n: int = 40) -> float:
    """sum_i i m_i - mu for a spot's masses m_i over cells [i - 1/2, i + 1/2), the spot at mu (mpmath)."""
    import mpmath as mp

    i = range(-n, n + 1)
    m = [cdf(k + mp.mpf(1) / 2 - mu) - cdf(k - mp.mpf(1) / 2 - mu) for k in i]
    return sum(k * w for k, w in zip(i, m)) / sum(m) - mu


def test_alias_any_symmetric_spot():
    """ref-alias: for any spot symmetric about mu (characteristic function chi, real), the binned mean is
    mu + sum_k (-1)^k chi(2 pi k) sin(2 pi k mu) / (pi k): a core-and-wing mixture and a Laplace spot, against their
    cells' masses at 30 digits. Over a uniform phase mu its mean is 0 and its mean square sum_k chi_k^2 / (2 pi^2 k^2)
    (the sines are orthogonal)."""
    import mpmath as mp

    mp.mp.dps = 20
    f, sc, sw = mp.mpf("0.28"), mp.mpf("0.25"), mp.mpf("1.3")
    spots = [
        (lambda x: (1 - f) * mp.ncdf(x / sc) + f * mp.ncdf(x / sw),
         lambda w: (1 - f) * mp.exp(-(w * sc) ** 2 / 2) + f * mp.exp(-(w * sw) ** 2 / 2)),
        (lambda x: mp.mpf(1) / 2 + mp.sign(x) * (1 - mp.exp(-abs(x) / mp.mpf("0.3"))) / 2,
         lambda w: 1 / (1 + (w * mp.mpf("0.3")) ** 2)),
    ]  # fmt: skip
    for cdf, chi in spots:
        for mu in ("0.1", "0.25", "0.37"):
            m = mp.mpf(mu)
            # terms fall as chi(2 pi k) / k: below 1 / (35 k^3) for the Laplace, so 4000 terms leave < 1e-12
            series = sum(
                (-1) ** k * chi(2 * mp.pi * k) * mp.sin(2 * mp.pi * k * m) / (mp.pi * k) for k in range(1, 4001)
            )
            assert abs(_binned_mean_bias(cdf, m, 200) - series) < mp.mpf("1e-10")  # Laplace tails: cells to +-200
    u = sp.Symbol("u", real=True)
    assert sp.integrate(sp.sin(2 * sp.pi * 3 * u), (u, 0, 1)) == 0
    assert sp.integrate(sp.sin(2 * sp.pi * 3 * u) * sp.sin(2 * sp.pi * 5 * u), (u, 0, 1)) == 0
    assert sp.integrate(sp.sin(2 * sp.pi * 3 * u) ** 2, (u, 0, 1)) == sp.Rational(1, 2)


def test_alias_numbers():
    """ref-alias's numbers: the core-and-wing point spread (0.25 px holding 72 %, 1.3 px holding 28 %) has
    chi(2 pi) = 0.210; rendered instead as one Gaussian of its variance (sd 0.720 px, chi = 3.6e-5) the binned mean is
    wrong by up to 0.0667 px (near a quarter pixel), 0.0472 px rms over the phase; the bound sum_k |d chi_k| / (pi k) is
    0.0676."""
    f, sc, sw = sp.Rational(28, 100), sp.Rational(1, 4), sp.Rational(13, 10)
    var = (1 - f) * sc**2 + f * sw**2
    w = sp.Symbol("w", positive=True)
    chi_t = (1 - f) * sp.exp(-((w * sc) ** 2) / 2) + f * sp.exp(-((w * sw) ** 2) / 2)
    chi_m = sp.exp(-(w**2) * var / 2)
    d = [sp.N((chi_t - chi_m).subs(w, 2 * sp.pi * k), 30) for k in range(1, 8)]
    assert rounds_to(sp.sqrt(var), "0.720") and rounds_to(chi_t.subs(w, 2 * sp.pi), "0.210")
    assert rounds_to(chi_m.subs(w, 2 * sp.pi) * 10**5, "3.6")
    assert rounds_to(sp.sqrt(sum(dk**2 / (2 * sp.pi**2 * k**2) for k, dk in enumerate(d, 1))), "0.0472")
    assert rounds_to(sum(abs(dk) / (sp.pi * k) for k, dk in enumerate(d, 1)), "0.0676")
    rad = sp.Rational(667, 10000) * 75 / 180000  # 0.0667 of a 75 um pixel at 180 mm, in 2 theta
    assert rounds_to(rad * 10**5, "2.8") and rounds_to(rad / (2 * sp.Rational(8, 100)) * 10**4, "1.7")
    assert rounds_to(sp.exp(-2 * sp.pi**2 * sp.Rational(1, 4)), "0.0072")  # a spot of sd 0.5 cell
    # the largest over the phase (k > 7 adds < 1e-40): a root of the bias's derivative, to 30 digits, from the best
    # of a coarse table of the phase; the bias is odd about 0 and 1/2, so the extremes are within [0, 1/2]
    m = sp.Symbol("m", real=True)
    a = sum((-1) ** k * d[k - 1] * sp.sin(2 * sp.pi * k * m) / (sp.pi * k) for k in range(1, 8))
    start = max((sp.Rational(i, 200) for i in range(101)), key=lambda x: abs(sp.N(a.subs(m, x), 15)))
    m0 = sp.nsolve(sp.diff(a, m), m, start, prec=30)
    assert rounds_to(abs(a.subs(m, m0)), "0.0667") and 0.2 < m0 % sp.Rational(1, 2) < 0.3


def _rodrigues2(dvec: sp.Matrix, t: sp.Symbol) -> sp.Matrix:
    """exp([t d]x) to second order in t, from Rodrigues' formula expanded (not assumed)."""
    th = sp.sqrt(dvec.dot(dvec))
    K = skew(dvec / th)
    R = sp.eye(3) + sp.sin(t * th) * K + (1 - sp.cos(t * th)) * K * K
    return R.applyfunc(lambda e: sp.series(e, t, 0, 3).removeO())


def _expect(expr: sp.Expr, d: sp.Matrix, S: sp.Matrix) -> sp.Expr:
    """E over a zero-mean d with covariance S of a polynomial of degree <= 2 in d (no other moments needed)."""
    p = sp.Poly(sp.expand(expr), *d)
    out = 0
    for mono, coef in p.terms():
        idx = [i for i, n in enumerate(mono) for _ in range(n)]
        if len(idx) == 0:
            out += coef
        elif len(idx) == 2:
            out += coef * S[idx[0], idx[1]]
    return out


def test_spread_mean():
    """ref-spread-mean: a population's lattice turned by exp([d]x), d of mean 0 and covariance S (any distribution):
    E[R] = I + (S - tr S I) / 2 to second order (Rodrigues' formula expanded; the third order term has a third moment
    of d, zero for a symmetric spread). So the mean of g = R g0 is g0 + (S g0 - tr S g0) / 2: along g0 it changes by
    -(tr S - g^T S g) / 2 |g0| (an apparent strain), across it by (I - g g^T) S g / 2. |R g0| = |g0| exactly."""
    t = sp.Symbol("t", positive=True)
    d = sp.Matrix(sp.symbols("d0:3", real=True))
    S = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"S_{min(i, j)}{max(i, j)}"))
    R2 = _rodrigues2(d, t)
    ER = R2.applyfunc(lambda e: _expect(sp.simplify(e), d, S)).subs(t, 1)
    assert is_zero(ER - (sp.eye(3) + (S - S.trace() * sp.eye(3)) / 2))
    gh = sp.Matrix(sp.symbols("g0:3", real=True))
    mean = ER * gh
    unit = {gh[2]: sp.sqrt(1 - gh[0] ** 2 - gh[1] ** 2)}
    along = (gh.T * (mean - gh))[0]
    assert is_zero((along + (S.trace() - (gh.T * S * gh)[0]) / 2).subs(unit))
    across = mean - gh - along * gh
    assert is_zero((across - (sp.eye(3) - gh * gh.T) * S * gh / 2).subs(unit))
    U = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"U{i}{j}"))
    assert is_zero((U * gh).dot(U * gh) - gh.dot(U.T * U * gh))  # with U^T U = I, |U g| = |g|


def test_spread_numbers():
    """ref-spread-mean's numbers: an isotropic spread sigma reads as a strain -sigma^2: -7.62e-5 at 0.5 deg, -3.05e-4
    at 1 deg, -1.22e-3 at 2 deg; a spread about one axis at 45 deg to g moves the mean direction by sigma^2 / 4:
    7.62e-5 rad at 1 deg; below 0.3 deg the ring's sigma^2 / 2 (ref-ring) is under 1.4e-5."""
    for deg, want in ((sp.Rational(1, 2), "7.62"), (1, "3.05"), (2, "1.22")):
        s2 = (sp.pi * deg / 180) ** 2
        S = s2 * sp.eye(3)
        g = sp.Matrix([0, 0, 1])
        app = -(S.trace() - (g.T * S * g)[0]) / 2
        scale = {"7.62": 10**5, "3.05": 10**4, "1.22": 10**3}[want]
        assert rounds_to(-app * scale, want)
    s2 = (sp.pi / 180) ** 2
    ax = sp.Matrix([1, 0, 1]) / sp.sqrt(2)
    g = sp.Matrix([0, 0, 1])
    shift = (sp.eye(3) - g * g.T) * (s2 * ax * ax.T) * g / 2
    assert rounds_to(sp.sqrt(shift.dot(shift)) * 10**5, "7.62")
    assert (sp.pi * sp.Rational(3, 10) / 180) ** 2 / 2 < sp.Rational(14, 10**6)


def test_tail_first_moment():
    """ref-tail: light beyond a face at t > 0 from a spot's mean carries at most N sigma^2 / t of its first moment
    about that mean, for any shape of variance sigma^2 (x 1(x >= t) <= x^2 / t pointwise); a two-point spot (mass p at
    t, the rest at -p t / (1 - p)) attains it."""
    x = sp.Symbol("x", real=True)
    t, p = sp.symbols("t p", positive=True)
    assert is_zero(sp.expand((x**2 / t - x) - x * (x - t) / t))  # >= 0 for x >= t; and x^2 / t >= 0 = LHS for x < t
    lo = -p * t / (1 - p)
    var = p * t**2 + (1 - p) * lo**2
    assert is_zero(p * lo + (1 - p) * lo - lo)  # (the mean is 0: p t + (1 - p) lo = 0)
    assert is_zero(sp.simplify(p * t + (1 - p) * lo))
    assert is_zero(sp.simplify(var / t - p * t - p**2 * t / (1 - p)))  # sigma^2 / t = p t (1 + p / (1 - p))
    assert sp.limit(var / t / (p * t), p, 0) == 1  # tight as p -> 0
    z = sp.Symbol("z")
    gauss_tail = sp.integrate(z * sp.exp(-(z**2) / 2) / sp.sqrt(2 * sp.pi), (z, 4, sp.oo))  # phi(4), sigma = 1
    assert rounds_to(gauss_tail * 10**4, "1.3")


def _log_pair_derivs(X: np.ndarray, a: float, s: float) -> tuple:
    """Gradient [2, n] and Hessian [2, 2, n] in (a, s) of log of (phi_s(x - a/2) + phi_s(x + a/2)) / 2, written as a
    log-sum-exp of l_pm = -u_pm^2 / (2 s^2) - log s (u_pm = x -+ a/2), so nothing underflows: with r_pm the softmax
    weights, grad = sum r grad l, Hess = sum r Hess l + sum r grad l grad l^T - grad grad^T."""
    u = np.stack([X - a / 2, X + a / 2])
    L = -(u**2) / (2 * s**2)
    r = np.exp(L - L.max(0))
    r /= r.sum(0)
    sg = np.array([1.0, -1.0])[:, None]
    gl = np.stack([sg * u / (2 * s**2), u**2 / s**3 - 1 / s])  # [param, branch, n]
    hl = np.array([[-np.ones_like(u) / (4 * s**2), -sg * u / s**3], [-sg * u / s**3, -3 * u**2 / s**4 + 1 / s**2]])
    g = (r * gl).sum(1)
    H = (r * hl).sum(2) + np.einsum("bn,ibn,jbn->ijn", r, gl, gl) - g[:, None] * g[None]
    return g, H


def _split(f: float, ratio: float, d: float) -> tuple:
    """Pseudo-true separation and width of two equal Gaussians of a free common width fitted (Poisson likelihood,
    continuous) to two equal spots of a core-and-wing shape of unit variance at -d/2, d/2: the root of the expected
    score, by Newton with its exact Jacobian; and the sandwich standard deviation of the separation, times sqrt(N). On
    a grid fine enough that the trapezoid rule is exact to float64 for these Gaussians."""
    X = np.linspace(-25, 25, 100001)  # the shapes' tails beyond are below 1e-30
    dx = X[1] - X[0]
    sc = 1 / np.sqrt((1 - f) + f * ratio**2)
    sw = ratio * sc

    def shape(u):
        return sum(w * np.exp(-(u**2) / (2 * v**2)) / (v * np.sqrt(2 * np.pi)) for w, v in ((1 - f, sc), (f, sw)))

    lt = (shape(X - d / 2) + shape(X + d / 2)) / 2
    th = np.array([d, 1.0])
    for _ in range(50):
        g, H = _log_pair_derivs(X, *th)
        step = np.linalg.solve((H * lt).sum(-1) * dx, (g * lt).sum(-1) * dx)
        th = th - step
        if np.max(np.abs(step)) < 1e-13:
            break
    g, H = _log_pair_derivs(X, *th)
    B, Hm = (g * lt) @ g.T * dx, (H * lt).sum(-1) * dx
    V = np.linalg.solve(Hm, np.linalg.solve(Hm, B).T)
    return th[0], th[1], np.sqrt(V[0, 0])


def test_log_pair_derivs():
    """_log_pair_derivs' log-sum-exp form is the gradient and Hessian of the pair's log density: with l_pm the two
    log terms and r_pm their softmax weights, grad log = sum r grad l and Hess log = sum r Hess l + sum r grad l
    grad l^T - grad grad^T, identically; and the code's numbers are that form's (16 ulps of its terms)."""
    x, a, s_ = sp.symbols("x a s", real=True)
    u = [x - a / 2, x + a / 2]
    lterms = [-(v**2) / (2 * s_**2) - sp.log(s_) for v in u]
    ll = sp.log((sp.exp(lterms[0]) + sp.exp(lterms[1])) / (2 * sp.sqrt(2 * sp.pi)))
    tot = sp.exp(lterms[0]) + sp.exp(lterms[1])
    r = [sp.exp(t) / tot for t in lterms]
    ps = (a, s_)
    grad = [sum(ri * sp.diff(t, p) for ri, t in zip(r, lterms)) for p in ps]
    for i, p in enumerate(ps):
        assert is_zero(sp.diff(ll, p) - grad[i])
        for j, q in enumerate(ps):
            hess = sum(ri * (sp.diff(t, p, q) + sp.diff(t, p) * sp.diff(t, q)) for ri, t in zip(r, lterms))
            assert is_zero(sp.diff(ll, p, q) - (hess - grad[i] * grad[j]))
    pts = np.array([-3.1, -0.4, 0.0, 0.7, 2.9])
    g, H = _log_pair_derivs(pts, 1.7, 0.8)
    for n, xv in enumerate(pts):
        at = {x: sp.Float(xv, 30), a: sp.Float(1.7, 30), s_: sp.Float(0.8, 30)}
        scale = 1 + sum(float(sp.N(abs(sp.diff(t, p)).subs(at), 30)) ** 2 for t in lterms for p in ps)
        for i, p in enumerate(ps):
            assert abs(float(sp.diff(ll, p).subs(at)) - g[i, n]) <= 16 * EPS * scale
            for j, q in enumerate(ps):
                assert abs(float(sp.diff(ll, p, q).subs(at)) - H[i, j, n]) <= 16 * EPS * scale


def test_split_bias():
    """ref-split: two spots d apart (in units of their own sd), each a core and a wing, fitted as two Gaussians of a
    free common width: the separation's bias, its noise (sandwich, times sqrt(N), N the photons of the pair) and the
    N at which they are equal. Quoted for the measured point spread's shape (wing 5.2 times the core, 28 %) and a mild
    one (twice, 10 %), d = 3, 4, 6; with Gaussian spots the fit is unbiased and the noise tends to 2 (each spot's
    centroid alone)."""
    quoted = {
        (0.28, 5.2): {3: ("0.210", "1.53", "54"), 4: ("0.129", "1.70", "170"), 6: ("0.0383", "1.88", "2400")},
        (0.1, 2.0): {3: ("0.0333", "2.02", "3700"), 4: ("0.0318", "1.94", "3700"), 6: ("0.0113", "1.97", "30000")},
    }
    for (f, ratio), rows in quoted.items():
        for d, (bias, noise, n_eq) in rows.items():
            a, _, sd = _split(f, ratio, d)
            b = a - d
            assert rounds_to(sp.Float(b, 15), bias) and rounds_to(sp.Float(sd, 15), noise)
            assert float(f"{(sd / b) ** 2:.2g}") == float(n_eq)  # to two significant digits
    # Gaussian spots: the model holds, so the truth is the root (a correct model's expected score is the derivative
    # of its normalisation, zero: test_score_mean_zero), which Newton reaches to its stopping step (1e-13); the noise
    # is 2 to the digits quoted (two spots of N / 2 photons, sd 1 each: 2 / sqrt(N))
    a, s, sd = _split(0.0, 1.0, 6.0)
    assert abs(a - 6) <= 1e-12 and abs(s - 1) <= 1e-12 and rounds_to(sp.Float(sd, 15), "2.00")


def test_score_mean_zero():
    """For any family of densities p(x; t) the expected score is zero: int p d log p / dt = d/dt int p = 0. For the
    pair of Gaussians, symbolically in both parameters."""
    x, a, s_ = sp.symbols("x a s", positive=True)
    gauss = lambda v: sp.exp(-(v**2) / (2 * s_**2)) / (s_ * sp.sqrt(2 * sp.pi))
    p = (gauss(x - a / 2) + gauss(x + a / 2)) / 2
    for t in (a, s_):
        assert sp.simplify(sp.integrate(sp.diff(p, t), (x, -sp.oo, sp.oo))) == 0


def test_tangent_spot_on_a_ring():
    """ref-ring: a spot spread along the tangent of a ring of radius rho (a Gaussian in t of sd sigma_t, as a
    first-order model renders it) has its cells' mean radius sqrt(rho^2 + t^2) at rho + sigma_t^2 / (2 rho) + O(sigma^4);
    with rho = L tan 2 theta that is d(2 theta) = (sigma_eta^2 / 2) sin 2 theta cos 2 theta (sigma_eta = sigma_t / rho),
    read as a strain -d(2 theta) / (2 tan theta) = -(sigma_eta^2 / 2) cos^2 theta cos 2 theta: -1.52e-4 at
    sigma_eta = 1 deg, small theta."""
    t, rho, st, L, th, e = sp.symbols("t rho sigma_t L theta epsilon", positive=True)
    r2 = sp.series(sp.sqrt(rho**2 + (e * t) ** 2), e, 0, 4).removeO().subs(e, 1)
    mean = sp.expand(r2).subs(t**2, st**2)  # E[t^2] = sigma_t^2 (no other moment below fourth order)
    assert is_zero(mean - rho - st**2 / (2 * rho))
    # d rho / rho = d(2 theta) / (sin 2 theta cos 2 theta) for rho = L tan 2 theta
    drho = sp.diff(L * sp.tan(2 * th), th) / 2  # per unit of 2 theta
    assert is_zero(sp.simplify(drho / (L * sp.tan(2 * th)) - 1 / (sp.sin(2 * th) * sp.cos(2 * th))))
    s_eta = sp.Symbol("s_eta", positive=True)
    d2th = s_eta**2 / 2 * sp.sin(2 * th) * sp.cos(2 * th)
    assert is_zero(sp.simplify(-d2th / (2 * sp.tan(th)) + s_eta**2 / 2 * sp.cos(th) ** 2 * sp.cos(2 * th)))
    assert rounds_to((sp.pi / 180) ** 2 / 2 * 10**4, "1.52")
