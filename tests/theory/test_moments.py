"""Proofs for docs/source/theory/moments.rst: properties of counts and moments of recorded light over a region."""

from __future__ import annotations

import itertools

import sympy as sp

from tests.theory.common import (
    is_zero,
    p_at_least,
    p_group,
    rank,
    rounds_to,
)


def test_ray_sum():
    """The moments of the light over cells are the beam- and density-weighted sums of each contributor's own binned
    spot moments, for any spot shape (symbolic: 2 contributors, 4 cells, f = 1, x, y, x^2, x y, y^2)."""
    n_v, n_c = 2, 4
    b = sp.symbols(f"b0:{n_v}", positive=True)  # beam weights
    rho = sp.symbols(f"rho0:{n_v}", positive=True)  # densities
    s = sp.Matrix(n_v, n_c, lambda v, p: sp.Symbol(f"s_{v}_{p}"))  # each contributor's light per cell, any shape
    x = sp.symbols(f"x0:{n_c}")
    y = sp.symbols(f"y0:{n_c}")
    lam = [sum(b[v] * rho[v] * s[v, p] for v in range(n_v)) for p in range(n_c)]
    for f in (lambda p: 1, lambda p: x[p], lambda p: y[p], lambda p: x[p] ** 2, lambda p: x[p] * y[p],
              lambda p: y[p] ** 2):  # fmt: skip
        lhs = sum(lam[p] * f(p) for p in range(n_c))
        rhs = sum(b[v] * rho[v] * sum(s[v, p] * f(p) for p in range(n_c)) for v in range(n_v))
        assert is_zero(lhs - rhs)


def test_cut_deficit():
    """lam - E[r] = lam P(n <= c - 1) + E[n 1(n > c)] sum_{k < m} P_k, for c = 0..3 (the cut's and the group rule's
    deficit); at c = 1 the cut's part is lam e^-lam."""
    lam, w = sp.symbols("lambda w", positive=True)  # w = 1 - sum_{k<m} P_k
    for c in range(4):
        e1 = lam * p_at_least(c, lam)
        deficit = lam - e1 * w
        assert is_zero(deficit - (lam * (1 - p_at_least(c, lam)) + e1 * (1 - w)))
    assert is_zero(lam * (1 - p_at_least(1, lam)) - lam * sp.exp(-lam))


def test_second_moments_see_q():
    """Raw moments of a voxel's density f_v(beta) (any shape, mean beta_v, covariance Sigma_v): N, N beta, N Q with
    Q = Sigma + beta beta^T. (beta, Sigma) -> (beta, Q) is a bijection, so with Sigma free the second moments' means do
    not depend on beta: a change d beta with d Sigma = -(d beta beta^T + beta d beta^T) leaves them unchanged (d = 3)."""
    d = 3
    beta = sp.Matrix(sp.symbols(f"beta0:{d}"))
    S = sp.Matrix(d, d, lambda i, j: sp.Symbol(f"S_{min(i, j)}{max(i, j)}"))
    Q = S + beta * beta.T
    db = sp.Matrix(sp.symbols(f"db0:{d}"))
    dS = -(db * beta.T + beta * db.T)
    t = sp.Symbol("t")
    Qt = (S + t * dS) + (beta + t * db) * (beta + t * db).T
    assert is_zero(sp.diff(Qt, t).subs(t, 0))
    assert is_zero(Q - beta * beta.T - S)  # inverse map


def _profiled(J: sp.Matrix, C: sp.Matrix, n_beta: int) -> sp.Matrix:
    """Fisher information on the first n_beta parameters, the rest profiled out: Schur complement of J^T C^-1 J."""
    info = J.T * C.inv() * J
    A, B, D = info[:n_beta, :n_beta], info[:n_beta, n_beta:], info[n_beta:, n_beta:]
    return A - B * D.inv() * B.T


def test_second_moments_information():
    """With Q free, second moments add information on beta only through their noise correlation with the first
    moments, and none when they are no more numbers than the free Q (square, invertible block): then the profiled
    information is J1^T C11^-1 J1, the first moments' alone, whatever C. With more sums than free Q and correlated noise
    they add some (exact rationals: one voxel, two blobs, scalar beta and Q)."""
    # square: 2 first-moment rows (beta, 2 numbers), 3 second-moment rows (vech Q, 3 numbers), any SPD C
    L = sp.Matrix(5, 5, lambda i, j: sp.Rational((3 * i + 7 * j) % 11 + 1, 7) if j <= i else 0)
    C = L * L.T
    N = sp.Rational(5)
    J = sp.diag(N * sp.eye(2), N * sp.eye(3))  # d mu / d (beta, vech Q)
    assert is_zero(_profiled(J, C, 2) - N * sp.eye(2) * C[:2, :2].inv() * N * sp.eye(2))
    # tall: blobs see the voxel with weights a1, a2; mu1_b = a_b N beta, mu2_b = a_b N Q
    a = [sp.Rational(1, 2), sp.Rational(3, 2)]
    J = sp.Matrix([[a[0] * N, 0], [a[1] * N, 0], [0, a[0] * N], [0, a[1] * N]])
    C = sp.Matrix([[4, 1, 2, 0], [1, 3, 0, 1], [2, 0, 5, 1], [0, 1, 1, 6]])  # S1 and S2 of a blob correlated
    first = J[:2, :1].T * C[:2, :2].inv() * J[:2, :1]
    prof = _profiled(J, C, 1)
    assert prof[0, 0] > first[0, 0]
    C0 = sp.diag(C[:2, :2], C[2:, 2:])  # uncorrelated: nothing added
    assert is_zero(_profiled(J, C0, 1) - first)


def _recorded_1d(bits: tuple, m: int) -> list:
    """Recorded pixels of a 1D strip (2-connected groups of >= m kept pixels)."""
    n, out, i = len(bits), [0] * len(bits), 0
    while i < n:
        if bits[i]:
            j = i
            while j < n and bits[j]:
                j += 1
            if j - i >= m:
                out[i:j] = [1] * (j - i)
            i = j
        else:
            i += 1
    return out


def test_selection_bias():
    """A region defined from the data (recorded pixels, dilated by one) holds every recorded pixel, so
    E[sum_{p in R} r_p] = sum_p E[r_p], while the model summed over that region has mean sum_p E[r_p] P(p in R): the
    data exceed the model by sum_p E[r_p] (1 - P(p in R)) >= 0, and E[r_p | p in R] = E[r_p] / P(p in R). Exact
    enumeration on a strip of 4 pixels, m = 1 and 2 (the group rule is not needed for the bias). A region fixed
    before the data are seen has no bias."""
    n = 4
    a = sp.symbols(f"a0:{n}", positive=True)
    e = sp.symbols(f"e0:{n}", positive=True)  # E[n 1(n > c)]; given p kept its mean count is e / a
    for m in (1, 2):
        data_sum, model_sum, fixed_sum = 0, 0, 0
        R0 = (1, 2)  # a region fixed before the data are seen
        er = [0] * n
        pin = [0] * n
        for bits in itertools.product((0, 1), repeat=n):
            w = sp.Mul(*[a[i] if b else 1 - a[i] for i, b in enumerate(bits)])
            rec = _recorded_1d(bits, m)
            R = [any(rec[j] for j in range(max(i - 1, 0), min(i + 2, n))) for i in range(n)]
            for i in range(n):
                er[i] += w * rec[i] * e[i] / a[i]
                pin[i] += w * int(R[i])
                data_sum += w * int(R[i]) * rec[i] * e[i] / a[i]
                fixed_sum += w * int(i in R0) * rec[i] * e[i] / a[i]
        for i in range(n):
            model_sum += er[i] * pin[i]
        assert is_zero(data_sum - sum(er))
        bias = sum(er[i] * (1 - pin[i]) for i in range(n))
        assert is_zero(data_sum - model_sum - bias)
        assert not is_zero(bias)
        assert is_zero(fixed_sum - sum(er[i] for i in R0))


def test_single_frame_rank():
    """Moments of a region lying in one frame: omega is the same constant in data and model, so the 10 features
    [1, w, s, f, ww, ss, ff, ws, wf, sf] satisfy four linear relations and their covariance has rank 6 (exact)."""
    w0 = sp.Rational(7, 3)
    V = sp.zeros(10, 10)
    for k in range(12):  # 12 cells with generic rational positions and variances
        s, f = sp.Rational((5 * k) % 13, 3), sp.Rational((7 * k + 2) % 11, 5)
        ft = sp.Matrix([1, w0, s, f, w0**2, s**2, f**2, w0 * s, w0 * f, s * f])
        V += sp.Rational(k + 1, 4) * ft * ft.T
    assert rank(V) == 6


def test_exclusion_stencil():
    """An error in one cell's light changes E[r_p] of the cells of its frame within m - 1 pixels (Chebyshev), and of
    no others: for m = 2 the 8 neighbours, for m = 3 all 24 cells of the 5 x 5 window (derivatives of 1 - P1 - P2)."""
    a = {(i, j): sp.Symbol(f"a_{i}_{j}") for i in range(-3, 4) for j in range(-3, 4)}
    p = (0, 0)
    w2 = 1 - p_group(p, 1, a)
    w3 = w2 - p_group(p, 2, a)
    point = {a[q]: sp.Rational(1 + (3 * q[0] + 5 * q[1]) % 7, 9) for q in a}  # a generic rational point
    for q, aq in a.items():
        if q == p:
            continue
        cheb = max(abs(q[0]), abs(q[1]))
        for w, reach in ((w2, 1), (w3, 2)):
            if cheb > reach:  # a_q does not occur at all
                assert aq not in w.free_symbols
            else:  # the derivative is a polynomial that is not zero at one rational point, so not zero
                assert sp.diff(w, aq).subs(point) != 0


def test_binned_gaussian_log_concave():
    """Cells of width h of a Gaussian (sd sigma): b_{k-1} b_{k+1} <= rho b_k^2, rho = (1 + exp(-2 h^2 / sigma^2)) / 2
    < 1. The pairing identity behind it: phi(x - h) phi(y + h) = phi(x) phi(y) exp(-(h (y - x) + h^2) / sigma^2), and
    symmetrised over x <-> y in one cell (|y - x| <= h) the factor is cosh(h (y - x) / sigma^2) exp(-h^2 / sigma^2)
    <= rho."""
    x, y, h, sig, mu = sp.symbols("x y h sigma mu", real=True)

    def logphi(z):
        return -((z - mu) ** 2) / (2 * sig**2)

    assert is_zero(logphi(x - h) + logphi(y + h) - logphi(x) - logphi(y) + (h * (y - x) + h**2) / sig**2)
    u = sp.Symbol("u", real=True)  # u = h (y - x) / sigma^2
    sym = (sp.exp(-u) + sp.exp(u)) / 2 * sp.exp(-(h**2) / sig**2)
    assert is_zero(sym.rewrite(sp.exp) - sp.cosh(u).rewrite(sp.exp) * sp.exp(-(h**2) / sig**2))
    # cosh is even and increasing in |u| <= h^2 / sigma^2: the bound at the end of the range
    rho = (sp.cosh(h**2 / sig**2) * sp.exp(-(h**2) / sig**2)).rewrite(sp.exp)
    assert is_zero(sp.expand(rho - (1 + sp.exp(-2 * h**2 / sig**2)) / 2))


def test_mixture_not_log_concave():
    """A two-Gaussian point spread (core sd 0.25 px with 72% of the light, wing sd 1.3 px with 28%, AGENTS.md's 43.6
    keV numbers), binned in 1 px cells about its centre, is not log-concave: b_1^2 < b_0 b_2, so the light beyond a
    face can exceed b r^n. Cell masses evaluated at 30 digits: b_0 b_2 / b_1^2 = 3.30, and two cells out the light is
    7.8 times b_1 r^2 (r = b_1 / b_0)."""

    def mass(lo, hi):
        tot = 0
        for frac, sd in ((sp.Rational(72, 100), sp.Rational(1, 4)), (sp.Rational(28, 100), sp.Rational(13, 10))):
            tot += frac * (sp.erf(hi / (sd * sp.sqrt(2))) - sp.erf(lo / (sd * sp.sqrt(2)))) / 2
        return tot

    b = [mass(k - sp.Rational(1, 2), k + sp.Rational(1, 2)) for k in range(5)]
    assert rounds_to(b[0] * b[2] / b[1] ** 2, "3.30")
    assert rounds_to(b[3] / (b[1] * (b[1] / b[0]) ** 2), "7.8")


def test_gauge():
    """Expected light of component k on ring h: w_k s_h G (G known). In logs the Jacobian rows are e_k + e_{K+h} over
    the observed (k, h) pairs: rank K + H - (connected parts of the component-ring graph), with the null vector
    (1, ..., 1, -1, ..., -1) on each part. Exact ranks: 3 components on 3 rings, connected (one gauge) and split in
    two parts (two gauges)."""

    def jac(pairs, K, H):
        J = sp.zeros(len(pairs), K + H)
        for i, (k, h) in enumerate(pairs):
            J[i, k] = 1
            J[i, K + h] = 1
        return J

    J = jac([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0)], 3, 3)
    assert rank(J) == 5
    assert is_zero(J * sp.Matrix([1, 1, 1, -1, -1, -1]))
    J = jac([(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)], 3, 3)  # component 2 alone on ring 2
    assert rank(J) == 4
