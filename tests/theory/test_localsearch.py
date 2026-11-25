"""Proofs for docs/source/theory/localsearch.rst: the row shift of an orientation error, the children of an
orientation, and the voxel width of each level."""

import math

import numpy as np
import sympy as sp

from tests.theory.common import (
    EPS,
    agree,
    evaluate,
    is_zero,
    rng,
)


def _worst(step: sp.Expr) -> sp.Expr:
    """srch-cubic-delta, degrees."""
    return sp.deg(2 * sp.atan(sp.sqrt(3) / 2 * sp.tan(sp.rad(step) / 2)))


def test_row_shift():
    """ls-row-shift: with dif-dty for a beam along x, dty - y0 = -(x sin w + y cos w) = -r sin(w + phi); its
    derivative in w is -r cos(w + phi), and (d dty / dw)^2 + (dty - y0)^2 = r^2, so |dy| <= r |dw|."""
    x, y, w = sp.symbols("x y omega", real=True)
    r, phi = sp.symbols("r phi", real=True)
    dty = -(x * sp.sin(w) + y * sp.cos(w))
    polar = {x: r * sp.cos(phi), y: r * sp.sin(phi)}
    assert is_zero(dty.subs(polar) + r * sp.sin(w + phi))
    assert is_zero(sp.diff(dty, w).subs(polar) + r * sp.cos(w + phi))
    assert is_zero(sp.diff(dty, w) ** 2 + dty**2 - (x**2 + y**2))


def test_worst_cases_quoted():
    """The page's 0.87 deg (1 deg grid) and 2.6 deg (3 deg grid) are srch-cubic-delta rounded; two rounds from 3 deg
    leave 0.29 deg and a third 0.096 deg."""
    assert round(float(sp.N(_worst(1), 30)), 2) == 0.87
    assert round(float(sp.N(_worst(3), 30)), 1) == 2.6
    assert round(float(sp.N(_worst(3) / 9, 30)), 2) == 0.29
    assert round(float(sp.N(_worst(3) / 27, 30)), 3) == 0.096


def test_level_rows_quoted():
    """ls-level-rows on DTL1z45 (R = 141 um, 0.25 um rows, refined to a 3 deg grid's worst case / 9, or a 1 deg grid's
    / 3: both 0.2887 deg): 3 rows (0.75 um), then 1."""
    R, ystep = 141, sp.Rational(1, 4)
    for e in (_worst(3) / 9, _worst(1) / 3):
        rows = [max(1, math.ceil(sp.N(R * sp.rad(e / 3**k) / ystep, 30))) for k in range(2)]
        assert rows == [3, 1]


def test_children_code():
    """anri's children are ls-children: R(s / 3 (a, b, c)) U, exact rotations about sample axes."""
    from anri.index.local import children
    from anri.maths.rotation import quat_to_mat

    q = rng(5).normal(size=4)
    U = quat_to_mat(q / np.linalg.norm(q))
    step = 1.5
    s = sp.rad(sp.Float(step, 30)) / 3
    want = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                v = s * sp.Matrix([a, b, c])
                t = sp.sqrt(v.dot(v))
                n = v / t if t != 0 else v
                K = sp.Matrix([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
                R = sp.eye(3) + sp.sin(t) * K + (1 - sp.cos(t)) * K * K  # Rodrigues' formula for exp([v]x)
                want.append(evaluate(R, {}) @ U)
    # U is float64 (2 ulps from orthogonal); each entry is a 3-term product of entries <= 1 with the rotation's
    # (a tan, a division and the Rodrigues matrix: ~10 roundings): within 16 ulps of 1, absolutely
    agree(np.array(want), children, U[None], step, rtol=0, atol=16 * EPS)


def test_level_rows_code():
    """anri's level_rows is ls-level-rows, the last level one row."""
    from anri.index.local import level_rows

    R, e, ystep = 141.0, 0.8660254037844386, 0.25
    want = [max(1, math.ceil(sp.N(R * sp.rad(sp.Float(e, 30) / 3**k) / ystep, 30))) for k in range(3)]
    agree(np.array([*want[:-1], 1], float), level_rows, R, e, ystep, 2, rtol=0, atol=0)  # integers: exact


def test_support_share():
    """ls-support: a feature filling n of a coarse voxel's g^2 fine voxels holds n / g^2 of it; at min_share 0.01 one
    fine voxel reaches the last level for g <= 10; a lamella one voxel thick in g = 3 is 1/3, one voxel alone 1/9."""
    ms = sp.Rational(1, 100)
    assert all((sp.Integer(1) / g**2 >= ms) == (g <= 10) for g in range(1, 20))
    g = 3
    assert sp.Integer(g) / g**2 == sp.Rational(1, 3) and sp.Integer(1) / g**2 == sp.Rational(1, 9)


def test_unique_optimum():
    """ls-start: the Poisson log-likelihood l = sum d log(A f) - sum A f has Hessian -A^T diag(d / mu^2) A (by
    differentiation, 3 bins x 2 unknowns), negative definite iff A's columns over the recorded bins (d > 0) are
    independent; with two equal columns it is singular."""
    A = sp.Matrix(3, 2, sp.symbols("a0:6", positive=True))
    f = sp.Matrix(sp.symbols("f0:2", positive=True))
    d = sp.Matrix(sp.symbols("d0:3", positive=True))
    mu = A * f
    ll = sum(d[b] * sp.log(mu[b]) - mu[b] for b in range(3))
    H = sp.Matrix(2, 2, lambda i, j: sp.diff(ll, f[i], f[j]))
    W = sp.diag(*[d[b] / mu[b] ** 2 for b in range(3)])
    assert sp.simplify(H + A.T * W * A) == sp.zeros(2, 2)
    a = sp.Matrix(sp.symbols("c0:3", positive=True))
    A2 = a.row_join(a)
    assert sp.simplify((A2.T * W.subs({A[i, j]: A2[i, j] for i in range(3) for j in range(2)}) * A2).det()) == 0


def test_start_persists():
    """ls-start: two candidates with the same column get the same MLEM factor A^T(d / A f) / A^T 1 at every update,
    whatever the data and the other candidates, so their ratio stays the start's (three updates, symbolic)."""
    a = sp.Matrix(sp.symbols("a0:3", positive=True))
    b = sp.Matrix(sp.symbols("b0:3", positive=True))
    A = a.row_join(a).row_join(b)
    d = sp.Matrix(sp.symbols("d0:3", positive=True))
    f = sp.Matrix(sp.symbols("f0:3", positive=True))
    r0 = f[0] / f[1]
    norm = A.T * sp.ones(3, 1)
    for _ in range(3):
        ratio = sp.Matrix([d[i] / (A * f)[i] for i in range(3)])
        f = sp.Matrix([f[j] * (A.T * ratio)[j] / norm[j] for j in range(3)])
        assert sp.simplify(f[0] / f[1] - r0) == 0


def test_tails_log_concave():
    """ls-concave: log Q(c + 1, mu) (an empty bin) and log P(d, mu) (a saturated bin) are concave in mu. With
    Q' = -p and p' = p (c / mu - 1), (log Q)'' = -p (Q (c / mu - 1) + p) / Q^2, so concavity is Q (1 - c / mu) <= p,
    trivially for mu <= c; for mu > c, c log u <= c log mu + c (u - mu) / mu (log y <= y - 1) bounds the integrand of
    Gamma(c + 1, mu) by an exponential whose integral is mu^c e^-mu / (1 - c / mu). The saturated bin likewise, on
    [0, mu] for mu < d - 1."""
    mu, u = sp.symbols("mu u", positive=True)
    c = sp.Symbol("c", positive=True)
    Q = sp.uppergamma(c + 1, mu) / sp.gamma(c + 1)
    p = mu**c * sp.exp(-mu) / sp.gamma(c + 1)
    assert sp.simplify(sp.diff(Q, mu) + p) == 0
    assert sp.simplify(sp.diff(p, mu) - p * (c / mu - 1)) == 0
    Qs, ps = sp.symbols("Q p", positive=True)  # (log Q)'' with Q' = -p, Q'' = -p (c / mu - 1)
    second = (Qs * (-ps * (c / mu - 1)) - ps**2) / Qs**2
    assert sp.simplify(second + ps * (Qs * (c / mu - 1) + ps) / Qs**2) == 0
    # the integrand's bound: c log(u / mu) - c (u - mu) / mu = c (log y - (y - 1)), y = u / mu
    y = sp.Symbol("y", positive=True)
    assert sp.simplify((c * sp.log(u / mu) - c * (u - mu) / mu).subs(u, y * mu) - c * (sp.log(y) - (y - 1))) == 0
    k = sp.Symbol("k", positive=True)  # k = 1 - c / mu > 0
    bound = sp.integrate(mu**c * sp.exp(-mu) * sp.exp(-k * (u - mu)), (u, mu, sp.oo))
    assert sp.simplify(bound - mu**c * sp.exp(-mu) / k) == 0
    # saturated: P' = g, g' = g ((d - 1) / mu - 1); concavity is P ((d - 1) / mu - 1) <= g; on [0, mu] the integrand
    # is below g's exponential continuation, whose integral to mu from -oo is g mu^(d-1)... / ((d - 1) / mu - 1)
    d = sp.Symbol("d", positive=True)
    P = sp.lowergamma(d, mu) / sp.gamma(d)
    g = mu ** (d - 1) * sp.exp(-mu) / sp.gamma(d)
    assert sp.simplify(sp.diff(P, mu) - g) == 0
    assert sp.simplify(sp.diff(g, mu) - g * ((d - 1) / mu - 1)) == 0
    k2 = sp.Symbol("k2", positive=True)  # (d - 1) / mu - 1 > 0
    bound2 = sp.integrate(sp.exp(k2 * (u - mu)), (u, -sp.oo, mu))
    assert sp.simplify(bound2 - 1 / k2) == 0


def test_certificate():
    """ls-kkt: for plain Poisson terms l = sum d log mu - mu, mu = A f, l(f) - l(f^) - grad l(f^) . (f - f^) =
    sum d (log y - (y - 1)) with y = mu / mu^, at most 0 (log y <= y - 1); and with grad_j <= 0 everywhere and
    grad_j f^_j = 0, grad . (f - f^) = sum grad_j f_j <= 0 for f >= 0."""
    A = sp.Matrix(3, 2, sp.symbols("a0:6", positive=True))
    f = sp.Matrix(sp.symbols("f0:2", positive=True))
    fh = sp.Matrix(sp.symbols("h0:2", positive=True))
    dd = sp.Matrix(sp.symbols("d0:3", positive=True))

    def ll(x: sp.Matrix) -> sp.Expr:
        m = A * x
        return sum(dd[b] * sp.log(m[b]) - m[b] for b in range(3))

    grad = sp.Matrix([sp.diff(ll(fh), fh[j]) for j in range(2)])
    lhs = ll(f) - ll(fh) - (grad.T * (f - fh))[0]
    mu, muh = A * f, A * fh
    yb = [mu[b] / muh[b] for b in range(3)]
    rhs = sum(dd[b] * (sp.log(yb[b]) - (yb[b] - 1)) for b in range(3))
    assert sp.simplify(sp.expand_log(lhs - rhs, force=True)) == 0
    gj = sp.symbols("g0:2", nonpositive=True)
    fj = sp.symbols("x0:2", nonnegative=True)
    # complementary slackness: grad . f^ = 0, so grad . (f - f^) = grad . f, a sum of non-positive terms
    assert sum(gj[j] * fj[j] for j in range(2)).is_nonpositive


def test_gain_bound():
    """ls-gain: phi(t) = sum d log(1 + t a / mu) - t sum a is concave, so 2 phi(t) <= 2 phi'(0) t = 2 (S - 1) N for the
    N = t sum a photons it adds; and phi'' = -sum d a^2 / (mu + t a)^2 >= -h,
    h = sum d a^2 / mu^2, for t >= 0, so phi(t) >= g t - h t^2 / 2 (phi - g t + h t^2 / 2 is convex with value and slope
    0 at 0) and 2 max phi >= g^2 / h. Two equal columns have equal scores, so a duplicate of an active column (score 1)
    has g = 0 and, phi being concave, gains nothing."""
    t = sp.Symbol("t", nonnegative=True)
    a = sp.symbols("a0:3", positive=True)
    m = sp.symbols("m0:3", positive=True)
    dd = sp.symbols("d0:3", positive=True)
    phi = sum(dd[b] * sp.log(1 + t * a[b] / m[b]) for b in range(3)) - t * sum(a)
    d2 = sp.diff(phi, t, 2)
    assert sp.simplify(d2 + sum(dd[b] * a[b] ** 2 / (m[b] + t * a[b]) ** 2 for b in range(3))) == 0
    for b in range(3):  # each term's curvature is at most its value at t = 0: (m + t a)^2 >= m^2 for t >= 0
        assert sp.factor((m[b] + t * a[b]) ** 2 - m[b] ** 2).is_nonnegative
    g, h = sp.symbols("g h", positive=True)
    q = g * t - h * t**2 / 2
    tstar = sp.solve(sp.diff(q, t), t)[0]
    assert sp.simplify(2 * q.subs(t, tstar) - g**2 / h) == 0
    g0 = sp.diff(phi, t).subs(t, 0)
    S = sum(dd[b] / m[b] * a[b] for b in range(3)) / sum(a)  # the column's score
    assert sp.simplify(g0 - sum(a) * (S - 1)) == 0
    # concavity's upper bound: phi(t) <= phi'(0) t, as phi'' <= 0 (each term of d2 is negative); N = t sum a photons
    assert all(term.is_negative for term in sp.Add.make_args(sp.expand(d2)))
    N = t * sum(a)
    assert sp.simplify(2 * g0 * t - 2 * (S - 1) * N) == 0


def test_gap_bound():
    """ls-gap: d log mu <= d log(d / r) - d + r mu for d, r, mu > 0 (log y <= y - 1 at y = r mu / d); with
    r~ = rho / s_max feasible (A^T r~ <= A^T 1 when every score is at most s_max), sum (r~ - 1) mu = f . A^T (r~ - 1) <= 0,
    and at rho = d / mu^ the bound minus l(f^) is (sum d) log s_max + sum mu^ - sum d."""
    d, r, mu = sp.symbols("d r mu", positive=True)
    y = r * mu / d
    gap = d * sp.log(d / r) - d + r * mu - d * sp.log(mu)
    assert sp.simplify(sp.expand_log(gap - d * ((y - 1) - sp.log(y)), force=True)) == 0
    dd = sp.symbols("d0:3", positive=True)
    mh = sp.symbols("m0:3", positive=True)
    s = sp.Symbol("s", positive=True)
    rt = [dd[b] / mh[b] / s for b in range(3)]
    dual = sum(dd[b] * sp.log(dd[b] / rt[b]) - dd[b] for b in range(3))
    primal = sum(dd[b] * sp.log(mh[b]) - mh[b] for b in range(3))
    want = sum(dd) * sp.log(s) + sum(mh) - sum(dd)
    assert sp.simplify(sp.expand_log(dual - primal - want, force=True)) == 0


def test_coarse_positions_code():
    """coarse_positions: the points of recon_positions' grid of the coarse step within 0.75 of a step of any fine
    voxel, wide enough to hold them all; by brute force over every pair."""
    from anri.geom.scan import recon_positions
    from anri.index.occupancy import coarse_positions

    fine = np.asarray(recon_positions(9, 1.0))
    fine = fine[np.hypot(fine[:, 0], fine[:, 1]) <= 3.2]
    step = 3.0
    n = 2 * int(np.ceil(np.abs(fine[:, :2]).max() / step)) + 3
    grid = np.asarray(recon_positions(n, step))
    near = [p for p in grid if np.min(np.hypot(*(fine[:, :2] - p[:2]).T)) <= 0.75 * step]
    got = coarse_positions(fine, step)
    key = lambda a: sorted(map(tuple, np.round(np.asarray(a)[:, :2] / step).astype(int).tolist()))
    assert key(got) == key(near)

