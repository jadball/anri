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
