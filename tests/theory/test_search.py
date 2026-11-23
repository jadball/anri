"""Proofs for docs/source/theory/search.rst: misorientations on the cubic Rodrigues grid, its fundamental zone, and
the matching tolerances."""

import numpy as np
import sympy as sp

from tests.theory.common import (
    EPS,
    agree,
    evaluate,
    is_zero,
    rng,
)


def _rtol(expr: sp.Expr, vals: dict, n_ops: int) -> float:
    """Float64 error bound of a formula: n_ops roundings, plus one rounding of each input (as the code converts it,
    e.g. to radians) times its condition number |x df/dx / f|."""
    f = float(evaluate(expr, vals))
    cond = sum(abs(float(evaluate(x * sp.diff(expr, x), vals))) for x in vals) / abs(f)
    return (n_ops + cond) * EPS


r1 = sp.Matrix(sp.symbols("r1:4", real=True))
r2 = sp.Matrix(sp.symbols("s1:4", real=True))


def test_rodrigues_misorientation():
    """Rotations with Rodrigues vectors r1, r2 (quaternions (1, r) / |(1, r)|) are misoriented by w with
    tan^2(w / 2) = (|r1 - r2|^2 + |r1 x r2|^2) / (1 + r1 . r2)^2; from the identity (r1 = 0), w = 2 arctan |r2|."""
    n1, n2, dot = 1 + r1.dot(r1), 1 + r2.dot(r2), 1 + r1.dot(r2)
    # cos(w / 2) = q1 . q2 = dot / sqrt(n1 n2), so tan^2(w / 2) = (n1 n2 - dot^2) / dot^2
    assert is_zero(n1 * n2 - dot**2 - ((r1 - r2).dot(r1 - r2) + r1.cross(r2).dot(r1.cross(r2))))


def test_rodrigues_metric_largest_at_origin():
    """A step dr at r turns by ds^2 = 4 |dq|^2 = 4 (|dr|^2 (1 + |r|^2) - (r . dr)^2) / (1 + |r|^2)^2
    <= 4 |dr|^2 / (1 + |r|^2): a grid regular in r is coarsest, in angle, at the origin."""
    rho2 = r1.dot(r1)
    q = sp.Matrix([1, *r1]) / sp.sqrt(1 + rho2)
    J = q.jacobian(r1)
    metric = (sp.eye(3) * (1 + rho2) - r1 * r1.T) / (1 + rho2) ** 2
    assert is_zero(J.T * J - metric)
    assert is_zero(sp.eye(3) / (1 + rho2) - metric - r1 * r1.T / (1 + rho2) ** 2)  # the gap: (r . dr)^2 >= 0


def test_cubic_delta():
    """srch-cubic-delta: a cell of side d = tan(s / 2) centred on the origin has its corners at |r| = sqrt(3) / 2 d,
    misoriented from the identity by 2 arctan of that (test_rodrigues_misorientation with r1 = 0). The step s is the
    misorientation 2 arctan d of neighbours 0 and d."""
    d = sp.symbols("d", positive=True)
    corner = d / 2 * sp.Matrix([1, 1, 1])
    assert is_zero(sp.sqrt(corner.dot(corner)) - sp.sqrt(3) / 2 * d)


def test_cubic_delta_code():
    """anri's cubic_grid_misorientation is srch-cubic-delta."""
    from anri.crystal.orientation import cubic_grid_misorientation

    s = sp.symbols("s", positive=True)
    expr = sp.deg(2 * sp.atan(sp.sqrt(3) / 2 * sp.tan(sp.rad(s) / 2)))
    for step in (1.0, 2.5, 3.0):
        rtol = _rtol(expr, {s: step}, 8)  # radians, tan, /2, *, arctan, *2, degrees
        agree(evaluate(expr, {s: step}), cubic_grid_misorientation, step, rtol=rtol, atol=0)


def test_cubic_zone_corners():
    """(sqrt 2 - 1, sqrt 2 - 1, 3 - 2 sqrt 2) lies on two faces |r_i| = tan(pi / 8) and on the plane sum |r_i| = 1."""
    c = [sp.sqrt(2) - 1, sp.sqrt(2) - 1, 3 - 2 * sp.sqrt(2)]
    assert is_zero(sp.tan(sp.pi / 8) - c[0])
    assert is_zero(sum(c) - 1)
    assert sp.simplify(c[2] - sp.tan(sp.pi / 8)) < 0


th, eta, D = sp.symbols("theta eta Delta", positive=True)
t, sn, cs = sp.tan(th), sp.sin(eta), sp.cos(eta)


def test_tolerance_bounds():
    """From dif-rotation-motion, the largest moves over rotations of size D are D |(1, -tan th / sin eta)| in eta and
    D |(1, cot eta)| = D / |sin eta| in omega (Cauchy-Schwarz). srch-tolerance's bounds hold them: the squares differ
    by (1 + tan^2) 2 tan |cot| + tan^4 cot^2 >= 0 in eta, and by a factor 1 / cos th >= 1 in omega."""
    move_e2 = 1 + t**2 / sn**2
    bound_e2 = (1 + t * cs / sn) ** 2 / sp.cos(th) ** 2  # for 0 < eta < 90 deg, where |cot eta| = cot eta
    assert is_zero(bound_e2 - move_e2 - ((1 + t**2) * 2 * t * cs / sn + t**4 * cs**2 / sn**2))
    assert is_zero(1 + (cs / sn) ** 2 - 1 / sn**2)


def test_ring_width_on_detector():
    """A width h in 2theta, as a displacement across the ring on a flat detector normal to the beam (R = L tan 2th),
    is an angle h / (sin 2th cos 2th) along the ring: dR / R."""
    L, h = sp.symbols("L h", positive=True)
    R = L * sp.tan(2 * th)
    assert is_zero(sp.diff(R, th) / 2 * h / R - h / (sp.sin(2 * th) * sp.cos(2 * th)))


def test_tolerance_code():
    """anri's match_tolerances is srch-tolerance (degrees)."""
    from anri.index.search import match_tolerances

    hr, fr, tth = sp.symbols("h_r frame tth", positive=True)
    e = sp.symbols("eta", real=True)
    tt = sp.rad(tth) / 2
    te = sp.rad(D) / sp.cos(tt) * (1 + sp.tan(tt) * sp.Abs(sp.cot(sp.rad(e)))) + sp.rad(hr) / (
        sp.sin(2 * tt) * sp.cos(2 * tt)
    )
    to = sp.rad(D) / (sp.cos(tt) * sp.Abs(sp.sin(sp.rad(e)))) + sp.rad(fr) / 2
    g = rng(4)
    etas = np.array([[35.0, -120.0, 75.0, 160.0]])
    ring_j = np.array([0, 1, 0, 1])
    ring_tth, ring_hw = np.array([6.0, 9.5]), np.array([0.03, 0.05])
    delta, frame = float(g.uniform(0.5, 2)), 0.25
    want_e, want_o, rtol = [], [], 0.0
    for x, j in zip(etas[0], ring_j):
        vals = {e: x, tth: ring_tth[j], hr: ring_hw[j], D: delta, fr: frame}
        want_e.append(float(evaluate(sp.deg(te), vals)))
        want_o.append(float(evaluate(sp.deg(to), vals)))
        rtol = max(rtol, _rtol(te, vals, 20), _rtol(to, vals, 20))  # ~20 operations, trig included
    args = (etas, ring_j, ring_tth, ring_hw, delta, frame)
    agree(np.array([want_e]), lambda *z: match_tolerances(*z)[0], *args, rtol=rtol, atol=0)
    agree(np.array([want_o]), lambda *z: match_tolerances(*z)[1], *args, rtol=rtol, atol=0)
