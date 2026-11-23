"""Helpers for the theory pages' proofs: each test proves an equation of docs/source/theory with sympy, or checks
that the function computing it agrees with the proved expression.

Two kinds of test, named in the pages' **Proof:** lines:

- a symbolic proof: an identity simplified to zero (`is_zero`), or an exact rank over the rationals (`rank`);
- a code check: the proved expression, evaluated by sympy at a few random points, against the anri function in
  float64 (`agree`).
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal

import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix

from anri.maths.utils import float64


def is_zero(expr: sp.Expr | sp.MatrixBase) -> bool:
    """True if an expression (or every entry of a matrix) simplifies to zero."""
    if isinstance(expr, sp.MatrixBase):
        return all(is_zero(e) for e in expr)
    return sp.simplify(sp.expand(expr)) == 0


def rank(M: sp.MatrixBase) -> int:
    """Exact rank over the rationals (entries must be rational)."""
    return DomainMatrix.from_Matrix(M).convert_to(sp.QQ).rank()


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def evaluate(expr: sp.Expr | sp.MatrixBase, values: dict) -> np.ndarray:
    """A sympy expression at numeric values, as float64 (evaluated by sympy at 30 digits, not lambdified)."""
    out = sp.N(sp.sympify(expr).subs(values), 30)
    return np.array(out.tolist() if isinstance(out, sp.MatrixBase) else out, dtype=np.float64)


def agree(expected: np.ndarray, fn: object, *args: object, rtol: float, atol: float) -> None:
    """Assert that fn(*args), computed by anri in float64, equals the value of the proved expression, within a
    tolerance the caller derives (ulps times the condition, e.g. `tol`): there is no default."""
    with float64():
        got = np.asarray(fn(*args), dtype=np.float64)  # ty: ignore[call-non-callable]
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


EPS = float(np.finfo(np.float64).eps)


def tol(scale: float, n: int = 16) -> dict:
    """agree's tolerance for float64 code against the exact value: n ulps of `scale`, the largest term on the code's
    path to the result (times the condition number where one enters, as the callers state). Each product or sum rounds
    by at most half an ulp of its result, so a path of k operations errs by at most k eps scale (Higham's gamma_k);
    n = 16 bounds the longest path here, a chain of three 3 x 3 matrix products."""
    return {"rtol": 0.0, "atol": n * EPS * scale}


def ev(expr: sp.MatrixBase, values: dict) -> np.ndarray:
    """A column vector's value, flat."""
    return evaluate(expr, values).reshape(-1)


def rx(t: sp.Expr) -> sp.Matrix:
    return sp.Matrix([[1, 0, 0], [0, sp.cos(t), -sp.sin(t)], [0, sp.sin(t), sp.cos(t)]])


def ry(t: sp.Expr) -> sp.Matrix:
    return sp.Matrix([[sp.cos(t), 0, sp.sin(t)], [0, 1, 0], [-sp.sin(t), 0, sp.cos(t)]])


def rz(t: sp.Expr) -> sp.Matrix:
    return sp.Matrix([[sp.cos(t), -sp.sin(t), 0], [sp.sin(t), sp.cos(t), 0], [0, 0, 1]])


def skew(v: sp.Matrix) -> sp.Matrix:
    return sp.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def vec(name: str) -> sp.Matrix:
    return sp.Matrix(sp.symbols(f"{name}_x {name}_y {name}_z", real=True))


def cayley(k: sp.Matrix) -> sp.Matrix:
    """A rotation, rational in the three entries of k: every rotation but those by 180 degrees."""
    K = skew(k)
    return (sp.eye(3) - K).inv() * (sp.eye(3) + K)


def metric(a, b, c, ca, cb, cg) -> sp.Matrix:
    """The metric tensor G = [a_i . a_j] of lattice parameters (cosines of the angles)."""
    return sp.Matrix(
        [[a * a, a * b * cg, a * c * cb], [a * b * cg, b * b, b * c * ca], [a * c * cb, b * c * ca, c * c]]
    )


def page_B(G: sp.Matrix, ca: sp.Expr) -> sp.Matrix:
    """frm-B, with the reciprocal parameters read off G* = G^-1: a* = sqrt(G*_00), b* cos(gamma*) = G*_01 / a*, ..."""
    Gs = G.inv()
    ast = sp.sqrt(Gs[0, 0])
    return sp.Matrix(
        [
            [ast, Gs[0, 1] / ast, Gs[0, 2] / ast],
            [0, sp.sqrt(Gs[1, 1] - Gs[0, 1] ** 2 / Gs[0, 0]), -sp.sqrt(Gs[2, 2] - Gs[0, 2] ** 2 / Gs[0, 0]) * ca],
            [0, 0, 1 / sp.sqrt(G[2, 2])],
        ]
    )


def beam_frame(psi: sp.Expr, tilt: sp.Expr) -> tuple:
    """A beam at azimuth psi from x and tilt out of the horizontal plane, and the beam frame of frames.rst."""
    k = sp.Matrix([sp.cos(tilt) * sp.cos(psi), sp.cos(tilt) * sp.sin(psi), sp.sin(tilt)])
    zk = sp.Matrix([0, 0, 1]).cross(k)
    e_h = sp.simplify(zk / sp.cos(tilt))  # |z x k| = cos(tilt) for |tilt| < 90 degrees
    return k, e_h, k.cross(e_h)


# ---- the threshold's group rule (recording.rst, moments.rst) ----


def rounds_to(expr: sp.Expr, printed: str) -> bool:
    """A number quoted in a page is the exact rounding, to its printed digits, of the proved expression."""
    return Decimal(str(sp.N(expr, 30))).quantize(Decimal(printed), ROUND_HALF_EVEN) == Decimal(printed)


N8 = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]


def animals(p: tuple, k: int) -> set:
    """Every 8-connected set of k pixels holding p."""
    out = {frozenset([p])}
    for _ in range(k - 1):
        out = {s | {(q[0] + a, q[1] + b)} for s in out for q in s for a, b in N8 if (q[0] + a, q[1] + b) not in s}
    return out


def boundary(s: frozenset) -> set:
    return {(q[0] + a, q[1] + b) for q in s for a, b in N8} - set(s)


def p_at_least(c: int, lam: sp.Symbol) -> sp.Expr:
    """P(n >= c) for n ~ Poisson(lam), as a finite sum."""
    return 1 - sum(sp.exp(-lam) * lam**j / sp.factorial(j) for j in range(max(c, 0)))


def p_group(p: tuple, k: int, a: dict) -> sp.Expr:
    """P(p's group has k pixels | p kept): sum over k-animals S holding p of prod_S\\p a prod_dS (1 - a). Pixels not
    in a (off the detector, masked) are never kept."""
    tot = 0
    for s in animals(p, k):
        if not all(q in a for q in s - {p}):
            continue
        term = sp.Mul(*[a[q] for q in s - {p}])
        term *= sp.Mul(*[1 - a[q] for q in boundary(s) if q in a])
        tot += term
    return tot
