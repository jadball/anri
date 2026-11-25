"""Proofs of docs/source/theory/crystal.rst: the reflection list, systematic absences, the Laue group and the
structure factor's s."""

import numpy as np
import sympy as sp

from anri.crystal import lattice, orientation, structure
from tests.theory.common import (
    EPS,
    agree,
    evaluate,
    is_zero,
    metric,
    page_B,
    tol,
    vec,
)

# a space group of each Laue class, its lattice (cosines of alpha, beta, gamma exact; b = a for tetragonal, hexagonal
# and cubic) and the number of rotations crystal.rst gives
LAUE = {
    "m-3m": (225, "cubic", 24),
    "m-3": (200, "cubic", 12),
    "6/mmm": (191, "hexagonal", 12),
    "6/m": (175, "hexagonal", 6),
    "-3m": (164, "hexagonal", 6),
    "-3": (147, "hexagonal", 3),
    "4/mmm": (123, "tetragonal", 8),
    "4/m": (83, "tetragonal", 4),
    "mmm": (47, "orthorhombic", 4),
    "2/m": (10, "monoclinic", 2),
    "-1": (2, "triclinic", 1),
}


def system_metric(system: str) -> sp.Matrix:
    a, b, c = sp.symbols("a b c", positive=True)
    ca, cb, cg = sp.symbols("c_alpha c_beta c_gamma", real=True)
    return {
        "cubic": metric(a, a, a, 0, 0, 0),
        "hexagonal": metric(a, a, c, 0, 0, sp.Rational(-1, 2)),
        "tetragonal": metric(a, a, c, 0, 0, 0),
        "orthorhombic": metric(a, b, c, 0, 0, 0),
        "monoclinic": metric(a, b, c, 0, cb, 0),  # unique axis b
        "triclinic": metric(a, b, c, ca, cb, cg),
    }[system]


def proper_parts(sg: int) -> list:
    """The distinct R det R of a space group's operations, as exact integer matrices."""
    R = np.rint(structure.symmetry_matrices(sg)[:, :3, :3]).astype(int)
    R = R * np.rint(np.linalg.det(R)).astype(int)[:, None, None]
    return [sp.Matrix(m) for m in np.unique(R, axis=0)]


def test_hkl_transform():
    """Reflections transform as h -> R^T h: h . (R x + t) = (R^T h) . x + h . t. If R^T h = h and h . t is not an
    integer, F(h) = exp(2 pi i h . t) F(h) is zero: a systematic absence."""
    h, x, t = vec("h"), vec("x"), vec("t")
    R = sp.Matrix(3, 3, sp.symbols("R0:9", real=True))
    assert is_zero(h.dot(R * x + t) - (R.T * h).dot(x) - h.dot(t))


def test_laue():
    """cry-laue: S = B R^T B^-1 gives S B h = B (R^T h), another reflection, and S^T S = B^-T R G* R^T B^-1, the
    identity exactly when R G* R^T = G* (G* = B^T B). That holds for every operation of each Laue class (on the
    exact metric of its lattice), det(R det R) = 1, and the class has the number of rotations the page gives."""
    B = sp.Matrix([[sp.Symbol(f"B{i}{j}", real=True) if j >= i else 0 for j in range(3)] for i in range(3)])
    R = sp.Matrix(3, 3, sp.symbols("R0:9", real=True))
    S = B * R.T * B.inv()
    assert is_zero(S.T * S - B.inv().T * R * (B.T * B) * R.T * B.inv())
    for name, (sg, system, n) in LAUE.items():
        Gs = system_metric(system).inv()
        ops = proper_parts(sg)
        assert len(ops) == n, name
        for P in ops:
            assert P.det() == 1
            assert is_zero(P * Gs * P.T - Gs), name


def test_laue_code():
    """cry-laue: laue_rotations against B R^T B^-1 (exact B of frm-B) for each Laue class."""
    for name, (sg, system, n) in LAUE.items():
        lp = {"cubic": "4 4 4 90 90 90", "hexagonal": "3 3 5 90 90 120", "tetragonal": "4 4 6 90 90 90"}.get(
            system, {"orthorhombic": "4 5 6 90 90 90", "monoclinic": "4 5 6 90 104 90"}.get(system, "4 5 6 81 97 112")
        )
        a, b, c, al, be, ga = [sp.Rational(x) for x in lp.split()]
        B = page_B(metric(a, b, c, sp.cos(sp.rad(al)), sp.cos(sp.rad(be)), sp.cos(sp.rad(ga))), sp.cos(sp.rad(al)))
        B = sp.N(B, 30)  # exact B, evaluated at 30 digits before the products
        expected = np.array([evaluate(B * P.T * B.inv(), {}) for P in proper_parts(sg)])
        Bv = evaluate(B, {})
        got = orientation.laue_rotations(structure.symmetry_matrices(sg), Bv)
        np.testing.assert_array_equal(got[0], np.eye(3))  # the identity first
        # match as sets. S = B R^T B^-1: entries are bounded by cond(B), and inv(B) errs by cond(B) eps relative,
        # so S errs by at most n ulps of cond(B)^2 (n = 16 covers the two products, as tol says)
        order = [int(np.argmin(np.abs(expected - g).sum((1, 2)))) for g in got]
        assert sorted(order) == list(range(n)), name
        np.testing.assert_allclose(got, expected[order], **tol(np.linalg.cond(Bv) ** 2))


def test_allowed_code():
    """Systematic absences, through allowed_hkls, against the reflection conditions of International Tables A: fcc
    (225) all even or all odd; bcc (229) h + k + l even; Fd-3m (227) as fcc, and 0kl with k + l divisible by 4 (the
    d glide). 222 of Fd-3m is not absent by symmetry: it is zero for the diamond structure only (atoms on 8a), so it
    stays, as crystal.rst says."""
    r = np.arange(-4, 5)
    hkl = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    hkl = hkl[np.any(hkl != 0, 1)]
    odd, even = np.all(hkl % 2 == 1, 1), np.all(hkl % 2 == 0, 1)
    zero = np.any(hkl == 0, 1)
    glide = ~zero | (hkl.sum(1) % 4 == 0)  # with one index zero, the other two sum to 4n (and h00: h = 4n)
    rules = {225: odd | even, 229: hkl.sum(1) % 2 == 0, 227: (odd | even) & glide}
    for sg, rule in rules.items():
        np.testing.assert_array_equal(lattice.allowed_hkls(hkl, structure.symmetry_matrices(sg)), rule)


def test_index_bound():
    """cry-bound: h_i = a_i . g, and |a_i|^2 |g|^2 - (a_i . g)^2 = |a_i x g|^2 >= 0 (Lagrange's identity), so
    |h_i| <= |a_i| d*."""
    a, g = vec("a"), vec("g")
    assert is_zero(a.dot(a) * g.dot(g) - a.dot(g) ** 2 - a.cross(g).dot(a.cross(g)))


def test_reflections_code():
    """reflections lists every h with d* = sqrt(h^T G* h) <= d*max (checked against a wider search than cry-bound's),
    with that d*, for a triclinic cell (P-1: no absences)."""
    lp = [4.1, 5.3, 6.7, 81.0, 97.0, 112.0]
    G = metric(*[sp.Rational(str(x)) for x in lp[:3]], *[sp.cos(sp.rad(sp.Rational(str(x)))) for x in lp[3:]])
    G = sp.N(G, 30)
    Gs = evaluate(G.inv(), {})
    out = structure.reflections(lp, 2, 0.3, 1.0)
    r = np.arange(-12, 13)
    hkl = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    ds = np.sqrt(np.einsum("ni,ij,nj->n", hkl, Gs, hkl))
    inside = (ds <= 1.0) & np.any(hkl != 0, 1)
    assert {tuple(h) for h in hkl[inside]} == {tuple(h) for h in out["hkl"]}
    exact = np.array([float(sp.sqrt((sp.Matrix(h).T * G.inv() * sp.Matrix(h))[0]).evalf(30)) for h in out["hkl"][:20]])
    t = tol(np.linalg.cond(evaluate(G, {})) * exact.max())  # B from G^-1: its condition number times d*
    agree(exact, lambda: out["ds"][:20], **t)


def test_structure_factor_s():
    """cry-F's s: sin(theta) / lambda = d* / 2, by dif-bragg."""
    lam, ds = sp.symbols("lambda d_s", positive=True)
    sin_theta = lam * ds / 2
    assert is_zero(sin_theta / lam - ds / 2)


def test_rings_code():
    """rings: a ring starts at the smallest d* not yet in a ring and takes every d* within tol of it, so rings never
    chain (five d* 5e-5 apart are two rings at tol 1e-4), and each ring's d* is its first member's; exactly."""
    ds = [sp.Rational(1), sp.Rational(100005, 100000), sp.Rational(10001, 10000), sp.Rational(100015, 100000),
          sp.Rational(10002, 10000), sp.Rational(11, 10)]  # fmt: skip
    tol_ = sp.Rational(1, 10**4)
    want, firsts = [], []
    for d in ds:
        if not firsts or d - firsts[-1] > tol_:
            firsts.append(d)
        want.append(len(firsts) - 1)
    ring, ring_ds = structure.rings(np.array([float(d) for d in ds]), tol=1e-4)
    np.testing.assert_array_equal(ring, want)
    np.testing.assert_array_equal(ring_ds, [float(d) for d in firsts])


def test_diamond_sum():
    """cry-F for diamond (Fd-3m, 8 atoms at fcc + (1/4, 1/4, 1/4)): F = f e^(-B s^2) G with |G|^2 = 64 for h, k, l all
    even with h + k + l = 4n, 32 all odd, and 0 otherwise."""
    h = sp.symbols("h k l", integer=True)
    fcc = [(0, 0, 0), (0, sp.Rational(1, 2), sp.Rational(1, 2)), (sp.Rational(1, 2), 0, sp.Rational(1, 2)),
           (sp.Rational(1, 2), sp.Rational(1, 2), 0)]  # fmt: skip
    pos = fcc + [tuple(c + sp.Rational(1, 4) for c in p) for p in fcc]
    G = sum(sp.exp(2 * sp.pi * sp.I * sum(a * b for a, b in zip(h, p))) for p in pos)
    for hkl, want in (((2, 2, 0), 64), ((4, 0, 0), 64), ((1, 1, 1), 32), ((3, 1, 1), 32), ((2, 0, 0), 0),
                      ((2, 2, 2), 0), ((1, 1, 0), 0), ((4, 4, 2), 0)):  # fmt: skip
        g = G.subs(dict(zip(h, hkl)))
        assert sp.simplify(g * sp.conjugate(g)) == want, hkl


def test_structure_factors_code():
    """structure_factors (Dans Diffraction's evaluation of cry-F) for silicon: zero where test_diamond_sum's G is,
    and equal at reflections of equal d* and equal |G| (333 and 511), whatever the form factor."""
    import os
    import warnings

    import Dans_Diffraction

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xtl = Dans_Diffraction.Crystal(os.path.join(os.path.dirname(__file__), "..", "data", "cif", "Si.cif"))
        hkl = np.array([[1, 1, 1], [2, 2, 0], [2, 0, 0], [2, 2, 2], [1, 1, 0], [4, 4, 2], [3, 3, 3], [5, 1, 1]])
        F2 = structure.structure_factors(xtl, hkl, 0.3)
    # a sum of 8 terms of size f in float64 errs by at most 8 x 16 eps f, and f^2 <= F2(allowed, all odd) / 32
    zero = (8 * 16 * EPS) ** 2 * F2[0] / 32
    np.testing.assert_array_less(F2[2:6], zero)
    # the same s, so the same f, B and dispersion: 16 ulps of each sum, twice in |F|^2
    np.testing.assert_allclose(F2[6], F2[7], rtol=2 * 2 * 8 * 16 * EPS, atol=0)

