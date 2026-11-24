"""Proofs for docs/source/theory/intensity.rst: Lorentz, polarisation, the beam's profile over a voxel and its fold
into a peak, and the absorption paths.

Tolerances of the code checks are derived where they are stated: float64 has eps = 2.2e-16, and an expression of n
operations with condition number c is good to about n c eps.
"""

from __future__ import annotations

import math
import unittest

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import sympy as sp

from anri.diffract.factors import lorentz, polarisation, polarisation_extinction
from anri.fwd.render import _beam_fold, _dty_at, _frame_weights, make_row
from anri.geom.beam import beam_offsets, beam_weight, chord_weight
from anri.geom.gonio import sample_to_lab
from anri.geom.paths import path_lengths, transmission
from anri.maths.utils import float64, ramp_integral, ramp_integral2
from tests.theory.common import (
    EPS,
    agree,
    beam_frame,
    ev,
    evaluate,
    is_zero,
    rng,
    rounds_to,
    rx,
    ry,
    rz,
    tol,
)


def Phi(z: sp.Expr) -> sp.Expr:
    return (1 + sp.erf(z / sp.sqrt(2))) / 2


def phi(z: sp.Expr) -> sp.Expr:
    return sp.exp(-(z**2) / 2) / sp.sqrt(2 * sp.pi)


def R1(x: sp.Expr, s: sp.Expr) -> sp.Expr:
    """:eq:`int-ramp`, first antiderivative of Phi(x / s)."""
    return x * Phi(x / s) + s * phi(x / s)


def R2(x: sp.Expr, s: sp.Expr) -> sp.Expr:
    """:eq:`int-ramp`, second antiderivative of Phi(x / s)."""
    return (x**2 + s**2) / 2 * Phi(x / s) + x * s / 2 * phi(x / s)


def _k_out(th: sp.Expr, eta: sp.Expr) -> sp.Matrix:
    """Diffracted direction for a beam along x (:eq:`frm-eta`)."""
    return sp.Matrix([sp.cos(2 * th), -sp.sin(2 * th) * sp.sin(eta), sp.sin(2 * th) * sp.cos(eta)])


def _random_k_out(r: np.random.Generator, n: int) -> np.ndarray:
    tth, eta = np.radians(r.uniform(2, 40, n)), np.radians(r.uniform(-170, 170, n))
    eta = np.where(np.abs(np.sin(eta)) < 0.2, eta + 0.5, eta)
    return np.stack([np.cos(tth), -np.sin(tth) * np.sin(eta), np.sin(tth) * np.cos(eta)], 1)


# ---------------------------------------------------------------------------------------------- int-lorentz


def test_lorentz_vertical():
    """n . (k_in x k_out) = -sin 2theta sin eta for a vertical axis and a beam along x."""
    th, eta = sp.symbols("theta eta", real=True)
    triple = sp.Matrix([0, 0, 1]).dot(sp.Matrix([1, 0, 0]).cross(_k_out(th, eta)))
    assert is_zero(triple + sp.sin(2 * th) * sp.sin(eta))


def test_lorentz_code():
    """lorentz against 1 / |n . (k_in x k_out)| with unit vectors, at the code's own float64 inputs."""
    k = sp.Matrix(sp.symbols("k0:3", real=True))
    expr = sp.sqrt(k.dot(k)) / sp.Abs(sp.Matrix([0, 0, 1]).dot(sp.Matrix([1, 0, 0]).cross(k)))
    for v in _random_k_out(rng(1), 8):
        # ~15 operations of condition O(1) (|sin eta| > 0.2 keeps the triple product away from 0): 64 ulps
        with float64():
            got = float(lorentz(jnp.array([1.0, 0, 0]), jnp.asarray(v), jnp.array([0, 0, 1.0])))
        np.testing.assert_allclose(got, float(evaluate(expr, dict(zip(k, v)))), rtol=64 * EPS)


# ---------------------------------------------------------------------------------------------- int-pol


def test_polarisation_forms():
    """:eq:`int-pol` is (1 + cos^2 2theta) / 2 unpolarised, and Kahn et al.'s form with rho = eta + 90 deg."""
    th, eta, p = sp.symbols("theta eta p", real=True)
    k = _k_out(th, eta)
    P = (1 + p) / 2 * (1 - k[1] ** 2) + (1 - p) / 2 * (1 - k[2] ** 2)  # e_h = y, e_v = z
    assert is_zero(P.subs(p, 0) - (1 + sp.cos(2 * th) ** 2) / 2)
    rho = eta + sp.pi / 2
    kahn = (1 + sp.cos(2 * th) ** 2) / 2 - p * sp.cos(2 * rho) * sp.sin(2 * th) ** 2 / 2
    assert is_zero(P - kahn)


def test_polarisation_code():
    """polarisation against :eq:`int-pol`, at the code's own float64 inputs."""
    k, p = sp.Matrix(sp.symbols("k0:3", real=True)), sp.symbols("p")
    n2 = k.dot(k)
    expr = (1 + p) / 2 * (1 - k[1] ** 2 / n2) + (1 - p) / 2 * (1 - k[2] ** 2 / n2)
    for v, f in zip(_random_k_out(rng(2), 6), (0.0, 0.3, 0.9, 1.0, 0.5, 0.99)):
        with float64():
            got = float(polarisation(jnp.array([1.0, 0, 0]), jnp.asarray(v), f))
        # ~15 operations, each term in [0, 1] and their sum >= cos^2 2theta > 0.5: 64 ulps
        np.testing.assert_allclose(got, float(evaluate(expr, {**dict(zip(k, v)), p: f})), rtol=64 * EPS)
    # a tilted beam: int-pol in its own frame (frames.rst), the same 64 ulps plus the frame's rounding (16 more)
    psi, tilt = 0.3, -0.15
    kh, e_h, e_v = beam_frame(sp.Float(psi, 30), sp.Float(tilt, 30))
    kin = 2.0 * np.array([float(x) for x in ev(kh, {})])
    for v, f in zip(_random_k_out(rng(27), 4), (0.0, 0.9, 1.0, 0.5)):
        o = sp.Matrix([sp.Float(x, 30) for x in v])
        R = (rz(sp.Float(psi, 30)) * ry(-sp.Float(tilt, 30)))  # takes x to the beam, y to e_h, z to e_v
        o = R * o
        u = o / sp.sqrt(o.dot(o))
        want = float(sp.N((1 + f) / 2 * (1 - u.dot(e_h) ** 2) + (1 - f) / 2 * (1 - u.dot(e_v) ** 2), 30))
        with float64():
            got = float(polarisation(jnp.asarray(kin), jnp.asarray(ev(o, {})), f))
        np.testing.assert_allclose(got, want, rtol=80 * EPS)


def test_extinction_polarisation():
    """:eq:`int-ext-pol` with no extinction is :eq:`int-pol`: w + (1 - w) cos^2 2theta, w = sum p_e (e . n)^2."""
    th, eta, p = sp.symbols("theta eta p", real=True)
    k = _k_out(th, eta)
    n = sp.Matrix([1, 0, 0]).cross(k) / sp.sin(2 * th)  # unit normal of the scattering plane (0 < 2theta < pi)
    assert is_zero(n.dot(n) - 1)
    w = (1 + p) / 2 * n[1] ** 2 + (1 - p) / 2 * n[2] ** 2
    P = (1 + p) / 2 * (1 - k[1] ** 2) + (1 - p) / 2 * (1 - k[2] ** 2)
    assert is_zero(w + (1 - w) * sp.cos(2 * th) ** 2 - P)


def test_extinction_polarisation_code():
    """polarisation_extinction with x = X = 0 (y = 1) is polarisation."""
    for v, f in zip(_random_k_out(rng(3), 4), (0.0, 0.5, 0.9, 1.0)):
        with float64():
            a = float(polarisation_extinction(jnp.array([1.0, 0, 0]), jnp.asarray(v), f, 0.0, 0.0, True))
            b = float(polarisation(jnp.array([1.0, 0, 0]), jnp.asarray(v), f))
        # two ~20-operation evaluations of the same O(1) quantity, one through arccos and cos^2: 128 ulps
        np.testing.assert_allclose(a, b, rtol=128 * EPS)


# ---------------------------------------------------------------------------------------------- int-profile, int-ramp


def test_ramp_integrals():
    """R1' = Phi(x / s), R2' = R1, both vanishing as x -> -inf."""
    x = sp.symbols("x", real=True)
    s = sp.symbols("sigma", positive=True)
    assert is_zero(sp.diff(R1(x, s), x) - Phi(x / s))
    assert is_zero(sp.diff(R2(x, s), x) - R1(x, s))
    assert sp.limit(R1(x, s), x, -sp.oo) == 0
    assert sp.limit(R2(x, s), x, -sp.oo) == 0


def test_profile():
    """:eq:`int-profile` is a flat top averaged over Gaussians: its cumulative distribution is
    (R1(u + w/2) - R1(u - w/2)) / w, and it integrates to 1."""
    u, c = sp.symbols("u c", real=True)
    s, w = sp.symbols("sigma w", positive=True)
    p = (Phi((u + w / 2) / s) - Phi((u - w / 2) / s)) / w
    gauss = phi((u - c) / s) / s
    assert is_zero(sp.integrate(gauss, (c, -w / 2, w / 2)) / w - p)
    cdf = (R1(u + w / 2, s) - R1(u - w / 2, s)) / w
    assert is_zero(sp.diff(cdf, u) - p)
    assert sp.limit(cdf, u, sp.oo) == 1 and sp.limit(cdf, u, -sp.oo) == 0


def test_fwhm():
    """:eq:`int-fwhm`: a Gaussian of sigma = FWHM / (2 sqrt(2 ln 2)) is at half its maximum at FWHM / 2."""
    f = sp.symbols("f", positive=True)
    s = f / (2 * sp.sqrt(2 * sp.log(2)))
    assert sp.simplify(phi(f / 2 / s) / phi(0) - sp.Rational(1, 2)) == 0


def test_fwhm_code():
    """fwhm_to_sigma and sigma_to_fwhm against :eq:`int-fwhm`."""
    from anri.maths.utils import fwhm_to_sigma, sigma_to_fwhm

    f = np.array([0.25, 1.0, 3.7])
    c = 2 * sp.sqrt(2 * sp.log(2))
    s = np.array([float(sp.N(sp.Float(x, 30) / c, 30)) for x in f])
    # one product or quotient by a rounded constant: two ulps of the result
    agree(s, fwhm_to_sigma, f, rtol=4 * EPS, atol=0)
    agree(np.array([float(sp.N(sp.Float(x, 30) * c, 30)) for x in s]), sigma_to_fwhm, s, rtol=4 * EPS, atol=0)


def test_ramp_integrals_code():
    """ramp_integral and ramp_integral2 against :eq:`int-ramp`."""
    x, s = sp.symbols("x sigma", real=True)
    for xv, sv in [(-1.3, 0.7), (0.0, 1.0), (0.4, 0.2), (2.5, 1.1), (-0.2, 3.0)]:
        vals = {x: xv, s: sv}
        e1, e2 = float(evaluate(R1(x, s), vals)), float(evaluate(R2(x, s), vals))
        with float64():
            g1, g2 = float(ramp_integral(xv, sv)), float(ramp_integral2(xv, sv))
        # a sum of two terms of ~6 operations each: condition (|term 1| + |term 2|) / |sum|
        t = [float(evaluate(e, vals)) for e in (x * Phi(x / s), s * phi(x / s), (x**2 + s**2) / 2 * Phi(x / s),
                                                 x * s / 2 * phi(x / s))]  # fmt: skip
        np.testing.assert_allclose(g1, e1, rtol=16 * EPS * (abs(t[0]) + abs(t[1])) / abs(e1))
        np.testing.assert_allclose(g2, e2, rtol=16 * EPS * (abs(t[2]) + abs(t[3])) / abs(e2))


def _ext_coeffs(c: sp.Expr) -> dict:
    """int-ext-ab: the A, B (and the factor on x) of each extinction factor, at c = cos 2 theta."""
    b_l = sp.Piecewise((sp.Rational(15, 100) - sp.Rational(2, 10) * (sp.Rational(3, 4) - c) ** 2, c > 0),
                       (-sp.Rational(45, 100) * c, True))  # fmt: skip
    return {
        "primary": (2, sp.Rational(20, 100) + sp.Rational(45, 100) * c,
                    sp.Rational(22, 100) - sp.Rational(12, 100) * (sp.Rational(1, 2) - c) ** 2),
        "gaussian": (sp.Rational(212, 100), sp.Rational(58, 100) + sp.Rational(48, 100) * c + sp.Rational(24, 100) * c**2,
                     sp.Rational(2, 100) - sp.Rational(25, 1000) * c),
        "lorentzian": (2, sp.Rational(25, 1000) + sp.Rational(285, 1000) * c, b_l),
    }  # fmt: skip


def _y(x: sp.Expr, k: sp.Expr, A: sp.Expr, B: sp.Expr) -> sp.Expr:
    """int-ext-y."""
    return (1 + k * x + A * x**2 / (1 + B * x)) ** sp.Rational(-1, 2)


def test_extinction_limits():
    """int-ext-y: y = 1 without extinction; y falls with x wherever A >= 0 and 1 + B x > 0; and y^2 x -> B / A, so
    y |F|^2 grows as |F| in a perfect crystal (x is proportional to |F|^2)."""
    x, k, A, B = sp.symbols("x k A B", positive=True)
    y = _y(x, k, A, B)
    assert y.subs(x, 0) == 1
    inner = 1 + k * x + A * x**2 / (1 + B * x)
    assert is_zero(sp.diff(inner, x) - (k + A * x * (2 + B * x) / (1 + B * x) ** 2))  # > 0: y = inner^(-1/2) falls
    assert sp.simplify(sp.limit(y**2 * x, x, sp.oo) - B / (A + k * B)) == 0


def test_extinction_code():
    """extinction against int-ext-y with int-ext-ab, primary times Gaussian and times Lorentzian secondary, either side
    of the Lorentzian's c = 0."""
    from anri.diffract.factors import extinction

    for x, X, tth in ((0.7, 2.5, 0.3), (12.0, 0.4, 1.2), (3.0, 25.0, 2.0)):
        c = sp.cos(sp.Float(tth, 30))
        co = _ext_coeffs(c)
        for gauss in (True, False):
            parts = [_y(sp.Float(x, 30), *co["primary"]), _y(sp.Float(X, 30), *co["gaussian" if gauss else "lorentzian"])]
            want = float(sp.N(parts[0] * parts[1], 30))
            # each factor: ~16 operations on positive terms (c's own rounding moves A, B by an ulp), its square root
            # halving the relative error of the sum; the product adds one more
            with float64():
                got = float(extinction(x, X, tth, gauss))
            np.testing.assert_allclose(got, want, rtol=3 * 16 * EPS, atol=0)


def test_extinction_parameters_code():
    """extinction_parameters against int-ext-q and int-ext-x (K = 1, radii in micrometres = 1e4 angstrom), Gaussian and
    Lorentzian mosaics; without absorption the mean-path ratio is 1 exactly (the same quadrature's two means)."""
    from anri.diffract.factors import _sphere_path_ratio, extinction_parameters

    r_e = sp.Float("2.8179403262e-5", 30)  # CODATA 2018, angstrom
    F2, V, lam, r, R, g = 4600.0, 160.2, 0.19, 0.5, 20.0, 300.0
    tth = np.array([0.1, 0.6, 1.4])
    for gauss in (True, False):
        want_x, want_X = [], []
        for t in tth:
            s2 = sp.sin(sp.Float(t, 30))
            Q = (r_e * sp.sqrt(sp.Float(F2, 30)) / sp.Float(V, 30)) ** 2 * sp.Float(lam, 30) ** 3 / s2
            a = 3 * sp.Float(r * 1e4, 30) * s2 / (2 * sp.Float(lam, 30))
            alpha = a / sp.sqrt(1 + a**2 / (2 * sp.Float(g, 30) ** 2)) if gauss else a / (1 + 2 * a / (3 * sp.Float(g, 30)))
            want_x.append(float(sp.N(sp.Rational(2, 3) * Q * a * 3 * sp.Float(r * 1e4, 30) / 2, 30)))
            want_X.append(float(sp.N(sp.Rational(2, 3) * Q * alpha * 3 * sp.Float(R * 1e4, 30) / 2, 30)))
        with float64():
            x, X = extinction_parameters(np.full(3, F2), V, lam, tth, r, R, g, gauss)
            one = np.array([float(_sphere_path_ratio(0.0, t)) for t in tth])
        # products and quotients of ~15 rounded factors: 16 ulps relative; the ratio's two quadratures sum 1728
        # positive terms each (16 ulps of the sum relative, each), and their quotient is 1 up to those
        np.testing.assert_allclose(np.asarray(x), want_x, rtol=16 * EPS, atol=0)
        np.testing.assert_allclose(np.asarray(X), want_X, rtol=16 * EPS + 2 * 1728 * EPS, atol=0)
        np.testing.assert_allclose(one, 1.0, rtol=2 * 1728 * EPS, atol=0)


def test_lorentz_polarisation_code():
    """lorentz_polarisation: int-lor about the omega axis tilted by wedge and chi (frm-stack's W C z), times int-pol."""
    from anri.diffract.factors import lorentz_polarisation

    k = sp.Matrix(sp.symbols("k0:3", real=True))
    w, ch, p = 2.0, -1.5, 0.9
    n = ry(sp.rad(sp.Float(w, 30))) * rx(sp.rad(sp.Float(ch, 30))) * sp.Matrix([0, 0, 1])
    L = sp.sqrt(k.dot(k)) / sp.Abs(n.dot(sp.Matrix([1, 0, 0]).cross(k)))
    n2 = k.dot(k)
    P = (1 + p) / 2 * (1 - k[1] ** 2 / n2) + (1 - p) / 2 * (1 - k[2] ** 2 / n2)
    for v in _random_k_out(rng(24), 4):
        want = float(evaluate(L * P, dict(zip(k, v))))
        # as test_lorentz_code and test_polarisation_code (|sin eta| > 0.2), and the axis's two rotations: 64 + 64 ulps
        with float64():
            got = float(lorentz_polarisation(jnp.array([1.0, 0, 0]), jnp.asarray(v), w, ch, p))
        np.testing.assert_allclose(got, want, rtol=128 * EPS, atol=0)


# ---------------------------------------------------------------------------------------------- int-chord


def _ramp(x: sp.Expr, region_sign: int) -> sp.Expr:
    """max(x, 0), given the sign of x in a region."""
    return x if region_sign > 0 else sp.Integer(0)


def test_chord_trapezoid():
    """A square of side a turned by omega: its chord along the beam is the trapezoid of half-widths
    A = a (c + s) / 2, B = a (c - s) / 2 (c = |cos|, s = |sin|, c >= s by symmetry) and plateau h = a / c, area a^2;
    and the trapezoid is h / (A - B) times ramp(y + A) - ramp(y + B) - ramp(y - B) + ramp(y - A)."""
    a = sp.symbols("a", positive=True)
    om = sp.symbols("omega", positive=True)  # 0 < omega < 45 deg
    c, s = sp.cos(om), sp.sin(om)
    A, B, h = a * (c + s) / 2, a * (c - s) / 2, a / c
    assert is_zero(h * (A + B) - a**2)  # the area of a trapezoid
    # the chord at y = 0 crosses the square from one face to the opposite one: a / cos(omega)
    y = sp.symbols("y", real=True)
    trap = {-2: 0, -1: h * (y + A) / (A - B), 0: h, 1: h * (A - y) / (A - B), 2: 0}  # regions y < -A, ..., y > A
    signs = {-2: (-1, -1, -1, -1), -1: (1, -1, -1, -1), 0: (1, 1, -1, -1), 1: (1, 1, 1, -1), 2: (1, 1, 1, 1)}
    for region, (s1, s2, s3, s4) in signs.items():
        ramps = _ramp(y + A, s1) - _ramp(y + B, s2) - _ramp(y - B, s3) + _ramp(y - A, s4)
        assert is_zero(h / (A - B) * ramps - trap[region])


def test_chord_convolution():
    """A ramp convolved with the Gaussian profile is R1, and with the flat top (R2(+) - R2(-)) / w: the four ramps
    give chord_weight's four terms."""
    v, c, x = sp.symbols("v c x", real=True)
    s, w = sp.symbols("sigma w", positive=True)
    # int_0^inf x phi_sigma(v - x) dx = R1(v)
    conv = sp.integrate(x * phi((v - x) / s) / s, (x, 0, sp.oo))
    assert is_zero((conv - R1(v, s)).rewrite(sp.erf))
    # the flat top is the average of Gaussians centred over [-w/2, w/2] (test_profile): average R1 the same way
    assert is_zero(sp.integrate(R1(v - c, s), (c, -w / 2, w / 2)) - (R2(v + w / 2, s) - R2(v - w / 2, s)))


def test_profile_even():
    """:eq:`int-even`: P1(x) + P1(-x) = 1 and P2(x) - P2(-x) = x for the Gaussian and the flat top (P2 - P2(-x) has
    derivative P1(x) + P1(-x) = 1 and vanishes at 0); so the four terms of :eq:`int-chord` at -u are those at u."""
    x, u, A, B = sp.symbols("x u A B", real=True)
    s, w = sp.symbols("sigma w", positive=True)
    gauss = (Phi(x / s), R1(x, s))
    flat = ((R1(x + w / 2, s) - R1(x - w / 2, s)) / w, (R2(x + w / 2, s) - R2(x - w / 2, s)) / w)
    for P1, P2 in (gauss, flat):
        assert is_zero(P1 + P1.subs(x, -x) - 1)
        assert is_zero(sp.diff(P2, x) - P1)
        odd = P2 - P2.subs(x, -x)
        assert is_zero(sp.diff(odd, x) - 1) and is_zero(odd.subs(x, 0))
    # any P2 with P2(y) = P2(-y) + y: the linear parts cancel in the four terms
    P2f, Q = sp.Function("P2"), sp.Function("Q")

    def terms(v: sp.Expr, F: object) -> sp.Expr:
        return F(v + A) - F(v + B) - F(v - B) + F(v - A)

    assert is_zero(terms(-u, lambda y: Q(-y) + y) - terms(u, Q))
    assert terms(u, P2f) != 0  # (the terms are not trivially zero)


def _monic_legendre_mean_square(n: int) -> sp.Expr:
    """The mean over [-1, 1] of prod (x - x_i)^2 over the n Gauss-Legendre nodes: the monic Legendre polynomial."""
    x = sp.symbols("x", real=True)
    P = sp.legendre(n, x)
    monic = sp.expand(P / sp.Poly(P, x).LC())
    return sp.integrate(monic**2, (x, -1, 1)) / 2


def test_chord_mean():
    """:eq:`int-chord-mean`. The four terms are the integral of f(t) = P1(u + t) - P1(u - t) over [B, A] (they vanish
    at A = B, and their derivative in A is f(A)). A Gauss rule integrates the Hermite interpolant of a function g at
    its nodes exactly (degree 2n - 1), and g minus it is g^(2n)(xi) / (2n)! prod (x - x_i)^2, of one sign: the error
    is g^(2n)(xi) / (2n)! times the mean of prod (x - x_i)^2, the monic Legendre polynomial squared. With
    g = f(m + L x / 2) that gives K_n L^(2n) M_2n; d_m^i d_L^j E is the same functional applied to
    g = (x / 2)^j f^(i + j)(m + L x / 2), whose 2n-th derivative is Leibniz's sum (|x| <= 1). phi^(k) = (-1)^k He_k
    phi, and |He_k phi| peaks where its derivative -He_(k+1) phi vanishes."""
    u, A, B = sp.symbols("u A B", real=True)
    s, w = sp.symbols("sigma w", positive=True)
    for P1, P2 in ((lambda y: Phi(y / s), lambda y: R1(y, s)),
                   (lambda y: (R1(y + w / 2, s) - R1(y - w / 2, s)) / w,
                    lambda y: (R2(y + w / 2, s) - R2(y - w / 2, s)) / w)):  # fmt: skip
        four = P2(u + A) - P2(u + B) - P2(u - B) + P2(u - A)
        assert is_zero(four.subs(A, B))
        assert is_zero(sp.diff(four, A) - (P1(u + A) - P1(u - A)))
    n, L, m, x = sp.symbols("n L m x", real=True)
    K = sp.factorial(n) ** 4 / ((2 * n + 1) * sp.factorial(2 * n) ** 3)
    for k in range(1, 7):  # c_n / (2n)! (L / 2)^(2n) = K_n L^(2n)
        assert is_zero(
            (L / 2) ** (2 * k) * _monic_legendre_mean_square(k) / sp.factorial(2 * k) - K.subs(n, k) * L ** (2 * k)
        )
    # Leibniz: the 2n-th derivative in x of (x / 2)^j f^(i+j)(m + L x / 2), n = 6, i + j <= 2, as the bound's sum
    f = sp.Function("f")
    nn = 6
    for i, j in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)):
        g = (x / 2) ** j * sp.diff(f(m + L * x / 2), m, i + j)
        lhs = sp.diff(g, x, 2 * nn)
        rhs = sum(sp.binomial(2 * nn, k) * sp.diff((x / 2) ** j, x, k) * (L / 2) ** (2 * nn - k)
                  * sp.diff(f(m + L * x / 2), m, 2 * nn + i + j - k) for k in range(j + 1))  # fmt: skip
        assert is_zero(lhs - rhs)
        for k in range(j + 1):  # |d^k (x / 2)^j| <= j! / (j - k)! 2^-j on [-1, 1], and c_n / (2n)! = K_n 2^(2n)
            assert is_zero(sp.factorial(j) / sp.factorial(j - k) / 2**j * (L / 2) ** (2 * nn - k) * 2 ** (2 * nn)
                           - sp.factorial(j) / sp.factorial(j - k) * sp.Integer(2) ** (k - j) * L ** (2 * nn - k))  # fmt: skip
    # probabilists' Hermite polynomials by their recurrence; max |phi^(k)| at a root of He_(k+1)
    He = [sp.Integer(1), x]
    for k in range(1, 15):
        He.append(sp.expand(x * He[k] - k * He[k - 1]))
    for k in (11, 12, 13):
        assert is_zero(sp.diff(phi(x), x, k) - (-1) ** k * He[k] * phi(x))
        assert is_zero(sp.diff(He[k] * phi(x), x) + He[k + 1] * phi(x))
    M = {k + 1: 2 * max(abs(sp.N((He[k] * phi(x)).subs(x, r), 40)) for r in sp.Poly(He[k + 1], x).real_roots())
         for k in (11, 12, 13)}  # M_k sigma^k  # fmt: skip
    K6 = K.subs(n, nn)
    assert rounds_to(K6 * M[12], "4.37e-13")  # the error, times (L / sigma)^12

    def bound(j: int, r: sp.Expr) -> sp.Expr:  # |d_L^j E| sigma^j / h at L = r sigma
        terms = (sp.binomial(2 * nn, k) * sp.factorial(j) / sp.factorial(j - k) * sp.Integer(2) ** (k - j)
                 * r ** (2 * nn - k) * M[2 * nn + j - k] for k in range(j + 1))  # fmt: skip
        return K6 * sum(terms)

    for r, quoted in (
        (sp.Rational(5, 2), ("2.6e-8", "1.7e-7", "1.1e-6")),
        (sp.Rational(1, 2), ("1.1e-16", "2.8e-15", "6.6e-14")),
    ):
        assert all(rounds_to(bound(j, r), q) for j, q in enumerate(quoted))  # at the switches
    # the code's nodes and weights are the 6-point rule's
    xs, ws = np.polynomial.legendre.leggauss(6)
    np.testing.assert_allclose(np.polynomial.legendre.legval(xs, [0] * 6 + [1]), 0, atol=8 * EPS)
    np.testing.assert_allclose(
        [np.sum(ws * xs**k) for k in range(12)], [2 / (k + 1) * (k % 2 == 0) for k in range(12)], atol=16 * EPS
    )


def _chord_expr(u: sp.Expr, om_deg: float, a: float, s: sp.Expr, w: float) -> sp.Expr:
    """:eq:`int-chord` (the four ramps convolved with the profile), or its rectangle limit at omega = 0."""
    c, sn = abs(sp.cos(sp.rad(sp.nsimplify(om_deg)))), abs(sp.sin(sp.rad(sp.nsimplify(om_deg))))
    A, B = a * (c + sn) / 2, a * abs(c - sn) / 2
    h = a**2 / (A + B)
    if w == 0:
        S2, S1 = (lambda t: R1(t, s)), (lambda t: Phi(t / s))
    else:

        def S2(t: sp.Expr) -> sp.Expr:
            return (R2(t + w / 2, s) - R2(t - w / 2, s)) / w

        def S1(t: sp.Expr) -> sp.Expr:
            return (R1(t + w / 2, s) - R1(t - w / 2, s)) / w

    if A == B:
        return h * (S1(u + A) - S1(u - A))
    return h / (A - B) * (S2(u + A) - S2(u + B) - S2(u - B) + S2(u - A))


def _chord_tol(u: float, om_deg: float, a: float, s: float, w: float) -> float:
    """The rounding of chord_weight in float64 at one point, from its own terms: the code evaluates :eq:`int-chord` at
    -|u| (:eq:`int-even`). Four-term form: each P2 is a sum of products of ~16 operations, rounding by 16 eps times the
    sum of its terms' magnitudes, and the difference is scaled by h / (A - B). Below A - B = 0.5 sigma it is the
    6-point mean (:eq:`int-chord-mean`), whose error is at most 4.37e-13 (L / sigma)^12 h, and its rounding that of
    h times masses <= 1."""
    c, sn = abs(np.cos(np.radians(om_deg))), abs(np.sin(np.radians(om_deg)))
    A, B = a * (c + sn) / 2, a * abs(c - sn) / 2
    h = a**2 / (A + B)
    if A - B < 0.5 * s:
        return h * (4.37e-13 * ((A - B) / s) ** 12 + 2 * 16 * EPS)
    x, sv = sp.symbols("x"), sp.Float(s, 30)
    if w == 0:
        parts = [x * Phi(x / sv), sv * phi(x / sv)]  # R1
    else:
        y = sp.symbols("y")
        r2 = [(y**2 + sv**2) / 2 * Phi(y / sv), y * sv / 2 * phi(y / sv)]
        parts = [t.subs(y, x + sp.Float(w, 30) / 2) / w for t in r2] + [t.subs(y, x - sp.Float(w, 30) / 2) / w for t in r2]
    v = -abs(u)
    mag = sum(abs(float(sp.N(t.subs(x, sp.Float(v + d, 30)), 30))) for d in (A, B, -B, -A) for t in parts)
    return 16 * EPS * h / (A - B) * mag


def test_chord_weight_code():
    """chord_weight against :eq:`int-chord`, Gaussian and flat-top profiles, and a square lined up with the beam."""
    for u, om, a, s, w in [(0.3, 20.0, 1.0, 0.5, 0.0), (-1.1, 63.0, 1.0, 0.7, 0.0), (0.2, 117.0, 2.0, 0.4, 1.5),
                           (0.6, 0.0, 1.0, 0.5, 0.0), (1.5, 200.0, 0.5, 1.0, 3.0)]:  # fmt: skip
        expected = float(sp.N(_chord_expr(sp.Float(u, 30), om, sp.Float(a, 30), sp.Float(s, 30), w), 30))
        with float64():
            got = float(chord_weight(u, om, a, s, w))
        np.testing.assert_allclose(got, expected, rtol=0, atol=_chord_tol(u, om, a, s, w))


def test_flat_top_small_width():
    """Anri's choice in :eq:`int-profile`: below w = 0.01 sigma the flat top is taken as the Gaussian. The flat top is
    the mean of the Gaussian over centres in [-w/2, w/2], so p_w = p_0 + (w^2 / 24) p_0'' + O(w^4); and
    int |phi''| = 4 phi(1) (phi'' = (x^2 - 1) phi changes sign at +-1). So |W_w - W_0| <= (w^2 / 24) h 4 phi(1) /
    sigma^2 to leading order: 4.0e-6 h at the switch."""
    u, c, x = sp.symbols("u c x", real=True)
    s, w = sp.symbols("sigma w", positive=True)
    g = phi((u - c) / s) / s
    taylor = sum(sp.diff(g, c, k).subs(c, 0) * c**k / sp.factorial(k) for k in range(5))  # g to O(c^5)
    mean = sp.expand(sp.integrate(taylor, (c, -w / 2, w / 2)) / w)  # the odd terms vanish
    left = sp.simplify(mean - g.subs(c, 0) - w**2 / 24 * sp.diff(g.subs(c, 0), u, 2))
    assert is_zero(left - w**4 / 1920 * sp.diff(g.subs(c, 0), u, 4))  # the next term, O(w^4)
    d2 = sp.diff(phi(x), x, 2)
    assert is_zero(d2 - (x**2 - 1) * phi(x))
    total = 2 * (sp.integrate(-d2, (x, 0, 1)) + sp.integrate(d2, (x, 1, sp.oo)))
    assert is_zero(total - 4 * phi(sp.Integer(1)))
    assert rounds_to(sp.Rational(1, 10**4) / 24 * 4 * phi(sp.Integer(1)) * 10**6, "4.0")


def test_flat_top_switch_code():
    """chord_weight at widths either side of the 0.01 sigma switch: just below it is the Gaussian's :eq:`int-chord`
    (test_flat_top_small_width bounds the difference), just above the flat top's."""
    u, om, a, s = 0.3, 30.0, 1.0, 0.4
    for w, page_w in ((0.0099 * s, 0.0), (0.0101 * s, 0.0101 * s)):
        expected = float(sp.N(_chord_expr(sp.Float(u, 30), om, sp.Float(a, 30), sp.Float(s, 30), page_w), 30))
        with float64():
            got = float(chord_weight(u, om, a, s, w))
        np.testing.assert_allclose(got, expected, rtol=0, atol=_chord_tol(u, om, a, s, page_w))


def _chord_sym(u: sp.Expr, om: sp.Symbol, a: sp.Expr, s: sp.Expr, w: float) -> sp.Expr:
    """:eq:`int-chord` with omega (degrees) a symbol, to differentiate; not at A = B."""
    w = sp.nsimplify(w)  # exact: a float's 53 bits would cap evalf's precision of every term that holds it
    c, sn = sp.Abs(sp.cos(sp.pi * om / 180)), sp.Abs(sp.sin(sp.pi * om / 180))
    A, B = a * (c + sn) / 2, a * sp.Abs(c - sn) / 2
    if w == 0:
        S2 = lambda t: R1(t, s)
    else:
        S2 = lambda t: (R2(t + w / 2, s) - R2(t - w / 2, s)) / w
    return a**2 / (A + B) / (A - B) * (S2(u + A) - S2(u + B) - S2(u - B) + S2(u - A))


def test_chord_weight_near_box_code():
    """chord_weight and its first two omega derivatives near omega = 90 deg, where the four terms cancel and the code
    takes :eq:`int-chord-mean`'s 6-point mean (below A - B = 0.5 sigma in float64), and just past the switch; and in a
    tail, where it evaluates :eq:`int-even` at -|u|."""
    om = sp.symbols("omega", real=True)
    a, s = sp.Rational(7, 10), sp.Rational(9, 20)
    av, sv = 0.7, 0.45
    pmax, dpmax = 1 / np.sqrt(2 * np.pi) / sv, np.exp(-0.5) / np.sqrt(2 * np.pi) / sv**2  # max p, max |p'| (Gaussian)
    with float64():  # value, slope and curvature in omega, compiled once for every point

        @jax.jit
        def code(u: jax.Array, o: jax.Array, wd: jax.Array) -> tuple:
            f = lambda t: chord_weight(u, t, av, sv, wd)
            return f(o), jax.grad(f)(o), jax.grad(jax.grad(f))(o)

    for w in (0.0, 1.5):
        for uv in (0.0, 2 * sv):
            W = _chord_sym(sp.Float(uv, 60), om, a, s, w)
            # the same expressions, evaluated by mpmath at 50 digits (sympy's evalf is ~50x slower here)
            # (Abs's second derivative holds DiracDelta terms, zero away from the kinks sampled here)
            ders = (W, sp.diff(W, om), sp.diff(W, om, 2))
            exprs = [sp.lambdify(om, e.replace(sp.DiracDelta, lambda *_: 0), "mpmath") for e in ders]
            for omv in (90 + 1e-6, 90 - 1e-3, 90 + 0.5, 90 - 3.0, 100.0, 110.0, 27.3):
                with mpmath.workdps(50):  # the four terms at A - B = 1e-8 a cancel 8 digits, 1 / L^3 24 more
                    ref = [float(e(mpmath.mpf(repr(omv)))) for e in exprs]
                with float64():
                    got = [float(v) for v in code(np.float64(uv), np.float64(omv), np.float64(w))]
                # the 6-point mean's error and its derivatives are below 1e-16 h at L <= 0.5 sigma (int-chord-mean);
                # its rounding: chain-rule terms of at most S0 = h, S1 = h max(p) a (pi / 180) and
                # S2 = h (max|p'| a^2 + max(p) a) (pi / 180)^2, each rounded over ~16 operations. Past the switch
                # (L >= 0.5 sigma), the four-term form's rounding, amplified by j! |dL / domega|^j / L^(j+1)
                c, sn = abs(np.cos(np.radians(omv))), abs(np.sin(np.radians(omv)))
                A, B = av * (c + sn) / 2, av * abs(c - sn) / 2
                h = av**2 / (A + B)
                four = A - B >= 0.5 * sv
                dL = np.sqrt(2) * av * np.pi / 180  # |dA / domega|, |dB / domega| <= a / sqrt(2) per radian
                scale = [h, h * pmax * av * np.pi / 180, h * (dpmax * av**2 + pmax * av) * (np.pi / 180) ** 2]
                for k in range(3):
                    extra = h * (A + sv) * math.factorial(k) * dL**k / (A - B) ** (k + 1) if four else 0.0
                    tol = 16 * EPS * (scale[k] + extra)
                    np.testing.assert_allclose(got[k], ref[k], rtol=0, atol=tol)
    # a tail: u = 5 sigma, every term at -|u| +- A, B small. R1(x) at x << 0 is x Phi + sigma phi, cancelling with
    # condition (|x Phi| + sigma phi) / R1 (int-ramp's code check), then the four terms with their own condition
    uv, omv = 5 * sv, 27.3
    expected = _chord_expr(sp.Float(uv, 40), omv, sp.Float(av, 40), sp.Float(sv, 40), 0.0)
    c, sn = abs(np.cos(np.radians(omv))), abs(np.sin(np.radians(omv)))
    pts = [-uv + av * (c + sn) / 2, -uv + av * abs(c - sn) / 2, -uv - av * abs(c - sn) / 2, -uv - av * (c + sn) / 2]
    x = sp.symbols("x", real=True)
    cond_r = max(
        float(sp.N((abs(xv * Phi(sp.Float(xv, 30) / s)) + s * phi(sp.Float(xv, 30) / s)) / R1(sp.Float(xv, 30), s), 30))
        for xv in pts
    )
    four = [float(sp.N(R1(x, s).subs(x, sp.Float(xv, 30)), 30)) for xv in pts]
    cond_4 = sum(abs(v) for v in four) / abs(four[0] - four[1] - four[2] + four[3])
    with float64():
        got = float(chord_weight(uv, omv, av, sv, 0.0))
    np.testing.assert_allclose(got, float(sp.N(expected, 30)), rtol=16 * EPS * cond_r * cond_4)
    assert cond_r * cond_4 < 1e3  # at +u the four terms would grow like u and cancel to the tail: no digits left


def test_beam_offsets():
    """beam_offsets: for a beam at azimuth psi and tilt alpha, k = (cos a cos psi, cos a sin psi, sin a), a point p is
    across = p . e_h from the beam's vertical plane, its horizontal distance along the beam is p . (cos psi, sin psi, 0),
    the centre line has risen tan(alpha) times that, and p less its offsets across and up is on the centre line."""
    psi, al = sp.symbols("psi alpha", real=True)
    k, e_h, _ = beam_frame(psi, al)
    p = sp.Matrix(sp.symbols("p_x p_y p_z", real=True))
    along = p[0] * sp.cos(psi) + p[1] * sp.sin(psi)
    rise = along * sp.tan(al)
    height = p[2] - rise
    on_line = p - p.dot(e_h) * e_h - height * sp.Matrix([0, 0, 1])
    assert is_zero(on_line - along / sp.cos(al) * k)


def _offsets(p: tuple, psi: float, al: float) -> tuple:
    """test_beam_offsets' across, height, rise and cos(alpha), exactly."""
    P = sp.Matrix([sp.Float(x, 30) for x in p])
    ps, a = sp.Float(psi, 30), sp.Float(al, 30)
    _, e_h, _ = beam_frame(ps, a)
    along = P[0] * sp.cos(ps) + P[1] * sp.sin(ps)
    return P.dot(e_h), P[2] - along * sp.tan(a), along * sp.tan(a), sp.cos(a)


def test_beam_offsets_code():
    """beam_offsets against test_beam_offsets, for an unnormalised k."""
    for p, psi, al in [((5.0, -2.0, 3.0), 0.0, 0.0), ((5.0, -2.0, 3.0), 0.3, 0.1), ((-40.0, 12.0, -7.0), -0.6, -0.2)]:
        k = 2.5 * np.array([np.cos(al) * np.cos(psi), np.cos(al) * np.sin(psi), np.sin(al)])
        want = np.array([float(sp.N(x, 30)) for x in _offsets(p, psi, al)])
        # products of |p| with unit vectors, and a quotient by cos(alpha) ~ 1: a few ulps of |p| (16 covers them)
        agree(want, lambda *z: np.array(beam_offsets(*z)), np.array(p), k, **tol(np.abs(p).max()))


def test_beam_weight_code():
    """beam_weight against :eq:`int-beam-weight`: a horizontal beam along x and at an azimuth, Gaussian and flat-top;
    a tilted beam through a column (1 / cos alpha) and through a cube (the vertical profile measured vertically)."""
    # NumPy, not jnp: a JAX array made outside float64() would be float32
    cases = [  # pos, omega, cube, psi, alpha, sig, width
        ((3.0, 0.4, 0.2), 25.0, False, 0.0, 0.0, 0.6, 0.0),
        ((-2.0, -0.3, 0.7), 140.0, True, 0.0, 0.0, 0.6, 0.0),
        ((1.0, 0.3, 0.0), 30.0, False, 20.0, 0.0, 0.4, 0.0),
        ((0.5, -0.5, 0.0), 0.0, False, 10.0, 0.0, 0.4, 1.5),
        ((2.0, 0.2, 0.5), 30.0, False, 0.0, 25.0, 0.4, 0.0),
        ((1.0, -0.4, 0.3), 50.0, True, 15.0, 10.0, 0.4, 0.0),
    ]
    for pos, om, cube, psi, al, sig, width in cases:
        pr, ar = np.radians(psi), np.radians(al)
        k = np.array([np.cos(ar) * np.cos(pr), np.cos(ar) * np.sin(pr), np.sin(ar)])
        geom = {"k_in_lab": k, "voxel_size": 1.0, "sig_beam": sig, "width_beam": width, "sig_beam_v": 0.8,
                "width_beam_v": 1.0, "voxel_3d": cube}  # fmt: skip
        across, dz, _, cos_a = _offsets(pos, pr, ar)
        W = _chord_expr(across, om - psi, sp.Integer(1), sp.Float(sig, 30), width) / cos_a
        sv, wv = sp.Float(0.8, 30) / cos_a, 1 / cos_a
        cdf = [(R1(dz + sgn / 2 + wv / 2, sv) - R1(dz + sgn / 2 - wv / 2, sv)) / wv for sgn in (1, -1)]
        V = cdf[0] - cdf[1] if cube else 1
        want = float(sp.N(W * V, 30))
        with float64():
            got = float(beam_weight(jnp.asarray(pos), om, geom))
        # the chord weight's rounding, plus that of its argument u (16 ulps of |p|), moving W by at most
        # max|dW/du| = h max p <= sqrt(2) a phi(0) / sigma per unit; over cos(alpha); a cube's share and the products
        # add 16 ulps of the result's terms
        u_err = 16 * EPS * np.abs(pos).max() * np.sqrt(2) * float(sp.N(phi(sp.Integer(0)), 30)) / sig
        t = (_chord_tol(float(across), om - psi, 1.0, sig, width) + u_err) / float(cos_a)
        np.testing.assert_allclose(got, want, rtol=0, atol=t + 2 * 16 * EPS * abs(float(sp.N(W, 30))))


# ---------------------------------------------------------------------------------------------- int-fold


def test_beam_fold():
    """A Gaussian omega profile times exp(l + beta d + gamma d^2 / 2): a Gaussian of variance 1 / (1 / oo - gamma)
    and mean mu + oo_b beta, of integral exp(l + beta^2 oo_b / 2) sqrt(oo_b / oo); (slow, fast) keep their regression
    on omega."""
    d = sp.symbols("d", real=True)
    oo = sp.symbols("Sigma_oo", positive=True)
    g = sp.symbols("gamma", negative=True)
    lb, beta = sp.symbols("l beta", real=True)
    oo_b = 1 / (1 / oo - g)
    integrand = sp.exp(-(d**2) / (2 * oo)) / sp.sqrt(2 * sp.pi * oo) * sp.exp(lb + beta * d + g * d**2 / 2)
    weight = sp.exp(lb + beta**2 * oo_b / 2) * sp.sqrt(oo_b / oo)
    assert is_zero(sp.simplify(sp.integrate(integrand, (d, -sp.oo, sp.oo)) - weight))
    # the normalised product is N(oo_b beta, oo_b): log-density matches up to a constant
    new = -((d - oo_b * beta) ** 2) / (2 * oo_b)
    assert is_zero(sp.diff(sp.log(integrand) - new, d))
    # slow keeps E[s | omega] = mu_s + (so / oo)(omega - mu_o) and Var(s | omega) = ss - so^2 / oo
    ss, so = sp.symbols("Sigma_ss Sigma_so", real=True)
    k = so / oo
    assert is_zero((ss - k**2 * (oo - oo_b)) - (ss - so**2 / oo + k**2 * oo_b))


def test_beam_fold_code():
    """_beam_fold's new centroid, shape and weight from its own expansion (l, beta, gamma), against :eq:`int-fold`."""
    geom = {"k_in_lab": np.array([1.0, 0.0, 0.0]), "wedge": 0.0, "chi": 0.0, "y0": 0.0, "voxel_size": 0.5,
            "sig_beam": 0.4, "width_beam": 0.0, "sig_beam_v": 1.0, "width_beam_v": 0.0, "voxel_3d": False}  # fmt: skip
    row = make_row(np.array([11.9, 12.0, 12.1]), np.full(3, 0.3))  # dty fixed: the path is the row's dty
    mu = np.array([100.2, 50.7, 12.0])
    shape = (0.5, 0.4, 0.04, 0.1, 0.05, -0.03)
    with float64():
        mu_b, shape_b, weight, (om_c, lb, beta, gam) = _beam_fold(jnp.array([5.0, 0.2, 0.0]), jnp.asarray(mu),
                                                                  tuple(jnp.asarray(x) for x in shape), row, geom)  # fmt: skip
    ss, ff, oo, sf, so, fo = (sp.Float(x, 30) for x in shape)
    lb, beta, gam = (sp.Float(float(x), 30) for x in (lb, beta, gam))
    assert float(gam) < 0 and abs(float(beta)) * float(sp.sqrt(oo)) < 3  # neither clamp is active here
    oo_b = 1 / (1 / oo - gam)
    ks, kf = so / oo, fo / oo
    d = oo_b * beta
    exp_mu = [mu[0] + ks * d, mu[1] + kf * d, mu[2] + d]
    exp_shape = [ss - ks**2 * (oo - oo_b), ff - kf**2 * (oo - oo_b), oo_b, sf - ks * kf * (oo - oo_b), ks * oo_b,
                 kf * oo_b]  # fmt: skip
    exp_w = sp.exp(lb + beta**2 * oo_b / 2) * sp.sqrt(oo_b / oo)
    # each is ~10 operations of condition O(1) on these inputs: 64 ulps
    np.testing.assert_allclose(np.asarray(mu_b), [float(sp.N(x, 30)) for x in exp_mu], rtol=64 * EPS)
    np.testing.assert_allclose([float(x) for x in shape_b], [float(sp.N(x, 30)) for x in exp_shape], rtol=64 * EPS)
    np.testing.assert_allclose(float(weight), float(sp.N(exp_w, 30)), rtol=64 * EPS)
    assert float(om_c) == mu[2]


def test_dty_motion():
    """A voxel moving uniformly by Delta across the beam during a frame, its light spread evenly over the frame, sees
    the profile convolved with a box: its variance grows by Delta^2 / 12, and its mean is unchanged."""
    x, D = sp.symbols("x Delta", positive=True)
    assert is_zero(sp.integrate(x / D, (x, -D / 2, D / 2)))
    assert is_zero(sp.integrate(x**2 / D, (x, -D / 2, D / 2)) - D**2 / 12)


def test_dty_path():
    """:eq:`int-dty-motion`: omega and dty uniform in time make dty linear in omega within a frame, and the time
    integral of a light g(omega, y) is the omega integral along that path, over Omega. With the light even over the
    frame (g = w(y)), the path's average is the box average over Delta = r delta_omega."""
    t, w0, y0 = sp.symbols("t omega_f y_f", real=True)
    Om, V, T = sp.symbols("Omega V T", positive=True)
    om = sp.symbols("omega", real=True)
    a, b, c, d = sp.symbols("a b c d", real=True)
    g = a * om**2 + b * om * sp.Symbol("y") + c * sp.Symbol("y") ** 2 + d  # any polynomial light g(omega, y)
    path = y0 + V / Om * (om - w0)
    by_time = sp.integrate(g.subs({om: w0 + Om * t, sp.Symbol("y"): y0 + V * t}), (t, -T / 2, T / 2))
    by_omega = sp.integrate(g.subs(sp.Symbol("y"), path), (om, w0 - Om * T / 2, w0 + Om * T / 2)) / Om
    assert is_zero(by_time - by_omega)
    # light even over the frame: the weight along the path averaged over the frame is the box average over the
    # frame's motion Delta = r delta_omega (delta_omega = Omega T), by u = (V / Omega)(omega - omega_f); checked on
    # w(y) = exp(k y), of which any weight is a superposition (its Fourier or Laplace transform)
    D = V * T
    u = sp.symbols("u", real=True)
    k = sp.symbols("k", positive=True)  # k < 0 is the same with y -> -y
    path_avg = sp.integrate(sp.exp(k * path), (om, w0 - Om * T / 2, w0 + Om * T / 2)) / (Om * T)
    box = sp.integrate(sp.exp(k * (y0 + u)), (u, -D / 2, D / 2)) / D
    assert is_zero(sp.simplify(path_avg - box))


def test_dty_rate():
    """:eq:`int-dty-rate`: the central difference for unequal steps is exact on a quadratic y(omega)."""
    hp, hm = sp.symbols("h_+ h_-", positive=True)
    w0, c0, c1, c2 = sp.symbols("omega_f c0 c1 c2", real=True)

    def y(o: sp.Expr) -> sp.Expr:
        return c0 + c1 * (o - w0) + c2 * (o - w0) ** 2

    r = (hm**2 * (y(w0 + hp) - y(w0)) + hp**2 * (y(w0) - y(w0 - hm))) / (hp * hm * (hp + hm))
    assert is_zero(r - sp.diff(y(sp.Symbol("o")), sp.Symbol("o")).subs(sp.Symbol("o"), w0))


def test_dty_path_code():
    """make_row's rate and _dty_at's path against :eq:`int-dty-rate` and :eq:`int-dty-motion` for readings linear in
    omega (unequal frames, unsorted), and _frame_weights' weight at the path's dty; a constant dty has rate 0."""
    omega = np.array([0.30, 0.0, 0.11, 0.25, 0.62, 0.43])
    dty = 1.7 - 2.3 * omega  # a fly scan's dty, linear in omega
    with float64():
        row = make_row(omega, dty)
        assert np.all(np.asarray(make_row(omega, np.full(6, 1.7))["dty_rate_sorted"]) == 0.0)
        # the rate: a few operations on differences of |dty| <= 2 over steps >= 0.05: ~16 ulps of 2 / 0.05
        np.testing.assert_allclose(np.asarray(row["dty_rate_sorted"]), -2.3, rtol=0, atol=16 * EPS * 2 / 0.05)
        om = np.array([0.02, 0.13, 0.299, 0.31, 0.7, 360.2])  # the last in the next turn: 0.2
        got = np.asarray(jax.vmap(_dty_at, in_axes=(0, None))(jnp.asarray(om), row))
        # the reading plus rate x offset: ~16 ulps of the largest term (|dty| <= 2, rate x offset <= 0.5)
        np.testing.assert_allclose(got, 1.7 - 2.3 * np.mod(om, 360.0), rtol=0, atol=16 * EPS * 2 + 16 * EPS * 360 * 2.3)
        geom = {"k_in_lab": jnp.array([1.0, 0.0, 0.0]), "wedge": 0.0, "chi": 0.0, "y0": 0.0, "voxel_size": 0.5,
                "sig_beam": 0.4, "width_beam": 0.0, "sig_beam_v": 1.0, "width_beam_v": 0.0, "voxel_3d": False}  # fmt: skip
        pos = jnp.array([5.0, 0.2, 0.0])
        fclip = jnp.array([1, 3, 4])  # sorted frames 0.11, 0.30, 0.43; mean omegas within them
        om_fr = jnp.array([0.12, 0.27, 0.40])
        fw = np.asarray(_frame_weights(pos, om_fr, fclip, row, geom))
        lab = [sample_to_lab(pos, o, 0.0, 0.0, 1.7 - 2.3 * o, 0.0) for o in (0.12, 0.27, 0.40)]
        want = np.asarray([beam_weight(x, o, geom) for x, o in zip(lab, (0.12, 0.27, 0.40))])
    # the same float64 operations but for the dty: its error (above, ~1e-14) moves a weight of slope ~1 / sigma
    np.testing.assert_allclose(fw, want, rtol=1e-13)


# ---------------------------------------------------------------------------------------------- int-absorption


def test_path_code():
    """path_lengths is D / cos(elevation) (the in-plane fraction floored at 0.05), transmission exp(-mu (L_in + L_out))."""
    D = np.full((1, 36), 50.0)  # the same edge distance in every direction, so the interpolation is exact
    d_in = np.array([[[np.cos(0.3) * np.cos(1.0), np.cos(0.3) * np.sin(1.0), np.sin(0.3)]]])
    d_out = np.array([[[0.0, np.sin(0.01), np.cos(0.01)]]])  # nearly vertical: floored
    with float64():
        L_in = float(path_lengths(jnp.asarray(D), jnp.asarray(d_in))[0, 0])
        L_out = float(path_lengths(jnp.asarray(D), jnp.asarray(d_out))[0, 0])
        T = float(transmission(jnp.asarray(D), jnp.asarray(d_in), jnp.asarray(d_out), 0.01)[0, 0])
    e_in = sp.Float(50, 30) / sp.sqrt(sp.Float(d_in[0, 0, 0], 30) ** 2 + sp.Float(d_in[0, 0, 1], 30) ** 2)
    e_out = sp.Float(50, 30) / sp.Rational(5, 100)
    # a hypot, a division and the interpolation's 1 - w + w: ~8 ulps; the exponential of an O(1) argument adds ~4
    np.testing.assert_allclose(L_in, float(e_in), rtol=16 * EPS)
    np.testing.assert_allclose(L_out, float(e_out), rtol=16 * EPS)
    np.testing.assert_allclose(T, float(sp.N(sp.exp(-sp.Float(0.01, 30) * (e_in + e_out)), 30)), rtol=64 * EPS)


def test_edge_trapezoid():
    """edge_distances sums the map along each direction as sum_{i >= 0} m_i - m_0 / 2 (unit steps): the integral from 0
    to the edge of the map's linear interpolant along the ray, exactly, whenever the ray's samples are the map's
    nodes (a direction along an axis from a voxel's centre)."""
    m = sp.symbols("m0:4", real=True)
    t = sp.symbols("t", real=True)
    pieces = [m[i] + (m[i + 1] - m[i]) * (t - i) if i + 1 < 4 else m[i] * (1 - (t - i)) for i in range(4)]
    exact = sum(sp.integrate(f, (t, i, i + 1)) for i, f in enumerate(pieces))
    assert is_zero(exact - (sum(m) - m[0] / 2))


def test_edge_distances_code():
    """edge_distances against test_edge_trapezoid's integral, along +x, +y, -x, -y from voxel centres, and from outside
    the map looking away (zero)."""
    from fractions import Fraction

    from anri.geom.paths import edge_distances

    n, step = 7, 2.0
    g = rng(15)
    m = g.integers(0, 9, (n, n)) / 8  # multiples of 1/8: exact in float32
    c0 = n // 2

    def node(x: int, y: int) -> Fraction:  # recon order: ri = x + n // 2, rj = n // 2 - y
        i, j = x + c0, c0 - y
        return Fraction(int(m[i, j] * 8), 8) if 0 <= i < n and 0 <= j < n else Fraction(0)

    pts = [(0, 0), (2, -1), (-3, 3), (5, 0)]  # the last outside the map, along +x
    want = []
    for x, y in pts:
        row = []
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            ms = [node(x + dx * i, y + dy * i) for i in range(2 * n)]
            row.append(float((sum(ms) - ms[0] / 2) * Fraction(step)))
        want.append(row)
    got = np.asarray(edge_distances(jnp.asarray(m, jnp.float32), step, jnp.asarray(pts, jnp.float32) * step, n_phi=4))
    # float32: the cumulative sum of K = 2h + 1 samples rounds by K eps32 times their sum; and cos, sin of the
    # directions in float32 are off by eps32, moving each sample by eps32 (|t| + |s|) <= 2 h eps32 voxels, which
    # changes it by at most max|m difference| = 1 per voxel
    h = int(np.ceil(n * 0.7072)) + 2
    K, e32 = 2 * h + 1, float(np.finfo(np.float32).eps)
    bound = K * e32 * step * (np.abs(m).sum() + K * 2 * h)
    np.testing.assert_allclose(got, want, rtol=0, atol=bound)
    assert np.all(got[3, 0] == 0)  # outside, looking away: nothing


def test_ray_directions_code():
    """ray_directions against frm-stack inverted (test_frames.py::test_stack_inverse): R^T C^T W^T of -k and of k_out,
    normalised."""
    from anri.geom.paths import ray_directions

    k_in = np.array([2.0, 0.3, -0.1])
    k_out = np.array([[1.0, -0.4, 0.5], [0.6, 0.7, -0.2]])
    om, wedge, chi = np.array([33.0, -140.0]), 1.5, -2.0
    want_in, want_out = [], []
    for o, ko in zip(om, k_out):
        M = rz(sp.rad(sp.Float(o, 30))).T * rx(sp.rad(sp.Float(chi, 30))).T * ry(sp.rad(sp.Float(wedge, 30))).T
        for v, out in ((-k_in, want_in), (ko, want_out)):
            V = sp.Matrix([sp.Float(x, 30) for x in v])
            out.append(evaluate(M * V / sp.sqrt(V.dot(V)), {}).reshape(3))
    d_in, d_out = ray_directions(k_in, k_out, om, wedge, chi)
    # float32 by design: unit vectors through three rotations and a normalisation, 16 ulps of 1 in float32
    e32 = float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(np.asarray(d_in), want_in, rtol=0, atol=16 * e32)
    np.testing.assert_allclose(np.asarray(d_out), want_out, rtol=0, atol=16 * e32)


def test_attenuation_map_code():
    """attenuation_map: each voxel holds its entries' mu weighted by their weights, sum w mu / sum w, exactly; voxels
    without entries hold 0."""
    from fractions import Fraction

    from anri.geom.paths import attenuation_map

    step = 2.0
    pos = np.array([[4.0, 2.0, 0], [4.2, 1.9, 0], [3.8, 2.3, 0], [-6.0, 0.0, 0]])
    mu = np.array([0.001, 0.003, 0.0025, 0.005])
    w = np.array([3.0, 1.0, 0.5, 2.0])
    got = attenuation_map(pos, mu, w, step)
    n = got.shape[0]
    want = np.zeros((n, n))
    groups = {}
    for p, a, b in zip(pos, mu, w):
        key = (round(p[0] / step) + n // 2, round(-p[1] / step) + n // 2)  # recon_to_sample, inverted
        groups.setdefault(key, []).append((Fraction(a), Fraction(b)))
    for key, ab in groups.items():
        want[key] = float(sum(a * b for a, b in ab) / sum(b for _, b in ab))
    # float64 sums of a few terms, then one rounding to float32: half an ulp of float32, relative
    np.testing.assert_allclose(got, want, rtol=float(np.finfo(np.float32).eps), atol=0)


def test_add_absorption_code():
    """add_absorption from the map's own entries: each entry's optical depth along +x is the trapezoid of mu over the
    voxels it crosses (test_edge_trapezoid), mu being each voxel's phase's (attenuation_map), layer by layer; two
    phases side by side, two layers. Exactly (multiples of 1/8 in float32)."""
    from anri.fwd.render import add_absorption
    from anri.geom.scan import recon_positions

    step = 2.0
    grid = np.asarray(recon_positions(5, step))
    grid = grid[(np.abs(grid[:, 0]) <= 2) & (np.abs(grid[:, 1]) <= 2)]  # 3 x 3 voxels
    left = grid[:, 0] < 0
    mu = [0.25, 0.5]  # phase 1 on the left column, phase 2 elsewhere
    layers = [grid, grid[:4] + [0, 0, step]]  # the second layer holds four voxels
    ents = []
    for m in (left, ~left):
        pos = np.concatenate([L[m[: len(L)]] for L in layers])
        ents.append(({"ubi": np.zeros((len(pos), 3, 3)), "pos": pos, "density": np.ones(len(pos))}, None, None))
    out = add_absorption(ents, step, mu=mu, n_phi=4)
    for (e, _, _), (o, _, _) in zip(ents, out):
        for p, d in zip(e["pos"], o["optical_depth"]):
            layer = [q for L in layers for q in L if q[2] == p[2]]
            row = sorted((q for q in layer if q[1] == p[1] and q[0] >= p[0]), key=lambda q: q[0])
            m_ = [mu[0] if q[0] < 0 else mu[1] for q in row]
            assert d[0] == (sum(m_) - m_[0] / 2) * step, (p, d[0])


if __name__ == "__main__":
    unittest.main()
