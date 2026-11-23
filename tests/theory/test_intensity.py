"""Proofs for docs/source/theory/intensity.rst: Lorentz, polarisation, the beam's profile over a voxel and its fold
into a peak, and the absorption paths.

Tolerances of the code checks are derived where they are stated: float64 has eps = 2.2e-16, and an expression of n
operations with condition number c is good to about n c eps.
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np
import sympy as sp

from anri.diffract.factors import lorentz, polarisation, polarisation_extinction
from anri.fwd.render import _beam_fold
from anri.geom.beam import beam_weight, chord_weight
from anri.geom.paths import path_lengths, transmission
from anri.maths.utils import float64, ramp_integral, ramp_integral2
from tests.theory.common import (
    EPS,
    evaluate,
    is_zero,
    rng,
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


def test_chord_weight_code():
    """chord_weight against :eq:`int-chord`, Gaussian and flat-top profiles, and a square lined up with the beam."""
    for u, om, a, s, w in [(0.3, 20.0, 1.0, 0.5, 0.0), (-1.1, 63.0, 1.0, 0.7, 0.0), (0.2, 117.0, 2.0, 0.4, 1.5),
                           (0.6, 0.0, 1.0, 0.5, 0.0), (1.5, 200.0, 0.5, 1.0, 3.0)]:  # fmt: skip
        expected = float(sp.N(_chord_expr(sp.Float(u, 30), om, sp.Float(a, 30), sp.Float(s, 30), w), 30))
        with float64():
            got = float(chord_weight(u, om, a, s, w))
        # the four-term difference cancels: condition ~ 4 h max|S2| / ((A - B) |W|), each S2 a few tens of
        # operations; with A - B >= a / 4 and |W| >= 0.1 here, the bound is below 1e-12
        np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_beam_weight_code():
    """beam_weight for a horizontal beam along x: the chord weight at u = y_lab, times, for a cube, the vertical
    profile's mass over its height."""
    # NumPy, not jnp: a JAX array made outside float64() would be float32
    geom = {"k_in_lab": np.array([1.0, 0.0, 0.0]), "voxel_size": 1.0, "sig_beam": 0.6, "width_beam": 0.0,
            "sig_beam_v": 0.8, "width_beam_v": 1.0}  # fmt: skip
    for pos, om, cube in [((3.0, 0.4, 0.2), 25.0, False), ((-2.0, -0.3, 0.7), 140.0, True)]:
        y, z = sp.Float(pos[1], 30), sp.Float(pos[2], 30)
        W = _chord_expr(y, om, sp.Integer(1), sp.Float(0.6, 30), 0.0)
        half, sv = sp.Rational(1, 2), sp.Float(0.8, 30)
        cdf = [R1(z + half + sgn * half, sv) - R1(z + half * sgn - half, sv) for sgn in (1, -1)]  # width 1: no / w
        V = cdf[0] - cdf[1] if cube else 1
        with float64():
            got = float(beam_weight(jnp.asarray(pos), om, {**geom, "voxel_3d": cube}))
        np.testing.assert_allclose(got, float(sp.N(W * V, 30)), rtol=1e-12)  # as test_chord_weight_code


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
    row = {"dty_mid": 0.3}
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
    """A voxel moving uniformly by Delta across the beam during a frame sees the profile convolved with a box: its
    variance grows by Delta^2 / 12, and its mean is unchanged."""
    x, D = sp.symbols("x Delta", positive=True)
    assert is_zero(sp.integrate(x / D, (x, -D / 2, D / 2)))
    assert is_zero(sp.integrate(x**2 / D, (x, -D / 2, D / 2)) - D**2 / 12)


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


if __name__ == "__main__":
    unittest.main()
