"""Proofs for docs/source/theory/populations.rst: the mean of a population, its spread as sig_rot, the grouping
radius, and the disc mask."""

import numpy as np
import sympy as sp

from tests.theory.common import (
    EPS,
    agree,
    evaluate,
    is_zero,
    rng,
)


def _rz(a: sp.Expr) -> sp.Matrix:
    return sp.Matrix([[sp.cos(a), -sp.sin(a), 0], [sp.sin(a), sp.cos(a), 0], [0, 0, 1]])


def test_mean_of_coaxial_rotations():
    """pop-mean for rotations about one axis: sum w_i Rz(a_i) = Rz(phi) diag(rho, rho, W), with rho = |(C, S)|,
    C = sum w cos a, S = sum w sin a, W = sum w. The second factor is symmetric positive definite, so Rz(phi) is the
    polar factor of the sum: the closest rotation, phi = atan2(S, C)."""
    w = sp.symbols("w0:3", positive=True)
    a = sp.symbols("a0:3", real=True)
    C = sum(wi * sp.cos(ai) for wi, ai in zip(w, a))
    S = sum(wi * sp.sin(ai) for wi, ai in zip(w, a))
    rho = sp.sqrt(C**2 + S**2)
    M = sum((wi * _rz(ai) for wi, ai in zip(w, a)), sp.zeros(3, 3))
    polar = sp.Matrix([[C / rho, -S / rho, 0], [S / rho, C / rho, 0], [0, 0, 1]])  # Rz(atan2(S, C))
    assert is_zero(M - polar * sp.diag(rho, rho, sum(w)))


def test_mean_rotation_code():
    """anri's mean_rotation of rotations about z is Rz(atan2(S, C))."""
    from anri.maths.rotation import mean_rotation

    g = rng(6)
    av, wv = g.uniform(-0.3, 0.3, 3), g.uniform(0.2, 1.0, 3)
    a, w = sp.symbols("a0:3"), sp.symbols("w0:3")
    C = sum(wi * sp.cos(ai) for wi, ai in zip(w, a))
    S = sum(wi * sp.sin(ai) for wi, ai in zip(w, a))
    vals = {**dict(zip(a, av)), **dict(zip(w, wv))}
    want = evaluate(_rz(sp.atan2(S, C)), vals)
    U = np.array([np.asarray(evaluate(_rz(x), {}), float) for x in map(sp.Float, av)])
    # an SVD of a well-conditioned 3 x 3 sum (singular values rho, rho, W of similar size): backward stable, a few
    # tens of ulps in the entries, which are <= 1
    agree(want, mean_rotation, U, wv, rtol=0, atol=64 * EPS)


def test_isotropic_rms_angle():
    """A rotation vector with independent components of standard deviation sigma has E|delta|^2 = 3 sigma^2: the
    rms angle is sqrt(3) sigma, so sig_rot = spread / sqrt(3)."""
    from sympy.stats import E, Normal

    s = sp.symbols("sigma", positive=True)
    X = [Normal(f"x{i}", 0, s) for i in range(3)]
    assert is_zero(E(sum(x**2 for x in X)) - 3 * s**2)


def test_radius_quoted():
    """A plain fit of a 2.5 deg grid groups within 1.8 x 2.5 = 4.5 deg; a fit with levels (0.1 deg) within one bin."""
    from anri.index.populations import radius

    assert sp.Rational(18, 10) * sp.Rational(25, 10) == sp.Rational(9, 2)
    assert radius(2.5) == 4.5
    assert radius(0.1) == 1.0


def test_disc_fraction():
    """A voxel at r > r_disc is within r_disc of the axis's row, |r cos u| <= r_disc, over the fraction
    (pi - 2 arccos(r_disc / r)) / pi = (2 / pi) arcsin(r_disc / r) of a turn: cos u in [-x, x] for u in
    [arccos x, pi - arccos x] and its mirror."""
    x = sp.symbols("x", positive=True)
    frac = 2 * (sp.pi - 2 * sp.acos(x)) / (2 * sp.pi)
    assert is_zero((frac - 2 / sp.pi * sp.asin(x)).rewrite(sp.asin))


def test_disc_half_turn():
    """A voxel within the farther end b of the scan (r <= b, nearer end a < b) is in a scanned row,
    -a <= -r cos u <= b, whenever cos u <= 0: at least half a turn, and for every direction one of u, u + pi."""
    u = sp.symbols("u", real=True)
    assert is_zero(sp.cos(u + sp.pi) + sp.cos(u))
