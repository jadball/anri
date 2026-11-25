"""Proofs for docs/source/theory/detector.rst: pixels as box masses and what that does to a spot's moments, the point
spread, the thick sensor's depth kernel, the spatial distortion, and splatting.

Tolerances of the code checks are derived where they are stated: float64 has eps = 2.2e-16, and an expression of n
operations with condition number c is good to about n c eps.
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import mpmath
import numpy as np
import sympy as sp

from anri.fwd.render import _map_shape, _peak_shape, _sensor_kernel, _undistort
from anri.maths.utils import bin_fractions, exponential_slices, float64
from tests.theory.common import (
    EPS,
    evaluate,
    is_zero,
    vec,
)


def Phi(z: sp.Expr) -> sp.Expr:
    return (1 + sp.erf(z / sp.sqrt(2))) / 2


def phi(z: sp.Expr) -> sp.Expr:
    return sp.exp(-(z**2) / 2) / sp.sqrt(2 * sp.pi)


# ---------------------------------------------------------------------------------------------- det-pixel


def test_box_masses_sum():
    """The box masses of a row of pixels telescope: they sum to the Gaussian's mass over the row's extent."""
    mu = sp.symbols("mu", real=True)
    s = sp.symbols("sigma", positive=True)
    n = 4
    total = sum(Phi((i + sp.Rational(1, 2) - mu) / s) - Phi((i - sp.Rational(1, 2) - mu) / s) for i in range(-n, n + 1))
    assert is_zero(total - (Phi((n + sp.Rational(1, 2) - mu) / s) - Phi((-n - sp.Rational(1, 2) - mu) / s)))


def test_sawtooth_series():
    """round(x) - x = -x on [-1/2, 1/2) has sine coefficients (-1)^k / (pi k); its square x^2 has mean 1/12 and cosine
    coefficients (-1)^k / (pi k)^2."""
    x = sp.symbols("x", real=True)
    k = sp.symbols("k", integer=True, positive=True)
    half = sp.Rational(1, 2)
    b_k = 2 * sp.integrate(-x * sp.sin(2 * sp.pi * k * x), (x, -half, half))
    assert is_zero(b_k - (-1) ** k / (sp.pi * k))
    assert sp.integrate(x**2, (x, -half, half)) == sp.Rational(1, 12)
    a_k = 2 * sp.integrate(x**2 * sp.cos(2 * sp.pi * k * x), (x, -half, half))
    assert is_zero(a_k - (-1) ** k / (sp.pi * k) ** 2)


def test_gaussian_fourier():
    """E[exp(i w x)] = exp(i w mu - sigma^2 w^2 / 2) for x ~ N(mu, sigma^2), and E[(x - mu) exp(i w x)] =
    i w sigma^2 E[exp(i w x)] (Stein's identity for each Fourier mode)."""
    x, mu, w = sp.symbols("x mu omega", real=True)
    s = sp.symbols("sigma", positive=True)
    pdf = phi((x - mu) / s) / s
    cf = sp.exp(sp.I * w * mu - s**2 * w**2 / 2)
    assert is_zero(sp.integrate(pdf * sp.exp(sp.I * w * x), (x, -sp.oo, sp.oo)) - cf)
    assert is_zero(sp.integrate(pdf * (x - mu) * sp.exp(sp.I * w * x), (x, -sp.oo, sp.oo)) - sp.I * w * s**2 * cf)


def _pixel_moments(mu: sp.Expr, s: sp.Expr, K: int) -> tuple:
    """Mean and variance of the pixel index round(x), x ~ N(mu, s^2): :eq:`det-pixel-moments`, to K terms."""
    rho = [sp.exp(-2 * sp.pi**2 * k**2 * s**2) for k in range(K + 1)]
    Es = sum((-1) ** k / (sp.pi * k) * rho[k] * sp.sin(2 * sp.pi * k * mu) for k in range(1, K + 1))
    rest = sum((-1) ** k * rho[k] * sp.cos(2 * sp.pi * k * mu) * (4 * s**2 + 1 / (sp.pi * k) ** 2)
               for k in range(1, K + 1))  # fmt: skip
    return mu + Es, s**2 + sp.Rational(1, 12) + rest - Es**2


def test_pixel_moments_code():
    """The mean and variance of bin_fractions' pixel masses against :eq:`det-pixel-moments`.

    The series is summed until rho_k = exp(-2 pi^2 k^2 sigma^2) < 1e-30, so it is exact at float64.
    """
    for mu, s in [(0.13, 0.15), (0.37, 0.25), (-0.21, 0.4), (0.5, 0.2)]:
        K = int(np.ceil(np.sqrt(30 * np.log(10) / (2 * np.pi**2 * s**2))))
        e_mean, e_var = (float(sp.N(x, 30)) for x in _pixel_moments(sp.Float(mu, 30), sp.Float(s, 30), K))
        i = np.arange(-20, 21, dtype=float)
        with float64():
            p = np.asarray(bin_fractions(i - 0.5, i + 0.5, mu, s))
        mean = p @ i
        var = p @ (i - mean) ** 2
        # each mass is good to a few ulps of the largest Phi in its tail form (< 1) absolutely; the sums weight them
        # by |i| <= 20 and i^2 <= 400: 41 x 400 x 4 eps bounds both
        np.testing.assert_allclose(mean, e_mean, rtol=0, atol=41 * 400 * 4 * EPS)
        np.testing.assert_allclose(var, e_var, rtol=0, atol=41 * 400 * 4 * EPS)


def test_pixel_bias_numbers():
    """The leading term of the centroid's bias, exp(-2 pi^2 sigma^2) / pi, at sigma = 0.2, 0.3, 0.5 pixel."""
    for s, text in ((sp.Rational(1, 5), "0.145"), (sp.Rational(3, 10), "0.0539"), (sp.Rational(1, 2), "0.00229")):
        assert f"{float(sp.N(sp.exp(-2 * sp.pi**2 * s**2) / sp.pi, 30)):.3g}" == text


# ---------------------------------------------------------------------------------------------- det-crlb


def test_poisson_fisher():
    """A Poisson count of mean lambda(mu) carries Fisher information lambda'^2 / lambda about mu; a pixel's mass has
    d/dmu = (phi(a) - phi(b)) / sigma; with infinitely small pixels the information per photon is 1 / sigma^2."""
    mu, n = sp.symbols("mu n", real=True)
    lam = sp.Function("lambda", positive=True)(mu)
    loglik = n * sp.log(lam) - lam - sp.loggamma(n + 1)
    fisher = -sp.diff(loglik, mu, 2).subs(n, lam)  # linear in n, so E[.] puts n = lambda
    assert is_zero(fisher - sp.diff(lam, mu) ** 2 / lam)
    lo, hi = sp.symbols("l u", real=True)
    s = sp.symbols("sigma", positive=True)
    p = Phi((hi - mu) / s) - Phi((lo - mu) / s)
    assert is_zero(sp.diff(p, mu) - (phi((lo - mu) / s) - phi((hi - mu) / s)) / s)
    z = sp.symbols("z", real=True)
    assert sp.integrate(sp.diff(phi(z), z) ** 2 / phi(z), (z, -sp.oo, sp.oo)) == 1  # per unit sigma^-2


def _crlb(mu: float, s: float) -> mpmath.mpf:
    """sqrt(N) times the Cramer-Rao bound on a spot's centroid from its pixel counts (:eq:`det-crlb`), at 30 digits:
    pixels -40..40 hold the spot to far below 1e-30 for sigma <= 1."""
    with mpmath.workdps(30):
        info = mpmath.mpf(0)
        for i in range(-40, 41):
            a, b = (i - mpmath.mpf(1) / 2 - mu) / s, (i + mpmath.mpf(1) / 2 - mu) / s
            mass = mpmath.ncdf(b) - mpmath.ncdf(a)
            if mass > 0:
                info += (mpmath.npdf(a) - mpmath.npdf(b)) ** 2 / s**2 / mass
        return 1 / mpmath.sqrt(info)


def test_crlb_numbers():
    """The table of :eq:`det-crlb`: a spot centred on a pixel, on an edge, and sqrt(sigma^2 + 1/12)."""
    table = {"0.1": ("25.5", "0.125", "0.306"), "0.2": ("0.636", "0.251", "0.351"), "0.3": ("0.466", "0.374", "0.416"),
             "0.5": ("0.579", "0.575", "0.577"), "1": ("1.04", "1.04", "1.04")}  # fmt: skip
    for s_text, (centre, edge, sheppard) in table.items():
        s = mpmath.mpf(s_text)
        assert f"{float(_crlb(mpmath.mpf(0), s)):.3g}" == centre
        assert f"{float(_crlb(mpmath.mpf(1) / 2, s)):.3g}" == edge
        assert f"{float(mpmath.sqrt(s**2 + mpmath.mpf(1) / 12)):.3g}" == sheppard


# ---------------------------------------------------------------------------------------------- det-psf


def test_gaussian_convolution():
    """A Gaussian spot convolved with a Gaussian point spread is the Gaussian of the summed variances."""
    x, y = sp.symbols("x y", real=True)
    s1, s2 = sp.symbols("sigma_1 sigma_2", positive=True)
    conv = sp.integrate(phi((x - y) / s1) / s1 * phi(y / s2) / s2, (y, -sp.oo, sp.oo))
    s = sp.sqrt(s1**2 + s2**2)
    assert is_zero(conv - phi(x / s) / s)


def test_box_once():
    """Integrating over a pixel adds a uniform's 1/12 to the variance, once: a pixel-integrated kernel convolved with
    pixel-integrated values adds it twice."""
    x = sp.symbols("x", real=True)
    half = sp.Rational(1, 2)
    var_box = sp.integrate(x**2, (x, -half, half)) - sp.integrate(x, (x, -half, half)) ** 2
    assert var_box == sp.Rational(1, 12)
    s2, k2 = sp.symbols("sigma2 k2", positive=True)
    right = s2 + k2 + var_box  # continuous spot * continuous kernel, then one box
    wrong = (s2 + var_box) + (k2 + var_box)  # boxed spot, convolved with a boxed kernel
    assert wrong - right == sp.Rational(1, 12)


def _geom() -> dict:
    """Beam along x, a detector of 75 um pixels normal to it at 150 mm (call inside float64())."""
    px = 75.0
    return {"wavelength": 0.3, "k_in_lab": jnp.array([1.0, 0.0, 0.0]), "wedge": 0.0, "chi": 0.0, "y0": 0.0,
            "s_step_lab": jnp.array([0.0, 0.0, -px]), "f_step_lab": jnp.array([0.0, -px, 0.0]),
            "det_origin_lab": jnp.array([150e3, 1000 * px, 1000 * px]), "voxel_3d": False,
            "sig_wavelength": 3e-4, "sig_ky": 1e-4, "sig_kz": 1e-4, "sig_psf": 0.0}  # fmt: skip


def test_psf_code():
    """_peak_shape adds sig_psf^2 to the slow and fast variances, before the pixels integrate (a continuous
    convolution), and nothing else."""
    ubi = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))[0] * 3.0
    with float64():
        geom = _geom()
        args = (jnp.asarray(ubi), jnp.zeros(3), jnp.array([1.0, 1.0, 0.0]), 1.0, None, {"dty_mid": 0.0})
        _, a, _ = _peak_shape(*args, geom)
        _, b, _ = _peak_shape(*args, {**geom, "sig_psf": 0.7})
    d = np.array([float(y) - float(x) for x, y in zip(a, b)])
    # one float64 addition of 0.49 to each variance, then the difference: an ulp of the larger, each
    bound = 2 * EPS * np.maximum(np.abs([float(x) for x in a]), np.abs([float(y) for y in b]))
    np.testing.assert_array_less(np.abs(d - [0.49, 0.49, 0, 0, 0, 0]), bound + 2 * EPS * 0.49)


# ---------------------------------------------------------------------------------------------- det-sensor


def test_exponential_slices():
    """Equal-probability slices of Exp(scale b) truncated at R: edges -b log(1 - k P / n), each of mass P / n; each
    slice's mean and second moment; and the slices' mixture has the truncated exponential's mean and variance."""
    x, b, R = sp.symbols("x b R", positive=True)
    n = 4
    P = 1 - sp.exp(-R / b)
    edges = [-b * sp.log(1 - sp.Rational(k, n) * P) for k in range(n)] + [R]
    for k in range(n):
        assert is_zero((1 - sp.exp(-edges[k] / b)) / P - sp.Rational(k, n))
    f = sp.exp(-x / b) / b
    lo, hi = sp.symbols("lo hi", positive=True)
    elo, ehi = sp.exp(-lo / b), sp.exp(-hi / b)
    Z = elo - ehi
    assert is_zero(sp.integrate(f, (x, lo, hi)) - Z)
    m = b + (lo * elo - hi * ehi) / Z
    m2 = (elo * (lo**2 + 2 * lo * b + 2 * b**2) - ehi * (hi**2 + 2 * hi * b + 2 * b**2)) / Z
    assert is_zero(sp.integrate(x * f, (x, lo, hi)) / Z - m)
    assert is_zero(sp.integrate(x**2 * f, (x, lo, hi)) / Z - m2)
    # each slice holds P / n, so the slices' mixture (weights 1 / n) is the truncated distribution itself
    for k in range(n):
        assert is_zero(Z.subs({lo: edges[k], hi: edges[k + 1]}) - P / n)
    mix_mean = sum(m.subs({lo: edges[k], hi: edges[k + 1]}) for k in range(n)) / n
    assert is_zero(mix_mean - sp.integrate(x * f, (x, 0, R)) / P)


def test_exponential_slices_code():
    """exponential_slices against the slices' mean and variance."""
    for bv, Rv in [(0.4, 3.0), (2.0, 1.5), (1.0, 10.0)]:
        b, R = sp.Float(bv, 30), sp.Float(Rv, 30)
        P = 1 - sp.exp(-R / b)
        edges = [-b * sp.log(1 - sp.Rational(k, 4) * P) for k in range(4)] + [R]
        ms, vs, conds = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            elo, ehi = sp.exp(-lo / b), sp.exp(-hi / b)
            Z = elo - ehi
            m = b + (lo * elo - hi * ehi) / Z
            m2 = (elo * (lo**2 + 2 * lo * b + 2 * b**2) - ehi * (hi**2 + 2 * hi * b + 2 * b**2)) / Z
            ms.append(float(sp.N(m, 30)))
            vs.append(float(sp.N(m2 - m**2, 30)))
            conds.append(float(sp.N(m2 / (m2 - m**2), 30)))  # the variance cancels m2 - m^2
        with float64():
            gm, gv = (np.asarray(y) for y in exponential_slices(bv, Rv, 4))
        # ~30 operations; Z = elo - ehi cancels by ~ (elo + ehi) / Z <= 2n / P < 30 here
        np.testing.assert_allclose(gm, ms, rtol=30 * 30 * 4 * EPS)
        for g, e, c in zip(gv, vs, conds):
            np.testing.assert_allclose(g, e, rtol=30 * 30 * 4 * EPS * c)


def test_in_plane_solve():
    """The ray's in-plane part d, in pixels: G d_pix = A^T d with G = A^T A, solved by the written-out 2 x 2 inverse,
    recovers d's coordinates exactly when d lies in the detector plane."""
    A = sp.Matrix(3, 2, sp.symbols("a0:6", real=True))
    c = sp.Matrix(sp.symbols("c0:2", real=True))
    G, bb = A.T * A, A.T * (A * c)
    det = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
    d_pix = sp.Matrix([G[1, 1] * bb[0] - G[0, 1] * bb[1], G[0, 0] * bb[1] - G[1, 0] * bb[0]]) / det
    assert is_zero(d_pix - c)


def test_sensor_kernel_code():
    """_sensor_kernel against :eq:`det-sensor`: direction, scale and cut in pixels, and the slices with the mean
    removed."""
    mu_abs, thick = 1.0 / 111.0, 750.0
    with float64():
        geom = {**_geom(), "sensor": (mu_abs, thick)}
        r, v, u = (np.asarray(y) for y in _sensor_kernel(jnp.array([300.0, 450.0]), geom))
    px = sp.Integer(75)
    p = sp.Matrix([150000, 1000 * 75, 1000 * 75]) + 300 * sp.Matrix([0, 0, -75]) + 450 * sp.Matrix([0, -75, 0])
    k = p / sp.sqrt(p.dot(p))
    nrm = sp.Matrix([1, 0, 0])  # (0, 0, -75) x (0, -75, 0) normalised: +x
    cos_phi = sp.Abs(nrm.dot(k))
    d = k - nrm.dot(k) * nrm
    d_pix = sp.Matrix([-d[2], -d[1]]) / px  # slow along -z, fast along -y
    L = sp.sqrt(d_pix.dot(d_pix))
    b, R = L / sp.Float(mu_abs, 30), L * thick / cos_phi
    P = 1 - sp.exp(-R / b)
    edges = [-b * sp.log(1 - sp.Rational(j, 4) * P) for j in range(4)] + [R]
    ms, vs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        elo, ehi = sp.exp(-lo / b), sp.exp(-hi / b)
        Z = elo - ehi
        m = b + (lo * elo - hi * ehi) / Z
        m2 = (elo * (lo**2 + 2 * lo * b + 2 * b**2) - ehi * (hi**2 + 2 * hi * b + 2 * b**2)) / Z
        ms.append(m)
        vs.append(m2 - m**2)
    mean = sum(ms) / 4
    e_r = [float(sp.N(x - mean, 30)) for x in ms]
    e_v = [float(sp.N(x, 30)) for x in vs]
    e_u = [float(sp.N(x / L, 30)) for x in d_pix]
    # ~100 operations of condition O(1) (the geometry), then the slices (as test_exponential_slices_code, whose
    # variances cancel by m2 / var < 20 here): 1e4 ulps of the largest of each
    np.testing.assert_allclose(u, e_u, rtol=0, atol=1e4 * EPS)
    np.testing.assert_allclose(r, e_r, rtol=0, atol=1e4 * EPS * max(abs(x) for x in e_r))
    np.testing.assert_allclose(v, e_v, rtol=0, atol=1e4 * EPS * max(e_v))


def test_sensor_mean_shift():
    """The kernel's mean shift, sin(phi) / mu (untruncated), is not a moved detector (which shifts by D tan(phi)):
    the best constant, 1 / mu, leaves (sin phi - tan phi) / mu = -phi^3 / (2 mu) + O(phi^5)."""
    ph = sp.symbols("phi", positive=True)
    assert is_zero(sp.series(sp.sin(ph) - sp.tan(ph), ph, 0, 5).removeO() + ph**3 / 2)
    x, m = sp.symbols("x mu", positive=True)
    assert sp.integrate(x * m * sp.exp(-m * x), (x, 0, sp.oo)) == 1 / m
    # at 2theta = 10 deg and 1 / mu = 111 um (43.6 keV in CdTe): -0.297 um
    val = (sp.sin(sp.pi / 18) - sp.tan(sp.pi / 18)) * 111
    assert f"{float(sp.N(val, 30)):.3g}" == "-0.297"


# ---------------------------------------------------------------------------------------------- det-spatial


def test_newton_affine():
    """Newton on c(x) = mu_c is exact in one step where c is affine, c(x) = c0 + M x."""
    M = sp.Matrix(2, 2, sp.symbols("m0:4", real=True))
    c0, x0, mu = (sp.Matrix(sp.symbols(f"{n}0:2", real=True)) for n in ("c", "x", "mu"))
    x1 = x0 - M.inv() * (c0 + M * x0 - mu)
    assert is_zero(c0 + M * x1 - mu)


def test_pullback():
    """A Gaussian N(mu_c, S) on the ideal detector, pulled back through an affine c with Jacobian A, is
    N(x0, A^-1 S A^-T) on the raw pixels, with the same mass (the density times |det A|), and its covariance with
    omega is A^-1 cov(c, omega)."""
    A = sp.Matrix([[sp.symbols("a"), sp.symbols("b")], [sp.symbols("c"), sp.symbols("d")]])
    s1, s2 = sp.symbols("s1 s2", positive=True)
    r = sp.symbols("r", real=True)
    S = sp.Matrix([[s1**2, r * s1 * s2], [r * s1 * s2, s2**2]])
    y = sp.Matrix(sp.symbols("y0:2", real=True))  # x - x0, so c - mu_c = A y
    Ai = A.inv()
    T = Ai * S * Ai.T
    assert is_zero((A * y).T * S.inv() * (A * y) - y.T * T.inv() * y)
    assert is_zero(T.det() - S.det() / A.det() ** 2)  # so N_c(c(x)) |det A| = N_x(x)
    # (c, omega) jointly Gaussian: x - x0 = A^-1 (c - mu_c) has covariance A^-1 cov(c, omega) with omega
    so, fo = sp.symbols("so fo", real=True)
    assert is_zero(Ai * sp.Matrix([so, fo]) - Ai * sp.Matrix([so, fo]))


def test_map_shape_code():
    """_map_shape against T S T^T and T c."""
    T = np.array([[1.02, 0.03], [-0.01, 0.97]])
    shape = (0.8, 0.5, 0.01, 0.2, 0.03, -0.02)
    with float64():
        got = [float(x) for x in _map_shape(tuple(jnp.asarray(x) for x in shape), jnp.asarray(T))]
    Ts = sp.Matrix(T)
    S = Ts * sp.Matrix([[shape[0], shape[3]], [shape[3], shape[1]]]) * Ts.T
    c = Ts * sp.Matrix([shape[4], shape[5]])
    expected = [S[0, 0], S[1, 1], shape[2], S[0, 1], c[0], c[1]]
    # two 2 x 2 products of positive-definite-ish terms, ~8 operations each: 32 ulps of the largest term
    np.testing.assert_allclose(got, [float(evaluate(x, {})) for x in expected], rtol=0, atol=32 * EPS)


def test_undistort_code():
    """_undistort against the exact inverse: an affine map (Newton exact in one step) and a bilinear map with a
    cross term (Newton converging quadratically: three steps from x = mu - d(mu) reach float64 here)."""
    n = 64
    i, j = np.mgrid[0:n, 0:n].astype(float)
    rr, cc = sp.symbols("r c", real=True)
    cases = [(0.4 + 0.01 * rr - 0.02 * cc, -0.3 + 0.015 * rr + 0.005 * cc),  # (dy, dx) as functions of (row, col)
             (0.4 + 0.01 * rr - 0.02 * cc + 1e-4 * rr * cc, -0.3 + 0.015 * rr + 2e-4 * rr * cc)]  # fmt: skip
    mu_c = (30.3, 20.6)
    for dy_e, dx_e in cases:
        dy = sp.lambdify((rr, cc), dy_e)(i, j) * np.ones_like(i)
        dx = sp.lambdify((rr, cc), dx_e)(i, j) * np.ones_like(i)
        with float64():
            x0, Ainv = _undistort(jnp.asarray(mu_c), {"spatial": (dx, dy)})
        sol = sp.nsolve([rr + dy_e - mu_c[0], cc + dx_e - mu_c[1]], [rr, cc], [30, 20], prec=30)
        A = sp.Matrix([[1 + sp.diff(dy_e, rr), sp.diff(dy_e, cc)], [sp.diff(dx_e, rr), 1 + sp.diff(dx_e, cc)]])
        A = A.subs({rr: sol[0], cc: sol[1]})
        # the interpolation of a (bi)linear field on integer nodes is exact up to rounding: a few ulps of |x| ~ 30
        # per Newton step: 3 steps x 4 operations x 4 ulps of 32
        np.testing.assert_allclose(np.asarray(x0), [float(sol[0]), float(sol[1])], rtol=0, atol=48 * 32 * EPS)
        # A^-1 from map_coordinates' gradient (differences of nodes ~ 1, scaled): 64 ulps
        np.testing.assert_allclose(np.asarray(Ainv), np.array(A.inv().evalf(30).tolist(), float), rtol=0,
                                   atol=64 * EPS)  # fmt: skip


# ---------------------------------------------------------------------------------------------- det-splat


def test_splat_bilinear():
    """Sharing a point at fractional offset d between its two neighbours keeps its mean and adds d (1 - d) <= 1/4 to
    its variance (per axis)."""
    d = sp.symbols("d", positive=True)
    mean = (1 - d) * 0 + d * 1
    var = (1 - d) * 0**2 + d * 1**2 - mean**2
    assert mean == d
    assert is_zero(var - d * (1 - d))
    assert sp.solve(sp.diff(var, d), d) == [sp.Rational(1, 2)] and var.subs(d, sp.Rational(1, 2)) == sp.Rational(1, 4)


def test_splat_bspline():
    """:eq:`det-splat-exact`: quadratic B-spline weights at -1, 0, 1 (offset e from the nearest grid point) are
    non-negative for |e| <= 1/2, sum to 1, have mean e and variance 1/4 whatever e; on a grid n times finer, 1 / (4 n^2)
    pixel^2. A kernel N(0, s^2 - 1 / (4 n^2)) shared so has mean e and variance s^2: the spot's two moments."""
    e = sp.symbols("e", real=True)
    w = [(sp.Rational(1, 2) - e) ** 2 / 2, sp.Rational(3, 4) - e**2, (sp.Rational(1, 2) + e) ** 2 / 2]
    x = [-1, 0, 1]
    assert is_zero(sum(w) - 1)
    mean = sum(wi * xi for wi, xi in zip(w, x))
    assert is_zero(mean - e)
    assert is_zero(sum(wi * xi**2 for wi, xi in zip(w, x)) - mean**2 - sp.Rational(1, 4))
    assert all(sp.solveset(wi < 0, e, sp.Interval(-sp.Rational(1, 2), sp.Rational(1, 2))) == sp.EmptySet for wi in w)
    n, s = sp.symbols("n s", positive=True)
    v_share = sp.Rational(1, 4) / n**2  # the same weights on a grid of step 1 / n
    assert is_zero((s**2 - v_share) + v_share - s**2)  # kernel then sharing: variances add (independent)
    # bilinear sharing's d (1 - d) reaches the B-spline's 1/4 only at d = 1/2: no constant kernel deficit fits it
    d = sp.symbols("d", real=True)
    assert sp.solve(d * (1 - d) - sp.Rational(1, 4), d) == [sp.Rational(1, 2)]


if __name__ == "__main__":
    unittest.main()


def test_spot_terms():
    """det-spot-terms. (1) Bragg, lambda = 2 d sin(theta): d 2theta / d lambda = 2 tan(theta) / lambda. (2) A ray
    o + t k meets the plane n . (P - d0) = 0 at P; moving o by e d (k fixed) moves P by e (d - k (n . d) / (n . k));
    turning a unit k by e dk (o = 0) moves it by e |P| (dk - k (n . dk) / (n . k)). (3) With g = k_out - x (unit
    vectors, Ewald), the beam turned by a x x and the crystal by d_omega about z keep |x + g| = 1 to first order, and
    so the diffracted ray's length."""
    lam, d = sp.symbols("lambda d", positive=True)
    th = sp.asin(lam / (2 * d))
    assert is_zero(sp.diff(2 * th, lam) * lam - 2 * sp.tan(th))
    e = sp.Symbol("e", real=True)
    o, k, n, d0, dd = vec("o"), vec("k"), vec("n"), vec("d"), vec("q")
    moved = sp.diff((o + e * dd) + n.dot(d0 - o - e * dd) / n.dot(k) * k, e).subs(e, 0)
    assert is_zero(moved - (dd - k * n.dot(dd) / n.dot(k)))
    ku = k / sp.sqrt(k.dot(k))
    Pk = n.dot(d0) / n.dot(ku) * ku  # o = 0, a unit direction
    dk = vec("dk")
    turned = sp.diff((n.dot(d0) / n.dot(ku + e * dk)) * (ku + e * dk), e).subs(e, 0)
    assert is_zero(turned - n.dot(d0) / n.dot(ku) * (dk - ku * n.dot(dk) / n.dot(ku)))
    assert is_zero(Pk.dot(Pk) - (n.dot(d0) / n.dot(ku)) ** 2)  # |P| = t for a unit ray (t > 0: the detector ahead)
    X, Z = sp.Matrix([1, 0, 0]), sp.Matrix([0, 0, 1])
    g, a = vec("g"), vec("a")
    dk_in = a.cross(X)
    zg = Z.cross(g)
    d_om = -g.dot(dk_in) / zg.dot(X)
    dg = d_om * zg
    ewald = sp.expand((X + e * dk_in + g + e * dg).dot(X + e * dk_in + g + e * dg) - (X + e * dk_in).dot(X + e * dk_in))
    for root in sp.solve(2 * g[0] + g.dot(g), g[0]):  # on the Ewald sphere at e = 0: 2 x . g + |g|^2 = 0
        assert is_zero(sp.diff(ewald, e).subs(e, 0).subs({g[0]: root}))


def test_spot_terms_code():
    """spot_terms against det-spot-terms, with the pixel's 2theta (frm-eta, from the origin, beam along x) and its
    exact gradient, for a tilted detector: the radial direction, the known variance and the box."""
    from anri.calib.detector import spot_terms
    from anri.io.imaged11 import detector_from_pars
    from anri.maths.utils import fwhm_to_sigma

    pars = {"y_center": 1049.9, "y_size": 75.0, "tilt_y": -2e-3, "z_center": 1116.5, "z_size": 75.0, "tilt_z": 3e-3,
            "tilt_x": 1e-3, "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1, "wavelength": 0.284}  # fmt: skip
    rows, cols = np.array([700, 1600, 1116]), np.array([1050, 900, 1700])
    bw, vox, thick, conv = 1e-3, 1.5, 100.0, (1e-4, 2e-4)
    with float64():
        det = {k: np.asarray(v, float) for k, v in detector_from_pars(pars).items()}
    got = spot_terms(rows, cols, pars, None, None, bw, vox, thick, conv)
    S, F, d0 = (sp.Matrix([sp.Float(x, 30) for x in det[k]]) for k in ("s_step_lab", "f_step_lab", "det_origin_lab"))
    A = sp.Matrix.hstack(S, F)
    Ap = (A.T * A).inv() * A.T
    n = S.cross(F)
    X, Y, Z = sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 0]), sp.Matrix([0, 0, 1])
    s_, f_ = sp.symbols("s f", real=True)
    sl = fwhm_to_sigma(sp.Float(bw, 30))
    for i, (r, c) in enumerate(zip(rows, cols)):
        P = d0 + s_ * S + f_ * F
        tth = sp.acos(P[0] / sp.sqrt(P.dot(P)))
        at = {s_: sp.Integer(int(r)), f_: sp.Integer(int(c))}
        grad = sp.Matrix([sp.diff(tth, v).subs(at) for v in (s_, f_)])
        u = grad / sp.sqrt(grad.dot(grad))
        Pv = P.subs(at)
        k = Pv / sp.sqrt(Pv.dot(Pv))
        m = lambda d, k=k: Ap * (d - k * n.dot(d) / n.dot(k))
        th = tth.subs(at) / 2
        var = (2 * sp.tan(th) * sl / sp.sqrt(grad.dot(grad))) ** 2 + (vox * u.dot(m(Y))) ** 2 / 12
        g = k - X
        for a, sig in zip((Z, Y), conv):
            dk = a.cross(X) - g.dot(a.cross(X)) / Z.cross(g).dot(X) * Z.cross(g)
            var += (sig * sp.sqrt(Pv.dot(Pv)) * u.dot(m(dk))) ** 2
        box = thick * abs(u.dot(m(X)))
        want_u = np.array([float(sp.N(x, 30)) for x in u])
        # ~100 operations from the pixel to each term, each of condition at most ~2 (the detector is near normal to
        # the rays, n . k ~ 1; A^T A ~ pixel^2 I): 256 ulps of each value; the radial direction, a unit vector, to
        # 256 ulps of 1
        np.testing.assert_allclose(got["u"][i], want_u, rtol=0, atol=256 * EPS)
        np.testing.assert_allclose(got["var_known"][i], float(sp.N(var, 30)), rtol=256 * EPS)
        np.testing.assert_allclose(got["box"][i], float(sp.N(box, 30)), rtol=256 * EPS)

