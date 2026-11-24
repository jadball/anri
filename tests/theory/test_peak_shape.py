"""Proofs for docs/source/theory/peak_shape.rst: propagation, the omega widths, truncated moments, the factorised box
mass and the windows.

Tolerances of the code checks are derived where they are stated: float64 has eps = 2.2e-16, and an expression of n
operations with condition number c is good to about n c eps.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from anri.fwd.peak import get_cov_in, make_propagator, propagate_cov
from anri.fwd.render import _peak_cov, _window_cells, make_row
from anri.maths.rotation import rmat_from_axis_angle, skew
from anri.maths.utils import bin_fractions, float64, truncated_moments
from tests.theory.common import (
    EPS,
    agree,
    evaluate,
    is_zero,
    rng,
    tol,
    vec,
)


def Phi(z: sp.Expr) -> sp.Expr:
    return (1 + sp.erf(z / sp.sqrt(2))) / 2


def phi(z: sp.Expr) -> sp.Expr:
    return sp.exp(-(z**2) / 2) / sp.sqrt(2 * sp.pi)


def truncated(lo: sp.Expr, hi: sp.Expr, mu: sp.Expr, sigma: sp.Expr) -> tuple:
    """Mass, mean and variance of N(mu, sigma^2) on [lo, hi): :eq:`ps-truncated`."""
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    Z = Phi(b) - Phi(a)
    r = (phi(a) - phi(b)) / Z
    return Z, mu + sigma * r, sigma**2 * (1 + (a * phi(a) - b * phi(b)) / Z - r**2)


def skew_sp(v: sp.MatrixBase) -> sp.Matrix:
    return sp.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# ---------------------------------------------------------------------------------------------- ps-propagate


def test_propagate_diagonal():
    """J S J^T = sum_k sigma_k^2 J_k J_k^T for a diagonal S."""
    J = sp.Matrix(3, 4, sp.symbols("j0:12"))
    s = sp.symbols("s0:4", positive=True)
    S = sp.diag(*[x**2 for x in s])
    assert is_zero(J * S * J.T - sum((s[k] ** 2 * J[:, k] * J[:, k].T for k in range(4)), sp.zeros(3)))


def _f_sp(x: sp.MatrixBase) -> sp.Matrix:
    return sp.Matrix([x[0] * x[1] + sp.sin(x[2]), x[1] ** 2 - x[0] * x[2], sp.exp(x[0]) + x[2]])


def _f_jax(x: jax.Array) -> jax.Array:
    return jnp.stack([x[0] * x[1] + jnp.sin(x[2]), x[1] ** 2 - x[0] * x[2], jnp.exp(x[0]) + x[2]])


def test_propagate_code():
    """make_propagator gives J S J^T, diagonal and full (an off-diagonal S, as sig_rot [3, 3])."""
    x = sp.Matrix(sp.symbols("x0:3"))
    J = _f_sp(x).jacobian(x)
    r = rng(1)
    x0 = r.normal(size=3)
    A = r.normal(size=(3, 3))
    S_full = A @ A.T
    S_diag = np.diag(np.diag(S_full))
    vals = dict(zip(x, x0))
    # ~20 operations with condition O(1) per element: 1e-14 is ~50 ulps
    for diagonal, S in ((True, S_diag), (False, S_full)):
        expected = evaluate(J * sp.Matrix(S) * J.T, vals)
        fn = make_propagator(_f_jax, argnums=(0,), diagonal=diagonal)
        with float64():
            got = np.asarray(fn(jnp.asarray(x0), jnp.asarray(S)))
            # propagate_cov takes jax.jacfwd's parts, as the box and scan propagators do
            parts = jax.jacfwd(_f_jax, argnums=(0,))(jnp.asarray(x0))
            got_parts = np.asarray(propagate_cov(parts, jnp.asarray(S)))
        # each J entry is ~5 operations of its inputs (rounded once each), J S J^T sums 9 products of 3: 16 ulps of
        # the sum of the terms' magnitudes
        Jv = evaluate(J, vals)
        t = 16 * EPS * (np.abs(Jv) @ np.abs(S) @ np.abs(Jv).T)
        np.testing.assert_array_less(np.abs(got - expected), t)
        np.testing.assert_array_less(np.abs(got_parts - evaluate(J * sp.Matrix(S) * J.T, vals)), t)


def test_cov_in_code():
    """get_cov_in is diag(sigma_x^2, sigma_y^2, sigma_z^2, sigma_lambda^2, sigma_ky^2, sigma_kz^2): squares, one
    rounding each."""
    o, sl, sky, skz = np.array([1.0, 2.0, 0.5]), 3e-4, 1e-4, 2e-4
    want = np.diag([float(sp.Float(x, 30) ** 2) for x in (*o, sl, sky, skz)])
    with float64():
        got = np.asarray(get_cov_in(jnp.asarray(o), sl, sky, skz))
    np.testing.assert_allclose(got, want, rtol=EPS, atol=0)


# ---------------------------------------------------------------------------------------------- ps-rotation


def test_rotation_first_order():
    """d/dt exp(t [delta]x) at t = 0 is [delta]x (Rodrigues), and (R UB)^-1 = UBI R^T."""
    t = sp.symbols("t", real=True)
    d = sp.Matrix(sp.symbols("d0:3", real=True))
    n = sp.sqrt(d.dot(d))
    K = skew_sp(d)
    R = sp.eye(3) + sp.sin(t * n) / n * K + (1 - sp.cos(t * n)) / n**2 * K * K
    assert is_zero(sp.diff(R, t).subs(t, 0) - K)
    assert is_zero(R.T * R - sp.eye(3))  # orthogonal, so R^-1 = R^T


def test_skew_code():
    """skew(v) is [v]x: its entries are v's, negated or not, so exactly."""
    v = rng(2).normal(size=3)
    with float64():
        got = np.asarray(skew(jnp.asarray(v)))
    np.testing.assert_array_equal(got, evaluate(skew_sp(sp.Matrix(v)), {}))


def _rodrigues(n: sp.MatrixBase, t: sp.Expr) -> sp.Matrix:
    """ps-rotation's exp(t [n]x) for a unit axis n, as Rodrigues' formula."""
    K = skew_sp(n)
    return sp.eye(3) + sp.sin(t) * K + (1 - sp.cos(t)) * K * K


def test_axis_angle():
    """ps-rotation: for a unit axis n, R(t) = I + sin t [n]x + (1 - cos t) [n]x^2 is exp(t [n]x): R(0) = I and
    R' = [n]x R (as [n]x^3 = -[n]x), the equation exp(t [n]x) alone solves. rmat_from_axis_angle's form,
    cos t I + sin t [n]x + (1 - cos t) n n^T, is the same matrix ([n]x^2 = n n^T - I)."""
    t = sp.symbols("t", real=True)
    a = vec("a")
    n = a / sp.sqrt(a.dot(a))
    K, R = skew_sp(n), _rodrigues(n, t)
    assert is_zero(R.subs(t, 0) - sp.eye(3))
    assert is_zero(sp.diff(R, t) - K * R)
    assert is_zero(sp.cos(t) * sp.eye(3) + sp.sin(t) * K + (1 - sp.cos(t)) * n * n.T - R)


def test_axis_angle_code():
    """rmat_from_axis_angle against exp(t [n]x) (test_axis_angle), unnormalised axes, one call for a batch."""
    r = rng(12)
    axes = r.normal(size=(4, 3)) * [[1.0], [0.3], [5.0], [2.0]]
    deg = np.array([-170.0, -12.5, 33.3, 179.0])
    want = []
    for ax, d in zip(axes, deg):
        a = sp.Matrix([sp.Float(x, 30) for x in ax])
        want.append(evaluate(_rodrigues(a / sp.sqrt(a.dot(a)), sp.rad(sp.Float(d, 30))), {}))
    # entries <= 1; the angle in radians (<= pi) and the axis's normalisation each round once
    agree(np.array(want), rmat_from_axis_angle, axes, deg, **tol(np.pi))


# ---------------------------------------------------------------------------------------------- ps-omega


def _ewald():
    lam, th, eta = sp.symbols("lambda theta eta", positive=True)
    k_in = sp.Matrix([1, 0, 0]) / lam
    k_out = sp.Matrix([sp.cos(2 * th), -sp.sin(2 * th) * sp.sin(eta), sp.sin(2 * th) * sp.cos(eta)]) / lam
    return lam, th, eta, k_in, k_out, k_out - k_in


def _domega(dk_in: sp.MatrixBase, dg: sp.MatrixBase, k_out: sp.MatrixBase, g: sp.MatrixBase) -> sp.Expr:
    """Omega turn (radians) that keeps 2 k_in.g + |g|^2 = 0 to first order, given changes of k_in and g."""
    w = sp.symbols("w")
    dF = 2 * dk_in.dot(g) + 2 * k_out.dot(dg + w * sp.Matrix([0, 0, 1]).cross(g))
    return sp.solve(dF, w)[0]


def test_omega_motion():
    """Each input's first-order omega shift: rotation, wavelength, horizontal and vertical beam tilt."""
    lam, th, eta, k_in, k_out, g = _ewald()
    assert is_zero(2 * k_in.dot(g) + g.dot(g))  # on the Ewald sphere
    d = sp.Matrix(sp.symbols("dx dy dz"))
    dlam, ky, kz = sp.symbols("dlambda k_y k_z")
    zero = sp.zeros(3, 1)
    # a lattice rotation delta (lab frame): g -> g + delta x g
    assert is_zero(_domega(zero, d.cross(g), k_out, g) - (-d[2] - d[1] * sp.cot(eta)))
    # wavelength: k_in = x / lambda
    assert is_zero(
        _domega(-sp.Matrix([1, 0, 0]) * dlam / lam**2, zero, k_out, g) + dlam / lam * sp.tan(th) / sp.sin(eta)
    )
    # beam tilts along e_h = y and e_v = z (beam_basis for a beam along x)
    assert is_zero(_domega(sp.Matrix([0, ky, 0]) / lam, zero, k_out, g) - ky)
    assert is_zero(_domega(sp.Matrix([0, 0, kz]) / lam, zero, k_out, g) + kz * sp.cot(eta))


def _omega_variance(eta: sp.Expr, th: sp.Expr, om: sp.Expr, S: sp.MatrixBase, sl: sp.Expr, sky: sp.Expr,
                    skz: sp.Expr) -> sp.Expr:  # fmt: skip
    """Var(omega) in degrees^2 from :eq:`ps-omega`: S the sample-frame rotation covariance, sl = sigma_lambda / lambda."""
    Rz = sp.Matrix([[sp.cos(om), -sp.sin(om), 0], [sp.sin(om), sp.cos(om), 0], [0, 0, 1]])
    J = sp.Matrix([[0, -sp.cot(eta), -1]])
    rad2 = (J * Rz * S * Rz.T * J.T)[0] + (sl * sp.tan(th) / sp.sin(eta)) ** 2 + sky**2 + (skz * sp.cot(eta)) ** 2
    return (180 / sp.pi) ** 2 * rad2


def _geom() -> dict:
    """Beam along x, a detector of 75 um pixels normal to it at 150 mm."""
    px = 75.0
    return {"wavelength": 0.3, "k_in_lab": jnp.array([1.0, 0.0, 0.0]), "wedge": 0.0, "chi": 0.0, "y0": 0.0,
            "s_step_lab": jnp.array([0.0, 0.0, -px]), "f_step_lab": jnp.array([0.0, -px, 0.0]),
            "det_origin_lab": jnp.array([150e3, 1000 * px, 1000 * px]), "voxel_3d": False,
            "sig_wavelength": 0.0, "sig_ky": 0.0, "sig_kz": 0.0}  # fmt: skip


def test_omega_variance_code():
    """_peak_cov's omega variance, from each spread and from all together, against :eq:`ps-omega`."""
    from anri.diffract.scatter import k_out_to_tth_eta
    from anri.fwd.peak import hkl_to_k_omega

    r = rng(3)
    eta_s, th_s, om_s = sp.symbols("eta theta omega")
    S_s = sp.Matrix(3, 3, sp.symbols("S0:9"))
    sl_s, sky_s, skz_s = sp.symbols("s_l s_ky s_kz")
    expr = _omega_variance(eta_s, th_s, om_s, S_s, sl_s, sky_s, skz_s)
    with float64():
        geom = _geom()
        ubi = jnp.asarray(np.linalg.qr(r.normal(size=(3, 3)))[0] * 3.0)
        for hkl in ([1.0, 1.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 1.0]):
            for etasign in (1.0, -1.0):
                k_in, k_out, om, valid = hkl_to_k_omega(ubi, jnp.asarray(hkl), etasign, 0.3, geom["k_in_lab"], 0.0,
                                                        0.0, 0.0, 0.0)  # fmt: skip
                assert bool(valid)
                tth, eta = k_out_to_tth_eta(k_out, k_in)
                A = r.normal(size=(3, 3)) * 1e-3
                S = A @ A.T
                sl, sky, skz = 1e-3, 2e-4, 3e-4
                g = {**geom, "sig_wavelength": 0.3 * sl, "sig_ky": sky, "sig_kz": skz}
                got = float(_peak_cov(ubi, jnp.zeros(3), jnp.asarray(hkl), etasign, 0.0, g, jnp.asarray(S))[2])
                vals = {eta_s: np.radians(float(eta)), th_s: np.radians(float(tth)) / 2, om_s: np.radians(float(om)),
                        sl_s: sl, sky_s: sky, skz_s: skz, **dict(zip(S_s, S.ravel()))}  # fmt: skip
                # forward mode differentiates the rounded chain: each derivative errs by N eps of itself, N the
                # chain's length (~200 operations, N = 256, as test_peak_cov_rotation_code); the variance, a sum of
                # squares of derivatives times spreads, by twice that of the sum of its terms' magnitudes
                Sabs = {**vals, **dict(zip(S_s, np.abs(S).ravel()))}
                mag = float(evaluate(_omega_variance(eta_s, th_s, om_s, S_s.applyfunc(sp.Abs), sl_s, sky_s, skz_s)
                                     .replace(sp.cot, lambda z: sp.Abs(sp.cot(z))), Sabs))  # fmt: skip
                np.testing.assert_allclose(got, evaluate(expr, vals), rtol=0, atol=2 * 256 * EPS * mag)


def test_peak_cov_rotation_code():
    """_peak_cov's orientation-spread term is J S J^T (ps-propagate) with J the centroid's exact derivative in a
    sample-frame rotation (ps-rotation, test_diffraction.py::centroid_rotation_jacobian): (s_c, f_c, omega), every
    element, for a [3, 3] covariance and an isotropic sd; tilted beam, goniometer and detector, off the axis."""
    from tests.theory.test_diffraction import _centroid_case, centroid_rotation_jacobian

    c = _centroid_case()
    S_det, F_det, d0 = c["det"]
    geom = {"wavelength": c["lam"], "k_in_lab": c["k"], "wedge": c["wedge"], "chi": c["chi"], "y0": c["y0"],
            "s_step_lab": S_det, "f_step_lab": F_det, "det_origin_lab": d0, "voxel_3d": False,
            "sig_wavelength": 0.0, "sig_ky": 0.0, "sig_kz": 0.0}  # fmt: skip
    A = rng(26).normal(size=(3, 3)) * 1e-3
    S = A @ A.T
    for hkl, es in (((1, 1, 0), 1.0), ((2, 1, 1), -1.0)):
        J = centroid_rotation_jacobian(c["ubi"], c["pos"], hkl, es, c["lam"], c["k"], c["wedge"], c["chi"], 1.3,
                                       c["y0"], c["det"], True)  # fmt: skip
        for sig in (S, 1.5e-3):
            Sm = S if np.ndim(sig) == 2 else sig**2 * np.eye(3)
            C = J @ Sm @ J.T
            want = np.array([C[0, 0], C[1, 1], C[2, 2], C[0, 1], C[0, 2], C[1, 2]])  # _COV_ELEMS
            with float64():
                got = np.asarray(_peak_cov(jnp.asarray(c["ubi"]), jnp.asarray(c["pos"]), jnp.asarray(hkl, float), es,
                                           1.3, {k: jnp.asarray(v) for k, v in geom.items()}, jnp.asarray(sig)))  # fmt: skip
            # forward mode differentiates the rounded chain: each derivative errs by N eps of |J|, N the chain's
            # length (~200 operations from inv3 to raytrace_to_det: N = 256), and J S J^T by twice that
            Ja = np.abs(J)
            t = 2 * 256 * EPS * (Ja @ np.abs(Sm) @ Ja.T)
            bound = np.array([t[0, 0], t[1, 1], t[2, 2], t[0, 1], t[0, 2], t[1, 2]])
            np.testing.assert_array_less(np.abs(got - want), bound)


# ---------------------------------------------------------------------------------------------- ps-truncated


def test_truncated_moments():
    """The first two moments of N(mu, sigma^2) on [l, u), by integration."""
    z, a, b = sp.symbols("z a b", real=True)
    Z = Phi(b) - Phi(a)
    assert is_zero(sp.integrate(phi(z), (z, a, b)) - Z)
    m1 = sp.integrate(z * phi(z), (z, a, b))
    m2 = sp.integrate(z**2 * phi(z), (z, a, b))
    assert is_zero(m1 - (phi(a) - phi(b)))
    assert is_zero(m2 - (Z + a * phi(a) - b * phi(b)))
    # so E[z] = m1 / Z and Var[z] = m2 / Z - (m1 / Z)^2, i.e. :eq:`ps-truncated` with x = mu + sigma z
    r = (phi(a) - phi(b)) / Z
    assert is_zero(m2 / Z - (m1 / Z) ** 2 - (1 + (a * phi(a) - b * phi(b)) / Z - r**2))


def test_upper_tail_mass():
    """Phi(-a) - Phi(-b) = Phi(b) - Phi(a): the form bin_fractions uses for a > 0."""
    a, b = sp.symbols("a b", real=True)
    assert is_zero(Phi(-a) - Phi(-b) - (Phi(b) - Phi(a)))


def test_truncated_moments_code():
    """truncated_moments and bin_fractions against :eq:`ps-truncated`, in both tails and for bins narrow and wide."""
    lo, hi, mu, sg = sp.symbols("lo hi mu sigma", real=True)
    mass, mean, var = truncated(lo, hi, mu, sg)
    for l_, h_, m_, s_ in [(-0.5, 0.5, 0.2, 0.3), (0.5, 1.5, 0.1, 0.4), (2.0, 2.5, -0.3, 0.6), (-3.0, -1.0, 0.0, 1.0),
                           (-0.05, 0.05, 0.3, 2.0), (4.0, 5.0, 0.0, 1.0)]:  # fmt: skip
        vals = {lo: l_, hi: h_, mu: m_, sg: s_}
        e_mass, e_mean, e_var = (float(evaluate(x, vals)) for x in (mass, mean, var))
        with float64():
            got = [float(x) for x in truncated_moments(l_, h_, m_, s_)]
            got_bf = float(bin_fractions(l_, h_, m_, s_))
        # the mass is a difference of two Phi of the same tail (a few ulps each), relative to the mass:
        # condition max(Phi(-|a|), Phi(-|b|)) / mass
        a_, b_ = (l_ - m_) / s_, (h_ - m_) / s_
        tail = max(float(evaluate(Phi(-abs(sp.Float(a_))), {})), float(evaluate(Phi(-abs(sp.Float(b_))), {})))
        c_mass = 1.0 + tail / e_mass
        np.testing.assert_allclose(got[0], e_mass, rtol=16 * EPS * c_mass)
        np.testing.assert_allclose(got_bf, e_mass, rtol=16 * EPS * c_mass)
        # the mean: mu + sigma r, r ~ (phi(a) - phi(b)) / mass, with the mass's error and the cancellation in
        # phi(a) - phi(b) (condition (phi(a) + phi(b)) / |phi(a) - phi(b)|), scaled by sigma |r| / |mean|
        pa, pb = (float(evaluate(phi(sp.Float(x)), {})) for x in (a_, b_))
        r_ = (pa - pb) / e_mass
        c_mean = (abs(m_) + s_ * abs(r_) * (c_mass + (pa + pb) / max(abs(pa - pb), 1e-300))) / max(abs(e_mean), 1e-300)
        np.testing.assert_allclose(got[1], e_mean, rtol=32 * EPS * c_mean)
        # the variance: sigma^2 (1 + t - r^2), cancelling to var: condition sigma^2 (1 + |t| + r^2) / var, with t
        # and r^2 themselves good to c_mean-like relative errors
        t_ = (a_ * pa - b_ * pb) / e_mass
        c_var = s_**2 * (1 + abs(t_) + r_**2) / e_var * c_mean
        np.testing.assert_allclose(got[2], e_var, rtol=32 * EPS * c_var)


# ---------------------------------------------------------------------------------------------- ps-condition


def test_condition_slow():
    """Slow given omega in a frame: the Gaussian regression (a Schur complement), then total mean and variance."""
    ss, oo, so = sp.symbols("Sigma_ss Sigma_oo Sigma_so", positive=True)
    S = sp.Matrix([[ss, so], [so, oo]])
    # Var(s | omega) = 1 / (S^-1)_ss
    assert is_zero(1 / S.inv()[0, 0] - (ss - so**2 / oo))
    # s = mu_s + kappa (omega - mu_o) + tau Z, omega with mean m, variance v, independent of Z
    mu_s, mu_o, m, v = sp.symbols("mu_s mu_o m v", real=True)
    kappa, tau2 = so / oo, ss - so**2 / oo
    E_s = mu_s + kappa * (m - mu_o)
    E_s2 = (mu_s - kappa * mu_o) ** 2 + 2 * (mu_s - kappa * mu_o) * kappa * m + kappa**2 * (v + m**2) + tau2
    assert is_zero(E_s2 - E_s**2 - (ss - so**2 / oo + kappa**2 * v))


def test_condition_fast():
    """Fast given (slow in a row, omega in a frame): the code's mean and variance from the within-frame joint
    Gaussian, by Schur complements and the law of total variance."""
    ss, ff, oo, sf, so, fo = sp.symbols("ss ff oo sf so fo", real=True)
    v = sp.symbols("v", positive=True)
    S2 = sp.Matrix([[ss, so], [so, oo]])
    g = sp.Matrix([[sf, fo]]) * S2.inv()
    g_s, g_o = g[0], g[1]
    det = ss * oo - so**2
    assert is_zero(g_s - (sf * oo - fo * so) / det) and is_zero(g_o - (fo * ss - sf * so) / det)
    V_f = ff - (g_s * sf + g_o * fo)
    assert is_zero(V_f - 1 / sp.Matrix([[ss, sf, so], [sf, ff, fo], [so, fo, oo]]).inv()[1, 1])
    # within the frame, centred: omega = sqrt(v) Z1, s = kappa omega + tau Z2, f = g_s s + g_o omega + sqrt(V_f) Z3
    kappa, tau2 = so / oo, ss - so**2 / oo
    L = sp.Matrix([[sp.sqrt(v), 0, 0], [kappa * sp.sqrt(v), sp.sqrt(tau2), 0], [0, 0, 0]])
    L[2, :] = g_s * L[1, :] + g_o * L[0, :]
    C = L * L.T
    C[2, 2] += V_f  # covariance of (omega, s, f)
    var_s = kappa**2 * v + tau2  # sd_s_o^2
    k = kappa * v / var_s
    assert is_zero(C[0, 1] / C[1, 1] - k)  # regression of omega on slow
    assert is_zero(C[0, 0] - C[0, 1] ** 2 / C[1, 1] - v * (1 - k * kappa))  # var_o_s
    var_f_s = C[2, 2] - C[2, 1] ** 2 / C[1, 1]
    assert is_zero(C[2, 1] / C[1, 1] - (g_s + g_o * k))  # dfds
    assert is_zero(var_f_s - (V_f + g_o**2 * v * (1 - k * kappa)))
    # then over slow in the row (mean m_s, variance v_s): law of total variance adds dfds^2 v_s


def _n(xs: tuple) -> tuple:
    return tuple(sp.N(x, 30) for x in xs)


def _cell_masses(mu: np.ndarray, shape: tuple, edges: np.ndarray, fclip: np.ndarray, rows: np.ndarray,
                 cols: np.ndarray) -> np.ndarray:  # fmt: skip
    """:eq:`ps-factor` with :eq:`ps-condition` and :eq:`ps-condition-f`, evaluated with sympy."""
    mu_s, mu_f, mu_o = (sp.Float(x, 30) for x in mu)
    ss, ff, oo, sf, so, fo = (sp.Float(x, 30) for x in shape)
    det = ss * oo - so**2
    g_s, g_o = (sf * oo - fo * so) / det, (fo * ss - sf * so) / det
    V_f = ff - (g_s * sf + g_o * fo)
    out = np.zeros((len(fclip), rows.shape[1], cols.shape[1]))
    for i, fr in enumerate(fclip):
        p_o, m_o, v_o = _n(truncated(sp.Float(edges[fr], 30), sp.Float(edges[fr + 1], 30), mu_o, sp.sqrt(oo)))
        kappa = so / oo
        mu_so, var_so = _n((mu_s + kappa * (m_o - mu_o), ss - so**2 / oo + kappa**2 * v_o))
        k = kappa * v_o / var_so
        for j, rw in enumerate(rows[i]):
            p_s, m_s, v_s = _n(truncated(sp.Float(rw - 0.5, 30), sp.Float(rw + 0.5, 30), mu_so, sp.sqrt(var_so)))
            mean_f = mu_f + g_s * (m_s - mu_s) + g_o * (m_o + k * (m_s - mu_so) - mu_o)
            var_f = sp.N(V_f + g_o**2 * v_o * (1 - k * kappa) + (g_s + g_o * k) ** 2 * v_s, 30)
            for c, cl in enumerate(cols[i]):
                p_f = truncated(sp.Float(cl - 0.5, 30), sp.Float(cl + 0.5, 30), mean_f, sp.sqrt(var_f))[0]
                out[i, j, c] = float(sp.N(p_o * p_s * p_f, 30))
    return out


def test_window_cells_code():
    """_window_cells' cell masses against the factorised box mass, for a correlated peak narrower than a frame."""
    omega = np.arange(-0.2, 0.2001, 0.05)
    mu = np.array([20.3, 30.6, 0.013])
    shape = (0.8, 0.5, 0.0009, 0.2, 0.012, -0.008)  # (ss, ff, oo, sf, so, fo): positive definite
    with float64():
        r = make_row(omega, np.zeros_like(omega))
        frac, _, fclip, rows, cols = _window_cells(jnp.asarray(mu), tuple(jnp.asarray(x) for x in shape), None, r,
                                                   (3, 5, 5), (64, 64))[:5]  # fmt: skip
    expected = _cell_masses(mu, shape, np.asarray(r["omega_edges"]), np.asarray(fclip), np.asarray(rows),
                            np.asarray(cols))  # fmt: skip
    # absolute: each cell is a product of three Phi differences of O(1) arguments (|z| < 10), each argument from
    # ~30 operations of condition O(10) (the regressions divide by det / (ss oo) ~ 0.8): ~1e3 ulps on a value <= 1
    np.testing.assert_allclose(np.asarray(frac), expected, rtol=0, atol=1e3 * EPS)


# ---------------------------------------------------------------------------------------------- ps-factor error


def test_cumulants_add():
    """The conditional of slow given omega in a frame is tau Z + kappa omega: its cumulants beyond the second are
    kappa^n times omega's (cumulant generating functions add); a uniform omega has kappa_4 = -Delta^4 / 120."""
    t, kappa, tau = sp.symbols("t kappa tau", positive=True)
    K_w = sp.Function("K")  # omega's cumulant generating function
    K_s = tau**2 * t**2 / 2 + K_w(kappa * t)
    for n in (3, 4):
        assert is_zero(sp.diff(K_s, t, n).subs(t, 0) - kappa**n * sp.diff(K_w(t), t, n).subs(t, 0))
    x, D = sp.symbols("x Delta", positive=True)
    mom = [sp.integrate(x**n / D, (x, -D / 2, D / 2)) for n in range(5)]
    assert is_zero(mom[2] - D**2 / 12)
    assert is_zero(mom[4] - 3 * mom[2] ** 2 + D**4 / 120)
    assert sp.nsimplify((mom[4] - 3 * mom[2] ** 2) / mom[2] ** 2) == sp.Rational(-6, 5)


def test_edgeworth_step():
    """The leading Edgeworth terms integrate in closed form over a bin: d/dz (phi He_{n-1}) = -phi He_n."""
    z = sp.symbols("z", real=True)
    He = {2: z**2 - 1, 3: z**3 - 3 * z, 4: z**4 - 6 * z**2 + 3}
    for n in (3, 4):
        assert is_zero(sp.diff(phi(z) * He[n - 1], z) + phi(z) * He[n])


def test_uniform_limit():
    """A streak uniform over one row (tau = 0, frame << sigma_omega) rendered as the Gaussian of equal variance:
    that row gets 2 Phi(sqrt(3)) - 1 = 0.917 of it, not 1."""
    D = sp.symbols("Delta", positive=True)
    sd = D / sp.sqrt(12)
    mass = Phi(D / 2 / sd) - Phi(-D / 2 / sd)
    assert is_zero(mass - sp.erf(sp.sqrt(sp.Rational(3, 2))))
    assert f"{float(sp.N(mass, 30)):.3f}" == "0.917"


# ---------------------------------------------------------------------------------------------- windows


def test_cantelli():
    """For any spot of variance sigma^2, P(x - mu >= t) <= sigma^2 / (sigma^2 + t^2): the bound
    (sigma^2 + u^2) / (t + u)^2 (Markov on (x - mu + u)^2) is least at u = sigma^2 / t."""
    s, t, u = sp.symbols("sigma t u", positive=True)
    bound = (s**2 + u**2) / (t + u) ** 2
    assert sp.solve(sp.diff(bound, u), u) == [s**2 / t]
    assert is_zero(bound.subs(u, s**2 / t) - s**2 / (s**2 + t**2))
    # at 3.5 sigma: 0.0755 for any spot, Phi(-3.5) = 0.000233 for a Gaussian
    assert f"{float(sp.N((s**2 / (s**2 + t**2)).subs(t, sp.Rational(7, 2) * s), 30)):.4f}" == "0.0755"
    assert f"{float(sp.N(Phi(-sp.Rational(7, 2)), 30)):.3g}" == "0.000233"


def test_captured_code():
    """A window wide on the detector captures the Gaussian's omega mass over its frames, exactly (the slow and fast
    masses of each frame sum to 1 up to their far tails, 1e-30 at 12 sd)."""
    omega = np.arange(-0.2, 0.2001, 0.05)
    mu = np.array([30.2, 30.7, 0.01])
    shape = (0.25, 0.25, 0.0016, 0.05, 0.004, -0.002)
    with float64():
        r = make_row(omega, np.zeros_like(omega))
        frac, _, fclip = _window_cells(jnp.asarray(mu), tuple(jnp.asarray(x) for x in shape), None, r, (3, 13, 13),
                                       (64, 64))[:3]  # fmt: skip
    edges = np.asarray(r["omega_edges"])
    fc = np.asarray(fclip)
    o = sp.symbols("o")
    expected = float(sp.N((Phi((o - mu[2]) / sp.sqrt(shape[2]))).subs(o, edges[fc[-1] + 1])
                          - Phi((o - mu[2]) / sp.sqrt(shape[2])).subs(o, edges[fc[0]]), 30))  # fmt: skip
    # a sum of 507 cells, each good to ~1e3 ulps absolute (test_window_cells_code)
    np.testing.assert_allclose(float(np.sum(np.asarray(frac))), expected, rtol=0, atol=507 * 1e3 * EPS)


def test_faces_partition():
    """:eq:`ps-faces` and :eq:`ps-face-sum`: the cells beyond each face telescope to a tail, and the window's cells and
    the six faces add up to the scan's light. A window of 2 frames (of a scan of 4), 2 rows and 2 columns per frame,
    with the tails as symbols: the cell masses of :eq:`ps-factor` are differences of Phi at the cells' edges."""
    F = sp.Function("Phi")
    k, K = sp.symbols("k K", integer=True, positive=True)
    z, h = sp.symbols("z h", real=True)
    # rows below a face at z (edges z - k h): K of them telescope to Phi(z) - Phi(z - K h), Phi(-oo) = 0
    assert is_zero(sp.summation(F(z - (k - 1) * h) - F(z - k * h), (k, 1, K)).doit() - (F(z) - F(z - K * h)))
    assert Phi(-sp.oo) == 0 and Phi(sp.oo) == 1  # K -> oo: no cells left beyond
    # omega: edges e0..e4, window frames 1 and 2
    e = sp.symbols("e0:5", real=True)
    P = [Phi(e[j + 1]) - Phi(e[j]) for j in range(4)]
    L0, L1 = Phi(e[1]) - Phi(e[0]), Phi(e[4]) - Phi(e[3])
    captured, L2, L3, L4, L5 = 0, 0, 0, 0, 0
    for j in (1, 2):
        r = sp.symbols(f"r{j}_0:3", real=True)  # the row window's edges, standardised by slow given frame j
        L2 += P[j] * Phi(r[0])
        L3 += P[j] * Phi(-r[2])  # the upper tail, 1 - Phi(r[2]) (test_upper_tail_mass)
        for i in (0, 1):
            p_s = Phi(r[i + 1]) - Phi(r[i])
            c = sp.symbols(f"c{j}{i}_0:3", real=True)  # the column window's edges, standardised by fast given the row
            L4 += P[j] * p_s * Phi(c[0])
            L5 += P[j] * p_s * Phi(-c[2])
            captured += P[j] * p_s * ((Phi(c[1]) - Phi(c[0])) + (Phi(c[2]) - Phi(c[1])))
    assert is_zero(captured + L0 + L1 + L2 + L3 + L4 + L5 - (Phi(e[4]) - Phi(e[0])))


def _faces_sp(mu: np.ndarray, shape: tuple, edges: np.ndarray, origin: tuple, window: tuple) -> tuple:
    """:eq:`ps-faces` evaluated with sympy at 30 digits: the six L_k, and per face the largest T(1) / T(0) over its
    Gaussians (:eq:`ps-face-reach`)."""
    mu_s, mu_f, mu_o = (sp.Float(x, 30) for x in mu)
    ss, ff, oo, sf, so, fo = (sp.Float(x, 30) for x in shape)
    jo, i0, j0 = (np.asarray(x) for x in origin)
    wo, ws, wf = window
    n = len(edges) - 1
    det = ss * oo - so**2
    g_s, g_o = (sf * oo - fo * so) / det, (fo * ss - sf * so) / det
    V_f = ff - (g_s * sf + g_o * fo)
    sd_o = sp.sqrt(oo)
    E = [sp.Float(x, 30) for x in edges]

    def t(x: sp.Expr) -> sp.Expr:
        return Phi((x - mu_o) / sd_o)

    lo, hi = min(max(int(jo), 0), n), min(max(int(jo) + wo, 0), n)
    L = [t(E[lo]) - t(E[0]), t(E[n]) - t(E[hi]), 0, 0, 0, 0]
    T1 = [t(E[max(lo - 1, 0)]) - t(E[0]), t(E[n]) - t(E[min(hi + 1, n)])]
    rho = [T1[0] / L[0] if L[0] > 0 else 0, T1[1] / L[1] if L[1] > 0 else 0, 0, 0, 0, 0]
    for a in range(wo):
        fr = int(jo) + a
        if not 0 <= fr < n:
            continue
        p_o, m_o, v_o = _n(truncated(E[fr], E[fr + 1], mu_o, sd_o))
        kappa = so / oo
        mu_so, var_so = _n((mu_s + kappa * (m_o - mu_o), ss - so**2 / oo + kappa**2 * v_o))
        sd_so = sp.sqrt(var_so)
        kk = kappa * v_o / var_so
        for face, x in ((2, (i0[a] - sp.Rational(1, 2) - mu_so) / sd_so),
                        (3, (mu_so - i0[a] - ws + sp.Rational(1, 2)) / sd_so)):  # fmt: skip
            L[face] += p_o * Phi(x)
            rho[face] = max(rho[face], Phi(x - 1 / sd_so) / Phi(x))
        for b in range(ws):
            rw = i0[a] + b
            p_s, m_s, v_s = _n(truncated(rw - sp.Rational(1, 2), rw + sp.Rational(1, 2), mu_so, sd_so))
            mean_f = mu_f + g_s * (m_s - mu_s) + g_o * (m_o + kk * (m_s - mu_so) - mu_o)
            sd_f = sp.sqrt(V_f + g_o**2 * v_o * (1 - kk * kappa) + (g_s + g_o * kk) ** 2 * v_s)
            for face, x in ((4, (j0[a] - sp.Rational(1, 2) - mean_f) / sd_f),
                            (5, (mean_f - j0[a] - wf + sp.Rational(1, 2)) / sd_f)):  # fmt: skip
                L[face] += p_o * p_s * Phi(x)
                rho[face] = max(rho[face], Phi(x - 1 / sd_f) / Phi(x))
    return np.array([float(sp.N(x, 30)) for x in L]), np.array([float(sp.N(x, 30)) for x in rho])


def test_faces_code():
    """_window_cells' light beyond each face of a correlated peak's window, and each face's decay, against
    :eq:`ps-faces` and :eq:`ps-face-reach`; with the window's cells they make up the scan's light (:eq:`ps-face-sum`).
    Two windows: one cutting the peak on every face, and one reaching past the scan's first frame."""
    omega = np.arange(-0.2, 0.2001, 0.05)
    shape = (0.8, 0.5, 0.0016, 0.2, 0.012, -0.008)  # (ss, ff, oo, sf, so, fo): positive definite
    window = (3, 3, 3)
    for mu in (np.array([20.3, 30.6, 0.013]), np.array([20.3, 30.6, -0.21])):
        with float64():
            r = make_row(omega, np.zeros_like(omega))
            out = _window_cells(jnp.asarray(mu), tuple(jnp.asarray(x) for x in shape), None, r, window, (64, 64))
        frac, origin, (f_o, f_fr, decay) = out[0], out[7], out[8]
        face = np.concatenate([np.asarray(f_o), np.asarray(f_fr).sum(0)])
        edges = np.asarray(r["omega_edges"])
        L, rho = _faces_sp(mu, shape, edges, origin, window)
        assert L.min() >= 0 and L[1] > 1e-3 and L[2:].min() > 1e-3  # every face but the clipped one holds light
        # each L_k is a sum of at most wo x ws = 9 products of Phi of O(1) arguments, each argument from ~30
        # operations of condition O(10) (as test_window_cells_code): 1e3 ulps per term, absolute
        np.testing.assert_allclose(face, L, rtol=0, atol=9 * 1e3 * EPS)
        # a ratio of two tails, each good to 1e3 ulps relative to its own size where it holds light (Phi is accurate
        # relative to itself in the lower tail): 2e3 ulps relative
        np.testing.assert_allclose(np.asarray(decay), rho, rtol=2e3 * EPS, atol=0)
        real = (int(origin[0]) + np.arange(window[0]) >= 0) & (int(origin[0]) + np.arange(window[0]) < len(omega))
        in_range = float(sp.N(Phi((edges[-1] - mu[2]) / sp.sqrt(sp.Float(shape[2], 30)))
                              - Phi((edges[0] - mu[2]) / sp.sqrt(sp.Float(shape[2], 30))), 30))  # fmt: skip
        # 27 cells and 6 faces, each good to the tolerances above
        total = np.asarray(frac)[real].sum() + face.sum()
        np.testing.assert_allclose(total, in_range, rtol=0, atol=(27 + 6 * 9) * 1e3 * EPS)
        if mu[2] < omega[0]:
            assert int(origin[0]) < 0 and face[0] == 0.0 and np.asarray(decay)[0] == 0.0  # no frames before the scan


def test_face_reach():
    """:eq:`ps-face-reach`: adding a value at least the mean raises a weighted mean, so the ratios of a log-concave
    sequence's tails fall, T(m) <= T(0) rho^m; checked on a binned Gaussian's tails; and a mixture's bound."""
    S, B, b, r_ = sp.symbols("S B b r", positive=True)
    assert is_zero((S + b * r_) / (B + b) - S / B - b * (r_ * B - S) / (B * (B + b)))
    bs = sp.symbols("b0:4", positive=True)
    rs = sp.symbols("r0:4", positive=True)
    S4, B4 = sum(x * y for x, y in zip(bs, rs)), sum(bs)
    assert is_zero(r_ * B4 - S4 - sum(x * (r_ - y) for x, y in zip(bs, rs)))  # >= 0 when r >= every r_k
    # a Gaussian of sd 0.7 in unit cells, its face 0.3 sd from its mean: the tails' ratios fall, T(m) <= T(0) rho^m
    T = [Phi(sp.Rational(3, 10) / sp.Rational(7, 10) * -1 - sp.Rational(10, 7) * m) for m in range(8)]
    ratios = [sp.N(T[m + 1] / T[m], 30) for m in range(7)]
    assert all(x > y for x, y in zip(ratios, ratios[1:]))
    assert all(sp.N(T[m] - T[0] * ratios[0] ** m, 30) <= 0 for m in range(8))
    # the mixture: sum_g T_g rho_g^m <= (sum_g T_g) max_g rho_g^m, term by term
    t1, t2, p1, p2 = sp.symbols("t1 t2 rho1 rho2", positive=True)
    m = sp.symbols("m", positive=True)
    assert is_zero((t1 + t2) * p1**m - (t1 * p1**m + t2 * p2**m) - t2 * (p1**m - p2**m))  # >= 0 for rho1 >= rho2


def test_make_row_edges():
    """Frame edges are the midpoints between sorted frame centres, the end frames as wide as their neighbours."""
    omega = np.array([0.3, 0.0, 0.1, 0.25, 0.6])
    with float64():
        row = make_row(omega, np.zeros(5))
    om = sorted(sp.nsimplify(x) for x in omega)
    mids = [(a + b) / 2 for a, b in zip(om[:-1], om[1:])]
    expected = [2 * om[0] - mids[0], *mids, 2 * om[-1] - mids[-1]]
    # the code computes the same sums and halvings in float64 from the binary inputs: a few ulps of |omega| <= 1
    np.testing.assert_allclose(np.asarray(row["omega_edges"]), [float(x) for x in expected], rtol=0, atol=4 * EPS)


def test_window_origin_code():
    """The window's origin (peak_shape.rst, Windows): its first frame the one holding mu_omega less half the window,
    each frame's first row round(mu_{s|F}) (ps-condition, with ps-truncated's mean omega in the frame) and first column
    round(mu_f + S_fo / S_oo (m_omega - mu_omega)), less half the window. Exact integers."""
    omega = np.arange(-0.2, 0.2001, 0.05)
    shape = (0.8, 0.5, 0.0016, 0.2, 0.012, -0.008)  # (ss, ff, oo, sf, so, fo)
    window = (3, 5, 5)
    for mu in (np.array([20.3, 30.6, 0.013]), np.array([20.7, 30.2, -0.11])):
        with float64():
            r = make_row(omega, np.zeros_like(omega))
            out = _window_cells(jnp.asarray(mu), tuple(jnp.asarray(x) for x in shape), None, r, window, (64, 64))
        jo, i0, j0 = (np.asarray(x) for x in out[7])
        edges = [sp.nsimplify(x) for x in np.asarray(r["omega_edges"])]
        k = sum(1 for e in edges if e <= sp.Float(mu[2], 30)) - 1
        assert int(jo) == k - window[0] // 2
        _, _, oo, _, so, fo = (sp.Float(x, 30) for x in shape)
        ms, mf, mo = (sp.Float(x, 30) for x in mu)
        for n, f in enumerate(range(int(jo), int(jo) + window[0])):
            _, m, _ = truncated(edges[f], edges[f + 1], mo, sp.sqrt(oo))
            s_mu = ms + so / oo * (m - mo)
            f_mu = mf + fo / oo * (m - mo)
            assert int(i0[n]) == int(sp.floor(s_mu + sp.Rational(1, 2))) - window[1] // 2, (n, s_mu)
            assert int(j0[n]) == int(sp.floor(f_mu + sp.Rational(1, 2))) - window[2] // 2, (n, f_mu)


if __name__ == "__main__":
    unittest.main()
