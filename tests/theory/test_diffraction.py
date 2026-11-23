"""Proofs of docs/source/theory/diffraction.rst: the Ewald condition, Bragg's law, the omega solutions and their
branches, the diffracted ray, dty in scanning 3DXRD, the ray trace and how a small rotation moves a peak."""

import jax
import numpy as np
import sympy as sp

from anri.diffract import scatter
from anri.geom import gonio
from anri.maths.rotation import rmat_from_axis_angle
from anri.maths.utils import float64
from tests.theory.common import (
    agree,
    beam_frame,
    ev,
    evaluate,
    is_zero,
    rng,
    rx,
    ry,
    rz,
    skew,
    tol,
    vec,
)

X = sp.Matrix([1, 0, 0])
Z = sp.Matrix([0, 0, 1])


def test_ewald():
    """dif-ewald: |g + k_in|^2 = |k_in|^2 (dif-laue with equal lengths) is 2 (g . k_in + |g|^2 / 2) = 0."""
    g, k = vec("g"), vec("k")
    assert is_zero((g + k).dot(g + k) - k.dot(k) - 2 * (g.dot(k) + g.dot(g) / 2))


def test_bragg():
    """dif-bragg: with 2 theta between k_in and k_out, both of length 1 / lambda, |g| = |k_out - k_in| = 2 sin(theta)
    / lambda."""
    th, eta, lam = sp.symbols("theta eta lambda", positive=True)
    k_in = X / lam
    k_out = sp.Matrix([sp.cos(2 * th), -sp.sin(2 * th) * sp.sin(eta), sp.sin(2 * th) * sp.cos(eta)]) / lam
    g = k_out - k_in
    assert is_zero(g.dot(g) - (2 * sp.sin(th) / lam) ** 2)


def test_bragg_code():
    ds, lam = sp.symbols("d_s lambda", positive=True)
    tth = 2 * sp.deg(sp.asin(lam * ds / 2))
    for d, w in ((0.4, 0.17), (1.1, 0.3), (2.0, 0.9)):
        # degrees up to 180; arcsin has condition 1 / cos(theta)
        agree(evaluate(tth, {ds: d, lam: w}), scatter.ds_to_tth, d, w, **tol(180.0 / np.sqrt(1 - (d * w / 2) ** 2)))


def test_omega_harmonic():
    """dif-omega-harmonic: R_z(omega) turns only g_perp, so k . R_z(omega) g = alpha cos(omega) + beta sin(omega) +
    k . g_par, and dif-ewald (|g| unchanged by the rotation) becomes alpha cos(omega) + beta sin(omega) = delta."""
    g, k = vec("g"), vec("k")
    om = sp.Symbol("omega", real=True)
    g_par = g.dot(Z) * Z
    g_perp = g - g_par
    alpha, beta = k.dot(g_perp), k.dot(Z.cross(g_perp))
    delta = -g.dot(g) / 2 - k.dot(g_par)
    g_om = rz(om) * g
    assert is_zero(g_om.dot(g_om) - g.dot(g))
    ewald = k.dot(g_om) + g_om.dot(g_om) / 2
    assert is_zero(ewald - (alpha * sp.cos(om) + beta * sp.sin(om) - delta))


def omega_solutions() -> tuple:
    al, be, de = sp.symbols("alpha beta delta", real=True)
    R, phi = sp.sqrt(al**2 + be**2), sp.atan2(al, be)
    return (al, be, de), (sp.asin(de / R) - phi, -sp.asin(de / R) - phi - sp.pi)


def test_omega_solutions():
    """dif-omega: both solutions satisfy alpha cos(omega) + beta sin(omega) = delta (when |delta / R| <= 1)."""
    (al, be, de), omegas = omega_solutions()
    for w in omegas:
        assert is_zero(sp.expand_trig(sp.expand(al * sp.cos(w) + be * sp.sin(w) - de)))


def test_clip_cost():
    """Anri's choice in dif-omega: arcsin(1) - arcsin(x) = arccos(x) (their sum has zero derivative and is pi/2 at
    x = 1), which at x = 1 - 1e-6 rounds to the 0.081 degrees diffraction.rst quotes."""
    x = sp.Symbol("x", real=True)
    assert is_zero((sp.asin(x) + sp.acos(x)).diff(x))
    assert sp.asin(1) + sp.acos(1) == sp.asin(1)
    value = sp.N(sp.deg(sp.acos(1 - sp.Rational(1, 10**6))), 30)
    assert sp.Float("0.0805", 30) <= value < sp.Float("0.0815", 30)


def test_branch_sign():
    """The branches (conventions.rst): d/d omega of alpha cos(omega) + beta sin(omega) is R cos(omega + phi), which is
    +R sqrt(1 - (delta/R)^2) at omega_1 and -R sqrt(1 - (delta/R)^2) at omega_2. That derivative is
    k . (z x g(omega)) = g_lab . (k x n): for a beam along x and an untilted axis, -g_y / lambda = sin(2 theta) sin(eta)
    / lambda^2. So omega_1 (etasign = +1) diffracts with eta > 0, omega_2 with eta < 0."""
    (al, be, de), (w1, w2) = omega_solutions()
    R = sp.sqrt(al**2 + be**2)
    om = sp.Symbol("omega", real=True)
    deriv = (al * sp.cos(om) + be * sp.sin(om)).diff(om)
    root = R * sp.sqrt(1 - de**2 / R**2)
    assert is_zero(sp.expand_trig(sp.expand(deriv.subs(om, w1) - root)))
    assert is_zero(sp.expand_trig(sp.expand(deriv.subs(om, w2) + root)))
    th, eta, lam = sp.symbols("theta eta lambda", positive=True)
    g_lab = (sp.Matrix([sp.cos(2 * th), -sp.sin(2 * th) * sp.sin(eta), sp.sin(2 * th) * sp.cos(eta)]) - X) / lam
    g_s, k = vec("g"), X / lam
    assert is_zero((k.dot(rz(om) * g_s)).diff(om) - (rz(om) * g_s).dot(k.cross(Z)))
    assert is_zero(g_lab.dot(k.cross(Z)) - sp.sin(2 * th) * sp.sin(eta) / lam**2)


def random_reflection(r: np.random.Generator, k_in: np.ndarray) -> np.ndarray:
    """A g in the turntable frame that meets the Ewald sphere of k_in during a turn."""
    while True:
        g = r.normal(size=3) * 0.4
        a = k_in @ (g - g[2] * np.array([0, 0, 1.0]))
        b = k_in @ np.cross([0, 0, 1.0], g)
        if abs((-g @ g / 2 - k_in[2] * g[2]) / np.hypot(a, b)) < 0.95:
            return g


def test_omega_code():
    """dif-omega: omega_solns for both branches (wrapped to (-180, 180]) against the proved solutions, beam tilted."""
    (al, be, de), omegas = omega_solutions()
    g, k = vec("g"), vec("k")
    g_par = g.dot(Z) * Z
    sub = {al: k.dot(g - g_par), be: k.dot(Z.cross(g - g_par)), de: -g.dot(g) / 2 - k.dot(g_par)}
    r = rng(10)
    for _ in range(4):
        kv = np.array([1.0, r.normal() * 0.01, r.normal() * 0.01]) / 0.3
        gv = random_reflection(r, kv)
        vals = dict(zip(g, gv))
        vals.update(dict(zip(k, kv)))
        # alpha, beta, delta sum terms up to |k| |g| + |g|^2; omega errs by those over R, times the condition
        # 1 / sqrt(1 - (delta / R)^2) of arcsin, in degrees up to 360 after the shifts
        x = [float(evaluate(sub[q], vals)) for q in (al, be, de)]
        R = np.hypot(x[0], x[1])
        t = tol(np.degrees((kv @ kv + gv @ gv) / R / np.sqrt(1 - (x[2] / R) ** 2)) + 360.0)
        for etasign, w in zip((1.0, -1.0), omegas):
            wv = float(evaluate(w.subs(sub), vals))
            expected = np.degrees(np.arctan2(np.sin(wv), np.cos(wv)))
            agree(expected, lambda *a: scatter.omega_solns(*a)[0], gv, etasign, kv, **t)


def test_branch_code():
    """The branches: etasign = +1 gives eta > 0 (beam along x, no tilts), through omega_solns and k_out_to_tth_eta."""
    r = rng(11)
    kv = np.array([1 / 0.3, 0, 0])
    with float64():
        for _ in range(6):
            g = random_reflection(r, kv)
            for etasign in (1.0, -1.0):
                om = scatter.omega_solns(g, etasign, kv)[0]
                k_out = np.asarray(gonio.sample_to_lab(g, om, 0.0, 0.0, 0.0, 0.0)) + kv
                eta = float(scatter.k_out_to_tth_eta(k_out, kv)[1])
                assert np.sign(eta) == etasign


def page_kout(th2: sp.Expr, eta: sp.Expr, frame: tuple) -> sp.Matrix:
    k, e_h, e_v = frame
    return sp.cos(th2) * k + sp.sin(th2) * (-sp.sin(eta) * e_h + sp.cos(eta) * e_v)


def test_kout():
    """dif-kout inverts frm-eta for any beam: k_out is a unit vector with k_out . k = cos(2 theta),
    -k_out . e_h = sin(2 theta) sin(eta) and k_out . e_v = sin(2 theta) cos(eta), so for 0 < 2 theta < 180 degrees
    arccos and atan2 give back 2 theta and eta."""
    th2, eta, psi, tilt = sp.symbols("2theta eta psi tilt", real=True)
    k, e_h, e_v = frame = beam_frame(psi, tilt)
    u = page_kout(th2, eta, frame)
    assert is_zero(u.dot(u) - 1)
    assert is_zero(u.dot(k) - sp.cos(th2))
    assert is_zero(-u.dot(e_h) - sp.sin(th2) * sp.sin(eta))
    assert is_zero(u.dot(e_v) - sp.sin(th2) * sp.cos(eta))


def test_kout_code():
    """dif-kout: tth_eta_to_k_out (scaled to 1 / lambda) for a tilted beam."""
    th2, eta, psi, tilt, lam = sp.symbols("2theta eta psi tilt lambda", real=True)
    frame = beam_frame(psi, tilt)
    k_out = page_kout(sp.rad(th2), sp.rad(eta), frame) / lam
    r = rng(12)
    for _ in range(3):
        vals = {th2: r.uniform(1, 170), eta: r.uniform(-180, 180), psi: r.normal() * 0.1, tilt: r.normal() * 0.1}
        vals[lam] = 0.25
        kin = 2.0 * ev(frame[0], vals)
        # |k_out| = 4; sines and cosines of radians(angle) <= pi
        agree(ev(k_out, vals), scatter.tth_eta_to_k_out, vals[th2], vals[eta], 0.25, kin, **tol(4 * np.pi))


def test_dty():
    """dif-dty: the point W C R(omega) v + (0, dty - y0, 0) is on the beam's horizontal line y = (k_y / k_x) x when
    dty = y0 + (k_y / k_x) x_lab - y_lab; for k along x and no tilts that is y0 - (x_s sin(omega) + y_s cos(omega))."""
    v = vec("v")
    om, w, ch, y0, dty, kx, ky = sp.symbols("omega w chi y0 dty k_x k_y", real=True)
    lab0 = ry(w) * rx(ch) * rz(om) * v
    sol = sp.solve(sp.Eq(lab0[1] + dty - y0, ky / kx * lab0[0]), dty)
    assert len(sol) == 1 and is_zero(sol[0] - (y0 + ky / kx * lab0[0] - lab0[1]))
    flat = sol[0].subs({w: 0, ch: 0, ky: 0})
    assert is_zero(flat - (y0 - (v[0] * sp.sin(om) + v[1] * sp.cos(om))))


def test_dty_code():
    v = vec("v")
    om, w, ch, y0, kx, ky = sp.symbols("omega w chi y0 k_x k_y", real=True)
    lab0 = ry(sp.rad(w)) * rx(sp.rad(ch)) * rz(sp.rad(om)) * v
    dty = y0 + ky / kx * lab0[0] - lab0[1]
    r = rng(13)
    for _ in range(3):
        vals = dict(zip(v, r.normal(size=3) * 50))
        vals.update(
            {om: r.uniform(-180, 180), w: r.normal(), ch: r.normal(), y0: r.normal(), kx: 3.0, ky: r.normal() * 0.03}
        )
        kin = np.array([3.0, vals[ky], 0.0])
        vs = np.array([vals[x] for x in v])
        args = (vals[om], vals[w], vals[ch], vals[y0])
        t = tol(np.abs(vs).max() * 3 + abs(vals[y0]))  # rotated components, each a sum of three terms <= |v|
        agree(evaluate(dty, vals), gonio.find_dty_for_beam_xy, vs, kin, *args, **t)


def test_column_height():
    """dif-column: the beam's centre line t k_hat passes horizontal distance d along the beam at t = d / cos(a), at
    height d tan(a), with cos(a) = |(k_x, k_y)| for a unit k_hat."""
    psi, a, d = sp.symbols("psi a d", real=True)
    k = sp.Matrix([sp.cos(a) * sp.cos(psi), sp.cos(a) * sp.sin(psi), sp.sin(a)])
    p = k * d / sp.cos(a)
    assert is_zero(p[0] ** 2 + p[1] ** 2 - d**2)
    assert is_zero((p[0] * sp.cos(psi) + p[1] * sp.sin(psi)) - d)
    assert is_zero(p[2] - d * sp.tan(a))


def test_column_code():
    """dif-column: origin_lab raises a column voxel to the beam's height over it; a cube stays at frm-stack."""
    v = vec("v")
    om, psi, a, dty, y0 = sp.symbols("omega psi a dty y0", real=True)
    lab = sp.Matrix([0, dty - y0, 0]) + rz(sp.rad(om)) * v
    d = lab[0] * sp.cos(psi) + lab[1] * sp.sin(psi)
    col = lab + Z * d * sp.tan(a)
    r = rng(14)
    for _ in range(3):
        vals = dict(zip(v, r.normal(size=3) * 50))
        vals.update({om: r.uniform(-180, 180), psi: r.normal() * 0.1, a: r.normal() * 0.05, dty: r.normal(), y0: 0.3})
        kin = 2.0 * np.array(
            [np.cos(vals[a]) * np.cos(vals[psi]), np.cos(vals[a]) * np.sin(vals[psi]), np.sin(vals[a])]
        )
        vs = np.array([vals[x] for x in v])
        args = (vals[om], 0.0, 0.0, vals[dty], vals[y0])
        t = tol(np.abs(vs).max() * 3 + abs(vals[dty]) + 1.0)  # as frm-stack, plus the rise, smaller than |v|
        agree(ev(col, vals), gonio.origin_lab, vs, kin, *args, True, **t)
        agree(ev(lab, vals), gonio.origin_lab, vs, kin, *args, False, **t)


def test_raytrace():
    """dif-raytrace: Cramer's rule for o + t k = d0 + s_c S + f_c F. The solution is unchanged by scaling k."""
    o, k, d0, S, F = vec("o"), vec("k"), vec("d"), vec("S"), vec("F")
    s_c, f_c, t = sp.symbols("s_c f_c t", real=True)
    sol = sp.solve(list(o + t * k - d0 - s_c * S - f_c * F), [s_c, f_c, t], dict=True)[0]
    n = k.dot(S.cross(F))
    assert is_zero(sol[s_c] - k.dot(F.cross(d0 - o)) / n)
    assert is_zero(sol[f_c] - k.dot((d0 - o).cross(S)) / n)


def test_raytrace_code():
    from anri.geom import detector

    o, k, d0, S, F = vec("o"), vec("k"), vec("d"), vec("S"), vec("F")
    n = k.dot(S.cross(F))
    sf = sp.Matrix([k.dot(F.cross(d0 - o)) / n, k.dot((d0 - o).cross(S)) / n])
    r = rng(15)
    for _ in range(3):
        kv = np.array([1.0, *(r.normal(size=2) * 0.2)])
        vals = {**dict(zip(o, r.normal(size=3) * 100)), **dict(zip(k, kv))}
        vals.update(dict(zip(d0, [150000.0, *(r.normal(size=2) * 1000)])))
        vals.update(dict(zip(S, [0.1, 0.3, 75.0])))
        vals.update(dict(zip(F, [0.2, -75.0, 0.4])))
        arrs = [np.array([vals[x] for x in m]) for m in (k, o, S, F, d0)]
        # numerators sum terms up to |k| |F| |d0 - o| (and |o| for the code's separate origin term); over |n|
        k_, o_, S_, F_, d_ = arrs
        t = tol(np.linalg.norm(k_) * 75.0 * (np.linalg.norm(d_) + np.linalg.norm(o_)) / abs(k_ @ np.cross(S_, F_)))
        agree(ev(sf, vals), lambda *a: np.array(detector.raytrace_to_det(*a)), *arrs, **t)


def motion() -> tuple:
    """The peak of g0 = k_out - k_in (beam along x, axis z, lab frame at the diffracting omega) after a rotation
    eps * d: the omega and eta derivatives in eps at 0, from the implicit function theorem on the Ewald condition."""
    th, eta, lam, om, eps = sp.symbols("theta eta lambda omega epsilon", real=True)
    d = vec("d")
    k_in = X / lam
    g0 = (sp.Matrix([sp.cos(2 * th), -sp.sin(2 * th) * sp.sin(eta), sp.sin(2 * th) * sp.cos(eta)]) - X) / lam
    g = rz(om) * (sp.eye(3) + eps * skew(d)) * g0  # first order in eps is exact for the derivative at 0
    ewald = k_in.dot(g) + g0.dot(g0) / 2  # |g| = |g0|: rotations keep it
    at0 = {om: 0, eps: 0}
    d_om = -ewald.diff(eps).subs(at0) / ewald.diff(om).subs(at0)
    k_out = g + k_in
    dk = (k_out.diff(om) * d_om + k_out.diff(eps)).subs(at0)
    k0 = k_out.subs(at0)
    d_eta = (-dk[1] * k0[2] + k0[1] * dk[2]) / (k0[1] ** 2 + k0[2] ** 2)  # d atan2(-k_y, k_z)
    return (th, eta, lam, d, g0), d_om, d_eta, dk


def test_rotation_motion():
    """dif-rotation-motion is the exact first derivative: d omega = -d_z - d_y cot(eta), d eta = d_x - d_y tan(theta)
    / sin(eta), and k_out . x (so 2 theta) does not change."""
    (th, eta, _, d, _), d_om, d_eta, dk = motion()
    assert is_zero(sp.trigsimp(d_om - (-d[2] - d[1] * sp.cot(eta))))
    assert is_zero(sp.trigsimp(d_eta - (d[0] - d[1] * sp.tan(th) / sp.sin(eta))))
    assert is_zero(dk[0])


def test_rotation_motion_consequences():
    """The consequences read off dif-rotation-motion's Jacobian J: an isotropic spread sigma widens a peak by
    sigma / |sin(eta)| in omega and sigma sqrt(1 + tan^2(theta) / sin^2(eta)) in eta (the diagonal of J J^T), and a
    rotation about g itself does not move it (J g0 = 0)."""
    (th, eta, _, _, g0), _, _, _ = motion()
    J = sp.Matrix([[0, -sp.cot(eta), -1], [1, -sp.tan(th) / sp.sin(eta), 0]])
    JJ = J * J.T
    assert is_zero(sp.trigsimp(JJ[0, 0] - 1 / sp.sin(eta) ** 2))
    assert is_zero(JJ[1, 1] - (1 + sp.tan(th) ** 2 / sp.sin(eta) ** 2))
    assert is_zero(sp.expand_trig(J * g0))


def test_rotation_motion_code():
    """dif-rotation-motion against JAX derivatives of omega_solns and k_out_to_tth_eta, for a lattice rotated about lab
    x, y and z at the diffracting omega (beam along x, no tilts)."""
    lam = 0.3
    kv = np.array([1 / lam, 0, 0])
    r = rng(16)

    def peak(eps: jax.Array, g: np.ndarray, om0: jax.Array, axis: np.ndarray, etasign: float) -> jax.Array:
        Rl = rmat_from_axis_angle(axis, jax.numpy.degrees(eps))
        g_new = gonio.lab_to_sample(Rl @ gonio.sample_to_lab(g, om0, 0.0, 0.0, 0.0, 0.0), om0, 0.0, 0.0, 0.0, 0.0)
        om = scatter.omega_solns(g_new, etasign, kv)[0]
        k_out = gonio.sample_to_lab(g_new, om, 0.0, 0.0, 0.0, 0.0) + kv
        tth, eta = scatter.k_out_to_tth_eta(k_out, kv)
        return jax.numpy.radians(jax.numpy.stack([om, eta, tth]))

    with float64():
        for _ in range(3):
            g = random_reflection(r, kv)
            for etasign in (1.0, -1.0):
                om0 = scatter.omega_solns(g, etasign, kv)[0]
                tth, eta = scatter.k_out_to_tth_eta(
                    np.asarray(gonio.sample_to_lab(g, om0, 0.0, 0.0, 0.0, 0.0)) + kv, kv
                )
                th, et = np.radians(float(tth)) / 2, np.radians(float(eta))
                expected = np.array([[0, -1 / np.tan(et), -1], [1, -np.tan(th) / np.sin(et), 0], [0, 0, 0]])
                got = np.stack([jax.jacfwd(peak)(0.0, g, om0, ax, etasign) for ax in np.eye(3)], 1)
                # forward-mode derivatives of the same chain: errors of the size of the largest derivative, which
                # grows as 1 / |sin(eta)|, times the 1 / |sin(eta)| amplification of omega's solution
                np.testing.assert_allclose(got, expected, **tol(np.abs(expected).max() / abs(np.sin(et))))
