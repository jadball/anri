"""Proofs for docs/source/theory/calibration.rst: what the rings and the scan's peaks can determine (tilt_x, distance
against wavelength, wavelength against lattice), the Schur-complement step, and the strain left."""

import jax.numpy as jnp
import numpy as np
import sympy as sp

from tests.theory.common import (
    EPS,
    agree,
    evaluate,
    is_zero,
    rng,
    tol,
)


def _rx(t: sp.Expr) -> sp.Matrix:
    return sp.Matrix([[1, 0, 0], [0, sp.cos(t), -sp.sin(t)], [0, sp.sin(t), sp.cos(t)]])


def _ry(t: sp.Expr) -> sp.Matrix:
    return sp.Matrix([[sp.cos(t), 0, sp.sin(t)], [0, 1, 0], [-sp.sin(t), 0, sp.cos(t)]])


def _rz(t: sp.Expr) -> sp.Matrix:
    return sp.Matrix([[sp.cos(t), -sp.sin(t), 0], [sp.sin(t), sp.cos(t), 0], [0, 0, 1]])


def test_tilt_x_moves_no_ring():
    """frm-det applies R_x last, about the beam through (L, 0, 0): a pixel's v_x and v_y^2 + v_z^2, so its 2theta
    (tan^2 2theta = (v_y^2 + v_z^2) / v_x^2 from the lab origin), do not depend on tilt_x. Rings cannot fit it."""
    L, y, z, tx, ty, tz = sp.symbols("L y z t_x t_y t_z", real=True)
    v = sp.Matrix([L, 0, 0]) + _rx(tx) * _ry(ty) * _rz(tz) * sp.Matrix([0, y, z])
    assert is_zero(sp.diff(v[0], tx))
    assert is_zero(sp.diff(v[1] ** 2 + v[2] ** 2, tx))


def test_tilt_x_code():
    """anri's pixel 2theta is the same at any tilt_x (to the roundings of the tilted pixel's position)."""
    from anri.calib.fit import base_pars
    from anri.calib.rings import pixel_tth_eta
    from anri.maths.utils import float64

    pars = {"y_center": 1049.9, "y_size": 75.0, "tilt_y": 0.01, "z_center": 1116.5, "z_size": 75.0, "tilt_z": -0.02,
            "tilt_x": 0.0, "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1,
            "wavelength": 0.28, "wedge": 0.0, "chi": 0.0}  # fmt: skip
    sc, fc = rng(7).uniform(0, 2000, (2, 5))
    p0 = base_pars(pars, 0.0)
    p1 = {**p0, "tilt_x": 0.3}
    with float64():
        tth0 = np.asarray(pixel_tth_eta(sc, fc, p0)[0])
    # 2theta = atan2(|v_yz|, v_x) of a position rotated by tilt_x: ~10 roundings in each coordinate, and atan2 of
    # condition ~1 (2theta well away from 0): 32 ulps
    agree(np.asarray(tth0, float), lambda *z: pixel_tth_eta(*z)[0], sc, fc, p1, rtol=32 * EPS, atol=0)


def test_rings_tie_distance_to_wavelength():
    """For an untilted detector, rho = atan(R / L) - 2 asin(lambda d / 2). At the match (R = L tan 2th,
    lambda d = 2 sin th) the columns L drho/dL and lambda drho/dlambda are -sin 2th cos 2th and -2 tan th: their
    ratio is cos^2 th cos 2th = 1 - 3 th^2 + O(th^4). Rings fix L lambda; they separate L from lambda only through
    th^2."""
    R, L, lam, d, th = sp.symbols("R L lambda d theta", positive=True)
    rho = sp.atan(R / L) - 2 * sp.asin(lam * d / 2)
    match = {R: L * sp.tan(2 * th), d: 2 * sp.sin(th) / lam}
    cl = (L * sp.diff(rho, L)).subs(match)
    cw = (lam * sp.diff(rho, lam)).subs(match)
    assert is_zero(cl + sp.sin(2 * th) * sp.cos(2 * th))
    assert is_zero(cw**2 - 4 * sp.tan(th) ** 2)  # and cw < 0: -lambda d / sqrt(1 - (lambda d / 2)^2)
    ratio = sp.cos(th) ** 2 * sp.cos(2 * th)
    assert is_zero(-sp.sin(2 * th) * sp.cos(2 * th) - ratio * (-2 * sp.tan(th)))
    assert is_zero(sp.series(ratio, th, 0, 4).removeO() - (1 - 3 * th**2))


def test_wavelength_lattice_gauge():
    """With k_in = k / lambda (|k| = 1) and g = G / s (lattice scale s), the Ewald condition |k_in + g|^2 = |k_in|^2
    times lambda^2 is 2 t k.G + t^2 |G|^2 with t = lambda / s, and k_out's direction is k + t G: both are killed by
    lambda d/dlambda + s d/ds. Only lambda / s is determined."""
    lam, s = sp.symbols("lambda s", positive=True)
    k = sp.Matrix(sp.symbols("k0:3", real=True))
    G = sp.Matrix(sp.symbols("G0:3", real=True))
    kin, g = k / lam, G / s
    ewald = lam**2 * ((kin + g).dot(kin + g) - kin.dot(kin))
    t = lam / s
    assert is_zero(ewald - (2 * t * k.dot(G) + t**2 * G.dot(G)))
    direction = lam * (kin + g)
    for e in [ewald, *direction]:
        assert is_zero(lam * sp.diff(e, lam) + s * sp.diff(e, s))


# the normal equations of 2 free parameters x and 2 orientations of one parameter each (no coupling between them)
_h = sp.symbols("h0:9", real=True)
_b = sp.Matrix(sp.symbols("b0:4", real=True))
Hxx = sp.Matrix([[_h[0], _h[1]], [_h[1], _h[2]]])
Hxo = sp.Matrix([[_h[3], _h[4]], [_h[5], _h[6]]])  # column o: x's coupling to orientation o
Hoo = sp.diag(_h[7], _h[8])


def _schur(Hxx: sp.Matrix, Hxo: sp.Matrix, Hoo: sp.Matrix, bx: sp.Matrix, bo: sp.Matrix) -> tuple:
    Hinv = Hoo.inv()
    S = Hxx - Hxo * Hinv * Hxo.T
    dx = S.inv() * (-bx + Hxo * Hinv * bo)
    da = -Hinv * (bo + Hxo.T * dx)
    return dx, da


def test_schur_step():
    """cal-schur solves the whole system H (dx, da) = -b: its x rows and its orientation rows both hold."""
    bx, bo = _b[:2, :], _b[2:, :]
    dx, da = _schur(Hxx, Hxo, Hoo, bx, bo)
    assert is_zero(Hxx * dx + Hxo * da + bx)
    assert is_zero(Hxo.T * dx + Hoo * da + bo)


def test_schur_code():
    """anri's _solve is cal-schur, with Levenberg-Marquardt's mu diag(H) on both blocks and 16 eps tr(H_oo) (plus the
    smallest normal number) added to each orientation's diagonal."""
    from anri.calib.fit import _solve

    g = rng(8)
    J = g.integers(-5, 6, (12, 4)).astype(float)  # rational, so the exact step is exact
    J[:6, 3] = J[6:, 2] = 0  # orientation 0 sees peaks 0-5 only, orientation 1 peaks 6-11
    r = g.integers(-5, 6, 12).astype(float)
    H, bv = J.T @ J, J.T @ r
    for mu in (0.0, 0.5):
        hxx, hxo, hoo = (sp.Matrix(x) for x in (H[:2, :2], H[:2, 2:], np.diag(np.diag(H)[2:])))
        m = sp.Rational(mu)
        hxx = hxx + m * sp.diag(*hxx.diagonal())
        ridge = 16 * sp.Float(EPS, 30) * hoo + sp.Float(float(np.finfo(float).tiny), 30) * sp.eye(2)  # hoo diagonal
        hoo = hoo * (1 + m) + ridge
        dx, da = _schur(hxx, hxo, hoo, sp.Matrix(bv[:2]), sp.Matrix(bv[2:]))
        Hc = (H[:2, :2], bv[:2], np.stack([H[:2, 2:3], H[:2, 3:4]]), np.diag(H)[2:, None, None], bv[2:, None])
        # a 2 x 2 solve after eliminating 1 x 1 blocks: a few roundings times the condition of H (computed here)
        rtol = 64 * EPS * np.linalg.cond(H)
        agree(np.array(evaluate(dx, {}), float).ravel(), lambda *z: _solve(*z)[0], Hc, mu, rtol=rtol, atol=0)
        agree(np.array(evaluate(da, {}), float).ravel(), lambda *z: _solve(*z)[1].ravel(), Hc, mu, rtol=rtol,
              atol=0)  # fmt: skip


def _rod(r: sp.Matrix) -> sp.Matrix:
    K = sp.Matrix([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    return sp.eye(3) + 2 / (1 + r.dot(r)) * (K + K * K)


def test_strain_from_ubi():
    """cal-strain: UBI = B^-1 (R U)^T (I + E) gives K = B UBI = (R U)^T (I + E) and K^T K = (I + E)^2 for any
    rotation (here a Rodrigues one, V^T V = I), so I + E is the positive square root of K^T K."""
    r = sp.Matrix(sp.symbols("r0:3", real=True))
    e = sp.symbols("e0:6", real=True)
    E = sp.Matrix([[e[0], e[5], e[4]], [e[5], e[1], e[3]], [e[4], e[3], e[2]]])
    V = _rod(r)
    assert is_zero(V.T * V - sp.eye(3))
    K = V.T * (sp.eye(3) + E)
    assert is_zero(K.T * K - (sp.eye(3) + E) ** 2)


def test_strain_code():
    """anri's lattice_strain recovers E and U from UBI = B^-1 U^T (I + E), for a cubic and a monoclinic B, strained and
    not."""
    from anri.calib.fit import lattice_strain
    from tests.theory.common import metric, page_B

    g = rng(9)
    r = sp.Matrix([sp.Rational(int(x), 100) for x in g.integers(-30, 30, 3)])
    U = _rod(r)
    E = sp.Matrix(3, 3, lambda i, j: sp.Rational(int(g.integers(-9, 9)), 10**4))
    E = (E + E.T) / 2
    a, b, c, be = sp.Integer(4), sp.Integer(5), sp.Integer(6), sp.rad(95)
    Bm = sp.N(page_B(metric(a, b, c, 0, sp.cos(be), 0), 0), 40)
    for B in (sp.eye(3) / 4, Bm):  # cubic of 4 A (exact), and monoclinic
        Bv = np.array(evaluate(B, {}), float)
        for Ev in (E, sp.zeros(3, 3)):
            ubi = B.inv() * U.T * (sp.eye(3) + Ev)
            ubi_f = np.array(evaluate(ubi, {}), float)[None]
            # ubi and B round once (relative 1 ulp each, of cond(B) in K = B ubi), K^T K and its eigh (backward
            # stable, ~10 ulps of |K|^2 = 1) and the square root: entries of E within 16 ulps of cond(B), absolutely
            t = {"rtol": 0, "atol": 16 * EPS * np.linalg.cond(Bv)}
            agree(np.array(evaluate(Ev, {}), float)[None], lambda *z: lattice_strain(*z)[0], ubi_f, Bv, **t)
            agree(np.array(evaluate(U, {}), float)[None], lambda *z: lattice_strain(*z)[1], ubi_f, Bv, **t)


def test_split_strain():
    """cal-strain's split: E_dev is traceless and E_h I + E_dev = E, for any E."""
    E = sp.Matrix(3, 3, sp.symbols("e0:9", real=True))
    h = E.trace() / 3
    assert is_zero((E - h * sp.eye(3)).trace())


def test_split_strain_code():
    """split_strain against cal-strain's split, and the peak-weighted means over the fitted orientations."""
    from anri.calib.fit import split_strain

    g = rng(25)
    Es = []
    for _ in range(2):
        E = sp.Matrix(3, 3, lambda i, j: sp.Rational(int(g.integers(-99, 99)), 10**5))
        Es.append((E + E.T) / 2)
    n = [10, 30]
    hyd = [E.trace() / 3 for E in Es]
    dev = [E - h * sp.eye(3) for E, h in zip(Es, hyd)]
    Ev = np.array([evaluate(E, {}) for E in Es])
    got = split_strain(Ev, np.array(n))
    # a trace, a third and a subtraction on entries <= 1e-3: a few ulps of them
    t = tol(1e-3)
    np.testing.assert_allclose(got["hydrostatic"], [float(h) for h in hyd], rtol=0, atol=t["atol"])
    np.testing.assert_allclose(got["deviatoric"], [evaluate(d, {}) for d in dev], rtol=0, atol=t["atol"])
    mean_h = sum(ni * h for ni, h in zip(n, hyd)) / sum(n)
    mean_d = sum((ni * d for ni, d in zip(n, dev)), sp.zeros(3, 3)) / sum(n)
    np.testing.assert_allclose(got["mean_hydrostatic"], float(mean_h), rtol=0, atol=t["atol"])
    np.testing.assert_allclose(got["mean_deviatoric"], evaluate(mean_d, {}), rtol=0, atol=t["atol"])


def test_ring_ds_code():
    """ring_ds: bcc iron's rings up to 2 theta max are the distinct d* = |h| / a, h + k + l even, with
    2 asin(lambda d* / 2) <= 2 theta max (dif-bragg); exactly the ones the rule keeps."""
    from anri.calib.rings import ring_ds

    a, lam, tmax = sp.Rational(28665, 10000), sp.Rational(3, 10), 20
    dsmax = 2 * sp.sin(sp.rad(tmax) / 2) / lam
    n2 = sorted({h * h + k * k + l * l for h in range(-9, 10) for k in range(-9, 10) for l in range(-9, 10)
                 if (h + k + l) % 2 == 0 and 0 < h * h + k * k + l * l and sp.sqrt(h * h + k * k + l * l) / a <= dsmax})  # fmt: skip
    want = np.array([float(sp.N(sp.sqrt(m) / a, 30)) for m in n2])
    got = ring_ds(np.array([float(a)] * 3 + [90.0] * 3), 229, float(lam), float(tmax))
    # rings 1e-4 apart are merged; bcc's are further apart. d* = sqrt(h^T G* h): a few ulps of it (test_reflections_code)
    assert len(got) == len(want)
    np.testing.assert_allclose(got, want, **tol(float(want.max())))


def test_mad_factor():
    """1.4826 is 1 / Phi^-1(3/4) rounded: the MAD of a normal distribution over its standard deviation."""
    q = sp.sqrt(2) * sp.erfinv(sp.Rational(1, 2))
    assert is_zero(sp.erf(q / sp.sqrt(2)) - sp.Rational(1, 2))  # Phi(q) = 3/4
    assert round(float(sp.N(1 / q, 30)), 4) == 1.4826


def test_pixel_angle_quoted():
    """One 75 um Eiger pixel at 150 mm subtends 0.03 deg (rounded)."""
    assert round(float(sp.N(sp.deg(sp.atan(sp.Rational(75, 150000))), 30)), 2) == 0.03


def test_ring_residual_code():
    """cal-ring-residual: assign puts each pixel at its nearest ring if within t, unless that ring is closer than 2t
    to a neighbour, and in sector floor((eta + 180) n / 360); ring_offsets is each (ring, sector)'s weighted mean 2theta
    less the ring's 2 asin(lambda d* / 2). The pixels' angles from pixel_tth_eta (test_tilt_x_code); the rest exactly."""
    from fractions import Fraction as Fr

    from anri.calib.fit import base_pars
    from anri.calib.rings import assign, pixel_tth_eta, ring_offsets
    from anri.maths.utils import float64

    pars = {"y_center": 1024.0, "y_size": 75.0, "tilt_y": 1e-3, "z_center": 1030.0, "z_size": 75.0, "tilt_z": -2e-3,
            "tilt_x": 0.0, "distance": 150e3, "o11": 1, "o12": 0, "o21": 0, "o22": 1, "wavelength": 0.3}  # fmt: skip
    g = rng(41)
    px = {"sc": g.uniform(0, 2048, 4000), "fc": g.uniform(0, 2048, 4000), "intensity": g.integers(1, 9, 4000) * 1.0}
    ds = np.array([0.30, 0.45, 0.452, 0.60])  # the middle two closer than 2 t: left out
    tol, n_eta = 0.1, 8
    with float64():
        p = {k: jnp.float64(v) for k, v in base_pars(pars, 0.0).items()}
        tth, eta = (np.asarray(a) for a in pixel_tth_eta(jnp.asarray(px["sc"]), jnp.asarray(px["fc"]), p))
    t = [float(sp.N(sp.deg(2 * sp.asin(sp.Float(pars["wavelength"], 30) * sp.Float(d, 30) / 2)), 30)) for d in ds]
    used = [all(abs(t[r] - t[q]) > 2 * tol for q in range(len(t)) if q != r) for r in range(len(t))]
    ring = np.full(len(tth), -1)
    for i, x in enumerate(tth):
        r = int(np.argmin([abs(x - y) for y in t]))
        if abs(x - t[r]) < tol and used[r]:
            ring[i] = r
    sector = np.floor((eta + 180.0) * n_eta / 360.0).astype(int)
    a = assign(px, ds, pars, tol, n_eta)
    np.testing.assert_array_equal(a["ring"], ring)
    np.testing.assert_array_equal(a["group"][ring >= 0], (ring * n_eta + sector)[ring >= 0])
    np.testing.assert_array_equal(a["used"], used)
    off = ring_offsets(px, ds, pars, tol, n_eta)
    for r in range(len(t)):
        for j in range(n_eta):
            m = (ring == r) & (sector == j)
            if not m.any():
                assert np.isnan(off["offset"][r, j])
                continue
            w = [Fr(float(v)) for v in px["intensity"][m]]
            mean = sum(wi * Fr(float(x)) for wi, x in zip(w, tth[m])) / sum(w)
            # sums of a few hundred products of 2theta (<= 30 deg) and integers, then a quotient and a subtraction:
            # that many ulps of 30 deg
            assert abs(off["offset"][r, j] - (float(mean) - t[r])) <= 16 * EPS * 30 * m.sum(), (r, j)

