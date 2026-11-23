"""Proofs of docs/source/theory/frames.rst: rotations, lattice metrics and B, the UBI, the goniometer stack, the beam
frame, eta and the detector."""

import numpy as np
import sympy as sp

from anri.crystal import lattice
from anri.geom import beam, detector, gonio
from anri.maths import rotation
from anri.maths.utils import float64
from tests.theory.common import (
    agree,
    beam_frame,
    cayley,
    ev,
    evaluate,
    is_zero,
    metric,
    page_B,
    rng,
    rx,
    ry,
    rz,
    skew,
    tol,
    vec,
)


def test_rotations_right_handed():
    """rot_x, rot_y, rot_z are proper rotations, and right-handed: d R / dt at t = 0 is [e]_x, so R v turns v."""
    t = sp.Symbol("t", real=True)
    for R, e in ((rx(t), (1, 0, 0)), (ry(t), (0, 1, 0)), (rz(t), (0, 0, 1))):
        assert is_zero(R.T * R - sp.eye(3))
        assert sp.simplify(R.det()) == 1
        assert is_zero(R.diff(t).subs(t, 0) - skew(sp.Matrix(e)))


def test_rotations_code():
    for fn, R in ((rotation.rot_x, rx), (rotation.rot_y, ry), (rotation.rot_z, rz)):
        for deg in rng(1).uniform(-180, 180, 3):
            agree(evaluate(R(sp.rad(sp.Float(deg))), {}), fn, deg, **tol(np.pi))  # entries <= 1; radians(deg) <= pi


def test_reciprocal_metric():
    """frm-hkl: with the real vectors as the rows of A, the reciprocal ones are the rows of A^-T, a_i . a*_j = delta_ij,
    and G* = [a*_i . a*_j] = G^-1."""
    A = sp.Matrix(3, 3, sp.symbols("A0:9", real=True))
    Astar = A.inv().T
    assert is_zero(A * Astar.T - sp.eye(3))
    assert is_zero(Astar * Astar.T - (A * A.T).inv())


def test_metric_code():
    """frm-hkl: lpars_to_mt, mt_to_rmt and mt_to_lpars against G, G^-1 and the parameters read back from G."""
    a, b, c, al, be, ga = sp.symbols("a b c alpha beta gamma", positive=True)
    G = metric(a, b, c, sp.cos(sp.rad(al)), sp.cos(sp.rad(be)), sp.cos(sp.rad(ga)))
    back = [sp.sqrt(G[0, 0]), sp.sqrt(G[1, 1]), sp.sqrt(G[2, 2])]
    back += [sp.deg(sp.acos(G[1, 2] / (back[1] * back[2]))), sp.deg(sp.acos(G[0, 2] / (back[0] * back[2])))]
    back += [sp.deg(sp.acos(G[0, 1] / (back[0] * back[1])))]
    for lp in ([4.1, 5.3, 6.7, 81.0, 97.0, 112.0], [3.0, 3.0, 5.0, 90.0, 90.0, 120.0]):
        vals = dict(zip((a, b, c, al, be, ga), lp))
        Gv = evaluate(G, vals)
        Gi = evaluate(G.inv(), vals)
        agree(Gv, lattice.lpars_to_mt, np.array(lp), **tol(np.abs(Gv).max()))
        # an inverse errs by its condition number times its size
        agree(Gi, lattice.mt_to_rmt, Gv, **tol(np.linalg.cond(Gv) * np.abs(Gi).max()))
        # arccos has condition 1 / sin(angle) <= 1 / sin(81 degrees) here, in degrees up to 180
        agree(ev(sp.Matrix(back), vals), lattice.mt_to_lpars, Gv, **tol(2 * 180.0))


def test_B_matrix():
    """frm-B: B^T B = G*, for any lattice. Entries (0, 0), (0, 1), (0, 2) and (1, 1) hold by construction (dot products
    in a frame with x along a*, y in the a* b* plane). The other two need two identities in G* = G^-1:

    - (2, 2): (G*_22 - G*_02^2 / G*_00) sin^2(alpha) = 1 / c^2, i.e. c* sin(beta*) sin(alpha) = 1 / c (FABLE's form);
    - (1, 2): Q = G*_12 - G*_01 G*_02 / G*_00 equals B_11 B_12. Their squares are equal, and both have the sign of
      -cos(alpha), since G*_00 G*_12 - G*_01 G*_02 = -G_12 / det G (a minor of the inverse) and det G > 0.
    """
    a, b, c = sp.symbols("a b c", positive=True)
    ca, cb, cg = sp.symbols("c_alpha c_beta c_gamma", real=True)
    G = metric(a, b, c, ca, cb, cg)
    Gs = G.inv()
    assert is_zero(sp.together((Gs[2, 2] - Gs[0, 2] ** 2 / Gs[0, 0]) * (1 - ca**2) - 1 / G[2, 2]))
    Q = Gs[1, 2] - Gs[0, 1] * Gs[0, 2] / Gs[0, 0]
    assert is_zero(
        sp.together(Q**2 - (Gs[1, 1] - Gs[0, 1] ** 2 / Gs[0, 0]) * (Gs[2, 2] - Gs[0, 2] ** 2 / Gs[0, 0]) * ca**2)
    )
    assert is_zero(sp.together((Gs[0, 0] * Gs[1, 2] - Gs[0, 1] * Gs[0, 2]) * G.det() + G[1, 2]))


def test_B_code():
    """frm-B: lpars_to_B and B_matrix against the page's B (exact sympy, reciprocal parameters from G^-1)."""
    from anri.crystal.structure import B_matrix

    for lp in (
        [4.1, 5.3, 6.7, 81.0, 97.0, 112.0],
        [3.0, 3.0, 5.0, 90.0, 90.0, 120.0],
        [5.0, 6.0, 7.0, 90.0, 104.0, 90.0],
    ):
        a, b, c, al, be, ga = [sp.Rational(str(x)) for x in lp]
        Bv = evaluate(
            page_B(metric(a, b, c, sp.cos(sp.rad(al)), sp.cos(sp.rad(be)), sp.cos(sp.rad(ga))), sp.cos(sp.rad(al))), {}
        )
        # B comes from G^-1, so it errs by cond(G) times its size
        t = tol(
            np.linalg.cond(evaluate(metric(a, b, c, sp.cos(sp.rad(al)), sp.cos(sp.rad(be)), sp.cos(sp.rad(ga))), {}))
        )
        t["atol"] *= np.abs(Bv).max()
        agree(Bv, lattice.lpars_to_B, np.array(lp), **t)
        agree(Bv, B_matrix, lp, **t)


def test_ubi():
    """frm-g: UBI = (U B)^-1 = B^-1 U^T takes g back to h (so its rows are a, b, c), and UBI UBI^T = (B^T B)^-1 = G,
    whatever the rotation U."""
    U = cayley(vec("k"))
    B = sp.Matrix([[sp.Symbol(f"B{i}{j}", real=True) if j >= i else 0 for j in range(3)] for i in range(3)])
    UBI = B.inv() * U.T
    assert is_zero(UBI * (U * B) - sp.eye(3))
    assert is_zero(UBI * UBI.T - (B.T * B).inv())


def test_ubi_code():
    k = vec("k")
    U = cayley(k)
    B = sp.Matrix([[0.25, 0.03, -0.02], [0, 0.21, 0.05], [0, 0, 0.18]])
    vals = dict(zip(k, rng(2).normal(size=3)))
    Uv = evaluate(U, vals)
    UBI = evaluate((U * B).inv(), vals)
    Bv = np.array(B, float)
    # an inverse errs by its condition number (that of B, as U is orthogonal) times its size
    agree(UBI, lattice.U_B_to_UBI, Uv, Bv, **tol(np.linalg.cond(Bv) * np.abs(UBI).max()))
    agree(evaluate((U * B).inv() * (U * B).inv().T, vals), lattice.UBI_to_mt, UBI, **tol(np.abs(UBI).max() ** 2))


def stack(v: sp.Matrix, om: sp.Expr, wedge: sp.Expr, chi: sp.Expr, dty: sp.Expr, y0: sp.Expr) -> sp.Matrix:
    """frm-stack, angles in radians."""
    return sp.Matrix([0, dty - y0, 0]) + ry(wedge) * rx(chi) * rz(om) * v


def test_stack_inverse():
    """frm-stack: lab_to_sample's R^T C^T W^T (v_lab - (0, dty - y0, 0)) undoes it."""
    v = vec("v")
    om, w, ch, dty, y0 = sp.symbols("omega w chi dty y0", real=True)
    lab = stack(v, om, w, ch, dty, y0)
    back = rz(om).T * rx(ch).T * ry(w).T * (lab - sp.Matrix([0, dty - y0, 0]))
    assert is_zero(back - v)


def test_stack_code():
    v = vec("v")
    om, w, ch, dty, y0 = sp.symbols("omega w chi dty y0", real=True)
    lab = stack(v, sp.rad(om), sp.rad(w), sp.rad(ch), dty, y0)
    r = rng(3)
    for _ in range(3):
        vals = dict(zip(v, r.normal(size=3) * 100))
        vals.update(
            {om: r.uniform(-180, 180), w: r.uniform(-2, 2), ch: r.uniform(-2, 2), dty: r.normal(), y0: r.normal()}
        )
        args = (vals[om], vals[w], vals[ch], vals[dty], vals[y0])
        vs = np.array([vals[s] for s in v])
        lv = ev(lab, vals)
        t = tol(np.abs(lv).max() + np.abs(vs).max())  # rotations keep lengths: terms no larger than |v| + |dty - y0|
        agree(lv, gonio.sample_to_lab, vs, *args, **t)
        agree(vs, gonio.lab_to_sample, lv, *args, **t)


def test_beam_frame():
    """The beam frame: k, e_h, e_v are orthonormal and right-handed (e_h x e_v = k), e_h is horizontal, and for a
    beam along x they are y and z."""
    psi, tilt = sp.symbols("psi tilt", real=True)
    k, e_h, e_v = beam_frame(psi, tilt)
    M = sp.Matrix.hstack(k, e_h, e_v)
    assert is_zero(M.T * M - sp.eye(3))
    assert is_zero(e_h.cross(e_v) - k)
    assert e_h[2] == 0
    assert is_zero(sp.Matrix.hstack(*beam_frame(0, 0)) - sp.eye(3))


def test_beam_frame_code():
    psi, tilt = sp.symbols("psi tilt", real=True)
    k, e_h, e_v = beam_frame(psi, tilt)
    for p, t in rng(4).uniform(-0.5, 0.5, (3, 2)):
        vals = {psi: p, tilt: t}
        kv = ev(k, vals)
        for i, e in enumerate((k, e_h, e_v)):
            agree(ev(e, vals), lambda x, i=i: beam.beam_basis(x)[i], 3.7 * kv, **tol(1.0))  # unit vectors


def test_eta_code():
    """frm-eta: k_out_to_tth_eta against arccos(k_out . k) and atan2(-k_out . e_h, k_out . e_v), for a tilted beam."""
    from anri.diffract import scatter

    psi, tilt = sp.symbols("psi tilt", real=True)
    kout = vec("o")
    k, e_h, e_v = beam_frame(psi, tilt)
    u = kout / sp.sqrt(kout.dot(kout))
    angles = sp.Matrix([sp.deg(sp.acos(u.dot(k))), sp.deg(sp.atan2(-u.dot(e_h), u.dot(e_v)))])
    r = rng(5)
    for _ in range(4):
        vals = {psi: r.uniform(-0.3, 0.3), tilt: r.uniform(-0.3, 0.3)}
        vals.update(dict(zip(kout, r.normal(size=3))))
        o = np.array([vals[s] for s in kout])
        expected = ev(angles, vals)
        # degrees up to 180; arccos has condition 1 / sin(2 theta), atan2 of a unit vector's components has 1
        t = tol(180.0 / abs(np.sin(np.radians(expected[0]))))
        agree(expected, lambda x, kk: np.array(scatter.k_out_to_tth_eta(x, kk)), o, ev(k, vals), **t)


def test_omatrix_code():
    """frm-omatrix: detector_orientation_matrix puts o in the top-left corner and maps (s, f, 0) to (z_det, y_det, 0)."""
    o = rng(6).choice([-1.0, 0.0, 1.0], 4)
    expected = np.array([[o[0], o[1], 0], [o[2], o[3], 0], [0, 0, 1.0]])
    agree(expected, detector.detector_orientation_matrix, *o, **tol(0.0))  # exact copies


def det_symbols() -> tuple:
    return sp.symbols("s f z_c y_c p_z p_y L t_x t_y t_z o11 o12 o21 o22", real=True)


def page_det(s, f, zc, yc, pz, py, L, tx, ty, tz, o11, o12, o21, o22) -> sp.Matrix:
    """frm-det, tilts in radians."""
    zy = sp.Matrix([[o11, o12], [o21, o22]]) * sp.Matrix([pz * (s - zc), py * (f - yc)])
    return sp.Matrix([L, 0, 0]) + rx(tx) * ry(ty) * rz(tz) * sp.Matrix([0, zy[1], zy[0]])


def test_detector_factors():
    """frm-det is detector_transforms' product R_det P O S applied to (s - z_c, f - y_c, 0), plus (L, 0, 0), with P the
    permutation taking (z', y', 0) to (0, y', z'). Its pixel steps are its derivatives in s and f (constant: the
    detector is a plane), which detector_basis_vectors_lab returns."""
    s, f, zc, yc, pz, py, L, tx, ty, tz, o11, o12, o21, o22 = sym = det_symbols()
    O3 = sp.Matrix([[o11, o12, 0], [o21, o22, 0], [0, 0, 1]])
    P = sp.Matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    S = sp.diag(pz, py, 1)
    code = rx(tx) * ry(ty) * rz(tz) * P * O3 * S * sp.Matrix([s - zc, f - yc, 0]) + sp.Matrix([L, 0, 0])
    page = page_det(*sym)
    assert is_zero(code - page)
    trans = rx(tx) * ry(ty) * rz(tz) * P * O3 * S
    assert is_zero(page.diff(s) - trans[:, 0])
    assert is_zero(page.diff(f) - trans[:, 1])


def test_detector_code():
    sym = det_symbols()
    s, f, zc, yc, pz, py, L, tx, ty, tz, o11, o12, o21, o22 = sym
    page = page_det(*sym)
    r = rng(7)
    for o in ([1, 0, 0, 1], [-1, 0, 0, -1], [0, 1, -1, 0]):
        vals = dict(zip(sym, [r.uniform(0, 2000), r.uniform(0, 2000), 1024.3, 1011.7, 75.0, 74.0, 150000.0]))
        vals.update(dict(zip((tx, ty, tz), r.normal(size=3) * 0.01)))
        vals.update(dict(zip((o11, o12, o21, o22), o)))
        pars = [vals[x] for x in (yc, py, ty, zc, pz, tz, tx, L, o11, o12, o21, o22)]
        with float64():
            tr = detector.detector_transforms(*pars)
        lab = ev(page, vals)
        t = tol(np.abs(lab).max())  # terms no larger than L
        agree(lab, detector.det_to_lab, vals[s], vals[f], *tr, **t)
        # back in pixels: the inverse of R P O S has condition p_z / p_y and scales lab lengths by 1 / p_y
        agree(np.array([vals[s], vals[f]]), detector.lab_to_det, *lab, *tr, **tol(np.abs(lab).max() / 74.0 * 75 / 74))
        steps = [ev(page.diff(x), vals) for x in (s, f)] + [ev(page.subs({s: 0, f: 0}), vals)]
        for i in range(3):
            agree(steps[i], lambda *a, i=i: detector.detector_basis_vectors_lab(*a)[i], *tr, **t)
