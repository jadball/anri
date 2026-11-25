"""Proofs for docs/source/theory/search.rst: misorientations on the cubic Rodrigues grid, its fundamental zone, and
the matching tolerances."""

import jax.numpy as jnp
import numpy as np
import sympy as sp

from tests.theory.common import (
    EPS,
    agree,
    evaluate,
    is_zero,
    rng,
)


def _rtol(expr: sp.Expr, vals: dict, n_ops: int) -> float:
    """Float64 error bound of a formula: n_ops roundings, plus one rounding of each input (as the code converts it,
    e.g. to radians) times its condition number |x df/dx / f|."""
    f = float(evaluate(expr, vals))
    cond = sum(abs(float(evaluate(x * sp.diff(expr, x), vals))) for x in vals) / abs(f)
    return (n_ops + cond) * EPS


r1 = sp.Matrix(sp.symbols("r1:4", real=True))
r2 = sp.Matrix(sp.symbols("s1:4", real=True))


def test_rodrigues_misorientation():
    """Rotations with Rodrigues vectors r1, r2 (quaternions (1, r) / |(1, r)|) are misoriented by w with
    tan^2(w / 2) = (|r1 - r2|^2 + |r1 x r2|^2) / (1 + r1 . r2)^2; from the identity (r1 = 0), w = 2 arctan |r2|."""
    n1, n2, dot = 1 + r1.dot(r1), 1 + r2.dot(r2), 1 + r1.dot(r2)
    # cos(w / 2) = q1 . q2 = dot / sqrt(n1 n2), so tan^2(w / 2) = (n1 n2 - dot^2) / dot^2
    assert is_zero(n1 * n2 - dot**2 - ((r1 - r2).dot(r1 - r2) + r1.cross(r2).dot(r1.cross(r2))))


def test_rodrigues_metric_largest_at_origin():
    """A step dr at r turns by ds^2 = 4 |dq|^2 = 4 (|dr|^2 (1 + |r|^2) - (r . dr)^2) / (1 + |r|^2)^2
    <= 4 |dr|^2 / (1 + |r|^2): a grid regular in r is coarsest, in angle, at the origin."""
    rho2 = r1.dot(r1)
    q = sp.Matrix([1, *r1]) / sp.sqrt(1 + rho2)
    J = q.jacobian(r1)
    metric = (sp.eye(3) * (1 + rho2) - r1 * r1.T) / (1 + rho2) ** 2
    assert is_zero(J.T * J - metric)
    assert is_zero(sp.eye(3) / (1 + rho2) - metric - r1 * r1.T / (1 + rho2) ** 2)  # the gap: (r . dr)^2 >= 0


def test_cubic_delta():
    """srch-cubic-delta: a cell of side d = tan(s / 2) centred on the origin has its corners at |r| = sqrt(3) / 2 d,
    misoriented from the identity by 2 arctan of that (test_rodrigues_misorientation with r1 = 0). The step s is the
    misorientation 2 arctan d of neighbours 0 and d."""
    d = sp.symbols("d", positive=True)
    corner = d / 2 * sp.Matrix([1, 1, 1])
    assert is_zero(sp.sqrt(corner.dot(corner)) - sp.sqrt(3) / 2 * d)


def test_cubic_delta_code():
    """anri's cubic_grid_misorientation is srch-cubic-delta."""
    from anri.crystal.orientation import cubic_grid_misorientation

    s = sp.symbols("s", positive=True)
    expr = sp.deg(2 * sp.atan(sp.sqrt(3) / 2 * sp.tan(sp.rad(s) / 2)))
    for step in (1.0, 2.5, 3.0):
        rtol = _rtol(expr, {s: step}, 8)  # radians, tan, /2, *, arctan, *2, degrees
        agree(evaluate(expr, {s: step}), cubic_grid_misorientation, step, rtol=rtol, atol=0)


def test_cubic_zone_corners():
    """(sqrt 2 - 1, sqrt 2 - 1, 3 - 2 sqrt 2) lies on two faces |r_i| = tan(pi / 8) and on the plane sum |r_i| = 1."""
    c = [sp.sqrt(2) - 1, sp.sqrt(2) - 1, 3 - 2 * sp.sqrt(2)]
    assert is_zero(sp.tan(sp.pi / 8) - c[0])
    assert is_zero(sum(c) - 1)
    assert sp.simplify(c[2] - sp.tan(sp.pi / 8)) < 0


th, eta, D = sp.symbols("theta eta Delta", positive=True)
t, sn, cs = sp.tan(th), sp.sin(eta), sp.cos(eta)


def test_tolerance_bounds():
    """From dif-rotation-motion, the largest moves over rotations of size D are D |(1, -tan th / sin eta)| in eta and
    D |(1, cot eta)| = D / |sin eta| in omega (Cauchy-Schwarz). srch-tolerance's bounds hold them: the squares differ
    by (1 + tan^2) 2 tan |cot| + tan^4 cot^2 >= 0 in eta, and by a factor 1 / cos th >= 1 in omega."""
    move_e2 = 1 + t**2 / sn**2
    bound_e2 = (1 + t * cs / sn) ** 2 / sp.cos(th) ** 2  # for 0 < eta < 90 deg, where |cot eta| = cot eta
    assert is_zero(bound_e2 - move_e2 - ((1 + t**2) * 2 * t * cs / sn + t**4 * cs**2 / sn**2))
    assert is_zero(1 + (cs / sn) ** 2 - 1 / sn**2)


def test_ring_width_on_detector():
    """A width h in 2theta, as a displacement across the ring on a flat detector normal to the beam (R = L tan 2th),
    is an angle h / (sin 2th cos 2th) along the ring: dR / R."""
    L, h = sp.symbols("L h", positive=True)
    R = L * sp.tan(2 * th)
    assert is_zero(sp.diff(R, th) / 2 * h / R - h / (sp.sin(2 * th) * sp.cos(2 * th)))


def test_tolerance_code():
    """anri's match_tolerances is srch-tolerance (degrees)."""
    from anri.index.search import match_tolerances

    hr, fr, tth = sp.symbols("h_r frame tth", positive=True)
    e = sp.symbols("eta", real=True)
    tt = sp.rad(tth) / 2
    te = sp.rad(D) / sp.cos(tt) * (1 + sp.tan(tt) * sp.Abs(sp.cot(sp.rad(e)))) + sp.rad(hr) / (
        sp.sin(2 * tt) * sp.cos(2 * tt)
    )
    to = sp.rad(D) / (sp.cos(tt) * sp.Abs(sp.sin(sp.rad(e)))) + sp.rad(fr) / 2
    g = rng(4)
    etas = np.array([[35.0, -120.0, 75.0, 160.0]])
    ring_j = np.array([0, 1, 0, 1])
    ring_tth, ring_hw = np.array([6.0, 9.5]), np.array([0.03, 0.05])
    delta, frame = float(g.uniform(0.5, 2)), 0.25
    want_e, want_o, rtol = [], [], 0.0
    for x, j in zip(etas[0], ring_j):
        vals = {e: x, tth: ring_tth[j], hr: ring_hw[j], D: delta, fr: frame}
        want_e.append(float(evaluate(sp.deg(te), vals)))
        want_o.append(float(evaluate(sp.deg(to), vals)))
        rtol = max(rtol, _rtol(te, vals, 20), _rtol(to, vals, 20))  # ~20 operations, trig included
    args = (etas, ring_j, ring_tth, ring_hw, delta, frame)
    agree(np.array([want_e]), lambda *z: match_tolerances(*z)[0], *args, rtol=rtol, atol=0)
    agree(np.array([want_o]), lambda *z: match_tolerances(*z)[1], *args, rtol=rtol, atol=0)


# ---------------------------------------------------------------------------------------------- the cubochoric grid


def _ball_pyramid(x: sp.Expr, y: sp.Expr, z: sp.Expr) -> sp.Matrix:
    """srch-cubochoric on the pyramid where z is the largest coordinate and |y| <= |x|: the scaling, the square to
    curved square map and the inverse Lambert projection."""
    sc = (sp.pi / 6) ** sp.Rational(1, 6)
    X, Y, Zs = sc * x, sc * y, sc * z
    beta = sp.pi ** sp.Rational(5, 6) / (2 * 6 ** sp.Rational(1, 6))
    prek = (3 * sp.pi / 4) ** sp.Rational(1, 3) * 2 ** sp.Rational(1, 4) / beta
    t = sp.pi / 12 * Y / X
    q = prek * X / sp.sqrt(sp.sqrt(2) - sp.cos(t))
    T1, T2 = (sp.sqrt(2) * sp.cos(t) - 1) * q, sp.sqrt(2) * sp.sin(t) * q
    cc = T1**2 + T2**2
    root = sp.sqrt(1 - sp.pi * cc / (24 * Zs**2))
    return sp.Matrix([T1 * root, T2 * root, sp.sqrt(6 / sp.pi) * Zs - sp.sqrt(sp.pi) * cc / sp.sqrt(24) / Zs])


def _ball(p: tuple) -> sp.Matrix:
    """srch-cubochoric at a point of the cube: the pyramid of its largest coordinate (that one taken as z, the others
    in cyclic order), the larger of the other two as x."""
    i = max(range(3), key=lambda k: abs(p[k]))
    if all(v == 0 for v in p):
        return sp.zeros(3, 1)
    order = [(i + 1) % 3, (i + 2) % 3, i]
    x, y, z = (p[k] for k in order)
    if x == 0 and y == 0:
        h = sp.Matrix([0, 0, sp.sqrt(6 / sp.pi) * (sp.pi / 6) ** sp.Rational(1, 6) * z])
    elif abs(y) <= abs(x):
        h = _ball_pyramid(x, y, z)
    else:
        h = _ball_pyramid(y, x, z)
        h = sp.Matrix([h[1], h[0], h[2]])
    out = [0, 0, 0]
    for k, v in zip(order, h):
        out[k] = v
    return sp.Matrix(out)


def test_cubochoric_volume():
    """srch-cubochoric: on each pyramid the map from the cube of edge pi^(2/3) has Jacobian determinant 1, the cube's
    volume pi^2 being the ball's, 4/3 pi (3 pi / 4); and a face of the cube goes onto the ball's surface. So the grid's
    points are spread evenly in volume, as [Rosca2014]_ constructs it."""
    x, y, z = sp.symbols("x y z", positive=True)
    h = _ball_pyramid(x, y, z)
    u = sp.symbols("u", positive=True)
    det = h.jacobian([x, y, z]).det()
    assert sp.simplify(det.subs(y, 12 * x * u / sp.pi) - 1) == 0
    face = h.subs(z, sp.pi ** sp.Rational(2, 3) / 2)
    assert sp.simplify((face.dot(face) - (3 * sp.pi / 4) ** sp.Rational(2, 3)).subs(y, 12 * x * u / sp.pi)) == 0


def test_cubochoric_code():
    """cubochoric_quaternions against srch-cubochoric and the homochoric angle, |h|^3 = 3/4 (w - sin w), at every point
    of a 4 x 4 x 4 grid (every pyramid, both halves of each, the edges and the identity)."""
    from anri.crystal.orientation import cubochoric_quaternions

    N = 2
    a = sp.pi ** sp.Rational(2, 3) / 2
    g = [k * a / N for k in range(-N + 1, N + 1)]
    pts = [(gx, gy, gz) for gx in g for gy in g for gz in g]  # the code's order (meshgrid, ij)
    got = cubochoric_quaternions(N)
    w_ = sp.symbols("w", positive=True)
    for p, q in zip(pts, got):
        h = _ball(p)
        n = sp.sqrt(h.dot(h))
        if n == 0:
            want = np.array([1.0, 0, 0, 0])
            w = 0.0
        else:
            w = sp.nsolve(sp.Rational(3, 4) * (w_ - sp.sin(w_)) - n**3, w_, 2.0, prec=40)
            want = np.array([float(sp.N(sp.cos(w / 2), 30))] + [float(sp.N(sp.sin(w / 2) * c / n, 30)) for c in h])
        # the map: ~30 operations on coordinates <= 2 (16 ulps x 2 covers them); the angle by 60 bisections
        # (pi 2^-60) of w - sin w, whose rounding, eps w, moves w by eps w / (3/4 (1 - cos w)) ~ 2.7 eps / w; the
        # quaternion moves by half that
        wf = float(w)
        dw = np.pi * 2.0**-60 + (2.7 * EPS / wf if wf > 0 else 0.0)
        np.testing.assert_allclose(q, want, rtol=0, atol=2 * 16 * EPS * 2 + dw / 2)


def _ops(name: str) -> np.ndarray:
    from anri.crystal.orientation import laue_rotations
    from anri.crystal.structure import B_matrix, symmetry_matrices

    lp, sg = {"m-3m": ([3.6] * 3 + [90] * 3, 225), "6/mmm": ([2.95, 2.95, 4.68, 90, 90, 120], 194),
              "-3": ([5.0, 5.0, 14.0, 90, 90, 120], 148), "2/m": ([4.0, 5.0, 6.0, 90, 100, 90], 14),
              "mmm": ([4.0, 5.0, 6.0, 90, 90, 90], 62), "-1": ([4.0, 5.0, 6.0, 80, 100, 95], 2)}[name]  # fmt: skip
    return laue_rotations(symmetry_matrices(sg), B_matrix(lp))


def test_grid_delta_measured():
    """**Measured** (search.rst, Anri's choice): every orientation is within delta of a grid point, and every grid point
    is in the zone or its half-step shell. Random orientations stand for "every": this is a measurement, not a proof."""
    from anri.crystal.orientation import disorientation, orientation_grid
    from anri.maths.rotation import random_rotations

    test = random_rotations(300, np.random.default_rng(5))
    for name, step in (("m-3m", 6.0), ("6/mmm", 8.0), ("-3", 10.0), ("2/m", 12.0)):
        S = _ops(name)
        U, delta = orientation_grid(step, S)
        U = U.astype(float)
        angle = np.degrees(np.arccos(np.clip((np.trace(U, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        least = disorientation(np.repeat(np.eye(3)[None], len(U), 0), U, S)
        assert (angle - least).max() <= 0.5 * step * (1 + 1e-6)  # in the zone, or its shell
        nearest = np.array([disorientation(np.repeat(t[None], len(U), 0), U, S).min() for t in test])
        assert nearest.max() <= delta


def test_zone_sizes():
    """A fundamental zone holds 1 / |group| of all rotations: the reduced cubochoric grid holds that share of the whole
    grid's points, plus its half-step shell (which adds at most the shell's share, under 30% at 6 deg)."""
    from anri.crystal.orientation import cubochoric_quaternions, orientation_grid

    n_all = len(cubochoric_quaternions(int(np.round(131.97049 / (6.0 - 0.03732)))))
    for name in ("6/mmm", "mmm", "-1"):
        S = _ops(name)
        U, _ = orientation_grid(6.0, S)
        ratio = len(U) * len(S) / n_all
        assert 1.0 <= ratio < 1.3


# ---------------------------------------------------------------------------------------------- completeness


def test_summed_area():
    """A summed-area table T[i, j] = sum of a[:i, :j] counts any box exactly: sum of a[i0:i1, j0:j1] =
    T[i1, j1] - T[i0, j1] - T[i1, j0] + T[i0, j0]. Checked for every box of a 3 x 3 array of symbols."""
    a = sp.Matrix(3, 3, sp.symbols("a0:9"))
    T = lambda i, j: sum((a[p, q] for p in range(i) for q in range(j)), sp.Integer(0))
    for i0 in range(4):
        for i1 in range(i0, 4):
            for j0 in range(4):
                for j1 in range(j0, 4):
                    box = sum((a[p, q] for p in range(i0, i1) for q in range(j0, j1)), sp.Integer(0))
                    assert sp.expand(T(i1, j1) - T(i0, j1) - T(i1, j0) + T(i0, j0) - box) == 0


def _lit(seed: int = 28) -> np.ndarray:
    return rng(seed).random((2, 12, 10)) < 0.15  # [ring, eta, omega]


def test_lit_table_code():
    """lit_table is test_summed_area's table of the lit map tiled three times along eta (so that a box may wrap past
    either end), with a zero first row and column: exactly, as integers."""
    from anri.index.data import lit_table

    lit = _lit()
    t = np.concatenate([lit, lit, lit], 1).astype(np.int64)
    want = np.zeros((2, t.shape[1] + 1, t.shape[2] + 1), np.int64)
    for r in range(2):
        for i in range(t.shape[1] + 1):
            for j in range(t.shape[2] + 1):
                want[r, i, j] = t[r, :i, :j].sum()
    np.testing.assert_array_equal(np.asarray(lit_table(jnp.asarray(lit))), want)


def test_completeness_code():
    """completeness: the share of a orientation's predictions on the scan whose box (eta bin +- ceil(tol_e / b_e),
    periodic; omega bin +- ceil(tol_o / b_o), clipped to the scan) holds a lit bin of its ring, counted directly on the
    lit map. Exactly."""
    from anri.index.data import lit_table
    from anri.index.search import completeness

    lit = _lit()
    n_e, n_o, b_e, b_o, om0 = 12, 10, 30.0, 2.0, 5.0
    g = rng(29)
    eta = g.uniform(-180, 180, (3, 6))
    om = g.uniform(om0 - 2, om0 + n_o * b_o + 2, (3, 6))  # some past the scan's ends
    ok = g.random((3, 6)) < 0.9
    ring_j = g.integers(0, 2, 6).astype(np.int32)
    te, to = g.uniform(1, 70, (3, 6)), g.uniform(0.5, 5, (3, 6))
    want = []
    for q in range(3):
        hit = n = 0
        for j in range(6):
            ie, io = int(np.floor((eta[q, j] + 180) / b_e)) % n_e, int(np.floor((om[q, j] - om0) / b_o))
            if not ok[q, j] or not 0 <= io < n_o:
                continue
            n += 1
            de, do = min(int(np.ceil(te[q, j] / b_e)), n_e - 1), int(np.ceil(to[q, j] / b_o))
            es = [(ie + k) % n_e for k in range(-de, de + 1)]
            os_ = range(max(io - do, 0), min(io + do + 1, n_o))
            hit += any(lit[ring_j[j], e, o] for e in es for o in os_)
        want.append(hit / max(n, 1))
    got, _ = completeness(lit_table(jnp.asarray(lit)), jnp.asarray(eta), jnp.asarray(om), jnp.asarray(ok),
                          jnp.asarray(ring_j), jnp.asarray(te), jnp.asarray(to), om0, b_e, b_o, n_e, n_o)  # fmt: skip
    np.testing.assert_array_equal(np.asarray(got, np.float64), np.asarray(want, np.float32).astype(np.float64))


def test_lit_map_code():
    """lit_map: the bins above lit times the median of the non-empty bins of the row-summed histogram, as
    test_lit_table_code's table."""
    from anri.index.data import lit_table
    from anri.index.search import lit_map

    H = np.floor(rng(30).exponential(3.0, (2, 12, 10))) * (rng(31).random((2, 12, 10)) < 0.6)
    scan = {"n_e": 12, "n_o": 10, "om0": 0.0, "omega_step": 0.25}
    med = np.median(H[H > 0])  # small integers: exact
    out = lit_map(H.ravel(), scan, 2, 1.5, 0.1)
    np.testing.assert_array_equal(np.asarray(out["table"]), np.asarray(lit_table(jnp.asarray(H > 1.5 * med))))


def test_tolerance_measured():
    """**Measured** (search.rst): exact spot moves of random orientations turned by 0.5 deg about random axes, from
    predict (the exact Ewald solution) in float64, stay within match_tolerances' first-order bounds (|sin eta| > 0.3)."""
    from anri.crystal.structure import B_matrix, reflections
    from anri.index.search import match_tolerances, predict
    from anri.maths.rotation import random_rotations, rmat_from_axis_angle
    from anri.maths.utils import float64

    a, lam = 2.8665, 12.398419843320026 / 43.0
    B = B_matrix(np.array([a] * 3 + [90.0] * 3))
    r = reflections([a] * 3 + [90.0] * 3, 229, lam, 2 * np.sin(np.radians(9.0)) / lam)
    hkls = r["hkl"].astype(float)
    ring_tth, ring_j = np.unique(np.round(r["tth"], 6), return_inverse=True)
    geom = {"wavelength": lam, "k_in_lab": np.array([1.0, 0, 0]), "wedge": 0.0, "chi": 0.0, "pol_factor": 1.0}
    g = np.random.default_rng(0)
    U0 = random_rotations(200, g)
    with float64():
        dU = np.asarray(rmat_from_axis_angle(g.normal(size=(200, 3)), np.full(200, 0.5)))
        e0, o0, k0, _ = (np.asarray(x) for x in predict(U0, B, hkls, geom))
        e1, o1, k1, _ = (np.asarray(x) for x in predict(dU @ U0, B, hkls, geom))
        te, to = (np.asarray(x) for x in match_tolerances(e0, np.repeat(ring_j, 2), ring_tth, np.zeros(len(ring_tth)), 1.0, 0.0))
    use = k0 & k1 & (np.abs(np.sin(np.radians(e0))) > 0.3)
    de = np.abs((e1 - e0 + 180) % 360 - 180)
    do = np.abs((o1 - o0 + 180) % 360 - 180)
    assert (de / te)[use].max() <= 1.0 and (do / to)[use].max() <= 1.0, ((de / te)[use].max(), (do / to)[use].max())


def test_grid_chance_code():
    """grid_chance: the median completeness over a random sample of the grid; with a sample as large as the grid it
    is the whole grid's median (a sample without replacement is then a permutation), exactly, with the grid's size and
    worst case."""
    from anri.crystal.lattice import lpars_to_B
    from anri.crystal.orientation import orientation_grid
    from anri.index.data import lit_table, ring_table
    from anri.index.search import completeness_of, grid_chance
    from anri.io.imaged11 import geom_from_pars

    ops = _ops("-1")
    lp = np.array([4.0, 5.0, 6.0, 80.0, 100.0, 95.0])
    B = np.asarray(lpars_to_B(lp), np.float32)
    rings = ring_table(lp, 2, 0.3, 3)
    rings["hw"] = np.full(len(rings["tth"]), 0.05)
    pars = {"y_center": 1024.0, "y_size": 75.0, "tilt_y": 0.0, "z_center": 1024.0, "z_size": 75.0, "tilt_z": 0.0,
            "tilt_x": 0.0, "distance": 150e3, "o11": 1, "o12": 0, "o21": 0, "o22": 1, "wavelength": 0.3}  # fmt: skip
    geom = geom_from_pars(pars, 0.0, 0.0, 0.0, 0.0, sig_beam=0.5, voxel_size=1.0)
    n_e, n_o = 720, 1440
    lit_map = rng(42).random((3, n_e, n_o)) < 0.05
    lit = {"table": lit_table(jnp.asarray(lit_map)), "om0": -180.0, "bins": (0.5, 0.25, n_e, n_o),
           "frame_step": 0.25, "etacut": 0.1}  # fmt: skip
    U, delta = orientation_grid(30.0, ops)
    chance, n, d = grid_chance(30.0, ops, B, rings, geom, lit, np.random.default_rng(0), n_sample=len(U))
    assert (n, d) == (len(U), delta)
    assert chance == float(np.median(completeness_of(U, delta, B, rings, geom, lit)))

