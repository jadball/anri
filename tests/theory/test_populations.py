"""Proofs for docs/source/theory/populations.rst: the mean of a population, its spread as sig_rot, the grouping
radius, and the disc mask."""

import numpy as np
import sympy as sp

from tests.theory.common import (
    EPS,
    agree,
    cayley,
    evaluate,
    is_zero,
    rng,
    skew,
    tol,
    vec,
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


def test_mean_of_symmetric_pairs():
    """pop-mean for any rotation R and rotations V_j (Cayley, about any axes, by less than 90 degrees): the pairs
    R V_j and R V_j^T, weighted w_j each, sum to R P with P = sum w_j (V_j + V_j^T). V + V^T = 2 c I + 2 (1 - c) n n^T
    (c the cosine of V's angle, n its axis), whose eigenvalues 2c, 2c and 2 are positive: P is symmetric positive
    definite, so R is the sum's polar factor, its closest rotation."""
    k = vec("k")
    V = cayley(k)
    kk = k.dot(k)
    c = (1 - kk) / (1 + kk)  # cos(2 atan |k|)
    n = k / sp.sqrt(kk)
    assert is_zero(V + V.T - 2 * c * sp.eye(3) - 2 * (1 - c) * n * n.T)
    assert is_zero((V + V.T) * n - 2 * n)  # eigenvalue 2 along the axis; 2c across it (the identity above)


def test_mean_rotation_general_code():
    """anri's mean_rotation of symmetric pairs about three axes is R (test_mean_of_symmetric_pairs)."""
    from anri.maths.rotation import mean_rotation

    g = rng(13)
    R = cayley(sp.Matrix([sp.Rational(int(x), 10) for x in g.integers(-9, 9, 3)]))
    Vs = [cayley(sp.Matrix([sp.Rational(int(x), 20) for x in g.integers(-9, 9, 3)])) for _ in range(3)]
    w = [sp.Rational(int(x), 10) for x in g.integers(2, 10, 3)]
    U = np.array([evaluate(R * M, {}) for V in Vs for M in (V, V.T)])
    wv = np.repeat(np.array(w, float), 2)
    P = evaluate(sum((wi * (V + V.T) for wi, V in zip(w, Vs)), sp.zeros(3, 3)), {})
    sv = np.linalg.eigvalsh(P)
    # the sum rounds by a few ulps of |M|_F = |P|_F, the SVD is backward stable, and the polar factor moves by at most
    # 2 / (sigma_2 + sigma_3) times the change in the sum (Higham, Functions of Matrices, theorem 8.9)
    t = tol(2 * np.linalg.norm(P) / (sv[0] + sv[1]))
    agree(evaluate(R, {}), mean_rotation, U, wv, **t)


def test_rotation_angle_of_cayley():
    """The rotation cayley(k) turns by theta = 2 atan |k| about k: its trace is 1 + 2 cos(theta) and its axial part
    (R - R^T) / 2 is sin(theta) [k / |k|]x, which rotation_angle reads with atan2."""
    k = vec("k")
    R = cayley(k)
    kk = k.dot(k)
    cos_t, sin_t = (1 - kk) / (1 + kk), 2 * sp.sqrt(kk) / (1 + kk)
    assert is_zero(R.trace() - 1 - 2 * cos_t)
    assert is_zero((R - R.T) / 2 - sin_t * skew(k / sp.sqrt(kk)))


def test_rotation_angle_code():
    """rotation_angle against 2 atan |k|, from 1e-8 rad to near 180 degrees."""
    from anri.maths.rotation import rotation_angle

    d = rng(14).normal(size=3)
    d /= np.linalg.norm(d)
    ks = [sp.Matrix([sp.nsimplify(x) for x in d]) * sp.Rational(s) for s in ("1/200000000", "1/100", "1/2", "40")]
    R = np.array([evaluate(cayley(k), {}) for k in ks])
    want = np.array([float(sp.N(2 * sp.atan(sp.sqrt(k.dot(k))), 30)) for k in ks])
    # R's entries round once (<= 1); atan2 of (|axial|, (trace - 1) / 2), a unit pair, moves by at most the change in
    # either: a few ulps, absolute
    agree(want, rotation_angle, R, **tol(1.0))


def _cubic_ops() -> list:
    """The 24 rotations of the cube: signed permutation matrices of determinant 1, exactly."""
    import itertools

    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = sp.zeros(3, 3)
            for i, (j, sg) in enumerate(zip(perm, signs)):
                M[i, j] = sg
            if M.det() == 1:
                out.append(M)
    return out


def test_closest_equivalent_code():
    """closest_equivalent, disorientation and to_fundamental_zone against their definitions, with exact rotations and
    the cube's 24 rotations: the equivalent U S with the largest trace(U_ref^T U S) (the smallest angle,
    test_rotation_angle_of_cayley: trace = 1 + 2 cos), its angle, the least angle over S, and U's own least."""
    from anri.crystal.orientation import closest_equivalent, disorientation, to_fundamental_zone

    ops = _cubic_ops()
    opsv = np.array([evaluate(S, {}) for S in ops])
    g = rng(16)
    R = [cayley(sp.Matrix([sp.Rational(int(x), 7) for x in g.integers(-9, 9, 3)])) for _ in range(3)]
    Uref = cayley(sp.Matrix([sp.Rational(1, 9), sp.Rational(-1, 5), sp.Rational(2, 11)]))
    def angle(M: sp.Matrix) -> sp.Expr:  # atan2(|axial|, (trace - 1) / 2), test_rotation_angle_of_cayley
        ax = sp.Matrix([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]) / 2
        return sp.atan2(sp.sqrt(ax.dot(ax)), (M.trace() - 1) / 2)

    want_U, want_a, want_dis, want_fz = [], [], [], []
    for U in R:
        tr = [(Uref.T * U * S).trace() for S in ops]
        best = U * ops[max(range(24), key=lambda k: tr[k])]
        want_U.append(evaluate(best, {}))
        want_a.append(float(sp.N(angle(Uref.T * best), 30)))
        want_dis.append(float(sp.N(sp.deg(min((angle(Uref.T * U * S) for S in ops), key=lambda e: sp.N(e))), 30)))
        fz = U * ops[max(range(24), key=lambda k: (U * ops[k]).trace())]
        want_fz.append(evaluate(fz, {}))
    Uv = np.array([evaluate(U, {}) for U in R])
    Urv = evaluate(Uref, {})
    # products of rotations (entries <= 1) and atan2 of a unit pair: a few ulps (as test_rotation_angle_code); the
    # disorientation in degrees, up to 63
    agree(np.array(want_U), lambda *z: closest_equivalent(*z)[0], np.broadcast_to(Urv, Uv.shape), Uv, opsv, **tol(3.0))
    agree(np.array(want_a), lambda *z: closest_equivalent(*z)[1], np.broadcast_to(Urv, Uv.shape), Uv, opsv, **tol(3.0))
    agree(np.array(want_dis), disorientation, np.broadcast_to(Urv, Uv.shape), Uv, opsv, **tol(3.0 * 180 / np.pi))
    agree(np.array(want_fz), to_fundamental_zone, Uv, opsv, **tol(3.0))


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
