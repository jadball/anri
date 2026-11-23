"""Proofs for docs/source/theory/limits.rst: what scanning 3DXRD with one rotation axis determines."""

import mpmath as mp
import numpy as np
import sympy as sp

from tests.theory.common import (
    is_zero,
    rank,
    skew,
)

# ---------------------------------------------------------------------------------------------------------------
# helpers: beta (3x3) as a 9-vector, row-major; delta g = beta g0


def test_ray_component():
    """lim-ray: g . xi / |g| = -sin(theta) for a ray along the incident beam xi."""
    th, k = sp.symbols("theta k", positive=True)
    kin = sp.Matrix([k, 0, 0])
    g = k * sp.Matrix([sp.cos(2 * th), sp.sin(2 * th), 0]) - kin
    # g . xi = -2 k sin^2(theta) and |g| = 2 k sin(theta), so the ratio is -sin(theta)
    assert is_zero(sp.expand_trig(g.dot(kin / k)) + 2 * k * sp.sin(th) ** 2)
    assert is_zero(sp.expand_trig(g.dot(g)) - 4 * k**2 * sp.sin(th) ** 2)


def test_fourier_slice():
    """lim-slice: for g in span(k, z), beta g depends on beta k and beta z only."""
    B = sp.Matrix(3, 3, sp.symbols("b0:9"))
    a, c = sp.symbols("a c")
    kh, zh = sp.Matrix([1, 0, 0]), sp.Matrix([0, 0, 1])
    assert is_zero(B * (a * kh + c * zh) - (a * B * kh + c * B * zh))


def test_seen_and_weak():
    """lim-seen-weak: the rotation/strain content of beta k, beta z (seen) and beta xi (weak), k = x, xi = y."""
    wx, wy, wz = sp.symbols("w_x w_y w_z")
    e = sp.symbols("e_xx e_yy e_zz e_xy e_xz e_yz")
    E = sp.Matrix([[e[0], e[3], e[4]], [e[3], e[1], e[5]], [e[4], e[5], e[2]]])
    beta = E + skew([wx, wy, wz])
    x, y, z = sp.eye(3).columnspace()
    assert is_zero(beta * x - sp.Matrix([e[0], e[3] + wz, e[4] - wy]))
    assert is_zero(beta * z - sp.Matrix([e[4] + wy, e[5] - wx, e[2]]))
    assert is_zero(beta * y - sp.Matrix([e[3] - wz, e[1], e[5] + wx]))


def test_nye():
    """lim-nye: i beta(k) xi = (curl beta)_{.z} for in-plane k, xi = z x k."""
    kx, ky = sp.symbols("k_x k_y", real=True)
    B = sp.Matrix(3, 3, sp.symbols("b0:9"))
    xi = sp.Matrix([0, 0, 1]).cross(sp.Matrix([kx, ky, 0]))
    curl_z = sp.Matrix([sp.I * (kx * B[i, 1] - ky * B[i, 0]) for i in range(3)])
    assert is_zero(sp.I * B * xi - curl_z)


def test_compatible_no_weak():
    """lim-compatible: beta = grad u (beta(k) = i u k^T) has no weak part."""
    kx, ky = sp.symbols("k_x k_y", real=True)
    u = sp.Matrix(sp.symbols("u0:3"))
    k = sp.Matrix([kx, ky, 0])
    xi = sp.Matrix([0, 0, 1]).cross(k)
    assert is_zero(sp.I * u * k.T * xi)


def test_beam_transfer():
    """lim-transfer: Fourier transform of a Gaussian beam profile of FWHM 1 at period Lambda."""
    x = sp.symbols("x", real=True)
    lam = sp.symbols("Lambda", positive=True)
    s = 1 / (2 * sp.sqrt(2 * sp.log(2)))
    prof = sp.exp(-(x**2) / (2 * s**2)) / (s * sp.sqrt(2 * sp.pi))
    T = sp.integrate(prof * sp.cos(2 * sp.pi * x / lam), (x, -sp.oo, sp.oo))
    assert is_zero(sp.log(T) + sp.pi**2 / (4 * sp.log(2) * lam**2))
    # the page's numbers, as exact roundings of 1 / T
    assert [mp.nstr(sp.N(1 / T.subs(lam, v), 30), 2) for v in (1, sp.Rational(3, 2), 2)] == ["35.0", "4.9", "2.4"]


def test_merged_information():
    """lim-mu: the two integrals behind mu(k): the merged segment and the coherent window of directions."""
    t, kap, D, b, K = sp.symbols("t kappa D beta K", positive=True)
    seg = sp.integrate(sp.exp(sp.I * kap * t), (t, -D / 2, D / 2))
    assert is_zero(sp.simplify((seg - 2 * sp.sin(kap * D / 2) / kap).rewrite(sp.exp)))
    win = sp.integrate((sp.sin(K * D * b / 2) / (K * D * b / 2)) ** 2, (b, -sp.oo, sp.oo))
    assert is_zero(sp.simplify(win - 2 * sp.pi / (K * D)))
    L, M = sp.symbols("L M", positive=True)
    mu = sp.simplify((1 / M) * 2 * M / (2 * sp.pi) * win).subs(K, 2 * sp.pi / L)
    assert is_zero(sp.simplify(mu - L / (sp.pi * D)))


def test_rotation_about_g_invisible():
    """lim-separation: a rotation delta moves g by delta x g; |delta x g| = |delta| |g| sin(angle)."""
    d = sp.Matrix(sp.symbols("d0:3", real=True))
    g = sp.Matrix(sp.symbols("g0:3", real=True))
    dg = skew(d) * g
    assert is_zero(dg - d.cross(g))
    assert is_zero(dg.dot(dg) - (d.dot(d) * g.dot(g) - d.dot(g) ** 2))
    assert is_zero(skew(g) * g)


def _sd_rows(step, x0, n=80):
    """sd(x0) sqrt(N) / sigma_beam from counts per row (Gaussian beam, sigma 1, Poisson, total free), at 30 digits.
    Rows beyond |j| = n carry < exp(-n^2 step^2 / 2) of the light: nothing at 30 digits for step >= 1."""
    mp.mp.dps = 30
    js = range(-n, n + 1)
    p = [mp.e ** (-((j * step - x0) ** 2) / 2) for j in js]
    dp = [q * (j * step - x0) for q, j in zip(p, js)]
    Z, dZ = mp.fsum(p), mp.fsum(dp)
    info = mp.fsum((d / Z - q * dZ / Z**2) ** 2 / (q / Z) for q, d in zip(p, dp))
    return 1 / mp.sqrt(info)


def test_localisation():
    """lim-localise: continuous sampling gives 1/sigma^2 per photon; in rows, the values the page quotes on a row
    (x0 = 0) and midway (x0 = step / 2), both stationary by symmetry, as exact roundings."""
    x, x0 = sp.symbols("x x0", real=True)
    p = sp.exp(-((x - x0) ** 2) / 2) / sp.sqrt(2 * sp.pi)
    info = sp.integrate(sp.diff(p, x0) ** 2 / p, (x, -sp.oo, sp.oo))
    assert is_zero(info - 1)
    fwhm = 2 * mp.sqrt(2 * mp.log(2))
    half = fwhm / 2
    assert abs(_sd_rows(half, 0) - 1) < mp.mpf("2e-5") and abs(_sd_rows(half, half / 2) - 1) < mp.mpf("2e-5")
    assert mp.nstr(_sd_rows(fwhm, fwhm / 2), 2) == "0.84" and mp.nstr(_sd_rows(fwhm, 0), 3) == "1.27"
    st = 3 * fwhm / 2
    assert mp.nstr(_sd_rows(st, st / 2), 2) == "0.57" and mp.nstr(_sd_rows(st, 0), 2) == "4.5"


def test_merged_length():
    """lim-merged: random-walk misorientation theta_c sqrt(r / c) exceeds n own widths W at r = D d."""
    th, c, d, n, W, r = sp.symbols("theta_c c d n W r", positive=True)
    D = sp.solve(sp.Eq(th * sp.sqrt(r / c), n * W), r)[0] / d
    assert is_zero(D - n**2 * W**2 * c / (d * th**2))
    assert is_zero(D.subs(W, th * sp.sqrt(d / c)) - n**2)


def test_second_moment_contrast_only():
    """lim-mom2-jac: d(central second moment)/d x_v = 2 w_v (x_v - m) / W; zero at a flat state."""
    x = sp.symbols("x1:4")
    w = sp.symbols("w1:4", positive=True)
    W = sum(w)
    m = sum(wi * xi for wi, xi in zip(w, x)) / W
    raw2 = sum(wi * xi**2 for wi, xi in zip(w, x)) / W
    cen2 = raw2 - m**2
    for v in range(3):
        assert is_zero(sp.diff(cen2, x[v]) - 2 * w[v] / W * (x[v] - m))
    flat = {x[1]: x[0], x[2]: x[0]}
    assert all(is_zero(sp.diff(cen2, xi).subs(flat)) for xi in x)
    # raw second moment at a flat state: 2 x0 times the first moment's row
    assert all(is_zero((sp.diff(raw2, xi) - 2 * x[0] * sp.diff(m, xi)).subs(flat)) for xi in x)
    # Gaussian moment noise, 3D isotropic C = s2 I: (1/2) tr((C^-1 dC)^2) with dC = 2 sym(a b^T)
    a = sp.Matrix(sp.symbols("a1:4"))
    b = sp.Matrix(sp.symbols("b1:4"))
    s2 = sp.symbols("s2", positive=True)
    dC = a * b.T + b * a.T
    assert is_zero((dC * dC).trace() / (2 * s2**2) - (a.dot(a) * b.dot(b) + a.dot(b) ** 2) / s2**2)


# ---- per-voxel absorption of the scatter term by a free spread (integer arithmetic, exact rank) ----------------

_E = np.eye(3, dtype=object)
_ROT = [np.array(skew(e).tolist(), dtype=object).ravel() for e in _E]
_STR = []
for i in range(3):
    for j in range(i, 3):
        S = np.zeros((3, 3), dtype=object)
        S[i, j] = S[j, i] = 1
        _STR.append(S.ravel())


def _outer(U, V, same):
    return [np.outer(u, v) + np.outer(v, u) for i, u in enumerate(U) for j, v in enumerate(V) if not same or j >= i]


_NUIS = {
    "a": [],
    "b": _outer(_ROT, _ROT, True),
    "c1": _outer(_ROT, _ROT, True) + _outer(_STR, _STR, True),
    "c2": _outer(_ROT, _ROT, True) + _outer(_STR, _STR, True) + _outer(_ROT, _STR, False),
}


def _G(g):
    return np.kron(np.eye(3, dtype=object), np.array(g, dtype=object)[None, :])


def _frame(g, e):
    t1 = np.cross(g, e)
    return [np.array(g, dtype=object), t1, np.cross(g, t1)]


def _jac(refl, c, basis, nuis, drop_rr):
    rows = []
    pairs = [(i, j) for i in range(3) for j in range(i, 3) if not (drop_rr and i == j == 0)]
    for g, fr in refl:
        G = _G(g)
        a = G.dot(c)
        cols = []
        for d in basis:
            bd = G.dot(d)
            X = np.outer(a, bd) + np.outer(bd, a)
            cols.append([fr[i].dot(X).dot(fr[j]) for i, j in pairs])
        for S in nuis:
            X = G.dot(S).dot(G.T)
            cols.append([fr[i].dot(X).dot(fr[j]) for i, j in pairs])
        rows += [list(r) for r in zip(*cols)]
    return sp.Matrix(rows)


def _seen(refl, c, basis, cls, drop_rr):
    J = _jac(refl, c, basis, _NUIS[cls], drop_rr)
    n = len(basis)
    return rank(J) - (rank(J[:, n:]) if J.shape[1] > n else 0)


def _reflections(gen, n=12):
    out = []
    while len(out) < n:
        g, e = gen.integers(-5, 6, 3), gen.integers(-5, 6, 3)
        if np.any(g) and np.any(np.cross(g, e)):
            out.append((g, _frame(g, e)))
    return out


def test_second_moment_absorption():
    """lim-mom2-rank: combinations of d beta_v (of 9) the second moments see beyond each class of free spread."""
    gen = np.random.default_rng(1)
    refl = _reflections(gen)
    E9 = list(np.eye(9, dtype=object).astype(int).astype(object))
    rot = np.array(skew(gen.integers(-5, 6, 3).tolist()).tolist(), dtype=object).ravel()
    S = gen.integers(-5, 6, (3, 3))
    strain = np.array(S + S.T, dtype=object).ravel()
    want = {("a", False): (9, 9), ("b", True): (6, 9), ("c1", False): (6, 3), ("c2", False): (0, 0)}
    for (cls, drop), (n_rot, n_str) in want.items():
        assert _seen(refl, rot, E9, cls, drop) == n_rot, cls
        assert _seen(refl, strain, E9, cls, drop) == n_str, cls
    # with rotation contrast, (b) sees exactly the strain part of d beta
    assert _seen(refl, rot, _ROT, "b", True) == 0
    assert _seen(refl, rot, _STR, "b", True) == 6


def test_common_offset_is_k0():
    """lim-common: an offset s shared by every voxel's spot in a reflection shifts each blob mean by s, whatever
    the ray: the data change is constant over rays."""
    w = sp.symbols("w1:4", positive=True)
    x = sp.symbols("x1:4")
    s = sp.symbols("s")
    m = sum(wi * xi for wi, xi in zip(w, x)) / sum(w)
    ms = sum(wi * (xi + s) for wi, xi in zip(w, x)) / sum(w)
    assert is_zero(ms - m - s)
