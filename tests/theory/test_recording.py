"""Proofs for docs/source/theory/recording.rst: what the detector records (photon counts, frames, the cut and the
group rule)."""

from __future__ import annotations

import itertools

import numpy as np
import sympy as sp

from tests.theory.common import (
    N8,
    agree,
    animals,
    boundary,
    evaluate,
    float64,
    is_zero,
    p_at_least,
    p_group,
    rng,
    rounds_to,
    tol,
)


def _poisson_sum(expr_k, lam: sp.Symbol, lo: int) -> sp.Expr:
    """sum_{k >= lo} expr_k(k) e^-lam lam^k / k!, done symbolically."""
    k = sp.Symbol("k", integer=True, nonnegative=True)
    return sp.simplify(sp.summation(expr_k(k) * sp.exp(-lam) * lam**k / sp.factorial(k), (k, lo, sp.oo)))


def test_displaced_counts_poisson():
    """Photons of one Poisson source counted in pixel 1 or 2 (a displaced count, e.g. escaped fluorescence) give
    independent Poisson counts: the joint generating function factorises."""
    lam, p, s, t = sp.symbols("lambda p s t", positive=True)
    joint = sp.exp(lam * (p * s + (1 - p) * t - 1))  # E[s^n1 t^n2]: each photon to pixel 1 with probability p
    assert is_zero(joint - sp.exp(lam * p * (s - 1)) * sp.exp(lam * (1 - p) * (t - 1)))


def test_sheppard_offset_average():
    """With the frame grid at a uniform random offset, the frame centre minus the true omega is uniform on
    [-h/2, h/2) and independent of omega: the binned mean is unbiased, the variance gains h^2 / 12, the fourth central
    moment 6 sigma^2 h^2 / 12 + h^4 / 80, odd central moments are unchanged."""
    h, t, s2, m3, m4 = sp.symbols("h t sigma2 mu3 mu4", positive=True)
    e = [sp.integrate((h * t) ** k, (t, -sp.Rational(1, 2), sp.Rational(1, 2))) for k in range(5)]
    assert e[0] == 1 and e[1] == 0 and is_zero(e[2] - h**2 / 12) and e[3] == 0 and is_zero(e[4] - h**4 / 80)
    # central moments of x + u, x (central moments 0, sigma2, mu3, mu4) and u independent
    mx = [1, 0, s2, m3, m4]
    mom = [sum(sp.binomial(n, j) * mx[j] * e[n - j] for j in range(n + 1)) for n in range(5)]
    assert mom[1] == 0
    assert is_zero(mom[2] - (s2 + h**2 / 12))
    assert is_zero(mom[3] - m3)
    assert is_zero(mom[4] - (m4 + 6 * s2 * h**2 / 12 + h**4 / 80))


def test_truncated_poisson_moments():
    """E[n 1(n > c)] = lam P(n >= c) and E[n^2 1(n > c)] = lam^2 P(n >= c - 1) + lam P(n >= c), for c = 0..4."""
    lam = sp.Symbol("lambda", positive=True)
    for c in range(5):
        e1 = _poisson_sum(lambda k: k, lam, c + 1)
        e2 = _poisson_sum(lambda k: k**2, lam, c + 1)
        assert is_zero(e1 - lam * p_at_least(c, lam))
        assert is_zero(e2 - (lam**2 * p_at_least(c - 1, lam) + lam * p_at_least(c, lam)))


def test_cut_not_linear():
    """The expected recorded value is not linear in the light: cutting a cell's total is not the sum of cutting its
    parts (c = 1)."""
    l1, l2 = sp.symbols("lambda1 lambda2", positive=True)

    def h(x):
        return x * p_at_least(1, x)

    assert not is_zero(h(l1 + l2) - h(l1) - h(l2))


# --- the group rule ---------------------------------------------------------------------------------------------


def _component(kept: set, p: tuple) -> set:
    """p's 8-connected group among the kept pixels (p kept)."""
    seen, todo = {p}, [p]
    while todo:
        q = todo.pop()
        for a, b in N8:
            r = (q[0] + a, q[1] + b)
            if r in kept and r not in seen:
                seen.add(r)
                todo.append(r)
    return seen


def test_group_probabilities():
    """P(|G_p| = k | p kept) by the animal formula, k = 1, 2, 3, and P(|G_p| >= m | p kept) = 1 - sum_{k<m} P_k.

    Both sides are multilinear polynomials in the a_q (the enumeration's weight holds each a_q or 1 - a_q once; an
    animal's S and dS are disjoint). Two multilinear polynomials are equal iff they agree at every 0 / 1 point, where
    the enumeration reduces to the indicator of the kept pattern. So checking every pattern of a 4 x 4 detector, with
    p at (1, 1) and (2, 2) (together every offset of the 5 x 5 window), and p on an edge and a corner, is an exact
    proof. Then the closed forms P1, P2 symbolically on the 5 x 5 window."""
    pixels = [(i, j) for i in range(4) for j in range(4)]
    bit = {q: 1 << n for n, q in enumerate(pixels)}

    def mask(qs):
        return sum(bit[q] for q in qs if q in bit)

    for p in ((1, 1), (2, 2), (0, 2), (0, 0)):
        # each animal at a 0 / 1 point: 1 iff S \ p all kept and its boundary (on the detector) none kept
        terms = {k: [(mask(set(s) - {p}), mask(boundary(s))) for s in animals(p, k) if all(q in bit for q in s)]
                 for k in (1, 2, 3)}  # fmt: skip
        for pattern in range(1 << len(pixels)):
            if not pattern & bit[p]:
                continue
            kept = {q for q in pixels if pattern & bit[q]}
            size = len(_component(kept, p))
            pk = [sum((pattern & i) == i and not pattern & b for i, b in terms[k]) for k in (1, 2, 3)]
            assert pk == [int(size == k) for k in (1, 2, 3)]
            for m in (1, 2, 3, 4):
                assert int(size >= m) == 1 - sum(pk[: m - 1])
    a = {(i, j): sp.Symbol(f"a_{i}_{j}") for i in range(-2, 3) for j in range(-2, 3)}
    p = (0, 0)
    P1 = sp.Mul(*[1 - a[q] for q in N8])
    P2 = 0
    for q in N8:
        near = ({(p[0] + x, p[1] + y) for x, y in N8} | {(q[0] + x, q[1] + y) for x, y in N8}) - {p, q}
        P2 += a[q] * sp.Mul(*[1 - a[r] for r in near])
    assert is_zero(p_group(p, 1, a) - P1)
    assert is_zero(p_group(p, 2, a) - P2)


def test_group_reach():
    """The groups of up to k pixels holding p, with their boundaries, lie within k pixels of p (Chebyshev), and some
    reach exactly k: the rule 'groups of >= m' needs the (2m - 1) x (2m - 1) window around p (5 x 5 for m = 3)."""
    p = (0, 0)
    for k in (1, 2, 3):
        reach = max(max(abs(q[0]), abs(q[1])) for s in animals(p, k) for q in set(s) | boundary(s))
        assert reach == k


def _recorded_moments(a: dict, e: dict, m: int, p: tuple, q: tuple) -> tuple:
    """E[r_p], E[r_q] and E[r_p r_q] (p != q) by enumerating every kept pattern of every pixel. Given the pattern, a
    recorded pixel's mean count is e / a (E[n | n > c]), counts independent."""
    pix = list(a)
    ep = eq = epq = 0
    for bits in itertools.product((0, 1), repeat=len(pix)):
        kept = {x for x, b in zip(pix, bits) if b}
        w = sp.Mul(*[a[x] if b else 1 - a[x] for x, b in zip(pix, bits)])
        rp = int(p in kept and len(_component(kept, p)) >= m)
        rq = int(q in kept and len(_component(kept, q)) >= m)
        ep += w * rp * e[p] / a[p]
        eq += w * rq * e[q] / a[q]
        epq += w * rp * rq * e[p] / a[p] * e[q] / a[q]
    return ep, eq, epq


def test_recorded_covariance():
    """Neighbours are not independent after the group rule (enumerated). An isolated pair, m = 2: Cov(r_p, r_q) =
    e_p e_q (1 - a_p a_q). A row of three, m = 3: Cov(r_1, r_3) = e_1 e_3 a_2 (1 - a_1 a_2 a_3), between pixels two
    apart. m = 1: no covariance. At lam = 2, c = 1 the pair's correlation is 0.82."""
    a = {(0, j): sp.Symbol(f"a{j}", positive=True) for j in range(3)}
    e = {(0, j): sp.Symbol(f"e{j}", positive=True) for j in range(3)}
    a2 = {x: a[x] for x in [(0, 0), (0, 1)]}
    ep, eq, epq = _recorded_moments(a2, e, 2, (0, 0), (0, 1))
    cov = sp.simplify(epq - ep * eq)
    assert is_zero(cov - e[0, 0] * e[0, 1] * (1 - a[0, 0] * a[0, 1]))
    ep, eq, epq = _recorded_moments(a2, e, 1, (0, 0), (0, 1))
    assert is_zero(epq - ep * eq)
    ep, eq, epq = _recorded_moments(a, e, 3, (0, 0), (0, 2))
    assert is_zero(epq - ep * eq - e[0, 0] * e[0, 2] * a[0, 1] * (1 - a[0, 0] * a[0, 1] * a[0, 2]))
    # the pair's correlation at lam = 2, c = 1: a = P(n > 1), e = lam P(n >= 1), E[n^2 1(n > 1)] = lam^2 + lam P(n >= 1)
    lam = sp.Integer(2)
    av, ev, fv = 1 - sp.exp(-lam) * (1 + lam), lam * p_at_least(1, lam), lam**2 + lam * p_at_least(1, lam)
    var = fv * av - (ev * av) ** 2  # Var(r_p) = E[r_p^2] - E[r_p]^2, with P(group >= 2 | p kept) = a_q
    rho = ev * ev * (1 - av * av) / var
    assert rounds_to(rho, "0.82")


def _groups_all(kept: set) -> list:
    """The 8-connected groups of a set of kept pixels."""
    left, out = set(kept), []
    while left:
        g = _component(left, next(iter(left)))
        out.append(g)
        left -= g
    return out


def test_frame_likelihood():
    """The probability that a frame's recorded set is K: prod_K a prod_dK (1 - a) Z_m(U), U the pixels neither in K
    nor next to it, Z_m(U) the probability that every kept group in U has fewer than m pixels. Exact enumeration of
    every kept pattern of a 2 x 3 detector, m = 1, 2, 3, every K."""
    pix = [(i, j) for i in range(2) for j in range(3)]
    a = {q: sp.Symbol(f"a_{q[0]}_{q[1]}") for q in pix}

    def weight(kept, among):
        return sp.Mul(*[a[q] if q in kept else 1 - a[q] for q in among])

    for m in (1, 2, 3):
        brute = {}
        for bits in itertools.product((0, 1), repeat=len(pix)):
            kept = {q for q, b in zip(pix, bits) if b}
            K = frozenset(q for g in _groups_all(kept) if len(g) >= m for q in g)
            brute[K] = brute.get(K, 0) + weight(kept, pix)
        for K, p_brute in brute.items():
            dK = {(q[0] + x, q[1] + y) for q in K for x, y in N8} & set(pix) - K
            U = [q for q in pix if q not in K and q not in dK]
            Z = 0
            for bits in itertools.product((0, 1), repeat=len(U)):
                kept = {q for q, b in zip(U, bits) if b}
                if all(len(g) < m for g in _groups_all(kept)):
                    Z += weight(kept, U)
            assert is_zero(p_brute - sp.Mul(*[a[q] for q in K]) * sp.Mul(*[1 - a[q] for q in dK]) * Z)


def test_segment_is_the_group_rule():
    """anri.io.imaged11.segment records exactly the pixels of rec-recorded: every kept pattern of a 4 x 4 detector
    (each pattern one frame, so frames must not join either), a kept pixel with count c + 1 and the others with c (so
    the cut is tested too), c = 1, for groups of at least m = 1 .. 5 pixels: against the groups of _component."""
    from anri.io.imaged11 import segment

    c, ns, nf = 1, 4, 4
    pix = [(i, j) for i in range(ns) for j in range(nf)]
    patterns = np.arange(1 << len(pix))
    bits = (patterns[:, None] >> np.arange(len(pix))) & 1  # [pattern, pixel]
    frame, pixel = np.nonzero(np.ones_like(bits))
    counts = c + bits[frame, pixel]
    size = np.zeros_like(bits)  # each kept pixel's group size
    for t in patterns:
        for g in _groups_all({pix[k] for k in range(len(pix)) if bits[t, k]}):
            for q in g:
                size[t, q[0] * nf + q[1]] = len(g)
    for m in (1, 2, 3, 4, 5):
        keep = segment(frame, pixel, counts, (ns, nf), c, m)
        np.testing.assert_array_equal(keep, ((bits == 1) & (size >= m))[frame, pixel])


def test_rounded_counts_numbers():
    """The page's numbers on simulated data: a cell of lam = 1.4 is kept at c = 1 with probability 0.41 (rounding
    never keeps it), and a cell below the renderer's min_value 1e-3 with at most lam^2 / 2 = 5e-7."""
    lam = sp.Rational(14, 10)
    assert rounds_to(p_at_least(2, lam), "0.41")  # P(n > 1) = P(n >= 2)
    assert sp.Rational(1, 1000) ** 2 / 2 == sp.Rational(5, 10**7)


def _joint_terms(p: tuple, q: tuple, m: int, det: set) -> list:
    """The terms of rec-joint on a detector (cells det): (kept cells, not-kept cells) of each, every cell other than
    p and q; groups must lie on the detector, boundaries are cut to it."""
    out = []
    for k in range(2, m):
        for s in animals(p, k):
            if q in s and s <= det:
                out.append((set(s) - {p, q}, boundary(s) & det))
    for k1 in range(1, m):
        for s in animals(p, k1):
            if not s <= det or q in s:
                continue
            for k2 in range(1, m):
                for t in animals(q, k2):
                    if t <= det and not s & t and not s & boundary(t) and not t & boundary(s):
                        out.append(((set(s) | set(t)) - {p, q}, (boundary(s) | boundary(t)) & det))
    return out


def _rho(p: tuple, q: tuple, m: int, a: dict) -> sp.Expr:
    """rec-cov-pair's rho_pq = 1 - s_p^(q) - s_q^(p) + J_pq, cells not in a never kept."""

    def s_at(x, y):  # s_x with a_y = 1
        b = {**a, y: sp.Integer(1)}
        return sum(p_group(x, k, b) for k in range(1, m))

    J = sum(sp.Mul(*[a[x] for x in kept]) * sp.Mul(*[1 - a[x] for x in out]) for kept, out in _joint_terms(p, q, m, set(a)))
    return 1 - s_at(p, q) - s_at(q, p) + J


def test_recorded_covariance_any_pair():
    """rec-cov-pair and rec-joint. (1) Symbolically, every pair of a 2 x 3 and of a 1 x 5 detector, m = 1, 2, 3:
    Cov(r_p, r_q) by enumerating every kept pattern equals e_p e_q (rho_pq - w_p w_q); the ends of a row of five,
    m = 3, have Cov = e_1 e_5 a_2 a_3 a_4 (1 - a_3) (correlated at distance 2 (m - 1)). (2) J_pq on every kept pattern
    of a 3 x 4 detector, every pair, m = 2, 3, 4: J at the 0 / 1 point equals the indicator that neither p's nor q's
    group (p, q kept) reaches m. J is multilinear (a term's kept and not-kept cells are disjoint), as is the
    probability, so agreeing on every 0 / 1 point is equality."""
    for shape in ((2, 3), (1, 5)):
        pix = [(i, j) for i in range(shape[0]) for j in range(shape[1])]
        a = {x: sp.Symbol(f"a_{x[0]}_{x[1]}", positive=True) for x in pix}
        e = {x: sp.Symbol(f"e_{x[0]}_{x[1]}", positive=True) for x in pix}
        for p, q in itertools.combinations(pix, 2):
            for m in (1, 2, 3):
                ep, eq, epq = _recorded_moments(a, e, m, p, q)
                w = {x: 1 - sum(p_group(x, k, a) for k in range(1, m)) for x in (p, q)}
                assert is_zero(epq - ep * eq - e[p] * e[q] * (_rho(p, q, m, a) - w[p] * w[q])), (shape, p, q, m)
    p, q = (0, 0), (0, 4)
    ep, eq, epq = _recorded_moments(a, e, 3, p, q)
    assert is_zero(epq - ep * eq - e[p] * e[q] * a[0, 1] * a[0, 2] * a[0, 3] * (1 - a[0, 2]))

    pix = [(i, j) for i in range(3) for j in range(4)]
    det = set(pix)
    for p, q in itertools.combinations(pix, 2):
        others = [x for x in pix if x not in (p, q)]
        for m in (2, 3, 4):
            terms = _joint_terms(p, q, m, det)
            for bits in itertools.product((0, 1), repeat=len(others)):
                kept = {x for x, b in zip(others, bits) if b} | {p, q}
                small = len(_component(kept, p)) < m and len(_component(kept, q)) < m
                J = sum(kp <= kept and not (out & kept) for kp, out in terms)
                assert J == int(small), (p, q, m, bits)


def test_recorded_moments_code():
    """anri.io.imaged11.recorded_moments is rec-expected with rec-trunc's moments and w = 1 - sum_{k<m} P_k from the
    animal sums (rec-animal, p_group): random light on a 3 x 4 detector (every cell's 5 x 5 window gathered by
    neighbourhoods, off-detector cells never kept), c = 0, 1, 2, m = 1, 2, 3, against the expressions evaluated at 30
    digits. Every value is a product and sum of at most ~20 numbers of size <= 1 times the light (<= 6 photons, its
    square for the variance)."""
    from anri.io.imaged11 import neighbourhoods, recorded_moments

    ns, nf = 3, 4
    lam = rng(3).uniform(0.05, 6.0, (ns, nf))
    frame, pixel = np.zeros(ns * nf, int), np.arange(ns * nf)
    nb = neighbourhoods(frame, pixel, lam.ravel(), (ns, nf))
    assert nb.shape == (ns * nf, 5, 5)
    np.testing.assert_array_equal(nb[:, 2, 2], lam.ravel())
    x = sp.Symbol("x", positive=True)
    for c in (0, 1, 2):
        a = {(i, j): p_at_least(c + 1, x).subs(x, sp.Float(lam[i, j], 30)) for i in range(ns) for j in range(nf)}
        for m in (1, 2, 3):
            mean, var = [], []
            for i in range(ns):
                for j in range(nf):
                    L = sp.Float(lam[i, j], 30)
                    e = L * sp.sympify(p_at_least(c, x)).subs(x, L)
                    e2 = L**2 * sp.sympify(p_at_least(c - 1, x)).subs(x, L) + e
                    w = 1 - sum(p_group((i, j), k, a) for k in range(1, m))
                    mean.append(e * w)
                    var.append(e2 * w - (e * w) ** 2)
            agree(evaluate(sp.Matrix(mean), {}).ravel(), lambda v, c=c, m=m: recorded_moments(v, c, m)[0], nb,
                  **tol(6.0, 64))  # fmt: skip
            agree(evaluate(sp.Matrix(var), {}).ravel(), lambda v, c=c, m=m: recorded_moments(v, c, m)[1], nb,
                  **tol(36.0, 64))  # fmt: skip


def _spot_share(n_photons: float, sd: float, offset: float, m: int, wing: float = 0.0) -> float:
    """Expected share of a Gaussian spot's light recorded (c = 1), the spot of sd `sd` px at `offset` px from a pixel
    centre in both directions, with a share `wing` of it widened by a 1.3 px Gaussian: recorded_moments on 15 x 15."""
    from scipy.stats import norm

    from anri.io.imaged11 import neighbourhoods, recorded_moments

    n = 15
    e = np.arange(n + 1) - n / 2

    def masses(s):
        return norm.cdf((e[1:] - offset) / s) - norm.cdf((e[:-1] - offset) / s)

    ws = np.hypot(sd, 1.3)
    img = n_photons * ((1 - wing) * np.outer(masses(sd), masses(sd)) + wing * np.outer(masses(ws), masses(ws)))
    nb = neighbourhoods(np.zeros(n * n, int), np.arange(n * n), img.ravel(), (n, n))
    with float64():
        mean = np.asarray(recorded_moments(nb, 1, m)[0])
    return float(mean.sum() / img.sum())


def test_compact_spot_numbers():
    """recording.rst's table of compact spots: the expected share of a compact spot's light recorded at c = 1, without the group rule
    (m = 1) and with ImageD11's (m = 3), from recorded_moments (proved by test_recorded_moments_code)."""
    table = {
        (0.25, 0.0, 0.0): ("0.928 / 0.003", "0.953 / 0.102", "0.988 / 0.860"),
        (0.25, 0.5, 0.0): ("0.918 / 0.734", "0.999 / 0.999", "1.000 / 1.000"),
        (0.5, 0.0, 0.0): ("0.766 / 0.375", "0.930 / 0.926", "0.987 / 0.987"),
        (0.25, 0.0, 0.28): ("0.733 / 0.022", "0.806 / 0.383", "0.904 / 0.897"),
    }
    for (sd, off, wing), row in table.items():
        for n_ph, quoted in zip((10, 30, 100), row):
            got = [_spot_share(n_ph, sd, off, m, wing) for m in (1, 3)]
            assert quoted == " / ".join(f"{g:.3f}" for g in got), (sd, off, wing, n_ph, got)
