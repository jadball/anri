"""Proofs for docs/source/theory/recording.rst: what the detector records (photon counts, frames, the cut and the
group rule)."""

from __future__ import annotations

import itertools

import sympy as sp

from tests.theory.common import (
    N8,
    animals,
    boundary,
    is_zero,
    p_at_least,
    p_group,
    rounds_to,
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
