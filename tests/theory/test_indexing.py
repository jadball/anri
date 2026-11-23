"""Proofs for docs/source/theory/indexing.rst: the bilinear sharing of a prediction over (eta, omega) bins."""

import numpy as np
import sympy as sp

from tests.theory.common import (
    EPS,
    agree,
    evaluate,
    is_zero,
)


def test_bilinear_keeps_weight_and_position():
    """idx-bilinear: with x = (a - a_0) / b - 1/2 and t = x - floor(x), the weights 1 - t, t of bins floor(x) and
    floor(x) + 1 (centres a_0 + (i + 1/2) b) sum to 1 and their weighted centre is a: the shared prediction keeps its
    weight and its position."""
    a, a0, b, i0 = sp.symbols("a a_0 b i_0", real=True)
    t = (a - a0) / b - sp.Rational(1, 2) - i0
    centre = [a0 + (i0 + k + sp.Rational(1, 2)) * b for k in (0, 1)]
    assert is_zero((1 - t) + t - 1)
    assert is_zero((1 - t) * centre[0] + t * centre[1] - a)


def test_bilinear_code():
    """anri's eta_omega_corners are idx-bilinear's bins and weights."""
    from anri.index.data import eta_omega_corners

    a, a0, b = sp.symbols("a a_0 b", real=True)
    x = (a - a0) / b - sp.Rational(1, 2)
    eta, om = np.array([37.3, -151.85]), np.array([12.6, 301.2])
    b_e, b_o, n_e, n_o, om0 = 1.0, 0.5, 360, 720, 0.0
    te = [evaluate(x - sp.floor(x), {a: e, a0: -180, b: b_e}) for e in eta]
    to = [evaluate(x - sp.floor(x), {a: o, a0: om0, b: b_o}) for o in om]
    want = []
    for ke in (0, 1):
        for ko in (0, 1):
            want.append([[t if ke else 1 - t for t in te], [t if ko else 1 - t for t in to]])
    # x is ((a - a_0) / b - 1/2): ~3 roundings of numbers up to 720, so t is within 4 ulps of 720, absolutely
    tol = 4 * EPS * 720

    def weights(*z: object) -> np.ndarray:  # [4 corners, eta / omega, predictions]
        return np.array([[np.asarray(c[2]), np.asarray(c[3])] for c in eta_omega_corners(*z)])

    agree(np.array(want, float), weights, eta, om, om0, b_e, b_o, n_e, n_o, rtol=0, atol=tol)
