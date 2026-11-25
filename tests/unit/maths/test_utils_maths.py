import unittest

import jax
import jax.numpy as jnp
import numpy as np

from anri.maths.utils import bin_fractions, truncated_moments

jax.config.update("jax_enable_x64", True)


class TestTailMass(unittest.TestCase):
    def test_upper_tail_float32(self):
        # Phi(b) - Phi(a) cancels in float32 far in the upper tail; the mass must stay accurate and never negative
        from scipy.stats import norm as sp_norm

        lo = np.array([3.0, 5.0, 6.0, -5.1, -0.1], np.float32)
        hi = lo + np.float32(0.1)
        x, y = lo.astype(float), hi.astype(float)
        # the reference in float64 takes the same branch as the code (upper tail from the survival function), so it
        # does not cancel either; cdf(hi) - cdf(lo) loses ~1 float32 ulp of the mass at lo = 6
        up = lo > 0
        expect = np.where(up, sp_norm.sf(x) - sp_norm.sf(y), sp_norm.cdf(y) - sp_norm.cdf(x))
        operands = np.where(up, sp_norm.sf(x) + sp_norm.sf(y), sp_norm.cdf(y) + sp_norm.cdf(x))
        f = jnp.float32
        for got in (bin_fractions(f(lo), f(hi), f(0.0), f(1.0)), truncated_moments(f(lo), f(hi), f(0.0), f(1.0))[0]):
            self.assertEqual(got.dtype, jnp.float32)
            # float32 inputs, exact in float64, and (x - 0) / 1 exact: the error is ndtr's on each operand, a few
            # float32 ulps of it (XLA's erf / erfc; up to ~6 seen on JAX 0.4.30), then the subtraction's rounding.
            # Relative to the mass that is the condition |operands| / mass: 3-6 in the tails, 25 at the centre
            tol = 8 * np.finfo(np.float32).eps * operands + np.finfo(np.float32).eps * expect
            np.testing.assert_array_less(np.abs(np.asarray(got, float) - expect), tol)
            self.assertTrue(np.all(np.asarray(got) >= 0))
