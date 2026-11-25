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
        expect = sp_norm.cdf(hi.astype(float)) - sp_norm.cdf(lo.astype(float))
        f = jnp.float32
        for got in (bin_fractions(f(lo), f(hi), f(0.0), f(1.0)), truncated_moments(f(lo), f(hi), f(0.0), f(1.0))[0]):
            self.assertEqual(got.dtype, jnp.float32)
            # float32 inputs (exact in float64); the mass from the tail's complementary function rounds by a few
            # float32 ulps relative, wherever it is (the point of the test: no cancellation)
            np.testing.assert_allclose(np.asarray(got, float), expect, rtol=16 * np.finfo(np.float32).eps)
            self.assertTrue(np.all(np.asarray(got) >= 0))
