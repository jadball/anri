import unittest

import jax
import jax.numpy as jnp
import numpy as np

from anri.maths.utils import bin_fractions, fwhm_to_sigma, inv3, sigma_to_fwhm, truncated_moments

jax.config.update("jax_enable_x64", True)


class TestInv3(unittest.TestCase):
    def test_matches_numpy(self):
        rng = np.random.default_rng(0)
        m = rng.normal(size=(1000, 3, 3))
        np.testing.assert_allclose(inv3(m), np.linalg.inv(m), rtol=1e-8, atol=1e-10)

    def test_ubi(self):
        # a typical UBI: inverse of U.B for a strained, rotated cell
        ubi = np.array([[2.1, 0.3, -0.4], [-0.2, 2.9, 0.1], [0.5, 0.05, 3.4]])
        np.testing.assert_allclose(inv3(ubi) @ ubi, np.eye(3), atol=1e-13)


class TestFwhm(unittest.TestCase):
    def test_gaussian(self):
        # the Gaussian of that sigma is at half its maximum at +- FWHM / 2
        sigma = fwhm_to_sigma(3.0)
        self.assertAlmostEqual(float(np.exp(-0.5 * (1.5 / sigma) ** 2)), 0.5, places=12)
        np.testing.assert_allclose(sigma_to_fwhm(fwhm_to_sigma(np.array([0.5, 2.0]))), [0.5, 2.0], rtol=1e-15)


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
            np.testing.assert_allclose(np.asarray(got, float), expect, rtol=1e-3)
            self.assertTrue(np.all(np.asarray(got) >= 0))
