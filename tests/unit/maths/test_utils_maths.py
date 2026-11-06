import unittest

import jax
import jax.numpy as jnp
import numpy as np

from anri.maths.utils import bin_fractions, exponential_slices, fwhm_to_sigma, inv3, sigma_to_fwhm, truncated_moments

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


class TestExponentialSlices(unittest.TestCase):
    def test_moments_and_equal_probability(self):
        """The slices' mixture has the truncated exponential's mean and variance; each slice holds 1 / n of it."""
        for scale, cut, n in ((1.0, 0.7, 4), (0.3, 5.0, 4), (2.0, 3.0, 7)):
            x = (np.arange(400000) + 0.5) * cut / 400000  # midpoints: plain sums (no trapz / trapezoid across NumPys)
            p = np.exp(-x / scale)
            p /= p.sum()
            mean = (x * p).sum()
            var = (x**2 * p).sum() - mean**2
            m, v = (np.asarray(a) for a in exponential_slices(scale, cut, n))
            self.assertAlmostEqual(m.mean(), mean, places=6)
            self.assertAlmostEqual(np.mean(v + m**2) - m.mean() ** 2, var, places=6)
            # equal probability: the cdf at each slice's upper edge is (k + 1) / n, so the medians lie inside
            cdf = np.cumsum(p) / p.sum()
            for k in range(n):
                q = np.interp((k + 0.5) / n, cdf, x)
                self.assertTrue(m[k] - 3 * np.sqrt(v[k]) - 1e-9 <= q <= m[k] + 3 * np.sqrt(v[k]) + 1e-9)
            self.assertTrue(np.all(np.diff(m) > 0))
