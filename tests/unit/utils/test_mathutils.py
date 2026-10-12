import unittest

import jax
import numpy as np

from anri.utils import fwhm_to_sigma, inv3, sigma_to_fwhm

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
