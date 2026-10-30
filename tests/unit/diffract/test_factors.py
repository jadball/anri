import unittest

import jax.numpy as jnp
import numpy as np

from anri.diffract import factors

# Becker and Coppens (1974), Tables 1 (primary), 3 (secondary, Gaussian) and 4 (secondary, Lorentzian): y at
# x = 1, 3, 10 for sin(theta) = 0.05 and 0.5
TABLES = {
    "primary": {1: (0.5185, 0.5343), 3: (0.2952, 0.3204), 10: (0.1536, 0.1727)},
    "gaussian": {1: (0.4660, 0.4837), 3: (0.2215, 0.2489), 10: (0.0802, 0.1011)},
    "lorentzian": {1: (0.5297, 0.5450), 3: (0.3237, 0.3454), 10: (0.1742, 0.1927)},
}


class TestExtinction(unittest.TestCase):
    def test_tables(self):
        """The analytic fits reproduce the numerical tables they were fitted to, within 5% at these angles."""
        for kind, rows in TABLES.items():
            for x, ref in rows.items():
                for s, r in zip((0.05, 0.5), ref):
                    tth = 2 * np.arcsin(s)
                    if kind == "primary":
                        y = float(factors.extinction(x, 0.0, tth, True))
                    else:
                        y = float(factors.extinction(0.0, x, tth, kind == "gaussian"))
                    self.assertAlmostEqual(y / r, 1.0, delta=0.05, msg=f"{kind} x {x} sin(theta) {s}")

    def test_perfect_crystal_limit(self):
        """For large x, y ~ c / sqrt(x): the intensity grows as |F|, not |F|^2 (Becker and Coppens, section II.2)."""
        y1, y2 = (float(factors.extinction(x, 0.0, 0.3, True)) for x in (1e3, 4e3))
        self.assertAlmostEqual(y1 * np.sqrt(1e3) / (y2 * np.sqrt(4e3)), 1.0, delta=0.01)

    def test_no_extinction(self):
        """With x = X = 0, y = 1 and the polarisation factor is unchanged, for any polarisation and direction."""
        self.assertEqual(float(factors.extinction(0.0, 0.0, 0.7, False)), 1.0)
        k_in = jnp.array([1.0, 0.0, 0.0])
        for tth, eta in ((0.1, 0.2), (0.6, 1.9), (1.2, -2.5)):
            k_out = jnp.array([np.cos(tth), -np.sin(tth) * np.sin(eta), np.sin(tth) * np.cos(eta)])
            for p in (0.0, 0.9):
                a = float(factors.polarisation(k_in, k_out, p))
                b = float(factors.polarisation_extinction(k_in, k_out, p, 0.0, 0.0, True))
                self.assertAlmostEqual(a, b, places=6)

    def test_extinction_lowers_the_polarisation_factor(self):
        k_in = jnp.array([1.0, 0.0, 0.0])
        k_out = jnp.array([np.cos(0.3), 0.0, np.sin(0.3)])
        p = float(factors.polarisation(k_in, k_out, 0.9))
        q = float(factors.polarisation_extinction(k_in, k_out, 0.9, 0.5, 0.5, True))
        self.assertLess(q, p)

    def test_parameters(self):
        """x = (3 / 2) (r_e |F| r lambda / V)^2 (Becker and Coppens, 35a with 5); a single aligned perfect domain
        (R = r, g infinite) gives X = x; absorption lowers X, not x."""
        F2, V, lam, tth, r = np.array([4600.0]), 160.2, 0.19, np.array([0.1]), 2.0
        x, X = factors.extinction_parameters(F2, V, lam, tth, r, r)
        expected = 1.5 * (factors._R_E * np.sqrt(F2) * r * 1e4 * lam / V) ** 2
        np.testing.assert_allclose(np.asarray(x), expected, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(X), np.asarray(x), rtol=1e-3)
        x_mu, X_mu = factors.extinction_parameters(F2, V, lam, tth, r, r, mu=0.5)
        np.testing.assert_allclose(np.asarray(x_mu), np.asarray(x), rtol=1e-6)
        self.assertLess(float(X_mu[0]), float(X[0]))


if __name__ == "__main__":
    unittest.main()
