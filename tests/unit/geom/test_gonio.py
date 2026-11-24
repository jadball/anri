import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.geom

jax.config.update("jax_enable_x64", True)


class TestOmegaWrap(unittest.TestCase):
    def test_wrap_and_nearest(self):
        om = np.array([-725.0, -10.0, 0.0, 179.9, 359.99, 360.0, 541.0])
        w = anri.geom.gonio.wrap_omega(om, -180.0)
        self.assertTrue(np.all((w >= -180.0) & (w < 180.0)))
        np.testing.assert_allclose(np.mod(w - om, 360.0) % 360.0, 0.0, atol=1e-9)
        n = anri.geom.gonio.nearest_omega(om, 90.0)
        self.assertTrue(np.all(np.abs(n - 90.0) <= 180.0))
        np.testing.assert_allclose(np.mod(n - om, 360.0) % 360.0, 0.0, atol=1e-9)
        # NumPy in, NumPy out; JAX in, JAX out
        self.assertIsInstance(w, np.ndarray)
        self.assertIsInstance(anri.geom.gonio.wrap_omega(jnp.asarray(om), 0.0), jax.Array)
