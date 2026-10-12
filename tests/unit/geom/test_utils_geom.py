import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.geom
import anri.maths.rotation

jax.config.update("jax_enable_x64", True)


class TestBeamBasis(unittest.TestCase):
    def test_along_x(self):
        for got, expected in zip(anri.geom.beam_basis(jnp.array([2.0, 0.0, 0.0])), np.eye(3)):
            np.testing.assert_allclose(got, expected, atol=1e-15)

    def test_orthonormal(self):
        rng = np.random.default_rng(0)
        for k in rng.normal(size=(20, 3)):
            k_hat, e_h, e_v = anri.geom.beam_basis(jnp.asarray(k))
            basis = np.stack([k_hat, e_h, e_v])
            np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(np.linalg.det(basis), 1.0, atol=1e-12)  # right-handed, like (x, y, z)
            np.testing.assert_allclose(k_hat, k / np.linalg.norm(k), atol=1e-12)
            self.assertAlmostEqual(float(e_h[2]), 0.0)  # horizontal


class TestBeamOffsets(unittest.TestCase):
    def test_beam_along_x(self):
        across, height, rise, cos_alpha = anri.geom.beam_offsets(
            jnp.array([5.0, -2.0, 3.0]), jnp.array([2.0, 0.0, 0.0])
        )
        np.testing.assert_allclose([across, height, rise, cos_alpha], [-2.0, 3.0, 0.0, 1.0], atol=1e-15)

    def test_tilted_beam(self):
        k = jnp.array([1.0, 0.3, 0.1])
        p = jnp.array([5.0, -2.0, 3.0])
        across, height, rise, cos_alpha = anri.geom.beam_offsets(p, k)
        k_hat, e_h, _ = anri.geom.beam_basis(k)
        self.assertAlmostEqual(float(across), float(p @ e_h), places=12)
        self.assertAlmostEqual(float(height + rise), float(p[2]), places=12)
        self.assertAlmostEqual(float(cos_alpha), float(jnp.hypot(k_hat[0], k_hat[1])), places=12)
        # a point on the beam's centre line is at zero height and zero distance across
        on_line = anri.geom.beam_offsets(7.0 * k_hat, k)
        np.testing.assert_allclose([on_line[0], on_line[1]], 0.0, atol=1e-12)
