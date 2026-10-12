import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.geom

jax.config.update("jax_enable_x64", True)


class TestRotZ(unittest.TestCase):
    def test_against_fable_geom_doc(self):
        # In FABLE, the Omega matrix is defined as:
        # Omega = [cos(w) -sin(w)       0]
        #         [sin(w)  cos(w)       0]
        #         [     0       0       1]
        # they say it is applied as:
        # vec_lab = Omega . vec(sample)
        # We have the same understanding about the rot_x, rot_y, rot_z functions.

        omega = 12.345
        romega = jnp.radians(omega)
        expected = jnp.array([[jnp.cos(romega), -jnp.sin(romega), 0], [jnp.sin(romega), jnp.cos(romega), 0], [0, 0, 1]])
        result = anri.geom.rot_z(omega)
        np.testing.assert_allclose(result, expected)


class TestRmatFromAxisAngle(unittest.TestCase):
    def test_axes(self):
        # the axis need not be normalised
        for axis, rot in (
            ([2.0, 0, 0], anri.geom.rot_x),
            ([0, 0.5, 0], anri.geom.rot_y),
            ([0, 0, 3.0], anri.geom.rot_z),
        ):
            for angle in (-30.0, 10.0, 135.0):
                np.testing.assert_allclose(
                    anri.geom.rmat_from_axis_angle(jnp.array(axis), angle), rot(angle), atol=1e-12
                )


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


class TestSkewAndRodrigues(unittest.TestCase):
    def test_skew(self):
        rng = np.random.default_rng(0)
        v, u = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
        K = anri.geom.skew(jnp.asarray(v))
        np.testing.assert_allclose(np.einsum("nij,nj->ni", K, u), np.cross(v, u), atol=1e-12)
        np.testing.assert_allclose(K @ K, v[:, :, None] * v[:, None, :] - (v * v).sum(1)[:, None, None] * np.eye(3),
                                   atol=1e-12)  # fmt: skip

    def test_batched_rodrigues(self):
        R = anri.geom.rmat_from_axis_angle(jnp.array([[0.0, 0.0, 2.0], [1.0, 1.0, 1.0]]), jnp.array([90.0, 120.0]))
        np.testing.assert_allclose(R[0] @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)  # right-handed, z
        np.testing.assert_allclose(R[1] @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)  # 3-fold, <111>
        from jax.scipy.spatial.transform import Rotation

        rng = np.random.default_rng(1)
        ax, ang = rng.normal(size=(50, 3)), rng.uniform(-180, 180, 50)
        n = ax / np.linalg.norm(ax, axis=1, keepdims=True)
        expected = Rotation.from_rotvec(jnp.asarray(n * ang[:, None]), degrees=True).as_matrix()
        np.testing.assert_allclose(anri.geom.rmat_from_axis_angle(jnp.asarray(ax), jnp.asarray(ang)), expected,
                                   atol=1e-12)  # fmt: skip
