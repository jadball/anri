import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.maths.rotation
from anri.maths.rotation import mat_to_quat, mat_to_rod, quat_mul, quat_to_mat, rod_to_mat

jax.config.update("jax_enable_x64", True)


def random_rotations(n, seed=0):
    return anri.maths.rotation.random_rotations(n, np.random.default_rng(seed))


class TestRmatFromAxisAngle(unittest.TestCase):
    def test_integer_axis(self):
        # an integer axis must not truncate a fractional angle (issue #34)
        np.testing.assert_allclose(anri.maths.rotation.rmat_from_axis_angle(jnp.array([0, 0, 1]), 30.5),
                                   anri.maths.rotation.rot_z(30.5), atol=1e-12)  # fmt: skip
        np.testing.assert_allclose(anri.maths.rotation.rmat_from_axis_angle(np.array([1, 0, 0]), np.array(0.5)),
                                   anri.maths.rotation.rot_x(0.5), atol=1e-12)  # fmt: skip


class TestBatching(unittest.TestCase):
    def test_batched_axis_angle(self):
        """A batch of axes and angles in one call gives each one's rotation (values: test_axis_angle_code)."""
        rng = np.random.default_rng(1)
        ax, ang = rng.normal(size=(5, 3)), rng.uniform(-180, 180, 5)
        R = anri.maths.rotation.rmat_from_axis_angle(jnp.asarray(ax), jnp.asarray(ang))
        self.assertEqual(R.shape, (5, 3, 3))
        eps = np.finfo(np.float64).eps  # the same formula, vectorised: XLA may order its roundings differently
        for i in range(5):
            one = anri.maths.rotation.rmat_from_axis_angle(jnp.asarray(ax[i]), ang[i])
            np.testing.assert_allclose(R[i], one, rtol=0, atol=16 * eps)


class TestConversions(unittest.TestCase):
    def test_round_trips(self):
        U = random_rotations(500)
        np.testing.assert_allclose(quat_to_mat(mat_to_quat(U)), U, atol=1e-12)
        small = U[np.trace(U, axis1=1, axis2=2) > -0.5]  # rotation angles well below 180 degrees
        np.testing.assert_allclose(np.asarray(rod_to_mat(mat_to_rod(small))), small, atol=1e-5)

    def test_quat_mul_is_matrix_product(self):
        a, b = random_rotations(50, 1), random_rotations(50, 2)
        np.testing.assert_allclose(quat_to_mat(quat_mul(mat_to_quat(a), mat_to_quat(b))), a @ b, atol=1e-12)


class TestRotationHelpers(unittest.TestCase):
    def test_rotation_angle_small_and_large(self):
        # arccos of the trace loses small angles in float32; the angle must not
        for deg in (1e-4, 1e-2, 1.0, 90.0, 179.0):
            R = jnp.asarray(anri.maths.rotation.rmat_from_axis_angle(jnp.array([0.3, -0.5, 0.8]), deg), jnp.float32)
            self.assertAlmostEqual(float(np.degrees(anri.maths.rotation.rotation_angle(R))) / deg, 1.0, delta=1e-3)

    def test_mean_rotation_no_weight(self):
        """All weights zero: the identity, as documented (values: tests/theory/test_populations.py)."""
        U = random_rotations(5)
        np.testing.assert_array_equal(anri.maths.rotation.mean_rotation(U, np.zeros(5)), np.eye(3))


class TestRotations(unittest.TestCase):
    def test_small_rotations(self):
        R = anri.maths.rotation.small_rotations(20000, 0.5, np.random.default_rng(0))
        angle = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        self.assertAlmostEqual(np.sqrt(np.mean(angle**2)), 0.5 * np.sqrt(3), delta=0.02)  # rms of a 3D normal

    def test_random_rotations(self):
        R = anri.maths.rotation.random_rotations(1000, np.random.default_rng(1))
        np.testing.assert_allclose(R @ np.swapaxes(R, 1, 2), np.broadcast_to(np.eye(3), R.shape), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
