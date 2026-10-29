import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.maths.rotation
from anri.maths.rotation import mat_to_quat, mat_to_rod, quat_mul, quat_to_mat, rod_to_mat

jax.config.update("jax_enable_x64", True)


def random_rotations(n, seed=0):
    return anri.maths.rotation.random_rotations(n, np.random.default_rng(seed))


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
        result = anri.maths.rotation.rot_z(omega)
        np.testing.assert_allclose(result, expected)


class TestRmatFromAxisAngle(unittest.TestCase):
    def test_axes(self):
        # the axis need not be normalised
        for axis, rot in (
            ([2.0, 0, 0], anri.maths.rotation.rot_x),
            ([0, 0.5, 0], anri.maths.rotation.rot_y),
            ([0, 0, 3.0], anri.maths.rotation.rot_z),
        ):
            for angle in (-30.0, 10.0, 135.0):
                np.testing.assert_allclose(
                    anri.maths.rotation.rmat_from_axis_angle(jnp.array(axis), angle), rot(angle), atol=1e-12
                )

    def test_integer_axis(self):
        # an integer axis must not truncate a fractional angle (issue #34)
        np.testing.assert_allclose(anri.maths.rotation.rmat_from_axis_angle(jnp.array([0, 0, 1]), 30.5),
                                   anri.maths.rotation.rot_z(30.5), atol=1e-12)  # fmt: skip
        np.testing.assert_allclose(anri.maths.rotation.rmat_from_axis_angle(np.array([1, 0, 0]), np.array(0.5)),
                                   anri.maths.rotation.rot_x(0.5), atol=1e-12)  # fmt: skip


class TestSkewAndRodrigues(unittest.TestCase):
    def test_skew(self):
        rng = np.random.default_rng(0)
        v, u = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
        K = anri.maths.rotation.skew(jnp.asarray(v))
        np.testing.assert_allclose(np.einsum("nij,nj->ni", K, u), np.cross(v, u), atol=1e-12)
        np.testing.assert_allclose(K @ K, v[:, :, None] * v[:, None, :] - (v * v).sum(1)[:, None, None] * np.eye(3),
                                   atol=1e-12)  # fmt: skip

    def test_batched_rodrigues(self):
        R = anri.maths.rotation.rmat_from_axis_angle(
            jnp.array([[0.0, 0.0, 2.0], [1.0, 1.0, 1.0]]), jnp.array([90.0, 120.0])
        )
        np.testing.assert_allclose(R[0] @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)  # right-handed, z
        np.testing.assert_allclose(R[1] @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)  # 3-fold, <111>
        from jax.scipy.spatial.transform import Rotation

        rng = np.random.default_rng(1)
        ax, ang = rng.normal(size=(50, 3)), rng.uniform(-180, 180, 50)
        n = ax / np.linalg.norm(ax, axis=1, keepdims=True)
        expected = Rotation.from_rotvec(jnp.asarray(n * ang[:, None]), degrees=True).as_matrix()
        np.testing.assert_allclose(anri.maths.rotation.rmat_from_axis_angle(jnp.asarray(ax), jnp.asarray(ang)), expected,
                                   atol=1e-12)  # fmt: skip


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

    def test_mean_rotation_is_proper(self):
        U = random_rotations(5)
        M = np.asarray(anri.maths.rotation.mean_rotation(U, np.ones(5)))
        self.assertAlmostEqual(float(np.linalg.det(M)), 1.0, places=6)
        np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(anri.maths.rotation.mean_rotation(U[:1], np.ones(1)), U[0], atol=1e-6)
        np.testing.assert_allclose(anri.maths.rotation.mean_rotation(U, np.zeros(5)), np.eye(3), atol=1e-12)


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
