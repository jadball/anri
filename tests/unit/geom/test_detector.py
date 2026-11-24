import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.diffract
import anri.geom

jax.config.update("jax_enable_x64", True)


class TestDetectorOrientationMatrix(unittest.TestCase):
    def test_identity(self):
        o11 = 1.0
        o12 = 0.0
        o21 = 0.0
        o22 = 1.0
        expected = jnp.eye(3)
        result = anri.geom.detector.detector_orientation_matrix(o11, o12, o21, o22)
        np.testing.assert_allclose(result, expected)

    def test_simple(self):
        o11 = 0.0
        o12 = 1.0
        o21 = 0.0
        o22 = 1.0
        expected = jnp.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]])
        result = anri.geom.detector.detector_orientation_matrix(o11, o12, o21, o22)
        np.testing.assert_allclose(result, expected)


class TestDetectorRotationMatrix(unittest.TestCase):
    def test_identity(self):
        tilt_x = 0.0
        tilt_y = 0.0
        tilt_z = 0.0
        expected = jnp.eye(3)
        result = anri.geom.detector.detector_rotation_matrix(tilt_x, tilt_y, tilt_z)
        np.testing.assert_allclose(result, expected)


class TestDetectorTransforms(unittest.TestCase):
    def test_simple(self):
        y_center = 1000.0
        z_center = 1100.0
        tilt_x = 0.0
        tilt_y = 0.0
        tilt_z = 0.0
        y_size = 0.5
        z_size = 0.6
        distance = 500.0
        o11 = 1.0
        o12 = 0.0
        o21 = 0.0
        o22 = 1.0
        expected_x_distance_shift = jnp.array([500.0, 0.0, 0.0])
        expected_beam_cen_shift = jnp.array([-1100.0, -1000.0, 0.0])
        expected_pixel_size_scale = jnp.array([[0.6, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])
        expected_det_flips = jnp.eye(3)
        expected_cob_matrix = jnp.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])  # hard-coded anyway
        expected_det_tilts = jnp.eye(3)
        expected_det_trans = expected_det_tilts @ expected_cob_matrix @ expected_det_flips @ expected_pixel_size_scale
        result_det_trans, result_beam_cen_shift, result_x_distance_shift = anri.geom.detector.detector_transforms(
            y_center=y_center,
            z_center=z_center,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            tilt_z=tilt_z,
            y_size=y_size,
            z_size=z_size,
            distance=distance,
            o11=o11,
            o12=o12,
            o21=o21,
            o22=o22,
        )
        np.testing.assert_allclose(result_beam_cen_shift, expected_beam_cen_shift)
        np.testing.assert_allclose(result_x_distance_shift, expected_x_distance_shift)
        np.testing.assert_allclose(result_det_trans, expected_det_trans)


class TestDetToLab(unittest.TestCase):
    def test_simple(self):
        y_center = 1000.0
        z_center = 1000.0
        tilt_x = 0.0
        tilt_y = 0.0
        tilt_z = 0.0
        y_size = 0.5
        z_size = 0.5
        distance = 500.0
        o11 = 1.0
        o12 = 0.0
        o21 = 0.0
        o22 = 1.0
        sc = 1100.0
        fc = 900.0

        det_trans, beam_cen_shift, x_distance_shift = anri.geom.detector.detector_transforms(
            y_center=y_center,
            z_center=z_center,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            tilt_z=tilt_z,
            y_size=y_size,
            z_size=z_size,
            distance=distance,
            o11=o11,
            o12=o12,
            o21=o21,
            o22=o22,
        )

        expected = jnp.array([500.0, -50.0, 50.0])
        result = anri.geom.detector.det_to_lab(sc, fc, det_trans, beam_cen_shift, x_distance_shift)
        np.testing.assert_allclose(result, expected)

    def test_round_trip(self):
        import time

        from jax import random

        key = random.key(time.time_ns())
        npks = 1000
        det_size = 2048.0
        sc = random.uniform(key, shape=(npks,), minval=0.0, maxval=det_size)
        fc = random.uniform(key, shape=(npks,), minval=0.0, maxval=det_size)

        y_center = 1000.0
        z_center = 900.0
        tilt_x = 0.2
        tilt_y = -0.36
        tilt_z = 0.17
        y_size = 50
        z_size = 65
        distance = 130e3
        o11 = 1.0
        o12 = 0.0
        o21 = 0.0
        o22 = 1.0

        det_trans, beam_cen_shift, x_distance_shift = anri.geom.detector.detector_transforms(
            y_center=y_center,
            z_center=z_center,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            tilt_z=tilt_z,
            y_size=y_size,
            z_size=z_size,
            distance=distance,
            o11=o11,
            o12=o12,
            o21=o21,
            o22=o22,
        )

        det_to_lab_vec = jax.vmap(anri.geom.detector.det_to_lab, in_axes=[0, 0, None, None, None])
        lab_to_det_vec = jax.vmap(anri.geom.detector.lab_to_det, in_axes=[0, 0, 0, None, None, None])

        vec_lab = det_to_lab_vec(sc, fc, det_trans, beam_cen_shift, x_distance_shift)
        vec_det = lab_to_det_vec(
            vec_lab[:, 0], vec_lab[:, 1], vec_lab[:, 2], det_trans, beam_cen_shift, x_distance_shift
        )
        np.testing.assert_allclose(vec_det[:, 0], fc)
        np.testing.assert_allclose(vec_det[:, 1], sc)


class TestRaytraceFloat32(unittest.TestCase):
    def test_float32_matches_float64(self):
        """The whole detector chain must not lose precision in float32 (the GPU default).

        Pixel positions are ~distance from the origin, so building pixel steps by subtracting positions
        loses ~2e-4 of a 75 um pixel at 150 mm in float32: several tenths of a pixel on the detector.
        """
        pars = {
            "y_center": 1049.9, "y_size": 75.0, "tilt_y": -2e-3, "z_center": 1116.5, "z_size": 75.0, "tilt_z": 3e-3,
            "tilt_x": 1e-3, "distance": 150e3, "o11": -1.0, "o12": 0.0, "o21": 0.0, "o22": -1.0,
        }  # fmt: skip
        rng = np.random.default_rng(0)
        # outgoing rays up to ~37 degrees two-theta, from origins within 50 um of the rotation axis
        tth = np.radians(rng.uniform(3, 37, 2000))
        eta = rng.uniform(-np.pi, np.pi, 2000)
        k_out = np.stack([np.cos(tth), np.sin(tth) * np.sin(eta), np.sin(tth) * np.cos(eta)], 1)
        origin = rng.uniform(-50, 50, (2000, 3)) * np.array([1.0, 0.0, 1.0])
        out = {}
        for dtype in (np.float32, np.float64):
            # under x64 the transforms promote to float64; cast them, as a GPU run would have them in float32
            transforms = [jnp.asarray(t, dtype) for t in anri.geom.detector.detector_transforms(*pars.values())]
            basis = anri.geom.detector.detector_basis_vectors_lab(*transforms)
            trace = jax.vmap(anri.geom.detector.raytrace_to_det, in_axes=(0, 0, None, None, None))
            sc, fc = trace(jnp.asarray(k_out, dtype), jnp.asarray(origin, dtype), *basis)
            self.assertEqual(sc.dtype, dtype)
            out[dtype] = np.stack([sc, fc], 1).astype(np.float64)
        np.testing.assert_allclose(out[np.float32], out[np.float64], atol=0.01)
