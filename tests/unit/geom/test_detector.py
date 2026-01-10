import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

import anri

jax.config.update("jax_enable_x64", True)


class TestDetectorOrientationMatrix(unittest.TestCase):
    def test_identity(self):
        o11 = 1.0
        o12 = 0.0
        o21 = 0.0
        o22 = 1.0
        expected = jnp.eye(3)
        result = anri.geom.detector_orientation_matrix(o11, o12, o21, o22)
        np.testing.assert_allclose(expected, result)

    def test_simple(self):
        o11 = 0.0
        o12 = 1.0
        o21 = 0.0
        o22 = 1.0
        expected = jnp.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]])
        result = anri.geom.detector_orientation_matrix(o11, o12, o21, o22)
        np.testing.assert_allclose(expected, result)


class TestDetectorRotationMatrix(unittest.TestCase):
    def test_identity(self):
        tilt_x = 0.0
        tilt_y = 0.0
        tilt_z = 0.0
        expected = jnp.eye(3)
        result = anri.geom.detector_rotation_matrix(tilt_x, tilt_y, tilt_z)
        np.testing.assert_allclose(expected, result)

    def test_id11(self):
        tilt_x = 2.5
        tilt_y = -0.5
        tilt_z = -1.4
        from ImageD11.transform import detector_rotation_matrix as detector_rotation_matrix_id11

        expected = detector_rotation_matrix_id11(tilt_x, tilt_y, tilt_z)
        result = anri.geom.detector_rotation_matrix(tilt_x, tilt_y, tilt_z)
        np.testing.assert_allclose(expected, result)


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
        result_det_trans, result_beam_cen_shift, result_x_distance_shift = anri.geom.detector_transforms(
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
        np.testing.assert_allclose(expected_beam_cen_shift, result_beam_cen_shift)
        np.testing.assert_allclose(expected_x_distance_shift, result_x_distance_shift)
        np.testing.assert_allclose(expected_det_trans, result_det_trans)


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

        det_trans, beam_cen_shift, x_distance_shift = anri.geom.detector_transforms(
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
        result = anri.geom.det_to_lab(sc, fc, det_trans, beam_cen_shift, x_distance_shift)
        np.testing.assert_allclose(expected, result)

    def test_id11(self):
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

        det_trans, beam_cen_shift, x_distance_shift = anri.geom.detector_transforms(
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

        from ImageD11.transform import compute_xyz_lab

        peaks = jnp.vstack((fc, sc))
        assert peaks.shape == (2, npks), peaks.shape
        expected = compute_xyz_lab(
            peaks,
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
        ).T
        det_to_lab_vec = jax.vmap(anri.geom.det_to_lab, in_axes=[0, 0, None, None, None])
        result = det_to_lab_vec(sc, fc, det_trans, beam_cen_shift, x_distance_shift)
        np.testing.assert_allclose(expected, result)

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

        det_trans, beam_cen_shift, x_distance_shift = anri.geom.detector_transforms(
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

        det_to_lab_vec = jax.vmap(anri.geom.det_to_lab, in_axes=[0, 0, None, None, None])
        lab_to_det_vec = jax.vmap(anri.geom.lab_to_det, in_axes=[0, 0, 0, None, None, None])

        vec_lab = det_to_lab_vec(sc, fc, det_trans, beam_cen_shift, x_distance_shift)
        vec_det = lab_to_det_vec(
            vec_lab[:, 0], vec_lab[:, 1], vec_lab[:, 2], det_trans, beam_cen_shift, x_distance_shift
        )
        np.testing.assert_allclose(vec_det[:, 0], fc)
        np.testing.assert_allclose(vec_det[:, 1], sc)


class TestRaytraceToDet(unittest.TestCase):
    def setUp(self):
        # perfom the ImageD11 raytrace

        # rng
        import time

        from jax import random

        key = random.key(time.time_ns())

        npks = 100

        tth = random.uniform(key, shape=(npks,), minval=0.0, maxval=30.0)
        eta = random.uniform(key, shape=(npks,), minval=-180.0, maxval=180.0)
        omega = random.uniform(key, shape=(npks,), minval=-180.0, maxval=180.0)
        self.wavelength = 0.1

        vec_lab = vmap(anri.diffract.tth_eta_to_k_out, in_axes=[0, 0, None])(tth, eta, self.wavelength)

        # set up some ImageD11 parameters
        from ImageD11.parameters import AnalysisSchema

        pars = AnalysisSchema.from_default().geometry_pars_obj

        # fiddle with the detector positions
        pars.set("tilt_x", 0.00123)
        pars.set("tilt_y", -0.0345)
        pars.set("tilt_z", 0.02)
        # fiddle with diffractometer tilts
        pars.set("chi", 1)
        pars.set("wedge", -3)
        pars_dict = pars.get_parameters()
        del pars_dict["t_x"]
        del pars_dict["t_y"]
        del pars_dict["t_z"]

        from ImageD11.transform import compute_xyz_from_tth_eta

        fc_id11, sc_id11 = compute_xyz_from_tth_eta(tth, eta, omega, **pars.get_parameters())

        self.pars = pars
        # self.origin_sample = origin_sample
        self.omega = omega
        self.vec_lab = vec_lab
        self.sc_id11 = sc_id11
        self.fc_id11 = fc_id11

    def test_raytrace_to_det_id11(self):
        # set up detector transforms

        pars_for_det = self.pars.get_parameters().copy()
        del pars_for_det["chi"]
        del pars_for_det["wedge"]
        del pars_for_det["omegasign"]
        del pars_for_det["fit_tolerance"]
        del pars_for_det["min_bin_prob"]
        del pars_for_det["weight_hist_intensities"]
        del pars_for_det["no_bins"]
        del pars_for_det["wavelength"]
        det_trans, beam_cen_shift, x_distance_shift = anri.geom.detector_transforms(**pars_for_det)
        sc_lab, fc_lab, norm_lab = anri.geom.detector_basis_vectors_lab(det_trans, beam_cen_shift, x_distance_shift)

        # vectorize our function
        raytrace_to_det_vec = vmap(anri.geom.raytrace_to_det, in_axes=(0, None, None, None, None))
        sc_anri, fc_anri = raytrace_to_det_vec(
            self.vec_lab,
            jnp.array([0.0, 0.0, 0.0]),  # origin_lab,
            sc_lab,
            fc_lab,
            norm_lab,
        )

        np.testing.assert_allclose(sc_anri, self.sc_id11)
        np.testing.assert_allclose(fc_anri, self.fc_id11)
