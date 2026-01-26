import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

import anri

jax.config.update("jax_enable_x64", True)


class TestScaleNormK(unittest.TestCase):
    def test_scale_norm_k(self):
        k_vec = jnp.array([2.0, 0.0, 0.0])
        wavelength = 0.1
        expected = jnp.array([10.0, 0.0, 0.0])
        result = anri.diffract.scale_norm_k(k_vec, wavelength)
        np.testing.assert_allclose(result, expected)


class TestKToQLab(unittest.TestCase):
    def setUp(self):
        self.k_in = jnp.array([1.0, 0.0, 0.0])
        self.k_out = jnp.array([0.0, 1.0, 0.0])
        self.q = jnp.array([-1.0, 1.0, 0.0])

    def test_k_to_q_lab(self):
        expected = self.q
        result = anri.diffract.k_to_q_lab(self.k_in, self.k_out)
        np.testing.assert_allclose(result, expected)

    def test_q_lab_to_k_out(self):
        expected = self.k_out
        result = anri.diffract.q_lab_to_k_out(self.q, self.k_in)
        np.testing.assert_allclose(result, expected)


class TestPeakLabToKOut(unittest.TestCase):
    def test_peak_lab_to_k_out(self):
        peak_lab = jnp.array([1.0, 0.0, 1.0])
        origin_lab = jnp.array([0.0, 0.0, 1.0])
        # k_out should be 1,0,0
        wavelength = 0.5
        expected = (1 / wavelength) * jnp.array([1.0, 0.0, 0.0])
        result = anri.diffract.peak_lab_to_k_out(peak_lab, origin_lab, wavelength)
        np.testing.assert_allclose(result, expected)


class TestQLabToTthEta(unittest.TestCase):
    def setUp(self):
        from ImageD11.transform import compute_k_vectors

        self.wavelength = 0.1
        self.k_in = jnp.array([1.0 / self.wavelength, 0.0, 0.0])
        npks = 100
        self.tth = np.random.random(npks) * 30.0  # 0 to 30 degrees
        self.eta = np.random.random(size=npks) * 360.0 - 180.0  # -180 to 180 degrees
        # compute q_lab from tth, eta with ImageD11:
        self.q_lab_id11 = compute_k_vectors(self.tth, self.eta, self.wavelength).T
        # vectorize our functions for testing
        self.q_lab_to_tth_eta_vec = vmap(anri.diffract.q_lab_to_tth_eta, in_axes=(0, None))
        self.tth_eta_to_k_out_vec = vmap(anri.diffract.tth_eta_to_k_out, in_axes=(0, 0, None))
        self.k_to_q_lab_vec = vmap(anri.diffract.k_to_q_lab, in_axes=(0, 0))

    def test_q_lab_to_tth_eta_id11(self):
        # when we convert our ImageD11 q_lab back to tth, eta we should recover the original values
        expected_tth = self.tth
        expected_eta = self.eta

        result_tth, result_eta = self.q_lab_to_tth_eta_vec(self.q_lab_id11, self.wavelength)
        np.testing.assert_allclose(result_tth, expected_tth)
        np.testing.assert_allclose(result_eta, expected_eta)

    def test_tth_eta_to_k_out_id11(self):
        # when we convert tth, eta to k_out and then to q_lab we should recover the original q_lab

        k_out_vec = self.tth_eta_to_k_out_vec(self.tth, self.eta, self.wavelength)
        q_lab_vec = self.k_to_q_lab_vec(jnp.repeat(self.k_in[None, :], k_out_vec.shape[0], axis=0), k_out_vec)

        np.testing.assert_allclose(q_lab_vec, self.q_lab_id11)

    def test_round_trip_tth_eta(self):
        # round trip tth, eta -> k_out -> q_lab -> tth, eta
        k_out_vec = self.tth_eta_to_k_out_vec(self.tth, self.eta, self.wavelength)
        q_lab_vec = self.k_to_q_lab_vec(jnp.repeat(self.k_in[None, :], k_out_vec.shape[0], axis=0), k_out_vec)
        result_tth, result_eta = self.q_lab_to_tth_eta_vec(q_lab_vec, self.wavelength)

        np.testing.assert_allclose(result_tth, self.tth)
        np.testing.assert_allclose(result_eta, self.eta)


class TestOmegaSolns(unittest.TestCase):
    def setUp(self):
        # generate some test UBI, gives us q_sample
        # TODO: replace with our own B matrix generator
        import time

        from ImageD11.gv_general import g_to_k, wedgechi
        from ImageD11.unitcell import unitcell
        from jax.scipy.spatial.transform import Rotation as jR

        self.g_to_k_id11 = g_to_k
        self.wedgechi_id11 = wedgechi

        from jax import random

        key = random.key(time.time_ns())

        # numerical stuff
        a, b, c = 3.0, 4.0, 5.0
        alpha, beta, gamma = 90.0, 90.0, 90.0
        self.wavelength = 0.1
        dsmax = 3.0

        self.k_in_lab = jnp.array([1.0 / self.wavelength, 0.0, 0.0])

        # prepare UB matrix
        uc = unitcell([a, b, c, alpha, beta, gamma], 225)
        uc.makerings(dsmax)
        B = uc.B
        random_euler = random.uniform(key, shape=(3,), minval=0.0, maxval=2 * jnp.pi)
        R = jR.from_euler("xyz", random_euler)
        U = R.as_matrix()
        UB = U @ B

        hkls = jnp.concatenate([jnp.array(hkl) for hkl in uc.ringhkls.values()])

        self.q_sample = UB @ hkls.T

        self.omega_solns_vec = vmap(anri.diffract.omega_solns, in_axes=(1, None, None))

    def test_omega_solns_id11_nochi_nowedge(self):
        chi = 0.0
        wedge = 0.0
        dty = 0.0
        y0 = 0.0
        # the rotation axis is always defined as +Z in the sample frame
        rot_axis_sample = jnp.array([0.0, 0.0, 1.0])

        # convert k_in to sample frame
        # just identity here
        k_in_sample = anri.geom.lab_to_sample(self.k_in_lab, omega=0.0, wedge=wedge, chi=chi, dty=dty, y0=y0)

        # we have to supply negative the axis to match ImageD11 convention here
        # this is because ImageD11 forces a negative k_in, whereas we can supply one
        omega1_id11, omega2_id11, valid_id11 = self.g_to_k_id11(
            self.q_sample, self.wavelength, rot_axis_sample * -1, pre=None, post=None
        )

        omega1_anri, valid1_anri = self.omega_solns_vec(self.q_sample, 1, k_in_sample)
        omega2_anri, valid2_anri = self.omega_solns_vec(self.q_sample, -1, k_in_sample)

        np.testing.assert_allclose(omega1_anri, omega1_id11)
        np.testing.assert_allclose(omega2_anri, omega2_id11)
        np.testing.assert_array_equal(valid1_anri, valid_id11)
        np.testing.assert_array_equal(valid2_anri, valid_id11)

    def test_omega_solns_id11_chi_wedge(self):
        chi = 10.0
        wedge = -5.0
        dty = 0.0
        y0 = 0.0
        post = self.wedgechi_id11(wedge, chi)

        # convert k_in to sample frame
        # this is handled by the post matrix in ImageD11
        k_in_sample = anri.geom.lab_to_sample(self.k_in_lab, omega=0.0, wedge=wedge, chi=chi, dty=dty, y0=y0)

        # the rotation axis is always defined as +Z in the sample frame
        rot_axis_sample = jnp.array([0.0, 0.0, 1.0])

        omega1_id11, omega2_id11, valid_id11 = self.g_to_k_id11(
            self.q_sample, self.wavelength, rot_axis_sample * -1, pre=None, post=post
        )

        omega1_anri, valid1_anri = self.omega_solns_vec(self.q_sample, 1, k_in_sample)
        omega2_anri, valid2_anri = self.omega_solns_vec(self.q_sample, -1, k_in_sample)

        np.testing.assert_allclose(omega1_anri, omega1_id11)
        np.testing.assert_allclose(omega2_anri, omega2_id11)
        np.testing.assert_array_equal(valid1_anri, valid_id11)
        np.testing.assert_array_equal(valid2_anri, valid_id11)


class TestDetToQSample(unittest.TestCase):
    def test_id11(self):
        import time

        import ImageD11.transform
        from ImageD11.columnfile import columnfile
        from ImageD11.parameters import AnalysisSchema
        from jax import random

        # get some basic ImageD11 geometry
        pars = AnalysisSchema.from_default().geometry_pars_obj
        # make it slightly spicier
        pars.set("tilt_x", 0.4)
        pars.set("tilt_y", -0.345)
        pars.set("tilt_z", 0.2)
        pars.set("chi", 10)
        pars.set("wedge", -17)

        # RNG
        key = random.key(time.time_ns())
        key, *subkeys = random.split(key, 4)

        # detector peaks
        npks = 100_000
        fc = random.uniform(subkeys[0], shape=(npks,), minval=0.0, maxval=2048.0)
        sc = random.uniform(subkeys[1], shape=(npks,), minval=0.0, maxval=2048.0)
        om = random.uniform(subkeys[2], shape=(npks,), minval=-180.0, maxval=180.0)

        # bung em in a columnfile
        cf = columnfile(new=True)
        cf.nrows = npks
        cf.addcolumn(fc, "fc")
        cf.addcolumn(sc, "sc")
        cf.addcolumn(om, "omega")

        # compute geometry with ImageD11
        cf.parameters = pars
        cf.updateGeometry()

        # incident wavevector, normalised
        k_in = anri.diffract.scale_norm_k(np.array([1.0, 0, 0]), pars.get("wavelength"))

        det_trans, beam_cen_shift, x_distance_shift = anri.geom.detector_transforms(
            pars.get("y_center"),
            pars.get("y_size"),
            pars.get("tilt_y"),
            pars.get("z_center"),
            pars.get("z_size"),
            pars.get("tilt_z"),
            pars.get("tilt_x"),
            pars.get("distance"),
            pars.get("o11"),
            pars.get("o12"),
            pars.get("o21"),
            pars.get("o22"),
        )

        # detector peak positions in lab frame
        det_to_lab_vec = vmap(
            anri.geom.det_to_lab,
            in_axes=(0, 0, None, None, None),
        )
        v_lab_me = det_to_lab_vec(cf.sc, cf.fc, det_trans, beam_cen_shift, x_distance_shift)
        v_lab_id11 = jnp.column_stack([cf.xl, cf.yl, cf.zl])
        np.testing.assert_allclose(v_lab_me, v_lab_id11)

        # q_lab - just subtract diffraction origins then normalise and scale by 1/wavelength
        peak_lab_to_k_out_vec = vmap(
            anri.diffract.peak_lab_to_k_out,
            in_axes=(0, None, None),
        )
        k_out_me = peak_lab_to_k_out_vec(v_lab_me, jnp.array([0.0, 0.0, 0.0]), pars.get("wavelength"))
        k_to_q_lab_vec = vmap(
            anri.diffract.k_to_q_lab,
            in_axes=(None, 0),  # constant k_in
        )
        q_lab_me = k_to_q_lab_vec(k_in, k_out_me)
        q_lab_id11 = ImageD11.transform.compute_k_vectors(cf.tth, cf.eta, pars.get("wavelength")).T
        np.testing.assert_allclose(q_lab_me, q_lab_id11)

        # q_sample - rotate q_lab to sample frame by applying omega, wedge, chi
        lab_to_sample_vec = vmap(
            anri.geom.lab_to_sample,
            in_axes=(0, 0, None, None, None, None),
        )
        q_sample_me = lab_to_sample_vec(
            q_lab_me,
            cf.omega,
            pars.get("wedge"),
            pars.get("chi"),
            0.0,
            0.0,
        )
        q_sample_id11 = jnp.column_stack([cf.gx, cf.gy, cf.gz])
        np.testing.assert_allclose(q_sample_me, q_sample_id11)
