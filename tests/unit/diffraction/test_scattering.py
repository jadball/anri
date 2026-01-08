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
        result = anri.diffraction.scale_norm_k(k_vec, wavelength)
        np.testing.assert_allclose(result, expected)


class TestKToQLab(unittest.TestCase):
    def setUp(self):
        self.k_in = jnp.array([1.0, 0.0, 0.0])
        self.k_out = jnp.array([0.0, 1.0, 0.0])
        self.q = jnp.array([-1.0, 1.0, 0.0])

    def test_k_to_q_lab(self):
        expected = self.q
        result = anri.diffraction.k_to_q_lab(self.k_in, self.k_out)
        np.testing.assert_allclose(result, expected)

    def test_q_lab_to_k_out(self):
        expected = self.k_out
        result = anri.diffraction.q_lab_to_k_out(self.q, self.k_in)
        np.testing.assert_allclose(result, expected)


class TestPeakLabToKOut(unittest.TestCase):
    def test_peak_lab_to_k_out(self):
        peak_lab = jnp.array([1.0, 0.0, 1.0])
        origin_lab = jnp.array([0.0, 0.0, 1.0])
        # k_out should be 1,0,0
        wavelength = 0.5
        expected = (1 / wavelength) * jnp.array([1.0, 0.0, 0.0])
        result = anri.diffraction.peak_lab_to_k_out(peak_lab, origin_lab, wavelength)
        np.testing.assert_allclose(result, expected)


class TestQLabToTthEta(unittest.TestCase):
    def setUp(self):
        from ImageD11.transform import compute_k_vectors

        self.wavelength = 0.1
        self.k_in = jnp.array([1.0 / self.wavelength, 0.0, 0.0])
        self.tth = np.random.random(100) * 30.0  # 0 to 30 degrees
        self.eta = np.random.random(size=100) * 360.0 - 180.0  # -180 to 180 degrees
        # compute q_lab from tth, eta with ImageD11:
        self.q_lab_id11 = compute_k_vectors(self.tth, self.eta, self.wavelength).T
        # vectorize our functions for testing
        self.q_lab_to_tth_eta_vec = vmap(anri.diffraction.q_lab_to_tth_eta, in_axes=(0, None))
        self.tth_eta_to_k_out_vec = vmap(anri.diffraction.tth_eta_to_k_out, in_axes=(0, 0, None))
        self.k_to_q_lab_vec = vmap(anri.diffraction.k_to_q_lab, in_axes=(0, 0))

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

        self.omega_solns_vec = vmap(anri.diffraction.omega_solns, in_axes=(1, None, None))

    def test_omega_solns_id11_nochi_nowedge(self):
        chi = 0.0
        wedge = 0.0
        # the rotation axis is always defined as +Z in the sample frame
        rot_axis_sample = jnp.array([0.0, 0.0, 1.0])

        # convert k_in to sample frame
        # just identity here
        k_in_sample = anri.geometry.lab_to_sample(self.k_in_lab, omega=0.0, wedge=wedge, chi=chi)

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
        post = self.wedgechi_id11(wedge, chi)

        # convert k_in to sample frame
        # this is handled by the post matrix in ImageD11
        k_in_sample = anri.geometry.lab_to_sample(self.k_in_lab, omega=0.0, wedge=wedge, chi=chi)

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
