import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from anri import diffract

jax.config.update("jax_enable_x64", True)


class TestScaleNormK(unittest.TestCase):
    def test_scale_norm_k(self):
        k_vec = jnp.array([2.0, 0.0, 0.0])
        wavelength = 0.1
        expected = jnp.array([10.0, 0.0, 0.0])
        result = diffract.scatter.scale_norm_k(k_vec, wavelength)
        np.testing.assert_allclose(result, expected)


class TestKToQLab(unittest.TestCase):
    def setUp(self):
        self.k_in = jnp.array([1.0, 0.0, 0.0])
        self.k_out = jnp.array([0.0, 1.0, 0.0])
        self.q = jnp.array([-1.0, 1.0, 0.0])

    def test_k_to_q_lab(self):
        expected = self.q
        result = diffract.scatter.k_to_q_lab(self.k_in, self.k_out)
        np.testing.assert_allclose(result, expected)

    def test_q_lab_to_k_out(self):
        expected = self.k_out
        result = diffract.scatter.q_lab_to_k_out(self.q, self.k_in)
        np.testing.assert_allclose(result, expected)


class TestPeakLabToKOut(unittest.TestCase):
    def test_peak_lab_to_k_out(self):
        peak_lab = jnp.array([1.0, 0.0, 1.0])
        origin_lab = jnp.array([0.0, 0.0, 1.0])
        # k_out should be 1,0,0
        wavelength = 0.5
        expected = (1 / wavelength) * jnp.array([1.0, 0.0, 0.0])
        result = diffract.scatter.peak_lab_to_k_out(peak_lab, origin_lab, wavelength)
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
        self.q_lab_to_tth_eta_vec = vmap(diffract.scatter.q_lab_to_tth_eta, in_axes=(0, None))
        self.tth_eta_to_k_out_vec = vmap(diffract.scatter.tth_eta_to_k_out, in_axes=(0, 0, None))
        self.k_to_q_lab_vec = vmap(diffract.scatter.k_to_q_lab, in_axes=(0, 0))

    def test_round_trip_tth_eta(self):
        # round trip tth, eta -> k_out -> q_lab -> tth, eta
        k_out_vec = self.tth_eta_to_k_out_vec(self.tth, self.eta, self.wavelength)
        q_lab_vec = self.k_to_q_lab_vec(jnp.repeat(self.k_in[None, :], k_out_vec.shape[0], axis=0), k_out_vec)
        result_tth, result_eta = self.q_lab_to_tth_eta_vec(q_lab_vec, self.wavelength)

        np.testing.assert_allclose(result_tth, self.tth)
        np.testing.assert_allclose(result_eta, self.eta)


class TestOmegaSolnsBoth(unittest.TestCase):
    def test_matches_single_solutions(self):
        rng = np.random.default_rng(0)
        k_in = diffract.scatter.scale_norm_k(jnp.array([1.0, 0.0, 0.0]), 0.3)
        both = vmap(diffract.scatter.omega_solns_both, in_axes=(0, None))
        single = vmap(diffract.scatter.omega_solns, in_axes=(0, None, None))
        q = jnp.asarray(rng.normal(size=(200, 3)))
        omega1, omega2, valid = both(q, k_in)
        for omega, etasign in ((omega1, 1.0), (omega2, -1.0)):
            expected, valid_single = single(q, etasign, k_in)
            np.testing.assert_array_equal(valid, valid_single)
            np.testing.assert_allclose(omega[valid], expected[valid])
        self.assertTrue(valid.any() and not valid.all())


class TestAnyBeam(unittest.TestCase):
    def test_round_trip_tilted_beam(self):
        rng = np.random.default_rng(0)
        k_in = jnp.array([1.0, 0.2, -0.1])
        tth, eta = jnp.asarray(rng.uniform(1, 60, 20)), jnp.asarray(rng.uniform(-179, 179, 20))
        k_out = vmap(diffract.scatter.tth_eta_to_k_out, in_axes=(0, 0, None, None))(tth, eta, 0.3, k_in)
        t2, e2 = diffract.scatter.k_out_to_tth_eta(k_out, k_in)
        np.testing.assert_allclose(t2, tth, atol=1e-6)
        np.testing.assert_allclose(e2, eta, atol=1e-6)
        # eta of Q = k_out - k_in is eta of k_out, for any beam
        q = k_out - k_in / jnp.linalg.norm(k_in) / 0.3
        _, e3 = vmap(diffract.scatter.q_lab_to_tth_eta, in_axes=(0, None, None))(q, 0.3, k_in)
        np.testing.assert_allclose(e3, eta, atol=1e-6)
