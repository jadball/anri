import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri

jax.config.update("jax_enable_x64", True)


# class TestChiMat(unittest.TestCase):
#     def test_identity(self):
#         chi = 0.0

#         expected = jnp.eye(3)
#         result = anri.geom.chimat(chi)

#         np.testing.assert_allclose(result, expected)

#     def test_id11(self):
#         chi = -10.3

#         from ImageD11.gv_general import chimat

#         expected = chimat(chi)
#         result = anri.geom.chimat(chi)

#         np.testing.assert_allclose(result, expected)


# class TestWedgeMat(unittest.TestCase):
#     def test_identity(self):
#         wedge = 0.0

#         expected = jnp.eye(3)
#         result = anri.geom.wedgemat(wedge)

#         np.testing.assert_allclose(result, expected)

#     def test_id11(self):
#         wedge = -4.6

#         from ImageD11.gv_general import wedgemat

#         expected = wedgemat(wedge)
#         result = anri.geom.wedgemat(wedge)

#         np.testing.assert_allclose(result, expected)


class TestSampleToLab(unittest.TestCase):
    def test_identity(self):
        # extremely simple test case
        omega = 0.0
        wedge = 0.0
        chi = 0.0
        dty = 0.0
        y0 = 0.0

        v_sample = jnp.array([1.0, 2.0, 3.0])
        result = anri.geom.sample_to_lab(v_sample, omega=omega, wedge=wedge, chi=chi, dty=dty, y0=y0)

        # just identity so v_lab = v_sample
        np.testing.assert_allclose(result, v_sample)

    def test_just_dty(self):
        # no rotations
        omega = 0.0
        wedge = 0.0
        chi = 0.0
        dty = 10.0
        y0 = 0.5

        v_sample = jnp.array([1.0, 2.0, 3.0])
        result = anri.geom.sample_to_lab(v_sample, omega=omega, wedge=wedge, chi=chi, dty=dty, y0=y0)
        desired = jnp.array([1.0, 2.0 + 10.0 - 0.5, 3.0])
        np.testing.assert_allclose(result, desired)

    def test_omega_90(self):
        omega = 90.0
        wedge = 0.0
        chi = 0.0
        dty = 0.0
        y0 = 0.0

        # sample x goes to lab y
        v_sample = jnp.array([1.0, 0.0, 0.0])
        result = anri.geom.sample_to_lab(v_sample, omega=omega, wedge=wedge, chi=chi, dty=dty, y0=y0)
        desired = jnp.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(result, desired, atol=1e-15)

    def test_chi(self):
        omega = 0.0
        wedge = 0.0
        chi = 90.0
        dty = 0.0
        y0 = 0.0

        # lean fully rightwards when looking down beam
        # sample z is along lab -y
        v_sample = jnp.array([0.0, 0.0, 1.0])
        result = anri.geom.sample_to_lab(v_sample, omega=omega, wedge=wedge, chi=chi, dty=dty, y0=y0)
        desired = jnp.array([0.0, -1.0, 0.0])
        np.testing.assert_allclose(result, desired, atol=1e-15)

    def test_wedge(self):
        omega = 0.0
        wedge = -90.0
        chi = 0.0
        dty = 0.0
        y0 = 0.0

        # as angle increases, diffractometer falls towards detector
        # sample z ends up along lab x
        # in ImageD11, the sign of the wedge is flipped.
        v_sample = jnp.array([0.0, 0.0, 1.0])
        result = anri.geom.sample_to_lab(v_sample, omega=omega, wedge=wedge, chi=chi, dty=dty, y0=y0)
        desired = jnp.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, desired, atol=1e-15)

    def test_id11(self):
        # test some more complicated vectors.
        import time

        from jax import random

        key = random.key(time.time_ns())
        npks = 1000

        vec_sample = random.uniform(key, shape=(npks, 3), minval=-1000.0, maxval=1000.0)
        # omega = random.uniform(key, shape=(npks,), minval=-180.0, maxval=180.0)
        # wedge = random.uniform(key, shape=(npks,), minval=-180.0, maxval=180.0)
        # chi = random.uniform(key, shape=(npks,), minval=-180.0, maxval=180.0)
        key = random.key(time.time_ns())
        omega_arr = random.uniform(key, shape=(npks,), minval=-180.0, maxval=180.0)
        # omega = random.uniform(key, shape=(1,), minval=-180.0, maxval=180.0)[0]
        key = random.key(time.time_ns())
        wedge = random.uniform(key, shape=(1,), minval=-180.0, maxval=180.0)[0]
        key = random.key(time.time_ns())
        chi = random.uniform(key, shape=(1,), minval=-180.0, maxval=180.0)[0]
        wedge_arr = jnp.repeat(wedge, npks)
        chi_arr = jnp.repeat(chi, npks)

        dty = jnp.repeat(0.0, npks)
        y0 = jnp.repeat(0.0, npks)

        from ImageD11.transform import compute_grain_origins

        sample_to_lab_vec = jax.vmap(anri.geom.sample_to_lab)
        result = sample_to_lab_vec(vec_sample, omega_arr, wedge_arr, chi_arr, dty, y0)
        desired = compute_grain_origins(
            omega=omega_arr, wedge=wedge, chi=chi, t_x=vec_sample[:, 0], t_y=vec_sample[:, 1], t_z=vec_sample[:, 2]
        ).T
        np.testing.assert_allclose(result, desired)
