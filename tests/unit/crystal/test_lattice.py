import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.crystal
import anri.maths.rotation

jax.config.update("jax_enable_x64", True)


class TestLParsToMT(unittest.TestCase):
    def test_cubic(self):
        a = b = c = 3.0
        alpha = beta = gamma = 90.0
        lpars = jnp.array([a, b, c, alpha, beta, gamma])
        expected = jnp.array([[9.0, 0, 0], [0, 9, 0], [0, 0, 9]])
        result = anri.crystal.lattice.lpars_to_mt(lpars)
        np.testing.assert_allclose(result, expected, atol=1e-15)


# class TestRMTToRLPars(unittest.TestCase):
#     def test_cubic(self):
#         # a*, b*, c* are just inverted a, b, c
#         # angles are the same
#         a = b = c = 3.0
#         alpha = beta = gamma = 90.0
#         lpars = jnp.array([a, b, c, alpha, beta, gamma])
#         mt = anri.crystal.lattice.lpars_to_mt(lpars)
#         rmt = anri.crystal.lattice.mt_to_rmt(mt)
#         result = anri.crystal.rmt_to_rlpars(rmt)
#         expected = jnp.array([1 / 3.0, 1 / 3.0, 1 / 3.0, 90.0, 90.0, 90.0])
#         np.testing.assert_allclose(result, expected, atol=1e-15)

#     def test_id11(self):
#         a = 2.5
#         b = 2.8
#         c = 11.3
#         alpha = 85.2
#         beta = 95.8
#         gamma = 101.6
#         lpars = jnp.array([a, b, c, alpha, beta, gamma])
#         mt = anri.crystal.lattice.lpars_to_mt(lpars)
#         rmt = anri.crystal.lattice.mt_to_rmt(mt)

#         from ImageD11.unitcell import unitcell

#         uc = unitcell(lpars, 1)
#         expected = jnp.array([uc.astar, uc.bstar, uc.cstar, uc.alphas, uc.betas, uc.gammas])
#         result = anri.crystal.rmt_to_rlpars(rmt)
#         np.testing.assert_allclose(result, expected, atol=1e-15)


class TestBuildBMat(unittest.TestCase):
    def test_cubic(self):
        a = b = c = 3.0
        alpha = beta = gamma = 90.0
        lpars = jnp.array([a, b, c, alpha, beta, gamma])
        mt = anri.crystal.lattice.lpars_to_mt(lpars)
        rmt = anri.crystal.lattice.mt_to_rmt(mt)
        rlpars = anri.crystal.lattice.mt_to_lpars(rmt)
        expected = jnp.array([[1 / 3.0, 0, 0], [0, 1 / 3.0, 0], [0, 0, 1 / 3.0]])
        result = anri.crystal.lattice.lpars_rlpars_to_B(lpars, rlpars)
        np.testing.assert_allclose(result, expected, atol=1e-15)


class TestUBToUBI(unittest.TestCase):
    def test_against_inverse(self):
        q = np.random.default_rng(0).normal(size=(5, 4))
        U = np.asarray(anri.maths.rotation.quat_to_mat(q / np.linalg.norm(q, axis=1, keepdims=True)))
        B = np.asarray(anri.crystal.lattice.lpars_to_B(jnp.array([3.0, 4.0, 5.0, 85.0, 95.0, 100.0])))
        np.testing.assert_allclose(anri.crystal.lattice.U_B_to_UBI(U, B), np.linalg.inv(U @ B), rtol=0, atol=1e-13)
        self.assertTrue(np.all(np.isnan(anri.crystal.lattice.U_B_to_UBI(np.full((3, 3), np.nan), B))))
