import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri

jax.config.update("jax_enable_x64", True)


class TestLParsToMT(unittest.TestCase):
    def test_cubic(self):
        a = b = c = 3.0
        alpha = beta = gamma = 90.0
        lpars = jnp.array([a, b, c, alpha, beta, gamma])
        expected = jnp.array([[9.0, 0, 0], [0, 9, 0], [0, 0, 9]])
        result = anri.crystal.lpars_to_mt(lpars)
        np.testing.assert_allclose(result, expected, atol=1e-15)

    def test_id11(self):
        a = 2.5
        b = 2.8
        c = 11.3
        alpha = 85.2
        beta = 95.8
        gamma = 101.6
        lpars = jnp.array([a, b, c, alpha, beta, gamma])

        from ImageD11.unitcell import unitcell

        uc = unitcell(lpars, 1)
        expected = uc.g
        result = anri.crystal.lpars_to_mt(lpars)
        np.testing.assert_allclose(result, expected, atol=1e-15)


class TestRMTToB(unittest.TestCase):
    def test_busing_levy(self):
        """Generate many random lattice parameters and ensure we get Busing-Levy back"""
        ntests = 1_000

        import time

        from Dans_Diffraction.functions_lattice import busingandlevy
        from jax import random

        key = random.key(time.time_ns())

        key_l, key_a = random.split(key)
        lengths = random.uniform(key_l, (ntests, 3), minval=1.0, maxval=20.0)
        angles = random.uniform(key_a, (ntests, 3), minval=45.0, maxval=135.0)
        lpars_batch = jnp.concatenate((lengths, angles), axis=1)

        lpars_to_B_vec = jax.vmap(anri.crystal.lpars_to_B)

        results = lpars_to_B_vec(lpars_batch)

        for i in range(ntests):
            expected = busingandlevy(lpars_batch[i])
            np.testing.assert_allclose(results[i], expected, atol=1e-12)


# class TestRMTToRLPars(unittest.TestCase):
#     def test_cubic(self):
#         # a*, b*, c* are just inverted a, b, c
#         # angles are the same
#         a = b = c = 3.0
#         alpha = beta = gamma = 90.0
#         lpars = jnp.array([a, b, c, alpha, beta, gamma])
#         mt = anri.crystal.lpars_to_mt(lpars)
#         rmt = anri.crystal.mt_to_rmt(mt)
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
#         mt = anri.crystal.lpars_to_mt(lpars)
#         rmt = anri.crystal.mt_to_rmt(mt)

#         from ImageD11.unitcell import unitcell

#         uc = unitcell(lpars, 1)
#         expected = jnp.array([uc.astar, uc.bstar, uc.cstar, uc.alphas, uc.betas, uc.gammas])
#         result = anri.crystal.rmt_to_rlpars(rmt)
#         np.testing.assert_allclose(result, expected, atol=1e-15)


class TestRMTToB(unittest.TestCase):
    def test_cubic(self):
        a = b = c = 3.0
        alpha = beta = gamma = 90.0
        lpars = jnp.array([a, b, c, alpha, beta, gamma])
        mt = anri.crystal.lpars_to_mt(lpars)
        rmt = anri.crystal.mt_to_rmt(mt)
        expected = jnp.array([[1 / 3.0, 0, 0], [0, 1 / 3.0, 0], [0, 0, 1 / 3.0]])
        result = anri.crystal.rmt_to_B(rmt)
        np.testing.assert_allclose(result, expected, atol=1e-15)

    def test_id11(self):
        a = 2.5
        b = 2.8
        c = 11.3
        alpha = 85.2
        beta = 95.8
        gamma = 101.6
        lpars = jnp.array([a, b, c, alpha, beta, gamma])
        mt = anri.crystal.lpars_to_mt(lpars)
        rmt = anri.crystal.mt_to_rmt(mt)

        from ImageD11.unitcell import unitcell

        uc = unitcell(lpars, 1)
        expected = uc.B
        result = anri.crystal.rmt_to_B(rmt)
        np.testing.assert_allclose(result, expected, atol=1e-15)

    def test_dd(self):
        a = 2.5
        b = 2.8
        c = 11.3
        alpha = 85.2
        beta = 95.8
        gamma = 101.6
        lpars = jnp.array([a, b, c, alpha, beta, gamma])
        mt = anri.crystal.lpars_to_mt(lpars)
        rmt = anri.crystal.mt_to_rmt(mt)

        from Dans_Diffraction.classes_crystal import Cell

        result = anri.crystal.rmt_to_B(rmt)
        expected = Cell(a, b, c, alpha, beta, gamma).Bmatrix()
        np.testing.assert_allclose(result, expected, atol=1e-15)


class TestBFOA(unittest.TestCase):
    def test_dd(self):
        """Test B matrix -> F matrix -> O matrix -> A matrix (direct basis vectors)"""
        a = 2.5
        b = 2.8
        c = 11.3
        alpha = 85.2
        beta = 95.8
        gamma = 101.6
        lpars = jnp.array([a, b, c, alpha, beta, gamma])
        B = anri.crystal.lpars_to_B(lpars)
        result = anri.crystal.B_to_A(B)

        from Dans_Diffraction.classes_crystal import Cell

        expected = Cell(a, b, c, alpha, beta, gamma).UV()
        np.testing.assert_allclose(result, expected, atol=1e-15)


class TestEverything(unittest.TestCase):
    def test_id11(self):
        # Test against the full ImageD11 unit cell class
        a = 3.4
        b = 2.6
        c = 9.5
        alpha = 83.2
        beta = 95.8
        gamma = 105.4
        lpars = jnp.array([a, b, c, alpha, beta, gamma])
        from ImageD11.unitcell import unitcell

        uc = unitcell(lpars, 1)

        mt = anri.crystal.lpars_to_mt(lpars)
        np.testing.assert_allclose(mt, uc.g, atol=1e-15)

        rmt = anri.crystal.mt_to_rmt(mt)
        np.testing.assert_allclose(rmt, uc.gi, atol=1e-15)

        B = anri.crystal.rmt_to_B(rmt)
        np.testing.assert_allclose(B, uc.B)
