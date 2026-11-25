"""Anri agrees with Dans_Diffraction on the basics: B matrices.

Not proofs (tests/theory) nor software tests (tests/unit): a comparison with older, trusted software.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.crystal

jax.config.update("jax_enable_x64", True)


class TestBuildBMat(unittest.TestCase):
    def test_dd(self):
        a = 2.5
        b = 2.8
        c = 11.3
        alpha = 85.2
        beta = 95.8
        gamma = 101.6
        lpars = jnp.array([a, b, c, alpha, beta, gamma])
        mt = anri.crystal.lattice.lpars_to_mt(lpars)
        rmt = anri.crystal.lattice.mt_to_rmt(mt)
        rlpars = anri.crystal.lattice.mt_to_lpars(rmt)

        from Dans_Diffraction.classes_crystal import Cell

        result = anri.crystal.lattice.lpars_rlpars_to_B(lpars, rlpars)
        expected = Cell(a, b, c, alpha, beta, gamma).Bmatrix()
        # the same float64 formula: 16 ulps of the largest entry
        np.testing.assert_allclose(result, expected, rtol=0, atol=16 * np.finfo(float).eps * np.abs(expected).max())


if __name__ == "__main__":
    unittest.main()
