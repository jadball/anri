import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri

jax.config.update("jax_enable_x64", True)


class TestUnitCell(unittest.TestCase):
    def test_id11(self):
        ntests = 100

        from Dans_Diffraction.functions_lattice import random_lattice
        from ImageD11.unitcell import unitcell as unitcell_id11

        lpars_list = [random_lattice(symmetry="triclinic") for _ in range(ntests)]
        lpars_batch = jnp.array(lpars_list)

        self.assertTupleEqual(lpars_batch.shape, (ntests, 6))
        self.assertTrue(~jnp.any(jnp.isnan(lpars_batch)))

        for i in range(ntests):
            uc_id11 = unitcell_id11(lpars_batch[i], "P")
            uc_anri = anri.crystal.UnitCell.from_lpars(lpars_batch[i])
            np.testing.assert_allclose(uc_id11.lattice_parameters, uc_anri.lattice_parameters)
            np.testing.assert_allclose(uc_id11.B, uc_anri.B)
            np.testing.assert_allclose(uc_id11.g, uc_anri.mt)
            np.testing.assert_allclose(uc_id11.gi, uc_anri.rmt)
            np.testing.assert_allclose(
                jnp.array([uc_id11.astar, uc_id11.bstar, uc_id11.cstar, uc_id11.alphas, uc_id11.betas, uc_id11.gammas]),
                uc_anri.reciprocal_lattice_parameters,
            )
