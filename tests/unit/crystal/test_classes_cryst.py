import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.crystal

jax.config.update("jax_enable_x64", True)


def cellvolume(latticepar):
    """
    ImageD11.forward_model.forward_projector
    calculate unit cell volume [angstrom^3]
    """
    a = latticepar[0]
    b = latticepar[1]
    c = latticepar[2]
    calp = np.cos(np.deg2rad(latticepar[3]))
    cbet = np.cos(np.deg2rad(latticepar[4]))
    cgam = np.cos(np.deg2rad(latticepar[5]))
    
    angular = np.sqrt(1 - calp**2 - cbet**2 - cgam**2 + 2*calp*cbet*cgam)
    
    Vcell = np.abs(a*b*c*angular)
    
    return Vcell


class TestUnitCell(unittest.TestCase):
    def test_id11(self):
        ntests = 100

        from Dans_Diffraction.functions_lattice import random_lattice
        from ImageD11.unitcell import unitcell as unitcell_id11
        # from ImageD11.forward_model.forward_projector import cellvolume

        symmetries = ['cubic', 'tetragonal', 'hexagonal', 'rhobohedral', 'monoclinic-a', 'monoclinic-b', 'monoclinic-c', 'triclinic']
        
        lpars_list = [random_lattice(symmetry=np.random.choice(symmetries)) for _ in range(ntests)]
        lpars_batch = jnp.array(lpars_list)

        self.assertTupleEqual(lpars_batch.shape, (ntests, 6))
        self.assertTrue(~jnp.any(jnp.isnan(lpars_batch)))

        for i in range(ntests):
            uc_id11 = unitcell_id11(lpars_batch[i], "P")
            uc_anri = anri.crystal.UnitCell.from_lpars(lpars_batch[i])
            np.testing.assert_allclose(uc_id11.lattice_parameters, uc_anri.lattice_parameters)
            np.testing.assert_allclose(uc_id11.B, uc_anri.B, rtol=1e-7, atol=1e-10)
            np.testing.assert_allclose(uc_id11.g, uc_anri.mt, rtol=1e-7, atol=1e-10)
            np.testing.assert_allclose(uc_id11.gi, uc_anri.rmt, rtol=1e-7, atol=1e-10)
            np.testing.assert_allclose(
                jnp.array([uc_id11.astar, uc_id11.bstar, uc_id11.cstar, uc_id11.alphas, uc_id11.betas, uc_id11.gammas]),
                uc_anri.reciprocal_lattice_parameters,
            )
            np.testing.assert_allclose(uc_anri.volume, cellvolume(uc_anri.lattice_parameters))


# class TestSymmetry(unittest.TestCase):
#     def test_id11(self):
#         ntests = 100

#         from Dans_Diffraction.functions_lattice import random_lattice

#         lpars_list = [random_lattice(symmetry="triclinic") for _ in range(ntests)]
#         lpars_batch = jnp.array(lpars_list)

#         self.assertTupleEqual(lpars_batch.shape, (ntests, 6))
#         self.assertTrue(~jnp.any(jnp.isnan(lpars_batch)))