import os
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
            try:
                uc_id11 = unitcell_id11(lpars_batch[i], "P")
            except np.linalg.LinAlgError:
                continue
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


CIF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cif")


class TestSymmetry(unittest.TestCase):
    def test_number_and_name(self):
        by_number = anri.crystal.Symmetry.from_number(225)
        by_name = anri.crystal.Symmetry.from_name("Fm-3m")
        for sym in (by_number, by_name):
            self.assertEqual(sym.sgno, 225)
            self.assertIn("Fm-3m", sym.sgname)
        ops = np.asarray(by_number.sym_ops)
        self.assertEqual(ops.shape, (192, 3, 3))  # 48 point operations x 4 F-centring translations
        np.testing.assert_allclose(ops @ ops.transpose(0, 2, 1), np.broadcast_to(np.eye(3), ops.shape), atol=1e-12)


class TestStructure(unittest.TestCase):
    def setUp(self):
        self.struc = anri.crystal.Structure.from_cif(os.path.join(CIF_DIR, "Fe.cif"))

    def test_needs_make_hkls(self):
        for name in ("allhkls", "alltth", "allds", "rings_table", "rings_dict", "ringhkls", "ringhkls_arr", "ringtth",
                     "ringds", "ringmult"):  # fmt: skip
            with self.subTest(name), self.assertRaises(AttributeError):
                getattr(self.struc, name)

    def test_expand_to_p1(self):
        self.struc.make_hkls(dsmax=1.0, wavelength=0.3)
        n_p1 = self.struc.allhkls.shape[0]
        self.assertEqual(self.struc.alltth.shape, (n_p1,))
        self.assertEqual(self.struc.allds.shape, (n_p1,))
        unique = anri.crystal.Structure.from_cif(os.path.join(CIF_DIR, "Fe.cif"))
        unique.make_hkls(dsmax=1.0, wavelength=0.3, expand_to_p1=False)
        # one hkl per set of symmetry-equivalent reflections: fewer hkls, the same d* values
        self.assertLess(unique.allhkls.shape[0], n_p1)
        np.testing.assert_allclose(np.unique(np.round(unique.allds, 5)), np.unique(np.round(self.struc.allds, 5)))

    def test_rings_id11(self):
        from ImageD11.unitcell import unitcell as unitcell_id11

        dsmax, wavelength = 1.2, 0.18
        self.struc.make_hkls(dsmax=dsmax, wavelength=wavelength)
        uc = unitcell_id11(np.asarray(self.struc.lattice_parameters), self.struc.sgno)
        uc.makerings(dsmax, tol=1e-4)
        np.testing.assert_allclose(self.struc.ringds, uc.ringds, rtol=1e-6)
        self.assertEqual(len(self.struc.rings_dict), len(uc.ringds))
        for ds, mult, hkls in zip(uc.ringds, self.struc.ringmult, self.struc.ringhkls.values()):
            self.assertEqual(int(mult), len(uc.ringhkls[ds]))
            self.assertEqual(sorted(map(tuple, np.asarray(hkls).astype(int).tolist())), sorted(uc.ringhkls[ds]))
        tth = np.degrees(2 * np.arcsin(np.asarray(self.struc.ringds) * wavelength / 2))
        np.testing.assert_allclose(self.struc.ringtth, tth, rtol=1e-5)
        np.testing.assert_array_equal(self.struc.ringhkls_arr, np.concatenate(list(self.struc.ringhkls.values())))
        self.assertIs(self.struc.rings_table, self.struc.rings_table)  # computed once

    def test_thermal_factor_warning(self):
        import warnings

        from Dans_Diffraction.classes_crystal import Crystal as dd_Crystal

        # read from a CIF without U_iso / B_iso: warn
        self.struc.make_hkls(dsmax=0.5, wavelength=0.3)
        with self.assertWarns(UserWarning):
            _ = self.struc.rings_table
        # built in code, with a non-zero U_iso: no warning
        built = anri.crystal.Structure(dd_Crystal())
        built.make_hkls(dsmax=0.5, wavelength=0.3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _ = built.rings_table


class TestGrain(unittest.TestCase):
    def test_u_and_ub(self):
        from ImageD11.grain import grain as grain_id11

        import anri.geom

        B = anri.crystal.lpars_to_B(jnp.array([4.9, 4.9, 5.4, 90.0, 90.0, 120.0]))
        U = anri.geom.rot_z(25.0) @ anri.geom.rot_x(35.0) @ anri.geom.rot_y(10.0)
        ubi = jnp.linalg.inv(U @ B)
        g = anri.crystal.Grain(ubi)
        np.testing.assert_allclose(g.UB, U @ B, atol=1e-12)
        np.testing.assert_allclose(g.U, U, atol=1e-12)
        np.testing.assert_allclose(g.U, grain_id11(np.asarray(ubi)).U, atol=1e-10)
        np.testing.assert_allclose(g.lattice_parameters, [4.9, 4.9, 5.4, 90.0, 90.0, 120.0], atol=1e-10)
