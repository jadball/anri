import os
import unittest
import warnings

import numpy as np

import anri.crystal

CIF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cif")


def read_cif(name):
    import Dans_Diffraction

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Dans_Diffraction.Crystal(os.path.join(CIF_DIR, name))


def cellvolume(lp):
    """Unit cell volume from lattice parameters (as ImageD11.forward_model.forward_projector)."""
    ca, cb, cg = np.cos(np.radians(lp[3:]))
    return lp[0] * lp[1] * lp[2] * np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)


class TestLattice(unittest.TestCase):
    def test_against_imaged11(self):
        """B, the metric tensors, reciprocal lattice and volume (the JAX definitions, in float64) match ImageD11."""
        from Dans_Diffraction.functions_lattice import random_lattice
        from ImageD11.unitcell import unitcell as unitcell_id11

        rng = np.random.default_rng(0)
        kinds = [
            "cubic",
            "tetragonal",
            "hexagonal",
            "rhobohedral",
            "monoclinic-a",
            "monoclinic-b",
            "monoclinic-c",
            "triclinic",
        ]
        for _ in range(50):
            np.random.seed(int(rng.integers(1 << 30)))
            lp = np.asarray(random_lattice(symmetry=kinds[int(rng.integers(len(kinds)))]), float)
            try:
                uc = unitcell_id11(lp, "P")
            except np.linalg.LinAlgError:
                continue
            np.testing.assert_allclose(anri.crystal.B_matrix(lp), uc.B, rtol=1e-7, atol=1e-10)
            with anri.crystal.float64():
                mt = np.asarray(anri.crystal.lpars_to_mt(lp))
                rmt = np.asarray(anri.crystal.mt_to_rmt(mt))
                np.testing.assert_allclose(mt, uc.g, rtol=1e-7, atol=1e-10)
                np.testing.assert_allclose(rmt, uc.gi, rtol=1e-7, atol=1e-10)
                np.testing.assert_allclose(
                    anri.crystal.mt_to_lpars(rmt), [uc.astar, uc.bstar, uc.cstar, uc.alphas, uc.betas, uc.gammas]
                )
                np.testing.assert_allclose(anri.crystal.metric_to_volume(mt), cellvolume(lp))

    def test_float64(self):
        """The B matrix is float64 (the JAX definition evaluated in float64), whatever JAX's global setting."""
        B = anri.crystal.B_matrix([3.1, 3.2, 5.1, 90.0, 95.0, 120.0])
        self.assertEqual(B.dtype, np.float64)
        with anri.crystal.float64():
            ref = np.asarray(anri.crystal.lpars_to_B(np.array([3.1, 3.2, 5.1, 90.0, 95.0, 120.0])))
        np.testing.assert_array_equal(B, ref)


class TestSpaceGroup(unittest.TestCase):
    def test_number_name_and_matrices(self):
        self.assertEqual(anri.crystal.space_group("Fm-3m"), 225)
        self.assertEqual(anri.crystal.space_group(read_cif("Fe.cif")), 229)
        S = anri.crystal.symmetry_matrices(225)
        self.assertEqual(S.shape, (192, 4, 4))  # 48 point operations x 4 F-centring translations
        R = S[:, :3, :3]
        np.testing.assert_allclose(R @ R.transpose(0, 2, 1), np.broadcast_to(np.eye(3), R.shape), atol=1e-12)
        np.testing.assert_allclose(S[:, 3], np.broadcast_to([0, 0, 0, 1], (192, 4)))

    def test_lattice_parameters(self):
        np.testing.assert_allclose(anri.crystal.lattice_parameters(read_cif("Ti.cif"))[3:], [90, 90, 120])


class TestReflections(unittest.TestCase):
    def test_bcc_absences_and_order(self):
        a, wl = 2.8665, 0.3
        r = anri.crystal.reflections([a, a, a, 90, 90, 90], 229, wl, 1.0)
        self.assertEqual(r["hkl"].dtype.kind, "i")
        self.assertTrue(np.all(r["hkl"].sum(1) % 2 == 0))  # I-centring: h + k + l even
        self.assertTrue(np.all(np.diff(r["ds"]) >= -1e-12))
        np.testing.assert_allclose(r["ds"], np.linalg.norm(r["hkl"], axis=1) / a)
        np.testing.assert_allclose(r["tth"], np.degrees(2 * np.arcsin(r["ds"] * wl / 2)))
        self.assertLessEqual(r["ds"].max(), 1.0)
        # every allowed reflection up to dsmax is there: count them directly
        n = 3
        g = np.stack(np.meshgrid(*(np.arange(-n, n + 1),) * 3, indexing="ij"), -1).reshape(-1, 3)
        g = g[np.any(g != 0, 1) & (g.sum(1) % 2 == 0) & (np.linalg.norm(g, axis=1) / a <= 1.0)]
        self.assertEqual(len(r["hkl"]), len(g))

    def test_against_imaged11(self):
        """Rings (d*, multiplicity and members) match ImageD11's unitcell for a structure with atoms."""
        from ImageD11.unitcell import unitcell as unitcell_id11

        xtl = read_cif("Fe.cif")
        lp, sg = anri.crystal.lattice_parameters(xtl), anri.crystal.space_group(xtl)
        dsmax, wl = 1.2, 0.18
        r = anri.crystal.reflections(lp, sg, wl, dsmax)
        ring, ring_ds = anri.crystal.rings(r["ds"])
        uc = unitcell_id11(lp, sg)
        uc.makerings(dsmax, tol=1e-4)
        np.testing.assert_allclose(ring_ds, uc.ringds, rtol=1e-6)
        for i, ds in enumerate(uc.ringds):
            self.assertEqual(sorted(map(tuple, r["hkl"][ring == i].tolist())), sorted(uc.ringhkls[ds]))

    def test_rings_do_not_chain(self):
        ring, ring_ds = anri.crystal.rings([1.0, 1.00005, 1.0001, 1.00015, 1.0002], tol=1e-4)
        np.testing.assert_array_equal(ring, [0, 0, 0, 1, 1])  # a ring is within tol of its first member
        np.testing.assert_allclose(ring_ds, [1.0, 1.00015])
        with self.assertRaises(ValueError):
            anri.crystal.rings([1.0, 0.5])


class TestStructureFactors(unittest.TestCase):
    def test_special_positions_and_warning(self):
        xtl = read_cif("Si.cif")
        lp, sg, wl = anri.crystal.lattice_parameters(xtl), anri.crystal.space_group(xtl), 0.3
        r = anri.crystal.reflections(lp, sg, wl, 1.0)
        with self.assertWarns(UserWarning):  # this CIF has no thermal factors
            F2 = anri.crystal.structure_factors(xtl, r["hkl"], wl)
        # {222} of diamond is allowed by the space group but zero from the atom positions
        zero = F2 < 1e-6 * F2.max()
        self.assertTrue(zero.any())
        self.assertTrue(all(sorted(np.abs(h)) == [2, 2, 2] or np.all(np.abs(h) % 2 == 0) for h in r["hkl"][zero]))
        self.assertTrue(np.all(F2[np.all(np.abs(r["hkl"]) == 1, 1)] > 0))  # {111} is strong

    def test_no_warning_with_thermal_factors(self):
        from Dans_Diffraction.classes_crystal import Crystal as DansCrystal

        built = DansCrystal()  # built in code, with a non-zero U_iso
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            anri.crystal.structure_factors(built, np.array([[1, 0, 0], [1, 1, 0]]), 0.3)


if __name__ == "__main__":
    unittest.main()
