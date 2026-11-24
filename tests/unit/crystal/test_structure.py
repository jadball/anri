import os
import unittest
import warnings

import numpy as np

import anri.crystal
import anri.maths.utils

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
    def test_float64(self):
        """The B matrix is float64 (the JAX definition evaluated in float64), whatever JAX's global setting."""
        B = anri.crystal.structure.B_matrix([3.1, 3.2, 5.1, 90.0, 95.0, 120.0])
        self.assertEqual(B.dtype, np.float64)
        with anri.maths.utils.float64():
            ref = np.asarray(anri.crystal.lattice.lpars_to_B(np.array([3.1, 3.2, 5.1, 90.0, 95.0, 120.0])))
        np.testing.assert_array_equal(B, ref)


class TestSpaceGroup(unittest.TestCase):
    def test_number_name_and_matrices(self):
        self.assertEqual(anri.crystal.structure.space_group("Fm-3m"), 225)
        self.assertEqual(anri.crystal.structure.space_group(read_cif("Fe.cif")), 229)
        S = anri.crystal.structure.symmetry_matrices(225)
        self.assertEqual(S.shape, (192, 4, 4))  # 48 point operations x 4 F-centring translations
        R = S[:, :3, :3]
        np.testing.assert_allclose(R @ R.transpose(0, 2, 1), np.broadcast_to(np.eye(3), R.shape), atol=1e-12)
        np.testing.assert_allclose(S[:, 3], np.broadcast_to([0, 0, 0, 1], (192, 4)))

    def test_lattice_parameters(self):
        np.testing.assert_allclose(anri.crystal.structure.lattice_parameters(read_cif("Ti.cif"))[3:], [90, 90, 120])


class TestReflections(unittest.TestCase):
    def test_integer_hkl_sorted_by_ds(self):
        """Integer hkl, sorted by d* (values: tests/theory/test_crystal.py::test_reflections_code, test_allowed_code)."""
        a, wl = 2.8665, 0.3
        r = anri.crystal.structure.reflections([a, a, a, 90, 90, 90], 229, wl, 1.0)
        self.assertEqual(r["hkl"].dtype.kind, "i")
        self.assertTrue(np.all(np.diff(r["ds"]) >= -1e-9))  # equal d* to rounding: sorted as rings accepts

    def test_rings_need_sorted_ds(self):
        """rings refuses unsorted d* (its rule: tests/theory/test_crystal.py::test_rings_code)."""
        with self.assertRaises(ValueError):
            anri.crystal.structure.rings([1.0, 0.5])


class TestStructureFactors(unittest.TestCase):
    def test_warns_without_thermal_factors(self):
        """A CIF without thermal factors warns (values: tests/theory/test_crystal.py::test_structure_factors_code)."""
        xtl = read_cif("Si.cif")
        with self.assertWarns(UserWarning):
            anri.crystal.structure.structure_factors(xtl, np.array([[1, 1, 1]]), 0.3)

    def test_no_warning_with_thermal_factors(self):
        from Dans_Diffraction.classes_crystal import Crystal as DansCrystal

        built = DansCrystal()  # built in code, with a non-zero U_iso
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            anri.crystal.structure.structure_factors(built, np.array([[1, 0, 0], [1, 1, 0]]), 0.3)


if __name__ == "__main__":
    unittest.main()
