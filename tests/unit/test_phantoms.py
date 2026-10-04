import os
import unittest

import numpy as np

QUARTZ_FLYXDM_H5 = os.path.join(
    os.path.dirname(__file__), "..", "data", "phantoms", "quartz_flyxdm", "quartz_flyxdm_tmap.h5"
)


class TestQuartzFlyxdm(unittest.TestCase):
    def setUp(self):
        from ImageD11.sinograms.tensor_map import TensorMap

        self.tmap = TensorMap.from_h5(QUARTZ_FLYXDM_H5)
        self.mask = self.tmap.labels >= 0

    def test_shape(self):
        self.assertEqual(self.tmap.shape, (1, 96, 106))
        self.assertEqual(self.mask.sum(), 6818)
        self.assertTrue(np.isfinite(self.tmap.UBI[self.mask]).all())
        self.assertTrue(np.isnan(self.tmap.UBI[~self.mask]).all())

    def test_no_2pi_in_B(self):
        # B[0, 0] = 1 / (a sin(gamma)) for a hexagonal cell, without 2pi
        a = self.tmap.phases[0].lattice_parameters[0]
        B00_expected = 2 / (np.sqrt(3) * a)
        np.testing.assert_allclose(self.tmap.B[self.mask][:, 0, 0], B00_expected, rtol=5e-3)

    def test_consistent(self):
        UBI = self.tmap.UBI[self.mask]
        UB = self.tmap.UB[self.mask]
        np.testing.assert_allclose(UB @ UBI, np.broadcast_to(np.eye(3), UB.shape), atol=1e-12)
        np.testing.assert_allclose(self.tmap.U[self.mask] @ self.tmap.B[self.mask], UB, atol=1e-12)


AM316L_H5 = os.path.join(os.path.dirname(__file__), "..", "data", "phantoms", "am316l", "am316l_tmap.h5")


class TestAM316L(unittest.TestCase):
    def setUp(self):
        from ImageD11.sinograms.tensor_map import TensorMap

        self.tmap = TensorMap.from_h5(AM316L_H5)
        self.mask = self.tmap.phase_ids >= 0

    def test_shape(self):
        self.assertEqual(self.tmap.shape, (1, 103, 103))
        self.assertEqual(self.mask.sum(), 7845)
        self.assertTrue(np.isfinite(self.tmap.UBI[self.mask]).all())
        self.assertEqual(self.tmap.twin.sum(), 253)

    def test_lattice(self):
        a = self.tmap.phases[0].lattice_parameters[0]
        np.testing.assert_allclose(a, 3.5966)
        np.testing.assert_allclose(self.tmap.B[self.mask][:, 0, 0], 1 / a, rtol=1e-6)  # no 2pi

    def test_entries(self):
        from anri.io import entries_from_tensormap

        e = entries_from_tensormap(self.tmap)
        self.assertEqual(len(e["pos"]), 7845)
        self.assertLessEqual(np.linalg.norm(e["pos"][:, :2], axis=1).max(), 25.0)
