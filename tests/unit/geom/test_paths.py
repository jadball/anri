import unittest

import numpy as np

import anri.geom


class TestAttenuationMap(unittest.TestCase):
    def test_weighted_mean_and_layout(self):
        """Each voxel: its entries' density-weighted mean mu, at the voxel recon_positions puts there; others empty."""
        step = 2.0
        pos = np.array([[4.0, 2.0, 0], [4.2, 1.9, 0], [-6.0, 0.0, 0]])
        m = anri.geom.paths.attenuation_map(pos, np.array([1.0, 3.0, 5.0]), np.array([3.0, 1.0, 2.0]), step)
        n = m.shape[0]
        grid = np.asarray(anri.geom.scan.recon_positions(n, step))
        at = {tuple(np.round(g[:2], 6)): v for g, v in zip(grid, m.ravel())}
        self.assertAlmostEqual(at[(4.0, 2.0)], 1.5, places=6)  # (3 x 1 + 1 x 3) / 4
        self.assertAlmostEqual(at[(-6.0, 0.0)], 5.0, places=6)
        self.assertEqual(int((m > 0).sum()), 2)
        np.testing.assert_array_equal(m[[0, -1]], 0)  # an empty border
