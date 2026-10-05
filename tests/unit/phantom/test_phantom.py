import unittest

import numpy as np

import anri.crystal
import anri.phantom


def cubic_ops():
    return anri.crystal.laue_rotations(
        anri.crystal.symmetry_matrices(225), anri.crystal.B_matrix([3.6] * 3 + [90.0] * 3)
    )


class TestRotations(unittest.TestCase):
    def test_axis_angle(self):
        R = anri.phantom.axis_angle([[0.0, 0.0, 2.0], [1.0, 1.0, 1.0]], [90.0, 120.0])
        np.testing.assert_allclose(R[0] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)  # right-handed about z
        np.testing.assert_allclose(R[1] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)  # 3-fold about <111>

    def test_small_rotations(self):
        R = anri.phantom.small_rotations(20000, 0.5, np.random.default_rng(0))
        angle = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        self.assertAlmostEqual(np.sqrt(np.mean(angle**2)), 0.5 * np.sqrt(3), delta=0.02)  # rms of a 3D normal

    def test_random_rotations(self):
        R = anri.phantom.random_rotations(1000, np.random.default_rng(1))
        np.testing.assert_allclose(R @ np.swapaxes(R, 1, 2), np.broadcast_to(np.eye(3), R.shape), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


class TestPolycrystal(unittest.TestCase):
    def setUp(self):
        self.ph = anri.phantom.polycrystal(41, 0.5, 10.0, 6, cell_size=1.5, cell_spread_deg=0.3, twin_grains=1, seed=3)

    def test_maps(self):
        ph = self.ph
        inside = ph["inside"]
        self.assertEqual(inside.shape, (41, 41))
        self.assertEqual(ph["pos"].shape, (41 * 41, 3))
        np.testing.assert_array_equal(inside.ravel(), np.linalg.norm(ph["pos"][:, :2], axis=1) <= 10.0)
        self.assertTrue(np.all(ph["grain"][~inside] == -1) and np.all(ph["grain"][inside] >= 0))
        self.assertTrue(np.isnan(ph["U"][~inside]).all())
        np.testing.assert_allclose(np.linalg.det(ph["U"][inside]), 1.0, atol=1e-9)
        self.assertGreater(ph["twin"].sum(), 0)

    def test_deterministic(self):
        again = anri.phantom.polycrystal(41, 0.5, 10.0, 6, cell_size=1.5, cell_spread_deg=0.3, twin_grains=1, seed=3)
        np.testing.assert_array_equal(again["grain"], self.ph["grain"])
        np.testing.assert_array_equal(again["U"], self.ph["U"])

    def test_cells_and_twins(self):
        ph, ops = self.ph, cubic_ops()
        U, grain, twin, cell = (ph[k].reshape(-1, *ph[k].shape[2:]) for k in ("U", "grain", "twin", "cell"))
        g = grain[twin][0]
        parent = np.flatnonzero((grain == g) & ~twin)
        twins = np.flatnonzero(twin)
        m = min(len(parent), len(twins))
        mis = anri.crystal.disorientation(U[parent[:m]], U[twins[:m]], ops)
        np.testing.assert_allclose(mis, 60.0, atol=3.0)  # Sigma3, give or take two cells' rotations
        # two voxels of one cell share an orientation; different cells of one grain are a little apart
        same = [np.flatnonzero((cell == c) & (grain == g) & ~twin)[:2] for c in np.unique(cell[(grain == g) & ~twin])]
        pairs = np.array([p for p in same if len(p) == 2])
        np.testing.assert_allclose(U[pairs[:, 0]], U[pairs[:, 1]], atol=1e-12)
        cells = np.array([p[0] for p in same])
        spread = anri.crystal.disorientation(U[cells[:-1]], U[cells[1:]], ops)
        self.assertTrue(0.05 < np.median(spread) < 2.0)


if __name__ == "__main__":
    unittest.main()
