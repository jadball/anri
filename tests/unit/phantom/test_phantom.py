import unittest

import numpy as np

import anri.crystal
import anri.io
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


class TestDeformation(unittest.TestCase):
    """Cells accumulating like a random walk, intrinsic spread and bent grains."""

    def test_brownian_field(self):
        xy = np.stack(np.meshgrid(np.arange(64.0), [0.0, 16.0, 32.0, 48.0]), -1).reshape(-1, 2)
        d = []
        for seed in range(4):
            f = anri.phantom.brownian_field(xy, 1.0, 2.0, 0.1, np.random.default_rng(seed)).reshape(4, 64, 3)
            d.append([np.mean((f[:, k:] - f[:, :-k]) ** 2) for k in (2, 8)])
        rms = np.sqrt(np.mean(d, 0))
        self.assertAlmostEqual(rms[0], 0.1, delta=0.02)  # 0.1 deg per component at the scale
        self.assertAlmostEqual(rms[1] / rms[0], 2.0, delta=0.4)  # 4x the distance, 2x the rms: a random walk

    def test_defaults_unchanged(self):
        a = anri.phantom.polycrystal(41, 0.5, 10.0, 6, seed=3)
        b = anri.phantom.polycrystal(41, 0.5, 10.0, 6, cell_walk_deg=0.0, cell_sig_deg=0.0, bend_grains=0, seed=3)
        np.testing.assert_array_equal(a["U"], b["U"])
        self.assertNotIn("sig_rot", a)

    def test_walk_keeps_cells(self):
        ph = anri.phantom.polycrystal(41, 0.5, 10.0, 2, cell_spread_deg=0.0, twin_grains=0, cell_walk_deg=0.2,
                                      cell_sig_deg=0.05, seed=1)  # fmt: skip
        U, cell, ins, grain = ph["U"].reshape(-1, 3, 3), ph["cell"].ravel(), ph["inside"].ravel(), ph["grain"].ravel()
        for c in np.unique(cell[ins])[:20]:  # constant inside each cell (cells cross grain boundaries: one grain)
            m = (cell == c) & ins & (grain == grain[(cell == c) & ins][0])
            np.testing.assert_allclose(U[m], np.broadcast_to(U[m][0], U[m].shape), atol=1e-12)
        np.testing.assert_allclose(ph["sig_rot"][ph["inside"]], np.radians(0.05))

    def test_bend(self):
        ph = anri.phantom.polycrystal(81, 0.25, 10.0, 1, cell_spread_deg=0.0, twin_grains=0, bend_grains=1,
                                      bend_deg=2.0, seed=4)  # fmt: skip
        U = ph["U"][ph["inside"]]
        angle = np.degrees(
            np.arccos(np.clip((np.trace(U @ np.swapaxes(U[:1], 1, 2), axis1=1, axis2=2) - 1) / 2, -1, 1))
        )
        self.assertGreater(angle.max(), 1.0)  # turns by up to ~2 deg per radius over the disk
        self.assertLess(angle.max(), 5.0)

    def test_tensormap(self):
        ph = anri.phantom.polycrystal(41, 0.5, 10.0, 6, cell_spread_deg=0.1, twin_grains=0, cell_walk_deg=0.1,
                                      cell_sig_deg=0.05, seed=3)  # fmt: skip
        tm = anri.phantom.tensormap(ph, [3.6] * 3 + [90.0] * 3, 225, "fcc", 0.5)
        self.assertEqual(tm.shape, (1, 41, 41))
        mis = tm.misorientation[0][tm.phase_ids[0] == 0]
        self.assertTrue(np.all(np.isfinite(mis)) and 0.0 < np.median(mis) < 2.0)
        e = anri.io.entries_from_tensormap(tm)
        np.testing.assert_allclose(e["sig_rot"], np.radians(0.05))


if __name__ == "__main__":
    unittest.main()
