import unittest

import jax
import numpy as np

import anri.geom

jax.config.update("jax_enable_x64", True)


class TestStepGrid(unittest.TestCase):
    """Against ImageD11's sinogram geometry, which these functions follow."""

    def test_downsampled_holds_zero(self):
        """A downsampled grid is symmetric and holds 0 in its middle, where step_to_recon puts it (issue #33)."""
        ybincens = np.arange(-10.5, 12.0, 0.75)  # half extent 17 steps from y0 = -1.2: odd, so rounded up to 18
        si, sj = anri.geom.scan.step_grid_from_ybincens(ybincens, 0.75, 2, -1.2)
        np.testing.assert_array_equal(si[:, 0], np.arange(-18, 19, 2))
        np.testing.assert_array_equal(sj, si.T)
        n = si.shape[0]
        ri, _ = anri.geom.scan.step_to_recon(si, sj, (n, n))
        self.assertEqual((int(si[n // 2, n // 2]), int(sj[n // 2, n // 2])), (0, 0))
        np.testing.assert_array_equal(ri[:, 0], si[:, 0] + n // 2)  # 0 at the middle index n // 2


class TestDtyToRow(unittest.TestCase):
    def test_rows(self):
        edges = np.array([-1.5, -0.5, 0.5, 1.5, 2.5])  # 4 rows
        dty = np.array([-2.0, -1.5, -0.6, 0.0, 0.49, 0.5, 2.4, 2.5, np.nan, np.inf])
        # bins are (lo, hi]: a dty on an edge is in the row below it
        np.testing.assert_array_equal(anri.geom.scan.dty_to_row(dty, edges), [-1, -1, 0, 1, 1, 1, 3, 3, -1, -1])
        np.testing.assert_array_equal(anri.geom.scan.dty_to_row(dty, edges, 2), [-1, -1, 0, 0, 0, 0, 1, 1, -1, -1])
        self.assertEqual(anri.geom.scan.dty_to_row(np.zeros((2, 3)), edges).shape, (2, 3))


class TestRegrid(unittest.TestCase):
    def test_round_trip(self):
        """Grids centred on the axis share their middle voxels: pad then crop gives the array back."""
        n, step = 11, 2.0
        a = np.arange(n * n, dtype=float).reshape(n, n)
        big = anri.geom.scan.regrid(a, 15)
        np.testing.assert_array_equal(big[2:-2, 2:-2], a)
        np.testing.assert_array_equal(anri.geom.scan.regrid(big, n), a)
        p_small = np.asarray(anri.geom.scan.recon_positions(n, step)).reshape(n, n, 3)
        p_big = np.asarray(anri.geom.scan.recon_positions(15, step)).reshape(15, 15, 3)
        np.testing.assert_allclose(anri.geom.scan.regrid(p_big, n), p_small)  # the same positions
        with self.assertRaises(ValueError):
            anri.geom.scan.regrid(a, 12)  # odd onto even: not centred alike
