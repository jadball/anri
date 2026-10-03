import unittest

import jax
import numpy as np

import anri.geom

jax.config.update("jax_enable_x64", True)


class TestStepGrid(unittest.TestCase):
    """Against ImageD11's sinogram geometry, which these functions follow."""

    def test_id11(self):
        from ImageD11.sinograms import geometry as g

        ybincens = np.arange(-10.5, 12.0, 0.75)
        for args in ((ybincens, 0.75, 1, 0.3), (ybincens, 0.75, 2, -1.2)):
            si, sj = anri.geom.step_grid_from_ybincens(*args)
            # ImageD11 gives a list of (si, sj) for si in steps for sj in steps
            expected = np.array(g.step_grid_from_ybincens(*args)).reshape(si.shape + (2,))
            np.testing.assert_array_equal(np.stack([si, sj], axis=-1), expected)
        si, sj = anri.geom.step_grid_from_ybincens(ybincens, 0.75, 1, 0.3)
        recon_shape = (si.shape[0], si.shape[1] + 2)
        ri, rj = anri.geom.step_to_recon(si, sj, recon_shape)
        np.testing.assert_array_equal(np.stack([ri, rj]), np.stack(g.step_to_recon(si, sj, recon_shape)))
        np.testing.assert_array_equal(np.stack(anri.geom.recon_to_step(ri, rj, recon_shape)), np.stack([si, sj]))
        np.testing.assert_allclose(
            np.stack(anri.geom.step_to_sample(si, sj, 0.75)), np.stack(g.step_to_sample(si, sj, 0.75))
        )
