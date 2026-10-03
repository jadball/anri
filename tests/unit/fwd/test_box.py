import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.crystal
import anri.fwd
import anri.geom

jax.config.update("jax_enable_x64", True)


class TestCentroidBoxBoth(unittest.TestCase):
    def test_matches_single_branches(self):
        # both Friedel solutions at once == two single-branch calls, with a tilted detector, wedge and chi
        B = anri.crystal.lpars_to_B(jnp.array([4.0, 4.0, 4.0, 90.0, 90.0, 90.0]))
        rng = np.random.default_rng(0)
        q, _ = np.linalg.qr(rng.normal(size=(5, 3, 3)))
        ubi = jnp.linalg.inv(jnp.asarray(q) @ B)
        origin = jnp.asarray(rng.uniform(-200, 200, size=(5, 3)))
        hkls = jnp.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 0], [3, 2, 1]], dtype=float)
        det_trans, beam_cen_shift, x_distance_shift = anri.geom.detector_transforms(
            1024.0, 75.0, 0.01, 1000.0, 75.0, -0.02, 0.005, 150e3, 1, 0, 0, 1
        )
        det = anri.geom.detector_basis_vectors_lab(det_trans, beam_cen_shift, x_distance_shift)
        beam = (0.3, jnp.array([1.0, 0.0, 0.0]), 1e-4, -2e-4, 1.5, -0.7)  # wavelength, k_in_lab, ky, kz, wedge, chi

        both, valid = anri.fwd.get_centroid_box_all_both(ubi, origin, hkls, *beam, *det)
        self.assertEqual(both.shape, (5, 5, 2, 3))
        for i, etasign in enumerate((1.0, -1.0)):
            single, valid_single = anri.fwd.get_centroid_box_all(ubi, origin, hkls, etasign, *beam, *det)
            np.testing.assert_array_equal(valid, valid_single)
            np.testing.assert_allclose(both[:, :, i][valid], single[valid], rtol=1e-12, atol=1e-9)
        self.assertTrue(valid.any())


if __name__ == "__main__":
    unittest.main()
