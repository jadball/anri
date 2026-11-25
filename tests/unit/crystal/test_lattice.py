import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.crystal

jax.config.update("jax_enable_x64", True)


class TestUBToUBI(unittest.TestCase):
    def test_nan_in_nan_out(self):
        """An empty voxel's NaN U gives a NaN UBI, not an error (values: tests/theory/test_frames.py::test_ubi_code)."""
        B = np.asarray(anri.crystal.lattice.lpars_to_B(jnp.array([3.0, 4.0, 5.0, 85.0, 95.0, 100.0])))
        self.assertTrue(np.all(np.isnan(anri.crystal.lattice.U_B_to_UBI(np.full((3, 3), np.nan), B))))


if __name__ == "__main__":
    unittest.main()
