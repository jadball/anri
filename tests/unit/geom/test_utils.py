import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri

jax.config.update("jax_enable_x64", True)


class TestRotZ(unittest.TestCase):
    def test_against_fable_geom_doc(self):
        # In FABLE, the Omega matrix is defined as:
        # Omega = [cos(w) -sin(w)       0]
        #         [sin(w)  cos(w)       0]
        #         [     0       0       1]
        # they say it is applied as:
        # vec_lab = Omega . vec(sample)
        # We have the same understanding about the rot_x, rot_y, rot_z functions.

        omega = 12.345
        romega = jnp.radians(omega)
        expected = jnp.array([[jnp.cos(romega), -jnp.sin(romega), 0], [jnp.sin(romega), jnp.cos(romega), 0], [0, 0, 1]])
        result = anri.geom.rot_z(omega)
        np.testing.assert_allclose(result, expected)
