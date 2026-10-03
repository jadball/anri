import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.geom

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


class TestRmatFromAxisAngle(unittest.TestCase):
    def test_axes(self):
        # the axis need not be normalised
        for axis, rot in (
            ([2.0, 0, 0], anri.geom.rot_x),
            ([0, 0.5, 0], anri.geom.rot_y),
            ([0, 0, 3.0], anri.geom.rot_z),
        ):
            for angle in (-30.0, 10.0, 135.0):
                np.testing.assert_allclose(
                    anri.geom.rmat_from_axis_angle(jnp.array(axis), angle), rot(angle), atol=1e-12
                )
