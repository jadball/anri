import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from anri import diffract

jax.config.update("jax_enable_x64", True)


class TestOmegaSolnsBoth(unittest.TestCase):
    def test_matches_single_solutions(self):
        rng = np.random.default_rng(0)
        k_in = diffract.scatter.scale_norm_k(jnp.array([1.0, 0.0, 0.0]), 0.3)
        both = vmap(diffract.scatter.omega_solns_both, in_axes=(0, None))
        single = vmap(diffract.scatter.omega_solns, in_axes=(0, None, None))
        q = jnp.asarray(rng.normal(size=(200, 3)))
        omega1, omega2, valid = both(q, k_in)
        for omega, etasign in ((omega1, 1.0), (omega2, -1.0)):
            expected, valid_single = single(q, etasign, k_in)
            np.testing.assert_array_equal(valid, valid_single)
            np.testing.assert_allclose(omega[valid], expected[valid])
        self.assertTrue(valid.any() and not valid.all())
