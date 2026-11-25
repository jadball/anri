import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.diffract
import anri.geom

jax.config.update("jax_enable_x64", True)


class TestRaytraceFloat32(unittest.TestCase):
    def test_float32_matches_float64(self):
        """The whole detector chain must not lose precision in float32 (the GPU default).

        Pixel positions are ~distance from the origin, so building pixel steps by subtracting positions
        loses ~2e-4 of a 75 um pixel at 150 mm in float32: several tenths of a pixel on the detector.
        """
        pars = {
            "y_center": 1049.9, "y_size": 75.0, "tilt_y": -2e-3, "z_center": 1116.5, "z_size": 75.0, "tilt_z": 3e-3,
            "tilt_x": 1e-3, "distance": 150e3, "o11": -1.0, "o12": 0.0, "o21": 0.0, "o22": -1.0,
        }  # fmt: skip
        rng = np.random.default_rng(0)
        # outgoing rays up to ~37 degrees two-theta, from origins within 50 um of the rotation axis
        tth = np.radians(rng.uniform(3, 37, 2000))
        eta = rng.uniform(-np.pi, np.pi, 2000)
        k_out = np.stack([np.cos(tth), np.sin(tth) * np.sin(eta), np.sin(tth) * np.cos(eta)], 1)
        origin = rng.uniform(-50, 50, (2000, 3)) * np.array([1.0, 0.0, 1.0])
        out = {}
        for dtype in (np.float32, np.float64):
            # under x64 the transforms promote to float64; cast them, as a GPU run would have them in float32
            transforms = [jnp.asarray(t, dtype) for t in anri.geom.detector.detector_transforms(*pars.values())]
            basis = anri.geom.detector.detector_basis_vectors_lab(*transforms)
            trace = jax.vmap(anri.geom.detector.raytrace_to_det, in_axes=(0, 0, None, None, None))
            sc, fc = trace(jnp.asarray(k_out, dtype), jnp.asarray(origin, dtype), *basis)
            self.assertEqual(sc.dtype, dtype)
            out[dtype] = np.stack([sc, fc], 1).astype(np.float64)
        # float32 rounds positions ~distance from the origin: 16 ulps of float32 of (distance + |origin|), in pixels
        bound = 16 * np.finfo(np.float32).eps * (pars["distance"] + 50.0 * np.sqrt(2)) / pars["y_size"]
        np.testing.assert_allclose(out[np.float32], out[np.float64], rtol=0, atol=bound)
