import os
import unittest
from unittest import mock

import jax
import jax.numpy as jnp

import anri.utils


class TestSetup(unittest.TestCase):
    def test_after_jax_started(self):
        jnp.zeros(1).block_until_ready()
        with self.assertRaises(RuntimeError):
            anri.utils.setup()

    def flags(self, xla_flags, jaxlib_version, **kwargs):
        """XLA_FLAGS that setup() would set, as if JAX hadn't started, leaving os.environ untouched."""
        env = {"XLA_FLAGS": xla_flags} if xla_flags is not None else {}
        started = mock.patch("jax._src.xla_bridge.backends_are_initialized", return_value=False)
        version = mock.patch("jaxlib.version.__version__", jaxlib_version)
        with mock.patch.dict(os.environ, env, clear=True), started, version:  # no parenthesised with: Python 3.9
            anri.utils.setup(**kwargs)
            return os.environ["XLA_FLAGS"], os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]

    def test_flags(self):
        ynn = "--xla_cpu_experimental_ynn_fusion_type="
        flags, prealloc = self.flags(None, "0.11.2")
        self.assertEqual(flags, f"--xla_force_host_platform_device_count=4 {ynn}")
        self.assertEqual(prealloc, "false")
        # the device count is replaced, other flags and a user's own YNN setting are kept
        flags, prealloc = self.flags(
            f"--xla_force_host_platform_device_count=2 {ynn}foo", "0.11.2", n_cpu=8, preallocate=True
        )
        self.assertEqual(flags.split(), [f"{ynn}foo", "--xla_force_host_platform_device_count=8"])
        self.assertEqual(prealloc, "true")
        # older jaxlib doesn't know the YNN flag
        flags, _ = self.flags("", "0.4.30")
        self.assertEqual(flags, "--xla_force_host_platform_device_count=4")

    def test_matmul_precision(self):
        """Full float32 matrix products (no TF32 on GPUs), unless the user chose a precision."""
        before = jax.config.jax_default_matmul_precision
        try:
            self.flags(None, "0.11.2")
            self.assertEqual(jax.config.jax_default_matmul_precision, "highest")
            jax.config.update("jax_default_matmul_precision", None)
            with mock.patch.dict(os.environ, {"JAX_DEFAULT_MATMUL_PRECISION": "default"}):
                started = mock.patch("jax._src.xla_bridge.backends_are_initialized", return_value=False)
                with started:
                    anri.utils.setup()
            self.assertIsNone(jax.config.jax_default_matmul_precision)
        finally:
            jax.config.update("jax_default_matmul_precision", before)


if __name__ == "__main__":
    unittest.main()
