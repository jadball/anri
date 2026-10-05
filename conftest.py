"""Pytest configuration shared by all tests."""

import logging
import os

# tests run on CPU, as in CI: on a GPU they mostly wait for XLA:GPU to compile small one-off shapes. Set JAX_PLATFORMS
# (e.g. cuda) to test on a GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import anri.utils

# configure JAX as scripts do, before any test starts it (e.g. turns off XLA:CPU fusions that miscompile)
anri.utils.setup()

# suppress JAX debug spam when there are test failures
logging.getLogger("jax").setLevel(logging.WARNING)
