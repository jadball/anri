"""Pytest configuration shared by all tests."""

import logging

import anri.utils

# configure JAX as scripts do, before any test starts it (e.g. turns off XLA:CPU fusions that miscompile)
anri.utils.setup()

# suppress JAX debug spam when there are test failures
logging.getLogger("jax").setLevel(logging.WARNING)
