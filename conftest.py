"""Pytest configuration shared by all tests."""

import logging
import os
import tempfile

# tests run on CPU, as in CI: on a GPU they mostly wait for XLA:GPU to compile small one-off shapes. Set JAX_PLATFORMS
# (e.g. cuda) to test on a GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Most of the tests' time is XLA compiling (single-threaded), so compiled programs are kept on disk between runs. Set
# JAX_COMPILATION_CACHE_DIR to choose the folder, or to "" to turn the cache off. JAX reads these when it is imported.
_user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"  # one folder per user on shared machines
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(tempfile.gettempdir(), f"anri-jax-cache-{_user}"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")  # the many small compiles add up too

import anri.utils

# configure JAX as scripts do, before any test starts it (e.g. turns off XLA:CPU fusions that miscompile)
anri.utils.setup()

# suppress JAX debug spam when there are test failures
logging.getLogger("jax").setLevel(logging.WARNING)
