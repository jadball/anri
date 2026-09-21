import logging

# suppress JAX debug spam when there are test failures
logging.getLogger("jax").setLevel(logging.WARNING)