"""Utilities: choosing devices for JAX and setting it up (call :func:`setup` first)."""

from ._impl.backend import mesh, setup

__all__ = ["mesh", "setup"]
