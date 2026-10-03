"""Utilities: choosing devices for JAX, and small linear-algebra helpers."""

from ._impl.backend import mesh, setup
from ._impl.mathutils import inv3

__all__ = ["inv3", "mesh", "setup"]
