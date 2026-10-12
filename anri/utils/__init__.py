"""Utilities: choosing devices for JAX, and small maths helpers."""

from ._impl.backend import mesh, setup
from ._impl.mathutils import fwhm_to_sigma, inv3, sigma_to_fwhm

__all__ = ["fwhm_to_sigma", "inv3", "mesh", "setup", "sigma_to_fwhm"]
