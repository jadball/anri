"""A Python package for analysing Three-Dimensional X-Ray Diffraction data."""

# from . import crystal, diffract, fwd, geom
from .version import VERSION, VERSION_SHORT

__all__ = [
    "VERSION",
    "VERSION_SHORT",
    "calib",
    "crystal",
    "diffract",
    "fwd",
    "geom",
    "index",
    "io",
    "maths",
    "phantom",
    "simulate",
    "utils",
]
