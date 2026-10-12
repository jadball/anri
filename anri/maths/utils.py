"""Small numerical helpers: 3x3 inverses, Gaussian widths, and evaluating JAX code in float64."""

from ._impl.utils import float64, fwhm_to_sigma, inv3, sigma_to_fwhm

__all__ = ["float64", "fwhm_to_sigma", "inv3", "sigma_to_fwhm"]
