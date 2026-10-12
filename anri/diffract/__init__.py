"""Diffraction: the Ewald condition, (2theta, eta) of scattered beams, and intensity factors.

- :mod:`anri.diffract.scatter`: Ewald conditions, omega solutions, scattering angles and wave-vectors
- :mod:`anri.diffract.factors`: Lorentz and polarisation factors
"""

from . import factors, scatter

__all__ = ["factors", "scatter"]
