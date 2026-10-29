"""Calibrate the detector and beam geometry on a scanning-3DXRD scan of an indexed sample.

:func:`anri.io.imaged11.read_peaks` reads ImageD11's peaks, one per 4D label and dty row; :func:`fit.calibrate`
matches them to an indexed map (each scattering from where the beam crosses its grain) and fits the geometry.
"""

from . import fit, stages

__all__ = ["fit", "stages"]
