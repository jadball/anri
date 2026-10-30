"""Calibrate the detector and beam geometry on a scanning-3DXRD scan.

:mod:`rings` fits the distance, beam centre and two tilts to powder rings, close enough to index.
:func:`anri.io.imaged11.read_peaks` reads ImageD11's peaks, one per 4D label and dty row; :func:`fit.calibrate`
matches them to an indexed map (each scattering from where the beam crosses its grain) and fits the geometry.
"""

from . import fit, rings, stages

__all__ = ["fit", "rings", "stages"]
