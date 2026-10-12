"""Geometry: transforms between the laboratory, detector, goniometer and sample frames.

- :mod:`anri.geom.gonio`: goniometer and sample frames, omega wrapping, and the dty that puts a voxel in the beam
- :mod:`anri.geom.detector`: detector frames
- :mod:`anri.geom.scan`: scanning-3DXRD grids: reconstruction, step and sample positions, dty rows
- :mod:`anri.geom.utils`: beam directions and offsets
"""

from . import detector, gonio, scan, utils

__all__ = ["detector", "gonio", "scan", "utils"]
