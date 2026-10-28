"""Geometry: transforms between the laboratory, detector, goniometer and sample frames.

- :mod:`anri.geom.gonio`: goniometer and sample frames, omega wrapping, and the dty that puts a voxel in the beam
- :mod:`anri.geom.detector`: detector frames
- :mod:`anri.geom.scan`: scanning-3DXRD grids: reconstruction, step and sample positions, dty rows
- :mod:`anri.geom.beam`: beam directions and offsets
- :mod:`anri.geom.paths`: paths through the sample to its edge, for absorption
"""

from . import beam, detector, gonio, paths, scan

__all__ = ["beam", "detector", "gonio", "paths", "scan"]
