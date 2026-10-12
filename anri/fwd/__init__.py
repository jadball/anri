"""Forward projection: peaks of a voxel map projected onto the detector, and rendered into sparse pixels.

- :mod:`anri.fwd.peak`: the centroid and covariance of one peak, for any beam and scan
- :mod:`anri.fwd.box`: the same for box-beam scans (near or far field)
- :mod:`anri.fwd.scan`: the same for scanning 3DXRD
- :mod:`anri.fwd.render`: the renderer, one dty row (one rotation scan) at a time
"""

from . import box, peak, render, scan

__all__ = ["box", "peak", "render", "scan"]
