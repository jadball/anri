"""Mathematics underneath anri, depending only on JAX and NumPy.

- :mod:`anri.maths.rotation`: rotation algebra (matrices, axis-angle, Rodrigues vectors, quaternions, angles, means)
- :mod:`anri.maths.utils`: small numerical helpers (3x3 inverses, Gaussian widths, float64 evaluation)

The rest of anri builds on these: :mod:`anri.geom` (frames), :mod:`anri.diffract`, then :mod:`anri.crystal`.
"""

from . import rotation, utils

__all__ = ["rotation", "utils"]
