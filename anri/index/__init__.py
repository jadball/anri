"""Index scanning 3DXRD data: orientation populations per voxel, fitted to the data with the forward model.

The steps, in order (``python -m anri.index`` runs them on an ImageD11 dataset):

1. :func:`ring_table` and :func:`ring_profile`: the rings, and how wide they are in the data.
2. :func:`histogram_pixels`: the sparse pixels binned once into ``H[ring, eta, omega, row]``, and a row-summed lit map
   (:func:`lit_table`).
3. :func:`choose_grid`, :func:`prune` and :func:`orientation_mlem`: an orientation grid over the fundamental zone
   (:func:`anri.crystal.orientation_grid`), first filtered by completeness with tolerances from the grid and the data
   (:func:`match_tolerances`), then kept by the likelihood ratio of a global, intensity-aware orientation fit.
4. :func:`fit_occupancy`: sparse occupancy of the kept orientations per voxel by MLEM, every voxel fitted jointly.
5. :func:`populations`: each voxel's occupancy grouped into orientation populations with fractions and spreads.
"""

from ._impl.data import (
    coarsen_rows,
    histogram,
    histogram_pixels,
    lit_table,
    pixel_angles,
    ring_profile,
    ring_table,
    ring_widths,
    tth_profile,
)
from ._impl.mask import backproject, draw_mask, otsu, polygon_mask, ramp_filter, reconstruct, threshold_mask
from ._impl.occupancy import (
    backward,
    beam_rows,
    block_voxels,
    candidates,
    candidates_from,
    censored_ratio,
    deviance,
    fit_occupancy,
    forward,
    inherit_candidates,
    mlem,
    pad_voxels,
    system,
)
from ._impl.orientations import orientation_mlem
from ._impl.populations import populations
from ._impl.predict import (
    GRID_STEPS,
    choose_grid,
    completeness,
    completeness_of,
    lorentz_polarisation,
    match_tolerances,
    predict,
    predictions,
    prune,
)

__all__ = [
    "GRID_STEPS",
    "backproject",
    "backward",
    "beam_rows",
    "block_voxels",
    "candidates",
    "candidates_from",
    "censored_ratio",
    "choose_grid",
    "coarsen_rows",
    "completeness",
    "completeness_of",
    "deviance",
    "draw_mask",
    "fit_occupancy",
    "forward",
    "histogram",
    "histogram_pixels",
    "inherit_candidates",
    "lit_table",
    "lorentz_polarisation",
    "match_tolerances",
    "mlem",
    "orientation_mlem",
    "otsu",
    "pad_voxels",
    "pixel_angles",
    "polygon_mask",
    "populations",
    "predict",
    "predictions",
    "prune",
    "ramp_filter",
    "reconstruct",
    "ring_profile",
    "ring_table",
    "ring_widths",
    "system",
    "threshold_mask",
    "tth_profile",
]
