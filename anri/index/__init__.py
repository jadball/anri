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

The same steps as five stages that each write one HDF5 file with its provenance, for the command line, notebooks and
workflow systems: :func:`stage_histogram` (step 1), :func:`stage_search` (completeness), :func:`stage_filter`
(likelihood or completeness), :func:`stage_fit` (step 4) and :func:`stage_populations` (step 5). :func:`show`
prints how any stage file was made, back to the dataset (:func:`read_provenance` and :func:`upstream` read the
chain). :func:`open_dataset` opens a dataset as the histogram stage does.
"""

from ._impl.data import (
    coarsen_rows,
    coarsen_scan,
    dty_offsets,
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
    match_tolerances,
    predict,
    predictions,
    prune,
)
from ._impl.stages import (
    lit_map,
    load_histogram,
    open_dataset,
    read_provenance,
    row_intensities,
    sample_groups,
    select_by_likelihood,
    select_populations,
    show,
    stage_filter,
    stage_fit,
    stage_histogram,
    stage_populations,
    stage_search,
    upstream,
    voxel_grid,
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
    "coarsen_scan",
    "completeness",
    "completeness_of",
    "deviance",
    "draw_mask",
    "dty_offsets",
    "fit_occupancy",
    "forward",
    "histogram",
    "histogram_pixels",
    "inherit_candidates",
    "lit_map",
    "lit_table",
    "load_histogram",
    "match_tolerances",
    "mlem",
    "open_dataset",
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
    "read_provenance",
    "reconstruct",
    "ring_profile",
    "ring_table",
    "ring_widths",
    "row_intensities",
    "sample_groups",
    "select_by_likelihood",
    "select_populations",
    "show",
    "stage_filter",
    "stage_fit",
    "stage_histogram",
    "stage_populations",
    "stage_search",
    "system",
    "threshold_mask",
    "tth_profile",
    "upstream",
    "voxel_grid",
]
