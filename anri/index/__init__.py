"""Index scanning 3DXRD data: orientation populations per voxel, fitted to the data with the forward model.

The steps, in order (``python -m anri.index`` runs them on an ImageD11 dataset):

1. :func:`data.ring_table` and :func:`data.ring_profile`: the rings, and how wide they are in the data.
2. :func:`data.histogram_pixels`: the sparse pixels binned once into ``H[ring, eta, omega, row]``, and a row-summed lit
   map (:func:`data.lit_table`).
3. :func:`search.choose_grid`, :func:`search.prune` and :func:`orientations.fit`: an orientation grid
   over the fundamental zone (:func:`anri.crystal.orientation.orientation_grid`), first filtered by completeness with
   tolerances from the grid and the data (:func:`search.match_tolerances`), then kept by the likelihood ratio of a
   global, intensity-aware orientation fit.
4. :func:`occupancy.fit_occupancy`: sparse occupancy of the kept orientations per voxel by MLEM, every voxel fitted
   jointly (:mod:`anri.index.mask` chooses which voxels).
5. :func:`local.children` and :func:`local.inherit`: a local search that sharpens the fitted orientations off the
   grid, around those each voxel uses, refitting each voxel over their children only.
6. :func:`populations.group`: each voxel's occupancy grouped into orientation populations with fractions and
   spreads.

:mod:`anri.index.stages` runs the same steps as six stages that each write one HDF5 file with its provenance, for the
command line, notebooks and workflow systems: :func:`stages.stage_histogram` (step 1), :func:`stages.stage_search`
(completeness), :func:`stages.stage_filter` (likelihood or completeness), :func:`stages.stage_fit` (step 4),
:func:`stages.stage_localsearch` (step 5) and :func:`stages.stage_populations` (step 6). :func:`anri.io.provenance.show` prints how any stage file was made, back to
the dataset, and :func:`stages.run_command` the command line that remakes it. :func:`stages.open_dataset` opens a
dataset as the histogram stage does. The numbers behind each option's choice are in :mod:`anri.index.diagnostics`, and
their plots in :mod:`anri.index.plots`.
"""

from . import data, diagnostics, local, mask, occupancy, orientations, plots, populations, search, stages

__all__ = [
    "data",
    "diagnostics",
    "local",
    "mask",
    "occupancy",
    "orientations",
    "plots",
    "populations",
    "search",
    "stages",
]
