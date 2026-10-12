"""Simulate scanning-3DXRD datasets from a map: three stages, each writing one HDF5 file with its provenance.

:func:`stages.stage_prepare` turns a map (an ImageD11 TensorMap with any number of phases, or the indexer's entries)
and the instrument (ID11's optics, or numbers) into map entries, reflections, structure factors, beam spreads and the
scan's motor positions. :func:`stages.stage_render` renders rows of the scan into sparse pixels
(:func:`anri.fwd.render.render_phases`); rows split between jobs go to separate files. :func:`stages.stage_assemble`
gathers them into an ImageD11 dataset that ``python -m anri.index`` reads. ``python -m anri.simulate`` runs them from the
command line; :func:`anri.io.provenance.show` prints how any of the files was made, and :func:`stages.run_command` the
command line that remakes a dataset. The numbers behind each option's choice are in :mod:`anri.simulate.diagnostics`,
and their plots in :mod:`anri.simulate.plots`.
"""

from . import diagnostics, plots, stages

__all__ = ["diagnostics", "plots", "stages"]
