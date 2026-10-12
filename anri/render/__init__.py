"""Simulate scanning-3DXRD datasets from a map: three stages, each writing one HDF5 file with its provenance.

:func:`stage_prepare` turns a map (an ImageD11 TensorMap with any number of phases, or the indexer's entries) and the
instrument (ID11's optics, or numbers) into map entries, reflections, structure factors, beam spreads and the scan's
motor positions. :func:`stage_render` renders rows of the scan into sparse pixels (:func:`anri.fwd.render_phases`);
rows split between jobs go to separate files. :func:`stage_assemble` gathers them into an ImageD11 dataset that
``python -m anri.index`` reads. ``python -m anri.render`` runs them from the command line; :func:`anri.io.show` prints
how any of the files was made.
"""

from ._impl.stages import (
    BANDWIDTH,
    EIGER_4M,
    OPTICS,
    load_prepare,
    parse_rows,
    read_map,
    render_parts,
    stage_assemble,
    stage_prepare,
    stage_render,
)

__all__ = [
    "BANDWIDTH",
    "EIGER_4M",
    "OPTICS",
    "load_prepare",
    "parse_rows",
    "read_map",
    "render_parts",
    "stage_assemble",
    "stage_prepare",
    "stage_render",
]
