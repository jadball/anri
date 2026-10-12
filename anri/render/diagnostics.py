"""The numbers behind each choice of the renderer's options; they draw nothing (:mod:`anri.render.plots` draws them)."""

from ._impl.diagnostics import (
    footprint,
    probe_rows,
    ring_coverage,
    row_image,
)

__all__ = [
    "footprint",
    "probe_rows",
    "ring_coverage",
    "row_image",
]
