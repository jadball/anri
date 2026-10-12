"""Plots of the renderer's diagnostics; those with options return ``update(**options) -> str`` to redraw."""

from ._impl.plots import (
    plot_capture,
    plot_footprint,
    plot_frame_compare,
    plot_rings,
    plot_row_image,
    plot_survival,
)

__all__ = [
    "plot_capture",
    "plot_footprint",
    "plot_frame_compare",
    "plot_rings",
    "plot_row_image",
    "plot_survival",
]
