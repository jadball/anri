"""The numbers behind each choice of the renderer's options, and measured frames against the model; they draw nothing."""

from ._impl.diagnostics import (
    footprint,
    frames_image,
    model_row,
    observed_row,
    probe_rows,
    raw_image,
    ring_coverage,
    row_image,
    row_scale,
)

__all__ = [
    "footprint",
    "frames_image",
    "model_row",
    "observed_row",
    "probe_rows",
    "raw_image",
    "ring_coverage",
    "row_image",
    "row_scale",
]
