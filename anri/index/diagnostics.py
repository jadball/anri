"""The numbers behind each choice of the indexer's options; they draw nothing (:mod:`anri.index.plots` draws them)."""

from ._impl.diagnostics import (
    back_projection,
    candidate_shares,
    convergence_recorder,
    eta_profile,
    grid_chances,
    grid_deviances,
    lit_curve,
    nearest_stronger,
    ring_profile_cake,
    ring_ratios,
    ring_windows,
    row_profile,
    y0_from_edges,
)

__all__ = [
    "back_projection",
    "candidate_shares",
    "convergence_recorder",
    "eta_profile",
    "grid_chances",
    "grid_deviances",
    "lit_curve",
    "nearest_stronger",
    "ring_profile_cake",
    "ring_ratios",
    "ring_windows",
    "row_profile",
    "y0_from_edges",
]
