"""Plots of the indexer's diagnostics; each returns ``update(**options) -> str`` to redraw for other values."""

from ._impl.plots import (
    plot_completeness,
    plot_convergence,
    plot_etacut,
    plot_grid,
    plot_grid_deviances,
    plot_likelihood,
    plot_lit,
    plot_main_population,
    plot_mask,
    plot_populations,
    plot_residual,
    plot_rings,
    plot_row_ratios,
    plot_rows,
    plot_y0,
)

__all__ = [
    "plot_completeness",
    "plot_convergence",
    "plot_etacut",
    "plot_grid",
    "plot_grid_deviances",
    "plot_likelihood",
    "plot_lit",
    "plot_main_population",
    "plot_mask",
    "plot_populations",
    "plot_residual",
    "plot_rings",
    "plot_row_ratios",
    "plot_rows",
    "plot_y0",
]
