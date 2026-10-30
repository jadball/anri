"""Plots of the calibration's diagnostics (matplotlib, imported only here).

As :mod:`anri.index.plots`: each ``plot_x(fig, ...)`` draws on a matplotlib figure it is given. These show results
with no option to move, so they return None. The numbers come from :mod:`anri.calib.diagnostics`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_centre_search(fig: Figure, d: dict) -> None:
    """Draw the beam-centre search: the radial histogram's sharpness about each centre tried, grid by grid.

    Parameters
    ----------
    fig
        A matplotlib figure
    d
        From :func:`anri.calib.diagnostics.read_rings`, of a run with the search
    """
    c = d["centre"]
    ax = np.atleast_1d(fig.subplots(1, len(c["levels"])))
    fitted = (d["geometry"]["z_center"], d["geometry"]["y_center"])
    for a, lv in zip(ax, c["levels"]):
        z, y = np.unique(lv["centres"][:, 0]), np.unique(lv["centres"][:, 1])
        img = lv["sharpness"].reshape(len(z), len(y)) / np.median(lv["sharpness"])
        im = a.imshow(img, origin="lower", aspect="equal", cmap="viridis",
                      extent=(y[0] - lv["step"] / 2, y[-1] + lv["step"] / 2, z[0] - lv["step"] / 2,
                              z[-1] + lv["step"] / 2))  # fmt: skip
        best = lv["centres"][np.argmax(lv["sharpness"])]
        a.plot(best[1], best[0], "r+", ms=12, label="best")
        if a is ax[0]:
            a.plot(c["start"][1], c["start"][0], "wx", ms=8, label="start")
        a.plot(fitted[1], fitted[0], "o", mfc="none", mec="orange", ms=10, label="fitted")
        a.set(xlabel="y_center (px)", ylabel="z_center (px)", title=f"grid {lv['step']:g} px")
        fig.colorbar(im, ax=a, shrink=0.8, label="sharpness / median")
    ax[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("centre search: where every ring is sharpest (the fit then refines it with the tilts)")


def plot_distance_search(fig: Figure, d: dict) -> None:
    """Draw the distance search: the intensity the rings' windows catch, above a flat background, at each distance.

    Parameters
    ----------
    fig
        A matplotlib figure
    d
        From :func:`anri.calib.diagnostics.read_rings`, of a run with the search
    """
    ax = fig.subplots()
    ax.plot(d["distance"] / 1e3, d["score"], color="k", lw=0.8)
    best = d["distance"][np.argmax(d["score"])]
    ax.axvline(d["start"]["distance"] / 1e3, color="0.5", ls="--", label=f"start {d['start']['distance'] / 1e3:.2f} mm")
    ax.axvline(best / 1e3, color="r", ls=":", label=f"best {best / 1e3:.2f} mm")
    ax.axvline(d["geometry"]["distance"] / 1e3, color="orange", lw=0.8,
               label=f"fitted {d['geometry']['distance'] / 1e3:.3f} mm")  # fmt: skip
    ax.set(xlabel="distance (mm)", ylabel="score", title="distance search: one clear peak means the rings are found")
    ax.legend(fontsize=8)


def plot_ring_offsets(fig: Figure, d: dict) -> None:
    """Draw each ring's mean 2theta offset in each eta sector, at the start and fitted: straight rings lie at 0.

    Parameters
    ----------
    fig
        A matplotlib figure
    d
        From :func:`anri.calib.diagnostics.read_rings`
    """
    from matplotlib import pyplot as plt

    ax = fig.subplots(1, 2, sharex=True)
    lit = np.flatnonzero(d["used"] & (d["intensity"].sum(1) > 0))
    colours = plt.colormaps["viridis"](np.linspace(0, 0.9, max(len(lit), 1)))
    tol = d["tols"]
    for a, key, title in ((ax[0], "offset_start", f"start (window {tol[0]:g}°)"),
                          (ax[1], "offset", f"fitted (window {tol[-1]:g}°)")):  # fmt: skip
        for col, r in zip(colours, lit):
            a.plot(d["eta"], d[key][r], ".", ms=2, color=col, label=f"{d['tth'][r]:.2f}°")
        a.axhline(0, color="k", lw=0.6)
        a.set(xlabel="η (°)", ylabel="2θ − ring (°)", title=title)
    lim = (
        5 * max(np.nanpercentile(np.abs(d["offset"][lit]), 90), 2e-4) if len(lit) else 0.01
    )  # not a few sparse sectors
    ax[1].set_ylim(-lim, lim)
    ax[0].legend(fontsize=7, markerscale=4, title="ring 2θ", ncol=2)
    fig.suptitle("each ring's offset per η sector. Roughly: a centre error is a sine in η, the same for every ring; "
                 "a tilt a sine growing with 2θ; the distance a constant growing with 2θ", fontsize=10)  # fmt: skip
