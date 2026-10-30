"""Plots of the calibration's diagnostics (matplotlib, imported only here).

As :mod:`anri.index.plots`: each ``plot_x(fig, ...)`` draws on a matplotlib figure it is given. Those of an option
(:func:`plot_min_occ`, :func:`plot_match`) return ``update(**option) -> str``, which redraws for another value and
returns a one-line summary; front-ends bind their own controls to it (ipywidgets in Jupyter, Qt in a workflow
widget). Plots of a result with no option to move return None. The numbers come from :mod:`anri.calib.diagnostics`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from . import diagnostics
from .stages import COMPONENTS, IJ

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


DIST_BINS = np.linspace(0.0, 0.87, 175)  # (h, k, l) is at most sqrt(3) / 2 from integers


def plot_min_occ(fig: Figure, occ: dict, min_occ: float = 0.01) -> Callable:
    """Draw the occupancies over the largest and the share of each voxel's kept; ``update(min_occ)`` moves the cut.

    Parameters
    ----------
    fig
        A matplotlib figure
    occ
        From :func:`anri.calib.diagnostics.read_occupancy`
    min_occ
        The cut drawn first
    """
    ax = fig.subplots(1, 2, width_ratios=(1.6, 1))
    ax[0].hist(occ["log_ratio"], bins=120, log=True, color="tab:blue")
    line = ax[0].axvline(np.log10(min_occ), color="tab:red")
    ax[0].set(xlabel="log10 occupancy / largest", ylabel="(voxel, candidate) pairs")
    im = ax[1].imshow(np.full((occ["nr"], occ["nr"]), np.nan), origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax[1].set_facecolor("0.85")
    ax[1].set(title="share of each voxel's occupancy kept", xticks=[], yticks=[])
    fig.colorbar(im, ax=ax[1], shrink=0.8)

    def update(min_occ: float) -> str:
        k = diagnostics.occupancy_kept(occ, min_occ)
        line.set_xdata([np.log10(min_occ)] * 2)
        im.set_data(k["share"])
        fig.canvas.draw_idle()
        return (f"{k['pairs_kept']} of {k['pairs']} (voxel, orientation) pairs kept on {k['voxels']} voxels, "
                f"holding {k['fraction'] * 100:.1f}% of the occupancy")  # fmt: skip

    return update


def plot_match(fig: Figure, dist: dict, tols: tuple) -> Callable:
    """Draw each peak's distance of (h, k, l) from integers, and against intensity; ``update(tol)`` moves the first.

    Parameters
    ----------
    fig
        A matplotlib figure
    dist
        From :func:`anri.calib.diagnostics.match_distances`, at the starting geometry
    tols
        The rounds' tolerances: the first drawn in red (``update`` moves it), the later ones in grey
    """
    err, intensity = dist["err"], dist["intensity"]
    ax = fig.subplots(1, 2)
    ax[0].hist(err, DIST_BINS, log=True, color="tab:blue")
    for t in tols[1:]:
        ax[0].axvline(t, color="0.5", ls=":")
    line = ax[0].axvline(tols[0], color="tab:red")
    ax[0].set(xlabel="distance of (h, k, l) from integers", ylabel="peaks")
    sub = np.random.default_rng(0).choice(len(err), min(len(err), 20000), replace=False)
    ax[1].scatter(intensity[sub], err[sub], s=2, alpha=0.3, color="tab:blue")
    hline = ax[1].axhline(tols[0], color="tab:red")
    ax[1].set(xscale="log", xlabel="peak intensity", ylabel="distance from integers")

    def update(tol: float) -> str:
        line.set_xdata([tol, tol])
        hline.set_ydata([tol, tol])
        fig.canvas.draw_idle()
        n = int(np.sum(err < tol))
        return f"{n} of {len(err)} peaks ({n / max(len(err), 1) * 100:.1f}%) within {tol:g} in the first round"

    return update


def plot_residuals(fig: Figure, trends: dict, clip: float) -> None:
    """Draw each matched peak's residual (sc, fc, omega) / sigma against omega, detector azimuth and dty, with medians.

    Parameters
    ----------
    fig
        A matplotlib figure
    trends
        From :func:`anri.calib.diagnostics.residual_trends`
    clip
        The stage's ``clip``: the y range
    """
    r = trends["r"]
    labels = {"omega": "ω (°)", "azimuth": "azimuth on the detector (°)", "dty": "dty"}
    ax = fig.subplots(3, 3, sharey="row")
    sub = np.random.default_rng(0).choice(len(r), min(len(r), 20000), replace=False)
    for j, (name, x) in enumerate(trends["against"].items()):
        for i, comp in enumerate(("sc", "fc", "ω")):
            ax[i, j].scatter(x[sub], r[sub, i], s=1, alpha=0.2, color="tab:blue")
            ax[i, j].plot(trends["centres"][name], trends["medians"][name][:, i], color="tab:red")
            ax[i, j].axhline(0, color="k", lw=0.5)
            ax[i, j].set_ylim(-clip, clip)
            ax[i, 0].set_ylabel(f"{comp}: residual / σ")
        ax[0, j].set_title(f"against {labels[name].split(' (')[0]}")
        ax[2, j].set_xlabel(labels[name])


def plot_residual_hists(fig: Figure, trends: dict, clip: float, start: dict | None = None,
                        calibrated: dict | None = None, tols: tuple = ()) -> None:  # fmt: skip
    """Draw the residuals' histograms against a unit Gaussian, and the distances from integers before and after.

    Parameters
    ----------
    fig
        A matplotlib figure
    trends
        From :func:`anri.calib.diagnostics.residual_trends`
    clip
        The stage's ``clip``: the histograms' range
    start, calibrated
        From :func:`anri.calib.diagnostics.match_distances`, at the starting and at the calibrated geometry (both, or
        neither: then the last panel is left out)
    tols
        The rounds' tolerances, drawn on the last panel
    """
    r = trends["r"]
    ax = fig.subplots(1, 3 if start is None else 4)
    z = np.linspace(-clip, clip, 200)
    for i, comp in enumerate(("sc", "fc", "ω")):
        ax[i].hist(r[:, i], bins=np.linspace(-clip, clip, 81), log=True, color="tab:blue")
        ax[i].plot(z, len(r) * (2 * clip / 80) * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi), color="k", lw=0.8)
        ax[i].set_xlabel(f"{comp}: residual / σ")
        ax[i].set_ylim(0.5, None)
    ax[0].set_ylabel("peaks")
    if start is not None and calibrated is not None:
        ax[3].hist(start["err"], DIST_BINS, log=True, histtype="step", label="start")
        ax[3].hist(calibrated["err"], DIST_BINS, log=True, histtype="step", label="calibrated")
        for t in tols:
            ax[3].axvline(t, color="0.5", ls=":")
        ax[3].set_xlabel("distance of (h, k, l) from integers")
        ax[3].legend()


def plot_strain_left(fig: Figure, st: dict) -> None:
    """Draw each orientation's strain left (hydrostatic and deviatoric) against its peaks, with peak-weighted means.

    Parameters
    ----------
    fig
        A matplotlib figure
    st
        From :func:`anri.calib.diagnostics.strain_left`
    """
    ax = fig.subplots(1, 2)
    ok, n_pk = st["fitted"], st["n_peaks"]
    if not ok.any():
        return
    ax[0].scatter(n_pk[ok], st["hydrostatic"][ok] * 1e5, s=8, color="tab:blue")
    ax[0].axhline(st["mean_hydrostatic"] * 1e5, color="k")
    ax[0].set_title("hydrostatic")
    for c, (i, j) in zip(COMPONENTS, IJ):
        pts = ax[1].scatter(n_pk[ok], st["deviatoric"][ok, i, j] * 1e5, s=8, label=c)
        ax[1].axhline(st["mean_deviatoric"][i, j] * 1e5, color=pts.get_facecolor()[0], lw=1)
    ax[1].set_title("deviatoric (sample frame)")
    ax[1].legend(ncol=3)
    for a in ax:
        a.set(xscale="log", xlabel="matched peaks of the orientation", ylabel="strain (1e-5)")
