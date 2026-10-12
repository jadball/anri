"""Plots of the renderer's diagnostics, for choosing its options by eye (matplotlib, imported only here).

As :mod:`anri.index.plots`: each ``plot_x(fig, ...)`` draws on a matplotlib figure it is given and returns
``update(**options) -> str``, which redraws for other values of the options it shows and returns a one-line summary;
plots with no option to move return None. The numbers come from :mod:`anri.render.diagnostics`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_footprint(fig: Figure, fp: dict) -> None:
    """Draw the sample's voxels, per phase, against the radius the rows reach at every omega.

    Parameters
    ----------
    fig
        A matplotlib figure
    fp
        From :func:`anri.render.diagnostics.footprint`
    """
    ax = fig.subplots(1, 2)
    for name, pos in zip(fp["names"], fp["pos"]):
        ax[0].plot(pos[:, 0], pos[:, 1], ".", ms=2, label=name)
    t = np.linspace(0, 2 * np.pi, 361)
    for r, style, label in ((fp["radius"], "k:", "sample radius"), (fp["covered"], "r-", "rows reach, every ω")):
        ax[0].plot(r * np.cos(t), r * np.sin(t), style, lw=0.8, label=label)
    ax[0].set(aspect="equal", xlabel="x", ylabel="y", title="sample (rotation axis at 0)")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[1].vlines(fp["dty"], 0, 1, color="0.6", lw=0.5)
    ax[1].axvspan(fp["y0"] - fp["radius"], fp["y0"] + fp["radius"], color="tab:blue", alpha=0.2, label="sample")
    ax[1].axvline(fp["y0"], color="k", lw=0.8, label="y0")
    ax[1].set(xlabel="dty", yticks=[], title=f"{len(fp['dty'])} rows; beam FWHM {fp['beam_fwhm']:g}")
    ax[1].legend(fontsize=8)


def plot_rings(fig: Figure, cov: dict) -> Callable:
    """Draw the fraction of each ring on the detector and each phase's reflections; ``update(tth_max)`` marks the cut.

    Parameters
    ----------
    fig
        A matplotlib figure
    cov
        From :func:`anri.render.diagnostics.ring_coverage`
    """
    ax = fig.subplots(2, 1, sharex=True, height_ratios=[2, 1])
    ax[0].plot(cov["tth"], cov["on_detector"], color="k")
    ax[0].set(ylabel="fraction of the ring on the detector", ylim=(0, 1.05))
    for i, (name, refl) in enumerate(zip(cov["names"], cov["reflections"])):
        tth, mult = np.unique(np.round(refl, 4), return_counts=True)
        ax[1].vlines(tth, i, i + 0.8 * mult / mult.max(), color=f"C{i}", lw=1)
    ax[1].set(xlabel="2θ (°)", yticks=np.arange(len(cov["names"])) + 0.4, yticklabels=cov["names"],
              title="reflections (height: multiplicity)")  # fmt: skip
    lines = [a.axvline(cov["tth_max"], color="tab:red") for a in ax]

    def update(tth_max: float) -> str:
        for ln in lines:
            ln.set_xdata([tth_max, tth_max])
        fig.canvas.draw_idle()
        on = np.interp(tth_max, cov["tth"], cov["on_detector"])
        counts = ", ".join(f"{n}: {int(np.sum(r <= tth_max))}" for n, r in zip(cov["names"], cov["reflections"]))
        return f"tth_max {tth_max:.2f} deg: reflections {counts}; {on * 100:.0f}% of the outermost ring on the detector"

    return update


def plot_counts(fig: Figure, probe: dict) -> Callable:
    """Draw the rendered intensities; ``update(density, cut)`` shows what survives the segmentation cut.

    Intensities scale with ``density`` (counts per unit density), so the probe at one density shows any other.

    Parameters
    ----------
    fig
        A matplotlib figure
    probe
        From :func:`anri.render.diagnostics.probe_rows`
    """
    v = probe["values"]
    ax = fig.subplots(1, 2)
    hist = ax[0]
    a1 = ax[1]
    drawn: list = []

    def update(density: float, cut: float) -> str:
        x = v * density / probe["density"]
        counts = np.round(x)
        keep = counts > cut
        for art in drawn:
            art.remove()
        bins = np.logspace(np.log10(max(x[x > 0].min(), 1e-3)), np.log10(max(x.max(), 1.0)) + 0.01, 120)
        drawn[:] = [*hist.hist(x, bins=bins, color="0.7", log=True)[2], *hist.hist(x[keep], bins=bins, color="tab:blue",
                    log=True)[2], hist.axvline(cut + 0.5, color="tab:red")]  # fmt: skip
        hist.set(xscale="log", xlabel="counts per pixel (rendered)", ylabel="pixels",
                 title=f"density {density:g}: blue kept above cut {cut:g}")  # fmt: skip
        cuts = np.arange(0, 51)
        n_kept = [(counts > c).mean() for c in cuts]
        i_kept = [x[counts > c].sum() / max(x.sum(), 1e-30) for c in cuts]
        drawn.extend(a1.plot(cuts, n_kept, color="tab:blue", label="pixels kept"))
        drawn.extend(a1.plot(cuts, i_kept, color="tab:orange", label="intensity kept"))
        drawn.append(a1.axvline(cut, color="tab:red"))
        a1.set(xlabel="cut (counts)", ylabel="fraction", ylim=(0, 1.02), title="against the cut")
        a1.legend(fontsize=8)
        fig.canvas.draw_idle()
        return (f"density {density:g}, cut {cut:g}: {keep.sum()} of {len(x)} rendered pixels kept "
                f"({keep.mean() * 100:.1f}%), holding {x[keep].sum() / max(x.sum(), 1e-30) * 100:.1f}% of the "
                f"intensity; brightest pixel {x.max():.3g} counts (rows {probe['rows']})")  # fmt: skip

    return update


def plot_capture(fig: Figure, probe: dict) -> None:
    """Draw how much of each peak its window holds, and how many frames the windows needed (``window``, ``max_frames``).

    Parameters
    ----------
    fig
        A matplotlib figure
    probe
        From :func:`anri.render.diagnostics.probe_rows`
    """
    ax = fig.subplots(1, 2)
    cap = probe["captured"]
    ax[0].hist(np.maximum(1 - cap, 1e-6), bins=np.logspace(-6, 0, 61), log=True, color="tab:blue")
    ax[0].set(xscale="log", xlabel="share of the peak in the scan's ω range outside its window", ylabel="peaks",
              title=f"{len(cap)} peaks; {probe['n_outside']} centred beyond the ends of the ω range left out")  # fmt: skip
    if probe["window_frames"] is not None:
        sizes, n = np.unique(probe["window_frames"], return_counts=True)
        ax[1].bar([str(s) for s in sizes], n)
        ax[1].set(yscale="log", xlabel="frames in the window", ylabel="peaks", title="window sizes (max_frames)")


def plot_row_image(fig: Figure, img: dict, title: str = "") -> None:
    """Draw a row summed over its frames (log counts), and its counts per frame.

    Parameters
    ----------
    fig
        A matplotlib figure
    img
        From :func:`anri.render.diagnostics.row_image`
    """
    from matplotlib.colors import LogNorm
    from matplotlib.pyplot import colormaps

    ax = fig.subplots(1, 2, width_ratios=[1.3, 1])
    im = img["image"]
    cmap = colormaps["magma"].with_extremes(bad="black")
    h = ax[0].imshow(np.ma.masked_less_equal(im, 0), cmap=cmap, norm=LogNorm(), interpolation="nearest")
    fig.colorbar(h, ax=ax[0], shrink=0.8)
    ax[0].set(title=title or "summed over the frames", xlabel="fast", ylabel="slow")
    ax[1].plot(img["omega"], img["per_frame"], lw=0.7)
    ax[1].set(xlabel="ω (°)", ylabel="counts", title="per frame")
