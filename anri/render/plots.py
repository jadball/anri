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
        tth, counts = np.unique(np.round(refl, 4), return_counts=True)
        mult = np.asarray(counts, dtype=float)
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


def plot_survival(fig: Figure, probe: dict) -> Callable:
    """Draw how many peaks the cut removes against density; ``update(density)`` shows what each peak keeps there.

    Left: the share of peaks with no pixel above the cut (gone from the data), and the median share of a peak's
    intensity above it, against density. Right: each peak's kept share at the chosen density. The faint ends and tails
    of peaks go first, then whole weak reflections and small grains.

    Parameters
    ----------
    fig
        A matplotlib figure
    probe
        From :func:`anri.render.diagnostics.probe_rows`
    """
    d, lost, kept = probe["densities"], probe["lost_below"], probe["kept_share"]
    ax = fig.subplots(1, 2)
    gone = np.array([(lost > x).mean() for x in d])
    ax[0].semilogx(d, gone, color="tab:red", label="peaks gone (no pixel above the cut)")
    ax[0].semilogx(d, np.median(kept, 0), color="tab:blue", label="median share of a peak kept")
    ax[0].semilogx(d, np.percentile(kept, 10, 0), color="tab:blue", ls=":", label="10th percentile")
    v = ax[0].axvline(probe["density"], color="k")
    ax[0].set(xlabel="density (counts per unit density)", ylabel="fraction", ylim=(0, 1.02),
              title=f"cut {probe['cut']:g}; {len(lost)} peaks, rows {probe['rows']}")  # fmt: skip
    ax[0].legend(fontsize=8)
    drawn: list = []

    def update(density: float) -> str:
        k = int(np.argmin(np.abs(np.log(d / density))))
        share = kept[:, k]
        for art in drawn:
            art.remove()
        drawn[:] = list(ax[1].hist(share, bins=np.linspace(0, 1, 51), color="tab:blue", log=True)[2])
        ax[1].set(xlabel="share of the peak's intensity above the cut", ylabel="peaks", title=f"density {d[k]:.3g}")
        v.set_xdata([density, density])
        fig.canvas.draw_idle()
        return (f"density {d[k]:.3g}: {(lost > d[k]).mean() * 100:.1f}% of the peaks gone; a peak keeps "
                f"{np.median(share) * 100:.0f}% of its intensity (median), {np.percentile(share, 10) * 100:.0f}% "
                f"(10th percentile)")  # fmt: skip

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


def plot_frame_compare(fig: Figure, images: Callable, shape: tuple) -> Callable:
    """Draw measured frames, the model's and both overlaid; ``update(row, frame, frames)`` picks the frames.

    The panels share their axes (zoom one, all follow) and one log colour scale. In the overlay, the data are red and
    the model cyan: where they agree the spots are white, data the model misses are red, model the data lack cyan.

    Parameters
    ----------
    fig
        A matplotlib figure
    images
        ``images(row, frames) -> (data, model, label)``: [n_slow, n_fast] images of those frames of a row summed,
        the model already scaled, and a line about them (e.g. :func:`anri.render.diagnostics.frames_image` of
        :func:`~anri.render.diagnostics.observed_row` and :func:`~anri.render.diagnostics.model_row`)
    shape
        (n_slow, n_fast)
    """
    from matplotlib.colors import LogNorm
    from matplotlib.pyplot import colormaps

    ax = fig.subplots(1, 3, sharex=True, sharey=True)
    cmap = colormaps["magma"].with_extremes(bad="black")
    z = np.ma.masked_all(shape)
    ims = [a.imshow(z, cmap=cmap, norm=LogNorm(1, 10), interpolation="nearest") for a in ax[:2]]
    rgb = ax[2].imshow(np.zeros((*shape, 3)), interpolation="nearest")
    for a, t in zip(ax, ("data", "model", "data (red) and model (cyan)")):
        a.set(title=t, xlabel="fast")
    ax[0].set_ylabel("slow")
    fig.colorbar(ims[1], ax=ax[1], shrink=0.7)

    def update(row: int, frame: int, frames: int = 1) -> str:
        sel = range(max(frame - frames // 2, 0), frame - frames // 2 + frames)
        data, model, label = images(row, sel)
        pos = np.concatenate([data[data > 0], model[model > 0]])
        top = max(float(np.percentile(pos, 99.9)) if pos.size else 2.0, 2.0)  # a few saturated pixels set no scale
        for im, x in zip(ims, (data, model)):
            im.set_data(np.ma.masked_less_equal(x, 0))
            im.set_norm(LogNorm(1, top))

        def unit(x: np.ndarray) -> np.ndarray:
            return np.clip(np.log1p(np.maximum(x, 0)) / np.log1p(top), 0, 1)

        d, m = unit(data), unit(model)
        rgb.set_data(np.stack([d, m, m], -1))
        fig.canvas.draw_idle()
        return f"row {row}, frames {sel.start}..{sel.stop - 1}: {label}"

    return update
