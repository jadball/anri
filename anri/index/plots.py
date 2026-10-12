"""Plots of the indexer's diagnostics, for choosing its options by eye (matplotlib, imported only here).

Each ``plot_x(fig, ...)`` draws on a matplotlib figure it is given and returns ``update(**options) -> str``: a function
that redraws for other values of the options it shows and returns a one-line summary. Front-ends bind their own
controls to ``update``: ipywidgets in Jupyter, Qt controls in a workflow widget. Plots of a result with no option to
move return None. The numbers come from :mod:`anri.index.diagnostics`; the selections from the stages' own
functions, so a value chosen here does what it does in the stage.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from .stages import B_E, B_O, N_POP, R_E, R_O, select_by_likelihood, select_populations

if TYPE_CHECKING:
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure


def _log_cmap() -> Colormap:
    from matplotlib import pyplot as plt

    return plt.colormaps["magma"].with_extremes(bad="black")  # empty (masked) bins black, not white


def plot_rings(fig: Figure, cake: dict) -> Callable:
    """Draw the 2theta profile, the cake and a close-up of each ring; ``update(n_rings, tth_tol)`` draws the windows.

    Parameters
    ----------
    fig
        A matplotlib figure
    cake
        From :func:`anri.index.diagnostics.ring_profile_cake`
    """
    from matplotlib.colors import LogNorm

    from .diagnostics import ring_windows

    tth, prof, C, rings = cake["tth"], cake["profile"], cake["cake"], cake["rings"]
    n_all = len(rings["tth"])
    sf_top, sf_bot = fig.subfigures(2, 1, squeeze=False, height_ratios=[2.3, 0.9]).ravel()
    ax_p, ax_c = sf_top.subplots(2, 1, sharex=True)
    ax_p.semilogy(tth, np.maximum(prof, 0.5), lw=0.7, color="k")
    for t in rings["tth"]:
        ax_p.axvline(t, color="0.6", ls="--", lw=0.6)
    ax_p.set(ylabel="counts", title="2θ profile: every ring (grey), windows of the rings used (orange)")
    k = 5  # 0.01 deg in 2theta
    n_t = len(tth)
    Cb = C[: n_t // k * k].reshape(-1, k, C.shape[1]).sum(1)
    ax_c.imshow(np.ma.masked_less_equal(Cb.T, 0), aspect="auto", origin="lower", cmap=_log_cmap(), norm=LogNorm(),
                extent=(cake["lo"], cake["lo"] + len(Cb) * k * cake["step"], -180, 180), interpolation="nearest")  # fmt: skip
    ax_c.set(xlabel="2θ (°)", ylabel="η (°)", title="cake: rings should be straight; windows in cyan")
    axz = np.atleast_1d(sf_bot.subplots(1, n_all, sharex=True, sharey=True))
    for r, a in enumerate(axz):
        z = np.abs(tth - rings["tth"][r]) < 0.15
        a.semilogy(tth[z] - rings["tth"][r], np.maximum(prof[z], 0.5), lw=0.8, color="k")
        a.set_title(f"{r}: {cake['hkl'][r]}", fontsize=9)
        a.tick_params(labelsize=7)
    axz[0].set_ylabel("counts")
    sf_bot.supxlabel("2θ − ring (°); orange: window, blue: measured centre; grey: ring not used", fontsize=9)
    drawn: list = []

    def update(n_rings: int, tth_tol: float | None = None) -> str:
        w = ring_windows(cake, n_rings, tth_tol or None)
        for art in drawn:
            art.remove()
        drawn.clear()
        for r in range(n_rings):
            t, tol = rings["tth"][r], w["tolerance"][r]
            drawn.append(ax_p.axvspan(t - tol, t + tol, color="tab:orange", alpha=0.4))
            drawn.extend(ax_c.axvline(t + s * tol, color="c", lw=0.6) for s in (-1, 1))
            drawn.append(axz[r].axvspan(-tol, tol, color="tab:orange", alpha=0.4))
            drawn.append(axz[r].axvline(w["offset"][r], color="tab:blue", lw=0.8))
        for r, a in enumerate(axz):
            a.set_facecolor("white" if r < n_rings else "0.9")
        fig.canvas.draw_idle()
        lines = [" ring  hkl        mult  |F|^2  2theta   offset  half-width  tolerance  gap to next  counts in window"]
        for r in range(n_rings):
            gap = f"{cake['gap'][r]:11.3f}" if r < len(cake["gap"]) else "          -"
            lines.append(f"{r:5d}  {cake['hkl'][r]!s:10s} {cake['multiplicity'][r]:4d}  {cake['F2'][r]:5.2f}  "
                         f"{rings['tth'][r]:6.3f}  {w['offset'][r]:+7.3f}  {w['half_width'][r]:10.3f}  "
                         f"{w['tolerance'][r]:9.3f}  {gap}  {w['counts'][r]:.3g}"
                         + ("  <- windows overlap" if w["overlap"][r] else ""))  # fmt: skip
        lines.append(f"{w['fraction'] * 100:.1f}% of the intensity up to the last ring used is in a window")
        return "\n".join(lines)

    return update


def plot_rows(fig: Figure, rows: dict, dty_within: np.ndarray | None = None, ystep: float | None = None) -> None:
    """Draw the intensity per dty row (and each row against its neighbours) and per omega bin.

    Parameters
    ----------
    fig
        A matplotlib figure
    rows
        From :func:`anri.index.diagnostics.row_profile`
    dty_within
        Optionally the DataSet's dty of every frame [rows, frames], and ystep, to draw how dty moves within a row
    """
    ax = fig.subplots(1, 3 if dty_within is not None else 2)
    ax[0].plot(rows["dty"], rows["row"], ".-", ms=3)
    a2 = ax[0].twinx()
    a2.plot(rows["dty"], rows["jump"], ".", color="tab:red", ms=3)
    a2.axhline(1, color="tab:red", lw=0.5)
    a2.set_ylabel("row / mean of its neighbours", color="tab:red")
    ax[0].set(xlabel="dty", ylabel="intensity", title="per dty row")
    ax[1].plot(rows["omega"], rows["per_omega"], lw=0.8)
    ax[1].set(xlabel="ω (°)", ylabel="intensity", title="per ω bin, all rows")
    if dty_within is not None:
        for v in dty_within[:: max(1, len(dty_within) // 10)]:
            ax[2].plot(np.linspace(0, 1, len(v)), (v - v.mean()) / ystep, lw=0.7)
        ax[2].set(xlabel="fraction of the row", ylabel="dty − row mean (dty steps)", title="the DataSet's dty in a row")


def plot_y0(fig: Figure, edges: dict, recon: Callable, y0: float, scan: dict) -> Callable:
    """Draw the sinogram with the y0 fits, and back-projections at the DataSet's y0 and at ``update(y0)``.

    Parameters
    ----------
    fig
        A matplotlib figure
    edges
        From :func:`anri.index.diagnostics.y0_from_edges`
    recon
        ``recon(y0)``: the back-projection for that y0, e.g. :func:`anri.index.diagnostics.back_projection` on one grid
    y0
        The starting value (the DataSet's)
    scan
        The histogram's scan
    """
    ax = fig.subplots(1, 3)
    ax[2].sharex(ax[1])
    ax[2].sharey(ax[1])
    om, good = edges["omega"], np.isfinite(edges["middle"])
    ext_sino = (om[0] - B_O * R_O / 2, om[-1] + B_O * R_O / 2, edges["dty"][0] - scan["ystep"] / 2,
                edges["dty"][-1] + scan["ystep"] / 2)  # fmt: skip
    ax[0].imshow(np.log1p(edges["sinogram"]).T, aspect="auto", origin="lower", cmap="magma", extent=ext_sino,
                 interpolation="nearest")  # fmt: skip
    ax[0].plot(om[good], edges["mean"][good], ".", color="0.6", ms=2, label="mean dty")
    ax[0].plot(om[good], edges["middle"][good], ".", color="c", ms=2, label="middle of the edges")
    ax[0].plot(om, edges["fit"], color="w", lw=1, label=f"fit: y0 = {edges['y0']:.3f}")
    line = ax[0].axhline(y0, color="tab:green", lw=1, ls="--", label="y0 (slider)")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set(xlabel="ω (°)", ylabel="dty", title="sinogram, all rings")
    rec0 = recon(y0)
    vmax = np.percentile(rec0, 99.5)
    n = rec0.shape[0]
    ext = (-n / 2 * scan["ystep"], n / 2 * scan["ystep"]) * 2
    ax[1].imshow(rec0, origin="lower", cmap="gray", vmin=0, vmax=vmax, extent=ext)
    ax[1].set(title=f"back-projection, y0 = {y0:.6g} (DataSet)", xlabel="voxels (dty units)")
    im = ax[2].imshow(rec0, origin="lower", cmap="gray", vmin=0, vmax=vmax, extent=ext)
    ax[2].set(xlabel="voxels (dty units)")

    def update(y0: float) -> str:
        im.set_data(recon(y0))
        line.set_ydata([y0, y0])
        ax[2].set_title(f"back-projection, y0 = {y0:.6g} (slider)")
        fig.canvas.draw_idle()
        return f"y0 {y0:.6g}: from the sample's edges {edges['y0']:.4f}, from the mean dty {edges['y0_mean']:.4f}" + (
            f"; the end rows hold {edges['edge'] * 100:.0f}% of the brightest: the sample may extend past the scan"
            if edges["edge"] > 0.1 else "")  # fmt: skip

    return update


def plot_mask(fig: Figure, rec: np.ndarray, disc: np.ndarray | None = None) -> Callable:
    """Draw a back-projection, its histogram, and ``update(threshold)``'s mask (as the fit stage's ``mask="auto"``).

    Parameters
    ----------
    fig
        A matplotlib figure
    rec
        A back-projection, e.g. :func:`anri.index.diagnostics.back_projection` at the y0 chosen
    disc
        The voxels the beam sees at every omega (:func:`anri.index.mask.disc_mask`): outlined in cyan, and the mask lies
        within it
    """
    from .mask import otsu, threshold_mask

    ax = fig.subplots(1, 3)
    ax[1].sharex(ax[0])
    ax[1].sharey(ax[0])
    vmax = np.percentile(rec, 99.5)
    ax[0].imshow(rec, origin="lower", cmap="gray", vmin=0, vmax=vmax)
    ax[0].set(title="back-projection" + ("; cyan: seen at every ω" if disc is not None else ""), xticks=[], yticks=[])
    im = ax[1].imshow(rec, origin="lower", cmap="gray", vmin=0, vmax=vmax)
    ax[1].set(xticks=[], yticks=[])
    if disc is not None:
        for a in ax[:2]:
            a.contour(disc.astype(float), levels=[0.5], colors="c", linewidths=0.8)
    ax[2].hist(rec.ravel(), bins=200, log=True, color="0.5")
    ax[2].axvline(otsu(rec), color="k", ls=":", label="Otsu")
    v = ax[2].axvline(otsu(rec), color="tab:red", label="threshold")
    ax[2].set(xlabel="back-projection", ylabel="voxels")
    ax[2].legend(fontsize=8)

    def update(threshold: float) -> str:
        m = threshold_mask(rec, threshold) & (True if disc is None else disc)
        im.set_data(np.where(m, rec, np.nan))
        v.set_xdata([threshold, threshold])
        ax[1].set_title(f"mask: {m.sum()} of {m.size} voxels")
        fig.canvas.draw_idle()
        return f"mask: {m.sum()} of {m.size} voxels" + ("" if disc is None else f" (the disc: {disc.sum()})")

    return update


def plot_lit(fig: Figure, curve: dict, scan: dict, etacut: float) -> Callable:
    """Draw the lit area and intensity held against the lit cut, and ``update(lit, ring)``'s lit map of one ring.

    Parameters
    ----------
    fig
        A matplotlib figure
    curve
        From :func:`anri.index.diagnostics.lit_curve`
    scan
        The histogram's scan
    etacut
        Drawn where ``|sin eta|`` = etacut
    """
    from matplotlib.colors import LogNorm

    ax = fig.subplots(1, 3, width_ratios=[1, 1.3, 1.3])
    ax[2].sharex(ax[1])
    ax[2].sharey(ax[1])
    Hs, med = curve["per_ring"], curve["median"]
    ax[0].loglog(curve["cuts"], curve["area"], label="lit area")
    ax[0].loglog(curve["cuts"], curve["held"], label="intensity held")
    v = ax[0].axvline(1.0, color="tab:red")
    ax[0].set(xlabel="lit cut (x median non-empty bin)", ylabel="fraction", title=f"median non-empty bin: {med:.3g}")
    ax[0].legend()
    e = (scan["om0"], scan["om0"] + scan["n_o"] * B_O, -180, 180)
    pos = Hs[Hs > 0]
    im_i = ax[1].imshow(np.ma.masked_less_equal(Hs[0], 0), aspect="auto", origin="lower", cmap=_log_cmap(),
                        norm=LogNorm(pos.min(), pos.max()), extent=e, interpolation="nearest")  # fmt: skip
    ax[1].set(xlabel="ω (°)", ylabel="η (°)")
    im_l = ax[2].imshow(np.zeros(Hs[0].shape), aspect="auto", origin="lower", cmap="gray_r", vmin=0, vmax=1, extent=e,
                        interpolation="nearest")  # fmt: skip
    ax[2].set(xlabel="ω (°)")
    ec = np.degrees(np.arcsin(etacut))
    for a in ax[1:]:
        for el in (ec, -ec, 180 - ec, ec - 180):
            a.axhline(el, color="tab:cyan", lw=0.6, ls=":")
    total = Hs.sum()

    def update(lit: float, ring: int = 0) -> str:
        on = Hs > lit * med
        v.set_xdata([lit, lit])
        im_i.set_data(np.ma.masked_less_equal(Hs[ring], 0))
        im_l.set_data(on[ring].astype(np.float32))
        ax[1].set_title(f"ring {ring}: intensity, rows summed")
        ax[2].set_title(f"ring {ring}: lit, {on[ring].mean() * 100:.2f}% of its bins")
        fig.canvas.draw_idle()
        return (f"lit {lit:.3g}: {on.mean() * 100:.2f}% of the bins, holding {Hs[on].sum() / total * 100:.1f}% of the "
                f"intensity; per ring {np.round(on.mean((1, 2)) * 100, 2)}%")  # fmt: skip

    return update


def plot_etacut(fig: Figure, prof: dict) -> Callable:
    """Draw the intensity per eta, and ``update(etacut)``'s eta bins left out.

    Parameters
    ----------
    fig
        A matplotlib figure
    prof
        From :func:`anri.index.diagnostics.eta_profile`
    """
    eta, p = prof["eta"], prof["profile"]
    ax = fig.subplots()
    ax.plot(eta, p, lw=0.8, color="k")
    ax.set(xlabel="η (°)", ylabel="intensity, all rings", xlim=(-180, 180), title="red: not used")
    a2 = ax.twinx()
    a2.semilogy(eta, 1 / np.maximum(np.abs(np.sin(np.radians(eta))), 1e-3), color="tab:blue", lw=0.8)
    a2.set_ylim(1, 50)
    a2.set_ylabel("ω smearing, 1 / |sin η|", color="tab:blue")
    shade: list = []

    def update(etacut: float) -> str:
        cut = np.abs(np.sin(np.radians(eta))) <= etacut
        for art in shade:
            art.remove()
        shade[:] = [ax.fill_between(eta, 0, p.max(), where=cut, color="tab:red", alpha=0.2)]
        fig.canvas.draw_idle()
        return (f"etacut {etacut:.2f}: {2 * np.arcsin(etacut) / np.pi * 100:.1f}% of the reflections and "
                f"{p[cut].sum() / p.sum() * 100:.1f}% of the measured intensity not used; spots smeared up to "
                f"{1 / max(etacut, 1e-3):.1f} x in ω are kept")  # fmt: skip

    return update


def plot_grid(fig: Figure, chances: dict, grid: float | None = None) -> Callable:
    """Draw the chance completeness and size of each grid step, and ``update(max_chance)``'s automatic step.

    Parameters
    ----------
    fig
        A matplotlib figure
    chances
        From :func:`anri.index.diagnostics.grid_chances`
    grid
        A fixed step, if given (then max_chance does not choose it)
    """
    from .predict import GRID_STEPS

    ax = fig.subplots()
    xs = [str(s) for s in chances]
    bars = ax.bar(xs, [c[0] for c in chances.values()])
    h = ax.axhline(0.5, color="k", ls=":")
    ax.set(xlabel="grid step (°)", ylabel="chance completeness", ylim=(0, 1))
    a2 = ax.twinx()
    a2.semilogy(xs, [c[1] for c in chances.values()], "o-", color="0.4")
    a2.set_ylabel("orientations", color="0.4")

    def update(max_chance: float) -> str:
        step = grid if grid is not None else next((s for s, c in chances.items() if c[0] <= max_chance),
                                                  GRID_STEPS[-1])  # fmt: skip
        for b, s in zip(bars, chances):
            b.set_color("tab:red" if s == step else "tab:blue")
        h.set_ydata([max_chance, max_chance])
        ax.set_title(f"grid step {step}°" + (" (fixed)" if grid is not None else " (automatic)"))
        fig.canvas.draw_idle()
        return f"grid step {step} deg"

    return update


def plot_grid_deviances(fig: Figure, deviances: dict, min_lr: float) -> None:
    """Draw the orientation fit's deviance and likely orientations against the grid step.

    Parameters
    ----------
    fig
        A matplotlib figure
    deviances
        From :func:`anri.index.diagnostics.grid_deviances`
    """
    steps = list(deviances)
    ax = fig.subplots(1, 2)
    ax[0].plot(steps, [deviances[s][1] / deviances[min(steps)][1] for s in steps], "o-")
    ax[0].set(xlabel="grid step (°)", ylabel=f"deviance / deviance at {min(steps)}°",
              title="orientation fit to the row-summed data")  # fmt: skip
    ax[1].plot(steps, [deviances[s][2] for s in steps], "o-", color="0.4")
    ax[1].set(xlabel="grid step (°)", ylabel=f"orientations with likelihood ratio > {min_lr:g}", yscale="log")
    for a in ax:
        a.invert_xaxis()


def plot_completeness(fig: Figure, comp: np.ndarray, chance: float, halfway: float, keep: int) -> Callable:
    """Draw the completeness of every grid orientation, and ``update(prune, min_comp)``'s cut (as the filter makes it).

    Parameters
    ----------
    fig
        A matplotlib figure
    comp
        A search stage file's completeness
    chance, halfway
        Its results: the default min_comp of prune="likelihood" and of prune="completeness"
    keep
        The filter's keep
    """
    from .predict import select_by_completeness

    ax = fig.subplots(1, 2, sharex=True)
    ax[0].hist(comp, bins=100, log=True, color="tab:blue")
    ax[0].axvline(chance, color="k", ls=":", label="chance")
    ax[0].axvline(halfway, color="0.5", ls="--", label="halfway")
    lines = [a.axvline(0.0, color="tab:red", label="min_comp") for a in ax]
    ax[0].set(xlabel="completeness", ylabel="orientations")
    ax[0].legend()
    srt = np.sort(comp)
    cs = np.linspace(0, 1, 201)
    ax[1].semilogy(cs, np.maximum(comp.size - np.searchsorted(srt, cs), 0.5), color="tab:blue")
    ax[1].set(xlabel="min_comp", ylabel="orientations above")

    def update(prune: str, min_comp: float) -> str:
        for ln in lines:
            ln.set_xdata([min_comp, min_comp])
        fig.canvas.draw_idle()
        if prune == "likelihood":
            return f"{int(np.sum(comp > min_comp))} orientations above {min_comp:.2f} go to the orientation fit"
        n = int(np.sum(comp >= min_comp))
        return f"{n} orientations at or above {min_comp:.2f}; {len(select_by_completeness(comp, min_comp, keep))} kept"

    return update


def plot_likelihood(fig: Figure, fitted: np.ndarray, lr: np.ndarray, comp: np.ndarray, near: dict, step: float,
                    min_comp: float) -> Callable:  # fmt: skip
    """Draw the likelihood ratios of the orientations fitted, and ``update(min_lr, keep)``'s choice.

    The choice is :func:`anri.index.stages.select_by_likelihood`, as the filter stage makes it.

    Parameters
    ----------
    fig
        A matplotlib figure
    fitted, lr, comp
        A filter stage file's orientations fitted (grid indices), their likelihood ratios and completeness
    near
        From :func:`anri.index.diagnostics.nearest_stronger` on them
    step
        The grid step (the population radius is 1.8 steps)
    min_comp
        The completeness above which orientations were fitted
    """
    ax = fig.subplots(1, 3)
    ax[2].sharey(ax[0])
    srt = np.sort(lr)
    ax[0].scatter(comp, np.maximum(lr, 0.1), s=1, alpha=0.4)
    h = [ax[0].axhline(25, color="tab:red", label="min_lr"), ax[2].axhline(25, color="tab:red")]
    ax[0].axvline(min_comp, color="tab:orange", ls="--", label="min_comp")
    ax[0].set(yscale="log", xlabel="completeness", ylabel="likelihood ratio")
    ax[0].legend(loc="upper left", fontsize=8)
    ts = np.logspace(-1, max(1, np.log10(max(lr.max(), 10))), 100)
    ax[1].loglog(ts, np.maximum(lr.size - np.searchsorted(srt, ts, side="right"), 0.5))
    v = ax[1].axvline(25, color="tab:red")
    hk = ax[1].axhline(1, color="0.5", ls=":", label="keep")
    ax[1].set(xlabel="min_lr", ylabel="orientations kept")
    ax[1].legend(fontsize=8)
    order, dist = near["order"], near["distance"]
    lr_top = np.maximum(lr[order], 0.1)
    out = ax[2].scatter(dist, lr_top, s=2, color="0.75", label="not kept")
    inn = ax[2].scatter(dist, lr_top, s=3, color="tab:blue", label="kept")
    ax[2].axvline(1.8 * step, color="0.4", ls="--", label="population radius")
    ax[2].set(xscale="log", xlabel="misorientation to the nearest stronger orientation (°)",
              title=f"the {len(order)} strongest")  # fmt: skip
    ax[2].legend(fontsize=8, loc="lower right")

    def update(min_lr: float, keep: int) -> str:
        kept = select_by_likelihood(fitted, lr, min_lr, keep)
        sel = np.isin(fitted[order], kept)
        for ln in h:
            ln.set_ydata([min_lr, min_lr])
        v.set_xdata([min_lr, min_lr])
        hk.set_ydata([keep, keep])
        ax[1].set_ylim(0.5, max(2 * len(lr), 2 * keep))
        inn.set_offsets(np.c_[dist[sel], lr_top[sel]])
        out.set_offsets(np.c_[dist[~sel], lr_top[~sel]])
        fig.canvas.draw_idle()
        n_ok = int(np.sum(lr > min_lr))
        return (f"{n_ok} with likelihood ratio > {min_lr:.3g}; {len(kept)} kept"
                + (f" (capped by keep = {keep})" if n_ok > keep else "")
                + f"; {int(np.sum(sel & (dist > 1.8 * step)))} kept peaks (beyond 1.8 grid steps from anything stronger)")  # fmt: skip

    return update


def plot_convergence(fig: Figure, record: dict, f: np.ndarray, fitted: np.ndarray, occupied: np.ndarray,
                     nr: int) -> None:  # fmt: skip
    """Draw how the occupancy fit converged, and whether the voxels wanted more candidates.

    Parameters
    ----------
    fig
        A matplotlib figure
    record
        From :func:`anri.index.diagnostics.convergence_recorder`, after the fit
    f
        [Nv, cand] the fit stage file's occupancies
    fitted, occupied
        [Nv] the voxels fitted (the mask) and those occupied
    nr
        Voxels per side
    """
    from matplotlib.colors import LogNorm

    from .diagnostics import candidate_shares

    dev = record["deviance"]
    its = list(record["moved"])
    n_fit = int(fitted.sum())
    moved = {i: m[:n_fit] for i, m in record["moved"].items()}  # the fit's arrays are padded past the voxels fitted
    ax = fig.subplots(1, 4)
    ax[0].plot(np.arange(1, len(dev) + 1), dev)
    ax[0].set(xscale="log", yscale="log", xlabel="iteration", ylabel="deviance")
    a2 = ax[0].twinx()
    a2.semilogy(np.arange(2, len(dev) + 1), np.maximum(-np.diff(dev) / np.array(dev[1:]), 1e-9), color="tab:red",
                lw=0.8)  # fmt: skip
    a2.set_ylabel("relative decrease per iteration", color="tab:red")
    occ_fit = occupied[fitted]
    if its:
        for q, ls in ((50, "-"), (90, "--"), (99, ":")):
            ax[1].semilogy(its, [np.percentile(moved[i][occ_fit], q) for i in its], "o" + ls, color="k",
                           ms=3, label=f"{q}th percentile")  # fmt: skip
        ax[1].set(xscale="log", xticks=its, xticklabels=[str(i) for i in its], xlabel="iteration",
                  ylabel="share moved per iteration", title="occupancy moving between orientations")  # fmt: skip
        ax[1].minorticks_off()
        ax[1].legend(fontsize=8)
        last = np.full(len(fitted), np.nan)
        last[fitted] = moved[its[-1]]
        im = ax[2].imshow(np.where(occupied, last, np.nan).reshape(nr, nr), origin="lower", cmap="Reds",
                          norm=LogNorm(1e-5, 1e-1))  # fmt: skip
        ax[2].set(title=f"share moved per iteration, at {its[-1]}", xticks=[], yticks=[])
        fig.colorbar(im, ax=ax[2], shrink=0.8)
    share = candidate_shares(f, occupied)
    n_cand = share.shape[1]
    for q, ls in ((50, "-"), (90, "--"), (99, ":")):
        ax[3].semilogy(np.arange(1, n_cand + 1), np.maximum(np.percentile(share, q, 0), 1e-6), ls, color="k",
                       label=f"{q}th percentile")  # fmt: skip
    ax[3].set(xlabel="candidate rank", ylabel="share of the voxel's occupancy", title=f"cand = {n_cand}")
    ax[3].legend(fontsize=8)


def plot_row_ratios(fig: Figure, dty: np.ndarray, row_ratio: np.ndarray, ring_ratio: np.ndarray) -> None:
    """Draw measured / fitted intensity per dty row, and fitted / measured per ring.

    Parameters
    ----------
    fig
        A matplotlib figure
    dty, row_ratio
        Row centres and a fit stage file's row_ratio
    ring_ratio
        From :func:`anri.index.diagnostics.ring_ratios`
    """
    ax = fig.subplots(1, 2, width_ratios=[2, 1])
    ax[0].plot(dty, row_ratio, ".-", ms=3)
    ax[0].axhline(1, color="k", lw=0.5)
    ax[0].set(xlabel="dty", ylabel="measured / fitted", title="per dty row")
    ax[1].bar(np.arange(len(ring_ratio)), ring_ratio)
    ax[1].set(xlabel="ring", ylabel="fitted / measured", title="per ring")


def plot_residual(fig: Figure, H: np.ndarray, model: np.ndarray, n_rings: int, scan: dict) -> Callable:
    """Draw data, model and residual; ``update(ring, eta)`` picks the ring and the eta window of the sinograms.

    Top: one ring with the rows summed (eta against omega). Bottom: the sinogram of an eta window. Compact positive
    spots in the residual are intensity no kept orientation explains; negative ones are predicted where there is
    nothing.

    Parameters
    ----------
    fig
        A matplotlib figure
    H, model
        A histogram stage file's data and a fit stage file's model (``save_model``)
    n_rings, scan
        The histogram's
    """
    from matplotlib.colors import LogNorm

    n_e, n_o, nk = scan["n_e"] // R_E, scan["n_o"] // R_O, scan["n_rows"]
    H4 = np.asarray(H).reshape(n_rings, n_e, n_o, nk)
    M4 = np.asarray(model).reshape(n_rings, n_e, n_o, nk)
    eta_c = -180.0 + (np.arange(n_e) + 0.5) * B_E * R_E
    ax = fig.subplots(2, 3, sharex=True)
    for row in range(2):
        for j in (1, 2):
            ax[row, j].sharey(ax[row, 0])
    om_ext = (scan["om0"], scan["om0"] + n_o * B_O * R_O)
    exts = (
        (*om_ext, -180, 180),
        (*om_ext, scan["dty0"] - scan["ystep"] / 2, scan["dty0"] + (nk - 0.5) * scan["ystep"]),
    )
    shapes = ((n_e, n_o), (nk, n_o))
    ims = []
    for row in range(2):
        z = np.zeros(shapes[row])
        ims.append([ax[row, j].imshow(z, aspect="auto", origin="lower", extent=exts[row], interpolation="nearest",
                                      cmap=_log_cmap() if j < 2 else "RdBu_r") for j in range(3)])  # fmt: skip
        fig.colorbar(ims[row][2], ax=ax[row, 2], shrink=0.8)
        ax[row, 0].set_ylabel(("η (°)", "dty")[row])
        for j in range(3):
            ax[row, j].set_xlabel("ω (°)")
    eta_lines = [[a.axhline(0.0, color="c", lw=0.6) for _ in range(2)] for a in ax[0]]

    def update(ring: int, eta: tuple) -> str:
        ie = (eta_c >= eta[0]) & (eta_c < eta[1])
        pairs = ((H4[ring].sum(2), M4[ring].sum(2)), (H4[ring, ie].sum(0).T, M4[ring, ie].sum(0).T))
        for row, (dd, mm) in enumerate(pairs):
            top = max(dd.max(), 1.0)
            for j, im in enumerate((dd, mm)):
                ims[row][j].set_data(np.ma.masked_less_equal(im, 0))
                ims[row][j].set_norm(LogNorm(top * 1e-2, top))
            r = dd - mm
            vv = max(np.percentile(np.abs(r), 99.9), 1e-9)
            ims[row][2].set_data(r)
            ims[row][2].set_clim(-vv, vv)
            for j, t in enumerate(("data", "model", "data − model")):
                ax[row, j].set_title(f"ring {ring}, {t}" + ("" if row == 0 else f", η {eta[0]}..{eta[1]}°"))
        for pair in eta_lines:
            for ln, e in zip(pair, eta):
                ln.set_ydata([e, e])
        fig.canvas.draw_idle()
        return f"ring {ring}, eta {eta[0]}..{eta[1]} deg"

    return update


def plot_populations(fig: Figure, tot: np.ndarray, mask: np.ndarray, frac: np.ndarray, spread: np.ndarray,
                     step: float, nr: int) -> Callable:  # fmt: skip
    """Draw the total occupancy and the populations, and ``update(occupied, min_frac)``'s choice.

    The choice is :func:`anri.index.stages.select_populations`, as the populations stage makes it.

    Parameters
    ----------
    fig
        A matplotlib figure
    tot, mask
        [Nv] each voxel's total occupancy and the voxels fitted (a fit stage file)
    frac, spread
        [Nv, P] the populations' fractions and spreads (:func:`anri.index.populations.populations`)
    step
        The grid step (spreads include it)
    nr
        Voxels per side
    """
    p99 = np.percentile(tot[mask], 99)
    ax = fig.subplots(2, 3)
    maps = (ax[0, 0], ax[0, 2], ax[1, 0], ax[1, 2])
    for a in maps[1:]:
        a.sharex(maps[0])
        a.sharey(maps[0])
    im = ax[0, 0].imshow(tot.reshape(nr, nr), origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax[0, 0], shrink=0.8)
    im_out = ax[0, 0].imshow(np.zeros((nr, nr, 4)), origin="lower")
    ax[0, 0].set(title="total occupancy; grey: not occupied")
    ax[0, 1].hist(tot / p99, bins=np.linspace(0, max(1.5, (tot / p99).max()), 150), log=True)
    v_occ = ax[0, 1].axvline(0.2, color="tab:red")
    ax[0, 1].set(xlabel="total occupancy / 99th percentile", ylabel="voxels")
    im_n = ax[0, 2].imshow(np.zeros((nr, nr)), origin="lower", cmap="Blues", vmin=0, vmax=N_POP)
    ax[0, 2].set(title="populations per voxel")
    fig.colorbar(im_n, ax=ax[0, 2], shrink=0.8)
    im_f2 = ax[1, 0].imshow(np.zeros((nr, nr)), origin="lower", cmap="magma", vmin=0, vmax=0.5)
    ax[1, 0].set(title="fraction of the second population")
    fig.colorbar(im_f2, ax=ax[1, 0], shrink=0.8)
    hist_f2 = ax[1, 1]
    v_frac = hist_f2.axvline(0.1, color="tab:red")
    hist_f2.set(xlabel="fraction of populations 2 to 4", ylabel="populations")
    im_sp = ax[1, 2].imshow(np.zeros((nr, nr)), origin="lower", cmap="viridis", vmin=0, vmax=max(2 * step, 1e-3))
    ax[1, 2].set(title="spread of the main population (°)")
    fig.colorbar(im_sp, ax=ax[1, 2], shrink=0.8)
    for a in maps:
        a.set(xticks=[], yticks=[])
    drawn: list = []

    def update(occupied: float, min_frac: float) -> str:
        occ, present = select_populations(tot, mask, frac, occupied, min_frac)
        n_pop = present.sum(1)
        grey = np.zeros((len(tot), 4))
        grey[~occ] = (0.85, 0.85, 0.85, 0.85)
        im_out.set_data(grey.reshape(nr, nr, 4))
        im_n.set_data(np.where(occ, n_pop, np.nan).reshape(nr, nr))
        im_f2.set_data(np.where(occ, np.where(present[:, 1], frac[:, 1], 0), np.nan).reshape(nr, nr))
        im_sp.set_data(np.where(occ, spread[:, 0], np.nan).reshape(nr, nr))
        for art in drawn:
            art.remove()
        f2 = frac[occ, 1:].ravel()
        drawn[:] = list(hist_f2.hist(f2[f2 > 0], bins=np.linspace(0.0, 0.5, 101), log=True, color="tab:blue")[2])
        v_occ.set_xdata([occupied, occupied])
        v_frac.set_xdata([min_frac, min_frac])
        fig.canvas.draw_idle()
        return (f"{occ.sum()} of {len(tot)} voxels occupied; populations per occupied voxel: "
                + ", ".join(f"{k}: {np.mean(n_pop[occ] == k) * 100:.1f}%" for k in range(1, N_POP + 1)))  # fmt: skip

    return update


def plot_main_population(fig: Figure, completeness: np.ndarray, ipf: np.ndarray | None, nr: int) -> None:
    """Draw the main population's IPF colours (if given) and its completeness.

    Parameters
    ----------
    fig
        A matplotlib figure
    completeness
        [Nv] the main population's completeness (NaN where not occupied)
    ipf
        [nr, nr, 3] IPF colours in reconstruction order, or None
    nr
        Voxels per side
    """
    ax = fig.subplots(1, 2, sharex=True, sharey=True)
    im = ax[1].imshow(completeness.reshape(nr, nr), origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax[1].set(title="completeness of the main population", xticks=[], yticks=[])
    fig.colorbar(im, ax=ax[1], shrink=0.8)
    if ipf is not None:
        ax[0].imshow(ipf, origin="lower")
        ax[0].set(title="main population, IPF-Z")
    else:
        ax[0].set(title="no IPF map (needs ImageD11 and orix)")
    ax[0].set(xticks=[], yticks=[])
