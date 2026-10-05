"""Diagnose the dty and omega positions of an ImageD11 DataSet, frame by frame (e.g. a 2D fly scan, dty moving).

    python diagnose_positions.py <dataset.h5>

Prints the shapes and ranges, how dty behaves within each row, the encoder's quantisation, whether omega restarts
each row, and a straight-line fit of dty against time (frame index): the speed per frame and per 360 degrees, the
residuals, and where the fitted positions fall relative to the DataSet's dty bins. Saves positions.png in the
current folder.

The line alone cannot tell a stepped scan from continuous motion (a staircase also has a slope, and residuals of
half a step): the span of dty within each row, and where the reading changes, decide that.
"""

import sys

import numpy as np
from ImageD11.sinograms.dataset import load

ds = load(sys.argv[1])
dty, om = np.asarray(ds.dty, float), np.asarray(ds.omega, float)
print(f"shape {dty.shape} (rows, frames per row); dtymotor {ds.dtymotor}, omegamotor {ds.omegamotor}")
print(f"y0 {getattr(ds, 'y0', None)}; ybincens {ds.ybincens[0]:.6g} .. {ds.ybincens[-1]:.6g} ({len(ds.ybincens)}), "
      f"step {np.median(np.diff(ds.ybincens)):.6g}; obinedges {ds.obinedges[0]:.6g} .. {ds.obinedges[-1]:.6g} "
      f"({len(ds.obinedges) - 1} bins), step {np.median(np.diff(ds.obinedges)):.6g}")  # fmt: skip
n_rows, n_fr = dty.shape

# --- within each row
span = dty.max(1) - dty.min(1)
n_unique = np.array([len(np.unique(r)) for r in dty])
om_dir = np.sign(np.median(np.diff(om, axis=1), axis=1))
print(f"\ndty within a row: span min / median / max {span.min():.6g} / {np.median(span):.6g} / {span.max():.6g}; "
      f"distinct values per row min / median / max {n_unique.min()} / {np.median(n_unique):.0f} / {n_unique.max()}")
print(f"omega direction per row: {np.sum(om_dir > 0)} rows increasing, {np.sum(om_dir < 0)} decreasing")
print(f"omega at the start of the first 5 rows: {np.round(om[:5, 0], 4).tolist()}; at their ends: {np.round(om[:5, -1], 4).tolist()}")
print(f"omega step: {np.unique(np.round(np.diff(om, axis=1), 5))[:8].tolist()}")
print(f"first 3 rows, first 12 dty values: {[np.round(r[:12], 6).tolist() for r in dty[:3]]}")

# --- the encoder's quantisation
d_all = dty.ravel()  # time order, if the rows were measured one after another
vals = np.unique(np.round(d_all, 9))
print(f"\ndistinct dty values over the scan: {len(vals)}; spacing between them: "
      f"{np.unique(np.round(np.diff(vals), 7))[:6].tolist()}")  # fmt: skip
change = np.flatnonzero(np.diff(d_all) != 0)
if len(change) > 1:
    gaps = np.diff(change)
    print(f"frames between changes of the reading: min {gaps.min()}, median {np.median(gaps):.0f}, max {gaps.max()}; "
          f"{len(change)} changes in {len(d_all)} frames")

# --- omega over the whole scan: continuous, or restarting each row?
om_all = om.ravel()
jumps = np.flatnonzero(np.abs(np.diff(om_all)) > 10 * np.median(np.abs(np.diff(om_all))))
print(f"\nomega jumps over the scan: {len(jumps)} (at frames {jumps[:5].tolist()}...); "
      f"omega unwrapped spans {np.ptp(np.unwrap(np.radians(om_all), period=2 * np.pi)) * 180 / np.pi:.1f} deg")  # fmt: skip

# --- straight line through dty against time (the frame index)
t = np.arange(len(d_all))
a, b = np.polyfit(t, d_all, 1)
fit = a * t + b
res = d_all - fit
frames_per_turn = 360.0 / np.median(np.abs(np.diff(om, axis=1)))
print(f"\nlinear fit: dty = {b:.6g} + {a:.6g} x frame; per 360 deg ({frames_per_turn:.0f} frames): {a * frames_per_turn:.6g}; "
      f"per row ({n_fr} frames): {a * n_fr:.6g}")  # fmt: skip
print(f"residual rms {res.std():.3g}, max |residual| {np.abs(res).max():.3g} "
      f"(a reading rounded to a quantum q would give max ~q/2)")  # fmt: skip
fit_rows = fit.reshape(n_rows, n_fr)
nominal = np.asarray(ds.ybincens)[: n_rows][:, None]
off = fit_rows - nominal
print(f"fitted dty - row's bin centre: at the row's first frame {np.median(off[:, 0]):.4g}, middle "
      f"{np.median(off[:, n_fr // 2]):.4g}, last {np.median(off[:, -1]):.4g} (median over rows)")  # fmt: skip
edges = np.asarray(ds.ybinedges)
k = np.searchsorted(edges, fit_rows) - 1
print(f"frames whose fitted dty falls outside their row's bin: {np.mean(k != np.arange(n_rows)[:, None]) * 100:.1f}%")

# --- plot
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(3, 1, figsize=(12, 9), layout="constrained")
m = slice(0, min(len(d_all), 5 * n_fr))
ax[0].plot(t[m], d_all[m], ".", ms=1, label="dty reading")
ax[0].plot(t[m], fit[m], label="linear fit")
ax[0].set_ylabel("dty")
ax[0].legend()
ax[0].set_title("first 5 rows")
ax[1].plot(t, res, ",")
ax[1].set_ylabel("reading - fit")
ax[2].plot(t[m], om_all[m], ".", ms=1)
ax[2].set_ylabel("omega")
ax[2].set_xlabel("frame (time order)")
fig.savefig("positions.png", dpi=100)
print("\n-> positions.png")
