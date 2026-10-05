"""Peak widths in omega from ImageD11's peaks table, against |sin eta|.

Is the width the instrument (beam convergence) or an orientation spread?

    python peak_widths.py <analysisroot> <sample> <dataset> [--phase NAME] [--rows 40]

The peaks table holds every 2D peak (one frame) with the label of the 3D peak it belongs to. For each 3D peak within one
dty row: its total intensity, mean eta and 2theta, number of frames, and FWHM in omega from the intensity-weighted rms
over its frames (less a frame's own width). Only clean peaks count: one 2D peak per frame (no overlap within a frame)
and not touching the ends of the scan.

How the omega width depends on eta tells the causes apart:

- a rotational spread (mosaic) widens omega as about 1 / |sin eta|: narrowest at 3 and 9 o'clock;
- horizontal beam convergence acts as a rotation about the omega axis: the same width at every eta.

Only the middle --rows rows are read (the table's slices for those rows), so memory stays small.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

FWHM = 2 * np.sqrt(2 * np.log(2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("analysisroot")
    p.add_argument("sample")
    p.add_argument("dataset")
    p.add_argument("--phase")
    p.add_argument("--rings", type=int, default=6)
    p.add_argument("--rows", type=int, default=40, help="dty rows, around the middle of the scan")
    args = p.parse_args()

    import h5py
    from ImageD11.sinograms.dataset import load

    import anri.index as ix
    import anri.io

    dsname = f"{args.sample}_{args.dataset}"
    dsfile = os.path.join(args.analysisroot, args.sample, dsname, f"{dsname}_dataset.h5")
    ds = load(dsfile)
    n_rows, n_frames = ds.omega.shape
    step = float(np.median(np.abs(np.diff(ds.obinedges))))
    k0 = max(0, n_rows // 2 - args.rows // 2)
    k1 = min(n_rows, k0 + args.rows)

    # the table is stored scan by scan (ipk points to each scan's first 2D peak): read only the middle rows' slice
    with h5py.File(ds.pksfile, "r") as h:
        g = h["pks2d"]
        ipk = g["ipk"][:]
        a, b = int(ipk[k0]), int(ipk[k1])
        s1, s_i, sr_i, sc_i, frm = g["pk_props"][:, a:b]
        label = g["glabel"][a:b]
    print(f"{dsname}: rows {k0}..{k1 - 1} of {n_rows}, {n_frames} frames of {step:g} deg; {b - a} 2D peaks")

    pk = {"s_raw": sr_i / s_i, "f_raw": sc_i / s_i, "omega": ds.omega_for_bins.flat[frm], "dty": ds.dty.flat[frm],
          "Number_of_pixels": s1, "sum_intensity": s_i.astype(float), "spot3d_id": label}  # fmt: skip
    cf = ds.get_colfile_from_peaks_dict(peaks_dict=pk)
    ds.update_colfile_pars(cf, phase_name=args.phase)
    row, frame = frm // n_frames, frm % n_frames

    # 3D peaks within one row: group the 2D peaks by (label, row)
    _, grp = np.unique(label.astype(np.int64) * n_rows + row, return_inverse=True)
    grp = grp.ravel()
    n = grp.max() + 1
    w = cf.sum_intensity
    I = np.bincount(grp, w, n)
    om = np.bincount(grp, w * cf.omega, n) / I
    var = np.maximum(np.bincount(grp, w * cf.omega**2, n) / I - om**2, 0.0)
    fwhm = FWHM * np.sqrt(np.maximum(var - step**2 / 12, 0.0))
    sin_eta = np.abs(np.bincount(grp, w * np.sin(np.radians(cf.eta)), n) / I)
    tth = np.bincount(grp, w * cf.tth, n) / I
    n2d = np.bincount(grp, minlength=n)
    n_fr = np.bincount(np.unique(grp.astype(np.int64) * n_frames + frame) // n_frames, minlength=n)
    clean = n2d == n_fr  # one 2D peak per frame
    edge = np.zeros(n, bool)
    edge[grp[(frame == 0) | (frame == n_frames - 1)]] = True
    clean &= ~edge
    print(
        f"{n} 3D peaks per row, {clean.sum()} clean ({100 * clean.mean():.0f}%: one 2D peak per frame, off the scan ends)"
    )

    geo, phase, cell = anri.io.read_pars_json(ds.parfile, args.phase)
    lpars = np.array([cell[k] for k in ("cell__a", "cell__b", "cell__c", "cell_alpha", "cell_beta", "cell_gamma")])
    rings = ix.ring_table(lpars, int(cell["cell_lattice_[P,A,B,C,I,F,R]"]), geo["wavelength"], args.rings)
    r_tth = np.asarray(rings["tth"])
    ring = np.argmin(np.abs(tth[:, None] - r_tth[None, :]), axis=1)
    near = np.abs(tth - r_tth[ring]) < 0.5 * np.min(np.diff(np.concatenate([[0.0], r_tth])))

    s_edges = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("omega FWHM (deg), median over the brighter half of clean peaks, by ring and |sin eta| "
          "(n peaks; median frames). Spread: ~1/|sin eta|. Convergence: flat.")  # fmt: skip
    print(
        "ring  2theta   " + "  ".join(f"|sin eta| {lo:.2f}-{hi:.2f}      " for lo, hi in zip(s_edges[:-1], s_edges[1:]))
    )
    for r in range(len(r_tth)):
        m = clean & near & (ring == r)
        if not m.any():
            continue
        m &= I >= np.median(I[m])
        cols = []
        for lo, hi in zip(s_edges[:-1], s_edges[1:]):
            mm = m & (sin_eta >= lo) & (sin_eta < hi + 1e-9)
            if mm.sum() < 10:
                cols.append(f"{'-':>8} ({mm.sum():5d};  -)         ")
                continue
            cols.append(f"{np.median(fwhm[mm]):8.3f} ({mm.sum():5d}; {np.median(n_fr[mm]):4.1f})        ")
        print(f"{r:4d} {r_tth[r]:7.3f}   " + "".join(cols))
    m = clean & near
    m &= I >= np.percentile(I[m], 90)
    for lo, hi in zip(s_edges[:-1], s_edges[1:]):
        mm = m & (sin_eta >= lo) & (sin_eta < hi + 1e-9)
        if mm.sum():
            q = np.percentile(fwhm[mm], [10, 50, 90])
            print(f"brightest 10%, |sin eta| {lo:.2f}-{hi:.2f}: {mm.sum():6d} peaks, FWHM {q[0]:.3f} / {q[1]:.3f} / "
                  f"{q[2]:.3f} deg (10/50/90th), frames {np.median(n_fr[mm]):.0f}")  # fmt: skip

    # Selection check: does the clean filter drop the broad peaks at low |sin eta|? Brighter half per ring of all peaks
    # off the scan ends, clean or not. Overlapping peaks' widths include their neighbours, so they read wide; what
    # matters is whether they are much wider at low |sin eta| than the clean ones.
    m = near & ~edge
    bright = np.zeros(n, bool)
    for r in range(len(r_tth)):
        mr = m & (ring == r)
        if mr.any():
            bright |= mr & (I >= np.median(I[mr]))
    hi_bin = bright & clean & (sin_eta >= s_edges[-2])
    w_ref, s_ref = np.median(fwhm[hi_bin]), np.median(sin_eta[hi_bin])
    print(f"\nselection check (brighter half per ring, clean or not). Isotropic spread: clean FWHM ~ {w_ref:.3f} x "
          f"{s_ref:.2f} / |sin eta|, from the clean peaks at |sin eta| > {s_edges[-2]}")  # fmt: skip
    print("|sin eta|    share (uniform eta)  clean   predicted   clean FWHM 10/50/90 (frames)    overlapping FWHM 10/50/90 (frames)")  # fmt: skip
    for lo, hi in zip(s_edges[:-1], s_edges[1:]):
        mm = bright & (sin_eta >= lo) & (sin_eta < hi + 1e-9)
        if mm.sum() < 10:
            continue
        uniform = 2 / np.pi * (np.arcsin(hi) - np.arcsin(lo))
        pred = w_ref * s_ref / max(np.median(sin_eta[mm]), 0.02)
        cols = []
        for sel in (mm & clean, mm & ~clean):
            if sel.sum() < 5:
                cols.append(f"{'-':>32}")
                continue
            q = np.percentile(fwhm[sel], [10, 50, 90])
            cols.append(f"{q[0]:6.3f} {q[1]:6.3f} {q[2]:6.3f} ({np.median(n_fr[sel]):3.0f}) n={sel.sum():6d}")
        print(f"{lo:.2f}-{hi:.2f}    {mm.sum() / bright.sum():5.1%} ({uniform:5.1%})     {(mm & clean).sum() / mm.sum():5.1%}"
              f"   {pred:7.3f}    {cols[0]}    {cols[1]}")  # fmt: skip
    print("If the clean median is well under the prediction at low |sin eta| and the overlapping peaks there are broad, "
          "the flat floor is a selection effect; if the overlapping peaks are as narrow, the widths really are flat.")  # fmt: skip
