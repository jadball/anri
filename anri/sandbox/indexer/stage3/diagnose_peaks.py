"""Measure peak widths in eta and omega from the data alone (no model), to choose stage 3's bins or peak widths.

    python diagnose_peaks.py <analysisroot> <sample> <dataset> [--phase NAME] [--rows 40] [--bins 0.1 <omega step>]

The sparse pixels of the middle --rows dty rows are binned finely in (ring, eta, omega) for each row. Two estimates
of the spots' FWHM, per ring:

- autocorrelation: each row's ring image against itself shifted along omega (or eta). A spot of FWHM w gives a
  central peak sqrt(2) w wide; overlapping spots add a pedestal, taken from the largest lags and subtracted. The bin
  width is corrected for. No threshold, but an intensity-weighted average over the spots.
- spot moments: the connected spots of each row's image and the rms width of each along omega and eta (less the bin's
  own). A distribution (10th, 50th, 90th percentile), but segmentation cuts the tails (too narrow) and overlapping
  spots merge (too wide).

If the two agree, the widths are measured. Stage 3 predicts a spot as a point: bins narrower than these widths let it
fake the width by mixing units (the bleeding at grain boundaries).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

GAUSS_HALF = np.sqrt(2 * np.log(2))  # half width at half maximum of a unit-sigma Gaussian
FWHM = 2 * GAUSS_HALF


def autocorr(ie: np.ndarray, io: np.ndarray, v: np.ndarray, n_o: int, axis: int, lags: np.ndarray) -> np.ndarray:
    """Sum over pixels of v(x) v(x + lag) along one axis (0: eta, 1: omega), from sparse bins (ie, io) of one image."""
    key = ie.astype(np.int64) * n_o + io
    order = np.argsort(key)
    key, v = key[order], v[order]
    step = n_o if axis == 0 else 1
    out = np.zeros(len(lags))
    for n, lag in enumerate(lags):
        target = key + lag * step
        j = np.clip(np.searchsorted(key, target), 0, len(key) - 1)
        hit = key[j] == target
        if axis == 1:
            hit &= (io[order] + lag >= 0) & (io[order] + lag < n_o)  # no wrapping from one eta row into the next
        out[n] = np.sum(v[hit] * v[j[hit]])
    return out


def autocorr_fwhm(a: np.ndarray, lags: np.ndarray, b: float, n_ped: int = 5) -> float:
    """Spot FWHM (deg) from an autocorrelation a at lags 0..L (in bins of b deg), its pedestal from the last n_ped."""
    c = a - a[-n_ped:].mean()
    c = c / c[0]
    below = np.nonzero(c < 0.5)[0]
    if len(below) == 0 or below[0] == 0 or below[0] >= len(c) - n_ped:
        return np.nan  # wider than the lags reach
    i = below[0]
    half = lags[i - 1] + (c[i - 1] - 0.5) / (c[i - 1] - c[i]) * (lags[i] - lags[i - 1])  # lag of half maximum (bins)
    s2 = (half * b / GAUSS_HALF) ** 2  # variance of the autocorrelation's central peak (deg^2)
    sig2 = (s2 - b**2 / 6) / 2  # less the bin's own (a box's autocorrelation has variance b^2 / 6); a spot's is half
    return FWHM * np.sqrt(sig2) if sig2 > 0 else 0.0


def spot_moments(ie: np.ndarray, io: np.ndarray, v: np.ndarray, n_e: int, n_o: int, b_e: float, b_o: float) -> tuple:
    """Connected spots of one image: total intensity, and FWHM (deg) along eta and omega from their rms widths."""
    from scipy import ndimage

    img = np.zeros((n_e, n_o), np.float32)
    np.add.at(img, (ie, io), v)
    lab, n = ndimage.label(img > 0, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    lab_px = lab[ie, io]
    w = np.bincount(lab_px, v, n + 1)[1:]
    me = np.bincount(lab_px, v * ie, n + 1)[1:] / w
    mo = np.bincount(lab_px, v * io, n + 1)[1:] / w
    ve = np.bincount(lab_px, v * ie.astype(float) ** 2, n + 1)[1:] / w - me**2
    vo = np.bincount(lab_px, v * io.astype(float) ** 2, n + 1)[1:] / w - mo**2
    fe = FWHM * b_e * np.sqrt(np.maximum(ve - 1 / 12, 0))
    fo = FWHM * b_o * np.sqrt(np.maximum(vo - 1 / 12, 0))
    return w, fe, fo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("analysisroot")
    p.add_argument("sample")
    p.add_argument("dataset")
    p.add_argument("--phase")
    p.add_argument("--parfile")
    p.add_argument("--rings", type=int, default=6)
    p.add_argument("--rows", type=int, default=40, help="dty rows, around the middle of the scan")
    p.add_argument("--bins", type=float, nargs=2, help="eta and omega bins (deg; default 0.1 and the omega step)")
    p.add_argument("--max-lag", type=float, nargs=2, default=(4.0, 2.0), help="largest lags, eta and omega (deg)")
    p.add_argument("--n-cpu", type=int, default=4)
    args = p.parse_args()

    import anri.utils

    anri.utils.setup(n_cpu=args.n_cpu)
    import h5py
    import jax.numpy as jnp

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import anri.index as ix
    import anri.io
    from fine import sparse_histogram

    dsname = f"{args.sample}_{args.dataset}"
    dsfile = os.path.join(args.analysisroot, args.sample, dsname, f"{dsname}_dataset.h5")
    ds = anri.io.read_dataset(dsfile)
    sparsefile = ds["sparsefile"] if ds["sparsefile"] and os.path.exists(ds["sparsefile"]) else dsfile.replace("_dataset.h5", "_sparse.h5")  # fmt: skip
    geo, phase, cell = anri.io.read_pars_json(args.parfile or ds["parfile"], args.phase)
    lpars = np.array([cell[k] for k in ("cell__a", "cell__b", "cell__c", "cell_alpha", "cell_beta", "cell_gamma")])
    sg = int(cell["cell_lattice_[P,A,B,C,I,F,R]"])
    ybin, yedge, oedge = ds["ybincens"], ds["ybinedges"], ds["obinedges"]
    NK, OM0 = len(ybin), float(oedge[0])
    geom = anri.io.geom_from_pars(geo, 0.0, 1e-4, 1e-4, 1e-4, sig_beam=1.0, voxel_size=float(np.median(np.diff(ybin))))
    geom = {k: jnp.asarray(v, jnp.float32) if np.issubdtype(np.asarray(v).dtype, np.floating) else v for k, v in geom.items()}  # fmt: skip
    rings = ix.ring_table(lpars, sg, geo["wavelength"], args.rings)
    B_E, B_O = args.bins if args.bins else (0.1, float(np.median(np.diff(oedge))))
    N_E, N_O = round(360 / B_E), round(float(oedge[-1] - oedge[0]) / B_O)
    print(f"{dsname}: {NK} rows; bins {B_E:g} x {B_O:g} deg (eta x omega)")

    with h5py.File(sparsefile, "r") as h:
        groups = list(h.keys())
        n_max = max(int(h[g]["nnz"][()].sum()) for g in groups)
    chunk = int(min(1 << 24, 1 << max(10, int(np.ceil(np.log2(max(n_max, 1)))))))

    def stream(gs):  # noqa: ANN001, ANN202
        return anri.io.stream_sparse(sparsefile, yedge, ds["omegamotor"], ds["dtymotor"], chunk, gs, 1, ds["dty"], ds["scans"])  # fmt: skip

    sample = [groups[i] for i in np.unique(np.linspace(0, len(groups) - 1, min(len(groups), 9)).round().astype(int))]
    off, hw = ix.ring_profile(stream(sample), geom, rings["tth"], chunk)
    data = sparse_histogram(stream(groups), geom, rings["tth"], np.abs(off) + hw, OM0, (B_E, B_O, N_E, N_O), NK, chunk)
    start, row, value = data["start"], data["row"], data["value"]
    k0 = max(0, NK // 2 - args.rows // 2)
    rows = np.arange(k0, min(NK, k0 + args.rows))
    lag_e = np.arange(int(round(args.max_lag[0] / B_E)) + 1)
    lag_o = np.arange(int(round(args.max_lag[1] / B_O)) + 1)

    print(f"rows {rows[0]}..{rows[-1]}; FWHM in deg. autocorr: from the intensity-weighted autocorrelation. spots: "
          "10th / 50th / 90th percentile over spots, and the intensity-weighted median")  # fmt: skip
    print(f"widths below about half a bin ({B_E / 2:g} eta, {B_O / 2:g} omega) are not resolved: the spots fit in one bin")
    for r in range(len(rings["tth"])):
        a, b = start[r * N_E * N_O], start[(r + 1) * N_E * N_O]
        cell_r = np.repeat(np.arange(N_E * N_O), np.diff(start[r * N_E * N_O : (r + 1) * N_E * N_O + 1]))
        ie_r, io_r = cell_r // N_O, cell_r % N_O
        row_r, v_r = row[a:b], value[a:b].astype(float)
        ac_e, ac_o = np.zeros(len(lag_e)), np.zeros(len(lag_o))
        W, FE, FO = [], [], []
        for k in rows:
            m = row_r == k
            if not m.any():
                continue
            ac_e += autocorr(ie_r[m], io_r[m], v_r[m], N_O, 0, lag_e)
            ac_o += autocorr(ie_r[m], io_r[m], v_r[m], N_O, 1, lag_o)
            w, fe, fo = spot_moments(ie_r[m], io_r[m], v_r[m], N_E, N_O, B_E, B_O)
            W.append(w), FE.append(fe), FO.append(fo)
        if not W:
            continue
        W, FE, FO = np.concatenate(W), np.concatenate(FE), np.concatenate(FO)
        strong = W >= np.percentile(W, 50)  # the brighter half: faint specks are mostly noise and cut by the threshold

        def wmed(x: np.ndarray, w: np.ndarray) -> float:
            o = np.argsort(x)
            return float(x[o][np.searchsorted(np.cumsum(w[o]), 0.5 * w.sum())])

        pe, po = np.percentile(FE[strong], [10, 50, 90]), np.percentile(FO[strong], [10, 50, 90])
        print(f"ring {r} (2theta {rings['tth'][r]:.3f}): {strong.sum()} spots (brighter half of {len(W)})\n"
              f"   omega: autocorr {autocorr_fwhm(ac_o, lag_o, B_O):.3f}; spots {po[0]:.3f} / {po[1]:.3f} / {po[2]:.3f}, "
              f"weighted {wmed(FO[strong], W[strong]):.3f}\n"
              f"   eta:   autocorr {autocorr_fwhm(ac_e, lag_e, B_E):.3f}; spots {pe[0]:.3f} / {pe[1]:.3f} / {pe[2]:.3f}, "
              f"weighted {wmed(FE[strong], W[strong]):.3f}")  # fmt: skip
