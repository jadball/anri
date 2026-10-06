"""Shapes of the peaks in some dty rows of a sparse dataset: are they single-maximum arcs (bananas).

    python peak_shapes.py <dataset.h5> [--rows i,j,...] [--png out.png] [--min-sum 200]

Works on any ImageD11 DataSet (a rendered phantom or real data). For each requested row (0-based scan groups, default:
all in the sparse file), the sparse pixels are joined into peaks: connected in (frame, slow, fast), 26-neighbourhood.
For each peak with summed intensity >= --min-sum:

- its extent in eta and omega: the range of pixels above 10% of the peak's maximum, and the intensity-weighted rms;
- its maxima: pixels at least as bright as all their neighbours (missing neighbours count as 0), and how many of
  those are at least half the peak's maximum ("major" maxima). These are counted on raw pixels, so detector pixel
  sampling (a pixel is ~0.1-0.2 deg wide in eta on low rings) and frame steps inflate them on long arcs: look at the
  gallery to judge shapes.

Printed: the share of peaks with 1, 2, 3+ major maxima and the extents, binned by |sin eta|. With --png, a gallery of
the largest peaks (summed over frames on the detector, and in (eta, omega)).
"""

import argparse

import anri.utils

anri.utils.setup()
import h5py
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import anri.index as ix
import anri.io

p = argparse.ArgumentParser()
p.add_argument("dataset", help="ImageD11 DataSet .h5")
p.add_argument("--rows", help="scan groups to look at, 0-based, comma-separated (default: all)")
p.add_argument("--png", help="write a gallery of the largest peaks here")
p.add_argument("--min-sum", type=float, default=200.0, help="ignore peaks with less summed intensity (default 200)")
args = p.parse_args()

ds = anri.io.read_dataset(args.dataset)
geo, _, _ = anri.io.read_pars_json(ds["parfile"])
geom = anri.io.geom_from_pars(geo, 0.0, 0.0, 0.0, 0.0, sig_beam=1.0, voxel_size=1.0)  # only the detector is used

peaks = []  # per peak: dict of arrays/values
with h5py.File(ds["sparsefile"], "r") as h:
    groups = sorted(h.keys(), key=lambda k: float(k))
    rows = [int(i) for i in args.rows.split(",")] if args.rows else range(len(groups))
    for r in rows:
        g = h[groups[r]]
        nnz = g["nnz"][()]
        frame = np.repeat(np.arange(len(nnz)), nnz)
        s, f, v = g["row"][()].astype(np.int64), g["col"][()].astype(np.int64), g["intensity"][()].astype(float)
        om = np.asarray(g[f"measurement/{ds['omegamotor']}"][()], float)[frame]
        ns, nf = int(s.max()) + 2, int(f.max()) + 2
        key = (frame * ns + s) * nf + f
        order = np.argsort(key)
        key, frame, s, f, v, om = key[order], frame[order], s[order], f[order], v[order], om[order]
        n = len(key)
        # 26 neighbours: edges for connected components, and the brightest neighbour of each pixel
        src, dst = [], []
        nbmax = np.zeros(n)
        for df in (-1, 0, 1):
            for dsl in (-1, 0, 1):
                for dfa in (-1, 0, 1):
                    if df == dsl == dfa == 0:
                        continue
                    ok = (s + dsl >= 0) & (f + dfa >= 0)
                    k2 = key + (df * ns + dsl) * nf + dfa
                    j = np.clip(np.searchsorted(key, k2), 0, n - 1)
                    hit = ok & (key[j] == k2)
                    src.append(np.flatnonzero(hit))
                    dst.append(j[hit])
                    nbmax[hit] = np.maximum(nbmax[hit], v[j[hit]])
        src, dst = np.concatenate(src), np.concatenate(dst)
        _, lab = connected_components(coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)), directed=False)
        ang = np.asarray(ix.pixel_angles(s.astype(np.float32), f.astype(np.float32), om.astype(np.float32), geom))
        eta = ang[:, 1]
        is_max = v >= nbmax
        tot = np.bincount(lab, v)
        for c in np.flatnonzero(tot >= args.min_sum):
            m = lab == c
            vm, em, om_c = v[m], eta[m], om[m]
            e0 = em[np.argmax(vm)]
            de = (em - e0 + 180.0) % 360.0 - 180.0  # eta relative to the brightest pixel, no wrap
            top = vm >= 0.1 * vm.max()
            w = vm / vm.sum()
            peaks.append({
                "row": r, "sum": vm.sum(), "eta": e0 + np.sum(w * de),
                "eta_ext": np.ptp(de[top]), "om_ext": np.ptp(om_c[top]),
                "eta_rms": np.sqrt(np.sum(w * (de - np.sum(w * de)) ** 2)),
                "om_rms": np.sqrt(np.sum(w * (om_c - np.sum(w * om_c)) ** 2)),
                "n_max": int(np.sum(is_max[m])), "n_major": int(np.sum(is_max[m] & (vm >= 0.5 * vm.max()))),
                "s": s[m], "f": f[m], "v": vm, "de": de, "om": om_c,
            })  # fmt: skip
        print(f"row {r}: {n} pixels, {lab.max() + 1} connected peaks, {np.sum(tot >= args.min_sum)} with sum >= "
              f"{args.min_sum:g}")  # fmt: skip

if not peaks:
    raise SystemExit("no peaks")
nmaj = np.array([q["n_major"] for q in peaks])
print(
    f"\n{len(peaks)} peaks: major maxima 1: {np.mean(nmaj == 1):.0%}, 2: {np.mean(nmaj == 2):.0%}, "
    f"3+: {np.mean(nmaj >= 3):.0%}; all local maxima per peak, median {np.median([q['n_max'] for q in peaks]):g}"
)
sin = np.abs(np.sin(np.radians([q["eta"] for q in peaks])))
print("\n|sin eta|   peaks  eta extent / rms (deg)   omega extent / rms (deg)   share with 1 major max")
for lo, hi in [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]:
    m = (sin >= lo) & (sin < hi)
    if not m.any():
        continue
    med = {
        k: np.median([q[k] for q, keep in zip(peaks, m) if keep]) for k in ("eta_ext", "eta_rms", "om_ext", "om_rms")
    }
    print(f"{lo:.1f}-{hi:.1f}   {m.sum():6d}   {med['eta_ext']:6.2f} / {med['eta_rms']:5.2f}"
          f"          {med['om_ext']:6.2f} / {med['om_rms']:5.2f}            {np.mean(nmaj[m] == 1):.0%}")  # fmt: skip

if args.png:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    big = sorted(peaks, key=lambda q: -q["sum"])[:8]
    fig, ax = plt.subplots(2, len(big), figsize=(2.6 * len(big), 5.4), layout="constrained")
    for i, q in enumerate(big):
        s0, f0 = q["s"].min(), q["f"].min()
        img = np.zeros((np.ptp(q["s"]) + 1, np.ptp(q["f"]) + 1))
        np.add.at(img, (q["s"] - s0, q["f"] - f0), q["v"])
        ax[0, i].imshow(img, origin="lower", cmap="magma")
        ax[0, i].set_title(f"eta {q['eta']:.0f}, {q['n_major']} major max", fontsize=8)
        ax[0, i].set_xlabel("fast (px)", fontsize=7)
        ax[1, i].hexbin(q["de"], q["om"], C=q["v"], reduce_C_function=np.sum, gridsize=30, cmap="magma")
        ax[1, i].set_xlabel("eta - eta of max (deg)", fontsize=7)
        ax[1, i].set_ylabel("omega (deg)", fontsize=7)
    ax[0, 0].set_ylabel("slow (px), summed over frames", fontsize=7)
    fig.savefig(args.png, dpi=110)
    print(f"-> {args.png}")
