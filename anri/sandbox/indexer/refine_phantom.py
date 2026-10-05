"""Refine the indexer's map entries on the am316l phantom with anri.refine, widening the peaks by a spread first.

    python refine_phantom.py <analysisroot> <sample> <dataset> <..._index_entries.npz> <truth_tmap.h5>
        [--sig 0.3 0.1 0.03] [--iter 5] [--rows N] [--max-frames 63] [--window 3 11 11]

Each round refines every entry's lattice with all entries' sig_rot set to the round's spread (deg), densities fixed
(the indexer decided who owns each voxel). Before and after each round: the main populations' misorientation to the
truth (the closest truth voxel inside the entry's voxel). --rows N uses only the N central dty rows (a quick look).
"""

import argparse
import os
import time

import numpy as np

T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f} s] {msg}", flush=True)


p = argparse.ArgumentParser()
p.add_argument("analysisroot")
p.add_argument("sample")
p.add_argument("dataset")
p.add_argument("entries")
p.add_argument("truth")
p.add_argument("--sig", type=float, nargs="+", default=[0.3, 0.1, 0.03], help="spread per round (deg, sigma)")
p.add_argument("--bin", type=int, nargs="+", default=[5, 2, 1], help="frames summed per round (one per --sig)")
p.add_argument(
    "--method",
    choices=("linear", "entry", "cg"),
    default="linear",
    help="linear: refine_orientations, geometry linearised, cheap sweeps (default; --iter = sweeps per round, frames "
    "not binned); entry: refine_per_entry; cg: refine",
)
p.add_argument("--iter", type=int, default=5)
p.add_argument("--rows", type=int, default=0)
p.add_argument("--max-frames", type=int, default=63)
p.add_argument("--window", type=int, nargs=3, default=(3, 11, 11))
p.add_argument("--beam", type=float, help="beam FWHM as rendered (default: the voxel)")
p.add_argument("--n-cg", type=int, default=15, help="conjugate gradient steps per iteration (default 15)")
p.add_argument("--relin", type=int, default=5, help="linear: accepted sweeps between linearisations")
p.add_argument("--win-px", type=int, default=7, help="linear: window size in pixels (slow and fast)")
p.add_argument("--n-cpu", type=int, default=8)
args = p.parse_args()

import anri.utils  # noqa: E402

anri.utils.setup(n_cpu=args.n_cpu)
import h5py  # noqa: E402
from ImageD11.sinograms.tensor_map import TensorMap  # noqa: E402
from scipy.spatial import KDTree  # noqa: E402

import anri.crystal  # noqa: E402
import anri.fwd  # noqa: E402
import anri.index as ix  # noqa: E402
import anri.io  # noqa: E402
import anri.refine  # noqa: E402

dsname = f"{args.sample}_{args.dataset}"
ds = anri.io.read_dataset(os.path.join(args.analysisroot, args.sample, dsname, f"{dsname}_dataset.h5"))
geo, _, cell = anri.io.read_pars_json(ds["parfile"])
a = cell["cell__a"]
lpars = np.array([a, a, a, 90.0, 90.0, 90.0])
B = anri.crystal.B_matrix(lpars)
ops = anri.crystal.laue_rotations(anri.crystal.symmetry_matrices(225), B)
wl = geo["wavelength"]
# the rendering's instrument (as render_phantom.py), with the indexer's voxels
r_idx = np.load(args.entries.replace("_entries.npz", ".npz"))
vsize = float(np.median(np.diff(np.unique(np.round(r_idx["pos"][:, 0], 6)))))
beam = args.beam or vsize
geom = anri.io.geom_from_pars(geo, ds["y0"], wl * 2e-4 / 2.355, 5e-5, 5e-5, sig_beam=beam / 2.355, voxel_size=vsize,
                              sig_psf=0.5)  # fmt: skip
rings = ix.ring_table(lpars, 225, wl, 8)
hkls, F2 = np.asarray(rings["hkls"], float), np.ones(len(rings["hkls"]))
det_shape = (2048, 2048)

# measured pixels, one dty row per scan
rows, meas, raw = [], [], []
with h5py.File(ds["sparsefile"], "r") as h:
    scans = sorted(h.keys(), key=float)
    if args.rows:
        mid = len(scans) // 2
        scans = scans[mid - args.rows // 2 : mid - args.rows // 2 + args.rows]
    for g in scans:
        nnz = h[g]["nnz"][()]
        frame = np.repeat(np.arange(len(nnz)), nnz)
        pixel = h[g]["row"][()].astype(np.int64) * det_shape[1] + h[g]["col"][()]
        om, dt, val = h[g]["measurement/rot_center"][()], h[g]["measurement/dty"][()], h[g]["intensity"][()]
        raw.append((om, dt, frame, pixel, val))
        rows.append(anri.fwd.make_row(om, dt))
        meas.append(anri.refine.measured(frame, pixel, val, len(nnz)))
log(f"{len(rows)} rows, {sum(m['value'].size for m in meas)} measured pixels")

r = np.load(args.entries)
vox, popn = r["voxel"], r["population"]
frac = r["density"] / np.bincount(vox, r["density"])[vox]
entries = {"ubi": r["ubi"].astype(np.float64), "pos": r["pos"].astype(np.float64), "density": 30.0 * frac}
log(f"{len(vox)} entries ({np.sum(popn == 0)} voxels)")

# one global intensity scale, from the central row
i = len(rows) // 2
_, _, val, _ = anri.fwd.render_row(entries, hkls, F2, geom, rows[i], det_shape, min_value=1.0, max_frames=31)
scale = float(meas[i]["value"].sum() / max(float(np.sum(val)), 1e-30))
entries["density"] = entries["density"] * scale
log(f"intensity scale from row {i}: {scale:.3f} (1 if the densities match the phantom's)")

truth = TensorMap.from_h5(args.truth)
te = anri.io.entries_from_tensormap(truth)
t_U = np.linalg.inv(te["ubi"]) @ np.linalg.inv(B)
inside = KDTree(te["pos"][:, :2]).query_ball_point(entries["pos"][:, :2], 0.5 * vsize, p=np.inf)
main = np.flatnonzero((popn == 0) & np.array([len(x) > 0 for x in inside]))


def accuracy(ubi: np.ndarray, label: str) -> None:
    """Main populations: misorientation to the closest truth voxel inside the entry's voxel."""
    u, _, vt = np.linalg.svd(np.linalg.inv(ubi[main]) @ np.linalg.inv(B))
    U = u @ vt
    err = np.array([anri.crystal.disorientation(np.repeat(U[k][None], len(inside[v]), 0), t_U[inside[v]], ops).min()
                    for k, v in enumerate(main)])  # fmt: skip
    log(f"{label}: main populations vs truth: median {np.median(err):.3f} deg; within 0.05 {np.mean(err < 0.05) * 100:.1f}%, "
        f"0.1 {np.mean(err < 0.1) * 100:.1f}%, 0.25 {np.mean(err < 0.25) * 100:.1f}%, 1 {np.mean(err < 1) * 100:.1f}%")  # fmt: skip


accuracy(entries["ubi"], "indexer")


def binned(k: int) -> tuple:
    """Rows and measured pixels with k frames summed (k = 1: as read)."""
    if k == 1:
        return rows, meas
    out_rows, out_meas = [], []
    for om, dt, frame, pixel, value in raw:
        om_b, dt_b, fr_b, px_b, val_b = anri.refine.bin_frames(om, dt, frame, pixel, value, k)
        out_rows.append(anri.fwd.make_row(om_b, dt_b))
        out_meas.append(anri.refine.measured(fr_b, px_b, val_b, len(om_b)))
    return out_rows, out_meas


if len(args.bin) != len(args.sig):
    raise SystemExit("--bin needs one value per --sig")
for sig, k in zip(args.sig, args.bin):
    e_in = {**entries, "sig_rot": np.full(len(vox), np.radians(sig))}
    rows_k, meas_k = binned(1 if args.method == "linear" else k)
    log(f"spread {sig} deg, frames summed in {k}s: {sum(m['value'].size for m in meas_k)} measured pixels")
    if args.method == "linear":  # frames as measured; the window wide enough for the spread
        ostep = float(np.median(np.diff(np.asarray(rows[0]["omega_edges"]))))
        wo = min(31, 2 * int(np.ceil(2.5 * sig / ostep)) + 3)
        log(f"  window {wo} frames x {args.win_px} x {args.win_px} pixels")
        out, hist = anri.refine.refine_orientations(e_in, hkls, F2, geom, rows, meas, det_shape,
                                                    window=(wo, args.win_px, args.win_px), n_sweeps=args.iter,
                                                    relinearise=args.relin, cut=1.0, log=lambda m: log("  " + m))  # fmt: skip
    elif args.method == "entry":
        out, hist = anri.refine.refine_per_entry(e_in, hkls, F2, geom, rows_k, meas_k, det_shape, n_iter=args.iter,
                                                 cut=1.0 * k, max_frames=args.max_frames, window=tuple(args.window),
                                                 log=lambda m: log("  " + m))  # fmt: skip
    else:
        out, hist = anri.refine.refine(e_in, hkls, F2, geom, rows_k, meas_k, det_shape, n_iter=args.iter,
                                       cut=1.0 * k, max_frames=args.max_frames, window=tuple(args.window),
                                       fit_density=False, n_cg=args.n_cg, log=lambda m: log("  " + m))  # fmt: skip
    entries["ubi"] = out["ubi"]
    accuracy(entries["ubi"], f"after spread {sig} deg ({hist[-1]['time']:.0f} s, capture {hist[-1]['capture']:.2f})")
np.savez(os.path.splitext(args.entries)[0] + "_refined.npz", **entries)
