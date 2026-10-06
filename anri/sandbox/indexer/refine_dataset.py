"""Refine the indexer's map entries on an ImageD11 dataset with anri.refine, widening the peaks by a spread first.

    python refine_dataset.py <analysisroot> <sample> <dataset> <..._index_entries.npz> [--phase NAME] [--cif CIF]
        [--monitor fpico6] [--rows 5] [--sig 0.3 0.1 0.03] [--iter 5] [--beam FWHM] [--det-shape 2162 2068]

Each round refines every entry's lattice with all entries' sig_rot set to the round's spread (deg, sigma), densities
fixed: the indexer decided which population owns each voxel. --rows N uses the N dty rows nearest the rotation axis
(start small: a few rows show the cost and whether the loss falls). The y0 and voxel size are the indexer's (from
the _index.npz beside the entries). Logs, per round, the loss and how far the main populations moved; writes
<tag>_refined.npz (entries) and <tag>_refined_tmap.h5 (main populations).
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
p.add_argument(
    "entries", help="<tag>_index_entries.npz of python -m anri.index (the <tag>_index.npz must be beside it)"
)
p.add_argument("--phase")
p.add_argument("--parfile")
p.add_argument("--cif", help="the CIF the indexer used, for |F|^2 (default 1)")
p.add_argument("--rings", type=int, default=6)
p.add_argument("--monitor")
p.add_argument("--rows", type=int, default=5, help="dty rows nearest the rotation axis (0: all)")
p.add_argument("--sig", type=float, nargs="+", default=[0.3, 0.1, 0.03], help="spread per round (deg, sigma)")
p.add_argument("--bin", type=int, nargs="+", default=[5, 2, 1], help="frames summed per round (one per --sig)")
p.add_argument(
    "--method",
    choices=("linear", "entry", "cg"),
    default="linear",
    help="linear: refine_orientations, geometry linearised, cheap sweeps (default; --iter = sweeps per round, frames "
    "not binned); entry: refine_per_entry; cg: refine",
)
p.add_argument("--iter", type=int, default=5, help="Levenberg-Marquardt iterations per round")
p.add_argument("--beam", type=float, help="beam FWHM, dty units (default: the row step)")
p.add_argument("--det-shape", type=int, nargs=2, default=(2162, 2068), help="detector (slow, fast) pixels")
p.add_argument("--cut", type=float, default=1.0, help="segmentation threshold of the sparse pixels")
p.add_argument("--max-frames", type=int, default=31)
p.add_argument("--window", type=int, nargs=3, default=(3, 7, 7))
p.add_argument("--n-cg", type=int, default=15, help="conjugate gradient steps per iteration (default 15)")
p.add_argument("--relin", type=int, default=5, help="linear: accepted sweeps between linearisations")
p.add_argument("--win-px", type=int, default=7, help="linear: window size in pixels (slow and fast)")
p.add_argument("--n-cpu", type=int, default=4)
args = p.parse_args()

import anri.utils  # noqa: E402

anri.utils.setup(n_cpu=args.n_cpu)
import h5py  # noqa: E402

import anri.crystal  # noqa: E402
import anri.fwd  # noqa: E402
import anri.index as ix  # noqa: E402
import anri.io  # noqa: E402
import anri.refine  # noqa: E402

dsname = f"{args.sample}_{args.dataset}"
ds = anri.io.read_dataset(os.path.join(args.analysisroot, args.sample, dsname, f"{dsname}_dataset.h5"))
geo, phase, cell = anri.io.read_pars_json(args.parfile or ds["parfile"], args.phase)
lpars = np.array([cell[k] for k in ("cell__a", "cell__b", "cell__c", "cell_alpha", "cell_beta", "cell_gamma")])
sg = int(cell["cell_lattice_[P,A,B,C,I,F,R]"])
B = anri.crystal.B_matrix(lpars)
ops = anri.crystal.laue_rotations(anri.crystal.symmetry_matrices(sg), B)
wl = geo["wavelength"]
structure = None
if args.cif:
    import Dans_Diffraction

    structure = Dans_Diffraction.Crystal(args.cif)
rings = ix.ring_table(lpars, sg, wl, args.rings, structure)
hkls, F2 = np.asarray(rings["hkls"], float), np.asarray(rings["F2"], float)

r_idx = np.load(args.entries.replace("_entries.npz", ".npz"))
y0 = float(r_idx["y0"])
xs = np.unique(np.round(r_idx["pos"][:, 0], 6))
vox = float(np.median(np.diff(xs)))
ystep = float(np.median(np.diff(ds["ybincens"])))
beam = args.beam or ystep
# instrument: Si(111)-like bandwidth, ~0.2 mrad convergence (Al CRLs), the detector's point spread ~0.5 px
geom = anri.io.geom_from_pars(geo, y0, wl * 1.4e-4 / 2.355, 1e-4, 1e-4, sig_beam=beam / 2.355, voxel_size=vox,
                              sig_psf=0.5)  # fmt: skip
det_shape = tuple(args.det_shape)
log(f"{dsname}: phase {phase}, {len(hkls)} hkls in {args.rings} rings; y0 {y0:.6g}, voxel {vox:g}, beam FWHM {beam:g}")

# measured pixels: the --rows scans nearest the rotation axis, one dty row each
scans = list(ds["scans"])
dty_scan = np.asarray(ds["dty"]).mean(1)
pick = np.argsort(np.abs(dty_scan - y0))[: (args.rows or len(scans))]
groups = [scans[i] for i in sorted(pick, key=lambda i: dty_scan[i])]
mons = anri.io.read_monitor(ds["sparsefile"], groups, args.monitor, ds["masterfile"]) if args.monitor else {}
mon_ref = float(np.mean(np.concatenate(list(mons.values())))) if args.monitor else 1.0
rows, meas, raw = [], [], []
with h5py.File(ds["sparsefile"], "r") as h:
    for g in groups:
        gr = h[g]
        nnz = gr["nnz"][()]
        frame = np.repeat(np.arange(len(nnz)), nnz)
        pixel = gr["row"][()].astype(np.int64) * det_shape[1] + gr["col"][()]
        if pixel.size and (gr["row"][()].max() >= det_shape[0] or gr["col"][()].max() >= det_shape[1]):
            raise SystemExit(f"pixels beyond --det-shape {det_shape} in {g}")
        value = gr["intensity"][()].astype(np.float32)
        if args.monitor:
            value = value * (mon_ref / np.maximum(mons[g], 1e-30))[frame]
        dty_f = (
            gr[f"measurement/{ds['dtymotor']}"][()]
            if ds["dtymotor"] in gr["measurement"]
            else ds["dty"][scans.index(g)]
        )
        om, dt = gr[f"measurement/{ds['omegamotor']}"][()], np.broadcast_to(dty_f, nnz.shape)
        raw.append((om, dt, frame, pixel, value))
        rows.append(anri.fwd.make_row(om, dt))
        meas.append(anri.refine.measured(frame, pixel, value, len(nnz)))
log(f"{len(rows)} rows (dty - y0 {dty_scan[pick].min() - y0:+.2f} .. {dty_scan[pick].max() - y0:+.2f}), "
    f"{sum(m['value'].size for m in meas)} measured pixels")  # fmt: skip

r = np.load(args.entries)
voxel, popn = r["voxel"], r["population"]
entries = {
    "ubi": r["ubi"].astype(np.float64),
    "pos": r["pos"].astype(np.float64),
    "density": r["density"].astype(float),
}
# one global intensity scale (the indexer's densities are in its own units), from the middle row
i = len(rows) // 2
_, _, val, _ = anri.fwd.render_row(entries, hkls, F2, geom, rows[i], det_shape, min_value=args.cut, max_frames=31)
scale = float(meas[i]["value"].sum() / max(float(np.sum(val)), 1e-30))
entries["density"] = entries["density"] * scale
log(f"{len(voxel)} entries ({np.sum(popn == 0)} voxels); intensity scale from the middle row {scale:.4g}")

main = popn == 0


def orientation(ubi: np.ndarray) -> np.ndarray:
    """Nearest rotation to UBI^-1 B^-1."""
    u, _, vt = np.linalg.svd(np.linalg.inv(ubi) @ np.linalg.inv(B))
    return u @ vt


start = orientation(entries["ubi"][main])


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


bins = [1] * len(args.sig) if args.method == "linear" else args.bin  # linear: frames as measured
if len(bins) != len(args.sig):
    raise SystemExit("--bin needs one value per --sig")
for sig, k in zip(args.sig, bins):
    e_in = {**entries, "sig_rot": np.full(len(voxel), np.radians(sig))}
    rows_k, meas_k = binned(k)
    log(f"spread {sig} deg, frames summed in {k}s: {sum(m['value'].size for m in meas_k)} measured pixels")
    if args.method == "linear":  # frames as measured; the window wide enough for the spread
        ostep = float(np.median(np.diff(np.asarray(rows[0]["omega_edges"]))))
        wo = min(31, 2 * int(np.ceil(2.5 * sig / ostep)) + 3)
        log(f"  window {wo} frames x {args.win_px} x {args.win_px} pixels")
        out, hist = anri.refine.refine_orientations(e_in, hkls, F2, geom, rows, meas, det_shape,
                                                    window=(wo, args.win_px, args.win_px), n_sweeps=args.iter,
                                                    relinearise=args.relin, cut=args.cut, log=lambda m: log("  " + m))  # fmt: skip
    elif args.method == "entry":
        out, hist = anri.refine.refine_per_entry(e_in, hkls, F2, geom, rows_k, meas_k, det_shape, n_iter=args.iter,
                                                 cut=args.cut * k, max_frames=args.max_frames, window=tuple(args.window),
                                                 log=lambda m: log("  " + m))  # fmt: skip
    else:
        out, hist = anri.refine.refine(e_in, hkls, F2, geom, rows_k, meas_k, det_shape, n_iter=args.iter,
                                       cut=args.cut * k, max_frames=args.max_frames, window=tuple(args.window),
                                       fit_density=False, n_cg=args.n_cg, log=lambda m: log("  " + m))  # fmt: skip
    moved = anri.crystal.disorientation(orientation(out["ubi"][main]), orientation(entries["ubi"][main]), ops)
    total = anri.crystal.disorientation(orientation(out["ubi"][main]), start, ops)
    entries["ubi"] = out["ubi"]
    log(f"spread {sig} deg: loss {hist[0]['loss']:.5g} -> {hist[-1]['loss']:.5g} ({hist[-1]['time']:.0f} s, capture "
        f"{hist[-1].get('capture', float('nan')):.2f}); main populations moved this round: median {np.median(moved):.3f} deg, 90th "
        f"{np.percentile(moved, 90):.3f}; from the indexer: median {np.median(total):.3f}, 90th {np.percentile(total, 90):.3f}")  # fmt: skip

tag = args.entries.replace("_index_entries.npz", "")
np.savez(f"{tag}_refined.npz", **entries, voxel=voxel, population=popn)
nv = len(r_idx["pos"])
NR = round(np.sqrt(nv))
ubi_map = np.full((nv, 3, 3), np.nan)
ubi_map[voxel[main]] = entries["ubi"][main]
ok = np.isfinite(ubi_map).all(axis=(1, 2))
maps = {"UBI": ubi_map.reshape(NR, NR, 3, 3), "phase_ids": np.where(ok, 0, -1).reshape(NR, NR)}
tmap = anri.io.tensormap_from_recon(maps, np.asarray(lpars), sg, phase, ystep)
try:
    tmap.get_ipf_maps()
except ImportError:
    pass
if os.path.exists(f"{tag}_refined_tmap.h5"):
    os.remove(f"{tag}_refined_tmap.h5")
tmap.to_h5(f"{tag}_refined_tmap.h5")
log(f"-> {tag}_refined.npz, {tag}_refined_tmap.h5")
