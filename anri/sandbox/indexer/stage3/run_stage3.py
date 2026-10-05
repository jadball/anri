"""Stage 3 prototype on any ImageD11 dataset: refine the populations of an anri.index result on a fine histogram.

    python run_stage3.py <analysisroot> <sample> <dataset> --npz <..._index.npz> --beam FWHM [--phase NAME]
        [--monitor fpico6] [--bins 0.25 0.05] [--iter 20] [--outdir .]

Each (voxel, population) of the npz is a unit with a local grid of orientations around it; MLEM fits the units'
occupancies to a sparse fine histogram (eta x omega x dty row), with parallax (each voxel's spots traced from its
own position) and the beam profile over rows. Pass 1: +-1 deg at 0.5 deg around the stage-2 means; pass 2: +-0.5 deg
at 0.25 deg around pass 1's. Writes <tag>_stage3.npz and <tag>_stage3_tmap.h5 (main population, IPF), and logs how far
each main orientation moved from stage 2.

Scratch quality: see ../DESIGN.md. Uses fine.py beside it.
"""

import argparse
import os
import sys
import time

T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f} s] {msg}", flush=True)


p = argparse.ArgumentParser()
p.add_argument("analysisroot")
p.add_argument("sample")
p.add_argument("dataset")
p.add_argument("--npz", required=True, help="the _index.npz of python -m anri.index for this dataset")
p.add_argument("--beam", type=float, required=True, help="beam FWHM, dty units")
p.add_argument("--phase")
p.add_argument("--parfile")
p.add_argument("--monitor")
p.add_argument("--rings", type=int, default=6)
p.add_argument("--bins", type=float, nargs=2, default=(0.25, 0.05), help="fine bins: eta and omega (deg)")
p.add_argument("--iter", type=int, default=20, help="MLEM iterations per pass")
p.add_argument("--pass1", type=float, nargs=2, default=(1.0, 0.5), help="pass 1 local grid: half-width and step (deg)")
p.add_argument("--pass2", type=float, nargs=2, default=(1.0, 0.25), help="pass 2 local grid: half-width and step (deg); "
               "a half-width of 0 skips it. Keep it at least as wide as the populations' spread")
p.add_argument("--min-frac", type=float, default=0.1)
p.add_argument("--block", type=int, default=0, help="units per block (default: from a ~1 GB budget)")
p.add_argument("--n-cpu", type=int, default=4)
p.add_argument("--outdir", default=".")
args = p.parse_args()

import anri.utils  # noqa: E402

anri.utils.setup(n_cpu=args.n_cpu)
import h5py  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anri.crystal  # noqa: E402
import anri.index as ix  # noqa: E402
import anri.io  # noqa: E402
import anri.phantom  # noqa: E402
from fine import fine_mlem, sparse_histogram  # noqa: E402

# ------------------------------------------------------------------------------------------------- inputs
dsname = f"{args.sample}_{args.dataset}"
dsfile = os.path.join(args.analysisroot, args.sample, dsname, f"{dsname}_dataset.h5")
ds = anri.io.read_dataset(dsfile)
sparsefile = ds["sparsefile"] if ds["sparsefile"] and os.path.exists(ds["sparsefile"]) else dsfile.replace("_dataset.h5", "_sparse.h5")
parfile = args.parfile or ds["parfile"]
geo, phase, cell = anri.io.read_pars_json(parfile, args.phase)
lpars = np.array([cell[k] for k in ("cell__a", "cell__b", "cell__c", "cell_alpha", "cell_beta", "cell_gamma")])
sg = int(cell["cell_lattice_[P,A,B,C,I,F,R]"])
B64 = anri.crystal.B_matrix(lpars)
B = B64.astype(np.float32)
ops = anri.crystal.laue_rotations(anri.crystal.symmetry_matrices(sg), B64)
WL = geo["wavelength"]
r = np.load(args.npz)
Y0 = ds["y0"]
ybin, yedge, oedge = ds["ybincens"], ds["ybinedges"], ds["obinedges"]
YSTEP, NK = float(np.median(np.diff(ybin))), len(ybin)
OM0 = float(oedge[0])
geom = anri.io.geom_from_pars(geo, Y0, WL * 2e-3 / 2.355, 1.5e-4, 1.5e-4, sig_beam=args.beam / 2.355, voxel_size=YSTEP)
geom = {k: jnp.asarray(v, jnp.float32) if np.issubdtype(np.asarray(v).dtype, np.floating) else v for k, v in geom.items()}
rings = ix.ring_table(lpars, sg, WL, args.rings)
B_E, B_O = args.bins
N_E, N_O = round(360 / B_E), round(float(oedge[-1] - oedge[0]) / B_O)
log(f"{dsname}: {NK} rows x {len(oedge) - 1} frames; fine bins {B_E} x {B_O} deg ({args.rings} rings x {N_E} x {N_O} cells "
    f"x {NK} rows); beam FWHM {args.beam}")  # fmt: skip

with h5py.File(sparsefile, "r") as h:
    groups = list(h.keys())
    n_max = max(int(h[g]["nnz"][()].sum()) for g in groups)
chunk = int(min(1 << 24, 1 << max(10, int(np.ceil(np.log2(max(n_max, 1)))))))
mon_ref = None
if args.monitor:
    mon_ref = float(np.mean(np.concatenate(list(anri.io.read_monitor(sparsefile, groups, args.monitor, ds["masterfile"]).values()))))


def stream(gs):  # noqa: ANN001, ANN201
    return anri.io.prefetch(anri.io.stream_sparse(sparsefile, yedge, ds["omegamotor"], ds["dtymotor"], chunk, gs, 1, ds["dty"],
                                                  ds["scans"], args.monitor, mon_ref, ds["masterfile"]))  # fmt: skip


sample = [groups[i] for i in np.unique(np.linspace(0, len(groups) - 1, min(len(groups), 9)).round().astype(int))]
off, hw = ix.ring_profile(stream(sample), geom, rings["tth"], chunk)
tth_tol = np.abs(off) + hw
t1 = time.perf_counter()
data = sparse_histogram(stream(groups), geom, rings["tth"], tth_tol, OM0, (B_E, B_O, N_E, N_O), NK, chunk)
log(f"fine histogram: {len(data['value']) / 1e6:.1f}M non-empty bins ({(len(data['value']) * 8 + len(data['start']) * 4) / 1e9:.2f} GB) "
    f"in {time.perf_counter() - t1:.0f} s")  # fmt: skip

# ------------------------------------------------------------------------------------------------- units
present, U_pop, pos = r["present"], r["U_pop"], r["pos"]
v_unit, p_unit = np.nonzero(present)
log(f"{present[:, 0].sum()} voxels, {len(v_unit)} (voxel, population) units")
n_side = int(np.ceil((0.5 * YSTEP + 3 * args.beam / 2.355) / YSTEP))
scan = {"y0": Y0, "dty0": float(ybin[0]), "ystep": YSTEP, "n_rows": NK, "om0": OM0, "B": jnp.asarray(B)}


def local_pass(U_centre, f_centre, half, step, label):  # noqa: ANN001, ANN201
    """MLEM on local grids (+-half at step) around each unit's orientation; returns each unit's mean, spread, total."""
    g = np.arange(-half, half + 1e-9, step)
    off3 = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    R_off = anri.phantom.axis_angle(off3 + (np.linalg.norm(off3, axis=1, keepdims=True) == 0), np.linalg.norm(off3, axis=1))
    K = len(R_off)
    U_c = (R_off[None] @ U_centre[:, None]).astype(np.float32)  # [Nu, K, 3, 3]
    vb = args.block or max(1, int(1e9 // (K * len(rings["ring_j"]) * 4 * (2 * n_side + 1) * 24)))
    n = len(U_c)
    pad = -n % vb

    def padded(a):  # noqa: ANN001, ANN202
        return np.concatenate([a, np.repeat(a[:1], pad, 0)])

    U_b = jnp.asarray(padded(U_c).reshape(-1, vb, K, 3, 3))
    pos_b = jnp.asarray(padded(pos[v_unit]).reshape(-1, vb, 3))
    f0 = np.repeat(f_centre[:, None], K, 1).astype(np.float32) / K
    f0_b = jnp.asarray(np.concatenate([f0, np.zeros((pad, K), np.float32)]).reshape(-1, vb, K))
    log(f"{label}: {n} units x {K} orientations (+-{half} at {step} deg), blocks of {vb}, rows +-{n_side}")
    t0 = time.perf_counter()
    f = fine_mlem(data, U_b, pos_b, f0_b, rings, geom, scan, (B_E, B_O, N_E, N_O), 0.2, n_side, args.iter, log=log)
    f = np.asarray(f).reshape(-1, K)[:n]
    log(f"{label}: {time.perf_counter() - t0:.0f} s")
    cand = np.arange(n * K).reshape(n, K)
    _, U_m, spread, _ = ix.populations(f, cand, U_c.reshape(-1, 3, 3), ops, 4 * half, p=1, eps=0.0)
    tot = f.sum(1)
    empty = tot <= 0  # the fit emptied the unit: no mean to take, so keep its input orientation
    log(f"{label}: {empty.sum()} of {n} units emptied by the fit")
    U_m, spread = U_m[:, 0], spread[:, 0]
    U_m[empty] = U_centre[empty]
    spread[empty] = np.nan
    return U_m, spread, tot


f_unit = r["frac"][v_unit, p_unit] * r["f"].sum(1)[v_unit]
U1, s1, t1_ = local_pass(U_pop[v_unit, p_unit].astype(float), f_unit, *args.pass1, "pass 1")
if args.pass2[0] > 0:
    U2, s2, tot2 = local_pass(U1.astype(float), t1_, *args.pass2, "pass 2")
else:
    U2, s2, tot2 = U1, s1, t1_

# ------------------------------------------------------------------------------------------------- per voxel
nv = len(present)
vox_tot = np.bincount(v_unit, weights=tot2, minlength=nv)
frac = tot2 / np.maximum(vox_tot[v_unit], 1e-30)
moved = anri.crystal.disorientation(U_pop[v_unit, p_unit].astype(float), U2.astype(float), ops)
live = tot2 > 0
log(f"{live.sum()} of {len(live)} units kept by the fit; for those: moved from stage 2 median {np.median(moved[live]):.3f} "
    f"deg, 90th {np.percentile(moved[live], 90):.3f}, max {moved[live].max():.2f}; spread median {np.nanmedian(s2[live]):.3f} "
    f"deg (stage 2: {np.median(r['spread'][v_unit, p_unit][live]):.3f})")  # fmt: skip
main = np.full(nv, -1)
order = np.lexsort((-frac, v_unit))  # by voxel, largest fraction first
first = np.unique(v_unit[order], return_index=True)[1]
main[v_unit[order][first]] = order[first]
occupied = main >= 0
n2 = np.bincount(v_unit[frac >= args.min_frac], minlength=nv)
log(f"{occupied.sum()} voxels; with 2+ populations >= {args.min_frac}: {np.mean(n2[occupied] >= 2) * 100:.1f}% (stage 2: "
    f"{np.mean(present[occupied].sum(1) >= 2) * 100:.1f}%)")  # fmt: skip
tag = os.path.join(args.outdir, f"{dsname}_stage3")
np.savez(f"{tag}.npz", voxel=v_unit, population=p_unit, U=U2, spread=s2, frac=frac, moved=moved, U_pass1=U1)
NR = int(round(np.sqrt(nv)))
Um = np.where(occupied[:, None, None], U2[np.maximum(main, 0)], np.nan)
maps = {"UBI": np.where(occupied[:, None, None], np.linalg.inv(Um @ B), np.nan).reshape(NR, NR, 3, 3),
        "phase_ids": np.where(occupied, 0, -1).reshape(NR, NR),
        "fraction": np.where(occupied, frac[np.maximum(main, 0)], np.nan).reshape(NR, NR),
        "spread": np.where(occupied, s2[np.maximum(main, 0)], np.nan).reshape(NR, NR),
        "moved": np.where(occupied, moved[np.maximum(main, 0)], np.nan).reshape(NR, NR)}  # fmt: skip
tmap = anri.io.tensormap_from_recon(maps, np.asarray(lpars), sg, phase, YSTEP)
try:
    tmap.get_ipf_maps()
except ImportError:
    pass
if os.path.exists(f"{tag}_tmap.h5"):
    os.remove(f"{tag}_tmap.h5")
tmap.to_h5(f"{tag}_tmap.h5")
tmap.to_paraview(f"{tag}_tmap.h5")
log(f"-> {tag}.npz, {tag}_tmap.h5")
