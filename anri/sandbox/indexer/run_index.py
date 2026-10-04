"""Index an ImageD11 S3DXRD dataset from scratch and write a TensorMap (the highest-occupancy orientation per voxel).

Coarse histograms of the sparse pixels, a cubic fundamental-zone grid pruned by completeness, then MLEM occupancy of
the kept orientations on an NR x NR voxel grid (voxel = dty step; NR = number of dty bins + ImageD11's
sino_shift_and_pad padding, so the grid matches ImageD11's reconstructions and is centred on the rotation axis).

    python run_index.py <analysisroot> <sample> <dataset> [--phase NAME] [--parfile pars.json] [--check] ...

Paths follow ImageD11's layout: {analysisroot}/{sample}/{sample}_{dataset}/{sample}_{dataset}_dataset.h5 and _sparse.h5.
Parameters: the geometry and the phase's lattice and space group (number) come from pars.json (the dataset's parfile
if it exists here, else pars/pars.json beside PROCESSED_DATA, or --parfile). The scan (y0, dty and omega bins, motor
names) comes from the dataset file (y0 can be overridden with --y0); each frame's row is the dty bin of its motor reading. Lengths are in the units of
the dataset's dty and the geometry file, which must agree. No spatial distortion correction yet; F^2 = 1 (no atoms).
"""

import argparse
import json
import os
import time

T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f} s] {msg}", flush=True)


p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("analysisroot")
p.add_argument("sample")
p.add_argument("dataset")
p.add_argument("--phase", help="phase name in pars.json (default: the only one)")
p.add_argument("--parfile", help="pars.json (default: the dataset's parfile, else pars/pars.json beside PROCESSED_DATA)")
p.add_argument("--rings", type=int, default=6, help="number of rings used (default 6)")
p.add_argument("--grid", type=float, default=2.5, help="orientation grid step, degrees (default 2.5)")
p.add_argument("--keep", type=int, default=3000, help="orientations kept for the occupancy fit (default 3000)")
p.add_argument("--iter", type=int, default=10, help="MLEM iterations (default 10)")
p.add_argument("--lit", type=float, default=1.0, help="lit threshold, x the median non-empty bin (default 1)")
p.add_argument("--etacut", type=float, default=0.1, help="use reflections with |sin eta| above this (default 0.1)")
p.add_argument("--beam", type=float, help="beam FWHM (dty units; default one dty step; not used by the fit yet)")
p.add_argument("--y0", type=float, help="dty where the rotation axis is in the beam (default: the dataset's y0)")
p.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
p.add_argument("--check", action="store_true", help="print the resolved paths and parameters, then stop")
args = p.parse_args()

import anri.utils  # noqa: E402

anri.utils.setup(n_cpu=4)
import jax  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
jax.config.update("jax_compilation_cache_dir", os.path.join(HERE, "..", ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
import h5py  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from ImageD11.sinograms.geometry import recon_to_sample  # noqa: E402
from ImageD11.sinograms.tensor_map import TensorMap  # noqa: E402
from ImageD11.unitcell import unitcell  # noqa: E402

import index as X  # noqa: E402
from anri.crystal import Crystal, Symmetry, UnitCell  # noqa: E402
from anri.io import geom_from_pars  # noqa: E402

F32 = jnp.float32
B_E, B_O = 0.5, 0.25  # lit-map bins (deg): eta, omega
R_E, R_O = 2, 4  # MLEM data bins = lit bins x these (1 x 1 deg)
TOL_E, TOL_O = 0.5, 1.0  # matching tolerance (deg) of the completeness test
CHUNK = 1 << 24


def read_par(path: str) -> dict:
    """An ImageD11 .par file as a dict (numbers as floats)."""
    out = {}
    for line in open(path):
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                out[parts[0]] = parts[1]
    return out


# ------------------------------------------------------------------------------------------------- paths
dsname = f"{args.sample}_{args.dataset}"
dsdir = os.path.join(args.analysisroot, args.sample, dsname)
dsfile = os.path.join(dsdir, f"{dsname}_dataset.h5")
sparsefile = os.path.join(dsdir, f"{dsname}_sparse.h5")
with h5py.File(dsfile, "r") as h:
    attrs = dict(h.attrs)
    Y0 = float(attrs["y0"]) if args.y0 is None else args.y0
    ybin, yedge = h["ybincens"][()], h["ybinedges"][()]
    oedge = h["obinedges"][()]
parfile = args.parfile
if parfile is None:
    parfile = str(attrs.get("parfile", ""))
    if not os.path.exists(parfile):  # e.g. the dataset was processed elsewhere: pars/ beside PROCESSED_DATA
        root = os.path.abspath(args.analysisroot)
        while os.path.basename(root) != "PROCESSED_DATA" and root != os.path.dirname(root):
            root = os.path.dirname(root)
        parfile = os.path.join(os.path.dirname(root), "pars", "pars.json")
pj = json.load(open(parfile))
pdir = os.path.dirname(os.path.abspath(parfile))
phases = pj["phases"]
phase = args.phase or (next(iter(phases)) if len(phases) == 1 else None)
if phase is None:
    raise SystemExit(f"several phases in {parfile}: {list(phases)}; choose one with --phase")
geo = read_par(os.path.join(pdir, pj["geometry"]["file"]))
ph = read_par(os.path.join(pdir, phases[phase]["file"]))

# ------------------------------------------------------------------------------------------------- scan
YSTEP, DTY0, NK = float(np.median(np.diff(ybin))), float(ybin[0]), len(ybin)
OM0 = float(oedge[0])
N_E, N_O = int(round(360 / B_E)), int(round((oedge[-1] - oedge[0]) / B_O))
N_O -= N_O % R_O
DTYM, OMM = str(attrs["dtymotor"]), str(attrs["omegamotor"])
BEAM = args.beam if args.beam is not None else YSTEP
WL = geo["wavelength"]
geom = geom_from_pars(geo, Y0, WL * 2e-3 / 2.355, 1.5e-4, 1.5e-4, sig_beam=BEAM / 2.355, voxel_size=YSTEP, sig_psf=0.5)
geom = {k: jnp.asarray(v, F32) if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating) else v for k, v in geom.items()}

# ------------------------------------------------------------------------------------------------- crystal and rings
lpars = [ph[k] for k in ("cell__a", "cell__b", "cell__c", "cell_alpha", "cell_beta", "cell_gamma")]
sg = ph["cell_lattice_[P,A,B,C,I,F,R]"]
if not isinstance(sg, float):
    raise SystemExit(f"{phase}: cell_lattice_[P,A,B,C,I,F,R] = {sg} is a centring letter; a space-group number is needed")
sg = int(sg)
if not 195 <= sg <= 230:
    raise SystemExit(f"{phase}: space group {sg} is not cubic; only the cubic fundamental zone is implemented")
crystal = Crystal(UnitCell.from_lpars(lpars), Symmetry.from_number(sg))
B = np.asarray(crystal.B, np.float32)
dsmax = 0.5
while True:  # enough d* range for the first args.rings allowed rings
    crystal.make_hkls(dsmax, WL)
    h_all = np.asarray(crystal.allhkls, np.float64)
    ok = X.allowed(h_all, np.asarray(crystal._sym.symmetry_matrices))
    tth_all = np.asarray(crystal.alltth)[ok]
    ring_tth_all, ring_all = np.unique(np.round(tth_all, 4), return_inverse=True)
    if len(ring_tth_all) > args.rings or dsmax > 5:
        break
    dsmax *= 1.5
sel = ring_all < args.rings
hkls = jnp.asarray(h_all[ok][sel], F32)
ring_of_h = jnp.asarray(np.repeat(ring_all[sel], 2), jnp.int32)  # j = h * 2 + branch
ring_tth = jnp.asarray(ring_tth_all[: args.rings], F32)
NJ = 2 * hkls.shape[0]


def sino_shift_and_pad(y0: float, ny: int, ymin: float, ystep: float) -> tuple:
    """ImageD11.sinograms.geometry.sino_shift_and_pad as of 2026-10 (older installed versions lack the odd-size rule)."""
    shift = ny // 2 - (y0 - ymin) / ystep
    pad = int(np.ceil(abs(shift) * 2)) + 1
    if (ny + pad) % 2 == 0:  # keep the reconstruction odd-sized
        pad += 1
    return shift, pad


_, PAD = sino_shift_and_pad(Y0, NK, float(ybin[0]), YSTEP)  # as ImageD11 pads its reconstructions
NR = NK + int(PAD)  # recon grid NR x NR, centred on the rotation axis
NV = NR * NR
QC = int(max(1, 2 ** np.floor(np.log2(max(1, 2e9 / (NV * NJ * 64))))))  # orientations per chunk: ~2 GB of system entries

log(f"dataset {dsfile}")
log(f"sparse  {sparsefile}")
log(f"pars    {parfile}: phase {phase}, lattice {', '.join(f'{v:g}' for v in lpars)}, space group {sg} ({crystal.sgname})")
log(f"geometry: wavelength {WL:.5f}, distance {geo['distance']:g}; y0 {Y0:.6g}"
    f"{' (override)' if args.y0 is not None else ''}; dty {DTY0:.6g} + {NK} x {YSTEP:.6g} "
    f"(motor {DTYM}); omega {OM0:.4g} .. {oedge[-1]:.4g} in {len(oedge) - 1} frames (motor {OMM}); beam FWHM {BEAM:g}"
    + ("" if args.beam is not None else " (placeholder: one dty step)"))
log(f"{hkls.shape[0]} hkls in {args.rings} rings at 2theta {', '.join(f'{v:.2f}' for v in ring_tth_all[: args.rings])}; "
    f"grid {args.grid} deg, keep {args.keep}, {args.iter} MLEM iterations, voxels {NR} x {NR} ({NK} dty bins + pad {int(PAD)}), {QC} orientations per chunk")
if args.check:
    raise SystemExit(0)

# ------------------------------------------------------------------------------------------------- 1. histograms
# the lit map (no rows, fine bins) for pruning; the MLEM data (1 x 1 deg, with rows). Every frame's row is the dty bin
# of its motor reading, so one group per row and one group for the whole scan both work.
H_lit = jnp.zeros(args.rings * N_E * N_O, F32)
H = jnp.zeros(args.rings * (N_E // R_E) * (N_O // R_O) * NK, F32)
t1 = time.perf_counter()
n_tot = 0
with h5py.File(sparsefile, "r") as h:
    for name in h.keys():
        gr = h[name]
        nnz = gr["nnz"][()]
        om_f = gr[f"measurement/{OMM}"][()].astype(np.float32)
        dty_f = np.broadcast_to(gr[f"measurement/{DTYM}"][()], nnz.shape)
        k_f = (np.searchsorted(yedge, dty_f) - 1).astype(np.int32)
        k_f[(k_f < 0) | (k_f >= NK)] = -1
        ends = np.cumsum(nnz)
        n = int(ends[-1]) if len(ends) else 0
        n_tot += n
        for s0 in range(0, n, CHUNK):
            m = min(CHUNK, n - s0)
            fr = np.searchsorted(ends, np.arange(s0, s0 + m), side="right")
            pad = lambda a, dt=np.float32: jnp.asarray(np.pad(np.asarray(a).astype(dt), (0, CHUNK - m)))  # noqa: E731
            x = X.pixels_to_x(pad(gr["row"][s0:s0 + m]), pad(gr["col"][s0:s0 + m]), pad(om_f[fr]), geom)
            live = jnp.arange(CHUNK) < m
            v = jnp.where(live, pad(gr["intensity"][s0:s0 + m]), 0.0)
            H_lit = H_lit + X.histogram(x, v, jnp.zeros(CHUNK, jnp.int32), ring_tth, 0.1, OM0, B_E, B_O, n_ring=args.rings,
                                        n_e=N_E, n_o=N_O, n_k=1)  # fmt: skip
            rows = jnp.where(live, pad(k_f[fr], np.int32), -1)
            H = H + X.histogram(x, v, rows, ring_tth, 0.1, OM0, B_E * R_E, B_O * R_O, n_ring=args.rings, n_e=N_E // R_E,
                                n_o=N_O // R_O, n_k=NK)  # fmt: skip
jax.block_until_ready(H)
log(f"histograms: {n_tot / 1e6:.0f}M pixels in {time.perf_counter() - t1:.0f} s; MLEM data {H.size / 1e6:.0f}M bins, "
    f"{float(jnp.mean(H > 0)) * 100:.1f}% non-empty")

# ------------------------------------------------------------------------------------------------- 2. pruning
Hs = H_lit.reshape(args.rings, N_E, N_O)
med = float(jnp.median(Hs[Hs > 0]))
for kk in (0.0, 1.0, 3.0, 10.0):
    log(f"  lit fraction at {kk:g} x median: {float(jnp.mean(Hs > kk * med)) * 100:.1f}%")
lit = X.dilate(Hs > args.lit * med, de=int(round(TOL_E / B_E)), do=int(round(TOL_O / B_O)))
log(f"lit (> {args.lit} x median, dilated +-{TOL_E} / +-{TOL_O} deg): {float(jnp.mean(lit)) * 100:.1f}%")
t1 = time.perf_counter()
rod = jnp.asarray(X.cubic_fz_grid(args.grid))
comp = []
for s0 in range(0, rod.shape[0], 1 << 15):
    eta, om, ok_ = X.predict(rod[s0:s0 + (1 << 15)], jnp.asarray(B), hkls, geom)
    comp.append(X.completeness(lit, eta, om, ok_, ring_of_h, OM0, B_E, B_O, n_e=N_E, n_o=N_O, etacut=args.etacut)[0])
comp = np.asarray(jnp.concatenate(comp))
keep = np.argsort(comp)[::-1][: args.keep]
log(f"completeness of {rod.shape[0]} orientations: {time.perf_counter() - t1:.1f} s; median {np.median(comp):.2f}, 99th "
    f"{np.percentile(comp, 99):.2f}, max {comp.max():.2f}; keeping the top {len(keep)} (completeness >= {comp[keep[-1]]:.2f})")

# ------------------------------------------------------------------------------------------------- 3. MLEM occupancy
nq = len(keep)
pad_q = -nq % QC
rod_s = jnp.asarray(np.concatenate([np.asarray(rod)[keep], np.repeat(np.asarray(rod)[keep[:1]], pad_q, 0)]))
eta, om, ok_, lp = X.predict_lp(rod_s, jnp.asarray(B), hkls, geom)
use = ok_ & (jnp.abs(jnp.sin(jnp.radians(eta))) > args.etacut) & (jnp.arange(nq + pad_q) < nq)[:, None]
pred = (eta, om, use, jnp.where(use, lp, 0.0))  # F^2 = 1
dims = (B_E * R_E, B_O * R_O, N_E // R_E, N_O // R_O)
ri, rj = np.meshgrid(np.arange(NR), np.arange(NR), indexing="ij")
sx, sy = recon_to_sample(ri, rj, (NR, NR), YSTEP)
pos = jnp.asarray(np.stack([sx.ravel(), sy.ravel(), np.zeros(NV)], 1), F32)
scan = {"y0": Y0, "dty0": DTY0, "ystep": YSTEP, "n_rows": NK, "om0": OM0}
f0 = jnp.ones((NV, nq + pad_q), F32)
t1 = time.perf_counter()
jax.block_until_ready(X.forward(f0, pred, ring_of_h, pos, scan, dims, H.shape[0], qc=QC))
t2 = time.perf_counter()
jax.block_until_ready(X.forward(f0, pred, ring_of_h, pos, scan, dims, H.shape[0], qc=QC))
t3 = time.perf_counter()
log(f"MLEM: {NV} voxels x {nq} orientations; one forward projection {t3 - t2:.1f} s (first {t2 - t1:.1f} s); "
    f"estimate ~{2.2 * (t3 - t2) * args.iter / 60:.1f} min for {args.iter} iterations")
t1 = time.perf_counter()
f = np.asarray(X.mlem(H, pred, ring_of_h, pos, scan, dims, f0, args.iter, log=log, qc=QC))[:, :nq]
log(f"MLEM {args.iter} iterations: {time.perf_counter() - t1:.0f} s")

# ------------------------------------------------------------------------------------------------- 4. TensorMap
best = np.argmax(f, 1)
tot = f.sum(1)
share = f[np.arange(NV), best] / np.maximum(tot, 1e-30)
ubi = np.linalg.inv(np.asarray(X.rod_to_mat(rod_s[:nq]))[best] @ B)
occupied = tot > 0.05 * np.percentile(tot, 99)
log(f"voxels with occupancy: {occupied.sum()} of {NV}; top share median {np.median(share[occupied]):.2f}; "
    f"distinct orientations used {len(np.unique(best[occupied]))}")
to_map = TensorMap.recon_order_to_map_order
tmap = TensorMap(maps={"UBI": to_map(np.where(occupied[:, None, None], ubi, np.nan).reshape(NR, NR, 3, 3)),
                       "phase_ids": to_map(np.where(occupied, 0, -1).reshape(NR, NR)),
                       "occupancy": to_map(tot.reshape(NR, NR)), "best_share": to_map(share.reshape(NR, NR)),
                       "orientation_id": to_map(np.where(occupied, best, -1).reshape(NR, NR))}, steps=[YSTEP] * 3)  # fmt: skip
tmap.phases = {0: unitcell(lpars, sg, name=phase)}
tmap.get_ipf_maps()
_ = tmap.euler
_ = tmap.eps_devia
_ = tmap.eps_crystal
tag = f"{dsname}_mlem_{args.keep}_{args.iter}_{args.lit:g}_{args.grid:g}"
out = os.path.join(args.outdir, f"{tag}_tmap.h5")
if os.path.exists(out):
    os.remove(out)
tmap.to_h5(out)
tmap.to_paraview(out)
np.savez(os.path.join(args.outdir, f"{tag}.npz"), f=f, rod=np.asarray(rod_s[:nq]), comp=comp[keep])
log(f"-> {out}")
