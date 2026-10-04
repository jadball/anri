"""Index an ImageD11 S3DXRD dataset from scratch and write a TensorMap (the highest-occupancy orientation per voxel).

Coarse histograms of the sparse pixels, a cubic fundamental-zone grid pruned by completeness, then MLEM occupancy of
the kept orientations on an NR x NR voxel grid (voxel = dty step; NR = number of dty bins + ImageD11's
sino_shift_and_pad padding, so the grid matches ImageD11's reconstructions and is centred on the rotation axis). The
occupancies are sparse: each voxel keeps its --cand best orientations by the first MLEM update, and the work runs over
blocks of voxels, so memory is set by --block-gb and the map's size x --cand, not by the number of orientations.

    python run_index.py <analysisroot> <sample> <dataset> [--phase NAME] [--parfile pars.json] [--check] ...

Paths follow ImageD11's layout: {analysisroot}/{sample}/{sample}_{dataset}/{sample}_{dataset}_dataset.h5 and _sparse.h5.
Parameters: the geometry and the phase's lattice and space group (number) come from pars.json (the dataset's parfile
if it exists here, else pars/pars.json beside PROCESSED_DATA, or --parfile). The scan (y0, dty and omega bins, motor
names) comes from the dataset file (y0 can be overridden with --y0, and must be if the dataset has none; if the
sparse file has no dty column, each scan's dty comes from the dataset); each frame's row is the dty bin of its motor reading. Lengths are in the units of
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
p.add_argument("--grid", type=float, help="orientation grid step, deg (default: the coarsest of 3, 2.5, 2, 1.5, 1 whose "
               "chance completeness is at most --max-chance)")
p.add_argument("--max-chance", type=float, default=0.3, help="chance completeness allowed by the automatic grid (default 0.3)")
p.add_argument("--keep", type=int, default=3000, help="at most this many orientations for the occupancy fit (default 3000)")
p.add_argument("--min-comp", type=float, help="keep orientations with at least this completeness (default: halfway "
               "between the grid's median, the chance level, and its maximum)")
p.add_argument("--iter", type=int, default=10, help="MLEM iterations (default 10)")
p.add_argument("--cand", type=int, default=64, help="candidate orientations per voxel (default 64)")
p.add_argument("--block-gb", type=float, default=1.0, help="memory for one block of voxels' system entries (default 1 GB)")
p.add_argument("--lit", type=float, default=1.0, help="lit threshold, x the median non-empty bin (default 1)")
p.add_argument("--etacut", type=float, default=0.2, help="use reflections with |sin eta| above this (default 0.2)")
p.add_argument("--tth-tol", type=float, help="2theta tolerance of the rings (deg; default: measured per ring)")
p.add_argument("--y0", type=float, help="dty where the rotation axis is in the beam (default: the dataset's y0)")
p.add_argument("--gridstep", type=int, default=1,
               help="voxel = gridstep x dty step; data rows are summed in groups of gridstep to match (default 1)")
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
GRID_STEPS = (3.0, 2.5, 2.0, 1.5, 1.0)  # tried by the automatic grid, coarsest first


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
    if args.y0 is None and "y0" not in attrs:
        raise SystemExit(f"{dsfile} has no y0: give it with --y0")
    Y0 = float(attrs["y0"]) if args.y0 is None else args.y0
    ybin, yedge = h["ybincens"][()], h["ybinedges"][()]
    ds_dty = h["dty"][()]  # [scans, frames]: for sparse files without a dty column
    ds_scans = [x.decode() if isinstance(x, bytes) else str(x) for x in h["scans"][()]]
    oedge = h["obinedges"][()]
parfile = args.parfile
if parfile is None:
    parfile = str(attrs.get("parfile", ""))
    if parfile and not os.path.isabs(parfile):
        parfile = os.path.normpath(os.path.join(dsdir, parfile))
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


def par_path(f: str) -> str:
    """A file named in pars.json: relative to it, or (absolute but missing here) by name beside it."""
    f = os.path.join(pdir, f)
    return f if os.path.exists(f) else os.path.join(pdir, os.path.basename(f))


geo = read_par(par_path(pj["geometry"]["file"]))
ph = read_par(par_path(phases[phase]["file"]))

# ------------------------------------------------------------------------------------------------- scan
G = args.gridstep
YSTEP0, NK0 = float(np.median(np.diff(ybin))), len(ybin)
YSTEP, DTY0, NK = G * YSTEP0, float(ybin[0]) + 0.5 * (G - 1) * YSTEP0, -(-NK0 // G)  # rows summed in groups of G
OM0 = float(oedge[0])
OSTEP = float(np.median(np.diff(oedge)))
N_E, N_O = int(round(360 / B_E)), int(round((oedge[-1] - oedge[0]) / B_O))
N_O -= N_O % R_O
DTYM, OMM = str(attrs["dtymotor"]), str(attrs["omegamotor"])
WL = geo["wavelength"]
# the geometry for predictions and pixel angles (the spreads are unused here)
geom = geom_from_pars(geo, Y0, WL * 2e-3 / 2.355, 1.5e-4, 1.5e-4, sig_beam=YSTEP / 2.355, voxel_size=YSTEP, sig_psf=0.5)
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
QC = 16  # orientations per chunk in the candidate pass
K = args.cand
VB = max(X.block_voxels(QC, NJ, args.block_gb * 1e9), X.block_voxels(K, NJ, args.block_gb * 1e9))  # voxels padded to this
N_CELLS = args.rings * (N_E // R_E) * (N_O // R_O) * NK

log(f"dataset {dsfile}")
log(f"sparse  {sparsefile}")
log(f"pars    {parfile}: phase {phase}, lattice {', '.join(f'{v:g}' for v in lpars)}, space group {sg} ({crystal.sgname})")
log(f"geometry: wavelength {WL:.5f}, distance {geo['distance']:g}; y0 {Y0:.6g}"
    f"{' (override)' if args.y0 is not None else ''}; dty {DTY0:.6g} + {NK} x {YSTEP:.6g} "
    f"(motor {DTYM}{'' if G == 1 else f', rows summed in groups of {G}'}); omega {OM0:.4g} .. {oedge[-1]:.4g} in "
    f"{len(oedge) - 1} frames of {OSTEP:.4g} (motor {OMM})")
log(f"memory: data {N_CELLS * 4 / 1e9:.2f} GB (x ~4 in MLEM); occupancies {NV} voxels x {K} candidates = "
    f"{NV * K * 8 / 1e9:.2f} GB; blocks of {X.block_voxels(QC, NJ, args.block_gb * 1e9)} / {X.block_voxels(K, NJ, args.block_gb * 1e9)} "
    f"voxels (candidates / MLEM), ~{args.block_gb:g} GB each")
log(f"{hkls.shape[0]} hkls in {args.rings} rings at 2theta {', '.join(f'{v:.2f}' for v in ring_tth_all[: args.rings])}; "
    f"grid {'auto' if args.grid is None else args.grid} deg, keep {args.keep}, {args.iter} MLEM iterations, voxels {NR} x {NR} ({NK} dty bins + pad {int(PAD)})")
if args.check:
    raise SystemExit(0)

# ------------------------------------------------------------------------------------------------- 1. histograms
def stream(h, names):  # noqa: ANN001, ANN201
    """(x, val, row) per chunk of CHUNK pixels (padded: val 0, row -1) of the groups names of the sparse file h."""
    for name in names:
        gr = h[name]
        nnz = gr["nnz"][()]
        om_f = gr[f"measurement/{OMM}"][()].astype(np.float32)
        if DTYM in gr["measurement"]:
            dty_f = np.broadcast_to(gr[f"measurement/{DTYM}"][()], nnz.shape)
        else:  # no dty column in this sparse file: the dataset's dty for this scan
            dty_f = ds_dty[ds_scans.index(name)][: len(nnz)]
        k_f = (np.searchsorted(yedge, dty_f) - 1).astype(np.int32)
        k_f = np.where((k_f >= 0) & (k_f < NK0), k_f // G, -1).astype(np.int32)
        ends = np.cumsum(nnz)
        n = int(ends[-1]) if len(ends) else 0
        for s0 in range(0, n, CHUNK):
            m = min(CHUNK, n - s0)
            fr = np.searchsorted(ends, np.arange(s0, s0 + m), side="right")
            pad = lambda a, dt=np.float32: jnp.asarray(np.pad(np.asarray(a).astype(dt), (0, CHUNK - m)))  # noqa: E731
            x = X.pixels_to_x(pad(gr["row"][s0:s0 + m]), pad(gr["col"][s0:s0 + m]), pad(om_f[fr]), geom)
            live = jnp.arange(CHUNK) < m
            yield x, jnp.where(live, pad(gr["intensity"][s0:s0 + m]), 0.0), jnp.where(live, pad(k_f[fr], np.int32), -1), m


with h5py.File(sparsefile, "r") as h:
    names = list(h.keys())
    n_max = max(int(h[nm]["nnz"][()].sum()) for nm in names)
CHUNK = int(min(1 << 24, 1 << max(10, int(np.ceil(np.log2(max(n_max, 1)))))))

# 1a. ring widths (2theta profile of a few groups): the 2theta tolerance per ring, and the rings' spread for eta
TTH_STEP = 0.002
tth_lo = float(ring_tth[0]) - 0.5
n_tth = int(np.ceil((float(ring_tth[-1]) + 0.5 - tth_lo) / TTH_STEP))
prof = jnp.zeros(n_tth, F32)
t1 = time.perf_counter()
with h5py.File(sparsefile, "r") as h:
    sample = [names[i] for i in np.unique(np.linspace(0, len(names) - 1, min(len(names), 9)).round().astype(int))]
    for x, v, _, _ in stream(h, sample):
        prof = prof + X.tth_profile(x, v, tth_lo, TTH_STEP, n=n_tth)
ring_off, ring_hw = X.ring_widths(np.asarray(prof), tth_lo, TTH_STEP, np.asarray(ring_tth))
TTH_TOL = np.abs(ring_off) + ring_hw if args.tth_tol is None else np.full(args.rings, args.tth_tol)
log(f"ring widths from {len(sample)} groups ({time.perf_counter() - t1:.0f} s): offset / half-width (95%) / tolerance, deg: "
    + "; ".join(f"{o:+.3f} / {w:.3f} / {t:.3f}" for o, w, t in zip(ring_off, ring_hw, TTH_TOL)))

# the lit map (no rows, fine bins) for pruning; the MLEM data (1 x 1 deg, with rows). Every frame's row is the dty bin
# of its motor reading, so one group per row and one group for the whole scan both work.
tth_tol = jnp.asarray(TTH_TOL, F32)
H_lit = jnp.zeros(args.rings * N_E * N_O, F32)
H = jnp.zeros(args.rings * (N_E // R_E) * (N_O // R_O) * NK, F32)
t1 = time.perf_counter()
n_tot = 0
with h5py.File(sparsefile, "r") as h:
    for x, v, rows, m in stream(h, names):
        n_tot += m
        H_lit = H_lit + X.histogram(x, v, jnp.zeros(CHUNK, jnp.int32), ring_tth, tth_tol, OM0, B_E, B_O, n_ring=args.rings,
                                    n_e=N_E, n_o=N_O, n_k=1)  # fmt: skip
        H = H + X.histogram(x, v, rows, ring_tth, tth_tol, OM0, B_E * R_E, B_O * R_O, n_ring=args.rings, n_e=N_E // R_E,
                            n_o=N_O // R_O, n_k=NK)  # fmt: skip
jax.block_until_ready(H)
log(f"histograms: {n_tot / 1e6:.0f}M pixels in {time.perf_counter() - t1:.0f} s; MLEM data {H.size / 1e6:.0f}M bins, "
    f"{float(jnp.mean(H > 0)) * 100:.1f}% non-empty")

# ------------------------------------------------------------------------------------------------- 2. pruning
Hs = H_lit.reshape(args.rings, N_E, N_O)
med = float(jnp.median(Hs[Hs > 0]))
for kk in (0.0, 1.0, 3.0, 10.0):
    log(f"  lit fraction at {kk:g} x median: {float(jnp.mean(Hs > kk * med)) * 100:.1f}%")
table = X.lit_table(Hs > args.lit * med)
ring_hw_j = jnp.asarray(ring_hw, F32)


def completeness(rod, step):  # noqa: ANN001, ANN202
    """Completeness [N] of orientations rod [N, 3], with the tolerances of a grid of this step (deg)."""
    out = []
    for s0 in range(0, rod.shape[0], 1 << 15):
        eta, om, ok_ = X.predict(rod[s0:s0 + (1 << 15)], jnp.asarray(B), hkls, geom)
        te, to = X.match_tolerances(eta, ring_of_h, ring_tth, ring_hw_j, X.grid_misorientation(step), OSTEP)
        ok_ = ok_ & (jnp.abs(jnp.sin(jnp.radians(eta))) > args.etacut)
        out.append(X.completeness_tol(table, eta, om, ok_, ring_of_h, te, to, OM0, B_E, B_O, n_e=N_E, n_o=N_O)[0])
    return np.asarray(jnp.concatenate(out))


# grid step: the coarsest whose chance completeness (the median over a random sample of the grid: most orientations
# are wrong) is at most --max-chance. A coarser grid has larger tolerances, so more chance matches.
GRID = args.grid
if GRID is None:
    rng = np.random.default_rng(0)
    for GRID in GRID_STEPS:
        g = X.cubic_fz_grid(GRID)
        c = float(np.median(completeness(jnp.asarray(g[rng.choice(len(g), min(len(g), 1 << 14), replace=False)]), GRID)))
        log(f"  grid {GRID} deg ({len(g)} orientations): chance completeness {c:.2f}")
        if c <= args.max_chance:
            break
    else:
        log(f"  no grid step reaches chance <= {args.max_chance}: using the finest, {GRID} deg")
DELTA = X.grid_misorientation(GRID)
th_ = np.radians(np.asarray(ring_tth) / 2)
log(f"grid {GRID} deg{'' if args.grid is not None else ' (auto)'}: up to {DELTA:.2f} deg from the truth; tolerances at "
    f"|sin eta| = 1 / {args.etacut}: eta "
    f"{DELTA / np.cos(th_[0]) + ring_hw[0] / np.sin(2 * th_[0]) / np.cos(2 * th_[0]):.2f} (ring 0) .. "
    f"{DELTA / np.cos(th_[-1]) + ring_hw[-1] / np.sin(2 * th_[-1]) / np.cos(2 * th_[-1]):.2f} (ring {args.rings - 1}) / "
    f"up to {DELTA / np.cos(th_[-1]) * (1 + np.tan(th_[-1]) / np.tan(np.arcsin(args.etacut))) + ring_hw[-1] / np.sin(2 * th_[-1]) / np.cos(2 * th_[-1]):.2f}; "
    f"omega {DELTA / np.cos(th_[0]) + OSTEP / 2:.2f} / {DELTA / np.cos(th_[-1]) / args.etacut + OSTEP / 2:.2f}")
t1 = time.perf_counter()
rod = jnp.asarray(X.cubic_fz_grid(GRID))
comp = completeness(rod, GRID)
chance = float(np.median(comp))
min_comp = args.min_comp if args.min_comp is not None else chance + 0.5 * (comp.max() - chance)
keep = np.flatnonzero(comp >= min_comp)
keep = keep[np.argsort(comp[keep])[::-1]][: args.keep]
log(f"completeness of {rod.shape[0]} orientations: {time.perf_counter() - t1:.1f} s; median (chance) {chance:.2f}, 99th "
    f"{np.percentile(comp, 99):.2f}, max {comp.max():.2f}; {int((comp >= min_comp).sum())} at >= {min_comp:.2f}"
    + (f", the top {args.keep} kept (--keep; completeness >= {comp[keep[-1]]:.2f})" if (comp >= min_comp).sum() > args.keep else ", all kept"))
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
pos_p = X.pad_voxels(pos, VB)
log(f"candidates: {NV} voxels x {nq} orientations, the top {K} per voxel")
t1 = time.perf_counter()
f0, cand = X.candidates(H, pred, ring_of_h, pos_p, scan, dims, K, X.block_voxels(QC, NJ, args.block_gb * 1e9), qc=QC, log=log)
t2 = time.perf_counter()
vb = X.block_voxels(K, NJ, args.block_gb * 1e9)
jax.block_until_ready(X.forward_sparse(f0, cand, pred, ring_of_h, pos_p, scan, dims, N_CELLS, vb))
t3 = time.perf_counter()
jax.block_until_ready(X.forward_sparse(f0, cand, pred, ring_of_h, pos_p, scan, dims, N_CELLS, vb))
t4 = time.perf_counter()
log(f"MLEM: one forward projection {t4 - t3:.1f} s (first {t3 - t2:.1f} s); estimate ~{2.2 * (t4 - t3) * args.iter / 60:.1f} "
    f"min for {args.iter} iterations")
t1 = time.perf_counter()
f = X.mlem_sparse(H, cand, pred, ring_of_h, pos_p, scan, dims, f0, args.iter, vb, log=log)
f, cand = np.asarray(f)[:NV], np.asarray(cand)[:NV]
log(f"MLEM {args.iter} iterations: {time.perf_counter() - t1:.0f} s")

# ------------------------------------------------------------------------------------------------- 4. TensorMap
kbest = np.argmax(f, 1)
best = cand[np.arange(NV), kbest]  # index into the orientation list
tot = f.sum(1)
share = f[np.arange(NV), kbest] / np.maximum(tot, 1e-30)
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
tag = f"{dsname}_mlem_{args.keep}_{args.iter}_{args.lit:g}_{GRID:g}"
out = os.path.join(args.outdir, f"{tag}_tmap.h5")
if os.path.exists(out):
    os.remove(out)
tmap.to_h5(out)
tmap.to_paraview(out)
np.savez(os.path.join(args.outdir, f"{tag}.npz"), f=f, cand=cand, rod=np.asarray(rod_s[:nq]), comp=comp[keep])
log(f"-> {out}")
