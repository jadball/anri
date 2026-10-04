"""Stage 1 on the fe_cells phantom: coarse histogram, cubic FZ grid, completeness pruning; checked against the truth.

python run_phantom.py <phantom> [grid_step_deg]
"""

import os
import sys
import time
import warnings

T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.2f} s] {msg}", flush=True)


import anri.utils  # noqa: E402

anri.utils.setup(n_cpu=4)
import jax  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
jax.config.update("jax_compilation_cache_dir", os.path.join(HERE, "..", ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from ImageD11.sinograms.tensor_map import TensorMap  # noqa: E402

import index as X  # noqa: E402
from anri.crystal import Structure, lpars_to_B  # noqa: E402
from anri.io import entries_from_tensormap, geom_from_pars, motor_grid  # noqa: E402

NAME = sys.argv[1]
STEP_DEG = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
TOL_E = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0  # matching tolerance (deg): eta
TOL_O = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5  # omega
ETACUT = float(sys.argv[5]) if len(sys.argv) > 5 else 0.4
OSTEP, STEP = 0.05, 0.1
B_E, B_O = 0.5, 0.25  # coarse bins (deg): eta, omega
N_RINGS = 4
CHUNK = 1 << 22
F32 = jnp.float32
PH = os.path.join(HERE, "..", "phantoms")
pars = {
    "y_center": 1049.9, "y_size": 75.0, "tilt_y": -2e-3, "z_center": 1116.5, "z_size": 75.0, "tilt_z": 3e-3, "tilt_x": 1e-3,
    "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1, "wavelength": 12.398419843320026 / 43.0, "wedge": 0.0,
    "chi": 0.0, "omegasign": 1.0, "t_x": 0.0, "t_y": 0.0, "t_z": 0.0,
}  # fmt: skip
WL = pars["wavelength"]
geom = geom_from_pars(pars, 0.0, WL * 2e-3 / 2.355, 1.5e-4, 1.5e-4, sig_beam=0.1 / 2.355, voxel_size=STEP, sig_psf=0.5)
geom = {k: jnp.asarray(v, F32) if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating) else v for k, v in geom.items()}

# hkls of the first rings
fe = Structure.from_cif(os.path.join(PH, "..", "..", "..", "tests", "data", "cif", "Fe.cif"))
fe.make_hkls(1.605, WL)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    t = fe.rings_table
hkls_all = np.stack([t["h"], t["k"], t["l"]], 1).astype(np.float32)
B = np.asarray(lpars_to_B(jnp.array([2.8694, 2.8694, 2.8694, 90.0, 90.0, 90.0])), np.float32)
tth_h = np.degrees(2 * np.arcsin(WL * np.linalg.norm(hkls_all @ B.T, axis=1) / 2))
ring_tth_all, ring_all = np.unique(np.round(tth_h, 4), return_inverse=True)
use_h = ring_all < N_RINGS
hkls, ring_of_h = jnp.asarray(hkls_all[use_h]), jnp.asarray(np.repeat(ring_all[use_h], 2), jnp.int32)  # j = h * 2 + branch
ring_tth = jnp.asarray(ring_tth_all[:N_RINGS], F32)
log(f"{int(use_h.sum())} hkls in the first {N_RINGS} rings (2theta {', '.join(f'{v:.2f}' for v in ring_tth_all[:N_RINGS])})")

# data (rendered by ../moments/refine_cells.py)
d = np.load(os.path.join(HERE, "..", "moments", f"data_{NAME}_{OSTEP:g}.npz"))
row_np, frame_np, pixel_np, val_np = d["row"].astype(np.int32), d["frame"].astype(np.int32), d["pixel"], d["val"]
omega, dty = motor_grid((0.0, 180.0), OSTEP, (-5.3, 5.3), STEP)
NK, NF = omega.shape
DET = (2162, 2068)
N_E, N_O = int(round(360 / B_E)), int(round(NF * OSTEP / B_O))
NPIX = len(val_np)
log(f"{NPIX / 1e6:.1f}M pixels; histogram {N_RINGS} x {N_E} x {N_O} x {NK} = {N_RINGS * N_E * N_O * NK / 1e6:.0f}M bins")

t1 = time.perf_counter()
H = jnp.zeros(N_RINGS * N_E * N_O * NK, F32)
om_pix = omega[row_np, frame_np].astype(np.float32)
for s0 in range(0, NPIX, CHUNK):
    n = min(CHUNK, NPIX - s0)
    pad = lambda a: jnp.asarray(np.pad(a[s0:s0 + n], (0, CHUNK - n)))  # noqa: E731
    x = X.pixels_to_x(pad((pixel_np // DET[1]).astype(np.float32)), pad((pixel_np % DET[1]).astype(np.float32)), pad(om_pix), geom)
    v = jnp.where(jnp.arange(CHUNK) < n, pad(val_np), 0.0)
    H = H + X.histogram(x, v, pad(row_np), ring_tth, 0.1, 0.0, B_E, B_O, n_ring=N_RINGS, n_e=N_E, n_o=N_O, n_k=NK)
H = H.reshape(N_RINGS, N_E, N_O, NK)
jax.block_until_ready(H)
log(f"histogram: {time.perf_counter() - t1:.2f} s; {float(jnp.mean(H > 0)) * 100:.2f}% of bins non-empty")

# row-summed: lit bins
Hs = H.sum(-1)
lit = Hs > 1e-3 * float(jnp.median(Hs[Hs > 0]))
log(f"row-summed: {float(jnp.mean(lit)) * 100:.2f}% of (ring, eta, omega) bins lit")
lit = X.dilate(lit, de=int(round(TOL_E / B_E)), do=int(round(TOL_O / B_O)))
log(f"dilated by +-{TOL_E} deg in eta, +-{TOL_O} deg in omega: {float(jnp.mean(lit)) * 100:.2f}% lit; |sin eta| > {ETACUT}")

# grid and completeness
t1 = time.perf_counter()
rod = jnp.asarray(X.cubic_fz_grid(STEP_DEG))
comp, n_pred = [], []
for s0 in range(0, rod.shape[0], 1 << 15):
    r_ = rod[s0:s0 + (1 << 15)]
    eta, om, ok = X.predict(r_, jnp.asarray(B), hkls, geom)
    c_, n_ = X.completeness(lit, eta, om, ok, ring_of_h, 0.0, B_E, B_O, n_e=N_E, n_o=N_O, etacut=ETACUT)
    comp.append(c_)
    n_pred.append(n_)
comp, n_pred = np.asarray(jnp.concatenate(comp)), np.asarray(jnp.concatenate(n_pred))
log(f"grid: {rod.shape[0]} orientations at {STEP_DEG} deg; completeness {time.perf_counter() - t1:.2f} s; "
    f"median {np.median(comp):.3f}, 99th {np.percentile(comp, 99):.3f}; predictions per orientation median {np.median(n_pred):.0f}")

# truth: grain mean orientations
tmap = TensorMap.from_h5(os.path.join(PH, NAME, f"{NAME}_tmap.h5"))
ent = entries_from_tensormap(tmap)
ri, rj = np.nonzero(TensorMap.map_order_to_recon_order(tmap.phase_ids) == 0)
grain = TensorMap.map_order_to_recon_order(tmap.labels)[ri, rj].astype(np.int32)
U_vox = np.linalg.inv(ent["ubi"]) @ np.linalg.inv(B)  # UB B^-1 (strain ~1e-4: close to a rotation)
U_vox = np.linalg.svd(U_vox)[0] @ np.linalg.svd(U_vox)[2]  # nearest rotation
NG = grain.max() + 1
U_g = np.stack([U_vox[grain == g].mean(0) for g in range(NG)])
U_g = np.linalg.svd(U_g)[0] @ np.linalg.svd(U_g)[2]
rod_true = jnp.asarray(X.to_fz(U_g), F32)
eta, om, ok = X.predict(rod_true, jnp.asarray(B), hkls, geom)
c_true, _ = X.completeness(lit, eta, om, ok, ring_of_h, 0.0, B_E, B_O, n_e=N_E, n_o=N_O, etacut=ETACUT)
c_true = np.asarray(c_true)
# nearest grid point of each true orientation
U_grid = np.asarray(X.rod_to_mat(rod))
near = []
for g in range(NG):
    dis = X.disorientation(np.repeat(U_g[g][None], len(U_grid), 0), U_grid)
    k = int(np.argmin(dis))
    near.append((k, dis[k]))
c_near = np.array([comp[k] for k, _ in near])
rank = np.array([(comp > comp[k]).sum() for k, _ in near])
log(f"true grain orientations: completeness at the truth median {np.median(c_true):.3f} (min {c_true.min():.3f}); "
    f"at the nearest grid point (median {np.median([d_ for _, d_ in near]):.2f} deg away) median {np.median(c_near):.3f} "
    f"(min {c_near.min():.3f}); their rank in the grid: median {np.median(rank):.0f}, max {rank.max()}")
for thr in (0.5, 0.7, 0.8, 0.9):
    log(f"  threshold {thr}: {(comp >= thr).sum()} orientations survive; true grains kept {(c_near >= thr).sum()} of {NG}")
np.savez(os.path.join(HERE, f"stage1_{NAME}_{STEP_DEG:g}.npz"), rod=np.asarray(rod), comp=comp, c_near=c_near, rank=rank)

# ---------------------------------------------------------------------------------- stage 2: MLEM occupancy
from anri.geom import sample_to_lab  # noqa: E402

THR, N_ITER, QC = 0.9, int(os.environ.get("N_ITER", "30")), 16
keep = np.flatnonzero(comp >= THR)
nq = len(keep)
pad_q = -nq % QC
rod_s = jnp.asarray(np.concatenate([np.asarray(rod)[keep], np.repeat(np.asarray(rod)[keep[:1]], pad_q, 0)]))
eta, om, ok, lp = X.predict_lp(rod_s, jnp.asarray(B), hkls, geom)
F2j = jnp.asarray(np.repeat(t["intensity"].to_numpy()[use_h], 2), F32)
use = ok & (jnp.abs(jnp.sin(jnp.radians(eta))) > ETACUT) & (jnp.arange(nq + pad_q) < nq)[:, None]
w = jnp.where(use, lp * F2j[None], 0.0)
pred = (eta, om, use, w)
# data at the grid's scale: 1 deg in eta and omega
R_E, R_O = 2, 4
d2 = H.reshape(N_RINGS, N_E // R_E, R_E, N_O // R_O, R_O, NK).sum((2, 4)).ravel()
dims = (B_E * R_E, B_O * R_O, N_E // R_E, N_O // R_O)
pos = jnp.asarray(ent["pos"], F32)
scan = {"y0": float(geom["y0"]), "dty0": float(dty[0, 0]), "ystep": STEP, "n_rows": NK, "om0": 0.0}
# check the row geometry against the renderer's sample_to_lab
p_t, o_t = pos[123], 37.0
y_ref = float(sample_to_lab(p_t, o_t, 0.0, 0.0, scan["y0"], scan["y0"])[1])
y_mine = float(p_t[0] * np.sin(np.radians(o_t)) + p_t[1] * np.cos(np.radians(o_t)))
assert abs(y_ref - y_mine) < 1e-4, (y_ref, y_mine)
log(f"stage 2: {nq} orientations x {pos.shape[0]} voxels; data {d2.shape[0] / 1e6:.1f}M cells at {dims[0]} x {dims[1]} deg")
t1 = time.perf_counter()
Af = X.forward(jnp.ones((pos.shape[0], nq + pad_q), F32), pred, ring_of_h, pos, scan, dims, d2.shape[0])
jax.block_until_ready(Af)
t2 = time.perf_counter()
Af = X.forward(jnp.ones((pos.shape[0], nq + pad_q), F32), pred, ring_of_h, pos, scan, dims, d2.shape[0])
jax.block_until_ready(Af)
log(f"forward projection: first {t2 - t1:.2f} s (with compile), then {time.perf_counter() - t2:.3f} s")
t1 = time.perf_counter()
f = X.mlem(d2, pred, ring_of_h, pos, scan, dims, jnp.ones((pos.shape[0], nq + pad_q), F32), N_ITER, log=log)
jax.block_until_ready(f)
log(f"MLEM {N_ITER} iterations: {time.perf_counter() - t1:.1f} s")
f = np.asarray(f)[:, :nq]
U_s = np.asarray(X.rod_to_mat(rod_s[:nq]))
best = np.argmax(f, 1)
ang_best = X.disorientation(U_s[best], U_vox)
# occupancy-weighted mean of the orientations near each voxel's best (within 2.5 deg), in Rodrigues space
rs = np.asarray(rod_s[:nq])
mean_rod = np.zeros((pos.shape[0], 3))
for v in range(pos.shape[0]):
    dv = np.linalg.norm(rs - rs[best[v]], axis=1)
    m = dv < np.tan(np.radians(2.5) / 2)
    mean_rod[v] = (f[v, m, None] * rs[m]).sum(0) / f[v, m].sum()
ang_mean = X.disorientation(np.asarray(X.rod_to_mat(jnp.asarray(mean_rod, F32))), U_vox)
share = np.sort(f, 1)[:, ::-1]
log(f"top orientation per voxel vs truth: median {np.median(ang_best):.3f}, 90th {np.percentile(ang_best, 90):.3f} deg; "
    f"< 1 deg {np.mean(ang_best < 1) * 100:.1f}%, < 2 deg {np.mean(ang_best < 2) * 100:.1f}%")
log(f"occupancy-weighted mean near the top: median {np.median(ang_mean):.3f}, 90th {np.percentile(ang_mean, 90):.3f} deg; "
    f"< 0.5 deg {np.mean(ang_mean < 0.5) * 100:.1f}%, < 1 deg {np.mean(ang_mean < 1) * 100:.1f}%")
log(f"occupancy: total per voxel median {np.median(f.sum(1)):.3g}; top share median {np.median(share[:, 0] / f.sum(1)):.2f}")
np.savez(os.path.join(HERE, f"stage2_{NAME}.npz"), f=f, rod=rs, ang_best=ang_best, ang_mean=ang_mean)
log("done")
