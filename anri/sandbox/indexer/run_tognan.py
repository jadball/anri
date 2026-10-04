"""Index Tognan AM 316L (AP1_1, z0) from scratch: coarse histogram, cubic FZ grid + completeness pruning, MLEM
occupancy on a 201 x 201 grid of 1 um voxels; write a TensorMap with the highest-occupancy orientation per voxel.

No spatial distortion correction yet (coarse bins of 0.5 deg ~ 19 pixels). Env: N_KEEP (orientations kept),
N_ITER (MLEM iterations), LIT_K (lit threshold, x the median non-empty row-summed bin).
"""

import os
import time

T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f} s] {msg}", flush=True)


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
from anri.crystal import lpars_to_B  # noqa: E402
from anri.io import geom_from_pars  # noqa: E402

ROOT = os.path.expanduser("~/Data/test_data_3DXRD/S3DXRD/Tognan_AM_316L")
DSDIR = os.path.join(ROOT, "PROCESSED_DATA/20261002_JADB/AP1_1/AP1_1_s3dxrd_z_1micron_z0")
SPARSE = os.path.join(DSDIR, "AP1_1_s3dxrd_z_1micron_z0_sparse.h5")
N_KEEP = int(os.environ.get("N_KEEP", "3000"))
N_ITER = int(os.environ.get("N_ITER", "10"))
LIT_K = float(os.environ.get("LIT_K", "1.0"))
F32 = jnp.float32
B_E, B_O = 0.5, 0.25
N_RINGS = 6
# GRID_DEG, TOL_E, TOL_O, ETACUT = 1.0, 0.5, 1.0, 0.1  # was 0.5 etacut (why???)
GRID_DEG, TOL_E, TOL_O, ETACUT = 2.5, 0.5, 1.0, 0.1  # was 0.5 etacut (why???)
CHUNK = 1 << 24
A_FCC = 3.5965991760810625

# geometry from ImageD11 parameters
pars = {}
for line in open(os.path.join(ROOT, "pars/geometry.par")):
    k, v = line.split()[:2]
    try:
        pars[k] = float(v)
    except ValueError:
        pars[k] = v
WL = pars["wavelength"]
Y0, YSTEP, DTY0, NK = -0.14, 1.0, -100.0, 201
geom = geom_from_pars(pars, Y0, WL * 2e-3 / 2.355, 1.5e-4, 1.5e-4, sig_beam=1.4 / 2.355, voxel_size=YSTEP, sig_psf=0.5)
geom = {k: jnp.asarray(v, F32) if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating) else v for k, v in geom.items()}

# FCC reflections of the first rings
B = np.asarray(lpars_to_B(jnp.array([A_FCC] * 3 + [90.0] * 3)), np.float32)
g = np.array([h for h in np.ndindex(13, 13, 13)]) - 6
g = g[(np.abs(g).sum(1) > 0) & ((g % 2 == 0).all(1) | (g % 2 == 1).all(1))]  # h, k, l all even or all odd
tth = np.degrees(2 * np.arcsin(WL * np.linalg.norm(g @ B.T, axis=1) / 2))
ring_tth_all, ring_all = np.unique(np.round(tth, 4), return_inverse=True)
sel = ring_all < N_RINGS
hkls = jnp.asarray(g[sel], F32)
ring_of_h = jnp.asarray(np.repeat(ring_all[sel], 2), jnp.int32)  # j = h * 2 + branch
ring_tth = jnp.asarray(ring_tth_all[:N_RINGS], F32)
log(f"FCC a = {A_FCC:.4f}, wavelength {WL:.5f}: {int(sel.sum())} hkls in rings at 2theta {', '.join(f'{v:.2f}' for v in ring_tth_all[:N_RINGS])}")

# 1. histogram, streamed row by row from the sparse file
OM0, N_E, N_O = -90.0, int(round(360 / B_E)), int(round(181.0 / B_O))
H = jnp.zeros(N_RINGS * N_E * N_O * NK, F32)
t1 = time.perf_counter()
n_tot = 0
with h5py.File(SPARSE, "r") as h:
    for name in h.keys():
        gr = h[name]
        rr, cc, ii, nnz = gr["row"][()], gr["col"][()], gr["intensity"][()], gr["nnz"][()]
        om_f = gr["measurement/rot_center"][()].astype(np.float32)
        k = int(round((float(gr["measurement/dty"][0]) - DTY0) / YSTEP))
        om_p = np.repeat(om_f, nnz)
        n = len(ii)
        n_tot += n
        for s0 in range(0, n, CHUNK):
            m = min(CHUNK, n - s0)
            pad = lambda a, dt=np.float32: jnp.asarray(np.pad(a[s0:s0 + m].astype(dt), (0, CHUNK - m)))  # noqa: E731
            x = X.pixels_to_x(pad(rr), pad(cc), pad(om_p), geom)
            v = jnp.where(jnp.arange(CHUNK) < m, pad(ii), 0.0)
            H = H + X.histogram(x, v, jnp.full(CHUNK, k, jnp.int32), ring_tth, 0.1, OM0, B_E, B_O, n_ring=N_RINGS, n_e=N_E,
                                n_o=N_O, n_k=NK)  # fmt: skip
H = H.reshape(N_RINGS, N_E, N_O, NK)
jax.block_until_ready(H)
log(f"histogram: {n_tot / 1e9:.2f}G pixels in {time.perf_counter() - t1:.0f} s; {float(jnp.mean(H > 0)) * 100:.1f}% of bins non-empty; "
    f"{float(H.sum()) / max(float(jnp.sum(jnp.asarray(1.0))), 1):.3g} counts on the rings")

# 2. completeness pruning on the cubic FZ grid
Hs = H.sum(-1)
med = float(jnp.median(Hs[Hs > 0]))
for kk in (0.0, 1.0, 3.0, 10.0):
    log(f"  lit fraction at {kk:g} x median: {float(jnp.mean(Hs > kk * med)) * 100:.1f}%")
lit = X.dilate(Hs > LIT_K * med, de=int(round(TOL_E / B_E)), do=int(round(TOL_O / B_O)))
log(f"lit (> {LIT_K} x median, dilated +-{TOL_E} / +-{TOL_O} deg): {float(jnp.mean(lit)) * 100:.1f}%")
t1 = time.perf_counter()
rod = jnp.asarray(X.cubic_fz_grid(GRID_DEG))
comp = []
for s0 in range(0, rod.shape[0], 1 << 15):
    eta, om, ok = X.predict(rod[s0:s0 + (1 << 15)], jnp.asarray(B), hkls, geom)
    comp.append(X.completeness(lit, eta, om, ok, ring_of_h, OM0, B_E, B_O, n_e=N_E, n_o=N_O, etacut=ETACUT)[0])
comp = np.asarray(jnp.concatenate(comp))
order = np.argsort(comp)[::-1]
keep = order[:N_KEEP]
log(f"completeness of {rod.shape[0]} orientations: {time.perf_counter() - t1:.1f} s; median {np.median(comp):.2f}, 99th "
    f"{np.percentile(comp, 99):.2f}, max {comp.max():.2f}; keeping the top {N_KEEP} (completeness >= {comp[keep[-1]]:.2f})")

# 3. MLEM occupancy on a 201 x 201 grid
QC = 4
nq = len(keep)
pad_q = -nq % QC
rod_s = jnp.asarray(np.concatenate([np.asarray(rod)[keep], np.repeat(np.asarray(rod)[keep[:1]], pad_q, 0)]))
eta, om, ok, lp = X.predict_lp(rod_s, jnp.asarray(B), hkls, geom)
use = ok & (jnp.abs(jnp.sin(jnp.radians(eta))) > ETACUT) & (jnp.arange(nq + pad_q) < nq)[:, None]
pred = (eta, om, use, jnp.where(use, lp, 0.0))  # F^2 = 1 at this level
R_E, R_O = 2, 4
d2 = H.reshape(N_RINGS, N_E // R_E, R_E, N_O // R_O, R_O, NK).sum((2, 4)).ravel()
dims = (B_E * R_E, B_O * R_O, N_E // R_E, N_O // R_O)
n = NK
ri, rj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
sx, sy = recon_to_sample(ri, rj, (n, n), YSTEP)
pos = jnp.asarray(np.stack([sx.ravel(), sy.ravel(), np.zeros(n * n)], 1), F32)
scan = {"y0": Y0, "dty0": DTY0, "ystep": YSTEP, "n_rows": NK, "om0": OM0}
f0 = jnp.ones((pos.shape[0], nq + pad_q), F32)
t1 = time.perf_counter()
Af = X.forward(f0, pred, ring_of_h, pos, scan, dims, d2.shape[0], qc=QC)
jax.block_until_ready(Af)
t2 = time.perf_counter()
Af = X.forward(f0, pred, ring_of_h, pos, scan, dims, d2.shape[0], qc=QC)
jax.block_until_ready(Af)
t3 = time.perf_counter()
log(f"MLEM: {pos.shape[0]} voxels x {nq} orientations; one forward projection {t3 - t2:.1f} s (first {t2 - t1:.1f} s); "
    f"estimate ~{2.2 * (t3 - t2) * N_ITER / 60:.1f} min for {N_ITER} iterations")
t1 = time.perf_counter()
f = X.mlem(d2, pred, ring_of_h, pos, scan, dims, f0, N_ITER, log=log, qc=QC)
f = np.asarray(f)[:, :nq]
log(f"MLEM {N_ITER} iterations: {time.perf_counter() - t1:.0f} s")

# 4. TensorMap: highest occupancy wins
best = np.argmax(f, 1)
tot = f.sum(1)
share = f[np.arange(len(best)), best] / np.maximum(tot, 1e-30)
U = np.asarray(X.rod_to_mat(rod_s[:nq]))[best]
ubi = np.linalg.inv(U @ B)
occupied = tot > 0.05 * np.percentile(tot, 99)
log(f"voxels with occupancy: {occupied.sum()} of {len(tot)}; top share median {np.median(share[occupied]):.2f}; "
    f"distinct orientations used {len(np.unique(best[occupied]))}")
ubi_r = np.where(occupied[:, None, None], ubi, np.nan).reshape(n, n, 3, 3)
to_map = TensorMap.recon_order_to_map_order
tmap = TensorMap(maps={"UBI": to_map(ubi_r), "phase_ids": to_map(np.where(occupied, 0, -1).reshape(n, n)),
                       "occupancy": to_map(tot.reshape(n, n)), "best_share": to_map(share.reshape(n, n)),
                       "orientation_id": to_map(np.where(occupied, best, -1).reshape(n, n))}, steps=[1.0, YSTEP, YSTEP])  # fmt: skip
tmap.phases = {0: unitcell([A_FCC] * 3 + [90.0] * 3, 225, name="Fe")}
tmap.get_ipf_maps()
_ = tmap.euler
_ = tmap.eps_devia
_ = tmap.eps_crystal
out = os.path.join(HERE, f"tognan_mlem_tmap_{N_KEEP}_{N_ITER}_{LIT_K}_{GRID_DEG}.h5")
if os.path.exists(out):
    os.remove(out)
tmap.to_h5(out)
np.savez(os.path.join(HERE, "tognan_mlem.npz"), f=f, rod=np.asarray(rod_s[:nq]), comp=comp[keep])
log(f"-> {out}")
