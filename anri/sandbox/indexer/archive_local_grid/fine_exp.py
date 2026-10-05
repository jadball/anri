"""Stage 3 on the fine sparse histogram: local grids around the first local pass's populations."""

import os
import sys
import time

import anri.utils

anri.utils.setup(n_cpu=8)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import anri.crystal
import anri.index as ix
import anri.io
import anri.phantom
from fine import fine_mlem, sparse_histogram

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.1f} s] {m}", flush=True)


root = sys.argv[1]
HALF, STEP_L = float(sys.argv[2]), float(sys.argv[3])
B_E, B_O = float(sys.argv[4]), float(sys.argv[5])  # fine bins (deg)
BEAM = float(sys.argv[6]) if len(sys.argv) > 6 else 1.4  # beam FWHM for the model
r = np.load(os.path.join(root, "stage2.npz"))
p1 = np.load(os.path.join(root, "local_1_0.5.npz"))
B, ops, pos = r["B"], r["ops"], r["pos"]
NV = int(os.environ.get("NVOX", "0"))  # profile on the first NV voxels
sub = slice(0, NV) if NV else slice(None)
occ = p1["occ"][sub]
wl = 0.2843
rings = ix.ring_table(np.array([3.5966] * 3 + [90.0] * 3), 225, wl, 6)
ds = anri.io.read_dataset(os.path.join(root, "phantom", "phantom_am316l", "phantom_am316l_dataset.h5"))
geo, _, _ = anri.io.read_pars_json(ds["parfile"])
geom = anri.io.geom_from_pars(geo, ds["y0"], wl * 2e-3 / 2.355, 1.5e-4, 1.5e-4, sig_beam=BEAM / 2.355, voxel_size=1.0)
geom = {k: jnp.asarray(v, jnp.float32) if np.issubdtype(np.asarray(v).dtype, np.floating) else v for k, v in geom.items()}
nk, om0 = int(r["nk"]), float(r["om0"])
n_e, n_o = round(360 / B_E), round(180 / B_O)
cache = os.path.join(root, f"fine_{B_E:g}_{B_O:g}.npz")
if os.path.exists(cache):
    data = dict(np.load(cache))
    data["n_rows"] = int(data["n_rows"])
else:
    chunk = 1 << 20
    stream = anri.io.stream_sparse(os.path.join(root, "sparse.h5"), ds["ybinedges"], ds["omegamotor"], ds["dtymotor"], chunk)
    data = sparse_histogram(stream, geom, rings["tth"], r["tth_tol"], om0, (B_E, B_O, n_e, n_o), nk, chunk)
    np.savez(cache, **data)
log(f"fine histogram {B_E} x {B_O} deg: {len(data['value']) / 1e6:.2f}M non-empty bins")

# local grids around pass 1's top two populations
g = np.arange(-HALF, HALF + 1e-9, STEP_L)
off = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
R_off = anri.phantom.axis_angle(off + (np.linalg.norm(off, axis=1, keepdims=True) == 0), np.linalg.norm(off, axis=1))
P = 2
use_p = p1["frac"][sub][:, :P] >= 0.1
use_p[:, 0] = True
base = np.where(use_p[:, :, None, None], p1["U_new"][sub][:, :P], p1["U_new"][sub][:, :1])
U_c = (R_off[None, None] @ base[:, :, None]).reshape(len(occ), -1, 3, 3).astype(np.float32)  # [No, K, 3, 3]
K = U_c.shape[1]
f0 = np.repeat(use_p.astype(np.float32), len(R_off), 1)
vb = 16
n_pad = -len(occ) % vb
pad = lambda a: np.concatenate([a, np.repeat(a[:1], n_pad, 0)])  # noqa: E731
U_b = jnp.asarray(pad(U_c).reshape(-1, vb, K, 3, 3))
pos_b = jnp.asarray(pad(pos[occ]).reshape(-1, vb, 3))
f0_b = jnp.asarray(np.concatenate([f0, np.zeros((n_pad, K), np.float32)]).reshape(-1, vb, K))
scan = {"y0": float(r["scan_y0"]), "dty0": float(r["dty0"]), "ystep": 1.0, "n_rows": nk, "om0": om0, "B": jnp.asarray(B)}
n_side = int(np.ceil((0.5 + 3 * BEAM / 2.355) / 1.0))
log(f"{len(occ)} voxels x {K} candidates; rows +-{n_side}")
f = fine_mlem(data, U_b, pos_b, f0_b, rings, geom, scan, (B_E, B_O, n_e, n_o), 0.2, n_side, 20, log=log)
f = np.asarray(f).reshape(-1, K)[: len(occ)]
log("fine MLEM done")
U_flat = U_c.reshape(-1, 3, 3)
cand = np.arange(len(U_flat)).reshape(len(occ), K)
frac, U_new, spread, _ = ix.populations(f, cand, U_flat, ops, 2 * HALF + STEP_L, p=4)

tp_pos, tp_U = r["truth_pos"], r["truth_U"]
near = np.argmin(np.linalg.norm(tp_pos[:, None, :2] - pos[None, occ, :2], axis=2), 1)
mean_err, spread_true, spread_fit = [], [], []
for i in range(len(occ)):
    m = near == i
    if m.sum() == 0:
        continue
    Ut = tp_U[m]
    if anri.crystal.disorientation(np.repeat(Ut[:1], len(Ut), 0), Ut, ops).max() > 2:
        continue
    u, _, vt = np.linalg.svd(Ut.mean(0))
    um = u @ vt
    mean_err.append(anri.crystal.disorientation(U_new[i : i + 1, 0], um[None], ops)[0])
    spread_true.append(np.sqrt(np.mean(anri.crystal.disorientation(np.repeat(um[None], len(Ut), 0), Ut, ops) ** 2)))
    spread_fit.append(spread[i, 0])
mean_err = np.array(mean_err)
pres = frac >= 0.1
pres[:, 0] = True
log(f"fine {HALF}/{STEP_L} on {B_E} x {B_O}: vs each voxel's mean truth ({len(mean_err)} voxels): median {np.median(mean_err):.3f} deg, "
    f"within 0.1 {np.mean(mean_err < 0.1) * 100:.0f}%, within 0.25 {np.mean(mean_err < 0.25) * 100:.0f}%, 90th {np.percentile(mean_err, 90):.2f}; "
    f"spread median {np.median(spread_fit):.2f} (true {np.median(spread_true):.2f}); 2+ populations {np.mean(pres.sum(1) >= 2) * 100:.0f}%")
np.savez(os.path.join(root, f"fine_result_{HALF:g}_{STEP_L:g}_{B_E:g}_{B_O:g}.npz"), f=f, U_new=U_new, frac=frac, spread=spread)
