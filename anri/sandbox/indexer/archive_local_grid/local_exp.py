"""Experiment: local grids around each voxel's populations, against the coarse 1-degree histogram."""

import os
import sys
import time

import anri.utils

anri.utils.setup(n_cpu=8)
import jax.numpy as jnp
import numpy as np

import anri.crystal
import anri.index as ix
import anri.io
import anri.phantom

T0 = time.time()
root = sys.argv[1]
HALF, STEP_L = float(sys.argv[2]), float(sys.argv[3])  # local grid half-width and step (deg)
r = np.load(os.path.join(root, "stage2.npz"))
H = jnp.asarray(np.load(os.path.join(root, "H.npy")))
B, ops = r["B"], r["ops"]
U_pop, present, pos = r["U_pop"], r["present"], r["pos"]
nk, om0 = int(r["nk"]), float(r["om0"])
wl = 0.2843
rings = ix.ring_table(np.array([3.5966] * 3 + [90.0] * 3), 225, wl, 6)
ds = anri.io.read_dataset(os.path.join(root, "phantom", "phantom_am316l", "phantom_am316l_dataset.h5"))
geo, _, _ = anri.io.read_pars_json(ds["parfile"])
geom = anri.io.geom_from_pars(geo, ds["y0"], wl * 2e-3 / 2.355, 1.5e-4, 1.5e-4, sig_beam=1.0, voxel_size=1.0)
geom = {k: jnp.asarray(v, jnp.float32) if np.issubdtype(np.asarray(v).dtype, np.floating) else v for k, v in geom.items()}

# local offsets: rotation vectors on a cube grid
g = np.arange(-HALF, HALF + 1e-9, STEP_L)
off = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
R_off = anri.phantom.axis_angle(off + (np.linalg.norm(off, axis=1, keepdims=True) == 0), np.linalg.norm(off, axis=1))
n_off = len(R_off)
P = 2  # populations per voxel refined
occ = np.flatnonzero(r["occupied"])
nv = len(pos)
# candidate orientations per occupied voxel: R_off @ U_pop[v, p] for its present populations (absent: repeat the main)
use_p = present[occ, :P]
base = np.where(use_p[:, :, None, None], U_pop[occ, :P], U_pop[occ, :1])  # [No, P, 3, 3]
U_loc = (R_off[None, None] @ base[:, :, None]).reshape(-1, 3, 3).astype(np.float32)  # [No * P * n_off]
K = P * n_off
cand_o = np.arange(len(U_loc)).reshape(len(occ), K)
f0_o = np.repeat(use_p.astype(np.float32), n_off, 1)  # absent populations start at 0 (MLEM keeps them there)
print(f"{len(occ)} voxels x {K} local candidates ({n_off} per population): {len(U_loc)} orientations", flush=True)
pred = ix.predictions(U_loc, B, rings, geom, 0.2)
scan = {"y0": float(r["scan_y0"]), "dty0": float(r["dty0"]), "ystep": 1.0, "n_rows": nk, "om0": om0}
bins = (1.0, 1.0, 360, 180)
vb = ix.block_voxels(K, len(rings["ring_j"]), 1e9)
pos_o = ix.pad_voxels(pos[occ], vb)
n_pad = len(pos_o) - len(occ)
cand_p = jnp.asarray(np.concatenate([cand_o, np.repeat(cand_o[:1], n_pad, 0)]), jnp.int32)
f0_p = jnp.asarray(np.concatenate([f0_o, np.zeros((n_pad, K), np.float32)]))
t1 = time.time()
f = ix.mlem(H, cand_p, pred, jnp.asarray(rings["ring_j"]), pos_o, scan, bins, f0_p, 20, vb)
f = np.asarray(f)[: len(occ)]
print(f"local MLEM: {time.time() - t1:.0f} s", flush=True)
frac, U_new, spread, _ = ix.populations(f, cand_o, U_loc, ops, 2 * HALF + STEP_L, p=4)
# compare with the truth
tp_pos, tp_U = r["truth_pos"], r["truth_U"]
near = np.argmin(np.linalg.norm(tp_pos[:, None, :2] - pos[None, occ, :2], axis=2), 1)
pres_new = frac >= 0.1
pres_new[:, 0] = True
err = np.stack([anri.crystal.disorientation(U_new[near, p], tp_U, ops) for p in range(4)], 1)
err = np.where(pres_new[near], err, np.inf)
print(f"local {HALF}/{STEP_L}: main within 1 deg {np.mean(err[:, 0] < 1) * 100:.1f}%, within 0.5 {np.mean(err[:, 0] < 0.5) * 100:.1f}%, "
      f"within 0.25 {np.mean(err[:, 0] < 0.25) * 100:.1f}%, median {np.median(err[:, 0]):.2f}; closest within 1 deg {np.mean(err.min(1) < 1) * 100:.1f}%; "
      f"spread median {np.median(spread[:, 0]):.2f}; voxels with 2+ populations {np.mean(pres_new.sum(1) >= 2) * 100:.0f}%")
# against each indexed voxel's mean true orientation (voxels whose phantom voxels lie within 2 deg of one another)
mean_err, spread_true = [], []
for i in range(len(occ)):
    m = near == i
    if m.sum() == 0:
        continue
    Ut = tp_U[m]
    d = anri.crystal.disorientation(np.repeat(Ut[:1], len(Ut), 0), Ut, ops)
    if d.max() > 2:
        continue  # a grain or twin boundary runs through it
    u, _, vt = np.linalg.svd(Ut.mean(0))  # phantom orientations of one grain are generated unreduced, so a plain mean works
    um = u @ vt
    mean_err.append(anri.crystal.disorientation(U_new[i : i + 1, 0], um[None], ops)[0])
    spread_true.append(np.sqrt(np.mean(anri.crystal.disorientation(np.repeat(um[None], len(Ut), 0), Ut, ops) ** 2)))
mean_err = np.array(mean_err)
print(f"vs each voxel's mean truth ({len(mean_err)} single-grain voxels): median {np.median(mean_err):.3f} deg, within 0.1 "
      f"{np.mean(mean_err < 0.1) * 100:.0f}%, within 0.25 {np.mean(mean_err < 0.25) * 100:.0f}%, 90th {np.percentile(mean_err, 90):.2f}; "
      f"true spread inside a voxel median {np.median(spread_true):.2f} deg")
np.savez(os.path.join(root, f"local_{HALF:g}_{STEP_L:g}.npz"), f=f, cand=cand_o, U_loc=U_loc, U_new=U_new, frac=frac, occ=occ)
