"""Where are twins lost? Checks an anri.index result (the _index.npz) for a twin relation, stage by stage.

    python diagnose_twins.py <..._index.npz> --lattice a b c alpha beta gamma --sg N --angle DEG --axis u v w

The twin of an orientation U is U R, with R a rotation by --angle about the crystal direction [u v w] (direct
lattice, e.g. 86.3 deg about [1 0 0] for {10-12} extension twins in Mg; 60 deg about [1 1 1] for Sigma3 in FCC).
For each occupied voxel's main population it asks:

1. pruning: is the twin (any symmetric variant) within delta of an orientation in the kept list?
2. candidates: is the twin among the voxel's fitted candidates, and with what share of the voxel's occupancy?
3. populations: the misorientation between each voxel's first and second population (a peak at the twin angle
   means twin orientations were fitted).

Beware twin ghosts: a twin shares reflections with its parent (a third of them for Sigma3), so where the grid fits
the parent imperfectly, the twin orientation explains some of the shared spots and passes pruning, and may be
fitted, without being in the sample. On the am316l phantom (one twinned grain in 38), the twin of 98% of the grains
was in the kept list, and most second populations were 60 deg from the first. Real twins sit in lamellae; ghosts
spread over every grain.
With --map: the twin share per voxel. On that phantom, voxels with true twin have a median share of 0.36, but the 37
untwinned grains still 0.087 (above 0.1 in 45% of their voxels): ghosts make a haze over every grain, and real twins
must stand out above it as lamellae.
"""

import argparse

import numpy as np
from scipy.spatial import KDTree

import anri.crystal
import anri.phantom

p = argparse.ArgumentParser()
p.add_argument("npz")
p.add_argument("--lattice", type=float, nargs=6, required=True)
p.add_argument("--sg", type=int, required=True)
p.add_argument("--angle", type=float, required=True)
p.add_argument("--axis", type=float, nargs=3, required=True)
p.add_argument("--voxels", type=int, default=20000, help="voxels sampled (default 20000)")
p.add_argument("--map", help="also save a map of each voxel's twin share (occupancy on candidates at the twin angle from "
               "its main orientation) to this png: real twins form lamellae, ghosts spread over whole grains")
args = p.parse_args()

r = np.load(args.npz)
B = anri.crystal.B_matrix(args.lattice)
ops = anri.crystal.laue_rotations(anri.crystal.symmetry_matrices(args.sg), B)
A = np.linalg.inv(B).T  # direct lattice vectors as columns: a direction [u v w] is A @ [u v w]
axis = A @ np.array(args.axis)
R = anri.phantom.axis_angle(axis, args.angle)  # crystal frame
U_kept, f, cand = r["U"].astype(float), r["f"], r["cand"]
occ = np.flatnonzero(r["occupied"])
rng = np.random.default_rng(0)
vox = rng.choice(occ, min(args.voxels, len(occ)), replace=False)
U_main = r["U_pop"][vox, 0].astype(float)
delta = float(r["delta"])

# all symmetric variants of the twin of each main orientation: U S R, for every S in the Laue group
T = (U_main[:, None] @ ops[None] @ R[None, None]).reshape(-1, 3, 3)
q_kept = anri.crystal.mat_to_quat((U_kept[:, None] @ ops[None]).reshape(-1, 3, 3))
tree = KDTree(np.concatenate([q_kept, -q_kept]))
d, _ = tree.query(anri.crystal.mat_to_quat(T))
ang = np.degrees(4 * np.arcsin(np.minimum(d, 2.0) / 2)).reshape(len(vox), len(ops))
best = ang.min(1)
print(f"{len(vox)} occupied voxels sampled; delta (grid) {delta:.2f} deg; twin {args.angle} deg about {args.axis}")
print(f"1. kept list: the twin of the voxel's main orientation is within delta of a kept orientation for "
      f"{np.mean(best <= delta) * 100:.1f}% of voxels (within 2 delta: {np.mean(best <= 2 * delta) * 100:.1f}%)")

# 2. among the voxel's candidates? the share of the voxel's occupancy on candidates near a twin variant
Tv = T.reshape(len(vox), len(ops), 3, 3)
share = np.zeros(len(vox))
has = np.zeros(len(vox), bool)
for i, v in enumerate(vox):
    Uc = U_kept[cand[v]]
    dis = anri.crystal.disorientation(np.repeat(Uc, len(ops), 0), np.tile(Tv[i], (len(Uc), 1, 1)), ops).reshape(len(Uc), -1).min(1)
    near = dis <= 1.5 * delta
    has[i] = near.any()
    share[i] = f[v, near].sum() / max(f[v].sum(), 1e-30)
print(f"2. candidates: a twin orientation is among the voxel's candidates for {np.mean(has) * 100:.1f}% of voxels; "
      f"its share of the occupancy: median {np.median(share[has]) if has.any() else 0:.3f}; "
      f"> 0.01 in {np.mean(share > 0.01) * 100:.1f}%, > 0.03 in {np.mean(share > 0.03) * 100:.1f}%, "
      f"> 0.1 (reported) in {np.mean(share > 0.1) * 100:.1f}% of voxels")  # fmt: skip

# 3. populations 1 vs 2
two = r["present"][vox, 1]
mis = anri.crystal.disorientation(r["U_pop"][vox[two], 0].astype(float), r["U_pop"][vox[two], 1].astype(float), ops)
h, e = np.histogram(mis, bins=np.arange(0, 95, 5))
print(f"3. populations: {two.sum()} voxels with a second population; misorientation of 1 vs 2 (5-deg bins from 0): "
      f"{h.tolist()}; within 3 deg of the twin angle: {np.sum(np.abs(mis - args.angle) < 3)}")  # fmt: skip

if args.map:  # the twin share of every occupied voxel, by misorientation angle from its main orientation
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nv = len(r["occupied"])
    n = int(round(np.sqrt(nv)))
    share_all = np.full(nv, np.nan)
    U0 = r["U_pop"][:, 0].astype(float)
    for s0 in range(0, len(occ), 2000):
        vv = occ[s0 : s0 + 2000]
        Uc = U_kept[cand[vv]]  # [b, K, 3, 3]
        k = Uc.shape[1]
        mis = anri.crystal.disorientation(np.repeat(U0[vv], k, 0), Uc.reshape(-1, 3, 3), ops).reshape(len(vv), k)
        tw = np.abs(mis - args.angle) <= 1.5 * delta
        share_all[vv] = (f[vv] * tw).sum(1) / np.maximum(f[vv].sum(1), 1e-30)
    img = share_all.reshape(n, n)  # reconstruction order: first axis x, second -y
    img = np.flip(img, 1).T  # map order (as TensorMap.recon_order_to_map_order): rows y, columns x
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    im = ax.imshow(img, origin="lower", cmap="magma", vmin=0, vmax=0.3)
    fig.colorbar(im, ax=ax, label=f"share at {args.angle} deg from the main orientation")
    ax.set_title("twin share per voxel (grey: unoccupied)")
    ax.set_facecolor("0.7")
    fig.savefig(args.map, dpi=100)
    print(f"-> {args.map}: twin share > 0.03 in {np.nanmean(share_all > 0.03) * 100:.1f}% of occupied voxels")
