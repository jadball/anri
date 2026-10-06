"""Did an anri.index run keep the orientations another map needs? Distance from each voxel's orientation to the run's kept set.

    python kept_coverage.py <reference_tmap.h5> <..._index.npz> [--out <file.h5>]

For every voxel of the reference map (e.g. an older index that did better somewhere), the misorientation (over the
Laue group) to the nearest orientation the run kept after pruning ("U" in its npz). Large values mark orientations
that were pruned before the voxel fit: no later setting (--cand, --occupied) can bring them back. Writes the
reference map with a "kept_distance_deg" map added.
"""

import argparse
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("reference")
p.add_argument("index")
p.add_argument("--out")
args = p.parse_args()

from ImageD11.sinograms.tensor_map import TensorMap
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import anri.crystal

ref = TensorMap.from_h5(args.reference)
uc = ref.phases[0]
B = anri.crystal.B_matrix(np.asarray(uc.lattice_parameters, float))
ops = anri.crystal.laue_rotations(anri.crystal.symmetry_matrices(int(uc.symmetry)), B)
r = np.load(args.index)


def nearest_rotation(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    return u @ vt


# every symmetry-equivalent of every kept orientation, as unit quaternions (q and -q are the same rotation)
U_kept = nearest_rotation(r["U"].astype(np.float64))
q = Rotation.from_matrix(np.einsum("kij,sjl->ksil", U_kept, ops).reshape(-1, 3, 3)).as_quat()
tree = cKDTree(np.concatenate([q, -q]))

ubi = np.asarray(ref.maps["UBI"], np.float64).reshape(-1, 3, 3)
ok = np.isfinite(ubi).all(axis=(1, 2)) & (np.asarray(ref.maps["phase_ids"]).ravel() >= 0)
dist = np.full(ok.size, np.nan)
U_ref = nearest_rotation(np.linalg.inv(ubi[ok]) @ np.linalg.inv(B))
qd, _ = tree.query(Rotation.from_matrix(U_ref).as_quat())
dist[ok] = np.degrees(4 * np.arcsin(np.clip(qd / 2, 0, 1)))  # chord between unit quaternions -> rotation angle
d = dist[ok]
print(f"{ok.sum()} reference voxels; {r['U'].shape[0]} kept orientations in the run "
      f"(grid step {float(r['grid_step']):g}, worst case {float(r['delta']):.2f} deg)")  # fmt: skip
print("distance to the nearest kept orientation, deg: percentiles 50/90/99: "
      + " / ".join(f"{np.percentile(d, k):.2f}" for k in (50, 90, 99))
      + f"; beyond 2 x the worst case ({2 * float(r['delta']):.2f}): {np.mean(d > 2 * float(r['delta'])):.1%} of voxels")
ref.maps["kept_distance_deg"] = dist.reshape(np.asarray(ref.maps["phase_ids"]).shape)
out = args.out or os.path.splitext(args.reference)[0] + "_kept_distance.h5"
if os.path.exists(out):
    os.remove(out)
ref.to_h5(out)
ref.to_paraview(out)
print(f"-> {out} (and .xdmf): open kept_distance_deg over the deformed grain")
