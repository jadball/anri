"""Compare Anri orientation maps with an ImageD11 map, voxel by voxel.

    python compare_imaged11.py <imaged11_tmap.h5> <anri_tmap.h5> [more anri tmaps ...] [--phase-id 0] [--png out.png]

The maps may be padded differently (e.g. ImageD11's reconstruction padded by 50): voxels are matched by their sample
positions, both grids being centred on the rotation axis, to within half the coarser step. Orientations are the
nearest rotations to UBI^-1 B^-1, with B and the Laue group from the reference map's phase, so strain in a refined
map does not count as misorientation. For each Anri map: the misorientation to the reference over the voxels both
maps have (median, 90th percentile, share within 0.1 / 0.25 / 0.5 / 1 deg), and a map of it in --png. Then, for the
voxels that disagree by more than --boundary deg: how far each is from the reference's nearest grain boundary. Mostly 0-1
voxels: the boundaries are drawn in slightly different places; further: wrong inside grains.
"""

import argparse

import numpy as np
from ImageD11.sinograms.tensor_map import TensorMap
from scipy import ndimage
from scipy.spatial import KDTree

import anri.crystal
from anri.geom import recon_to_step, step_to_sample

p = argparse.ArgumentParser()
p.add_argument("reference", help="ImageD11 TensorMap (e.g. pbp or refined), the reference")
p.add_argument("anri", nargs="+", help="Anri TensorMaps (python -m anri.index)")
p.add_argument("--phase-id", type=int, default=0, help="phase in the reference map (default 0)")
p.add_argument("--png", default="compare_imaged11.png")
p.add_argument("--boundary", type=float, default=2.0, help="misorientation (deg) that makes a boundary (default 2)")
p.add_argument("--sg", type=int, help="space-group number, if the reference phase has a centring letter instead")
args = p.parse_args()


def voxels(path: str, phase_id: int) -> tuple:
    """Sample positions [N, 2], UBIs [N, 3, 3] and step of a 2D TensorMap's voxels of one phase."""
    t = TensorMap.from_h5(path)
    ubi = np.asarray(TensorMap.map_order_to_recon_order(t.UBI, 0), float)
    ph = np.asarray(TensorMap.map_order_to_recon_order(t.phase_ids, 0))
    ri, rj = np.nonzero((ph == phase_id) & np.isfinite(ubi).all(axis=(-1, -2)))
    step = float(t.steps[1])
    si, sj = recon_to_step(ri, rj, ph.shape)
    sx, sy = step_to_sample(si, sj, step)
    return np.stack([np.asarray(sx), np.asarray(sy)], 1), ubi[ri, rj], step, t, ph.shape, (ri, rj)


def rotations(ubi: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Nearest rotations to UBI^-1 B^-1 (the orientation, without strain)."""
    u, _, vt = np.linalg.svd(np.linalg.inv(ubi) @ np.linalg.inv(B))
    return u @ vt


ref_pos, ref_ubi, ref_step, ref_t, ref_shape, (rri, rrj) = voxels(args.reference, args.phase_id)
uc = ref_t.phases[args.phase_id]
lpars = np.asarray(uc.lattice_parameters, float)
sg = args.sg if args.sg is not None else uc.symmetry
if not isinstance(sg, (int, np.integer)) and not str(sg).isdigit():
    raise SystemExit(f"the reference phase's symmetry is {sg!r}: give the space-group number with --sg")
B = anri.crystal.B_matrix(lpars)
ops = anri.crystal.laue_rotations(anri.crystal.symmetry_matrices(int(sg)), B)
ref_U = rotations(ref_ubi, B)
tree = KDTree(ref_pos)

# the reference's grain boundaries (a 4-neighbour more than --boundary deg away), and each voxel's distance to one
Ug = np.full(ref_shape + (3, 3), np.nan)
Ug[rri, rrj] = ref_U
bnd = np.zeros(ref_shape, bool)
for a, b in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
    both = np.isfinite(Ug[a]).all(axis=(-1, -2)) & np.isfinite(Ug[b]).all(axis=(-1, -2))
    big = np.zeros(both.shape, bool)
    big[both] = anri.crystal.disorientation(Ug[a][both], Ug[b][both], ops) > args.boundary
    bnd[a] |= big
    bnd[b] |= big
ref_dist = ndimage.distance_transform_edt(~bnd)[rri, rrj]  # in reference voxels
print(f"reference {args.reference}: {len(ref_pos)} voxels of phase {args.phase_id} ({uc.lattice_parameters}, "
      f"space group {sg}), step {ref_step:g}; {len(ops)} Laue-group rotations")  # fmt: skip

results = []
for path in args.anri:
    pos, ubi, step, _, shape, (ri, rj) = voxels(path, 0)
    d, i = tree.query(pos)
    ok = d <= 0.5 * max(step, ref_step) + 1e-9
    mis = anri.crystal.disorientation(rotations(ubi[ok], B), ref_U[i[ok]], ops)
    img = np.full(shape, np.nan)  # the map's own grid, reconstruction order
    img[ri[ok], rj[ok]] = mis
    results.append((path, img, mis))
    print(f"{path}: {ok.sum()} of {len(pos)} voxels matched; misorientation to the reference: median {np.median(mis):.3f} "
          f"deg, 90th {np.percentile(mis, 90):.3f}; within 0.1 {np.mean(mis < 0.1) * 100:.1f}%, 0.25 "
          f"{np.mean(mis < 0.25) * 100:.1f}%, 0.5 {np.mean(mis < 0.5) * 100:.1f}%, 1 {np.mean(mis < 1) * 100:.1f}%")  # fmt: skip
    dist = ref_dist[i[ok]]
    wrong = mis > args.boundary
    bins = [(0, 0.5, "on a boundary"), (0.5, 1.5, "1"), (1.5, 2.5, "2"), (2.5, 5.5, "3-5"), (5.5, np.inf, "> 5")]
    print(f"  {wrong.sum()} voxels ({wrong.mean() * 100:.1f}%) more than {args.boundary} deg off; by distance to the "
          "reference's nearest boundary (reference voxels): " + "; ".join(
              f"{name}: {wrong[(dist >= lo) & (dist < hi)].sum()} of {((dist >= lo) & (dist < hi)).sum()} "
              f"({np.mean(wrong[(dist >= lo) & (dist < hi)]) * 100:.1f}%)" for lo, hi, name in bins if ((dist >= lo) & (dist < hi)).any()))  # fmt: skip

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

n = len(results)
fig, ax = plt.subplots(1, n, figsize=(min(5.5 * n, 19), 5), layout="constrained", squeeze=False)
for a, (path, img, mis) in zip(ax[0], results):
    # one pixel per voxel, in map order as TensorMap.plot shows it (origin lower)
    sc = a.imshow(TensorMap.recon_order_to_map_order(img)[0], origin="lower", cmap="viridis", vmin=0, vmax=1,
                  interpolation="nearest")  # fmt: skip
    a.set_facecolor("0.85")
    a.set_title(f"{path.split('/')[-1][:40]}\nmedian {np.median(mis):.2f} deg", fontsize=9)
    a.set_xticks([]), a.set_yticks([])
fig.colorbar(sc, ax=ax[0, -1], label="misorientation to ImageD11 (deg)", shrink=0.8)
fig.savefig(args.png, dpi=100)
print(f"-> {args.png}")
