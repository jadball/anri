"""Write the joint fit's result as a TensorMap on the indexer's grid, to look at next to the indexer's.

    python write_tmap.py <..._index_entries.npz> <match_..._joint.npz> [--truth <tmap.h5>] [--out <file.h5>]

Per voxel, of all its entries (the indexer's populations), the one with the largest occupancy after joint.py wins; its
chosen hypothesis gives the UBI. Maps: occupancy (the voxel's total), fraction (the winner's share of it), share (how
decisive the choice among the winner's hypotheses was), n_populations (entries above 10%), and with --truth
error_deg (refined) and error_index_deg (the indexer's main population), each against the truth voxel there.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("entries")
p.add_argument("joint")
p.add_argument("--truth")
p.add_argument("--out")
p.add_argument("--owner", choices=("indexer", "joint"), default="indexer", help="indexer: each voxel shows the "
               "indexer's main population (refined); joint: the population with the largest occupancy after joint.py")
args = p.parse_args()

from ImageD11.sinograms.tensor_map import TensorMap

import anri.crystal
import anri.io

ent = np.load(args.entries)
jt = np.load(args.joint)
idx = np.load(args.entries.replace("_entries.npz", ".npz"))
itm = TensorMap.from_h5(args.entries.replace("_entries.npz", "_tmap.h5"))
uc = itm.phases[0]
n_vox = idx["pos"].shape[0]
NR = int(round(np.sqrt(n_vox)))
vox = ent["voxel"]
tot_e = jt["x"].sum(1)  # each entry's occupancy
tot_v = np.bincount(vox, tot_e, minlength=n_vox)
win = np.full(n_vox, -1)
order = np.lexsort((tot_e, vox))  # by voxel, then occupancy: the last of each voxel wins
win[vox[order]] = order
changed = (win >= 0) & (ent["population"][np.maximum(win, 0)] != 0)  # the fit prefers another population
n_changed = int(changed.sum())
if args.owner == "indexer":
    win = np.full(n_vox, -1)
    win[vox[ent["population"] == 0]] = np.flatnonzero(ent["population"] == 0)
B = anri.crystal.B_matrix(np.asarray(uc.lattice_parameters, float))
ops = anri.crystal.laue_rotations(anri.crystal.symmetry_matrices(int(uc.symmetry)), B)


def rot(ubi: np.ndarray) -> np.ndarray:
    """Nearest rotation to UBI^-1 B^-1: UBIs stored in float32 are off orthogonal by ~1e-7, which near zero
    misorientation reads as ~0.02 deg."""
    u, _, vt = np.linalg.svd(np.linalg.inv(ubi.astype(np.float64)) @ np.linalg.inv(B))
    return u @ vt


def kam(ubi: np.ndarray, ok: np.ndarray, cut: float = 5.0) -> np.ndarray:
    """Mean misorientation (deg) to the 4 neighbouring voxels within `cut` (the same grain), on the NR x NR grid."""
    U = np.full((n_vox, 3, 3), np.nan)
    U[ok] = rot(ubi[ok])
    U = U.reshape(NR, NR, 3, 3)
    okg = ok.reshape(NR, NR)
    tot, cnt = np.zeros((NR, NR)), np.zeros((NR, NR))
    for a, b in (((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                 ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):  # fmt: skip
        both = okg[a] & okg[b]
        m = np.full(both.shape, np.inf)
        m[both] = anri.crystal.disorientation(U[a][both], U[b][both], ops)
        near = m < cut
        for sl in (a, b):
            tot[sl] += np.where(near, m, 0.0)
            cnt[sl] += near
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).ravel()


main = np.full((n_vox, 3, 3), np.nan)  # the indexer's main population per voxel
m0 = ent["population"] == 0
main[vox[m0]] = ent["ubi"][m0]
occ = idx["occupied"] & (win >= 0)
w = np.where(occ, win, 0)
ubi = np.where(occ[:, None, None], jt["ubi"][w], np.nan)
frac = np.where(occ, tot_e[w] / np.maximum(tot_v, 1e-30), np.nan)
n_pop = np.bincount(vox, tot_e / np.maximum(tot_v[vox], 1e-30) > 0.1, minlength=n_vox)
maps = {
    "UBI": ubi.reshape(NR, NR, 3, 3),
    "phase_ids": np.where(occ, 0, -1).reshape(NR, NR),
    "occupancy": tot_v.reshape(NR, NR),
    "fraction": frac.reshape(NR, NR),
    "share": np.where(occ, jt["share"][w], np.nan).reshape(NR, NR),
    "n_populations": np.where(occ, n_pop, 0).reshape(NR, NR),
    "owner_changed": np.where(occ, changed, 0).astype(np.int32).reshape(NR, NR),
}
# how far refinement moved each voxel from the indexer, and the local misorientation (KAM) of both maps
ok_m = occ & np.isfinite(main).all(axis=(1, 2))
moved = np.full(n_vox, np.nan)
moved[ok_m] = anri.crystal.disorientation(rot(ubi[ok_m]), rot(main[ok_m]), ops)
k_ref, k_idx = kam(ubi, occ), kam(main, ok_m)
maps["moved_deg"], maps["kam_deg"], maps["kam_index_deg"] = (x.reshape(NR, NR) for x in (moved, k_ref, k_idx))
print(f"  moved from the indexer: median {np.nanmedian(moved):.3f} deg, 90th {np.nanpercentile(moved, 90):.3f}; KAM "
      f"(4 neighbours < 5 deg) median: indexer {np.nanmedian(k_idx):.3f}, refined {np.nanmedian(k_ref):.3f} deg")  # fmt: skip
print(f"{occ.sum()} voxels, owner from the {args.owner}; the joint fit's largest population is not the indexer's main "
      f"one in {n_changed} voxels")  # fmt: skip
if args.truth:  # misorientation to the truth voxel at the same place
    from scipy.spatial import cKDTree

    te = anri.io.entries_from_tensormap(TensorMap.from_h5(args.truth))
    dist, it = cKDTree(te["pos"][:, :2]).query(idx["pos"][:, :2])
    has = occ & (dist < 0.5 * itm.steps[1])
    U_t = rot(te["ubi"][it])

    def mis(u: np.ndarray) -> np.ndarray:
        e = np.full(n_vox, np.nan)
        e[has] = anri.crystal.disorientation(rot(u[has]), U_t[has], ops)
        return e

    e_ref = mis(ubi)
    e_idx = mis(np.where(np.isfinite(main), main, np.eye(3)))
    maps["error_deg"] = e_ref.reshape(NR, NR)
    maps["error_index_deg"] = e_idx.reshape(NR, NR)
    for lab, e in (("indexer", e_idx[has]), ("refined", e_ref[has])):
        print(f"  {lab:>8}: median {np.median(e):.4f} deg; within 0.05 {np.mean(e < 0.05):.1%}, 0.25 "
              f"{np.mean(e < 0.25):.1%}, 1 {np.mean(e < 1):.1%}")  # fmt: skip
tmap = anri.io.tensormap_from_recon(maps, np.asarray(uc.lattice_parameters), int(uc.symmetry), uc.name, itm.steps[1])
try:
    tmap.get_ipf_maps()
except ImportError:
    print("orix is not installed: no IPF maps")
_ = tmap.euler
out = args.out or args.joint.replace(".npz", "_tmap.h5")
if os.path.exists(out):
    os.remove(out)
tmap.to_h5(out)
tmap.to_paraview(out)
print(f"-> {out} (and .xdmf for ParaView)")
