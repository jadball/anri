"""Compare anri.index results on a phantom with its truth, voxel by voxel, with attention to grain boundaries.

    python compare_truth.py <truth_tmap.h5> <run1_index.npz> [<run2_index.npz> ...] [--boundary 2]

Each reconstructed voxel covers several truth voxels (e.g. 4 of 0.5 um in a 1 um voxel). A voxel is pure if they all
lie within --boundary deg of each other, mixed otherwise (a real boundary through it). For pure voxels any error is
the method's: the share whose main population is more than --boundary deg from the truth, by distance from the voxel
to the nearest true boundary. For mixed voxels: the share whose main population matches one of the truths in it.
Also: boundary voxels and isolated flips of each map.
"""

import argparse

import numpy as np
from ImageD11.sinograms.tensor_map import TensorMap
from scipy.spatial import KDTree

import anri.crystal
import anri.io

p = argparse.ArgumentParser()
p.add_argument("truth", help="the phantom's TensorMap")
p.add_argument("runs", nargs="+", help="_index.npz of python -m anri.index")
p.add_argument("--boundary", type=float, default=2.0)
p.add_argument("--sg", type=int, default=225)
args = p.parse_args()

truth = TensorMap.from_h5(args.truth)
lp = np.asarray(truth.phases[0].lattice_parameters, float)
B = anri.crystal.B_matrix(lp)
ops = anri.crystal.laue_rotations(anri.crystal.symmetry_matrices(args.sg), B)
te = anri.io.entries_from_tensormap(truth)
t_pos, t_U = te["pos"][:, :2], np.linalg.inv(te["ubi"]) @ np.linalg.inv(B)
t_step = float(np.min(np.diff(np.unique(np.round(t_pos[:, 0], 6)))))
t_tree = KDTree(t_pos)

# true boundaries: truth voxels with a 4-neighbour more than --boundary deg away
pairs = t_tree.query_pairs(1.01 * t_step, output_type="ndarray")
mis = anri.crystal.disorientation(t_U[pairs[:, 0]], t_U[pairs[:, 1]], ops)
t_bnd = np.zeros(len(t_pos), bool)
t_bnd[pairs[mis > args.boundary].ravel()] = True
bnd_tree = KDTree(t_pos[t_bnd])
print(f"truth: {len(t_pos)} voxels of {t_step:g}; {t_bnd.mean() * 100:.1f}% on a boundary (> {args.boundary} deg)")


def flips(U: np.ndarray, ok: np.ndarray) -> tuple:
    """Boundary voxels and isolated flips of a square map of main orientations [NR, NR, 3, 3]."""
    nr = U.shape[0]
    edge, flip = np.zeros((nr, nr), bool), np.zeros((nr, nr), bool)
    for axis in (0, 1):
        a, b = (np.s_[:-1, :], np.s_[1:, :]) if axis == 0 else (np.s_[:, :-1], np.s_[:, 1:])
        both = ok[a] & ok[b]
        big = np.zeros(both.shape, bool)
        big[both] = anri.crystal.disorientation(U[a][both], U[b][both], ops) > args.boundary
        edge[a] |= big
        edge[b] |= big
        if axis == 0:
            flip[1:-1] |= big[:-1] & big[1:]
        else:
            flip[:, 1:-1] |= big[:, :-1] & big[:, 1:]
    return edge[ok].mean(), flip[ok].mean()


for path in args.runs:
    r = np.load(path)
    pos, U_main, occ = r["pos"][:, :2], r["U_pop"][:, 0], r["occupied"]
    vox = float(np.min(np.diff(np.unique(np.round(pos[:, 0], 6)))))
    inside = t_tree.query_ball_point(pos, 0.5 * vox, p=np.inf)  # truth voxels within each voxel's square
    has = occ & np.array([len(i) > 0 for i in inside])
    err, pure = np.full(len(pos), np.nan), np.zeros(len(pos), bool)
    for v in np.flatnonzero(has):
        tu = t_U[inside[v]]
        e = anri.crystal.disorientation(np.repeat(U_main[v][None], len(tu), 0), tu, ops)
        err[v] = e.min()
        pure[v] = anri.crystal.disorientation(np.repeat(tu[:1], len(tu), 0), tu, ops).max() <= args.boundary
    dist = bnd_tree.query(pos)[0]
    wrong = err > args.boundary
    pm = has & pure
    print(f"\n{path}: {has.sum()} occupied voxels over the truth, {pm.sum()} pure, {(has & ~pure).sum()} mixed")
    e = err[pm]
    print(f"  pure: main within 0.5 deg {np.mean(e < 0.5) * 100:.1f}%, 1 deg {np.mean(e < 1) * 100:.1f}%, median "
          f"{np.median(e):.2f} deg; wrong (> {args.boundary} deg) {np.mean(wrong[pm]) * 100:.2f}%")  # fmt: skip
    bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, np.inf)]
    print("  pure, wrong by distance to the true boundary (um): " + "; ".join(
        f"{lo:g}-{hi:g}: {wrong[pm & (dist >= lo) & (dist < hi)].sum()} of {(pm & (dist >= lo) & (dist < hi)).sum()}"
        for lo, hi in bins if (pm & (dist >= lo) & (dist < hi)).any()))  # fmt: skip
    mx = has & ~pure
    print(f"  mixed: main matches one of the truths in the voxel {np.mean(~wrong[mx]) * 100:.1f}%")
    nr = round(np.sqrt(len(pos)))
    Ug = np.where(occ[:, None, None], U_main, np.nan).reshape(nr, nr, 3, 3)
    e_frac, f_frac = flips(Ug, occ.reshape(nr, nr))
    print(f"  map: boundary voxels {e_frac * 100:.1f}%, isolated flips {f_frac * 100:.2f}%")
