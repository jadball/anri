"""Compare a stage-3 result with the stage-2 result it started from, in numbers (no images needed).

    python compare_stage3.py <dataset>_index.npz <dataset>_stage3.npz --sg 194 --lattice a b c alpha beta gamma

Prints, for stage 2 and stage 3:
- the main population's fraction per voxel (quantiles, and how many voxels are below 0.9 / 0.7 / 0.55);
- boundary voxels: those whose main orientation is more than --boundary deg from a 4-neighbour's. Fuzzier boundaries
  mean more of them, and more voxels that disagree with both neighbours along a line (isolated flips).
"""

import argparse

import numpy as np

import anri.crystal

p = argparse.ArgumentParser()
p.add_argument("stage2", help="_index.npz of python -m anri.index")
p.add_argument("stage3", help="_stage3.npz of run_stage3.py")
p.add_argument("--sg", type=int, required=True)
p.add_argument("--lattice", type=float, nargs=6, required=True)
p.add_argument("--boundary", type=float, default=2.0, help="misorientation (deg) that makes a boundary (default 2)")
args = p.parse_args()

B = anri.crystal.B_matrix(args.lattice)
ops = anri.crystal.laue_rotations(anri.crystal.symmetry_matrices(args.sg), B)
r2, r3 = np.load(args.stage2), np.load(args.stage3)
present, frac2, U_pop = r2["present"], r2["frac"], r2["U_pop"]
nv = len(present)
NR = round(np.sqrt(nv))

# stage 2: populations are sorted by fraction, so population 0 is the main one
main2_u = np.where(present[:, :1, None, None], U_pop[:, :1], np.nan)[:, 0]
main2_f = np.where(present[:, 0], frac2[:, 0], np.nan)

# stage 3: the unit with the largest fraction in each voxel
v, f3, U3 = r3["voxel"], r3["frac"], r3["U"]
order = np.lexsort((-f3, v))
first = np.unique(v[order], return_index=True)[1]
main3_f = np.full(nv, np.nan)
main3_u = np.full((nv, 3, 3), np.nan)
main3_f[v[order][first]] = f3[order][first]
main3_u[v[order][first]] = U3[order][first]
n_units3 = np.bincount(v, minlength=nv)


def fraction_stats(name: str, f: np.ndarray) -> None:
    """Print quantiles of the main population's fraction."""
    f = f[np.isfinite(f)]
    q = np.percentile(f, [1, 5, 25, 50])
    print(f"{name}: {len(f)} voxels; main fraction 1/5/25/50th percentile {q[0]:.3f} {q[1]:.3f} {q[2]:.3f} {q[3]:.3f}; "
          f"below 0.9 {np.mean(f < 0.9) * 100:.1f}%, 0.7 {np.mean(f < 0.7) * 100:.1f}%, 0.55 {np.mean(f < 0.55) * 100:.1f}%")  # fmt: skip


def boundaries(name: str, U: np.ndarray) -> np.ndarray:
    """Print how many voxels sit on a boundary, and return which."""
    U = U.reshape(NR, NR, 3, 3)
    ok = np.isfinite(U).all(axis=(-1, -2))
    mis = {}
    for axis in (0, 1):
        a, b = [np.s_[:-1, :], np.s_[1:, :]] if axis == 0 else [np.s_[:, :-1], np.s_[:, 1:]]
        both = ok[a] & ok[b]
        m = np.full(both.shape, np.nan)
        m[both] = anri.crystal.disorientation(U[a][both], U[b][both], ops)
        mis[axis] = m
    edge = np.zeros((NR, NR), bool)  # voxel has a boundary to at least one 4-neighbour
    flip = np.zeros((NR, NR), bool)  # voxel differs from both neighbours along a line: an isolated flip
    for axis, m in mis.items():
        big = m > args.boundary
        if axis == 0:
            edge[:-1] |= big
            edge[1:] |= big
            flip[1:-1] |= big[:-1] & big[1:]
        else:
            edge[:, :-1] |= big
            edge[:, 1:] |= big
            flip[:, 1:-1] |= big[:, :-1] & big[:, 1:]
    n = ok.sum()
    print(f"{name}: boundary voxels (> {args.boundary} deg to a neighbour) {edge[ok].sum()} ({edge[ok].mean() * 100:.1f}%); "
          f"isolated flips {flip[ok].sum()} ({flip[ok].sum() / n * 100:.2f}%)")  # fmt: skip
    return edge


fraction_stats("stage 2", main2_f)
fraction_stats("stage 3", main3_f)
print(f"stage 3 units per occupied voxel: {np.bincount(n_units3[n_units3 > 0])[1:]} (1, 2, 3, ... units)")
e2 = boundaries("stage 2", main2_u)
e3 = boundaries("stage 3", main3_u)
print(f"boundary voxels in stage 3 that were not in stage 2: {(e3 & ~e2).sum()}; in stage 2 only: {(e2 & ~e3).sum()}")
both = np.isfinite(main2_u).all(axis=(-1, -2)) & np.isfinite(main3_u).all(axis=(-1, -2))
d = anri.crystal.disorientation(main2_u[both], main3_u[both], ops)
switched = d > args.boundary
print(f"voxels whose main orientation changed by more than {args.boundary} deg (a different population became main): "
      f"{switched.sum()} ({switched.mean() * 100:.1f}%); of those on a stage-3 boundary "
      f"{np.mean(e3.ravel()[both][switched]) * 100:.0f}%")  # fmt: skip
