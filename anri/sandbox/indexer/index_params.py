"""Recover what can be recovered of an anri.index run's settings from its outputs.

    python index_params.py <..._index.npz> [<another ..._index.npz> ...]

The npz does not store the command line. From its arrays: the grid step and its worst case, the candidates per
voxel (--cand), the orientations kept after pruning (set by --min-lr / --keep), y0, and the --occupied cut: a voxel is
occupied when its total occupancy exceeds --occupied x the 99th percentile, so the cut lies between the largest
unoccupied total and the smallest occupied one. Also --min-frac from the smallest reported population.
"""

import sys

import numpy as np

for path in sys.argv[1:]:
    r = np.load(path)
    tot = r["f"].sum(1)
    p99 = np.percentile(tot, 99)
    occ = r["occupied"]
    lo = tot[~occ].max() / p99 if np.any(~occ) else 0.0
    hi = tot[occ].min() / p99 if np.any(occ) else np.inf
    frac, present = r["frac"], r["present"]
    print(path)
    print(f"  grid step {float(r['grid_step']):g} deg (worst case {float(r['delta']):.2f}); --cand {r['cand'].shape[1]}; "
          f"orientations kept {r['U'].shape[0]}; y0 {float(r['y0']) if 'y0' in r else 'not saved'}")  # fmt: skip
    print(f"  --occupied between {lo:.3f} and {hi:.3f} (x the 99th percentile of the voxel totals, {p99:.4g}); "
          f"{occ.sum()} of {occ.size} voxels occupied")  # fmt: skip
    print(f"  absolute cut in the TensorMap's occupancy map: {hi * p99:.4g}; at --occupied 0.1: {0.1 * p99:.4g}, "
          f"0.05: {0.05 * p99:.4g}")  # fmt: skip
    print(f"  --min-frac <= {frac[present].min():.3f} (smallest reported population); populations per occupied voxel: "
          + ", ".join(f"{k}: {np.mean(present[occ].sum(1) == k):.1%}" for k in range(1, present.shape[1] + 1)))
