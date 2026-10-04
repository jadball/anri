# AM-like 316L phantom (am316l)

A 2D slice of a 316L-like stainless steel polycrystal with the features of additively manufactured and annealed
metals: grains, misoriented cells inside them, and annealing twins. Load it with
`ImageD11.sinograms.tensor_map.TensorMap.from_h5`.

## Contents

`am316l_tmap.h5` is an ImageD11 `TensorMap` with shape `(1, 103, 103)` and step `0.5` (microns).

- 7845 voxels in a disk of radius 25 µm (`phase_ids == 0`); outside, `phase_ids == labels == -1` and `UBI` is NaN.
- 38 grains (`labels`), 721 cells of about 1.5 µm (`cell`), each turned from its grain by a small random rotation
  (0.3 degrees standard deviation per rotation-vector component).
- Twin lamellae (`twin == 1`, 253 voxels) in the largest grain: turned 60 degrees about the grain's <111>, 2 µm thick
  every 5 µm, with the twin plane normal to that <111>.
- `misorientation`: each voxel's misorientation (degrees) from its grain's mean orientation.
- Phase 0: `3.5966 3.5966 3.5966 90 90 90`, space group 225. `B` has no 2π factor.

## Provenance

Made with `anri.phantom.polycrystal(n=103, step=0.5, radius=25.0, n_grains=40, cell_size=1.5, cell_spread_deg=0.3,
twin_grains=1, seed=0)` in `docs/source/tutorials/phantom.ipynb`. Two of the 40 Voronoi seeds give no voxels in the
disk. `docs/source/tutorials/indexing.ipynb` simulates a scan of it and indexes it.
