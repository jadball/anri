# Deformed 316L-like phantom (def316l)

A 2D slice of a deformed FCC polycrystal, for testing refinement of sub-grain orientations: orientation varies along
every ray, so peaks smear into single-maximum arcs (bananas), as in deformed metals (see `AGENTS.md`, "What a peak
is"). Load it with `ImageD11.sinograms.tensor_map.TensorMap.from_h5`.

## Contents

`def316l_tmap.h5` is an ImageD11 `TensorMap` with shape `(1, 403, 403)` and step `0.25` (microns).

- 125,629 voxels in a disk of radius 50 µm (`phase_ids == 0`); outside, `phase_ids == labels == -1` and `UBI` is NaN.
- 10 grains (`labels`), 4-55 µm across (most 19-55 µm: 40-110 voxels of a 0.5 µm scan). Meant to be scanned with
  0.5 µm steps and beam: disk radius / beam = 100, as on real samples.
- Dislocation walls: straight walls in random directions, on average 0.25 µm (one voxel) apart along any line, each
  turning one side by a small random rotation (0.041° standard deviation per rotation-vector component). The
  misorientation between two points grows like a random walk with their distance: rms 0.10° at 0.5 µm, 0.17° at
  1.5 µm, 0.31° at 5 µm, 0.53° at 15 µm (bends included). Walls 1.5 µm apart with 0.1° each (the same walk on
  longer scales) gave lumpy peaks of separate sub-spots in this 2D slice; one-voxel walls give smooth arcs.
- Peaks (`anri/sandbox/indexer/peak_shapes.py` on the centre row, 0.5 µm scan, 0.05° frames): curved arcs (bananas)
  about 1-4° long in omega and 1-2° in eta.
- Two bent grains: 2° per 50 µm about one random axis each (the second- and third-largest grains).
- Twin lamellae (`twin == 1`) in the largest grain: 60° about the grain's <111>, 2 µm thick every 8 µm.
- `misorientation`: each voxel's misorientation (degrees) from its grain's mean orientation; median 0.40°, 99th
  percentile 1.1° (non-twin voxels).
- `cell` is unused here (no independent cell rotations).
- Phase 0: `3.5966 3.5966 3.5966 90 90 90`, space group 225. `B` has no 2π factor.

## Provenance

```python
ph = anri.phantom.polycrystal(
    n=403,
    step=0.25,
    radius=50.0,
    n_grains=10,
    cell_size=1.5,
    cell_spread_deg=0.0,
    twin_grains=1,
    twin_period=8.0,
    twin_thickness=2.0,
    wall_spacing=0.25,
    wall_spread_deg=0.0408,
    bend_grains=2,
    bend_deg=2.0,
    seed=0,
)
tmap = anri.phantom.tensormap(ph, [3.5966, 3.5966, 3.5966, 90.0, 90.0, 90.0], 225, "316L", 0.25)
tmap.to_h5("def316l_tmap.h5")
```

Render it as an ImageD11 dataset with
`anri/sandbox/indexer/render_phantom.py <out> --tmap <this file> --step 0.5 --ostep 0.05`.
