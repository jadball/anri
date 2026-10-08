# Deformed 316L-like phantom (def316l)

A 2D slice of a deformed FCC polycrystal, for testing refinement of sub-grain orientations. It is built like
`am316l` (grains, ~1.5 µm cells with sharp boundaries, twin lamellae), with orientation varying along every ray as in
deformed metals, so peaks smear into arcs (bananas; see `AGENTS.md`, "What a peak is"). Load it with
`ImageD11.sinograms.tensor_map.TensorMap.from_h5`.

## Contents

`def316l_tmap.h5` is an ImageD11 `TensorMap` with shape `(1, 203, 203)` and step `0.5` (microns).

- 31,417 voxels in a disk of radius 50 µm (`phase_ids == 0`); outside, `phase_ids == labels == -1` and `UBI` is NaN.
  Scanned on its own grid (0.5 µm steps and beam), the disk radius is 100 beam widths, as on real samples.
- 10 grains (`labels`), 4-55 µm across (most 19-55 µm, 40-110 voxels).
- 2,818 cells of about 1.5 µm (`cell`). Each is turned from its grain by its own small random rotation (0.1° standard
  deviation per rotation-vector component) and by a field that accumulates from cell to cell like a random walk (0.1°
  rms per component between cells 1.5 µm apart, growing as the square root of distance). Cells are constant inside.
- `sig_rot`: an intrinsic orientation spread inside every voxel, 0.05° (in radians in the map) per component, which
  the renderer applies to each voxel's peaks.
- Two bent grains: 2° per 50 µm about one random axis each (the second- and third-largest grains).
- Twin lamellae (`twin == 1`, 3,785 voxels) in the largest grain: 60° about the grain's <111>, 2 µm thick every 5 µm.
- `misorientation`: each voxel's misorientation (degrees) from its grain's mean orientation; median 0.42°, 90th
  percentile 0.84°, 99th 1.3° (non-twin voxels). Between voxels of one grain: 0.30° rms at 1.5 µm, 0.42° at 5 µm,
  0.61° at 15 µm.
- Peaks (`anri/sandbox/indexer/peak_shapes.py` on the centre row, 0.05° frames): smeared arcs about 0.5-1.5° across
  in eta and omega.
- Phase 0: `3.5966 3.5966 3.5966 90 90 90`, space group 225. `B` has no 2π factor.
- ImageD11's derived maps, computed from `UBI`: `B`, `U`, `UB`, `mt`, `unitcell`, `euler` and the IPF colours
  `ipf_x`, `ipf_y`, `ipf_z`. No strain maps: every `UBI` is a rotation of the nominal lattice.

## Provenance

```python
ph = anri.phantom.polycrystal(
    n=203,
    step=0.5,
    radius=50.0,
    n_grains=10,
    cell_size=1.5,
    cell_spread_deg=0.1,
    twin_grains=1,
    twin_period=5.0,
    twin_thickness=2.0,
    cell_walk_deg=0.1,
    cell_sig_deg=0.05,
    bend_grains=2,
    bend_deg=2.0,
    seed=0,
)
tmap = anri.phantom.tensormap(ph, [3.5966, 3.5966, 3.5966, 90.0, 90.0, 90.0], 225, "316L", 0.5)
for name in ("B", "U", "UB", "mt", "unitcell", "euler"):
    getattr(tmap, name)  # computed from UBI and kept in the maps
tmap.get_ipf_maps()  # needs orix
tmap.to_h5("def316l_tmap.h5")
```

Render it as an ImageD11 dataset with
`anri/sandbox/indexer/render_phantom.py <out> --tmap <this file> --ostep 0.05 --max-frames 61`. Without
`--max-frames`, each peak gets 3 frames, which cuts peaks broader than a frame in omega (about a third of the
intensity is lost).
