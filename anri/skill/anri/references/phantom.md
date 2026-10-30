# Making a phantom to test a method

A phantom is a map whose answer you know: render it with [simulate](simulate.md), index or refine the rendered
dataset, and compare with the truth. It is the only way to back a claim about a method (rule 17). A phantom that is
easier than the data proves nothing, so build one with the microstructure the method will meet. The tutorial "A
phantom microstructure" in the manual makes one step by step.

```python
import anri.utils

anri.utils.setup()
import anri.phantom

ph = anri.phantom.polycrystal(
    n=203,
    step=0.5,
    radius=50.0,
    n_grains=10,
    cell_size=1.5,
    cell_spread_deg=0.1,
    twin_grains=1,
    cell_walk_deg=0.1,
    cell_sig_deg=0.05,
    bend_grains=2,
    bend_deg=2.0,
    seed=0,
)
tmap = anri.phantom.tensormap(ph, [3.5966, 3.5966, 3.5966, 90, 90, 90], 225, "316L", 0.5)  # same step
tmap.to_h5("my_phantom_tmap.h5")
```

`anri.phantom.polycrystal` returns the truth maps ("U", "grain", "cell", "twin", "inside", and "sig_rot" with an
intrinsic spread); `anri.phantom.tensormap` turns them into an ImageD11 TensorMap that `python -m anri.simulate run`
renders, with each voxel's "misorientation" from its grain's mean. Read the parameters from the docstrings, not from
memory.

## Before you make one

1. **Ask the human which microstructure the method is for**, and the material (lattice, space group). Never pick
   one for them: a method tested on clean grains says nothing about deformed cells.
2. **Ask which scan step and beam the phantom stands in for.** Make the phantom's `step` finer than the scan's
   `--dty-step` (prepare's default is the map's own step), so that one scan voxel holds several cells, or a parent
   and its twin, as real voxels do.
3. **Estimate the cost.** The renderer works per phantom voxel and reflection: a finer `step` or a larger `radius`
   multiplies the rendering time. Render a few rows first and time them ([simulate](simulate.md)).

## Choosing the options

What the microstructures in [physics](physics.md) look like in `anri.phantom.polycrystal`'s options. The numbers
are the ranges real materials span; agree the values with the human.

| Microstructure | Options |
|---|---|
| annealed grains, single crystals | `cell_spread_deg=0`, `twin_grains=0`: constant grains with sharp boundaries |
| annealing twins (Σ3 in FCC) | `twin_grains`, `twin_axis=(1, 1, 1)`, `twin_angle_deg=60`; `twin_thickness` and `twin_period` from ~0.1 µm to tens of µm |
| deformation and nano-twins | `twin_thickness` below the scan step but above the phantom's `step`: a scan voxel holds two orientations with a volume fraction |
| deformed FCC/BCC (dislocation cells) | `cell_size` 0.5-2 µm, `cell_spread_deg` for each cell's own turn, `cell_walk_deg` for misorientation that accumulates like a random walk, `cell_sig_deg` for the spread inside a cell |
| additively manufactured (solidification cells) | `cell_size` 0.3-1 µm, `cell_spread_deg` ≲ 0.5; drift along a column only as `bend_grains` (columns and melt pools are not modelled) |
| lattice curvature with one dominant axis, elastic bending | `bend_grains`, `bend_deg` (degrees per `radius` of distance) |

The repository's `tests/data/phantoms` holds three made this way, each with a README giving the call that made it:
`am316l` (cells and twins), `def316l` (accumulating cells, intrinsic spread, bent grains, twins) and a quartz one.
Start from these when they fit.

**What the generator cannot make.** Say so, and ask before building any of these by hand:

- a third dimension: it is one 2D slice;
- several phases: `anri.phantom.tensormap` writes one phase (the renderer takes several, from any TensorMap);
- strain: every UBI is a rotation of the nominal lattice, so peaks do not move in 2θ;
- density changes: every voxel has the same density;
- martensite, bainite or α-lath variants related by an orientation relationship;
- an anisotropic spread inside a voxel: `cell_sig_deg` is the same for each rotation component (bent grains and the
  cells' rotations give anisotropy between voxels).

## Before you use one

4. **Check the structure is what you asked for.** Count grains, cells and twin voxels from the truth maps, and give
   the median and 90th percentile of the TensorMap's "misorientation" (non-twin voxels), as the phantoms' READMEs
   do. A `cell_walk_deg` grows as the square root of distance, so a large grain can drift further than you meant.
5. **Check that its peaks are bananas.** Render a few rows (`anri.simulate.diagnostics.model_row` on a prepare
   file, summed with `anri.simulate.diagnostics.frames_image`; or `anri.simulate.diagnostics.row_image` on the
   rendered dataset, drawn by `plot_row_image`) and look at peaks near and away from η = 0 and 180°. With orientation
   spread along the rays they must be smeared arcs with a single maximum, widest in ω near η = 0 and 180° (σ / |sin
   η|). A ladder of separate sub-peaks means each voxel's orientation is uniform on a grid too coarse for its
   gradient: make `step` finer, or add `cell_sig_deg`. Separate sharp spots are right only where a few cells or
   grains fill the ray.
6. **Make it as hard as the data.** Render with the data's optics and counts (prepare's `--optics` and
   `--density`, render's `--cut`), and with absorption (`--cif` or `--mu`), so that the faint ends of the bananas
   are lost to the cut as in real data.

## Recording it

The phantom is not a stage: no file records how it was made. Write the exact calls (every option, the seed, the
anri version) in a README beside the TensorMap, as `tests/data/phantoms` does. The same seed and options give the
same phantom. Accumulating cells, intrinsic spread and bent grains draw from their own random stream, so turning
them on leaves the grains, cells and twins unchanged.

## Judging a method on it

- Compare voxel by voxel with the truth maps: grain ("labels"), "cell", "twin", "misorientation".
- **Path averaging** is the failure to look for: each voxel's orientation shrunk towards its grain's mean, so the
  method's misorientations from the grain mean come out smaller than the truth's. Compare their distributions, not
  only the mean error.
- Twin lamellae a voxel or two thick, and voxels at grain boundaries, are where errors gather: report them apart
  from grain interiors.
- Populations closer than the instrument resolves cannot be separated, in a phantom too: judge them as one
  population with a spread.
