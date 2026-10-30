# Simulating a dataset, and the forward model

`python -m anri.simulate` renders a map into a scanning-3DXRD dataset, as ImageD11 files that ImageD11 and
`python -m anri.index` read. It is the command line over `anri.fwd`. Option defaults are in
`python -m anri.simulate <stage> -h`. The manual's guides "How to simulate a scanning-3DXRD dataset" and "How to
compare measured frames with a map's forward model" run the same diagnostics with plots.

```
python -m anri.simulate prepare <map> <pars>                    # -> <map name>_prepare.h5
python -m anri.simulate render <..._prepare.h5> [--rows a:b]    # -> ..._render.h5 (or _render_<a>-<b>.h5)
python -m anri.simulate assemble <..._prepare.h5> <analysisroot> <sample> <dataset>
python -m anri.simulate run <map> <pars> <analysisroot> <sample> <dataset>
python -m anri.simulate show <any stage file> --log
```

The map is an ImageD11 TensorMap (every phase in it; each needs a space group) or the `_entries.npz` the indexer's
populations stage writes, so an indexed map can be rendered back and compared with its data. `<pars>` is an ImageD11
`.par` file or `pars.json`.

## Choices

| Stage | Option | Look at | Function (`anri.simulate.diagnostics`; plot in `anri.simulate.plots`) |
|---|---|---|---|
| prepare | `--optics`, `--divergence`, `--bandwidth`, `--beam-fwhm` | the beamline's optics: ask the human (see [physics](physics.md)) | |
| prepare | `--dty-step`, `--margin` | every voxel inside the radius the rows reach at every ω | `footprint`; `plot_footprint` |
| prepare | `--tth-max` | the share of each ring on the detector: a ring running off it loses spots an indexer reads as weak reflections | `ring_coverage`; `plot_rings` |
| prepare | `--dataset` | simulate a measured scan exactly: its frames' ω and dty, detector, y0 and geometry | |
| prepare | `--cif`, `--mu`, `--mu-map`, `--no-absorption` | absorption along the incoming and diffracted paths, as the indexer's fit models it: from each phase's CIF, a coefficient per phase, or a map (e.g. from absorption tomography). Ask the human for the sample's composition | |
| render | `--cut`, with prepare's `--density` | how peaks survive the segmentation cut at each density: below the density where peaks stop disappearing, small and thin features vanish | `probe_rows`; `plot_survival` |
| render | `--window`, `--max-frames` | the share of each peak its window misses; peaks broad in ω (near η = 0 and 180°, or with a spread) need more frames | `probe_rows`; `plot_capture` |
| render | `--rows a:b` | rows are independent: split them between jobs, then `assemble` gathers every part and refuses a missing or doubled row | |

Set `--cut` to the threshold the segmentation used on the data you mimic. To compare a map with measured frames,
prepare with `--dataset` and the y0 the map's fit used (`show` prints it), then compare `observed_row` (or
`raw_image`) with `model_row`, scaled by `row_scale`; `plot_frame_compare` overlays them.

## Phantoms

`anri.phantom.polycrystal` makes a 2D polycrystal of grains split into misoriented cells, with twin lamellae and,
optionally, cell orientations that accumulate like a random walk, a spread inside each cell, and bent grains;
`anri.phantom.tensormap` turns it into a TensorMap to render. Use a phantom with orientation spread along the rays
to test a method: a uniform orientation per voxel turns a gradient into an unphysical ladder of sub-peaks. Before
using a phantom to judge a method, check that its peaks render as single-maximum arcs (bananas). The repository's
`tests/data/phantoms` holds phantom TensorMaps ready to render (an AM-like 316L, a deformed 316L, quartz).

## The forward model from Python (`anri.fwd`)

- `anri.fwd.render.render_row`: every peak of one phase that reaches one dty row, as sparse pixels; `render_phases`
  sums several phases.
- `anri.fwd.render.make_row`: the row dict from one scan's per-frame ω and dty. Frames integrate over their ω (and
  dty) range: never treat a frame as one angle.
- `anri.fwd.render.guess_batch_size`: the largest `batch` that fits a share of free memory. Use it rather than
  guessing on a shared GPU.
- `anri.fwd.render.check_render`: checks rendered peaks against a Monte Carlo simulation of the beam's spreads.
  Run it after changing the beam or geometry.
- `anri.fwd.peak.get_centroid_both`: a reflection's two peak centroids (both η branches) in (sc, fc, ω), e.g. to
  check a grain against ImageD11's peaks.

Render a few rows first and time them: the first call compiles. Then size the whole job from that.
