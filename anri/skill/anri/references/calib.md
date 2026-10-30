# Calibrating the geometry

`python -m anri.calib` calibrates the detector and beam geometry in two steps, as ImageD11's notebooks do: powder
rings from the motors' first guess, close enough to index; then the indexed scan, which separates the wavelength from
the distance and models each peak's parallax. Option defaults are in `python -m anri.calib <stage> -h`. The manual's
guide "How to calibrate the geometry" runs the whole chain with a plot for each choice.

```
python -m anri.calib rings <..._dataset.h5 | image> [--parfile pars.json]   # -> <sample>_<dataset>_rings.h5
python -m anri.calib pars <..._rings.h5> <folder>                          # -> <folder>/pars.json, to index with
python -m anri.index histogram <analysisroot> <sample> <dataset> --parfile <folder>/pars.json
python -m anri.index search <..._histogram.h5>
python -m anri.index filter <..._search.h5>
python -m anri.index fit <..._filter.h5>                                   # no local search
python -m anri.calib fit <..._fit.h5>                                      # -> <...>_calib.h5
python -m anri.calib pars <..._calib.h5> <folder>                          # the calibrated pars.json
python -m anri.calib show <any stage file> --log
```

## Before you run it

1. **Ask the human** whether the sample is a calibrant (a certified lattice, given with `--lattice`) and what the
   energy was. A calibrant frees the wavelength (the `--free` default); any other sample frees `lattice`, its scale,
   instead: the two cannot both be free. Ask also for the detector's distortion files and mask, if the DataSet does
   not name them.
2. **Know the scope.** The scan fit is for calibrants and little-deformed samples. Where a voxel holds orientations
   closer together than the geometry's error (cells, subgrains), matching absorbs part of that error and the rounds
   converge slowly; the stage warns. A deformed sample's geometry is refined with its strains, not here.
3. **Wedge, chi and y0 stay fixed.** y0 is the indexing fit's; choose it there ([index](index.md)).
4. **Index with the fit stage only.** The calibration refuses a local search's file: its orientations, a fraction of
   a degree apart, let each round's matching absorb part of the geometry's error.

## Choices

| Stage | Option | Look at | Function (`anri.calib.diagnostics`; plot in `anri.calib.plots`) |
|---|---|---|---|
| rings | `--max-shift`, `--no-search` | the centre search: one bright spot that the finer grids close in on; a centre further than `--max-shift` from the start is not searched | `read_rings`; `plot_centre_search` |
| rings | `--span` | the distance search's score: one clear peak. Two of similar height mean the rings could be matched one ring off: narrow `--span` around the right one | `read_rings`; `plot_distance_search` |
| rings | `--tols`, `--free` | each (ring, η sector)'s offset from its ring before and after: every ring flat at zero after the fit. A sine in η, the same for every ring, is the centre; one growing with 2θ, a tilt; a constant growing with 2θ, the distance. Keep `--tols` a few pixels wide (one Eiger pixel at 150 mm is 0.03°); rings closer than twice the widest to a neighbour are left out. tilt_x moves no ring, and the wavelength cannot be told from the distance by rings: leave both for `fit` | `read_rings`; `plot_ring_offsets` |
| rings | `--scans`, `--threshold` | a scan's spots fill its rings well enough summed over every frame; fewer scans for speed | |
| rings | `--merge` | reflections closer in d* than this are one ring; rings the data cannot separate are merged, not split | |
| fit | `--min-occ` | the histogram of the fit file's occupancies over the largest: the occupied orientations stand apart from a floor of near-zero candidates. Put the cut in the gap; each voxel should keep nearly all its occupancy | `read_occupancy`, `occupancy_kept`; `plot_min_occ` |
| fit | `--tols`, `--lattice` | each peak's distance of (h, k, l) from integers at the starting geometry: real matches pile up near 0 above a floor of chance matches. The first tolerance takes in the whole pile and little of the floor. No pile: the geometry is too far off, the lattice is wrong, or the peaks are not this scan's | `match_distances` on `anri.calib.stages.load_inputs`; `plot_match` |
| fit | `--extra-rounds` | the log's line per round of how far each parameter moved, in its error bars. A `WARNING` that the last round still moved one by more than 1 sigma: the errors quoted are too small | |
| fit | `--row-step` | on a CPU, every Nth dty row (e.g. 10): matching costs in proportion to the peaks | |
| fit | `--min-peaks`, `--clip`, `--iter` | the strain left and the residuals (below) | `read_calib`, `residual_trends`, `strain_left` |

## Before you believe a calibration

5. **Read its log** (`python -m anri.calib show <file> --log`). From round to round the matched peaks should grow or
   stay, and the standard deviations in sc, fc and ω fall to the peaks' own precision. `last_round_moved_sigma` in
   the results must be below 1.
6. **Residuals** (`anri.calib.diagnostics.residual_trends` on `anri.calib.diagnostics.read_calib`; `plot_residuals`,
   `plot_residual_hists`). Each matched peak's observed minus predicted (sc, fc, ω), in standard deviations, against
   ω, the azimuth about the fitted beam centre and dty: with the right geometry the medians are flat at zero.
   - a sinusoid against azimuth: the centre or the tilts; with them free, a higher-order pattern is detector
     distortion (check the distortion files);
   - a trend against ω: something fixed here that turns with the sample (wedge, chi, a wobble of the axis);
   - a trend against dty: y0, or large grains the map places wrongly;
   - heavy tails: wrong matches or overlapping peaks (a lower `--clip`, tighter `--tols`).

   The distances from integers again at the calibrated geometry (`anri.calib.diagnostics.match_distances` with the
   calibration's "ubi" and "pars") should pile up much tighter than at the start.
7. **Strain left** (`anri.calib.diagnostics.strain_left`; `plot_strain_left`; `strain_left_hydrostatic`,
   `strain_left_deviatoric_max` in the results): each orientation with at least `--min-peaks` peaks, its lattice
   freed with the geometry fixed. On a strain-free calibrant a right geometry leaves only noise: a deviatoric strain
   shared by every orientation points to the tilts or the centre, a hydrostatic one to the wavelength or distance
   against the reference lattice (absorbed by the scale when `lattice` is free). `chi2 x` in the log near 1 means the
   geometry explains the peaks as well as free lattices do. On an undeformed Si cube the calibration left 0.2e-5
   hydrostatic and at most 2e-5 deviatoric. On a sample that is not strain-free, the strain left is the sample's plus
   the calibration's error.
8. **Check it on a phantom** when the claim matters: the guide renders one with a first guess far from its true
   geometry ([phantom](phantom.md)).

## Report

The calib file, the commands that remake it from the dataset (`anri.calib.stages.commands`: rings, pars, the
indexer's run and fit; `pars.json` names the file it was written from, so the indexer's histogram stage links the
rings into the chain), the `pars.json` written from it (`anri.calib.stages.write_pars`, as `python -m anri.calib
pars`), the free parameters with their errors, the
residual and strain-left checks above, and any `WARNING` in the log. Then index again with the new `pars.json`
(`--parfile`): the map was made with the old geometry.
