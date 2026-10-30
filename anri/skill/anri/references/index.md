# The indexer, stage by stage

Each section names a stage's options, the numbers that choose them, and what the stage writes. Option defaults are
in `python -m anri.index <stage> -h`; this file does not repeat them. The guide "How to choose the indexing
parameters" in the manual runs the same diagnostics with plots, and explains each at length.

The diagnostics (`anri.index.diagnostics`) take arrays from a histogram file:

```python
import anri.index as ix

d = ix.stages.load_histogram("<sample>_<dataset>_histogram.h5")
# d["H"] the fit's data [ring, eta, omega, row] (flat), d["H_lit"] the lit histogram (rows summed),
# d["n_rings"], d["scan"], d["rings"], d["geom"], d["prov"] (its provenance)
```

The plots (`anri.index.plots`) draw them: `update = plot_x(fig, diag)` draws once, and `update(**values)` redraws
for other option values. Save the figure with `fig.savefig` and look at it.

## 1. histogram: `--rings`, `--tth-tol`, `--cif`, `--monitor`, `--gridstep`, `--saturation`, `--detector-mask`

The one pass over the whole dataset: every pixel binned into the lit histogram and the fit's histogram. Re-run it
after changing the rings.

| Choice | Look at | Function |
|---|---|---|
| which rings, how wide | the 2θ profile and the (2θ, η) cake: rings that touch or overlap, intensity between rings (another phase), a ring that wobbles with η (the geometry is off: fix the calibration, do not widen `--tth-tol`) | `ring_profile_cake`, `ring_windows`; `plot_rings` |
| `--rings` as a list, `--ring-merge` | the table `plot_rings` returns lists every ring shown, used or not, with multiplicity, `\|F\|^2` and gap: for low symmetry pick strong rings well apart (`--rings 9,10,11`); merge rings the data cannot separate | `ring_windows(cake, [9, 10, 11])`; `plot_rings` |
| `--detector-mask` | the measured share of each ring per η bin: dips are module gaps and the mask, 0 is off the detector | `rings/coverage` in the histogram file; `plot_coverage` |
| `--monitor` | intensity per dty row: steps from one row to the next are flux changes between the rows' scans | `row_profile`; `plot_rows` |
| `--gridstep` | the number of voxels the fit can afford: rows summed in groups make voxels larger and the fit much cheaper | the log's `memory:` line |
| `--saturation` | the detector's saturation count; saturated bins become lower bounds in the fit | ask the human |
| `--cif` | structure factors for intensities, and the attenuation the fit's absorption uses | ask the human for the phase |

Six to eight rings are usually enough; more help small grains and lower symmetry, and each ring costs time in every
later stage. `--grid-pad` widens the voxel grid for a sample reaching past the scanned range. The grid is fixed here
for every later stage.

Writes `H`, `H_lit`, `H_saturated` (if any), and the groups `rings`, `scan`, `geometry`, `grid`.

## 2. search: `--lit`, `--etacut`, `--grid`, `--max-chance`

Scores every orientation of a grid over the fundamental zone by its completeness: the share of its predicted spots
that land on lit bins.

| Choice | Look at | Function |
|---|---|---|
| `--lit` | lit area against the intensity it holds as the cut rises: put the cut near the elbow, where the area falls fast and the intensity slowly | `lit_curve`; `plot_lit` |
| `--etacut` | intensity per η: reflections near η = 0 and 180° smear in ω by 1/\|sin η\| and are left out | `eta_profile`; `plot_etacut` |
| `--grid`, `--max-chance` | chance completeness (the median over the grid, where most orientations are wrong) per grid step; then the deviance per step, flat where the data stop telling finer grids apart | `grid_chances`, `grid_deviances`; `plot_grid`, `plot_grid_deviances` |

The search's log reports the grid step chosen, its worst case from the truth, and the chance completeness. Writes
`U_grid` and `completeness`; `results` holds `chance_completeness` and `completeness_halfway`.

## 3. filter: `--prune`, `--min-comp`, `--min-lr`, `--keep`, `--censor`

Chooses the orientations for the fit. `--prune likelihood` fits one occupancy per orientation above `--min-comp` to
the row-summed data and keeps those whose likelihood ratio is above `--min-lr` (25 is about 5σ).

| Choice | Look at | Function |
|---|---|---|
| `--min-comp` | the completeness of every grid orientation (the search file's): real ones well above the chance level | `plot_completeness` |
| `--min-lr`, `--keep` | likelihood ratio against the misorientation to the nearest stronger orientation: a grain is one strong point far from anything stronger, with a halo of weak grid neighbours | `nearest_stronger`; `plot_likelihood` |
| `--censor` | the segmentation cut, times the pixel-frames a faint spot covers: empty bins below it agree with the model | ask the human for the cut |

`anri.index.orientations.select_by_likelihood` redoes the choice for other cuts without fitting again: the file keeps every
ratio. Writes `kept`, `U`, `completeness`, and for likelihood pruning `fitted` and `likelihood_ratio`.

## 4. fit: `--y0`, `--mask`, `--pad`, `--cand`, `--iter`, `--coarse`, `--beam`, `--mu`, `--censor`, `--block-gb`

Fits each voxel's occupancy of its `--cand` best orientations by MLEM, every voxel jointly.

| Choice | Look at | Function |
|---|---|---|
| `--y0` | the sinogram's middle between the sample's edges at every ω; a back-projection at that y0 (a wrong y0 blurs the edge and doubles grain boundaries) | `y0_from_edges`, `back_projection`; `plot_y0` |
| `--mask`, `--pad` | `disc` fits the voxels the beam sees at every ω; `auto` the sample's support in the projections; or a mask file (`anri.index.mask.save_mask`, `draw_mask`). A sample a little larger than the disc piles onto its edge without `--pad` | `back_projection`; `plot_mask` |
| `--iter` | deviance per iteration (logged), and how much occupancy still moves between orientations | `convergence_recorder` (a callback for `stage_fit`); `plot_convergence` |
| `--cand` | each voxel's sorted occupancy shares: the last candidates should hold little | `candidate_shares` |
| `--beam` | the beam's FWHM across dty, when wider than the dty step (e.g. overfocused) | ask the human |
| `--coarse` | for large maps: fits coarse voxels first (4 is about 16 times cheaper), at the risk of missing a grain the coarse fit missed | the log |

Time grows with voxels × candidates × reflections × iterations. More iterations sharpen thin features; fewer keep
the populations' means a little closer to the truth between grid points. Absorption is modelled when the histogram
had `--cif` or the fit gets `--mu`; the log says which.

Writes `f` and `cand` (occupancies and orientation indices per voxel), `U`, `comp`, `pos`, `mask`, and per dty row
`row_data`, `row_model` and `row_ratio`; `support` with `--mask auto`; and with `save_model=True` (from Python)
the fitted histogram `model`. `ring_ratios` compares fitted and measured intensity per ring, `plot_row_ratios` draws
both, and `plot_residual` draws data, model and residual ring by ring: compact positive spots are intensity no kept
orientation explains (a missed grain, or another phase), negative ones are predicted where there is nothing.

## 5. localsearch: `--levels`, `--cover`, `--per-voxel`

Replaces each orientation a voxel uses by 27 children a third of the step apart and fits again, level by level (2
levels take a 1° grid to 0.11°). It removes the ghost twins a coarse grid makes. Each level costs about one fit.
Writes a file like the fit's, which the populations stage reads.

| Choice | Look at | Function |
|---|---|---|
| `--cover`, `--per-voxel` | before running, on the fit file: how many orientations each occupied voxel needs to hold `cover` of its occupancy; red over whole grains means raise `--per-voxel`. A level predicts 27 children per orientation used | `orientations_needed`; `plot_local_slots` |
| `--levels`, and whether it worked | the deviance at each level (`level_recorder` as the stage's callback); how far each voxel's main population moved: up to the grid's worst case is expected, a twin's misorientation means the main population changed (a ghost lost, or a thin twin won) | `local_change`; `plot_local` |

## 6. populations: `--occupied`, `--min-frac`

Groups each voxel's occupied candidates into populations (within 1.8 grid steps of the most occupied one), each
with a fraction, mean orientation, spread and completeness.

| Choice | Look at |
|---|---|
| `--occupied` | the histogram of total occupancy in the mask: the sample well apart from the air; put the cut in the gap |
| `--min-frac` | second populations: real at boundaries and lamellae, a weak scatter everywhere is the floor; put the cut above it |

Both are cheap to redo: `anri.index.populations.select_populations` recomputes the choice from the file. Writes `frac`,
`U_pop`, `spread` (deg, an upper bound), `n`, `comp_pop`, `occupied`, `present`; beside it `<stem>_entries.npz`
(every population as map entries, which `python -m anri.simulate` renders) and, with ImageD11, `<stem>_tmap.h5` (a
TensorMap of the main population, with `.xdmf` for ParaView).

## powder (an export, not a stage of the indexer)

`python -m anri.index powder <analysisroot> <sample> <dataset>` sums every frame into a powder pattern
(`_powder.h5` and a `.xye`), e.g. to identify phases or refine a structure elsewhere.

## Reading a stage file

```python
import h5py

from anri.io.provenance import read_provenance

prov = read_provenance("<file>.h5")  # stage, id, source, options, resolved, results, log, ...
with h5py.File("<file>.h5", "r") as h:
    f = h["f"][()]  # any dataset named above
```

`anri.index.stages.run_command(path)` returns the `python -m anri.index run` command that remakes a file and every
stage before it.
