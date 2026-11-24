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

## 1. histogram: `--rings`, `--tth-tol`, `--cif`, `--monitor`, `--gridstep`, `--saturation`, `--detector-mask`, `--pol-factor`

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
| `--pol-factor` | the beam's degree of horizontal polarisation, in the predicted intensities (default 1, a synchrotron) | ask the human for a lab source |

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
| `--censor` | default measured: 3 pixels per spot x the smallest recorded pixel (logged). Ask the human for the segmentation's `pixels_in_spot` if not ImageD11's default 3, and give that x the logged pixel. Never raise it to where model and data disagree (e.g. a fit's lit curve): orientations whose spots fall in empty bins below it cost nothing | `anri.index.occupancy.censor_level` |

`anri.index.orientations.select_by_likelihood` redoes the choice for other cuts without fitting again: the file keeps every
ratio. Writes `kept`, `U`, `completeness`, and for likelihood pruning `fitted` and `likelihood_ratio`.

## 4. fit: `--y0`, `--mask`, `--pad`, `--cand`, `--iter`, `--tol`, `--price-rounds`, `--price-eps`, `--refine-to`, `--levels`, `--min-share`, `--per-voxel`, `--beam`, `--mu`, `--censor`, `--block-gb`

Fits each voxel's occupancy of orientations by MLEM, every voxel jointly, coarse to fine in orientation and space:

1. The filter's orientations are refined on the row-summed data, without tomography, to a worst case of
   `--refine-to` (27 children a third of the step apart per round, kept by the filter's `--min-lr` and `--keep`). A
   coarser orientation puts its spots in the wrong 1° bins: on half-sample scans that starved the voxels near the axis
   and made streaks, and even on full scans left a checkerboard.
2. Level 0: each voxel's `--cand` best orientations among all of them, on voxels about R × the worst case wide (R
   the mask's radius: an orientation error e moves a voxel's predicted row by up to R e).
3. Each of `--levels` more: orientations subdivided by 3 and voxels down to the dty step. A voxel's orientations
   holding at least `--min-share` of it (at most `--per-voxel`) pass their 27 children to the finer voxels of its
   3 × 3 neighbourhood, each started at its parent's occupancy / 27; a thin twin lamella keeps its own.

| Choice | Look at | Function |
|---|---|---|
| `--y0` | over a full turn, `pairs`: from the Friedel pairs (each reflection at dty, ω, η is seen again at 2 y0 − dty, ω + 180°, 180° − η), with the drift (`--drift`); the fit logs it. Otherwise the sinogram's middle between the sample's edges at every ω (needs the whole sample in the scan). Check either on a back-projection at that y0 (a wrong y0 blurs the edge and doubles grain boundaries; it uses the histogram's sinogram at the frames' ω step, and joins a full turn's two halves, so a half-sample scan reconstructs whole; give it the drift's offsets) | `anri.index.drift.measure`, `y0_from_edges`, `back_projection`; `plot_drift`, `plot_y0` |
| `--drift` | over a full turn, `pairs` follows the sample's drift against the rotation axis, measured per dty row from the same Friedel pairs (the part odd about the time row y0 was measured) and smoothed over 31 rows for the fit (per-row noise would make the rows' sensitivities wrong). The fit's log gives its range, its row-to-row change against its noise (~1 is smooth) and the rows where it jumps (a slip shows at its time and at the mirrored row's); the fit file keeps it (`drift_s`, `drift_c`, `drift_err`). Use it when the second populations make wide bands along every boundary at fine voxels | `anri.index.drift.measure`, `anri.index.drift.roughness`; `plot_drift` |
| `--mask`, `--pad` | `disc` fits the voxels the scan sees from every direction (over a full turn, out to the scan's farther end: a half-sample scan's whole sample); `auto` the sample's support in the projections; or a mask file (`anri.index.mask.save_mask`, `draw_mask`). A sample a little larger than the disc piles onto its edge without `--pad` | `back_projection`; `plot_mask` |
| `--iter`, `--tol` | each level's log line "converged in N iterations" or "NOT converged": a level that stops at `--iter` first is a filter (raise `--iter`, never lower it to regularise). `--tol` is the estimated deviance still to fall (default 1); also the deviance per iteration of the last level, and how much occupancy still moves between orientations | `convergence_recorder` (a callback for `stage_fit`); `plot_convergence` |
| `--price-rounds`, `--price-eps` | the log's last "pricing:" line: "certificate holds" means the map is the maximum likelihood over every orientation of the last level, whatever the levels did (to `--price-eps`: no orientation outside the candidates could lower the deviance by more than 2 x it per photon); otherwise how many voxels still have an orientation outside their candidates that would raise it, and how many ran out of slots (raise `--cand`) or rounds (raise `--price-rounds`) | the log; `anri.index.occupancy.price` |
| `--cand` | each voxel's sorted occupancy shares: the last candidates should hold little; and at each later level the log's share of the coarse voxel's occupancy left out (each kept orientation takes 27 slots): raise `--cand` if many voxels lose over 5%; likewise the share not subdivided: raise `--per-voxel` (broad, deformed grains) | `candidate_shares` |
| `--refine-to`, `--levels` | the log: each refinement round's orientations kept, each level's voxel size and worst case, and a warning when the last level's orientations still move a voxel at the radius by more than a dty step (add a level). Defaults: 1° grid, refined to 0.29°, one level to 0.096°. Finer orientations come from `--levels`, not a smaller `--refine-to`: refined globally, a grain's near-duplicate orientations crowd each voxel's `--cand` best (on Renk_a of ihma936, 0.1 deg made the deviance 62% worse, 256 candidates covering 67 distinct parents) | the log |
| `--beam` | the beam's FWHM across dty, when wider than the dty step (e.g. overfocused) | ask the human |

`--refine-to 0 --levels 0` fits the filter's grid orientations as they are, on the dty step: the map
`python -m anri.calib fit` needs (it refuses refined orientations). Time grows with voxels × candidates × reflections
× iterations, per level; the coarse first level is cheap. More iterations sharpen thin features. Absorption is
modelled when the histogram had `--cif` or the fit gets `--mu`; the log says which.

Writes `f` and `cand` (occupancies and orientation indices per voxel), `U` (the last level's orientations), `comp`,
`pos`, `mask`, and per dty row `row_data`, `row_model` and `row_ratio`; `support` with `--mask auto`; and with
`save_model=True` (from Python) the fitted histogram `model`. `ring_ratios` compares fitted and measured intensity per
ring, `plot_row_ratios` draws both, and `plot_residual` draws data, model and residual ring by ring: compact positive
spots are intensity no kept orientation explains (a missed grain, or another phase), negative ones are predicted where
there is nothing.

## 5. populations: `--occupied`, `--min-frac`

Groups each voxel's occupied candidates into populations (within `anri.index.populations.radius` of the most
occupied one: a 1° histogram bin, or 1.8 × the fitted orientations' spacing if coarser; logged), each
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
