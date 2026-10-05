# Indexer: current state and next steps (2026-10-05, evening)

Read this first. Below it is the design history (the first draft); where they differ, this section is current.

## Where things are

- **Code:** `anri.index` (data, predict, orientations, occupancy, populations), `anri.crystal` (Laue groups, fundamental zones, orientation grids), `anri.phantom` (polycrystals with cells and twins), ImageD11 readers in `anri.io`.
- **Command line:** `python -m anri.index <analysisroot> <sample> <dataset>` on any ImageD11 dataset. `--check` prints paths, sizes and memory without running.
- **Tutorials (stored runs):** `docs/source/tutorials/phantom.ipynb` and `indexing.ipynb`. Test phantom: `tests/data/phantoms/am316l`.
- **Tests:** `tests/unit/index`, `tests/unit/crystal/test_orientation.py`, `tests/unit/phantom` (an end-to-end two-grain index takes ~13 s).

## The pipeline and its defaults

1. **Rings:** the first `--rings` (6) allowed rings; |F|^2 from `--cif` (default 1). Each ring's 2theta window is measured from 9 sample rows.
2. **Histograms**, built once:
   - the lit map: 0.5 deg eta x 0.25 deg omega, rows summed; lit above `--lit` x the median non-empty bin;
   - the fit data: 1 deg x 1 deg per dty row.
3. **Grid:** Rodrigues for cubic, cubochoric reduced to the fundamental zone (plus a half-step shell) for other symmetries. Step: `--grid`, or the coarsest of 3, 2.5, 2, 1.5, 1 deg with chance completeness <= `--max-chance` (0.5).
4. **Pruning:**
   - completeness (tolerances per prediction from the grid's worst case, ring widths, |sin eta|), as a cheap filter: everything above chance;
   - then `orientation_mlem`: one occupancy per orientation fitted to the row-summed data, kept if its likelihood ratio > `--min-lr` (25);
   - `--prune completeness` gives the old behaviour.
5. **Occupancy:** each voxel's `--cand` (64) candidates by the first MLEM update, then sparse MLEM, in voxel blocks of `--block-gb`. `--coarse G` takes the candidates from a fit G x coarser (~G^2 cheaper on large maps).
6. **Populations:** grouped within 1.8 grid steps, up to 4 per voxel, reported above `--min-frac` (0.1). Voxels count as occupied above `--occupied` (0.2) x the 99th percentile of the occupancy.

## What was learned (with numbers)

- **Tolerances** follow from first order: `delta / cos(theta) (1 + tan(theta) |cot eta|)` in eta and `delta / (cos(theta) |sin eta|)` in omega, checked numerically. Fixed tolerances were too tight for a 2.5 deg grid.
- **Grids:** for cubic, Rodrigues needs about half the orientations of cubochoric for the same worst case. Cubochoric matches orix point for point. Without the shell, the -3 group had gaps up to 1.86 x step.
- **Sparse occupancy:**
  - K = 32 lost twins (77% against 93% within 1 deg); K = 64 matches the dense fit.
  - `--coarse 4` gave the same accuracy as the direct run on the phantom.
- **Pruning by likelihood ratio** (crowded phantom, r = 50 um, 249 grains, 2 deg grid):
  - 4059 orientations recall 99.6% of grains (90% of those under 5 um^2), where completeness needs 19,222 for 98.8% (70%);
  - the voxel fit with the likelihood list took < 10 min on CPU; with the completeness list it did not finish within 30 min.
  - The cost: the global fit also keeps **decoys** that soak up intensity the grid cannot fit (the truth lies between grid points). They have low completeness but high likelihood ratio. On the sparse 25 um phantom, accuracy drops from 88% to 85% within 1 deg, and more voxels get a second population.
  - Gating by completeness removes the decoys, but also small grains (90% -> 30-70% recall under 5 um^2). So the gate is optional (`--min-comp`), and the real fix is item 1 below.
- **Second populations** before the 0.1 cut: a third are real (two orientations in one voxel), a quarter are a neighbour's orientation within 1.5 um, a third are decoys (median fraction 0.05).
  - The neighbour leak is a model gap: the system matrix spreads a voxel over 2 rows, with no beam profile (the phantom's beam is 1.4 um FWHM on 1 um voxels).
- **Precision:** the main population's mean is ~0.25-0.3 x the grid step from the truth. The spread includes the grid spacing, so it is too large as `sig_rot` for `anri.refine`.
- **Beam size** is not measurable from edges along dty: inclined boundaries widen every edge.
- **Small grains under 5 um^2:** 90% are in the pruned list, but only 40% survive into the final map, so the 1 um voxel fit loses half of them.
- **Real data** (maintainer's runs):
  - Dataset A: chance completeness 0.98 at `--lit 1` and 2.5 deg. `--lit 10` and a 1 deg grid gave chance 0.46, and the deviance fell 3.4x.
  - Dataset B: the old `--keep 3000` cap dropped small grains; keeping all 10,354 cut the deviance by 31%.
  - The likelihood pruning has not been run on real data yet: it is planned for the beamtime.

## Since this morning (2026-10-05)

- **Real data:**
  - Dataset A with likelihood pruning: 21k orientations, final deviance 1.265e12 (completeness pruning: 1.360e12 with 27k), 3.5 min on the L40S.
  - An HCP dataset (P63/mmc, cubochoric grid at 1.5 deg) indexed first time. Fine deformation twins were missed (diagnose with `diagnose_twins.py`).
  - `--occupied 0.2` cuts holes into real samples (tuned on the uniform phantom); a rule from the data is wanted.
- **Speed:**
  - The histograms are read-bound: 94 s to read dataset A's 3.77G pixels on one core (HDF5 decompression under h5py's lock), against 95 s for the whole histogram step with `anri.io.prefetch`.
  - Beyond that: decompress raw chunks in threads (`read_direct_chunk` + bitshuffle/LZ4).
  - `inherit_candidates` now uses a KD-tree: 2 min -> 4 s.
- **Twin ghosts:** a twin shares reflections with its parent (a third for Sigma3). Where the grid fits the parent imperfectly, the twin orientation explains part of the shared spots, passes pruning (likelihood ratio > 25) and is fitted as a second population.
  - On the am316l phantom (one twinned grain in 38): the twin of 98% of grains was kept, and most second populations sat 60 deg from the first.
  - These are most of the "decoys" seen before. Local refinement cut voxels with 2+ populations from 57% to 23%: a well-fitted parent leaves nothing to steal.
- **Parent/twin split per voxel** is poorly determined near boundaries on the phantom: pure parent voxels get ~30% twin, the correlation with the truth is 0.66 (fine stage 0.45). It shows as stripes one voxel wide where the main population flips.
- **Ring artefacts** (concentric about the rotation axis) on a helical-scan dataset, from per-row misfit. The CLI logs and saves `row_ratio` (measured / fitted intensity per dty row).
  - Phantom: smooth radial bias of +-7% from the 2-row model.
  - Helical dataset: row-to-row rms 0.19 (HCP dataset: 0.018). The monitor (`--monitor`, now supported, with a master-file fallback) varies only +-6% there, so flux is not the cause.
  - Cause: **it is a helical scan**: dty moves continuously, 2 um per 360 deg turn (from the slope of the rows' readings; the encoder reading steps once per row, so its phase within a turn is unknown and degenerate with y0). The model must shift row r's beam by rate x (omega - omega_ref).
  - The maintainer asked to do this on a machine where the data can be read and debugged. `diagnose_positions.py` shows a DataSet's dty and omega per frame. Its "outside their bin" line is wrong for rows in descending dty.
- **Fine stage (prototype, not in anri.index):** a sparse fine histogram (CSR per (ring, eta, omega) cell over rows, int32 lookups), parallax by ray tracing from each voxel, and the beam profile over rows, with local grids around populations.
  - On the 25 um phantom: the local grid on the coarse 1 deg data already gave main population within 0.5 deg 33.6% -> 73.2%, median 0.60 -> 0.36 deg, and removed most decoys.
  - The fine data did not improve further: 0.25 deg median against each voxel's mean truth. That phantom's truth varies inside a 1 um voxel (0.5 um cells), so it may be the test's limit. **Next: a phantom on the indexing grid with cells larger than a voxel.**
  - Speed: ~100 s per iteration on a laptop CPU for 2060 voxels x 250 candidates. A subset of voxels cannot be fitted alone (other voxels' spots share the rays).
  - Prototype code: `anri/sandbox/indexer/stage3/` (scratch quality, see its README).

## Stage 3 on real data (2026-10-05, evening)

`stage3/run_stage3.py` on an HCP dataset (481 rows of 1.5 um, 3620 frames of 0.05 deg, zigzag, no CIF,
no gridstep). Its default fine bins (0.25 x 0.05 deg) made the map **worse than stage 2**: grain boundaries bled into the
neighbours (along stage 2's voxels with fraction < 1), and the main fraction went to ~1 everywhere.
`compare_stage3.py` (stage 2 vs 3 in numbers): isolated flips 0.96% -> 3.4-4.9%, 11% of voxels changed their main
population.

- **Cause: the orientation grid is ~10x coarser than the bins.** A true orientation sits up to half a step from its
  nearest grid node, which misplaces its spots by, in omega, median 0.14 deg (90th 0.32) for a 0.5 deg grid and 0.07
  (0.17) for 0.25 deg; in eta about 0.8 x that. With 0.05 deg omega bins the grid cannot put a spot in the right bin,
  so MLEM mixes nodes and borrows intensity from other units (the neighbour's minor populations at boundaries).
  - **Rule: bins no finer than the spot error of the grid step** (roughly bins >= the grid step).
  - `--bins 1.0 1.0` (pass 1 only, 0.5 deg grid): grain shapes decent, boundaries and twins slightly better than
    stage 2. Pass 2 (0.25 deg grid) at 1 deg bins gained 1% deviance for 813 s and looked the same: skip it there
    (`--pass2 0 0`).
  - Untested: one pass at 0.25 deg with bins to match, `--pass1 1.0 0.25 --pass2 0 0 --bins 0.25 0.25 --iter 10`.
  - Bins as fine as frames need continuous orientations per unit (a refiner like `anri.refine`), not a grid.
- **Ruled out or minor:**
  - y0: stage 3 took the DataSet's y0 while stage 2 had `--y0`. Fixed (stage 2 saves y0 in its npz; stage 3 reads it,
    `--y0` overrides), but the fine-bin map stayed bad.
  - zigzag omega offset: real but 0.004 deg (8% of a frame), negligible (`diagnose_zigzag.py`).
  - over-iteration: 5 -> 20 iterations raised flips 3.4 -> 4.7%; the deviance was flat after ~10.
  - the beam: the scan that favoured wider beams (deviance falling up to 5 um FWHM) was run with the wrong y0 and the
    mismatched bins, so it is void. The maintainer's tomo map is sharp at 1.5 um. **Redo it** with the right y0 and
    `--bins 1.0 1.0 --pass2 0 0 --iter 5`, beams 1.0 / 1.5 / 2.5.
- **Model gaps found:** stage 3 had no structure factors (`--cif` added; hcp rings differ in |F|^2 by up to 8x, so a
  CIF should go to both stages), and modelled voxels as one row step (now stage 2's voxel size). Both are logged.
- **Twins** are still almost absent: stage 3 only refines stage 2's units within +-1 deg. Stage 2 fits them but under
  `--min-frac`; `--unit-frac 0.03` makes units of populations down to 3% (stage 2 keeps up to 4 per voxel, above 2%).
  Untested. If not enough: seed twin orientations into stage 2 (new anri.index API: ask first).
- **Peak widths** (instrument and lens parameters: AGENTS.md). `peak_widths.py` measures the omega FWHM of each clean
  3D peak in ImageD11's peaks table, by ring and |sin eta|, and checks whether the "clean" filter biases them.
  (`diagnose_peaks.py` is superseded: its spot moments are inflated by merged spots.) Across real datasets so far:
  - with the Al CRLs, an undeformed single crystal has cores within one 0.05 deg frame at every eta (both stations):
    widths beyond that are the sample's;
  - sample spreads range from below ~0.02 deg to ~0.1 deg FWHM, growing as 1 / |sin eta| (isotropic). Brighter peaks
    (longer chords through larger grains) are often wider: a peak in one row sums the voxels along the ray, so the
    measured spread includes orientation changes along the chord, and a voxel's own spread may be smaller;
  - at |sin eta| < 0.25 broad peaks overlap and fail the "clean" filter, so medians there read low; the share of
    peaks there is low anyway (g near the rotation axis never diffracts);
  - widths under a frame are unresolved: a sub-frame peak split over two frames reads ~0.9 frame.
- **Consequences:** the spot model is the frame width, the vertical convergence (`sig_kz`) and, where the sample needs
  it, a spread per entry (`sig_rot`, isotropic, exists). Stage 3's bins must not be finer than its grid's spot error.
  `anri.refine`'s basin is a tenth of a peak width, or a fraction of a frame for sub-frame peaks (omega is then fixed
  only by how a peak splits between frames, and the detector position carries most of the orientation): stage 3's
  grid must be followed by a continuous stage (see "Proposed structure" below).
- **Speed (L40S, 116k units):** pass 1 (125 orientations) 22-29 s/iteration, pass 2 (729) 118-147 s/iteration. Coarser
  bins were slower, probably contention in the scatter-add (unchecked).

## Proposed structure after the initial index (2026-10-05, late; not yet agreed)

1. Histograms (`anri.index`, exists).
2. Global index (`anri.index`, exists): ~0.3 x the grid step; ghosts and decoys remain.
3. **One local-grid pass** (stage 3 prototype): +-1.5 steps around each population, bins matched to the step (a grid
   cannot go finer: its spots are misplaced by ~a quarter step). Joint MLEM removes the ghosts; ~0.1-0.25 deg.
   Untested on the HCP dataset: `--pass1 1.0 0.25 --pass2 0 0 --bins 0.25 0.25 --iter 10`.
4. **Continuous coarse-to-fine (new):** each entry's rotation (3) and log density, Gauss-Newton, joint over voxels,
   against the histograms with Gaussian spots of width w (`bin_fractions`), w shrinking (e.g. 0.25 deg -> the measured
   width) with the bins. Each level starts within a fraction of its own w. Drop entries whose density goes to 0, merge
   entries that converge together (mean + spread). Rotation only: the ring windows integrate over 2theta.
5. `anri.refine` on pixels (exists): F and density, censoring, a spread per entry where the peaks need it.

Open: where stage 4 lives (`anri.index`, or a histogram mode of `anri.refine`): ask before adding API.

## Next steps, in order

1. **Local refinement (stage 3)**, see the section above for its state on real data. Also the fix for twin ghosts and grid-limited spreads. For each voxel's populations, a local grid (e.g. +-1.5 deg at 0.25 deg) against finer data (omega at frame resolution, finer eta), fitted again by sparse MLEM: each voxel's candidates become its local grid, so `fit_occupancy`'s machinery mostly applies.
   - Fixes: the decoys, precision (~0.1 deg needed for wide samples like dataset B), and the grid-inflated spread.
   - Open: memory of the finer histogram on large maps; it probably needs blocks of rows.
2. **Beam profile in the system matrix:** a voxel spread over the rows the beam reaches, not 2. Do it together with 1, as both change `system`.
3. **Censoring** below the segmentation cut in both MLEMs (as `anri.refine` does): weak predictions are biased to zero now.
4. **Validation on real data** (dataset A, dataset B) against ImageD11's pbp and refined maps:
   - Is `--min-lr 25` right under strain and distortion mismatch?
   - How many small grains does each method find?
5. **Smaller items:**
   - mask empty voxels from the coarse fit (~2x on half acquisitions such as dataset B);
   - a better automatic grid rule (`--max-chance 0.5` picks 3 deg on sparse phantoms, where 2 deg is better);
   - a non-cubic phantom, end to end.

## Reproducing the test cases

- **25 um phantom:** the tutorial `indexing.ipynb` renders and indexes it in ~3 min on a laptop CPU.
- **Crowded, small-grain phantom:** `anri.phantom.polycrystal(n=203, step=0.5, radius=50, n_grains=300, cell_size=1.5, cell_spread_deg=0.3, twin_grains=3, seed=11)`, rendered as in the tutorial (dataset A geometry, 8 rings, dty +-55 um in 1 um steps, 1800 frames of 0.1 deg).
  - 31k entries, ~5 min to render on CPU.
  - Compare pruning per grain by checking whether a kept orientation lies within delta + 0.5 deg of each grain's mean, binned by grain area.

## History: the first draft and its status notes (2026-10-04)

Status (2026-10-04 20:10). The text below is the first draft. Since then the maintainer decided:

- **Dropped:** the moment stage (stage 4).
- **Core:** MLEM occupancy over a global, pruned orientation list (Jon Wright's idea); observables coarsened to the grid scale.

Implemented in index.py, run_phantom.py and a run script for dataset A:

- **Phantom:** 99.3% of voxels within 1°.
- **Dataset A:** a map that matches ImageD11's well but with poor fidelity. Pruning is arbitrary on crowded data; (a memory note of the time, not in the repo).

Since then (2026-10-04, run_index.py):

- **Tolerances from the data:** ring widths are measured in a pre-pass (2theta tolerance per ring); each prediction gets its own eta and omega box from the grid's worst-case misorientation, its ring and |sin eta| (`match_tolerances`), checked with a summed-area table of the lit map (`completeness_tol`). Orientations are kept above a completeness threshold halfway between chance (the grid median) and the maximum.
- **Test phantom:** `make_phantom.py` renders an AM-like 316L phantom (grains, 1.5 um cells, a twinned grain) as an ImageD11 dataset. On a crowded one (r = 25 um, 40 grains, 0.25 um voxels), at the same number of orientations, the weighted-mean orientation is within 1 deg for 82% of voxels (71% with the old fixed tolerances).
- **Sparse occupancy:** each voxel keeps its --cand (64) best orientations by the first MLEM update from unit occupancy (two blocked passes over every voxel and orientation), then MLEM runs on [voxels, K] in blocks of voxels sized by --block-gb. Crowded phantom: 72-120 s against 438 s dense, same accuracy at K = 64 (K = 32 loses twins: 77% against 93% within 1 deg).
- **Populations:** each voxel's occupied candidates are grouped greedily (seed = most occupied, members within 1.8 grid steps over the cubic symmetry) into up to 4 populations with fraction, mean orientation, rms spread (includes the grid spacing: an upper bound) and completeness; those under --min-frac (0.1) are not reported. Output: the TensorMap holds population 1's mean, and `_entries.npz` holds every population as anri map entries. Crowded phantom, 2 deg grid: population 1 within 1 deg for 91% (the top grid point: 55%), parent and twin both found where a voxel holds both. Of the second populations before the cut, a third are real mixtures in the voxel, a quarter are a neighbour's orientation within 1.5 um (no beam profile in the system matrix: blur is explained by neighbours), a third are decoys (median fraction 0.05).
- **Coarse to fine (--coarse G):** the full candidate pass and MLEM on voxels G x larger (data rows summed in groups of G), then each voxel scores only the 2K orientations its 3 x 3 coarse neighbourhood occupied most. Crowded phantom at G = 4: the same accuracy as the direct run (population 1 within 1 deg: 90.6% against 90.8%; twins found). Dataset B (1043 x 1043 voxels, 7410 orientations): the direct candidate pass took 17 min.
- **Beam size:** not measurable from edges along dty: inclined boundaries widen every edge (1.4 um beam read as ~2-2.4 um). Measure it at the beamline, or fit it in the refinement.

Written 2026-10-04 after the refinement study (`anri/sandbox/math/NOTES.md`, `anri/sandbox/moments/`).

### Goal

Find each voxel's orientation populations from the data and the forward model alone:

- **Known in advance:** phases and lattices.
- **Not known:** grains, tomo shapes, point-by-point maps.

Each voxel gets a few entries with fractions, located to within the basin of the local moment refinement, ~beam / r (~0.1° for real samples). The moment refinement then finishes the job: orientation to the instrument limit, plus strain and spread.

**First real target: dataset A (FCC).**

- 201 rows (dty ±100 µm, 1 µm steps) × 3620 frames (0.05°, ω −90° to 91°).
- ~31.7k voxels in the disk at 1 µm.
- 3.77G sparse pixels (18.7M per row).
- FCC, a = 3.597 Å, λ = 0.284 Å.

### What the study established (the constraints)

1. **The data are rich.** Per dislocation cell: ~5e-4° in orientation, ~1e-5 in strain at 100 counts per sub-peak. Pixels hold about the same information as the moments of separated sub-peaks.
2. **ω is both the Bragg angle and the projection angle.** A voxel that is Δω off lights rows r·Δω away. Every row-wise local method therefore has a basin of ~beam / r.
3. **Scoring voxels independently** (ImageD11 pbp, the local grid search) is biased by other voxels on the same rays. That's merging; in the test, the score preferred a wrong rotation in 73% of failures.

### Principle

The data are linear in the sample's density over position and orientation, f(v, q) (voxel v, orientation q):

    d = A f        A: the forward model (renderer geometry: Bragg condition, beam, detector)

- **The adjoint is a back-projection.** Aᵀd(v, q) collects, over the reflections of q, the data at q's predicted (η, ω) in the row where voxel v sits at that ω.
- **ImageD11 pbp and the grid search are one back-projection,** which is why they suffer from merging.
- **Iterating on the residual removes the blur,** as in tomography:

      f ← P( f + τ Aᵀ(d − A f) )       P: f ≥ 0, keep the few largest orientations per voxel

  This handles merging by construction. There is no start map and no basin: each step is linear.

A useful structure: for a fixed orientation q and reflection h, the predicted (η, ω) does not depend on the voxel (parallax aside). So Aᵀd for one q is a sum over reflections of one-angle back-projections of a 1D row profile (the data at (η_qh, ω_qh) as a function of dty) along angle ω_qh. That is an ordinary tomographic back-projection, one per orientation.

### Stages

1. **Histogram the data once** into H[ring, η, ω, row], at a resolution matched to the orientation grid (coarse: η 0.5°, ω 0.25°).
   - Dataset A, first 4 rings: 4 × 720 × 724 × 201 ≈ 4.2e8 cells (1.7 GB float32).
   - One streaming pass over the 3.77G pixels, in chunks on the GPU. The 8 GB read from disk dominates.
2. **Coarse lifted reconstruction** on a fundamental-zone grid:
   - 2°: ~78k orientations for cubic; 1°: ~620k.
   - Low-order rings only (~50 reflections × 2 branches) at this stage.
   - Each iteration = one back-projection and one forward projection. Dataset A: 31.7k voxels × 78k orientations × ~100 reflections ≈ 2.5e11 lookups, ~1 min on the L40S.
   - Keep the top K (~4) orientations per voxel with weight. Sparsity keeps A f cheap: forward-project only the kept (v, q).
3. **Local refinement of each kept orientation:** the same lifted iteration on a local grid around it (±1° at 0.25°, then ±0.25° at 0.05°), with finer histograms (one frame in ω) and all rings. This replaces the independent grid search; merging is still handled jointly.
4. **Hand over to the moment refinement** (the existing prototype): entries = (orientation, fraction) per voxel, then refine rotation, then strain, then spread.

Grain shapes come out as regions of voxels sharing an orientation; no tomo step is needed.

### Cost and scale

- **Dataset A (31.7k voxels):** minutes per stage, a few GB of GPU memory. Fine.
- **400 × 400 laptop case (~125k voxels):** 4× dataset A. Fine on CPU if slow; chunk over orientations.
- **3k × 3k (~7M voxels):** the full coarse product is ~2e14 lookups, too much. Use multi-resolution in space: run stage 2 on voxels 4–8× coarser, then only refine the surviving orientations per region at full resolution.

### Test plan

1. **An AM-like phantom at the dataset A geometry,** first small (r = 25 µm, 1 µm voxels): columnar grains, solidification cells of 0.5–1 µm with ≲0.5° misorientation, a few degrees of drift along columns. Render it with Anri, then index from nothing. Score per voxel against the truth: fraction within 0.1°, and shapes.
2. **Dataset A:** compare with the ImageD11 pbp map and the refined TensorMap that are already in PROCESSED_DATA.

### Code

Plain functions in a new `anri/index` module:

- `histogram` (pixels → H);
- `backproject` and `forward` (H ↔ f, sparse in q);
- `fz_grid` (fundamental-zone and local grids);
- `reconstruct` (the iteration).

No classes. Reuse the renderer's geometry functions (`hkl_to_k_omega`, `raytrace_to_det`, `sample_to_lab`, `beam_weight`).

### Open questions for the maintainer

1. **Beam size and profile for dataset A:** 1 µm FWHM? Al CRL or Si lenses?
2. **Coarse grid:** is 2° fine enough to see AM solidification cells (≲0.5°) as one population at stage 2? If not, stage 3 has to split them.
3. **K, populations per voxel:** 4 at the coarse stage, then pruned by fraction?
4. **Rings for the coarse stage:** the first 4 FCC rings (111, 200, 220, 311)?
