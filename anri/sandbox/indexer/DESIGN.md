# A forward-model indexer for scanning 3DXRD (draft for review)

Status (2026-10-04 20:10). The text below is the first draft. Since then the maintainer decided:

- **Dropped:** the moment stage (stage 4).
- **Core:** MLEM occupancy over a global, pruned orientation list (Jon Wright's idea); observables coarsened to the grid scale.

Implemented in index.py, run_phantom.py and run_tognan.py:

- **Phantom:** 99.3% of voxels within 1°.
- **Tognan:** a map that matches ImageD11's well but with poor fidelity. Pruning is arbitrary on crowded data; see the memory note "indexer" for the issues and next steps.

Since then (2026-10-04, run_index.py):

- **Tolerances from the data:** ring widths are measured in a pre-pass (2theta tolerance per ring); each prediction gets its own eta and omega box from the grid's worst-case misorientation, its ring and |sin eta| (`match_tolerances`), checked with a summed-area table of the lit map (`completeness_tol`). Orientations are kept above a completeness threshold halfway between chance (the grid median) and the maximum.
- **Test phantom:** `make_phantom.py` renders an AM-like 316L phantom (grains, 1.5 um cells, a twinned grain) as an ImageD11 dataset. On a crowded one (r = 25 um, 40 grains, 0.25 um voxels), at the same number of orientations, the weighted-mean orientation is within 1 deg for 82% of voxels (71% with the old fixed tolerances).
- **Sparse occupancy:** each voxel keeps its --cand (64) best orientations by the first MLEM update from unit occupancy (two blocked passes over every voxel and orientation), then MLEM runs on [voxels, K] in blocks of voxels sized by --block-gb. Crowded phantom: 72-120 s against 438 s dense, same accuracy at K = 64 (K = 32 loses twins: 77% against 93% within 1 deg).
- **Populations:** each voxel's occupied candidates are grouped greedily (seed = most occupied, members within 1.8 grid steps over the cubic symmetry) into up to 4 populations with fraction, mean orientation, rms spread (includes the grid spacing: an upper bound) and completeness; those under --min-frac (0.1) are not reported. Output: the TensorMap holds population 1's mean, and `_entries.npz` holds every population as anri map entries. Crowded phantom, 2 deg grid: population 1 within 1 deg for 91% (the top grid point: 55%), parent and twin both found where a voxel holds both. Of the second populations before the cut, a third are real mixtures in the voxel, a quarter are a neighbour's orientation within 1.5 um (no beam profile in the system matrix: blur is explained by neighbours), a third are decoys (median fraction 0.05).
- **Beam size:** not measurable from edges along dty: inclined boundaries widen every edge (1.4 um beam read as ~2-2.4 um). Measure it at the beamline, or fit it in the refinement.

Written 2026-10-04 after the refinement study (`anri/sandbox/math/NOTES.md`, `anri/sandbox/moments/`).

## Goal

Find each voxel's orientation populations from the data and the forward model alone:

- **Known in advance:** phases and lattices.
- **Not known:** grains, tomo shapes, point-by-point maps.

Each voxel gets a few entries with fractions, located to within the basin of the local moment refinement, ~beam / r (~0.1° for real samples). The moment refinement then finishes the job: orientation to the instrument limit, plus strain and spread.

**First real target: Tognan AM 316L (AP1_1, z0).**

- 201 rows (dty ±100 µm, 1 µm steps) × 3620 frames (0.05°, ω −90° to 91°).
- ~31.7k voxels in the disk at 1 µm.
- 3.77G sparse pixels (18.7M per row).
- FCC, a = 3.597 Å, λ = 0.284 Å.

## What the study established (the constraints)

1. **The data are rich.** Per dislocation cell: ~5e-4° in orientation, ~1e-5 in strain at 100 counts per sub-peak. Pixels hold about the same information as the moments of separated sub-peaks.
2. **ω is both the Bragg angle and the projection angle.** A voxel that is Δω off lights rows r·Δω away. Every row-wise local method therefore has a basin of ~beam / r.
3. **Scoring voxels independently** (ImageD11 pbp, the local grid search) is biased by other voxels on the same rays. That's merging; in the test, the score preferred a wrong rotation in 73% of failures.

## Principle

The data are linear in the sample's density over position and orientation, f(v, q) (voxel v, orientation q):

    d = A f        A: the forward model (renderer geometry: Bragg condition, beam, detector)

- **The adjoint is a back-projection.** Aᵀd(v, q) collects, over the reflections of q, the data at q's predicted (η, ω) in the row where voxel v sits at that ω.
- **ImageD11 pbp and the grid search are one back-projection,** which is why they suffer from merging.
- **Iterating on the residual removes the blur,** as in tomography:

      f ← P( f + τ Aᵀ(d − A f) )       P: f ≥ 0, keep the few largest orientations per voxel

  This handles merging by construction. There is no start map and no basin: each step is linear.

A useful structure: for a fixed orientation q and reflection h, the predicted (η, ω) does not depend on the voxel (parallax aside). So Aᵀd for one q is a sum over reflections of one-angle back-projections of a 1D row profile (the data at (η_qh, ω_qh) as a function of dty) along angle ω_qh. That is an ordinary tomographic back-projection, one per orientation.

## Stages

1. **Histogram the data once** into H[ring, η, ω, row], at a resolution matched to the orientation grid (coarse: η 0.5°, ω 0.25°).
   - Tognan, first 4 rings: 4 × 720 × 724 × 201 ≈ 4.2e8 cells (1.7 GB float32).
   - One streaming pass over the 3.77G pixels, in chunks on the GPU. The 8 GB read from disk dominates.
2. **Coarse lifted reconstruction** on a fundamental-zone grid:
   - 2°: ~78k orientations for cubic; 1°: ~620k.
   - Low-order rings only (~50 reflections × 2 branches) at this stage.
   - Each iteration = one back-projection and one forward projection. Tognan: 31.7k voxels × 78k orientations × ~100 reflections ≈ 2.5e11 lookups, ~1 min on the L40S.
   - Keep the top K (~4) orientations per voxel with weight. Sparsity keeps A f cheap: forward-project only the kept (v, q).
3. **Local refinement of each kept orientation:** the same lifted iteration on a local grid around it (±1° at 0.25°, then ±0.25° at 0.05°), with finer histograms (one frame in ω) and all rings. This replaces the independent grid search; merging is still handled jointly.
4. **Hand over to the moment refinement** (the existing prototype): entries = (orientation, fraction) per voxel, then refine rotation, then strain, then spread.

Grain shapes come out as regions of voxels sharing an orientation; no tomo step is needed.

## Cost and scale

- **Tognan (31.7k voxels):** minutes per stage, a few GB of GPU memory. Fine.
- **400 × 400 laptop case (~125k voxels):** 4× Tognan. Fine on CPU if slow; chunk over orientations.
- **3k × 3k (~7M voxels):** the full coarse product is ~2e14 lookups, too much. Use multi-resolution in space: run stage 2 on voxels 4–8× coarser, then only refine the surviving orientations per region at full resolution.

## Test plan

1. **An AM-like phantom at the Tognan geometry,** first small (r = 25 µm, 1 µm voxels): columnar grains, solidification cells of 0.5–1 µm with ≲0.5° misorientation, a few degrees of drift along columns. Render it with Anri, then index from nothing. Score per voxel against the truth: fraction within 0.1°, and shapes.
2. **Tognan:** compare with the ImageD11 pbp map and the refined TensorMap that are already in PROCESSED_DATA.

## Code

Plain functions in a new `anri/index` module:

- `histogram` (pixels → H);
- `backproject` and `forward` (H ↔ f, sparse in q);
- `fz_grid` (fundamental-zone and local grids);
- `reconstruct` (the iteration).

No classes. Reuse the renderer's geometry functions (`hkl_to_k_omega`, `raytrace_to_det`, `sample_to_lab`, `beam_weight`).

## Open questions for the maintainer

1. **Beam size and profile for Tognan:** 1 µm FWHM? Al CRL or Si lenses?
2. **Coarse grid:** is 2° fine enough to see AM solidification cells (≲0.5°) as one population at stage 2? If not, stage 3 has to split them.
3. **K, populations per voxel:** 4 at the coarse stage, then pruned by fraction?
4. **Rings for the coarse stage:** the first 4 FCC rings (111, 200, 220, 311)?
