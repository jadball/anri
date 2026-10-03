# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `anri.geom.beam_basis`: unit vectors along a beam and across it (horizontal and vertical).
- `anri.io.beam_from_pars` and `geom_from_pars` take a beam direction, `k_in_lab` (default lab x).
- `anri.fwd.guess_batch_size`: the largest `batch` for `render_row` that fits in a fraction (default 25%) of the free GPU or host memory, from XLA's memory analysis of the compiled render step. Adds `psutil` as a dependency. Raises `RuntimeError` where XLA gives no usable memory analysis (e.g. jaxlib 0.4.28 on macOS).
- `anri.fwd.check_render`: checks rendered peaks of your own map and geometry against a Monte Carlo simulation of the beam spreads through the forward model, and reports the worst cell error and window capture for each peak.
- The renderer is public: `anri.fwd.render_row`, `make_row`, `select_peaks`, `render_peaks`, `dty_weight`, `lorentz` and `polarisation` (previously only importable from `anri.fwd._impl.render`).
- `anri.io.detector_from_pars`, `gonio_from_pars` and `beam_from_pars`: the detector, goniometer and beam parts of `geom_from_pars`, usable on their own (e.g. without the renderer's spreads). `detector_from_pars` also returns the pixel-to-lab transforms for `anri.geom.det_to_lab`, and so does `geom_from_pars`.
- `anri.fwd.get_centroid_box_both` (and `_all_grains_both`, `_all_both`): both Friedel peak centroids of a box-beam forward projection from one call, like `get_centroid_scan_both`.

### Changed

- For a beam tilted out of the horizontal plane, the scattering origin in the scanning model is where the pencil crosses the voxel's column: the beam is taken to cross the rotation axis at the voxel's own height (the layer's height in a 2D map), so the origin is raised by (k_z / k_x) x. Nothing changes for a horizontal beam.
- `anri.fwd.dty_weight` takes the beam direction (`k_in_lab`, default lab x), and the renderer passes it: a beam turned by psi in the horizontal plane sees the voxel rotated by omega - psi and a dty offset moves the voxel delta cos(psi) across it; a pencil tilted by alpha out of the plane travels 1/cos(alpha) further through the voxel column. The dty selection margin scales by 1/cos(psi).
- `anri.fwd.polarisation(k_in, k_out, factor)` takes the beam direction: horizontal polarisation is across the beam, wherever it points.

### Fixed

- Beam divergence (`ky`, `kz`) was applied along lab y and z, which is only across the beam when it is along lab x: for a tilted beam the vertical divergence shrank by cos(tilt). It is now applied along the beam's own horizontal and vertical (`anri.geom.beam_basis`), in every forward model.
- `anri.geom.step_grid_from_ybincens` always raised: it was jitted, but the size of its grid depends on its inputs.
- `Structure` warned about missing thermal factors for crystals built in code (not read from a CIF), even when their atoms had U_iso.
- `render_peaks` put up to ~3% of a peak's intensity in the wrong cells (and lost up to ~7% from broad, truncated peaks): when conditioning fast on slow it held omega at its frame mean, ignoring that slow and omega are correlated within the frame. It now integrates omega out within the frame. Errors against a Monte Carlo of the beam spreads went from 1% (median worst cell) to 0.13%, the Monte Carlo noise.
- Wrong renders on CPU with jaxlib >= 0.11: XLA:CPU's YNNPACK fusions miscompiled `render_peaks` for batches of more than a few thousand peaks (in float64, most peaks squeezed into one pixel; in float32, NaNs). `anri.utils.setup()` now turns them off with `--xla_cpu_experimental_ynn_fusion_type=`. CPU renders made with jaxlib 0.11.x without this fix should be redone.
- `anri.crystal.__all__` and `anri.fwd.__all__` listed functions that don't exist, so `from anri.crystal import *` failed. `anri.fwd` now exports `propagate_cov_scan_both`.
