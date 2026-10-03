# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `anri.fwd.guess_batch_size`: the largest `batch` for `render_row` that fits in a fraction (default 25%) of the free GPU or host memory, from XLA's memory analysis of the compiled render step. Adds `psutil` as a dependency.
- `anri.fwd.check_render`: checks rendered peaks of your own map and geometry against a Monte Carlo simulation of the beam spreads through the forward model, and reports the worst cell error and window capture for each peak.
- The renderer is public: `anri.fwd.render_row`, `make_row`, `select_peaks`, `render_peaks`, `dty_weight`, `lorentz` and `polarisation` (previously only importable from `anri.fwd._impl.render`).
- `anri.io.detector_from_pars`, `gonio_from_pars` and `beam_from_pars`: the detector, goniometer and beam parts of `geom_from_pars`, usable on their own (e.g. without the renderer's spreads). `detector_from_pars` also returns the pixel-to-lab transforms for `anri.geom.det_to_lab`, and so does `geom_from_pars`.
- `anri.fwd.get_centroid_box_both` (and `_all_grains_both`, `_all_both`): both Friedel peak centroids of a box-beam forward projection from one call, like `get_centroid_scan_both`.

### Fixed

- `anri.geom.step_grid_from_ybincens` always raised: it was jitted, but the size of its grid depends on its inputs.
- `Structure` warned about missing thermal factors for crystals built in code (not read from a CIF), even when their atoms had U_iso.
- `render_peaks` put up to ~3% of a peak's intensity in the wrong cells (and lost up to ~7% from broad, truncated peaks): when conditioning fast on slow it held omega at its frame mean, ignoring that slow and omega are correlated within the frame. It now integrates omega out within the frame. Errors against a Monte Carlo of the beam spreads went from 1% (median worst cell) to 0.13%, the Monte Carlo noise.
- Wrong renders on CPU with jaxlib >= 0.11: XLA:CPU's YNNPACK fusions miscompiled `render_peaks` for batches of more than a few thousand peaks (in float64, most peaks squeezed into one pixel; in float32, NaNs). `anri.utils.setup()` now turns them off with `--xla_cpu_experimental_ynn_fusion_type=`. CPU renders made with jaxlib 0.11.x without this fix should be redone.
- `anri.crystal.__all__` and `anri.fwd.__all__` listed functions that don't exist, so `from anri.crystal import *` failed. `anri.fwd` now exports `propagate_cov_scan_both`.
