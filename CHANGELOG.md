# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `anri.fwd.get_centroid_box_both` (and `_all_grains_both`, `_all_both`): both Friedel peak centroids of a box-beam forward projection from one call, like `get_centroid_scan_both`.

### Fixed

- Wrong renders on CPU with jaxlib >= 0.11: XLA:CPU's YNNPACK fusions miscompiled `render_peaks` for batches of more than a few thousand peaks (in float64, most peaks squeezed into one pixel; in float32, NaNs). `anri.utils.setup()` now turns them off with `--xla_cpu_experimental_ynn_fusion_type=`. CPU renders made with jaxlib 0.11.x without this fix should be redone.
- `anri.crystal.__all__` and `anri.fwd.__all__` listed functions that don't exist, so `from anri.crystal import *` failed. `anri.fwd` now exports `propagate_cov_scan_both`.
