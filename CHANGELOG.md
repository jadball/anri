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
- The renderer is public: `anri.fwd.render_row`, `make_row`, `select_peaks`, `render_peaks`, `beam_weight`, `lorentz` and `polarisation` (previously only importable from `anri.fwd._impl.render`).
- `anri.io.detector_from_pars`, `gonio_from_pars` and `beam_from_pars`: the detector, goniometer and beam parts of `geom_from_pars`, usable on their own (e.g. without the renderer's spreads). `detector_from_pars` also returns the pixel-to-lab transforms for `anri.geom.det_to_lab`, and so does `geom_from_pars`.
- `anri.fwd.get_centroid_box_both` (and `_all_grains_both`, `_all_both`): both Friedel peak centroids of a box-beam forward projection from one call, like `get_centroid_scan_both`.

### Changed

- `anri.fwd.render_row` is faster on GPUs: duplicate pixels (neighbouring voxels light up the same pixels, ~100x for a grain) are summed on the device, so only unique pixels go to the host, and batches are 1024 peaks per device times a power of 4, so a scan compiles at most a few shapes (XLA:GPU took up to a minute to compile very small batches). A 153-row scan of a 6818-voxel grain renders in 1 minute instead of 3. Output is unchanged up to float rounding; CPUs keep the host merge, as XLA's CPU sort is slower than NumPy's.
- For a beam tilted out of the horizontal plane, the scattering origin in the scanning model is where the pencil crosses the voxel's column: the beam is taken to cross the rotation axis at the voxel's own height (the layer's height in a 2D map), so the origin is raised by (k_z / k_x) x. Nothing changes for a horizontal beam.
- The renderer handles pencil, line and box beams (e.g. DCT) with one model. Every voxel sits at its real lab position for the row's dty (it used to be moved onto the pencil's centre line), and `anri.fwd.beam_weight` (replacing `dty_weight`) integrates the beam's profile across it over the voxel: horizontally and vertically, each a flat top blurred by a Gaussian (`geom_from_pars(..., sig_beam, width_beam, sig_beam_v, width_beam_v)`). Voxels are columns for 2D maps or cubes for 3D maps (`voxel_3d`). The beam can point anywhere (`k_in_lab`): it sees the voxel rotated by omega - psi, and travels 1/cos(alpha) further through a column. `select_peaks` keeps peaks whose voxel is within reach of the beam.
- `anri.fwd.polarisation(k_in, k_out, factor)` takes the beam direction: horizontal polarisation is across the beam, wherever it points.

### Fixed

- Wrong spot positions in float32 on recent NVIDIA GPUs: JAX's default precision for float32 matrix products there is TF32 (10-bit mantissa), which moved rendered spots ~1000 px from the beam centre by up to 0.3 px on an L40S, a strain error of ~1e-5. `anri.utils.setup()` now sets `jax_default_matmul_precision` to `"highest"` (unless `JAX_DEFAULT_MATMUL_PRECISION` is set). GPU renders made in float32 without this fix should be redone.
- Beam divergence (`ky`, `kz`) was applied along lab y and z, which is only across the beam when it is along lab x: for a tilted beam the vertical divergence shrank by cos(tilt). It is now applied along the beam's own horizontal and vertical (`anri.geom.beam_basis`), in every forward model.
- `anri.geom.step_grid_from_ybincens` always raised: it was jitted, but the size of its grid depends on its inputs.
- `Structure` warned about missing thermal factors for crystals built in code (not read from a CIF), even when their atoms had U_iso.
- `render_peaks` put up to ~3% of a peak's intensity in the wrong cells (and lost up to ~7% from broad, truncated peaks): when conditioning fast on slow it held omega at its frame mean, ignoring that slow and omega are correlated within the frame. It now integrates omega out within the frame. Errors against a Monte Carlo of the beam spreads went from 1% (median worst cell) to 0.13%, the Monte Carlo noise.
- Wrong renders on CPU with jaxlib >= 0.11: XLA:CPU's YNNPACK fusions miscompiled `render_peaks` for batches of more than a few thousand peaks (in float64, most peaks squeezed into one pixel; in float32, NaNs). `anri.utils.setup()` now turns them off with `--xla_cpu_experimental_ynn_fusion_type=`. CPU renders made with jaxlib 0.11.x without this fix should be redone.
- `anri.crystal.__all__` and `anri.fwd.__all__` listed functions that don't exist, so `from anri.crystal import *` failed. `anri.fwd` now exports `propagate_cov_scan_both`.
