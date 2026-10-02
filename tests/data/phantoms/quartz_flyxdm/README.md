# Deformed alpha-quartz phantom (flyxdm)

A single 2D slice of one alpha-quartz grain with intragranular misorientation and strain.
Load it with `ImageD11.sinograms.tensor_map.TensorMap.from_h5`.

## Contents

`quartz_flyxdm_tmap.h5` is an ImageD11 `TensorMap` with shape `(1, 96, 106)`.

- 6818 voxels inside the grain (`labels == 0`). Outside the grain, `labels == phase_ids == -1` and the tensors are NaN.
- Step is `(1, 0.787, 0.787)`, in the same units as `X` and `Y` from flyxdm (microns).
- Phase 0: `4.926 4.926 5.4189 90 90 120`, lattice number 154.
- The flyxdm simulation used a wavelength of 0.2845704100778472 Å.

## Convention

`B` has **no 2π factor** (ImageD11/anri convention), so `B[0, 0] ≈ 2 / (sqrt(3) * a) ≈ 0.2345`.
flyxdm's `ub_field.npy` comes from `xfab.tools.form_b_mat`, which includes 2π, so it was divided by 2π before building the `TensorMap`.

## Provenance

Source: `demo/data/{ub_field,X,Y}.npy` from Axel Henningsson's [flyxdm](https://github.com/AxelHenningsson/flyxdm)
(commit `36162e1`, MIT License, Copyright (c) 2023 Axel Henningsson).
Converted to a `TensorMap` in `anri/sandbox/fwd_quartz/alpha_quartz_Axel.ipynb`.
