# Stage 3 prototype (local refinement)

Scratch code from 2026-10-05, kept so it is not lost; see `../DESIGN.md` ("Since this morning").

- `stage3_setup.py <out>`: render the am316l phantom (as the indexing tutorial), run stages 1-2, save `stage2.npz` and `H.npy`.
- `local_exp.py <out> HALF STEP`: local grids (+-HALF at STEP deg) around each voxel's top two populations, against the coarse 1 deg histogram.
- `fine.py`: the sparse fine histogram (CSR over rows, int32 lookups), the fine system (parallax, beam profile over rows) and MLEM.
- `fine_exp.py <out> HALF STEP B_ETA B_OMEGA [BEAM]`: the fine pass on top of `local_exp.py`'s result. `NVOX=n` limits it to n voxels, for timing only: a subset cannot be fitted alone.

They use absolute paths from the session they were written in; adjust before running.
