# Archived: local-grid refinement prototype

**Retired 2026-10-05.** It re-fitted which population owns each voxel, with ~125 free occupancies per population on a
private fine grid against fine bins, and made grain boundaries worse than the indexer's on real data. Refinement now
starts from the indexer's map entries in `anri.refine` (see `../DESIGN.md`). Kept for reference; paths and APIs in
these scripts may be out of date.

## Original notes

**To run on a dataset:** `run_stage3.py`, on the `_index.npz` of `python -m anri.index` (see its docstring). Tested
end to end on the am316l phantom (1 MLEM iteration per pass: spread 0.82 -> 0.45 deg, voxels with 2+ populations
57% -> 32%); not yet for accuracy at full iterations, nor on real data.

Scratch code from 2026-10-05, kept so it is not lost; see `../DESIGN.md` ("Since this morning").

- `stage3_setup.py <out>`: render the am316l phantom (as the indexing tutorial), run stages 1-2, save `stage2.npz` and `H.npy`.
- `local_exp.py <out> HALF STEP`: local grids (+-HALF at STEP deg) around each voxel's top two populations, against the coarse 1 deg histogram.
- `fine.py`: the sparse fine histogram (CSR over rows, int32 lookups), the fine system (parallax, beam profile over rows) and MLEM.
- `fine_exp.py <out> HALF STEP B_ETA B_OMEGA [BEAM]`: the fine pass on top of `local_exp.py`'s result. `NVOX=n` limits it to n voxels, for timing only: a subset cannot be fitted alone.

They use absolute paths from the session they were written in; adjust before running.
