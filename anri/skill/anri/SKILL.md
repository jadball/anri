---
name: anri
description: >-
  Index scanning-3DXRD (S3DXRD) data from scratch, calibrate its geometry, and simulate datasets from a grain map,
  with the anri Python package (python -m anri.index, python -m anri.calib, python -m anri.simulate, anri.fwd). Read
  it before running an anri stage on a dataset; before calibrating the detector and beam geometry; before choosing
  the indexer's options (rings, y0, sample mask, lit cut, orientation grid, pruning, occupancy fit, local search,
  populations); before judging an anri stage file or the map it holds; before rendering
  a phantom or an indexed map back into detector frames; before making a phantom (anri.phantom) to test a method;
  and before comparing anri's maps with ImageD11's. Not for changing anri's own code: that is the repository's
  AGENTS.md.
license: GPL-3.0
compatibility: >-
  Requires the anri package (JAX, h5py; ImageD11 to read and write ImageD11 datasets). Runs on CPU, or on a GPU when
  JAX has one. This file and its references ship inside the package; the manual is at https://jadball.github.io/anri/
---

# Indexing, calibrating and simulating scanning-3DXRD data with anri

anri indexes a scanning-3DXRD dataset into orientation populations per voxel, calibrates the geometry on it, and
renders a map back into detector frames. All three run as **stages**: each reads files, writes one HDF5 file, and records in it how it was made. The files
are the record of what you did. This file is the order of work, and the checks to make before you believe a number.

## Routing

| When | Load |
|---|---|
| choosing an indexer option, or reading what a stage wrote | [index](references/index.md) |
| simulating a dataset, rendering a map or phantom, calling `anri.fwd` | [simulate](references/simulate.md) |
| calibrating the geometry: powder rings, then the indexed scan | [calib](references/calib.md) |
| making a phantom to test a method (`anri.phantom`), and judging a method on it | [phantom](references/phantom.md) |
| what a peak, a population or a spread can and cannot tell you | [physics](references/physics.md) |

Read options and signatures from the installed code, never from memory: `python -m anri.index <stage> -h`,
`python -m anri.calib <stage> -h`, `python -m anri.simulate <stage> -h`, and the stage functions' docstrings
(`anri.index.stages.stage_fit` and the others). The manual's Guides and Theory sections explain each choice at length.

---

## 1. Before you run anything

1. **Ask the human what the data files do not record**: the focusing optics (Al CRLs or Si nanolenses), the energy,
   any overfocusing, the beam's FWHM across dty against the dty step, the station (NSCOPE or TDXRD), the
   segmentation cut, and the monitor counter. They set `--beam`, `--censor` and `--monitor`, and what a peak width
   means. Never guess them.
2. **Never assume the microstructure.** Maps can come from phantoms, simulations or experiments: sharp boundaries,
   twins, deformation cells, several phases. If a step you plan relies on smoothness or on one orientation per
   grain, ask first.
3. **Estimate memory and time before a run, and check the machine** (`free -g`, `nvidia-smi`). Machines are shared,
   and running out of memory can force a restart. The histogram stage logs `memory: data … GB (x ~4 in MLEM)`, the
   fit `occupancies … voxels x … candidates = … GB`.
4. **Start small.** `--check` (histogram, powder) resolves the paths and parameters, logs them and writes nothing.
   `--gridstep 2` (or more) sums dty rows into larger voxels; fewer `--rings` and a coarse `--grid` cut the cost
   further. Go to full size once the small run is understood.
5. **Run under `timeout`, pin CPU runs, and log to a file**: `timeout 1800 taskset -c 0-15 python -m anri.index …
   --n-cpu 16 > run.log 2>&1`. Every message carries its time. Near-zero CPU or GPU use, or one busy core, usually
   means XLA is compiling.
6. **In your own scripts, call `anri.utils.setup()` before anything touches JAX.** It stops JAX taking 75% of the
   GPU at start-up.

## 2. The indexer: stages in order, choices at decision points

```
python -m anri.index histogram <analysisroot> <sample> <dataset>   # -> <sample>_<dataset>_histogram.h5
python -m anri.index search <sample>_<dataset>_histogram.h5         # -> ..._search.h5
python -m anri.index filter <sample>_<dataset>_search.h5            # -> ..._filter.h5
python -m anri.index fit <sample>_<dataset>_filter.h5               # -> ..._fit.h5
python -m anri.index localsearch <sample>_<dataset>_fit.h5          # -> ..._localsearch.h5
python -m anri.index populations <sample>_<dataset>_localsearch.h5  # -> ..._populations.h5, _entries.npz, _tmap.h5
```

The dataset is ImageD11's layout: `{analysisroot}/{sample}/{sample}_{dataset}/{sample}_{dataset}_dataset.h5`, with
the geometry and phase from its `pars.json` (`--parfile`, `--phase`). `run` does all six in one go, once the options
are known.

If the geometry is only the motors' first guess, calibrate first ([calib](references/calib.md)): fit the powder
rings on the dataset's own pixels (`python -m anri.calib rings`), index with the `pars.json` they give, fit the
geometry on the fit stage's file (`python -m anri.calib fit`), and index again with the calibrated `pars.json`.

7. **An option is chosen by looking at the output of the stage before it.** y0 and the sample mask (fit options)
   are read off the histogram; the lit cut, eta cut and grid (search) off the histogram's lit map; the pruning cuts
   (filter) off the search's completeness; the occupied and population cuts (populations) off the fit.
   [index](references/index.md) gives, for each option, the number to look at and the function that computes it.
   Decide from those numbers and say which ones you used.
8. **Look at the plots too.** Most diagnostics have a plot in `anri.index.plots` (`plot_x(fig, diag)` draws it; save
   it with `fig.savefig` and read the PNG). The pictures show things a summary number hides: a ring that wobbles in
   eta, a flux step between rows, a doubled sample edge from a wrong y0.
9. **Variants are new files.** Re-run a stage with other options and `--out`; never overwrite a source (stages
   refuse to). Each file records the id of the file it was made from, so a remade upstream file is refused
   downstream: remake the later stages as well.

## 3. Before you believe a file

10. **Read how it was made, and its log**: `python -m anri.index show <file> --log` (`anri.io.provenance.show(path,
    log=True)`, or `read_provenance(path)` for a dict). It covers every stage back to the dataset. Look for lines
    saying something was assumed or not modelled: `detector mask: none`, `absorption: not modelled`,
    `spatial correction: none`, `saturated`, `exposure per (row, omega bin) uneven`, `dty moves within the rows`.
    Each changes what the map can be trusted for, and each belongs in your report.
11. **Measured against fitted intensity per dty row** (logged by fit and localsearch; `row_ratio` in the file,
    `row_ratio_rms` in its results). Steps from row to row mean the flux changed between the rows' scans: use
    `--monitor`. A smooth trend with radius points to the model: absorption, or a mask of the wrong shape.
12. **Convergence.** Every MLEM iteration is logged with its deviance (`not taken`: SQUAREM's extrapolated step was
    rejected, which is normal). A flat deviance does not mean thin features have converged: twins one or two voxels
    thick keep changing long after. `anri.index.diagnostics.convergence_recorder` measures how much occupancy still
    moves.
13. **Enough candidates?** If each voxel's last candidates still hold a noticeable share of its occupancy
    (`candidate_shares`), raise `--cand`. If the local search's log shows nearly every slot used, raise
    `--per-voxel`.
14. **Low completeness inside the sample** (`comp_pop`) means a wrong orientation, or a voxel the data barely see.

## 4. What you may report

15. **Report a mean and a spread for what the instrument cannot separate.** Populations closer than the resolution
    are one population. The populations stage's `spread` includes the orientation grid's own spacing: it is an
    upper bound, not a measured spread.
16. **Second populations are real at boundaries and in lamellae**; a scatter of weak ones all over the grains is the
    model's noise floor. Put `--min-frac` above that floor, and say where you put it.
17. **A good-looking map is not a check.** To back a claim about a method, index a phantom whose truth you know:
    [simulate](references/simulate.md) renders one with cells and twins, and `python -m anri.index` reads the
    dataset back; [phantom](references/phantom.md) says how to build one for the microstructure at hand.
    Orientation shifts shrunk towards each grain's mean are path averaging, the failure to look for.
18. **An input is not a result.** A value a stage took as a default (y0 from the DataSet, the automatic grid step)
    is an assumption you made. Say so.
19. **Report** the files, the command that remakes them (`anri.index.stages.run_command(path)` gives a `run` command
    with every option), the options you chose and the numbers that chose them, and every log line from rule 10.

---

## 5. A worked start

A first, cheap pass on a dataset, then the numbers behind the fit's choices. Paths are placeholders.

```bash
timeout 900 python -m anri.index histogram /data/analysis S1 slice0 --rings 6 --gridstep 2 --check
timeout 3600 python -m anri.index histogram /data/analysis S1 slice0 --rings 6 --gridstep 2 > histogram.log 2>&1
python -m anri.index show S1_slice0_histogram.h5 --log
```

```python
import anri.utils

anri.utils.setup()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import anri.index as ix

d = ix.stages.load_histogram("S1_slice0_histogram.h5")
edges = ix.diagnostics.y0_from_edges(d["H"], d["n_rings"], d["scan"])  # rule 7: y0 for the fit
rows = ix.diagnostics.row_profile(d["H"], d["n_rings"], d["scan"])  # rule 11: flux steps -> --monitor
print("y0 from the sample's edges:", edges["y0"], "DataSet's:", d["prov"]["resolved"].get("y0_dataset"))
fig = plt.figure(figsize=(13, 4))
ix.plots.plot_rows(fig, rows)
fig.savefig("rows.png")  # look at it
```

Then `search`, `filter`, `fit --y0 <value>` and onwards, each checked as in [index](references/index.md).
