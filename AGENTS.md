# AGENTS.md

Ground rules for AI coding agents working on Anri. See `README.md` for what Anri is and how to install it.

## Physics: ask, don't assume

Never make assumptions about the grain maps Anri will see. They can come from phantoms, simulations or experiments, and may be:

- not locally smooth (sharp boundaries, twins, large intragranular gradients)
- multi-phase
- not constant in density

**If a method depends on an assumption about the physics, ask the maintainer before relying on it.**

### Microstructures, and why there are no smoothness priors

Anri targets everything from perfect crystals to heavily deformed metals. At the scales that matter (voxels of 0.1-1 µm, rays tens to hundreds of µm long):

- **Annealed grains and single crystals:** orientation constant to below the instrument resolution, with sharp boundaries.
- **Annealing twins** (Σ3 in 316L, Cu, Ni): sharp planar boundaries (60° about ⟨111⟩), and constant domains from ~100 nm to tens of µm.
- **Deformed FCC/BCC metals:** lattice curvature is carried by dislocations, which pattern into cells and walls. Orientation is piecewise nearly constant: cells of 0.5-2 µm, walls with 0.1-10° misorientation, accumulating like a random walk. Peaks are sharp sub-spots plus a diffuse cloud (Jakobsen et al., Science 2006), not smooth streaks.
- **Additively manufactured metals** (e.g. L-PBF 316L): melt pools, then columnar grains, then solidification cells of 0.3-1 µm with dislocation walls and ≲0.5° misorientations, plus degrees of drift along a column and large cell-scale residual stresses.
- **Lath martensite, bainite, Ti α laths:** a few discrete variants per voxel, related by an orientation relationship, each with 1-2° spread; sometimes two phases.
- **Deformation and nano-twins:** lamellae much thinner than a voxel, so a voxel holds two orientations with a volume fraction.
- **Genuinely smooth fields** (elastic bending, undulose extinction, elastic strain away from defects): the exception, not the rule.

Consequences for models:

- **A voxel's orientation distribution is a small mixture of populations** (cells, domains, variants), each nearly constant with a small intrinsic spread. Represent it with multiple map entries per voxel, each with its own density (volume fraction) and, where needed, its own spread.
- **No coupling between voxels:** no smoothness, interpolation, total variation or basis-function priors on orientation or strain. They suppress exactly the local variation Anri exists to measure, as ImageD11's path averaging does.
- **Report what the data cannot resolve:** populations closer than the instrument resolution cannot be separated, so report a mean and a spread rather than inventing structure.
- **Test on realistic phantoms:** with orientation spread along the rays (cell structures, twins, AM-like hierarchies), not uniform grains or linear gradients on a voxel grid. A uniform orientation per voxel turns a gradient into an unphysical ladder of sub-peaks.

### The instrument: ESRF ID11 focusing optics

Most data come from scanning 3DXRD at ID11. The beam is focused by one of these, or both together:

- **Si planar nanolenses:** two 1D lenses (one horizontal, one vertical). f ~ 10 cm, physical aperture 50 µm, effective aperture ~38 µm at 40-55 keV (Snigirev et al., Proc. SPIE 6705, 670506, 2007).
- **Al CRL boxes:** two 2D boxes, 96 m from a source of 60 µm (h) × 20 µm (v); lenses of R = 30 µm, 10 µm between apexes.

  | Energy | Lenses | f | Effective aperture |
  |---|---|---|---|
  | 43 keV | 102 | 50.4 cm | 118 µm |
  | 56 keV | 173 | 50.5 cm | 115 µm |
  | 70 keV | 275 (both boxes) | 49.6 cm | 105 µm |

- **Overfocusing:** a transfocator ~60 m from the source is sometimes set to overfocus, sending a divergent beam into the lenses. This enlarges the beam at the sample, by an amount chosen per experiment.

Consequences:

- **Convergence at the sample** is at most about effective aperture / f: ~0.2 mrad (0.012°) for the Al CRLs, ~0.4-0.5 mrad for the Si lenses, whatever comes in upstream.
- **Horizontal convergence acts as a spread in omega** of the same size at every eta (the renderer's `sig_ky`). An isotropic orientation spread instead widens omega as ~1 / |sin eta|.
- **Which optics, energy and overfocus a dataset used** is not in the data files: ask the maintainer. The beam size across dty depends on it.

## Use JAX

Anri code should use JAX wherever possible: `jit`, `vmap`, `lax.map`/`scan`, sharding.
Don't give up on JAX and wrap work in Python multiprocessing or threading instead.
Keep functions differentiable where it is reasonable to do so.

## Scale

Anri must scale from a laptop to a cluster.

- **Largest experimental datasets so far:** 3k × 3k voxels and 5 billion peaks. Design for this.
- **GPU:** sometimes available, not always. Code must run on CPU-only machines.
- **Memory:** we don't always have lots of RAM. Avoid holding everything in memory at once. Prefer chunking and batching, and streaming where needed.
- **Laptop target:** ImageD11 can analyse a small(ish) S3DXRD dataset (400 × 400 px) on a laptop with < 64 GB RAM. Anri should match that.
- **Big problems:** we have a SLURM cluster with multiple GPUs. JAX multi-device and multi-host (`jax.distributed`, sharding) is the preferred route there.

## Shared machines: be a good neighbour

Development happens on shared machines (e.g. an ESRF node with one L40S GPU and other users) and sometimes on a laptop.
**Running out of memory can force a full machine restart.**

- **Estimate first.** Before running anything, estimate peak host-RAM and GPU-memory footprint from the array shapes and dtypes. Start with a small problem, measure, then scale up.
- **Check usage.** Look at what others are using (`free -g`, `nvidia-smi`) before a big run.
- **Don't let JAX take the whole GPU.** Call `anri.utils.setup()` at the top of scripts. It sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX doesn't grab 75% of the GPU at start-up. Set `XLA_PYTHON_CLIENT_MEM_FRACTION` too if needed.
- **Don't hog the CPU.** Don't use every core: pin CPU runs with `taskset` (e.g. `taskset -c 0-15`).
- **Clean up.** Don't leave long-running or idle processes holding memory.

## Running jobs: small, timed, visible

The maintainer is often watching. A machine that sits idle while they wait is a failure.

- **Test small first.** Use a cut-down case (one dty row, a few hundred voxels) that compiles and runs in seconds, so you understand the answer quickly. Go to full size only once the small case is understood.
- **Estimate the run time before you start**, and say what it is. Count XLA compile time: it is single-threaded and often longer than the run.
- **Always set a timeout** a little above your estimate (e.g. `timeout 120 python ...`), so a hung or still-compiling job dies instead of blocking for 30 minutes.
- **Log, don't tail.** Write timestamped progress to a log file. Don't pipe output through `tail` or `grep` that buffers it until the end. Watch the log and report each step as it starts and finishes.
- **Watch the machine.** If CPU or GPU use is near zero, or one core is busy, find out why (usually compiling) and fix it. Don't just wait.

## Keep it simple (no "Claudish" code)

No bloated frameworks, layers of abstraction or complicated APIs that the maintainer can't follow.
Earlier AI-written attempts were removed for exactly this reason: they were hard to understand, and slow.

- Prefer small, plain functions in the style of the existing `anri/` modules.
- Keep code functional: pure functions on arrays, no OOP. The classes in `anri.crystal` are an existing exception and may be refactored to match.
- Build incrementally and explain design choices.
- Check in before adding new abstractions or API surface.
- Going slower with better understanding beats a fast, opaque result.

## Installing software

**Never install anything without asking the maintainer first:** no `pip`, `conda`/`mamba`, `npm` or downloaded binaries, not even into a temporary folder.
Packages are how malware gets in. Use what is already installed, or ask.

## Git

- **Never push without asking the maintainer first.** Committing locally is fine.
- Keep commits small and focused, one topic each.
- Don't commit notebook metadata churn (kernel name, Python version, widget state) or files written by running the docs notebooks.

## CI and compatibility

CI (`.github/workflows/main.yml`) tests Python 3.9 and 3.14 on Ubuntu for pushes and PRs. Release tags (`v*.*.*`) also test Linux, Windows and macOS (Intel and ARM). It runs `ruff check .` and `ty check .` with the **latest** ruff and ty, and has no GPU. Each job times out after 15 minutes (30 for release tags). The docs deploy from `main` only when every check passes.

- **Python 3.9+.** No `match`, `zip(..., strict=True)`, parenthesised context managers, or `X | Y` types evaluated at runtime (fine in annotations with `from __future__ import annotations`).
- **Old JAX and NumPy.** Python 3.9 gets JAX 0.4.30 and NumPy 1.26. So:
  - no `np.trapezoid` (NumPy 2+);
  - `jax.shard_map` needs a fallback to `jax.experimental.shard_map`;
  - check that any newer JAX API exists in 0.4.30 before using it.
- **Cross-platform.** Avoid Linux-only calls (e.g. `os.sched_getaffinity`). Use `os.path`/`tempfile` for paths, not hard-coded `/tmp`.
- **Lint the whole repo.** Run `ruff check .` from the repo root, not just `anri tests`: root files like `conftest.py` are checked too. Local ruff/ty may lag the versions CI installs.
- **Don't start JAX at import time.** No module-level `jnp` arrays or computations; use Python or NumPy constants. Starting the backend on import stops `anri.utils.setup()` from working.
- **Keep tests small.** They run on CPU-only runners, so keep each test to seconds and modest memory. Don't pad small problems up to production batch sizes.
- **PRs that touch `anri/**` must update `CHANGELOG.md`** (`pr_checks.yml`).
- **Testing on Python 3.9 locally:** build a throwaway env the way CI does: a conda-forge `python=3.9` env, then `unidep install -p <env> ".[dev]"`.

## Development

- Lint and format with `ruff`, type-check with `ty`, test with `pytest tests/`. All code outside `anri/sandbox` must pass.
- Hooks in `.githooks/` catch CI failures early: pre-commit runs `ruff check .`; pre-push builds the docs with `sphinx-build -W` (notebooks not executed, ~30 s). Enable them once per clone with `git config core.hooksPath .githooks`.
- If `ty` can't find the environment (e.g. `VIRTUAL_ENV` points at a conda env), run `env -u VIRTUAL_ENV ty check --python <env>/bin/python`.
- Shared test data lives in `tests/data/`. Docs notebooks load it by relative path, e.g. `../../../tests/data/cif/Si.cif`.
- `anri/sandbox` is unstable and gitignored.
