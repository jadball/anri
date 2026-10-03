# AGENTS.md

Ground rules for AI coding agents working on Anri. See `README.md` for what Anri is and how to install it.

## Physics: ask, don't assume

Never make assumptions about the grain maps Anri will see. They can come from phantoms, simulations or experiments, and may be:

- not locally smooth (sharp boundaries, twins, large intragranular gradients)
- multi-phase
- not constant in density

**If a method depends on an assumption about the physics, ask the maintainer before relying on it.**

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
- **Don't let JAX take the whole GPU.** Call `anri.backend.setup()` at the top of scripts. It sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX doesn't grab 75% of the GPU at start-up. Set `XLA_PYTHON_CLIENT_MEM_FRACTION` too if needed.
- **Don't hog the CPU.** Don't use every core: pin CPU runs with `taskset` (e.g. `taskset -c 0-15`).
- **Clean up.** Don't leave long-running or idle processes holding memory.

## Keep it simple (no "Claudish" code)

No bloated frameworks, layers of abstraction or complicated APIs that the maintainer can't follow.
Earlier AI-written attempts were removed for exactly this reason: they were hard to understand, and slow.

- Prefer small, plain functions in the style of the existing `anri/` modules.
- Keep code functional: pure functions on arrays, no OOP. The classes in `anri.crystal` are an existing exception and may be refactored to match.
- Build incrementally and explain design choices.
- Check in before adding new abstractions or API surface.
- Going slower with better understanding beats a fast, opaque result.

## Git

- **Never push without asking the maintainer first.** Committing locally is fine.
- Keep commits small and focused, one topic each.
- Don't commit notebook metadata churn (kernel name, Python version, widget state) or files written by running the docs notebooks.

## CI and compatibility

CI (`.github/workflows/main.yml`) tests Python 3.9 and 3.14 on Linux, Windows and macOS (Intel and ARM). It runs `ruff check .` and `ty check .` with the **latest** ruff and ty, and has no GPU. Each job times out after 15 minutes.

- **Python 3.9+.** No `match`, `zip(..., strict=True)`, parenthesised context managers, or `X | Y` types evaluated at runtime (fine in annotations with `from __future__ import annotations`).
- **Old JAX and NumPy.** Python 3.9 gets JAX 0.4.30 and NumPy 1.26. So:
  - no `np.trapezoid` (NumPy 2+);
  - `jax.shard_map` needs a fallback to `jax.experimental.shard_map`;
  - check that any newer JAX API exists in 0.4.30 before using it.
- **Cross-platform.** Avoid Linux-only calls (e.g. `os.sched_getaffinity`). Use `os.path`/`tempfile` for paths, not hard-coded `/tmp`.
- **Lint the whole repo.** Run `ruff check .` from the repo root, not just `anri tests`: root files like `conftest.py` are checked too. Local ruff/ty may lag the versions CI installs.
- **Don't start JAX at import time.** No module-level `jnp` arrays or computations; use Python or NumPy constants. Starting the backend on import stops `anri.backend.setup()` from working.
- **Keep tests small.** They run on CPU-only runners, so keep each test to seconds and modest memory. Don't pad small problems up to production batch sizes.
- **PRs that touch `anri/**` must update `CHANGELOG.md`** (`pr_checks.yml`).
- **Testing on Python 3.9 locally:** build a throwaway env the way CI does: a conda-forge `python=3.9` env, then `unidep install -p <env> ".[dev]"`.

## Development

- Lint and format with `ruff`, type-check with `ty`, test with `pytest tests/`. All code outside `anri/sandbox` must pass.
- A pre-commit hook in `.githooks/` runs `ruff check .` like CI. Enable it once per clone with `git config core.hooksPath .githooks`.
- If `ty` can't find the environment (e.g. `VIRTUAL_ENV` points at a conda env), run `env -u VIRTUAL_ENV ty check --python <env>/bin/python`.
- Shared test data lives in `tests/data/`. Docs notebooks load it by relative path, e.g. `../../../tests/data/cif/Si.cif`.
- `anri/sandbox` is unstable and gitignored.
