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

## Development

- Lint and format with `ruff`, type-check with `ty`, test with `pytest tests/`. All code outside `anri/sandbox` must pass.
- Shared test data lives in `tests/data/`. Docs notebooks load it by relative path, e.g. `../../../tests/data/cif/Si.cif`.
- `anri/sandbox` is unstable and gitignored.
