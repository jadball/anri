"""Measurements the anri renderer design depends on. Run on each machine and send back the output.

    python benchmarks/render_bench.py all              # everything that applies to this machine
    python benchmarks/render_bench.py scaling          # one section
    python benchmarks/render_bench.py all --quick      # small sizes, to check it runs

Sections
--------
kernel   contributions/s of the peak kernel against window size and n_sub (no scatter)
extract  per-frame cost of summing + thresholding a batch of frames: sparse (owner trick) vs dense
scaling  CPU only: does throughput grow with the number of XLA CPU devices? Kernel + scatter-add, and
         covariance propagation, each under shard_map. Runs one subprocess per device count.
fwd      anri forward model: centroids/s and covariances/s on one device
sort     argsort throughput (int32 keys), used to group contributions by frame

Everything is synthetic; nothing is read from disk. Results also go to render_bench_<host>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

WORKER_FLAG = "--scaling-worker"


def _setup(n_cpu_devices: int | None = None) -> None:
    import anri.backend

    anri.backend.configure(n_cpu_devices=n_cpu_devices)


def _best_time(f, *args, repeat: int = 3) -> float:  # noqa: ANN001
    import numpy as np

    np.asarray(f(*args))  # compile + warm
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        np.asarray(f(*args))
        best = min(best, time.perf_counter() - t0)
    return best


def _synthetic(n: int, det: int = 2048, seed: int = 0):  # noqa: ANN202
    """Physically shaped 4D peaks in grid units, each paired with a frame that cuts it."""
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(seed)
    sc, fc = rng.uniform(20, det - 20, n), rng.uniform(20, det - 20, n)
    eta = np.arctan2(fc - det / 2, sc - det / 2)
    ss = rng.uniform(0.1, 1.5, n)  # wavelength streak, px
    sol = rng.uniform(0.2, 2.0, n)  # omega spread tied to it, frames
    sod = rng.uniform(0.0, 1.0, n)  # divergence omega spread, frames
    slope = rng.uniform(-1.3, 1.3, n)  # dty rows per omega frame
    sy = rng.uniform(0.2, 0.5, n)  # beam / voxel, rows
    J = np.stack([ss * np.cos(eta), ss * np.sin(eta), sol, slope * sol], -1)
    C = J[:, :, None] * J[:, None, :]
    C[:, 0, 0] += 1 / 12
    C[:, 1, 1] += 1 / 12
    C[:, 2, 2] += sod**2
    C[:, 3, 3] += sy**2 + (slope * sod) ** 2
    C[:, 2, 3] += slope * sod**2
    C[:, 3, 2] += slope * sod**2
    om, dt = rng.uniform(100, 7000, n), rng.uniform(100, 2900, n)
    mu = np.stack([sc, fc, om, dt], -1)
    fo = np.round(om) + rng.integers(-1, 2, n)
    fy = np.round(dt) + rng.integers(-1, 2, n)
    lo = np.stack([fo - 0.5, fy - 0.5], -1)
    hi = lo + 1.0
    amp = rng.exponential(100.0, n)
    f32 = lambda a: jnp.asarray(a, jnp.float32)  # noqa: E731
    return f32(mu), f32(C), f32(amp), f32(lo), f32(hi)


SHAPE = (2048, 2048, 7200, 3000)
PERIODS = (0.0, 0.0, 7200.0, 0.0)


def _kernel_fn(win: int, n_sub: int):  # noqa: ANN202
    import jax

    from anri.render._impl.kernel import splat_peaks

    @jax.jit
    def f(mu, cov, amp, lo, hi):  # noqa: ANN001, ANN202
        blocks, starts = splat_peaks(mu, cov, amp, lo, hi, SHAPE, (0, 1), (win, win), n_sub, PERIODS)
        return blocks.sum() + starts.sum()

    return f


def bench_kernel(quick: bool, out: dict) -> None:
    import jax

    sizes = [2**12] if quick else ([2**16, 2**18, 2**20] if _is_gpu() else [2**14, 2**16])
    print("\n== kernel: contributions per second (one device, no scatter) ==")
    print(f"{'batch':>9s} {'win':>4s} {'n_sub':>5s} {'contrib/s':>12s} {'window px/s':>12s}")
    res = []
    for n in sizes:
        data = _synthetic(n)
        for win in (7, 9, 13):
            for n_sub in (0, 1, 2, 4):
                t = _best_time(_kernel_fn(win, n_sub), *data)
                res.append(dict(batch=n, win=win, n_sub=n_sub, contrib_per_s=n / t))
                print(f"{n:9d} {win:4d} {n_sub:5d} {n / t:12.3g} {n * win * win / t:12.3g}")
    out["kernel"] = dict(device=str(jax.devices()[0]), rows=res)


def bench_extract(quick: bool, out: dict) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from anri.render._impl.sparse import accumulate_extract, new_canvases

    det = 2048
    npix = det * det
    per_frame = 1000
    win = 9
    slots_list = [4] if quick else ([16, 64, 256] if _is_gpu() else [4, 16, 64])
    print(f"\n== extract: {per_frame} peaks x {win}x{win} px per frame, {det}x{det} detector ==")
    print(f"{'frames':>7s} {'sparse ms/frame':>16s} {'dense ms/frame':>15s}")
    rng = np.random.default_rng(0)
    res = []
    for slots in slots_list:
        n = slots * per_frame
        slot = np.repeat(np.arange(slots), per_frame)
        r = rng.integers(0, det - win, n)
        c = rng.integers(0, det - win, n)
        dr, dc = np.meshgrid(np.arange(win), np.arange(win), indexing="ij")
        keys = (slot[:, None] * npix + (r[:, None] + dr.ravel()) * det + c[:, None] + dc.ravel()).reshape(-1)
        g = np.exp(-0.5 * ((dr - 4) ** 2 + (dc - 4) ** 2) / 1.5**2).ravel()
        vals = (rng.exponential(200.0, n)[:, None] * g / g.sum()).reshape(-1)
        keys = jnp.asarray(keys, jnp.int32)
        vals = jnp.asarray(vals, jnp.float32)
        max_nnz = slots * per_frame * 30
        state = {"c": new_canvases(slots, npix)}

        def sparse(k, v):  # noqa: ANN001, ANN202
            cv, ow = state["c"]
            cv, ow, ok, ov, nnz = accumulate_extract(cv, ow, k, v, 2.0, slots, npix, max_nnz)
            state["c"] = (cv, ow)
            return nnz + ov.sum()

        @jax.jit
        def dense(k, v):  # noqa: ANN001, ANN202
            canvas = jnp.zeros((slots + 1) * npix, jnp.float32).at[k].add(v)
            keep = canvas[: slots * npix] > 2.0
            (sel,) = jnp.nonzero(keep, size=max_nnz, fill_value=0)
            return keep.sum() + canvas[sel].sum()

        ts = _best_time(sparse, keys, vals)
        td = _best_time(dense, keys, vals)
        res.append(dict(frames=slots, sparse_ms_per_frame=1e3 * ts / slots, dense_ms_per_frame=1e3 * td / slots))
        print(f"{slots:7d} {1e3 * ts / slots:16.3f} {1e3 * td / slots:15.3f}")
        del state
    out["extract"] = dict(device=str(jax.devices()[0]), rows=res)


def _scaling_worker(n_dev: int, per_dev: int) -> None:
    """Runs in a subprocess with JAX_NUM_CPU_DEVICES already set."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, PartitionSpec

    try:
        from jax import shard_map
    except ImportError:  # older jax
        from jax.experimental.shard_map import shard_map

    from anri.render._impl.kernel import flat_index, splat_peaks

    assert jax.device_count() == n_dev, (jax.device_count(), n_dev)
    mesh = Mesh(np.array(jax.devices()), ("d",))
    spec = PartitionSpec("d")
    win = 9

    def render_local(mu, cov, amp, lo, hi):  # noqa: ANN001, ANN202
        blocks, starts = splat_peaks(mu, cov, amp, lo, hi, SHAPE, (0, 1), (win, win), 2, PERIODS)
        idx = flat_index(starts, (win, win), (2048, 2048), (False, False))
        canvas = jnp.zeros(2048 * 2048, jnp.float32).at[idx.reshape(-1)].add(blocks.reshape(-1))
        return canvas.sum(keepdims=True)

    def scatter_local(keys, vals):  # noqa: ANN001, ANN202
        return jnp.zeros(2048 * 2048, jnp.float32).at[keys].add(vals).sum(keepdims=True)

    try:
        smap = lambda f: jax.jit(shard_map(f, mesh=mesh, in_specs=spec, out_specs=spec))  # noqa: E731
        render = smap(render_local)
        scatter = smap(scatter_local)
    except TypeError:
        smap = lambda f: jax.jit(shard_map(f, mesh=mesh, in_specs=spec, out_specs=spec, check_rep=False))  # noqa: E731
        render = smap(render_local)
        scatter = smap(scatter_local)

    n = n_dev * per_dev
    data = _synthetic(n)
    t_render = _best_time(render, *data)

    rng = np.random.default_rng(0)
    m = n * win * win
    keys = jnp.asarray(rng.integers(0, 2048 * 2048, m), jnp.int32)
    vals = jnp.asarray(rng.random(m), jnp.float32)
    t_scatter = _best_time(scatter, keys, vals)

    t_cov = _fwd_cov_sharded(mesh, spec, smap, per_dev_vox=max(1, per_dev // 64))
    print(json.dumps(dict(n_dev=n_dev, render_contrib_per_s=n / t_render, scatter_per_s=m / t_scatter,
                          fwd_cov_per_s=t_cov)))


def _fwd_setup(n_vox: int, n_hkl: int):  # noqa: ANN202
    import jax.numpy as jnp
    import numpy as np
    from scipy.spatial.transform import Rotation

    import anri.fwd
    import anri.geom

    rng = np.random.default_rng(0)
    a = 5.43
    ubi = np.asarray(a * Rotation.random(n_vox, random_state=1).as_matrix().transpose(0, 2, 1), np.float32)
    pos = np.asarray(np.column_stack([rng.uniform(-500, 500, (n_vox, 2)), np.zeros(n_vox)]), np.float32)
    hkl = rng.integers(-8, 9, (n_hkl, 3)).astype(np.float32)
    hkl[np.all(hkl == 0, axis=1)] = 1
    det = anri.geom.detector_transforms(1024.0, 75.0, 0.0, 1024.0, 75.0, 0.0, 0.0, 120e3, 1, 0, 0, 1)
    sc_lab, fc_lab, norm_lab = anri.geom.detector_basis_vectors_lab(*det)
    cov_in = anri.fwd.get_cov_in(jnp.array([1.0, 1.0, 1.0]), 0.285 * 5e-4, 5e-5, 5e-5)
    geom = (0.285, jnp.array([1.0, 0.0, 0.0]), 0.0, 0.0, 0.0, 0.0, 0.0)
    f32 = lambda x: jnp.asarray(x, jnp.float32)  # noqa: E731
    dets = tuple(f32(v) for v in (sc_lab, fc_lab, norm_lab))
    return f32(ubi), f32(pos), f32(hkl), geom, dets, f32(cov_in)


def _fwd_cov_sharded(mesh, spec, smap, per_dev_vox: int) -> float:  # noqa: ANN001
    import jax
    import jax.numpy as jnp

    import anri.fwd

    n_dev = len(mesh.devices)
    n_hkl = 64
    ubi, pos, hkl, geom, dets, cov_in = _fwd_setup(n_dev * per_dev_vox, n_hkl)
    wl, kin, ky, kz, wedge, chi, y0 = geom

    def local(u, p):  # noqa: ANN001, ANN202
        c = anri.fwd.propagate_cov_scan_all_both(u, p, hkl, wl, kin, ky, kz, wedge, chi, y0, *dets, cov_in)
        return jnp.sum(c, keepdims=True)[None, ...].reshape(1)

    f = smap(local)
    t = _best_time(f, ubi, pos)
    return n_dev * per_dev_vox * n_hkl * 2 / t


def bench_scaling(quick: bool, out: dict) -> None:
    import anri.backend

    if _is_gpu():
        print("\n== scaling: skipped (GPU machine) ==")
        return
    cores = anri.backend._n_cores()
    counts = [1, 2] if quick else sorted({1, 2, 4, 8, 16, 32, cores} & set(range(1, cores + 1)))
    per_dev = 2**10 if quick else 2**14
    variants = {
        "default": {},
        "1 thread per op": {"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"},
    }
    print(f"\n== scaling: weak scaling over XLA CPU devices ({cores} cores), {per_dev} contributions per device ==")
    res = []
    for vname, extra in variants.items():
        print(f"-- {vname}")
        print(f"{'devices':>8s} {'render/s':>10s} {'speedup':>8s} {'scatter/s':>10s} {'speedup':>8s} {'fwd cov/s':>10s} {'speedup':>8s}")
        base = None
        for d in counts:
            env = dict(os.environ, JAX_NUM_CPU_DEVICES=str(d), JAX_PLATFORMS="cpu", **extra)
            cmd = [sys.executable, os.path.abspath(__file__), WORKER_FLAG, str(d), str(per_dev)]
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
            line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            if p.returncode or not line:
                print(f"{d:8d}  FAILED:\n{p.stderr[-2000:]}")
                continue
            r = json.loads(line[-1])
            base = base or r
            r["variant"] = vname
            res.append(r)
            print(f"{d:8d} {r['render_contrib_per_s']:10.3g} {r['render_contrib_per_s'] / base['render_contrib_per_s']:8.2f}"
                  f" {r['scatter_per_s']:10.3g} {r['scatter_per_s'] / base['scatter_per_s']:8.2f}"
                  f" {r['fwd_cov_per_s']:10.3g} {r['fwd_cov_per_s'] / base['fwd_cov_per_s']:8.2f}")
    out["scaling"] = dict(cores=cores, rows=res)


def bench_fwd(quick: bool, out: dict) -> None:
    import anri.fwd

    n_vox, n_hkl = (256, 64) if quick else ((2**14, 256) if _is_gpu() else (2**11, 128))
    ubi, pos, hkl, geom, dets, cov_in = _fwd_setup(n_vox, n_hkl)
    wl, kin, ky, kz, wedge, chi, y0 = geom
    n = n_vox * n_hkl * 2

    def cen():  # noqa: ANN202
        c, v = anri.fwd.get_centroid_scan_all_both(ubi, pos, hkl, wl, kin, ky, kz, wedge, chi, y0, *dets)
        return c.sum() + v.sum()

    def cov():  # noqa: ANN202
        return anri.fwd.propagate_cov_scan_all_both(ubi, pos, hkl, wl, kin, ky, kz, wedge, chi, y0, *dets, cov_in).sum()

    tc, tv = _best_time(cen), _best_time(cov)
    print(f"\n== fwd: {n_vox} voxels x {n_hkl} hkls x 2 branches = {n} peaks, float32, one device ==")
    print(f"centroids   {n / tc:10.3g} peaks/s")
    print(f"covariances {n / tv:10.3g} peaks/s   ({tv / tc:.1f}x a centroid)")
    out["fwd"] = dict(peaks=n, centroid_per_s=n / tc, cov_per_s=n / tv)


def bench_sort(quick: bool, out: dict) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np

    sizes = [2**20] if quick else ([2**22, 2**24, 2**26] if _is_gpu() else [2**20, 2**22, 2**24])
    print("\n== sort: argsort of int32 keys, one device ==")
    res = []
    f = jax.jit(lambda k: jnp.argsort(k).sum())
    for n in sizes:
        k = jnp.asarray(np.random.default_rng(0).integers(0, 2**31 - 1, n), jnp.int32)
        t = _best_time(f, k)
        res.append(dict(n=n, keys_per_s=n / t))
        print(f"{n:12d} {n / t:12.3g} keys/s")
    out["sort"] = res


_GPU = None


def _is_gpu() -> bool:
    global _GPU  # noqa: PLW0603
    if _GPU is None:
        import jax

        _GPU = jax.devices()[0].platform != "cpu"
    return _GPU


SECTIONS = {"kernel": bench_kernel, "extract": bench_extract, "scaling": bench_scaling, "fwd": bench_fwd, "sort": bench_sort}


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == WORKER_FLAG:
        _scaling_worker(int(sys.argv[2]), int(sys.argv[3]))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("section", choices=["all", *SECTIONS])
    ap.add_argument("--quick", action="store_true", help="small sizes, to check the script runs")
    args = ap.parse_args()

    # the main process renders on one device; scaling spawns its own
    _setup(n_cpu_devices=1)
    import anri.backend

    info = anri.backend.check()
    out = dict(host=socket.gethostname(), backend=info._asdict(), quick=args.quick)
    for name, fn in SECTIONS.items():
        if args.section in ("all", name):
            fn(args.quick, out)
    path = f"render_bench_{socket.gethostname()}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\nresults written to {path}")


if __name__ == "__main__":
    main()
