"""Set up JAX for the machine it is running on, and check that it worked.

Call :func:`configure` before any JAX operation has run (ideally before ``import jax``), then
:func:`check` once JAX is up::

    import anri.backend
    anri.backend.configure()      # CPU: one XLA device per core. GPU: leave JAX alone.
    import jax
    info = anri.backend.check()   # prints what JAX actually got, warns if it is wrong

Why this matters: XLA:CPU runs some operations (notably scatter-add, which the renderer is built
around) on a single thread. Anri gets multi-core CPU throughput by giving JAX one virtual device per
core and sharding work across them. That device count is fixed when JAX initialises its backend and
cannot be changed afterwards.

This module does not import JAX at module level, so importing it never initialises a backend.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import warnings
from typing import NamedTuple

__all__ = ["Backend", "check", "configure"]

_CUDA_PLUGINS = ("jax_cuda13_plugin", "jax_cuda12_plugin")
_JAX_DEFAULT_GPU_MEM_FRACTION = 0.75


class Backend(NamedTuple):
    """What JAX is running on. Returned by :func:`check`."""

    kind: str
    """``"gpu"`` or ``"cpu"``."""
    n_devices: int
    """``jax.device_count()``."""
    n_cores: int
    """CPU cores this process may run on."""
    bytes_per_device: int
    """Estimated memory one device can use for work. GPU: total x JAX's memory fraction.
    CPU: available host memory / ``n_devices``. An estimate for sizing batches, not a guarantee."""
    warnings: tuple[str, ...]
    """Everything :func:`check` found wrong. Empty if the setup looks right."""


def _n_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS, Windows
        return os.cpu_count() or 1


def _cuda_hidden() -> bool:
    return os.environ.get("CUDA_VISIBLE_DEVICES", None) in ("", "-1")


def _nvidia_memory_bytes() -> list[int]:
    """Total memory of each visible NVIDIA GPU, via nvidia-smi. Empty if none or unavailable."""
    if _cuda_hidden() or shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    mem = [int(line.strip()) * 2**20 for line in out.splitlines() if line.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        idx = [int(i) for i in visible.split(",") if i.strip().isdigit()]
        mem = [mem[i] for i in idx if i < len(mem)]
    return mem


def _has_cuda_plugin() -> bool:
    return any(importlib.util.find_spec(p) is not None for p in _CUDA_PLUGINS)


def _host_available_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def configure(device: str = "auto", n_cpu_devices: int | None = None) -> str:
    """Prepare JAX for this machine. Must run before JAX executes its first operation.

    Parameters
    ----------
    device
        ``"auto"`` (GPU if an NVIDIA GPU is visible and JAX's CUDA plugin is installed, else CPU),
        ``"gpu"`` or ``"cpu"``. ``"cpu"`` on a GPU machine hides the GPU from JAX.
    n_cpu_devices
        Number of XLA CPU devices. Defaults to the number of cores this process may use.
        Ignored on GPU.

    Returns
    -------
    kind: str
        ``"gpu"`` or ``"cpu"``: what was requested of JAX.

    Notes
    -----
    Settings the user has already made through environment variables are respected, except that an
    explicit ``device``/``n_cpu_devices`` argument always wins.

    GPU memory is left at JAX's defaults (preallocate 75%). Preallocation is what you want for a
    long render: the batch sizer reads the same fraction back in :func:`check`. Set
    ``XLA_PYTHON_CLIENT_MEM_FRACTION`` yourself to change it.

    If JAX has already initialised its backend this cannot change anything; it warns instead, and
    :func:`check` will say what JAX actually has.
    """
    if device not in ("auto", "gpu", "cpu"):
        msg = f"device must be 'auto', 'gpu' or 'cpu', got {device!r}"
        raise ValueError(msg)

    gpus = _nvidia_memory_bytes()
    plugin = _has_cuda_plugin()
    if device == "auto":
        device = "gpu" if (gpus and plugin) else "cpu"
        if gpus and not plugin:
            warnings.warn(
                f"{len(gpus)} NVIDIA GPU(s) visible but no JAX CUDA plugin is installed; "
                "falling back to CPU. Install jax[cuda12] or jax[cuda13] to use the GPU.",
                stacklevel=2,
            )
    elif device == "gpu" and not (gpus and plugin):
        msg = f"device='gpu' requested but found {len(gpus)} visible GPU(s), CUDA plugin installed: {plugin}"
        raise RuntimeError(msg)

    n = int(n_cpu_devices) if n_cpu_devices is not None else _n_cores()
    if n < 1:
        msg = f"n_cpu_devices must be >= 1, got {n}"
        raise ValueError(msg)

    if "jax" not in sys.modules:
        if device == "cpu":
            if n_cpu_devices is not None or "JAX_NUM_CPU_DEVICES" not in os.environ:
                os.environ["JAX_NUM_CPU_DEVICES"] = str(n)
            if gpus or plugin:
                os.environ["JAX_PLATFORMS"] = "cpu"
        return device

    # jax is imported. Its backend may still be uninitialised, in which case config updates work.
    import jax

    if device == "cpu":
        try:
            jax.config.update("jax_num_cpu_devices", n)
        except RuntimeError:
            warnings.warn(
                "anri.backend.configure() was called after JAX initialised its backend, so the "
                "CPU device count could not be set. Call it before the first JAX operation.",
                stacklevel=2,
            )
            return device
        if gpus or plugin:
            jax.config.update("jax_platforms", "cpu")
    return device


def check(verbose: bool = True) -> Backend:
    """Report what JAX is running on and warn about setups that will be slow.

    This initialises JAX's backend if nothing has yet, so call :func:`configure` first.

    Parameters
    ----------
    verbose
        Print a one-line summary and any warnings.

    Returns
    -------
    backend: Backend
    """
    import jax

    devices = jax.devices()
    platform = devices[0].platform
    kind = "gpu" if platform in ("cuda", "gpu", "rocm") else "cpu"
    n_dev = len(devices)
    cores = _n_cores()
    gpus = _nvidia_memory_bytes()
    problems = []

    if kind == "gpu":
        frac = float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", _JAX_DEFAULT_GPU_MEM_FRACTION))
        per_dev = int(min(gpus) * frac) if gpus else 0
        if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() in ("0", "false"):
            problems.append(
                "XLA_PYTHON_CLIENT_PREALLOCATE is off: memory is allocated on demand, which "
                "fragments over a long render. Leave it on unless the GPU is shared."
            )
        if not gpus:
            problems.append("could not query GPU memory (nvidia-smi); batch sizes cannot be estimated")
    else:
        per_dev = _host_available_bytes() // max(n_dev, 1)
        if gpus and _has_cuda_plugin() and os.environ.get("JAX_PLATFORMS") != "cpu":
            problems.append(f"{len(gpus)} GPU(s) and the CUDA plugin are present but JAX is on CPU")
        elif gpus and not _has_cuda_plugin():
            problems.append(f"{len(gpus)} GPU(s) visible but no JAX CUDA plugin installed, running on CPU")
        if n_dev == 1 and cores > 1:
            problems.append(
                f"1 CPU device on {cores} cores: scatter-heavy work will use one core. Call "
                "anri.backend.configure() before the first JAX operation."
            )
        if n_dev > cores:
            problems.append(f"{n_dev} CPU devices on {cores} cores is oversubscribed")

    if jax.config.jax_enable_x64:
        problems.append("jax_enable_x64 is on: anri's renderer works in float32/int32; x64 doubles memory")

    info = Backend(kind, n_dev, cores, per_dev, tuple(problems))
    if verbose:
        print(
            f"anri backend: jax {jax.__version__}, {kind}, {n_dev} device(s), {cores} core(s), "
            f"~{per_dev / 2**30:.1f} GiB per device"
        )
        for p in problems:
            print(f"  WARNING: {p}")
    return info
