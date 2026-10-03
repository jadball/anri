"""Where JAX runs: the GPU(s) if there are any, otherwise a few XLA CPU devices.

XLA:CPU already spreads large element-wise operations over several threads, but runs others (scatter, sort)
on one thread per device. Asking XLA for a few CPU devices and sharding work across them runs those in
parallel too. The number of devices is fixed when JAX starts its backend, so call :func:`setup` at the
top of a script or notebook, before any JAX computation::

    import anri.utils
    anri.utils.setup()

On a GPU machine the CPU device count is unused, and work is sharded over the GPUs instead.
"""

from __future__ import annotations

import os
import re

import jax
import numpy as np
from jax.sharding import Mesh


def setup(n_cpu: int = 4, preallocate: bool = False) -> None:
    """Configure JAX for this machine. Must run before JAX has computed anything.

    Parameters
    ----------
    n_cpu
        Number of XLA CPU devices. Each device also uses several threads, so this is not the number of
        cores used. Rendering one row of the quartz phantom on 16 cores took 9.0 s with 1 device,
        6.3 s with 4 and 8.2 s with 16: more devices than that compete for the same cores.
        Limit the cores themselves with ``taskset`` or a SLURM allocation.
    preallocate
        If ``False`` (default), JAX allocates GPU memory as needed instead of reserving 75% of it at start-up.
        Only applied if ``XLA_PYTHON_CLIENT_PREALLOCATE`` is not already set.
    """
    from jax._src import xla_bridge

    if xla_bridge.backends_are_initialized():
        msg = "JAX has already started; call anri.utils.setup() before any JAX computation"
        raise RuntimeError(msg)
    flags = re.sub(r"--xla_force_host_platform_device_count=\d+", "", os.environ.get("XLA_FLAGS", ""))
    os.environ["XLA_FLAGS"] = f"{flags} --xla_force_host_platform_device_count={n_cpu}".strip()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", str(preallocate).lower())


def mesh() -> Mesh:
    """Return a 1D mesh, axis ``"d"``, over the devices of the default backend (all GPUs, or all CPU devices)."""
    return Mesh(np.array(jax.devices()), ("d",))
