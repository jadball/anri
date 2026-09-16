"""Sum window pixels into frames and pull out the ones above threshold, in O(contributions).

A batch of frames is rendered into a persistent scratch canvas of ``n_slots`` frames. Thresholding
that canvas densely costs ``n_slots * n_pixels`` per batch no matter how few pixels were touched;
for a 2048x2048 detector and ~1000 peaks per frame that is ~50x the cost of rendering the peaks.
Instead only the touched pixels are read back and reset.

Duplicates (overlapping windows) are resolved without sorting: each window pixel scatter-maxes its
own position in the batch into an ``owner`` canvas, so exactly one window pixel per canvas pixel reads
its own position back. That one is the pixel's representative.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


def new_canvases(n_slots: int, n_pixels: int, dtype: jax.typing.DTypeLike = jnp.float32) -> tuple[jax.Array, jax.Array]:
    """Zeroed value canvas and ``-1`` owner canvas for :func:`accumulate_extract`, with a padding slot."""
    size = (n_slots + 1) * n_pixels
    return jnp.zeros(size, dtype), jnp.full(size, -1, jnp.int32)


@functools.partial(jax.jit, static_argnames=("n_slots", "n_pixels", "max_nnz"), donate_argnames=("canvas", "owner"))
def accumulate_extract(
    canvas: jax.Array,
    owner: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    threshold: jax.typing.ArrayLike,
    n_slots: int,
    n_pixels: int,
    max_nnz: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Add ``values`` at ``keys``, return the summed pixels above ``threshold``, reset what was touched.

    Parameters
    ----------
    canvas
        [(n_slots + 1) * n_pixels] float scratch canvas, all zero on entry. Donated.
    owner
        [(n_slots + 1) * n_pixels] int32 scratch canvas, all -1 on entry. Donated.
    keys
        [N] int32 ``slot * n_pixels + pixel``, with ``N < 2**31``. Padding must use slot ``n_slots``
        (the extra one), which is never extracted, so padding cannot take ownership of a real pixel.
    values
        [N] intensities
    threshold
        Pixels whose summed value is ``<= threshold`` are dropped.
    n_slots, n_pixels
        Static. Frames per batch and pixels per frame.
    max_nnz
        Static. Output capacity. If ``nnz > max_nnz`` the output is truncated: re-run with fewer slots.

    Returns
    -------
    canvas, owner: jax.Array
        The scratch canvases, zero and -1 again.
    out_keys: jax.Array
        [max_nnz] int32 keys of the extracted pixels, in no particular order. Valid up to ``nnz``.
    out_values: jax.Array
        [max_nnz] summed values of the extracted pixels.
    nnz: jax.Array
        Number of pixels above threshold. May exceed ``max_nnz``.
    """
    pos = jnp.arange(keys.shape[0], dtype=jnp.int32)
    canvas = canvas.at[keys].add(values, mode="promise_in_bounds")
    owner = owner.at[keys].max(pos, mode="promise_in_bounds")
    total = canvas[keys]
    keep = (owner[keys] == pos) & (total > threshold) & (keys < n_slots * n_pixels)
    (sel,) = jnp.nonzero(keep, size=max_nnz, fill_value=0)
    nnz = keep.sum(dtype=jnp.int32)
    out_keys = keys[sel]
    out_values = total[sel]
    canvas = canvas.at[keys].set(0.0, mode="promise_in_bounds")
    owner = owner.at[keys].set(-1, mode="promise_in_bounds")
    return canvas, owner, out_keys, out_values, nnz
