"""Sum window pixels into frames and pull out the ones above threshold.

A batch of frames is rendered into a persistent scratch canvas of ``n_slots`` frames plus one dump
pixel for padding. Thresholding that canvas densely would cost ``n_slots * n_pixels`` per batch
however few pixels were touched, so only the touched pixels are read back and reset.

On CPU every pass over the touched pixels is a cache miss into a canvas of tens of MB, so passes are
kept to two: one gather of the summed values, one reset. Only the few pixels above threshold (a
fraction of a percent of window pixels in practice) are deduplicated, by sorting them.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

_NONE = jnp.iinfo(jnp.int32).max


def new_canvas(n_slots: int, n_pixels: int, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
    """Zeroed canvas for :func:`accumulate_extract`: ``n_slots`` frames and a dump pixel."""
    return jnp.zeros(n_slots * n_pixels + 1, dtype)


def _extract(
    canvas: jax.Array,
    keys: jax.Array,
    threshold: jax.typing.ArrayLike,
    n_slots: int,
    n_pixels: int,
    max_nnz: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pixels above ``threshold`` among ``keys``, for a canvas already holding the values; then reset.

    Returns
    -------
    canvas: jax.Array
        Zero again at ``keys``.
    out_keys: jax.Array
        [max_nnz] int32 sorted distinct keys above threshold, then padding.
    out_values: jax.Array
        [max_nnz] their values.
    nnz: jax.Array
        Number of distinct keys above threshold (valid entries of the outputs).
    hits: jax.Array
        Number of window pixels above threshold, duplicates included. If ``hits > max_nnz`` the output
        is incomplete: run again with more room.
    """
    total = canvas[keys]
    keep = (total > threshold) & (keys < n_slots * n_pixels)
    hits = keep.sum(dtype=jnp.int32)
    (sel,) = jnp.nonzero(keep, size=max_nnz, fill_value=0)
    cand = jnp.where(jnp.arange(max_nnz) < hits, keys[sel], _NONE)
    out_keys = jnp.unique(cand, size=max_nnz, fill_value=_NONE)
    valid = out_keys != _NONE
    nnz = valid.sum(dtype=jnp.int32)
    out_values = jnp.where(valid, canvas[jnp.where(valid, out_keys, 0)], 0.0)
    canvas = canvas.at[keys].set(0.0, mode="promise_in_bounds")
    return canvas, out_keys, out_values, nnz, hits


def _accumulate_extract(
    canvas: jax.Array,
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
        [n_slots * n_pixels + 1] float scratch canvas, all zero on entry. Donated.
    keys
        [N] int32 ``slot * n_pixels + pixel``. Padding uses ``n_slots * n_pixels``, the dump pixel.
    values
        [N] intensities
    threshold
        Pixels whose summed value is ``<= threshold`` are dropped.
    n_slots, n_pixels
        Static. Frames per batch and pixels per frame.
    max_nnz
        Static. Output capacity; see ``hits``.

    Returns
    -------
    See :func:`_extract`.
    """
    canvas = canvas.at[keys].add(values, mode="promise_in_bounds")
    return _extract(canvas, keys, threshold, n_slots, n_pixels, max_nnz)


accumulate_extract = functools.partial(
    jax.jit, static_argnames=("n_slots", "n_pixels", "max_nnz"), donate_argnames=("canvas",)
)(_accumulate_extract)
accumulate_extract.__doc__ = _accumulate_extract.__doc__
