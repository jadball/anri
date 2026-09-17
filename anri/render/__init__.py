"""Render forward-projected peaks into detector frames, sinograms and sparse pixel lists.

The pieces, in the order you use them:

:class:`Frames`
    the measurement grid and every recorded frame on it (from an ImageD11 dataset or arrays)
:class:`PeakSource`
    the anri forward model, evaluated for any list of peak ids
:class:`PeakTable`
    which peaks reach which rows; built once per sample model
:func:`sparse_frames` / :func:`write_sparse`
    every frame, thresholded, as ImageD11 sparse pixels
:func:`render_frames`
    dense detector images of chosen frames, or sums of frames
:func:`render_sum`
    one image with axes integrated out: whole-scan detector image, sinogram, ROI sinogram
:func:`loss_and_grad`
    sparse-to-sparse image loss against measured data, differentiable in the source parameters
:func:`splat`
    the kernel on its own: peaks in grid units onto one dense image, no culling
"""

from ._impl.frames import Frames
from ._impl.kernel import splat
from ._impl.pipeline import Measured, loss_and_grad, render_frames, render_sum, sparse_frames, write_sparse
from ._impl.table import PeakSource, PeakTable

__all__ = [
    "Frames",
    "Measured",
    "PeakSource",
    "PeakTable",
    "loss_and_grad",
    "render_frames",
    "render_sum",
    "sparse_frames",
    "splat",
    "write_sparse",
]
