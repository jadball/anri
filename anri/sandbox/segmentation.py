from typing import Optional

import h5py
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from ImageD11 import cImageD11
from jax import lax
from typing_extensions import override


def simple_cut(image: np.ndarray, cut: float) -> npt.NDArray[np.bool_]:
    """Return a boolean array mask of pixels >= cut.

    :param image: 2D array
    :param cut: minimum pixel value to keep
    :return: Boolean array mask
    """
    return image >= cut


def correct_flat_dark(image: np.ndarray, flat: Optional[np.ndarray] = None, dark: Optional[np.ndarray] = None):
    """Scale image by flat and subtract dark if supplied.

    Args:
        image: Raw detector image.
        flat: Detector sensitivity image. Defaults to None.
        dark: Detector dark current image (i.e. no beam). Defaults to None.

    Returns:
        Corrected image.
    """
    cor = image.copy()
    if dark is not None:
        cor -= dark
    if flat is not None:
        cor /= flat
    return cor


def scale_bg(bg: np.ndarray, m_offset_thresh: int = 70, m_ratio_thresh: int = 150):
    """Scale background image by offset and ratio.

    Background image pixels < m_offset_thresh are subtracted from the background image.
    Pixels greater than m_ratio_thresh will be later scaled.

    Args:
        bg: Detector background image. This should be corrected for flat and dark.
        m_offset_thresh: bgfile less than this is constant. Defaults to 70.
        m_ratio_thresh: bgfile greather than this is scaled. Defaults to 150.

    Returns:
        A tuple of `(cor, m_offset, m_ratio, invbg)` where:
            cor: the background image with offset subtracted.
            m_offset: a boolean array of pixels < m_offset_thresh.
            m_ratio: a boolean array of pixels > m_ratio_thresh.
            invbg: the inverse of the scaled background image masked to m_ratio.
    """
    cor = bg.copy()

    m_offset = cor < m_offset_thresh
    m_ratio = cor > m_ratio_thresh
    mbg = np.mean(cor[m_offset])
    cor -= mbg
    invbg = 1 / cor[m_ratio]
    return cor, m_offset, m_ratio, invbg


class Corrector:
    """Image corrector base class, with a reference implementation for flat-field and dark-field correction."""

    def __init__(
        self,
        dark: Optional[np.ndarray] = None,
        flat: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ):
        self.dark = dark
        self.flat = flat

        # normalise flat if present
        if self.flat is not None:
            self.flat /= self.flat.mean()
        self.mask = mask

    def __call__(self, image: np.ndarray):
        cor = correct_flat_dark(image, self.flat, self.dark)
        if self.mask is not None:
            cor[self.mask == 1] = 0
        return cor


class SimpleThresholdCorrector(Corrector):
    """Corrector class that applies a simple threshold to an image.

    This class applies a simple threshold to an image, zeroing out all pixels below a certain value.

    Attributes:
        threshold: ADU below which to zero out image. Defaults to 100.
    """

    def __init__(self, threshold: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    @override
    def __call__(self, image: np.ndarray):
        cor = image.copy()
        cor[cor < self.threshold] = 0
        return cor


def uniform_filter_jax(input, size, axis=None):
    output_dtype = input.dtype  # Ensure output matches input dtype
    compute_dtype = jnp.promote_types(input.dtype, jnp.float32)  # Use float for computation
    input = jnp.asarray(input, dtype=compute_dtype)  # Preserve original dtype

    if axis is None:
        size = (size,) * input.ndim if isinstance(size, int) else tuple(size)
        kernel = jnp.ones(size, dtype=compute_dtype) / jnp.prod(jnp.array(size))

        # Reflective padding matching SciPy
        pad_width = [(s // 2, s // 2) for s in size]
        input_padded = jnp.pad(input, pad_width, mode="edge")

        result = lax.conv_general_dilated(
            input_padded[None, None],  # Add batch and channel dims
            kernel[None, None],  # Add batch and channel dims
            window_strides=(1,) * input.ndim,
            padding="VALID",  # Since we manually pad
        )[0, 0]  # Remove batch and channel dims
        return jnp.round(result).astype(output_dtype)  # Round and cast back to original dtype
    else:
        shape = [1] * input.ndim
        shape[axis] = size
        kernel = jnp.ones(tuple(shape), dtype=input.dtype) / size

        # Reflective padding matching SciPy for the given axis
        pad_width = [(0, 0)] * input.ndim
        pad_width[axis] = (size // 2, size // 2)
        input_padded = jnp.pad(input, pad_width, mode="edge")

        result = lax.conv_general_dilated(
            input_padded[None, None],  # Add batch and channel dims
            kernel[None, None],  # Add batch and channel dims
            window_strides=(1,) * input.ndim,
            padding="VALID",  # Since we manually pad
        )[0, 0]  # Remove batch and channel dims
        return jnp.round(result).astype(output_dtype)  # Round and cast back to original dtype


import functools
from typing import Sequence

import jax
from jax._src.typing import ArrayLike
from scipy.ndimage import _ni_support


def _gaussian(x, sigma):
    return jnp.exp(-0.5 / sigma**2 * x**2) / jnp.sqrt(2 * jnp.pi * sigma**2)


def _grad_order(func, order):
    """Compute higher order grads recursively"""
    if order == 0:
        return func

    return jax.grad(_grad_order(func, order - 1))


def _gaussian_kernel1d(sigma, order, radius):
    """Computes a 1-D Gaussian convolution kernel"""
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    x = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    func = _grad_order(functools.partial(_gaussian, sigma=sigma), order)
    kernel = jax.vmap(func)(x)

    if order == 0:
        return kernel / jnp.sum(kernel)

    return kernel


def gaussian_filter1d(
    input: ArrayLike,
    sigma: float,
    axis: int = -1,
    order: int = 0,
    mode: str = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
    *,
    radius: int | None = None,
    method: str = "auto",
):
    """Compute a 1D Gaussian filter on the input array along the specified axis.

    Args:
        input: N-dimensional input array to filter.
        sigma: The standard deviation of the Gaussian filter.
        axis: The axis along which to apply the filter.
        order: The order of the Gaussian filter.
        mode: The mode to use for padding the input array. See :func:`jax.numpy.pad` for more details.
        cval: The value to use for padding the input array.
        truncate: The number of standard deviations to include in the filter.
        radius: The radius of the filter. Overrides `truncate` if provided.
        method: The method to use for the convolution.

    Returns:
        The filtered array.

    Examples:
        >>> from jax import numpy as jnp
        >>> import jax
        >>> input = jnp.arange(12.0).reshape(3, 4)
        >>> input
        Array([[ 0.,  1.,  2.,  3.],
               [ 4.,  5.,  6.,  7.],
               [ 8.,  9., 10., 11.]], dtype=float32)
        >>> jax.scipy.ndimage.gaussian_filter1d(input, sigma=1.0, axis=0, order=0)
       Array([[2.8350844, 3.8350847, 4.8350844, 5.8350844],
              [4.0000005, 5.       , 6.       , 7.0000005],
              [5.1649156, 6.1649156, 7.164916 , 8.164916 ]], dtype=float32)
    """
    if radius is None:
        radius = int(truncate * sigma + 0.5)

    if radius < 0:
        raise ValueError(f"Radius must be non-negative, got {radius}")

    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")

    pad_width = [(0, 0)] * input.ndim
    pad_width[axis] = (int(radius), int(radius))

    pad_kwargs = {"mode": mode}

    if mode == "constant":
        # jnp.pad errors if constant_values is provided and mode is not 'constant'
        pad_kwargs["constant_values"] = cval

    input_pad = jnp.pad(input, pad_width=pad_width, **pad_kwargs)

    kernel = _gaussian_kernel1d(sigma, order=order, radius=radius)

    axes = list(range(input.ndim))
    axes.pop(axis)
    kernel = jnp.expand_dims(kernel, axes)

    # boundary handling is done by jnp.pad, so we use the fixed valid mode
    return jax.scipy.signal.convolve(input_pad, kernel, mode="valid", method=method)


def gaussian_filter_jax(
    input: ArrayLike,
    sigma: float | Sequence[float],
    order: int | Sequence[int] = 0,
    mode: str = "reflect",
    cval: float | Sequence[float] = 0.0,
    truncate: float | Sequence[float] = 4.0,
    *,
    radius: None | Sequence[int] = None,
    axes: Sequence[int] = None,
    method="auto",
):
    """Gaussian filter for N-dimensional input

    Args:
       input: N-dimensional input array to filter.
       sigma: The standard deviation of the Gaussian filter.
       order: The order of the Gaussian filter.
       mode: The mode to use for padding the input array. See :func:`jax.numpy.pad` for more details.
       cval: The value to use for padding the input array.
       truncate: The number of standard deviations to include in the filter.
       radius: The radius of the filter. Overrides `truncate` if provided.
       method: The method to use for the convolution.
    """
    axes = _ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    orders = _ni_support._normalize_sequence(order, num_axes)
    sigmas = _ni_support._normalize_sequence(sigma, num_axes)
    modes = _ni_support._normalize_sequence(mode, num_axes)
    radii = _ni_support._normalize_sequence(radius, num_axes)

    # the loop goes over the input axes, so it is always low-dimensional and
    # keeping a Python loop is ok
    for idx in range(input.ndim):
        input = gaussian_filter1d(
            input,
            sigmas[idx],
            axis=idx,
            order=orders[idx],
            mode=modes[idx],
            cval=cval,
            truncate=truncate,
            radius=radii[idx],
            method=method,
        )

    return input


class LocalBGCorrector(Corrector):
    """Corrector class that applies per-peak background removal.

    This class, based on the ImageD11 frelon_peaksearch.py script, applies a per-peak background removal.
    The following steps are taken, assuming a background image is supplied:
    1. Correct the background image for flat and dark.
    2. Background is scaled via the function `scale_bg`
    3. The image is corrected for flat and dark
    3. The offset determined in `scale_bg` is applied to the image
    4. Some more sophisticated background subtraction is applied
    5. The image is smoothed via a Gaussian filter with sigma `smoothsigma`
    6. The smoothed image is thresholded via `threshold`
    7. The smoothed image is labelled to locate peak positions
    8. The properties of each labelled region is computed
    9. A per-peak background is computed for each blob
    10. The per-peak background is applied to the image, multiplied by `bgc`
    11. The image is masked by the per-peak background

    Attributes:
        threshold: ADU below which to zero out image. Defaults to 100.
        smoothsigma: sigma for Gaussian filter before labelleing. Defaults to 1.0.
        bgc: fractional part of bg per peak to remove. Defaults to 0.9.
        bg: detector background image (with beam, no sample). Defaults to None.
        dark: detector dark current (no beam). Defaults to None.
        flat: detector sensitivity image. Defaults to None.
        mask: Detector image mask. Active pixels are 0, masked pixels are 1. Defaults to None.
        m_offset_thresh: bgfile less than this is constant. Defaults to 70.
        m_ratio_thresh: bgfile greather than this is scaled. Defaults to 150.
    """

    def __init__(
        self,
        threshold: int = 100,
        smoothsigma: float = 1.0,
        bgc: float = 0.9,
        bg: Optional[np.ndarray] = None,
        dark: Optional[np.ndarray] = None,
        flat: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        m_offset_thresh: Optional[int] = 70,
        m_ratio_thresh: Optional[int] = 150,
    ):
        super().__init__(dark, flat)
        self.bg = bg
        self.mask = mask
        self.m_offset_thresh = m_offset_thresh
        self.m_ratio_thresh = m_ratio_thresh
        self.threshold = threshold
        self.smoothsigma = smoothsigma
        self.bgc = bgc

        # scale the background if we have one
        if self.bg is not None:
            # first, correct the background image
            self.bg = correct_flat_dark(self.bg, self.flat, self.dark)
            # then scale it
            if self.bg is not None:
                # need to ensure that m_offset_thresh and m_ratio_thresh values are valid
                if self.m_offset_thresh is None:
                    self.m_offset_thresh = 70
                if self.m_ratio_thresh is None:
                    self.m_ratio_thresh = 150
                self.bg, self.m_offset, self.m_ratio, self.invbg = scale_bg(
                    self.bg,
                    self.m_offset_thresh,
                    self.m_ratio_thresh,
                )

        # pre-compute constants for background subtraction
        self.bins = np.linspace(0, 2, 256)  # bin edges
        self.bc = (self.bins[1:] + self.bins[:-1]) / 2  # bin centres
        self.wrk = None
        self.labels = None

    @override
    def __call__(self, image: np.ndarray):
        # prepare outputs for localmaxlabel:
        # malloc and free stuff...
        if self.wrk is None:
            self.wrk = np.empty(image.shape, "b")  # temporary array for labeling, can be ignored
            self.labels = np.empty(image.shape, "i")  # 32-bit integer array of labels

        # first, correct the image for flat and dark
        cor = correct_flat_dark(image, self.flat, self.dark)
        # compute offsets
        if self.bg is None:
            # try to estimate a constant background from median of minima along different image axes
            offset = np.median((cor[0].min(), cor[-1].min(), cor[:, 0].min(), cor[:, -1].min()))
            cor -= offset
        else:
            # more sophisticated background subtraction
            offset = np.mean(cor[self.m_offset])
            cor -= offset
            ratio = cor[self.m_ratio] * self.invbg

            # TODO: document this better
            h, b = np.histogram(ratio, bins=self.bins)
            htrim = np.where(h < h.max() * 0.05, 0, h)
            r = (htrim * self.bc).sum() / htrim.sum()
            cor -= r * self.bg

        # apply mask
        if self.mask is not None:
            cor[self.mask == 1] = 0

        # smooth the corrected image
        # smoothed = gaussian_filter(cor, self.smoothsigma)
        smoothed = gaussian_filter_jax(cor, self.smoothsigma)
        # where is the smoothed corrected image less than the user-supplied threshold?
        smooth_under_thresh = smoothed < self.threshold
        smoothed[smooth_under_thresh] = 0

        # label on smoothed image
        # modifies wrk and labels in-place:

        npks = cImageD11.localmaxlabel(smoothed, self.labels, self.wrk)  # type: ignore
        # wipe out labels under threshold
        self.labels[smooth_under_thresh] = 0
        # TODO - what happens here?
        # l3 = uniform_filter(self.labels * 7, 3)
        l3 = uniform_filter_jax(self.labels * 7, 3)
        borders = (self.labels * 7) != l3
        # border properties
        blobsouter = cImageD11.blobproperties(cor, self.labels * borders, npks)  # type: ignore
        # Computed background per peak
        per_peak_bg = np.concatenate(
            (
                [
                    0,
                ],
                blobsouter[:, cImageD11.mx_I],  # type: ignore
            )
        )
        # make new calculated background from per-peak backgrounds
        bgcalc = per_peak_bg[self.labels]
        # apply bgc scale factor to bgcalc
        m_top = cor > bgcalc * self.bgc
        # make sure mask is zero where smoothed image is below threshold
        m_top[smooth_under_thresh] = 0
        # finally mask corrected input image
        cor = cor * m_top

        return cor


def load_test_image(h5file) -> npt.NDArray:
    """Load test image from h5 path.

    Args:
        h5file: H5 path.
    """
    with h5py.File(h5file, "r") as f:
        dataset = f.get("/1.1/measurement/frelon3")
        if dataset is not None and isinstance(dataset, h5py.Dataset):
            image = dataset[158]
        else:
            raise ValueError("Dataset '/1.1/measurement/frelon3' not found or invalid.")

    return image
