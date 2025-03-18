from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt
from ImageD11 import cImageD11
from scipy.ndimage import gaussian_filter, uniform_filter
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
        smoothed = gaussian_filter(cor, self.smoothsigma)
        # where is the smoothed corrected image less than the user-supplied threshold?
        smooth_under_thresh = smoothed < self.threshold
        smoothed[smooth_under_thresh] = 0

        # label on smoothed image
        # prepare outputs for localmaxlabel:
        wrk = np.empty(cor.shape, "b")  # temporary array for labeling, can be ignored
        labels = np.empty(cor.shape, "i")  # 32-bit integer array of labels
        # modifies wrk and labels in-place:
        npks = cImageD11.localmaxlabel(smoothed, labels, wrk)  # type: ignore
        # wipe out labels under threshold
        labels[smooth_under_thresh] = 0
        # TODO - what happens here?
        l3 = uniform_filter(labels * 7, 3)
        borders = (labels * 7) != l3
        # border properties
        blobsouter = cImageD11.blobproperties(cor, labels * borders, npks)  # type: ignore
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
        bgcalc = per_peak_bg[labels]
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
