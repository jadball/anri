"""Forward-projection functions, that use the rest of anri to project peaks onto detector images."""

from ._impl.base import get_cov_in, hkl_to_k_omega, propagate_cov
from ._impl.box import (
    get_centroid_box,
    get_centroid_box_all,
    get_centroid_box_all_grains,
    propagate_cov_box,
    propagate_cov_box_all,
    propagate_cov_box_all_grains,
)
from ._impl.scan import (
    get_centroid_scan,
    get_centroid_scan_all,
    get_centroid_scan_all_grains,
    propagate_cov_scan,
    propagate_cov_scan_all,
    propagate_cov_scan_all_grains,
)
from ._impl.splat import peak_to_pixels, peaks_to_pixels, prepare_gaussian_bin, sample_gaussian_bins

# fmt: off
__all__ = [
    "hkl_to_k_omega",
    "get_cov_in",
    "propagate_cov",
    "propagate_cov_box",
    "propagate_cov_box_all_grains",
    "propagate_cov_box_all",
    "propagate_cov_scan",
    "propagate_cov_scan_all_grains",
    "propagate_cov_scan_all",
    "get_centroid_box",
    "get_centroid_box_all_grains",
    "get_centroid_box_all",
    "get_centroid_scan",
    "get_centroid_scan_all_grains",
    "get_centroid_scan_all",
    "prepare_gaussian_bin",
    "sample_gaussian_bins",
    "peak_to_pixels",
    "peaks_to_pixels"
]
# fmt: on
