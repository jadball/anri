"""Forward-projection functions, that use the rest of anri to project peaks onto detector images."""

from ._impl.base import get_cov_in, hkl_to_k_omega, hkl_to_k_omega_both, inv3, propagate_cov
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
    get_centroid_scan_pairs_both,
    get_centroid_scan_all,
    get_centroid_scan_all_both,
    get_centroid_scan_all_grains,
    get_centroid_scan_all_grains_both,
    get_centroid_scan_both,
    propagate_cov_scan,
    propagate_cov_scan_all,
    propagate_cov_scan_all_both,
    propagate_cov_scan_all_both_diag,
    propagate_cov_scan_all_grains,
    propagate_cov_scan_all_grains_both,
    propagate_cov_scan_both,
    propagate_cov_scan_both_diag,
    propagate_cov_scan_both_sf,
    propagate_cov_scan_all_both_sf,
    propagate_cov_scan_pairs_both_sf,
    propagate_cov_scan_pairs_both,
    propagate_cov_scan_pairs_both_diag,
)

# from ._impl.splat import peak_to_pixels, peaks_to_pixels, prepare_gaussian_bin, sample_gaussian_bins

# fmt: off
__all__ = [
    "get_centroid_box",
    "get_centroid_box_all",
    "get_centroid_box_all_grains",
    "get_centroid_scan",
    "get_centroid_scan_all",
    "get_centroid_scan_all_grains",
    "get_cov_in",
    "hkl_to_k_omega",
#    "peak_to_pixels",
#    "peaks_to_pixels",
#    "prepare_gaussian_bin",
    "get_centroid_scan_all_both",
    "get_centroid_scan_pairs_both",
    "get_centroid_scan_all_grains_both",
    "get_centroid_scan_both",
    "hkl_to_k_omega_both",
    "inv3",
    "propagate_cov_scan_all_both",
    "propagate_cov_scan_all_both_diag",
    "propagate_cov_scan_all_grains_both",
    "propagate_cov_scan_both",
    "propagate_cov_scan_both_diag",
    "propagate_cov_scan_both_sf",
    "propagate_cov_scan_all_both_sf",
    "propagate_cov_scan_pairs_both_sf",
    "propagate_cov_scan_pairs_both",
    "propagate_cov_scan_pairs_both_diag",
    "propagate_cov",
    "propagate_cov_box",
    "propagate_cov_box_all",
    "propagate_cov_box_all_grains",
    "propagate_cov_scan",
    "propagate_cov_scan_all",
    "propagate_cov_scan_all_grains",
#    "sample_gaussian_bins"
]
# fmt: on
