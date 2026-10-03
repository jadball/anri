"""Forward-projection functions, that use the rest of anri to project peaks onto detector images, and the renderer."""

from ._impl.base import get_cov_in, hkl_to_k_omega, hkl_to_k_omega_both, propagate_cov
from ._impl.box import (
    get_centroid_box,
    get_centroid_box_all,
    get_centroid_box_all_both,
    get_centroid_box_all_grains,
    get_centroid_box_all_grains_both,
    get_centroid_box_both,
    propagate_cov_box,
    propagate_cov_box_all,
    propagate_cov_box_all_grains,
)
from ._impl.render import dty_weight, lorentz, make_row, polarisation, render_peaks, render_row, select_peaks
from ._impl.scan import (
    get_centroid_scan,
    get_centroid_scan_all,
    get_centroid_scan_all_both,
    get_centroid_scan_all_grains,
    get_centroid_scan_all_grains_both,
    get_centroid_scan_both,
    propagate_cov_scan,
    propagate_cov_scan_all,
    propagate_cov_scan_all_both,
    propagate_cov_scan_all_grains,
    propagate_cov_scan_all_grains_both,
    propagate_cov_scan_both,
)

# fmt: off
__all__ = [
    "dty_weight",
    "get_centroid_box",
    "get_centroid_box_all",
    "get_centroid_box_all_both",
    "get_centroid_box_all_grains",
    "get_centroid_box_all_grains_both",
    "get_centroid_box_both",
    "get_centroid_scan",
    "get_centroid_scan_all",
    "get_centroid_scan_all_both",
    "get_centroid_scan_all_grains",
    "get_centroid_scan_all_grains_both",
    "get_centroid_scan_both",
    "get_cov_in",
    "hkl_to_k_omega",
    "hkl_to_k_omega_both",
    "lorentz",
    "make_row",
    "polarisation",
    "propagate_cov",
    "propagate_cov_box",
    "propagate_cov_box_all",
    "propagate_cov_box_all_grains",
    "propagate_cov_scan",
    "propagate_cov_scan_all",
    "propagate_cov_scan_all_both",
    "propagate_cov_scan_all_grains",
    "propagate_cov_scan_all_grains_both",
    "propagate_cov_scan_both",
    "render_peaks",
    "render_row",
    "select_peaks",
]
# fmt: on
