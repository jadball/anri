"""Diffraction-related utilities and functions."""

from ._impl.scatter import (
    ds_to_tth,
    k_to_q_lab,
    omega_solns,
    peak_lab_to_k_out,
    q_lab_to_k_out,
    q_lab_to_tth_eta,
    q_to_ds,
    scale_norm_k,
    tth_eta_to_k_out,
)

__all__ = [
    "k_to_q_lab",
    "omega_solns",
    "peak_lab_to_k_out",
    "q_lab_to_k_out",
    "q_lab_to_tth_eta",
    "scale_norm_k",
    "tth_eta_to_k_out",
    "q_to_ds",
    "ds_to_tth",
]
