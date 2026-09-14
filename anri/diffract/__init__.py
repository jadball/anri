"""Diffraction-related utilities and functions."""

from ._impl.scatter import (
    ds_to_tth,
    k_to_q_lab,
    omega_from_core,
    omega_solns,
    omega_solns_both,
    omega_solns_core,
    peak_lab_to_k_out,
    q_lab_to_k_out,
    q_lab_to_tth_eta,
    q_to_ds,
    scale_norm_k,
    tth_eta_to_k_out,
)

__all__ = [
    "ds_to_tth",
    "k_to_q_lab",
    "omega_from_core",
    "omega_solns",
    "omega_solns_both",
    "omega_solns_core",
    "peak_lab_to_k_out",
    "q_lab_to_k_out",
    "q_lab_to_tth_eta",
    "q_to_ds",
    "scale_norm_k",
    "tth_eta_to_k_out",
]
