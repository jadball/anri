"""Forward projection code for Scanning 3DXRD case."""

import jax
import jax.numpy as jnp

from anri.geom import find_dty_for_beam_xy, raytrace_to_det, sample_to_lab

from .base import hkl_to_k_omega, make_propagator


@jax.jit
def get_active_peak_indices(
    dty: float, omega: float, centroids: jax.Array, dty_tol: float, omega_tol: float
) -> jax.Array:
    """Return a boolean mask of peaks that contribute to a specific (dty, omega) coordinate within tolerances.

    Parameters
    ----------
    dty
        Base diffractometer Y translation value (same units as v_sample)
    omega
        Omega motor value (degrees)
    centroids
        [N,4] 4D peak centroids [sc, fc, omega, dty]
    dty_tol
        Tolerance of dty
    omega_tol
        Tolerance of omega

    Returns
    -------
    is_active: jax.Array
        [N] bools - mask to centroids
    """
    # 1. Extract motor centroids
    # omega is index 2, dty is index 3
    omega_mus = centroids[:, 2]
    dty_mus = centroids[:, 3]

    # 3. Check boundaries for both motors independently
    # |val - mu| <= margin * sqrt(var)
    is_omega_active = jnp.abs(omega - omega_mus) <= omega_tol
    is_dty_active = jnp.abs(dty - dty_mus) <= dty_tol

    # 4. Peak is active only if it falls within the window for BOTH motors
    is_active = is_omega_active & is_dty_active

    return is_active


@jax.jit
def get_centroid_scan(
    ubi: jax.Array,  # grain stuff
    origin_sample: jax.Array,
    hkl: jax.Array,  # peak stuff
    etasign: float,
    wavelength: float,  # beam
    k_in_lab: jax.Array,
    ky: float,
    kz: float,
    wedge: float,  # gonio
    chi: float,
    y0: float,
    sc_lab: jax.Array,  # detector
    fc_lab: jax.Array,
    norm_lab: jax.Array,
) -> jax.Array:
    """Forward project (ubi, hkl) to get 4D peak centroid (sc, fc, omega, dty) in the Scanning 3DXRD case.

    This can be vectorised over ubis and origin_samples, see :func:`get_centroid_scan_all_grains`.
    It can then be vectorised in an outer loop over hkl, see :func:`get_centroid_scan_all`.

    Parameters
    ----------
    ubi
        [3,3] (U.B)^(-1) matrix of the grain/voxel
    origin_sample
        [3] origin position of the voxel in the sample reference frame
    hkl
        [3] (h,k,l) reciprocal space vector
    etasign
        +1 (omega1 in ImageD11) or -1 (omega2 in ImageD11) to select which omega solution to return
    wavelength
        Wavelength in angstroms
    k_in_lab:
        [3] Unperturbed unit vector of incoming beam, lab frame
    ky
        y-component of the beam in the lab frame. Represents horizontal beam divergence, usually zero.
    kz
        z-component of the beam in the lab frame. Represents vertical beam divergence, usually zero.
    wedge
        Wedge motor value (degrees)
    chi
        Chi motor value (degrees)
    y0
        The true value of dty when the rotation axis (untilted by wedge, chi) intersects the beam
    sc_lab
        [3] Laboratory basis vector for the slow direction on the detector from :func:`anri.geom.detector_basis_vectors_lab`.
    fc_lab
        [3] Laboratory basis vector for the fast direction on the detector from :func:`anri.geom.detector_basis_vectors_lab`.
    norm_lab
        [3] Laboratory basis vector for the detector normal from :func:`anri.geom.detector_basis_vectors_lab`.

    Returns
    -------
    centroid: jax.Array
        [4] Peak centre-of-mass in (sc, fc, omega, dty)

    Notes
    -----
    Propagates (h,k,l) into k-vectors using :func:`anri.fwd.hkl_to_k_omega`

    Then computes the origin in the lab frame, and ray-traces into the detector.
    """
    k_in_lab, k_out_lab, omega, valid = hkl_to_k_omega(
        ubi,  # grain stuff
        origin_sample,
        hkl,  # peak stuff
        etasign,
        wavelength,  # beam
        k_in_lab,
        ky,
        kz,
        wedge,  # gonio
        chi,
        y0,
    )

    dty = find_dty_for_beam_xy(origin_sample, k_in_lab, omega, wedge, chi, y0)
    origin_lab = sample_to_lab(origin_sample, omega, wedge, chi, dty, y0)
    sc, fc = raytrace_to_det(k_out_lab, origin_lab, sc_lab, fc_lab, norm_lab)

    centroid = jnp.array([sc, fc, omega, dty])

    return centroid, valid


propagate_cov_scan = make_propagator(get_centroid_scan, argnums=(1, 4, 6, 7), has_aux=True)

### vmaps
# vmap over grains
get_centroid_scan_all_grains = jax.vmap(
    get_centroid_scan, in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None, None]
)

# vmap over hkls
get_centroid_scan_all = jax.vmap(
    get_centroid_scan_all_grains, in_axes=[None, None, 0, None, None, None, None, None, None, None, None, None, None, None]
)

# vmap over grains
propagate_cov_scan_all_grains = jax.vmap(
    propagate_cov_scan,
    in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None],
)
# vmap over hkls
propagate_cov_scan_all = jax.vmap(
    propagate_cov_scan_all_grains,
    in_axes=[None, None, 0, None, None, None, None, None, None, None, None, None, None, None, None],
)
