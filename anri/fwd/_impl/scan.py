"""Forward projection code for Scanning 3DXRD case."""

import jax
import jax.numpy as jnp

from anri.geom import dty_and_origin_lab, raytrace_to_det

from .base import hkl_to_k_omega, hkl_to_k_omega_both, make_propagator


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
    s_step_lab: jax.Array,  # detector
    f_step_lab: jax.Array,
    det_origin_lab: jax.Array,
) -> tuple[jax.Array, bool]:
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
        [3] Direction of the incoming beam before divergence, lab frame (any length, not vertical)
    ky
        Horizontal beam divergence: small tilt of the beam (radians) along the horizontal across it, see
        :func:`anri.geom.beam_basis`. Usually zero.
    kz
        Vertical beam divergence: small tilt of the beam (radians) along the vertical across it, see
        :func:`anri.geom.beam_basis`. Usually zero.
    wedge
        Wedge motor value (degrees)
    chi
        Chi motor value (degrees)
    y0
        The true value of dty when the rotation axis (untilted by wedge, chi) intersects the beam
    s_step_lab
        [3] Lab-frame step of one pixel along the slow direction, from :func:`anri.geom.detector_basis_vectors_lab`.
    f_step_lab
        [3] Lab-frame step of one pixel along the fast direction, from :func:`anri.geom.detector_basis_vectors_lab`.
    det_origin_lab
        [3] Lab-frame position of pixel (0, 0), from :func:`anri.geom.detector_basis_vectors_lab`.

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
        hkl,  # peak stuff
        etasign,
        wavelength,  # beam
        k_in_lab,
        ky,
        kz,
        wedge,  # gonio
        chi,
    )

    dty, origin_lab = dty_and_origin_lab(origin_sample, k_in_lab, omega, wedge, chi, y0)
    sc, fc = raytrace_to_det(k_out_lab, origin_lab, s_step_lab, f_step_lab, det_origin_lab)

    centroid = jnp.array([sc, fc, omega, dty])

    return centroid, valid


@jax.jit
def get_centroid_scan_both(
    ubi: jax.Array,  # grain stuff
    origin_sample: jax.Array,
    hkl: jax.Array,  # peak stuff
    wavelength: float,  # beam
    k_in_lab: jax.Array,
    ky: float,
    kz: float,
    wedge: float,  # gonio
    chi: float,
    y0: float,
    s_step_lab: jax.Array,  # detector
    f_step_lab: jax.Array,
    det_origin_lab: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Forward project (ubi, hkl) to both Friedel 4D peak centroids in the Scanning 3DXRD case.

    This can be vectorised over ubis and origin_samples, see
    :func:`get_centroid_scan_all_grains_both`. It can then be vectorised in an
    outer loop over hkl, see :func:`get_centroid_scan_all_both`.

    Parameters
    ----------
    ubi
        [3,3] (U.B)^(-1) matrix of the grain/voxel
    origin_sample
        [3] origin position of the voxel in the sample reference frame
    hkl
        [3] (h,k,l) reciprocal space vector
    wavelength
        Wavelength in angstroms
    k_in_lab:
        [3] Direction of the incoming beam before divergence, lab frame (any length, not vertical)
    ky
        Horizontal beam divergence: small tilt of the beam (radians) along the horizontal across it, see
        :func:`anri.geom.beam_basis`. Usually zero.
    kz
        Vertical beam divergence: small tilt of the beam (radians) along the vertical across it, see
        :func:`anri.geom.beam_basis`. Usually zero.
    wedge
        Wedge motor value (degrees)
    chi
        Chi motor value (degrees)
    y0
        The true value of dty when the rotation axis (untilted by wedge, chi) intersects the beam
    s_step_lab
        [3] Lab-frame step of one pixel along the slow direction, from :func:`anri.geom.detector_basis_vectors_lab`.
    f_step_lab
        [3] Lab-frame step of one pixel along the fast direction, from :func:`anri.geom.detector_basis_vectors_lab`.
    det_origin_lab
        [3] Lab-frame position of pixel (0, 0), from :func:`anri.geom.detector_basis_vectors_lab`.

    Returns
    -------
    centroids: jax.Array
        [2,4] Peak centres of mass in (sc, fc, omega, dty). Index 0 is the
        ``etasign = +1`` solution, index 1 is ``etasign = -1``.
    valid: jax.Array
        Boolean indicating if a valid solution exists, shared by both branches

    Notes
    -----
    There is no ``etasign`` argument. Both solutions come from one call to
    :func:`anri.fwd.hkl_to_k_omega_both`, which evaluates the geometry shared
    between the branches once. Only ray-tracing to the detector, and the dty
    that follows from each omega, are done per branch.

    See Also
    --------
    get_centroid_scan : Single-solution version, taking an ``etasign`` argument.
    """
    k_in_lab, k_out_labs, omegas, valid = hkl_to_k_omega_both(
        ubi,  # grain stuff
        hkl,  # peak stuff
        wavelength,  # beam
        k_in_lab,
        ky,
        kz,
        wedge,  # gonio
        chi,
    )

    centroids = []
    for i in range(2):
        dty, origin_lab = dty_and_origin_lab(origin_sample, k_in_lab, omegas[i], wedge, chi, y0)
        sc, fc = raytrace_to_det(k_out_labs[i], origin_lab, s_step_lab, f_step_lab, det_origin_lab)
        centroids.append(jnp.array([sc, fc, omegas[i], dty]))

    return jnp.stack(centroids), valid


propagate_cov_scan = make_propagator(get_centroid_scan, argnums=(1, 4, 6, 7), has_aux=True)

### vmaps
# The fully-vectorised entry points are wrapped in jax.jit so that repeated calls
# reuse one compiled program instead of re-tracing the nested vmaps each time.

# vmap over grains
get_centroid_scan_all_grains = jax.vmap(
    get_centroid_scan, in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None, None]
)

# vmap over hkls
get_centroid_scan_all = jax.jit(
    jax.vmap(
        get_centroid_scan_all_grains,
        in_axes=[None, None, 0, None, None, None, None, None, None, None, None, None, None, None],
    )
)

# vmap over grains
get_centroid_scan_all_grains_both = jax.vmap(
    get_centroid_scan_both, in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None]
)

# vmap over hkls
get_centroid_scan_all_both = jax.jit(
    jax.vmap(
        get_centroid_scan_all_grains_both,
        in_axes=[None, None, 0, None, None, None, None, None, None, None, None, None, None],
    )
)

# vmap over grains
propagate_cov_scan_all_grains = jax.vmap(
    propagate_cov_scan,
    in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None],
)

# vmap over hkls
propagate_cov_scan_all = jax.jit(
    jax.vmap(
        propagate_cov_scan_all_grains,
        in_axes=[None, None, 0, None, None, None, None, None, None, None, None, None, None, None, None],
    )
)

# argnums are one lower than propagate_cov_scan: get_centroid_scan_both takes no etasign
propagate_cov_scan_both = make_propagator(get_centroid_scan_both, argnums=(1, 3, 5, 6), has_aux=True)

# vmap over grains
propagate_cov_scan_all_grains_both = jax.vmap(
    propagate_cov_scan_both,
    in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None, None],
)

# vmap over hkls
propagate_cov_scan_all_both = jax.jit(
    jax.vmap(
        propagate_cov_scan_all_grains_both,
        in_axes=[None, None, 0, None, None, None, None, None, None, None, None, None, None, None],
    )
)
