"""Forward projection code for box-beam scans (near or far-field)."""

import jax
import jax.numpy as jnp

from anri.geom import raytrace_to_det, sample_to_lab

from .base import hkl_to_k_omega, hkl_to_k_omega_both, make_propagator


@jax.jit
def get_centroid_box(
    ubi: jax.Array,
    origin_sample: jax.Array,
    hkl: jax.Array,
    etasign: int,
    wavelength: float,
    k_in_lab: jax.Array,
    ky: float,
    kz: float,
    wedge: float,
    chi: float,
    s_step_lab: jax.Array,
    f_step_lab: jax.Array,
    det_origin_lab: jax.Array,
) -> tuple[jax.Array, bool]:
    r"""Forward project (ubi, hkl) to get 3D peak centroid on detector (sc, fc, omega) in the box-beam case.

    This can be vectorised over ubis and origin_samples, see :func:`get_centroid_box_all_grains`.
    It can then be vectorised in an outer loop over hkl, see :func:`get_centroid_box_all`.

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
    s_step_lab
        [3] Lab-frame step of one pixel along the slow direction, from :func:`anri.geom.detector_basis_vectors_lab`.
    f_step_lab
        [3] Lab-frame step of one pixel along the fast direction, from :func:`anri.geom.detector_basis_vectors_lab`.
    det_origin_lab
        [3] Lab-frame position of pixel (0, 0), from :func:`anri.geom.detector_basis_vectors_lab`.

    Returns
    -------
    centroid: jax.Array
        [3] Peak centre-of-mass in (sc, fc, omega)
    valid: bool
        Boolean indicating if a valid solution exists

    Notes
    -----
    Propagates (h,k,l) into k-vectors using :func:`anri.fwd.hkl_to_k_omega`

    Then computes the origin in the lab frame, and ray-traces into the detector.
    """
    k_in_lab, k_out_lab, omega, valid = hkl_to_k_omega(
        ubi,
        hkl,
        etasign,
        wavelength,
        k_in_lab,
        ky,
        kz,
        wedge,
        chi,
    )

    origin_lab = sample_to_lab(origin_sample, omega, wedge, chi, 0.0, 0.0)

    sc, fc = raytrace_to_det(k_out_lab, origin_lab, s_step_lab, f_step_lab, det_origin_lab)

    centroid = jnp.array([sc, fc, omega])

    return centroid, valid


@jax.jit
def get_centroid_box_both(
    ubi: jax.Array,
    origin_sample: jax.Array,
    hkl: jax.Array,
    wavelength: float,
    k_in_lab: jax.Array,
    ky: float,
    kz: float,
    wedge: float,
    chi: float,
    s_step_lab: jax.Array,
    f_step_lab: jax.Array,
    det_origin_lab: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Forward project (ubi, hkl) to both Friedel 3D peak centroids on the detector in the box-beam case.

    This can be vectorised over ubis and origin_samples, see :func:`get_centroid_box_all_grains_both`.
    It can then be vectorised in an outer loop over hkl, see :func:`get_centroid_box_all_both`.

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
    s_step_lab
        [3] Lab-frame step of one pixel along the slow direction, from :func:`anri.geom.detector_basis_vectors_lab`.
    f_step_lab
        [3] Lab-frame step of one pixel along the fast direction, from :func:`anri.geom.detector_basis_vectors_lab`.
    det_origin_lab
        [3] Lab-frame position of pixel (0, 0), from :func:`anri.geom.detector_basis_vectors_lab`.

    Returns
    -------
    centroids: jax.Array
        [2,3] Peak centres of mass in (sc, fc, omega). Index 0 is the ``etasign = +1`` solution,
        index 1 is ``etasign = -1``.
    valid: jax.Array
        Boolean indicating if a valid solution exists, shared by both branches

    Notes
    -----
    There is no ``etasign`` argument. Both solutions come from one call to
    :func:`anri.fwd.hkl_to_k_omega_both`, which evaluates the geometry shared
    between the branches once. Only ray-tracing to the detector is done per branch.

    See Also
    --------
    get_centroid_box : Single-solution version, taking an ``etasign`` argument.
    """
    _, k_out_labs, omegas, valid = hkl_to_k_omega_both(ubi, hkl, wavelength, k_in_lab, ky, kz, wedge, chi)

    centroids = []
    for i in range(2):
        origin_lab = sample_to_lab(origin_sample, omegas[i], wedge, chi, 0.0, 0.0)
        sc, fc = raytrace_to_det(k_out_labs[i], origin_lab, s_step_lab, f_step_lab, det_origin_lab)
        centroids.append(jnp.array([sc, fc, omegas[i]]))

    return jnp.stack(centroids), valid


propagate_cov_box = make_propagator(get_centroid_box, argnums=(1, 4, 6, 7), has_aux=True)

### vmaps
# Fully-vectorised entry points are jitted so repeated calls reuse one compiled
# program instead of re-tracing the nested vmaps.
# vmap over grains
get_centroid_box_all_grains = jax.vmap(
    get_centroid_box, in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None]
)

# vmap over hkls
get_centroid_box_all = jax.jit(
    jax.vmap(
        get_centroid_box_all_grains,
        in_axes=[None, None, 0, None, None, None, None, None, None, None, None, None, None],
    )
)


# vmap over grains
get_centroid_box_all_grains_both = jax.vmap(
    get_centroid_box_both, in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None]
)

# vmap over hkls
get_centroid_box_all_both = jax.jit(
    jax.vmap(
        get_centroid_box_all_grains_both,
        in_axes=[None, None, 0, None, None, None, None, None, None, None, None, None],
    )
)


# vmap over grains
propagate_cov_box_all_grains = jax.vmap(
    propagate_cov_box, in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None, None]
)

# vmap over hkls
propagate_cov_box_all = jax.jit(
    jax.vmap(
        propagate_cov_box_all_grains,
        in_axes=[None, None, 0, None, None, None, None, None, None, None, None, None, None, None],
    )
)
