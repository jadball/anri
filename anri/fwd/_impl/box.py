"""Forward projection code for box-beam scans (near or far-field)."""

import jax
import jax.numpy as jnp

import anri

from .base import hkl_to_k_omega, propagate_cov


@jax.jit
def get_centroid_box(
    ubi: jax.Array,
    origin_sample: jax.Array,
    hkl: jax.Array,
    etasign: int,
    wavelength: float,
    ky: float,
    kz: float,
    wedge: float,
    chi: float,
    y0: float,
    sc_lab: jax.Array,
    fc_lab: jax.Array,
    norm_lab: jax.Array,
) -> jax.Array:
    """Forward project (ubi, hkl) to get 3D peak centroid on detector (sc, fc, omega) in the box-beam case.

    This can be vectorised over ubis and origin_samples - :func:`get_centroid_box_all_grains`.
    It can then be vectorised in an outer loop over hkl - :func:`get_centroid_box_all`.

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
    wavelength:
        Wavelength in angstroms
    ky:
        y-component of the beam in the lab frame. Represents horizontal beam divergence, usually zero.
    kz:
        z-component of the beam in the lab frame. Represents vertical beam divergence, usually zero.
    wedge:
        Wedge motor value (degrees)
    chi:
        Chi motor value (degrees)
    y0:
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
        [3] Peak centre-of-mass in (sc, fc, omega)

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
        ky,
        kz,
        wedge,
        chi,
        y0,
    )

    origin_lab = anri.geom.sample_to_lab(origin_sample, omega, wedge, chi, 0.0, 0.0)

    sc, fc = anri.geom.raytrace_to_det(k_out_lab, origin_lab, sc_lab, fc_lab, norm_lab)

    centroid = jnp.array([sc, fc, omega])

    return centroid


# Function to get the Jacobian of get_centroid_box
J_get_centroid_box = jax.jacfwd(get_centroid_box, argnums=(1, 4, 5, 6))


@jax.jit
def propagate_cov_box(
    ubi: jax.Array,
    origin_sample: jax.Array,
    hkl: jax.Array,
    etasign: int,
    wavelength: float,
    ky: float,
    kz: float,
    wedge: float,
    chi: float,
    y0: float,
    sc_lab: jax.Array,
    fc_lab: jax.Array,
    norm_lab: jax.Array,
    ostep: float,
    cov_in: jax.Array,
) -> jax.Array:
    r"""Get output covariance matrix for a given forward-projected box-beam peak.

    This can be vectorised over ubis and origin_samples - :func:`propagate_cov_box_all_grains`.
    It can then be vectorised in an outer loop over hkl - :func:`propagate_cov_box_all`.

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
    wavelength:
        Wavelength in angstroms
    ky:
        y-component of the beam in the lab frame. Represents horizontal beam divergence, usually zero.
    kz:
        z-component of the beam in the lab frame. Represents vertical beam divergence, usually zero.
    wedge:
        Wedge motor value (degrees)
    chi:
        Chi motor value (degrees)
    y0:
        The true value of dty when the rotation axis (untilted by wedge, chi) intersects the beam
    sc_lab
        [3] Laboratory basis vector for the slow direction on the detector from :func:`anri.geom.detector_basis_vectors_lab`.
    fc_lab
        [3] Laboratory basis vector for the fast direction on the detector from :func:`anri.geom.detector_basis_vectors_lab`.
    norm_lab
        [3] Laboratory basis vector for the detector normal from :func:`anri.geom.detector_basis_vectors_lab`.
    ostep
        Omega step size in degrees
    cov_in
        [6,6] Input covariance matrix - build with :func:`anri.fwd.get_cov_in`

    Returns
    -------
    cov_integrated: jax.Array
        [3,3] Output covariance matrix for this peak.

    Notes
    -----
    Gets Jacobian of :func:`anri.fwd.get_centroid_box`, then uses that to propagate cov_in via :func:`anri.fwd.propagate_cov`.
    Adds single pixel widths (in sc, fc, ostep) as variances to outputs to "spread" the signal over 1 pixel.
    """
    J_func_out = J_get_centroid_box(
        ubi,
        origin_sample,
        hkl,
        etasign,
        wavelength,
        ky,
        kz,
        wedge,
        chi,
        y0,
        sc_lab,
        fc_lab,
        norm_lab,
    )
    cov_out = propagate_cov(J_func_out, cov_in)

    # add single pixel width as variances
    voxel_var = jnp.diag(jnp.array([1 / 12, 1 / 12, (ostep**2) / 12]))
    cov_integrated = cov_out + voxel_var

    return cov_integrated


### vmaps
# vmap over grains
get_centroid_box_all_grains = jax.vmap(
    get_centroid_box, in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None]
)

# vmap over hkls
get_centroid_box_all = jax.vmap(
    get_centroid_box_all_grains, in_axes=[None, None, 0, None, None, None, None, None, None, None, None, None, None]
)

# vmap over grains
propagate_cov_box_all_grains = jax.vmap(
    propagate_cov_box, in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None]
)

# vmap over hkls
propagate_cov_box_all = jax.vmap(
    propagate_cov_box_all_grains,
    in_axes=[None, None, 0, None, None, None, None, None, None, None, None, None, None, None, None],
)
