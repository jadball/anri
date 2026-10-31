"""Forward projection code for box-beam scans (near or far-field)."""

import jax

from .peak import get_centroid, make_propagator


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

    Vectorise it with ``jax.vmap``: over voxels (ubi, origin_sample), hkls, or etasign for both peaks.

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
        :func:`anri.geom.beam.beam_basis`. Usually zero.
    kz
        Vertical beam divergence: small tilt of the beam (radians) along the vertical across it, see
        :func:`anri.geom.beam.beam_basis`. Usually zero.
    wedge
        Wedge motor value (degrees)
    chi
        Chi motor value (degrees)
    s_step_lab
        [3] Lab-frame step of one pixel along the slow direction, from :func:`anri.geom.detector.detector_basis_vectors_lab`.
    f_step_lab
        [3] Lab-frame step of one pixel along the fast direction, from :func:`anri.geom.detector.detector_basis_vectors_lab`.
    det_origin_lab
        [3] Lab-frame position of pixel (0, 0), from :func:`anri.geom.detector.detector_basis_vectors_lab`.

    Returns
    -------
    centroid: jax.Array
        [3] Peak centre-of-mass in (sc, fc, omega)
    valid: bool
        Boolean indicating if a valid solution exists

    Notes
    -----
    This is :func:`anri.fwd.peak.get_centroid` with the rotation axis in the beam (dty = y0) and the voxel a cube,
    scattering from its centre.
    """
    # the voxel at its real position with the rotation axis in the beam (dty = y0), scattering from its centre
    return get_centroid(
        ubi, origin_sample, hkl, etasign, wavelength, k_in_lab, ky, kz, wedge, chi, 0.0, 0.0,
        s_step_lab, f_step_lab, det_origin_lab, column=False,
    )  # fmt: skip


propagate_cov_box = make_propagator(get_centroid_box, argnums=(1, 4, 6, 7), has_aux=True)
