"""Forward projection code for Scanning 3DXRD case."""

import jax
import jax.numpy as jnp

from anri.geom.detector import raytrace_to_det
from anri.geom.gonio import dty_and_origin_lab

from .peak import hkl_to_k_omega, make_propagator


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
    y0
        The true value of dty when the rotation axis (untilted by wedge, chi) intersects the beam
    s_step_lab
        [3] Lab-frame step of one pixel along the slow direction, from :func:`anri.geom.detector.detector_basis_vectors_lab`.
    f_step_lab
        [3] Lab-frame step of one pixel along the fast direction, from :func:`anri.geom.detector.detector_basis_vectors_lab`.
    det_origin_lab
        [3] Lab-frame position of pixel (0, 0), from :func:`anri.geom.detector.detector_basis_vectors_lab`.

    Returns
    -------
    centroid: jax.Array
        [4] Peak centre-of-mass in (sc, fc, omega, dty)

    Notes
    -----
    Propagates (h,k,l) into k-vectors using :func:`anri.fwd.peak.hkl_to_k_omega`

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


propagate_cov_scan = make_propagator(get_centroid_scan, argnums=(1, 4, 6, 7), has_aux=True)
