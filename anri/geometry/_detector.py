import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._utils import rmat_from_axis_angle


@jax.jit
def detector_orientation_matrix(o11: float, o12: float, o21: float, o22: float) -> Float[Array, "3 3"]:
    """Return (3D) detector orientation matrix from 2D orientation elements.

    In ImageD11.transform.compute_xyz_lab
    """
    return jnp.array([[o11, o12, 0], [o21, o22, 0], [0, 0, 1]])


@jax.jit
def detector_rotation_matrix(tilt_x: float, tilt_y: float, tilt_z: float) -> Float[Array, "3 3"]:
    """Return rotation matrix for detector tilts about x, y, z axes.

    R1 = Z, R2 = Y, R3 = X
    tilt_x, tilt_y, tilt_z are in radians
    but chi and wedge are in degrees

    ImageD11.transform.detector_rotation_matrix
    """
    R1 = rmat_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.degrees(tilt_z))
    R2 = rmat_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), jnp.degrees(tilt_y))
    R3 = rmat_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.degrees(tilt_x))

    # combine in order R3 @ R2 @ R1
    return R3 @ R2 @ R1


@jax.jit
def detector_transforms(
    y_center: float,
    y_size: float,
    tilt_y: float,
    z_center: float,
    z_size: float,
    tilt_z: float,
    tilt_x: float,
    distance: float,
    o11: float,
    o12: float,
    o21: float,
    o22: float,
) -> tuple[Float[Array, "3 3"], Float[Array, "3"], Float[Array, "3"]]:
    """Return required transformation matrices and shifts to convert between detector and lab coordinates.

    v_lab = (det_tilts @ (cob_matrix @ (det_flips @ (pixel_size_scale @ (v_det + beam_cen_shift))))) + x_distance_shift
    v_lab = M(v_det + beam_cen_shift) + x_distance_shift
    v_det = M^-1(v_lab - x_distance_shift) - beam_cen_shift.

    In ImageD11.transform.compute_xyz_lab
    """
    beam_cen_shift = jnp.array([-z_center, -y_center, 0])  # shift to beam center in detector coords
    pixel_size_scale = jnp.array([[z_size, 0, 0], [0, y_size, 0], [0, 0, 1]])  # change pixel units to units of y_size
    det_flips = detector_orientation_matrix(o11, o12, o21, o22)  # detector orientation flips
    cob_matrix = jnp.array(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    )  # map (sc, fc) from (x, y) in detector frame to (z, y) in lab frame
    det_tilts = detector_rotation_matrix(tilt_x, tilt_y, tilt_z)
    x_distance_shift = jnp.array([distance, 0, 0])  # shift along lab x by sample-to-detector distance

    det_trans = det_tilts @ cob_matrix @ det_flips @ pixel_size_scale  # combine all transforms except shifts

    return det_trans, beam_cen_shift, x_distance_shift


@jax.jit
def det_to_lab(sc, fc, det_trans, beam_cen_shift, x_distance_shift):
    """Convert detector (sc, fc) coordinates to lab (lx, ly, lz) coordinates.

    If diffraction is from the lab origin, the lab vector is parallel to k_out.
    ImageD11.transform.compute_xyz_lab
    """
    v_det = jnp.array([sc, fc, 0])
    v_lab = det_trans @ (v_det + beam_cen_shift) + x_distance_shift
    return v_lab


@jax.jit
def lab_to_det(xl, yl, zl, det_trans, beam_cen_shift, x_distance_shift):
    """Convert lab (lx, ly, lz) coordinates to detector (sc, fc) coordinates.

    Inverse of ImageD11.transform.compute_xyz_lab
    """
    v_lab = jnp.array([xl, yl, zl])
    v_det = jnp.linalg.inv(det_trans) @ (v_lab - x_distance_shift) - beam_cen_shift
    return v_det[:2]


@jax.jit
def detector_basis_vectors_lab(det_trans, beam_cen_shift, x_distance_shift):
    """Get detector basis vectors (slow, fast, normal) in lab coordinates."""
    # Find vectors in the fast, slow directions in the detector plane
    sc = jnp.array([1.0, 0.0, 0])
    fc = jnp.array([0.0, 1.0, 0])

    sc_lab = det_to_lab(sc[0], fc[0], det_trans, beam_cen_shift, x_distance_shift)
    fc_lab = det_to_lab(sc[1], fc[1], det_trans, beam_cen_shift, x_distance_shift)
    norm_lab = det_to_lab(sc[2], fc[2], det_trans, beam_cen_shift, x_distance_shift)

    return sc_lab, fc_lab, norm_lab


@jax.jit
def raytrace_to_det(
    vec_lab: jax.Array,
    origin_lab: jax.Array,
    sc_lab: jax.Array,
    fc_lab: jax.Array,
    norm_lab: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Ray-trace from vector in lab frame (unknown length) to detector coordinates (sc, fc).

    Adapted from :func:`ImageD11.transform.compute_xyz_from_tth_eta`.

    Parameters
    ----------
    vec_lab
        [3] Outgoing scaled normalised wave-vector in lab frame
    origin_lab
        [3] Origin of diffraction in lab frame
    sc_lab
        [3] Slow coordinate axis of detector in lab frame (pixel unit length)
    fc_lab
        [3] Fast coordinate axis of detector in lab frame (pixel unit length)
    norm_lab
        [3] Normal vector of detector in lab frame

    Returns
    -------
    sc: jax.Array
        Slow coordinate on detector in pixels
    fc: jax.Array
        Fast coordinate on detector in pixels

    See Also
    --------
    anri.geometry.detector_basis_vectors_lab : Compute detector basis vectors in lab frame, needed for `sc_lab`, `fc_lab`, `norm_lab`.

    Notes
    -----
    Originally from section 5 of (Thomas, 1992) [3]_ .

    References
    ----------
    .. [3] Thomas, D.J., 1992. Modern equations of diffractometry. Diffraction geometry. Acta Crystallographica Section A 48, 134–158. https://doi.org/10.1107/S0108767391008577
    """
    # we assume diffraction happens from the origin
    # then we account for the shifts later in the detector plane

    # ensure vec_lab is unit vector
    unit_vec_lab = vec_lab / jnp.linalg.norm(vec_lab)

    ds = sc_lab - norm_lab  # 1,0 in plane is (1,0)-(0,0)
    df = fc_lab - norm_lab  # 0,1 in plane
    dO = norm_lab  # origin pixel

    # Cross products to get the detector normal
    det_norm = jnp.cross(ds, df)

    # Scattered rays on detector normal
    norm = jnp.dot(det_norm, unit_vec_lab)
    # Check for divide by zero
    msk = norm == 0
    norm += msk

    # Intersect ray on detector plane
    sc = jnp.dot(jnp.cross(df, dO), unit_vec_lab) / norm
    fc = jnp.dot(jnp.cross(dO, ds), unit_vec_lab) / norm

    # project lab origin onto the detector face to give shifts in px
    sct = (unit_vec_lab * jnp.cross(df, origin_lab)).sum(axis=0) / norm
    fct = (unit_vec_lab * jnp.cross(origin_lab, ds)).sum(axis=0) / norm
    sc -= sct
    fc -= fct

    fc = jnp.where(msk, 0, fc)
    sc = jnp.where(msk, 0, sc)

    return sc, fc
