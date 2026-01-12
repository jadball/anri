"""Functions to convert between detector and laboratory frames."""

import jax
import jax.numpy as jnp

from .utils import rot_x, rot_y, rot_z


@jax.jit
def detector_orientation_matrix(o11: float, o12: float, o21: float, o22: float) -> jax.Array:
    r"""Return a matrix that transforms detector (slow, fast) pixel values to (z_det, y_det).

    This accounts for the varying possible readout directions and origins of the detector.
    We get a vector coming back in the format $\left(z_{\text{det}}, y_{\text{det}}, 0\right)$.
    See :ref:`tut_geom` for more details about FABLE geometry.
    In ImageD11, you can find this in :func:`ImageD11.transform.compute_xyz_lab`

    Parameters
    ----------
    o11:
        11 element of orientation matrix
    o12:
        12 element of orientation matrix
    o21:
        21 element of orientation matrix
    o22:
        22 element of orientation matrix

    Returns
    -------
    jax.Array
        3x3 image orientation matrix, with the 4 specified elements in the top-left corner.
    
    Notes
    -----

    .. math::
        \begin{pmatrix}
        o_{11} & o_{12} & 0 \\
        o_{21} & o_{22} & 0 \\
        0      & 0      & 1
        \end{pmatrix}
        \begin{pmatrix}
        s \\
        f \\
        0
        \end{pmatrix}
        =
        \begin{pmatrix}
        z_{\text{det}} \\
        y_{\text{det}} \\
        0
        \end{pmatrix}
    
    """
    return jnp.array([[o11, o12, 0], [o21, o22, 0], [0, 0, 1]])


@jax.jit
def detector_rotation_matrix(tilt_x: float, tilt_y: float, tilt_z: float) -> jax.Array:
    r"""Return a 3D orientation matrix that transforms a point in the detector reference frame to the lab frame.

    Adapted from :func:`ImageD11.transform.detector_rotation_matrix`.
    Applies rotations in order (Z,Y,X).

    Parameters
    ----------
    tilt_x
        Tilt of the detector around lab x vector (right-handed rotation) in radians.
    tilt_y
        Tilt of the detector around lab y vector (right-handed rotation) in radians.
    tilt_z
        Tilt of the detector around lab z vector (right-handed rotation) in radians.

    Returns
    -------
    jax.Array
        [3,3] Rotation matrix

    Notes
    -----
    We have $\matr{R} = \matr{R_x} \matr{R_y} \matr{R_z}$ around $x,y,z$ respectively.
    """
    R_x = rot_x(jnp.degrees(tilt_x))
    R_y = rot_y(jnp.degrees(tilt_y))
    R_z = rot_z(jnp.degrees(tilt_z))

    return R_x @ R_y @ R_z


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
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Return all transformations required to convert from detector pixel to lab space.

    The units of pixel size and distance need to match, but you can choose what they are.
    In ImageD11, they are normally microns, but we don't need to enforce it as they cancel out in the end.
    Adapted from :func:`ImageD11.transform.compute_xyz_lab`
    To use these transformations, see :func:`det_to_lab` and :func:`lab_to_det`

    Parameters
    ----------
    y_center
        Detector Y position where the laboratory x vector hits the detector (pixels)
    y_size
        Detector pixel size in Y.
    tilt_y
        Tilt of the detector around lab y vector (right-handed rotation) in radians. See :func:`detector_rotation_matrix`
    z_center
        Detector Z position where the laboratory x vector hits the detector (pixels)
    z_size
        Detector pixel size in Z.
    tilt_z
        Tilt of the detector around lab z vector (right-handed rotation) in radians. See :func:`detector_rotation_matrix`
    tilt_x
        Tilt of the detector around lab x vector (right-handed rotation) in radians. See :func:`detector_rotation_matrix`
    distance
        The distance from the centre of rotation (i.e. rotation axis) to the (y_center, z_center) position.
    o11:
        11 element of orientation matrix. See :func:`detector_orientation_matrix`
    o12:
        12 element of orientation matrix. See :func:`detector_orientation_matrix`
    o21:
        21 element of orientation matrix. See :func:`detector_orientation_matrix`
    o22:
        22 element of orientation matrix. See :func:`detector_orientation_matrix`

    Returns
    -------
    det_trans: jax.Array
        [3,3] Combined transformation matrix (see Notes)
    beam_cen_shift: jax.Array
        [3] Translation vector to beam center
    x_distance_shift: jax.Array
        [3] Translation vector to move detector along X axis

    Notes
    -----
    We can convert a detector value (s, f) to a vector in the lab frame:

    $\vec{v_{\text{det}}} = \left(s, f, 0\right)$

    $\vec{v_{\text{lab}}} = \left(x_{\text{lab}}, y_{\text{lab}}, z_{\text{lab}}\right)$

    $\vec{v_{\text{lab}}} = \matr{D_{\text{trans}}} \cdot \left(\vec{v_{\text{det}} + \vec{v_{\text{cen shift}}}}\right) + \vec{v_{\text{dist shift}}}$

    where:

    $\vec{v_{\text{cen shift}}} = \left(-z_{\text{center}}, -y_{\text{center}}, 0\right)$

    $\vec{v_{\text{dist shift}}} = \left(\text{distance}, 0, 0\right)$

    $\matr{D_{\text{trans}}} = \matr{R_{\text{det}}} \cdot \matr{R_{\text{c.o.b}}} \cdot \matr{R_{\text{orien}}} \cdot \matr{S_{\text{px size}}}$

    where:

    $\matr{R_{\text{det}}}$ comes from :func:`detector_rotation_matrix`

    $\matr{R_{\text{c.o.b}}} = \begin{bmatrix} 0 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 0 & 0 \end{bmatrix}$ which is a change-of-basis matrix that maps $\left(z_{\text{det}}, y_{\text{det}}, 0\right)$ to $\left(0_{\text{det}}, y_{\text{det}}, z_{\text{det}}\right)$

    $\matr{R_{\text{orien}}}$ comes from :func:`detector_orientation_matrix`

    $\matr{S_{\text{px size}}} = \begin{bmatrix} z_{\text{size}} & 0 & 0 \\ 0 & y_{\text{size}} & 0 \\ 0 & 0 & 1 \end{bmatrix}$ scales the pixel units to the (y_size, z_size) units.
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
def det_to_lab(
    sc: float, fc: float, det_trans: jax.Array, beam_cen_shift: jax.Array, x_distance_shift: jax.Array
) -> jax.Array:
    """Convert detector (sc, fc) coordinates to lab (lx, ly, lz) coordinates.

    Applies detector transformations from :func:`detector_transforms`.
    Adapted from :func:`ImageD11.transform.compute_xyz_lab`

    Parameters
    ----------
    sc
        Slow diretion pixel value
    fc
        Fast diretion pixel value
    det_trans
        [3,3] detector transformation matrix from :func:`detector_transforms`
    beam_cen_shift
        [3] beam center shift from :func:`detector_transforms`
    x_distance_shift
        [3] X distance shift from :func:`detector_transforms`

    Returns
    -------
    v_lab: jax.Array
        [3] (x_lab, y_lab, z_lab) vector
    """
    v_det = jnp.array([sc, fc, 0])
    v_lab = det_trans @ (v_det + beam_cen_shift) + x_distance_shift
    return v_lab


@jax.jit
def lab_to_det(
    xl: float, yl: float, zl: float, det_trans: jax.Array, beam_cen_shift: jax.Array, x_distance_shift: jax.Array
) -> jax.Array:
    """Convert lab (lx, ly, lz) coordinates to detector (sc, fc) coordinates.

    Inverse of :func:`det_to_lab`

    Parameters
    ----------
    xl
        X component of lab vector
    yl
        Y component of lab vector
    zl
        Z component of lab vector
    det_trans
        [3,3] detector transformation matrix from :func:`detector_transforms`
    beam_cen_shift
        [3] beam center shift from :func:`detector_transforms`
    x_distance_shift
        [3] X distance shift from :func:`detector_transforms`

    Returns
    -------
    v_det: jax.Array
        [2] (sc, fc) vector
    """
    v_lab = jnp.array([xl, yl, zl])
    v_det = jnp.linalg.inv(det_trans) @ (v_lab - x_distance_shift) - beam_cen_shift
    return v_det[:2]


@jax.jit
def detector_basis_vectors_lab(
    det_trans: jax.Array, beam_cen_shift: jax.Array, x_distance_shift: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Get laboratory basis vectors for detector (slow, fast, normal).

    Needed for :func:`raytrace_to_det`

    Parameters
    ----------
    det_trans
        [3,3] detector transformation matrix from :func:`detector_transforms`
    beam_cen_shift
        [3] beam center shift from :func:`detector_transforms`
    x_distance_shift
        [3] X distance shift from :func:`detector_transforms`

    Returns
    -------
    sc_lab: jax.Array
        [3] Laboratory basis vector for the slow direction on the detector.
    fc_lab: jax.Array
        [3] Laboratory basis vector for the fast direction on the detector.
    norm_lab: jax.Array
        [3] Laboratory basis vector for the detector normal.
    """
    # Find vectors in the fast, slow directions in the detector plane
    # 3 basis vectors as pixels in the detector
    sc = jnp.array([1.0, 0.0, 0])
    fc = jnp.array([0.0, 1.0, 0])

    # 3 basis vectors in the lab frame
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
    ssc_lab: jax.Array
        [3] Laboratory basis vector for the slow direction on the detector from :func:`detector_basis_vectors_lab`.
    fc_lab: jax.Array
        [3] Laboratory basis vector for the fast direction on the detector from :func:`detector_basis_vectors_lab`.
    norm_lab: jax.Array
        [3] Laboratory basis vector for the detector normal from :func:`detector_basis_vectors_lab`.

    Returns
    -------
    sc: jax.Array
        Slow coordinate on detector in pixels
    fc: jax.Array
        Fast coordinate on detector in pixels

    See Also
    --------
    anri.geom.detector_basis_vectors_lab : Compute detector basis vectors in lab frame, needed for `sc_lab`, `fc_lab`, `norm_lab`.

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
