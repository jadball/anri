import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


### coordinate systems
# lab frame: arbitrary, right-handed. beam is normally along x. with no base tilts or base transformations, rotation axis defines z, and origin is where axis intersects beam
# sample frame: top of diffractometer stack. with no diffractometer rotations, aligned with lab frame

### mathematical conventions
# k-vectors need to be normalised and then scaled
# we need a notation for this
# k_in and k_out as written are always normalised and scaled correctly
# therefore q_lab is always scaled correctly


### scalar functions


@jax.jit
def _rmat_from_axis_angle(axis, angle):
    """Return rotation matrix for rotation about axis by angle (degrees)."""
    rom = jnp.radians(angle)
    som = jnp.sin(rom)
    com = jnp.cos(rom)
    C = 1 - com
    # normalise axis
    axis = axis / jnp.linalg.norm(axis)
    x, y, z = axis

    Q = jnp.array(
        [
            [x * x * C + com, x * y * C - z * som, x * z * C + y * som],
            [y * x * C + z * som, y * y * C + com, y * z * C - x * som],
            [z * x * C - y * som, z * y * C + x * som, z * z * C + com],
        ]
    )
    return Q


rmat_from_axis_angle = jax.jit(jax.vmap(_rmat_from_axis_angle, in_axes=[None, 0]))


@jax.jit
def detector_rotation_matrix(tilt_x, tilt_y, tilt_z):
    """Return rotation matrix for detector tilts about x, y, z axes.

    R1 = Z, R2 = Y, R3 = X
    tilt_x, tilt_y, tilt_z are in radians
    but chi and wedge are in degrees

    ImageD11.transform.detector_rotation_matrix
    """
    R1 = _rmat_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.degrees(tilt_z))
    R2 = _rmat_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), jnp.degrees(tilt_y))
    R3 = _rmat_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.degrees(tilt_x))

    # combine in order R3 @ R2 @ R1
    return R3 @ R2 @ R1


@jax.jit
def chimat(chi):
    """Return rotation matrix for rotation about x axis by chi (degrees).

    ImageD11.gv_general.chimat
    """
    # negative rotation about x-axis
    return _rmat_from_axis_angle(jnp.array([-1.0, 0.0, 0.0]), chi)


@jax.jit
def wedgemat(wedge):
    """Return rotation matrix for rotation about y axis by wedge (degrees).

    ImageD11.gv_general.wedgemat
    """
    return _rmat_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), wedge)


@jax.jit
def detector_orientation_matrix(o11, o12, o21, o22):
    """Return (3D) detector orientation matrix from 2D orientation elements.

    In ImageD11.transform.compute_xyz_lab
    """
    return jnp.array([[o11, o12, 0], [o21, o22, 0], [0, 0, 1]])


@jax.jit
def detector_transforms(y_center, y_size, tilt_y, z_center, z_size, tilt_z, tilt_x, distance, o11, o12, o21, o22):
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


### DETECTOR <-> LAB TRANSFORMS


@jax.jit
def _det_to_lab(sc, fc, det_trans, beam_cen_shift, x_distance_shift):
    """Convert detector (sc, fc) coordinates to lab (lx, ly, lz) coordinates.

    If diffraction is from the lab origin, the lab vector is parallel to k_out.
    ImageD11.transform.compute_xyz_lab
    """
    v_det = jnp.array([sc, fc, 0])
    v_lab = det_trans @ (v_det + beam_cen_shift) + x_distance_shift
    return v_lab


det_to_lab = jax.jit(jax.vmap(_det_to_lab, in_axes=[0, 0, None, None, None]))


@jax.jit
def _lab_to_det(xl, yl, zl, det_trans, beam_cen_shift, x_distance_shift):
    """Convert lab (lx, ly, lz) coordinates to detector (sc, fc) coordinates.

    Inverse of ImageD11.transform.compute_xyz_lab
    """
    v_lab = jnp.array([xl, yl, zl])
    v_det = jnp.linalg.inv(det_trans) @ (v_lab - x_distance_shift) - beam_cen_shift
    return v_det[:2]


lab_to_det = jax.jit(jax.vmap(_lab_to_det, in_axes=[0, 0, 0, None, None, None]))


### LAB <-> SAMPLE TRANSFORMS


@jax.jit
def _sample_to_lab(v_sample, omega, wedge, chi):
    """Convert from sample to lab coordinates (apply the diffractometer stack).

    v_lab = W.T @ C.T @ R.T @ v_sample
    Adapted from ImageD11.transform.compute_g_from_k and compute_grain_origins
    """
    W = wedgemat(wedge)
    C = chimat(chi)

    R = _rmat_from_axis_angle(jnp.array([0.0, 0.0, -1.0]), omega)

    v_lab = W.T @ C.T @ R.T @ v_sample

    return v_lab


@jax.jit
def sample_to_lab(v_sample, omega, wedge, chi):
    """As per Jake Vanderplas.

    jit is compatible with if statements whose truth or falsehood is decidable at trace time. x is None fits this criterion, because it concerns the identity of the Python variable.
    Other statements that can be used within if statements are static attributes of the an array, such as shape, size, or dtype.
    What will not work within JIT is attempting to branch based on dynamic quantites, such as values within arrays.
    https://github.com/jax-ml/jax/discussions/14224#discussioncomment-4831741
    """
    if v_sample.shape == (3,):
        # just a 3-vector
        return jax.jit(jax.vmap(_sample_to_lab, in_axes=[None, 0, None, None]))(v_sample, omega, wedge, chi)
    else:
        # vec_sample is (N,3), omega is (N)
        return jax.jit(jax.vmap(_sample_to_lab, in_axes=[0, 0, None, None]))(v_sample, omega, wedge, chi)


@jax.jit
def _lab_to_sample(v_lab, omega, wedge, chi):
    """Convert from lab to sample coordinates (apply the diffractometer stack).

    v_sample = R @ C @ W @ v_lab
    Adapted from ImageD11.transform.compute_g_from_k and compute_grain_origins
    Equivalent to ImageD11.transform.compute_g_from_k
    t_x, t_y, t_z are in the sample frame!
    """
    W = wedgemat(wedge)
    C = chimat(chi)

    R = _rmat_from_axis_angle(jnp.array([0.0, 0.0, -1.0]), omega)

    v_sample = R @ C @ W @ v_lab

    return v_sample


lab_to_sample = jax.jit(jax.vmap(_lab_to_sample, in_axes=[0, 0, None, None]))


@jax.jit
def q_lab_to_q_sample(q_lab, omega, wedge, chi):
    """Convert from q-vector in lab frame to q-vector in sample frame."""
    q_sample = lab_to_sample(q_lab, omega, wedge, chi)
    return q_sample


# Tricky functions first, then we have a graveyard of cross-transforms at the end.


@jax.jit
def _scale_norm_k(k_vec, wavelength):
    """Normalise then scale k-vector to 1/wavelength."""
    k = 1 / wavelength  # ImageD11 convention
    k_vec_norm = k_vec / jnp.linalg.norm(k_vec)
    k_vec_scaled = k * k_vec_norm
    return k_vec_scaled


scale_norm_k = jax.jit(jax.vmap(_scale_norm_k, in_axes=[0, None]))


@jax.jit
def _k_to_q_lab(k_in, k_out):
    """Convert from scaled normalised (k_in, k_out) to q-vector in the lab frame.

    q = k_out - k_in
    """
    q_lab = k_out - k_in
    return q_lab


k_to_q_lab = jax.jit(jax.vmap(_k_to_q_lab, in_axes=[None, 0]))


@jax.jit
def _q_lab_to_k_out(q_lab, k_in):
    """Convert from q-vector and scaled normalised k_in to scaled normalised k_out in the lab frame.

    k_out = q + k_in
    """
    k_out = q_lab + k_in
    return k_out


q_lab_to_k_out = jax.jit(jax.vmap(_q_lab_to_k_out, in_axes=[0, None]))


@jax.jit
def peak_lab_to_k_out(peak_lab, origin_lab, wavelength):
    """Convert from vector of peak in lab frame to normalised scaled k_out in the lab frame.

    We determine k_out = peak_lab - origin_lab
    """
    k_out_vec = peak_lab - origin_lab  # unscaled, un-normalised
    k_out = scale_norm_k(k_out_vec, wavelength)
    return k_out


@jax.jit
def _tth_eta_to_k_out(tth, eta, wavelength):
    """Convert from (tth, eta) in degrees to k_out in the lab frame.

    ImageD11.transform.compute_k_vectors
    """
    tth = jnp.radians(tth)
    eta = jnp.radians(eta)

    k_out_vec = jnp.array(
        [
            jnp.cos(tth),
            -jnp.sin(tth) * jnp.sin(eta),
            jnp.sin(tth) * jnp.cos(eta),
        ]
    )

    k_out = _scale_norm_k(k_out_vec, wavelength)

    # c = jnp.cos(tth / 2)  # cos theta
    # s = jnp.sin(tth / 2)  # sin theta
    # ds = 2 * s / wavelength

    # k1 = -ds * s  # this is negative x
    # k2 = -ds * c * jnp.sin(eta)  # CHANGED eta to HFP convention 4-9-2007
    # k3 = ds * c * jnp.cos(eta)

    return k_out


tth_eta_to_k_out = jax.jit(jax.vmap(_tth_eta_to_k_out, in_axes=[0, 0, None]))


@jax.jit
def _q_lab_to_tth_eta(q_lab, wavelength):
    """Convert from Q in the lab frame to (tth, eta).

    Inverse of ImageD11.transform.compute_k_vectors
    """
    q1, q2, q3 = q_lab
    ds = jnp.linalg.norm(q_lab)
    s = ds * wavelength / 2.0  # sin(theta)
    tth = 2.0 * jnp.degrees(jnp.arcsin(s))
    eta = jnp.degrees(jnp.arctan2(-q2, q3))
    return tth, eta


q_lab_to_tth_eta = jax.jit(jax.vmap(_q_lab_to_tth_eta, in_axes=[0, None]))


@jax.jit
def _omega_solns(q_sample, k_in, wavelength, axis, pre, post):
    """Computes omega rotation angles needed for each g to diffract."""
    rg = pre @ q_sample

    krot = jnp.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1.0]])  # works for wedges but not others

    beam = krot @ k_in

    rb = jnp.dot(post.T, beam)

    a1 = jnp.cross(axis, rg)
    a2 = jnp.cross(a1, axis)
    a0 = rg - a2

    rbda0 = rb @ a0
    rbda1 = rb @ a1
    rbda2 = rb @ a2

    kdotbeam = -jnp.sum(q_sample * q_sample) / 2.0

    phi = jnp.arctan2(rbda2, rbda1)
    den = jnp.sqrt(rbda1 * rbda1 + rbda2 * rbda2)

    quot = (kdotbeam - rbda0) / den
    valid = (quot >= -1) & (quot <= 1)
    quot = jnp.where(valid, quot, 0)
    x_plus_p = jnp.arcsin(quot)
    sol1 = x_plus_p + phi
    sol2 = jnp.pi - x_plus_p + phi

    angmod_sol1 = jnp.arctan2(jnp.sin(sol1), jnp.cos(sol1))
    angmod_sol2 = jnp.arctan2(jnp.sin(sol2), jnp.cos(sol2))

    return jnp.degrees(angmod_sol1), jnp.degrees(angmod_sol2), valid


omega_solns = jax.jit(jax.vmap(_omega_solns, in_axes=[0, None, None, None, None, None]))


# fully working
@jax.jit
def _q_lab_to_det(q_lab, omega, origin_lab, k_in, wavelength, det_trans, beam_cen_shift, x_distance_shift):
    # xyz = unit vectors along the scattered vectors

    # we assume diffraction happens from the origin
    # then we account for the shifts later in the detector plane

    # add k_in to get k_out
    k_out = _q_lab_to_k_out(q_lab, k_in)
    # rescale by wavelength which gives unit vector
    unit_vec_lab = k_out * wavelength

    # Find vectors in the fast, slow directions in the detector plane
    sc = jnp.array([1.0, 0.0, 0])
    fc = jnp.array([0.0, 1.0, 0])

    dxyzl = det_to_lab(sc, fc, det_trans, beam_cen_shift, x_distance_shift).T

    # == [xpos, ypos, zpos] shape (3,n)
    #
    # This was based on the recipe from Thomas in Acta Cryst ...
    #  ... Modern Equations of ...

    ds = dxyzl[:, 0] - dxyzl[:, 2]  # 1,0 in plane is (1,0)-(0,0)
    df = dxyzl[:, 1] - dxyzl[:, 2]  # 0,1 in plane
    dO = dxyzl[:, 2]  # origin pixel

    # Cross products to get the detector normal
    # Thomas uses an inverse matrix, but then divides out the determinant anyway
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


q_lab_to_det = jax.jit(jax.vmap(_q_lab_to_det, in_axes=[0, 0, 0, None, None, None, None, None]))

### Cross-transforms graveyard


@jax.jit
def tth_eta_to_q_lab(tth, eta, k_in, wavelength):
    k_out = tth_eta_to_k_out(tth, eta, wavelength)
    q_lab = k_to_q_lab(k_in, k_out)
    return q_lab


@jax.jit
def peak_lab_to_q_lab(peak_lab, origin_lab, k_in, wavelength):
    k_out = peak_lab_to_k_out(peak_lab, origin_lab, wavelength)
    q_lab = k_to_q_lab(k_in, k_out)
    return q_lab


@jax.jit
def q_lab_to_peak_lab(q_lab, omega, origin_lab, k_in, wavelength, det_trans, beam_cen_shift, x_distance_shift):
    sc, fc = q_lab_to_det(q_lab, omega, origin_lab, k_in, wavelength, det_trans, beam_cen_shift, x_distance_shift)
    peak_lab = det_to_lab(sc, fc, det_trans, beam_cen_shift, x_distance_shift)
    return peak_lab


@jax.jit
def peak_lab_to_q_sample(peak_lab, omega, origin_lab, k_in, wedge, chi, wavelength):
    k_out = peak_lab_to_k_out(peak_lab, origin_lab)
    q_lab = k_to_q_lab(k_in, k_out, wavelength)
    q_sample = q_lab_to_q_sample(q_lab, omega, wedge, chi)
    return q_sample


@jax.jit
def _q_sample_to_q_lab(q_sample, k_in, wedge, chi, wavelength):
    """Get Q in the lab frame from Q in the sample frame.

    There are two solutions for Q_lab for each Q_sample.

    ImageD11.gv_general.g_to_k
    """
    # wedge and chi matrices
    W = wedgemat(wedge)
    C = chimat(chi)

    post = W @ C

    # work out valid omega angles for g-vectors given a wavelength and rotation axis
    omega1, omega2, valid = _omega_solns(
        q_sample,
        k_in,
        wavelength,
        jnp.array([0, 0, -1.0]),
        jnp.eye(3),
        post,
    )

    # invalidate incorrect omega angles
    omega1 = jnp.where(valid, omega1, jnp.nan)
    omega2 = jnp.where(valid, omega2, jnp.nan)

    # now get q_lab
    q_lab1 = _sample_to_lab(q_sample, omega1, wedge, chi)
    q_lab2 = _sample_to_lab(q_sample, omega2, wedge, chi)

    return [q_lab1, q_lab2], [omega1, omega2], valid


q_sample_to_q_lab = jax.jit(jax.vmap(_q_sample_to_q_lab, in_axes=[0, None, None, None, None]))


@jax.jit
def q_sample_to_det(q_sample, origin_lab, k_in, wedge, chi, wavelength, det_trans, beam_cen_shift, x_distance_shift):
    [q_lab1, q_lab2], [omega1, omega2], valid = q_sample_to_q_lab(q_sample, k_in, wedge, chi, wavelength)
    sc1, fc1 = q_lab_to_det(q_lab1, omega1, origin_lab, k_in, wavelength, det_trans, beam_cen_shift, x_distance_shift)
    sc2, fc2 = q_lab_to_det(q_lab2, omega2, origin_lab, k_in, wavelength, det_trans, beam_cen_shift, x_distance_shift)

    return [sc1, sc2], [fc1, fc2], [omega1, omega2], valid


def q_and_origin_sample_to_det(
    q_sample, origin_sample, k_in, wedge, chi, wavelength, det_trans, beam_cen_shift, x_distance_shift
):
    """Like q_sample_to_det, but origin is given in sample frame. Useful for forward simulating."""
    [q_lab1, q_lab2], [omega1, omega2], valid = q_sample_to_q_lab(q_sample, k_in, wedge, chi, wavelength)
    # now we have omega angles, we can get origin_lab
    origin_lab_1 = sample_to_lab(origin_sample, omega1, wedge, chi)
    origin_lab_2 = sample_to_lab(origin_sample, omega2, wedge, chi)
    sc1, fc1 = q_lab_to_det(q_lab1, omega1, origin_lab_1, k_in, wavelength, det_trans, beam_cen_shift, x_distance_shift)
    sc2, fc2 = q_lab_to_det(q_lab2, omega2, origin_lab_2, k_in, wavelength, det_trans, beam_cen_shift, x_distance_shift)

    return [sc1, sc2], [fc1, fc2], [omega1, omega2], valid


@jax.jit
def q_sample_to_peak_lab(
    q_sample, origin_lab, k_in, wedge, chi, wavelength, det_trans, beam_cen_shift, x_distance_shift
):
    [sc1, sc2], [fc1, fc2], [omega1, omega2], valid = q_sample_to_det(
        q_sample, origin_lab, k_in, wedge, chi, wavelength, det_trans, beam_cen_shift, x_distance_shift
    )
    peak_lab1 = det_to_lab(sc1, fc1, det_trans, beam_cen_shift, x_distance_shift)
    peak_lab2 = det_to_lab(sc2, fc2, det_trans, beam_cen_shift, x_distance_shift)
    return [peak_lab1, peak_lab2], [omega1, omega2], valid


@jax.jit
def peak_lab_to_tth_eta(
    peak_lab,
    origin_lab,
    k_in,
    wavelength,
):
    q_lab = peak_lab_to_q_lab(peak_lab, origin_lab, k_in, wavelength)
    tth, eta = q_lab_to_tth_eta(q_lab, wavelength)
    return tth, eta


@jax.jit
def q_sample_to_tth_eta_omega(q_sample, k_in, wedge, chi, wavelength):
    [q_lab1, q_lab2], [omega1, omega2], valid = q_sample_to_q_lab(q_sample, k_in, wedge, chi, wavelength)
    tth1, eta1 = q_lab_to_tth_eta(q_lab1, wavelength)
    tth2, eta2 = q_lab_to_tth_eta(q_lab2, wavelength)
    return tth1, [eta1, eta2], [omega1, omega2], valid


@jax.jit
def tth_eta_omega_to_q_sample(tth, eta, omega, k_in, wedge, chi, wavelength):
    q_lab = tth_eta_to_q_lab(tth, eta, k_in, wavelength)
    q_sample = q_lab_to_q_sample(q_lab, omega, wedge, chi)
    return q_sample


@jax.jit
def det_to_q_lab(sc, fc, omega, origin_lab, k_in, wedge, chi, wavelength, det_trans, beam_cen_shift, x_distance_shift):
    peak_lab = det_to_lab(sc, fc, det_trans, beam_cen_shift, x_distance_shift)
    q_lab = peak_lab_to_q_lab(peak_lab, origin_lab, k_in, wavelength)
    return q_lab


@jax.jit
def det_to_q_sample(
    sc,
    fc,
    omega,
    origin_lab,
    k_in,
    wedge,
    chi,
    wavelength,
    det_trans,
    beam_cen_shift,
    x_distance_shift,
):
    q_lab = det_to_q_lab(
        sc,
        fc,
        omega,
        origin_lab,
        k_in,
        wedge,
        chi,
        wavelength,
        det_trans,
        beam_cen_shift,
        x_distance_shift,
    )
    q_sample = q_lab_to_q_sample(q_lab, omega, wedge, chi)
    return q_sample


@jax.jit
def tth_eta_omega_to_det(
    tth,
    eta,
    omega,
    origin_lab,
    k_in,
    wedge,
    chi,
    wavelength,
    det_trans,
    beam_cen_shift,
    x_distance_shift,
):
    q_lab = tth_eta_to_q_lab(tth, eta, k_in, wavelength)
    sc, fc = q_lab_to_det(q_lab, omega, origin_lab, k_in, wavelength, det_trans, beam_cen_shift, x_distance_shift)

    return sc, fc
