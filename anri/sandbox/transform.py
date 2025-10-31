import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


### scalar functions


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

    M = det_tilts @ cob_matrix @ det_flips @ pixel_size_scale  # combine all transforms except shifts

    return M, beam_cen_shift, x_distance_shift


### DETECTOR <-> LAB TRANSFORMS


def _det_to_xyz_lab(sc, fc, y_center, y_size, tilt_y, z_center, z_size, tilt_z, tilt_x, distance, o11, o12, o21, o22):
    """Convert detector (sc, fc) coordinates to lab (lx, ly, lz) coordinates.

    ImageD11.transform.compute_xyz_lab
    """
    M, beam_cen_shift, x_distance_shift = detector_transforms(
        y_center, y_size, tilt_y, z_center, z_size, tilt_z, tilt_x, distance, o11, o12, o21, o22
    )
    v_det = jnp.array([sc, fc, 0])
    v_lab = M @ (v_det + beam_cen_shift) + x_distance_shift
    return v_lab


det_to_xyz_lab = jax.jit(
    jax.vmap(_det_to_xyz_lab, in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None, None])
)


def _xyz_lab_to_det(
    xl, yl, zl, y_center, y_size, tilt_y, z_center, z_size, tilt_z, tilt_x, distance, o11, o12, o21, o22
):
    """Convert lab (lx, ly, lz) coordinates to detector (sc, fc) coordinates.

    Inverse of ImageD11.transform.compute_xyz_lab
    """
    M, beam_cen_shift, x_distance_shift = detector_transforms(
        y_center, y_size, tilt_y, z_center, z_size, tilt_z, tilt_x, distance, o11, o12, o21, o22
    )
    v_lab = jnp.array([xl, yl, zl])
    v_det = jnp.linalg.inv(M) @ (v_lab - x_distance_shift) - beam_cen_shift
    return v_det[:2]


xyz_lab_to_det = jax.jit(
    jax.vmap(_xyz_lab_to_det, in_axes=[0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None])
)

### LAB <-> SAMPLE TRANSFORMS


def _sample_to_lab(vec_sample, omega, wedge, chi):
    """Convert from lab to sample coordinates (apply the diffractometer stack).

    v_sample = W.T @ C.T @ R.T @ v_lab
    Adapted from ImageD11.transform.compute_g_from_k and compute_grain_origins
    """
    W = wedgemat(wedge)
    C = chimat(chi)

    R = _rmat_from_axis_angle(jnp.array([0.0, 0.0, -1.0]), omega)

    vec_lab = W.T @ C.T @ R.T @ vec_sample

    return vec_lab


def _lab_to_sample(vec_lab, omega, wedge, chi):
    """Convert from sample to lab coordinates (apply the diffractometer stack).

    v_lab = R @ C @ W @ v_sample
    Adapted from ImageD11.transform.compute_g_from_k and compute_grain_origins
    Equivalent to ImageD11.transform.compute_g_from_k
    t_x, t_y, t_z are in the sample frame!
    """
    W = wedgemat(wedge)
    C = chimat(chi)

    R = _rmat_from_axis_angle(jnp.array([0.0, 0.0, -1.0]), omega)

    vec_sample = R @ C @ W @ vec_lab

    return vec_sample


lab_to_sample = jax.jit(jax.vmap(_lab_to_sample, in_axes=[0, 0, None, None]))


def sample_to_lab(vec_sample, omega, wedge, chi):
    if vec_sample.shape == (3,):
        # just a 3-vector
        return jax.jit(jax.vmap(_sample_to_lab, in_axes=[None, 0, None, None]))(vec_sample, omega, wedge, chi)
    else:
        # vec_sample is (N,3), omega is (N)
        return jax.jit(jax.vmap(_sample_to_lab, in_axes=[0, 0, None, None]))(vec_sample, omega, wedge, chi)


# sample_to_lab = jax.jit(jax.vmap(_sample_to_lab, in_axes=[0, 0, None, None]))

### LAB XYZ TO (TTH, ETA) TRANSFORMS


def _xyz_lab_to_tth_eta(xyz_lab, omega, origin_sample, wedge, chi):
    """Compute tth and eta from lab xyz coordinates on the detector and diffraction origins in the sample frame.

    ImageD11.transform.compute_xyz_from_tth_eta
    """
    origin_lab = _sample_to_lab(origin_sample, omega, wedge, chi)
    scatter_vec_lab = xyz_lab - origin_lab
    eta = jnp.degrees(jnp.arctan2(-scatter_vec_lab[1], scatter_vec_lab[2]))
    s1_perp_x = jnp.sqrt(scatter_vec_lab[1] * scatter_vec_lab[1] + scatter_vec_lab[2] * scatter_vec_lab[2])
    tth = jnp.degrees(jnp.arctan2(s1_perp_x, scatter_vec_lab[0]))
    return tth, eta


def xyz_lab_to_tth_eta(xyz_lab, omega, origin_sample, wedge, chi):
    # origin_sample can either be a 3-vector or a (n, 3) array of origins same length as omega
    # if it's a 3-vector, we don't need to vectorize over it
    if origin_sample.shape == (3,):
        # just a 3-vector
        return jax.jit(jax.vmap(_xyz_lab_to_tth_eta, in_axes=[0, 0, None, None, None]))(
            xyz_lab, omega, origin_sample, wedge, chi
        )
    else:
        # array of origins
        return jax.jit(jax.vmap(_xyz_lab_to_tth_eta, in_axes=[0, 0, 0, None, None]))(
            xyz_lab, omega, origin_sample, wedge, chi
        )


### K-VECTOR TO (TTH,ETA,OMEGA) TRANSFORMS


def _tth_eta_to_k(tth, eta, wvln):
    """Convert from (tth, eta) to k-vector.

    ImageD11.transform.compute_k_vectors
    """
    tth = jnp.radians(tth)
    eta = jnp.radians(eta)
    c = jnp.cos(tth / 2)  # cos theta
    s = jnp.sin(tth / 2)  # sin theta
    ds = 2 * s / wvln

    k1 = -ds * s  # this is negative x
    k2 = -ds * c * jnp.sin(eta)  # CHANGED eta to HFP convention 4-9-2007
    k3 = ds * c * jnp.cos(eta)

    return jnp.array([k1, k2, k3])


tth_eta_to_k = jax.jit(jax.vmap(_tth_eta_to_k, in_axes=[0, 0, None]))


def _k_to_tth_eta(k, wvln):
    """Convert from k-vector to (tth, eta).

    Inverse of ImageD11.transform.compute_k_vectors
    """
    k1, k2, k3 = k
    ds = jnp.linalg.norm(k)
    s = ds * wvln / 2.0  # sin(theta)
    tth = 2.0 * jnp.degrees(jnp.arcsin(s))
    eta = jnp.degrees(jnp.arctan2(-k2, k3))
    return tth, eta


k_to_tth_eta = jax.jit(jax.vmap(_k_to_tth_eta, in_axes=[0, None]))

### K-VECTOR TO G-VECTOR TRANSFORMS


def k_omega_to_g(k, omega, wedge, chi):
    """ImageD11.transform.compute_g_from_k."""
    return lab_to_sample(k, omega, wedge, chi)


def _omega_solns_for_g(g, wavelength, axis, pre, post):
    """Computes omega rotation angles needed for each g to diffract."""
    rg = pre @ g

    beam = jnp.array([-1.0 / wavelength, 0, 0])

    rb = jnp.dot(post.T, beam)

    a1 = jnp.transpose(jnp.cross(axis, rg.T))
    a2 = jnp.transpose(jnp.cross(a1.T, axis))
    a0 = rg - a2

    rbda0 = jnp.sum(rb * a0)
    rbda1 = jnp.sum(rb * a1)
    rbda2 = jnp.sum(rb * a2)

    modg = jnp.sqrt(jnp.sum(g * g))
    kdotbeam = -modg * modg / 2.0

    phi = jnp.arctan2(rbda2, rbda1)
    den = jnp.sqrt(rbda1 * rbda1 + rbda2 * rbda2)
    msk = den <= 0
    quot = (kdotbeam - rbda0) / (den + msk)
    valid = (~msk) & (quot >= -1) & (quot <= 1)
    quot = jnp.where(valid, quot, 0)
    x_plus_p = jnp.arcsin(quot)
    sol1 = x_plus_p + phi
    sol2 = jnp.pi - x_plus_p + phi

    angmod_sol1 = jnp.arctan2(jnp.sin(sol1), jnp.cos(sol1))
    angmod_sol2 = jnp.arctan2(jnp.sin(sol2), jnp.cos(sol2))

    return jnp.degrees(angmod_sol1), jnp.degrees(angmod_sol2), valid


omega_solns_for_g = jax.jit(jax.vmap(_omega_solns_for_g, in_axes=[0, None, None, None, None]))


def _g_to_k_omega(g, wavelength, wedge, chi):
    """Get k-vectors and omega angles from g-vectors.

    There are two solutions for k for each g-vector.

    ImageD11.gv_general.g_to_k
    """
    # wedge and chi matrices
    W = wedgemat(wedge)
    C = chimat(chi)

    post = W @ C

    # work out valid omega angles for g-vectors given a wavelength and rotation axis
    omega1, omega2, valid = _omega_solns_for_g(g, wavelength, jnp.array([0, 0, -1.0]), jnp.eye(3), post)

    # work out k-vectors (in rotated sample frame) with these angles
    k_one = _sample_to_lab(g, omega1, wedge, chi)
    k_two = _sample_to_lab(g, omega2, wedge, chi)

    return [k_one, k_two], [omega1 * valid, omega2 * valid], valid


g_to_k_omega = jax.jit(
    jax.vmap(
        _g_to_k_omega,
        in_axes=[
            0,
            None,
            None,
            None,
        ],
    )
)


# Below are cross-transforms


def g_to_tth_eta_omega(g, wavelength, wedge, chi):
    [k_one, k_two], [omega1, omega2], valid = g_to_k_omega(g, wavelength, wedge, chi)
    tth, eta_one = k_to_tth_eta(k_one, wavelength)
    tth, eta_two = k_to_tth_eta(k_two, wavelength)
    return tth, [eta_one * valid, eta_two * valid], [omega1, omega2]


def tth_eta_omega_to_g(tth, eta, omega, wavelength, wedge, chi):
    k = tth_eta_to_k(tth, eta, wavelength)
    g = k_omega_to_g(k, omega, wedge, chi)
    return g


def _k_to_det(
    k,
    omega,
    origin_sample,
    wedge,
    chi,
    wavelength,
    y_center,
    y_size,
    tilt_y,
    z_center,
    z_size,
    tilt_z,
    tilt_x,
    distance,
    o11,
    o12,
    o21,
    o22,
):
    # xyz = unit vectors along the scattered vectors

    xyz = _k_to_xyz_lab_direc(k, wavelength)

    # Find vectors in the fast, slow directions in the detector plane
    sc = jnp.array([1.0, 0.0, 0])
    fc = jnp.array([0.0, 1.0, 0])

    dxyzl = det_to_xyz_lab(
        sc, fc, y_center, y_size, tilt_y, z_center, z_size, tilt_z, tilt_x, distance, o11, o12, o21, o22
    ).T

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
    norm = jnp.dot(det_norm, xyz)
    # Check for divide by zero
    msk = norm == 0
    norm += msk

    # Intersect ray on detector plane
    sc = jnp.dot(jnp.cross(df, dO), xyz) / norm
    fc = jnp.dot(jnp.cross(dO, ds), xyz) / norm

    go = _sample_to_lab(origin_sample, omega, wedge, chi)
    # project these onto the detector face to give shifts
    sct = (xyz * jnp.cross(df, go.T).T).sum(axis=0) / norm
    fct = (xyz * jnp.cross(go.T, ds).T).sum(axis=0) / norm
    sc -= sct
    fc -= fct

    fc = jnp.where(msk, 0, fc)
    sc = jnp.where(msk, 0, sc)

    return sc, fc


k_to_det = jax.jit(
    jax.vmap(
        _k_to_det,
        in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    )
)


def tth_eta_omega_to_det(
    tth,
    eta,
    omega,
    origin_sample,
    wedge,
    chi,
    wavelength,
    y_center,
    y_size,
    tilt_y,
    z_center,
    z_size,
    tilt_z,
    tilt_x,
    distance,
    o11,
    o12,
    o21,
    o22,
):
    k = tth_eta_to_k(tth, eta, wavelength)
    sc, fc = k_to_det(
        k,
        omega,
        origin_sample,
        wedge,
        chi,
        wavelength,
        y_center,
        y_size,
        tilt_y,
        z_center,
        z_size,
        tilt_z,
        tilt_x,
        distance,
        o11,
        o12,
        o21,
        o22,
    )

    return sc, fc


def tth_eta_omega_to_xyz_lab(
    tth,
    eta,
    omega,
    origin_sample,
    wedge,
    chi,
    y_center,
    y_size,
    tilt_y,
    z_center,
    z_size,
    tilt_z,
    tilt_x,
    distance,
    o11,
    o12,
    o21,
    o22,
):
    sc, fc = tth_eta_omega_to_det(
        tth,
        eta,
        omega,
        origin_sample,
        wedge,
        chi,
        y_center,
        y_size,
        tilt_y,
        z_center,
        z_size,
        tilt_z,
        tilt_x,
        distance,
        o11,
        o12,
        o21,
        o22,
    )
    xyz_lab = det_to_xyz_lab(
        sc, fc, y_center, y_size, tilt_y, z_center, z_size, tilt_z, tilt_x, distance, o11, o12, o21, o22
    )
    return xyz_lab


# there are faster ways to go from xl, yl, zl to k


def _xyz_lab_to_k(xyz_lab, omega, origin_sample, wedge, chi, wavelength):
    """Convert from lab (lx, ly, lz) coordinates to k-vector.

    ImageD11/src/cdiffraction.c
    """
    origin_lab = _sample_to_lab(origin_sample, omega, wedge, chi)
    scatter_vec_lab = xyz_lab - origin_lab
    # normalize this
    scatter_vec_lab = scatter_vec_lab / jnp.linalg.norm(scatter_vec_lab)

    s0 = jnp.array([1.0, 0.0, 0.0])
    k = (scatter_vec_lab - s0) / wavelength

    return k


xyz_lab_to_k = jax.jit(jax.vmap(_xyz_lab_to_k, in_axes=[0, 0, None, None, None, None]))


def _k_to_xyz_lab_direc(k, wavelength):
    """Convert from k-vector to a vector parallel to s.

    Not quite the inverse of ImageD11/src/cdiffraction.c
    """
    scatter_vec_lab = k * wavelength + jnp.array([1.0, 0.0, 0.0])
    return scatter_vec_lab / jnp.linalg.norm(scatter_vec_lab)


def k_to_xyz_lab(
    k,
    omega,
    origin_sample,
    wedge,
    chi,
    wavelength,
    y_center,
    y_size,
    tilt_y,
    z_center,
    z_size,
    tilt_z,
    tilt_x,
    distance,
    o11,
    o12,
    o21,
    o22,
):
    sc, fc = k_to_det(
        k,
        omega,
        origin_sample,
        wedge,
        chi,
        wavelength,
        y_center,
        y_size,
        tilt_y,
        z_center,
        z_size,
        tilt_z,
        tilt_x,
        distance,
        o11,
        o12,
        o21,
        o22,
    )
    xyz_lab = det_to_xyz_lab(
        sc, fc, y_center, y_size, tilt_y, z_center, z_size, tilt_z, tilt_x, distance, o11, o12, o21, o22
    )

    return xyz_lab


def xyz_lab_to_g(xyz_lab, omega, origin_sample, wedge, chi, wavelength):
    k = xyz_lab_to_k(xyz_lab, omega, origin_sample, wedge, chi, wavelength)
    g = k_omega_to_g(k, omega, wedge, chi)
    return g


def g_to_xyz_lab(g, omega, origin_sample, wedge, chi, wavelength):
    [k_one, k_two], [omega1, omega2], _ = g_to_k_omega(g, wavelength, wedge, chi)
    xyz_lab_one = k_to_xyz_lab(k_one, omega1, origin_sample, wedge, chi, wavelength)
    xyz_lab_two = k_to_xyz_lab(k_two, omega2, origin_sample, wedge, chi, wavelength)

    return xyz_lab_one, xyz_lab_two


def det_to_g(
    sc,
    fc,
    omega,
    origin_sample,
    y_center,
    y_size,
    tilt_y,
    z_center,
    z_size,
    tilt_z,
    tilt_x,
    distance,
    o11,
    o12,
    o21,
    o22,
    wedge,
    chi,
    wavelength,
):
    """Convert detector (sc, fc) coordinates to g-vectors."""
    xyz_lab = det_to_xyz_lab(
        sc,
        fc,
        y_center,
        y_size,
        tilt_y,
        z_center,
        z_size,
        tilt_z,
        tilt_x,
        distance,
        o11,
        o12,
        o21,
        o22,
    )
    k = xyz_lab_to_k(xyz_lab, omega, origin_sample, wedge, chi, wavelength)
    g = k_omega_to_g(k, omega, wedge, chi)
    return g


def g_to_det(
    g,
    omega,
    origin_sample,
    y_center,
    y_size,
    tilt_y,
    z_center,
    z_size,
    tilt_z,
    tilt_x,
    distance,
    o11,
    o12,
    o21,
    o22,
    wedge,
    chi,
    wavelength,
):
    xyz_lab_one, xyz_lab_two = g_to_xyz_lab(g, omega, origin_sample, wedge, chi, wavelength)
    sc_one, fc_one = xyz_lab_to_det(
        xyz_lab_one,
        y_center,
        y_size,
        tilt_y,
        z_center,
        z_size,
        tilt_z,
        tilt_x,
        distance,
        o11,
        o12,
        o21,
        o22,
    )

    sc_two, fc_two = xyz_lab_to_det(
        xyz_lab_two,
        y_center,
        y_size,
        tilt_y,
        z_center,
        z_size,
        tilt_z,
        tilt_x,
        distance,
        o11,
        o12,
        o21,
        o22,
    )

    return (sc_one, fc_one), (sc_two, fc_two)
