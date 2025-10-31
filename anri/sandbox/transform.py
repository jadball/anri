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


@jax.jit
def detector_rotation_matrix(tilt_x, tilt_y, tilt_z):
    """Return rotation matrix for detector tilts about x, y, z axes.

    R1 = Z, R2 = Y, R3 = X
    tilt_x, tilt_y, tilt_z are in radians
    but chi and wedge are in degrees
    """
    R1 = _rmat_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.degrees(tilt_z))
    R2 = _rmat_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), jnp.degrees(tilt_y))
    R3 = _rmat_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.degrees(tilt_x))

    # combine in order R3 @ R2 @ R1
    return R3 @ R2 @ R1


@jax.jit
def chimat(chi):
    """Return rotation matrix for rotation about x axis by chi (degrees)."""
    # negative rotation about x-axis
    return _rmat_from_axis_angle(jnp.array([-1.0, 0.0, 0.0]), chi)


@jax.jit
def wmat(wedge):
    """Return rotation matrix for rotation about y axis by wedge (degrees)."""
    return _rmat_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), wedge)


@jax.jit
def detector_orientation_matrix(o11, o12, o21, o22):
    """Return (3D) detector orientation matrix from 2D orientation elements."""
    return jnp.array([[o11, o12, 0], [o21, o22, 0], [0, 0, 1]])


@jax.jit
def detector_transforms(y_center, y_size, tilt_y, z_center, z_size, tilt_z, tilt_x, distance, o11, o12, o21, o22):
    """Return required transformation matrices and shifts to convert between detector and lab coordinates.

    v_lab = (det_tilts @ (cob_matrix @ (det_flips @ (pixel_size_scale @ (v_det + beam_cen_shift))))) + x_distance_shift
    v_lab = M(v_det + beam_cen_shift) + x_distance_shift
    v_det = M^-1(v_lab - x_distance_shift) - beam_cen_shift.
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


def _det_to_xyz_lab(sc, fc, y_center, y_size, tilt_y, z_center, z_size, tilt_z, tilt_x, distance, o11, o12, o21, o22):
    """Convert detector (sc, fc) coordinates to lab (lx, ly, lz) coordinates."""
    M, beam_cen_shift, x_distance_shift = detector_transforms(
        y_center, y_size, tilt_y, z_center, z_size, tilt_z, tilt_x, distance, o11, o12, o21, o22
    )
    v_det = jnp.array([sc, fc, 0])
    v_lab = M @ (v_det + beam_cen_shift) + x_distance_shift
    return v_lab


def _xyz_lab_to_det(
    xl, yl, zl, y_center, y_size, tilt_y, z_center, z_size, tilt_z, tilt_x, distance, o11, o12, o21, o22
):
    """Convert lab (lx, ly, lz) coordinates to detector (sc, fc) coordinates."""
    M, beam_cen_shift, x_distance_shift = detector_transforms(
        y_center, y_size, tilt_y, z_center, z_size, tilt_z, tilt_x, distance, o11, o12, o21, o22
    )
    v_lab = jnp.array([xl, yl, zl])
    v_det = jnp.linalg.inv(M) @ (v_lab - x_distance_shift) - beam_cen_shift
    return v_det[:2]


### functions for vectorization (not jitted yet)
# each of these are written for one vector only
# private versions are for a single vector
# public versions are for many vectors


def _lab_to_sample(vec_lab, omega, wedge, chi):
    """Convert from lab to sample coordinates.

    v_sample = W.T @ C.T @ R.T @ v_lab.
    """
    W = wmat(wedge)
    C = chimat(chi)

    R = _rmat_from_axis_angle(jnp.array([0.0, 0.0, -1.0]), omega)

    vec_sample = W.T @ C.T @ R.T @ vec_lab

    return vec_sample


def _sample_to_lab(vec_sample, omega, wedge, chi):
    """Convert from sample to lab coordinates.

    v_lab = R @ C @ W @ v_sample
    """
    W = wmat(wedge)
    C = chimat(chi)

    R = _rmat_from_axis_angle(jnp.array([0.0, 0.0, -1.0]), omega)

    vec_lab = R @ C @ W @ vec_sample

    return vec_lab


def _tth_eta_to_k(tth, eta, wvln):
    tth = jnp.radians(tth)
    eta = jnp.radians(eta)
    c = jnp.cos(tth / 2)  # cos theta
    s = jnp.sin(tth / 2)  # sin theta
    ds = 2 * s / wvln

    k1 = -ds * s  # this is negative x
    k2 = -ds * c * jnp.sin(eta)  # CHANGED eta to HFP convention 4-9-2007
    k3 = ds * c * jnp.cos(eta)

    return jnp.array([k1, k2, k3])


def _k_to_tth_eta(k, wvln):
    k1, k2, k3 = k
    ds = jnp.sqrt(k1**2 + k2**2 + k3**2)
    s = ds * wvln / 2.0  # sin(theta)
    tth = 2.0 * jnp.degrees(jnp.arcsin(s))
    eta = jnp.degrees(jnp.arctan2(-k2, k3))
    return tth, eta


def k_omega_to_g(k, omega, wedge, chi):
    return sample_to_lab(k, omega, wedge, chi)


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


def _g_to_k_omega(g, wavelength, wedge, chi):
    """Get k-vectors and omega angles from g-vectors."""
    # wedge and chi matrices
    W = wmat(wedge)
    C = chimat(chi)

    post = W @ C

    # work out valid omega angles for g-vectors given a wavelength and rotation axis
    omega1, omega2, valid = _omega_solns_for_g(g, wavelength, jnp.array([0, 0, -1.0]), jnp.eye(3), post)

    # work out k-vectors (in rotated sample frame) with these angles
    k_one = _lab_to_sample(g, omega1, wedge, chi)
    k_two = _lab_to_sample(g, omega2, wedge, chi)

    return [k_one, k_two], [omega1 * valid, omega2 * valid], valid


def _g_to_tth_eta_omega(g, wavelength, wedge, chi):
    [k_one, k_two], [omega1, omega2], valid = _g_to_k_omega(g, wavelength, wedge, chi)
    tth, eta_one = _k_to_tth_eta(k_one, wavelength)
    tth, eta_two = _k_to_tth_eta(k_two, wavelength)

    # eta_one = jnp.arctan2(-k_one[1], k_one[2])
    # eta_two = jnp.arctan2(-k_two[1], k_two[2])

    # ds = jnp.sqrt(jnp.sum(g * g))
    # s = ds * wavelength / 2.0  # sin theta
    # tth = jnp.degrees(jnp.arcsin(s) * 2.0) * valid
    # eta1 = jnp.degrees(eta_one) * valid
    # eta2 = jnp.degrees(eta_two) * valid
    # omega1 = omega1 * valid
    # omega2 = omega2 * valid
    return tth, [eta_one * valid, eta_two * valid], [omega1, omega2]


def tth_eta_omega_to_g(tth, eta, omega, wavelength, wedge, chi):
    k = tth_eta_to_k(tth, eta, wavelength)
    g = k_omega_to_g(k, omega, wedge, chi)
    return g


def _xyz_lab_to_tth_eta(xyz_lab, omega, origin_lab, wedge, chi):
    """Compute tth and eta from lab xyx coordinates on the detector and diffraction origins in the sample frame."""
    origin_lab = _lab_to_sample(origin_lab, omega, wedge, chi)
    scatter_vec_lab = xyz_lab - origin_lab
    eta = jnp.degrees(jnp.arctan2(-scatter_vec_lab[1], scatter_vec_lab[2]))
    s1_perp_x = jnp.sqrt(scatter_vec_lab[1] * scatter_vec_lab[1] + scatter_vec_lab[2] * scatter_vec_lab[2])
    tth = jnp.degrees(jnp.arctan2(s1_perp_x, scatter_vec_lab[0]))
    return tth, eta


def _tth_eta_omega_to_det(
    tth,
    eta,
    omega,
    origin_lab,
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
    # xyz = unit vectors along the scattered vectors

    rtth = jnp.radians(tth)
    reta = jnp.radians(eta)

    xyz = jnp.array([jnp.cos(rtth), -jnp.sin(rtth) * jnp.sin(reta), jnp.sin(rtth) * jnp.cos(reta)])

    # xyz[0, :] = jnp.cos(rtth)
    # #  eta = np.degrees(np.arctan2(-s1[1, :], s1[2, :]))
    # xyz[1, :] = -jnp.sin(rtth) * jnp.sin(reta)
    # xyz[2, :] = jnp.sin(rtth) * jnp.cos(reta)

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

    go = _lab_to_sample(origin_lab, omega, wedge, chi)
    # project these onto the detector face to give shifts
    sct = (xyz * jnp.cross(df, go.T).T).sum(axis=0) / norm
    fct = (xyz * jnp.cross(go.T, ds).T).sum(axis=0) / norm
    sc -= sct
    fc -= fct

    fc = jnp.where(msk, 0, fc)
    sc = jnp.where(msk, 0, sc)

    return sc, fc


def tth_eta_omega_to_xyz_lab(
    tth,
    eta,
    omega,
    origin_lab,
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
        origin_lab,
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


def xyz_lab_to_k(xyz_lab, omega, origin_sample, wedge, chi, wavelength):
    tth, eta = xyz_lab_to_tth_eta(xyz_lab, omega, origin_sample, wedge, chi)
    k = tth_eta_to_k(tth, eta, wavelength)

    return k


def k_to_xyz_lab(
    k,
    omega,
    origin_lab,
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
    tth, eta = _k_to_tth_eta(k, wavelength)
    xyz_lab = tth_eta_omega_to_xyz_lab(
        tth,
        eta,
        omega,
        origin_lab,
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
    return xyz_lab


def xyz_lab_to_g(xyz_lab, omega, origin_sample, wedge, chi, wavelength):
    tth, eta = xyz_lab_to_tth_eta(xyz_lab, omega, origin_sample, wedge, chi)
    k = tth_eta_to_k(tth, eta, wavelength)
    g = k_omega_to_g(k, omega, wedge, chi)
    return g


def g_to_xyz_lab(
    g,
    omega,
    origin_lab,
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
    tth, [eta_one, eta_two], [omega1, omega2] = g_to_tth_eta_omega(g, wavelength, wedge, chi)

    xyz_lab_one = tth_eta_omega_to_xyz_lab(
        tth,
        eta_one,
        omega1,
        origin_lab,
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

    xyz_lab_two = tth_eta_omega_to_xyz_lab(
        tth,
        eta_two,
        omega2,
        origin_lab,
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
    tth, eta = xyz_lab_to_tth_eta(xyz_lab, omega, origin_sample, wedge, chi)
    k = tth_eta_to_k(tth, eta, wavelength)
    g = k_omega_to_g(k, omega, wedge, chi)
    return g


def g_to_det(
    g,
    omega,
    origin_lab,
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
    xyz_lab_one, xyz_lab_two = g_to_xyz_lab(
        g,
        omega,
        origin_lab,
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


rmat_from_axis_angle = jax.jit(jax.vmap(_rmat_from_axis_angle, in_axes=[None, 0]))

det_to_xyz_lab = jax.jit(
    jax.vmap(_det_to_xyz_lab, in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None, None])
)
xyz_lab_to_det = jax.jit(
    jax.vmap(_xyz_lab_to_det, in_axes=[0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None])
)


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


def tth_eta_omega_to_det(
    tth,
    eta,
    omega,
    origin_lab,
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
    if origin_lab.shape == (3,):
        return jax.jit(
            jax.vmap(
                _tth_eta_omega_to_det,
                in_axes=[
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            )
        )(
            tth,
            eta,
            omega,
            origin_lab,
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
    else:
        return jax.jit(
            jax.vmap(
                _tth_eta_omega_to_det,
                in_axes=[
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            )
        )(
            tth,
            eta,
            omega,
            origin_lab,
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


sample_to_lab = jax.jit(jax.vmap(_sample_to_lab, in_axes=[0, 0, None, None]))
lab_to_sample = jax.jit(jax.vmap(_lab_to_sample, in_axes=[0, 0, None, None]))
omega_solns_for_g = jax.jit(jax.vmap(_omega_solns_for_g, in_axes=[0, None, None, None, None]))
tth_eta_to_k = jax.jit(jax.vmap(_tth_eta_to_k, in_axes=[0, 0, None]))
g_to_tth_eta_omega = jax.jit(
    jax.vmap(
        _g_to_tth_eta_omega,
        in_axes=[
            0,
            None,
            None,
            None,
        ],
    )
)

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
