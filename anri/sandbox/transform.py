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
    """Convert from sample to lab coordinates
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


def k_omega_to_g(k, omega, wedge, chi):
    return sample_to_lab(k, omega, wedge, chi)


def _omega_solns_for_g(g, wavelength, axis, pre, post):
    """Computes omega rotation angles needed for each g to diffract"""
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


def _g_to_tth_eta_omega(g, wavelength, wedge, chi):
    # wedge and chi matrices
    W = wmat(wedge)
    C = chimat(chi)

    post = W @ C

    # work out valid omega angles for g-vectors given a wavelength and rotation axis
    omega1, omega2, valid = _omega_solns_for_g(g, wavelength, jnp.array([0, 0, -1.0]), jnp.eye(3), post)

    # work out k-vectors (in rotated sample frame) with these angles
    k_one = _lab_to_sample(g, omega1, wedge, chi)
    k_two = _lab_to_sample(g, omega2, wedge, chi)

    eta_one = jnp.arctan2(-k_one[1], k_one[2])
    eta_two = jnp.arctan2(-k_two[1], k_two[2])

    ds = jnp.sqrt(jnp.sum(g * g))
    s = ds * wavelength / 2.0  # sin theta
    tth = jnp.degrees(jnp.arcsin(s) * 2.0) * valid
    eta1 = jnp.degrees(eta_one) * valid
    eta2 = jnp.degrees(eta_two) * valid
    omega1 = omega1 * valid
    omega2 = omega2 * valid
    return tth, [eta1, eta2], [omega1, omega2]


def g_to_tth_eta_omega(tth, eta, omega, wavelength, wedge, chi):
    k = k_from_tth_eta(tth, eta, wavelength)
    g = k_omega_to_g(k, omega, wedge, chi)
    return g


rmat_from_axis_angle = jax.jit(jax.vmap(_rmat_from_axis_angle, in_axes=[None, 0]))

det_to_xyz_lab = jax.jit(
    jax.vmap(_det_to_xyz_lab, in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None, None])
)
xyz_lab_to_det = jax.jit(
    jax.vmap(_xyz_lab_to_det, in_axes=[0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None])
)

sample_to_lab = jax.jit(jax.vmap(_sample_to_lab, in_axes=[0, 0, None, None]))
lab_to_sample = jax.jit(jax.vmap(_lab_to_sample, in_axes=[0, 0, None, None]))
omega_solns_for_g = jax.jit(jax.vmap(_omega_solns_for_g, in_axes=[0, None, None, None, None]))
k_from_tth_eta = jax.jit(jax.vmap(_tth_eta_to_k, in_axes=[0, 0, None]))
tth_eta_omega_from_g = jax.jit(
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
