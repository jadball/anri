import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


# transformations to implement:
# (x_l, y_l, z_l) from (sc, fc) and detector params (ImageD11.transform.compute_xyz_lab)
# (tth, eta) from (sc, fc) and detector params (ImageD11.transform.compute_tth_eta)
# (tth, eta) from (x_l, y_l, z_l) and detector params ((ImageD11.transform.compute_tth_eta_from_xyz)
# (x_l, y_l, z_l) from (tth, eta) and detector params and crystal translations

### scalar functions


@jax.jit
def detector_rotation_matrix(tilt_x, tilt_y, tilt_z):
    # R1 = Z, R2 = Y, R3 = X
    # tilt_x, tilt_y, tilt_z are in radians
    # but chi and wedge are in degrees
    R1 = _rmat_from_axis_angle(jnp.array([0., 0., 1.]), jnp.degrees(tilt_z))
    R2 = _rmat_from_axis_angle(jnp.array([0., 1., 0.]), jnp.degrees(tilt_y))
    R3 = _rmat_from_axis_angle(jnp.array([1., 0., 0.]), jnp.degrees(tilt_x))

    # combine in order R3 @ R2 @ R1
    return R3 @ R2 @ R1


def _rmat_from_axis_angle(axis, angle):
    rom = jnp.radians(angle)
    som = jnp.sin(rom)
    com = jnp.cos(rom)
    C = 1 - com
    # normalise axis
    axis = axis / jnp.linalg.norm(axis)
    x, y, z = axis

    Q = jnp.array([
        [x*x*C +   com, x*y*C - z*som, x*z*C + y*som],
        [y*x*C + z*som, y*y*C +   com, y*z*C - x*som],
        [z*x*C - y*som, z*y*C + x*som, z*z*C +   com]
        ])
    return Q

rmat_from_axis_angle = jax.jit(jax.vmap(_rmat_from_axis_angle, in_axes=[None, 0]))


def chimat(chi):
    # rotation about -x-axis
    return _rmat_from_axis_angle(jnp.array([-1., 0., 0.]), chi)

def wmat(wedge):
    # rotation about y-axis
    return _rmat_from_axis_angle(jnp.array([0., 1., 0.]), wedge)


### functions for vectorization (not jitted yet)
# each of these are written for one vector only
# private versions are for a single vector
# public versions are for many vectors


@jax.jit
def get_detector_affine(y_center, y_size, tilt_y,
                         z_center, z_size, tilt_z,
                         tilt_x,
                         distance,
                         o11, o12, o21, o22):
    """
    Gets a single 4x4 affine transformation matrix that maps from (sc, fc) to (xl, yl, zl) via:
    v_det = (sc, fc, 1, 1)  # homogeneous 2d in 3d
    (xl, yl, zl) = det_affine @ v_det
    """
    # most of this transform is a series of 2D affines (3x3 homogeneous matrices):
    r2r1 = detector_rotation_matrix(tilt_x, tilt_y, tilt_z)

    shift = jnp.array([[1., 0, -z_center],
                       [0., 1, -y_center],
                       [0., 0,         1]])

    scale = jnp.array([[z_size,      0, 0],
                       [    0., y_size, 0],
                       [    0.,      0, 1]])

    orient = jnp.array([[o11, o12, 0],
                        [o21, o22, 0],
                        [  0,   0, 1]])

    lift = jnp.array([[0., 0., 0.],
                      [0., 1., 0.],
                      [1., 0., 0.]])

    linear3x3 = r2r1 @ lift @ orient @ scale @ shift

    # 4x4 affine including translation along x
    det_affine = jnp.eye(4)
    det_affine = det_affine.at[:3,:3].set(linear3x3)
    det_affine = det_affine.at[0,3].set(distance)

    return det_affine


def _xyz_lab_from_sc_fc_aff(sc, fc,
                     y_center, y_size, tilt_y,
                     z_center, z_size, tilt_z,
                     tilt_x,
                     distance,
                     o11, o12, o21, o22):

    # homogeneous input vector, 2D detector coordinates
    v_det_h = jnp.array([sc, fc, 1, 1])

    # get affine detector transformation matrix
    det_affine = get_detector_affine(y_center, y_size, tilt_y,
                                     z_center, z_size, tilt_z,
                                     tilt_x,
                                     distance,
                                     o11, o12, o21, o22)

    # this is (xl, yl, zl, 1)
    final = det_affine @ v_det_h

    return final[:3]


xyz_lab_from_sc_fc = jax.jit(jax.vmap(_xyz_lab_from_sc_fc_aff, in_axes=[
                                                                0, 0,
                                                                None, None, None,
                                                                None, None, None,
                                                                None,
                                                                None,
                                                                None, None, None, None
                                                            ]))


def _lab_to_sample(vec_lab, omega, wedge, chi):
    """
    Convert from lab to sample coordinates
    v_sample = W.T @ C.T @ R.T @ v_lab
    """
    W = wmat(wedge)
    C = chimat(chi)

    R = _rmat_from_axis_angle(jnp.array([0.,0.,-1.]), omega)

    vec_sample = W.T @ C.T @ R.T @ vec_lab

    return vec_sample

lab_to_sample = jax.jit(jax.vmap(_lab_to_sample, in_axes=[0, 0, None, None]))


def _sample_to_lab(vec_sample, omega, wedge, chi):
    """
    Convert from sample to lab coordinates
    v_lab = R @ C @ W @ v_sample
    """
    W = wmat(wedge)
    C = chimat(chi)

    R = _rmat_from_axis_angle(jnp.array([0.,0.,-1.]), omega)

    vec_lab = R @ C @ W @ vec_sample

    return vec_lab

sample_to_lab = jax.jit(jax.vmap(_sample_to_lab, in_axes=[0, 0, None, None]))


def _k_from_tth_eta(tth, eta, wvln):
    tth = jnp.radians(tth)
    eta = jnp.radians(eta)
    c = jnp.cos(tth / 2)  # cos theta
    s = jnp.sin(tth / 2)  # sin theta
    ds = 2 * s / wvln

    k1 = -ds * s  # this is negative x
    k2 = -ds * c * jnp.sin(eta)  # CHANGED eta to HFP convention 4-9-2007
    k3 =  ds * c * jnp.cos(eta)

    return jnp.array([k1, k2, k3])

k_from_tth_eta = jax.jit(jax.vmap(_k_from_tth_eta, in_axes=[0, 0, None]))


# just sample to lab
# 
# def _g_from_k_omega(k, omega, axis, pre, post):
#     """
#     g = pre . rot(axis, angle) . post . k
#     g = omega . chi . wedge . k
#     """
#     rot = _rmat_from_axis_angle(axis, omega)
#     g = pre @ rot @ post @ k
#     return g

# g_from_k_omega = jax.jit(jax.vmap(_g_from_k_omega, in_axes=[0, 0, None, None, None]))

# def _g_from_k_omega_pars(k, omega, wedge, chi):
#     # this whole thing is just sample to lab
#     # """
#     # Common case - compute g from k vectors given wedge, chi in degrees
#     # This is doing g = R @ C @ W @ k
#     # """
#     # # axis is normally -Z - why?
#     # axis = jnp.array([0., 0., -1])

#     # W = wmat(wedge)
#     # C = chimat(chi)

#     # pre = jnp.eye(3)
#     # post = C @ W
#     # g = _g_from_k_omega(k, omega, axis, pre, post)

#     g = _sample_to_lab(k, omega, wedge, chi)
    
#     return g

# g_from_k_omega_pars = jax.jit(jax.vmap(_g_from_k_omega_pars, in_axes=[0, 0, None, None]))


def g_from_k_omega(k, omega, wedge, chi):
    return sample_to_lab(k, omega, wedge, chi)


def _omega_solns_for_g(g,
             wavelength,
             axis,
             pre,
             post):
    """
    Computes omega rotation angles needed for each g to diffract
    """
    rg = pre @ g

    beam = jnp.array([-1.0/wavelength, 0, 0])

    rb = jnp.dot(post.T, beam)

    a1 = jnp.transpose(jnp.cross(axis, rg.T))
    a2 = jnp.transpose(jnp.cross(a1.T, axis))
    a0 = rg - a2

    rbda0 = jnp.sum(rb * a0)
    rbda1 = jnp.sum(rb * a1)
    rbda2 = jnp.sum(rb * a2)

    modg = jnp.sqrt(jnp.sum(g * g))
    kdotbeam = -modg*modg/2.

    phi = jnp.arctan2(rbda2, rbda1)
    den = jnp.sqrt(rbda1*rbda1 + rbda2*rbda2)
    msk = (den <= 0)
    quot = (kdotbeam - rbda0)/(den + msk)
    valid = (~msk) & (quot >= -1) & (quot <= 1)
    quot = jnp.where(valid, quot, 0) 
    x_plus_p = jnp.arcsin(quot)
    sol1 = x_plus_p + phi
    sol2 = jnp.pi - x_plus_p + phi

    angmod_sol1 = jnp.arctan2(jnp.sin(sol1), jnp.cos(sol1))
    angmod_sol2 = jnp.arctan2(jnp.sin(sol2), jnp.cos(sol2))
    
    return jnp.degrees(angmod_sol1), jnp.degrees(angmod_sol2), valid

omega_solns_for_g = jax.jit(jax.vmap(_omega_solns_for_g, in_axes=[0, None, None, None, None]))


def _tth_eta_omega_from_g(g, wavelength, wedge, chi):
    # wedge and chi matrices
    W = wmat(wedge)
    C = chimat(chi)
    
    post = W @ C

    omega1, omega2, valid = _k_from_g(g, wavelength, jnp.array([0,0,-1.]), jnp.eye(3), post)

    k_one = _lab_to_sample(g, omega1, wedge, chi)
    k_two = _lab_to_sample(g, omega2, wedge, chi)

    eta_one = jnp.arctan2(-k_one[1], k_one[2])
    eta_two = jnp.arctan2(-k_two[1], k_two[2])

    ds = jnp.sqrt(jnp.sum(g * g))
    s = ds * wavelength / 2.0  # sin theta
    tth = jnp.degrees(jnp.arcsin(s) * 2.) * valid
    eta1 = jnp.degrees(eta_one) * valid
    eta2 = jnp.degrees(eta_two) * valid
    omega1 = omega1 * valid
    omega2 = omega2 * valid
    return tth, [eta1, eta2], [omega1, omega2]

tth_eta_omega_from_g = jax.jit(jax.vmap(_tth_eta_omega_from_g, in_axes=[0, None, None, None,]))


def g_from_tth_eta_omega(tth, eta, omega, wavelength, wedge, chi):
    k = k_from_tth_eta(tth, eta, wavelength)
    g = g_from_k_omega(k, omega, wedge, chi)
    return g