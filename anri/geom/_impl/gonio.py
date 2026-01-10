import jax
import jax.numpy as jnp

from .utils import rmat_from_axis_angle


@jax.jit
def chimat(chi):
    """Return rotation matrix for rotation about x axis by chi (degrees).

    ImageD11.gv_general.chimat
    """
    # negative rotation about x-axis
    return rmat_from_axis_angle(jnp.array([-1.0, 0.0, 0.0]), chi)


@jax.jit
def wedgemat(wedge):
    """Return rotation matrix for rotation about y axis by wedge (degrees).

    ImageD11.gv_general.wedgemat
    """
    return rmat_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), wedge)


@jax.jit
def sample_to_lab(v_sample, omega, wedge, chi):
    """Convert from sample to lab coordinates (apply the diffractometer stack).

    v_lab = W.T @ C.T @ R.T @ v_sample
    Adapted from ImageD11.transform.compute_g_from_k and compute_grain_origins
    """
    W = wedgemat(wedge)
    C = chimat(chi)

    R = rmat_from_axis_angle(jnp.array([0.0, 0.0, -1.0]), omega)

    v_lab = W.T @ C.T @ R.T @ v_sample

    return v_lab


@jax.jit
def lab_to_sample(v_lab, omega, wedge, chi):
    """Convert from lab to sample coordinates (apply the diffractometer stack).

    v_sample = R @ C @ W @ v_lab
    Adapted from ImageD11.transform.compute_g_from_k and compute_grain_origins
    Equivalent to ImageD11.transform.compute_g_from_k
    t_x, t_y, t_z are in the sample frame!
    """
    W = wedgemat(wedge)
    C = chimat(chi)

    R = rmat_from_axis_angle(jnp.array([0.0, 0.0, -1.0]), omega)

    v_sample = R @ C @ W @ v_lab

    return v_sample
