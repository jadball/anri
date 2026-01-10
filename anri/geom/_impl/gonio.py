"""Functions to convert between goniometer and laboratory frames."""

import jax

from .utils import rot_x, rot_y, rot_z


@jax.jit
def chimat(chi: float) -> jax.Array:
    """Get the rotation matrix that applies the chi motor (roll the gonio around the beam).

    Equivalent to :func:`ImageD11.gv_general.chimat`

    Parameters
    ----------
    chi
        Chi motor angle (degrees)

    Returns
    -------
    jax.Array
        [3,3] Rotation matrix by applying the chi motor
    """
    # negative rotation about x-axis
    return -rot_x(chi)


@jax.jit
def wedgemat(wedge: float) -> jax.Array:
    """Get the rotation matrix that applies the wedge (roll the gonio around the y axis).

    Equivalent to :func:`ImageD11.gv_general.wedgemat`

    Parameters
    ----------
    wedge
        Wedge motor angle (degrees)

    Returns
    -------
    jax.Array
        [3,3] Rotation matrix by applying the wedge motor
    """
    return rot_y(wedge)


@jax.jit
def sample_to_lab(v_sample: jax.Array, omega: float, wedge: float, chi: float) -> jax.Array:
    r"""Convert from sample to lab coordinates (apply the diffractometer stack).

    Adapted from :func:`ImageD11.transform.compute_g_from_k` and :func:`ImageD11.transform.compute_grain_origins`

    Parameters
    ----------
    v_sample
        [3] Vector in sample coordinates
    omega
        Omega motor value (degrees)
    wedge
        Wedge motor value (degrees)
    chi
        Chi motor value (degrees)

    Returns
    -------
    v_lab: jax.Array
        [3] Vector in lab coordinates

    Notes
    -----
    Given rotation matrices $\matr{W}$, $\matr{C}$, $\matr{R}$ for wedge, chi and omega motors respectively:

    $$\vec{v_{\text{lab}}} = \matr{W^\dagger} \cdot \matr{C^\dagger} \cdot \matr{R^\dagger} \cdot \vec{v_{\text{sample}}}$$
    """
    W = wedgemat(wedge)
    C = chimat(chi)

    R = -rot_z(omega)

    v_lab = W.T @ C.T @ R.T @ v_sample

    return v_lab


@jax.jit
def lab_to_sample(v_lab: jax.Array, omega: float, wedge: float, chi: float) -> jax.Array:
    r"""Convert from lab to sample coordinates (apply the diffractometer stack).

    Adapted from :func:`ImageD11.transform.compute_g_from_k` and :func:`ImageD11.transform.compute_grain_origins`

    Parameters
    ----------
    v_lab
        [3] Vector in lab coordinates
    omega
        Omega motor value (degrees)
    wedge
        Wedge motor value (degrees)
    chi
        Chi motor value (degrees)

    Returns
    -------
    v_sample: jax.Array
        [3] Vector in sample coordinates

    Notes
    -----
    Given rotation matrices $\matr{W}$, $\matr{C}$, $\matr{R}$ for wedge, chi and omega motors respectively:

    $$\vec{v_{\text{sample}}} = \matr{R} \cdot \matr{C} \cdot \matr{W} \cdot \vec{v_{\text{lab}}}$$
    """
    W = wedgemat(wedge)
    C = chimat(chi)

    R = -rot_z(omega)

    v_sample = R @ C @ W @ v_lab

    return v_sample
