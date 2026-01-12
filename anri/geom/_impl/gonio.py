"""Functions to convert between goniometer and laboratory frames."""

import jax
import jax.numpy as jnp

from .utils import rot_x, rot_y, rot_z


@jax.jit
def sample_to_lab(v_sample: jax.Array, omega: float, wedge: float, chi: float, dty: float, y0: float) -> jax.Array:
    r"""Convert from sample to lab coordinates (apply the diffractometer stack).

    Adapted from :func:`ImageD11.transform.compute_g_from_k` and :func:`ImageD11.transform.compute_grain_origins`.
    See :ref:`tut_geom` for more details about our geometry.

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
    dty:
        Base diffractometer Y translation value (same units as v_sample)
    y0:
        The true value of dty when the rotation axis (untilted by wedge, chi) intersects the beam

    Returns
    -------
    v_lab: jax.Array
        [3] Vector in lab coordinates

    Notes
    -----
    Given right-handed rotation matrices $\matr{W}$, $\matr{C}$, $\matr{R}$ for wedge, chi and omega motors respectively, which all follow:

    $$\matr{M} \cdot \vec{v_{\text{sample}}} = \vec{v_{\text{lab}}}$$

    Then we get:

    $$\vec{v_{\text{lab}}} = \left(0, \text{dty} - y_0, 0\right) + \matr{W} \cdot \matr{C} \cdot \matr{R} \cdot \vec{v_{\text{sample}}}$$
    """
    v_dty = jnp.array([0.0, dty - y0, 0.0])

    C = rot_x(chi)
    W = rot_y(-wedge)
    R = rot_z(omega)

    v_lab = v_dty + (W @ C @ R @ v_sample)

    return v_lab


@jax.jit
def lab_to_sample(v_lab: jax.Array, omega: float, wedge: float, chi: float, dty: float, y0: float) -> jax.Array:
    r"""Convert from lab to sample coordinates (apply the diffractometer stack).

    Adapted from :func:`ImageD11.transform.compute_g_from_k` and :func:`ImageD11.transform.compute_grain_origins`.
    See :ref:`tut_geom` for more details about our geometry.

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
    dty:
        Base diffractometer Y translation value (same units as v_sample)
    y0:
        The true value of dty when the rotation axis (untilted by wedge, chi) intersects the beam

    Returns
    -------
    v_sample: jax.Array
        [3] Vector in sample coordinates

    Notes
    -----
    Given right-handed rotation matrices $\matr{W}$, $\matr{C}$, $\matr{R}$ for wedge, chi and omega motors respectively, which all follow:

    $$\matr{M} \cdot \vec{v_{\text{sample}}} = \vec{v_{\text{lab}}}$$

    Then we get:

    $$\vec{v_{\text{sample}}} = \matr{R^\dagger} \cdot \matr{C^\dagger} \cdot \matr{W^\dagger} \cdot \left(\vec{v_{\text{lab}}} - \left(0, \text{dty} - y_0, 0\right)\right)$$
    """
    v_dty = jnp.array([0.0, dty - y0, 0.0])

    C = rot_x(chi)
    W = rot_y(-wedge)
    R = rot_z(omega)

    v_sample = R.T @ C.T @ W.T @ (v_lab - v_dty)

    return v_sample
