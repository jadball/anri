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
    W = rot_y(wedge)
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

    $$\vec{v_{\text{sample}}} = \matr{R^T} \cdot \matr{C^T} \cdot \matr{W^T} \cdot \left(\vec{v_{\text{lab}}} - \left(0, \text{dty} - y_0, 0\right)\right)$$
    """
    v_dty = jnp.array([0.0, dty - y0, 0.0])

    C = rot_x(chi)
    W = rot_y(wedge)
    R = rot_z(omega)

    v_sample = R.T @ C.T @ W.T @ (v_lab - v_dty)

    return v_sample


@jax.jit
def find_dty_for_beam_xy(
    v_sample: jax.Array, k_in_lab: jax.Array, omega: float, wedge: float, chi: float, y0: float
) -> jax.Array:
    """Find the dty value required to make the beam intersect a specific point v_sample at a given omega.

    This is only valid for the scanning case (beam can be approximated as a ray).

    Parameters
    ----------
    v_sample
        [3] Vector in sample coordinates
    k_in_lab
        [3] Incoming wave-vector in lab frame
    omega
        Omega motor value (degrees)
    wedge
        Wedge motor value (degrees)
    chi
        Chi motor value (degrees)
    y0:
        The true value of dty when the rotation axis (untilted by wedge, chi) intersects the beam

    Returns
    -------
    dty_required: float
        dty value that brings v_sample into beam at angle omega
    """
    # Rotate v_sample into v_lab, ignoring y0 and dty for now (just find angles)
    v_lab = sample_to_lab(v_sample, omega, wedge, chi, 0.0, 0.0)

    # Find slope of ray in lab frame
    beam_slope = k_in_lab[1] / k_in_lab[0]

    # Find y value of ray at lab x coordinate
    y_ray = beam_slope * v_lab[0]

    # Find dty required
    # When dty = y0, y_ray
    dty_required = y_ray - v_lab[1] + y0

    return dty_required
