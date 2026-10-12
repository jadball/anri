"""Functions to convert between goniometer and laboratory frames."""

import jax
import jax.numpy as jnp

from .utils import beam_offsets, rot_x, rot_y, rot_z


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
    dty
        Base diffractometer Y translation value (same units as v_sample)
    y0
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

    # Parenthesised right to left so every step is a matrix-vector product.
    v_lab = v_dty + W @ (C @ (R @ v_sample))

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
    dty
        Base diffractometer Y translation value (same units as v_sample)
    y0
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

    # Parenthesised right to left so every step is a matrix-vector product.
    v_sample = R.T @ (C.T @ (W.T @ (v_lab - v_dty)))

    return v_sample


@jax.jit
def origin_lab(
    v_sample: jax.Array,
    k_in_lab: jax.Array,
    omega: float,
    wedge: float,
    chi: float,
    dty: float,
    y0: float,
    column: bool = True,
) -> jax.Array:
    """Where a voxel scatters, in the lab frame: its real position at this omega and dty.

    Parameters
    ----------
    v_sample
        [3] Voxel centre in sample coordinates
    k_in_lab
        [3] Direction of the incoming beam (need not be normalised; must not be vertical)
    omega
        Omega motor value (degrees)
    wedge
        Wedge motor value (degrees)
    chi
        Chi motor value (degrees)
    dty
        Base diffractometer Y translation value (same units as v_sample)
    y0
        The true value of dty when the rotation axis (untilted by wedge, chi) intersects the beam
    column
        True if the voxel is a column (a 2D map): it scatters at the height where the beam crosses it. False for a
        cube (a 3D map), which scatters from its centre.

    Returns
    -------
    origin_lab: jax.Array
        [3] ``sample_to_lab(v_sample, omega, wedge, chi, dty, y0)``, raised for a column to the height of a beam
        tilted out of the horizontal plane

    Notes
    -----
    A tilted beam is taken to cross the rotation axis at the voxel's own height (in a 2D map, the layer's height),
    so that a voxel's height never depends on other layers. For a beam in the horizontal plane, or a cube, the
    origin is ``sample_to_lab(v_sample, omega, wedge, chi, dty, y0)``.
    """
    lab = sample_to_lab(v_sample, omega, wedge, chi, dty, y0)
    _, _, rise, _ = beam_offsets(lab, k_in_lab)
    return lab + jnp.where(column, 1.0, 0.0) * jnp.array([0.0, 0.0, 1.0]) * rise


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
    y0
        The true value of dty when the rotation axis (untilted by wedge, chi) intersects the beam

    Returns
    -------
    dty_required: float
        dty value that brings v_sample into beam at angle omega
    """
    dty_required, _ = dty_and_origin_lab(v_sample, k_in_lab, omega, wedge, chi, y0)

    return dty_required


@jax.jit
def dty_and_origin_lab(
    v_sample: jax.Array, k_in_lab: jax.Array, omega: float, wedge: float, chi: float, y0: float
) -> tuple[jax.Array, jax.Array]:
    """Find the dty that brings v_sample into the beam, and the lab position it then occupies.

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
    y0
        The true value of dty when the rotation axis (untilted by wedge, chi) intersects the beam

    Returns
    -------
    dty_required: jax.Array
        dty value that brings v_sample into beam at angle omega
    origin_lab: jax.Array
        [3] Where the beam passes through v_sample's column at that dty: v_sample in lab coordinates
        (``sample_to_lab(v_sample, omega, wedge, chi, dty_required, y0)``), raised to the height of a tilted beam

    Notes
    -----
    Applying dty shifts the lab position by ``(0, dty - y0, 0)``, so dty is chosen to put the point on the beam's
    horizontal line. The origin is then :func:`origin_lab` at that dty, for a column: a beam tilted out of the
    horizontal plane is taken to cross the rotation axis at the point's own height, and the scattering happens
    where the pencil crosses the column.

    See Also
    --------
    find_dty_for_beam_xy : Returns the dty value alone.
    origin_lab : The scattering origin for a given dty.
    """
    # Rotate v_sample into v_lab, ignoring y0 and dty for now (just find angles)
    v_lab = sample_to_lab(v_sample, omega, wedge, chi, 0.0, 0.0)

    # y of the beam's horizontal line at the point's lab x; dty moves the point there (when dty = y0, no shift)
    y_ray = k_in_lab[1] / k_in_lab[0] * v_lab[0]
    dty_required = y_ray - v_lab[1] + y0

    return dty_required, origin_lab(v_sample, k_in_lab, omega, wedge, chi, dty_required, y0)
