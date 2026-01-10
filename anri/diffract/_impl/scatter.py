import jax
import jax.numpy as jnp


@jax.jit
def scale_norm_k(k_vec: jax.Array, wavelength: float) -> jax.Array:
    r"""Normalise and scale k-vector according to ImageD11 convention (1/wavelength).

    Parameters
    ----------
    k_vec:
        [3] Unscaled wave-vector
    wavelength:
        wavelength in angstroms

    Returns
    -------
    k_vec_scaled: jax.Array
        [3] Scaled normalised k-vector

    Notes
    -----
    .. math::
        \vec{k_{\text{scaled}}} = \frac{1}{\lambda}\frac{\vec{k}}{\abs{\vec{k}}}

    """
    k = 1 / wavelength  # ImageD11 convention
    k_vec_norm = k_vec / jnp.linalg.norm(k_vec)
    k_vec_scaled = k * k_vec_norm
    return k_vec_scaled


@jax.jit
def k_to_q_lab(k_in: jax.Array, k_out: jax.Array) -> jax.Array:
    r"""Convert from scaled normalised $\vec{k_{\text{in}}}$ and $\vec{k_{\text{out}}}$ to scattering vector $\vec{Q}$.

    Parameters
    ----------
    k_in:
        [3] Incoming scaled normalised wave-vector
    k_out:
        [3] Outgoing scaled normalised wave-vector

    Returns
    -------
    q_lab: jax.Array
        [3] Scattering vector

    Notes
    -----
    Just the simple Laue equation:

    .. math::
        \vec{\Delta k} = \vec{k_{\text{out}}} - \vec{k_{\text{in}}} = \vec{G} = \vec{Q}
    """
    q_lab = k_out - k_in
    return q_lab


@jax.jit
def q_lab_to_k_out(q_lab: jax.Array, k_in: jax.Array) -> jax.Array:
    r"""Convert from scattering vector $\vec{Q}$ and scaled normalised $\vec{k_{\text{in}}}$ to scaled normalised $\vec{k_{\text{out}}}$.

    Parameters
    ----------
    q_lab:
        [3] Scattering vector
    k_in:
        [3] Incoming scaled normalised wave-vector

    Returns
    -------
    k_out: jax.Array
        [3] Outgoing scaled normalised wave-vector
    """
    k_out = q_lab + k_in
    return k_out


@jax.jit
def peak_lab_to_k_out(peak_lab: jax.Array, origin_lab: jax.Array, wavelength: float) -> jax.Array:
    r"""Convert from vector of peak in lab frame to normalised scaled $\vec{k_{\text{out}}}$ in the lab frame.

    If peak_lab is an observed peak position in the lab frame, we subtract the origin_lab to get the scattering vector, then normalise it.

    Parameters
    ----------
    peak_lab:
        [3] Vector in the lab frame
    origin_lab:
        [3] Origin of diffraction in the lab frame
    wavelength:
        wavelength in angstroms

    Returns
    -------
    k_out: jax.Array
        [3] Outgoing scaled normalised wave-vector
    """
    k_out_vec = peak_lab - origin_lab  # unscaled, un-normalised
    k_out = scale_norm_k(k_out_vec, wavelength)
    return k_out


@jax.jit
def tth_eta_to_k_out(tth: jax.Array, eta: jax.Array, wavelength: float) -> jax.Array:
    r"""Convert from (tth, eta) angles to scaled normalised $\vec{k_{\text{out}}}$ in the lab frame.

    Parameters
    ----------
    tth:
        :math:`2\theta` angle in degrees
    eta:
        :math:`\eta` angle in degrees
    wavelength:
        wavelength in angstroms

    Returns
    -------
    k_out: jax.Array
        [3] Outgoing scaled normalised wave-vector
    
    Notes
    -----
    From equation 39 [1]_:

    .. math::
        \vec{k_{\text{out}}} = \frac{1}{\lambda}
        \begin{bmatrix}
        \cos(2\theta) \\
        -\sin(2\theta) \sin(\eta) \\
        \sin(2\theta) \cos(\eta)
        \end{bmatrix}

    References
    ----------
    .. [1] Poulsen, H.F., Jakobsen, A.C., Simons, H., Ahl, S.R., Cook, P.K., Detlefs, C., 2017. X-ray diffraction microscopy based on refractive optics. J Appl Crystallogr 50, 1441–1456. https://doi.org/10.1107/S1600576717011037
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

    k_out = scale_norm_k(k_out_vec, wavelength)

    return k_out


@jax.jit
def q_lab_to_tth_eta(q_lab: jax.Array, wavelength: float) -> tuple[jax.Array, jax.Array]:
    r"""Convert from scattering vector $\vec{Q}$ in lab frame to (tth, eta) angles.

    Adapted from :func:`ImageD11.transform.uncompute_g_vectors`

    Parameters
    ----------
    q_lab
        [3] Scattering vector in lab frame
    wavelength
        wavelength in angstroms

    Returns
    -------
    tth: jax.Array
        :math:`2\theta` angle in degrees
    eta: jax.Array
        :math:`\eta` angle in degrees
    
    Notes
    -----
    .. math::
        \begin{aligned}
            d^* &= \abs{\vec{Q}} \\
            \sin{\theta} &= \frac{d^* \lambda}{2} \\
            2\theta &= 2 \arcsin\left(\frac{\abs{\vec{Q}}\lambda}{2}\right) = 2 \arcsin\left(\frac{d^* \lambda}{2}\right)\\
            \eta &= \arctan2\left(-Q_2, Q_3\right)
        \end{aligned}
    """
    q1, q2, q3 = q_lab
    ds = jnp.linalg.norm(q_lab)
    s = ds * wavelength / 2.0  # sin(theta)
    tth = 2.0 * jnp.degrees(jnp.arcsin(s))
    eta = jnp.degrees(jnp.arctan2(-q2, q3))
    return tth, eta


@jax.jit
def omega_solns(
    q_sample: jax.Array,
    etasign: float,
    k_in_sample: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Compute omega angles that satisfy the Ewald condition for a given $\vec{Q}$ in the sample frame.

    Due to Friedel pairs, there are always two possible omega solutions for a given $\vec{Q}$ (when they exist).
    We pass in an `etasign` parameter to select which solution to return.

    Parameters
    ----------
    q_sample
        [3] Scattering vector in sample frame
    etasign
        +1 (omega1 in ImageD11) or -1 (omega2 in ImageD11) to select which omega solution to return
    k_in_sample
        Incoming scaled normalised wave-vector in sample frame

    Returns
    -------
    omega: jax.Array
        Omega angle in degrees
    valid: jax.Array
        Boolean indicating if a valid solution exists

    Notes
    -----
    This is adapted from :func:`ImageD11.gv_general.g_to_k`. You can find a nice writeup of this in (Milch and Minor, 1974) [2]_.

    The Ewald condition is defined by:

    $\vec{Q} = \vec{k}_{out} - \vec{k}_{in}$

    so:

    $\vec{Q} + \vec{k}_{in} = \vec{k}_{out}$

    The Ewald condition requires that the magnitudes of the incoming and outgoing wave-vectors are equal:

    $\abs{\vec{k}_{out}} = \abs{\vec{k}_{in}} = \frac{1}{\lambda}$

    So we can square both sides:

    $\abs{\vec{k}_{out}}^2 = \abs{\vec{k}_{in}}^2 = \frac{1}{\lambda^2}$

    We also square both sides here:

    $\abs{\vec{Q} + \vec{k}_{in}}^2 = \abs{\vec{k}_{out}}^2$

    Substituting the Ewald condition into this gives:

    $\abs{\vec{Q} + \vec{k}_{in}}^2 = \abs{\vec{k}_{in}}^2$

    Expanding the left-hand side:

    $\abs{\vec{Q}}^2 + 2 \vec{Q} \cdot \vec{k}_{in} + \abs{\vec{k}_{in}}^2 = \abs{\vec{k}_{in}}^2$

    Subtracting $\abs{\vec{k}_{in}}^2$ from both sides:

    $\abs{\vec{Q}}^2 + 2 \vec{Q} \cdot \vec{k}_{in} = 0$

    Rearrange:

    $\vec{Q} \cdot \vec{k}_{in} = -\frac{\abs{\vec{Q}}^2}{2}$

    We decompose $\vec{Q}$ into components parallel and perpendicular to the rotation axis $\hat{n}$.

    $\vec{Q}(\omega)$ is the rotated Q vector at angle omega about axis $\hat{n}$:

    $\vec{Q}(\omega) = \vec{Q}_{perp} \cos(\omega) + (\hat{n} \times \vec{Q}_{perp}) \sin(\omega) + \vec{Q}_{par}$

    These are formed via basis vectors:

    $\vec{Q}_{par} = (\vec{Q} \cdot \hat{n}) \hat{n}$

    $\vec{Q}_{perp} = \vec{Q} - \vec{Q}_{par}$

    Now dot product with $\vec{k}_{in}$:

    $\vec{Q}(\omega) \cdot \vec{k}_{in} = \vec{Q}_{perp} \cdot \vec{k}_{in} \cos(\omega) + (\hat{n} \times \vec{Q}_{perp}) \cdot \vec{k}_{in} \sin(\omega) + \vec{Q}_{par} \cdot \vec{k}_{in}$

    We define some constants here:

    $\alpha \cos(\omega) + \beta \sin(\omega) + \gamma = \vec{Q}(\omega) \cdot \vec{k}_{in}$

    where:

    .. math::
        \begin{aligned}
        \alpha &= \vec{Q}_{perp} \cdot \vec{k}_{in} \\
        \beta &= (\hat{n} \times \vec{Q}_{perp}) \cdot \vec{k}_{in} \\
        \gamma &= \vec{Q}_{par} \cdot \vec{k}_{in}
        \end{aligned}
    
    Setting equal to the Ewald condition:

    $\alpha \cos(\omega) + \beta \sin(\omega) + \gamma = -\frac{\abs{\vec{Q}}^2}{2}$

    Subtracting $\gamma$ from both sides:

    $\alpha \cos(\omega) + \beta \sin(\omega) = -\frac{\abs{\vec{Q}}^2}{2} - \gamma$

    We call the right-hand side $\delta$:

    $\alpha \cos(\omega) + \beta \sin(\omega) = \delta$
    
    We now solve the harmonic addition:

    $R \sin(\omega + \phi) = \delta$

    with $R = \sqrt{\alpha^2 + \beta^2}$ and $\phi = \arctan2(\alpha, \beta)$.

    The two solutions for $\omega$ are:

    $\omega_1 = \arcsin\left(\frac{\delta}{R}\right) - \phi$

    $\omega_2 = -\arcsin\left(\frac{\delta}{R}\right) - \phi - \pi$

    References
    ----------
    .. [2] Milch, J.R., Minor, T.C., 1974. The indexing of single-crystal X-ray rotation photographs. Journal of Applied Crystallography 7, 502–505. https://doi.org/10.1107/S0021889874010284
    """
    q_0 = q_sample
    axis_sample = jnp.array([0.0, 0.0, 1.0])  # rotation axis is always +Z in sample frame

    # split Q into components parallel and perpendicular to rotation axis
    # when we rotate, only the perpendicular component changes

    q_par = jnp.dot(q_0, axis_sample) * axis_sample
    q_perp = q_0 - q_par

    # Q(w) = R(w) @ Q0
    # R(w) rotates Q_perp about axis by w, leaves Q_par unchanged

    alpha = jnp.dot(k_in_sample, q_perp)
    beta = jnp.dot(k_in_sample, jnp.cross(axis_sample, q_perp))
    gamma = jnp.dot(k_in_sample, q_par)
    delta = -jnp.sum(q_0 * q_0) / 2.0 - gamma

    # trig identity (phased cosine wave):
    # https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Sine_and_cosine

    # alpha cos(w) + beta sin(w) = R * sin(w + phi)
    # where
    # R = sqrt(alpha^2 + beta^2)
    # phi = arctan2(alpha, beta)

    phi = jnp.arctan2(alpha, beta)  # cos term / sin term
    R = jnp.sqrt(alpha * alpha + beta * beta)
    # handle cases where R is very close to zero
    eps = 1e-12
    R_safe = jnp.where(R < eps, eps, R)
    # valid solutions occur if |delta / R| <= 1
    quot = delta / R_safe

    valid = jnp.where(
        R < eps,
        jnp.abs(delta) < eps,  # any w works only if delta≈0
        (quot >= -1.0) & (quot <= 1.0),
    )

    # asin only valid for -1 <= quot <= 1
    asin_term = jnp.where(valid, jnp.arcsin(jnp.clip(quot, -1.0, 1.0)), 0.0)
    omega1 = asin_term - phi
    omega2 = -asin_term - phi - jnp.pi

    # map into -pi to pi
    angmod_omega1 = jnp.arctan2(jnp.sin(omega1), jnp.cos(omega1))  # postive eta
    angmod_omega2 = jnp.arctan2(jnp.sin(omega2), jnp.cos(omega2))  # negative eta

    # Convert the etasign to a boolean condition
    condition = etasign == 1

    angmod_result = jnp.where(condition, angmod_omega1, angmod_omega2)

    omega = jnp.degrees(angmod_result)

    return omega, valid
