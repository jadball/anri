"""Forward projection code for producing dense images from centroids and covariances."""

import jax
import jax.numpy as jnp


@jax.jit
def prepare_gaussian_bin(cov: jax.Array) -> jax.Array:
    r"""Precompute per-axis scaling factors for :func:`sample_gaussian_bin`.

    Parameters
    ----------
    cov : jax.Array
        ``[n, n]`` covariance matrix of the multivariate Gaussian.

    Returns
    -------
    erf_scale : jax.Array
        ``[n]`` per-axis scaling factor :math:`\frac{1}{\sigma_i \sqrt{2}}`
        applied to bin edge residuals in :func:`sample_gaussian_bin`.

    Notes
    -----
    The integrated intensity of an :math:`n`-dimensional Gaussian
    :math:`\mathcal{N}(\vec{\mu}, \Sigma)` over an axis-aligned bin centred
    at :math:`\vec{x}` with half-widths :math:`\vec{h} = \vec{w}/2` factorises
    over axes using the marginal distributions:

    .. math::

        \int_{\vec{x} - \vec{h}}^{\vec{x} + \vec{h}}
        \mathcal{N}(\vec{u} \mid \vec{\mu}, \Sigma)\, d\vec{u}
        \approx \prod_{i=1}^{n}
        \frac{1}{2}\left[
            \operatorname{erf}\!\left(\frac{x_i + h_i - \mu_i}{\sigma_i\sqrt{2}}\right)
            -
            \operatorname{erf}\!\left(\frac{x_i - h_i - \mu_i}{\sigma_i\sqrt{2}}\right)
        \right]

    where :math:`\sigma_i = \sqrt{\Sigma_{ii}}` is the marginal standard
    deviation along axis :math:`i`. This is exact when :math:`\Sigma` is
    diagonal and accurate when bin widths are small relative to the
    correlation length of :math:`\Sigma` [1]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Error_function#Integral_of_a_Gaussian_over_an_interval
    """
    marginal_std = jnp.sqrt(jnp.diag(cov))
    erf_scale = 1.0 / (marginal_std * jnp.sqrt(2.0))
    return erf_scale


@jax.jit
def sample_gaussian_bins(
    mu: jax.Array,
    erf_scale: jax.Array,
    bin_centres: jax.Array,
    half_widths: jax.Array,
) -> jax.Array:
    r"""Integrated intensity of a multivariate Gaussian over an array of axis-aligned bins.

    Parameters
    ----------
    mu : jax.Array
        ``[n]`` centroid of the Gaussian in ``(slow, fast, omega[, dty])`` coordinates.
    erf_scale : jax.Array
        ``[n]`` precomputed per-axis scaling factors from :func:`prepare_gaussian_bin`,
        equal to :math:`\frac{1}{\sigma_i \sqrt{2}}`.
    bin_centres : jax.Array
        ``[m, n]`` coordinates of ``m`` bin centres to evaluate.
    half_widths : jax.Array
        ``[n]`` half-width of each bin along each axis, derived from the bin
        spacing of ``(slow, fast, omega[, dty])``.

    Returns
    -------
    intensities : jax.Array
        ``[m]`` integrated intensity at each bin, fully differentiable
        with respect to ``mu`` and ``erf_scale``.

    Notes
    -----
    Each bin integral factorises over axes as a product of 1-D marginal integrals:

    .. math::

        I_j = \prod_{i=1}^{n}
            \frac{1}{2}\left[
                \operatorname{erf}\!\left((x_{ji} + h_i - \mu_i) \cdot s_i\right)
                -
                \operatorname{erf}\!\left((x_{ji} - h_i - \mu_i) \cdot s_i\right)
            \right]

    where :math:`h_i` is the per-axis bin half-width and
    :math:`s_i = \frac{1}{\sigma_i\sqrt{2}}` is the precomputed scale.
    Each axis carries its own physical units --- pixels for ``slow`` and
    ``fast``, degrees for ``omega``, mm for ``dty`` --- so half-widths must
    not be assumed uniform across axes.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Error_function#Integral_of_a_Gaussian_over_an_interval
    """
    lo = (bin_centres - half_widths - mu) * erf_scale   # [m, n]
    hi = (bin_centres + half_widths - mu) * erf_scale   # [m, n]
    marginals = 0.5 * (jax.scipy.special.erf(hi) - jax.scipy.special.erf(lo))  # [m, n]
    return jnp.prod(marginals, axis=-1)                  # [m]


@jax.jit(static_argnums=(4,))
def peak_to_pixels(
    centroid: jax.Array,
    erf_scale: jax.Array,
    amplitude: float,
    bins: tuple[jax.Array, ...],
    window_size: int,
) -> tuple[jax.Array, jax.Array]:
    r"""Compute the intensity of a single Gaussian peak over a local pixel window.

    Parameters
    ----------
    centroid : jax.Array
        ``[n]`` centroid of the Gaussian peak in ``(slow, fast, omega[, dty])``
        coordinates.
    erf_scale : jax.Array
        ``[n]`` precomputed per-axis scaling factors from :func:`prepare_gaussian_bin`,
        equal to :math:`\frac{1}{\sigma_i\sqrt{2}}`.
    amplitude : float
        Integrated intensity of the peak.
    bins : tuple[jax.Array, ...]
        Tuple of ``n`` coordinate arrays giving bin centres along each of
        ``(slow, fast, omega[, dty])``. Bin widths are derived from the
        spacing of each array, so uniform spacing per axis is assumed.
    window_size : int
        Number of bins along each axis. Static for JIT and vmap compilation.

    Returns
    -------
    intensities : jax.Array
        ``[window_size] * n`` dense block of integrated intensities scaled
        by ``amplitude``.
    starts : jax.Array
        ``[n]`` start indices locating the window within each axis of ``bins``,
        for scattering the output back into a global array.

    Notes
    -----
    Per-axis bin half-widths :math:`h_i` are derived from the spacing of each
    bin array as :math:`h_i = (b_i[1] - b_i[0]) / 2`. A local window of
    ``window_size`` bins is extracted around ``centroid`` along each axis via
    :func:`jax.lax.dynamic_slice`. The resulting local bin centres are formed
    into an ``[window_size^n, n]`` grid via :func:`jnp.meshgrid` and passed to
    :func:`sample_gaussian_bins`.

    See Also
    --------
    prepare_gaussian_bin : Precomputes ``erf_scale`` from a covariance matrix.
    peaks_to_pixels : Vectorised form of this function over a batch of peaks.
    """
    n = centroid.shape[0]

    # Per-axis bin half-widths — each axis has its own physical units
    half_widths = jnp.array([
        (bins[i][1] - bins[i][0]) / 2.0
        for i in range(n)
    ])

    starts = jnp.array([
        jnp.clip(
            jnp.argmin(jnp.abs(bins[i] - centroid[i])) - window_size // 2,
            0,
            len(bins[i]) - window_size,
        )
        for i in range(n)
    ])

    local_axes = [
        jax.lax.dynamic_slice(bins[i], (starts[i],), (window_size,))
        for i in range(n)
    ]

    # [window_size^n, n] grid of bin centres within the local window
    grid = jnp.stack(
        [g.ravel() for g in jnp.meshgrid(*local_axes, indexing='ij')],
        axis=-1,
    )

    intensities = sample_gaussian_bins(centroid, erf_scale, grid, half_widths)
    intensities = intensities.reshape((window_size,) * n) * amplitude

    return intensities, starts


#: Vectorised form of :func:`peak_to_pixels` over a batch of peaks.
#:
#: Parameters map identically to :func:`peak_to_pixels` with a leading batch
#: dimension on ``centroid``, ``erf_scale``, and ``amplitude``.
#: ``bins`` and ``window_size`` are shared across all peaks.
peaks_to_pixels = jax.vmap(peak_to_pixels, in_axes=[0, 0, 0, None, None])
