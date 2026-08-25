"""Forward projection code for producing dense images from centroids and covariances."""

import jax
import jax.numpy as jnp


@jax.jit
def prepare_gaussian_bin(cov: jax.Array) -> jax.Array:
    r"""Precompute per-axis scaling factors for :func:`sample_gaussian_bins`.

    Parameters
    ----------
    cov : jax.Array
        ``[n, n]`` covariance matrix of the multivariate Gaussian.

    Returns
    -------
    erf_scale : jax.Array
        ``[n]`` per-axis scaling factor :math:`s_i = \frac{1}{\sigma_i \sqrt{2}}`,
        where :math:`\sigma_i = \sqrt{\Sigma_{ii}}` is the marginal standard
        deviation along axis :math:`i`. See :func:`sample_gaussian_bins` for
        usage.
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
        ``[n]`` per-axis scaling factors :math:`s_i = \frac{1}{\sigma_i \sqrt{2}}`
        precomputed by :func:`prepare_gaussian_bin`.
    bin_centres : jax.Array
        ``[m, n]`` coordinates of ``m`` bin centres to evaluate.
    half_widths : jax.Array
        ``[n]`` half-width :math:`h_i` of each bin along each axis, derived from
        the bin spacing of ``(slow, fast, omega[, dty])``. Each axis carries its
        own physical units --- pixels for ``slow`` and ``fast``, degrees for
        ``omega``, um for ``dty`` --- so half-widths are not assumed uniform
        across axes.

    Returns
    -------
    intensities : jax.Array
        ``[m]`` integrated intensity at each bin, fully differentiable
        with respect to ``mu`` and ``erf_scale``.

    Notes
    -----
    For a 1-D Gaussian with mean :math:`\mu_i` and standard deviation
    :math:`\sigma_i`, the probability mass in the interval :math:`(a, b]` is [1]_:

    .. math::

        \operatorname{P}(a < x \leq b) =
            \frac{1}{2}\left[
                \operatorname{erf}\!\left(\frac{b - \mu_i}{\sigma_i\sqrt{2}}\right)
                -
                \operatorname{erf}\!\left(\frac{a - \mu_i}{\sigma_i\sqrt{2}}\right)
            \right]

    Substituting bin edges :math:`a = x_{ji} - h_i` and :math:`b = x_{ji} + h_i`
    and writing :math:`s_i = \frac{1}{\sigma_i\sqrt{2}}`:

    .. math::

        \operatorname{P}(x_{ji} - h_i < x \leq x_{ji} + h_i) =
            \frac{1}{2}\left[
                \operatorname{erf}\!\left((x_{ji} + h_i - \mu_i) \cdot s_i\right)
                -
                \operatorname{erf}\!\left((x_{ji} - h_i - \mu_i) \cdot s_i\right)
            \right]

    The full :math:`n`-dimensional bin intensity is the product of marginal
    integrals over each axis:

    .. math::

        I_j = \prod_{i=1}^{n}
            \frac{1}{2}\left[
                \operatorname{erf}\!\left((x_{ji} + h_i - \mu_i) \cdot s_i\right)
                -
                \operatorname{erf}\!\left((x_{ji} - h_i - \mu_i) \cdot s_i\right)
            \right]

    This factorisation is exact when :math:`\Sigma` is diagonal and accurate
    when bin widths are small relative to the correlation length of :math:`\Sigma`.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function

    See Also
    --------
    prepare_gaussian_bin : Precomputes :math:`s_i` from a covariance matrix.
    peak_to_pixels : Evaluates a single peak over a local window using this function.
    """
    lo = (bin_centres - half_widths - mu) * erf_scale  # [m, n]
    hi = (bin_centres + half_widths - mu) * erf_scale  # [m, n]
    marginals = 0.5 * (jax.scipy.special.erf(hi) - jax.scipy.special.erf(lo))  # [m, n]
    return jnp.prod(marginals, axis=-1)  # [m]


@jax.jit(static_argnums=(4,))
def peak_to_pixels(
    centroid: jax.Array,
    erf_scale: jax.Array,
    amplitude: float,
    bins: tuple[jax.Array, ...],
    window_sizes: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    r"""Compute the intensity of a single Gaussian peak over a local pixel window.

    Parameters
    ----------
    centroid : jax.Array
        ``[n]`` centroid of the Gaussian peak in ``(slow, fast, omega[, dty])``
        coordinates.
    erf_scale : jax.Array
        ``[n]`` per-axis scaling factors :math:`s_i = \frac{1}{\sigma_i\sqrt{2}}`
        precomputed by :func:`prepare_gaussian_bin`.
    amplitude : float
        Integrated intensity of the peak.
    bins : tuple[jax.Array, ...]
        Tuple of ``n`` coordinate arrays giving bin centres along each of
        ``(slow, fast, omega[, dty])``. Bin half-widths :math:`h_i` are derived
        from the spacing of each array as :math:`h_i = (b_i[1] - b_i[0]) / 2`,
        so uniform spacing per axis is assumed.
    window_sizes : tuple[int, ...]
        Number of bins along each axis, one per axis. Each entry must not
        exceed the length of the corresponding bin array. Static for JIT
        and vmap compilation.

    Returns
    -------
    intensities : jax.Array
        Dense block of integrated intensities scaled by ``amplitude``, with
        shape ``window_sizes``.
    starts : jax.Array
        ``[n]`` start indices locating the window within each axis of ``bins``,
        for scattering the output back into a global array.

    Notes
    -----
    A local window of ``window_sizes[i]`` bins is extracted around ``centroid``
    along each axis ``i`` via :func:`jax.lax.dynamic_slice`. The resulting local
    bin centres are assembled into an ``[prod(window_sizes), n]`` grid and passed
    to :func:`sample_gaussian_bins` — see that function for the underlying maths.

    See Also
    --------
    prepare_gaussian_bin : Precomputes :math:`s_i` from a covariance matrix.
    sample_gaussian_bins : Core bin-integration function called internally.
    peaks_to_pixels : Vectorised form of this function over a batch of peaks.
    """
    n = centroid.shape[0]

    half_widths = jnp.array([(bins[i][1] - bins[i][0]) / 2.0 for i in range(n)])

    starts = jnp.array(
        [
            jnp.clip(
                jnp.argmin(jnp.abs(bins[i] - centroid[i])) - window_sizes[i] // 2,
                0,
                len(bins[i]) - window_sizes[i],
            )
            for i in range(n)
        ]
    )

    local_axes = [jax.lax.dynamic_slice(bins[i], (starts[i],), (window_sizes[i],)) for i in range(n)]

    # [prod(window_sizes), n] grid of bin centres within the local window
    grid = jnp.stack(
        [g.ravel() for g in jnp.meshgrid(*local_axes, indexing="ij")],
        axis=-1,
    )

    intensities = sample_gaussian_bins(centroid, erf_scale, grid, half_widths)
    intensities = intensities.reshape(window_sizes) * amplitude

    return intensities, starts


peaks_to_pixels = jax.vmap(peak_to_pixels, in_axes=[0, 0, 0, None, None])
