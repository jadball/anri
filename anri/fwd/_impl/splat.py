"""Forward projection code for producing dense images from centroids and covariances."""

import jax
import jax.numpy as jnp


@jax.jit
def mahalanobis_sq(centroid: jax.Array, coord: jax.Array, inv_cov: jax.Array) -> jax.Array:
    r"""Get squared Mahalanobis distance for a coordinate given a centroid and an inverse covariance matrix.

    See :func:`prepare_cov` for preparation of inv_cov.

    Parameters
    ----------
    centroid
        [N] Coordinate of peak centroid in N dimensions
    coord
        [N] Query coordinate in N dimensions
    inv_cov
        Inverse covariance matrix, from :func:`prepare_cov`

    Returns
    -------
    d2: jax.Array
        Squared Mahalanobis distance

    Notes
    -----
    From Wikipedia [1]_:

    Given two points $\vec{x}$ and $\vec{y}$, the squared Mahalanobis distance between them with respect to a probability distribution $Q$, that has a positiuve semi-definite covariance matrix $\matr{\Sigma}$ is:

    $$d_{M}\left(\vec{x},\vec{y},Q\right)^2 = \left(\vec{x}-\vec{y}\right)^{\intercal}\matr{\Sigma}^{-1}\left(\vec{x}-\vec{y}\right)$$

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mahalanobis_distance#Definition
    """
    diff = coord - centroid
    d2 = diff @ inv_cov @ diff
    return d2


@jax.jit
def sample_intensities(centroid: jax.Array, coord: jax.Array, inv_cov: jax.Array, norm_const: float) -> jax.Array:
    r"""Evaluate probability density of Mahalanobis distance.

    See :func:`prepare_cov` for preparation of inv_cov and norm_const.

    Parameters
    ----------
    centroid
        [N] Coordinate of peak centroid in N dimensions
    coord
        [N] Query coordinate in N dimensions
    inv_cov
        Inverse covariance matrix, from :func:`prepare_cov`
    norm_const
        Normalisation constant, from :func:`prepare_cov`

    Returns
    -------
    jax.Array
        Probability density of an observation

    Notes
    -----
    See :func:`prepare_cov` for maths.

    """
    d2 = mahalanobis_sq(centroid, coord, inv_cov)
    return norm_const * jnp.exp(-0.5 * d2)


@jax.jit
def prepare_cov(cov: jax.Array) -> tuple[jax.Array, jax.Array]:
    r"""Get inverse covariance matrix and normalisation constant for :func:`sample_intensities`.

    Parameters
    ----------
    cov
        [3,3] or [3,4] output covariance matrix from :func:`anri.fwd.propagate_cov_box` or :func:`anri.fwd.propagate_cov_scan`

    Returns
    -------
    inv_cov: jax.Array
        Inverse covariance matrix
    norm_const: jax.Array
        Normalisation constant for :func:`sample_intensities`

    Notes
    -----
    From Wikipedia [1]_:

    For a $N$-dimensional normal distribution, the probability density of an observation $\vec{x}$ can be determined:

    $$\Pr[{\vec{x}}]\,d{\vec{x}} = \frac{1}{\sqrt{\det{2\pi\matr{\Sigma}}}}\exp{\left(-\frac{d_{M}\left(\vec{x},\vec{y},Q\right)^2}{2}\right)}$$

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mahalanobis_distance#Normal_distributions
    """
    inv_cov = jnp.linalg.inv(cov)

    twopi_cov = (2.0 * jnp.pi) * cov
    det_twopi_cov = jnp.linalg.det(twopi_cov)

    norm_const = 1.0 / jnp.sqrt(det_twopi_cov)

    return inv_cov, norm_const


@jax.jit(static_argnums=(4,))
def peak_to_pixels(
    centroid: jax.Array, cov: jax.Array, amplitude: float, bins: tuple[jax.Array], window_size: int
) -> tuple[jax.Array, jax.Array]:
    """Get N*(window_size,) image of normalised pixel intensities around centroid in N dimensions.

    This can be vectorised over centroids - :func:`peaks_to_pixels`.

    Parameters
    ----------
    centroid
        [N] Coordinate of peak centroid in N dimensions
    cov
        [3, N] output covariance matrix from :func:`anri.fwd.propagate_cov_box` or :func:`anri.fwd.propagate_cov_scan`
    amplitude
        Total integrated amplitude of the peak
    bins
        The bin centres for each dimension ( e.g (sc, fc, omega, [dty]) )
    window_size
        The window size to evaluate pixel intensities

    Returns
    -------
    intensities_dense: jax.Array
        [N*(window_size,)] Dense image of intensities around the centroid
    starts: jax.Array
        [N] Start indices (of bins) to the intensity image
    """
    k = centroid.shape[0]
    inv_cov, norm_const = prepare_cov(cov)

    # 1. Find start indices using a list comprehension instead of vmap
    # This handles bins with different lengths (e.g., 2048 vs 3600)
    starts = jnp.array(
        [
            jnp.clip(jnp.argmin(jnp.abs(bins[i] - centroid[i])) - window_size // 2, 0, len(bins[i]) - window_size)
            for i in range(k)
        ]
    )

    # 2. Extract local coordinates
    local_bins = [jax.lax.dynamic_slice(bins[i], (starts[i],), (window_size,)) for i in range(k)]

    # 3. Create the k-dimensional grid
    grid = jnp.meshgrid(*local_bins, indexing="ij")
    coords = jnp.stack([g.ravel() for g in grid], axis=-1)

    # 4. Calculate intensities
    # We vmap over the flattened 'coords'
    v_sample = jax.vmap(sample_intensities, in_axes=(None, 0, None, None))
    intensities = v_sample(centroid, coords, inv_cov, norm_const) * amplitude

    # 5. Reshape back to the dense block shape: (window_size, window_size, ...)
    block_shape = (window_size,) * k
    intensities_dense = intensities.reshape(block_shape)

    return intensities_dense, starts


## vmaps
peaks_to_pixels = jax.vmap(peak_to_pixels, in_axes=[0, 0, 0, None, None])
