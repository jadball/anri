"""Base functions for forward projection code."""

from collections.abc import Callable, Iterable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from anri.diffract import omega_from_core, omega_solns_core, q_lab_to_k_out, scale_norm_k
from anri.geom import lab_to_sample, sample_to_lab


@jax.jit
def hkl_to_k_omega(
    ubi: jax.Array,  # grain stuff
    hkl: jax.Array,  # peak stuff
    etasign: int,  # beam stuff
    wavelength: float,
    k_in_lab: jax.Array,
    ky: float,
    kz: float,  # gonio stuff
    wedge: float,
    chi: float,
) -> tuple[jax.Array, jax.Array, float, bool]:
    r"""Forward-project a reciprocal space vector (h,k,l) with basis vectors (a*, b*, c*) into k-vectors and omega angles.

    This just chains together various transforms from :func:`anri.diffract` and :func:`anri.geom`.

    Parameters
    ----------
    ubi:
        [3,3] (U.B)^(-1) matrix of the grain/voxel
    hkl:
        [3] (h,k,l) reciprocal space vector
    etasign:
        +1 (omega1 in ImageD11) or -1 (omega2 in ImageD11) to select which omega solution to return
    wavelength:
        Wavelength in angstroms
    k_in_lab:
        [3] Unperturbed unit vector of incoming beam, lab frame
    ky:
        y-component of the beam in the lab frame. Represents horizontal beam divergence, usually zero.
    kz:
        z-component of the beam in the lab frame. Represents vertical beam divergence, usually zero.
    wedge:
        Wedge motor value (degrees)
    chi:
        Chi motor value (degrees)

    Returns
    -------
    k_in_lab: jax.Array
        [3] k-in vector in laboratory frame (incoming beam) - not scaled or normalised!
    k_out_lab: jax.Array
        [3] k_out vector in laboratory frame
    omega: float
        Omega angle where diffraction occurs in degrees
    valid: bool
        Boolean indicating if a valid solution exists
    """
    q_sample = jnp.linalg.inv(ubi) @ hkl

    # perturb k_in_lab by divergence
    k_in_lab = k_in_lab + jnp.array([0.0, ky, kz])
    k_in_lab_norm = scale_norm_k(k_in_lab, wavelength)
    k_in_sample_norm = lab_to_sample(k_in_lab_norm, 0.0, wedge, chi, 0.0, 0.0)

    asin_term, phi, valid = omega_solns_core(q_sample, k_in_sample_norm)
    omega = omega_from_core(asin_term, phi, etasign)

    q_lab = sample_to_lab(q_sample, omega, wedge, chi, 0.0, 0.0)

    k_out_lab = q_lab_to_k_out(q_lab, k_in_lab_norm)

    return k_in_lab, k_out_lab, omega, valid


@jax.jit
def hkl_to_k_omega_both(
    ubi: jax.Array,
    hkl: jax.Array,
    wavelength: float,
    k_in_lab: jax.Array,
    ky: float,
    kz: float,
    wedge: float,
    chi: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Forward-project (h,k,l) into k-vectors and omega angles for both Friedel solutions.

    Parameters
    ----------
    ubi:
        [3,3] (U.B)^(-1) matrix of the grain/voxel
    hkl:
        [3] (h,k,l) reciprocal space vector
    wavelength:
        Wavelength in angstroms
    k_in_lab:
        [3] Unperturbed unit vector of incoming beam, lab frame
    ky:
        y-component of the beam in the lab frame. Represents horizontal beam divergence, usually zero.
    kz:
        z-component of the beam in the lab frame. Represents vertical beam divergence, usually zero.
    wedge:
        Wedge motor value (degrees)
    chi:
        Chi motor value (degrees)

    Returns
    -------
    k_in_lab: jax.Array
        [3] k-in vector in laboratory frame (incoming beam) - not scaled or normalised!
    k_out_lab: jax.Array
        [2,3] k_out vectors in laboratory frame, index 0 for ``etasign = +1``
    omega: jax.Array
        [2] Omega angles in degrees, index 0 for ``etasign = +1``
    valid: jax.Array
        Boolean indicating if a valid solution exists, shared by both branches

    Notes
    -----
    Q in the sample frame, the beam normalisation and the sample-frame beam
    vector are all properties of the geometry rather than of the branch, as is
    everything :func:`anri.diffract.omega_solns_core` computes. Producing both
    solutions together evaluates that shared part once. Only the omega rotation
    of Q and the resulting k_out differ per branch.

    See Also
    --------
    hkl_to_k_omega : Single-solution version, taking an ``etasign`` argument.
    """
    q_sample = jnp.linalg.inv(ubi) @ hkl

    # perturb k_in_lab by divergence
    k_in_lab = k_in_lab + jnp.array([0.0, ky, kz])
    k_in_lab_norm = scale_norm_k(k_in_lab, wavelength)
    k_in_sample_norm = lab_to_sample(k_in_lab_norm, 0.0, wedge, chi, 0.0, 0.0)

    asin_term, phi, valid = omega_solns_core(q_sample, k_in_sample_norm)
    omegas = jnp.stack([omega_from_core(asin_term, phi, 1.0), omega_from_core(asin_term, phi, -1.0)])

    k_outs = jnp.stack([
        q_lab_to_k_out(sample_to_lab(q_sample, omegas[i], wedge, chi, 0.0, 0.0), k_in_lab_norm)
        for i in range(2)
    ])

    return k_in_lab, k_outs, omegas, valid


@jax.jit
def get_cov_in(sig_origin: jax.Array, sig_wavelength: float, sig_ky: float, sig_kz: float) -> jax.Array:
    r"""Generate the input variance-covariance matrix from your sigma values.

    Parameters
    ----------
    sig_origin: jax.Array
        [3] Array of standard deviations on diffraction origin position. This is often your position uncertainty
    sig_wavelength: float
        Standard deviation on beam wavelength
    sig_ky
        Standard deviation on beam horizontal divergence
    sig_kz
        Standard deviation on beam vertical divergence

    Returns
    -------
    cov_in: jax.Array
        [6,6] Diagonal input variance-covariance matrix.

    Notes
    -----
    Builds a 6x6 input variance-covariance matrix.
    For $\vec{\sigma_{\text{origin}}} = \left(\sigma_x, \sigma_y, \sigma_z\right)$:
    $\matr{\Sigma}^{\text{in}} = \begin{bmatrix} \sigma_x^2 & 0 & 0 & 0 & 0 & 0 \\ 0 & \sigma_y^2 & 0 & 0 & 0 & 0\\0 & 0 & \sigma_z^2 & 0 & 0 & 0\\ 0 & 0 & 0 &\sigma_\lambda^2 & 0 & 0\\ 0 & 0 & 0 & 0 & \sigma_{k_y}^2 & 0 \\ 0 & 0 & 0& 0& 0 & \sigma_{k_z}^2\end{bmatrix} $

    """
    cov_in = jnp.diag(jnp.array([sig_origin[0]**2, sig_origin[1]**2, sig_origin[2]**2, sig_wavelength**2, sig_ky**2, sig_kz**2]))

    return cov_in


@jax.jit
def propagate_cov(J_func_out: Iterable[jax.Array], cov_in: jax.Array) -> jax.Array:
    r"""Propagate an input covariance matrix with a Jacobian to yield an output covariance matrix.

    Parameters
    ----------
    J_func_out
        The output of calling :func:`jax.jacfwd` on a JAX jitted function.
    cov_in
        [6,6] The input covariance matrix - build with :func:`get_cov_in`. Must have the same dimensionality as J_func_out

    Returns
    -------
    cov_out: jax.Array
        [3,3] Output covariance matrix - the covariance in the outputs of the JAX jitted function

    Notes
    -----
    Propagation goes as:
    $\mathbf{\Sigma}^{\text{out}} = \mathbf {J_{f}} \mathbf{\Sigma}^{\text{in}}  \mathbf {J_{f}}^T$

    This handles multi-dimensional outputs - e.g. if one of the function inputs is a 3-vector, we get a 3x3 Jacobian for it.
    """
    J = jnp.concatenate([j if j.ndim > 1 else j[..., jnp.newaxis] for j in J_func_out], axis=-1)
    cov_out = J @ cov_in @ J.T
    return cov_out


def make_propagator(
    centroid_fn: Callable,
    argnums: tuple[int, ...],
    has_aux: bool = False,
    diagonal: bool = True,
    active_dims: tuple[int, ...] | None = None,
    diag_out: bool = False,
    out_elems: tuple[tuple[int, int], ...] | None = None,
) -> Callable:
    r"""Build a JIT'd covariance propagation function for a given centroid function.

    Parameters
    ----------
    centroid_fn
        A JIT'd function that forward-projects (ubi, hkl) to a peak centroid.
        Its signature must have ``cov_in`` as the final argument.
    argnums
        Argument indices to differentiate with respect to.
        Should correspond to the uncertain inputs (origin, wavelength, divergence).
    has_aux
        If ``True``, ``centroid_fn`` returns a ``(centroid, aux)`` tuple (e.g. a validity bool),
        and the auxiliary output is discarded before propagation.
    diagonal
        If ``True`` (default), treat :math:`\mathbf{\Sigma}^{\text{in}}` as diagonal,
        which is what :func:`get_cov_in` produces. Off-diagonal entries of ``cov_in``
        are then ignored. Set ``False`` for a hand-built correlated input covariance.
    diag_out
        If ``True``, return only the diagonal of :math:`\\mathbf{\\Sigma}^{\\text{out}}`,
        i.e. the marginal variances, with shape ``[..., 4]`` instead of ``[..., 4, 4]``.
        Callers that splat axis-aligned Gaussians only ever use
        ``jnp.diagonal(cov)``; accumulating the full outer product
        ``col[:, None] * col[None, :]`` computes sixteen numbers per peak per
        input dimension to keep four, and returns an array four times larger for
        the host to gather from. Requires ``diagonal=True``.
    out_elems
        Static tuple of ``(i, j)`` index pairs selecting which elements of
        :math:`\\mathbf{\\Sigma}^{\\text{out}}` to accumulate, returned with shape
        ``[..., len(out_elems)]``. ``diag_out=True`` is shorthand for the four
        diagonal entries.

        The four variances alone are enough only if whatever consumes them
        renders an axis-aligned peak. They are not enough on a detector: the
        dominant broadening in a monochromatic scanning experiment is the
        wavelength spread, which displaces the spot along the radial direction
        of the Debye-Scherrer ring, so at azimuth 45 degrees the slow-fast
        covariance is comparable to the variances themselves and the peak is a
        tilted streak. Pass ``((0, 0), (1, 1), (2, 2), (3, 3), (0, 1))`` to get
        the variances plus the detector-plane covariance for five accumulations
        instead of sixteen.
    active_dims
        Static tuple selecting which input dimensions to propagate, indexing the
        flattened concatenation of the ``argnums`` entries. With
        ``argnums=(1, 4, 6, 7)`` on the scanning model these are ``0,1,2`` for
        origin xyz, ``3`` for wavelength, ``4`` for ky and ``5`` for kz. ``None``
        selects all of them. Only used when ``diagonal`` is ``True``.

        Selection happens at trace time, so an omitted dimension costs nothing at
        all, whereas a zero entry in ``cov_in`` is a runtime value that still pays
        for its Jacobian column. The origin dimensions are much cheaper than the
        beam ones: :func:`hkl_to_k_omega` does not depend on ``origin_sample``, so
        those tangents are zero through the expensive half of the model and fold
        away.

    Returns
    -------
    propagate_fn: Callable
        A JIT'd function with the same signature as ``centroid_fn`` (plus ``cov_in``
        as the final argument) that returns an output covariance matrix.

    Notes
    -----
    Propagation goes as:

    .. math::
        \mathbf{\Sigma}^{\text{out}} = \mathbf{J}_f \, \mathbf{\Sigma}^{\text{in}} \, \mathbf{J}_f^T

    where :math:`\mathbf{J}_f` is the Jacobian of ``centroid_fn`` with respect to
    ``argnums``. For diagonal :math:`\mathbf{\Sigma}^{\text{in}}` this is a sum over
    the Jacobian columns:

    .. math::
        \mathbf{\Sigma}^{\text{out}} = \sum_k \sigma_k^2 \, \mathbf{J}_{:,k} \mathbf{J}_{:,k}^T

    so the diagonal path takes one column at a time as a JVP against a tangent that
    is zero everywhere except entry :math:`k`. Those zeros are compile-time
    constants, letting XLA drop whichever parts of the model a given column does not
    reach, and the sum of outer products needs neither the assembled Jacobian nor
    the two matrix products. Forward-mode costs one pass per input dimension either
    way, so covariance propagation remains several times more expensive than a
    centroid: ``argnums=(1, 4, 6, 7)`` is six dimensions.

    This is a Python-level factory; it executes once at definition time.
    The returned ``propagate_fn`` is safe to call standalone or inside another :func:`jax.jit`.
    """
    argnums = tuple(argnums)

    if out_elems is None and diag_out:
        out_elems = ((0, 0), (1, 1), (2, 2), (3, 3))
    if out_elems is not None:
        out_elems = tuple((int(i), int(j)) for i, j in out_elems)
        if not diagonal:
            msg = "out_elems / diag_out require diagonal=True"
            raise ValueError(msg)

    if not diagonal:
        J_fn = jax.jacfwd(centroid_fn, argnums=argnums, has_aux=has_aux)

        def _propagate_full(*args: Any) -> jax.Array:  # noqa: ANN401
            J_out = J_fn(*args[:-1])
            if has_aux:
                J_out, _ = J_out
            return propagate_cov(J_out, args[-1])

        return _propagate_full

    def _propagate(*args: Any) -> jax.Array:  # noqa: ANN401
        fargs, cov_in = list(args[:-1]), args[-1]
        sigmas = jnp.sqrt(jnp.diagonal(cov_in))

        def f(*diff_args: Any) -> jax.Array:  # noqa: ANN401
            full = list(fargs)
            for n, v in zip(argnums, diff_args, strict=True):
                full[n] = v
            out = centroid_fn(*full)
            return out[0] if has_aux else out

        prim = tuple(jnp.asarray(fargs[n]) for n in argnums)
        shapes = [pr.shape for pr in prim]
        sizes = [int(np.prod(sh, dtype=int)) for sh in shapes]
        dims = active_dims if active_dims is not None else tuple(range(sum(sizes)))

        zeros = [jnp.zeros(sh, dtype=pr.dtype) for sh, pr in zip(shapes, prim, strict=True)]

        acc = None
        for k in dims:
            # Locate dimension k within the flattened argnums, then build a tangent
            # that is sigma_k there and a literal zero everywhere else.
            a, off = 0, k
            while off >= sizes[a]:
                off -= sizes[a]
                a += 1
            tan = list(zeros)
            if shapes[a] == ():
                tan[a] = sigmas[k].astype(prim[a].dtype)
            else:
                tan[a] = zeros[a].reshape(-1).at[off].set(sigmas[k]).reshape(shapes[a])

            col = jax.jvp(f, prim, tuple(tan))[1]
            # Indexed outer product rather than jnp.outer, so a centroid function
            # returning several solutions (shape [2,4]) yields [2,4,4] instead of
            # flattening into a single [8,8]. With diag_out only the diagonal of
            # that outer product is formed.
            if out_elems is None:
                outer = col[..., :, None] * col[..., None, :]
            else:
                outer = jnp.stack([col[..., i] * col[..., j]
                                   for i, j in out_elems], axis=-1)
            acc = outer if acc is None else acc + outer

        if acc is None:
            msg = "active_dims selected no input dimensions, nothing to propagate"
            raise ValueError(msg)
        return acc

    return _propagate
