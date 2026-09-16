r"""Bin-integrated rendering of correlated multivariate Gaussian peaks.

Everything here works in **grid units**: along every axis, bin ``i`` is centred on ``i`` and spans
``[i - 1/2, i + 1/2)``. Detector axes already are (pixels). Motor axes are converted by the caller,
``x -> (x - first_centre) / step``, with the covariance scaled to match.

A peak is a Gaussian :math:`\mathcal N(\mu, \Sigma)` in ``D`` dimensions (``D = 4`` for scanning,
``(sc, fc, omega, dty)``; ``D = 3`` for box beam) with integrated intensity :math:`A`. A rendered
image has two *image* axes, binned, and ``D - 2`` *collapsed* axes, each integrated over one
interval. A detector frame collapses omega and dty onto the frame's own intervals; a sinogram
collapses the two detector axes onto the whole detector, or onto an ROI.

Two steps:

1. :func:`truncate` integrates the collapsed axes. Each interval is applied to the full Gaussian as a
   box truncation, replaced by the Gaussian with the same first two moments. For a single interval
   those moments are exact; after that it is an approximation. The result is a weight :math:`W` and
   a new :math:`(\mu', \Sigma')`. An interval covering the whole peak leaves :math:`(\mu, \Sigma)`
   unchanged, so collapsing a whole axis gives the exact marginal. A narrow interval gives the
   conditional, which is how an off-centre frame gets its shifted, narrower spot.
2. :func:`window` integrates the image axes over a fixed pixel window. Each row, split into
   ``n_sub`` sub-rows, is a truncation of the row axis; the column axis is then integrated under that
   sub-row's conditional. ``n_sub = 0`` ignores the row/column correlation and uses the separable
   outer product.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import erfc

_SQRT2 = math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)
_ZMAX = 12.0  # standardised bounds are clipped here: Phi(-12) ~ 2e-33
_TINY_MASS = 1e-30
_TINY_VAR = 1e-12  # grid units squared. A zero-variance axis behaves as a delta function.
_NARROW = 1e-2  # interval width (in sigma) below which moments use the midpoint expansion


def _cut(x):  # noqa: ANN001, ANN202
    """Break XLA:CPU's elementwise fusion here. A no-op on accelerators.

    XLA:CPU compiles long chains of fused elementwise ops over lazily broadcast inputs very badly
    (measured 7x slower for this kernel than with the chains cut at these points). On GPU the same
    cuts would only stop useful fusion, so they are applied on CPU only, decided at trace time.
    """
    return jax.lax.optimization_barrier(x) if jax.default_backend() == "cpu" else x


def _edges(z: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Sign, upper tail and density at standardised points ``z``.

    With :math:`s = \operatorname{sign}(z)` and :math:`q = \Phi(-|z|)`,
    :math:`\Phi(z) = (1 + s)/2 - s\,q`. Differences of :math:`\Phi` built from ``q`` do not cancel
    in either tail, which plain ``erf`` differences do in float32. ``s * z`` rather than ``abs(z)``
    keeps the derivative right at ``z = 0``.
    """
    z = jnp.clip(z, -_ZMAX, _ZMAX)
    s = jnp.where(z < 0, -1.0, 1.0).astype(z.dtype)
    q = 0.5 * erfc(s * z / _SQRT2)
    p = _INV_SQRT2PI * jnp.exp(-0.5 * z * z)
    return _cut((s, q, p))


def _mass(sa: jax.Array, qa: jax.Array, sb: jax.Array, qb: jax.Array) -> jax.Array:
    r""":math:`\Phi(b) - \Phi(a)` from :func:`_edges` of ``a <= b``."""
    return 0.5 * (sb - sa) - sb * qb + sa * qa


def _moments(za: jax.Array, zb: jax.Array, ea: tuple, eb: tuple) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Mass, mean and variance of a standard normal truncated to ``[za, zb]``.

    Returns
    -------
    mass
        :math:`Z = \Phi(z_b) - \Phi(z_a)`
    lam
        Truncated mean, :math:`(\phi(z_a) - \phi(z_b)) / Z`
    nu
        Truncated variance, :math:`1 + (z_a\phi(z_a) - z_b\phi(z_b))/Z - \lambda^2`, in ``(0, 1]``

    Notes
    -----
    Narrow intervals use the midpoint expansion (mean :math:`c(1 - w^2/12)`, variance
    :math:`w^2/12`), where the closed form cancels to nothing in float32. Intervals with no mass
    return the nearest point of the interval and its uniform variance; they are multiplied by
    ``mass = 0`` downstream, so this only has to be finite, including its gradient.
    """
    za = jnp.clip(za, -_ZMAX, _ZMAX)
    zb = jnp.clip(zb, -_ZMAX, _ZMAX)
    (sa, qa, pa), (sb, qb, pb) = ea, eb
    mass = _mass(sa, qa, sb, qb)

    ok = mass > _TINY_MASS
    safe = jnp.where(ok, mass, 1.0)
    lam = (pa - pb) / safe
    nu = 1.0 + (za * pa - zb * pb) / safe - lam * lam

    width = zb - za
    mid = 0.5 * (za + zb)
    uniform_var = width * width / 12.0
    narrow = width < _NARROW
    lam = jnp.where(narrow, mid * (1.0 - uniform_var), lam)
    nu = jnp.where(narrow, uniform_var, nu)

    mass = jnp.where(narrow, _INV_SQRT2PI * jnp.exp(-0.5 * mid * mid) * width, mass)

    fallback = ~(ok | narrow)
    lam = jnp.where(fallback, jnp.clip(0.0, za, zb), lam)
    nu = jnp.where(fallback, uniform_var, nu)
    return mass, lam, jnp.clip(nu, 1e-9, 1.0)


def _truncation_sweep(
    mu: list[jax.Array], cov: list[list[jax.Array]], lo: list[jax.Array], hi: list[jax.Array], periods: list[jax.Array]
) -> tuple[jax.Array, list[jax.Array], list[list[jax.Array]]]:
    """Truncate axis 0, then 1, ... ``len(lo) - 1`` of a Gaussian given as nested lists of scalars.

    Step ``k`` only touches axes ``>= k``, like a Cholesky sweep. Everything is scalar elementwise
    arithmetic, which XLA:CPU fuses; indexing, small reductions and batched matrix products do not.
    Returns the weight and the moments of the untruncated trailing axes.
    """
    d = len(mu)
    mu = list(mu)
    cov = [list(row) for row in cov]
    weight = 1.0
    for k in range(len(lo)):
        var = jnp.maximum(cov[k][k], _TINY_VAR)
        sd = jnp.sqrt(var)
        a, b, period = lo[k], hi[k], periods[k]
        has_period = period > 0
        safe_period = jnp.where(has_period, period, 1.0)
        shift = jnp.where(has_period, safe_period * jnp.round((mu[k] - 0.5 * (a + b)) / safe_period), 0.0)
        za = (a + shift - mu[k]) / sd
        zb = (b + shift - mu[k]) / sd
        mass, lam, nu = _cut(_moments(za, zb, _edges(za), _edges(zb)))
        weight = weight * mass
        gain = {i: cov[i][k] / var for i in range(k + 1, d)}
        step = sd * lam
        shrink = (1.0 - nu) * var
        for i in range(k + 1, d):
            mu[i] = mu[i] + gain[i] * step
        for i in range(k + 1, d):
            for j in range(i, d):
                cov[i][j] = cov[i][j] - shrink * gain[i] * gain[j]
                cov[j][i] = cov[i][j]
        if k + 1 < len(lo):
            weight, mu, cov = _cut((weight, mu, cov))
    return weight, mu, cov


def _as_lists(mu: jax.Array, cov: jax.Array) -> tuple[list[jax.Array], list[list[jax.Array]]]:
    d = mu.shape[-1]
    return [mu[..., i] for i in range(d)], [[cov[..., i, j] for j in range(d)] for i in range(d)]


def truncate(
    mu: jax.Array, cov: jax.Array, lo: jax.Array, hi: jax.Array, axes: tuple[int, ...], periods: tuple[float, ...]
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Integrate one peak over an interval along each of ``axes``, in that order.

    Parameters
    ----------
    mu
        [..., D] peak centres, grid units
    cov
        [..., D, D] peak covariances, grid units
    lo, hi
        [..., K] interval bounds for each of ``axes``, grid units, ``lo <= hi``
    axes
        Static. The ``K`` axes to integrate.
    periods
        Static, length ``D``. Period of each axis in grid units, ``0`` if not periodic. On a periodic
        axis the interval is shifted by whole periods to sit as close to ``mu`` as possible.

    Returns
    -------
    weight: jax.Array
        [...] fraction of each peak inside the intervals.
    mu: jax.Array
        [..., D] means after truncation
    cov: jax.Array
        [..., D, D] covariances after truncation

    Notes
    -----
    For a truncation along axis :math:`c` with standardised truncated mean :math:`\lambda` and
    variance :math:`\nu`, and gain :math:`g = \Sigma_{:,c} / \Sigma_{cc}`:

    .. math::
        \mu' = \mu + g\,\sigma_c\lambda, \qquad
        \Sigma' = \Sigma - (1 - \nu)\,\Sigma_{cc}\, g g^T

    These are the exact moments of the truncated joint, because the other axes are Gaussian and
    linear given axis :math:`c`. Each further truncation treats the result as Gaussian.
    """
    d = mu.shape[-1]
    order = list(axes) + [i for i in range(d) if i not in axes]
    m, c = _as_lists(mu, cov)
    m = [m[i] for i in order]
    c = [[c[i][j] for j in order] for i in order]
    w, m, c = _truncation_sweep(m, c, [lo[..., k] for k in range(len(axes))], [hi[..., k] for k in range(len(axes))],
                                [jnp.asarray(periods[i], mu.dtype) for i in axes])
    back = {ax: n for n, ax in enumerate(order)}
    w = jnp.broadcast_to(w, mu.shape[:-1])
    mu_out = jnp.stack([jnp.broadcast_to(m[back[i]], mu.shape[:-1]) for i in range(d)], axis=-1)
    cov_out = jnp.stack(
        [jnp.stack([jnp.broadcast_to(c[back[i]][back[j]], mu.shape[:-1]) for j in range(d)], axis=-1) for i in range(d)],
        axis=-2)
    return w, mu_out, cov_out


def _col_mass(ec: jax.Array, mean: jax.Array, var: jax.Array) -> jax.Array:
    """[..., n] Gaussian mass between consecutive edges ``ec`` [n+1]."""
    z = (ec - mean[..., None]) / jnp.sqrt(jnp.maximum(var, _TINY_VAR))[..., None]
    s, q, _ = _edges(z)
    return _mass(s[..., :-1], q[..., :-1], s[..., 1:], q[..., 1:])


def _select(onehot: list[jax.Array], values: list) -> jax.Array:  # noqa: D103
    """``values[i]`` where ``onehot[i]`` is 1, as arithmetic."""
    out = onehot[0] * values[0]
    for h, v in zip(onehot[1:], values[1:]):
        out = out + h * v
    return out


def splat_peaks(
    mu: jax.Array,
    cov: jax.Array,
    amp: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    shape: tuple[int, ...],
    image: tuple[int, int],
    win: tuple[int, int],
    n_sub: int,
    periods: tuple[float, ...],
) -> tuple[jax.Array, jax.Array]:
    r"""Integrate peaks over their collapsed intervals and a pixel window of the image axes.

    Arguments as :func:`splat`, except that ``lo`` and ``hi`` must be [P,D-2].

    Returns
    -------
    blocks: jax.Array
        [P, win[0], win[1]] intensity in each window pixel
    starts: jax.Array
        [P, 2] int32 bin index of each window's first (row, column). Not wrapped on periodic axes.

    Notes
    -----
    The window is centred on the peak's mean *given* the collapsed intervals, so an off-centre frame
    gets its spot where the conditional puts it.

    With ``n_sub = 0`` the collapsed axes are applied first and the image axes are then treated as
    independent: an outer product, ``win[0] + win[1] + 2`` erfcs.

    Otherwise one image axis (the *row* axis) is cut into ``n_sub`` sub-bins per bin. For each
    sub-bin the row interval and the collapsed intervals are applied as successive truncations
    (:func:`truncate`), and the other image axis is integrated exactly under what remains. The row
    axis is the wider image axis for a square window, else the axis with the longer window: sub-bins
    of the wide axis pin the peak down, sub-bins of the narrow one do not.

    Truncation order matters. A truncation replaces what it leaves with a Gaussian, which is a poor
    stand-in when the interval cuts through the middle of the peak, and a good one when the interval
    is narrow compared with the peak. So the truncations run in order of interval width over marginal
    sigma, narrowest first. For a streak whose motion across the detector is tied to omega (wavelength
    spread), detector sub-rows come first and pin omega before the frame interval is applied; for a
    peak much wider in omega than one frame, the frame comes first. Over 324 physically shaped peaks
    (wavelength streak, divergence, 1 px PSF, frames cutting through the peak) this rule matched the
    better of the two fixed orders in every case; either fixed order alone is up to 20% wrong at
    the brightest pixel. With ``n_sub = 2`` the error is 5e-4 at p90 and 1.4% at worst (a pure
    wavelength streak at 45 degrees azimuth, cut mid-peak by the frame).

    The same argument applies between collapsed axes: two strongly correlated motor intervals (dty
    tied to omega near the sample edge, with coarse omega steps) make the second truncation cut a
    thin diagonal ridge, and the box weight alone can be 20% off. So the first collapsed interval in
    the order is also cut into ``n_sub`` equal pieces; each sub-row's pieces are merged back into one
    Gaussian before the column is integrated, so the column cost does not grow.

    All arithmetic is done relative to the peak's rounded centre, so float32 keeps its precision on
    long axes (omega bin 7000 has a float32 resolution of 5e-4 bins; an offset of 0.3 has 3e-8).

    Costs about ``win_r * n_sub * (win_c + 1 + 2 * n_sub * (K + 1))`` erfcs for ``K`` collapsed axes.

    This is a batched function rather than a per-peak one for speed on CPU. XLA:CPU generates tight
    loops for 1-D arrays of one shape; a per-peak function under ``vmap`` broadcasts per-peak values
    against the (sub-row, piece) grid inside every operation, which was 5x slower. So per-peak values
    are repeated and the grid is flattened before the expensive parts.
    """
    n, d = mu.shape
    r0, c0 = image
    collapsed = tuple(i for i in range(d) if i not in image)
    kc = len(collapsed)
    dt = mu.dtype

    ref = jnp.floor(mu + 0.5)
    mu = mu - ref
    ref_c = ref[:, list(collapsed)] if kc else jnp.zeros((n, 0), dt)
    lo = lo - ref_c
    hi = hi - ref_c

    w, m, p = truncate(mu, cov, lo, hi, collapsed, periods)
    sizes = (shape[r0], shape[c0])
    periodic = (bool(periods[r0]), bool(periods[c0]))
    ref_img = jnp.stack([ref[:, r0], ref[:, c0]], axis=-1)
    centre = jnp.stack([m[:, r0], m[:, c0]], axis=-1)
    start = jnp.floor(centre + 0.5).astype(jnp.int32) + ref_img.astype(jnp.int32)
    start = start - jnp.array([win[0] // 2, win[1] // 2], jnp.int32)
    start = jnp.stack(
        [start[:, i] if periodic[i] else jnp.clip(start[:, i], 0, sizes[i] - win[i]) for i in range(2)], axis=-1)
    fstart = start.astype(dt) - ref_img  # relative to ref

    if n_sub == 0:
        row = _col_mass(fstart[:, 0, None] - 0.5 + jnp.arange(win[0] + 1, dtype=dt), m[:, r0], p[:, r0, r0])
        col = _col_mass(fstart[:, 1, None] - 0.5 + jnp.arange(win[1] + 1, dtype=dt), m[:, c0], p[:, c0, c0])
        return (amp * w)[:, None, None] * (row[:, :, None] * col[:, None, :]), start

    if win[0] != win[1]:
        swap = jnp.full(n, win[1] > win[0])
    else:
        swap = p[:, c0, c0] > p[:, r0, r0]
    wr, wc = (win[1], win[0]) if win[1] > win[0] else win
    fs_r = jnp.where(swap, fstart[:, 1], fstart[:, 0])
    fs_c = jnp.where(swap, fstart[:, 0], fstart[:, 1])
    ns = n_sub
    n_steps = 1 + kc
    n_pieces = ns if kc else 1
    n_rows = wr * ns
    grid = n_rows * n_pieces

    # Axes of the n_steps intervals as one-hot lists over the d axes: interval 0 is the row
    # sub-bin, then the collapsed axes.
    zero = jnp.zeros(n, dt)
    one = jnp.ones(n, dt)
    is_c0 = swap.astype(dt)
    row_onehot = [zero] * d
    row_onehot[r0] = 1.0 - is_c0
    row_onehot[c0] = is_c0
    col_onehot = [zero] * d
    col_onehot[c0] = 1.0 - is_c0
    col_onehot[r0] = is_c0
    interval_axes = [row_onehot] + [[one if i == ax else zero for i in range(d)] for ax in collapsed]
    var_diag = [cov[:, i, i] for i in range(d)]

    # Order: narrowest interval (relative to its marginal sigma) first. Rank by pairwise comparison.
    widths = [jnp.full(n, 1.0 / ns, dt)] + [hi[:, k] - lo[:, k] for k in range(kc)]
    ratio = [widths[j] / jnp.sqrt(jnp.maximum(_select(interval_axes[j], var_diag), _TINY_VAR))
             for j in range(n_steps)]
    rank = [sum(((ratio[i] < ratio[j]) | ((ratio[i] == ratio[j]) & (i < j))).astype(jnp.int32)
                for i in range(n_steps) if i != j) + jnp.zeros(n, jnp.int32)
            for j in range(n_steps)]
    at_step = [[(rank[j] == k).astype(dt) for j in range(n_steps)] for k in range(n_steps)]  # [step][interval]

    # Axis permutation: the step axes in order, then the column axis, as one-hot rows over d axes.
    perm_rows = [[_select(at_step[k], [interval_axes[j][i] for j in range(n_steps)]) for i in range(d)]
                 for k in range(n_steps)] + [col_onehot]
    mu_l, cov_l = _as_lists(mu, cov)
    mu_p = [_select(perm_rows[a], mu_l) for a in range(d)]
    cov_p = [[None] * d for _ in range(d)]
    for a in range(d):
        row_a = [_select(perm_rows[a], [cov_l[i][jj] for i in range(d)]) for jj in range(d)]
        for b in range(a, d):
            cov_p[a][b] = cov_p[b][a] = _select(perm_rows[b], row_a)
    per_arr = [jnp.full(n, periods[i], dt) for i in range(d)]
    per_p = [_select(perm_rows[k], per_arr) for k in range(n_steps)]

    # Intervals on the [n, n_rows, n_pieces] grid, placed in step order.
    edges = fs_r[:, None] - 0.5 + jnp.arange(n_rows + 1, dtype=dt) / ns
    frac = jnp.arange(n_pieces + 1, dtype=dt) / n_pieces
    full = jnp.zeros((n, n_rows, n_pieces), dt)
    ivl_lo = [edges[:, :-1, None] + full]
    ivl_hi = [edges[:, 1:, None] + full]
    if kc:
        min_rank = rank[1]
        for j in range(2, n_steps):
            min_rank = jnp.minimum(min_rank, rank[j])
        for j in range(1, n_steps):
            split = (rank[j] == min_rank)[:, None, None]
            l, h = lo[:, j - 1, None, None], hi[:, j - 1, None, None]
            ivl_lo.append(jnp.where(split, l + (h - l) * frac[:-1], l) + full)
            ivl_hi.append(jnp.where(split, l + (h - l) * frac[1:], h) + full)
    flat = lambda x: x.reshape(-1)  # noqa: E731
    rep = lambda x: jnp.repeat(x, grid)  # noqa: E731
    step_lo = [flat(_select([h[:, None, None] for h in at_step[k]], ivl_lo)) for k in range(n_steps)]
    step_hi = [flat(_select([h[:, None, None] for h in at_step[k]], ivl_hi)) for k in range(n_steps)]

    sweep_in = _cut(
        ([rep(x) for x in mu_p], [[rep(cov_p[i][j]) for j in range(d)] for i in range(d)], step_lo, step_hi,
         [rep(x) for x in per_p]))
    ws, m_end, c_end = _truncation_sweep(*sweep_in)
    # Merge each sub-row's pieces into one Gaussian along the column. Everything below stays 1-D:
    # pieces are strided slices of the (peak, sub-row, piece) layout.
    mq = m_end[-1]  # column mean, relative to ref
    vq = c_end[-1][-1]
    w1, w2 = _cut((ws * mq, ws * (vq + mq * mq)))
    ws, w1, w2 = ws.reshape(-1, n_pieces), w1.reshape(-1, n_pieces), w2.reshape(-1, n_pieces)
    wt, s1, s2 = ws[:, 0], w1[:, 0], w2[:, 0]
    for q in range(1, n_pieces):
        wt = wt + ws[:, q]
        s1 = s1 + w1[:, q]
        s2 = s2 + w2[:, q]
    safe = jnp.where(wt > _TINY_MASS, wt, 1.0)
    mean_c = s1 / safe
    inv_sd = 1.0 / jnp.sqrt(jnp.maximum(s2 / safe - mean_c * mean_c, _TINY_VAR))
    wt, mean_c, inv_sd = _cut((wt, mean_c, inv_sd))

    # Column integral on a (peak, row, column, sub-row) layout, so that summing sub-rows is a
    # strided slice. Per-(peak, sub-row) values are tiled across the columns.
    per_col = lambda x: jnp.tile(x.reshape(n * wr, ns), (1, wc)).reshape(-1)  # noqa: E731
    col_idx = jnp.repeat(jnp.arange(wc, dtype=dt), ns)
    left = jnp.repeat(fs_c, wr)[:, None] - 0.5 + col_idx[None, :]
    left, k_mean, k_inv, k_w = _cut(
        (left.reshape(-1), per_col(mean_c), per_col(inv_sd), per_col(wt)))
    zl = (left - k_mean) * k_inv
    zr = zl + k_inv
    sl, ql, _ = _edges(zl)
    sr, qr, _ = _edges(zr)
    val = _cut(k_w * _mass(sl, ql, sr, qr))
    block = val[0::ns]
    for t in range(1, ns):
        block = block + val[t::ns]
    block = block.reshape(n, wr, wc)
    if win[0] == win[1]:
        block = jnp.where(swap[:, None, None], block.transpose(0, 2, 1), block)
    elif win[1] > win[0]:
        block = block.transpose(0, 2, 1)
    return amp[:, None, None] * block, start


def flat_index(start: jax.Array, win: tuple[int, int], sizes: tuple[int, int], periodic: tuple[bool, bool]) -> jax.Array:
    """[..., win[0]*win[1]] int32 row-major pixel index of each window pixel, wrapping periodic axes."""
    rows = start[..., 0, None] + jnp.arange(win[0], dtype=jnp.int32)
    cols = start[..., 1, None] + jnp.arange(win[1], dtype=jnp.int32)
    if periodic[0]:
        rows = rows % sizes[0]
    if periodic[1]:
        cols = cols % sizes[1]
    return (rows[..., :, None] * sizes[1] + cols[..., None, :]).reshape(*start.shape[:-1], -1)


@functools.partial(jax.jit, static_argnames=("shape", "image", "win", "n_sub", "periods"))
def splat(
    mu: jax.Array,
    cov: jax.Array,
    amp: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    shape: tuple[int, ...],
    image: tuple[int, int] = (0, 1),
    win: tuple[int, int] = (9, 9),
    n_sub: int = 2,
    periods: tuple[float, ...] | None = None,
) -> jax.Array:
    r"""Render peaks into one dense image. No culling, no sharding: every peak is evaluated.

    Differentiable with respect to ``mu``, ``cov``, ``amp``, ``lo`` and ``hi``.

    Parameters
    ----------
    mu
        [P,D] peak centres, grid units
    cov
        [P,D,D] peak covariances, grid units
    amp
        [P] integrated intensities
    lo, hi
        [D-2] or [P,D-2] interval of each collapsed axis, in increasing axis order, grid units
    shape
        Static. Number of bins along each of the ``D`` axes.
    image
        Static. The two image axes, (row, column).
    win
        Static. Window per peak, in bins. Intensity outside it is lost.
    n_sub
        Static. Sub-rows per row for the row/column correlation, 0 to ignore it.
    periods
        Static. Period of each axis in grid units, 0 if not periodic. Default: none periodic.

    Returns
    -------
    image: jax.Array
        [shape[image[0]], shape[image[1]]]
    """
    periods = periods if periods is not None else (0.0,) * len(shape)
    lo = jnp.broadcast_to(lo, (mu.shape[0], len(shape) - 2))
    hi = jnp.broadcast_to(hi, (mu.shape[0], len(shape) - 2))
    blocks, starts = splat_peaks(mu, cov, amp, lo, hi, shape, image, win, n_sub, periods)
    sizes = (shape[image[0]], shape[image[1]])
    periodic = (bool(periods[image[0]]), bool(periods[image[1]]))
    idx = flat_index(starts, win, sizes, periodic)
    out = jnp.zeros(sizes[0] * sizes[1], blocks.dtype)
    out = out.at[idx.reshape(-1)].add(blocks.reshape(-1), mode="promise_in_bounds")
    return out.reshape(sizes)
