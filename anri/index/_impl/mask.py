"""A sample mask on the indexer's voxel grid: a quick reconstruction of the histogram, then a threshold or a polygon.

The reconstruction is the one ImageD11's ``tomo_2_map`` makes its whole-sample mask from: a sinogram of log
intensities (every spot counts about the same however bright), filtered back-projection with a Hamming-windowed
ramp. The mask is either thresholded (Otsu, then the convex hull, as ImageD11's ``threshold_mask``) or drawn by hand
(:func:`draw_mask`, as ImageD11's ``InteractiveMask``). Images are in reconstruction order, so ``mask.ravel()`` lines
up with the voxels of :func:`anri.geom.recon_positions`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def ramp_filter(sino: ArrayLike) -> np.ndarray:
    """Filter each projection (row of ``sino`` [omega, dty]) along dty with a Hamming-windowed ramp.

    As ImageD11's ``run_iradon`` with ``filter_name="hamming"``, for filtered back-projection.
    """
    sino = np.asarray(sino, float)
    n = sino.shape[1]
    nf = 2 ** int(np.ceil(np.log2(2 * n)))
    fr = np.fft.fftfreq(nf)
    filt = np.abs(fr) * (0.54 + 0.46 * np.cos(2 * np.pi * fr))
    return np.real(np.fft.ifft(np.fft.fft(sino, nf, axis=1) * filt, axis=1))[:, :n]


@jax.jit
def backproject(
    sino: jax.Array, om_deg: jax.Array, pos: jax.Array, y0: float, dty0: float, ystep: float,
    ddty: jax.Array | None = None,
) -> jax.Array:  # fmt: skip
    """Back-project a sinogram [omega, row] onto voxels at ``pos`` [Nv, 3], linearly between rows.

    A voxel is in the beam at dty = y0 - (x sin(omega) + y cos(omega)), as in :func:`anri.index.system`; ``ddty``
    [row, omega] is each row's offset from its nominal dty (fly, helical scans: :func:`anri.index.dty_offsets`).
    """
    n_k = sino.shape[1]
    dd = jnp.zeros((sino.shape[0], n_k), sino.dtype) if ddty is None else jnp.asarray(ddty, sino.dtype).T

    def step(acc: jax.Array, so: tuple) -> tuple:
        row, o, d = so
        fk = (y0 - (pos[:, 0] * jnp.sin(o) + pos[:, 1] * jnp.cos(o)) - dty0) / ystep
        fk = fk - d[jnp.clip(jnp.round(fk).astype(jnp.int32), 0, n_k - 1)] / ystep
        k0 = jnp.floor(fk).astype(jnp.int32)
        t = fk - k0

        def at(k: jax.Array) -> jax.Array:
            return jnp.where((k >= 0) & (k < n_k), row[jnp.clip(k, 0, n_k - 1)], 0.0)

        return acc + (1 - t) * at(k0) + t * at(k0 + 1), None

    acc, _ = jax.lax.scan(step, jnp.zeros(pos.shape[0], jnp.float32), (sino, jnp.radians(om_deg), dd))
    return acc / sino.shape[0]


def reconstruct(H: ArrayLike, n_rings: int, n_e: int, n_o: int, scan: dict, b_o: float, nr: int) -> np.ndarray:
    """Reconstruct the whole sample from the indexer's histogram, for a mask (or to check y0).

    Parameters
    ----------
    H
        [n_rings * n_e * n_o * n_rows] histogram, rows last (:func:`anri.index.histogram_pixels`)
    n_rings, n_e, n_o
        Its rings, eta bins and omega bins
    scan
        "y0", "dty0", "ystep", "n_rows", "om0" and optionally "ddty" (as :func:`anri.index.system`)
    b_o
        Omega bin width (degrees)
    nr
        Voxels per side of the reconstruction grid (:func:`anri.geom.recon_positions`)

    Returns
    -------
    np.ndarray
        [nr, nr] image in reconstruction order
    """
    from anri.geom import recon_positions

    H4 = np.asarray(H).reshape(n_rings, n_e, n_o, scan["n_rows"])
    if "exposure" in scan:  # bins with fewer or more frames than a full one: scaled to one
        ex = np.asarray(scan["exposure"]).T  # [omega, row]
        H4 = H4 / np.where(ex > 0, ex, 1.0)
    sino = np.log1p(H4).sum((0, 1))  # [omega, row]
    om = scan["om0"] + (np.arange(n_o) + 0.5) * b_o
    pos = jnp.asarray(recon_positions(nr, scan["ystep"]), jnp.float32)
    rec = backproject(jnp.asarray(ramp_filter(sino), jnp.float32), jnp.asarray(om, jnp.float32), pos,
                      scan["y0"], scan["dty0"], scan["ystep"], scan.get("ddty"))  # fmt: skip
    return np.asarray(rec).reshape(nr, nr)


def otsu(values: ArrayLike, n_bins: int = 256) -> float:
    """Otsu's threshold of some values: the cut that best separates them into two classes."""
    values = np.asarray(values, float).ravel()
    h, e = np.histogram(values, n_bins)
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    s = np.cumsum(h * c)
    m0 = s / np.maximum(w0, 1)
    m1 = (s[-1] - s) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def threshold_mask(image: ArrayLike, threshold: float | None = None) -> np.ndarray:
    """Mask of the sample: the convex hull of the largest connected region above a threshold (default Otsu's).

    As ImageD11's ``threshold_mask``, but only the largest region (8-connected) goes into the hull: a few noisy
    pixels above the threshold away from the sample would otherwise stretch the hull over them.

    Parameters
    ----------
    image
        [ny, nx] reconstruction, e.g. from :func:`reconstruct`
    threshold
        Pixels above this are sample; default :func:`otsu` of the image

    Returns
    -------
    np.ndarray
        [ny, nx] bool
    """
    from scipy import ndimage
    from scipy.spatial import Delaunay

    image = np.asarray(image, float)
    t = otsu(image) if threshold is None else threshold
    lab, n = ndimage.label(image > t, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(image.shape, bool)
    above = lab == 1 + np.argmax(np.bincount(lab.ravel())[1:])  # the largest region
    pts = np.argwhere(above)
    if len(pts) < 3:
        return above
    grid = np.indices(image.shape).reshape(2, -1).T
    hull = (Delaunay(pts).find_simplex(grid, tol=1e-6) >= 0).reshape(image.shape)
    return hull | above  # pixel centres on the hull's edges, and the region's own pixels


def polygon_mask(shape: tuple, vertices: ArrayLike) -> np.ndarray:
    """Mask [ny, nx] of the pixels inside a polygon given by (x, y) = (column, row) vertices."""
    from matplotlib.path import Path

    yy, xx = np.indices(shape)
    inside = Path(np.asarray(vertices, float)).contains_points(np.stack([xx.ravel(), yy.ravel()], 1))
    return inside.reshape(shape)


def draw_mask(image: ArrayLike, ax: Axes | None = None) -> Callable[[], np.ndarray]:
    """Draw the sample's outline on an image; returns a function that gives the mask of what was drawn.

    As ImageD11's ``InteractiveMask``: click the polygon's corners on the image, close it on the first corner, and
    move corners by dragging. Call the returned function (``get_mask()``) once done, e.g. in the next notebook cell.
    Needs an interactive matplotlib backend (``%matplotlib widget`` in Jupyter).

    Parameters
    ----------
    image
        [ny, nx] image, e.g. from :func:`reconstruct`, shown with ``origin="lower"``
    ax
        Axes to draw on (default: a new figure)

    Returns
    -------
    Callable
        ``get_mask() -> [ny, nx] bool``: inside the polygon (all False until one is drawn)
    """
    from matplotlib import pyplot as plt
    from matplotlib.widgets import PolygonSelector

    image = np.asarray(image)
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    ax.imshow(image, origin="lower", cmap="gray")
    ax.set_title("click the sample's outline; close it on the first corner")
    verts: list = []

    def on_select(v: object, *_: object) -> None:  # called with the polygon's (x, y) vertices
        verts[:] = [tuple(p) for p in np.asarray(v)]

    selector = PolygonSelector(ax, on_select)

    def get_mask() -> np.ndarray:
        if len(verts) < 3:
            return np.zeros(image.shape, bool)
        return polygon_mask(image.shape, np.asarray(verts))

    # the selector must stay alive as long as the figure
    get_mask.selector = selector  # ty: ignore[unresolved-attribute]
    return get_mask
