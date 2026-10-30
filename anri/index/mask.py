"""A sample mask on the indexer's voxel grid: the sample's support in the projections, the scanned disc, or a polygon.

The default (:func:`support_mask`) keeps the voxels whose row is lit at every omega the scan covers them: a voxel in
air has some omega where the beam through it misses the sample. Within the scanned dty range that is the sample's
convex hull, wherever the beam reaches it, also outside the disc it sees at every omega (:func:`disc_mask`), so a
sample larger than the disc is fitted whole instead of piling its intensity onto the disc's edge. A quick
reconstruction (:func:`reconstruct`, as ImageD11's ``tomo_2_map``: log intensities, filtered back-projection) can be
thresholded instead (Otsu, then the convex hull, as ImageD11's ``threshold_mask``), or a mask drawn by hand
(:func:`draw_mask`, as ImageD11's ``InteractiveMask``). Images are in reconstruction order, so ``mask.ravel()`` lines
up with the voxels of :func:`anri.geom.scan.recon_positions`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from scipy.ndimage import median_filter

from anri.geom.gonio import find_dty_for_beam_xy
from anri.geom.scan import recon_positions, regrid

from .data import scan_beam

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def ramp_filter(sino: ArrayLike) -> np.ndarray:
    """Filter each projection (row of ``sino`` [omega, dty]) along dty with a Hamming-windowed ramp.

    As ImageD11's ``run_iradon`` with ``filter_name="hamming"``, for filtered back-projection (Kak and Slaney,
    1988) [1]_.

    References
    ----------
    .. [1] Kak, A.C., Slaney, M., 1988. Principles of Computerized Tomographic Imaging. IEEE Press, New York. Reprinted
       2001, SIAM, Philadelphia. https://doi.org/10.1137/1.9780898719277
    """
    sino = np.asarray(sino, float)
    n = sino.shape[1]
    nf = 2 ** int(np.ceil(np.log2(2 * n)))
    fr = np.fft.fftfreq(nf)
    filt = np.abs(fr) * (0.54 + 0.46 * np.cos(2 * np.pi * fr))
    return np.real(np.fft.ifft(np.fft.fft(sino, nf, axis=1) * filt, axis=1))[:, :n]


def _rows(pos: jax.Array, k_in: jax.Array, o: jax.Array, d: jax.Array, wedge: float, chi: float, y0: float,
          dty0: float, ystep: float, n_k: int) -> tuple[jax.Array, jax.Array]:  # fmt: skip
    """Row below each voxel's beam position at omega o, and how far towards the next (``d``: the rows' dty offsets)."""
    fk = (find_dty_for_beam_xy(pos, k_in, o, wedge, chi, y0) - dty0) / ystep
    fk = fk - d[jnp.clip(jnp.round(fk).astype(jnp.int32), 0, n_k - 1)] / ystep
    k0 = jnp.floor(fk).astype(jnp.int32)
    return k0, fk - k0


@jax.jit
def backproject(
    sino: jax.Array, om_deg: jax.Array, pos: jax.Array, y0: float, dty0: float, ystep: float,
    ddty: jax.Array | None = None, k_in_lab: ArrayLike | None = None, wedge: float = 0.0, chi: float = 0.0,
) -> jax.Array:  # fmt: skip
    """Back-project a sinogram [omega, row] onto voxels at ``pos`` [Nv, 3], linearly between rows.

    A voxel is in the beam at the dty of :func:`anri.geom.gonio.find_dty_for_beam_xy` (beam ``k_in_lab``, rotation axis
    tilted by ``wedge`` and ``chi``), as in :func:`anri.index.occupancy.system`; ``ddty`` [row, omega] is each row's offset
    from its nominal dty (fly, helical scans: :func:`anri.index.data.dty_offsets`).
    """
    n_k = sino.shape[1]
    dd = jnp.zeros((sino.shape[0], n_k), sino.dtype) if ddty is None else jnp.asarray(ddty, sino.dtype).T

    def step(acc: jax.Array, so: tuple) -> tuple:
        row, o, d = so
        k0, t = _rows(pos, k_in, o, d, wedge, chi, y0, dty0, ystep, n_k)

        def at(k: jax.Array) -> jax.Array:
            return jnp.where((k >= 0) & (k < n_k), row[jnp.clip(k, 0, n_k - 1)], 0.0)

        return acc + ((1 - t) * at(k0) + t * at(k0 + 1)).astype(acc.dtype), None

    k_in = jnp.asarray((1.0, 0.0, 0.0) if k_in_lab is None else k_in_lab)  # default: along lab x
    acc, _ = jax.lax.scan(step, jnp.zeros(pos.shape[0], jnp.float32), (sino, jnp.asarray(om_deg), dd))
    return acc / sino.shape[0]


def reconstruct(H: ArrayLike, n_rings: int, n_e: int, n_o: int, scan: dict, b_o: float, nr: int) -> np.ndarray:
    """Reconstruct the whole sample from the indexer's histogram, for a mask (or to check y0).

    Parameters
    ----------
    H
        [n_rings * n_e * n_o * n_rows] histogram, rows last (:func:`anri.index.data.histogram_pixels`)
    n_rings, n_e, n_o
        Its rings, eta bins and omega bins
    scan
        "y0", "dty0", "ystep", "n_rows", "om0" and optionally "ddty" (as :func:`anri.index.occupancy.system`)
    b_o
        Omega bin width (degrees)
    nr
        Voxels per side of the reconstruction grid (:func:`anri.geom.scan.recon_positions`)

    Returns
    -------
    np.ndarray
        [nr, nr] image in reconstruction order
    """
    from anri.geom.scan import recon_positions

    H4 = np.asarray(H).reshape(n_rings, n_e, n_o, scan["n_rows"])
    if "exposure" in scan:  # bins with fewer or more frames than a full one: scaled to one
        ex = np.asarray(scan["exposure"]).T  # [omega, row]
        H4 = H4 / np.where(ex > 0, ex, 1.0)
    sino = np.log1p(H4).sum((0, 1))  # [omega, row]
    om = scan["om0"] + (np.arange(n_o) + 0.5) * b_o
    pos = jnp.asarray(recon_positions(nr, scan["ystep"]), jnp.float32)
    k_in, wedge, chi = scan_beam(scan)
    rec = backproject(jnp.asarray(ramp_filter(sino), jnp.float32), jnp.asarray(om, jnp.float32), pos,
                      scan["y0"], scan["dty0"], scan["ystep"], scan.get("ddty"), k_in, wedge, chi)  # fmt: skip
    return np.asarray(rec).reshape(nr, nr)


@jax.jit
def support(
    sino: jax.Array, om_deg: jax.Array, pos: jax.Array, y0: float, dty0: float, ystep: float,
    ddty: jax.Array | None = None, k_in_lab: ArrayLike | None = None, wedge: float = 0.0, chi: float = 0.0,
) -> jax.Array:  # fmt: skip
    """Smallest value of a sinogram [omega, row] along each voxel's rows, over the omegas the scan covers it.

    The same rows as :func:`backproject` (linearly between two); omegas where the voxel is outside the scanned rows
    are skipped. A voxel the scan never covers gets 0.
    """
    n_k = sino.shape[1]
    dd = jnp.zeros((sino.shape[0], n_k), sino.dtype) if ddty is None else jnp.asarray(ddty, sino.dtype).T

    def step(acc: jax.Array, so: tuple) -> tuple:
        row, o, d = so
        k0, t = _rows(pos, k_in, o, d, wedge, chi, y0, dty0, ystep, n_k)
        inside = (k0 >= 0) & (k0 + 1 < n_k)
        v = (1 - t) * row[jnp.clip(k0, 0, n_k - 1)] + t * row[jnp.clip(k0 + 1, 0, n_k - 1)]
        return jnp.where(inside, jnp.minimum(acc, v.astype(acc.dtype)), acc), None

    k_in = jnp.asarray((1.0, 0.0, 0.0) if k_in_lab is None else k_in_lab)
    acc, _ = jax.lax.scan(step, jnp.full(pos.shape[0], jnp.inf, jnp.float32), (sino, jnp.asarray(om_deg), dd))
    return jnp.where(jnp.isfinite(acc), acc, 0.0)


def support_mask(
    H: ArrayLike, n_rings: int, n_e: int, n_o: int, scan: dict, b_o: float, nr: int, level: float = 0.03,
    median_omega: int = 5,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Mask of the sample: the voxels whose row is lit at every omega the scan covers them (its convex hull).

    Each omega's projection (the histogram summed over rings and eta, per row) is scaled to its 99th-percentile row
    and median-filtered along omega; a voxel is sample if its scaled row stays above ``level`` at every omega it is
    scanned at
    (:func:`support`). Then, as :func:`threshold_mask`, the largest region and its convex hull. Absorption only dims
    the rows through a thick sample (here never below ~10%), so it does not cut into the support.

    Parameters
    ----------
    H, n_rings, n_e, n_o, scan, b_o, nr
        As :func:`reconstruct`
    level
        Fraction of an omega's bright rows below which a row has missed the sample
    median_omega
        Omega bins of the median filter applied to each row first (a few bins: removes isolated dark bins, keeps the
        sample's edge)

    Returns
    -------
    mask, image: np.ndarray
        [nr, nr] bool and the support values (smallest scaled row), in reconstruction order
    """
    H4 = np.asarray(H).reshape(n_rings, n_e, n_o, scan["n_rows"])
    if "exposure" in scan:
        ex = np.asarray(scan["exposure"]).T
        H4 = H4 / np.where(ex > 0, ex, 1.0)
    sino = H4.sum((0, 1))  # [omega, row]
    sino = sino / np.maximum(np.percentile(sino, 99, axis=1, keepdims=True), 1e-30)
    # a single dark (omega, row) bin would cut a line through the mask; the sample's edge is smooth in omega
    sino = median_filter(sino, size=(median_omega, 1), mode="wrap")
    om = scan["om0"] + (np.arange(n_o) + 0.5) * b_o
    pos = jnp.asarray(recon_positions(nr, scan["ystep"]), jnp.float32)
    k_in, wedge, chi = scan_beam(scan)
    img = np.asarray(support(jnp.asarray(sino, jnp.float32), jnp.asarray(om, jnp.float32), pos, scan["y0"],
                             scan["dty0"], scan["ystep"], scan.get("ddty"), k_in, wedge, chi)).reshape(nr, nr)  # fmt: skip
    return threshold_mask(img, level), img


def otsu(values: ArrayLike, n_bins: int = 256) -> float:
    """Otsu's threshold of some values: the cut that best separates them into two classes.

    The cut that maximises the between-class variance (Otsu, 1979) [1]_.

    References
    ----------
    .. [1] Otsu, N., 1979. A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man,
       and Cybernetics 9, 62–66. https://doi.org/10.1109/TSMC.1979.4310076
    """
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


def disc_mask(scan: dict, y0: float, nr: int, pad: float = 0.0) -> np.ndarray:
    """Mask of the voxels the beam sees at every omega: within the scanned dty range all the way round the axis.

    A voxel at distance r from the rotation axis is in the beam at dty = y0 - r cos(omega - phi). It stays within the
    rows' range, from dty0 - ystep / 2 to dty0 + (n_rows - 1/2) ystep, at every omega when r is at most the distance
    from y0 to the nearer end. The voxels further out (the corners of the square grid, and more of it when y0 is off
    the middle of the scan) are seen at some omegas only, so the data hardly constrain them: MLEM can give them large
    occupancies that explain almost nothing.

    Parameters
    ----------
    scan
        The histogram's scan ("dty0", "ystep", "n_rows")
    y0
        dty where the rotation axis is in the beam
    nr
        Voxels per side (:func:`anri.index.stages.voxel_grid`)
    pad
        Grow the disc by this (sample units): a sample a little larger than the disc is then fitted whole, instead of
        its outer part's intensity piling onto the disc's edge. A voxel at radius r is in the beam over a fraction
        (2 / pi) asin(r_disc / r) of the rotation (e.g. 2/3 at r = 1.15 r_disc), so a modest pad stays well
        constrained

    Returns
    -------
    np.ndarray
        [nr, nr] bool, in reconstruction order
    """
    lo = scan["dty0"] - 0.5 * scan["ystep"]
    r_max = min(y0 - lo, lo + scan["n_rows"] * scan["ystep"] - y0) + pad
    pos = np.asarray(recon_positions(nr, scan["ystep"]))
    return (np.hypot(pos[:, 0], pos[:, 1]) <= r_max).reshape(nr, nr)


def save_mask(path: str, mask: ArrayLike, step: float) -> None:
    """Save a mask in reconstruction order with its voxel size (an .npz), for :func:`load_mask`."""
    np.savez(path, mask=np.asarray(mask, bool), step=float(step))


def load_mask(path: str, n: int, step: float, log: Callable = print) -> np.ndarray:
    """Load a mask onto an n x n grid of voxels of ``step``, centred on the rotation axis.

    An .npz from :func:`save_mask` carries its voxel size, which must be ``step``; a bare .npy (e.g. an ImageD11
    mask, on its reconstruction grid) is taken to have voxels of ``step``. Either is cropped or padded about the
    axis to n x n (:func:`anri.geom.scan.regrid`).
    """
    if str(path).endswith(".npz"):
        z = np.load(path)
        mask, mstep = z["mask"], float(z["step"])
        if abs(mstep - step) > 1e-6 * step:
            msg = f"mask {path} has voxels of {mstep:g}, the grid {step:g}"
            raise ValueError(msg)
    else:
        mask = np.load(path)
    mask = np.asarray(mask, bool)
    if mask.shape != (n, n):
        log(f"mask {path}: {mask.shape[0]} x {mask.shape[1]} voxels, centred onto the {n} x {n} grid")
        mask = regrid(mask, n, fill=False)
    return mask


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
