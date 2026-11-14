"""The detector's response, measured on isolated spots: a point spread that is a core and a wing.

A spot on the detector, in one frame, is its source's intensity spread by what is known (the thick sensor's depth
kernel, the beam's energy bandwidth, the illuminated voxel's width across the beam) and by the detector's own point
spread, which is what is fitted here: a mixture (1 - f) N(0, s1^2) + f N(0, s2^2), the same in every direction. A
photon-counting CdTe detector is a sharp core and a wing of about a pixel (fluorescence, counted when the lower
threshold is below the Cd / Te K lines), which one Gaussian cannot be. :func:`anri.fwd.render.render_row` takes the
result as ``geom["psf_mix"]``.

Each spot is fitted in a small window of its raw frame (every pixel observed: no segmentation cut), in its own radial
(u, the direction in which 2theta grows) and tangential (t) frame, where every term is a Gaussian (the sensor kernel
as :data:`anri.fwd.render.SENSOR_SLICES` slices, as the renderer has it), so the model of a pixel is a sum of
separable Gaussians integrated over the pixel (sub-samples). Per spot: intensity, radial and tangential centre,
tangential sd (its orientation spread and the beam's), background, and optionally an extra radial sd (a deformed
sample's strain spread: then only a tail heavier than a Gaussian is the detector's). Shared: s1, s2, f.

:func:`find_spots` picks bright, isolated spots from the sparse pixels; :func:`spot_terms` works out each spot's
known blurs; :func:`fit_response` fits one Gaussian, then the core and wing.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SUB = 4  # sub-samples per pixel side


def find_spots(frame: np.ndarray, row: np.ndarray, col: np.ndarray, value: np.ndarray, shape: tuple, min_counts: float,
               window: int) -> dict:  # fmt: skip
    """Bright spots alone in their window: 8-connected blobs of sparse pixels, in one frame each.

    Parameters
    ----------
    frame, row, col, value
        Sparse pixels (e.g. :func:`anri.io.imaged11.read_sparse_frames`)
    shape
        (n_slow, n_fast)
    min_counts
        A blob's total counts
    window
        Window side (odd): the spot's brightest pixel at its centre, inside the detector, with no other blob's pixel

    Returns
    -------
    dict
        "frame", "row", "col" (the window's centre) and "counts" of each spot
    """
    ns, nf = shape
    h = window // 2
    f, r, c = (np.asarray(x, np.int64) for x in (frame, row, col))
    v = np.asarray(value, np.float64)
    key = (f * ns + r) * nf + c
    o = np.argsort(key)
    ks = key[o]
    src, dst = [], []
    for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):  # each neighbour pair once
        q = key + dr * nf + dc
        at = np.searchsorted(ks, q)
        hit = (at < len(ks)) & (ks[np.minimum(at, len(ks) - 1)] == q) & (c + dc >= 0) & (c + dc < nf)
        src.append(np.nonzero(hit)[0])
        dst.append(o[at[hit]])
    src, dst = np.concatenate(src), np.concatenate(dst)
    _, lab = connected_components(coo_matrix((np.ones(len(src)), (src, dst)), shape=(len(v),) * 2), directed=False)
    tot = np.bincount(lab, v)
    peak = np.full(len(tot), -1)
    order = np.lexsort((v, lab))  # per blob, its brightest pixel last
    peak[lab[order]] = order
    fo = np.argsort(f, kind="stable")  # each frame's pixels, by binary search: a spot looks only at its own frame
    fs = f[fo]
    out = {"frame": [], "row": [], "col": [], "counts": []}
    for b in np.flatnonzero(tot >= min_counts):
        k = peak[b]
        fr, ci, cj = f[k], r[k], c[k]
        if ci < h or cj < h or ci >= ns - h or cj >= nf - h:
            continue
        idx = fo[np.searchsorted(fs, fr) : np.searchsorted(fs, fr, side="right")]
        near = (np.abs(r[idx] - ci) <= h) & (np.abs(c[idx] - cj) <= h)
        if np.any(near & (lab[idx] != b)):
            continue
        for name, x in (("frame", fr), ("row", ci), ("col", cj), ("counts", tot[b])):
            out[name].append(x)
    return {k: np.asarray(x) for k, x in out.items()}


def spot_terms(rows: np.ndarray, cols: np.ndarray, pars: dict, spatial: tuple | None, sensor: tuple | None,
               bandwidth: float, voxel: float, thickness: float = 0.0, convergence: tuple = (0.0, 0.0)) -> dict:  # fmt: skip
    """Each spot's known radial blurs and directions, at its window centre.

    Parameters
    ----------
    rows, cols
        The windows' centres (raw pixels)
    pars
        ImageD11 geometry parameters
    spatial
        (dx, dy) distortion maps (:func:`anri.io.imaged11.read_spatial`), or None
    sensor
        (absorption length, thickness) of the sensor, um (the par file's units), or None
    bandwidth
        FWHM of d(lambda) / lambda
    voxel
        Width of the illuminated volume across the beam (horizontal), um
    convergence
        (horizontal, vertical) standard deviation of the incoming beam's direction, rad (the renderer's sig_ky,
        sig_kz). A tilted beam diffracts at a slightly different omega, within the frame, and moves the spot: radially
        at eta = +-90 deg for the horizontal one, at eta = 0, 180 for the vertical one
    thickness
        Its length along the beam, um: a source further along the beam lands at another radius (about
        thickness x tan(2theta) for a detector normal to the beam), a box the model convolves exactly

    Returns
    -------
    dict
        "u" [n, 2] radial unit (slow, fast); "tth", "eta" (deg); "r" [n, K] and "v" [n, K] the sensor slices'
        radial offsets and variances (px^2), K = 1 without a sensor; "var_known" [n] the bandwidth's, the voxel's and
        the convergence's radial variance (px^2); "box" [n] the thickness's radial width (px)
    """
    import anri.io.imaged11 as io
    from anri.calib.fit import base_pars
    from anri.calib.rings import pixel_tth_eta
    from anri.fwd.render import SENSOR_SLICES, _sensor_kernel
    from anri.maths.utils import float64

    sc, fc = np.asarray(rows, np.float64), np.asarray(cols, np.float64)
    if spatial is not None:
        dx, dy = spatial
        sc, fc = sc + dy[rows, cols], fc + dx[rows, cols]
    p = base_pars(pars, 0.0)
    det = io.detector_from_pars(pars)
    with float64():
        pj = {k: jnp.float64(x) for k, x in p.items()}

        def tth_eta(s: np.ndarray, f: np.ndarray) -> tuple:
            return tuple(np.asarray(a) for a in pixel_tth_eta(jnp.asarray(s), jnp.asarray(f), pj))

        tth, eta = tth_eta(sc, fc)
        g = np.stack([(tth_eta(sc + 0.5, fc)[0] - tth_eta(sc - 0.5, fc)[0]),
                      (tth_eta(sc, fc + 0.5)[0] - tth_eta(sc, fc - 0.5)[0])], 1)  # fmt: skip  d 2theta / d pixel, deg
        dtth = np.linalg.norm(g, axis=1)  # deg per pixel along u
        u = g / dtth[:, None]
        th = np.radians(tth / 2)
        bw = np.degrees(2 * bandwidth / (2 * np.sqrt(2 * np.log(2))) * np.tan(th)) / dtth  # px
        A = np.stack([np.asarray(det["s_step_lab"]), np.asarray(det["f_step_lab"])], 1)  # pixels to lab
        e_h = np.linalg.solve(A.T @ A, A.T @ np.array([0.0, 1.0, 0.0]))  # lab y (horizontal, across the beam) in px
        horiz = np.abs(u @ (e_h / np.linalg.norm(e_h)))
        var_known = bw**2 + (voxel / float(pars["y_size"]) * horiz) ** 2 / 12  # the voxel's width as a box
        # a source moved along the beam (lab x) by 1 um: where its ray to this spot meets the detector, in pixels
        o_lab, n = np.asarray(det["det_origin_lab"]), np.cross(A[:, 0], A[:, 1])
        p_lab = o_lab[None] + sc[:, None] * A[:, 0] + fc[:, None] * A[:, 1]
        k = p_lab / np.linalg.norm(p_lab, axis=1, keepdims=True)
        dp = np.array([1.0, 0.0, 0.0])[None] - k * (n[0] / (k @ n))[:, None]
        along = np.linalg.solve(A.T @ A, (dp @ A).T).T  # [n, 2] pixels per um along the beam
        box = thickness * np.abs(np.sum(along * u, 1))
        # convergence: the beam tilted by a small rotation d (about lab z: horizontal; about lab y: vertical), the
        # crystal turned by d_omega about z so that Bragg holds again (|k_in + g| = |k_in|), the ray's change on the
        # detector
        k_in, z = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        g = k - k_in[None]  # [n, 3], units of |k|
        zg = np.cross(z[None], g)
        rad_conv = []
        for axis in (z, np.array([0.0, 1.0, 0.0])):
            dk_in = np.cross(axis, k_in)
            d_om = -(g @ dk_in) / (zg @ k_in)
            dk = dk_in[None] + d_om[:, None] * zg
            dP = np.linalg.norm(p_lab, axis=1, keepdims=True) * (dk - k * ((dk @ n) / (k @ n))[:, None])
            rad_conv.append(np.sum(np.linalg.solve(A.T @ A, (dP @ A).T).T * u, 1))  # radial px per rad
        var_known = var_known + (convergence[0] * rad_conv[0]) ** 2 + (convergence[1] * rad_conv[1]) ** 2
        if sensor is None:
            r, v = np.zeros((len(sc), 1)), np.zeros((len(sc), 1))
        else:
            geom: dict = {k: jnp.asarray(det[k], jnp.float64) for k in ("s_step_lab", "f_step_lab", "det_origin_lab")}
            geom["sensor"] = (jnp.float64(1.0 / sensor[0]), jnp.float64(sensor[1]))
            mus = jnp.stack([jnp.asarray(sc), jnp.asarray(fc), jnp.zeros(len(sc))], 1)
            r, v = (np.asarray(a).reshape(-1, SENSOR_SLICES) for a in jax.vmap(lambda m: _sensor_kernel(m, geom)[:2])(mus))
    return {"u": u, "tth": tth, "eta": eta, "r": r, "v": v, "var_known": var_known, "box": box}


def _sub_offsets(window: int) -> tuple:
    """(slow, fast) offsets of every pixel's sub-samples from the window's centre: [w, w, SUB, SUB] each."""
    h = window // 2
    o = (np.arange(window) - h)[:, None] + (np.arange(SUB) + 0.5) / SUB - 0.5  # [w, SUB]
    return (np.broadcast_to(o[:, None, :, None], (window,) * 2 + (SUB,) * 2),
            np.broadcast_to(o[None, :, None, :], (window,) * 2 + (SUB,) * 2))  # fmt: skip


def _radial_tangential(u: np.ndarray, window: int) -> tuple:
    """Every pixel sub-sample's radial and tangential offset from its window's centre: [n, w, w, SUB, SUB] each."""
    ps, pf = _sub_offsets(window)
    return (u[:, 0, None, None, None, None] * ps + u[:, 1, None, None, None, None] * pf,
            -u[:, 1, None, None, None, None] * ps + u[:, 0, None, None, None, None] * pf)  # fmt: skip


def model_data(counts: np.ndarray, good: np.ndarray, terms: dict) -> dict:
    """Return the data :func:`model` takes (inside float64): windows' counts and mask, sub-sample offsets, known terms."""
    rho, tau = _radial_tangential(terms["u"], counts.shape[1])
    return {"rho": jnp.asarray(rho), "tau": jnp.asarray(tau), "counts": jnp.asarray(counts, jnp.float64),
            "good": jnp.asarray(good), "r": jnp.asarray(terms["r"], jnp.float64),
            "v": jnp.asarray(terms["v"], jnp.float64), "var_known": jnp.asarray(terms["var_known"], jnp.float64),
            "box": jnp.asarray(terms.get("box", np.zeros(len(counts))), jnp.float64)}  # fmt: skip


def _unpack(theta: jax.Array, n: int, two: bool, free_radial: bool) -> tuple:
    s1 = jnp.exp(theta[0])
    s2 = jnp.exp(theta[1]) if two else s1
    f = jax.nn.sigmoid(theta[2]) if two else 0.0
    b = theta[3:].reshape(n, 6)
    sr2 = jnp.exp(2 * b[:, 5]) if free_radial else jnp.zeros(n)
    return s1, s2, f, jnp.exp(b[:, 0]), b[:, 1], b[:, 2], jnp.exp(b[:, 3]), jnp.exp(b[:, 4]), sr2


def model(theta: jax.Array, d: dict, two: bool, free_radial: bool) -> jax.Array:
    """Return the expected counts in every window pixel [n, w, w] (:func:`fit_response`'s parameters theta, data d)."""
    n = d["rho"].shape[0]
    s1, s2, f, I, r0, t0, st, bg, sr2 = _unpack(theta, n, two, free_radial)
    rad_var = d["var_known"] + sr2  # [n]
    out = 0.0
    for w_p, s_p in ((1.0 - f, s1), (f, s2)):
        b = st**2 + s_p**2  # tangential variance [n]
        tan = jnp.exp(-0.5 * (d["tau"] - t0[:, None, None, None, None]) ** 2 / b[:, None, None, None, None]) / jnp.sqrt(
            2 * jnp.pi * b)[:, None, None, None, None]  # fmt: skip
        rad = 0.0
        w = jnp.maximum(d["box"], 1e-3)[:, None, None, None, None]  # the thickness's box, convolved exactly
        has_box = (d["box"] > 1e-3)[:, None, None, None, None]
        for k in range(d["r"].shape[1]):  # sensor slices
            a = (rad_var + d["v"][:, k] + s_p**2)[:, None, None, None, None]
            x = d["rho"] - (r0 + d["r"][:, k])[:, None, None, None, None]
            gauss = jnp.exp(-0.5 * x**2 / a) / jnp.sqrt(2 * jnp.pi * a)
            boxed = (ndtr((x + w / 2) / jnp.sqrt(a)) - ndtr((x - w / 2) / jnp.sqrt(a))) / w
            rad = rad + jnp.where(has_box, boxed, gauss) / d["r"].shape[1]
        out = out + w_p * jnp.mean(rad * tan, (3, 4))
    return I[:, None, None] * out + bg[:, None, None]


def _nll(theta: jax.Array, d: dict, use: jax.Array, two: bool, free_radial: bool) -> jax.Array:
    lam = model(theta, d, two, free_radial)
    ll = d["counts"] * jnp.log(lam) - lam
    return -jnp.sum(jnp.where(d["good"] & use[:, None, None], ll, 0.0))


def fit_response(counts: np.ndarray, good: np.ndarray, terms: dict, free_radial: bool = True,
                 subsets: dict | None = None, log: Callable = print) -> dict:  # fmt: skip
    """Fit the detector's point spread to spot windows: one Gaussian, then a core and a wing.

    Parameters
    ----------
    counts, good
        [n, w, w] each spot's window of raw counts, and which pixels are measured (not masked)
    terms
        :func:`spot_terms` of the window centres
    free_radial
        Each spot its own extra radial sd (a deformed sample's strain spread); False for an undeformed crystal
    subsets
        {name: [n] bool}: also fit these subsets of the spots alone (e.g. low and high 2theta), same shapes
    log
        Progress messages

    Returns
    -------
    dict
        For "all" and each subset: "one" (sd), "core", "wing", "fraction", "nll_one", "nll_two" and "n";
        "theta_one", "theta_two" (all spots' fits) and "lam_one", "lam_two" [n, w, w] their models
    """
    from anri.maths.utils import float64

    n, w, _ = counts.shape
    rho, tau = _radial_tangential(terms["u"], w)
    # start: each window's own moments
    rho_c, tau_c = rho.mean((3, 4)), tau.mean((3, 4))
    bg0 = np.maximum(np.nanmedian(np.where(good, counts, np.nan).reshape(n, -1), 1), 0.05)
    sig = np.maximum(counts - bg0[:, None, None], 0) * good
    tot = np.maximum(sig.sum((1, 2)), 1.0)
    r_c, t_c = (sig * rho_c).sum((1, 2)) / tot, (sig * tau_c).sum((1, 2)) / tot
    st = np.sqrt(np.maximum((sig * (tau_c - t_c[:, None, None]) ** 2).sum((1, 2)) / tot - 1 / 12, 0.05))
    per = np.stack([np.log(tot), r_c, t_c, np.log(st), np.log(bg0), np.full(n, np.log(0.2))], 1)

    fits = {}
    with float64():
        d = model_data(counts, good, terms)
        f1 = jax.jit(jax.value_and_grad(lambda t, m: _nll(t, d, m, False, free_radial)))
        f2 = jax.jit(jax.value_and_grad(lambda t, m: _nll(t, d, m, True, free_radial)))

        def solve(f: Callable, th0: np.ndarray, m: np.ndarray) -> tuple:
            mj = jnp.asarray(m)
            res = minimize(lambda t: tuple(np.asarray(a, np.float64) for a in f(jnp.asarray(t), mj)), th0, jac=True,
                           method="L-BFGS-B", options={"maxiter": 20000, "maxfun": 40000})  # fmt: skip
            return res.x, float(res.fun), res

        for name, m in {"all": np.ones(n, bool), **(subsets or {})}.items():
            th1, nll1, r1 = solve(f1, np.concatenate([[np.log(0.5), 0.0, 0.0], per.ravel()]), m)
            th2, nll2, r2 = solve(f2, np.concatenate([[th1[0] - 0.4, np.log(1.3), -2.0], th1[3:]]), m)
            s1, s2, fr = np.exp(th2[0]), np.exp(th2[1]), 1 / (1 + np.exp(-th2[2]))
            if s1 > s2:  # name the narrower component the core
                s1, s2, fr = s2, s1, 1 - fr
            fits[name] = {"one": float(np.exp(th1[0])), "core": float(s1), "wing": float(s2), "fraction": float(fr),
                          "nll_one": nll1, "nll_two": nll2, "n": int(m.sum())}  # fmt: skip
            log(f"{name} ({int(m.sum())} spots): one Gaussian sd {np.exp(th1[0]):.3f} px; core {s1:.3f} px, wing "
                f"{s2:.3f} px holding {fr:.3f}; log-likelihood gain {nll1 - nll2:.1f} for 2 parameters "
                f"({r1.nit} + {r2.nit} iterations)")  # fmt: skip
            if name == "all":
                fits["theta_one"], fits["theta_two"] = th1, th2
                fits["lam_one"] = np.asarray(model(jnp.asarray(th1), d, False, free_radial))
                fits["lam_two"] = np.asarray(model(jnp.asarray(th2), d, True, free_radial))
                fits["radial_offset"] = rho_c - th2[3:].reshape(n, 6)[:, 1, None, None]
                fits["radial_sd"] = np.sqrt(np.asarray(_unpack(jnp.asarray(th2), n, True, free_radial)[8]))
    return fits
