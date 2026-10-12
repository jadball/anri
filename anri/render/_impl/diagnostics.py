"""The numbers behind each choice of the renderer's options, for notebooks, workflow widgets or logs.

Pure functions of a prepare stage file (read with :func:`anri.render.load_prepare`) or of a rendered dataset; they draw
nothing. The plots of :mod:`anri.render.plots` draw them.
"""

from __future__ import annotations

from collections.abc import Callable

import h5py
import numpy as np

from .stages import _LATTICE_KEYS


def footprint(p: dict) -> dict:
    """Return where the sample is against the scanned dty range, to check the rows cover it (``dty_step``, ``margin``).

    Parameters
    ----------
    p
        From :func:`anri.render.load_prepare`

    Returns
    -------
    dict
        "names", "pos" (per phase [N, 2] sample positions), "radius" (of the sample), "covered" (the radius the rows
        reach at every omega, from y0 to the nearer end of the rows), "dty" (row centres), "y0", "beam_fwhm"
    """
    r = p["prov"]["resolved"]
    dty = p["dty"][:, 0]
    step = r["dty_step"]
    y0 = p["prov"]["options"]["y0"]
    covered = min(y0 - (dty.min() - step / 2), dty.max() + step / 2 - y0)
    return {"names": p["names"], "pos": [np.asarray(e["pos"])[:, :2] for e, _, _ in p["phases"]],
            "radius": r["sample_radius"], "covered": covered, "dty": dty, "y0": y0, "beam_fwhm": r["beam_fwhm"]}  # fmt: skip


def ring_coverage(p: dict, n_tth: int = 300, n_eta: int = 720) -> dict:
    """Return how much of each 2theta's ring lands on the detector, and where each phase's reflections are (``tth_max``).

    Spots on a ring that leaves the detector are lost wherever they fall off it, so the reflections are usually kept
    to rings wholly on it.

    Parameters
    ----------
    p
        From :func:`anri.render.load_prepare`
    n_tth, n_eta
        Rings and points per ring to trace (from the sample at the origin)

    Returns
    -------
    dict
        "tth" [n_tth] and "on_detector" [n_tth] (fraction of each ring), "names", "reflections" (per phase, each
        reflection's 2theta), "tth_max" (the prepare file's)
    """
    import jax
    import jax.numpy as jnp

    from anri.crystal import B_matrix
    from anri.diffract import tth_eta_to_k_out
    from anri.geom import raytrace_to_det

    g, wl = p["geom"], float(p["pars"]["wavelength"])
    ns, nf = p["det_shape"]
    refl = []
    for name, (_, hkls, _) in zip(p["names"], p["phases"]):
        B = B_matrix(np.array([p["cells"][name][k] for k in _LATTICE_KEYS]))
        ds = np.linalg.norm(np.asarray(hkls) @ B.T, axis=1)
        refl.append(np.degrees(2 * np.arcsin(np.clip(ds * wl / 2, 0, 1))))
    top = max(max(float(r.max()) for r in refl), float(p["prov"]["resolved"]["tth_max"])) * 1.15
    tth = np.linspace(0.5, top, n_tth)
    eta = np.linspace(-180, 180, n_eta, endpoint=False)
    tt, ee = np.meshgrid(tth, eta, indexing="ij")
    k_in = jnp.asarray(g["k_in_lab"])

    def one(t: jax.Array, e: jax.Array) -> tuple[jax.Array, jax.Array]:
        k = tth_eta_to_k_out(t, e, wl, k_in)
        return raytrace_to_det(k, jnp.zeros(3), g["s_step_lab"], g["f_step_lab"], g["det_origin_lab"])

    sc, fc = (np.asarray(x).reshape(tt.shape) for x in jax.vmap(one)(jnp.asarray(tt.ravel()), jnp.asarray(ee.ravel())))
    on = (sc >= 0) & (sc <= ns - 1) & (fc >= 0) & (fc <= nf - 1)
    return {"tth": tth, "on_detector": on.mean(1), "names": p["names"], "reflections": refl,
            "tth_max": float(p["prov"]["resolved"]["tth_max"])}  # fmt: skip


def probe_rows(p: dict, rows: list, cut: float = 1.0, window: tuple = (3, 7, 7), max_frames: int | None = 31,
               densities: np.ndarray | None = None, batch: int = 8192, log: Callable = print) -> dict:  # fmt: skip
    """Render a few rows without writing them: how each peak survives the cut at any density, and its window.

    Each peak is rendered on its own (as :func:`anri.fwd.render_peaks`, on the peaks :func:`anri.fwd.row_peaks`
    selects, as the renderer does) and looked up in the row's summed pixels, so overlapping peaks count. A peak is
    judged in the row that sees its voxel best: where the voxel crosses the beam within half a dty step of the row's
    dty, at the peak's omega (peaks of voxels at the beam's edge are faint in this row and bright in their own), and
    centred in the scan's omega range (a peak centred beyond its ends has its Friedel mate in the scan).
    Intensities scale with the density, so one render gives every density: a peak vanishes below the density at which
    its brightest pixel rounds to the cut.

    Parameters
    ----------
    p
        From :func:`anri.render.load_prepare`
    rows
        Row indices to render (e.g. the middle row and one near the sample's edge)
    cut, window, max_frames
        See :func:`anri.render.stage_render`
    densities
        The densities to evaluate (default: 1/100 to 1000 times the prepare file's, log-spaced)
    batch
        Peaks rendered at once

    Returns
    -------
    dict
        "densities" [F], "lost_below" [n] (each peak's density below which no pixel of it is above the cut),
        "kept_share" [n, F] (share of each peak's intensity in pixels above the cut, at each density), "rows",
        "density" (the prepare file's), "cut"; and for the windows "captured" (the share of each peak within the
        scan's omega range that its window holds), "n_outside" (peaks centred beyond its ends), "window_frames"
    """
    import time

    import jax
    import jax.numpy as jnp

    from anri.fwd import make_row, render_phases, row_peaks
    from anri.fwd._impl.render import _as_jax, _render_peaks
    from anri.geom import find_dty_for_beam_xy

    d0 = float(p["prov"]["options"]["density"])
    densities = d0 * np.logspace(-2, 3, 101) if densities is None else np.asarray(densities, float)
    factors = densities / d0
    step, y0 = p["prov"]["resolved"]["dty_step"], p["prov"]["options"]["y0"]
    npix = p["det_shape"][0] * p["det_shape"][1]
    _, ws, wf = window
    render = jax.jit(_render_peaks, static_argnums=(8, 9))  # as render_peaks, also giving each peak's in_range
    lost, kept, captured, frames, n_out = [], [], [], [], 0
    for i in rows:
        t0 = time.perf_counter()
        row = make_row(p["omega"][i], p["dty"][i])
        frame, pixel, value, st = render_phases(p["phases"], p["geom"], row, p["det_shape"], window=tuple(window),
                                                max_frames=max_frames)  # fmt: skip
        key_m = np.asarray(frame).astype(np.int64) * npix + np.asarray(pixel)  # sorted by (frame, pixel)
        value_m = np.asarray(value)
        inr = np.asarray(st["in_range"])
        inside = inr >= 0.5
        captured.append(np.asarray(st["captured"])[inside] / inr[inside])
        n_out += int(np.sum(~inside))
        if "window_frames" in st:
            frames.append(np.asarray(st["window_frames"])[inside])
        n_row = 0
        for entries, hkls, F2 in p["phases"]:
            ent, hk, f2, geom, row_j = _as_jax(entries, hkls, F2, p["geom"], row)
            e, h, b, cls, classes = row_peaks(ent, hk, geom, row_j, p["det_shape"], tuple(window), max_frames)
            pos = np.asarray(entries["pos"])
            for k, wo in enumerate(classes):
                sel = np.flatnonzero(cls == k)
                for s0 in range(0, sel.size, batch):
                    idx = np.pad(sel[s0 : s0 + batch], (0, batch - len(sel[s0 : s0 + batch])))
                    n = min(batch, sel.size - s0)
                    fr, px, val, _, in_scan, _ = render(jnp.asarray(e[idx]), jnp.asarray(h[idx]), jnp.asarray(b[idx]),
                                                        ent, hk, f2, geom, row_j, (wo, ws, wf), p["det_shape"], None)  # fmt: skip
                    fr, px, v = np.asarray(fr)[:n], np.asarray(px)[:n], np.asarray(val)[:n].astype(np.float64)
                    v[fr < 0] = 0.0
                    total = v.sum(1)
                    key = fr.astype(np.int64) * npix + px
                    M = v  # the row's summed pixel where the row has it, else the peak's own value
                    if len(key_m):
                        j = np.minimum(np.searchsorted(key_m, key), len(key_m) - 1)
                        M = np.where(key_m[j] == key, value_m[j], v)
                    # the peak's omega: its brightest frame; its voxel's dty at that omega
                    best = np.pad(fr[np.arange(n), np.argmax(v, 1)], (0, batch - n))  # fixed shapes: one compile
                    om = np.asarray(p["omega"][i])[np.maximum(best, 0)]
                    dty_v = np.asarray(find_dty_for_beam_xy(jnp.asarray(pos[e[idx]]), geom["k_in_lab"], jnp.asarray(om),
                                                            geom["wedge"], geom["chi"], y0))[:n]  # fmt: skip
                    own = (np.abs(dty_v - float(np.mean(p["dty"][i]))) <= step / 2) & (total > 0)
                    own &= np.asarray(in_scan)[:n] >= 0.5  # peaks centred beyond the scan's ends: their mates count
                    if not own.any():
                        continue
                    v, M, total = v[own], M[own], total[own]
                    lost.append(d0 * (cut + 0.5) / np.maximum(M.max(1), 1e-30))
                    kept.append(np.stack([(v * (M * f >= cut + 0.5)).sum(1) for f in factors], 1) / total[:, None])
                    n_row += int(own.sum())
        log(f"row {i}: {st['n_peaks']} peaks, {n_row} of them judged here ({time.perf_counter() - t0:.1f} s)")
    return {"densities": densities, "lost_below": np.concatenate(lost), "kept_share": np.concatenate(kept),
            "rows": list(rows), "density": d0, "cut": cut, "captured": np.concatenate(captured), "n_outside": n_out,
            "window_frames": np.concatenate(frames) if frames else None}  # fmt: skip


def row_image(sparse_file: str, row: int, omega_motor: str = "rot_center") -> dict:
    """Return one row of a rendered (or measured) ImageD11 sparse file summed over its frames, and per frame.

    Parameters
    ----------
    sparse_file
        The sparse pixels file
    row
        Row index (the scan group "<row + 1>.1")
    omega_motor
        The omega motor's name in the scan's measurement group

    Returns
    -------
    dict
        "image" [n_slow, n_fast] summed over the frames, "per_frame" [n_frames] total counts, "omega" [n_frames]
    """
    with h5py.File(sparse_file, "r") as h:
        g = h[f"{row + 1}.1"]
        shape = (int(g.attrs["shape0"]), int(g.attrs["shape1"]))
        r, c, v = g["row"][()], g["col"][()], g["intensity"][()].astype(np.float64)
        nnz = g["nnz"][()]
        omega = g[f"measurement/{omega_motor}"][()]
    image = np.zeros(shape)
    np.add.at(image, (r, c), v)
    frame = np.repeat(np.arange(len(nnz)), nnz)
    return {"image": image, "per_frame": np.bincount(frame, weights=v, minlength=len(nnz)), "omega": omega}
