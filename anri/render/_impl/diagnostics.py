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


def probe_rows(p: dict, rows: list, window: tuple = (3, 7, 7), max_frames: int | None = 31,
               log: Callable = print) -> dict:  # fmt: skip
    """Render a few rows without writing them: the counts before the cut, and how well the windows hold the peaks.

    Parameters
    ----------
    p
        From :func:`anri.render.load_prepare`
    rows
        Row indices to render (e.g. the middle row and one near the edge)
    window, max_frames
        See :func:`anri.render.stage_render`

    Returns
    -------
    dict
        "rows", "values" (every rendered pixel's intensity before rounding and the cut, all rows), "captured" (the share
        of each peak within the scan's omega range that its window holds, for peaks centred in it), "n_outside" (peaks
        centred beyond its ends),
        "window_frames" (frames in each peak's window), "n_peaks" per row, and "density" (the prepare file's)
    """
    import time

    from anri.fwd import make_row, render_phases

    values, captured, frames, n_peaks, n_out = [], [], [], [], 0
    for i in rows:
        t0 = time.perf_counter()
        row = make_row(p["omega"][i], p["dty"][i])
        _, _, v, st = render_phases(p["phases"], p["geom"], row, p["det_shape"], window=tuple(window),
                                    max_frames=max_frames)  # fmt: skip
        inr = np.asarray(st["in_range"])
        inside = inr >= 0.5
        values.append(np.asarray(v))
        captured.append(np.asarray(st["captured"])[inside] / inr[inside])
        n_out += int(np.sum(~inside))
        if "window_frames" in st:
            frames.append(np.asarray(st["window_frames"])[inside])
        n_peaks.append(int(st["n_peaks"]))
        log(f"row {i}: {st['n_peaks']} peaks, {len(v)} pixels ({time.perf_counter() - t0:.1f} s)")
    return {"rows": list(rows), "values": np.concatenate(values), "captured": np.concatenate(captured),
            "n_outside": n_out, "window_frames": np.concatenate(frames) if frames else None, "n_peaks": n_peaks,
            "density": float(p["prov"]["options"]["density"])}  # fmt: skip


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
