"""Shared pieces of match_spots.py and joint.py: an ImageD11 dataset with its 2D peaks, and spot geometry.

A spot is one (entry, hkl, branch). Its prediction (sc, fc, omega) is made with the voxel centred in the beam at its
own omega; rotations are small sample-frame rotation vectors w, UBI -> UBI (I + [w]x)^T.
"""

from __future__ import annotations

import argparse
import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from ImageD11.sinograms.dataset import load as load_ds

import anri.index as ix
import anri.io
from anri.fwd._impl.base import hkl_to_k_omega
from anri.fwd._impl.render import _beam_offsets, _scattering_origin, _wrap_omega
from anri.geom import raytrace_to_det, sample_to_lab


def add_args(p: argparse.ArgumentParser) -> None:
    """The dataset and phase arguments shared by match_spots.py and joint.py."""
    p.add_argument("analysisroot")
    p.add_argument("sample")
    p.add_argument("dataset")
    p.add_argument("--parfile", help="pars.json (default: the DataSet's)")
    p.add_argument("--phase", help="phase name in pars.json (default: the only one)")
    p.add_argument("--cif", help="CIF of the phase, for structure factors (default: |F|^2 = 1)")
    p.add_argument("--rings", type=int, default=8, help="rings used")
    p.add_argument("--y0", type=float, help="dty where the rotation axis is in the beam (default: the DataSet's)")
    p.add_argument("--det-shape", type=int, nargs=2, default=(2048, 2048), help="detector (slow, fast) pixels")
    p.add_argument("--beam", type=float, help="beam FWHM (default: the dty step)")
    p.add_argument("--voxel", type=float, help="voxel size (default: the dty step)")


def load_args(args: argparse.Namespace) -> dict:
    return load(args.analysisroot, args.sample, args.dataset, beam=args.beam, voxel=args.voxel, n_rings=args.rings,
                parfile=args.parfile, phase=args.phase, cif=args.cif, y0=args.y0, det_shape=tuple(args.det_shape))  # fmt: skip


def load(analysisroot: str, sample: str, dataset: str, beam: float | None = None, voxel: float | None = None,
         n_rings: int = 8, parfile: str | None = None, phase: str | None = None, cif: str | None = None,
         y0: float | None = None, det_shape: tuple = (2048, 2048)) -> dict:  # fmt: skip
    """Rows, geometry, the phase's reflections and ImageD11's 2D peaks (sorted by (row, file frame), each cell's start
    in "cstart")."""
    dsname = f"{sample}_{dataset}"
    dsfile = os.path.join(analysisroot, sample, dsname, f"{dsname}_dataset.h5")
    ds = load_ds(dsfile)
    dsd = anri.io.read_dataset(dsfile)
    geo, phase, cell = anri.io.read_pars_json(parfile or dsd["parfile"], phase)
    lpars = np.array([cell[k] for k in ("cell__a", "cell__b", "cell__c", "cell_alpha", "cell_beta", "cell_gamma")])
    sg = cell["cell_lattice_[P,A,B,C,I,F,R]"]
    if not isinstance(sg, (float, int)):
        raise SystemExit(f"{phase}: cell_lattice_[P,A,B,C,I,F,R] = {sg} is a centring letter; a space-group number is needed")
    sg = int(sg)
    structure = None
    if cif:
        import Dans_Diffraction

        structure = Dans_Diffraction.Crystal(cif)
    wl = geo["wavelength"]
    rings = ix.ring_table(lpars, sg, wl, n_rings, structure)
    y0 = dsd["y0"] if y0 is None else y0
    if y0 is None:
        raise SystemExit(f"{dsfile} has no y0: give it with --y0")
    n_rows, n_frames = ds.omega.shape
    row_dty = ds.dty.mean(axis=1)
    ystep = float(np.median(np.abs(np.diff(np.sort(row_dty)))))
    beam = beam or ystep
    voxel = voxel or ystep
    geom = anri.io.geom_from_pars(geo, y0, wl * 2e-4 / 2.355, 5e-5, 5e-5, sig_beam=beam / 2.355,
                                  voxel_size=voxel, sig_psf=0.5)  # fmt: skip
    geom = jax.tree.map(
        lambda x: jnp.asarray(x, jnp.float32) if np.asarray(x).dtype.kind == "f" else jnp.asarray(x), geom
    )
    om_all = np.sort(ds.omega, axis=1)
    om_sorted = np.median(om_all, axis=0).astype(np.float32)  # one omega grid for all rows
    ostep = float(np.median(np.diff(om_sorted)))
    om_dev = float(np.abs(om_all - om_sorted).max())
    if om_dev > 0.25 * ostep:
        raise SystemExit(f"rows' omegas differ from their median by up to {om_dev:.4g} deg (> a quarter frame)")
    edges = np.concatenate([[om_sorted[0] - ostep / 2], 0.5 * (om_sorted[1:] + om_sorted[:-1]),
                            [om_sorted[-1] + ostep / 2]]).astype(np.float32)  # fmt: skip
    rsort = np.argsort(row_dty).astype(np.int32)
    with h5py.File(ds.pksfile, "r") as h:
        s1, s_i, sr_i, sc_i, frm = h["pks2d"]["pk_props"][:]
    row_of, frm = frm // n_frames, frm % n_frames  # the id is row * n_frames + frame
    cellid = row_of.astype(np.int64) * n_frames + frm
    o = np.argsort(cellid, kind="stable")
    blobs = np.stack([sr_i / s_i, sc_i / s_i, ds.omega[row_of, frm]]).T[o].astype(np.float32)
    return {
        "geom": geom, "lpars": lpars, "sg": sg, "phase": phase, "hkls": np.asarray(rings["hkls"], np.float32),
        "F2": np.asarray(rings["F2"], np.float32), "det_shape": tuple(det_shape), "om_dev": om_dev,
        "n_rows": n_rows, "n_frames": n_frames, "om_sorted": om_sorted, "edges": edges, "ostep": ostep,
        "order": np.argsort(ds.omega, axis=1, kind="stable").astype(np.int32),  # sorted index -> file frame, per row
        "rsort": rsort, "dty_sorted": row_dty[rsort].astype(np.float32), "ystep": ystep,
        "blobs": blobs, "blob_i": s_i[o].astype(np.float32), "blob_cell": cellid[o],
        "cstart": np.searchsorted(cellid[o], np.arange(n_rows * n_frames + 1)).astype(np.int32),
    }  # fmt: skip


def skew(w: jax.Array) -> jax.Array:
    z = jnp.zeros((), w.dtype)
    return jnp.stack([jnp.stack([z, -w[2], w[1]]), jnp.stack([w[2], z, -w[0]]), jnp.stack([-w[1], w[0], z])])


def centre_dty(pos: jax.Array, om: jax.Array, geom: dict) -> jax.Array:
    """Return the dty that puts the voxel on the beam's centre line at this omega (across is linear in dty)."""
    y0 = geom["y0"]
    a0 = _beam_offsets(sample_to_lab(pos, om, geom["wedge"], geom["chi"], y0, y0), geom)[0]
    a1 = _beam_offsets(sample_to_lab(pos, om, geom["wedge"], geom["chi"], y0 + 1.0, y0), geom)[0]
    return y0 - a0 / (a1 - a0)


def spot(ubi: jax.Array, pos: jax.Array, hkl: jax.Array, etasign: jax.Array, d: dict) -> tuple:
    """Return a spot's (sc, fc, omega), its derivative by a rotation vector [3, 3], whether it is in the scan, |sin eta|."""
    geom = d["geom"]

    def centroid(w: jax.Array) -> tuple:
        u = ubi @ (jnp.eye(3, dtype=ubi.dtype) + skew(w)).T
        _, k_out, om, valid = hkl_to_k_omega(u, hkl, etasign, geom["wavelength"], geom["k_in_lab"], 0.0, 0.0,
                                             geom["wedge"], geom["chi"])  # fmt: skip
        origin = _scattering_origin(pos, om, centre_dty(pos, om, geom), geom)
        sc, fc = raytrace_to_det(k_out, origin, geom["s_step_lab"], geom["f_step_lab"], geom["det_origin_lab"])
        return jnp.stack([sc, fc, om]), (valid, k_out)

    w0 = jnp.zeros(3, ubi.dtype)
    mu, (valid, k_out) = centroid(w0)
    J = jax.jacfwd(lambda w: centroid(w)[0])(w0)
    edges = d["edges"]
    mu = mu.at[2].set(_wrap_omega(mu[2], jnp.float32(0.5 * (edges[0] + edges[-1]))))
    sin_eta = jnp.abs(k_out[1]) / jnp.hypot(k_out[1], k_out[2])
    dty_c = centre_dty(pos, mu[2], geom)
    dty = d["dty_sorted"]
    ok = valid & (mu[2] > edges[0]) & (mu[2] < edges[-1]) & (dty_c > dty[0] - 1.0) & (dty_c < dty[-1] + 1.0)
    ns, nf = d["det_shape"]
    ok = ok & (mu[0] > -5) & (mu[0] < ns + 4) & (mu[1] > -5) & (mu[1] < nf + 4)
    return mu, J, ok, sin_eta
