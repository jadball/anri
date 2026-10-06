"""Refine the orientations of the indexer's populations against the pixels: voxel-centric, in rotation space.

    python refine_rot.py <dataset.h5> <index_entries.npz> [--truth T] [--check] [--rings N] [--out refined.npz]

Rotation only (no strain, densities and populations fixed: the indexer's ownership is kept). Each pass:

1. Every population's predicted spots (both omega solutions of every reflection) are listed in every dty row where
   its voxel is lit within the search window, sorted by (row, ring, omega bin, eta bin).
2. E-step (one jitted function per block of pixels): each pixel takes the predicted spots in its own and the
   neighbouring bins, within its omega window, as candidates; the (pixel, candidate) pairs are laid end to end and
   processed in fixed-size chunks, twice (totals per pixel, then responsibilities), so none is dropped. For each, the pixel's g-vector is computed from that voxel's own position at
   the frame's omega (parallax exact), rotated back to the sample frame, and compared with the population's
   predicted direction U B h: the misfit, in the plane normal to g, is split into the direction omega moves g (widened
   by the frame step) and the one normal to it. The pixel's intensity is shared among the candidates in proportion to
   density x beam weight x Lorentz-polarisation x a Gaussian of the misfit, plus a background.
3. M-step: each population's orientation is the weighted best rotation taking its reflections' directions B h onto
   the observed ones (Wahba's problem, by SVD). Populations with too little support keep their orientation.

The kernel narrows over the passes (--sigmas), from about the indexer's error to the instrument. With --truth, each
pass prints the main populations' errors against the phantom, and how much of the truth's sub-grain field they
recover (slope 1: all of it; 0: each voxel at its grain's mean).
"""

from __future__ import annotations

import argparse
import time

import anri.utils

anri.utils.setup()
import h5py
import jax
import jax.numpy as jnp
import numpy as np

import anri.crystal
import anri.index as ix
import anri.io
from anri.fwd import beam_weight

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f} s] {msg}", flush=True)


def rot_z(om_deg):
    """[..., 3, 3] rotations about z (right-handed), omega in degrees (works for numpy and jax arrays)."""
    xp = jnp if isinstance(om_deg, jax.Array) else np
    c, s = xp.cos(xp.radians(om_deg)), xp.sin(xp.radians(om_deg))
    z, o = xp.zeros_like(c), xp.ones_like(c)
    return xp.stack([xp.stack([c, -s, z], -1), xp.stack([s, c, z], -1), xp.stack([z, z, o], -1)], -2)


# ----------------------------------------------------------------------------------------------------------------------
# data


def load_pixels(ds: dict, geom: dict, rings: dict, n_rings: int, tth_tol: float, etacut: float) -> dict:
    """Every sparse pixel near a ring and with |sin eta| > etacut: lab position, omega, row, ring, eta, intensity."""
    tth_r = np.asarray(rings["tth"])[:n_rings]
    out = {k: [] for k in ("P", "om", "row", "ring", "eta", "I")}
    o, s_step, f_step = (np.asarray(geom[k], float) for k in ("det_origin_lab", "s_step_lab", "f_step_lab"))
    with h5py.File(ds["sparsefile"], "r") as h:
        groups = sorted(h.keys(), key=float)
        for r, name in enumerate(groups):
            g = h[name]
            nnz = g["nnz"][()]
            frame = np.repeat(np.arange(len(nnz)), nnz)
            s, f = g["row"][()].astype(float), g["col"][()].astype(float)
            om = np.asarray(g[f"measurement/{ds['omegamotor']}"][()], float)[frame]
            P = o + s[:, None] * s_step + f[:, None] * f_step
            ang = np.asarray(ix.pixel_angles(jnp.asarray(s, jnp.float32), jnp.asarray(f, jnp.float32),
                                             jnp.asarray(om, jnp.float32), geom))  # fmt: skip
            ring = np.argmin(np.abs(ang[:, :1] - tth_r[None]), 1)
            keep = (np.abs(ang[:, 0] - tth_r[ring]) < tth_tol) & (np.abs(np.sin(np.radians(ang[:, 1]))) > etacut)
            for k, v in (("P", P), ("om", om), ("row", np.full(len(s), r)), ("ring", ring), ("eta", ang[:, 1]),
                         ("I", g["intensity"][()].astype(float))):  # fmt: skip
                out[k].append(v[keep])
    return {k: np.concatenate(v) for k, v in out.items()}


# ----------------------------------------------------------------------------------------------------------------------
# predicted spots, listed per lit row and sorted by bin


def instances(U, pos, B, rings, n_rings, geom, scan, etacut, sig_tot, bins, max_rows=9):
    """Predicted spots of every population, listed in each row where its voxel is lit within the window, sorted by key.

    Returns sorted int64 keys and, per entry, population, hkl index and Lorentz-polarisation, plus a count of entries
    that hit max_rows (their rows were cut).
    """
    nc = len(U)
    eta, om, ok, lp = [], [], [], []
    hkls = jnp.asarray(rings["hkls"])
    for c0 in range(0, nc, 4096):  # fixed chunk: one compile
        u = np.asarray(U[c0 : c0 + 4096], np.float32)
        n = len(u)
        u = np.concatenate([u, np.repeat(u[:1], 4096 - n, 0)])
        e, o_, k = ix.predict(jnp.asarray(u), jnp.asarray(B, jnp.float32), hkls, geom)
        w = ix.lorentz_polarisation(jnp.asarray(u), jnp.asarray(B, jnp.float32), hkls, geom)
        eta.append(np.asarray(e)[:n]), om.append(np.asarray(o_)[:n]), ok.append(np.asarray(k)[:n])
        lp.append(np.asarray(w)[:n])
    eta, om, ok, lp = (np.concatenate(a) for a in (eta, om, ok, lp))
    om = np.mod(om, 360.0)
    ring_j = np.asarray(rings["ring_j"])
    use = ok & (np.abs(np.sin(np.radians(eta))) > etacut) & (ring_j[None] < n_rings)
    use &= (om >= scan["om0"]) & (om < scan["om1"])
    c, j = np.nonzero(use)
    om_c, eta_c = om[c, j], eta[c, j]
    # the voxel's dty at that omega, and the rows within reach: beam, row width, and the omega window x radius
    x, y = pos[c, 0], pos[c, 1]
    dty_c = scan["y0"] - (x * np.sin(np.radians(om_c)) + y * np.cos(np.radians(om_c)))
    k_mid = (dty_c - scan["dty0"]) / scan["ystep"]
    w_om = 3.0 * np.radians(sig_tot) / np.maximum(np.abs(np.sin(np.radians(eta_c))), etacut)  # radians
    reach = (3.0 * scan["sig_beam"] + 0.5 * scan["ystep"] + np.hypot(x, y) * w_om) / scan["ystep"]
    k_lo, k_hi = np.ceil(k_mid - reach).astype(int), np.floor(k_mid + reach).astype(int)
    n_cut = int(np.sum(k_hi - k_lo + 1 > max_rows))
    mid = np.round(k_mid).astype(int)
    k_lo, k_hi = np.maximum(k_lo, mid - max_rows // 2), np.minimum(k_hi, mid + max_rows // 2)
    rows = k_lo[:, None] + np.arange(max_rows)[None]
    ok_r = (rows <= k_hi[:, None]) & (rows >= 0) & (rows < scan["n_rows"])
    e_idx, r_off = np.nonzero(ok_r)
    row = rows[e_idx, r_off]
    bo = np.floor((om_c[e_idx] - scan["om0"]) / bins["om"]).astype(np.int64)
    be = np.floor((eta_c[e_idx] + 180.0) / bins["eta"]).astype(np.int64)
    key = make_key(row, ring_j[j[e_idx]], be, bo, bins)
    order = np.argsort(key, kind="stable")
    return {
        "key": key[order],
        "c": c[e_idx][order].astype(np.int32),
        "h": (j[e_idx][order] // 2).astype(np.int32),
        "lp": lp[c, j][e_idx][order].astype(np.float32),
        "n_cut": n_cut,
    }


def make_key(row, ring, be, bo, bins):
    """Sort key of (row, ring, eta bin, omega bin), omega last so a pixel's omega window is one contiguous range."""
    return ((row * bins["n_ring"] + ring) * bins["n_eta"] + be) * bins["n_om"] + bo


# ----------------------------------------------------------------------------------------------------------------------
# E-step over a block of pixels, as ragged (pixel, candidate) pairs, with the M-step's sums


def pixel_ranges(px, key, prm, scan, bins):
    """Each pixel's candidates: 3 contiguous ranges of the sorted keys (its eta bin and the two either side, each over
    its omega window: 3 sigma of the kernel over |sin eta|, plus a frame). Returns lo, cnt [N, 3] int32."""
    tol = prm["tol_deg"] / jnp.abs(jnp.sin(jnp.radians(px["eta"]))) + scan["ostep"]
    bo_lo = jnp.clip(jnp.floor((px["om"] - tol - scan["om0"]) / bins["om"]), 0, bins["n_om"] - 1).astype(jnp.int32)
    bo_hi = jnp.clip(jnp.floor((px["om"] + tol - scan["om0"]) / bins["om"]), 0, bins["n_om"] - 1).astype(jnp.int32)
    be = jnp.floor((px["eta"] + 180.0) / bins["eta"]).astype(jnp.int32)
    lo, cnt = [], []
    for d in (-1, 0, 1):
        a = jnp.searchsorted(key, make_key(px["row"], px["ring"], be + d, bo_lo, bins), side="left")
        b = jnp.searchsorted(key, make_key(px["row"], px["ring"], be + d, bo_hi, bins), side="right")
        lo.append(a)
        cnt.append(jnp.where(px["I"] > 0, b - a, 0))
    return jnp.stack(lo, 1).astype(jnp.int32), jnp.stack(cnt, 1).astype(jnp.int32)


def pair_terms(t0, P, end, lo, cnt, px, inst, pops, bhat, geom, scan, prm):
    """The pairs t0 .. t0 + P - 1 of a block: pixel, population, hkl, observed direction, misfits and weights."""
    t = t0 + jnp.arange(P, dtype=jnp.int32)
    ok = t < end[-1]
    p = jnp.minimum(jnp.searchsorted(end, t, side="right"), end.shape[0] - 1)
    k = t - (end[p] - jnp.sum(cnt[p], 1))
    c0, c1 = cnt[p, 0], cnt[p, 1]
    i = jnp.where(k < c0, lo[p, 0] + k, jnp.where(k < c0 + c1, lo[p, 1] + k - c0, lo[p, 2] + k - c0 - c1))
    i = jnp.clip(i, 0, inst["key"].shape[0] - 1)
    c, h, lp = inst["c"][i], inst["h"][i], inst["lp"][i]
    om = px["om"][p]
    R = rot_z(om)  # [P, 3, 3]
    p_lab = jnp.einsum("pij,pj->pi", R, pops["pos"][c]) + jnp.stack(
        [jnp.zeros(P), px["dty"][p] - scan["y0"], jnp.zeros(P)], -1
    )
    w_beam = jax.vmap(lambda q, o: beam_weight(q, o, geom))(p_lab, om)
    k_out = px["P"][p] - p_lab
    k_out = k_out / jnp.linalg.norm(k_out, axis=-1, keepdims=True)
    g = jnp.einsum("pji,pj->pi", R, k_out - geom["k_in_lab"][None])  # sample frame: R^T g
    g = g / jnp.linalg.norm(g, axis=-1, keepdims=True)
    gp = jnp.einsum("pij,pj->pi", pops["U"][c], bhat[h])  # predicted direction
    # misfit in the plane normal to the prediction: along the direction omega moves it (z x g), and normal to that
    zxg = jnp.stack([-gp[:, 1], gp[:, 0], jnp.zeros(P)], -1)
    s_om = jnp.linalg.norm(zxg, axis=-1)
    e1 = zxg / jnp.maximum(s_om, 1e-6)[:, None]
    e2 = jnp.cross(gp, e1)
    d1, d2 = jnp.sum(g * e1, -1), jnp.sum(g * e2, -1)
    base = prm["sig_a"] ** 2 + prm["sig_inst"] ** 2
    v1 = base + (prm["sig_frame"] * s_om) ** 2
    norm = pops["rho"][c] * w_beam * lp / (2 * jnp.pi * jnp.sqrt(v1 * base))
    use = ok & (jnp.sum(g * gp, -1) > 0.0)
    a = jnp.where(use, norm * jnp.exp(-0.5 * (d1**2 / v1 + d2**2 / base)), 0.0)
    return p, c, h, g, a, jnp.where(use, norm, 0.0), d1**2 + d2**2


def block_estep(px, inst, pops, bhat, geom, scan, bins, prm, P: int, n_pop: int):
    """E-step of one block of pixels: every (pixel, candidate) pair, in chunks of P pairs, twice.

    First pass: each pixel's total weight and its best candidate's peak density (for the background). Second pass: the
    responsibilities, and their sums per population: K [Np, 3, 3] (sum W g_obs bhat^T), S [Np] (sum W) and Q [Np]
    (sum W misfit^2), W = intensity x responsibility. Also diagnostics: intensity explained, total, pairs.
    """
    lo, cnt = pixel_ranges(px, inst["key"], prm, scan, bins)
    end = jnp.cumsum(jnp.sum(cnt, 1))
    n_chunks = (end[-1] + P - 1) // P
    nb = px["om"].shape[0]
    args = (P, end, lo, cnt, px, inst, pops, bhat, geom, scan, prm)

    def first(j, carry):
        tot, ref = carry
        p, _, _, _, a, norm, _ = pair_terms(j * P, *args)
        return tot.at[p].add(a), ref.at[p].max(norm)

    tot, ref = jax.lax.fori_loop(0, n_chunks, first, (jnp.zeros(nb), jnp.zeros(nb)))
    den = tot + prm["bg"] * ref * jnp.exp(-4.5) + 1e-30

    def second(j, carry):
        K, S, Q = carry
        p, c, h, g, a, _, m2 = pair_terms(j * P, *args)
        W = px["I"][p] * a / den[p]
        K = K + jax.ops.segment_sum(W[:, None, None] * g[:, :, None] * bhat[h][:, None, :], c, n_pop)
        return K, S + jax.ops.segment_sum(W, c, n_pop), Q + jax.ops.segment_sum(W * m2, c, n_pop)

    zero = (jnp.zeros((n_pop, 3, 3)), jnp.zeros(n_pop), jnp.zeros(n_pop))
    K, S, Q = jax.lax.fori_loop(0, n_chunks, second, zero)
    diag = jnp.stack([jnp.sum(px["I"] * tot / den), jnp.sum(px["I"]), end[-1].astype(jnp.float32)])
    return K, S, Q, diag


def wahba(K: np.ndarray) -> np.ndarray:
    """[N, 3, 3] rotations U maximising tr(U^T K), by SVD."""
    u, _, vt = np.linalg.svd(K)
    d = np.sign(np.linalg.det(u @ vt))
    u[:, :, 2] *= d[:, None]
    return u @ vt


# ----------------------------------------------------------------------------------------------------------------------
# truth comparison (phantoms)


def truth_report(U, d, truth, B, ops, first, tag):
    from ImageD11.sinograms.tensor_map import TensorMap
    from scipy.spatial import KDTree

    te = anri.io.entries_from_tensormap(truth)
    tU = np.linalg.inv(te["ubi"]) @ np.linalg.inv(B)
    dist, it = KDTree(te["pos"][:, :2]).query(d["pos"][first, :2])
    m = dist < 0.1
    e = np.asarray(anri.crystal.disorientation(U[first][m], tU[it[m]], ops))
    # sub-grain field: deviation from the grain mean, refined vs truth, in the truth's symmetry setting
    lab = TensorMap.map_order_to_recon_order(truth.labels, 0)
    ri, rj = np.nonzero(TensorMap.map_order_to_recon_order(truth.phase_ids, 0) == 0)
    grain = lab[ri, rj][it[m]]
    Ut, Ur = tU[it[m]], U[first][m]
    # bring each refined orientation to the symmetric equivalent nearest the truth
    M = Ut.transpose(0, 2, 1) @ Ur  # tr(M O) largest: Ur O nearest Ut
    Ur = Ur @ ops[np.argmax(np.einsum("nij,oji->no", M, ops), 1)]
    num = den = rr = 0.0
    for gi in np.unique(grain):
        k = grain == gi
        u, _, vt = np.linalg.svd(Ut[k].mean(0))
        Um = u @ vt
        vt_ = rotvec(Ut[k] @ Um.T)
        vr_ = rotvec(Ur[k] @ Um.T)
        num += np.sum(vt_ * vr_)
        den += np.sum(vt_ * vt_)
        rr += np.sum(vr_ * vr_)
    log(f"{tag}: vs truth median {np.median(e):.3f} deg; within 0.02/0.05/0.1/0.25 deg "
        f"{np.mean(e < 0.02):.1%} / {np.mean(e < 0.05):.1%} / {np.mean(e < 0.1):.1%} / {np.mean(e < 0.25):.1%}; "
        f"sub-grain field: slope {num / den:.2f}, correlation {num / np.sqrt(den * rr):.2f}, "
        f"rms true {np.degrees(np.sqrt(den / m.sum())):.3f} deg")  # fmt: skip


def rotvec(R):
    """Rotation vectors (radians) of [N, 3, 3] rotations."""
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    t = np.arccos(c)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    return v * (t / np.maximum(2 * np.sin(t), 1e-12))[:, None]


# ----------------------------------------------------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("entries")
    p.add_argument("--truth", help="phantom TensorMap, to score each pass")
    p.add_argument("--rings", type=int, default=8)
    p.add_argument("--etacut", type=float, default=0.3)
    p.add_argument("--tth-tol", type=float, default=0.15, help="pixels within this of a ring (deg)")
    p.add_argument("--sigmas", default="0.3,0.3,0.2,0.2,0.1,0.1,0.05,0.05,0.03,0.03,0.02,0.02",
                   help="kernel width added per pass, deg")  # fmt: skip
    p.add_argument("--sig-inst", type=float, default=0.01, help="instrument width per component, deg")
    p.add_argument(
        "--bg", type=float, default=1.0, help="background level (x the 3-sigma density of the best candidate)"
    )
    p.add_argument("--block", type=int, default=1 << 20, help="pixels per block (one jitted call each)")
    p.add_argument("--pairs", type=int, default=1 << 18, help="(pixel, candidate) pairs per chunk within a block")
    p.add_argument("--min-support", type=float, default=50.0, help="keep a population's orientation below this")
    p.add_argument("--check", action="store_true", help="sizes, candidates per pixel, and one pass at the truth")
    p.add_argument("--check-sigma", type=float, default=0.05, help="kernel for --check at the truth (deg)")
    p.add_argument("--out")
    args = p.parse_args()

    ds = anri.io.read_dataset(args.dataset)
    geo, _, cell = anri.io.read_pars_json(ds["parfile"])
    lpars = np.array([cell[k] for k in ("cell__a", "cell__b", "cell__c", "cell_alpha", "cell_beta", "cell_gamma")])
    sg = int(cell["cell_lattice_[P,A,B,C,I,F,R]"])
    d = dict(np.load(args.entries))
    y0 = float(ds["y0"])
    ybin, oedge = ds["ybincens"], ds["obinedges"]
    ystep = float(np.median(np.diff(ybin)))
    geom = anri.io.geom_from_pars(geo, y0, 0.0, 1e-4, 1e-4, sig_beam=ystep / 2.355, voxel_size=ystep)
    geom = {k: jnp.asarray(v, jnp.float32) if np.issubdtype(np.asarray(v).dtype, np.floating) else v
            for k, v in geom.items()}  # fmt: skip
    rings = ix.ring_table(lpars, sg, geo["wavelength"], args.rings)
    B = anri.crystal.B_matrix(lpars)
    ops = anri.crystal.laue_rotations(anri.crystal.symmetry_matrices(sg), B)
    bh = np.asarray(rings["hkls"], float) @ B.T
    bhat = jnp.asarray(bh / np.linalg.norm(bh, axis=1, keepdims=True), jnp.float32)
    ostep = float(np.median(np.diff(oedge)))
    scan = {"y0": y0, "dty0": float(ybin[0]), "ystep": ystep, "n_rows": len(ybin), "om0": float(oedge[0]),
            "om1": float(oedge[-1]), "sig_beam": ystep / 2.355}  # fmt: skip

    px = load_pixels(ds, geom, rings, args.rings, args.tth_tol, args.etacut)
    px["dty"] = np.asarray(ybin, float)[px["row"]]
    n_px = len(px["om"])
    log(f"{n_px} pixels near the first {args.rings} rings with |sin eta| > {args.etacut}, "
        f"{np.sum(px['I']):.4g} counts")  # fmt: skip

    U = np.linalg.inv(d["ubi"].astype(float)) @ np.linalg.inv(B)
    pos = d["pos"].astype(float)
    n_pop = len(U)
    n_pop_pad = -(-n_pop // 1024) * 1024
    order = np.lexsort((-d["density"], d["voxel"]))
    first = order[np.r_[True, d["voxel"][order][1:] != d["voxel"][order][:-1]]]  # each voxel's main population
    truth = None
    if args.truth:
        from ImageD11.sinograms.tensor_map import TensorMap

        truth = TensorMap.from_h5(args.truth)
        truth_report(U, d, truth, B, ops, first, "indexer")

    sigmas = [float(s) for s in args.sigmas.split(",")]
    scan["ostep"] = ostep
    bins0 = {"om": max(0.25, ostep), "n_ring": args.rings}
    bins0["n_om"] = int(np.ceil((scan["om1"] - scan["om0"]) / bins0["om"])) + 1

    def bins_for(sig_tot):  # eta bins at least the kernel's 3-sigma reach in eta, and no finer than 0.5 deg
        b = dict(bins0, eta=max(0.5, 3.0 * sig_tot * (1 + 0.2 / args.etacut)))
        b["n_eta"] = int(np.ceil(360.0 / b["eta"])) + 1
        return b

    log(f"{n_pop} populations; omega bins {bins0['om']:g} deg; kernels {sigmas} deg")

    # pixels on the device, in blocks of fixed size (I = 0 in the padding): pairs per block stay within int32
    C = args.block
    n_ch = -(-n_px // C)
    pad = n_ch * C - n_px

    def padded(a, dt):
        a = np.asarray(a)
        return jnp.asarray(np.concatenate([a, np.repeat(a[:1], pad, 0)]).astype(dt))

    pxd = {k: padded(px[k], np.float32) for k in ("P", "om", "dty", "eta")}
    pxd["row"], pxd["ring"] = padded(px["row"], np.int32), padded(px["ring"], np.int32)
    pxd["I"] = jnp.asarray(np.concatenate([px["I"], np.zeros(pad)]).astype(np.float32))
    step = jax.jit(block_estep, static_argnames=("P", "n_pop"))

    def one_pass(U_now, sig_a):
        sig_tot = np.sqrt(sig_a**2 + args.sig_inst**2)
        bins = bins_for(sig_tot)
        inst = instances(U_now, pos, B, rings, args.rings, geom, scan, args.etacut, sig_tot, bins)
        n_i = len(inst["key"])
        n_ipad = -(-n_i // (1 << 20)) * (1 << 20)  # bucket: few compiles
        big = np.iinfo(np.int32).max
        if inst["key"].max() >= big:
            raise SystemExit("sort keys overflow int32: fewer rows, rings or bins")
        instd = {"key": jnp.asarray(np.concatenate([inst["key"], np.full(n_ipad - n_i, big)]).astype(np.int32)),
                 "c": jnp.asarray(np.concatenate([inst["c"], np.zeros(n_ipad - n_i, np.int32)])),
                 "h": jnp.asarray(np.concatenate([inst["h"], np.zeros(n_ipad - n_i, np.int32)])),
                 "lp": jnp.asarray(np.concatenate([inst["lp"], np.zeros(n_ipad - n_i, np.float32)]))}  # fmt: skip
        pops = {"U": jnp.asarray(np.concatenate([U_now, np.repeat(U_now[:1], n_pop_pad - n_pop, 0)]), jnp.float32),
                "pos": jnp.asarray(np.concatenate([pos, np.zeros((n_pop_pad - n_pop, 3))]), jnp.float32),
                "rho": jnp.asarray(np.concatenate([d["density"], np.zeros(n_pop_pad - n_pop)]), jnp.float32)}  # fmt: skip
        prm = {"sig_a": jnp.float32(np.radians(sig_a)), "sig_inst": jnp.float32(np.radians(args.sig_inst)),
               "sig_frame": jnp.float32(np.radians(ostep) / np.sqrt(12)), "bg": jnp.float32(args.bg),
               "tol_deg": jnp.float32(3.0 * sig_tot)}  # fmt: skip
        K = jnp.zeros((n_pop_pad, 3, 3))
        S, Q, dg = jnp.zeros(n_pop_pad), jnp.zeros(n_pop_pad), jnp.zeros(3)
        for i in range(n_ch):
            sl = slice(i * C, (i + 1) * C)
            k_, s_, q_, d_ = step({k: v[sl] for k, v in pxd.items()}, instd, pops, bhat, geom, scan, bins, prm,
                                  P=args.pairs, n_pop=n_pop_pad)  # fmt: skip
            K, S, Q, dg = K + k_, S + s_, Q + q_, dg + d_
        K, S, Q, dg = (np.asarray(a, float) for a in (K, S, Q, dg))  # the pass's one host sync
        return K[:n_pop], S[:n_pop], Q[:n_pop], dg, inst["n_cut"], n_i

    if args.check:
        log("check: one pass from the indexer at the widest kernel")
        _, S, _, dg, n_cut, n_i = one_pass(U, sigmas[0])
        log(f"  {n_i} listed spots ({n_cut} cut by max_rows); candidates per pixel {dg[2] / n_px:.1f}; "
            f"intensity explained {dg[0] / dg[1]:.1%}")  # fmt: skip
        if truth is not None:
            from scipy.spatial import KDTree

            te = anri.io.entries_from_tensormap(truth)
            _, it = KDTree(te["pos"][:, :2]).query(pos[:, :2])
            Ut = np.linalg.inv(te["ubi"][it]) @ np.linalg.inv(B)
            for name, Uc in (("truth", Ut), ("indexer", U)):
                _, S, Q, dg, _, _ = one_pass(Uc, args.check_sigma)
                ok = S > 0
                log(f"  at the {name} orientations, kernel {args.check_sigma:g} deg: intensity explained "
                    f"{dg[0] / dg[1]:.1%}, rms misfit {np.degrees(np.sqrt(np.sum(Q[ok]) / (2 * np.sum(S[ok])))):.4f} deg")  # fmt: skip
        return

    for it_, sig_a in enumerate(sigmas):
        K, S, Q, dg, n_cut, n_i = one_pass(U, sig_a)
        ok = S > args.min_support
        U = np.where(ok[:, None, None], wahba(K), U)
        msg = (f"pass {it_ + 1}/{len(sigmas)} kernel {sig_a:g} deg: explained {dg[0] / dg[1]:.1%}, "
               f"candidates/pixel {dg[2] / n_px:.1f}, moved {ok.mean():.1%} of populations")  # fmt: skip
        log(msg)
        if truth is not None:
            truth_report(U, d, truth, B, ops, first, f"  pass {it_ + 1}")
    if args.out:
        np.savez(args.out, U=U, ubi=np.linalg.inv(U @ B), pos=d["pos"], density=d["density"], voxel=d["voxel"],
                 population=d["population"], support=S)  # fmt: skip
        log(f"-> {args.out}")


if __name__ == "__main__":
    main()
