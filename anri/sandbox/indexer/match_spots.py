"""Match every predicted spot of a map to ImageD11's 2D peaks along the voxel's own sinusoid, and vote per voxel.

    python match_spots.py <analysisroot> <sample> <dataset> (--entries <..._entries.npz> | --truth <tmap.h5>)
        [--perturb DEG] [--tol DEG] [--max-omega DEG] [--n-hyp 4]

For each frame within --max-omega of a spot's predicted omega, the voxel's position across the beam at that frame
picks the two dty rows that bracket it (ImageD11's voxel_mask predicate), and every 2D peak in those (row, frame)
cells is tested: it is a candidate if a rotation |delta| < --tol maps the prediction onto it, with the part no
rotation can explain (essentially 2theta) under 3 sigma.

Per voxel, each candidate allows a line of rotations (the turn about G is free); the lines vote in a grid of
rotations. The --n-hyp best distinct peaks of the vote are refined by least squares on each spot's nearest candidate
and saved as the voxel's hypotheses (<analysisroot>/match_<perturb>.npz) for joint.py.

With --truth, --perturb turns each entry by a random rotation of that size (deg); candidates are scored "right" or
"wrong" against the known correction, and hypotheses by their error.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f} s] {msg}", flush=True)


p = argparse.ArgumentParser()
p.add_argument("analysisroot")
p.add_argument("sample")
p.add_argument("dataset")
p.add_argument("--entries")
p.add_argument("--truth")
p.add_argument("--perturb", type=float, default=0.0, help="random rotation of each entry (deg), with --truth")
p.add_argument("--start", help="with --truth: start from these entries' ubi (e.g. joint.py's output) instead")
p.add_argument("--out", help="output npz (default <analysisroot>/match_<perturb>.npz)")
p.add_argument("--tol", type=float, default=0.6, help="largest rotation a match may need (deg)")
p.add_argument("--max-omega", type=float, default=2.0, help="omega window each side of a prediction (deg)")
p.add_argument("--sig", type=float, nargs=3, default=(0.5, 0.5, 0.05), help="sigma of sc, fc (px), omega (deg)")
p.add_argument("--max-blobs", type=int, default=32, help="2D peaks tested per (row, frame) cell")
p.add_argument("--max-cand", type=int, default=64, help="candidates kept per spot for the vote")
p.add_argument("--step", type=float, help="vote grid step (deg, default tol / 10)")
p.add_argument("--n-hyp", type=int, default=4, help="hypotheses kept per voxel")
p.add_argument("--n-cpu", type=int, default=12)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

import anri.utils

anri.utils.setup(n_cpu=args.n_cpu)
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation

import anri.io
from spotlib import centre_dty, load, spot

d = load(args.analysisroot, args.sample, args.dataset)
n_rows, n_frames, hkls = d["n_rows"], d["n_frames"], d["hkls"]
cs = np.diff(d["cstart"])
M = args.max_blobs
log(f"{n_rows} rows, {n_frames} frames of {d['ostep']:g} deg; {d['blobs'].shape[0]} 2D peaks, per (row, frame) "
    f"mean {cs.mean():.2f}, max {cs.max()} ({np.sum(cs > M)} cells over {M})")  # fmt: skip

# --- entries -----------------------------------------------------------------------------------------------------
rng = np.random.default_rng(args.seed)
ubi_true = None
if args.entries:  # e.g. the indexer's: every population; with --truth, scored against the truth voxel at its place
    r = np.load(args.entries)
    ubi, pos, dens = r["ubi"].astype(np.float64), r["pos"].astype(np.float64), r["density"].astype(np.float64)
    main = r["population"] == 0 if "population" in r else np.ones(len(ubi), bool)
    if args.truth:
        from ImageD11.sinograms.tensor_map import TensorMap
        from scipy.spatial import cKDTree

        import anri.crystal

        te = anri.io.entries_from_tensormap(TensorMap.from_h5(args.truth))
        B = anri.crystal.B_matrix(d["lpars"])
        ops = anri.crystal.laue_rotations(anri.crystal.symmetry_matrices(225), B)
        dist, it = cKDTree(te["pos"][:, :2]).query(pos[:, :2])
        has = dist < 0.5 * d["ystep"]
        U_s = np.linalg.inv(ubi) @ np.linalg.inv(B)
        U_t = (np.linalg.inv(te["ubi"]) @ np.linalg.inv(B))[it]
        var = np.einsum("nij,kjl->nkil", U_t, ops)  # U S: the same reflections
        k = np.argmax(np.einsum("nij,nkij->nk", U_s, var), axis=1)  # the variant closest to the start
        ubi_true = np.linalg.inv(var[np.arange(len(ubi)), k] @ B)
        ubi_true[~has] = np.nan
elif args.truth:
    from ImageD11.sinograms.tensor_map import TensorMap

    ent = anri.io.entries_from_tensormap(TensorMap.from_h5(args.truth))
    ubi_true, pos = ent["ubi"].astype(np.float64), ent["pos"].astype(np.float64)
    dens, main = np.ones(len(pos)), np.ones(len(pos), bool)
    axis = rng.normal(size=(len(pos), 3))
    w = axis / np.linalg.norm(axis, axis=1, keepdims=True) * np.radians(args.perturb)
    ubi = ubi_true @ np.swapaxes(Rotation.from_rotvec(w).as_matrix(), 1, 2)  # UB -> R UB, so UBI -> UBI R^T
    if args.start:
        ubi = np.load(args.start)["ubi"].astype(np.float64)
d_true = np.zeros((len(ubi), 3))
if ubi_true is not None:  # the correction back: ubi_true = ubi R(d)^T, R(d) the nearest rotation to (ubi^-1 ubi_true)^T
    ok_t = np.isfinite(ubi_true).all(axis=(1, 2)) & np.isfinite(ubi).all(axis=(1, 2))
    u_, _, vt_ = np.linalg.svd(np.swapaxes(np.linalg.inv(ubi[ok_t]) @ ubi_true[ok_t], 1, 2))
    d_true[ok_t] = Rotation.from_matrix(u_ @ vt_).as_rotvec()
    d_true[~ok_t] = np.nan
    e_start = np.degrees(np.linalg.norm(d_true, axis=1))
    sc_ = main & ok_t
    print(f"  start vs truth (main populations, {sc_.sum()}): median {np.median(e_start[sc_]):.3f} deg; within 0.05 "
          f"{np.mean(e_start[sc_] < 0.05):.1%}, 0.25 {np.mean(e_start[sc_] < 0.25):.1%}, 1 {np.mean(e_start[sc_] < 1):.1%}",
          flush=True)  # fmt: skip
n_ent, n_hkl = len(ubi), len(hkls)
log(f"{n_ent} entries x {n_hkl} hkls x 2 branches; perturbed by {args.perturb} deg; tolerance {args.tol} deg")

# --- spot geometry: one compile, a few big calls -----------------------------------------------------------------
spots_chunk = jax.jit(jax.vmap(lambda u, x, h, s: spot(u, x, h, s, d)))
e, h, b = (x.ravel().astype(np.int32) for x in np.meshgrid(np.arange(n_ent), np.arange(n_hkl), [0, 1], indexing="ij"))
BIG = 1 << 16
n_all = -(-e.size // BIG) * BIG
ep, hp, bp = (np.pad(x, (0, n_all - x.size)) for x in (e, h, b))
outs = [spots_chunk(jnp.asarray(ubi[ep[s:s + BIG]], jnp.float32), jnp.asarray(pos[ep[s:s + BIG]], jnp.float32),
                    jnp.asarray(hkls[hp[s:s + BIG]]), jnp.asarray(1.0 - 2.0 * bp[s:s + BIG], jnp.float32))
        for s in range(0, n_all, BIG)]  # fmt: skip
mu, J, ok, sin_eta = (np.concatenate([np.asarray(o[i]) for o in outs])[: e.size] for i in range(4))
keep = np.nonzero(ok)[0]
n_sp = keep.size
log(f"geometry: {n_sp} spots in the scan (of {e.size})")

# --- matching ----------------------------------------------------------------------------------------------------
K = 2 * int(np.ceil(args.max_omega / d["ostep"])) + 1
CM = args.max_cand
tol = np.radians(args.tol)
W = jnp.asarray(1.0 / np.asarray(args.sig, np.float32))
blobs = jnp.asarray(d["blobs"])
cstart_j, order_j, rsort_j = jnp.asarray(d["cstart"]), jnp.asarray(d["order"]), jnp.asarray(d["rsort"])
dty_j, om_j, edges_j = jnp.asarray(d["dty_sorted"]), jnp.asarray(d["om_sorted"]), jnp.asarray(d["edges"])


def match(mu: jax.Array, J: jax.Array, pos: jax.Array, dtrue: jax.Array) -> tuple:
    """Candidates of one spot: counts (all, right) and the scaled offsets z of the best CM by |delta|."""
    A = W[:, None] * J
    U, S, Vt = jnp.linalg.svd(A)
    Sinv = jnp.where(S > 1e-3 * S[0], 1.0 / S, 0.0)
    P = Vt.T @ (Sinv[:, None] * U.T)  # pseudo-inverse: measurement (scaled) -> rotation
    n = U[:, 2]  # the direction no rotation reaches (essentially 2theta)
    s = jnp.searchsorted(edges_j, mu[2]) - 1 - K // 2 + jnp.arange(K)  # sorted frames of the window
    fin = (s >= 0) & (s < n_frames)
    s = jnp.clip(s, 0, n_frames - 1)
    y = jax.vmap(lambda o: centre_dty(pos, o, d["geom"]))(om_j[s])  # the voxel's sinusoid: its dty at each frame
    r_hi = jnp.clip(jnp.searchsorted(dty_j, y), 1, n_rows - 1)
    row = rsort_j[jnp.stack([r_hi - 1, r_hi], 1)]  # [K, 2] the rows bracketing it
    c = row * n_frames + order_j[row, s[:, None]]
    lo, hi = cstart_j[c], cstart_j[c + 1]
    idx = lo[..., None] + jnp.arange(M)  # [K, 2, M]
    live = (idx < hi[..., None]) & fin[:, None, None]
    z = W * (blobs[jnp.minimum(idx, blobs.shape[0] - 1)] - mu)
    dl = z @ P.T  # implied rotation
    cand = live & (jnp.abs(z @ n) < 3.0) & (jnp.linalg.norm(dl, axis=-1) < tol)
    right = cand & (jnp.linalg.norm(z - A @ dtrue, axis=-1) < 3.0)
    sv, si = jax.lax.top_k(jnp.where(cand, -jnp.linalg.norm(dl, axis=-1), -jnp.inf).ravel(), CM)
    return cand.sum(), right.sum(), z.reshape(-1, 3)[si], jnp.isfinite(sv)


match_chunk = jax.jit(jax.vmap(match))
t = time.perf_counter()
MC = 1 << 13
n_pad = -(-n_sp // MC) * MC
sel = np.pad(keep, (0, n_pad - n_sp))
mu_k, J_k, e_k = mu[sel], J[sel].astype(np.float32), e[sel]
outs = [match_chunk(*(jnp.asarray(x[s:s + MC]) for x in (mu_k, J_k, pos[e_k].astype(np.float32),
                                                         d_true[e_k].astype(np.float32))))
        for s in range(0, n_pad, MC)]  # fmt: skip
cand, right, zc, cv = (np.concatenate([np.asarray(o[i]) for o in outs])[:n_sp] for i in range(4))
wrong = cand - right
se = sin_eta[keep]
log(f"matching: {time.perf_counter() - t:.1f} s, window {K} frames x 2 rows x {M} peaks")


def report(m: np.ndarray, label: str) -> None:
    c, rt, wr = cand[m], right[m], wrong[m]
    hist = [np.mean(c == k) for k in (0, 1, 2, 3)] + [np.mean((c >= 4) & (c < 9)), np.mean(c >= 9)]
    line = f"  {label:>16}: {m.sum():8d} spots; candidates 0/1/2/3/4-8/9+: " + " ".join(f"{x:5.1%}" for x in hist)
    if args.truth:
        line += (f"; right found {np.mean(rt > 0):5.1%}, any wrong {np.mean(wr > 0):5.1%}, "
                 f"unambiguous {np.mean((rt > 0) & (wr == 0)):5.1%}")  # fmt: skip
    print(line, flush=True)


report(np.ones(n_sp, bool), "all")
for lo_, hi_ in ((0.0, 0.2), (0.2, 0.5), (0.5, 1.01)):
    report((se >= lo_) & (se < hi_), f"|sin eta| {lo_}-{hi_:.1f}")
print(f"  candidates per spot: median {np.median(cand):.0f}, 90th {np.percentile(cand, 90):.0f}, max {cand.max()}; "
      f"over --max-cand {CM}: {np.mean(cand > CM):.1%}", flush=True)  # fmt: skip

# --- per-voxel vote: each candidate allows a line of rotations; the lines vote in a grid; the best n_hyp distinct
# peaks are refined by least squares on each spot's nearest candidate ----------------------------------------------
step = np.radians(args.step or args.tol / 10)
G = 2 * int(np.ceil(tol / step)) + 1
tl = jnp.linspace(-tol, tol, G)
NH = args.n_hyp
supp = max(1, int(round(np.radians(0.15) / step)))  # votes cleared around a peak before taking the next
counts = np.bincount(e_k[:n_sp], minlength=n_ent)
S_max = int(counts.max())
table = np.full((n_ent, S_max), n_sp, np.int32)  # n_sp: a dead spot
table[e_k[:n_sp], np.arange(n_sp) - np.concatenate([[0], np.cumsum(counts)[:-1]])[e_k[:n_sp]]] = np.arange(n_sp)
J_all = jnp.asarray(np.concatenate([J_k[:n_sp], np.eye(3, dtype=np.float32)[None]]))
zc_all = jnp.asarray(np.concatenate([zc, np.zeros((1, CM, 3), np.float32)]))
cv_all = jnp.asarray(np.concatenate([cv, np.zeros((1, CM), bool)]))


def vote(sp: jax.Array, J_all: jax.Array, zc_all: jax.Array, cv_all: jax.Array) -> tuple:
    """One voxel's hypotheses. The tables are arguments: closed over, ~1 GB would be compiled in as constants."""
    J, z, ok = J_all[sp], zc_all[sp], cv_all[sp]  # [S, 3, 3], [S, CM, 3], [S, CM]
    A = W[None, :, None] * J
    U, S, Vt = jnp.linalg.svd(A)
    Sinv = jnp.where(S > 1e-3 * S[:, :1], 1.0 / S, 0.0)
    P = jnp.einsum("sji,sj,skj->sik", Vt, Sinv, U)
    g = Vt[:, 2]  # [S, 3] the rotation that does not move the spot (about G)
    d0 = jnp.einsum("sij,scj->sci", P, z)
    pts = d0[:, :, None] + tl[None, None, :, None] * g[:, None, None]  # [S, CM, G, 3]
    bi = jnp.round(pts / step).astype(jnp.int32) + G // 2
    inn = jnp.all((bi >= 0) & (bi < G), -1) & ok[:, :, None]
    flat = jnp.where(inn, (bi[..., 0] * G + bi[..., 1]) * G + bi[..., 2], G**3)
    # empty slots point past the end and are dropped: an in-bounds dummy bin would take millions of atomic adds
    hist = jnp.zeros(G**3).at[flat.ravel()].add(1.0, mode="drop")
    grid = jnp.stack(jnp.unravel_index(jnp.arange(G**3), (G, G, G)), 1)

    def fit(delta: jax.Array, thr: float, first: bool) -> tuple:
        if first:  # distance from delta to each candidate's line, in grid steps
            v = delta - d0
            dist = jnp.linalg.norm(v - jnp.einsum("sck,sk->sc", v, g)[..., None] * g[:, None], axis=-1) / step
        else:
            dist = jnp.linalg.norm(jnp.einsum("sij,j->si", A, delta)[:, None] - z, axis=-1)
        dist = jnp.where(ok, dist, jnp.inf)
        zi = jnp.take_along_axis(z, jnp.argmin(dist, axis=1)[:, None, None], 1)[:, 0]
        inl = jnp.min(dist, axis=1) < thr
        N = jnp.einsum("s,sji,sjk->ik", inl, A, A)
        bv = jnp.einsum("s,sji,sj->i", inl, A, zi)
        return jnp.linalg.solve(N + 1e-6 * jnp.eye(3), bv), inl.sum()

    deltas, n_ins, votes = [], [], []
    for _ in range(NH):
        best = jnp.argmax(hist)
        votes.append(hist[best])
        bg = grid[best]
        delta = (bg - G // 2) * step
        delta, _ = fit(delta, 1.0, True)
        delta, _ = fit(delta, 6.0, False)
        delta, n_in = fit(delta, 3.0, False)
        deltas.append(delta)
        n_ins.append(n_in)
        hist = jnp.where(jnp.all(jnp.abs(grid - bg) <= supp, axis=1), -1.0, hist)
    return jnp.stack(deltas), jnp.stack(n_ins), jnp.stack(votes), jnp.sum(ok.any(1))


vote_chunk = jax.jit(jax.vmap(vote, in_axes=(0, None, None, None)))
t = time.perf_counter()
VC = 256
n_vp = -(-n_ent // VC) * VC
tab = np.concatenate([table, np.full((n_vp - n_ent, S_max), n_sp, np.int32)])
outs = [vote_chunk(jnp.asarray(tab[:VC]), J_all, zc_all, cv_all)]
jax.block_until_ready(outs)
log(f"vote: first call (compile + run) {time.perf_counter() - t:.1f} s")
outs += [vote_chunk(jnp.asarray(tab[v0 : v0 + VC]), J_all, zc_all, cv_all) for v0 in range(VC, n_vp, VC)]
d_hyp, n_in, votes, n_sp_v = (np.concatenate([np.asarray(o[i]) for o in outs])[:n_ent] for i in range(4))
log(f"vote: {time.perf_counter() - t:.1f} s, grid {G}^3 of {np.degrees(step):.3f} deg, {S_max} spots per voxel at most")
print(f"  entries with a non-finite hypothesis: {np.mean(~np.isfinite(d_hyp).all(axis=(1, 2))):.2%}; "
      f"with no inliers at all: {np.mean(n_in.max(1) == 0):.2%}", flush=True)
d_hyp[~np.isfinite(d_hyp) | (n_in[..., None] == 0)] = 0.0  # no fit: keep the start
# a hypothesis that refined onto an earlier one (within 0.03 deg) is a duplicate
valid = n_in > 0
for k in range(1, NH):
    for j in range(k):
        valid[:, k] &= np.degrees(np.linalg.norm(d_hyp[:, k] - d_hyp[:, j], axis=1)) > 0.03
err = np.degrees(np.linalg.norm(d_hyp - d_true[:, None], axis=2))  # [n_ent, NH]
score = main & np.isfinite(err[:, 0])  # main populations with a truth
err, valid_s = err[score], valid[score]
e0 = err[:, 0]
print(f"  vote's best: median error {np.median(e0):.4f} deg; within 0.01 {np.mean(e0 < 0.01):.1%}, 0.02 "
      f"{np.mean(e0 < 0.02):.1%}, 0.05 {np.mean(e0 < 0.05):.1%}, 0.1 {np.mean(e0 < 0.1):.1%}", flush=True)  # fmt: skip
any_ok = np.any((err < 0.05) & valid_s, axis=1)
print(f"  hypotheses per entry: mean {valid.sum(1).mean():.2f}; truth (within 0.05) among them: {np.mean(any_ok):.1%}; "
      f"for the voxels the vote gets wrong: {np.mean(any_ok[e0 > 0.05]):.1%}", flush=True)  # fmt: skip
R = Rotation.from_rotvec(d_hyp.reshape(-1, 3)).as_matrix().reshape(n_ent, NH, 3, 3)
ubi_hyp = np.einsum("nij,nklj->nkil", ubi, R)  # UBI R^T
out = args.out or os.path.join(args.analysisroot, f"match_{args.perturb}.npz")
err = np.degrees(np.linalg.norm(d_hyp - d_true[:, None], axis=2))
np.savez(out, ubi_hyp=ubi_hyp, valid=valid, err=err, main=main, n_in=n_in, votes=votes, n_sp=n_sp_v, pos=pos, density=dens)
log(f"-> {out}")
