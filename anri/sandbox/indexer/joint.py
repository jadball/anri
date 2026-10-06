"""Choose each voxel's orientation jointly: MLEM over the hypotheses of match_spots.py against the 2D peak intensities.

    python joint.py <analysisroot> <sample> <dataset> <match_....npz> [--iter 200] [--beam FWHM] [--voxel SIZE]

Each voxel has a few hypotheses (refined vote peaks). A hypothesis predicts, for every spot, intensity into each
nearby (row, frame): Lorentz-polarisation x the beam's weight on the voxel at that row x the fraction of the peak in
that frame. That intensity goes to the 2D peak in the cell nearest the predicted (sc, fc), within --match-px, or to
nothing. The occupancies x (one per hypothesis, >= 0) are fitted by MLEM to the 2D peaks' summed intensities, all
voxels at once, so a voxel's rows are explained jointly with the other voxels on the same rays: a wrong hypothesis
predicts peaks into rows where its orientation is absent, and claims peaks that the true owners explain.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f} s] {msg}", flush=True)


import spotlib  # noqa: E402  (defines functions only: JAX starts after anri.utils.setup)

p = argparse.ArgumentParser()
spotlib.add_args(p)
p.add_argument("match")
p.add_argument("--iter", type=int, default=200)
p.add_argument("--hold", choices=("voxel", "entry"), default="voxel", help="voxel: each voxel's total density held, "
               "its populations may trade (they re-decide who owns boundary voxels); entry: each entry's density held, "
               "only the choice among its own hypotheses is fitted")
p.add_argument("--free", action="store_true", help="each hypothesis' occupancy free (the old way): densities trade "
               "along rays and the map is noisy. Default: each voxel's total held at the indexer's density x one scale")
p.add_argument("--match-px", type=float, default=3.0, help="largest distance from a prediction to its 2D peak (px)")
p.add_argument("--max-blobs", type=int, default=32)
p.add_argument("--sig-omega", type=float, default=0.0, help="extra omega spread of each predicted peak (deg): a soft "
               "split between frames where the instrument's peak is far narrower than a frame")
p.add_argument("--eta-cut", type=float, default=0.0, help="drop spots with |sin eta| below this (their omega moves as "
               "1 / |sin eta| with orientation, so their frames and rows are the least certain)")
p.add_argument("--save-terms", action="store_true", help="also save the fit's terms (for diagnostics)")
p.add_argument("--n-cpu", type=int, default=12)
args = p.parse_args()
if args.y0 is None and "y0" in np.load(args.match):  # as match_spots.py used
    args.y0 = float(np.load(args.match)["y0"])

import anri.utils

anri.utils.setup(n_cpu=args.n_cpu)
import jax
import jax.numpy as jnp

from anri.fwd._impl.render import _peak_cov, _peak_factors, beam_weight, bin_fractions
from anri.geom import sample_to_lab
from spotlib import band_start, centre_dty, spot

d = spotlib.load_args(args)
geom, hkls, n_rows, n_frames = d["geom"], d["hkls"], d["n_rows"], d["n_frames"]
n_blob = d["blobs"].shape[0]
r = np.load(args.match)
valid, err = r["valid"], r["err"]
vv, kk = np.nonzero(valid)
ubi = r["ubi_hyp"][vv, kk]
pos = r["pos"][vv]
n_hyp, n_vox, n_hkl = vv.size, valid.shape[0], len(hkls)
log(f"{n_vox} voxels, {n_hyp} hypotheses ({n_hyp / n_vox:.2f} per voxel), {n_blob} 2D peaks")

# --- each (hypothesis, spot): its intensity in each of 3 rows x 3 frames, and the 2D peak it lands on --------------
M = args.max_blobs
# tables passed to the jitted function as arguments: closed over, the peaks would be compiled in as constants
tb = {k: jnp.asarray(d[k]) for k in ("blobs", "cstart", "order", "rsort", "dty_sorted", "edges")}
NS = d["n_search"]


def predict(u: jax.Array, x: jax.Array, hkl: jax.Array, f2: jax.Array, etasign: jax.Array, tb: dict) -> tuple:
    blobs, cstart, order, rsort, dty_s, edges = (tb[k] for k in ("blobs", "cstart", "order", "rsort", "dty_sorted",
                                                                 "edges"))  # fmt: skip
    mu, _, ok, sin_eta = spot(u, x, hkl, etasign, d)
    ok = ok & (sin_eta >= args.eta_cut)
    om = mu[2]
    yc = centre_dty(x, om, geom)
    sig_om = jnp.sqrt(_peak_cov(u, x, hkl, etasign, yc, geom)[2] + args.sig_omega**2 + 1e-8)
    lp = f2 * _peak_factors(u, hkl, etasign, geom)  # |F|^2 x Lorentz-polarisation
    f0 = jnp.searchsorted(edges, om) - 1
    s = jnp.clip(f0 + jnp.arange(-1, 2), 0, n_frames - 1)  # sorted frames
    pf = bin_fractions(edges[s], edges[s + 1], om, sig_om) * (f0 + jnp.arange(-1, 2) == s)  # [3]
    rn = jnp.clip(jnp.argmin(jnp.abs(dty_s - yc)) + jnp.arange(-1, 2), 0, n_rows - 1)  # [3] rows, dty order
    wb = jax.vmap(lambda y: beam_weight(sample_to_lab(x, om, geom["wedge"], geom["chi"], y, geom["y0"]), om, geom))(
        dty_s[rn]
    ) * (jnp.argmin(jnp.abs(dty_s - yc)) + jnp.arange(-1, 2) == rn)  # [3]
    row = rsort[rn]
    c = row[:, None] * n_frames + order[row[:, None], s[None, :]]  # [3 rows, 3 frames]
    lo, hi = cstart[c], cstart[c + 1]
    start = band_start(blobs[:, 0], lo, hi, mu[0] - args.match_px, NS)  # the cell's peaks within match_px in slow
    idx = start[..., None] + jnp.arange(M)
    bl = blobs[jnp.minimum(idx, n_blob - 1)]
    dist = jnp.where(idx < hi[..., None], jnp.hypot(bl[..., 0] - mu[0], bl[..., 1] - mu[1]), jnp.inf)
    k = jnp.argmin(dist, axis=-1)
    hit = jnp.take_along_axis(dist, k[..., None], -1)[..., 0] < args.match_px
    blob = jnp.where(hit, jnp.take_along_axis(idx, k[..., None], -1)[..., 0], n_blob)  # n_blob: lands on nothing
    a = jnp.where(ok, lp * wb[:, None] * pf[None, :], 0.0)
    return a.ravel(), blob.ravel()


pred = jax.jit(jax.vmap(predict, in_axes=(0, 0, 0, 0, 0, None)))
jh, hh, bb = (x.ravel().astype(np.int32) for x in np.meshgrid(np.arange(n_hyp), np.arange(n_hkl), [0, 1], indexing="ij"))
BIG = 1 << 15
n_all = -(-jh.size // BIG) * BIG
jp, hp, bp = (np.pad(x, (0, n_all - x.size)) for x in (jh, hh, bb))
A, B, Jx = [], [], []
t = time.perf_counter()
for s0 in range(0, n_all, BIG):
    sl = slice(s0, s0 + BIG)
    a, blob = pred(jnp.asarray(ubi[jp[sl]], jnp.float32), jnp.asarray(pos[jp[sl]], jnp.float32),
                   jnp.asarray(hkls[hp[sl]]), jnp.asarray(d["F2"][hp[sl]]),
                   jnp.asarray(1.0 - 2.0 * bp[sl], jnp.float32), tb)  # fmt: skip
    a, blob = np.array(a), np.asarray(blob)
    a[max(0, jh.size - s0) :] = 0.0  # padding
    nz = np.nonzero(a > 1e-4 * a.max())
    A.append(a[nz])
    B.append(blob[nz])
    Jx.append(jp[sl][nz[0]])
A, B, Jx = np.concatenate(A).astype(np.float32), np.concatenate(B).astype(np.int32), np.concatenate(Jx).astype(np.int32)
hit = B < n_blob
log(f"predictions: {A.size} (hypothesis, spot, row, frame) terms, {time.perf_counter() - t:.1f} s; "
    f"{np.sum(A[hit]) / np.sum(A):.1%} of the predicted intensity lands on a 2D peak; "
    f"{np.unique(B[hit]).size} of {n_blob} peaks are claimed")  # fmt: skip

# --- MLEM over the occupancies, one jitted loop ------------------------------------------------------------------
I = np.concatenate([d["blob_i"], [0.0]]).astype(np.float32)  # the last: "nothing", never fits anything
Aj, Bj, Jj = jnp.asarray(A), jnp.asarray(B), jnp.asarray(Jx)
sens = jax.ops.segment_sum(Aj, Jj, n_hyp)
k_per = np.bincount(vv, minlength=n_vox)[vv]
# voxels: entries at the same place; each voxel's total density is the indexer's (sum over its populations)
_, vox_e = np.unique(np.round(r["pos"][:, :2] / (0.25 * d["ystep"])).astype(np.int64), axis=0, return_inverse=True)
vox_e = vox_e.ravel()
if args.hold == "entry":  # each entry is its own group: populations cannot trade
    vox_e = np.arange(n_vox)
n_v = int(vox_e.max()) + 1
dens_e = r["density"].astype(np.float64)
d_v = np.bincount(vox_e, dens_e, minlength=n_v)
vh = vox_e[vv]  # each hypothesis' voxel
pi0 = (dens_e[vv] / np.maximum(d_v[vh], 1e-30) / k_per).astype(np.float32)  # fractions within the voxel
base = (d_v[vh]).astype(np.float32)
lam0 = np.bincount(B, A * (base * pi0)[Jx], minlength=n_blob + 1)[:n_blob]
claimed = lam0 > 0
s0 = np.float32(I[:n_blob][claimed].sum() / lam0[claimed].sum())  # one global scale


@jax.jit
def mlem(pi: jax.Array, s: jax.Array, base: jax.Array, vh: jax.Array, Aj: jax.Array, Bj: jax.Array, Jj: jax.Array,
         sens: jax.Array, Ij: jax.Array) -> tuple:  # fmt: skip
    """Run --iter EM updates; the arrays are arguments, not constants baked into the compiled loop.

    Occupancy x = s * base * pi. Free: pi takes the plain MLEM update. Fixed (default): the mixture-weight update,
    pi * g / sens renormalised to sum to 1 within each voxel (the hypotheses of a voxel predict nearly the same
    total, so this is the EM step for its fractions), and s its own EM update.
    """

    def step(i: int, carry: tuple) -> tuple:
        pi, s, hist = carry
        x = s * base * pi
        lam = jax.ops.segment_sum(Aj * x[Jj], Bj, n_blob + 1)
        ratio = jnp.where((lam > 0) & (jnp.arange(n_blob + 1) < n_blob), Ij / jnp.maximum(lam, 1e-30), 0.0)
        ll = jnp.sum(jnp.where(lam[:-1] > 0, Ij[:-1] * jnp.log(jnp.maximum(lam[:-1], 1e-30)), 0.0)) - jnp.sum(sens * x)
        g = jax.ops.segment_sum(Aj * ratio[Bj], Jj, n_hyp)
        if args.free:
            pi = pi * g / jnp.maximum(sens, 1e-30)
        else:
            s = s * jnp.sum(x * g) / jnp.maximum(jnp.sum(x * sens), 1e-30)
            pi = pi * g / jnp.maximum(sens, 1e-30)
            pi = pi / jnp.maximum(jax.ops.segment_sum(pi, vh, n_v)[vh], 1e-30)
        return pi, s, hist.at[i].set(ll)

    return jax.lax.fori_loop(0, args.iter, step, (pi, s, jnp.zeros(args.iter)))


t = time.perf_counter()
pi, s_fit, ll = mlem(jnp.asarray(pi0), jnp.float32(s0), jnp.asarray(base), jnp.asarray(vh), Aj, Bj, Jj, sens,
                     jnp.asarray(I))  # fmt: skip
x, ll = np.asarray(s_fit * jnp.asarray(base) * pi), np.asarray(ll)
log(f"  {'free occupancies' if args.free else args.hold + ' totals held'}; global scale {float(s_fit):.4g} (start {s0:.4g})")
log(f"MLEM: {args.iter} iterations, {time.perf_counter() - t:.1f} s; log-likelihood at 1, 10, 50, last: "
    f"{ll[0]:.6g} {ll[min(9, len(ll) - 1)]:.6g} {ll[min(49, len(ll) - 1)]:.6g} {ll[-1]:.6g}")  # fmt: skip

lam = np.bincount(B, A * x[Jx], minlength=n_blob + 1)[:n_blob]
k_ = lam > 0
r_ = d["blob_i"][k_] / lam[k_]
q_ = np.percentile(np.repeat(r_, np.clip((d["blob_i"][k_] / np.median(d["blob_i"][k_])).astype(int), 1, 50)), [10, 50, 90])
log(f"  measured / predicted per 2D peak (intensity-weighted) 10/50/90: {np.round(q_, 3)}")

# --- each voxel's choice: the hypothesis with the largest occupancy ----------------------------------------------
X = np.zeros(valid.shape)
X[vv, kk] = x
best = np.argmax(X, axis=1)
share = X[np.arange(n_vox), best] / np.maximum(X.sum(1), 1e-30)
e_best = err[np.arange(n_vox), best]
e_vote = err[:, 0]
oracle = np.min(np.where(valid, err, np.inf), axis=1)
score = (r["main"] if "main" in r else np.ones(n_vox, bool)) & np.isfinite(e_vote)  # main populations with a truth


def acc(e: np.ndarray, label: str) -> None:
    print(f"  {label:>22}: median {np.median(e):.4f} deg; within 0.02 {np.mean(e < 0.02):.1%}, 0.05 "
          f"{np.mean(e < 0.05):.1%}, 0.1 {np.mean(e < 0.1):.1%}, 0.25 {np.mean(e < 0.25):.1%}", flush=True)  # fmt: skip


if np.any(score):  # with a truth (match_spots.py --truth)
    acc(e_vote[score], "vote's best")
    acc(e_best[score], "joint (largest x)")
    acc(oracle[score], "best hypothesis (bound)")
tot = X.sum(1)
if "main" in r and np.any(~r["main"]):  # the indexer's other populations: do they keep their share?
    print(f"  occupancy of the indexer's other populations relative to the main one in their voxel: median "
          f"{np.median(tot[~r['main']] / np.maximum(np.median(tot[r['main']]), 1e-30)):.2f} (main populations: "
          f"{np.median(tot[r['main']]) / np.median(tot[r['main']]):.2f})", flush=True)  # fmt: skip
main_ = r["main"] if "main" in r else np.ones(n_vox, bool)
line = (f"  share of the chosen hypothesis: median {np.median(share):.2f}, 10th {np.percentile(share, 10):.2f}; "
        f"changed from the vote: {np.mean(best[main_] != 0):.1%}")
if np.any(score):
    line += (f" (fixed {np.mean((e_vote[score] > 0.05) & (e_best[score] < 0.05)):.1%}, broken "
             f"{np.mean((e_vote[score] < 0.05) & (e_best[score] > 0.05)):.1%})")
print(line, flush=True)
# the chosen hypothesis as map entries (usable as match_spots.py --entries / --start)
np.savez(args.match.replace(".npz", "_joint.npz"), x=X, best=best, share=share, err=e_best,
         ubi=r["ubi_hyp"][np.arange(n_vox), best], pos=r["pos"], density=r["density"],
         population=np.where(r["main"], 0, 1) if "main" in r else np.zeros(n_vox, int))
if args.save_terms:
    np.savez(args.match.replace(".npz", "_terms.npz"), A=A, B=B, J=Jx, I=I, vv=vv, kk=kk, vh=vh, base=base,
             pi=np.asarray(pi), s=float(s_fit), sens=np.asarray(sens), blobs=d["blobs"], blob_cell=d["blob_cell"])
