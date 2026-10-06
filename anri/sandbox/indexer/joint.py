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


p = argparse.ArgumentParser()
p.add_argument("analysisroot")
p.add_argument("sample")
p.add_argument("dataset")
p.add_argument("match")
p.add_argument("--iter", type=int, default=200)
p.add_argument("--beam", type=float, help="beam FWHM (default: the dty step)")
p.add_argument("--voxel", type=float, help="voxel size (default: the dty step)")
p.add_argument("--match-px", type=float, default=3.0, help="largest distance from a prediction to its 2D peak (px)")
p.add_argument("--max-blobs", type=int, default=32)
p.add_argument("--n-cpu", type=int, default=12)
args = p.parse_args()

import anri.utils

anri.utils.setup(n_cpu=args.n_cpu)
import jax
import jax.numpy as jnp

from anri.fwd._impl.render import _peak_cov, _peak_factors, beam_weight, bin_fractions
from anri.geom import sample_to_lab
from spotlib import centre_dty, load, spot

d = load(args.analysisroot, args.sample, args.dataset, beam=args.beam, voxel=args.voxel)
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
blobs = jnp.asarray(d["blobs"])
cstart, order, rsort = jnp.asarray(d["cstart"]), jnp.asarray(d["order"]), jnp.asarray(d["rsort"])
dty_s, edges = jnp.asarray(d["dty_sorted"]), jnp.asarray(d["edges"])


def predict(u: jax.Array, x: jax.Array, hkl: jax.Array, etasign: jax.Array) -> tuple:
    mu, _, ok, _ = spot(u, x, hkl, etasign, d)
    om = mu[2]
    yc = centre_dty(x, om, geom)
    sig_om = jnp.sqrt(_peak_cov(u, x, hkl, etasign, yc, geom)[2] + 1e-8)
    lp = _peak_factors(u, hkl, etasign, geom)
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
    idx = lo[..., None] + jnp.arange(M)
    bl = blobs[jnp.minimum(idx, n_blob - 1)]
    dist = jnp.where(idx < hi[..., None], jnp.hypot(bl[..., 0] - mu[0], bl[..., 1] - mu[1]), jnp.inf)
    k = jnp.argmin(dist, axis=-1)
    hit = jnp.take_along_axis(dist, k[..., None], -1)[..., 0] < args.match_px
    blob = jnp.where(hit, jnp.take_along_axis(idx, k[..., None], -1)[..., 0], n_blob)  # n_blob: lands on nothing
    a = jnp.where(ok, lp * wb[:, None] * pf[None, :], 0.0)
    return a.ravel(), blob.ravel()


pred = jax.jit(jax.vmap(predict))
jh, hh, bb = (x.ravel().astype(np.int32) for x in np.meshgrid(np.arange(n_hyp), np.arange(n_hkl), [0, 1], indexing="ij"))
BIG = 1 << 15
n_all = -(-jh.size // BIG) * BIG
jp, hp, bp = (np.pad(x, (0, n_all - x.size)) for x in (jh, hh, bb))
A, B, Jx = [], [], []
t = time.perf_counter()
for s0 in range(0, n_all, BIG):
    sl = slice(s0, s0 + BIG)
    a, blob = pred(jnp.asarray(ubi[jp[sl]], jnp.float32), jnp.asarray(pos[jp[sl]], jnp.float32),
                   jnp.asarray(hkls[hp[sl]]), jnp.asarray(1.0 - 2.0 * bp[sl], jnp.float32))  # fmt: skip
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
x0 = np.full(n_hyp, 1.0, np.float32) / k_per
lam0 = np.bincount(B, A * x0[Jx], minlength=n_blob + 1)[:n_blob]
claimed = lam0 > 0
x0 *= I[:n_blob][claimed].sum() / lam0[claimed].sum()  # one global scale


@jax.jit
def mlem(x: jax.Array, Aj: jax.Array, Bj: jax.Array, Jj: jax.Array, sens: jax.Array, Ij: jax.Array) -> tuple:
    """Run --iter MLEM updates; the arrays are arguments, not constants baked into the compiled loop."""

    def step(i: int, carry: tuple) -> tuple:
        x, hist = carry
        lam = jax.ops.segment_sum(Aj * x[Jj], Bj, n_blob + 1)
        ratio = jnp.where((lam > 0) & (jnp.arange(n_blob + 1) < n_blob), Ij / jnp.maximum(lam, 1e-30), 0.0)
        ll = jnp.sum(jnp.where(lam[:-1] > 0, Ij[:-1] * jnp.log(jnp.maximum(lam[:-1], 1e-30)), 0.0)) - jnp.sum(sens * x)
        x = x * jax.ops.segment_sum(Aj * ratio[Bj], Jj, n_hyp) / jnp.maximum(sens, 1e-30)
        return x, hist.at[i].set(ll)

    return jax.lax.fori_loop(0, args.iter, step, (x, jnp.zeros(args.iter)))


t = time.perf_counter()
x, ll = mlem(jnp.asarray(x0), Aj, Bj, Jj, sens, jnp.asarray(I))
x, ll = np.asarray(x), np.asarray(ll)
log(f"MLEM: {args.iter} iterations, {time.perf_counter() - t:.1f} s; log-likelihood at 1, 10, 50, last: "
    f"{ll[0]:.6g} {ll[min(9, len(ll) - 1)]:.6g} {ll[min(49, len(ll) - 1)]:.6g} {ll[-1]:.6g}")  # fmt: skip

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


acc(e_vote[score], "vote's best")
acc(e_best[score], "joint (largest x)")
acc(oracle[score], "best hypothesis (bound)")
tot = X.sum(1)
if "main" in r and np.any(~r["main"]):  # the indexer's other populations: do they keep their share?
    print(f"  occupancy of the indexer's other populations relative to the main one in their voxel: median "
          f"{np.median(tot[~r['main']] / np.maximum(np.median(tot[r['main']]), 1e-30)):.2f} (main populations: "
          f"{np.median(tot[r['main']]) / np.median(tot[r['main']]):.2f})", flush=True)  # fmt: skip
print(f"  share of the chosen hypothesis: median {np.median(share):.2f}, 10th {np.percentile(share, 10):.2f}; "
      f"changed from the vote: {np.mean(best[score] != 0):.1%} (fixed "
      f"{np.mean((e_vote[score] > 0.05) & (e_best[score] < 0.05)):.1%}, broken "
      f"{np.mean((e_vote[score] < 0.05) & (e_best[score] > 0.05)):.1%})", flush=True)  # fmt: skip
# the chosen hypothesis as map entries (usable as match_spots.py --entries / --start)
np.savez(args.match.replace(".npz", "_joint.npz"), x=X, best=best, share=share, err=e_best,
         ubi=r["ubi_hyp"][np.arange(n_vox), best], pos=r["pos"], density=r["density"],
         population=np.where(r["main"], 0, 1) if "main" in r else np.zeros(n_vox, int))
