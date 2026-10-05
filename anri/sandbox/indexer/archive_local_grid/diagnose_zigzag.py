"""Is omega offset between the forward and backward rows of a zigzag scan? A check on the data alone, no model.

    python diagnose_zigzag.py <analysisroot> <sample> <dataset> [--phase NAME] [--rows 40] [--ring 0]

Adjacent dty rows see nearly the same spots. For each pair of adjacent rows in the middle of the scan, the ring's
(eta, omega) image of one row is cross-correlated with the next row's along omega, and the lag of the best match is
found to a fraction of a bin. With no offset the lags scatter around 0. If the rows rotated in opposite directions
and their omega readings are offset (e.g. a frame's start recorded instead of its centre, or encoder lag), the lags
alternate +d, -d: the offset between the two directions is d (in degrees; one frame is the omega step).
"""

import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("analysisroot")
p.add_argument("sample")
p.add_argument("dataset")
p.add_argument("--phase")
p.add_argument("--parfile")
p.add_argument("--rings", type=int, default=6)
p.add_argument("--ring", type=int, default=-1, help="ring to correlate (default: the one with the most intensity)")
p.add_argument("--rows", type=int, default=40, help="adjacent row pairs, around the middle of the scan")
p.add_argument("--bins", type=float, nargs=2, default=(0.25, 0.05), help="eta and omega bins (deg)")
p.add_argument("--max-lag", type=int, default=6, help="largest lag tried, in omega bins")
p.add_argument("--n-cpu", type=int, default=4)
args = p.parse_args()

import anri.utils  # noqa: E402

anri.utils.setup(n_cpu=args.n_cpu)
import sys  # noqa: E402

import h5py  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anri.index as ix  # noqa: E402
import anri.io  # noqa: E402
from fine import sparse_histogram  # noqa: E402

dsname = f"{args.sample}_{args.dataset}"
dsfile = os.path.join(args.analysisroot, args.sample, dsname, f"{dsname}_dataset.h5")
ds = anri.io.read_dataset(dsfile)
sparsefile = ds["sparsefile"] if ds["sparsefile"] and os.path.exists(ds["sparsefile"]) else dsfile.replace("_dataset.h5", "_sparse.h5")  # fmt: skip
geo, phase, cell = anri.io.read_pars_json(args.parfile or ds["parfile"], args.phase)
lpars = np.array([cell[k] for k in ("cell__a", "cell__b", "cell__c", "cell_alpha", "cell_beta", "cell_gamma")])
sg = int(cell["cell_lattice_[P,A,B,C,I,F,R]"])
ybin, yedge, oedge = ds["ybincens"], ds["ybinedges"], ds["obinedges"]
NK, OM0 = len(ybin), float(oedge[0])
geom = anri.io.geom_from_pars(geo, 0.0, 1e-4, 1e-4, 1e-4, sig_beam=1.0, voxel_size=float(np.median(np.diff(ybin))))
geom = {k: jnp.asarray(v, jnp.float32) if np.issubdtype(np.asarray(v).dtype, np.floating) else v for k, v in geom.items()}
rings = ix.ring_table(lpars, sg, geo["wavelength"], args.rings)
B_E, B_O = args.bins
N_E, N_O = round(360 / B_E), round(float(oedge[-1] - oedge[0]) / B_O)
print(f"{dsname}: omega motor {ds['omegamotor']!r}, dty motor {ds['dtymotor']!r}; {NK} rows; bins {B_E} x {B_O} deg")

# the rows around the middle of the scan, and the scans (groups) that hold them
k0 = max(0, NK // 2 - args.rows // 2)
k1 = min(NK, k0 + args.rows + 1)
with h5py.File(sparsefile, "r") as h:
    groups = list(h.keys())
    n_max = max(int(h[g]["nnz"][()].sum()) for g in groups)
chunk = int(min(1 << 24, 1 << max(10, int(np.ceil(np.log2(max(n_max, 1)))))))


def stream(gs):  # noqa: ANN001, ANN201
    return anri.io.stream_sparse(sparsefile, yedge, ds["omegamotor"], ds["dtymotor"], chunk, gs, 1, ds["dty"], ds["scans"])


sample = [groups[i] for i in np.unique(np.linspace(0, len(groups) - 1, min(len(groups), 9)).round().astype(int))]
off, hw = ix.ring_profile(stream(sample), geom, rings["tth"], chunk)
data = sparse_histogram(stream(groups), geom, rings["tth"], np.abs(off) + hw, OM0, (B_E, B_O, N_E, N_O), NK, chunk)
start, row, value = data["start"], data["row"], data["value"]

per_ring = [value[start[r * N_E * N_O] : start[(r + 1) * N_E * N_O]].sum() for r in range(len(rings["tth"]))]
ring = args.ring if args.ring >= 0 else int(np.argmax(per_ring))
a, b = start[ring * N_E * N_O], start[(ring + 1) * N_E * N_O]
cell = np.repeat(np.arange(N_E * N_O), np.diff(start[ring * N_E * N_O : (ring + 1) * N_E * N_O + 1]))
r_row, r_val = row[a:b], value[a:b]


def image(k: int) -> np.ndarray:
    """The ring's (eta, omega) image of row k."""
    img = np.zeros(N_E * N_O)
    m = r_row == k
    np.add.at(img, cell[m], r_val[m])
    return img.reshape(N_E, N_O)


L = np.arange(-args.max_lag, args.max_lag + 1)
lags, ks = [], []
nxt = image(k0)
for k in range(k0, k1 - 1):
    cur, nxt = nxt, image(k + 1)
    if cur.sum() == 0 or nxt.sum() == 0:
        continue
    c = np.array([np.sum(cur * np.roll(nxt, -lag, axis=1)) for lag in L])  # nxt shifted back by lag matches cur
    i = int(np.argmax(c))
    if 0 < i < len(L) - 1:  # parabola through the peak and its neighbours: the lag to a fraction of a bin
        d = c[i - 1] - 2 * c[i] + c[i + 1]
        sub = 0.5 * (c[i - 1] - c[i + 1]) / d if d != 0 else 0.0
    else:
        sub = 0.0
    lags.append((L[i] + sub) * B_O)
    ks.append(k)
lags, ks = np.array(lags), np.array(ks)
print(f"ring {ring} (2theta {rings['tth'][ring]:.3f} deg); omega lag (deg) of row k+1 against row k, rows {k0}..{k1 - 1}:")
for k, lag in zip(ks, lags):
    print(f"  {k:4d} -> {k + 1:4d}: {lag:+.4f}")
even, odd = lags[ks % 2 == 0], lags[ks % 2 == 1]
print(f"from even rows: mean {even.mean():+.4f} (sd {even.std():.4f}); from odd rows: mean {odd.mean():+.4f} "
      f"(sd {odd.std():.4f}); omega step {B_O}")  # fmt: skip
d = (odd.mean() - even.mean()) / 2  # half the swing between the two kinds of pair
alternating = np.sign(even.mean()) != np.sign(odd.mean()) and abs(d) > 3 * max(even.std(), odd.std()) / np.sqrt(len(lags) / 2)
if not alternating:
    print("=> no alternating offset between the rows")
else:
    print(f"=> forward and backward rows are offset by about {abs(d):.4f} deg ({abs(d) / B_O * 100:.0f}% of an omega bin): "
          + ("negligible for these bins" if abs(d) < 0.25 * B_O else "large enough to blur peaks across rows: correct it"))
