"""Stage-3 test case: render the am316l phantom (as the indexing tutorial does), run stage 1-2, save everything."""

import os
import sys
import time

import anri.utils

anri.utils.setup(n_cpu=8)
import jax.numpy as jnp
import numpy as np
from ImageD11.sinograms.tensor_map import TensorMap

import anri.crystal
import anri.geom
import anri.index as ix
import anri.io

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.1f} s] {m}", flush=True)


out = sys.argv[1]
os.makedirs(out, exist_ok=True)
truth = TensorMap.from_h5("/home/james/Code/anri/tests/data/phantoms/am316l/am316l_tmap.h5")
a = truth.phases[0].lattice_parameters[0]
crystal = anri.crystal.Crystal(anri.crystal.UnitCell.from_lpars(jnp.asarray([a, a, a, 90.0, 90.0, 90.0])), anri.crystal.Symmetry.from_number(225))
B = np.asarray(crystal.B, np.float32)
ops = anri.crystal.laue_rotations(np.asarray(crystal.sym_ops), B)
wl = 0.2843
pars = {
    "y_center": 1023.5, "y_size": 75.0, "tilt_y": 1e-3, "z_center": 1023.5, "z_size": 75.0, "tilt_z": -2e-3,
    "tilt_x": 0.0, "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1, "wavelength": wl, "wedge": 0.0,
    "chi": 0.0, "omegasign": 1.0, "t_x": 0.0, "t_y": 0.0, "t_z": 0.0,
}  # fmt: skip
y0 = 0.3
sparse = os.path.join(out, "sparse.h5")
if not os.path.exists(sparse):
    geom_r = anri.io.geom_from_pars(pars, y0, wl * 2e-4 / 2.355, 5e-5, 5e-5, sig_beam=1.4 / 2.355, voxel_size=0.5, sig_psf=0.5)
    entries = anri.io.entries_from_tensormap(truth)
    entries["density"] = np.full(len(entries["pos"]), 30.0)
    rings8 = ix.ring_table(crystal, wl, 8)
    omega, dty = anri.io.motor_grid((0.0, 180.0), 0.1, (y0 - 30.0, y0 + 30.0), 1.0)
    anri.io.simulate_sparse(sparse, entries, rings8["hkls"], np.ones(len(rings8["hkls"])), geom_r, omega, dty, (2048, 2048))
    cell = {"cell__a": a, "cell__b": a, "cell__c": a, "cell_alpha": 90.0, "cell_beta": 90.0, "cell_gamma": 90.0, "cell_lattice_[P,A,B,C,I,F,R]": 225}
    parfile = anri.io.write_pars(os.path.join(out, "pars"), pars, {"316L": cell})
    anri.io.write_dataset(sparse, out, "phantom", "am316l", y0=y0, parfile=parfile)
    log("rendered")
ds = anri.io.read_dataset(os.path.join(out, "phantom", "phantom_am316l", "phantom_am316l_dataset.h5"))
geo, _, _ = anri.io.read_pars_json(ds["parfile"])
geom = anri.io.geom_from_pars(geo, ds["y0"], wl * 2e-3 / 2.355, 1.5e-4, 1.5e-4, sig_beam=1.0, voxel_size=1.0)
geom = {k: jnp.asarray(v, jnp.float32) if np.issubdtype(np.asarray(v).dtype, np.floating) else v for k, v in geom.items()}
rings = ix.ring_table(crystal, wl, 6)
chunk = 1 << 20


def stream(groups=None):
    return anri.io.stream_sparse(sparse, ds["ybinedges"], ds["omegamotor"], ds["dtymotor"], chunk, groups)


off, hw = ix.ring_profile(stream(["16.1", "31.1", "46.1"]), geom, rings["tth"], chunk)
rings["hw"] = hw
tth_tol = np.abs(off) + hw
om0, ostep, nk = float(ds["obinedges"][0]), float(np.median(np.diff(ds["obinedges"]))), len(ds["ybincens"])
lit_bins, bins = (0.5, 0.25, 720, 720), (1.0, 1.0, 360, 180)
H_lit = ix.histogram_pixels(stream(), geom, rings["tth"], tth_tol, om0, lit_bins, 1, chunk).reshape(6, 720, 720)
H = ix.histogram_pixels(stream(), geom, rings["tth"], tth_tol, om0, bins, nk, chunk)
med = float(jnp.median(H_lit[H_lit > 0]))
lit = {"table": ix.lit_table(H_lit > med), "om0": om0, "bins": lit_bins, "frame_step": ostep, "etacut": 0.2}
step = 2.0
U_grid, delta = anri.crystal.orientation_grid(step, ops)
_, comp, info = ix.prune(U_grid, delta, B, rings, geom, lit)
pre = np.flatnonzero(comp > info["chance"])
g, lr = ix.orientation_mlem(np.asarray(H).reshape(-1, nk).sum(1), U_grid[pre], B, rings, geom, (*bins, om0), log=log)
U_kept = U_grid[pre[lr > 25]]
pred = ix.predictions(U_kept, B, rings, geom, 0.2)
_, pad = anri.geom.sino_shift_and_pad(ds["y0"], nk, float(ds["ybincens"][0]), 1.0)
nr = nk + pad
pos = np.asarray(anri.geom.recon_positions(nr, 1.0), np.float32)
scan = {"y0": ds["y0"], "dty0": float(ds["ybincens"][0]), "ystep": 1.0, "n_rows": nk, "om0": om0}
f, cand = ix.fit_occupancy(H, pred, rings["ring_j"], pos, scan, bins, k=64, n_iter=10, log=log)
frac, U_pop, spread, _ = ix.populations(f, cand, U_kept, ops, 1.8 * step)
tot = f.sum(1)
occupied = tot > 0.2 * np.percentile(tot, 99)
present = (frac >= 0.1) & occupied[:, None]
present[:, 0] = occupied
tp = anri.io.entries_from_tensormap(truth)
np.savez(os.path.join(out, "stage2.npz"), f=f, cand=cand, U_kept=U_kept, frac=frac, U_pop=U_pop, spread=spread,
         present=present, occupied=occupied, pos=pos, nr=nr, scan_y0=ds["y0"], dty0=float(ds["ybincens"][0]), nk=nk,
         om0=om0, ostep=ostep, tth_tol=tth_tol, hw=hw, B=B, ops=ops, step=step, delta=delta,
         truth_U=np.linalg.inv(tp["ubi"]) @ np.linalg.inv(B), truth_pos=tp["pos"])  # fmt: skip
np.save(os.path.join(out, "H.npy"), np.asarray(H))
near = np.argmin(np.linalg.norm(tp["pos"][:, None, :2] - pos[None, :, :2], axis=2), 1)
U_true = np.linalg.inv(tp["ubi"]) @ np.linalg.inv(B)
err = np.stack([anri.crystal.disorientation(U_pop[near, p], U_true, ops) for p in range(4)], 1)
err = np.where(present[near], err, np.inf)
log(f"stage 2: {len(U_kept)} orientations; main within 1 deg {np.mean(err[:, 0] < 1) * 100:.1f}%, within 0.5 deg "
    f"{np.mean(err[:, 0] < 0.5) * 100:.1f}%, median {np.median(err[:, 0]):.2f}; closest within 1 deg {np.mean(err.min(1) < 1) * 100:.1f}%")
