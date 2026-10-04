"""Render a small AM-like 316L phantom as an ImageD11 S3DXRD dataset, to test run_index.py against a known truth.

    python make_phantom.py <outdir> [--beam FWHM] [--radius R] [--ostep DEG]

Writes {outdir}/PROCESSED_DATA/phantom/phantom_am/phantom_am_dataset.h5 and _sparse.h5, pars/ beside PROCESSED_DATA,
and truth.npz (voxel positions, orientations, grain and cell ids). Index it with

    python run_index.py {outdir}/PROCESSED_DATA phantom am

The map: a disk of 1 um voxels split into Voronoi grains with random orientations; each grain into Voronoi cells of
~1.5 um misoriented by ~0.3 deg (rms per axis) from the grain; one grain carries Sigma3 twin lamellae 2 um thick every
5 um. 316L-like FCC at the Tognan wavelength and a 150 mm, 75 um-pixel detector. Counts ~ Poisson.
"""

import argparse
import json
import os
import time

T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f} s] {msg}", flush=True)


p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("outdir")
p.add_argument("--beam", type=float, default=1.4, help="beam FWHM, um (default 1.4)")
p.add_argument("--radius", type=float, default=20.0, help="sample radius, um (default 20)")
p.add_argument("--ostep", type=float, default=0.1, help="frame step, deg (default 0.1)")
p.add_argument("--grains", type=int, default=6)
p.add_argument("--voxel", type=float, default=1.0, help="phantom voxel size, um (default 1; the dty step is 1)")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

import anri.utils  # noqa: E402

anri.utils.setup(n_cpu=8)
import h5py  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from anri.crystal import Crystal, Symmetry, UnitCell  # noqa: E402
from anri.fwd import make_row, render_row  # noqa: E402
from anri.io import geom_from_pars  # noqa: E402

A, SG, WL = 3.5966, 225, 0.2843
DET = (2048, 2048)
pars = {
    "y_center": 1023.5, "y_size": 75.0, "tilt_y": 1e-3, "z_center": 1023.5, "z_size": 75.0, "tilt_z": -2e-3, "tilt_x": 0.0,
    "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1, "wavelength": WL, "wedge": 0.0, "chi": 0.0,
    "omegasign": 1.0, "t_x": 0.0, "t_y": 0.0, "t_z": 0.0,
}  # fmt: skip
Y0, YSTEP = 0.3, 1.0  # rotation axis slightly off the middle row
NK = 2 * int(np.ceil(args.radius + 3 * args.beam)) + 1
dty = Y0 + (np.arange(NK) - NK // 2) * YSTEP
omega = np.arange(0.0, 180.0, args.ostep) + args.ostep / 2
rng = np.random.default_rng(args.seed)

# ------------------------------------------------------------------------------------------------- map
g1 = np.arange(-np.ceil(args.radius), np.ceil(args.radius) + args.voxel / 2, args.voxel)
xy = np.stack(np.meshgrid(g1, g1, indexing="ij"), -1).reshape(-1, 2)
xy = xy[np.linalg.norm(xy, axis=1) <= args.radius]
nv = len(xy)
seeds = rng.uniform(-args.radius, args.radius, (args.grains, 2))
grain = np.argmin(np.linalg.norm(xy[:, None] - seeds[None], axis=2), 1)
U_g = Rotation.random(args.grains, random_state=args.seed).as_matrix()
n_cells = int(np.pi * args.radius**2 / 1.5**2)  # ~1.5 um cells
cseeds = rng.uniform(-args.radius, args.radius, (n_cells, 2))
cell = np.argmin(np.linalg.norm(xy[:, None] - cseeds[None], axis=2), 1)
dU_c = Rotation.from_rotvec(np.radians(0.3) * rng.normal(size=(n_cells, 3))).as_matrix()
U = dU_c[cell] @ U_g[grain]  # cell misorientation, sample frame
# Sigma3 twin lamellae in grain 0: 60 deg about a <111> of that grain, lamella normal along that <111>
n111 = U_g[0] @ (np.array([1.0, 1.0, 1.0]) / np.sqrt(3))
twin = (grain == 0) & (np.mod(xy @ n111[:2] / max(np.linalg.norm(n111[:2]), 1e-6), 5.0) < 2.0)
R_tw = Rotation.from_rotvec(np.radians(60.0) * n111).as_matrix()
U[twin] = R_tw @ U[twin]

crystal = Crystal(UnitCell.from_lpars([A, A, A, 90.0, 90.0, 90.0]), Symmetry.from_number(SG))
B = np.asarray(crystal.B)
crystal.make_hkls(2 * np.sin(np.radians(10.6)) / WL, WL)  # 2theta < 21 deg
h_all = np.asarray(crystal.allhkls, np.float64)
h_all = h_all[np.all(np.mod(h_all, 2) == np.mod(h_all[:, :1], 2), 1)]  # FCC: h, k, l unmixed
tth = np.degrees(2 * np.arcsin(WL * np.linalg.norm(h_all @ B.T, axis=1) / 2))
hkls = h_all[tth < 21.0]
entries = {"ubi": np.linalg.inv(U @ B), "pos": np.column_stack([xy, np.zeros(nv)]), "density": np.full(nv, args.voxel**2)}
geom = geom_from_pars(pars, Y0, WL * 2e-4 / 2.355, 5e-5, 5e-5, sig_beam=args.beam / 2.355, voxel_size=args.voxel, sig_psf=0.5)
log(f"{nv} voxels, {args.grains} grains ({twin.sum()} twin voxels), {n_cells} cells; {len(hkls)} hkls; "
    f"{NK} rows x {len(omega)} frames; beam FWHM {args.beam}")

# ------------------------------------------------------------------------------------------------- files
root = os.path.abspath(args.outdir)
dsdir = os.path.join(root, "PROCESSED_DATA", "phantom", "phantom_am")
pdir = os.path.join(root, "pars")
os.makedirs(dsdir, exist_ok=True)
os.makedirs(pdir, exist_ok=True)
with open(os.path.join(pdir, "geometry.par"), "w") as f:
    f.writelines(f"{k} {v}\n" for k, v in pars.items())
with open(os.path.join(pdir, "316L.par"), "w") as f:
    f.write(f"cell__a {A}\ncell__b {A}\ncell__c {A}\ncell_alpha 90.0\ncell_beta 90.0\ncell_gamma 90.0\n")
    f.write(f"cell_lattice_[P,A,B,C,I,F,R] {SG}\n")
with open(os.path.join(pdir, "pars.json"), "w") as f:
    json.dump({"geometry": {"file": "geometry.par"}, "phases": {"316L": {"file": "316L.par"}}}, f)
np.savez(os.path.join(root, "truth.npz"), pos=entries["pos"], U=U, grain=grain, cell=cell, twin=twin, B=B,
         beam=args.beam, y0=Y0)  # fmt: skip
with h5py.File(os.path.join(dsdir, "phantom_am_dataset.h5"), "w") as h:
    h.attrs.update({"y0": Y0, "dtymotor": "dty", "omegamotor": "rot_center", "parfile": os.path.join(pdir, "pars.json")})
    h["ybincens"] = dty
    h["ybinedges"] = np.concatenate([dty - YSTEP / 2, dty[-1:] + YSTEP / 2])
    h["obinedges"] = np.concatenate([omega - args.ostep / 2, omega[-1:] + args.ostep / 2])

scale = 2000.0  # counts per unit rendered intensity, roughly
with h5py.File(os.path.join(dsdir, "phantom_am_sparse.h5"), "w") as h:
    for k in range(NK):
        row = make_row(omega, np.full_like(omega, dty[k]))
        fr, px, val, st = render_row(entries, hkls, np.ones(len(hkls)), geom, row, DET, min_value=1e-4)
        counts = rng.poisson(val * scale)
        m = counts > 0
        fr, px, counts = fr[m], px[m], counts[m]
        gr = h.create_group(f"{k + 1}.1")
        gr["row"] = (px // DET[1]).astype(np.uint16)
        gr["col"] = (px % DET[1]).astype(np.uint16)
        gr["intensity"] = counts.astype(np.float32)
        gr["nnz"] = np.bincount(fr, minlength=len(omega)).astype(np.int32)
        gr["measurement/rot_center"] = omega
        gr["measurement/dty"] = np.full(len(omega), dty[k])
        if k % 10 == 0 or k == NK - 1:
            log(f"row {k + 1}/{NK} (dty {dty[k]:+.1f}): {st['n_peaks']} peaks, {m.sum() / 1e6:.2f}M pixels")
log(f"-> {dsdir}")
