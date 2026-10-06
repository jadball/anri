"""Render the am316l test phantom as an ImageD11 dataset, for the indexer's tests.

    python render_phantom.py <out> [--step S] [--beam FWHM]

Writes <out>/sparse.h5, <out>/pars and <out>/phantom/phantom_am316l/ (the DataSet). Then, for example:

    python -m anri.index <out> phantom am316l --grid 1 --outdir <out>/index
    python compare_truth.py <repo>/tests/data/phantoms/am316l/am316l_tmap.h5 <out>/index/phantom_am316l_index.npz

The phantom (tests/data/phantoms/am316l): grains with ~1.5 um cells misoriented by a few tenths of a degree, and a
twinned grain, on 0.5 um voxels. By default it is scanned on its own grid (dty steps of one phantom voxel, a beam
that wide), so the indexer's voxels are the phantom's and every comparison with the truth is voxel to voxel.
1800 frames of 0.1 deg.
"""

import argparse
import os

import anri.utils

anri.utils.setup()
import numpy as np  # noqa: E402
from ImageD11.sinograms.tensor_map import TensorMap  # noqa: E402

import anri.index as ix  # noqa: E402
import anri.io  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--step", type=float, help="dty step (default: the phantom's voxel)")
p.add_argument("--beam", type=float, help="beam FWHM (default: the step)")
args = p.parse_args()
out = args.out
os.makedirs(out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
truth = TensorMap.from_h5(os.path.join(here, "../../../tests/data/phantoms/am316l/am316l_tmap.h5"))
vox = float(truth.steps[1])
step = args.step or vox
beam = args.beam or step
a = truth.phases[0].lattice_parameters[0]
lpars = np.array([a, a, a, 90.0, 90.0, 90.0])
wl = 0.2843
pars = {
    "y_center": 1023.5, "y_size": 75.0, "tilt_y": 1e-3, "z_center": 1023.5, "z_size": 75.0, "tilt_z": -2e-3,
    "tilt_x": 0.0, "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1, "wavelength": wl, "wedge": 0.0,
    "chi": 0.0,
}  # fmt: skip
y0 = 0.3
sparse = os.path.join(out, "sparse.h5")
geom = anri.io.geom_from_pars(
    pars, y0, wl * 2e-4 / 2.355, 5e-5, 5e-5, sig_beam=beam / 2.355, voxel_size=vox, sig_psf=0.5
)
entries = anri.io.entries_from_tensormap(truth)
entries["density"] = np.full(len(entries["pos"]), 30.0)
rings8 = ix.ring_table(lpars, 225, wl, 8)
# rows so that the indexer's grid (rows + its 2 padding voxels) is the phantom's own: n x n voxels, same centres
n = truth.UBI.shape[1]
half = 0.5 * (n - 3) * vox
omega, dty = anri.io.motor_grid((0.0, 180.0), 0.1, (y0 - half, y0 + half), step)
anri.io.simulate_sparse(sparse, entries, rings8["hkls"], np.ones(len(rings8["hkls"])), geom, omega, dty, (2048, 2048))
cell = {"cell__a": a, "cell__b": a, "cell__c": a, "cell_alpha": 90.0, "cell_beta": 90.0, "cell_gamma": 90.0,
        "cell_lattice_[P,A,B,C,I,F,R]": 225}  # fmt: skip
parfile = anri.io.write_pars(os.path.join(out, "pars"), pars, {"316L": cell})
anri.io.write_dataset(sparse, out, "phantom", "am316l", y0=y0, parfile=parfile)
print(f"-> {out}: {len(np.unique(dty))} rows of {step:g}, beam FWHM {beam:g}; phantom {n} x {n} voxels of {vox:g}")
