import os
import shlex
import tempfile
import unittest

import h5py
import numpy as np

import anri.crystal
import anri.io
import anri.maths.rotation
import anri.simulate as sim
from anri.simulate import diagnostics as dg

PARS = {
    "y_center": 1049.9, "y_size": 75.0, "tilt_y": 0.0, "z_center": 1116.5, "z_size": 75.0, "tilt_z": 0.0,
    "tilt_x": 0.0, "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1,
    "wavelength": 12.398419843320026 / 43.0, "wedge": 0.0, "chi": 0.0,
}  # fmt: skip


def two_phase_map(folder):
    """A 4 x 4 TensorMap of 2 um voxels: Fe in one half, Ni in the other."""
    from ImageD11.sinograms.tensor_map import TensorMap
    from ImageD11.unitcell import unitcell

    lp = {0: [2.8665] * 3 + [90.0] * 3, 1: [3.524] * 3 + [90.0] * 3}
    U = anri.maths.rotation.random_rotations(2, np.random.default_rng(3))
    phase = np.full((4, 4), -1)
    phase[:, :2], phase[:, 2:] = 0, 1
    ubi = np.full((4, 4, 3, 3), np.nan)
    for p in (0, 1):
        ubi[phase == p] = np.linalg.inv(np.asarray(U[p]) @ anri.crystal.structure.B_matrix(np.array(lp[p])))
    maps = {"UBI": ubi, "phase_ids": phase}
    tmap = TensorMap(maps={k: TensorMap.recon_order_to_map_order(v) for k, v in maps.items()}, steps=[2.0] * 3)
    tmap.phases = {0: unitcell(lp[0], 229, name="Fe"), 1: unitcell(lp[1], 225, name="Ni")}
    path = os.path.join(folder, "fe_ni.h5")
    tmap.to_h5(path)
    return path


class TestRenderStages(unittest.TestCase):
    def test_prepare_render_assemble(self):
        """Two phases, rendered in two parts and assembled into an ImageD11 dataset with the provenance chain."""
        with tempfile.TemporaryDirectory() as tmp:
            parfile = os.path.join(tmp, "geometry.par")
            anri.io.imaged11.write_par(parfile, PARS)
            map_file = two_phase_map(tmp)
            prep = sim.stages.stage_prepare(map_file, parfile, os.path.join(tmp, "x_prepare.h5"), omega_step=2.0,
                                    dty_step=8.0, margin=0.25, density=50.0, log=lambda m: None)  # fmt: skip
            p = sim.stages.load_prepare(prep)
            self.assertEqual(p["names"], ["Fe", "Ni"])
            self.assertEqual(p["omega"].shape, (3, 90))  # radius 4.2 + 0.25 x 8: one row either side of y0
            self.assertEqual(p["cells"]["Ni"]["cell_lattice_[P,A,B,C,I,F,R]"], 225)
            self.assertEqual(list(sim.stages.parse_rows("1:", 3)), [1, 2])
            a = sim.stages.stage_render(prep, rows=":1", log=lambda m: None)
            self.assertTrue(a.endswith("x_render_0000-0000.h5"))
            with self.assertRaises(ValueError):  # row 1 and 2 missing
                sim.stages.stage_assemble(prep, tmp, "s", "d", log=lambda m: None)
            b = sim.stages.stage_render(prep, rows="1:", log=lambda m: None)
            self.assertEqual(sim.stages.render_parts(prep), [a, b])
            out = sim.stages.stage_assemble(prep, tmp, "s", "d", log=lambda m: None)
            prov = anri.io.provenance.read_provenance(out)
            ds = anri.io.imaged11.read_dataset(prov["resolved"]["dataset_file"])
            self.assertEqual(len(ds["ybincens"]), 3)
            with h5py.File(prov["resolved"]["sparse_file"], "r") as h:
                n = sum(int(h[g]["nnz"][()].sum()) for g in h)
                self.assertEqual(sorted(h), ["1.1", "2.1", "3.1"])
            self.assertEqual(n, prov["results"]["pixels"])
            self.assertGreater(n, 0)
            self.assertEqual(anri.io.provenance.chain(out), [out, prep])
            _, _, cell = anri.io.imaged11.read_pars_json(prov["resolved"]["parfile"], "Ni")
            self.assertEqual(cell["cell_lattice_[P,A,B,C,I,F,R]"], 225)
            # the command line that remakes it parses back to the same options
            from anri.simulate.__main__ import parse_args

            args = parse_args(shlex.split(sim.stages.run_command(out))[3:])
            self.assertEqual((args.command, args.sample, args.omega_step, args.dty_step, args.density),
                             ("run", "s", 2.0, 8.0, 50.0))  # fmt: skip
            self.assertEqual((list(args.det_shape), list(args.window), args.cut), ([2162, 2068], [3, 7, 7], 1.0))
            # diagnostics: the rows reach past the sample, and a row's image holds the pixels written
            fp = dg.footprint(p)
            self.assertGreater(fp["covered"], fp["radius"])
            img = dg.row_image(prov["resolved"]["sparse_file"], 1)
            with h5py.File(prov["resolved"]["sparse_file"], "r") as h:
                self.assertEqual(img["image"].sum(), h["2.1"]["intensity"][()].sum())
            pr = dg.probe_rows(p, [1], cut=1.0, densities=[1.0, 50.0, 1e4], log=lambda m: None)
            self.assertGreater(len(pr["lost_below"]), 0)
            gone = [(pr["lost_below"] > d).mean() for d in pr["densities"]]
            self.assertEqual(gone, sorted(gone, reverse=True))  # fewer peaks gone as the density rises
            self.assertEqual(gone[-1], 0.0)
            self.assertTrue((pr["kept_share"] <= 1 + 1e-9).all() and (np.diff(pr["kept_share"], axis=1) >= 0).all())
            cov = dg.ring_coverage(p, n_tth=50, n_eta=90)
            self.assertEqual(cov["on_detector"][0], 1.0)  # the innermost ring is wholly on the detector
            # the map prepared from the dataset it made, rendered again, matches that dataset's measured row
            dsfile = prov["resolved"]["dataset_file"]
            again = sim.stages.load_prepare(sim.stages.stage_prepare(map_file, None, os.path.join(tmp, "y_prepare.h5"),
                                                     dataset=dsfile, density=50.0, log=lambda m: None))  # fmt: skip
            np.testing.assert_array_equal(again["omega"], p["omega"])
            ds = anri.io.imaged11.read_dataset(dsfile)
            obs, mod = dg.observed_row(ds, 1), dg.model_row(again, 1)
            self.assertAlmostEqual(dg.row_scale(obs, mod), 1.0, delta=0.02)
            npix = 2162 * 2068
            k_obs = obs["frame"].astype(np.int64) * npix + obs["pixel"]
            k_mod = mod["frame"].astype(np.int64) * npix + mod["pixel"]
            self.assertTrue(np.isin(k_obs, k_mod).all())  # every measured pixel is one the model lights
            self.assertEqual(dg.frames_image(obs, range(90)).sum(), obs["value"].sum())


if __name__ == "__main__":
    unittest.main()
