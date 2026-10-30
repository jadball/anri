import os
import shlex
import tempfile
import unittest

import h5py
import numpy as np

import anri.crystal
import anri.geom
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


class TestCommandLine(unittest.TestCase):
    def test_run_dataset_names(self):
        # run's positional output dataset and --dataset (an input DataSet to simulate) are separate (issue #25)
        from anri.simulate.__main__ import parse_args

        args = parse_args(["run", "map.npz", "geometry.par", "results", "sample", "out_name"])
        self.assertEqual((args.dataset, args.input_dataset), ("out_name", None))
        args = parse_args(["run", "map.npz", "geometry.par", "results", "sample", "out_name", "--dataset", "in.h5"])
        self.assertEqual((args.dataset, args.input_dataset), ("out_name", "in.h5"))
        self.assertEqual(parse_args(["prepare", "map.npz", "--dataset", "in.h5"]).input_dataset, "in.h5")


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
            with self.assertRaises(ValueError):  # an output that is the source is refused, the source kept (#26)
                sim.stages.stage_render(prep, out=prep, log=lambda m: None)
            self.assertEqual(sim.stages.load_prepare(prep)["names"], ["Fe", "Ni"])
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
            # each stage keeps its log, though nothing was printed
            for f in (prep, a, out):
                self.assertIn(f"-> {f}", anri.io.provenance.read_provenance(f)["log"][-1])
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
            # (into a folder that does not exist yet: the stage makes it)
            again = sim.stages.load_prepare(sim.stages.stage_prepare(map_file, None,
                                                                     os.path.join(tmp, "new", "y_prepare.h5"),
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


class TestGrainMap(unittest.TestCase):
    def test_prepare_and_render(self):
        """A grain map .npz (grain, grain_ubi) goes through prepare and render with its grains, absorption included."""
        from anri.fwd.render import grain_entries

        lp = np.array([2.8665] * 3 + [90.0] * 3)
        U = anri.maths.rotation.random_rotations(2, np.random.default_rng(3))
        ubi = np.linalg.inv(np.asarray(U) @ anri.crystal.structure.B_matrix(lp))
        labels = np.zeros((4, 4), int)
        labels[:, 2:] = 1
        labels[0, 0] = -1
        with tempfile.TemporaryDirectory() as tmp:
            parfile = os.path.join(tmp, "geometry.par")
            anri.io.imaged11.write_par(parfile, PARS)
            map_file = os.path.join(tmp, "grains.npz")
            np.savez(map_file, **grain_entries(labels, ubi, 2.0), voxel_size=2.0, lattice=lp, space_group=229,
                     phase="Fe")  # fmt: skip
            prep = sim.stages.stage_prepare(map_file, parfile, os.path.join(tmp, "g_prepare.h5"), omega_step=2.0,
                                            dty_step=8.0, margin=0.25, mu=0.01, log=lambda m: None)  # fmt: skip
            entries = sim.stages.load_prepare(prep)["phases"][0][0]
            self.assertEqual(sorted(entries), ["density", "grain", "grain_ubi", "optical_depth", "pos"])
            np.testing.assert_array_equal(entries["grain"], labels[labels >= 0])
            np.testing.assert_allclose(entries["grain_ubi"], ubi)
            out = sim.stages.stage_render(prep, rows="1:2", log=lambda m: None)
            self.assertGreater(anri.io.provenance.read_provenance(out)["results"]["pixels"], 0)


class TestAbsorption(unittest.TestCase):
    def test_mu_per_phase_and_map(self):
        """mu per phase (built from the map) or a mu map file: the same optical depths for the same sample; absorption
        lowers the intensities, and the command line that remakes it keeps the option."""
        from anri.fwd.render import make_row, render_phases

        quiet = {"omega_step": 2.0, "dty_step": 8.0, "margin": 0.25, "log": lambda m: None}
        with tempfile.TemporaryDirectory() as tmp:
            parfile = os.path.join(tmp, "geometry.par")
            anri.io.imaged11.write_par(parfile, PARS)
            map_file = two_phase_map(tmp)
            plain = sim.stages.load_prepare(sim.stages.stage_prepare(
                map_file, parfile, os.path.join(tmp, "a_prepare.h5"), absorption=False, **quiet))  # fmt: skip
            self.assertNotIn("optical_depth", plain["phases"][0][0])
            per = sim.stages.load_prepare(sim.stages.stage_prepare(
                map_file, parfile, os.path.join(tmp, "b_prepare.h5"), mu={"Fe": 0.05, "Ni": 0.05}, **quiet))  # fmt: skip
            self.assertEqual(per["prov"]["resolved"]["mu"], ["Fe=0.05", "Ni=0.05"])
            pos = np.concatenate([e["pos"] for e, _, _ in per["phases"]])
            np.save(os.path.join(tmp, "mu.npy"), anri.geom.paths.attenuation_map(pos, np.full(len(pos), 0.05),
                                                                                  np.ones(len(pos)), 2.0))  # fmt: skip
            given = sim.stages.load_prepare(sim.stages.stage_prepare(
                map_file, parfile, os.path.join(tmp, "c_prepare.h5"), mu_map=os.path.join(tmp, "mu.npy"), **quiet))  # fmt: skip
            for (a, _, _), (b, _, _) in zip(per["phases"], given["phases"]):
                np.testing.assert_allclose(a["optical_depth"], b["optical_depth"], atol=1e-6)
            self.assertGreater(per["phases"][0][0]["optical_depth"].max(), 0.1)
            row = make_row(plain["omega"][1], plain["dty"][1])
            totals = [render_phases(p["phases"], p["geom"], row, p["det_shape"], min_value=0.0, batch=1024)[2].sum()
                      for p in (plain, per)]  # fmt: skip
            self.assertLess(totals[1], 0.95 * totals[0])
            self.assertGreater(totals[1], np.exp(-0.05 * 2 * 12) * totals[0])  # paths under 2 x (8 / cos 2theta)
            from anri.simulate.__main__ import _mu, parse_args

            args = parse_args(["prepare", "m.h5", "p.par", "--mu", "Fe=0.05", "Ni=0.05", "--no-absorption"])
            self.assertEqual((_mu(args.mu), args.no_absorption), ({"Fe": 0.05, "Ni": 0.05}, True))


if __name__ == "__main__":
    unittest.main()
