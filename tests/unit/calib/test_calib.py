import unittest

import jax
import jax.numpy as jnp
import numpy as np

from anri.calib import fit
from anri.crystal.lattice import lpars_to_B
from anri.maths.rotation import random_rotations

jax.config.update("jax_enable_x64", True)

TRUE = {"y_center": 1030.0, "y_size": 75.0, "tilt_y": -0.0015, "z_center": 1110.0, "z_size": 75.0, "tilt_z": 0.002,
        "tilt_x": -0.0019, "distance": 151000.0, "o11": -1.0, "o12": 0.0, "o21": 0.0, "o22": -1.0,
        "wavelength": 0.19, "wedge": 0.0, "chi": 0.0}  # fmt: skip
FREE = ("distance", "y_center", "z_center", "tilt_x", "tilt_y", "tilt_z", "wavelength")


def _crystal() -> tuple:
    """A silicon-like crystal at a random orientation, its reflections and its peaks seen from the rotation axis."""
    B = np.asarray(lpars_to_B(np.array([5.43, 5.43, 5.43, 90.0, 90.0, 90.0])))
    U = random_rotations(1, np.random.default_rng(3))[0]
    ubi = np.linalg.inv(U @ B)
    h = np.stack(np.meshgrid(*[np.arange(-5, 6)] * 3, indexing="ij"), -1).reshape(-1, 3)
    fcc = np.all(h % 2 == h[:, :1] % 2, 1) & np.any(h != 0, 1)
    h = h[fcc & (np.linalg.norm(h @ np.linalg.inv(ubi).T, axis=1) < 1.2)].astype(float)
    pars = fit.base_pars(TRUE, y0=0.0)
    c, ok = fit._predict(jnp.asarray(np.repeat(ubi[None], len(h), 0)), jnp.zeros((len(h), 3)), jnp.asarray(h),
                         jnp.zeros(len(h)), {k: jnp.float64(v) for k, v in pars.items()})  # fmt: skip
    c = np.asarray(c).reshape(-1, 3)[np.repeat(np.asarray(ok), 2)]
    c = c[(c[:, 0] > 0) & (c[:, 0] < 2000) & (c[:, 1] > 0) & (c[:, 1] < 2000)]
    rng = np.random.default_rng(4)
    c = c + rng.normal(size=c.shape) * (0.05, 0.05, 0.002)  # centroid noise: pixels, pixels, degrees
    peaks = {"sc": c[:, 0], "fc": c[:, 1], "omega": c[:, 2], "dty": np.zeros(len(c)), "row": np.zeros(len(c), int)}
    return B, U, ubi, peaks, pars


class TestCalibrate(unittest.TestCase):
    def test_recovers_geometry(self):
        """From a wrong geometry and a turned crystal, the fit returns to the true geometry within its error bars."""
        B, _, ubi, peaks, true = _crystal()
        grid_o = np.full((9, 1), -1, np.int32)
        grid_o[4] = 0  # one voxel, on the rotation axis, of a 3 x 3 grid
        grid_w = (grid_o >= 0).astype(float)
        start = dict(true, distance=true["distance"] * 1.005, wavelength=true["wavelength"] * 0.995,
                     y_center=true["y_center"] + 1.0, z_center=true["z_center"] - 1.0, tilt_x=0.0,
                     tilt_y=-0.0005, tilt_z=0.003)  # fmt: skip
        turned = np.linalg.inv(np.asarray(jax.vmap(fit._deform)(jnp.asarray(ubi[None]),
                                                                    jnp.asarray([[1e-3, 0.0, 0.0]]))))  # fmt: skip
        res = fit.calibrate(peaks, np.linalg.inv(turned), grid_o, grid_w, 10.0, start, FREE,
                            tols=(0.1, 0.05, 0.025), log=lambda _: None)  # fmt: skip
        self.assertGreater(len(res["matched"]["index"]), 0.9 * len(peaks["sc"]))
        err = np.sqrt(np.diag(res["cov"]))
        for k, e in zip(FREE, err):
            self.assertLess(abs(res["pars"][k] - true[k]), 5 * e, k)
        self.assertLess(abs(res["pars"]["wavelength"] / true["wavelength"] - 1), 1e-4)

        st = fit.strain_left(peaks, res["matched"], res["ubi"], B, res["pars"], res["sig"], log=lambda _: None)
        self.assertLess(np.abs(st["strain"][0]).max(), 1e-4)

    def test_lattice_and_wavelength_not_both(self):
        _, _, ubi, peaks, true = _crystal()
        with self.assertRaises(ValueError):
            fit.calibrate(peaks, ubi[None], np.zeros((1, 1), np.int32), np.ones((1, 1)), 10.0, true,
                          ("wavelength", "lattice"))  # fmt: skip

    def test_lattice_strain(self):
        """lattice_strain recovers the strain and orientation a UBI was made from."""
        B = np.asarray(lpars_to_B(np.array([4.0, 5.0, 6.0, 90.0, 95.0, 90.0])))
        U = random_rotations(2, np.random.default_rng(0))
        E = np.array([[[1e-3, 2e-4, 0.0], [2e-4, -5e-4, 1e-4], [0.0, 1e-4, 3e-4]], np.zeros((3, 3))])
        ubi = np.linalg.inv(U @ B) @ (np.eye(3) + E)
        E_fit, U_fit = fit.lattice_strain(ubi, B)
        np.testing.assert_allclose(E_fit, E, atol=1e-12)
        np.testing.assert_allclose(U_fit, U, atol=1e-12)


class TestDiagnostics(unittest.TestCase):
    def test_split_strain(self):
        """The parts add back to the strain; unfitted orientations are left out of the peak-weighted means."""
        rng = np.random.default_rng(0)
        E = rng.normal(size=(3, 3, 3)) * 1e-4
        E = (E + E.transpose(0, 2, 1)) / 2
        E[2] = np.nan
        s = fit.split_strain(E, np.array([10, 30, 50]))
        np.testing.assert_allclose(s["deviatoric"][:2] + s["hydrostatic"][:2, None, None] * np.eye(3), E[:2])
        np.testing.assert_allclose(np.trace(s["deviatoric"][:2], axis1=1, axis2=2), 0, atol=1e-18)
        self.assertEqual(s["fitted"].tolist(), [True, True, False])
        self.assertAlmostEqual(s["mean_hydrostatic"], (10 * s["hydrostatic"][0] + 30 * s["hydrostatic"][1]) / 40)
        self.assertTrue(np.isnan(fit.split_strain(E[2:], np.array([5]))["mean_hydrostatic"]))

    def test_occupancy_kept(self):
        """occupancy_kept applies the stage's cut: a voxel keeps its share above min_occ x the largest."""
        from anri.calib.diagnostics import occupancy_kept

        f = np.array([[1.0, 0.001], [0.5, 0.5], [0.0, 0.0], [0.004, 0.0]])
        k = occupancy_kept({"f": f, "nr": 2}, 0.01)
        np.testing.assert_allclose(k["share"].ravel(), [1 / 1.001, 1.0, np.nan, 0.0])
        self.assertEqual((k["pairs_kept"], k["pairs"], k["voxels"]), (3, 5, 2))

    def test_residual_trends(self):
        """Residuals over sig, binned against omega, azimuth (about the fitted centre) and dty."""
        from anri.calib.diagnostics import residual_trends

        n = 400
        rng = np.random.default_rng(1)
        omega = np.linspace(0, 180, n)
        mt = {"residual": np.stack([np.full(n, 0.2), omega / 180 * 0.4, np.zeros(n)], 1), "omega": omega,
              "dty": rng.uniform(-5, 5, n), "sc": np.full(n, 1100.0), "fc": np.full(n, 1000.0)}  # fmt: skip
        tr = residual_trends({"matched": mt, "sig": np.array([0.1, 0.2, 1.0]),
                              "geometry": {"z_center": 1000.0, "y_center": 1000.0}}, n_bins=4)  # fmt: skip
        np.testing.assert_allclose(tr["r"][:, 0], 2.0)
        np.testing.assert_allclose(tr["against"]["azimuth"], 90.0)
        # a residual of 0.4 * omega / 180 over sig 0.2: omega / 90
        np.testing.assert_allclose(tr["medians"]["omega"][:, 1], tr["centres"]["omega"] / 90, atol=1e-2)


class TestRunCommand(unittest.TestCase):
    def test_parses_back(self):
        """run_command gives every option a calibration recorded, and the command line parses each one back."""
        import inspect
        import os
        import shlex
        import tempfile

        import h5py

        from anri.calib.__main__ import parse_args
        from anri.calib.stages import run_command, stage_fit
        from anri.io.provenance import write_provenance

        values = {"peaks": "/data/a b/pks.h5", "lattice": [5.43, 5.43, 5.43, 90.0, 90.0, 90.0],
                  "free": ["distance", "y_center", "lattice"], "tols": [0.2, 0.05], "n_iter": 7, "min_occ": 0.05}  # fmt: skip
        params = inspect.signature(stage_fit).parameters
        opts = {k: values.get(k, p.default) for k, p in params.items()
                if p.kind == p.KEYWORD_ONLY and k not in ("command", "log")}  # fmt: skip
        opts = {k: list(v) if isinstance(v, tuple) else v for k, v in opts.items()}  # as stage_fit records them
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "x_fit.h5")
            with h5py.File(source, "w") as h:
                write_provenance(h, "fit", None, {}, {})
            for name, out in (("x_calib.h5", None), ("other.h5", "other.h5")):
                path = os.path.join(tmp, name)
                with h5py.File(path, "w") as h:
                    write_provenance(h, "calib", source, opts, {})
                args = parse_args(shlex.split(run_command(path))[3:])
                self.assertEqual(args.command, "fit")
                self.assertEqual(args.source, source)
                self.assertEqual(args.out, None if out is None else path)
                self.assertEqual(args.peaks, opts["peaks"])
                self.assertEqual(args.lattice, opts["lattice"])
                self.assertEqual(args.free.split(","), opts["free"])
                self.assertEqual([float(t) for t in args.tols.split(",")], opts["tols"])
                for k in ("n_iter", "clip", "min_occ", "min_peaks"):
                    self.assertEqual(getattr(args, {"n_iter": "iter"}.get(k, k)), opts[k], msg=k)

    def test_rings_parses_back(self):
        """run_command gives every option a rings file recorded, and the command line parses each one back."""
        import inspect
        import os
        import shlex
        import tempfile

        import h5py

        from anri.calib.__main__ import parse_args
        from anri.calib.stages import run_command, stage_rings
        from anri.io.provenance import write_provenance

        values = {"parfile": "start/pars.json", "phase": "Si", "detector_mask": "/m/mask.edf",
                  "spatial": ["/m/e2dx.edf", "/m/e2dy.edf"], "scans": 5, "threshold": 2.0, "free": ["distance"],
                  "tols": [0.3, 0.1], "n_iter": 12, "n_eta": 180, "clip": 3.0, "merge": 2e-4, "search": False,
                  "max_shift": 50.0, "span": [0.8, 1.2]}  # fmt: skip
        params = inspect.signature(stage_rings).parameters
        opts = {k: values.get(k, p.default) for k, p in params.items()
                if p.kind == p.KEYWORD_ONLY and k not in ("command", "log")}  # fmt: skip
        opts = {k: list(v) if isinstance(v, tuple) else v for k, v in opts.items()}  # as stage_rings records them
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "si_rings.h5")
            with h5py.File(path, "w") as h:
                write_provenance(h, "rings", None, {"source": "si.edf", **opts},
                                 {"dataset_file": "", "image": "/x/si.edf", "parfile": "/x/start/pars.json"})  # fmt: skip
            args = parse_args(shlex.split(run_command(path))[3:])
        self.assertEqual((args.command, args.source, args.out), ("rings", "/x/si.edf", path))
        self.assertEqual(args.parfile, "/x/start/pars.json")  # as resolved: absolute
        self.assertEqual((args.mask, args.spatial, args.span), (opts["detector_mask"], opts["spatial"], opts["span"]))
        self.assertEqual(args.free.split(","), opts["free"])
        self.assertEqual([float(t) for t in args.tols.split(",")], opts["tols"])
        self.assertIs(args.no_search, True)
        for k in ("phase", "scans", "threshold", "n_iter", "n_eta", "clip", "merge", "max_shift"):
            self.assertEqual(getattr(args, {"n_iter": "iter"}.get(k, k)), opts[k], msg=k)


class TestChain(unittest.TestCase):
    def test_rings_to_calibration(self):
        """pars.json names its calibration; the histogram stage links it; commands remakes the whole chain."""
        import os
        import shlex
        import tempfile

        import h5py

        import anri.calib.__main__ as calib_cli
        import anri.index.__main__ as index_cli
        import anri.io.imaged11 as io
        from anri.calib.stages import commands, write_pars
        from anri.index.stages import STAGES, _pars_source
        from anri.io.provenance import chain, read_provenance, write_provenance

        cell = {"cell__a": 5.43, "cell__b": 5.43, "cell__c": 5.43, "cell_alpha": 90.0, "cell_beta": 90.0,
                "cell_gamma": 90.0, "cell_lattice_[P,A,B,C,I,F,R]": 227}  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            start = io.write_pars(os.path.join(tmp, "start"), TRUE, {"Si": cell})
            self.assertIsNone(io.read_pars_source(start))  # written from no stage file
            rings_file = os.path.join(tmp, "si_x_rings.h5")
            with h5py.File(rings_file, "w") as h:
                write_provenance(h, "rings", None, {"source": "x.h5", "tols": [0.2, 0.1], "span": [0.7, 1.4]},
                                 {"dataset_file": "/d/si_x_dataset.h5", "image": "", "parfile": start, "phase": "Si",
                                  "lattice": [5.43] * 3 + [90.0] * 3})  # fmt: skip
                g = h.create_group("geometry")
                for k, v in TRUE.items():
                    g.attrs[k] = v
                g.attrs["lattice_scale"] = 1.0
            parfile = write_pars(rings_file, os.path.join(tmp, "rings_pars"))
            self.assertEqual(io.read_pars_source(parfile)["file"], os.path.abspath(rings_file))
            self.assertEqual(io.read_pars_json(parfile)[1], "Si")  # ImageD11's entries are unchanged
            logged = []
            self.assertEqual(_pars_source(parfile, logged.append), os.path.abspath(rings_file))
            # the indexer's chain, its histogram made from those parameters, then the calibration on its fit
            source = _pars_source(parfile, logged.append)
            first = {"analysisroot": "/d", "sample": "si", "dataset": "x", "parfile": parfile}
            for stage in STAGES[:4]:
                path = os.path.join(tmp, f"si_x_{stage}.h5")
                with h5py.File(path, "w") as h:
                    if stage == "histogram":
                        write_provenance(h, stage, source, first, {"parfile": parfile})
                    else:
                        write_provenance(h, stage, source, {}, {})
                source = path
            calib_file = os.path.join(tmp, "si_x_calib.h5")
            with h5py.File(calib_file, "w") as h:
                write_provenance(h, "calib", source, {"tols": [0.1, 0.05]}, {})
            self.assertEqual(chain(calib_file)[-1], os.path.abspath(rings_file))
            cmds = commands(calib_file)
            self.assertEqual(len(cmds), 4)
            rings = calib_cli.parse_args(shlex.split(cmds[0])[3:])
            self.assertEqual((rings.command, rings.source, rings.out), ("rings", "/d/si_x_dataset.h5", rings_file))
            self.assertEqual(shlex.split(cmds[1])[3:], ["pars", rings_file, os.path.dirname(parfile)])
            run = index_cli.parse_args(shlex.split(cmds[2])[3:])  # the indexer's options only, not the rings'
            self.assertEqual((run.command, run.parfile, run.outdir, run.levels), ("run", parfile, tmp, 0))
            fit_cmd = calib_cli.parse_args(shlex.split(cmds[3])[3:])
            self.assertEqual((fit_cmd.command, fit_cmd.source), ("fit", source))
            # remade since the parameters were written: not linked
            with h5py.File(rings_file, "w") as h:
                write_provenance(h, "rings", None, {}, {})
            self.assertIsNone(_pars_source(parfile, logged.append))
            self.assertIn("remade since", logged[-1])
            os.remove(rings_file)
            self.assertIsNone(_pars_source(parfile, logged.append))
            self.assertIn("not there", logged[-1])
            self.assertEqual(read_provenance(calib_file)["stage"], "calib")


if __name__ == "__main__":
    unittest.main()
