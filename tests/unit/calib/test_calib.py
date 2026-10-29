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


if __name__ == "__main__":
    unittest.main()
