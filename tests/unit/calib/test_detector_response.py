"""anri.calib.detector: the detector's point spread, fitted to spots the renderer makes with a known one."""

import unittest

import jax
import numpy as np

import anri.crystal
import anri.maths.rotation
from anri.calib.detector import find_spots, fit_response, spot_terms
from anri.fwd.render import make_row, render_row
from anri.io.imaged11 import geom_from_pars

jax.config.update("jax_enable_x64", True)

PARS = {
    "y_center": 1049.9, "y_size": 75.0, "tilt_y": -2e-3, "z_center": 1116.5, "z_size": 75.0, "tilt_z": 3e-3,
    "tilt_x": 1e-3, "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1,
    "wavelength": 12.398419843320026 / 43.6, "wedge": 0.0, "chi": 0.0,
}  # fmt: skip
CORE, WING, FRACTION = 0.29, 1.28, 0.32
SENSOR = (111.0, 750.0)  # absorption length, thickness (um): CdTe at 43.6 keV
BANDWIDTH = 1e-3


def rendered_spots(n_max: int = 40, seed: int = 0) -> tuple:
    """Isolated Fe spots of one point voxel, rendered with the core + wing point spread and the sensor, Poisson."""
    rng = np.random.default_rng(seed)
    U = np.asarray(anri.maths.rotation.random_rotations(1, rng)[0])
    ubi = np.linalg.inv(U @ anri.crystal.structure.B_matrix(np.array([2.8665] * 3 + [90.0] * 3)))
    hkls = np.array([h for h in np.ndindex(7, 7, 7) if sum(h) % 2 == 0 and any(h)], float) - 3  # bcc, |h| <= 3
    hkls = hkls[np.abs(hkls).sum(1) > 0]
    geom = geom_from_pars(PARS, 0.0, PARS["wavelength"] * BANDWIDTH / 2.3548, 1e-6, 1e-6, 0.5, 1.0,
                          psf_mix=((1 - FRACTION, CORE), (FRACTION, WING)), sensor=(1 / SENSOR[0], SENSOR[1]))  # fmt: skip
    omega = np.arange(0.025, 180, 0.05)
    shape = (2162, 2068)
    frame, pixel, value, _ = render_row({"ubi": ubi[None], "pos": np.zeros((1, 3)), "density": np.ones(1)}, hkls,
                                        np.full(len(hkls), 4e3), geom, make_row(omega, np.zeros_like(omega)), shape,
                                        window=(5, 13, 13), min_value=1e-3, batch=256)  # fmt: skip
    row, col = pixel // shape[1], pixel % shape[1]
    spots = find_spots(frame, row, col, value, shape, min_counts=500, window=11)
    k = np.argsort(-spots["counts"])[:n_max]
    spots = {key: x[k] for key, x in spots.items()}
    dense = {}
    for f, r, c, v in zip(frame, row, col, value):
        dense[(f, r, c)] = v
    windows = np.zeros((len(k), 11, 11))
    for s, (f, r, c) in enumerate(zip(spots["frame"], spots["row"], spots["col"])):
        for i in range(11):
            for j in range(11):
                windows[s, i, j] = dense.get((f, r - 5 + i, c - 5 + j), 0.0)
    return spots, rng.poisson(windows + 0.05).astype(float)


class TestDetectorResponse(unittest.TestCase):
    def test_recovers_rendered_core_and_wing(self):
        spots, counts = rendered_spots()
        self.assertGreater(len(counts), 20)
        terms = spot_terms(spots["row"], spots["col"], PARS, None, SENSOR, BANDWIDTH, 1e-3)
        fits = fit_response(counts, np.ones(counts.shape, bool), terms, free_radial=False, log=lambda m: None)
        a = fits["all"]
        self.assertAlmostEqual(a["core"], CORE, delta=0.04)
        self.assertAlmostEqual(a["wing"], WING, delta=0.12)
        self.assertAlmostEqual(a["fraction"], FRACTION, delta=0.04)
        self.assertGreater(a["nll_one"] - a["nll_two"], 100)  # one Gaussian is clearly worse


class TestThickness(unittest.TestCase):
    def test_box_width(self):
        """A source's length along the beam: about thickness x tan(2theta) / pixel radially (detector ~ normal)."""
        rows, cols = np.array([700.0, 1600.0, 1116.0]), np.array([1050.0, 900.0, 1700.0])
        t = spot_terms(rows.astype(int), cols.astype(int), PARS, None, None, BANDWIDTH, 1e-3, 100.0)
        np.testing.assert_allclose(t["box"], 100.0 * np.tan(np.radians(t["tth"])) / 75.0, rtol=0.02)

    def test_fit_with_and_without_the_box(self):
        """Spots smeared by a box: the fit that knows it recovers the point spread; one that does not, a wider core."""
        from anri.calib.detector import model, model_data
        from anri.maths.utils import float64

        rng = np.random.default_rng(1)
        n = 30
        rows, cols = rng.integers(300, 1800, n), rng.integers(300, 1800, n)
        t = spot_terms(rows, cols, PARS, None, None, BANDWIDTH, 1e-3, 600.0)  # a ~1 px box
        theta = np.concatenate([[np.log(CORE), np.log(WING), np.log(FRACTION / (1 - FRACTION))],
                                np.tile([np.log(3e4), 0.1, -0.2, np.log(0.6), np.log(0.05), 0.0], n)])  # fmt: skip
        with float64():
            lam = np.asarray(model(theta, model_data(np.zeros((n, 11, 11)), np.ones((n, 11, 11), bool), t), True,
                                   False))  # fmt: skip
        counts = rng.poisson(lam).astype(float)
        good = np.ones(counts.shape, bool)
        known = fit_response(counts, good, t, free_radial=False, log=lambda m: None)["all"]
        blind = fit_response(counts, good, {**t, "box": np.zeros(n)}, free_radial=False, log=lambda m: None)["all"]
        self.assertGreater(np.median(t["box"]), 0.5)
        self.assertAlmostEqual(known["core"], CORE, delta=0.04)
        self.assertAlmostEqual(known["wing"], WING, delta=0.1)
        self.assertAlmostEqual(known["fraction"], FRACTION, delta=0.03)
        self.assertGreater(blind["core"], known["core"] + 0.05)


class TestProfile(unittest.TestCase):
    def test_profile_and_plot(self):
        """detector_profile bins a stage file's windows by radial offset; plot_detector_profile draws it."""
        import os
        import tempfile

        import h5py
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from anri.calib.diagnostics import detector_profile
        from anri.calib.plots import plot_detector_profile

        rng = np.random.default_rng(0)
        off = rng.uniform(-5, 5, (3, 11, 11))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x_detector.h5")
            with h5py.File(path, "w") as h:
                for name in ("response", "response/low", "response/high"):
                    g = h.create_group(name)
                    for k, v in {"one": 0.4, "core": 0.3, "wing": 1.3, "fraction": 0.3, "nll_one": 2.0,
                                 "nll_two": 1.0, "n": 3}.items():  # fmt: skip
                        g.attrs[k] = v
                g = h.create_group("windows")
                g["counts"], g["one"], g["two"] = np.ones((3, 11, 11)), np.full((3, 11, 11), 2.0), np.ones((3, 11, 11))
                g["good"], g["radial_offset"] = np.ones((3, 11, 11), bool), off
            d = detector_profile(path)
        self.assertEqual(d["data"].sum(), np.sum(np.abs(off) < 5))
        np.testing.assert_allclose(d["one"], 2 * d["data"])
        self.assertAlmostEqual(d["response"]["core"], 0.3)
        plot_detector_profile(Figure(), d)


if __name__ == "__main__":
    unittest.main()
