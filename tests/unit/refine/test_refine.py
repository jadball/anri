import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import anri.crystal
from anri.fwd import make_row, render_row
from anri.io import geom_from_pars
from anri.refine import bin_frames, measured, refine, refine_per_entry


def _problem():
    """A 2 x 2 voxel iron grain, its {110} reflections, two dty rows of 0.5 degree frames, broad peaks (small, for CI)."""
    a = 2.8665
    B = np.asarray(anri.crystal.lpars_to_B(np.array([a, a, a, 90.0, 90.0, 90.0])))
    U = Rotation.from_euler("xyz", [12.0, 34.0, 56.0], degrees=True).as_matrix()
    xy = np.stack(np.meshgrid([-0.5, 0.5], [-0.5, 0.5], indexing="ij"), -1).reshape(-1, 2)
    entries = {"ubi": np.repeat(np.linalg.inv(U @ B)[None], 4, 0), "pos": np.column_stack([xy, np.zeros(4)]),
               "density": np.ones(4)}  # fmt: skip
    hkls = np.array([[h, k, l] for h in (-1, 0, 1) for k in (-1, 0, 1) for l in (-1, 0, 1)
                     if h * h + k * k + l * l == 2], float)  # fmt: skip
    wavelength = 12.398419843320026 / 43.0
    pars = {
        "y_center": 255.5, "y_size": 75.0, "tilt_y": 0.0, "z_center": 255.5, "z_size": 75.0, "tilt_z": 0.0,
        "tilt_x": 0.0, "distance": 60e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1, "wavelength": wavelength,
        "wedge": 0.0, "chi": 0.0,
    }  # fmt: skip
    geom = geom_from_pars(pars, 0.0, wavelength * 1e-3, 1e-3, 1e-3, sig_beam=0.5, voxel_size=1.0, sig_psf=0.7)
    omega = np.arange(0.0, 180.0, 0.5) + 0.25
    rows = [make_row(omega, np.full_like(omega, dty)) for dty in (-0.5, 0.5)]
    return entries, hkls, np.ones(len(hkls)), geom, rows, (512, 512)


class TestRefine(unittest.TestCase):
    def test_recovers_perturbed_grain(self):
        entries, hkls, F2, geom, rows, det_shape = _problem()
        meas = []
        for row in rows:  # measured data: the truth, rendered
            frame, pixel, value, _ = render_row(entries, hkls, F2, geom, row, det_shape, min_value=0.0, max_frames=3)
            meas.append(measured(frame, pixel, value, len(row["order"])))
        rng = np.random.default_rng(0)
        rot = Rotation.from_rotvec(np.radians(0.01) * rng.normal(size=(4, 3)) / np.sqrt(3)).as_matrix()
        start = {**entries, "ubi": entries["ubi"] @ np.swapaxes(rot, -1, -2), "density": np.full(4, 1.1)}

        def misorientation(ubi):
            """Degrees from the truth; orientation = polar part of (UBI)^-1 = U B (B is a multiple of I here)."""

            def orientation(m):
                u, _, vt = np.linalg.svd(np.linalg.inv(m))
                return u @ vt

            r = orientation(ubi) @ np.swapaxes(orientation(entries["ubi"]), -1, -2)
            return np.degrees(np.linalg.norm(Rotation.from_matrix(r).as_rotvec(), axis=1))

        out, history = refine(
            start, hkls, F2, geom, rows, meas, det_shape, n_iter=8, cut=0.0, max_frames=3, n_cg=8, log=None
        )
        self.assertLess(history[-1]["loss"], 1e-3 * history[0]["loss"])
        self.assertLess(
            np.sqrt((misorientation(out["ubi"]) ** 2).mean()), 0.2 * np.sqrt((misorientation(start["ubi"]) ** 2).mean())
        )
        self.assertLess(np.abs(out["density"] - 1.0).max(), 0.02)

    def test_spread_and_fixed_density(self):
        """With a spread (sig_rot) the peaks are wider, so a start 10x further off still comes in; fixed densities stay."""
        entries, hkls, F2, geom, rows, det_shape = _problem()
        entries["sig_rot"] = np.full(4, np.radians(0.3))
        meas = []
        for row in rows:
            frame, pixel, value, _ = render_row(entries, hkls, F2, geom, row, det_shape, min_value=0.0, max_frames=3)
            meas.append(measured(frame, pixel, value, len(row["order"])))
        rng = np.random.default_rng(1)
        rot = Rotation.from_rotvec(np.radians(0.1) * rng.normal(size=(4, 3)) / np.sqrt(3)).as_matrix()
        start = {**entries, "ubi": entries["ubi"] @ np.swapaxes(rot, -1, -2)}

        def misorientation(ubi):
            def orientation(m):
                u, _, vt = np.linalg.svd(np.linalg.inv(m))
                return u @ vt

            r = orientation(ubi) @ np.swapaxes(orientation(entries["ubi"]), -1, -2)
            return np.degrees(np.linalg.norm(Rotation.from_matrix(r).as_rotvec(), axis=1))

        out, history = refine(start, hkls, F2, geom, rows, meas, det_shape, n_iter=6, cut=0.0, max_frames=3, n_cg=8,
                              fit_density=False, log=None)  # fmt: skip
        self.assertLess(history[-1]["loss"], 1e-2 * history[0]["loss"])
        self.assertLess(np.sqrt((misorientation(out["ubi"]) ** 2).mean()), 0.01)
        np.testing.assert_array_equal(out["density"], entries["density"])

    def test_per_entry(self):
        """One sweep per iteration, each entry its own rotation step: a start 0.11 deg off, peaks widened by a spread."""
        entries, hkls, F2, geom, rows, det_shape = _problem()
        entries = {k: v[:1] for k, v in entries.items()}
        entries["sig_rot"] = np.full(1, np.radians(0.3))
        meas = []
        for row in rows:
            frame, pixel, value, _ = render_row(entries, hkls, F2, geom, row, det_shape, min_value=0.0, max_frames=7)
            meas.append(measured(frame, pixel, value, len(row["order"])))
        rot = Rotation.from_rotvec(np.radians([0.06, -0.08, 0.05])).as_matrix()[None]
        start = {**entries, "ubi": entries["ubi"] @ np.swapaxes(rot, -1, -2)}
        out, history = refine_per_entry(start, hkls, F2, geom, rows, meas, det_shape, n_iter=6, cut=0.0, max_frames=7,
                                        log=None)  # fmt: skip

        def orientation(m):
            u, _, vt = np.linalg.svd(np.linalg.inv(m))
            return u @ vt

        r = orientation(out["ubi"]) @ np.swapaxes(orientation(entries["ubi"]), -1, -2)
        err = np.degrees(np.linalg.norm(Rotation.from_matrix(r).as_rotvec(), axis=1))
        self.assertLess(history[-1]["loss"], 1e-4 * history[0]["loss"])
        self.assertLess(err.max(), 1e-3)
        np.testing.assert_array_equal(out["density"], entries["density"])


class TestBinFrames(unittest.TestCase):
    def test_sums_runs_of_frames(self):
        omega = np.array([0.25, 0.75, 1.25, 1.75, 2.25])  # file order
        dty = np.zeros(5)
        frame = np.array([0, 1, 1, 3, 4])
        pixel = np.array([7, 7, 9, 2, 2])
        value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        om_b, _, fr, px, val = bin_frames(omega, dty, frame, pixel, value, 2)
        np.testing.assert_allclose(om_b, [0.5, 1.5, 2.25])  # bins: frames 0-1, 2-3, 4
        got = {(int(f), int(p)): float(v) for f, p, v in zip(fr, px, val)}
        self.assertEqual(got, {(0, 7): 3.0, (0, 9): 3.0, (1, 2): 4.0, (2, 2): 5.0})


if __name__ == "__main__":
    unittest.main()
