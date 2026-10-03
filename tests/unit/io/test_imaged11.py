import os
import tempfile
import unittest

import h5py
import jax
import numpy as np

from anri.io import (
    detector_from_pars,
    entries_from_tensormap,
    geom_from_pars,
    motor_grid,
    simulate_sparse,
    write_dataset,
    write_pars,
    write_peaks_table,
    write_scan,
    write_zero_distortion,
)

jax.config.update("jax_enable_x64", True)

QUARTZ_FLYXDM_H5 = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "phantoms", "quartz_flyxdm", "quartz_flyxdm_tmap.h5"
)


class TestWriteScan(unittest.TestCase):
    def test_imaged11_reads_back(self):
        from ImageD11.sparseframe import SparseScan

        det_shape = (20, 30)
        omega, dty = np.arange(5) + 0.5, np.full(5, 2.0)
        frame = np.array([0, 0, 2, 2, 2, 4])
        pixel = np.array([3, 31, 0, 45, 599, 100])
        value = np.array([5.2, 0.9, 1.6, 300.4, 2.5, 1.4])  # rounds to 5, 1, 2, 300, 2, 1
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sparse.h5")
            with h5py.File(path, "w") as hout:
                n = write_scan(hout, "1.1", frame, pixel, value, omega, dty, det_shape, cut=1)
            self.assertEqual(n, 4)  # counts of 1 are not > cut
            scan = SparseScan(path, "1.1")
            np.testing.assert_array_equal(scan.nnz, [1, 0, 3, 0, 0])
            np.testing.assert_array_equal(scan.row, [0, 0, 1, 19])
            np.testing.assert_array_equal(scan.col, [3, 0, 15, 29])
            np.testing.assert_array_equal(scan.intensity, [5, 2, 300, 2])
            self.assertEqual(scan.shape, (5, 20, 30))


class TestDataset(unittest.TestCase):
    def test_load(self):
        from ImageD11.sinograms.dataset import load

        omega, dty = motor_grid((0.0, 2.0), 0.5, (-1.0, 1.0), 0.5)
        self.assertEqual(omega.shape, (5, 4))
        with tempfile.TemporaryDirectory() as tmp:
            sparse = os.path.join(tmp, "s_sparse.h5")
            with h5py.File(sparse, "w") as hout:
                for i in range(5):
                    write_scan(
                        hout, f"{i + 1}.1", np.zeros(0, int), np.zeros(0, int), np.zeros(0), omega[i], dty[i], (4, 4)
                    )
            dsfile = write_dataset(sparse, tmp, "s", "d", y0=0.25)
            self.assertEqual(dsfile, os.path.join(tmp, "s", "s_d", "s_d_dataset.h5"))
            ds = load(dsfile)
            self.assertEqual(os.path.dirname(ds.pksfile), os.path.dirname(dsfile))
            self.assertEqual(ds.shape, (5, 4))
            np.testing.assert_allclose(ds.omega, omega)
            np.testing.assert_allclose(ds.dty, dty)
            np.testing.assert_allclose(ds.ybincens, dty[:, 0])
            with h5py.File(dsfile, "r") as f:
                self.assertEqual(f.attrs["y0"], 0.25)


class TestEntriesFromTensorMap(unittest.TestCase):
    def test_quartz(self):
        from ImageD11.sinograms.geometry import recon_to_sample
        from ImageD11.sinograms.tensor_map import TensorMap

        tmap = TensorMap.from_h5(QUARTZ_FLYXDM_H5)
        entries = entries_from_tensormap(tmap)
        self.assertEqual(len(entries["ubi"]), 6818)
        # map cell (y, x) is recon cell (x, ny - 1 - y)
        _, ny, nx = tmap.shape
        y, x = np.argwhere(tmap.labels[0] >= 0)[0]
        sx, sy = recon_to_sample(x, ny - 1 - y, (nx, ny), tmap.steps[1])
        k = np.flatnonzero(np.all(np.isclose(entries["pos"][:, :2], [sx, sy]), axis=1))
        self.assertEqual(k.size, 1)
        np.testing.assert_array_equal(entries["ubi"][k[0]], tmap.UBI[0, y, x])
        np.testing.assert_array_equal(entries["density"], 1.0)

    def test_density_map(self):
        from ImageD11.sinograms.tensor_map import TensorMap

        tmap = TensorMap.from_h5(QUARTZ_FLYXDM_H5)
        density = np.where(tmap.labels >= 0, 0.5, 0.0)
        density[0, 40:50, 40:50] = 0.0  # a pore
        tmap.add_map("density", density)
        entries = entries_from_tensormap(tmap)
        n_pore = int(np.sum(tmap.labels[0, 40:50, 40:50] >= 0))
        self.assertEqual(np.sum(entries["density"] == 0.0), n_pore)
        self.assertEqual(np.sum(entries["density"] == 0.5), len(entries["density"]) - n_pore)


class TestSimulateSparse(unittest.TestCase):
    def test_small_grain(self):
        from ImageD11.sparseframe import SparseScan

        import anri.crystal
        import anri.geom

        pars = {
            "y_center": 1049.9, "y_size": 75.0, "tilt_y": -2e-3,
            "z_center": 1116.5, "z_size": 75.0, "tilt_z": 3e-3, "tilt_x": 1e-3,
            "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1,
            "wavelength": 12.398419843320026 / 43.0, "wedge": 0.0, "chi": 0.0,
        }  # fmt: skip
        geom = geom_from_pars(pars, 0.0, pars["wavelength"] * 1e-3, 1e-3, 1e-3, sig_beam=0.5, voxel_size=1.0)
        U = np.asarray(anri.geom.rot_z(25.0) @ anri.geom.rot_x(35.0) @ anri.geom.rot_y(10.0))
        B = np.asarray(anri.crystal.lpars_to_B(np.array([2.8694, 2.8694, 2.8694, 90.0, 90.0, 90.0])))
        i, j = np.mgrid[0:4, 0:4]
        pos = np.stack([i.ravel() - 1.5, j.ravel() - 1.5, np.zeros(16)], 1)
        entries = {"ubi": np.repeat(np.linalg.inv(U @ B)[None], 16, 0), "pos": pos, "density": np.full(16, 50.0)}
        hkls = np.array([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [2.0, 0.0, 0.0]])
        omega, dty = motor_grid((0.0, 180.0), 0.25, (-3.0, 3.0), 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sparse.h5")
            stats = simulate_sparse(path, entries, hkls, np.ones(3), geom, omega, dty, (2162, 2068), window=(5, 17, 17))
            self.assertGreater(stats["n_pixels"].sum(), 0)
            with h5py.File(path, "r") as f:
                self.assertEqual(sorted(f, key=float), [f"{k}.1" for k in range(1, 8)])
            for k, n in enumerate(stats["n_pixels"]):
                scan = SparseScan(path, f"{k + 1}.1")
                self.assertEqual(scan.nnz.sum(), n)
                self.assertTrue(np.all(scan.intensity > 1))

    def test_imaged11_peaks(self):
        """The whole pipeline: simulate, write pars, zero distortion, DataSet and peaks table; ImageD11 makes the 2D peaks."""
        from ImageD11.sinograms.dataset import load

        import anri.crystal
        import anri.geom

        pars = {
            "y_center": 1049.9, "y_size": 75.0, "tilt_y": 0.0,
            "z_center": 1116.5, "z_size": 75.0, "tilt_z": 0.0, "tilt_x": 0.0,
            "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1,
            "wavelength": 12.398419843320026 / 43.0, "wedge": 0.0, "chi": 0.0,
        }  # fmt: skip
        a = 2.8694
        cell = {"cell__a": a, "cell__b": a, "cell__c": a, "cell_alpha": 90.0, "cell_beta": 90.0, "cell_gamma": 90.0}
        cell["cell_lattice_[P,A,B,C,I,F,R]"] = 229
        geom = geom_from_pars(pars, 0.0, pars["wavelength"] * 1e-3, 1e-3, 1e-3, sig_beam=0.5, voxel_size=1.0)
        B = np.asarray(anri.crystal.lpars_to_B(np.array([a, a, a, 90.0, 90.0, 90.0])))
        U = np.asarray(anri.geom.rot_z(25.0) @ anri.geom.rot_x(35.0))
        entries = {"ubi": np.linalg.inv(U @ B)[None], "pos": np.zeros((1, 3)), "density": np.full(1, 100.0)}
        hkls = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        omega, dty = motor_grid((0.0, 180.0), 0.25, (-1.0, 1.0), 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            sparse = os.path.join(tmp, "sparse.h5")
            simulate_sparse(sparse, entries, hkls, np.ones(3), geom, omega, dty, (2162, 2068), window=(5, 17, 17))
            parfile = write_pars(os.path.join(tmp, "pars"), pars, {"Fe": cell})
            e2dx, e2dy = write_zero_distortion(os.path.join(tmp, "pars"), (2162, 2068))
            dsfile = write_dataset(sparse, tmp, "fe", "sim", y0=0.0, parfile=parfile, e2dxfile=e2dx, e2dyfile=e2dy)
            pksfile = write_peaks_table(dsfile, nproc=1)
            ds = load(dsfile)
            self.assertEqual(pksfile, ds.pksfile)
            self.assertIn("Fe", ds.get_phases_from_disk().unitcells)
            cf = ds.get_cf_2d()
            self.assertGreater(cf.nrows, 0)
            np.testing.assert_allclose(cf.sc, cf.s_raw)
            np.testing.assert_allclose(cf.fc, cf.f_raw)


class TestGeomFromPars(unittest.TestCase):
    def test_detector_from_pars(self):
        """Pixel to lab matches ImageD11, and ray-tracing the lab point back from the origin gives the same pixel."""
        import jax.numpy as jnp
        from ImageD11 import transform

        import anri.geom

        pars = {
            "y_center": 1049.9, "y_size": 75.0, "tilt_y": -2e-3,
            "z_center": 1116.5, "z_size": 75.0, "tilt_z": 3e-3, "tilt_x": 1e-3,
            "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1,
        }  # fmt: skip
        det = detector_from_pars(pars)
        for sc, fc in ((1500.3, 700.8), (12.0, 2000.5)):
            xyz = anri.geom.det_to_lab(sc, fc, det["det_trans"], det["beam_cen_shift"], det["x_distance_shift"])
            np.testing.assert_allclose(xyz, transform.compute_xyz_lab(np.array([[sc], [fc]]), **pars).ravel(), atol=1e-6)
            back = anri.geom.raytrace_to_det(
                xyz / jnp.linalg.norm(xyz), jnp.zeros(3), det["s_step_lab"], det["f_step_lab"], det["det_origin_lab"]
            )
            np.testing.assert_allclose(back, (sc, fc), atol=1e-8)

    def test_wedge_chi_against_imaged11(self):
        """A peak computed by anri from geom_from_pars maps back to its hkl through ImageD11's own geometry."""
        import jax.numpy as jnp
        from ImageD11 import transform

        import anri.crystal
        import anri.geom
        from anri.fwd._impl.scan import get_centroid_scan

        pars = {
            "y_center": 1049.9, "y_size": 75.0, "tilt_y": -2e-3,
            "z_center": 1116.5, "z_size": 75.0, "tilt_z": 3e-3, "tilt_x": 1e-3,
            "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1,
            "wavelength": 0.2883, "wedge": 2.5, "chi": -1.5,
        }  # fmt: skip
        geom = geom_from_pars(pars, 0.0, 0.0, 0.0, 0.0, sig_beam=0.5, voxel_size=1.0)
        self.assertEqual(geom["wedge"], -2.5)
        U = np.asarray(anri.geom.rot_z(25.0) @ anri.geom.rot_x(35.0) @ anri.geom.rot_y(10.0))
        B = np.asarray(anri.crystal.lpars_to_B(np.array([2.8694, 2.8694, 2.8694, 90.0, 90.0, 90.0])))
        ubi = np.linalg.inv(U @ B)
        checked = 0
        for hkl in ([1.0, 1.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 1.0], [-1.0, 2.0, 1.0], [3.0, 1.0, 0.0]):
            for etasign in (1.0, -1.0):
                c, valid = get_centroid_scan(
                    jnp.asarray(ubi), jnp.zeros(3), jnp.asarray(hkl), etasign, geom["wavelength"], geom["k_in_lab"],
                    0.0, 0.0, geom["wedge"], geom["chi"], geom["y0"],
                    geom["s_step_lab"], geom["f_step_lab"], geom["det_origin_lab"],
                )  # fmt: skip
                if not valid or not (0 < c[0] < 2162 and 0 < c[1] < 2068):
                    continue
                xyz = transform.compute_xyz_lab(np.array([[c[0]], [c[1]]]), **pars)
                tth, eta = transform.compute_tth_eta_from_xyz(
                    xyz, np.array([c[2]]), wedge=pars["wedge"], chi=pars["chi"]
                )
                g = transform.compute_g_vectors(
                    tth, eta, np.array([c[2]]), pars["wavelength"], pars["wedge"], pars["chi"]
                )
                np.testing.assert_allclose((ubi @ g).ravel(), hkl, atol=1e-6)
                checked += 1
        self.assertGreater(checked, 3)
