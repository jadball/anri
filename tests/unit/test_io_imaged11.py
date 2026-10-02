import os
import tempfile
import unittest

import h5py
import jax
import numpy as np

from anri.io_imaged11 import (
    entries_from_tensormap,
    geom_from_pars,
    motor_grid,
    simulate_sparse,
    write_dataset,
    write_par,
    write_scan,
)

jax.config.update("jax_enable_x64", True)

QUARTZ_FLYXDM_H5 = os.path.join(
    os.path.dirname(__file__), "..", "data", "phantoms", "quartz_flyxdm", "quartz_flyxdm_tmap.h5"
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
            sparse, dsfile, par = (os.path.join(tmp, f) for f in ("s_sparse.h5", "s_dataset.h5", "s.par"))
            with h5py.File(sparse, "w") as hout:
                for i in range(5):
                    write_scan(
                        hout, f"{i + 1}.1", np.zeros(0, int), np.zeros(0, int), np.zeros(0), omega[i], dty[i], (4, 4)
                    )
            write_par(par, {"wavelength": 0.2, "distance": 150e3})
            write_dataset(sparse, dsfile, y0=0.25, parfile=par)
            ds = load(dsfile)
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
