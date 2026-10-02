import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.crystal
import anri.geom
from anri.fwd._impl.render import dty_weight, lorentz, make_row, polarisation, render_row
from anri.fwd._impl.scan import get_centroid_scan

jax.config.update("jax_enable_x64", True)


class TestDtyWeight(unittest.TestCase):
    def test_area(self):
        # integrated over dty, the weight is the voxel area for any omega and beam size
        delta = np.linspace(-20, 20, 40001)
        for omega in [0.0, 10.0, 45.0, 90.0, 123.4]:
            for sig in [0.05, 0.5, 3.0]:
                w = dty_weight(jnp.asarray(delta), omega, 2.0, sig)
                np.testing.assert_allclose(np.trapezoid(w, delta), 4.0, rtol=1e-6)

    def test_convolution(self):
        # compare with a brute-force convolution of the chord length with a Gaussian beam
        size, sig, omega = 1.5, 0.4, 30.0
        u = np.linspace(-3, 3, 6001)
        c, s = np.abs(np.cos(np.radians(omega))), np.abs(np.sin(np.radians(omega)))
        a, b = 0.5 * size * (c + s), 0.5 * size * abs(c - s)
        chord = np.clip((a - np.abs(u)) / (a - b), 0, 1) * size / max(c, s)
        for d in [-1.0, 0.0, 0.3, 1.2]:
            g = np.exp(-0.5 * ((d - u) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
            np.testing.assert_allclose(dty_weight(d, omega, size, sig), np.trapezoid(chord * g, u), rtol=1e-4)

    def test_continuous_near_box(self):
        # the rectangle branch at omega = 0 joins the trapezoid smoothly
        w = [dty_weight(0.3, om, 1.0, 0.4) for om in (0.0, 1e-3, 1e-1)]
        np.testing.assert_allclose(w[0], w[1], rtol=1e-5)
        np.testing.assert_allclose(w[0], w[2], rtol=1e-3)


class TestIntensityFactors(unittest.TestCase):
    def test_id11(self):
        from ImageD11.refinegrains import lf, polarization

        rng = np.random.default_rng(0)
        tth, eta = rng.uniform(2, 30, 100), rng.uniform(-180, 180, 100)
        t, e = np.radians(tth), np.radians(eta)
        # ImageD11 eta = atan2(-y, z)
        k_out = np.stack([np.cos(t), -np.sin(t) * np.sin(e), np.sin(t) * np.cos(e)], 1)
        k_in, axis = jnp.array([1.0, 0, 0]), jnp.array([0, 0, 1.0])
        L = jax.vmap(lorentz, in_axes=(None, 0, None))(k_in, k_out, axis)
        np.testing.assert_allclose(L * lf(tth, eta), 1.0, rtol=1e-10)
        for f in [1.0, 0.9, 0.0]:
            P = jax.vmap(polarisation, in_axes=(0, None))(k_out, f)
            np.testing.assert_allclose(P, polarization(tth, eta, factor=f), atol=1e-12)


def _single_peak_setup():
    """A 10 x 10 undeformed Fe grain, Eiger 4M at 150 mm, 43 keV, and one hkl well away from eta = 0, 180."""
    pars = {
        "y_center": 1049.9, "y_size": 75.0, "tilt_y": -2e-3,
        "z_center": 1116.5, "z_size": 75.0, "tilt_z": 3e-3, "tilt_x": 1e-3,
        "distance": 150e3, "o11": -1.0, "o12": 0.0, "o21": 0.0, "o22": -1.0,
        "wavelength": 12.398419843320026 / 43.0, "wedge": 0.0, "chi": 0.0,
    }  # fmt: skip
    det_shape = (2162, 2068)
    det_trans, shift, xshift = anri.geom.detector_transforms(
        *(pars[k] for k in ("y_center", "y_size", "tilt_y", "z_center", "z_size", "tilt_z", "tilt_x", "distance")),
        *(pars[k] for k in ("o11", "o12", "o21", "o22")),
    )
    sc_lab, fc_lab, norm_lab = anri.geom.detector_basis_vectors_lab(det_trans, shift, xshift)
    voxel = 1.0
    geom = {
        "wavelength": pars["wavelength"], "k_in_lab": jnp.array([1.0, 0.0, 0.0]),
        "wedge": 0.0, "chi": 0.0, "y0": 0.0,
        "sc_lab": sc_lab, "fc_lab": fc_lab, "norm_lab": norm_lab,
        # broad enough that spots cover a few pixels and frames, so moments are unbiased
        "sig_wavelength": pars["wavelength"] * 1e-3, "sig_ky": 1e-3, "sig_kz": 1e-3,
        "sig_beam": 0.5, "voxel_size": voxel, "pol_factor": 1.0,
    }  # fmt: skip

    U = np.asarray(anri.geom.rot_z(25.0) @ anri.geom.rot_x(35.0) @ anri.geom.rot_y(10.0))
    B = np.asarray(anri.crystal.lpars_to_B(jnp.array([2.8694, 2.8694, 2.8694, 90.0, 90.0, 90.0])))
    ubi = np.linalg.inv(U @ B)
    i, j = np.mgrid[0:10, 0:10]
    pos = np.stack([(i.ravel() - 4.5) * voxel, (j.ravel() - 4.5) * voxel, np.zeros(100)], 1)
    entries = {"ubi": np.repeat(ubi[None], 100, 0), "pos": pos, "density": np.ones(100)}

    # first {110} reflection, etasign +1, that lands well inside the detector with |sin(eta)| > 0.5
    centroid_fn = jax.vmap(get_centroid_scan, in_axes=(None, None, 0) + (None,) * 11)
    hkls = np.array([h for h in np.ndindex(3, 3, 3) if sorted(np.abs(np.array(h) - 1)) == [0, 1, 1]]) - 1
    args = (1.0, geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, 0.0, 0.0, 0.0, sc_lab, fc_lab, norm_lab)
    cen, valid = centroid_fn(ubi, jnp.zeros(3), jnp.asarray(hkls, float), *args)
    xyz = jax.vmap(anri.geom.det_to_lab, in_axes=(0, 0, None, None, None))(
        cen[:, 0], cen[:, 1], det_trans, shift, xshift
    )
    eta = np.degrees(np.arctan2(-xyz[:, 1], xyz[:, 2]))
    ok = (
        np.asarray(valid)
        & (np.abs(np.sin(np.radians(eta))) > 0.5)
        & (cen[:, 0] > 50) & (cen[:, 0] < det_shape[0] - 50)
        & (cen[:, 1] > 50) & (cen[:, 1] < det_shape[1] - 50)
    )  # fmt: skip
    k = int(np.flatnonzero(ok)[0])
    return pars, det_shape, geom, entries, hkls[k].astype(float), float(cen[k, 2])


class TestSinglePeak(unittest.TestCase):
    def test_id11_labelling(self):
        """Render one reflection from a small grain; ImageD11's sparse labelling must recover its centroid.

        Then push the measured centroid back through ImageD11's own geometry: UBI . g must return the hkl.
        """
        from ImageD11 import cImageD11, sparseframe, transform

        pars, det_shape, geom, entries, hkl, omega_c = _single_peak_setup()
        ostep = 0.05
        omega = (np.floor(omega_c / ostep) + np.arange(-40, 41)) * ostep + 0.5 * ostep
        nf = det_shape[1]
        ubi = entries["ubi"][0]

        # check the moment column naming on a toy frame: one pixel at row 3, col 7
        toy = sparseframe.sparse_frame([3], [7], (10, 10))
        toy.set_pixels("intensity", np.array([2.0], np.float32))
        sparseframe.sparse_connected_pixels(toy, threshold=0, label_name="cp")
        m = sparseframe.sparse_moments(toy, "intensity", "cp")[0]
        self.assertEqual((m[cImageD11.s2D_sI] / m[cImageD11.s2D_I], m[cImageD11.s2D_fI] / m[cImageD11.s2D_I]), (3, 7))

        centroids, _ = jax.vmap(get_centroid_scan, in_axes=(0, 0) + (None,) * 12)(
            jnp.asarray(entries["ubi"]), jnp.asarray(entries["pos"]), jnp.asarray(hkl), 1.0,
            geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, 0.0, 0.0, geom["y0"],
            geom["sc_lab"], geom["fc_lab"], geom["norm_lab"],
        )  # fmt: skip
        centroids = np.asarray(centroids)
        n_rows_checked = 0
        for dty in np.arange(-7.0, 8.0):
            row = make_row(omega, np.full_like(omega, dty))
            frame, pixel, value, stats = render_row(
                entries, hkl[None], np.ones(1), geom, row, det_shape, window=(9, 17, 17), min_value=0.0
            )
            if stats["n_peaks"] == 0:
                continue
            self.assertGreater(stats["captured"].min(), 0.999)

            # expected: centroids of the voxels weighted by how much of each sits in the beam
            w = np.asarray(dty_weight(dty - centroids[:, 3], centroids[:, 2], geom["voxel_size"], geom["sig_beam"]))
            if w.sum() < 0.05 * geom["voxel_size"] ** 2:
                continue  # the grain is barely in this row
            expected = (w[:, None] * centroids[:, :3]).sum(0) / w.sum()

            # ImageD11: connected pixels and moments per frame, then merge frames by intensity
            sumI = srI = sfI = soI = 0.0
            for f in np.unique(frame):
                sel = frame == f
                spf = sparseframe.sparse_frame(pixel[sel] // nf, pixel[sel] % nf, det_shape)
                spf.set_pixels("intensity", value[sel].astype(np.float32))
                n = sparseframe.sparse_connected_pixels(spf, threshold=0, label_name="cp")
                self.assertEqual(n, 1)
                m = sparseframe.sparse_moments(spf, "intensity", "cp")[0]
                sumI += m[cImageD11.s2D_I]
                srI += m[cImageD11.s2D_sI]
                sfI += m[cImageD11.s2D_fI]
                soI += m[cImageD11.s2D_I] * omega[f]
            measured = np.array([srI, sfI, soI]) / sumI
            np.testing.assert_allclose(measured[:2], expected[:2], atol=0.02)  # pixels
            np.testing.assert_allclose(measured[2], expected[2], atol=2e-3)  # degrees

            # ImageD11 geometry: lab xyz of the measured spot, shift x by the diffraction origin, g, then hkl
            sx, sy = entries["pos"][:, 0], entries["pos"][:, 1]
            om = np.radians(measured[2])
            lx = (w * (sx * np.cos(om) - sy * np.sin(om))).sum() / w.sum()
            xyz = transform.compute_xyz_lab(np.array([[measured[0]], [measured[1]]]), **pars)
            xyz[0] -= lx
            tth, eta = transform.compute_tth_eta_from_xyz(xyz, np.array([measured[2]]))
            g = transform.compute_g_vectors(tth, eta, np.array([measured[2]]), pars["wavelength"])
            np.testing.assert_allclose((ubi @ g).ravel(), hkl, atol=2e-4)
            n_rows_checked += 1
        self.assertGreater(n_rows_checked, 5)
