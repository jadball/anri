import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.crystal
import anri.geom
from anri.fwd import (
    beam_weight,
    check_render,
    guess_batch_size,
    lorentz,
    make_row,
    polarisation,
    render_peaks,
    render_row,
    select_peaks,
)
from anri.fwd._impl.render import _centroid, _compact, _free_memory, _merge
from anri.fwd._impl.scan import get_centroid_scan
from anri.geom import sample_to_lab

jax.config.update("jax_enable_x64", True)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoid rule (np.trapezoid needs numpy >= 2)."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _geom(sig, width=0.0, size=1.2, k=(1.0, 0.0, 0.0), sig_v=0.0, width_v=0.0, cube=False):
    """The beam_weight part of a geometry dict."""
    return {"k_in_lab": jnp.asarray(k), "sig_beam": sig, "width_beam": width, "sig_beam_v": sig_v,
            "width_beam_v": width_v, "voxel_3d": cube, "voxel_size": size}  # fmt: skip


def _weight(delta, omega, geom, z=0.0):
    """beam_weight of a voxel at lab (0, delta, z): delta along lab y from the beam's centre line."""
    return beam_weight(jnp.array([0.0, delta, z]), omega, geom)


class TestBeamWeight(unittest.TestCase):
    def test_area(self):
        # a column's weight integrated over its distance from the beam is its area, for any beam: the profile
        # integrates to 1
        delta = np.linspace(-20, 20, 40001)
        for omega in [0.0, 10.0, 45.0, 90.0, 123.4]:
            for sig, width in [(0.05, 0.0), (0.5, 0.0), (3.0, 0.0), (0.05, 2.0), (0.5, 6.0)]:
                g = _geom(sig, width, size=2.0)
                w = jax.vmap(_weight, in_axes=(0, None, None))(jnp.asarray(delta), omega, g)
                np.testing.assert_allclose(_trapezoid(np.asarray(w), delta), 4.0, rtol=1e-6)

    def test_convolution(self):
        # compare with a brute-force convolution of the chord length with the profile, Gaussian and flat-top
        from scipy.special import ndtr

        size, omega = 1.5, 30.0
        u = np.linspace(-6, 6, 12001)
        c, s = np.abs(np.cos(np.radians(omega))), np.abs(np.sin(np.radians(omega)))
        a, b = 0.5 * size * (c + s), 0.5 * size * abs(c - s)
        chord = np.clip((a - np.abs(u)) / (a - b), 0, 1) * size / max(c, s)
        for sig, width in [(0.4, 0.0), (0.1, 2.0), (0.4, 1.0)]:
            for d in [-1.0, 0.0, 0.3, 1.2]:
                if width == 0:
                    p = np.exp(-0.5 * ((d - u) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
                else:
                    p = (ndtr((d - u + width / 2) / sig) - ndtr((d - u - width / 2) / sig)) / width
                np.testing.assert_allclose(
                    _weight(d, omega, _geom(sig, width, size)), _trapezoid(chord * p, u), rtol=1e-4
                )

    def test_continuous(self):
        # the rectangle branch at omega = 0 joins the trapezoid, and the Gaussian limit joins the flat top
        w = [_weight(0.3, om, _geom(0.4, size=1.0)) for om in (0.0, 1e-3, 1e-1)]
        np.testing.assert_allclose(w[0], w[1], rtol=1e-5)
        np.testing.assert_allclose(w[0], w[2], rtol=1e-3)
        w = [_weight(0.3, 30.0, _geom(0.4, width)) for width in (0.0, 0.0039, 0.0041)]  # switch at 0.01 sigma
        np.testing.assert_allclose(w[0], w[1], rtol=1e-6)
        np.testing.assert_allclose(w[0], w[2], rtol=1e-5)


def _brute_force_weight(delta, omega, size, sig_h, k_in, sig_v=None, width_h=0.0, cube_z=None, n=400, nz=600):
    """Integrate the beam profile over a square voxel rotated by omega, shifted by delta along lab y.

    The beam passes through the origin along k_in, across it horizontally a flat top of width_h blurred by a
    Gaussian of sig_h. With sig_v, it is also Gaussian vertically, and the voxel is a column, integrated over height
    on a grid; with cube_z too, the voxel is a cube centred at that height.
    """
    from scipy.special import ndtr

    k = np.asarray(k_in, float) / np.linalg.norm(k_in)
    e_h = np.cross([0.0, 0.0, 1.0], k)
    e_h /= np.linalg.norm(e_h)
    e_v = np.cross(k, e_h)
    t = (np.arange(n) + 0.5) / n * size - size / 2
    s, u = np.meshgrid(t, t, indexing="ij")
    c, sn = np.cos(np.radians(omega)), np.sin(np.radians(omega))
    x, y = c * s - sn * u, sn * s + c * u + delta  # voxel rotated by +omega about z, then moved by delta along y
    area = (size / n) ** 2

    def profile(d, sig, width=0.0):
        if width == 0:
            return np.exp(-0.5 * (d / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
        return (ndtr((d + width / 2) / sig) - ndtr((d - width / 2) / sig)) / width

    p_h = profile(x * e_h[0] + y * e_h[1], sig_h, width_h)
    if sig_v is None:
        return np.sum(p_h) * area
    if cube_z is None:
        zmax = 8 * (sig_v + size) / e_v[2]
        z = (np.arange(nz) + 0.5) / nz * 2 * zmax - zmax
    else:
        z = cube_z + (np.arange(nz) + 0.5) / nz * size - size / 2
    p_v = profile(x[..., None] * e_v[0] + y[..., None] * e_v[1] + z * e_v[2], sig_v)
    return np.sum(p_h[..., None] * p_v) * area * (z[1] - z[0])


class TestBeamWeightGeometry(unittest.TestCase):
    def test_horizontal_beam(self):
        # a beam at psi from lab x in the horizontal plane, Gaussian and flat-top
        for psi, omega, delta, width in ((20.0, 30.0, 0.3, 0.0), (-35.0, 70.0, -0.5, 0.0), (10.0, 0.0, 0.0, 1.5)):
            k = [np.cos(np.radians(psi)), np.sin(np.radians(psi)), 0.0]
            expected = _brute_force_weight(delta, omega, 1.2, 0.4, k, width_h=width)
            np.testing.assert_allclose(_weight(delta, omega, _geom(0.4, width, k=k)), expected, rtol=2e-4)

    def test_tilted_pencil(self):
        # a column, tilted beam: the pencil's vertical profile integrates out
        for alpha, psi, omega, delta in ((25.0, 0.0, 30.0, 0.2), (40.0, 15.0, 50.0, -0.4)):
            a, p = np.radians(alpha), np.radians(psi)
            k = [np.cos(a) * np.cos(p), np.cos(a) * np.sin(p), np.sin(a)]
            expected = _brute_force_weight(delta, omega, 1.2, 0.4, k, sig_v=0.3, n=200)
            np.testing.assert_allclose(_weight(delta, omega, _geom(0.4, k=k)), expected, rtol=1e-3)

    def test_cube(self):
        # a cube (3D map), horizontal beam: the vertical profile over the cube's height
        for omega, delta, z in ((30.0, 0.3, 0.0), (10.0, -0.2, 0.5), (60.0, 0.1, -1.0)):
            expected = _brute_force_weight(delta, omega, 1.2, 0.4, [1, 0, 0], sig_v=0.3, cube_z=z, n=200, nz=400)
            g = _geom(0.4, sig_v=0.3, cube=True)
            np.testing.assert_allclose(_weight(delta, omega, g, z=z), expected, rtol=1e-3)


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
            P = jax.vmap(polarisation, in_axes=(None, 0, None))(k_in, k_out, f)
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
    s_step_lab, f_step_lab, det_origin_lab = anri.geom.detector_basis_vectors_lab(det_trans, shift, xshift)
    voxel = 1.0
    geom = {
        "wavelength": pars["wavelength"], "k_in_lab": jnp.array([1.0, 0.0, 0.0]),
        "wedge": 0.0, "chi": 0.0, "y0": 0.0,
        "s_step_lab": s_step_lab, "f_step_lab": f_step_lab, "det_origin_lab": det_origin_lab,
        # broad enough that spots cover a few pixels and frames, so moments are unbiased
        "sig_wavelength": pars["wavelength"] * 1e-3, "sig_ky": 1e-3, "sig_kz": 1e-3,
        "sig_beam": 0.5, "width_beam": 0.0, "sig_beam_v": 0.0, "width_beam_v": 0.0, "voxel_3d": False,
        "voxel_size": voxel, "pol_factor": 1.0, "sig_psf": 0.0,
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
    args = (1.0, geom["wavelength"], geom["k_in_lab"], 0.0, 0.0, 0.0, 0.0, 0.0, s_step_lab, f_step_lab, det_origin_lab)
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

        n_rows_checked = 0
        for dty in np.arange(-7.0, 8.0):
            row = make_row(omega, np.full_like(omega, dty))
            frame, pixel, value, stats = render_row(
                entries, hkl[None], np.ones(1), geom, row, det_shape, window=(9, 17, 17), min_value=0.0
            )
            if stats["n_peaks"] == 0:
                continue
            self.assertGreater(stats["captured"].min(), 0.999)

            # expected: centroids of the voxels, at their real positions for this dty, weighted by how much of each
            # the beam illuminates
            g = jax.tree.map(jnp.asarray, geom)
            pos = jnp.asarray(entries["pos"])
            in_axes = (None, 0) + (None,) * 7
            centroids = np.asarray(
                jax.vmap(_centroid, in_axes=in_axes)(jnp.asarray(ubi), pos, jnp.asarray(hkl), 1.0, g["wavelength"],
                                                     0.0, 0.0, dty, g)[0]
            )  # fmt: skip
            om = centroids[0, 2]  # one orientation: every voxel diffracts at the same omega
            lab = jax.vmap(sample_to_lab, in_axes=(0,) + (None,) * 5)(pos, om, 0.0, 0.0, dty, g["y0"])
            w = np.asarray(jax.vmap(beam_weight, in_axes=(0, None, None))(lab, om, g))
            lab = np.asarray(lab)
            if w.sum() < 0.05 * geom["voxel_size"] ** 2:
                continue  # the grain is barely in this row
            expected = (w[:, None] * centroids).sum(0) / w.sum()

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

            # ImageD11 geometry: lab xyz of the measured spot, less the (weighted) diffraction origin, g, then hkl
            origin = (w[:, None] * lab).sum(0) / w.sum()
            xyz = transform.compute_xyz_lab(np.array([[measured[0]], [measured[1]]]), **pars)
            xyz[:, 0] -= origin
            tth, eta = transform.compute_tth_eta_from_xyz(xyz, np.array([measured[2]]))
            g = transform.compute_g_vectors(tth, eta, np.array([measured[2]]), pars["wavelength"])
            np.testing.assert_allclose((ubi @ g).ravel(), hkl, atol=2e-4)
            n_rows_checked += 1
        self.assertGreater(n_rows_checked, 5)


class TestPointSpread(unittest.TestCase):
    def test_sharp_spot_centroid(self):
        """A spot much narrower than a pixel has its pixel centroid snapped to the pixel centre; a PSF fixes that."""
        _, det_shape, geom, entries, hkl, omega_c = _single_peak_setup()
        one = {"ubi": entries["ubi"][:1], "pos": np.zeros((1, 3)), "density": np.ones(1)}
        sharp = {**geom, "sig_wavelength": geom["wavelength"] * 1e-5, "sig_ky": 1e-5, "sig_kz": 1e-5}
        centroid, _ = get_centroid_scan(
            jnp.asarray(one["ubi"][0]), jnp.zeros(3), jnp.asarray(hkl), 1.0, geom["wavelength"], geom["k_in_lab"],
            0.0, 0.0, 0.0, 0.0, geom["y0"], geom["s_step_lab"], geom["f_step_lab"], geom["det_origin_lab"],
        )  # fmt: skip
        omega = (np.floor(omega_c / 0.05) + np.arange(-20, 21)) * 0.05 + 0.025
        row = make_row(omega, np.zeros_like(omega))
        errors = {}
        for psf in (0.0, 0.7):
            _, pixel, value, _ = render_row(
                one, hkl[None], np.ones(1), {**sharp, "sig_psf": psf}, row, det_shape, window=(5, 9, 9), min_value=0.0
            )
            s, f = pixel // det_shape[1], pixel % det_shape[1]
            measured = np.array([(s * value).sum(), (f * value).sum()]) / value.sum()
            errors[psf] = np.abs(measured - np.asarray(centroid[:2])).max()
        self.assertLess(errors[0.7], 0.01)
        self.assertGreater(errors[0.0], errors[0.7])


class TestOmegaSpread(unittest.TestCase):
    def test_sig_omega_broadens_omega(self):
        """sig_omega adds its variance to every peak's omega variance, and leaves its centroid in place."""
        _, det_shape, geom, entries, hkl, omega_c = _single_peak_setup()
        one = {"ubi": entries["ubi"][:1], "pos": np.zeros((1, 3)), "density": np.ones(1)}
        omega = (np.floor(omega_c / 0.01) + np.arange(-60, 61)) * 0.01 + 0.005  # fine frames around the peak
        row = make_row(omega, np.zeros_like(omega))
        moments = {}
        for sig in (0.0, 0.05):
            frame, _, value, _ = render_row(
                one, hkl[None], np.ones(1), {**geom, "sig_omega": sig}, row, det_shape, window=(61, 7, 7), min_value=0.0
            )
            w = np.bincount(frame, weights=value, minlength=omega.size)
            mean = (w * omega).sum() / w.sum()
            moments[sig] = (mean, (w * (omega - mean) ** 2).sum() / w.sum() - 0.01**2 / 12)  # minus the frame width
        self.assertAlmostEqual(moments[0.05][0], moments[0.0][0], delta=1e-4)
        self.assertAlmostEqual(moments[0.05][1] - moments[0.0][1], 0.05**2, delta=0.02 * 0.05**2)


class TestOrientationSpread(unittest.TestCase):
    """An entry's "sig_rot" (intrinsic orientation spread) against a cloud of explicitly rotated sub-entries."""

    @staticmethod
    def _moments(frame, pixel, value, omega, nf):
        x = np.stack([omega[frame], pixel // nf, pixel % nf], 1).astype(float)  # (omega, slow, fast)
        w = value / value.sum()
        mean = w @ x
        d = x - mean
        return mean, (d * w[:, None]).T @ d

    def test_matches_rotated_sub_entries(self):
        """One entry with sig_rot adds the same to the peak's moments as 400 sub-entries with rotations drawn from it."""
        from scipy.spatial.transform import Rotation

        _, det_shape, geom, entries, hkl, omega_c = _single_peak_setup()
        sig = 2e-3  # rad per component, 0.11 degrees: comparable to this setup's beam spreads
        omega = (np.floor(omega_c / 0.04) + np.arange(-15, 16)) * 0.04 + 0.02
        row = make_row(omega, np.zeros_like(omega))
        win = (31, 17, 17)  # wide enough (+-4 sigma) that no peak is clipped: clipping would bias the moments
        ubi = entries["ubi"][:1]
        one = {"ubi": ubi, "pos": np.zeros((1, 3)), "density": np.ones(1), "sig_rot": np.array([sig])}
        rng = np.random.default_rng(0)
        rv = rng.normal(size=(200, 3)) * sig
        rv = np.concatenate([rv, -rv])  # antithetic pairs: first-order errors in the mean cancel
        R = Rotation.from_rotvec(rv).as_matrix()  # UB -> R UB, so UBI -> UBI R^T
        cloud = {"ubi": ubi[0] @ np.swapaxes(R, -1, -2), "pos": np.zeros((400, 3)), "density": np.full(400, 1 / 400)}
        base = {k: v for k, v in one.items() if k != "sig_rot"}
        res = []
        for ent in (base, one, cloud):
            frame, pixel, value, stats = render_row(ent, hkl[None], np.ones(1), geom, row, det_shape, window=win,
                                                    min_value=0.0)  # fmt: skip
            self.assertGreater(stats["captured"].min(), 0.999)
            res.append(self._moments(frame, pixel, value, omega, det_shape[1]))
        (_, c0), (m1, c1), (m2, c2) = res
        for a, b_, tol in zip(m1, m2, (2e-3, 0.02, 0.02)):  # degrees, pixels
            self.assertLess(abs(a - b_), tol)
        d1, d2 = c1 - c0, c2 - c0  # what the spread adds, over the beam's own spreads
        self.assertGreater(d2[0, 0], 0.5 * c0[0, 0])  # the spread matters in omega here
        self.assertAlmostEqual(d1[0, 0] / d2[0, 0], 1.0, delta=0.2)  # Monte Carlo, 400 samples: ~7% statistical
        self.assertLess(np.linalg.norm(d1 - d2) / np.linalg.norm(d2), 0.2)


    def test_absent_equals_zero(self):
        """No "sig_rot" and sig_rot = 0 render the same pixels."""
        _, det_shape, geom, entries, hkl, omega_c = _single_peak_setup()
        omega = (np.floor(omega_c / 0.05) + np.arange(-5, 6)) * 0.05 + 0.025
        row = make_row(omega, np.zeros_like(omega))
        base = {"ubi": entries["ubi"][:1], "pos": np.zeros((1, 3)), "density": np.ones(1)}
        a = render_row(base, hkl[None], np.ones(1), geom, row, det_shape, window=(5, 7, 7), min_value=0.0)
        b = render_row({**base, "sig_rot": np.zeros(1)}, hkl[None], np.ones(1), geom, row, det_shape, window=(5, 7, 7),
                       min_value=0.0)  # fmt: skip
        for x, y in zip(a[:3], b[:3]):
            np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)


class TestWindowClasses(unittest.TestCase):
    def test_broad_peak_gets_more_frames(self):
        """With max_frames, a peak broad in omega is rendered in a larger window and nothing is lost."""
        _, det_shape, geom, entries, hkl, omega_c = _single_peak_setup()
        one = {"ubi": entries["ubi"][:1], "pos": np.zeros((1, 3)), "density": np.ones(1)}
        omega = (np.floor(omega_c / 0.05) + np.arange(-40, 41)) * 0.05 + 0.025
        row = make_row(omega, np.zeros_like(omega))
        broad = {**geom, "sig_omega": 0.2}  # 4 frames
        win = (3, 21, 21)  # this peak is a few pixels wide on the detector: wide enough in pixels for all of it
        _, _, value_3, stats_3 = render_row(
            one, hkl[None], np.ones(1), broad, row, det_shape, window=win, min_value=0.0
        )
        _, _, value_31, stats_31 = render_row(
            one, hkl[None], np.ones(1), broad, row, det_shape, window=win, min_value=0.0, max_frames=31
        )
        self.assertLess(stats_3["captured"].max(), 0.5)
        self.assertGreater(stats_31["captured"].min(), 0.999)
        self.assertEqual(int(stats_31["window_frames"][0]), 31)
        self.assertGreater(value_31.sum(), 2 * value_3.sum())


class TestLargeBatch(unittest.TestCase):
    def test_matches_small_batches(self):
        """A batch of 4096 peaks renders the same as batches of 256.

        jaxlib 0.11's YNNPACK fusions miscompiled render_peaks for large batches (most peaks squeezed into one
        pixel); anri.utils.setup() turns them off, and conftest.py calls it.
        """
        import os

        from ImageD11.sinograms.tensor_map import TensorMap

        from anri.io import entries_from_tensormap, geom_from_pars

        data = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        tmap = TensorMap.from_h5(os.path.join(data, "phantoms", "quartz_flyxdm", "quartz_flyxdm_tmap.h5"))
        entries = {k: jnp.asarray(v[:300]) for k, v in entries_from_tensormap(tmap).items()}
        wl = 0.2845704100778472
        struc = anri.crystal.Structure.from_cif(os.path.join(data, "cif", "SiO2.cif"))
        struc.make_hkls(dsmax=1.0, wavelength=wl)
        table = struc.rings_table
        hkls = jnp.asarray(np.stack([table["h"], table["k"], table["l"]], 1), dtype=float)
        F2 = jnp.asarray(table["intensity"], dtype=float)
        pars = {
            "y_center": 1049.9, "y_size": 75.0, "tilt_y": 0.0, "z_center": 1116.5, "z_size": 75.0, "tilt_z": 0.0,
            "tilt_x": 0.0, "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1, "wavelength": wl,
        }  # fmt: skip
        geom = jax.tree.map(
            jnp.asarray, geom_from_pars(pars, 0.0, wl * 1e-4, 1e-4, 1e-4, sig_beam=0.5, voxel_size=0.787, sig_psf=0.3)
        )
        omega = np.arange(0.0, 180.0, 0.25) + 0.125
        row = jax.tree.map(jnp.asarray, make_row(omega, np.zeros_like(omega)))
        det_shape, window = (2162, 2068), (3, 7, 7)
        margin = jnp.array([4.0, 4.0, 0.5, 2.787])
        mask = select_peaks(entries["ubi"], entries["pos"], hkls, geom, row, margin, det_shape)
        e, h, b = (x[:4096].astype(np.int32) for x in np.nonzero(np.asarray(mask)))
        self.assertEqual(e.size, 4096)

        args = (entries, hkls, F2, geom, row, window, det_shape)
        big = np.asarray(render_peaks(e, h, b, *args)[2])
        small = np.concatenate([np.asarray(render_peaks(e[i : i + 256], h[i : i + 256], b[i : i + 256], *args)[2])
                                for i in range(0, 4096, 256)])  # fmt: skip
        np.testing.assert_allclose(big, small, rtol=0, atol=1e-9 * small.max())


class TestCheckRender(unittest.TestCase):
    """Rendered peaks against a Monte Carlo simulation of the beam spreads through the forward model."""

    def setUp(self):
        pars, self.det_shape, self.geom, self.entries, _, _ = _single_peak_setup()
        self.wavelength = pars["wavelength"]
        hkls = np.array(list(np.ndindex(7, 7, 7)), dtype=float) - 3
        self.hkls = hkls[(np.abs(hkls).sum(1) > 0) & (hkls.sum(1) % 2 == 0)]  # bcc
        omega = np.arange(0.0, 180.0, 0.25) + 0.125
        self.row = make_row(omega, np.zeros_like(omega))

    def test_narrow_peaks(self):
        # realistic spreads: peaks much narrower in omega than a frame, so slow and omega are strongly coupled within it
        geom = dict(self.geom, sig_wavelength=self.wavelength * 1e-4, sig_ky=1e-4, sig_kz=1e-4, sig_psf=0.3)
        r = check_render(self.entries, self.hkls, geom, self.row, self.det_shape, n_peaks=24, n_samples=200_000)
        self.assertGreater(len(r["entry"]), 10)
        # Monte Carlo noise is ~0.002 per cell; holding omega at its frame mean gave a median of 0.011
        self.assertLess(np.median(r["max_cell_error"]), 0.004)
        self.assertLess(r["max_cell_error"].max(), 0.012)

    def test_tilted_beam(self):
        # a beam turned 10 degrees in the horizontal plane and tilted 3 degrees up
        a, p = np.radians(3.0), np.radians(10.0)
        k = jnp.array([np.cos(a) * np.cos(p), np.cos(a) * np.sin(p), np.sin(a)])
        geom = dict(self.geom, sig_wavelength=self.wavelength * 1e-4, sig_ky=1e-4, sig_kz=1e-4, sig_psf=0.3, k_in_lab=k)
        r = check_render(self.entries, self.hkls, geom, self.row, self.det_shape, n_peaks=24, n_samples=200_000)
        self.assertGreater(len(r["entry"]), 10)
        self.assertLess(np.median(r["max_cell_error"]), 0.004)

    def test_broad_peaks_captured(self):
        # broad spreads (~2 px, truncated by the window): the fraction inside the window must match
        geom = dict(self.geom, sig_psf=0.3)
        r = check_render(
            self.entries, self.hkls, geom, self.row, self.det_shape, window=(7, 11, 11), n_peaks=24, n_samples=200_000
        )
        self.assertGreater(len(r["entry"]), 10)
        self.assertLess(r["captured"].min(), 0.995)  # the window does truncate these peaks
        np.testing.assert_allclose(r["captured"], r["captured_mc"], atol=0.01)


class TestGuessBatchSize(unittest.TestCase):
    def setUp(self):
        _, self.det_shape, self.geom, self.entries, hkl, _ = _single_peak_setup()
        self.hkls, self.F2 = hkl[None], np.ones(1)
        omega = np.arange(0.0, 180.0, 0.25) + 0.125
        self.row = make_row(omega, np.zeros_like(omega))
        self.nd = len(jax.devices())

    def guess(self, **kwargs):
        try:
            return guess_batch_size(self.entries, self.hkls, self.F2, self.geom, self.row, self.det_shape, **kwargs)
        except RuntimeError as err:  # some jaxlib builds have no usable memory analysis (tested below with mocks)
            self.skipTest(str(err))

    def test_power_of_two_per_device(self):
        batch = self.guess()
        per_device = batch // self.nd
        self.assertEqual(batch % self.nd, 0)
        self.assertEqual(per_device & (per_device - 1), 0)
        self.assertGreaterEqual(per_device, 64)
        # a bigger window costs more memory per peak
        self.assertLessEqual(self.guess(window=(5, 11, 11)), batch)

    def test_minimum(self):
        self.assertEqual(self.guess(memory_fraction=1e-12), 64 * self.nd)

    def test_device_memory(self):
        # GPUs: free device memory, without the host copy of the output that CPU devices share memory with
        from unittest import mock

        with mock.patch("anri.fwd._impl.render._free_memory", return_value=(2**30, True)):
            host = self.guess()
        with mock.patch("anri.fwd._impl.render._free_memory", return_value=(2**30, False)):
            device = self.guess()
        self.assertGreaterEqual(device, host)

    def test_no_memory_analysis(self):
        from unittest import mock

        args = (self.entries, self.hkls, self.F2, self.geom, self.row, self.det_shape)
        fake = mock.MagicMock()
        analysis = fake.lower.return_value.compile.return_value.memory_analysis
        analysis.return_value = None  # none at all
        with mock.patch("anri.fwd._impl.render._render_sharded", fake), self.assertRaises(RuntimeError):
            guess_batch_size(*args)
        sizes = mock.MagicMock(temp_size_in_bytes=1000, argument_size_in_bytes=0, output_size_in_bytes=0)
        sizes.alias_size_in_bytes = 0
        analysis.return_value = sizes  # the same for any batch
        with mock.patch("anri.fwd._impl.render._render_sharded", fake), self.assertRaises(RuntimeError):
            guess_batch_size(*args)

    def test_free_memory(self):
        free, host = _free_memory(jax.devices())
        self.assertTrue(host)
        self.assertGreater(free, 0)

        class FakeGPU:
            platform = "gpu"

            def __init__(self, in_use):
                self.in_use = in_use

            def memory_stats(self):
                return {"bytes_limit": 1000, "bytes_in_use": self.in_use}

        self.assertEqual(_free_memory([FakeGPU(100), FakeGPU(300)]), (700.0, False))


class TestMerge(unittest.TestCase):
    """_merge (used on GPUs) sums duplicates on the device; it must match _compact plus a host merge (CPUs)."""

    def test_matches_host_merge(self):
        rng = np.random.default_rng(0)
        n = 5000
        frame = rng.integers(-1, 6, n).astype(np.int32)  # -1: unused cell
        pixel = rng.integers(0, 40, n).astype(np.int32)  # few pixels, so many duplicates
        value = rng.exponential(1.0, n)
        min_value = jnp.asarray(0.5)
        fr, px, val, count = (np.asarray(x) for x in _merge(frame, pixel, value, min_value))
        fr, px, val = fr[:count], px[:count], val[:count]
        c_fr, c_px, c_val, c_count = (np.asarray(x) for x in _compact(frame, pixel, value, min_value))
        key = c_fr[:c_count].astype(np.int64) * 40 + c_px[:c_count]
        uniq, inverse = np.unique(key, return_inverse=True)
        expected = np.bincount(inverse, weights=c_val[:c_count])
        np.testing.assert_array_equal(fr.astype(np.int64) * 40 + px, uniq)  # sorted by (frame, pixel), unique
        np.testing.assert_allclose(val, expected, rtol=1e-6)
        self.assertTrue(np.all(fr >= 0))


class TestPolarisationDirection(unittest.TestCase):
    def test_rotating_the_setup(self):
        # rotating beam and scattered ray together about lab z, or about lab y (a beam tilted up or down, e.g.
        # grazing incidence), keeps horizontal polarisation horizontal: the factor must not change
        rng = np.random.default_rng(0)
        k_in = jnp.array([1.0, 0.0, 0.0])
        k_out = jnp.asarray(rng.normal(size=(50, 3)) + [3.0, 0.0, 0.0])
        pol = jax.vmap(polarisation, in_axes=(None, 0, None))
        for factor in (1.0, 0.9, 0.0):
            reference = pol(k_in, k_out, factor)
            for R in (anri.geom.rot_z(40.0), anri.geom.rot_y(-25.0)):
                np.testing.assert_allclose(pol(R @ k_in, k_out @ R.T, factor), reference, rtol=1e-12)


class TestMakeRow(unittest.TestCase):
    def test_unsorted_with_transmission(self):
        omega = np.array([0.75, 0.25, 1.25, 1.75])  # file order is not omega order
        row = make_row(omega, np.full(4, 2.0), transmission=np.array([0.9, 0.8, 0.7, 0.6]))
        np.testing.assert_array_equal(row["order"], [1, 0, 2, 3])
        np.testing.assert_allclose(row["omega_edges"], [0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(row["transmission_sorted"], [0.8, 0.9, 0.7, 0.6])
        self.assertEqual((row["omega_min"], row["omega_max"], row["dty_min"], row["dty_max"]), (0.0, 2.0, 2.0, 2.0))


class TestEmptyRow(unittest.TestCase):
    def test_no_peaks(self):
        _, det_shape, geom, entries, hkl, _ = _single_peak_setup()
        omega = np.arange(0.0, 10.0, 0.25) + 0.125
        row = make_row(omega, np.full_like(omega, 1000.0))  # the beam misses the grain
        frame, pixel, value, stats = render_row(entries, hkl[None], np.ones(1), geom, row, det_shape)
        self.assertEqual((frame.size, pixel.size, value.size, stats["n_peaks"], stats["captured"].size), (0,) * 5)


class TestBoxBeam(unittest.TestCase):
    def test_dct_spot(self):
        """A cube of voxels in a box beam bigger than it, near-field detector: every voxel is lit equally, and the
        spot is the cube projected along the diffracted beam."""
        from anri.fwd._impl.render import _peak_cov
        from anri.io import geom_from_pars

        wl = 0.2845704100778472
        pars = {
            "y_center": 1023.5, "z_center": 1023.5, "y_size": 1.0, "z_size": 1.0, "tilt_x": 0.0, "tilt_y": 0.0,
            "tilt_z": 0.0, "distance": 5000.0, "o11": -1, "o12": 0, "o21": 0, "o22": -1, "wavelength": wl,
        }  # fmt: skip
        det_shape = (2048, 2048)
        geom = geom_from_pars(
            pars, 0.0, wl * 1e-4, 1e-4, 1e-4, sig_beam=0.5, voxel_size=1.0, sig_psf=0.3,
            width_beam=1000.0, sig_beam_v=0.5, width_beam_v=1000.0, voxel_3d=True,
        )  # fmt: skip
        g = jax.tree.map(jnp.asarray, geom)
        B = anri.crystal.lpars_to_B(jnp.array([2.8665, 2.8665, 2.8665, 90.0, 90.0, 90.0]))
        ubi = jnp.linalg.inv(anri.geom.rot_z(25.0) @ anri.geom.rot_x(35.0) @ anri.geom.rot_y(10.0) @ B)
        i, j, k = np.mgrid[0:6, 0:6, 0:6]
        pos = (np.stack([i.ravel(), j.ravel(), k.ravel()], 1) - 2.5).astype(float)
        entries = {"ubi": np.repeat(np.asarray(ubi)[None], len(pos), 0), "pos": pos, "density": np.ones(len(pos))}

        # a 110-type spot well inside the detector, from the grain's centre
        for hkl in np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, -1, 0], [1, 0, -1], [0, 1, -1]], float):
            c, valid = _centroid(ubi, jnp.zeros(3), jnp.asarray(hkl), 1.0, g["wavelength"], 0.0, 0.0, 0.0, g)
            if valid and 200 < c[0] < 1850 and 200 < c[1] < 1850:
                break
        omega = np.arange(float(c[2]) - 2.0, float(c[2]) + 2.0, 0.25) + 0.125
        row = make_row(omega, np.zeros_like(omega))
        _, pixel, value, stats = render_row(
            entries, hkl[None], np.ones(1), geom, row, det_shape, window=(3, 9, 9), min_value=0.0
        )
        self.assertEqual(stats["n_peaks"], len(pos))  # every voxel is in the beam
        self.assertGreater(stats["captured"].min(), 0.999)

        # every voxel is lit by the same fraction of the beam: its volume over the beam's cross-section
        lab = jax.vmap(sample_to_lab, in_axes=(0,) + (None,) * 5)(jnp.asarray(pos), c[2], 0.0, 0.0, 0.0, 0.0)
        w = jax.vmap(beam_weight, in_axes=(0, None, None))(lab, c[2], g)
        np.testing.assert_allclose(w, 1.0 / 1000.0**2, rtol=1e-9)

        # the spot: the voxel centres projected along the diffracted beam, blurred by the point spread and spreads
        proj = jax.vmap(_centroid, in_axes=(None, 0) + (None,) * 7)(
            ubi, jnp.asarray(pos), jnp.asarray(hkl), 1.0, g["wavelength"], 0.0, 0.0, 0.0, g
        )[0]
        proj = np.asarray(proj)[:, :2]
        cov6 = np.asarray(_peak_cov(ubi, jnp.zeros(3), jnp.asarray(hkl), 1.0, 0.0, g))
        blur = np.array([[cov6[0], cov6[3]], [cov6[3], cov6[1]]]) + np.eye(2) * (0.3**2 + 1e-4)
        s, f = pixel // det_shape[1], pixel % det_shape[1]
        sf = np.stack([s, f], 1).astype(float)
        mean = (value[:, None] * sf).sum(0) / value.sum()
        cov = ((sf - mean).T * value) @ (sf - mean) / value.sum()
        np.testing.assert_allclose(mean, proj.mean(0), atol=0.02)
        np.testing.assert_allclose(cov, np.cov(proj.T, bias=True) + blur + np.eye(2) / 12, atol=0.02 * np.trace(cov))
