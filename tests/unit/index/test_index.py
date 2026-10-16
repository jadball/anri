import os
import tempfile
import unittest

import jax.numpy as jnp
import numpy as np

import anri.crystal
import anri.geom
import anri.index as ix
import anri.io
import anri.maths.rotation

PARS = {
    "y_center": 1049.9, "y_size": 75.0, "tilt_y": 0.0, "z_center": 1116.5, "z_size": 75.0, "tilt_z": 0.0,
    "tilt_x": 0.0, "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1,
    "wavelength": 12.398419843320026 / 43.0, "wedge": 0.0, "chi": 0.0,
}  # fmt: skip
A_FE = 2.8665


def geometry(sig_beam=0.5):
    g = anri.io.imaged11.geom_from_pars(
        PARS, 0.0, PARS["wavelength"] * 1e-4, 5e-5, 5e-5, sig_beam=sig_beam, voxel_size=1.0
    )
    return {
        k: jnp.asarray(v, jnp.float32) if np.issubdtype(np.asarray(v).dtype, np.floating) else v for k, v in g.items()
    }


def iron():
    lpars = np.array([A_FE] * 3 + [90.0] * 3)
    B = anri.crystal.structure.B_matrix(lpars)
    return (
        (lpars, 229),
        B.astype(np.float32),
        anri.crystal.orientation.laue_rotations(anri.crystal.structure.symmetry_matrices(229), B),
    )


def rotations(n, seed):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return anri.maths.rotation.quat_to_mat(q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


class TestRings(unittest.TestCase):
    def test_bcc_rings(self):
        c, _, _ = iron()
        rings = ix.data.ring_table(*c, PARS["wavelength"], 3)
        h = rings["hkls"]
        self.assertTrue(np.all(np.sum(h, 1) % 2 == 0))  # I-centred: h + k + l even
        self.assertEqual(sorted(np.bincount(rings["ring_j"]) // 2), [6, 12, 24])  # {200}, {110}, {211}
        np.testing.assert_allclose(np.sort(np.sum(h * h, 1))[[0, 12, 18]], [2, 4, 6])

    def test_ring_widths(self):
        ring_tth = np.array([5.0, 6.0])
        lo, step = 4.5, 0.002
        x = lo + (np.arange(1000) + 0.5) * step
        prof = 100 * np.exp(-0.5 * ((x - 5.01) / 0.02) ** 2) + 50 * np.exp(-0.5 * ((x - 6.0) / 0.04) ** 2) + 1.0
        off, hw = ix.data.ring_widths(prof, lo, step, ring_tth)
        np.testing.assert_allclose(off, [0.01, 0.0], atol=0.002)
        np.testing.assert_allclose(hw, [1.96 * 0.02, 1.96 * 0.04], rtol=0.1)  # 95% of a Gaussian


class TestTolerances(unittest.TestCase):
    def test_first_order_bound(self):
        """A rotation by delta moves each reflection by at most the tolerances (|sin eta| > 0.3)."""
        c, B, _ = iron()
        rings = ix.data.ring_table(*c, PARS["wavelength"], 3)
        geom = geometry()
        U0 = rotations(200, 0)
        axis = np.random.default_rng(1).normal(size=(200, 3))
        dU = anri.maths.rotation.quat_to_mat(np.column_stack([np.full(200, np.cos(np.radians(0.5))),
                                                       np.sin(np.radians(0.5)) * axis / np.linalg.norm(axis, axis=1, keepdims=True)]))  # fmt: skip
        e0, o0, k0, _ = ix.search.predict(jnp.asarray(U0), jnp.asarray(B), jnp.asarray(rings["hkls"]), geom)
        e1, o1, k1, _ = ix.search.predict(
            jnp.asarray((dU @ U0).astype(np.float32)), jnp.asarray(B), jnp.asarray(rings["hkls"]), geom
        )
        te, to = ix.search.match_tolerances(
            e0, jnp.asarray(rings["ring_j"]), jnp.asarray(rings["tth"]), jnp.zeros(3), 1.0, 0.0
        )
        use = np.asarray(k0 & k1) & (np.abs(np.sin(np.radians(np.asarray(e0)))) > 0.3)
        de = np.abs((np.asarray(e1) - np.asarray(e0) + 180) % 360 - 180)
        do = np.abs((np.asarray(o1) - np.asarray(o0) + 180) % 360 - 180)
        self.assertLessEqual((de / np.asarray(te))[use].max(), 1.02)
        self.assertLessEqual((do / np.asarray(to))[use].max(), 1.02)

    def test_completeness_boxes(self):
        lit = np.zeros((1, 360, 180), bool)
        lit[0, 190, 50] = True  # eta 10..11, omega 50..51
        table = ix.data.lit_table(jnp.asarray(lit))
        eta = jnp.array([[12.2, 10.5, 10.5]])
        om = jnp.array([[50.5, 52.7, 50.5]])
        ok = jnp.ones((1, 3), bool)

        def comp(te, to):
            c, n = ix.search.completeness(table, eta, om, ok, jnp.zeros(3, jnp.int32), jnp.full((1, 3), te), jnp.full((1, 3), to),
                                  0.0, 1.0, 1.0, 360, 180)  # fmt: skip
            self.assertEqual(int(n[0]), 3)
            return float(c[0])

        self.assertAlmostEqual(comp(0.1, 0.1), 1 / 3, places=6)  # only the prediction in the lit bin
        self.assertAlmostEqual(comp(2.0, 0.1), 2 / 3, places=6)  # the eta neighbour too
        self.assertAlmostEqual(comp(2.0, 2.5), 1.0, places=6)  # and the omega neighbour


class TestOccupancy(unittest.TestCase):
    def test_adjoint(self):
        """<A f, r> = <f, A^T r>."""
        c, B, _ = iron()
        rings = ix.data.ring_table(*c, PARS["wavelength"], 2)
        geom = geometry()
        pred = ix.search.predictions(rotations(16, 2), B, rings, geom, 0.2)
        rng = np.random.default_rng(3)
        pos = jnp.asarray(np.column_stack([rng.uniform(-4, 4, (32, 2)), np.zeros(32)]), jnp.float32)
        scan = {"y0": 0.0, "dty0": -6.0, "ystep": 1.0, "n_rows": 13, "om0": 0.0}
        dims = (1.0, 1.0, 360, 180)
        n_cells = 2 * 360 * 180 * 13
        cand = jnp.asarray(rng.integers(0, 16, (32, 4)), jnp.int32)
        f = jnp.asarray(rng.uniform(size=(32, 4)), jnp.float32)
        r = jnp.asarray(rng.uniform(size=n_cells), jnp.float32)
        Af = ix.occupancy.forward(f, cand, pred, jnp.asarray(rings["ring_j"]), pos, scan, dims, n_cells, 8)
        ATr = ix.occupancy.backward(r, cand, pred, jnp.asarray(rings["ring_j"]), pos, scan, dims, 8)
        self.assertGreater(float(jnp.sum(Af)), 0)
        np.testing.assert_allclose(float(jnp.sum(Af * r)), float(jnp.sum(f * ATr)), rtol=1e-4)

    def test_adjoint_beam_profile(self):
        """<A f, r> = <f, A^T r> with a beam profile over the rows too."""
        c, B, _ = iron()
        rings = ix.data.ring_table(*c, PARS["wavelength"], 2)
        pred = ix.search.predictions(rotations(16, 2), B, rings, geometry(), 0.2)
        rng = np.random.default_rng(3)
        pos = jnp.asarray(np.column_stack([rng.uniform(-4, 4, (32, 2)), np.zeros(32)]), jnp.float32)
        scan = {"y0": 0.0, "dty0": -6.0, "ystep": 1.0, "n_rows": 13, "om0": 0.0, "sig_beam": 0.6, "width_beam": 0.0,
                "voxel": 1.0}  # fmt: skip
        dims = (1.0, 1.0, 360, 180, ix.occupancy.beam_rows(scan))
        n_cells = 2 * 360 * 180 * 13
        cand = jnp.asarray(rng.integers(0, 16, (32, 4)), jnp.int32)
        f = jnp.asarray(rng.uniform(size=(32, 4)), jnp.float32)
        r = jnp.asarray(rng.uniform(size=n_cells), jnp.float32)
        Af = ix.occupancy.forward(f, cand, pred, jnp.asarray(rings["ring_j"]), pos, scan, dims, n_cells, 8)
        ATr = ix.occupancy.backward(r, cand, pred, jnp.asarray(rings["ring_j"]), pos, scan, dims, 8)
        self.assertGreater(float(jnp.sum(Af)), 0)
        np.testing.assert_allclose(float(jnp.sum(Af * r)), float(jnp.sum(f * ATr)), rtol=1e-4)

    def test_beam_profile_rows(self):
        """Row weights: sum 1, centred on the voxel, variance of the beam plus the voxel's chord (omega 0: a box)."""
        sig, ystep, vox = 0.8, 1.0, 1.0
        scan = {"y0": 0.0, "dty0": -10.0, "ystep": ystep, "n_rows": 21, "om0": -180.0, "sig_beam": sig,
                "width_beam": 0.0, "voxel": vox}  # fmt: skip
        n_beam = ix.occupancy.beam_rows(scan)
        n_k, n_e, n_o = scan["n_rows"], 360, 360
        for x, y in ((0.0, 0.3), (0.0, -2.45), (0.0, 4.0)):
            pos = jnp.asarray([[x, y, 0.0]], jnp.float32)
            one = jnp.ones((1, 1, 1))
            idx, wt = ix.occupancy.system(one * 10.0, one * 0.0, one > 0, one, jnp.asarray([0]), pos, scan, 1.0, 1.0, n_e, n_o,
                                n_beam)  # fmt: skip
            idx, wt = np.asarray(idx).ravel(), np.asarray(wt).ravel()
            rows = idx[idx >= 0] % n_k
            w = np.bincount(rows, weights=wt[idx >= 0], minlength=n_k)
            dty = scan["dty0"] + np.arange(n_k) * ystep
            centre = scan["y0"] - y  # at omega 0 the voxel is in the beam at dty = y0 - y
            self.assertAlmostEqual(w.sum(), 1.0, delta=0.01)
            self.assertAlmostEqual((w * dty).sum() / w.sum(), centre, delta=0.01)
            var = (w * (dty - centre) ** 2).sum() / w.sum()
            self.assertAlmostEqual(var, sig**2 + vox**2 / 12, delta=0.05 * (sig**2 + vox**2 / 12))

    def test_dty_offsets(self):
        """Rows away from their nominal dty: a helical scan's offsets, and system putting a voxel in the right rows."""
        n_rows, n_fr, ystep = 5, 360, 1.0
        om = np.tile((np.arange(n_fr) + 0.5) * 360.0 / n_fr, (n_rows, 1))
        dty = np.arange(n_rows)[:, None] * ystep + ystep * (om / 360.0 - 0.5)  # one step per turn, row mean on the row
        edges = (np.arange(n_rows + 1) - 0.5) * ystep
        dd = ix.data.dty_offsets(dty, om, edges, 1, 0.0, 1.0, 360, 0.0, ystep)
        np.testing.assert_allclose(dd[2], ystep * ((np.arange(360) + 0.5) / 360.0 - 0.5), atol=1e-9)
        for n_beam in (0, 7):  # the 2-row model and a beam profile
            # a beam wide enough that rows sample its profile evenly (a narrow one, sampled at whole rows, has its
            # centroid a little off its centre)
            scan = {"y0": 0.0, "dty0": -10.0, "ystep": ystep, "n_rows": 21, "om0": -180.0, "sig_beam": 0.8,
                    "width_beam": 0.0, "voxel": 1.0}  # fmt: skip
            one, pos = jnp.ones((1, 1, 1)), jnp.asarray([[0.0, 0.3, 0.0]], jnp.float32)
            rows = []
            for ddty in (None, np.full((21, 360), 0.3)):
                sc = scan if ddty is None else {**scan, "ddty": jnp.asarray(ddty, jnp.float32)}
                idx, wt = ix.occupancy.system(one * 10.0, one * 0.0, one > 0, one, jnp.asarray([0]), pos, sc, 1.0, 1.0, 360, 360,
                                    n_beam)  # fmt: skip
                idx, wt = np.asarray(idx).ravel(), np.asarray(wt).ravel()
                w = np.bincount(idx[idx >= 0] % 21, weights=wt[idx >= 0], minlength=21)
                rows.append((w * (scan["dty0"] + np.arange(21) * ystep)).sum() / w.sum())
            # every row 0.3 further along dty: the voxel is seen by rows whose nominal dty is 0.3 lower
            self.assertAlmostEqual(rows[1] - rows[0], -0.3, delta=0.02)

    def test_exposure(self):
        """Frames binned by their own dty: bins without frames or with two rotations' worth, and system scaling by it."""
        n_rows, n_fr, ystep = 6, 360, 1.0
        om = np.tile((np.arange(n_fr) + 0.5) * 360.0 / n_fr, (n_rows, 1))
        # 1.2 steps per turn, rows a little off their nominal dty (as measured): near a row's edges a frame lands in
        # the next row, so some (row, omega) bins get two rotations' frames and some none
        jitter = np.array([0.0, 0.3, -0.2, 0.25, -0.3, 0.1])[:, None]
        dty = (np.arange(n_rows)[:, None] + jitter) * ystep + 1.2 * ystep * (om / 360.0 - 0.5)
        edges = (np.arange(n_rows + 1) - 0.5) * ystep
        _, ex = ix.data.dty_offsets(dty, om, edges, 1, 0.0, 1.0, 360, 0.0, ystep, bin_dty=dty, return_exposure=True)
        self.assertTrue(np.any(ex[1:-1] == 0) and np.any(ex[1:-1] == 2))  # gaps and overlaps between rows
        _, ex1 = ix.data.dty_offsets(dty, om, edges, 1, 0.0, 1.0, 360, 0.0, ystep, return_exposure=True)
        np.testing.assert_allclose(ex1, 1.0)  # binned by the row means: one rotation per row
        scan = {"y0": 0.0, "dty0": -10.0, "ystep": ystep, "n_rows": 21, "om0": -180.0}
        one, pos = jnp.ones((1, 1, 1)), jnp.asarray([[0.0, 0.3, 0.0]], jnp.float32)
        for ex2, total in ((np.ones((21, 360)), 1.0), (np.zeros((21, 360)), 0.0), (np.full((21, 360), 2.0), 2.0)):
            sc = {**scan, "exposure": jnp.asarray(ex2, jnp.float32)}
            _, wt = ix.occupancy.system(
                one * 10.0, one * 0.0, one > 0, one, jnp.asarray([0]), pos, sc, 1.0, 1.0, 360, 360
            )
            self.assertAlmostEqual(float(jnp.sum(wt)), total, places=5)

    def test_squarem(self):
        """mlem (SQUAREM): the deviance never rises, and it fits at least as well as plain MLEM with the same updates."""
        c, B, _ = iron()
        rings = ix.data.ring_table(*c, PARS["wavelength"], 2)
        pred = ix.search.predictions(rotations(16, 2), B, rings, geometry(), 0.2)
        rng = np.random.default_rng(5)
        pos = jnp.asarray(np.column_stack([rng.uniform(-4, 4, (32, 2)), np.zeros(32)]), jnp.float32)
        scan = {"y0": 0.0, "dty0": -6.0, "ystep": 1.0, "n_rows": 13, "om0": 0.0}
        dims = (1.0, 1.0, 360, 180)
        n_cells = 2 * 360 * 180 * 13
        ring_j = jnp.asarray(rings["ring_j"])
        cand = jnp.asarray(np.argsort(rng.uniform(size=(32, 16)), 1)[:, :4], jnp.int32)
        f_true = jnp.asarray(rng.uniform(size=(32, 4)) * (rng.uniform(size=(32, 4)) > 0.5), jnp.float32)
        d = ix.occupancy.forward(f_true, cand, pred, ring_j, pos, scan, dims, n_cells, 8)
        f0 = jnp.ones((32, 4), jnp.float32)
        devs = []
        f = ix.occupancy.mlem(d, cand, pred, ring_j, pos, scan, dims, f0, 10, 8, log=lambda m: None,
                    callback=lambda it, f_, dev: devs.append(dev))  # fmt: skip
        self.assertEqual(len(devs), 10)
        self.assertTrue(np.all(np.diff(devs) <= 1e-6 * devs[0]))
        dev_sq = float(
            ix.occupancy.deviance(d, ix.occupancy.forward(f, cand, pred, ring_j, pos, scan, dims, n_cells, 8), 0.0)
        )
        norm = ix.occupancy.backward(jnp.ones(n_cells), cand, pred, ring_j, pos, scan, dims, 8)
        fp = f0
        for _ in range(30):  # plain MLEM, as many updates
            Af = ix.occupancy.forward(fp, cand, pred, ring_j, pos, scan, dims, n_cells, 8)
            fp = (
                fp
                * ix.occupancy.backward(ix.occupancy.censored_ratio(d, Af, 0.0), cand, pred, ring_j, pos, scan, dims, 8)
                / norm
            )
        dev_plain = float(
            ix.occupancy.deviance(d, ix.occupancy.forward(fp, cand, pred, ring_j, pos, scan, dims, n_cells, 8), 0.0)
        )
        self.assertLessEqual(dev_sq, dev_plain * 1.0001)

    def test_censored_ratio(self):
        d = jnp.asarray([4.0, 0.0, 0.0, 0.0])
        mu = jnp.asarray([2.0, 1.0, 10.0, 0.0])
        np.testing.assert_allclose(np.asarray(ix.occupancy.censored_ratio(d, mu, 0.0)), [2.0, 0.0, 0.0, 0.0])
        # empty bins: agreeing below the censoring level, censor / 2 above it
        np.testing.assert_allclose(np.asarray(ix.occupancy.censored_ratio(d, mu, 5.0)), [2.0, 1.0, 0.25, 0.0])
        plain = 2 * float(jnp.sum(jnp.where(d > 0, d * jnp.log(d / jnp.maximum(mu, 1e-30)), 0.0) - d + mu))
        self.assertAlmostEqual(float(ix.occupancy.deviance(d, mu, 0.0)), plain, places=4)
        self.assertAlmostEqual(float(ix.occupancy.deviance(d, mu, 5.0)), plain - 2 * (1.0 + 5.0), places=4)

    def test_masks(self):
        yy, xx = np.indices((41, 41))
        disk = ((yy - 20) ** 2 + (xx - 20) ** 2 <= 12**2).astype(float)
        image = disk * 10 + np.random.default_rng(0).normal(0, 0.5, disk.shape)
        m = ix.mask.threshold_mask(image)
        self.assertFalse(np.any(disk.astype(bool) & ~m))  # nothing of the sample missed
        self.assertLess(
            np.sum(m & ~disk.astype(bool)), 0.1 * disk.sum()
        )  # the hull of a pixelated disk is a little more
        sq = ix.mask.polygon_mask((10, 10), [(2, 3), (7, 3), (7, 8), (2, 8)])  # (x, y) = (column, row)
        self.assertTrue(sq[5, 4] and not sq[1, 4] and not sq[5, 9])

    def test_tilted_geometry_rows(self):
        """With wedge, chi and a beam off lab x, system puts a voxel in the rows where the renderer's beam crosses it."""
        k, wedge, chi, y0 = jnp.array([1.0, 0.1, 0.05]), 3.0, -2.0, 0.4
        rng = np.random.default_rng(3)
        pos = jnp.asarray(np.column_stack([rng.uniform(-5, 5, (6, 2)), np.zeros(6)]), jnp.float32)
        om = jnp.asarray(rng.uniform(-180, 180, (6, 1, 1)), jnp.float32)
        dty = anri.geom.gonio.find_dty_for_beam_xy(pos[:, None, None, :], k, om, wedge, chi, y0)[:, 0, 0]
        for v, o, d in zip(pos, om[:, 0, 0], dty):
            across = anri.geom.beam.beam_offsets(anri.geom.gonio.sample_to_lab(v, o, wedge, chi, d, y0), k)[0]
            self.assertAlmostEqual(float(across), 0.0, places=4)  # the renderer's beam passes through the voxel
        scan = {"y0": y0, "dty0": -10.0, "ystep": 1.0, "n_rows": 21, "om0": -180.0, "k_in_lab": k, "wedge": wedge,
                "chi": chi}  # fmt: skip
        one = jnp.ones((6, 1, 1))
        idx, wt = ix.occupancy.system(one * 10.0, om, one > 0, one, jnp.asarray([0]), pos, scan, 1.0, 1.0, 360, 360)
        rows = np.asarray(idx) % 21
        centre = (np.asarray(wt) * (scan["dty0"] + rows)).sum((1, 2, 3)) / np.asarray(wt).sum((1, 2, 3))
        np.testing.assert_allclose(centre, dty, atol=1e-4)

    def test_coarsen_scan(self):
        """Coarse rows: the geometry of summed rows, and ddty as dty_offsets gives it for the summed rows directly."""
        n_fr, ystep = 360, 1.0
        for n_rows, g in ((6, 2), (5, 2), (7, 3)):  # with a partial last group too
            om = np.tile((np.arange(n_fr) + 0.5) * 360.0 / n_fr, (n_rows, 1))
            dty = np.arange(n_rows)[:, None] * ystep + 1.2 * ystep * (om / 360.0 - 0.5)  # helical, 1.2 steps a turn
            edges = (np.arange(n_rows + 1) - 0.5) * ystep
            fine = {"dty0": 0.0, "ystep": ystep, "n_rows": n_rows, "voxel": ystep, "width_beam": 0.0,
                    "ddty": jnp.asarray(ix.data.dty_offsets(dty, om, edges, 1, 0.0, 1.0, 360, 0.0, ystep))}  # fmt: skip
            c = ix.data.coarsen_scan(fine, g)
            self.assertEqual((c["n_rows"], c["ystep"], c["voxel"], c["width_beam"]), (-(-n_rows // g), g, g, g - 1))
            self.assertAlmostEqual(c["dty0"], 0.5 * (g - 1) * ystep)
            direct = ix.data.dty_offsets(dty, om, edges, g, 0.0, 1.0, 360, c["dty0"], c["ystep"])
            np.testing.assert_allclose(c["ddty"], direct, atol=1e-6)

    def test_coarsen_rows(self):
        H = np.arange(2 * 7, dtype=float)
        Hc, n = ix.data.coarsen_rows(H, 7, 3)
        self.assertEqual(n, 3)
        np.testing.assert_allclose(np.asarray(Hc), [0 + 1 + 2, 3 + 4 + 5, 6, 7 + 8 + 9, 10 + 11 + 12, 13])


class TestPopulations(unittest.TestCase):
    def test_parent_twin_and_decoy(self):
        _, _, ops = iron()
        U0 = rotations(1, 4)[0].astype(float)
        twin = anri.maths.rotation.quat_to_mat(
            np.r_[np.cos(np.radians(30)), np.sin(np.radians(30)) * np.ones(3) / np.sqrt(3)]
        )
        small = [anri.maths.rotation.quat_to_mat(np.r_[np.cos(np.radians(a / 2)), np.sin(np.radians(a / 2)) * np.array(v)])
                 for a, v in ((0.8, [1, 0, 0]), (0.8, [0, 1, 0]), (0.8, [0, 0, -1]))]  # fmt: skip
        U_list = np.stack([U0, *(s @ U0 for s in small), twin @ U0, rotations(1, 5)[0]])  # parent x 4, twin, decoy
        f = np.array([[0.3, 0.2, 0.15, 0.15, 0.17, 0.03]])
        cand = np.arange(6)[None]
        frac, U, spread, n = ix.populations.group(f, cand, U_list, ops, 2.0)
        np.testing.assert_allclose(frac[0, :3], [0.8, 0.17, 0.03], atol=1e-5)
        np.testing.assert_array_equal(n[0, :3], [4, 1, 1])
        self.assertLess(anri.crystal.orientation.disorientation(U[0, :1], U0[None], ops)[0], 0.5)
        self.assertLess(anri.crystal.orientation.disorientation(U[0, 1:2], (twin @ U0)[None], ops)[0], 0.05)  # float32
        self.assertGreater(spread[0, 0], 0.2)


class TestEndToEnd(unittest.TestCase):
    def test_two_grains(self):
        """Simulate two grains to ImageD11 files, then index them from scratch."""
        c, B, ops = iron()
        wl = PARS["wavelength"]
        rings = ix.data.ring_table(*c, wl, 3)
        U_true = rotations(2, 7)
        i, j = np.mgrid[0:6, 0:6]
        pos = np.stack([i.ravel() - 2.5, j.ravel() - 2.5, np.zeros(36)], 1)
        grain = (pos[:, 0] > 0).astype(int)
        entries = {"ubi": np.linalg.inv(U_true[grain] @ B), "pos": pos, "density": np.full(36, 30.0)}
        geom = geometry()
        omega, dty = anri.io.imaged11.motor_grid((0.0, 180.0), 0.5, (-5.0, 5.0), 1.0)
        a = A_FE
        cell = {"cell__a": a, "cell__b": a, "cell__c": a, "cell_alpha": 90.0, "cell_beta": 90.0, "cell_gamma": 90.0,
                "cell_lattice_[P,A,B,C,I,F,R]": 229}  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            sparse = os.path.join(tmp, "sparse.h5")
            anri.io.imaged11.simulate_sparse(sparse, entries, rings["hkls"], np.ones(len(rings["hkls"])), geom, omega, dty,
                                    (2162, 2068), window=(5, 9, 9))  # fmt: skip
            parfile = anri.io.imaged11.write_pars(os.path.join(tmp, "pars"), PARS, {"Fe": cell})
            dsfile = anri.io.imaged11.write_dataset(sparse, tmp, "fe", "sim", y0=0.0, parfile=parfile)
            ds = anri.io.imaged11.read_dataset(dsfile)
            self.assertEqual(os.path.realpath(ds["sparsefile"]), os.path.realpath(sparse))
            _, phase, cell_read = anri.io.imaged11.read_pars_json(ds["parfile"])
            self.assertEqual(phase, "Fe")
            self.assertEqual(cell_read["cell_lattice_[P,A,B,C,I,F,R]"], 229)

            def stream():
                return anri.io.imaged11.stream_sparse(
                    sparse, ds["ybinedges"], ds["omegamotor"], ds["dtymotor"], 1 << 16
                )

            _, hw = ix.data.ring_profile(stream(), geom, rings["tth"], 1 << 16)
            rings["hw"] = hw
            nk = len(ds["ybincens"])
            om0 = float(ds["obinedges"][0])
            H_lit, H = ix.data.histogram_pixels(
                stream(), geom, rings["tth"], 0.1, om0, [((0.5, 0.5, 720, 360), 1), ((1.0, 1.0, 360, 180), nk)], 1 << 16
            )
        Hs = H_lit.reshape(3, 720, 360)
        lit = {
            "table": ix.data.lit_table(Hs > 0),
            "om0": om0,
            "bins": (0.5, 0.5, 720, 360),
            "frame_step": 0.5,
            "etacut": 0.2,
        }
        U_grid, delta = anri.crystal.orientation.orientation_grid(5.0, ops)
        kept, _, info = ix.search.prune(U_grid, delta, B, rings, geom, lit)
        self.assertGreater(info["min_comp"], info["chance"])
        U_kept = U_grid[kept]
        pred = ix.search.predictions(U_kept, B, rings, geom, 0.2)
        _, pad = anri.geom.scan.sino_shift_and_pad(0.0, nk, float(ds["ybincens"][0]), 1.0)
        nr = nk + pad
        vox = np.asarray(anri.geom.scan.recon_positions(nr, 1.0), np.float32)
        scan = {"y0": 0.0, "dty0": float(ds["ybincens"][0]), "ystep": 1.0, "n_rows": nk, "om0": om0}
        f, cand = ix.occupancy.fit_occupancy(
            H, pred, rings["ring_j"], vox, scan, (1.0, 1.0, 360, 180), k=16, n_iter=2, log=lambda m: None
        )
        frac, U_pop, _, _ = ix.populations.group(f, cand, U_kept, ops, 1.8 * 5.0)
        # voxels well inside each grain find it as their main population, to within the grid's spacing
        for g, sel in ((0, (vox[:, 0] < -0.9) & (vox[:, 0] > -2.6)), (1, (vox[:, 0] > 0.9) & (vox[:, 0] < 2.6))):
            sel &= np.abs(vox[:, 1]) < 2.6
            err = anri.crystal.orientation.disorientation(U_pop[sel, 0], np.repeat(U_true[g][None], sel.sum(), 0), ops)
            self.assertLess(np.median(err), delta / 2, msg=f"grain {g}")
            self.assertGreater(np.median(frac[sel, 0]), 0.5)


class TestProvenance(unittest.TestCase):
    def test_chain(self):
        """Each stage file points back to its source by path and id; show prints the chain, first stage first."""
        import shutil

        import h5py

        from anri.index.stages import STAGES
        from anri.io.provenance import write_provenance

        with tempfile.TemporaryDirectory() as tmp:
            paths, source = [], None
            for stage in STAGES:
                paths.append(os.path.join(tmp, f"x_{stage}.h5"))
                with h5py.File(paths[-1], "w") as h:
                    write_provenance(h, stage, source, {"cand": 64, "y0": None, "path": "C:\\data\\x"},
                                     {"lattice": [2.87, 2.87, 2.87], "flag": True})  # fmt: skip
                source = paths[-1]
            self.assertEqual(anri.io.provenance.upstream(paths[-1], "histogram"), paths[0])
            prov = anri.io.provenance.read_provenance(paths[2])
            self.assertEqual((prov["stage"], prov["options"]["cand"], prov["options"]["path"]),
                             ("filter", 64, "C:\\data\\x"))  # fmt: skip
            self.assertNotIn("y0", prov["options"])  # None is left out: HDF5 has no null
            self.assertEqual(prov["resolved"]["lattice"], [2.87, 2.87, 2.87])
            self.assertIs(prov["resolved"]["flag"], True)
            text = anri.io.provenance.show(paths[-1])
            self.assertLess(text.index("[histogram]"), text.index("[search]"))
            self.assertLess(text.index("[filter]"), text.index("[populations]"))
            # moved together to another folder: the sources are found beside each other
            moved = os.path.join(tmp, "moved")
            os.mkdir(moved)
            for p in paths:
                shutil.move(p, moved)
            self.assertEqual(anri.io.provenance.upstream(os.path.join(moved, "x_populations.h5"), "histogram"),
                             os.path.join(moved, "x_histogram.h5"))  # fmt: skip
            # a remade search is not the one the later stages were made from
            with h5py.File(os.path.join(moved, "x_search.h5"), "w") as h:
                write_provenance(h, "search", os.path.join(moved, "x_histogram.h5"), {}, {})
            with self.assertRaises(ValueError):
                anri.io.provenance.upstream(os.path.join(moved, "x_populations.h5"), "histogram")

    def test_run_command(self):
        """run_command gives every option recorded along the chain, and the command line parses each one back."""
        import inspect
        import shlex

        import h5py

        from anri.index.__main__ import parse_args
        from anri.index.stages import STAGES
        from anri.io.provenance import write_provenance

        values = {"analysisroot": "/data/a b", "sample": "s", "dataset": "d", "tth_tol": 0.02, "frame_dty": False,
                  "grid": 1.5, "min_comp": 0.3, "y0": -1.25, "mask": "auto"}  # fmt: skip
        recorded = {}
        with tempfile.TemporaryDirectory() as tmp:
            source = None
            for stage in STAGES:
                params = inspect.signature(getattr(ix.stages, f"stage_{stage}")).parameters
                opts = {k: values.get(k, p.default) for k, p in params.items()
                        if k in values or (p.kind == p.KEYWORD_ONLY and k not in ("command", "log", "callback", "check"))}  # fmt: skip
                recorded.update(opts)
                path = os.path.join(tmp, f"x_{stage}.h5")
                with h5py.File(path, "w") as h:
                    write_provenance(h, stage, source, opts, {})
                source = path
            args = parse_args(shlex.split(ix.stages.run_command(source))[3:])
        self.assertEqual(args.command, "run")
        for k, v in recorded.items():
            if k == "save_model":  # not an option of the command line
                continue
            if k == "frame_dty":
                self.assertEqual(args.no_frame_dty, not v)
                continue
            self.assertEqual(getattr(args, {"n_iter": "iter"}.get(k, k)), v, msg=k)


class TestDiscMask(unittest.TestCase):
    def test_disc(self):
        """The voxels within the scanned dty range all round the axis: a disc, smaller when y0 is off the middle."""
        scan = {"dty0": -5.0, "ystep": 1.0, "n_rows": 11}  # rows -5..5, edges -5.5..5.5
        nr = ix.stages.voxel_grid(scan, 0.0)
        m = ix.mask.disc_mask(scan, 0.0, nr)
        r = np.hypot(*np.asarray(anri.geom.scan.recon_positions(nr, 1.0))[:, :2].T).reshape(nr, nr)
        np.testing.assert_array_equal(m, r <= 5.5)
        self.assertFalse(m[0, 0])  # a corner
        self.assertTrue(m[nr // 2, nr // 2])
        off = ix.mask.disc_mask(scan, 2.0, ix.stages.voxel_grid(scan, 2.0))
        self.assertLess(off.sum(), m.sum())  # 3.5 from the nearer end


class TestSelections(unittest.TestCase):
    def test_completeness(self):
        """The filter's completeness cut: at or above min_comp, the most complete first, at most keep."""
        comp = np.array([0.1, 0.5, 0.3, 0.5, 0.9, 0.2, 0.1])
        chance, halfway = ix.search.completeness_levels(comp)
        np.testing.assert_allclose((chance, halfway), (0.3, 0.6))  # median, and halfway to 0.9
        np.testing.assert_array_equal(ix.search.select_by_completeness(comp, 0.3, 3)[:1], [4])
        self.assertEqual(set(ix.search.select_by_completeness(comp, 0.3, 10)), {1, 2, 3, 4})
        self.assertEqual(len(ix.search.select_by_completeness(comp, 0.3, 2)), 2)


if __name__ == "__main__":
    unittest.main()


class TestStructureFactors(unittest.TestCase):
    def test_f2_from_cif(self):
        c, _, _ = iron()
        cif = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cif", "Fe.cif")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rings = ix.data.ring_table(*c, PARS["wavelength"], 3, __import__("Dans_Diffraction").Crystal(cif))
        F2 = rings["F2"]
        self.assertAlmostEqual(float(F2.mean()), 1.0, places=5)
        per_ring = [F2[rings["ring_j"][::2] == r].mean() for r in range(3)]
        self.assertGreater(per_ring[0], per_ring[1])  # the form factor falls with 2theta
        self.assertGreater(per_ring[1], per_ring[2])
        self.assertTrue(np.all(F2 > 0))
        np.testing.assert_allclose(ix.data.ring_table(*c, PARS["wavelength"], 3)["F2"], 1.0)


class TestOrientationFit(unittest.TestCase):
    def test_finds_the_orientations_in_the_data(self):
        """Data from two orientations among 200: those two have the largest likelihood ratios, and their occupancies."""
        from anri.index.orientations import _forward

        c, B, _ = iron()
        rings = ix.data.ring_table(*c, PARS["wavelength"], 3)
        geom = geometry()
        U = rotations(200, 9)
        bins = (1.0, 1.0, 360, 180, 0.0)
        g_true = np.zeros(200, np.float32)
        g_true[[17, 123]] = [100.0, 25.0]
        F2 = jnp.ones(2 * len(rings["hkls"]), jnp.float32)
        args = (jnp.asarray(B), jnp.asarray(rings["hkls"]), F2, jnp.asarray(rings["ring_j"]), geom, 0.2)
        d = _forward(
            jnp.asarray(g_true.reshape(-1, 8)), jnp.asarray(U.reshape(-1, 8, 3, 3)), *args, bins, 3 * 360 * 180
        )
        g, lr = ix.orientations.fit(np.asarray(d), U, B, rings, geom, bins, 0.2, n_iter=50, qc=8, log=lambda m: None)
        self.assertEqual(set(np.argsort(lr)[-2:]), {17, 123})
        self.assertGreater(lr[[17, 123]].min(), 25.0)
        np.testing.assert_allclose(g[[17, 123]], [100.0, 25.0], rtol=0.1)

    def test_likelihood_ratio_precise_at_large_counts(self):
        """With ~1e6 counts per bin, weak orientations' likelihood ratios match float64 at the same fit.

        Those near a min_lr cut matter: log(m / rest) in float32 made them noisy (0.02-0.04 here, up to ~5 on
        AP1_1). The true orientations' ratios (~1e9, set by the floor on rest where they are alone) are not checked.
        """
        from anri.index.orientations import _forward, _system

        c, B, _ = iron()
        rings = ix.data.ring_table(*c, PARS["wavelength"], 3)
        geom = geometry()
        U = rotations(64, 9)
        bins = (1.0, 1.0, 360, 180, 0.0)
        g_true = np.zeros(64, np.float32)
        g_true[[5, 40]] = [1e6, 3e5]
        F2 = jnp.ones(2 * len(rings["hkls"]), jnp.float32)
        args = (jnp.asarray(B), jnp.asarray(rings["hkls"]), F2, jnp.asarray(rings["ring_j"]), geom, 0.2)
        d = _forward(
            jnp.asarray(g_true.reshape(-1, 8)), jnp.asarray(U.reshape(-1, 8, 3, 3)), *args, bins, 3 * 360 * 180
        )
        d = np.asarray(d)
        g, lr, mu = ix.orientations.fit(d, U, B, rings, geom, bins, 0.2, n_iter=5, qc=8, log=lambda m: None,
                                        return_model=True)  # fmt: skip
        idx, wt = (np.asarray(a) for a in _system(jnp.asarray(U), *args, bins))
        mu64, d64 = np.asarray(mu, np.float64), d.astype(np.float64)
        for q in np.flatnonzero(g_true == 0):
            ok = (idx[q] >= 0) & (mu64[np.clip(idx[q], 0, None)] > 0)
            m, dd = mu64[idx[q][ok]], d64[idx[q][ok]]
            ga = float(g[q]) * wt[q][ok].astype(np.float64)
            rest = np.maximum(m - ga, 1e-6 * m)
            ref = 2 * np.sum(np.where(dd > 0, dd * np.log(m / rest) - ga, -ga))
            self.assertLess(abs(lr[q] - ref), 1e-3, f"orientation {q}")


class TestInherit(unittest.TestCase):
    def test_neighbourhood_candidates(self):
        """A fine voxel gets the orientations its 3 x 3 coarse neighbourhood occupied, most occupied first, once each."""
        gc = np.arange(-2, 3) * 4.0
        xx, yy = np.meshgrid(gc, gc, indexing="ij")
        pos_c = np.stack([xx.ravel(), yy.ravel(), np.zeros(25)], 1)
        f_c = np.zeros((25, 2), np.float32)
        cand_c = np.zeros((25, 2), np.int32)
        for c in range(25):
            cand_c[c] = [c, 100]  # every coarse voxel holds its own orientation and a shared one
            f_c[c] = [1.0 + c, 0.5]
        fine = np.array([[0.0, 0.0, 0.0], [7.0, 7.0, 0.0]])  # the centre, and near coarse voxel (2, 2) -> index 18
        cand = np.asarray(ix.occupancy.inherit_candidates(fine, pos_c, f_c, cand_c, 10))
        centre = {6, 7, 8, 11, 12, 13, 16, 17, 18}
        self.assertEqual(set(cand[0][:9]), centre)  # the neighbours' own orientations come first
        self.assertEqual(list(cand[0][:9]), sorted(centre, reverse=True))  # by occupancy
        self.assertEqual(cand[0][9], 100)  # the shared one once
        self.assertIn(24, cand[1])
