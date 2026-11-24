import os
import tempfile
import unittest

import h5py
import jax.numpy as jnp
import numpy as np

import anri.crystal
import anri.geom
import anri.index as ix
import anri.io
import anri.maths.rotation
import anri.simulate

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

    def test_chosen_rings(self):
        """A list of ring indices keeps those rings, numbered 0.. in order, and the same reflections as the first N."""
        c, _, _ = iron()
        first = ix.data.ring_table(*c, PARS["wavelength"], 4)
        some = ix.data.ring_table(*c, PARS["wavelength"], [3, 1])
        np.testing.assert_array_equal(some["index"], [1, 3])
        np.testing.assert_allclose(some["tth"], first["tth"][[1, 3]])
        np.testing.assert_allclose(some["tth_all"], first["tth"])
        keep = np.isin(first["ring_j"][::2], [1, 3])
        np.testing.assert_array_equal(some["hkls"], first["hkls"][keep])
        np.testing.assert_array_equal(some["ring_j"][::2], np.searchsorted([1, 3], first["ring_j"][::2][keep]))

    def test_merge_rings(self):
        """Monoclinic Ag2S: (1 0 -2) and (-1 -1 0) are 1e-4 apart in d*; a wider merge makes them one ring."""
        c = ([4.2263, 6.92759, 8.2814, 90, 110.552, 90], 14)
        split = ix.data.ring_table(*c, 0.17163, 8, merge=5e-5)
        merged = ix.data.ring_table(*c, 0.17163, 8, merge=5e-4)
        self.assertLess(len(merged["ds_all"]), len(split["ds_all"]) + 1)
        d = np.abs(merged["ds_all"] - 0.2909).argmin()
        self.assertEqual(np.sum(merged["ring_j"][::2] == d), 6)  # 2 + 4 reflections in one ring
        self.assertTrue(np.all(np.diff(merged["ds_all"]) > 5e-4))

    def test_ring_widths(self):
        ring_tth = np.array([5.0, 6.0])
        lo, step = 4.5, 0.002
        x = lo + (np.arange(1000) + 0.5) * step
        prof = 100 * np.exp(-0.5 * ((x - 5.01) / 0.02) ** 2) + 50 * np.exp(-0.5 * ((x - 6.0) / 0.04) ** 2) + 1.0
        off, hw = ix.data.ring_widths(prof, lo, step, ring_tth)
        np.testing.assert_allclose(off, [0.01, 0.0], atol=0.002)
        np.testing.assert_allclose(hw, [1.96 * 0.02, 1.96 * 0.04], rtol=0.1)  # 95% of a Gaussian


class TestDiagnostics(unittest.TestCase):
    def test_ring_windows(self):
        """Windows of chosen rings are measured between every ring's neighbours, as the histogram stage measures them."""
        tth = np.array([5.0, 5.3, 6.0, 7.0])
        lo, step = 4.5, 0.002
        x = lo + (np.arange(1500) + 0.5) * step
        prof = sum(100 * np.exp(-0.5 * ((x - t) / 0.02) ** 2) for t in tth) + 1.0
        cake = {"rings": {"tth": tth}, "profile": prof, "tth": x, "lo": lo, "step": step, "gap": np.diff(tth)}
        w = ix.diagnostics.ring_windows(cake, [2, 0])
        np.testing.assert_array_equal(w["index"], [0, 2])
        off, hw = ix.data.ring_widths(prof, lo, step, tth[:3])  # up to the last ring used, ring 1 included
        np.testing.assert_allclose(w["tolerance"], (np.abs(off) + hw)[[0, 2]])
        self.assertEqual(w["overlap"], [False, False])
        self.assertLess(w["fraction"], 0.8)  # ring 1 is in no window
        np.testing.assert_allclose(ix.diagnostics.ring_windows(cake, 3)["tolerance"], np.abs(off) + hw)
        self.assertEqual(ix.diagnostics.ring_windows(cake, [0], tth_tol=0.2)["overlap"], [True])  # meets ring 1
        with self.assertRaises(ValueError):
            ix.diagnostics.ring_windows(cake, [4])


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
    def test_fewer_orientations_than_candidates(self):
        """Fewer orientations than the candidates asked for: as many candidates as orientations (issue #32)."""
        pred = (jnp.full((16, 1), 90.0), jnp.full((16, 1), 0.5), jnp.ones((16, 1), bool), jnp.ones((16, 1)))
        scan = {"n_rows": 1, "y0": 0.0, "dty0": 0.0, "ystep": 1.0, "om0": 0.0}
        args = (np.ones(4, np.float32), pred, np.array([0]), np.zeros((1, 3), np.float32), scan, (90.0, 1.0, 4, 1))
        f, cand = ix.occupancy.fit_occupancy(*args, n_iter=0, block_bytes=4096, log=lambda _: None)
        self.assertEqual(f.shape, (1, 16))
        self.assertEqual(sorted(cand[0].tolist()), list(range(16)))
        with self.assertRaises(ValueError):
            ix.occupancy.fit_occupancy(args[0], tuple(p[:0] for p in pred), *args[2:], n_iter=0, log=lambda _: None)

    def test_absorption(self):
        """With edge distances after the positions, each weight is scaled by exp(-mu (L_in + L_out))."""
        n, step, r, mu = 81, 2.0, 60.0, 0.005
        grid = np.asarray(anri.geom.scan.recon_positions(n, step))
        disc = (np.hypot(grid[:, 0], grid[:, 1]) <= r).reshape(n, n).astype(np.float32)
        pos = np.zeros((1, 3), np.float32)  # the disc's centre
        D = np.asarray(anri.geom.paths.edge_distances(disc, step, pos[:, :2]))
        tth, eta = 5.0, np.array([[[30.0, 150.0, -60.0]]])
        om, use, w = np.zeros_like(eta), np.ones(eta.shape, bool), np.ones_like(eta)
        scan = {
            "y0": 0.0,
            "dty0": -30.0,
            "ystep": step,
            "n_rows": 31,
            "om0": -90.0,
            "mu": mu,
            "ring_tth": np.array([tth]),
        }
        args = (jnp.asarray(eta), jnp.asarray(om), jnp.asarray(use), jnp.asarray(w), jnp.zeros(3, jnp.int32))
        _, w0 = ix.occupancy.system(*args, jnp.asarray(pos), scan, 1.0, 1.0, 360, 180)
        _, wa = ix.occupancy.system(*args, jnp.asarray(np.concatenate([pos, D], 1)), scan, 1.0, 1.0, 360, 180)
        t, e = np.radians(tth), np.radians(eta[0, 0])
        horizontal = np.hypot(np.cos(t), np.sin(t) * np.sin(e))  # k_out at omega = 0, beam along x
        expected = np.exp(-mu * (r + r / horizontal))
        ratio = np.asarray(wa).sum(-1)[0, 0] / np.asarray(w0).sum(-1)[0, 0]
        np.testing.assert_allclose(ratio, expected, rtol=0.02)

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
        # 0.5 um beam on 0.2 um rows: 9 rows; the fit's coarse level (rows summed in pairs) needs 7 of 0.4 um
        fine = {**scan, "ystep": 0.2, "sig_beam": 0.5 / 2.3548, "voxel": 0.2}
        self.assertEqual(ix.occupancy.beam_rows(fine), 9)
        self.assertEqual(ix.occupancy.beam_rows(ix.data.coarsen_scan(fine, 2)), 7)

    def test_padding_is_invisible(self):
        """Padded voxels get no entries: a far-away one (1e6, 1e6) would cross the axis's ray at omega 135, 315."""
        pred = (jnp.array([[10.0, 10.0]]), jnp.array([[135.0, 315.0]]), jnp.ones((1, 2), bool), jnp.ones((1, 2)))
        ring_j = jnp.zeros(2, jnp.int32)
        scan = {"y0": 0.0, "dty0": -3.0, "ystep": 1.0, "n_rows": 7, "om0": 0.0}
        dims, n_cells = (90.0, 1.0, 4, 360), 4 * 360 * 7
        pos = ix.occupancy.pad_voxels(jnp.zeros((1, 3)), 2)
        _, wt = ix.occupancy.system(*(p[None] for p in pred), ring_j, pos, scan, *dims)
        np.testing.assert_array_equal(wt[1], 0.0)
        data = ix.occupancy.forward(jnp.ones((1, 1)), jnp.zeros((1, 1), jnp.int32), pred, ring_j, pos[:1], scan, dims,
                                    n_cells, 1)  # fmt: skip
        f, _ = ix.occupancy.candidates(data, pred, ring_j, pos, scan, dims, 1, 1, qc=1, log=lambda _: None)
        np.testing.assert_allclose(f, [[1.0], [0.0]], atol=1e-6)

    def test_full_turn_seam(self):
        """A full turn wraps omega's corners at the seam; a partial scan drops the corners past its ends."""
        om0 = 17.25
        for n_o, b_o in ((360, 1.0), (3600, 0.1)):
            om = om0 + b_o * jnp.array([0.1, n_o - 0.1])  # 0.4 of a bin either side of the seam
            corners = ix.data.eta_omega_corners(jnp.zeros(2), om, om0, 1.0, b_o, 360, n_o)
            np.testing.assert_allclose(sum(we * wo * inside for _, _, we, wo, inside in corners), 1.0, atol=1e-4)
            np.testing.assert_array_equal(corners[0][1], [n_o - 1, n_o - 1])
            np.testing.assert_array_equal(corners[1][1], [0, 0])
        corners = ix.data.eta_omega_corners(jnp.zeros(1), jnp.array([om0 + 0.1]), om0, 1.0, 1.0, 360, 180)
        np.testing.assert_array_equal(corners[0][1], [-1])  # half a turn: the corner before om0 is outside
        np.testing.assert_allclose(sum(we * wo * inside for _, _, we, wo, inside in corners), 0.6, atol=1e-5)

    def test_dty_offsets_match_histogram(self):
        """Offsets and exposure leave out the frames past the scan's ends, as the histogram does, not wrap them."""
        omega = jnp.array([[-0.25, 0.25, 0.75, 1.25, 1.75, 360.25]])
        dty = np.array([[0.99, 0.2, 0.4, 0.6, 0.8, 0.99]])
        x = jnp.stack([jnp.full(omega.size, 5.0), jnp.zeros(omega.size), omega.ravel()], 1)
        counts = ix.data.histogram(x, jnp.ones(omega.size), jnp.zeros(omega.size, jnp.int32), jnp.array([5.0]), 0.1,
                                   0.0, 360.0, 1.0, 1, 1, 360, 1)  # fmt: skip
        offsets, exposure = ix.data.dty_offsets(dty, omega, [-0.5, 0.5], 1, 0.0, 1.0, 360, 0.0, 1.0,
                                               bin_dty=np.zeros_like(dty), return_exposure=True)  # fmt: skip
        np.testing.assert_allclose(exposure[0], counts / 2)
        np.testing.assert_allclose(offsets[0, :2], [0.3, 0.7])
        np.testing.assert_array_equal(offsets[0, 2:], 0.0)

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

    def test_crowding(self):
        """Near-duplicate orientations crowd the first-update candidates out of a weak second orientation; pricing at
        the fit adds it back, the certificate then holds, and the deviance falls to near the optimum's (0, as the data
        are the model's) (theory page, coarse to fine: ls-kkt, ls-gain)."""
        c, B, _ = iron()
        rings = ix.data.ring_table(*c, PARS["wavelength"], 2)
        base = rotations(4, 3)
        dup = np.asarray(anri.maths.rotation.rot_z(0.05), np.float32) @ base  # 0.05 deg: far below the bins
        pred = ix.search.predictions(np.concatenate([base, dup]), B, rings, geometry(), 0.2)
        rng = np.random.default_rng(7)
        nv = 8
        pos = np.column_stack([rng.uniform(-2, 2, (nv, 2)), np.zeros(nv)]).astype(np.float32)
        scan = {"y0": 0.0, "dty0": -3.0, "ystep": 1.0, "n_rows": 7, "om0": 0.0}
        dims = (4.0, 4.0, 90, 45)
        n_cells = 2 * 90 * 45 * 7
        ring_j = jnp.asarray(rings["ring_j"])
        cand_true = np.tile(np.array([[0, 1]], np.int32), (nv, 1))  # a strong orientation 0 and a weak 1 everywhere
        f_true = np.tile(np.array([[1.0, 0.15]], np.float32), (nv, 1))
        d = ix.occupancy.forward(jnp.asarray(f_true), jnp.asarray(cand_true), pred, ring_j, jnp.asarray(pos), scan,
                                 dims, n_cells, 8)  # fmt: skip

        def fit(rounds: "int | None") -> tuple:
            lines = []
            f, cand, mu = ix.occupancy.fit_occupancy(d, pred, ring_j, pos, scan, dims, k=2, n_iter=200, tol=1e-2,
                                                     qc=8, block_bytes=1e5, price_rounds=rounds, return_model=True,
                                                     log=lines.append)  # fmt: skip
            return np.asarray(f), np.asarray(cand), float(ix.occupancy.deviance(d, jnp.asarray(mu), 0.0)), lines

        _, cand_raw, dev_raw, _ = fit(None)
        self.assertTrue(all(set(cr) == {0, 4} for cr in cand_raw))  # the strong one and its duplicate
        f_p, cand_p, dev_p, lines = fit(2)
        checks = [m for m in lines if "certificate holds" in m or "voxels have an orientation" in m]
        self.assertIn("certificate holds", checks[-1])
        self.assertLess(dev_p, 1e-3 * dev_raw)
        weak = np.array([f_v[(c_v == 1) | (c_v == 5)].sum() for f_v, c_v in zip(f_p, cand_p)])
        np.testing.assert_allclose(weak / f_p.sum(1), 0.15 / 1.15, atol=0.01)  # the weak one, or its duplicate

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
        devs, msgs = [], []
        f, info = ix.occupancy.mlem(d, cand, pred, ring_j, pos, scan, dims, f0, 10, 8, log=msgs.append,
                    callback=lambda it, f_, dev: devs.append(dev), tol=0.0)  # fmt: skip
        self.assertEqual(info["iterations"], 10)
        self.assertEqual(info["converged"], info["remaining"] <= 0.0)
        # a tolerance every iteration meets stops after two (twice running), and says so
        _, info1 = ix.occupancy.mlem(d, cand, pred, ring_j, pos, scan, dims, f0, 10, 8, log=lambda m: None,
                                      tol=float("inf"))  # fmt: skip
        self.assertEqual(info1["iterations"], 2)  # twice running
        self.assertTrue(info1["converged"])
        # the estimate of the deviance still to fall decreases as the fit converges, and is never negative
        rem = [float(m.split("~")[1].split(" ")[0]) for m in msgs]
        self.assertTrue(all(x >= 0 for x in rem))
        self.assertLess(rem[-1], rem[0])
        self.assertEqual(len(devs), 10)
        self.assertEqual(len(msgs), 10)  # one line per iteration
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

    def test_censor_level(self):
        # pixels of 2 counts and up (cut 1), monitor-normalised by 0.95: the smallest bin is one pixel
        H = np.zeros((3, 4, 5), np.float32)
        H[0, 1, 2], H[2, 3, 4], H[1, 0, 0] = 2 * 0.95, 7.0, 40.0
        censor, smallest = ix.occupancy.censor_level(H)
        self.assertAlmostEqual(smallest, 1.9, places=5)
        self.assertAlmostEqual(censor, 3 * 1.9, places=5)
        self.assertAlmostEqual(ix.occupancy.censor_level(H, pixels_in_spot=5)[0], 5 * 1.9, places=5)
        self.assertEqual(ix.occupancy.censor_level(np.zeros(4, np.float32)), (0.0, 0.0))

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

    def test_coarsen_scan_partial_group(self):
        """Without exposure, a partial last group gets the share of g rows it sums, as coarsen_rows sums the data."""
        for n_rows, g in ((5, 2), (7, 3), (541, 9), (6, 2)):
            c = ix.data.coarsen_scan({"dty0": 0.0, "ystep": 1.0, "n_rows": n_rows}, g)
            Hc, _ = ix.data.coarsen_rows(np.ones(n_rows), n_rows, g)
            if n_rows % g:
                np.testing.assert_allclose(np.asarray(c["exposure"])[:, 0], np.asarray(Hc) / g)
            else:
                self.assertNotIn("exposure", c)  # whole groups: nothing to correct
        # system() takes a per-row exposure [n_rows, 1]: the weight of a prediction in a row scales with it
        scan = {"y0": 0.0, "dty0": -1.0, "ystep": 1.0, "n_rows": 3, "om0": -180.0, "k_in_lab": jnp.array([1.0, 0.0, 0.0]),
                "wedge": 0.0, "chi": 0.0}  # fmt: skip
        pos = jnp.zeros((1, 3), jnp.float32)  # on the axis: row 1 (dty 0) at every omega
        args = (jnp.full((1, 1, 1), 10.0), jnp.full((1, 1, 1), 0.3), jnp.ones((1, 1, 1), bool), jnp.ones((1, 1, 1)),
                jnp.asarray([0]), pos)  # fmt: skip
        _, w1 = ix.occupancy.system(*args, scan, 1.0, 1.0, 360, 360)
        _, w2 = ix.occupancy.system(
            *args, {**scan, "exposure": jnp.asarray([[1.0], [0.25], [1.0]])}, 1.0, 1.0, 360, 360
        )
        self.assertAlmostEqual(float(np.asarray(w2).sum()), 0.25 * float(np.asarray(w1).sum()), places=5)

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

    def test_largest_first(self):
        """Populations come out largest first, though the most occupied candidate seeds the smaller one (#29)."""
        U = np.asarray(anri.maths.rotation.rmat_from_axis_angle(jnp.array([[0.0, 0.0, 1.0]]), jnp.array([0, 10, 10.1])))
        f = np.array([[0.3, 0.25, 0.25]], np.float32)
        frac, mean, _, n = ix.populations.group(f, np.arange(3)[None], U, np.eye(3)[None], 1.0, p=2, m=3, vb=1)
        np.testing.assert_allclose(frac[0], [0.625, 0.375], rtol=1e-5)
        np.testing.assert_array_equal(n[0], [2, 1])
        self.assertAlmostEqual(float(np.degrees(anri.maths.rotation.rotation_angle(mean[0, 0]))), 10.05, places=2)

    def test_spread_thin(self):
        """Occupancy spread thinly over many candidates is one population with a spread, never the identity (#27)."""
        U = np.asarray(
            anri.maths.rotation.rmat_from_axis_angle(jnp.array([[0.0, 0.0, 1.0]]), 40 + jnp.arange(64) * 0.01)
        )
        frac, mean, spread, n = ix.populations.group(
            np.ones((1, 64)), np.arange(64)[None], U, np.eye(3)[None], 1.0, vb=1
        )
        self.assertEqual(int(n[0, 0]), 64)  # every candidate considered: the whole occupancy is one population
        np.testing.assert_allclose(frac[0, 0], 1.0, rtol=1e-5)
        frac16 = ix.populations.group(np.ones((1, 64)), np.arange(64)[None], U, np.eye(3)[None], 1.0, m=16)[0]
        np.testing.assert_allclose(frac16[0, 0], 16 / 64, rtol=1e-5)  # m = 16 sees a quarter of it
        self.assertAlmostEqual(float(np.degrees(anri.maths.rotation.rotation_angle(mean[0, 0]))), 40.315, delta=0.01)
        self.assertGreater(spread[0, 0], 0.0)

    def test_occupancy_weighted(self):
        """IPF colours dimmed by occupancy, full brightness at the 99th percentile of the occupied voxels."""
        occ = np.array([0.0, 0.5, 1.0, 2.0])
        rgb = np.ones((4, 3))
        w = ix.populations.occupancy_weighted(rgb, occ, percentile=100)
        np.testing.assert_allclose(w[:, 0], [0.0, 0.25, 0.5, 1.0])
        np.testing.assert_allclose(ix.populations.occupancy_weighted(rgb, np.zeros(4)), 0.0)


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
            # pixels_in_spot 1: this phantom's spots are mostly 1-2 pixels, which ImageD11's default group rule (3) would
            # drop whole (65% of the counts)
            anri.simulate.stages.simulate_sparse(sparse, entries, rings["hkls"], np.ones(len(rings["hkls"])), geom, omega, dty,
                                    (2162, 2068), window=(5, 9, 9), pixels_in_spot=1)  # fmt: skip
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
            # saturated pixels: flagged on the raw values, counted in the fit's bins; the histograms do not change
            cap = float(np.percentile(np.concatenate([c[4] for c in stream()]), 99.9))
            sat_stream = anri.io.imaged11.stream_sparse(
                sparse, ds["ybinedges"], ds["omegamotor"], ds["dtymotor"], 1 << 16, saturation=cap
            )
            H_lit2, H2, S = ix.data.histogram_pixels(
                sat_stream, geom, rings["tth"], 0.1, om0, [((0.5, 0.5, 720, 360), 1), ((1.0, 1.0, 360, 180), nk)],
                1 << 16, saturated_into=1,
            )  # fmt: skip
            np.testing.assert_array_equal(np.asarray(H2), np.asarray(H))
            np.testing.assert_array_equal(np.asarray(H_lit2), np.asarray(H_lit))
            n_cap = sum(int(np.sum(c[4] >= cap)) for c in stream())
            self.assertGreater(float(np.asarray(S).sum()), 0)
            self.assertLessEqual(float(np.asarray(S).sum()), n_cap)  # those in the rings' windows
            self.assertTrue(np.all(np.asarray(H)[np.asarray(S) > 0] > 0))  # in bins with data
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
        # finer levels: two around each voxel's orientations, 5 deg -> 0.56 deg, each voxel over its own children
        U_l, f_l, c_l, step = U_kept, f, cand, 5.0
        for _ in range(2):
            use, f0, c0 = ix.local.inherit(f_l, c_l, len(U_l), min_share=0.05, m=2)  # two grains: at most 2 per voxel
            given = (f0, c0)
            U_l, step = ix.local.children(U_l[use], step), step / 3
            pred = ix.search.predictions(U_l, B, rings, geom, 0.2)
            f_l, c_l = ix.occupancy.fit_occupancy(H, pred, rings["ring_j"], vox, scan, (1.0, 1.0, 360, 180), n_iter=2,
                                                  log=lambda m: None, given=given)  # fmt: skip
        _, U_pop_l, _, _ = ix.populations.group(f_l, c_l, U_l, ops, 1.8 * 5.0)
        for g, sel in ((0, (vox[:, 0] < -0.9) & (vox[:, 0] > -2.6)), (1, (vox[:, 0] > 0.9) & (vox[:, 0] < 2.6))):
            sel &= np.abs(vox[:, 1]) < 2.6
            truth = np.repeat(U_true[g][None], sel.sum(), 0)
            err = anri.crystal.orientation.disorientation(U_pop[sel, 0], truth, ops)
            err_l = anri.crystal.orientation.disorientation(U_pop_l[sel, 0], truth, ops)
            print(f"grain {g}: grid {np.median(err):.3f} deg, local search {np.median(err_l):.3f} deg")
            self.assertLess(np.median(err_l), np.median(err), msg=f"grain {g}")


class TestFitStage(unittest.TestCase):
    def test_half_scan(self):
        """The stages on a half-sample scan of two grains: refined orientations, no voxel starved near the axis."""
        import h5py

        c, B, ops = iron()
        rings = ix.data.ring_table(*c, PARS["wavelength"], 3)
        U_true = rotations(2, 7)
        i, j = np.mgrid[0:7, 0:7]
        pos = np.stack([i.ravel() - 3.0, j.ravel() - 3.0, np.zeros(49)], 1)
        grain = (pos[:, 0] > 0.5).astype(int)
        entries = {"ubi": np.linalg.inv(U_true[grain] @ B), "pos": pos, "density": np.full(49, 30.0)}
        omega, dty = anri.io.imaged11.motor_grid((0.0, 360.0), 1.0, (-5.0, 1.0), 1.0)  # one edge to a row past y0
        a = A_FE
        cell = {"cell__a": a, "cell__b": a, "cell__c": a, "cell_alpha": 90.0, "cell_beta": 90.0, "cell_gamma": 90.0,
                "cell_lattice_[P,A,B,C,I,F,R]": 229}  # fmt: skip
        quiet = lambda m: None
        with tempfile.TemporaryDirectory() as tmp:
            sparse = os.path.join(tmp, "sparse.h5")
            anri.simulate.stages.simulate_sparse(sparse, entries, rings["hkls"], np.ones(len(rings["hkls"])),
                                                 geometry(), omega, dty, (2162, 2068), window=(5, 9, 9),
                                                 pixels_in_spot=1)  # fmt: skip (spots of 1-2 pixels: see test_two_grains)
            parfile = anri.io.imaged11.write_pars(os.path.join(tmp, "pars"), PARS, {"Fe": cell})
            anri.io.imaged11.write_dataset(sparse, tmp, "fe", "half", y0=0.0, parfile=parfile)
            hist = ix.stages.stage_histogram(tmp, "fe", "half", os.path.join(tmp, "h.h5"), rings=3, log=quiet)
            search = ix.stages.stage_search(hist, grid=3.0, log=quiet)
            filt = ix.stages.stage_filter(search, min_comp=0.3, log=quiet)  # fewer to fit than all above chance
            fit = ix.stages.stage_fit(filt, cand=32, n_iter=3, log=quiet)  # level 1 and plain: one shape, one compile
            res = anri.io.provenance.read_provenance(fit)["resolved"]
            self.assertAlmostEqual(res["worst_case_deg"], 2.6 / 27, delta=0.01)  # refined 2.6 -> 0.29, then a level
            pop = ix.stages.stage_populations(fit, log=quiet)
            with h5py.File(fit, "r") as h:
                f, vox, sel = h["f"][()], h["pos"][()], h["mask"][()]
            with h5py.File(pop, "r") as h:
                U_pop, frac = h["U_pop"][:, 0], h["frac"][:, 0]
            # the grid's own orientations, as a calibration needs: not refined, one level
            plain = ix.stages.stage_fit(filt, os.path.join(tmp, "plain.h5"), cand=32, n_iter=3, refine_to=0, levels=0,
                                        log=quiet)  # fmt: skip
            self.assertEqual(anri.io.provenance.read_provenance(plain)["resolved"]["orientation_step"], 3.0)
        tot = f.sum(1)
        inside = sel & (np.abs(vox[:, 0]) <= 3.0) & (np.abs(vox[:, 1]) <= 3.0)
        near_axis = inside & (np.hypot(vox[:, 0], vox[:, 1]) < 1.5)
        self.assertGreater(tot[near_axis].min(), 0.5 * np.median(tot[inside]))  # not starved by the half scan
        for g, cut in ((0, vox[:, 0] < -0.5), (1, vox[:, 0] > 1.5)):
            v = inside & cut
            err = anri.crystal.orientation.disorientation(U_pop[v], np.repeat(U_true[g][None], v.sum(), 0), ops)
            self.assertLess(np.median(err), 0.15, msg=f"grain {g}")  # within the last level's 0.096 deg worst case
            self.assertGreater(np.median(frac[v]), 0.8)


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
            # the log: kept by the recorder as well as passed on, stored in the file, shown on request
            shown = []
            log, lines = anri.io.provenance.recorder(shown.append)
            log("first")
            log("second")
            self.assertEqual(shown, ["first", "second"])
            anri.io.provenance.write_log(paths[1], lines)
            self.assertEqual(anri.io.provenance.read_provenance(paths[0])["log"], [])  # none written
            kept = anri.io.provenance.read_provenance(paths[1])["log"]
            self.assertEqual([k.split("] ", 1)[1] for k in kept], ["first", "second"])
            self.assertNotIn("[search.log]", text)
            self.assertIn("] second", anri.io.provenance.show(paths[-1], log=True))
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
                  "grid": 1.5, "min_comp": 0.3, "y0": -1.25, "mask": "auto", "rings": [8, 9, 15], "pad": 10.0,
                  "absorption": False, "mu": 0.002}  # fmt: skip
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
            # a plain fit from before the levels: the same fit, of the grid's orientations on one level
            path = os.path.join(tmp, "old_fit.h5")
            with h5py.File(path, "w") as h:
                write_provenance(h, "fit", os.path.join(tmp, "x_filter.h5"), {"cand": 64, "coarse": 1}, {})
            old = parse_args(shlex.split(ix.stages.run_command(path))[3:])
            self.assertEqual((old.refine_to, old.levels, old.cand), (0.0, 0, 64))
            # a local search or a coarse fit: no longer in the fit stage, so run cannot remake them
            for name, stage, opts in (("ls", "localsearch", {"cover": 0.95}), ("coarse", "fit", {"coarse": 4})):
                path = os.path.join(tmp, f"{name}.h5")
                with h5py.File(path, "w") as h:
                    write_provenance(h, stage, os.path.join(tmp, "x_filter.h5"), opts, {})
                with self.assertRaises(ValueError):
                    ix.stages.run_command(path)
            # one ring is a list of one, not the number of first rings
            path = os.path.join(tmp, "one_histogram.h5")
            with h5py.File(path, "w") as h:
                write_provenance(h, "histogram", None, {"analysisroot": "/a", "sample": "s", "dataset": "d",
                                                        "rings": [9]}, {})  # fmt: skip
            self.assertEqual(parse_args(shlex.split(ix.stages.run_command(path))[3:]).rings, [9])
        self.assertEqual(args.command, "run")
        for k, v in recorded.items():
            if k == "save_model":  # not an option of the command line
                continue
            if k in ("frame_dty", "absorption"):  # on by default, turned off on the command line
                self.assertEqual(getattr(args, "no_" + k), not v, msg=k)
                continue
            self.assertEqual(getattr(args, k), v, msg=k)


class TestHalfScanReconstruction(unittest.TestCase):
    def test_stitch_turn(self):
        """Omega + 180 is the same ray mirrored about y0: the halves join into one projection per direction."""
        n_o, rows = 8, np.arange(-6, 3)  # a half scan: rows -6..2 about y0 0
        om = (np.arange(n_o) + 0.5) * 45.0  # a full turn
        prof = lambda d: np.maximum(0.0, 25.0 - d**2)  # the same projection at every omega
        sino = np.tile(prof(rows.astype(float)), (n_o, 1))
        joined, om_h, dty0 = ix.mask.stitch_turn(sino, om, 0.0, -6.0, 1.0)
        self.assertEqual(joined.shape, (4, 13))  # 4 directions, rows -6..6
        np.testing.assert_allclose(om_h, om[:4])
        self.assertEqual(dty0, -6.0)
        np.testing.assert_allclose(joined, np.tile(prof(np.arange(-6.0, 7.0)), (4, 1)))

    def test_half_scan_disc(self):
        """A uniform disc on the axis, scanned to 3 steps past it over a full turn: uniform, no bright centre."""
        R0, n_o = 20.0, 720
        rows = np.arange(-24.0, 4.0)  # one edge of the sample to 3 past the axis
        p = 2 * np.sqrt(np.maximum(R0**2 - rows**2, 0.0))  # the disc's projection, at every omega
        H = np.expm1(np.tile(p, (n_o, 1)))  # reconstruct takes log1p of its histogram
        scan = {"y0": 0.0, "dty0": rows[0], "ystep": 1.0, "n_rows": len(rows), "om0": 0.0,
                "k_in_lab": jnp.array([1.0, 0.0, 0.0]), "wedge": 0.0, "chi": 0.0}  # fmt: skip
        nr = 51
        rec = ix.mask.reconstruct(H.ravel(), 1, 1, n_o, scan, 0.5, nr)
        r = np.hypot(*np.asarray(anri.geom.scan.recon_positions(nr, 1.0))[:, :2].T).reshape(nr, nr)
        inner, centre = rec[r < 15], rec[r < 3]
        self.assertLess(inner.std() / inner.mean(), 0.05)
        self.assertLess(abs(centre.mean() / inner.mean() - 1), 0.03)
        self.assertLess(np.abs(rec[r > 23]).mean() / inner.mean(), 0.15)  # air: a filtered edge rings a little


class TestDrift(unittest.TestCase):
    @staticmethod
    def _render(ddty, y0, dty, seed=0):
        """Points with 4 reflections and their Friedel mates (180 - eta, omega + 180), each row at its dty + ddty."""
        rng = np.random.default_rng(seed)
        n_e, n_o, n_k = 360, 360, len(dty)
        om = np.radians(-180.0 + np.arange(n_o) + 0.5)
        H = np.zeros((1, n_e, n_o, n_k))
        pts = rng.uniform(-5, 5, (600, 2))
        for x, y in pts[np.hypot(*pts.T) < 5]:
            for _ in range(4):
                ie, io, inten = rng.integers(n_e), rng.integers(n_o // 2), rng.uniform(50, 500)
                for e, o in ((ie, io), ((n_e // 2 - 1 - ie) % n_e, io + n_o // 2)):
                    at = y0 - (np.sin(om[o]) * x + np.cos(om[o]) * y)  # the point's dty
                    w = np.exp(-0.5 * ((dty + ddty[:, o] - at) / 0.16) ** 2)  # each row where it really was
                    H[0, e, o] += inten * w / w.sum()
        scan = {"n_e": 2 * n_e, "n_o": 4 * n_o, "n_rows": n_k, "om0": -180.0, "dty0": dty[0], "ystep": dty[1] - dty[0]}
        return rng.poisson(H).astype(np.float32).ravel(), scan

    def test_friedel_pairs(self):
        """y0 and the drift's part odd about row y0's time from the Friedel pairs; a slip is flagged at both ends."""
        y0, dty = 0.13, np.arange(-6.4, 6.4 + 1e-9, 0.2)
        s_t, c_t = -0.004 * (dty - y0), 0.01 * (dty - y0) + np.where(dty > 3.0, 0.1, 0.0)
        om = np.radians(-180.0 + np.arange(360) + 0.5)
        ddty = s_t[:, None] * np.sin(om) + c_t[:, None] * np.cos(om)
        H, scan = self._render(ddty, y0, dty)
        p = ix.drift.from_pairs(ix.drift.friedel_shifts(H, 1, scan, 0.0, 0.13), 0.0)
        self.assertAlmostEqual(p["y0"], y0, delta=0.005)

        def odd(v):  # the part odd about row y0's time, as the pairs see it
            return 0.5 * (v - np.interp(2 * y0 - dty, dty, v))

        use = p["measured"] & (np.abs(dty - y0) < 4.5)
        err = np.median(p["err"][use])
        for est, true in ((p["s"], s_t), (p["c"], c_t)):
            far = use & (np.abs(np.abs(dty - y0) - 2.9) > 0.5)  # away from the slip, which the pairs spread over 2 rows
            self.assertLess(np.sqrt(np.mean((est[far] - odd(true)[far]) ** 2)), 2 * err)
        rough = ix.drift.roughness(p["s"], p["c"], p["err"], p["measured"])
        jumps = dty[rough["jumps"]]
        self.assertTrue(np.any(np.abs(jumps - 3.0) < 0.5) and np.any(np.abs(jumps - (2 * y0 - 3.0)) < 0.5), jumps)
        self.assertTrue(np.all(np.abs(np.abs(jumps - y0) - 2.9) < 0.6), jumps)  # nowhere else
        # with the drift known (a fly scan's offsets, scan["ddty"]), nothing remains
        q = ix.drift.friedel_shifts(H, 1, {**scan, "ddty": ddty}, y0, 0.13)
        self.assertTrue(np.all(np.abs(q["coef"][far]) < 5 * q["err"][far]))
        # the projections resampled onto the nominal rows, from where the rows really were
        smooth = (0.01 * (dty - y0))[:, None] * np.cos(om)  # [row, omega], up to 1/3 of a row
        prof = np.exp(-0.5 * ((dty[None, :] + smooth.T - 1.0) / 0.5) ** 2)  # [omega, row], a blob at dty 1
        back = ix.mask.shift_rows(prof, smooth, 0.2)
        want = np.exp(-0.5 * ((dty[10:-10] - 1.0) / 0.5) ** 2)
        np.testing.assert_allclose(back[:, 10:-10], np.broadcast_to(want, back[:, 10:-10].shape), atol=0.02)
        with self.assertRaises(ValueError):  # half a turn: no pairs
            ix.drift.friedel_shifts(H.reshape(1, 360, 360, -1)[:, :, :170].ravel(), 1, {**scan, "n_o": 4 * 170}, y0,
                                    0.13)  # fmt: skip


class TestDatasetGroups(unittest.TestCase):
    def test_only_the_datasets_scans(self):
        """The histogram reads the sparse groups the DataSet names, in its order; a missing one is an error."""
        keys = ["1.1", "10.1", "2.1", "3.1"]  # h5py order
        self.assertEqual(ix.stages._dataset_groups(keys, ["1.1", "2.1"], "f.h5"), ["1.1", "2.1"])
        self.assertEqual(
            ix.stages._dataset_groups(keys, ["3.1::[0:10]", "3.1::[10:20]", "1.1"], "f.h5"), ["3.1", "1.1"]
        )
        self.assertEqual(ix.stages._dataset_groups(keys, None, "f.h5"), keys)
        with self.assertRaises(ValueError):
            ix.stages._dataset_groups(keys, ["1.1", "4.1"], "f.h5")


class TestDiscMask(unittest.TestCase):
    def test_disc(self):
        """The voxels within the scanned dty range all round the axis: a disc, smaller when y0 is off the middle."""
        scan = {"dty0": -5.0, "ystep": 1.0, "n_rows": 11}  # rows -5..5, edges -5.5..5.5
        nr = ix.mask.voxel_grid(scan, 0.0)
        m = ix.mask.disc_mask(scan, 0.0, nr)
        r = np.hypot(*np.asarray(anri.geom.scan.recon_positions(nr, 1.0))[:, :2].T).reshape(nr, nr)
        np.testing.assert_array_equal(m, r <= 5.5)
        self.assertFalse(m[0, 0])  # a corner
        self.assertTrue(m[nr // 2, nr // 2])
        off = ix.mask.disc_mask(scan, 2.0, ix.mask.voxel_grid(scan, 2.0))
        self.assertLess(off.sum(), m.sum())  # 3.5 from the nearer end
        np.testing.assert_array_equal(ix.mask.disc_mask(scan, 0.0, nr, pad=1.5), r <= 7.0)
        # a half-sample scan (rows -8..2, 0.5 past the axis edge): over a full turn out to the farther end, over half
        # a turn to the nearer end
        half = {"dty0": -8.0, "ystep": 1.0, "n_rows": 11}
        nh = ix.mask.voxel_grid(half, 0.0)
        rh = np.hypot(*np.asarray(anri.geom.scan.recon_positions(nh, 1.0))[:, :2].T).reshape(nh, nh)
        turn = {**half, "n_o": round(360 / ix.data.B_O)}
        np.testing.assert_array_equal(ix.mask.disc_mask(turn, 0.0, nh), rh <= 8.5)
        np.testing.assert_array_equal(ix.mask.disc_mask({**half, "n_o": round(180 / ix.data.B_O)}, 0.0, nh), rh <= 2.5)
        np.testing.assert_array_equal(ix.mask.disc_mask(half, 0.0, nh), rh <= 2.5)  # no omega range: half a turn
        self.assertFalse(ix.mask.disc_mask(turn, 3.0, nh).any())  # y0 past the scan's end: no voxel seen all round
        wide = ix.mask.voxel_grid(scan, 0.0, grid_pad=2.5)  # 3 voxels more on every side
        self.assertEqual(wide, nr + 6)
        inner = np.asarray(anri.geom.scan.recon_positions(wide, 1.0)).reshape(wide, wide, 3)[3:-3, 3:-3]
        np.testing.assert_allclose(inner.reshape(-1, 3), np.asarray(anri.geom.scan.recon_positions(nr, 1.0)))

    def test_back_projection_grid(self):
        """The back-projection is on the histogram's grid (with --grid-pad), as the disc and the fit; else voxel_grid."""
        scan = {"dty0": -5.0, "ystep": 1.0, "n_rows": 11, "om0": -90.0}
        scan.update(n_e=2 * ix.data.R_E, n_o=8 * ix.data.R_O)  # 2 x 8 fit bins
        nr = ix.mask.voxel_grid(scan, 0.0)
        d = {"H": np.ones(2 * 8 * 11), "n_rings": 1, "scan": scan, "grid": {"n": nr + 6, "step": 1.0}}
        self.assertEqual(ix.diagnostics.back_projection(d, 0.0).shape, (nr + 6, nr + 6))
        self.assertEqual(ix.diagnostics.back_projection({**d, "grid": None}, 0.0).shape, (nr, nr))
        self.assertEqual(ix.diagnostics.back_projection(d, 0.0, nr).shape, (nr, nr))

    def test_support(self):
        """A square sample poking out of the disc is recovered from its projections, corners outside the disc too."""
        import jax.numpy as jnp

        from anri.geom.gonio import find_dty_for_beam_xy

        scan = {"y0": 0.0, "dty0": -20.0, "ystep": 1.0, "n_rows": 41, "om0": -90.0}
        nr, n_o, b_o = ix.mask.voxel_grid(scan, 0.0), 180, 1.0
        pos = np.asarray(anri.geom.scan.recon_positions(nr, 1.0))
        square = ((np.abs(pos[:, 0]) <= 16) & (np.abs(pos[:, 1]) <= 16)).reshape(nr, nr)  # corners at 22.6 > 20.5
        sino = np.zeros((n_o, 41))
        for i, om in enumerate(-90 + (np.arange(n_o) + 0.5) * b_o):  # chord length of each row: voxels per row
            dty = np.asarray(
                find_dty_for_beam_xy(jnp.asarray(pos[square.ravel()]), jnp.array([1.0, 0, 0]), om, 0.0, 0.0, 0.0)
            )
            sino[i] = np.bincount(np.clip(np.round(dty + 20).astype(int), 0, 40), minlength=41)
        m, _ = ix.mask.support_mask(sino[None, None].ravel(), 1, 1, n_o, scan, b_o, nr, level=0.03)
        self.assertGreater((m & square).sum() / (m | square).sum(), 0.9)  # the square, to about a voxel
        self.assertGreater((m & square & ~ix.mask.disc_mask(scan, 0.0, nr)).sum(), 10)  # its corners, outside the disc


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
        args = (jnp.asarray(B), jnp.asarray(rings["hkls"]), F2, jnp.asarray(rings["ring_j"]), geom, 0.2, None)
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
        args = (jnp.asarray(B), jnp.asarray(rings["hkls"]), F2, jnp.asarray(rings["ring_j"]), geom, 0.2, None)
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


class TestDetectorCoverage(unittest.TestCase):
    def test_masked_half(self):
        """A full detector covers each ring; with half its columns masked each ring is half covered, bins 0 or 1."""
        geom = geometry()
        tth = np.degrees(np.arctan(np.array([300.0, 600.0]) * 75.0 / PARS["distance"]))  # rings of 300, 600 pixels
        good = np.ones((2200, 2100), bool)
        cov = ix.data.detector_coverage(good, geom, tth, np.full(2, 0.05), 1.0, 360, sub=1)
        np.testing.assert_allclose(cov, 1.0)
        good[:, : int(PARS["y_center"])] = False
        cov = ix.data.detector_coverage(good, geom, tth, np.full(2, 0.05), 1.0, 360, sub=1)
        np.testing.assert_allclose(cov.mean(1), 0.5, atol=0.02)
        self.assertLessEqual(np.sum((cov > 0.01) & (cov < 0.99)), 8)  # only the bins the mask's edge crosses
        np.testing.assert_allclose(ix.data.coarsen_coverage(cov, 180), cov.reshape(2, 180, 2).mean(2))

    def test_detector_edge(self):
        """A ring window past the detector's edge is part covered, whether or not masked pixels extend it (#30)."""
        geom = {"det_origin_lab": jnp.array([100.0, -10.0, -10.0]), "s_step_lab": jnp.array([0.0, 0.0, 1.0]),
                "f_step_lab": jnp.array([0.0, 1.0, 0.0]), "k_in_lab": jnp.array([1.0, 0.0, 0.0])}  # fmt: skip
        tth = [np.degrees(np.arctan(0.1))]  # a ring of radius 10 pixels, its window 8.3-11.8, on 21 x 21 pixels
        cropped = ix.data.detector_coverage(np.ones((21, 21), bool), geom, tth, [1.0], 10.0, 36, sub=4, rows=16)
        good = np.zeros((21, 31), bool)
        good[:, :21] = True  # the same measured pixels, and masked ones beyond the edge
        extended = ix.data.detector_coverage(good, geom, tth, [1.0], 10.0, 36, sub=4, rows=16)
        self.assertLess(cropped[0, 9], 0.7)  # was 1: only the points on the detector were counted
        np.testing.assert_allclose(cropped, extended, atol=0.05)
        whole = {**geom, "det_origin_lab": jnp.array([100.0, -30.0, -30.0])}
        np.testing.assert_allclose(ix.data.detector_coverage(np.ones((61, 61), bool), whole, tth, [1.0], 10.0, 36), 1.0)

    def test_scales_the_system(self):
        """Coverage 1 changes nothing; coverage 0 on a ring removes its predictions' weight."""
        from anri.index.orientations import _system

        c, B, _ = iron()
        rings = ix.data.ring_table(*c, PARS["wavelength"], 3)
        F2 = jnp.ones(2 * len(rings["hkls"]), jnp.float32)
        args = (jnp.asarray(B), jnp.asarray(rings["hkls"]), F2, jnp.asarray(rings["ring_j"]), geometry(), 0.2)
        bins = (1.0, 1.0, 360, 180, 0.0)
        U = jnp.asarray(rotations(4, 3))
        _, w0 = _system(U, *args, None, bins)
        _, w1 = _system(U, *args, jnp.ones((3, 360), jnp.float32), bins)
        np.testing.assert_allclose(np.asarray(w1), np.asarray(w0))
        cov = np.ones((3, 360), np.float32)
        cov[1] = 0.0
        _, w2 = _system(U, *args, jnp.asarray(cov), bins)
        on1 = np.asarray(rings["ring_j"]) == 1
        self.assertEqual(float(np.abs(np.asarray(w2)[:, on1]).max()), 0.0)
        np.testing.assert_allclose(np.asarray(w2)[:, ~on1], np.asarray(w0)[:, ~on1])


class TestLocalSearch(unittest.TestCase):
    def test_children(self):
        """27 rotations around each orientation, the middle one itself, step / 3 apart along each sample axis."""
        u, _, vt = np.linalg.svd(rotations(2, 3).astype(np.float64))
        U = u @ vt  # exact rotations (rotations() gives float32)
        ch = ix.local.children(U, 3.0)
        self.assertEqual(ch.shape, (54, 3, 3))
        np.testing.assert_allclose(ch[13], U[0], atol=1e-12)
        np.testing.assert_allclose(ch[27 + 13], U[1], atol=1e-12)
        np.testing.assert_allclose(np.einsum("nji,njk->nik", ch, ch), np.broadcast_to(np.eye(3), ch.shape), atol=1e-12)
        ang = np.degrees(np.arccos(np.clip((np.einsum("nij,ij->n", ch[:27], U[0]) - 1) / 2, -1, 1)))
        self.assertAlmostEqual(np.sort(ang)[1], 1.0, places=6)  # nearest neighbours 1 deg off
        self.assertAlmostEqual(ang.max(), np.sqrt(3), places=6)  # the corners

    def test_kept_and_inherit(self):
        """A voxel's orientations holding min_share of its occupancy are subdivided; it inherits their children only."""
        cand = np.array([[5, 2, 7, -1], [2, 9, -1, -1]], np.int32)
        f = np.array([[0.6, 0.38, 0.005, 0.0], [1.0, 0.0, 0.0, 0.0]], np.float32)
        use, f0, c0 = ix.local.inherit(f, cand, 10, min_share=0.01, m=2)
        np.testing.assert_array_equal(np.flatnonzero(use), [2, 5])  # 7 holds under 1%, 9 nothing
        self.assertEqual(c0.shape, (2, 54))
        np.testing.assert_array_equal(c0[0, :27], np.arange(27, 54))  # orientation 5: the second subdivided
        np.testing.assert_array_equal(c0[0, 27:], np.arange(27))  # orientation 2: the first
        np.testing.assert_allclose(f0[0], np.repeat([0.6 / 27, 0.38 / 27], 27), rtol=1e-6)
        np.testing.assert_array_equal(c0[1, :27], np.arange(27))
        np.testing.assert_array_equal(c0[1, 27:], -1)
        np.testing.assert_array_equal(f0[1, 27:], 0)
        # a small share is kept however much more the others hold (a thin twin), unlike a cover of the largest
        use, _, _ = ix.local.inherit(np.array([[0.97, 0.03]], np.float32), np.array([[0, 1]], np.int32), 2)
        np.testing.assert_array_equal(use, [True, True])

    def test_population_radius(self):
        """Populations are grouped within a histogram bin, or 1.8 x the fitted orientations' spacing if coarser."""
        self.assertEqual(ix.populations.radius(0.0926), 1.0)  # a fit with levels: the bin
        self.assertAlmostEqual(ix.populations.radius(2.5), 4.5)  # a plain fit of a 2.5 deg grid

    def test_level_rows(self):
        """Each level's voxels are about radius x its worst case wide, the last on the dty step."""
        # 141.5 um radius, 0.289 deg: 0.71 um, 3 rows of 0.25 um; then 0.096 deg: 1 row (DTL1z45)
        self.assertEqual(ix.local.level_rows(141.5, 0.289, 0.25, 1), [3, 1])
        self.assertEqual(ix.local.level_rows(141.5, 2.6, 0.25, 2), [26, 9, 1])  # the last forced to one row
        self.assertEqual(ix.local.level_rows(5.0, 0.29, 1.0, 0), [1])

    def test_predictions_in_chunks(self):
        """Predicting in chunks gives the same predictions, padded to whole chunks that predict nothing."""
        c, B, _ = iron()
        rings = ix.data.ring_table(*c, PARS["wavelength"], 3)
        U = rotations(40, 5)
        whole = ix.search.predictions(U, B, rings, geometry(), 0.2)
        chunked = ix.search.predictions(U, B, rings, geometry(), 0.2, chunk=32)
        self.assertEqual(chunked[0].shape[0], 64)
        for a, b in zip(whole, chunked):
            np.testing.assert_array_equal(np.asarray(a)[:40], np.asarray(b)[:40])
        self.assertFalse(np.asarray(chunked[2])[40:].any())


class TestInherit(unittest.TestCase):
    def test_neighbourhood_candidates(self):
        """A fine voxel gets the orientations its 3 x 3 coarse neighbourhood occupied, its own coarse voxel's first."""
        gc = np.arange(-2, 3) * 4.0
        xx, yy = np.meshgrid(gc, gc, indexing="ij")
        pos_c = np.stack([xx.ravel(), yy.ravel(), np.zeros(25)], 1)
        f_c = np.zeros((25, 2), np.float32)
        cand_c = np.zeros((25, 2), np.int32)
        for c in range(25):
            cand_c[c] = [c, 100]  # every coarse voxel holds its own orientation and a shared one
            f_c[c] = [1.0 + c, 0.5]
        fine = np.array([[0.0, 0.0, 0.0], [7.0, 7.0, 0.0]])  # the centre (coarse voxel 12), and near coarse voxel 18
        cand, f0, lost = ix.occupancy.inherit_candidates(fine, pos_c, f_c, cand_c, 12)
        self.assertEqual(list(cand[0][:2]), [12, 100])  # its own coarse voxel's first, by occupancy
        others = [6, 7, 8, 11, 13, 16, 17, 18]
        self.assertEqual(list(cand[0][2:10]), sorted(others, reverse=True))  # then the neighbours', by occupancy
        np.testing.assert_allclose(f0[0][2:10], 1.0 + cand[0][2:10])  # starting from their occupancy there
        self.assertEqual(f0[0][1], 0.5)  # the shared one once
        np.testing.assert_array_equal(cand[0][10:], -1)  # nothing else: empty slots, never an orientation -1
        np.testing.assert_array_equal(f0[0][10:], 0)
        self.assertIn(24, cand[1])
        np.testing.assert_array_equal(lost, 0)
        # too few slots: the neighbours' go before the voxel's own
        cand, _, lost = ix.occupancy.inherit_candidates(fine[:1], pos_c, f_c, cand_c, 1)
        self.assertEqual(cand[0, 0], 12)
        self.assertAlmostEqual(float(lost[0]), 0.5 / 13.5, places=6)

    def test_coarse_positions(self):
        """A coarse grid centred on the axis, its voxels within 0.75 coarse steps of the fine ones."""
        fine = np.asarray(anri.geom.scan.recon_positions(9, 1.0))
        fine = fine[np.hypot(fine[:, 0], fine[:, 1]) <= 3.0]
        pos_c = ix.occupancy.coarse_positions(fine, 3.0)
        np.testing.assert_allclose(np.sort(np.unique(pos_c[:, 0])), [-3.0, 0.0, 3.0], atol=1e-6)
        self.assertEqual(len(pos_c), 9)  # the 3 x 3 coarse voxels around a disc of radius 3

    def test_duplicates(self):
        """An orientation in several neighbours ranks by its largest occupancy, and comes once (issue #28)."""
        near = jnp.arange(9)[None]
        c = np.arange(18, dtype=np.int32).reshape(9, 2)
        c[8, 0] = 0  # orientation 0: weak in neighbour 0, strong in neighbour 8
        f = np.full((9, 2), 2.0, np.float32)
        f[0, 0], f[8, 0] = 0.01, 100.0
        out, w, _ = ix.occupancy._inherit(near, jnp.asarray(f), jnp.asarray(c), 2)
        self.assertEqual((int(out[0, 0]), float(w[0, 0])), (0, 100.0))  # in the nearest (0), at its largest (8)
        c = np.tile(np.array([0, 1], np.int32), (9, 1))  # two orientations in all: the other slots empty
        out = np.asarray(ix.occupancy._inherit(near, jnp.ones((9, 2)), jnp.asarray(c), 4)[0])
        np.testing.assert_array_equal(np.sort(out[0, :2]), [0, 1])
        np.testing.assert_array_equal(out[0, 2:], [-1, -1])


class TestMaskFiles(unittest.TestCase):
    def test_load_mask(self):
        """A mask carries its voxel size; another size of grid is centred on, another voxel size refused."""
        m = np.zeros((9, 9), bool)
        m[3:6, 2:7] = True
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "m.npz")
            ix.mask.save_mask(p, m, 2.0)
            got = ix.mask.load_mask(p, 13, 2.0, log=lambda s: None)
            self.assertEqual(got.shape, (13, 13))
            np.testing.assert_array_equal(got[2:-2, 2:-2], m)
            with self.assertRaises(ValueError):
                ix.mask.load_mask(p, 13, 1.0)
            q = os.path.join(tmp, "m.npy")
            np.save(q, m)
            np.testing.assert_array_equal(ix.mask.load_mask(q, 7, 2.0, log=lambda s: None), m[1:-1, 1:-1])


class TestPowder(unittest.TestCase):
    def _pattern(self, good, step=0.1, n_eta=36, n_frames=2):
        """Every measured pixel lit in n_frames frames with c x its solid angle: a uniform 2 per sr per frame."""
        geom = geometry()
        n_t = int(30 / step)
        Om, npx = ix.data.detector_cake(good, geom, 0.0, step, n_t, n_eta)
        s, f = np.nonzero(good)
        s, f = s.astype(np.float32), f.astype(np.float32)
        val = 2.0 * np.asarray(ix.data.pixel_solid_angle(jnp.asarray(s), jnp.asarray(f), geom))
        chunk = [(s, f, np.zeros_like(s), np.zeros_like(s), val)] * n_frames
        I, hits = ix.data.pixel_cake(chunk, geom, 0.0, step, n_t, len(s), n_eta, counts=True)  # [2, n_t, n_eta]
        return ix.data.powder_pattern(I, Om, npx, hits, n_frames, 0.0, step)

    def test_uniform_intensity(self):
        """A uniform intensity per sr comes out flat; whole rings have coverage 1 (their analytic solid angle)."""
        p = self._pattern(np.ones((2200, 2100), bool))
        whole = (p["tth"] > 1.0) & (p["tth"] < 20.0)  # rings entirely on the detector
        np.testing.assert_allclose(p["intensity"][whole], 2.0, rtol=1e-4)
        np.testing.assert_allclose(p["coverage"][whole], 1.0, atol=0.02)
        np.testing.assert_allclose(p["lit_fraction"][whole], 1.0)
        np.testing.assert_allclose(p["sigma"][whole], p["sigma_counting"][whole])  # no spread over eta
        self.assertTrue(np.all(p["coverage"][p["tth"] > 28.5] < 0.98))  # rings past the nearest edge (27.7 deg)

    def test_masked_half(self):
        """Masking half the detector halves the coverage, not the intensity."""
        good = np.ones((2200, 2100), bool)
        good[:, : int(PARS["y_center"])] = False
        p = self._pattern(good)
        whole = (p["tth"] > 1.0) & (p["tth"] < 20.0)
        np.testing.assert_allclose(p["intensity"][whole], 2.0, rtol=1e-4)
        np.testing.assert_allclose(p["coverage"][whole], 0.5, atol=0.02)

    def test_write_xye(self):
        """The .xye holds the covered bins, with the wavelength in its header."""
        from anri.io.provenance import write_provenance

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x_powder.h5")
            with h5py.File(path, "w") as h:
                write_provenance(h, "powder", None, {}, {"wavelength": 0.2})
                h["tth"], h["intensity"], h["sigma"] = np.arange(4.0), np.arange(4.0) * 10, np.ones(4)
                h["coverage"] = np.array([0.0, 0.5, 1.0, 0.1])
                h["intensity"].attrs["units"] = "per sr per frame"
            xye = ix.stages.write_xye(path, min_coverage=0.3)
            np.testing.assert_allclose(np.loadtxt(xye), [[1.0, 10.0, 1.0], [2.0, 20.0, 1.0]])
            with open(xye) as fh:
                self.assertIn("wavelength 0.200000", fh.readline())
