import os
import tempfile
import unittest

import jax.numpy as jnp
import numpy as np

import anri.crystal
import anri.geom
import anri.index as ix
import anri.io

PARS = {
    "y_center": 1049.9, "y_size": 75.0, "tilt_y": 0.0, "z_center": 1116.5, "z_size": 75.0, "tilt_z": 0.0,
    "tilt_x": 0.0, "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1,
    "wavelength": 12.398419843320026 / 43.0, "wedge": 0.0, "chi": 0.0,
}  # fmt: skip
A_FE = 2.8665


def geometry(sig_beam=0.5):
    g = anri.io.geom_from_pars(PARS, 0.0, PARS["wavelength"] * 1e-4, 5e-5, 5e-5, sig_beam=sig_beam, voxel_size=1.0)
    return {
        k: jnp.asarray(v, jnp.float32) if np.issubdtype(np.asarray(v).dtype, np.floating) else v for k, v in g.items()
    }


def iron():
    c = anri.crystal.Crystal(
        anri.crystal.UnitCell.from_lpars([A_FE] * 3 + [90.0] * 3), anri.crystal.Symmetry.from_number(229)
    )
    return c, np.asarray(c.B, np.float32), anri.crystal.laue_rotations(np.asarray(c.sym_ops), np.asarray(c.B))


def rotations(n, seed):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return anri.crystal.quat_to_mat(q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


class TestRings(unittest.TestCase):
    def test_bcc_rings(self):
        c, _, _ = iron()
        rings = ix.ring_table(c, PARS["wavelength"], 3)
        h = rings["hkls"]
        self.assertTrue(np.all(np.sum(h, 1) % 2 == 0))  # I-centred: h + k + l even
        self.assertEqual(sorted(np.bincount(rings["ring_j"]) // 2), [6, 12, 24])  # {200}, {110}, {211}
        np.testing.assert_allclose(np.sort(np.sum(h * h, 1))[[0, 12, 18]], [2, 4, 6])

    def test_ring_widths(self):
        ring_tth = np.array([5.0, 6.0])
        lo, step = 4.5, 0.002
        x = lo + (np.arange(1000) + 0.5) * step
        prof = 100 * np.exp(-0.5 * ((x - 5.01) / 0.02) ** 2) + 50 * np.exp(-0.5 * ((x - 6.0) / 0.04) ** 2) + 1.0
        off, hw = ix.ring_widths(prof, lo, step, ring_tth)
        np.testing.assert_allclose(off, [0.01, 0.0], atol=0.002)
        np.testing.assert_allclose(hw, [1.96 * 0.02, 1.96 * 0.04], rtol=0.1)  # 95% of a Gaussian


class TestTolerances(unittest.TestCase):
    def test_first_order_bound(self):
        """A rotation by delta moves each reflection by at most the tolerances (|sin eta| > 0.3)."""
        c, B, _ = iron()
        rings = ix.ring_table(c, PARS["wavelength"], 3)
        geom = geometry()
        U0 = rotations(200, 0)
        axis = np.random.default_rng(1).normal(size=(200, 3))
        dU = anri.crystal.quat_to_mat(np.column_stack([np.full(200, np.cos(np.radians(0.5))),
                                                       np.sin(np.radians(0.5)) * axis / np.linalg.norm(axis, axis=1, keepdims=True)]))  # fmt: skip
        e0, o0, k0 = ix.predict(jnp.asarray(U0), jnp.asarray(B), jnp.asarray(rings["hkls"]), geom)
        e1, o1, k1 = ix.predict(
            jnp.asarray((dU @ U0).astype(np.float32)), jnp.asarray(B), jnp.asarray(rings["hkls"]), geom
        )
        te, to = ix.match_tolerances(
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
        table = ix.lit_table(jnp.asarray(lit))
        eta = jnp.array([[12.2, 10.5, 10.5]])
        om = jnp.array([[50.5, 52.7, 50.5]])
        ok = jnp.ones((1, 3), bool)

        def comp(te, to):
            c, n = ix.completeness(table, eta, om, ok, jnp.zeros(3, jnp.int32), jnp.full((1, 3), te), jnp.full((1, 3), to),
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
        rings = ix.ring_table(c, PARS["wavelength"], 2)
        geom = geometry()
        pred = ix.predictions(rotations(16, 2), B, rings, geom, 0.2)
        rng = np.random.default_rng(3)
        pos = jnp.asarray(np.column_stack([rng.uniform(-4, 4, (32, 2)), np.zeros(32)]), jnp.float32)
        scan = {"y0": 0.0, "dty0": -6.0, "ystep": 1.0, "n_rows": 13, "om0": 0.0}
        dims = (1.0, 1.0, 360, 180)
        n_cells = 2 * 360 * 180 * 13
        cand = jnp.asarray(rng.integers(0, 16, (32, 4)), jnp.int32)
        f = jnp.asarray(rng.uniform(size=(32, 4)), jnp.float32)
        r = jnp.asarray(rng.uniform(size=n_cells), jnp.float32)
        Af = ix.forward(f, cand, pred, jnp.asarray(rings["ring_j"]), pos, scan, dims, n_cells, 8)
        ATr = ix.backward(r, cand, pred, jnp.asarray(rings["ring_j"]), pos, scan, dims, 8)
        self.assertGreater(float(jnp.sum(Af)), 0)
        np.testing.assert_allclose(float(jnp.sum(Af * r)), float(jnp.sum(f * ATr)), rtol=1e-4)

    def test_coarsen_rows(self):
        H = np.arange(2 * 7, dtype=float)
        Hc, n = ix.coarsen_rows(H, 7, 3)
        self.assertEqual(n, 3)
        np.testing.assert_allclose(np.asarray(Hc), [0 + 1 + 2, 3 + 4 + 5, 6, 7 + 8 + 9, 10 + 11 + 12, 13])


class TestPopulations(unittest.TestCase):
    def test_parent_twin_and_decoy(self):
        _, _, ops = iron()
        U0 = rotations(1, 4)[0].astype(float)
        twin = anri.crystal.quat_to_mat(np.r_[np.cos(np.radians(30)), np.sin(np.radians(30)) * np.ones(3) / np.sqrt(3)])
        small = [anri.crystal.quat_to_mat(np.r_[np.cos(np.radians(a / 2)), np.sin(np.radians(a / 2)) * np.array(v)])
                 for a, v in ((0.8, [1, 0, 0]), (0.8, [0, 1, 0]), (0.8, [0, 0, -1]))]  # fmt: skip
        U_list = np.stack([U0, *(s @ U0 for s in small), twin @ U0, rotations(1, 5)[0]])  # parent x 4, twin, decoy
        f = np.array([[0.3, 0.2, 0.15, 0.15, 0.17, 0.03]])
        cand = np.arange(6)[None]
        frac, U, spread, n = ix.populations(f, cand, U_list, ops, 2.0)
        np.testing.assert_allclose(frac[0, :3], [0.8, 0.17, 0.03], atol=1e-5)
        np.testing.assert_array_equal(n[0, :3], [4, 1, 1])
        self.assertLess(anri.crystal.disorientation(U[0, :1], U0[None], ops)[0], 0.5)
        self.assertLess(anri.crystal.disorientation(U[0, 1:2], (twin @ U0)[None], ops)[0], 0.05)  # float32
        self.assertGreater(spread[0, 0], 0.2)


class TestEndToEnd(unittest.TestCase):
    def test_two_grains(self):
        """Simulate two grains to ImageD11 files, then index them from scratch."""
        c, B, ops = iron()
        wl = PARS["wavelength"]
        rings = ix.ring_table(c, wl, 3)
        U_true = rotations(2, 7)
        i, j = np.mgrid[0:6, 0:6]
        pos = np.stack([i.ravel() - 2.5, j.ravel() - 2.5, np.zeros(36)], 1)
        grain = (pos[:, 0] > 0).astype(int)
        entries = {"ubi": np.linalg.inv(U_true[grain] @ B), "pos": pos, "density": np.full(36, 30.0)}
        geom = geometry()
        omega, dty = anri.io.motor_grid((0.0, 180.0), 0.5, (-5.0, 5.0), 1.0)
        a = A_FE
        cell = {"cell__a": a, "cell__b": a, "cell__c": a, "cell_alpha": 90.0, "cell_beta": 90.0, "cell_gamma": 90.0,
                "cell_lattice_[P,A,B,C,I,F,R]": 229}  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            sparse = os.path.join(tmp, "sparse.h5")
            anri.io.simulate_sparse(sparse, entries, rings["hkls"], np.ones(len(rings["hkls"])), geom, omega, dty,
                                    (2162, 2068), window=(5, 9, 9))  # fmt: skip
            parfile = anri.io.write_pars(os.path.join(tmp, "pars"), PARS, {"Fe": cell})
            dsfile = anri.io.write_dataset(sparse, tmp, "fe", "sim", y0=0.0, parfile=parfile)
            ds = anri.io.read_dataset(dsfile)
            _, phase, cell_read = anri.io.read_pars_json(ds["parfile"])
            self.assertEqual(phase, "Fe")
            self.assertEqual(cell_read["cell_lattice_[P,A,B,C,I,F,R]"], 229)

            def stream():
                return anri.io.stream_sparse(sparse, ds["ybinedges"], ds["omegamotor"], ds["dtymotor"], 1 << 16)

            _, hw = ix.ring_profile(stream(), geom, rings["tth"], 1 << 16)
            rings["hw"] = hw
            nk = len(ds["ybincens"])
            om0 = float(ds["obinedges"][0])
            H_lit = ix.histogram_pixels(stream(), geom, rings["tth"], 0.1, om0, (0.5, 0.5, 720, 360), 1, 1 << 16)
            H = ix.histogram_pixels(stream(), geom, rings["tth"], 0.1, om0, (1.0, 1.0, 360, 180), nk, 1 << 16)
        Hs = H_lit.reshape(3, 720, 360)
        lit = {
            "table": ix.lit_table(Hs > 0),
            "om0": om0,
            "bins": (0.5, 0.5, 720, 360),
            "frame_step": 0.5,
            "etacut": 0.2,
        }
        U_grid, delta = anri.crystal.orientation_grid(5.0, ops)
        kept, _, info = ix.prune(U_grid, delta, B, rings, geom, lit)
        self.assertGreater(info["min_comp"], info["chance"])
        U_kept = U_grid[kept]
        pred = ix.predictions(U_kept, B, rings, geom, 0.2)
        _, pad = anri.geom.sino_shift_and_pad(0.0, nk, float(ds["ybincens"][0]), 1.0)
        nr = nk + pad
        vox = np.asarray(anri.geom.recon_positions(nr, 1.0), np.float32)
        scan = {"y0": 0.0, "dty0": float(ds["ybincens"][0]), "ystep": 1.0, "n_rows": nk, "om0": om0}
        f, cand = ix.fit_occupancy(
            H, pred, rings["ring_j"], vox, scan, (1.0, 1.0, 360, 180), k=16, n_iter=10, log=lambda m: None
        )
        frac, U_pop, _, _ = ix.populations(f, cand, U_kept, ops, 1.8 * 5.0)
        # voxels well inside each grain find it as their main population, to within the grid's spacing
        for g, sel in ((0, (vox[:, 0] < -0.9) & (vox[:, 0] > -2.6)), (1, (vox[:, 0] > 0.9) & (vox[:, 0] < 2.6))):
            sel &= np.abs(vox[:, 1]) < 2.6
            err = anri.crystal.disorientation(U_pop[sel, 0], np.repeat(U_true[g][None], sel.sum(), 0), ops)
            self.assertLess(np.median(err), delta / 2, msg=f"grain {g}")
            self.assertGreater(np.median(frac[sel, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
