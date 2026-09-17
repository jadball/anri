import os
import subprocess
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import phantom

from anri.render import (
    Frames,
    Measured,
    PeakTable,
    loss_and_grad,
    render_frames,
    render_sum,
    sparse_frames,
    splat,
    write_sparse,
)
from anri.render._impl.sparse import accumulate_extract, new_canvas

jax.config.update("jax_enable_x64", True)
THR = 0.05
EXACT = {
    "windows": (9,),
    "tolerance": 0.0,
    "pixel_cut": 0.0,
}  # same kernel as brute_frame, nothing culled by brightness


def brute_frame(frames, g, gc, amp, idx):
    lo, hi = frames.intervals(np.array([idx]))
    return np.asarray(
        splat(g, gc, amp, jnp.asarray(lo[0]), jnp.asarray(hi[0]), frames.shape, (0, 1), (9, 9), 2, frames.periods)
    )


def collect(results):
    return [np.concatenate([r[i] for r in results]) for i in range(5)]


class TestFrames(unittest.TestCase):
    def test_interlaced_and_lookup(self):
        fr = phantom.scan_frames(n_dty=3, ostep=10.0)
        self.assertEqual(fr.n_frames, 3 * 36)
        row1 = slice(fr.row_offsets[1], fr.row_offsets[2])
        # odd scan runs backwards: omega bin 0 is its last frame
        self.assertEqual(fr.frame[row1][0], 35)
        np.testing.assert_array_equal(fr.obin[row1], np.arange(36))
        img = fr.frame_image()
        self.assertEqual(img.shape, (36, 3))
        self.assertTrue((img >= 0).all())
        np.testing.assert_array_equal(fr.obin[img[:, 2]], np.arange(36))

    def test_intervals_and_grid(self):
        fr = phantom.scan_frames(n_dty=3, ostep=10.0)
        lo, hi = fr.intervals(np.arange(fr.n_frames))
        np.testing.assert_allclose(hi - lo, 1.0)
        np.testing.assert_allclose(0.5 * (lo + hi)[:, 0], fr.obin)
        mu = jnp.array([[1.0, 2.0, -10.0, 50.0]])
        cov = jnp.eye(4)[None]
        g, gc = fr.to_grid(mu, cov)
        np.testing.assert_allclose(np.asarray(g[0]), [1.0, 2.0, 35.0, 2.0])  # -10 deg wraps to bin 35
        np.testing.assert_allclose(np.diag(np.asarray(gc[0])), [1.0, 1.0, 0.01, 1 / 2500])

    def test_partial_scan_wraps_to_scan(self):
        obincens = np.arange(100.0, 150.0, 1.0)
        fr = Frames.from_arrays((10, 10), obincens, obincens)
        self.assertFalse(fr.omega_periodic)
        g, _ = fr.to_grid(jnp.array([[0.0, 0.0, -240.0]]), jnp.eye(3)[None])
        self.assertAlmostEqual(float(g[0, 2]), 20.0)  # -240 == 120 degrees


class _Scan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = phantom.scan_source()
        cls.fr = phantom.scan_frames()
        cls.tab = PeakTable.build(cls.src, cls.fr, chunk=4096, n_sigma=5)
        mu, cov, cls.amp = cls.src(cls.src.params, cls.src.device_ids(np.arange(cls.src.n_peaks)))
        cls.g, cls.gc = cls.fr.to_grid(mu, cov)


class TestSource(_Scan):
    def test_omegasign_flips_omega(self):
        s = self.src
        flipped = type(s)(
            np.asarray(s.params["ubi"]),
            np.asarray(s.params["origin"]),
            np.asarray(s.params["weight"]),
            np.asarray(s.hkl),
            np.asarray(s.intensity),
            phantom.geometry(True),
            np.asarray(s.cov_in),
            extra_var=np.asarray(s.extra_var),
            omegasign=-1.0,
            dtype=jnp.float64,
        )
        ids = s.device_ids(np.arange(200))
        mu0, cov0, amp0 = s(s.params, ids)
        mu1, cov1, amp1 = flipped(flipped.params, ids)
        np.testing.assert_allclose(np.asarray(mu1[:, 2]), -np.asarray(mu0[:, 2]))
        np.testing.assert_allclose(np.asarray(mu1[:, [0, 1, 3]]), np.asarray(mu0[:, [0, 1, 3]]))
        np.testing.assert_allclose(np.asarray(cov1[:, 2, 0]), -np.asarray(cov0[:, 2, 0]))
        np.testing.assert_allclose(np.asarray(cov1[:, 2, 2]), np.asarray(cov0[:, 2, 2]))
        np.testing.assert_array_equal(np.asarray(amp1), np.asarray(amp0))


class TestTable(_Scan):
    def test_candidates_cover_reachable_peaks(self):
        g, gc, amp = (np.asarray(x) for x in (self.g, self.gc, self.amp))
        reach = 5 * np.sqrt(gc[:, 3, 3])
        for r in (0, 4, 8):
            ids, lo, hi = self.tab.candidates(r, r + 1)
            self.assertEqual(ids.size, np.unique(ids).size)
            expected = np.flatnonzero((amp > 0) & (np.abs(g[:, 3] - r) <= reach + 0.5))
            self.assertTrue(np.isin(expected, ids).all())
            self.assertTrue(((lo <= r) & (hi >= r)).all())


class TestSparseAndDense(_Scan):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.results = list(
            sparse_frames(
                cls.src,
                cls.tab,
                cls.fr,
                THR,
                batch=4096,
                slots=16,
                weight_cut=1e-6,
                blocks=cls.tab.blocks(rows_per_block=4),
                **EXACT,
            )
        )
        cls.stats = {}
        cls.classed = list(sparse_frames(cls.src, cls.tab, cls.fr, THR, batch=4096, slots=16, stats=cls.stats))

    def test_every_frame_once(self):
        done = np.concatenate([r[4] for r in self.results])
        np.testing.assert_array_equal(np.sort(done), np.arange(self.fr.n_frames))

    def test_sparse_matches_brute_force(self):
        f, r, c, v, _ = collect(self.results)
        busy = np.bincount(f).argsort()[-4:]
        for idx in list(busy) + [0, 1000]:
            ref = brute_frame(self.fr, self.g, self.gc, self.amp, idx)
            got = np.zeros(self.fr.det_shape)
            m = f == idx
            got[r[m], c[m]] = v[m]
            ref = np.where(ref > THR, ref, 0.0)
            clear = np.abs(ref - THR) > 0.01 * THR  # pixels right at threshold may flip with the weight cut
            np.testing.assert_allclose(got[clear], ref[clear], rtol=1e-4, atol=1e-6)

    def test_default_classes_within_tolerance(self):
        f0, r0, c0, v0, _ = collect(self.results)
        f1, r1, c1, v1, _ = collect(self.classed)
        a = np.zeros((self.fr.n_frames, *self.fr.det_shape))
        b = np.zeros_like(a)
        a[f0, r0, c0] = v0
        b[f1, r1, c1] = v1
        clear = np.abs(a - THR) > 0.05 * THR
        self.assertLess(np.abs(a - b)[clear].max(), 0.03 * a.max())
        self.assertGreater(self.stats["kept 5px separable"], 0)
        self.assertLess(self.stats["contributions kept"], self.stats["contributions tried"])

    def test_pixels_sorted(self):
        f, r, c, _, _ = collect(self.results)
        key = (f * self.fr.det_shape[0] + r) * self.fr.det_shape[1] + c
        self.assertTrue((np.diff(key) > 0).all())

    def test_writer_round_trip(self):
        import h5py

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sparse.h5")
            write_sparse(path, self.fr, iter(self.results), intensity_dtype=np.float32)
            with h5py.File(path, "r") as h:
                self.assertEqual(sorted(h, key=float), [f"{i + 1}.1" for i in range(9)])
                g = h["2.1"]
                self.assertEqual(int(g.attrs["nframes"]), 360)
                self.assertEqual((int(g.attrs["shape0"]), int(g.attrs["shape1"])), self.fr.det_shape)
                np.testing.assert_allclose(g["measurement/rot_center"][:3], [359.0, 358.0, 357.0])
                np.testing.assert_allclose(g["rot_center"][:3], [359.0, 358.0, 357.0])
                self.assertAlmostEqual(float(g["instrument/positioners/dty"][()]), -150.0)
                self.assertEqual(int(g["nnz"][:].sum()), g["row"].shape[0])
            meas = Measured.from_sparse_file(path, self.fr)
            counts_path = os.path.join(tmp, "counts.h5")
            write_sparse(counts_path, self.fr, iter(self.results), max_counts=3)
            with h5py.File(counts_path, "r") as h:
                ints = np.concatenate([h[n]["intensity"][:] for n in h])
            self.assertEqual(ints.dtype, np.uint32)
            self.assertGreaterEqual(int(ints.min()), 1)
            self.assertLessEqual(int(ints.max()), 3)
        f, r, c, v, _ = collect(self.results)
        self.assertEqual(meas.pixel.size, f.size)
        order = np.lexsort((c, r, f))
        np.testing.assert_allclose(np.sort(meas.value), np.sort(v[order].astype(np.float32)), rtol=1e-6)

    def test_render_frames(self):
        idx = int(np.bincount(collect(self.results)[0]).argmax())
        near = np.flatnonzero((self.fr.row == self.fr.row[idx]) & (np.abs(self.fr.obin - self.fr.obin[idx]) <= 2))
        near = near[near != idx]
        imgs = render_frames(
            self.src, self.tab, self.fr, [[idx], list(near)], weight_cut=1e-9, batch=4096, windows=(9,), tolerance=0.0
        )
        ref0 = brute_frame(self.fr, self.g, self.gc, self.amp, idx)
        ref1 = sum(brute_frame(self.fr, self.g, self.gc, self.amp, j) for j in near)
        np.testing.assert_allclose(imgs[0], ref0, atol=1e-6 * ref0.max())
        np.testing.assert_allclose(imgs[1], ref1, atol=1e-4 * ref1.max())

    def test_render_sum(self):
        det = render_sum(self.src, self.tab, self.fr, "detector")
        ref = np.asarray(
            splat(
                self.g,
                self.gc,
                self.amp,
                jnp.array([-1e9, -1e9]),
                jnp.array([1e9, 1e9]),
                self.fr.shape,
                (0, 1),
                (9, 9),
                2,
                (0.0,) * 4,
            )
        )
        np.testing.assert_allclose(det, ref, atol=1e-9 * ref.max())
        sino = render_sum(self.src, self.tab, self.fr, "sinogram")
        self.assertEqual(sino.shape, (360, 9))
        self.assertAlmostEqual(sino.sum(), float(self.amp.sum()), delta=1e-3)
        dty_band = render_sum(self.src, self.tab, self.fr, "detector", dty=(-25.0, 25.0))
        self.assertLess(dty_band.sum(), det.sum())
        self.assertGreater(dty_band.sum(), 0.0)


class TestLoss(_Scan):
    def test_zero_at_truth_and_gradient(self):
        res = list(sparse_frames(self.src, self.tab, self.fr, THR, batch=4096, slots=64, blocks=[(3, 5)], **EXACT))
        f, r, c, v, _ = collect(res)
        meas = Measured.from_arrays(self.fr, f, r, c, v)
        kw = {"batch": 4096, "slots": 64, "blocks": [(3, 5)]}
        loss0, _ = loss_and_grad(self.src.params, self.src, self.tab, self.fr, meas, THR, **kw)
        self.assertLess(loss0, 1e-9)

        p = dict(self.src.params)
        p["origin"] = p["origin"] + jnp.array([3.0, 0.0, 0.0])
        loss1, grad = loss_and_grad(p, self.src, self.tab, self.fr, meas, THR, **kw)
        self.assertGreater(loss1, 1e-3)
        rng = np.random.default_rng(0)
        direction = {k: jnp.zeros_like(x) for k, x in p.items()}
        direction["origin"] = jnp.asarray(rng.normal(size=p["origin"].shape)) * jnp.array([1.0, 1.0, 0.0])
        direction["weight"] = jnp.asarray(rng.normal(size=p["weight"].shape)) * 0.01
        eps = 1e-4
        plus, _ = loss_and_grad(
            jax.tree.map(lambda a, d: a + eps * d, p, direction), self.src, self.tab, self.fr, meas, THR, **kw
        )
        minus, _ = loss_and_grad(
            jax.tree.map(lambda a, d: a - eps * d, p, direction), self.src, self.tab, self.fr, meas, THR, **kw
        )
        analytic = sum(float(jnp.vdot(grad[k], direction[k])) for k in p)
        self.assertAlmostEqual((plus - minus) / (2 * eps), analytic, delta=1e-4 * abs(analytic) + 1e-8)


class TestBoxBeam(unittest.TestCase):
    def test_sparse_across_omega_wrap(self):
        src, fr = phantom.box_source(), phantom.box_frames()
        tab = PeakTable.build(src, fr, chunk=1024)
        mu, cov, amp = src(src.params, src.device_ids(np.arange(src.n_peaks)))
        g, gc = fr.to_grid(mu, cov)
        f, r, c, v, done = collect(
            list(
                sparse_frames(
                    src,
                    tab,
                    fr,
                    THR,
                    batch=2048,
                    slots=16,
                    weight_cut=1e-6,
                    blocks=tab.blocks(rows_per_block=90),
                    **EXACT,
                )
            )
        )
        np.testing.assert_array_equal(np.sort(done), np.arange(fr.n_frames))
        for idx in (0, 359, int(np.bincount(f).argmax())):
            ref = np.where((x := brute_frame(fr, g, gc, amp, idx)) > THR, x, 0.0)
            got = np.zeros(fr.det_shape)
            m = f == idx
            got[r[m], c[m]] = v[m]
            np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)
        det = render_sum(src, tab, fr, "detector")
        self.assertAlmostEqual(det.sum(), float(amp.sum()), delta=1e-6 * float(amp.sum()))


class TestAccumulateExtract(unittest.TestCase):
    def test_matches_dense(self):
        rng = np.random.default_rng(0)
        slots, npix = 4, 64 * 64
        canvas = new_canvas(slots, npix)
        for cap in (2000, 64):
            n = 5000
            keys = rng.integers(0, 300, n) + rng.integers(0, slots, n) * npix
            keys[rng.random(n) < 0.1] = slots * npix  # padding
            keys = keys.astype(np.int32)
            vals = rng.exponential(1.0, n).astype(np.float32)
            canvas, k, v, nnz, hits = accumulate_extract(
                canvas, jnp.asarray(keys), jnp.asarray(vals), 3.0, slots, npix, cap
            )
            dense = np.zeros(slots * npix + 1)
            np.add.at(dense, keys, vals)
            expect = np.flatnonzero(dense[: slots * npix] > 3.0)
            self.assertEqual(int(hits), int(np.isin(keys, expect).sum()))
            self.assertEqual(float(jnp.abs(canvas).max()), 0.0)
            if int(hits) > cap:
                continue  # caller must retry with more room
            np.testing.assert_array_equal(np.asarray(k)[: int(nnz)], expect)
            np.testing.assert_allclose(np.asarray(v)[: int(nnz)], dense[expect], rtol=1e-5)


class TestBackend(unittest.TestCase):
    def test_configure_sets_cpu_devices(self):
        code = (
            "import anri.backend as b; b.configure(device='cpu', n_cpu_devices=3); import jax; "
            "i = b.check(verbose=False); print(i.kind, i.n_devices)"
        )
        env = {k: v for k, v in os.environ.items() if k not in ("JAX_NUM_CPU_DEVICES", "XLA_FLAGS")}
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
        self.assertEqual(out.stdout.split()[-2:], ["cpu", "3"])

    def test_check_reports(self):
        import anri.backend

        info = anri.backend.check(verbose=False)
        self.assertIn(info.kind, ("cpu", "gpu"))
        self.assertEqual(info.n_devices, jax.device_count())
        self.assertGreater(info.bytes_per_device, 0)


if __name__ == "__main__":
    unittest.main()
