import unittest

import jax
import jax.numpy as jnp
import numpy as np

from anri.render._impl.kernel import flat_index, splat, splat_peaks, truncate

jax.config.update("jax_enable_x64", True)

try:
    from scipy.stats import multivariate_normal, norm
except ImportError:  # pragma: no cover
    multivariate_normal = norm = None


def exact_block(mu, cov, start, win):
    """Bivariate normal mass in each pixel of a window, from the CDF."""
    er = start[0] - 0.5 + np.arange(win[0] + 1)
    ec = start[1] - 0.5 + np.arange(win[1] + 1)
    r, c = np.meshgrid(er, ec, indexing="ij")
    f = multivariate_normal(mean=mu, cov=cov).cdf(np.stack([r, c], -1).reshape(-1, 2)).reshape(r.shape)
    return f[1:, 1:] - f[:-1, 1:] - f[1:, :-1] + f[:-1, :-1]


def render(mu, cov, lo, hi, shape, image=(0, 1), win=(15, 15), n_sub=2, periods=None, amp=1.0, dtype=np.float32):
    """One peak, as (dense canvas, block, start)."""
    periods = periods or (0.0,) * len(shape)
    f = lambda x: jnp.asarray(np.asarray(x, dtype))
    blocks, starts = splat_peaks(
        f(mu)[None],
        f(cov)[None],
        f([amp]),
        f(lo).reshape(1, -1),
        f(hi).reshape(1, -1),
        shape,
        image,
        win,
        n_sub,
        periods,
    )
    return np.asarray(blocks[0], np.float64), np.asarray(starts[0])


def place(block, start, size):
    out = np.zeros(size)
    out[start[0] : start[0] + block.shape[0], start[1] : start[1] + block.shape[1]] += block
    return out


@unittest.skipIf(multivariate_normal is None, "needs scipy")
class TestImageAxes(unittest.TestCase):
    CASES = ((0.4, 0.9, 0.6), (0.9, 0.4, 0.6), (0.3, 1.1, 0.88), (1.5, 1.5, 0.95), (0.29, 0.29, 0.0), (2.5, 0.5, -0.8))

    def _check(self, n_sub, tol, rho_zero_only=False):
        for sr, sc, rho in self.CASES:
            if rho_zero_only and rho != 0.0:
                continue
            mu = np.array([100.37, 200.62])
            cov = np.array([[sr * sr, rho * sr * sc], [rho * sr * sc, sc * sc]])
            block, start = render(mu, cov, [], [], (512, 512), n_sub=n_sub)
            ref = exact_block(mu, cov, start, block.shape)
            err = np.abs(block - ref).max() / ref.max()
            self.assertLess(err, tol, f"sigma=({sr}, {sc}) rho={rho}")

    def test_correlated(self):
        self._check(n_sub=2, tol=1e-3)

    def test_separable_is_exact_without_correlation(self):
        self._check(n_sub=0, tol=1e-4, rho_zero_only=True)

    def test_float64(self):
        mu = np.array([100.37, 200.62])
        cov = np.array([[0.16, 0.2], [0.2, 0.81]])
        block, start = render(mu, cov, [], [], (512, 512), n_sub=4, dtype=np.float64)
        ref = exact_block(mu, cov, start, block.shape)
        self.assertLess(np.abs(block - ref).max() / ref.max(), 1e-4)

    def test_non_square_window(self):
        # a streak along columns, with a window that is long along columns
        mu = np.array([100.2, 200.7])
        cov = np.array([[0.1, 0.3], [0.3, 2.0]])
        block, start = render(mu, cov, [], [], (512, 512), win=(7, 17))
        self.assertEqual(block.shape, (7, 17))
        ref = exact_block(mu, cov, start, block.shape)
        self.assertLess(np.abs(block - ref).max() / ref.max(), 1e-3)


@unittest.skipIf(multivariate_normal is None, "needs scipy")
class TestCollapsedAxes(unittest.TestCase):
    def test_whole_axis_collapse_is_marginal(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(4, 4))
        cov = a @ a.T * 0.3 + np.diag([0.1, 0.1, 0.5, 0.4])
        mu = np.array([100.3, 200.7, 50.2, 30.6])
        block, start = render(mu, cov, [-1e6, -1e6], [1e6, 1e6], (512, 512, 100, 100), n_sub=4)
        ref = exact_block(mu[:2], cov[:2, :2], start, block.shape)
        self.assertLess(np.abs(block - ref).max() / ref.max(), 1e-4)

    def test_frames_sum_to_amplitude(self):
        # box beam: (sc, fc, omega). Rendering every omega frame and adding them up gives the peak.
        cov = np.array([[0.5, 0.1, 0.4], [0.1, 0.3, -0.2], [0.4, -0.2, 0.9]])
        mu = np.array([100.3, 200.7, 50.2])
        total = np.zeros((512, 512))
        for f in range(44, 57):
            block, start = render(mu, cov, [f - 0.5], [f + 0.5], (512, 512, 100), amp=3.0)
            total += place(block, start, (512, 512))
        self.assertAlmostEqual(total.sum(), 3.0, delta=3e-4)
        ref = 3.0 * exact_block(mu[:2], cov[:2, :2], np.array([90, 190]), (21, 21))
        self.assertLess(np.abs(total[90:111, 190:211] - ref).max() / ref.max(), 1e-2)

    def test_frame_cutting_a_wavelength_streak(self):
        # detector position and omega both driven by wavelength: a frame cuts a streak.
        # reference integrates omega by quadrature over exact conditional 2D Gaussians.
        eta = np.radians(30)
        jac = np.array([1.1 * np.cos(eta), 1.1 * np.sin(eta), 0.5])
        cov = np.outer(jac, jac) + np.diag([1 / 12, 1 / 12, 0.01])
        mu = np.array([200.2, 200.6, 50.0 - 0.7 * np.sqrt(cov[2, 2])])
        block, start = render(mu, cov, [49.5], [50.5], (512, 512, 100), win=(17, 17))
        nodes = 49.5 + (np.arange(200) + 0.5) / 200
        gain = cov[:2, 2] / cov[2, 2]
        cond = cov[:2, :2] - np.outer(cov[:2, 2], cov[:2, 2]) / cov[2, 2]
        ref = np.zeros_like(block)
        for om in nodes:
            w = norm.pdf(om, mu[2], np.sqrt(cov[2, 2])) / 200
            ref += w * exact_block(mu[:2] + gain * (om - mu[2]), cond, start, block.shape)
        self.assertLess(np.abs(block - ref).max() / ref.max(), 1e-2)
        self.assertAlmostEqual(block.sum(), ref.sum(), delta=1e-4 * ref.sum())

    def test_correlated_motor_box_weight(self):
        # omega and dty tied together (voxel far from the rotation axis): the frame weight is a
        # correlated 2D box probability. Checked against the bivariate CDF.
        so, sy, rho = 0.31, 0.94, 0.97
        cov = np.diag([0.3, 0.3, 0.0, 0.0])
        cov[2:, 2:] = [[so * so, rho * so * sy], [rho * so * sy, sy * sy]]
        mu = np.array([100.3, 200.7, 50.0 - 0.33, 30.0 + 0.9])
        d = multivariate_normal(mean=mu[2:], cov=cov[2:, 2:])
        box = d.cdf([50.5, 30.5]) - d.cdf([49.5, 30.5]) - d.cdf([50.5, 29.5]) + d.cdf([49.5, 29.5])
        for n_sub, tol in ((2, 1e-2), (4, 1e-3)):
            block, _ = render(mu, cov, [49.5, 29.5], [50.5, 30.5], (512, 512, 100, 100), n_sub=n_sub)
            self.assertAlmostEqual(block.sum() / box, 1.0, delta=tol)

    def test_periodic_collapsed_axis(self):
        cov = jnp.diag(jnp.array([0.5, 0.5, 0.3]))
        mu = jnp.array([10.0, 10.0, 7199.8])
        w, _, _ = truncate(mu, cov, jnp.array([-0.5]), jnp.array([0.5]), (2,), (0.0, 0.0, 7200.0))
        sd = np.sqrt(0.3)
        self.assertAlmostEqual(float(w), norm.cdf(0.7, scale=sd) - norm.cdf(-0.3, scale=sd), places=5)

    def test_periodic_image_axis(self):
        # sinogram (omega, dty) of a peak sitting on the omega wrap: collapse the whole detector
        cov = np.diag([0.3, 0.3, 4.0, 1.0])
        cov[2, 3] = cov[3, 2] = 1.0
        mu = np.array([100.0, 100.0, 0.4, 20.3])
        shape = (512, 512, 360, 40)
        periods = (0.0, 0.0, 360.0, 0.0)
        img = np.asarray(
            splat(
                jnp.asarray(mu[None]),
                jnp.asarray(cov[None]),
                jnp.ones(1),
                jnp.array([-0.5, -0.5]),
                jnp.array([511.5, 511.5]),
                shape,
                image=(2, 3),
                win=(21, 21),
                n_sub=2,
                periods=periods,
            )
        )
        self.assertEqual(img.shape, (360, 40))
        self.assertAlmostEqual(img.sum(), 1.0, delta=1e-4)
        self.assertGreater(img[355:].sum(), 0.2)  # wrapped around to the end of omega
        self.assertGreater(img[:5].sum(), 0.4)


class TestGradients(unittest.TestCase):
    def test_matches_finite_differences(self):
        shape = (64, 64, 100, 50)
        periods = (0.0, 0.0, 100.0, 0.0)
        lo, hi = jnp.array([[49.5, 19.5]]), jnp.array([[50.5, 20.5]])
        a = np.array([[0.6, 0.1, 0.3, 0.0], [0.2, 0.5, -0.2, 0.1], [0.3, 0.0, 0.5, 0.2], [0.0, 0.1, 0.4, 0.4]])
        cov0 = jnp.asarray(a @ a.T + 0.05 * np.eye(4))[None]
        mu0 = jnp.array([[30.23, 31.61, 49.87, 20.21]])
        amp0 = jnp.array([2.0])
        target = jnp.asarray(np.random.default_rng(1).random((9, 9)))

        def loss(mu, cov, amp):
            blocks, _ = splat_peaks(mu, cov, amp, lo, hi, shape, (0, 1), (9, 9), 2, periods)
            return jnp.sum((blocks[0] - 0.1 * target) ** 2)

        grads = jax.grad(loss, argnums=(0, 1, 2))(mu0, cov0, amp0)
        for g in grads:
            self.assertTrue(bool(jnp.isfinite(g).all()))
        eps = 1e-6
        for arg, g in ((0, grads[0]), (2, grads[2])):
            x0 = (mu0, cov0, amp0)[arg]
            for i in range(x0.size):
                dx = jnp.zeros(x0.size).at[i].set(eps).reshape(x0.shape)
                args_p = [mu0, cov0, amp0]
                args_m = [mu0, cov0, amp0]
                args_p[arg] = x0 + dx
                args_m[arg] = x0 - dx
                fd = (loss(*args_p) - loss(*args_m)) / (2 * eps)
                self.assertAlmostEqual(float(g.reshape(-1)[i]), float(fd), delta=1e-5 + 1e-4 * abs(float(fd)))
        for i, j in ((0, 0), (0, 2), (2, 3), (3, 3)):
            dx = jnp.zeros((4, 4)).at[i, j].add(eps).at[j, i].add(eps if i != j else 0.0)[None]
            fd = (loss(mu0, cov0 + dx, amp0) - loss(mu0, cov0 - dx, amp0)) / (2 * eps)
            an = float(grads[1][0, i, j] + (grads[1][0, j, i] if i != j else 0.0))
            self.assertAlmostEqual(an, float(fd), delta=1e-5 + 1e-4 * abs(float(fd)))

    def test_finite_far_from_frame_and_degenerate(self):
        shape = (64, 64, 100, 50)
        cov = jnp.diag(jnp.array([0.3, 0.3, 0.2, 0.0]))[None]  # zero dty variance

        def loss(mu):
            blocks, _ = splat_peaks(
                mu,
                cov,
                jnp.ones(1),
                jnp.array([[49.5, 19.5]]),
                jnp.array([[50.5, 20.5]]),
                shape,
                (0, 1),
                (9, 9),
                2,
                (0.0,) * 4,
            )
            return jnp.sum(blocks**2)

        for om in (50.1, 90.0):
            g = jax.grad(loss)(jnp.array([[30.2, 31.6, om, 20.0]]))
            self.assertTrue(bool(jnp.isfinite(g).all()))


class TestSplat(unittest.TestCase):
    def test_dense_equals_placed_blocks(self):
        rng = np.random.default_rng(3)
        n = 50
        mu = np.column_stack([rng.uniform(0, 64, n), rng.uniform(0, 48, n), rng.uniform(10, 12, n)])
        cov = np.tile(np.diag([0.5, 0.4, 0.3]), (n, 1, 1))
        cov[:, 0, 2] = cov[:, 2, 0] = 0.2
        args = (jnp.asarray(mu), jnp.asarray(cov), jnp.ones(n), jnp.array([10.5]), jnp.array([11.5]))
        img = np.asarray(splat(*args, (64, 48, 20), win=(7, 7)))
        blocks, starts = splat_peaks(
            *args[:3], jnp.full((n, 1), 10.5), jnp.full((n, 1), 11.5), (64, 48, 20), (0, 1), (7, 7), 2, (0.0, 0.0, 0.0)
        )
        ref = np.zeros((64, 48))
        for b, s in zip(np.asarray(blocks), np.asarray(starts)):
            ref[s[0] : s[0] + 7, s[1] : s[1] + 7] += b
        np.testing.assert_allclose(img, ref, rtol=1e-10, atol=1e-12)
        self.assertTrue((np.asarray(starts) >= 0).all())
        self.assertTrue((np.asarray(starts) <= [64 - 7, 48 - 7]).all())

    def test_flat_index_wraps_periodic_axes(self):
        idx = np.asarray(flat_index(jnp.array([[8, -2]]), (3, 3), (10, 5), (True, True)))
        rows, cols = np.divmod(idx[0], 5)
        np.testing.assert_array_equal(rows, [8, 8, 8, 9, 9, 9, 0, 0, 0])
        np.testing.assert_array_equal(cols, [3, 4, 0] * 3)


if __name__ == "__main__":
    unittest.main()
