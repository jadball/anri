import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.crystal
import anri.fwd
import anri.geom
import anri.io
from anri.fwd._impl.base import make_propagator

jax.config.update("jax_enable_x64", True)


def _box_args():
    """Arguments of get_centroid_box for one valid peak: a rotated Fe grain away from the origin."""
    pars = {
        "y_center": 1024.0, "y_size": 75.0, "tilt_y": 0.0, "z_center": 1024.0, "z_size": 75.0, "tilt_z": 0.0,
        "tilt_x": 0.0, "distance": 150e3, "o11": 1, "o12": 0, "o21": 0, "o22": 1,
    }  # fmt: skip
    det = anri.io.detector_from_pars(pars)
    B = anri.crystal.lpars_to_B(jnp.array([2.8694, 2.8694, 2.8694, 90.0, 90.0, 90.0]))
    ubi = jnp.linalg.inv(anri.geom.rot_z(25.0) @ anri.geom.rot_x(35.0) @ B)
    detector = (det["s_step_lab"], det["f_step_lab"], det["det_origin_lab"])
    # (ubi, origin, hkl, etasign, wavelength, k_in_lab, ky, kz, wedge, chi, s_step, f_step, det_origin)
    return (ubi, jnp.array([20.0, -10.0, 5.0]), jnp.array([1.0, 1.0, 0.0]), 1.0, 0.3, jnp.array([1.0, 0.0, 0.0]),
            0.0, 0.0, 1.0, -2.0, *detector)  # fmt: skip


def _jacobian(args):
    """[3, 6] Jacobian of the centroid w.r.t. origin (3), wavelength, ky and kz, by jax.jacfwd."""

    def f(origin, wavelength, ky, kz):
        a = list(args)
        a[1], a[4], a[6], a[7] = origin, wavelength, ky, kz
        return anri.fwd.get_centroid_box(*a)[0]

    j = jax.jacfwd(f, argnums=(0, 1, 2, 3))(args[1], args[4], args[6], args[7])
    return j, jnp.concatenate([j[0]] + [x[:, None] for x in j[1:]], axis=1)


class TestPropagate(unittest.TestCase):
    def setUp(self):
        self.args = _box_args()
        self.assertTrue(anri.fwd.get_centroid_box(*self.args)[1])
        self.j_parts, self.J = _jacobian(self.args)
        self.cov_in = anri.fwd.get_cov_in(jnp.array([1.0, 2.0, 0.5]), 3e-4, 1e-4, 2e-4)

    def test_diagonal(self):
        expected = self.J @ self.cov_in @ self.J.T
        np.testing.assert_allclose(anri.fwd.propagate_cov_box(*self.args, self.cov_in), expected, rtol=1e-10)
        np.testing.assert_allclose(anri.fwd.propagate_cov(self.j_parts, self.cov_in), expected, rtol=1e-10)

    def test_diag_out(self):
        # diag_out keeps the four variances of the scanning centroid (sc, fc, omega, dty)
        scan_args = (*self.args[:10], 0.0, *self.args[10:])  # y0 = 0
        full = anri.fwd.propagate_cov_scan(*scan_args, self.cov_in)
        diag = make_propagator(anri.fwd.get_centroid_scan, argnums=(1, 4, 6, 7), has_aux=True, diag_out=True)
        np.testing.assert_allclose(diag(*scan_args, self.cov_in), jnp.diagonal(full), rtol=1e-10)
        elems = make_propagator(
            anri.fwd.get_centroid_box, argnums=(1, 4, 6, 7), has_aux=True, out_elems=((0, 0), (1, 1), (0, 1))
        )
        box = self.J @ self.cov_in @ self.J.T
        np.testing.assert_allclose(elems(*self.args, self.cov_in), [box[0, 0], box[1, 1], box[0, 1]], rtol=1e-10)

    def test_full_covariance(self):
        # a correlated input covariance needs diagonal=False
        cov_in = self.cov_in.at[0, 3].set(1e-4).at[3, 0].set(1e-4)
        expected = self.J @ cov_in @ self.J.T
        full = make_propagator(anri.fwd.get_centroid_box, argnums=(1, 4, 6, 7), has_aux=True, diagonal=False)
        np.testing.assert_allclose(full(*self.args, cov_in), expected, rtol=1e-10)

        def centroid_only(*a):
            return anri.fwd.get_centroid_box(*a)[0]

        no_aux = make_propagator(centroid_only, argnums=(1, 4, 6, 7), diagonal=False)
        np.testing.assert_allclose(no_aux(*self.args, cov_in), expected, rtol=1e-10)

    def test_errors(self):
        with self.assertRaises(ValueError):
            make_propagator(anri.fwd.get_centroid_box, argnums=(1,), diagonal=False, diag_out=True)
        nothing = make_propagator(anri.fwd.get_centroid_box, argnums=(1, 4, 6, 7), has_aux=True, active_dims=())
        with self.assertRaises(ValueError):
            nothing(*self.args, self.cov_in)


if __name__ == "__main__":
    unittest.main()
