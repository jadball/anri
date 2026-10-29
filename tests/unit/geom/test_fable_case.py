"""The test case of the FABLE geometry document (version 1.0.8, pp. 10-11), through Anri's forward model.

docs/source/theory/frames.rst quotes this check.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.fwd
import anri.io
from anri.diffract.scatter import k_out_to_tth_eta

jax.config.update("jax_enable_x64", True)

# FABLE's input. Its detector is given in "tilt specification II"; the document converts the origin to
# specification I, ImageD11's (L', y_det(0)', z_det(0)'), which is what these parameters are. Lengths in mm.
ENERGY = 69.533  # keV
A = 4.05  # space group 225
U = np.array([[0.872986, 0.436832, -0.216965], [-0.334822, 0.860185, 0.384678], [0.354669, -0.263174, 0.897190]])
POSITION = (-0.0602, 0.215, 0.0)
PARS = {"distance": 9.284758, "y_center": 724.953252, "z_center": 531.210607, "y_size": 0.0043, "z_size": 0.0043,
        "tilt_x": np.radians(1.0988), "tilt_y": np.radians(2.085), "tilt_z": np.radians(3.473),
        "o11": 1, "o12": 0, "o21": 0, "o22": 1}  # fmt: skip


class TestFableCase(unittest.TestCase):
    def test_omega_theta_eta_and_pixel(self):
        wavelength = 12.398419843320026 / ENERGY
        ubi = jnp.asarray(np.linalg.inv(U / A))  # B = I / a for a cubic cell
        hkl = jnp.array([-2.0, -2.0, 2.0])
        k_in = jnp.array([1.0, 0.0, 0.0])
        det = anri.io.imaged11.detector_from_pars(PARS)
        centroid, valid = anri.fwd.peak.get_centroid(
            ubi, jnp.array(POSITION), hkl, 1.0, wavelength, k_in, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            det["s_step_lab"], det["f_step_lab"], det["det_origin_lab"], False,
        )  # fmt: skip
        _, k_out, _, _ = anri.fwd.peak.hkl_to_k_omega(ubi, hkl, 1.0, wavelength, k_in, 0.0, 0.0, 0.0, 0.0)
        tth, eta = k_out_to_tth_eta(k_out, k_in)
        self.assertTrue(bool(valid))
        # FABLE: omega 79.793676, theta 4.373314, eta 62.190963; its hc differs from ours in the 5th digit
        self.assertAlmostEqual(float(centroid[2]), 79.793676, delta=1e-3)
        self.assertAlmostEqual(float(tth) / 2, 4.373314, delta=1e-3)
        self.assertAlmostEqual(float(eta), 62.190963, delta=1e-3)
        # FABLE: (y_det, z_det) = (418, 698) as whole pixels; with the identity o-matrix slow is z_det, fast y_det
        self.assertLess(abs(float(centroid[0]) - 698), 0.5)
        self.assertLess(abs(float(centroid[1]) - 418), 0.5)


if __name__ == "__main__":
    unittest.main()
