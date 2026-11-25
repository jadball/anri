"""Anri agrees with published tables and worked examples: FABLE's geometry test case, NIST XCOM's attenuation and
Becker and Coppens' extinction.

Not proofs (tests/theory) nor software tests (tests/unit): each tolerance is the source's own accuracy (the digits it
gives, or the accuracy of a fit), not rounding.
"""

import os
import unittest
import warnings

import jax
import jax.numpy as jnp
import numpy as np

import anri.crystal
import anri.fwd
import anri.io
from anri.diffract import factors
from anri.diffract.scatter import k_out_to_tth_eta

jax.config.update("jax_enable_x64", True)

CIF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cif")


def read_cif(name):
    import Dans_Diffraction

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Dans_Diffraction.Crystal(os.path.join(CIF_DIR, name))


TABLES = {
    "primary": {1: (0.5185, 0.5343), 3: (0.2952, 0.3204), 10: (0.1536, 0.1727)},
    "gaussian": {1: (0.4660, 0.4837), 3: (0.2215, 0.2489), 10: (0.0802, 0.1011)},
    "lorentzian": {1: (0.5297, 0.5450), 3: (0.3237, 0.3454), 10: (0.1742, 0.1927)},
}


ENERGY = 69.533  # keV


HC = 12.398  # keV angstrom: FABLE's value, which its theta implies (CODATA: 12.39842)


A = 4.05  # space group 225


U = np.array([[0.872986, 0.436832, -0.216965], [-0.334822, 0.860185, 0.384678], [0.354669, -0.263174, 0.897190]])


POSITION = (-0.0602, 0.215, 0.0)


PARS = {"distance": 9.284758, "y_center": 724.953252, "z_center": 531.210607, "y_size": 0.0043, "z_size": 0.0043,
        "tilt_x": np.radians(1.0988), "tilt_y": np.radians(2.085), "tilt_z": np.radians(3.473),
        "o11": 1, "o12": 0, "o21": 0, "o22": 1}  # fmt: skip


class TestAttenuation(unittest.TestCase):
    def test_against_nist(self):
        """mu / rho of iron and silicon (total, with coherent scattering) as NIST XCOM tabulates it."""
        for name, energy, mu_rho in (("Fe.cif", 50.0, 1.958), ("Fe.cif", 80.0, 0.5952), ("Si.cif", 50.0, 0.4385)):
            st = read_cif(name)
            mu = anri.crystal.structure.attenuation(st, energy)  # 1 / micrometre
            self.assertAlmostEqual(mu * 1e4 / st.Properties.density(), mu_rho, delta=0.002 * mu_rho)


class TestExtinction(unittest.TestCase):
    def test_tables(self):
        """The analytic fits reproduce the numerical tables they were fitted to, within 5% at these angles."""
        for kind, rows in TABLES.items():
            for x, ref in rows.items():
                for s, r in zip((0.05, 0.5), ref):
                    tth = 2 * np.arcsin(s)
                    if kind == "primary":
                        y = float(factors.extinction(x, 0.0, tth, True))
                    else:
                        y = float(factors.extinction(0.0, x, tth, kind == "gaussian"))
                    self.assertAlmostEqual(y / r, 1.0, delta=0.05, msg=f"{kind} x {x} sin(theta) {s}")


class TestFableCase(unittest.TestCase):
    def test_omega_theta_eta_and_pixel(self):
        wavelength = HC / ENERGY
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
        # FABLE: omega 79.793676, theta 4.373314, eta 62.190963. Left: ~1e-5 degrees, as FABLE's U is given to 6 digits
        self.assertAlmostEqual(float(centroid[2]), 79.793676, delta=1e-4)
        self.assertAlmostEqual(float(tth) / 2, 4.373314, delta=1e-4)
        self.assertAlmostEqual(float(eta), 62.190963, delta=1e-4)
        # FABLE: (y_det, z_det) = (418, 698) as whole pixels; with the identity o-matrix slow is z_det, fast y_det
        self.assertLess(abs(float(centroid[0]) - 698), 0.5)
        self.assertLess(abs(float(centroid[1]) - 418), 0.5)


if __name__ == "__main__":
    unittest.main()
