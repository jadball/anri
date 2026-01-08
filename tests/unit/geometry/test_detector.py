import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

import anri

jax.config.update("jax_enable_x64", True)


class TestRaytraceToDet(unittest.TestCase):
    def setUp(self):
        # perfom the ImageD11 raytrace

        # rng
        import time

        from jax import random

        key = random.key(time.time_ns())

        npks = 100

        tth = random.uniform(key, shape=(npks,), minval=0.0, maxval=30.0)
        eta = random.uniform(key, shape=(npks,), minval=-180.0, maxval=180.0)
        omega = random.uniform(key, shape=(npks,), minval=-180.0, maxval=180.0)
        self.wavelength = 0.1

        vec_lab = vmap(anri.diffraction.tth_eta_to_k_out, in_axes=[0, 0, None])(tth, eta, self.wavelength)

        # set up some ImageD11 parameters
        from ImageD11.parameters import AnalysisSchema

        pars = AnalysisSchema.from_default().geometry_pars_obj

        # fiddle with the detector positions
        pars.set("tilt_x", 0.00123)
        pars.set("tilt_y", -0.0345)
        pars.set("tilt_z", 0.02)
        # fiddle with diffractometer tilts
        pars.set("chi", 1)
        pars.set("wedge", -3)
        pars_dict = pars.get_parameters()
        del pars_dict["t_x"]
        del pars_dict["t_y"]
        del pars_dict["t_z"]

        from ImageD11.transform import compute_xyz_from_tth_eta

        fc_id11, sc_id11 = compute_xyz_from_tth_eta(tth, eta, omega, **pars.get_parameters())

        self.pars = pars
        # self.origin_sample = origin_sample
        self.omega = omega
        self.vec_lab = vec_lab
        self.sc_id11 = sc_id11
        self.fc_id11 = fc_id11

    def test_raytrace_to_det(self):
        # set up detector transforms

        pars_for_det = self.pars.get_parameters().copy()
        del pars_for_det["chi"]
        del pars_for_det["wedge"]
        del pars_for_det["omegasign"]
        del pars_for_det["fit_tolerance"]
        del pars_for_det["min_bin_prob"]
        del pars_for_det["weight_hist_intensities"]
        del pars_for_det["no_bins"]
        del pars_for_det["wavelength"]
        det_trans, beam_cen_shift, x_distance_shift = anri.geometry.detector_transforms(**pars_for_det)
        sc_lab, fc_lab, norm_lab = anri.geometry.detector_basis_vectors_lab(det_trans, beam_cen_shift, x_distance_shift)

        # vectorize our function
        raytrace_to_det_vec = vmap(anri.geometry.raytrace_to_det, in_axes=(0, None, None, None, None))
        sc_anri, fc_anri = raytrace_to_det_vec(
            self.vec_lab,
            jnp.array([0.0, 0.0, 0.0]),  # origin_lab,
            sc_lab,
            fc_lab,
            norm_lab,
        )

        np.testing.assert_allclose(sc_anri, self.sc_id11)
        np.testing.assert_allclose(fc_anri, self.fc_id11)
