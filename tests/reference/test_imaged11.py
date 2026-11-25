"""Anri agrees with ImageD11 on the basics: lattices, geometry, intensity factors and segmentation.

Not proofs (tests/theory) nor software tests (tests/unit): these compare anri with older, trusted software, whose
conventions anri follows except where frames.rst says it differs (the wedge's sign). Each function is also checked
against the maths in tests/theory.
"""

import os
import tempfile
import unittest
import warnings

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

import anri.crystal
import anri.diffract
import anri.geom
import anri.io
import anri.maths.rotation
import anri.maths.utils
from anri import diffract
from anri.diffract.factors import lorentz, polarisation
from anri.geom.detector import det_to_lab, detector_transforms
from anri.geom.gonio import lab_to_sample
from anri.io.imaged11 import detector_from_pars, geom_from_pars

jax.config.update("jax_enable_x64", True)

CIF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cif")


def _close(actual: object, desired: object, n: int = 16) -> None:
    """Anri and the other software compute the same float64 formula, in their own order: they agree to n ulps of the
    reference's largest element (measured: at most 4)."""
    d = np.asarray(desired, float)
    np.testing.assert_allclose(actual, d, rtol=0, atol=n * np.finfo(float).eps * float(np.max(np.abs(d))))


def cellvolume(lp):
    """Unit cell volume from lattice parameters (as ImageD11.forward_model.forward_projector)."""
    ca, cb, cg = np.cos(np.radians(lp[3:]))
    return lp[0] * lp[1] * lp[2] * np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)


def read_cif(name):
    import Dans_Diffraction

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Dans_Diffraction.Crystal(os.path.join(CIF_DIR, name))


class TestLParsToMT(unittest.TestCase):
    def test_id11(self):
        a = 2.5
        b = 2.8
        c = 11.3
        alpha = 85.2
        beta = 95.8
        gamma = 101.6
        lpars = jnp.array([a, b, c, alpha, beta, gamma])

        from ImageD11.unitcell import unitcell

        uc = unitcell(lpars, 1)
        expected = uc.g
        result = anri.crystal.lattice.lpars_to_mt(lpars)
        _close(result, expected)


class TestBuildBMat(unittest.TestCase):
    def test_id11(self):
        a = 2.5
        b = 2.8
        c = 11.3
        alpha = 85.2
        beta = 95.8
        gamma = 101.6
        lpars = jnp.array([a, b, c, alpha, beta, gamma])
        mt = anri.crystal.lattice.lpars_to_mt(lpars)
        rmt = anri.crystal.lattice.mt_to_rmt(mt)
        rlpars = anri.crystal.lattice.mt_to_lpars(rmt)

        from ImageD11.unitcell import unitcell

        uc = unitcell(lpars, 1)
        expected = uc.B
        result = anri.crystal.lattice.lpars_rlpars_to_B(lpars, rlpars)
        _close(result, expected)

    def test_busing_levy(self):
        """Generate many random lattice parameters using Dan's logic and check Busing-Levy"""
        ntests = 1_000

        from Dans_Diffraction.functions_lattice import random_lattice
        from ImageD11.unitcell import unitcell

        np.random.seed(0)  # random_lattice draws from NumPy's global generator
        lpars_list = [random_lattice(symmetry="triclinic") for _ in range(ntests)]
        lpars_batch = jnp.array(lpars_list)

        self.assertTupleEqual(lpars_batch.shape, (ntests, 6))
        self.assertTrue(~jnp.any(jnp.isnan(lpars_batch)))

        lpars_to_B_vec = jax.vmap(anri.crystal.lattice.lpars_to_B)
        results = lpars_to_B_vec(lpars_batch)

        for i in range(ntests):
            uc = unitcell(lpars_batch[i], "P")
            expected = uc.B

            self.assertFalse(jnp.any(jnp.isnan(results[i])), f"NaN found at index {i} with input {lpars_batch[i]}")

            # B comes from G^-1: both codes err by cond(G) ulps of it (as tests/theory/test_frames.py::test_B_code)
            cond = np.linalg.cond(np.asarray(uc.g))
            _close(results[i], expected, int(np.ceil(16 * cond)))


class TestEverything(unittest.TestCase):
    def test_id11(self):
        # Test against the full ImageD11 unit cell class
        a = 3.4
        b = 2.6
        c = 9.5
        alpha = 83.2
        beta = 95.8
        gamma = 105.4
        lpars = jnp.array([a, b, c, alpha, beta, gamma])
        from ImageD11.unitcell import unitcell

        uc = unitcell(lpars, 1)

        mt = anri.crystal.lattice.lpars_to_mt(lpars)
        _close(mt, uc.g)

        rmt = anri.crystal.lattice.mt_to_rmt(mt)
        _close(rmt, uc.gi)

        B = anri.crystal.lattice.lpars_to_B(lpars)
        _close(B, uc.B)


class TestMetricTensors(unittest.TestCase):
    def test_id11(self):
        from ImageD11.unitcell import unitcell as unitcell_id11

        import anri.geom

        lpars = [4.9, 5.1, 5.4, 88.0, 92.0, 119.0]
        uc = unitcell_id11(lpars, "P")
        B = anri.crystal.lattice.lpars_to_B(jnp.array(lpars))
        _close(anri.crystal.lattice.B_to_rmt(B), uc.gi)
        _close(anri.crystal.lattice.rmt_to_mt(jnp.asarray(uc.gi)), uc.g)
        # the metric tensor does not depend on the orientation
        U = anri.maths.rotation.rot_z(25.0) @ anri.maths.rotation.rot_x(35.0)
        _close(anri.crystal.lattice.UBI_to_mt(jnp.linalg.inv(U @ B)), uc.g)


class TestLattice(unittest.TestCase):
    def test_against_imaged11(self):
        """B, the metric tensors, reciprocal lattice and volume (the JAX definitions, in float64) match ImageD11."""
        from Dans_Diffraction.functions_lattice import random_lattice
        from ImageD11.unitcell import unitcell as unitcell_id11

        rng = np.random.default_rng(0)
        kinds = [
            "cubic",
            "tetragonal",
            "hexagonal",
            "rhobohedral",
            "monoclinic-a",
            "monoclinic-b",
            "monoclinic-c",
            "triclinic",
        ]
        for _ in range(50):
            np.random.seed(int(rng.integers(1 << 30)))
            lp = np.asarray(random_lattice(symmetry=kinds[int(rng.integers(len(kinds)))]), float)
            if not np.isfinite(cellvolume(lp)):
                continue  # angles no cell can have (older Dans_Diffraction draws them): both give NaN, which is no check
            try:
                uc = unitcell_id11(lp, "P")
            except np.linalg.LinAlgError:
                continue
            _close(anri.crystal.structure.B_matrix(lp), uc.B)
            with anri.maths.utils.float64():
                mt = np.asarray(anri.crystal.lattice.lpars_to_mt(lp))
                rmt = np.asarray(anri.crystal.lattice.mt_to_rmt(mt))
                _close(mt, uc.g)
                _close(rmt, uc.gi)
                _close(
                    anri.crystal.lattice.mt_to_lpars(rmt),
                    [uc.astar, uc.bstar, uc.cstar, uc.alphas, uc.betas, uc.gammas],
                )
                _close(anri.crystal.lattice.metric_to_volume(mt), cellvolume(lp))


class TestReflections(unittest.TestCase):
    def test_against_imaged11(self):
        """Rings (d*, multiplicity and members) match ImageD11's unitcell for a structure with atoms."""
        from ImageD11.unitcell import unitcell as unitcell_id11

        xtl = read_cif("Fe.cif")
        lp, sg = anri.crystal.structure.lattice_parameters(xtl), anri.crystal.structure.space_group(xtl)
        dsmax, wl = 1.2, 0.18
        r = anri.crystal.structure.reflections(lp, sg, wl, dsmax)
        ring, ring_ds = anri.crystal.structure.rings(r["ds"])
        uc = unitcell_id11(lp, sg)
        uc.makerings(dsmax, tol=1e-4)
        _close(ring_ds, uc.ringds)
        for i, ds in enumerate(uc.ringds):
            self.assertEqual(sorted(map(tuple, r["hkl"][ring == i].tolist())), sorted(uc.ringhkls[ds]))


class TestQLabToTthEta(unittest.TestCase):
    def setUp(self):
        from ImageD11.transform import compute_k_vectors

        self.wavelength = 0.1
        self.k_in = jnp.array([1.0 / self.wavelength, 0.0, 0.0])
        npks = 100
        self.tth = np.random.default_rng(0).random(npks) * 30.0  # 0 to 30 degrees
        self.eta = np.random.default_rng(1).random(npks) * 360.0 - 180.0  # -180 to 180 degrees
        # compute q_lab from tth, eta with ImageD11:
        self.q_lab_id11 = compute_k_vectors(self.tth, self.eta, self.wavelength).T
        # vectorize our functions for testing
        self.q_lab_to_tth_eta_vec = vmap(diffract.scatter.q_lab_to_tth_eta, in_axes=(0, None))
        self.tth_eta_to_k_out_vec = vmap(diffract.scatter.tth_eta_to_k_out, in_axes=(0, 0, None))
        self.k_to_q_lab_vec = vmap(diffract.scatter.k_to_q_lab, in_axes=(0, 0))

    def test_q_lab_to_tth_eta_id11(self):
        # when we convert our ImageD11 q_lab back to tth, eta we should recover the original values
        expected_tth = self.tth
        expected_eta = self.eta

        result_tth, result_eta = self.q_lab_to_tth_eta_vec(self.q_lab_id11, self.wavelength)
        _close(result_tth, expected_tth)
        _close(result_eta, expected_eta)

    def test_tth_eta_to_k_out_id11(self):
        # when we convert tth, eta to k_out and then to q_lab we should recover the original q_lab

        k_out_vec = self.tth_eta_to_k_out_vec(self.tth, self.eta, self.wavelength)
        q_lab_vec = self.k_to_q_lab_vec(jnp.repeat(self.k_in[None, :], k_out_vec.shape[0], axis=0), k_out_vec)

        # atol: components of q near zero (the angles are random) differ by rounding only
        _close(q_lab_vec, self.q_lab_id11)


class TestOmegaSolns(unittest.TestCase):
    def setUp(self):
        # generate some test UBI, gives us q_sample
        # TODO: replace with our own B matrix generator

        from ImageD11.gv_general import g_to_k, wedgechi
        from ImageD11.unitcell import unitcell
        from jax.scipy.spatial.transform import Rotation as jR

        self.g_to_k_id11 = g_to_k
        self.wedgechi_id11 = wedgechi

        from jax import random

        key = random.key(1)

        # numerical stuff
        a, b, c = 3.0, 4.0, 5.0
        alpha, beta, gamma = 90.0, 90.0, 90.0
        self.wavelength = 0.1
        dsmax = 3.0

        self.k_in_lab = jnp.array([1.0 / self.wavelength, 0.0, 0.0])

        # prepare UB matrix
        uc = unitcell([a, b, c, alpha, beta, gamma], 225)
        uc.makerings(dsmax)
        B = uc.B
        random_euler = random.uniform(key, shape=(3,), minval=0.0, maxval=2 * jnp.pi)
        R = jR.from_euler("xyz", random_euler)
        U = R.as_matrix()
        UB = U @ B

        hkls = jnp.concatenate([jnp.array(hkl) for hkl in uc.ringhkls.values()])

        self.q_sample = UB @ hkls.T

        self.omega_solns_vec = vmap(diffract.scatter.omega_solns, in_axes=(1, None, None))

    def test_omega_solns_id11_nochi_nowedge(self):
        chi = 0.0
        wedge = 0.0
        dty = 0.0
        y0 = 0.0
        # the rotation axis is always defined as +Z in the sample frame
        rot_axis_sample = jnp.array([0.0, 0.0, 1.0])

        # convert k_in to sample frame
        # just identity here
        k_in_sample = lab_to_sample(self.k_in_lab, omega=0.0, wedge=wedge, chi=chi, dty=dty, y0=y0)

        # we have to supply negative the axis to match ImageD11 convention here
        # this is because ImageD11 forces a negative k_in, whereas we can supply one
        omega1_id11, omega2_id11, valid_id11 = self.g_to_k_id11(
            self.q_sample, self.wavelength, rot_axis_sample * -1, pre=None, post=None
        )

        omega1_anri, valid1_anri = self.omega_solns_vec(self.q_sample, 1, k_in_sample)
        omega2_anri, valid2_anri = self.omega_solns_vec(self.q_sample, -1, k_in_sample)

        # we have different non-valid handling than ImageD11 to ensure differentiability
        # so we only check that valid solutions match, and that the validity arrays are the same

        np.testing.assert_array_equal(valid1_anri, valid_id11)
        np.testing.assert_array_equal(valid2_anri, valid_id11)
        _close(omega1_anri[valid1_anri], omega1_id11[valid_id11])
        _close(omega2_anri[valid2_anri], omega2_id11[valid_id11])

    def test_omega_solns_id11_chi_wedge(self):
        chi = 10.0
        wedge = -5.0
        dty = 0.0
        y0 = 0.0
        # imaged11 wedge definition is negative
        post = self.wedgechi_id11(-wedge, chi)

        # convert k_in to sample frame
        # this is handled by the post matrix in ImageD11
        k_in_sample = lab_to_sample(self.k_in_lab, omega=0.0, wedge=wedge, chi=chi, dty=dty, y0=y0)

        # the rotation axis is always defined as +Z in the sample frame
        rot_axis_sample = jnp.array([0.0, 0.0, 1.0])

        omega1_id11, omega2_id11, valid_id11 = self.g_to_k_id11(
            self.q_sample, self.wavelength, rot_axis_sample * -1, pre=None, post=post
        )

        omega1_anri, valid1_anri = self.omega_solns_vec(self.q_sample, 1, k_in_sample)
        omega2_anri, valid2_anri = self.omega_solns_vec(self.q_sample, -1, k_in_sample)

        # we have different non-valid handling than ImageD11 to ensure differentiability
        # so we only check that valid solutions match, and that the validity arrays are the same

        np.testing.assert_array_equal(valid1_anri, valid_id11)
        np.testing.assert_array_equal(valid2_anri, valid_id11)
        _close(omega1_anri[valid1_anri], omega1_id11[valid_id11])
        _close(omega2_anri[valid2_anri], omega2_id11[valid_id11])


class TestDetToQSample(unittest.TestCase):
    def test_id11(self):

        import ImageD11.transform
        from ImageD11.columnfile import columnfile
        from ImageD11.parameters import AnalysisSchema
        from jax import random

        # get some basic ImageD11 geometry
        pars = AnalysisSchema.from_default().geometry_pars_obj
        # make it slightly spicier
        pars.set("tilt_x", 0.4)
        pars.set("tilt_y", -0.345)
        pars.set("tilt_z", 0.2)
        pars.set("chi", 10)
        pars.set("wedge", -17)

        # RNG
        key = random.key(2)
        key, *subkeys = random.split(key, 4)

        # detector peaks
        npks = 100_000
        fc = random.uniform(subkeys[0], shape=(npks,), minval=0.0, maxval=2048.0)
        sc = random.uniform(subkeys[1], shape=(npks,), minval=0.0, maxval=2048.0)
        om = random.uniform(subkeys[2], shape=(npks,), minval=-180.0, maxval=180.0)

        # bung em in a columnfile
        cf = columnfile(new=True)
        cf.nrows = npks
        cf.addcolumn(fc, "fc")
        cf.addcolumn(sc, "sc")
        cf.addcolumn(om, "omega")

        # compute geometry with ImageD11
        cf.parameters = pars
        cf.updateGeometry()

        # incident wavevector, normalised
        k_in = diffract.scatter.scale_norm_k(np.array([1.0, 0, 0]), pars.get("wavelength"))

        det_trans, beam_cen_shift, x_distance_shift = detector_transforms(
            pars.get("y_center"),
            pars.get("y_size"),
            pars.get("tilt_y"),
            pars.get("z_center"),
            pars.get("z_size"),
            pars.get("tilt_z"),
            pars.get("tilt_x"),
            pars.get("distance"),
            pars.get("o11"),
            pars.get("o12"),
            pars.get("o21"),
            pars.get("o22"),
        )

        # detector peak positions in lab frame
        det_to_lab_vec = vmap(
            det_to_lab,
            in_axes=(0, 0, None, None, None),
        )
        v_lab_me = det_to_lab_vec(cf.sc, cf.fc, det_trans, beam_cen_shift, x_distance_shift)
        v_lab_id11 = jnp.column_stack([cf.xl, cf.yl, cf.zl])
        _close(v_lab_me, v_lab_id11)

        # q_lab - just subtract diffraction origins then normalise and scale by 1/wavelength
        peak_lab_to_k_out_vec = vmap(
            diffract.scatter.peak_lab_to_k_out,
            in_axes=(0, None, None),
        )
        k_out_me = peak_lab_to_k_out_vec(v_lab_me, jnp.array([0.0, 0.0, 0.0]), pars.get("wavelength"))
        k_to_q_lab_vec = vmap(
            diffract.scatter.k_to_q_lab,
            in_axes=(None, 0),  # constant k_in
        )
        q_lab_me = k_to_q_lab_vec(k_in, k_out_me)
        q_lab_id11 = ImageD11.transform.compute_k_vectors(cf.tth, cf.eta, pars.get("wavelength")).T
        _close(q_lab_me, q_lab_id11)

        # q_sample - rotate q_lab to sample frame by applying omega, wedge, chi
        lab_to_sample_vec = vmap(
            lab_to_sample,
            in_axes=(0, 0, None, None, None, None),
        )
        q_sample_me = lab_to_sample_vec(
            q_lab_me,
            cf.omega,
            -pars.get("wedge"),  # ImageD11 wedge sign is negative
            pars.get("chi"),
            0.0,
            0.0,
        )
        q_sample_id11 = jnp.column_stack([cf.gx, cf.gy, cf.gz])
        _close(q_sample_me, q_sample_id11)


class TestDetectorRotationMatrix(unittest.TestCase):
    def test_id11(self):
        tilt_x = 2.5
        tilt_y = -0.5
        tilt_z = -1.4
        from ImageD11.transform import detector_rotation_matrix as detector_rotation_matrix_id11

        expected = detector_rotation_matrix_id11(tilt_x, tilt_y, tilt_z)
        result = anri.geom.detector.detector_rotation_matrix(tilt_x, tilt_y, tilt_z)
        _close(result, expected)


class TestDetToLab(unittest.TestCase):
    def test_id11(self):

        from jax import random

        key = random.key(3)
        npks = 1000
        det_size = 2048.0
        sc = random.uniform(key, shape=(npks,), minval=0.0, maxval=det_size)
        fc = random.uniform(key, shape=(npks,), minval=0.0, maxval=det_size)

        y_center = 1000.0
        z_center = 900.0
        tilt_x = 0.2
        tilt_y = -0.36
        tilt_z = 0.17
        y_size = 50
        z_size = 65
        distance = 130e3
        o11 = 1.0
        o12 = 0.0
        o21 = 0.0
        o22 = 1.0

        det_trans, beam_cen_shift, x_distance_shift = anri.geom.detector.detector_transforms(
            y_center=y_center,
            z_center=z_center,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            tilt_z=tilt_z,
            y_size=y_size,
            z_size=z_size,
            distance=distance,
            o11=o11,
            o12=o12,
            o21=o21,
            o22=o22,
        )

        from ImageD11.transform import compute_xyz_lab

        peaks = jnp.vstack((fc, sc))
        assert peaks.shape == (2, npks), peaks.shape
        expected = compute_xyz_lab(
            peaks,
            y_center=y_center,
            z_center=z_center,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            tilt_z=tilt_z,
            y_size=y_size,
            z_size=z_size,
            distance=distance,
            o11=o11,
            o12=o12,
            o21=o21,
            o22=o22,
        ).T
        det_to_lab_vec = jax.vmap(anri.geom.detector.det_to_lab, in_axes=[0, 0, None, None, None])
        result = det_to_lab_vec(sc, fc, det_trans, beam_cen_shift, x_distance_shift)
        _close(result, expected)


class TestRaytraceToDet(unittest.TestCase):
    def setUp(self):
        # perfom the ImageD11 raytrace

        # rng

        from jax import random

        key = random.key(4)

        npks = 100

        tth = random.uniform(key, shape=(npks,), minval=0.0, maxval=30.0)
        eta = random.uniform(key, shape=(npks,), minval=-180.0, maxval=180.0)
        omega = random.uniform(key, shape=(npks,), minval=-180.0, maxval=180.0)
        self.wavelength = 0.1

        vec_lab = vmap(anri.diffract.scatter.tth_eta_to_k_out, in_axes=[0, 0, None])(tth, eta, self.wavelength)

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

    def test_raytrace_to_det_id11(self):
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
        det_trans, beam_cen_shift, x_distance_shift = anri.geom.detector.detector_transforms(**pars_for_det)
        s_step_lab, f_step_lab, det_origin_lab = anri.geom.detector.detector_basis_vectors_lab(
            det_trans, beam_cen_shift, x_distance_shift
        )

        # vectorize our function
        raytrace_to_det_vec = vmap(anri.geom.detector.raytrace_to_det, in_axes=(0, None, None, None, None))
        sc_anri, fc_anri = raytrace_to_det_vec(
            self.vec_lab,
            jnp.array([0.0, 0.0, 0.0]),  # origin_lab,
            s_step_lab,
            f_step_lab,
            det_origin_lab,
        )

        # **Measured:** ImageD11 goes from (2theta, eta) to the pixel through its own detector formulas, whose rounding
        # is its own (not derived here): 555 ulps of the largest pixel at most
        _close(sc_anri, self.sc_id11, 1000)
        _close(fc_anri, self.fc_id11, 1000)


class TestSampleToLab(unittest.TestCase):
    def test_id11(self):
        # test some more complicated vectors.

        from jax import random

        key = random.key(5)
        npks = 1000

        vec_sample = random.uniform(key, shape=(npks, 3), minval=-1000.0, maxval=1000.0)
        # omega = random.uniform(key, shape=(npks,), minval=-180.0, maxval=180.0)
        # wedge = random.uniform(key, shape=(npks,), minval=-180.0, maxval=180.0)
        # chi = random.uniform(key, shape=(npks,), minval=-180.0, maxval=180.0)
        key = random.key(6)
        omega_arr = random.uniform(key, shape=(npks,), minval=-180.0, maxval=180.0)
        # omega = random.uniform(key, shape=(1,), minval=-180.0, maxval=180.0)[0]
        key = random.key(7)
        wedge = random.uniform(key, shape=(1,), minval=-180.0, maxval=180.0)[0]
        key = random.key(8)
        chi = random.uniform(key, shape=(1,), minval=-180.0, maxval=180.0)[0]
        wedge_arr = jnp.repeat(wedge, npks)
        chi_arr = jnp.repeat(chi, npks)

        dty = jnp.repeat(0.0, npks)
        y0 = jnp.repeat(0.0, npks)

        from ImageD11.transform import compute_grain_origins

        sample_to_lab_vec = jax.vmap(anri.geom.gonio.sample_to_lab)
        result = sample_to_lab_vec(vec_sample, omega_arr, wedge_arr, chi_arr, dty, y0)
        desired = compute_grain_origins(
            omega=omega_arr, wedge=-wedge, chi=chi, t_x=vec_sample[:, 0], t_y=vec_sample[:, 1], t_z=vec_sample[:, 2]
        ).T
        _close(result, desired)


class TestFindDtyForBeamXY(unittest.TestCase):
    def test_imaged11(self):
        wedge = 0.0
        chi = 0.0
        omega = 12.4
        y0 = 15.6

        # fmt: off
        v_sample = jnp.array([0.2, 0.3, -0.4])
        k_in_lab = jnp.array([1.0, 0.0, 0.0])
        # fmt: on

        # fmt: off
        from ImageD11.sinograms.geometry import dty_values_grain_in_beam
        dty_desired = dty_values_grain_in_beam(v_sample[0], v_sample[1], y0, omega)
        # fmt: on
        dty_result = anri.geom.gonio.find_dty_for_beam_xy(v_sample, k_in_lab, omega, wedge, chi, y0)
        self.assertAlmostEqual(dty_desired, dty_result)


class TestStepGrid(unittest.TestCase):
    """Against ImageD11's sinogram geometry, which these functions follow."""

    def test_id11(self):
        from ImageD11.sinograms import geometry as g

        ybincens = np.arange(-10.5, 12.0, 0.75)
        si, sj = anri.geom.scan.step_grid_from_ybincens(ybincens, 0.75, 1, 0.3)
        # ImageD11 gives a list of (si, sj) for si in steps for sj in steps
        expected = np.array(g.step_grid_from_ybincens(ybincens, 0.75, 1, 0.3)).reshape(si.shape + (2,))
        np.testing.assert_array_equal(np.stack([si, sj], axis=-1), expected)
        si, sj = anri.geom.scan.step_grid_from_ybincens(ybincens, 0.75, 1, 0.3)
        recon_shape = (si.shape[0], si.shape[1] + 2)
        ri, rj = anri.geom.scan.step_to_recon(si, sj, recon_shape)
        np.testing.assert_array_equal(np.stack([ri, rj]), np.stack(g.step_to_recon(si, sj, recon_shape)))
        np.testing.assert_array_equal(np.stack(anri.geom.scan.recon_to_step(ri, rj, recon_shape)), np.stack([si, sj]))
        _close(np.stack(anri.geom.scan.step_to_sample(si, sj, 0.75)), np.stack(g.step_to_sample(si, sj, 0.75)))
        _close(
            np.stack(anri.geom.scan.recon_to_sample(ri, rj, recon_shape, 0.75)),
            np.stack(g.recon_to_sample(ri, rj, recon_shape, 0.75)),
        )


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
        _close(L * lf(tth, eta), 1.0)
        for f in [1.0, 0.9, 0.0]:
            P = jax.vmap(polarisation, in_axes=(None, 0, None))(k_in, k_out, f)
            _close(P, polarization(tth, eta, factor=f))


class TestSegment(unittest.TestCase):
    def test_as_imaged11(self):
        """segment records the pixels ImageD11's lima_segmenter records (select, then clean), frame by frame: Poisson
        frames with spots of every size, faint and bright, touching the detector's edges, for cut 0, 1, 2 and groups
        of at least 1, 2, 3, 4 pixels."""
        import ImageD11.sinograms.lima_segmenter as lima

        from anri.io.imaged11 import segment

        rng = np.random.default_rng(0)
        ns, nf, n_frames = 24, 31, 40
        ii, jj = np.mgrid[:ns, :nf]
        lam = np.zeros((n_frames, ns, nf))
        for k in range(n_frames):
            for _ in range(6):
                ci, cj, w, a = (
                    rng.uniform(-2, ns + 2),
                    rng.uniform(-2, nf + 2),
                    rng.uniform(0.3, 3),
                    rng.uniform(0.5, 30),
                )
                lam[k] += a * np.exp(-((ii - ci) ** 2 + (jj - cj) ** 2) / (2 * w**2))
        img = rng.poisson(lam)
        frame, pixel = np.nonzero(img.reshape(n_frames, -1))  # every cell with counts, sorted by (frame, pixel)
        counts = img.reshape(n_frames, -1)[frame, pixel]
        saved = lima.OPTIONS
        try:
            for cut in (0, 1, 2):
                for m in (1, 2, 3, 4):
                    opts = lima.SegmenterOptions(cut=cut, pixels_in_spot=m)
                    opts.setup()
                    opts.mask = np.ones((ns, nf), np.uint8)
                    lima.OPTIONS = opts
                    tosparse = lima.frmtosparse(opts.mask, np.uint32)
                    want = []
                    for k in range(n_frames):
                        spf = lima.clean(*tosparse(img[k].astype(np.uint32), cut))
                        if spf is not None:
                            want += [(k, int(r) * nf + int(c)) for r, c in zip(spf.row, spf.col)]
                    keep = segment(frame, pixel, counts, (ns, nf), cut, m)
                    got = list(zip(frame[keep].tolist(), pixel[keep].tolist()))
                    self.assertEqual(got, want, (cut, m))
                    self.assertGreater(len(want), 50)
        finally:
            lima.OPTIONS = saved


class TestSpatial(unittest.TestCase):
    def test_detector_h5_as_imaged11(self):
        """dx, dy from a pyFAI detector file, as ImageD11's get_e2dx_from_h5."""
        from ImageD11.blobcorrector import get_e2dx_from_h5

        from anri.io.imaged11 import read_spatial

        rng = np.random.default_rng(0)
        n_s, n_f, ps = 6, 5, 75e-6
        s, f = np.mgrid[0:n_s, 0:n_f]
        corners = np.zeros((n_s, n_f, 4, 3))
        for c, (a, b) in enumerate(((0, 0), (0, 1), (1, 1), (1, 0))):
            corners[:, :, c, 1] = (s + a) * ps + rng.normal(0, 0.1 * ps, (n_s, n_f))
            corners[:, :, c, 2] = (f + b) * ps + rng.normal(0, 0.1 * ps, (n_s, n_f))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "det.h5")
            with h5py.File(path, "w") as h:
                h["entry_0000/pyFAI/Eiger2_CdTe_4M/pixel_size"] = np.array([ps, ps])
                h["entry_0000/pyFAI/Eiger2_CdTe_4M/pixel_corners"] = corners
            dx, dy = read_spatial({"detectorh5": path, "detector": "eiger"})
            ex, ey = get_e2dx_from_h5(path, detector="eiger")
        # ImageD11 returns float32 maps: half a float32 ulp of each value, after the same float64 arithmetic
        e32 = float(np.finfo(np.float32).eps)
        np.testing.assert_allclose(dx, ex, rtol=0, atol=e32 * float(np.abs(ex).max()))
        np.testing.assert_allclose(dy, ey, rtol=0, atol=e32 * float(np.abs(ey).max()))


class TestGeomFromPars(unittest.TestCase):
    def test_detector_from_pars(self):
        """Pixel to lab matches ImageD11, and ray-tracing the lab point back from the origin gives the same pixel."""
        import jax.numpy as jnp
        from ImageD11 import transform

        import anri.geom

        pars = {
            "y_center": 1049.9, "y_size": 75.0, "tilt_y": -2e-3,
            "z_center": 1116.5, "z_size": 75.0, "tilt_z": 3e-3, "tilt_x": 1e-3,
            "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1,
        }  # fmt: skip
        det = detector_from_pars(pars)
        for sc, fc in ((1500.3, 700.8), (12.0, 2000.5)):
            xyz = anri.geom.detector.det_to_lab(
                sc, fc, det["det_trans"], det["beam_cen_shift"], det["x_distance_shift"]
            )
            _close(xyz, transform.compute_xyz_lab(np.array([[sc], [fc]]), **pars).ravel())
            back = anri.geom.detector.raytrace_to_det(
                xyz / jnp.linalg.norm(xyz), jnp.zeros(3), det["s_step_lab"], det["f_step_lab"], det["det_origin_lab"]
            )
            _close(back, (sc, fc))

    def test_wedge_chi_against_imaged11(self):
        """A peak computed by anri from geom_from_pars maps back to its hkl through ImageD11's own geometry."""
        import jax.numpy as jnp
        from ImageD11 import transform

        import anri.crystal
        import anri.geom
        from anri.fwd.scan import get_centroid_scan

        pars = {
            "y_center": 1049.9, "y_size": 75.0, "tilt_y": -2e-3,
            "z_center": 1116.5, "z_size": 75.0, "tilt_z": 3e-3, "tilt_x": 1e-3,
            "distance": 150e3, "o11": -1, "o12": 0, "o21": 0, "o22": -1,
            "wavelength": 0.2883, "wedge": 2.5, "chi": -1.5,
        }  # fmt: skip
        geom = geom_from_pars(pars, 0.0, 0.0, 0.0, 0.0, sig_beam=0.5, voxel_size=1.0)
        self.assertEqual(geom["wedge"], -2.5)
        U = np.asarray(
            anri.maths.rotation.rot_z(25.0) @ anri.maths.rotation.rot_x(35.0) @ anri.maths.rotation.rot_y(10.0)
        )
        B = np.asarray(anri.crystal.lattice.lpars_to_B(np.array([2.8694, 2.8694, 2.8694, 90.0, 90.0, 90.0])))
        ubi = np.linalg.inv(U @ B)
        checked = 0
        for hkl in ([1.0, 1.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 1.0], [-1.0, 2.0, 1.0], [3.0, 1.0, 0.0]):
            for etasign in (1.0, -1.0):
                c, valid = get_centroid_scan(
                    jnp.asarray(ubi), jnp.zeros(3), jnp.asarray(hkl), etasign, geom["wavelength"], geom["k_in_lab"],
                    0.0, 0.0, geom["wedge"], geom["chi"], geom["y0"],
                    geom["s_step_lab"], geom["f_step_lab"], geom["det_origin_lab"],
                )  # fmt: skip
                if not valid or not (0 < c[0] < 2162 and 0 < c[1] < 2068):
                    continue
                xyz = transform.compute_xyz_lab(np.array([[c[0]], [c[1]]]), **pars)
                tth, eta = transform.compute_tth_eta_from_xyz(
                    xyz, np.array([c[2]]), wedge=pars["wedge"], chi=pars["chi"]
                )
                g = transform.compute_g_vectors(
                    tth, eta, np.array([c[2]]), pars["wavelength"], pars["wedge"], pars["chi"]
                )
                _close((ubi @ g).ravel(), hkl)
                checked += 1
        self.assertGreater(checked, 3)


if __name__ == "__main__":
    unittest.main()
