import os
import tempfile
import unittest

import jax
import numpy as np

from anri.calib import fit, rings

jax.config.update("jax_enable_x64", True)

# a small detector (1024 x 1024 pixels of 150 um) so the rings reach 2theta ~ 40 degrees
TRUE = {"y_center": 515.0, "y_size": 150.0, "tilt_y": -0.0015, "z_center": 508.0, "z_size": 150.0, "tilt_z": 0.002,
        "tilt_x": 0.0, "distance": 150000.0, "o11": -1.0, "o12": 0.0, "o21": 0.0, "o22": -1.0, "wavelength": 0.19,
        "wedge": 0.0, "chi": 0.0}  # fmt: skip
SI = (np.array([5.43094, 5.43094, 5.43094, 90.0, 90.0, 90.0]), 227)


def _image(width: float = 0.03) -> np.ndarray:
    """Silicon's powder rings on the detector, Gaussian in 2theta (width in degrees), seen from the true geometry."""
    s, f = (a.ravel().astype(float) for a in np.mgrid[0:1024, 0:1024])
    tth = np.asarray(rings.pixel_tth_eta(s, f, fit.base_pars(TRUE, 0.0))[0])
    t = np.degrees(2 * np.arcsin(rings.ring_ds(*SI, TRUE["wavelength"], 30.0) * TRUE["wavelength"] / 2))
    img = np.exp(-0.5 * ((tth[:, None] - t[None]) / width) ** 2).sum(1)
    return np.where(img > 1e-3, img, 0.0).reshape(1024, 1024)


def _start() -> dict:
    p = fit.base_pars(TRUE, 0.0)
    return dict(p, distance=p["distance"] * 1.005, y_center=p["y_center"] + 2.0, z_center=p["z_center"] - 1.5,
                tilt_y=p["tilt_y"] + 0.002, tilt_z=p["tilt_z"] - 0.002)  # fmt: skip


class TestFitRings(unittest.TestCase):
    def test_recovers_geometry(self):
        """From a wrong distance, centre and tilts, the rings come back to the true geometry."""
        px = rings.lit_pixels(_image())
        ds = rings.ring_ds(*SI, TRUE["wavelength"], 30.0)
        res = rings.fit_rings(px, ds, _start(), log=lambda _: None)
        self.assertLess(abs(res["pars"]["distance"] / TRUE["distance"] - 1), 2e-5)
        for k in ("y_center", "z_center"):
            self.assertLess(abs(res["pars"][k] - TRUE[k]), 0.02, k)
        for k in ("tilt_y", "tilt_z"):
            self.assertLess(abs(res["pars"][k] - TRUE[k]), 2e-5, k)
        o = rings.ring_offsets(px, ds, res["pars"], 0.05, sep=0.2)  # straight rings: sectors on their ring
        self.assertLess(np.nansum(np.abs(o["offset"]) * o["intensity"]) / o["intensity"].sum(), 2e-3)

    def test_search_from_far(self):
        """From a centre 70 pixels and a distance 8% off, beyond any ring assignment, the search comes back."""
        px = rings.lit_pixels(_image())
        ds = rings.ring_ds(*SI, TRUE["wavelength"], 30.0)
        p = dict(fit.base_pars(TRUE, 0.0), y_center=TRUE["y_center"] + 40.0, z_center=TRUE["z_center"] - 60.0,
                 distance=TRUE["distance"] * 1.08)  # fmt: skip
        p, _ = rings.find_centre(px, p, log=lambda _: None)
        self.assertLess(np.hypot(p["y_center"] - TRUE["y_center"], p["z_center"] - TRUE["z_center"]), 1.5)
        p, _ = rings.find_distance(px, ds, p, log=lambda _: None)
        self.assertLess(abs(p["distance"] / TRUE["distance"] - 1), 2e-3)
        res = rings.fit_rings(px, ds, p, log=lambda _: None)
        self.assertLess(abs(res["pars"]["y_center"] - TRUE["y_center"]), 0.02)

    def test_stage_on_an_image(self):
        """stage_rings on a dense image writes a geometry that write_pars turns into a pars.json."""
        import fabio

        import anri.io.imaged11 as io
        from anri.calib.diagnostics import read_rings
        from anri.calib.stages import stage_rings, write_pars
        from anri.io.provenance import read_provenance

        cell = {"cell__a": SI[0][0], "cell__b": SI[0][1], "cell__c": SI[0][2], "cell_alpha": 90.0,
                "cell_beta": 90.0, "cell_gamma": 90.0, "cell_lattice_[P,A,B,C,I,F,R]": SI[1]}  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            start = fit.imaged11_pars(_start())
            parfile = io.write_pars(os.path.join(tmp, "start"), start, {"Si": cell})
            img = os.path.join(tmp, "si.edf")
            fabio.edfimage.EdfImage(data=_image().astype(np.float32)).write(img)
            out = stage_rings(img, os.path.join(tmp, "si_rings.h5"), parfile=parfile, log=lambda _: None)
            pars = io.read_pars_json(write_pars(out, os.path.join(tmp, "fitted")))[0]
            d = read_rings(out)  # what the plots draw: the searches and the rings' offsets
            log = read_provenance(out)["log"]  # the stage keeps its log, though nothing was printed
        self.assertEqual(len(d["centre"]["levels"]), 3)
        self.assertAlmostEqual(d["distance"][np.argmax(d["score"])] / TRUE["distance"], 1.0, delta=0.01)
        self.assertEqual(d["offset"].shape, (len(d["tth"]), 360))
        self.assertLess(abs(pars["distance"] / TRUE["distance"] - 1), 2e-5)
        self.assertLess(abs(pars["y_center"] - TRUE["y_center"]), 0.02)
        self.assertEqual(pars["wavelength"], TRUE["wavelength"])  # not fitted
        self.assertIn(f"-> {out}", log[-1])
