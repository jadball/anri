import unittest

import jax
import jax.numpy as jnp
import numpy as np

import anri.geom
import anri.maths.rotation
from anri.geom.beam import beam_weight

jax.config.update("jax_enable_x64", True)


class TestBeamBasis(unittest.TestCase):
    def test_along_x(self):
        for got, expected in zip(anri.geom.beam.beam_basis(jnp.array([2.0, 0.0, 0.0])), np.eye(3)):
            np.testing.assert_allclose(got, expected, atol=1e-15)

    def test_orthonormal(self):
        rng = np.random.default_rng(0)
        for k in rng.normal(size=(20, 3)):
            k_hat, e_h, e_v = anri.geom.beam.beam_basis(jnp.asarray(k))
            basis = np.stack([k_hat, e_h, e_v])
            np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(np.linalg.det(basis), 1.0, atol=1e-12)  # right-handed, like (x, y, z)
            np.testing.assert_allclose(k_hat, k / np.linalg.norm(k), atol=1e-12)
            self.assertAlmostEqual(float(e_h[2]), 0.0)  # horizontal


class TestBeamOffsets(unittest.TestCase):
    def test_beam_along_x(self):
        across, height, rise, cos_alpha = anri.geom.beam.beam_offsets(
            jnp.array([5.0, -2.0, 3.0]), jnp.array([2.0, 0.0, 0.0])
        )
        np.testing.assert_allclose([across, height, rise, cos_alpha], [-2.0, 3.0, 0.0, 1.0], atol=1e-15)

    def test_tilted_beam(self):
        k = jnp.array([1.0, 0.3, 0.1])
        p = jnp.array([5.0, -2.0, 3.0])
        across, height, rise, cos_alpha = anri.geom.beam.beam_offsets(p, k)
        k_hat, e_h, _ = anri.geom.beam.beam_basis(k)
        self.assertAlmostEqual(float(across), float(p @ e_h), places=12)
        self.assertAlmostEqual(float(height + rise), float(p[2]), places=12)
        self.assertAlmostEqual(float(cos_alpha), float(jnp.hypot(k_hat[0], k_hat[1])), places=12)
        # a point on the beam's centre line is at zero height and zero distance across
        on_line = anri.geom.beam.beam_offsets(7.0 * k_hat, k)
        np.testing.assert_allclose([on_line[0], on_line[1]], 0.0, atol=1e-12)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoid rule (np.trapezoid needs numpy >= 2)."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _geom(sig, width=0.0, size=1.2, k=(1.0, 0.0, 0.0), sig_v=0.0, width_v=0.0, cube=False):
    """The beam_weight part of a geometry dict."""
    return {"k_in_lab": jnp.asarray(k), "sig_beam": sig, "width_beam": width, "sig_beam_v": sig_v,
            "width_beam_v": width_v, "voxel_3d": cube, "voxel_size": size}  # fmt: skip


def _weight(delta, omega, geom, z=0.0):
    """beam_weight of a voxel at lab (0, delta, z): delta along lab y from the beam's centre line."""
    return beam_weight(jnp.array([0.0, delta, z]), omega, geom)


class TestBeamWeight(unittest.TestCase):
    def test_area(self):
        # a column's weight integrated over its distance from the beam is its area, for any beam: the profile
        # integrates to 1
        delta = np.linspace(-20, 20, 40001)
        for omega in [0.0, 10.0, 45.0, 90.0, 123.4]:
            for sig, width in [(0.05, 0.0), (0.5, 0.0), (3.0, 0.0), (0.05, 2.0), (0.5, 6.0)]:
                g = _geom(sig, width, size=2.0)
                w = jax.vmap(_weight, in_axes=(0, None, None))(jnp.asarray(delta), omega, g)
                np.testing.assert_allclose(_trapezoid(np.asarray(w), delta), 4.0, rtol=1e-6)

    def test_convolution(self):
        # compare with a brute-force convolution of the chord length with the profile, Gaussian and flat-top
        from scipy.special import ndtr

        size, omega = 1.5, 30.0
        u = np.linspace(-6, 6, 12001)
        c, s = np.abs(np.cos(np.radians(omega))), np.abs(np.sin(np.radians(omega)))
        a, b = 0.5 * size * (c + s), 0.5 * size * abs(c - s)
        chord = np.clip((a - np.abs(u)) / (a - b), 0, 1) * size / max(c, s)
        for sig, width in [(0.4, 0.0), (0.1, 2.0), (0.4, 1.0)]:
            for d in [-1.0, 0.0, 0.3, 1.2]:
                if width == 0:
                    p = np.exp(-0.5 * ((d - u) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
                else:
                    p = (ndtr((d - u + width / 2) / sig) - ndtr((d - u - width / 2) / sig)) / width
                np.testing.assert_allclose(
                    _weight(d, omega, _geom(sig, width, size)), _trapezoid(chord * p, u), rtol=1e-4
                )

    def test_continuous(self):
        # the rectangle branch at omega = 0 joins the trapezoid, and the Gaussian limit joins the flat top
        w = [_weight(0.3, om, _geom(0.4, size=1.0)) for om in (0.0, 1e-3, 1e-1)]
        np.testing.assert_allclose(w[0], w[1], rtol=1e-5)
        np.testing.assert_allclose(w[0], w[2], rtol=1e-3)
        w = [_weight(0.3, 30.0, _geom(0.4, width)) for width in (0.0, 0.0039, 0.0041)]  # switch at 0.01 sigma
        np.testing.assert_allclose(w[0], w[1], rtol=1e-6)
        np.testing.assert_allclose(w[0], w[2], rtol=1e-5)


def _brute_force_weight(delta, omega, size, sig_h, k_in, sig_v=None, width_h=0.0, cube_z=None, n=400, nz=600):
    """Integrate the beam profile over a square voxel rotated by omega, shifted by delta along lab y.

    The beam passes through the origin along k_in, across it horizontally a flat top of width_h blurred by a
    Gaussian of sig_h. With sig_v, it is also Gaussian vertically, and the voxel is a column, integrated over height
    on a grid; with cube_z too, the voxel is a cube centred at that height.
    """
    from scipy.special import ndtr

    k = np.asarray(k_in, float) / np.linalg.norm(k_in)
    e_h = np.cross([0.0, 0.0, 1.0], k)
    e_h /= np.linalg.norm(e_h)
    e_v = np.cross(k, e_h)
    t = (np.arange(n) + 0.5) / n * size - size / 2
    s, u = np.meshgrid(t, t, indexing="ij")
    c, sn = np.cos(np.radians(omega)), np.sin(np.radians(omega))
    x, y = c * s - sn * u, sn * s + c * u + delta  # voxel rotated by +omega about z, then moved by delta along y
    area = (size / n) ** 2

    def profile(d, sig, width=0.0):
        if width == 0:
            return np.exp(-0.5 * (d / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
        return (ndtr((d + width / 2) / sig) - ndtr((d - width / 2) / sig)) / width

    p_h = profile(x * e_h[0] + y * e_h[1], sig_h, width_h)
    if sig_v is None:
        return np.sum(p_h) * area
    if cube_z is None:
        zmax = 8 * (sig_v + size) / e_v[2]
        z = (np.arange(nz) + 0.5) / nz * 2 * zmax - zmax
    else:
        z = cube_z + (np.arange(nz) + 0.5) / nz * size - size / 2
    p_v = profile(x[..., None] * e_v[0] + y[..., None] * e_v[1] + z * e_v[2], sig_v)
    return np.sum(p_h[..., None] * p_v) * area * (z[1] - z[0])


class TestBeamWeightGeometry(unittest.TestCase):
    def test_horizontal_beam(self):
        # a beam at psi from lab x in the horizontal plane, Gaussian and flat-top
        for psi, omega, delta, width in ((20.0, 30.0, 0.3, 0.0), (-35.0, 70.0, -0.5, 0.0), (10.0, 0.0, 0.0, 1.5)):
            k = [np.cos(np.radians(psi)), np.sin(np.radians(psi)), 0.0]
            expected = _brute_force_weight(delta, omega, 1.2, 0.4, k, width_h=width)
            np.testing.assert_allclose(_weight(delta, omega, _geom(0.4, width, k=k)), expected, rtol=2e-4)

    def test_tilted_pencil(self):
        # a column, tilted beam: the pencil's vertical profile integrates out
        for alpha, psi, omega, delta in ((25.0, 0.0, 30.0, 0.2), (40.0, 15.0, 50.0, -0.4)):
            a, p = np.radians(alpha), np.radians(psi)
            k = [np.cos(a) * np.cos(p), np.cos(a) * np.sin(p), np.sin(a)]
            expected = _brute_force_weight(delta, omega, 1.2, 0.4, k, sig_v=0.3, n=200)
            np.testing.assert_allclose(_weight(delta, omega, _geom(0.4, k=k)), expected, rtol=1e-3)

    def test_cube(self):
        # a cube (3D map), horizontal beam: the vertical profile over the cube's height
        for omega, delta, z in ((30.0, 0.3, 0.0), (10.0, -0.2, 0.5), (60.0, 0.1, -1.0)):
            expected = _brute_force_weight(delta, omega, 1.2, 0.4, [1, 0, 0], sig_v=0.3, cube_z=z, n=200, nz=400)
            g = _geom(0.4, sig_v=0.3, cube=True)
            np.testing.assert_allclose(_weight(delta, omega, g, z=z), expected, rtol=1e-3)
