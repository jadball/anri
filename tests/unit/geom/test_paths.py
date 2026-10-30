import unittest

import numpy as np

import anri.geom


def disc(n, radius, step):
    """A disc of the given radius on an n x n reconstruction grid of voxel size step (fraction filled ~ 0 or 1)."""
    pos = np.asarray(anri.geom.scan.recon_positions(n, step))
    return (np.hypot(pos[:, 0], pos[:, 1]) <= radius).reshape(n, n).astype(np.float32), pos


def chord(x0, y0, phi, radius):
    """Exact path from (x0, y0) inside a circle to its edge along (cos phi, sin phi)."""
    u = np.array([np.cos(phi), np.sin(phi)])
    b = x0 * u[0] + y0 * u[1]
    return -b + np.sqrt(b * b - (x0 * x0 + y0 * y0 - radius * radius))


class TestEdgeDistances(unittest.TestCase):
    def test_disc(self):
        """Paths to the edge of a disc, from its centre and off it, in every direction, to about a voxel."""
        n, step, r = 101, 2.0, 80.0
        m, _ = disc(n, r, step)
        pts = np.array([[0.0, 0.0], [30.0, -20.0], [-60.0, 10.0]], np.float32)
        D = np.asarray(anri.geom.paths.edge_distances(m, step, pts, n_phi=72))
        phis = np.arange(72) * 2 * np.pi / 72
        for p, d in zip(pts, D):
            np.testing.assert_allclose(d, chord(p[0], p[1], phis, r), atol=1.5 * step)

    def test_outside(self):
        """Outside the sample, looking away from it: no path."""
        m, _ = disc(61, 30.0, 2.0)
        D = np.asarray(anri.geom.paths.edge_distances(m, 2.0, np.array([[50.0, 0.0]], np.float32), n_phi=4))
        self.assertLess(D[0, 0], 1e-3)  # phi = 0: along +x, away from the disc
        self.assertGreater(D[0, 2], 50.0)  # phi = 180: through it (a 60 um chord, less the voxel edges)


class TestTransmission(unittest.TestCase):
    def test_centre_of_disc(self):
        """At the disc's centre, beam along x, small 2theta: both paths ~ the radius; out of plane: / cos."""
        n, step, r, mu = 101, 2.0, 80.0, 0.004
        m, _ = disc(n, r, step)
        D = anri.geom.paths.edge_distances(m, step, np.zeros((1, 2), np.float32), n_phi=360)
        k_out = np.array([[[np.cos(0.05), 0.0, np.sin(0.05)]]])  # 2theta ~ 2.9 deg, straight up (eta = 0)
        d_in, d_out = anri.geom.paths.ray_directions(np.array([1.0, 0, 0]), k_out, np.array([[30.0]]))
        np.testing.assert_allclose(
            np.asarray(d_in)[0, 0], [-np.cos(np.radians(30)), np.sin(np.radians(30)), 0], atol=1e-6
        )
        L_out = np.asarray(anri.geom.paths.path_lengths(D, d_out))
        np.testing.assert_allclose(L_out, r / np.cos(0.05), atol=1.5 * step)
        T = np.asarray(anri.geom.paths.transmission(D, d_in, d_out, mu))
        np.testing.assert_allclose(T, np.exp(-mu * r * (1 + 1 / np.cos(0.05))), rtol=0.02)


class TestAttenuationMap(unittest.TestCase):
    def test_weighted_mean_and_layout(self):
        """Each voxel: its entries' density-weighted mean mu, at the voxel recon_positions puts there; others empty."""
        step = 2.0
        pos = np.array([[4.0, 2.0, 0], [4.2, 1.9, 0], [-6.0, 0.0, 0]])
        m = anri.geom.paths.attenuation_map(pos, np.array([1.0, 3.0, 5.0]), np.array([3.0, 1.0, 2.0]), step)
        n = m.shape[0]
        grid = np.asarray(anri.geom.scan.recon_positions(n, step))
        at = {tuple(np.round(g[:2], 6)): v for g, v in zip(grid, m.ravel())}
        self.assertAlmostEqual(at[(4.0, 2.0)], 1.5, places=6)  # (3 x 1 + 1 x 3) / 4
        self.assertAlmostEqual(at[(-6.0, 0.0)], 5.0, places=6)
        self.assertEqual(int((m > 0).sum()), 2)
        np.testing.assert_array_equal(m[[0, -1]], 0)  # an empty border
