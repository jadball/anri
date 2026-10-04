import unittest

import numpy as np

from anri.crystal import (
    Crystal,
    Symmetry,
    UnitCell,
    cubochoric_quaternions,
    disorientation,
    laue_rotations,
    mat_to_quat,
    mat_to_rod,
    orientation_grid,
    quat_mul,
    quat_to_mat,
    rod_to_mat,
    to_fundamental_zone,
)


def ops_of(lpars, sg):
    c = Crystal(UnitCell.from_lpars(lpars), Symmetry.from_number(sg))
    return laue_rotations(np.asarray(c.sym_ops), np.asarray(c.B))


def random_rotations(n, seed=0):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return quat_to_mat(q / np.linalg.norm(q, axis=1, keepdims=True))


CASES = {  # name: (lattice parameters, space group, Laue-group rotations)
    "m-3m": ([3.6, 3.6, 3.6, 90, 90, 90], 225, 24),
    "m-3": ([5.4, 5.4, 5.4, 90, 90, 90], 205, 12),
    "6/mmm": ([2.95, 2.95, 4.68, 90, 90, 120], 194, 12),
    "6/m": ([9.4, 9.4, 6.9, 90, 90, 120], 176, 6),
    "-3m": ([4.9, 4.9, 13.0, 90, 90, 120], 167, 6),
    "-3": ([5.0, 5.0, 14.0, 90, 90, 120], 148, 3),
    "4/mmm": ([3.0, 3.0, 5.0, 90, 90, 90], 139, 8),
    "mmm": ([4.0, 5.0, 6.0, 90, 90, 90], 62, 4),
    "2/m": ([4.0, 5.0, 6.0, 90, 100, 90], 14, 2),
    "-1": ([4.0, 5.0, 6.0, 80, 100, 95], 2, 1),
}


class TestConversions(unittest.TestCase):
    def test_round_trips(self):
        U = random_rotations(500)
        np.testing.assert_allclose(quat_to_mat(mat_to_quat(U)), U, atol=1e-12)
        small = U[np.trace(U, axis1=1, axis2=2) > -0.5]  # rotation angles well below 180 degrees
        np.testing.assert_allclose(np.asarray(rod_to_mat(mat_to_rod(small))), small, atol=1e-5)

    def test_quat_mul_is_matrix_product(self):
        a, b = random_rotations(50, 1), random_rotations(50, 2)
        np.testing.assert_allclose(quat_to_mat(quat_mul(mat_to_quat(a), mat_to_quat(b))), a @ b, atol=1e-12)


class TestLaueRotations(unittest.TestCase):
    def test_groups(self):
        for name, (lpars, sg, n) in CASES.items():
            with self.subTest(name):
                S = ops_of(lpars, sg)
                self.assertEqual(len(S), n)
                np.testing.assert_allclose(S[0], np.eye(3), atol=1e-6)
                np.testing.assert_allclose(S @ np.swapaxes(S, 1, 2), np.broadcast_to(np.eye(3), S.shape), atol=1e-5)
                np.testing.assert_allclose(np.linalg.det(S), 1.0, atol=1e-5)
                for a in S:  # closed under products
                    self.assertTrue(np.all([np.any(np.all(np.isclose(S, a @ b, atol=1e-5), axis=(1, 2))) for b in S]))

    def test_equivalent_orientations_give_the_same_reflections(self):
        lpars, sg, _ = CASES["6/mmm"]
        c = Crystal(UnitCell.from_lpars(lpars), Symmetry.from_number(sg))
        B, S = np.asarray(c.B, float), ops_of(lpars, sg)
        U = random_rotations(1, 3)[0]
        h = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 2], [2, -1, 3]], float)
        g = U @ B @ h.T  # scattering vectors of U
        for s in S:  # U S makes the same set of vectors from the symmetry-equivalent reflections
            gs = U @ s @ B
            hp = np.linalg.solve(B, s.T @ B @ h.T)  # the reflections that U S takes to g
            np.testing.assert_allclose(gs @ hp, g, atol=1e-6)
            np.testing.assert_allclose(hp, np.round(hp), atol=1e-6)


class TestFundamentalZone(unittest.TestCase):
    def test_disorientation(self):
        S = ops_of(*CASES["m-3m"][:2])
        U = random_rotations(100)
        np.testing.assert_allclose(disorientation(U, U @ S[7], S), 0.0, atol=1e-4)
        V = random_rotations(100, 1)
        np.testing.assert_allclose(disorientation(U, V, S), disorientation(V, U, S), atol=1e-6)
        self.assertLessEqual(disorientation(U, V, S).max(), 62.81)  # the largest cubic disorientation

    def test_to_fundamental_zone(self):
        S = ops_of(*CASES["4/mmm"][:2])
        U = to_fundamental_zone(random_rotations(200), S)
        np.testing.assert_allclose(to_fundamental_zone(U, S), U, atol=1e-12)  # already there


class TestGrids(unittest.TestCase):
    def test_cubochoric(self):
        q = cubochoric_quaternions(6)
        self.assertEqual(len(q), 12**3)
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-12)
        self.assertTrue(np.any(np.all(np.isclose(np.abs(q), [1, 0, 0, 0]), axis=1)))  # the identity
        # uniform over SO(3): the rotation angle w has density (1 - cos w) / pi
        w = 2 * np.arccos(np.clip(np.abs(cubochoric_quaternions(20)[:, 0]), 0, 1))
        self.assertAlmostEqual(np.mean(w), np.pi / 2 + 2 / np.pi, delta=0.01)

    def test_grids_cover_the_fundamental_zone(self):
        """Every grid point is in the zone, and every orientation is within delta of one."""
        test = random_rotations(300, 5)
        for name, step in (("m-3m", 6.0), ("6/mmm", 8.0), ("-3", 10.0), ("2/m", 12.0)):
            with self.subTest(name):
                S = ops_of(*CASES[name][:2])
                U, delta = orientation_grid(step, S)
                U = U.astype(float)
                angle = np.degrees(np.arccos(np.clip((np.trace(U, axis1=1, axis2=2) - 1) / 2, -1, 1)))
                least = disorientation(np.repeat(np.eye(3)[None], len(U), 0), U, S)  # smallest equivalent angle
                self.assertLessEqual((angle - least).max(), 0.5 * step + 1e-3)  # in the zone, or its thin shell
                nearest = np.array([disorientation(np.repeat(t[None], len(U), 0), U, S).min() for t in test])
                self.assertLessEqual(nearest.max(), delta)

    def test_cubochoric_zone_sizes(self):
        """A zone holds 1 / |group| of the rotations (fewer for the cubic Rodrigues grid, which is finer)."""
        n_all = len(cubochoric_quaternions(int(np.round(131.97049 / (6.0 - 0.03732)))))
        for name in ("6/mmm", "mmm", "-1"):
            S = ops_of(*CASES[name][:2])
            U, _ = orientation_grid(6.0, S)
            ratio = len(U) * len(S) / n_all
            self.assertGreaterEqual(ratio, 0.99)
            self.assertLess(ratio, 1.3)  # plus the shell


if __name__ == "__main__":
    unittest.main()
