import unittest

from anri.crystal.orientation import cubochoric_quaternions


class TestGrids(unittest.TestCase):
    def test_cubochoric_size(self):
        """(2 N)^3 points (values: tests/theory/test_search.py::test_cubochoric_code)."""
        self.assertEqual(cubochoric_quaternions(6).shape, (12**3, 4))


if __name__ == "__main__":
    unittest.main()
