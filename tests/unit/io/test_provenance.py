import os
import tempfile
import unittest

from anri.io.provenance import check_out


class TestCheckOut(unittest.TestCase):
    def test_refuses_an_input(self):
        # a stage's output must not be one of its inputs: opening it for writing would erase it (issue #26)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "a_fit.h5")
            open(src, "w").close()
            check_out(os.path.join(tmp, "a_populations.h5"), [src, None])
            for out in (src, os.path.join(tmp, ".", "a_fit.h5")):
                with self.assertRaises(ValueError):
                    check_out(out, [None, src])
            link = os.path.join(tmp, "link.h5")
            try:
                os.symlink(src, link)
            except (OSError, NotImplementedError):  # e.g. Windows without the privilege
                return
            with self.assertRaises(ValueError):
                check_out(link, [src])


if __name__ == "__main__":
    unittest.main()
