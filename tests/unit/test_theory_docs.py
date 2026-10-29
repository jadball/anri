"""Every function and module the theory pages name must exist, so a rename fails here instead of leaving a dead link."""

import importlib
import os
import re
import unittest

THEORY = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "source", "theory")
ROLE = re.compile(r":(?:func|mod|class|data):`~?(anri\.[\w.]+)`")


def resolve(name: str) -> object:
    """Import the longest module prefix of a dotted name, then get the rest as attributes."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(name)


class TestTheoryNames(unittest.TestCase):
    def test_names_import(self):
        names = set()
        for fn in sorted(os.listdir(THEORY)):
            if fn.endswith(".rst"):
                with open(os.path.join(THEORY, fn), encoding="utf-8") as f:
                    names |= set(ROLE.findall(f.read()))
        self.assertGreater(len(names), 50)
        missing = []
        for name in sorted(names):
            try:
                resolve(name)
            except (ImportError, AttributeError):
                missing.append(name)
        self.assertEqual(missing, [], "names in docs/source/theory that no longer exist")


if __name__ == "__main__":
    unittest.main()
