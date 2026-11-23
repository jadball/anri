"""The theory pages are the single source of truth for Anri's maths: every numbered equation is proved (a test in
tests/theory), defined, or cited, and every proof a page names exists."""

import ast
import os
import re
import unittest

HERE = os.path.dirname(__file__)
THEORY = os.path.join(HERE, "..", "..", "docs", "source", "theory")
ROOT = os.path.join(HERE, "..", "..")
LABEL = re.compile(r"^\s+:label:\s*(\S+)", re.MULTILINE)
MATH = re.compile(r"^\.\. math::|^\s+\.\. math::", re.MULTILINE)
PROOF = re.compile(r"``(tests/theory/test_\w+\.py)::(test_\w+)``")
# after each numbered equation, before the next one, one of these must appear
STATUS = re.compile(r"\*\*(Proof|Definition|Cited|Measured):\*\*")


def _pages() -> dict:
    pages = {}
    for fn in sorted(os.listdir(THEORY)):
        if fn.endswith(".rst"):
            with open(os.path.join(THEORY, fn), encoding="utf-8") as f:
                pages[fn] = f.read()
    return pages


def _tests(path: str) -> set:
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}


class TestTheoryPages(unittest.TestCase):
    def test_every_equation_has_a_status(self):
        missing = []
        for fn, text in _pages().items():
            starts = [m.start() for m in MATH.finditer(text)] + [len(text)]
            for a, b in zip(starts[:-1], starts[1:]):
                block = text[a:b]
                labels = LABEL.findall(block)
                if labels and not STATUS.search(block):
                    missing.append(f"{fn}: {labels[0]}")
        self.assertEqual(
            missing, [], "numbered equations with no **Proof:**, **Definition:**, **Cited:** or **Measured:**"
        )

    def test_proofs_exist(self):
        missing = []
        for fn, text in _pages().items():
            for path, name in PROOF.findall(text):
                if not os.path.exists(os.path.join(ROOT, path)) or name not in _tests(path):
                    missing.append(f"{fn}: {path}::{name}")
        self.assertEqual(missing, [], "proofs named in docs/source/theory that do not exist")


if __name__ == "__main__":
    unittest.main()
