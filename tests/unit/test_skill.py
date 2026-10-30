"""The agent skill (anri/skill/anri) must match the code: its commands, options, names and links.

The skill is read by agents that run anri, so a renamed option or function would mislead them silently. The format is
the open Agent Skills specification (https://agentskills.io/specification): ``SKILL.md`` with a ``name`` equal to its
directory and a ``description`` of at most 1024 characters, under 500 lines, and reference files it links to.
"""

import contextlib
import importlib
import io
import os
import re
import unittest

import anri

SKILL = os.path.join(os.path.dirname(anri.__file__), "skill", "anri")
COMMAND = re.compile(r"python -m anri\.(index|simulate) ([a-z]+)([^`\n]*)")
FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")
DOTTED = re.compile(r"`(anri\.[\w.]+)")
PLOT = re.compile(r"`(plot_\w+)`")
LINK = re.compile(r"\]\(([^)#:]+\.md)\)")
# where the bare names in the skill's tables live
MODULES = ("anri.index.diagnostics", "anri.index.plots", "anri.index.stages", "anri.index.mask",
           "anri.simulate.diagnostics", "anri.simulate.plots")  # fmt: skip


def skill_files() -> dict:
    """Every markdown file of the skill, by path relative to its directory."""
    out = {}
    for root, _, files in os.walk(SKILL):
        for fn in files:
            if fn.endswith(".md"):
                path = os.path.join(root, fn)
                with open(path, encoding="utf-8") as f:
                    out[os.path.relpath(path, SKILL)] = f.read()
    return out


def options(cli: str, sub: str) -> set:
    """The options a subcommand of python -m anri.<cli> takes, read from its help."""
    parse_args = importlib.import_module(f"anri.{cli}.__main__").parse_args
    text = io.StringIO()
    with contextlib.redirect_stdout(text), contextlib.suppress(SystemExit):
        parse_args([sub, "-h"])
    return set(FLAG.findall(text.getvalue()))


def resolve(name: str) -> object:
    """Import the longest module prefix of a dotted name, then get the rest as attributes."""
    parts = name.rstrip(".").split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(name)


def column(text: str, header: str) -> list:
    """The cells of every markdown table's column whose header starts with ``header``."""
    cells, col = [], None
    for line in text.splitlines():
        row = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("|") else None
        if row is None:
            col = None
        elif col is None:
            col = next((i for i, c in enumerate(row) if c.startswith(header)), -1)
        elif col >= 0 and not set(row[0]) <= set("-: "):
            cells.append(row[col])
    return cells


class TestSkill(unittest.TestCase):
    def setUp(self):
        self.files = skill_files()

    def test_format(self):
        text = self.files["SKILL.md"]
        self.assertLess(len(text.splitlines()), 500)
        head = text.split("---\n")[1]  # the frontmatter
        self.assertIn("\nname: anri\n", "\n" + head)
        lines = head.split("description: >-\n")[1].splitlines()
        desc = " ".join(line.strip() for line in lines[: next(i for i, x in enumerate(lines) if not x.startswith(" "))])
        self.assertGreater(len(desc), 100)
        self.assertLessEqual(len(desc), 1024)

    def test_links(self):
        for rel, text in self.files.items():
            for link in LINK.findall(text):
                path = os.path.join(SKILL, os.path.dirname(rel), link)
                self.assertTrue(os.path.exists(path), f"{rel} links to {link}, which is not there")

    def test_commands_and_options(self):
        """Each command names a real subcommand and its own options; every other option is one of some subcommand."""
        subs = {"index": ("histogram", "search", "filter", "fit", "localsearch", "populations", "run", "show",
                          "powder"),
                "simulate": ("prepare", "render", "assemble", "run", "show")}  # fmt: skip
        known = {(cli, sub): options(cli, sub) for cli, names in subs.items() for sub in names}
        every = set().union(*known.values())
        n = 0
        for rel, text in self.files.items():
            for cli, sub, rest in COMMAND.findall(text):
                self.assertIn((cli, sub), known, f"{rel}: python -m anri.{cli} {sub}")
                for flag in FLAG.findall(rest):
                    self.assertIn(flag, known[cli, sub], f"{rel}: python -m anri.{cli} {sub} ... {flag}")
                n += 1
            for flag in FLAG.findall(text):
                self.assertIn(flag, every, f"{rel}: {flag} is no option of python -m anri.index or anri.simulate")
        self.assertGreater(n, 10)

    def test_names(self):
        """Every dotted anri name, plot, and function in a table's "Function" column exists."""
        missing = []
        for rel, text in self.files.items():
            names = set(DOTTED.findall(text))
            for name in sorted(names):
                try:
                    resolve(name)
                except (ImportError, AttributeError):
                    missing.append(f"{rel}: {name}")
            bare = set(PLOT.findall(text))
            for cell in column(text, "Function"):
                bare |= set(re.findall(r"`(\w+)`", cell))
            for name in sorted(bare):
                if not any(hasattr(importlib.import_module(m), name) for m in MODULES):
                    missing.append(f"{rel}: {name}")
        self.assertEqual(missing, [], "names in the skill that no longer exist")


if __name__ == "__main__":
    unittest.main()
