"""Provenance of anri's stage files: what made each file and from what, as plain HDF5 attributes.

A method's stages each write one HDF5 file with a ``provenance`` group. Its attributes hold the stage, a unique id,
the file it was made from and that file's id (checked when the chain is followed, so a remade or replaced source is
refused), the anri version, git commit and dirty flag, the date, working directory and command. Three subgroups hold
attributes too: ``options`` (as given; None left out), ``resolved`` (worked out from the data) and ``results`` (a
summary). The dataset ``provenance/log`` holds every message the stage logged, timed from its start (:func:`recorder`,
:func:`write_log`), whichever front-end ran it. :func:`show` prints the chain back to the first stage.
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import time
import uuid
from collections.abc import Callable

import h5py
import numpy as np

import anri


def _git_commit() -> dict:
    """Return the git commit of the running anri code and whether it has uncommitted changes ({} outside git)."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                                check=False)  # fmt: skip
        status = subprocess.run(["git", "-C", here, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=10, check=False)  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return {}
    if commit.returncode != 0:
        return {}
    return {"git_commit": commit.stdout.strip(), "git_dirty": bool(status.stdout.strip())}


def write_attrs(group: h5py.Group, values: dict) -> None:
    """Store a flat dict as attributes; None (which HDF5 cannot hold) is left out, lists become arrays."""
    for k, v in values.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            strings = len(v) > 0 and all(isinstance(x, str) for x in v)
            v = np.asarray(v, dtype=h5py.string_dtype() if strings else None)
        group.attrs[k] = v


def read_attrs(group: h5py.Group) -> dict:
    """Read attributes back as Python values (str, bool, int, float, lists)."""
    out = {}
    for k, v in group.attrs.items():
        if isinstance(v, bytes):
            v = v.decode()
        v = v.tolist() if hasattr(v, "tolist") else v
        out[k] = [x.decode() if isinstance(x, bytes) else x for x in v] if isinstance(v, list) else v
    return out


def write_provenance(
    h: h5py.File, stage: str, source: str | None, options: dict, resolved: dict, command: str | None = None
) -> str:
    """Write a stage file's provenance group: what made it and from what. Returns the file's new id.

    Parameters
    ----------
    h
        The stage file, open for writing
    stage
        The stage's name
    source
        The file this stage read (None for a first stage), whose id is recorded too
    options, resolved
        Every option of this stage as given, and every value worked out from the data
    command
        The command line, if run from one
    """
    file_id = uuid.uuid4().hex
    p = h.create_group("provenance")
    write_attrs(p, {"stage": stage, "id": file_id, "anri_version": anri.VERSION, **_git_commit(),
                    "date": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
                    "cwd": os.getcwd(), "command": command})  # fmt: skip
    if source is not None:
        p.attrs["source"] = os.path.abspath(source)
        p.attrs["source_id"] = read_provenance(source)["id"]
    write_attrs(p.create_group("options"), options)
    write_attrs(p.create_group("resolved"), resolved)
    p.create_group("results")
    return file_id


def write_results(path: str, results: dict) -> None:
    """Add a summary of a stage's results to its provenance (``provenance/results``)."""
    with h5py.File(path, "a") as h:
        write_attrs(h["provenance/results"], results)


def recorder(log: Callable) -> tuple[Callable, list]:
    """Return a log callable that passes each message on to ``log`` and keeps it, and the list it keeps them in.

    A stage wraps the ``log`` it is given with this, so that its file holds what it logged (:func:`write_log`)
    whatever ran it: the command line, a notebook or a workflow system. Each message is kept as
    ``"[  12.3 s] message"``, timed from the call to this function.
    """
    lines: list = []
    t0 = time.perf_counter()

    def record(msg: str) -> None:
        lines.append(f"[{time.perf_counter() - t0:7.1f} s] {msg}")
        log(msg)

    return record, lines


def write_log(path: str, lines: list) -> None:
    """Store a stage's log messages (from :func:`recorder`) in its file, as the dataset ``provenance/log``."""
    with h5py.File(path, "a") as h:
        p = h["provenance"]
        if "log" in p:
            del p["log"]
        p.create_dataset("log", data=np.asarray(lines, dtype=h5py.string_dtype()))


def read_provenance(path: str) -> dict:
    """Read a stage file's provenance: attributes, with "options", "resolved", "results" as dicts and "log" as a list.

    "log" is empty for a file written before stages kept their log.
    """
    with h5py.File(path, "r") as h:
        p = h["provenance"]
        out = read_attrs(p)
        for g in ("options", "resolved", "results"):
            out[g] = read_attrs(p[g])
        out["log"] = [str(x) for x in p["log"].asstr()[()]] if "log" in p else []
    return out


def source_of(path: str) -> str | None:
    """Return the file a stage file was made from (None for a first stage), checking that it is that file.

    The recorded path is tried first, then the same name beside the file (files moved together). Its id must be the
    one recorded: a source remade or replaced since is refused.
    """
    prov = read_provenance(path)
    if "source" not in prov:
        return None
    src = prov["source"]
    if not os.path.exists(src):
        src = os.path.join(os.path.dirname(os.path.abspath(path)), os.path.basename(prov["source"]))
        if not os.path.exists(src):
            msg = f"{path} was made from {prov['source']}, which is not there (nor beside {path})"
            raise FileNotFoundError(msg)
    src_id = read_provenance(src)["id"]
    if src_id != prov["source_id"]:
        msg = (f"{src} is not the file {path} was made from (its id is {src_id}, {path} records "
               f"{prov['source_id']}): it was remade or replaced since")  # fmt: skip
        raise ValueError(msg)
    return src


def chain(path: str) -> list[str]:
    """Return a stage file and every file it was made from, back to the first stage (this file first)."""
    files = [path]
    while (src := source_of(files[-1])) is not None:
        files.append(src)
    return files


def upstream(path: str, stage: str) -> str:
    """Follow a stage file's sources back to the file of a stage (this file's own or an earlier one)."""
    for p in chain(path):
        if read_provenance(p)["stage"] == stage:
            return p
    msg = f"{path} has no {stage} stage upstream"
    raise ValueError(msg)


def show(path: str, log: bool = False) -> str:
    """Return how a stage file was made, all the way back to the first stage, as text: one section per stage.

    Parameters
    ----------
    path
        Any stage file
    log
        Also print each stage's log
    """
    lines = []
    for p in reversed(chain(path)):
        prov = read_provenance(p)
        lines.append(f"[{prov['stage']}]")
        lines.append(f"file = {os.path.abspath(p)!r}")
        for k, v in prov.items():
            if k not in ("stage", "options", "resolved", "results", "log"):
                lines.append(f"{k} = {v!r}")
        for g in ("options", "resolved", "results"):
            if prov[g]:
                lines.append(f"\n[{prov['stage']}.{g}]")
                lines += [f"{k} = {v!r}" for k, v in prov[g].items()]
        if log and prov["log"]:
            lines.append(f"\n[{prov['stage']}.log]")
            lines += prov["log"]
        lines.append("")
    return "\n".join(lines)


def check_out(out: str, inputs: list) -> None:
    """Refuse an output path that is one of a stage's inputs, before it is opened for writing (which would erase it).

    Parameters
    ----------
    out
        The stage's output file
    inputs
        The files the stage reads (None entries are skipped), e.g. :func:`chain` of its source
    """
    for f in inputs:
        if f is None:
            continue
        if os.path.exists(out) and os.path.exists(f):
            same = os.path.samefile(out, f)  # also through links
        else:
            same = os.path.normcase(os.path.realpath(out)) == os.path.normcase(os.path.realpath(f))
        if same:
            msg = f"the output {out} is the input {f}: give another output file"
            raise ValueError(msg)


def stage_path(source: str, stage: str, out: str | None = None) -> str:
    """Return a stage's output path: ``out`` if given, else the source's name with its stage replaced, beside it."""
    if out is not None:
        return out
    prev = read_provenance(source)["stage"]
    base = os.path.basename(source)
    stem = base[: -len(f"_{prev}.h5")] if base.endswith(f"_{prev}.h5") else os.path.splitext(base)[0]
    return os.path.join(os.path.dirname(os.path.abspath(source)), f"{stem}_{stage}.h5")


def command_line(parser: argparse.ArgumentParser, command: str, positionals: list, options: dict) -> list[str]:
    """Return the command line that gives a stage its recorded options again: the program, command and arguments.

    Each option is the subcommand's ``--name`` (``_`` as ``-``), or its option whose dest is the name; a flag the
    command line can only turn off (``--no-name``) is given when the option is False. A list is given as its option
    takes it: separate values (``nargs``), the flag once per value (``append``), or comma-separated (one value; a
    one-item list keeps its comma, ``9,``, so that the command can tell it from a number).

    Parameters
    ----------
    parser
        The method's command line, with one subcommand per stage
    command
        The subcommand, e.g. "run"
    positionals
        Its positional arguments, as given (anything else to put first, e.g. "--out", path)
    options
        Recorded options (:func:`read_provenance`), each one the subcommand takes

    Returns
    -------
    list[str]
        For ``shlex.join``
    """
    subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    actions = [a for a in subparsers.choices[command]._actions if a.option_strings]
    by_flag = {s: a for a in actions for s in a.option_strings}
    by_dest = {a.dest: a for a in actions}
    out = [*parser.prog.split(), command, *(str(x) for x in positionals)]
    for k, v in options.items():
        off = "--no-" + k.replace("_", "-")
        if off in by_flag:  # on by default; the command line can only turn it off
            out += [] if v else [off]
            continue
        a = by_flag.get("--" + k.replace("_", "-"), by_dest.get(k))
        if a is None:
            msg = f"{parser.prog} {command} has no option for {k!r}"
            raise KeyError(msg)
        flag = a.option_strings[0]
        if a.nargs == 0:  # a flag
            out += [flag] if v else []
            continue
        values = [str(x) for x in (v if isinstance(v, list) else [v])]
        if isinstance(a, argparse._AppendAction):
            out += [x for value in values for x in (flag, value)]
        elif a.nargs is None:
            out += [flag, ",".join(values) + ("," if isinstance(v, list) and len(values) == 1 else "")]
        else:
            out += [flag, *values]
    return out
