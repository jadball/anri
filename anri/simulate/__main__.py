"""Simulate a scanning-3DXRD dataset from a map, as ImageD11 files.

A dataset is simulated in three stages, each a subcommand that writes one HDF5 file the next one reads:

    python -m anri.simulate prepare <map> <pars> [options]                 -> <map name>_prepare.h5
    python -m anri.simulate render <..._prepare.h5> [--rows a:b] [options] -> <...>_render.h5 (or _render_<a>-<b>.h5)
    python -m anri.simulate assemble <..._prepare.h5> <analysisroot> <sample> <dataset>
                                                                         -> the ImageD11 dataset, and <...>_assemble.h5
    python -m anri.simulate run <map> <pars> <analysisroot> <sample> <dataset> [options]   all three
    python -m anri.simulate show <any stage file>                          how it was made

The map is an ImageD11 TensorMap (every phase in it) or the entries ``python -m anri.index populations`` writes.
Rows are independent: split them between jobs with ``render --rows``, then ``assemble`` gathers every part made from
the prepare file.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time

T0 = time.perf_counter()


def log(msg: str) -> None:
    """Print a timestamped progress message."""
    print(f"[{time.perf_counter() - T0:7.1f} s] {msg}", flush=True)


def _cif(values: list | None) -> str | dict | None:
    """--cif PATH (one phase) or --cif NAME=PATH (repeated, one per phase)."""
    if not values:
        return None
    if len(values) == 1 and "=" not in values[0]:
        return values[0]
    return dict(v.split("=", 1) for v in values)


def _prepare_options(p: argparse.ArgumentParser) -> None:
    from anri.simulate.stages import BANDWIDTH, OPTICS

    p.add_argument("--dataset", dest="input_dataset", help="an ImageD11 DataSet to simulate: its frames' omega and dty, detector shape, y0 and "
                   "geometry replace the scan options")  # fmt: skip
    p.add_argument("--remap", nargs="+", metavar="OLD=NEW",
                   help="rewrite the path prefixes the DataSet records, e.g. where the data are mounted elsewhere")  # fmt: skip
    p.add_argument("--cif", action="append", help="CIF for the structure factors: PATH for a single phase, or "
                   "NAME=PATH per phase (repeat it); phases without one get |F|^2 = 1")  # fmt: skip
    p.add_argument("--optics", choices=sorted(OPTICS), default="al-crl", help="ID11's focusing optics, which set the "
                   "beam's convergence: al-crl (Al CRL boxes, sigma 8e-5 rad; default) or si-nanolens (Si planar "
                   "nanolenses, 1.5e-4 rad)")  # fmt: skip
    p.add_argument("--divergence", type=float, help="the convergence's standard deviation (rad, H and V) instead")
    p.add_argument("--bandwidth", type=float, default=BANDWIDTH,
                   help=f"FWHM of dlambda/lambda (default {BANDWIDTH:g}: ID11's Laue Si(111) monochromator)")  # fmt: skip
    p.add_argument("--beam-fwhm", type=float, help="FWHM of the beam across dty (default: the dty step)")
    p.add_argument("--psf", type=float, default=0.5, help="detector point spread, sigma in pixels (default 0.5)")
    p.add_argument("--pol-factor", type=float, default=1.0, help="degree of horizontal polarisation (default 1)")
    p.add_argument("--y0", type=float, help="dty where the rotation axis is in the beam (default: --dataset's, else 0)")
    p.add_argument("--omega-range", type=float, nargs=2, default=(0.0, 180.0), help="rotation, deg (default 0 180)")
    p.add_argument("--omega-step", type=float, default=0.05, help="frame width, deg (default 0.05)")
    p.add_argument("--dty-step", type=float, help="between rows (default: the map's voxel size)")
    p.add_argument("--margin", type=float, default=3.0,
                   help="rows cover the sample and this many beam FWHM either side (default 3)")  # fmt: skip
    p.add_argument("--density", type=float, default=1.0, help="counts scale: every density x this (default 1)")
    p.add_argument("--det-shape", type=int, nargs=2, default=(2162, 2068),
                   help="detector pixels, slow fast (default 2162 2068: an Eiger 4M)")  # fmt: skip
    p.add_argument("--tth-max", type=float, help="largest 2theta of the reflections (default: the largest ring "
                   "wholly on the detector)")  # fmt: skip


def _render_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--cut", type=float, default=1.0, help="keep counts above this, as ImageD11's segmentation (default 1)"
    )
    p.add_argument("--window", type=int, nargs=3, default=(3, 7, 7),
                   help="peak window: frames, slow and fast pixels (default 3 7 7)")  # fmt: skip
    p.add_argument("--max-frames", type=int, default=31,
                   help="widest window in frames, for peaks broad in omega (default 31)")  # fmt: skip
    p.add_argument("--batch", type=int, default=2**16, help="peaks per batch (default 65536)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line (default: ``sys.argv``)."""
    p = argparse.ArgumentParser(prog="python -m anri.simulate", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--n-cpu", type=int, default=4, help="CPU devices for JAX (default 4)")
    inputs = argparse.ArgumentParser(add_help=False)
    inputs.add_argument("map", help="an ImageD11 TensorMap (.h5) or python -m anri.index's _entries.npz")
    inputs.add_argument("pars", nargs="?", help="ImageD11 geometry: a .par file or pars.json (default: --dataset's)")
    dataset = argparse.ArgumentParser(add_help=False)
    for a in ("analysisroot", "sample", "dataset"):
        dataset.add_argument(a)

    q = sub.add_parser("prepare", parents=[inputs, common], help="map entries, reflections, beam and scan")
    _prepare_options(q)
    q.add_argument("--out", help="output file (default <map name>_prepare.h5 in --outdir)")
    q.add_argument("--outdir", default=".")

    r = sub.add_parser("render", parents=[common], help="rows of the scan into sparse pixels")
    r.add_argument("source", help="a prepare stage file")
    r.add_argument("--rows", help="a:b, the rows to render (a Python slice; default all)")
    _render_options(r)
    r.add_argument("--out", help="output file (default: the source's name with _render, and the rows)")

    a = sub.add_parser("assemble", parents=[common], help="the rendered rows as an ImageD11 dataset")
    a.add_argument("source", help="a prepare stage file")
    for name in ("analysisroot", "sample", "dataset"):
        a.add_argument(name)
    a.add_argument("--parts", nargs="+", help="render stage files (default: all made from the source, beside it)")
    a.add_argument("--peaks-table", action="store_true", help="also write ImageD11's peaks table")
    a.add_argument("--out", help="this stage's file (default: the source's name with _assemble)")

    u = sub.add_parser("run", parents=[inputs, dataset, common], help="all three stages")
    _prepare_options(u)
    _render_options(u)
    u.add_argument("--peaks-table", action="store_true", help="also write ImageD11's peaks table")
    u.add_argument("--outdir", default=".", help="where the stage files go (default .)")

    w = sub.add_parser("show", help="how a stage file was made")
    w.add_argument("file")
    return p.parse_args(argv)


def main() -> None:
    """Run the subcommand named on the command line."""
    args = parse_args()
    if args.command == "show":
        from anri.io.provenance import show

        print(show(args.file))
        return
    import anri.utils

    anri.utils.setup(n_cpu=args.n_cpu)
    import anri.simulate as sim

    cmd = args.command
    command = "python -m anri.simulate " + shlex.join(sys.argv[1:])
    if cmd in ("prepare", "run"):
        out = getattr(args, "out", None) or os.path.join(
            args.outdir, os.path.splitext(os.path.basename(args.map))[0] + "_prepare.h5"
        )
        source = sim.stages.stage_prepare(
            args.map, args.pars, out, dataset=args.input_dataset,
            remap=dict(v.split("=", 1) for v in args.remap) if args.remap else None, cif=_cif(args.cif), optics=args.optics, divergence=args.divergence,
            bandwidth=args.bandwidth, beam_fwhm=args.beam_fwhm, psf=args.psf, pol_factor=args.pol_factor, y0=args.y0,
            omega_range=tuple(args.omega_range), omega_step=args.omega_step, dty_step=args.dty_step,
            margin=args.margin, density=args.density, det_shape=tuple(args.det_shape), tth_max=args.tth_max,
            command=command, log=log,
        )  # fmt: skip
    else:
        source = args.source
    if cmd in ("render", "run"):
        sim.stages.stage_render(source, None if cmd == "run" else args.out, rows=getattr(args, "rows", None), cut=args.cut,
                        window=tuple(args.window), max_frames=args.max_frames, batch=args.batch, command=command,
                        log=log)  # fmt: skip
    if cmd in ("assemble", "run"):
        sim.stages.stage_assemble(source, args.analysisroot, args.sample, args.dataset, None if cmd == "run" else args.out,
                          parts=getattr(args, "parts", None), peaks_table=args.peaks_table, command=command,
                          log=log)  # fmt: skip
    log(f"done in {time.perf_counter() - T0:.0f} s")


if __name__ == "__main__":
    main()
