"""Index an ImageD11 scanning-3DXRD dataset from scratch: orientation populations per voxel.

The indexer runs in five stages, each a subcommand that writes one HDF5 file the next one reads:

    python -m anri.index histogram <analysisroot> <sample> <dataset> [options]   -> <sample>_<dataset>_histogram.h5
    python -m anri.index search  <..._histogram.h5> [options]                    -> <...>_search.h5
    python -m anri.index filter  <..._search.h5> [options]                       -> <...>_filter.h5
    python -m anri.index fit     <..._filter.h5> [options]                       -> <...>_fit.h5
    python -m anri.index populations <..._fit.h5> [options]                      -> <...>_populations.h5, _entries.npz,
                                                                                    _tmap.h5
    python -m anri.index run <analysisroot> <sample> <dataset> [options]         all five
    python -m anri.index show <any stage file>                                   how it was made, back to the dataset

Each file records its options, the values resolved from the data, the file it was made from (checked by id when
read) and the anri version and git commit. Re-run a stage with other options and a different ``--out`` to keep
variants side by side. Paths follow ImageD11's layout: ``{analysisroot}/{sample}/{sample}_{dataset}/
{sample}_{dataset}_dataset.h5`` and ``_sparse.h5``; the geometry and phase come from ``pars.json``.
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


def _rings(s: str) -> int | list[int]:
    """--rings: a number (the first rings) or comma-separated ring indices."""
    return [int(v) for v in s.split(",") if v.strip()] if "," in s else int(s)


def _histogram_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--phase", help="phase name in pars.json (default: the only one)")
    p.add_argument(
        "--parfile", help="pars.json (default: the DataSet's parfile, else pars/pars.json beside PROCESSED_DATA)"
    )
    p.add_argument(
        "--rings",
        type=_rings,
        default=6,
        help="number of rings used, the first ones (default 6); "
        "or which rings, as comma-separated indices from the lowest 2theta (0 first), e.g. 9,10,11",
    )
    p.add_argument(
        "--ring-merge",
        type=float,
        default=1e-4,
        help="reflections within this in d* (1/angstrom) "
        "are one ring (default 1e-4): merge rings the data cannot separate",
    )
    p.add_argument("--cif", help="CIF of the phase, for structure factors (default: |F|^2 = 1)")
    p.add_argument("--tth-tol", type=float, help="2theta tolerance of the rings (deg; default: measured per ring)")
    p.add_argument("--monitor", help="normalise intensities by this counter in each scan's measurement, frame by frame, "
                   "as ImageD11's DataSet.set_monitor does (e.g. fpico6; default: no normalisation)")  # fmt: skip
    p.add_argument("--dty-source", choices=("dataset", "frames"), default="dataset", help="which dty bins the frames "
                   "into rows: the DataSet's (default; for a fly or helical scan one row per rotation, at its mean), or "
                   "each frame's own (the DataSet's plus the motion within its row, anri.io.imaged11.read_frame_dty: a helical "
                   "rotation then spreads over 2-3 rows)")  # fmt: skip
    p.add_argument("--no-frame-dty", action="store_true", help="ignore dty's motion within each row (fly and helical "
                   "scans): by default each frame's dty is the DataSet's for its row plus the motion read from the "
                   "sparse file (anri.io.imaged11.read_frame_dty), and the model puts each voxel in the rows that really saw it")  # fmt: skip
    p.add_argument(
        "--saturation",
        type=float,
        help="detector counts at which a pixel saturated (e.g. 65535): bins "
        "holding one are lower bounds in the fit (default: none)",
    )
    p.add_argument("--detector-mask", help="detector mask, non-zero for masked pixels, as ImageD11's segmenter reads "
                   "it (default: the DataSet's maskfile): predictions are scaled by the measured share of their "
                   "(ring, eta) bin, so reflections on module gaps or off the detector are not taken as missing")  # fmt: skip
    p.add_argument(
        "--grid-pad",
        type=float,
        default=0.0,
        help="widen the voxel grid by this on every side (sample "
        "units, e.g. um), for a sample reaching past the scanned range; fixed here for every later stage",
    )
    p.add_argument("--gridstep", type=int, default=1, help="voxel = gridstep x dty step; data rows are summed in groups "
                   "of gridstep to match (default 1)")  # fmt: skip
    p.add_argument("--check", action="store_true", help="print the resolved paths and parameters, then stop")


def _search_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--grid", type=float, help="orientation grid step, deg (default: the coarsest of 3, 2.5, 2, 1.5, 1 "
                   "whose chance completeness is at most --max-chance)")  # fmt: skip
    p.add_argument(
        "--max-chance", type=float, default=0.5, help="chance completeness allowed by the automatic grid (default 0.5)"
    )
    p.add_argument("--lit", type=float, default=1.0, help="lit threshold, x the median non-empty bin (default 1)")
    p.add_argument("--etacut", type=float, default=0.2, help="use reflections with |sin eta| above this (default 0.2)")


def _keep_option(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--keep", type=int, default=100000, help="at most this many orientations for the occupancy fit (default 100000)"
    )


def _censor_option(p: argparse.ArgumentParser) -> None:
    p.add_argument("--censor", type=float, default=0.0, help="counts per histogram bin below which an empty bin "
                   "(every pixel below the segmentation cut) counts as agreeing with the model, in both MLEMs "
                   "(default 0: empty bins are zeros)")  # fmt: skip


def _filter_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prune", choices=("likelihood", "completeness"), default="likelihood",
                   help="keep orientations by the likelihood ratio of a global orientation fit (default; intensity-aware), "
                   "or by completeness alone")  # fmt: skip
    p.add_argument("--min-comp", type=float, help="completeness needed: with --prune completeness, default halfway "
                   "between the grid's median (the chance level) and its maximum; with --prune likelihood, default the "
                   "chance level (raising it drops decoys, and small grains, which have lower completeness)")  # fmt: skip
    p.add_argument("--min-lr", type=float, default=25.0, help="likelihood ratio needed to keep an orientation (default 25, "
                   "about 5 sigma)")  # fmt: skip


def _fit_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--y0", type=float, help="dty where the rotation axis is in the beam (default: the DataSet's y0)")
    p.add_argument(
        "--iter",
        type=int,
        default=10,
        help="MLEM iterations, SQUAREM-accelerated (3 MLEM updates each; default 10: thin features such as twins "
        "converge long after the deviance flattens, while fewer keep the populations' means a little closer to the truth "
        "between grid points)",
    )
    p.add_argument("--cand", type=int, default=64, help="candidate orientations per voxel (default 64)")
    p.add_argument("--coarse", type=int, default=1, help="first fit voxels this many times larger, then give each voxel "
                   "the candidates of its coarse neighbourhood (default 1: off; 4 is ~16x cheaper on large maps)")  # fmt: skip
    p.add_argument(
        "--block-gb", type=float, default=1.0, help="memory for one block of voxels' system entries (default 1 GB)"
    )
    p.add_argument("--beam", type=float, default=0.0, help="FWHM of the beam across dty, as dty: each voxel is spread "
                   "over the rows by this Gaussian profile integrated over the voxel. Default 0: linearly over the 2 "
                   "nearest rows, close to a beam of FWHM = the dty step (as fast, and maps as good there); set it for "
                   "a beam wider than the step, e.g. overfocused")  # fmt: skip
    p.add_argument(
        "--mask",
        default="disc",
        help="fit only the voxels in a sample mask: 'disc' (default: the voxels "
        "the beam sees at every omega, within the scanned dty range all round the rotation axis, grown by --pad), "
        "'auto' (the sample's support in the projections: voxels whose row is lit at every omega it is scanned at, "
        "largest region, convex hull; also outside the disc), or a .npy file of [voxels, voxels] booleans in "
        "reconstruction order, e.g. drawn with anri.index.mask.draw_mask or an ImageD11 mask on the same grid",
    )
    p.add_argument(
        "--pad",
        type=float,
        default=0.0,
        help="grow the disc mask by this (sample units, e.g. um): for a "
        "sample a little larger than the disc, whose outer part would otherwise pile onto the disc's edge",
    )
    p.add_argument(
        "--mu",
        type=float,
        help="linear attenuation coefficient (1 / sample units, e.g. 1/um) for absorption along the incoming and "
        "diffracted paths through the mask (default: from the histogram stage's --cif)",
    )
    p.add_argument(
        "--no-absorption",
        action="store_true",
        help="do not model absorption, even with a CIF",
    )


def _populations_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--occupied", type=float, default=0.2, help="voxels count as occupied (in the TensorMap and entries) "
                   "above this x the 99th percentile of the total occupancy in the mask (default 0.2; the raw occupancy is always saved)")  # fmt: skip
    p.add_argument("--min-frac", type=float, default=0.1, help="report populations holding at least this fraction of a "
                   "voxel's occupancy (default 0.1)")  # fmt: skip


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line (default: ``sys.argv``)."""
    p = argparse.ArgumentParser(prog="python -m anri.index", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    dataset = argparse.ArgumentParser(add_help=False)
    for a in ("analysisroot", "sample", "dataset"):
        dataset.add_argument(a)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--n-cpu", type=int, default=4, help="CPU devices for JAX (default 4)")

    h = sub.add_parser("histogram", parents=[dataset, common], help="bin the dataset's pixels into the histograms")
    _histogram_options(h)
    h.add_argument("--out", help="output file (default <sample>_<dataset>_histogram.h5 in --outdir)")
    h.add_argument("--outdir", default=".")

    s = sub.add_parser("search", parents=[common], help="completeness of every orientation of a grid")
    s.add_argument("source", help="a histogram stage file")
    _search_options(s)
    s.add_argument("--out", help="output file (default: the source's name with _search)")

    f = sub.add_parser("filter", parents=[common], help="choose the orientations for the map")
    f.add_argument("source", help="a search stage file")
    _filter_options(f)
    _keep_option(f)
    _censor_option(f)
    f.add_argument("--out", help="output file (default: the source's name with _filter)")

    m = sub.add_parser("fit", parents=[common], help="occupancies of the kept orientations per voxel")
    m.add_argument("source", help="a filter stage file")
    _fit_options(m)
    _censor_option(m)
    m.add_argument("--out", help="output file (default: the source's name with _fit)")

    q = sub.add_parser("populations", parents=[common], help="orientation populations, TensorMap and entries")
    q.add_argument("source", help="a fit stage file")
    _populations_options(q)
    q.add_argument("--out", help="output file (default: the source's name with _populations)")

    r = sub.add_parser("run", parents=[dataset, common], help="all five stages, into --outdir")
    _histogram_options(r)
    _search_options(r)
    _keep_option(r)
    _filter_options(r)
    _fit_options(r)
    _populations_options(r)
    _censor_option(r)
    r.add_argument("--outdir", default=".")

    w = sub.add_parser("show", help="how a stage file was made, all the way back to the dataset")
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
    import anri.index as ix

    cmd = args.command
    command = "python -m anri.index " + shlex.join(sys.argv[1:])
    if cmd in ("histogram", "run"):
        out = getattr(args, "out", None) or os.path.join(args.outdir, f"{args.sample}_{args.dataset}_histogram.h5")
        source = ix.stages.stage_histogram(
            args.analysisroot, args.sample, args.dataset, out, phase=args.phase, parfile=args.parfile,
            rings=args.rings, ring_merge=args.ring_merge, cif=args.cif, tth_tol=args.tth_tol, monitor=args.monitor, dty_source=args.dty_source,
            frame_dty=not args.no_frame_dty, gridstep=args.gridstep, grid_pad=args.grid_pad, saturation=args.saturation, detector_mask=args.detector_mask, check=args.check, command=command, log=log,
        )  # fmt: skip
        if args.check or cmd == "histogram":
            return
    else:
        source = args.source
    out = None if cmd == "run" else args.out
    if cmd in ("search", "run"):
        source = ix.stages.stage_search(source, out, grid=args.grid, max_chance=args.max_chance, lit=args.lit,
                                 etacut=args.etacut, command=command, log=log)  # fmt: skip
    if cmd in ("filter", "run"):
        source = ix.stages.stage_filter(source, out, prune=args.prune, min_comp=args.min_comp, min_lr=args.min_lr,
                                 keep=args.keep, censor=args.censor, command=command, log=log)  # fmt: skip
    if cmd in ("fit", "run"):
        source = ix.stages.stage_fit(source, out, y0=args.y0, cand=args.cand, n_iter=args.iter, coarse=args.coarse,
                              beam=args.beam, mask=args.mask, pad=args.pad, absorption=not args.no_absorption, mu=args.mu, block_gb=args.block_gb, censor=args.censor,
                              command=command, log=log)  # fmt: skip
    if cmd in ("populations", "run"):
        ix.stages.stage_populations(
            source, out, occupied=args.occupied, min_frac=args.min_frac, command=command, log=log
        )
    log(f"done in {time.perf_counter() - T0:.0f} s")


if __name__ == "__main__":
    main()
