"""Calibrate the detector and beam geometry on a scanning-3DXRD scan.

    python -m anri.calib rings <..._dataset.h5 | image> [options] -> <...>_rings.h5
    python -m anri.calib fit <..._fit.h5> [options]      -> <...>_calib.h5
    python -m anri.calib pars <..._calib.h5> <folder>    ImageD11 parameter files (geometry.par, <phase>.par, pars.json)
    python -m anri.calib show <any stage file>           how it was made, back to the dataset

rings fits the distance, beam centre, tilt_y and tilt_z to a known phase's powder rings (ImageD11's powder
calibration), so that the sample indexes; ``pars`` turns the result into the pars.json to index with.

The map comes from ``python -m anri.index`` (its fit stage), indexed with a geometry good enough to index; the peaks
from ImageD11's peaks table (the DataSet's). A calibrant frees the wavelength (``--free`` default); any other sample
frees the lattice scale instead (``--free distance,y_center,z_center,tilt_x,tilt_y,tilt_z,lattice``).
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time

T0 = time.perf_counter()


def log(msg: str) -> None:
    """Print a timestamped progress message."""
    print(f"[{time.perf_counter() - T0:7.1f} s] {msg}", flush=True)


def parse_args(argv: list | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(prog="python -m anri.calib", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit", help="fit the geometry to an indexed map and the scan's peaks")
    f.add_argument("source", help="anri.index fit stage file")
    f.add_argument("--out", help="output file (default: the source's name with _calib)")
    f.add_argument("--peaks", help="ImageD11 peaks table (default: the DataSet's pksfile)")
    f.add_argument("--lattice", type=float, nargs=6, metavar=("A", "B", "C", "ALPHA", "BETA", "GAMMA"),
                   help="reference lattice, e.g. a calibrant's certified one (default: the phase's)")  # fmt: skip
    f.add_argument("--free", default="distance,y_center,z_center,tilt_x,tilt_y,tilt_z,wavelength",
                   help="comma-separated free parameters (default: detector and wavelength); 'lattice' (the lattice "
                   "scale) instead of 'wavelength' for a sample that is not a calibrant")  # fmt: skip
    f.add_argument("--tols", default="0.1,0.05,0.025,0.025", help="(h, k, l) matching tolerance of each round")
    f.add_argument("--iter", type=int, default=10, help="Gauss-Newton iterations per round (default 10)")
    f.add_argument("--clip", type=float, default=4.0, help="leave out peaks beyond this many sigma (default 4)")
    f.add_argument("--extra-rounds", type=int, default=6, help="repeat the last tolerance while a round still "
                   "moves the geometry by more than its errors, at most this many times (default 6)")  # fmt: skip
    f.add_argument("--row-step", type=int, default=1, help="use the peaks of every Nth dty row only, e.g. 10 on "
                   "a CPU (default 1: all)")  # fmt: skip
    f.add_argument("--min-occ", type=float, default=0.01, help="leave out occupancies below this x the largest")
    f.add_argument("--min-peaks", type=int, default=20, help="free the lattice of orientations with this many "
                   "peaks (the strain left; default 20)")  # fmt: skip
    f.add_argument("--n-cpu", type=int, default=4, help="CPU devices for JAX (default 4)")

    r = sub.add_parser("rings", help="fit the geometry to powder rings, close enough to index")
    r.add_argument("source", help="ImageD11 DataSet (..._dataset.h5; its sparse pixels are summed) or a detector image")
    r.add_argument("--out", help="output file (default <sample>_<dataset>_rings.h5 or <image>_rings.h5)")
    r.add_argument("--parfile", help="pars.json: starting geometry and phase (default: the DataSet's)")
    r.add_argument("--phase", help="phase whose rings are fitted (default: the only one)")
    r.add_argument("--mask", help="detector mask, non-zero where masked (default: the DataSet's)")
    r.add_argument("--spatial", nargs=2, metavar=("E2DX", "E2DY"), help="distortion files for an image")
    r.add_argument("--scans", type=int, help="sum only this many scans, spread evenly (default: all)")
    r.add_argument("--threshold", type=float, default=0.0, help="keep pixels above this (default 0)")
    r.add_argument("--free", default="distance,y_center,z_center,tilt_y,tilt_z", help="comma-separated free "
                   "parameters (default: distance, centre, tilt_y, tilt_z)")  # fmt: skip
    r.add_argument("--tols", default="0.2,0.1", help="ring tolerance (deg 2theta) of each round: a few pixels "
                   "wide at least (each sector's mean is over the pixels within it)")  # fmt: skip
    r.add_argument("--iter", type=int, default=30, help="Gauss-Newton iterations per round at most (default 30)")
    r.add_argument("--n-eta", type=int, default=360, help="eta sectors, one residual per ring and sector")
    r.add_argument("--clip", type=float, default=4.0, help="leave out sectors beyond this many sigma (default 4)")
    r.add_argument("--no-search", action="store_true", help="skip the centre and distance search (start close)")
    r.add_argument("--max-shift", type=float, default=200.0, help="centre search range, pixels (default 200)")
    r.add_argument("--span", type=float, nargs=2, default=(0.7, 1.4), metavar=("LO", "HI"),
                   help="distance search range, x the start (default 0.7 1.4)")  # fmt: skip
    r.add_argument("--n-cpu", type=int, default=4, help="CPU devices for JAX (default 4)")

    w = sub.add_parser("pars", help="write a calibration as ImageD11 parameter files")
    w.add_argument("source", help="calibration file")
    w.add_argument("folder", help="where to write geometry.par, <phase>.par and pars.json")

    s = sub.add_parser("show", help="how a stage file was made, all the way back to the dataset")
    s.add_argument("file")
    return p.parse_args(argv)


def main() -> None:
    """Run the subcommand named on the command line."""
    args = parse_args()
    if args.command == "show":
        from anri.io.provenance import show

        print(show(args.file))
        return
    if args.command == "pars":
        from anri.calib.stages import write_pars

        print(write_pars(args.source, args.folder))
        return
    import anri.utils

    anri.utils.setup(n_cpu=args.n_cpu)
    cmd = "python -m anri.calib " + shlex.join(sys.argv[1:])
    if args.command == "rings":
        from anri.calib.stages import stage_rings

        stage_rings(args.source, args.out, parfile=args.parfile, phase=args.phase, detector_mask=args.mask,
                    spatial=args.spatial, scans=args.scans, threshold=args.threshold,
                    free=tuple(k for k in args.free.split(",") if k),
                    tols=tuple(float(t) for t in args.tols.split(",")), n_iter=args.iter, n_eta=args.n_eta,
                    clip=args.clip, search=not args.no_search, max_shift=args.max_shift, span=tuple(args.span),
                    command=cmd, log=log)  # fmt: skip
        log(f"done in {time.perf_counter() - T0:.0f} s")
        return
    from anri.calib.stages import stage_fit

    stage_fit(args.source, args.out, peaks=args.peaks, lattice=args.lattice,
              free=tuple(k for k in args.free.split(",") if k),
              tols=tuple(float(t) for t in args.tols.split(",")), n_iter=args.iter, clip=args.clip,
              extra_rounds=args.extra_rounds, row_step=args.row_step, min_occ=args.min_occ, min_peaks=args.min_peaks,
              command=cmd, log=log)  # fmt: skip
    log(f"done in {time.perf_counter() - T0:.0f} s")


if __name__ == "__main__":
    main()
