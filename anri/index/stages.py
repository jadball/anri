"""The indexer as five stages, each writing one HDF5 file that the next one reads, with its provenance.

1. :func:`stage_histogram`: an ImageD11 dataset binned into the fit histogram ``H[ring, eta, omega, row]`` and a
   row-summed lit histogram, with the measured ring widths and the scan resolved from the data.
2. :func:`stage_search`: the completeness of every orientation of a grid over the fundamental zone.
3. :func:`stage_filter`: the orientations kept for the fit, by a global likelihood fit or by completeness.
4. :func:`stage_fit`: occupancies of the kept orientations per voxel.
5. :func:`stage_populations`: the occupancies grouped into orientation populations, a TensorMap and map entries for
   the renderer.

Each stage is split where a choice is made by looking at the previous one's output: y0 and the sample mask are read
off the histogram, so they are options of the fit; the occupied and population cuts are read off the fit, so they are
cheap to redo without fitting again. The selections that sliders redo are functions of their own
(:func:`select_by_likelihood`, :func:`select_populations`), used by the stages too.

Each file records how it was made in a ``provenance`` group: the stage, a unique id, the file it was made from and
that file's id (checked when it is read), the anri version and git commit, the date and command, and as attributes of
``provenance/options``, ``provenance/resolved`` and ``provenance/results`` every option given, every value worked out
from the data, and a summary of the results. :func:`~anri.io.provenance.show` prints the whole chain. Variants (e.g. fits with different
options from one filter) are separate files, each pointing back to its input.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator

import h5py
import numpy as np

from anri.io.provenance import (
    chain,
    read_attrs,
    read_provenance,
    stage_path,
    upstream,
    write_attrs,
    write_provenance,
    write_results,
)

STAGES = ("histogram", "search", "filter", "fit", "populations")
B_E, B_O = 0.5, 0.25  # lit histogram bins (deg): eta, omega
R_E, R_O = 2, 4  # fit histogram bins = lit bins x these (1 x 1 deg)
N_POP = 4  # populations per voxel at most
_LATTICE_KEYS = ("cell__a", "cell__b", "cell__c", "cell_alpha", "cell_beta", "cell_gamma")


# ------------------------------------------------------------------------------------------------------- provenance
def run_command(path: str) -> str:
    """Return the ``python -m anri.index run`` command that remakes a stage file and those before it.

    Every option recorded along the chain is given, defaults included, so that the command runs the same on another
    dataset (change the dataset's three arguments). Options the command line cannot take (save_model) are left out;
    where two stages recorded different values of one option (censor, in filter and fit), the later stage's is given.

    Parameters
    ----------
    path
        Any stage file
    """
    import shlex

    opts: dict = {}
    for p in reversed(chain(path)):  # the first stage first, so that a later stage's value wins
        opts.update(read_provenance(p)["options"])
    cmd = ["python", "-m", "anri.index", "run", *(str(opts.pop(k)) for k in ("analysisroot", "sample", "dataset"))]
    for k, v in opts.items():
        if k == "frame_dty":  # on by default; the command line can only turn it off
            cmd += [] if v else ["--no-frame-dty"]
        elif k != "save_model" and v is not False:
            cmd += ["--" + {"n_iter": "iter"}.get(k, k).replace("_", "-")] + ([] if v is True else [str(v)])
    return shlex.join(cmd)


# ----------------------------------------------------------------------------------------- shared setup (one source)
def _geometry(geo: dict, y0: float, ystep: float) -> dict:
    """Build the indexer's geometry dict (float32 JAX scalars and arrays) from ImageD11 parameters.

    y0 does not enter the binning or the predictions (no parallax), only where a voxel is in the beam, which takes
    its y0 from the scan dict.
    """
    import jax.numpy as jnp

    import anri.io
    from anri.maths.utils import fwhm_to_sigma

    geom = anri.io.imaged11.geom_from_pars(
        geo, y0, fwhm_to_sigma(geo["wavelength"] * 2e-3), 1.5e-4, 1.5e-4, sig_beam=fwhm_to_sigma(ystep),
        voxel_size=ystep,
    )  # fmt: skip
    return {k: jnp.asarray(v, jnp.float32) if np.issubdtype(np.asarray(v).dtype, np.floating) else v
            for k, v in geom.items()}  # fmt: skip


def _lattice(lattice: np.ndarray, sg: int) -> tuple:
    """Return (B float32, B float64, Laue-group rotations) of a phase."""
    import anri.crystal

    B64 = anri.crystal.structure.B_matrix(lattice)
    return (
        B64.astype(np.float32),
        B64,
        anri.crystal.orientation.laue_rotations(anri.crystal.structure.symmetry_matrices(sg), B64),
    )


def lit_map(H_lit: np.ndarray, scan: dict, n_rings: int, lit: float, etacut: float) -> dict:
    """Build the lit map for completeness: bins above lit x the median non-empty bin of the row-summed lit histogram.

    Parameters
    ----------
    H_lit
        The lit histogram of a histogram stage file (:func:`load_histogram`)
    scan
        Its scan ("om0", "omega_step", "n_e", "n_o")
    n_rings
        Its number of rings
    lit, etacut
        The search stage's options (:func:`stage_search`)
    """
    import jax.numpy as jnp

    from .data import lit_table

    Hs = jnp.asarray(H_lit).reshape(n_rings, scan["n_e"], scan["n_o"])
    med = float(jnp.median(Hs[Hs > 0]))
    return {"table": lit_table(Hs > lit * med), "om0": scan["om0"], "bins": (B_E, B_O, scan["n_e"], scan["n_o"]),
            "frame_step": scan["omega_step"], "etacut": etacut}  # fmt: skip


def voxel_grid(scan: dict, y0: float) -> int:
    """Return NR: the indexer's voxel grid is NR x NR, centred on the rotation axis, as ImageD11 pads its recons."""
    import anri.geom

    _, pad = anri.geom.scan.sino_shift_and_pad(y0, scan["n_rows"], scan["dty0"], scan["ystep"])
    return scan["n_rows"] + pad


def load_histogram(path: str) -> dict:
    """Read a histogram stage file back: setup, rings, scan and both histograms, ready for the later stages."""
    import jax.numpy as jnp

    prov = read_provenance(path)
    r = prov["resolved"]
    with h5py.File(path, "r") as h:
        geo = read_attrs(h["geometry"])
        rings = {k: h["rings"][k][()] for k in h["rings"]}
        scan = read_attrs(h["scan"])
        for k in ("ddty", "exposure"):
            if k in h["scan"]:
                scan[k] = jnp.asarray(h["scan"][k][()], jnp.float32)
        H = jnp.asarray(h["H"][()])
        H_lit = jnp.asarray(h["H_lit"][()])
    geom = _geometry(geo, r.get("y0_dataset", 0.0), r["voxel_size"])
    scan.update({"k_in_lab": geom["k_in_lab"], "wedge": geom["wedge"], "chi": geom["chi"]})
    B, B64, ops = _lattice(np.asarray(r["lattice"]), r["space_group"])
    return {"prov": prov, "geom": geom, "rings": rings, "scan": scan, "H": H, "H_lit": H_lit, "B": B, "B64": B64,
            "ops": ops, "n_rings": len(rings["tth"])}  # fmt: skip


def open_dataset(
    analysisroot: str, sample: str, dataset: str, *, phase: str | None = None, parfile: str | None = None,
    monitor: str | None = None, dty_source: str = "dataset", gridstep: int = 1, log: Callable = print,
) -> dict:  # fmt: skip
    """Open an ImageD11 dataset as the indexer reads it: its files, phase, geometry, rows, and a pixel stream.

    Paths follow ImageD11's layout: ``{analysisroot}/{sample}/{sample}_{dataset}/{sample}_{dataset}_dataset.h5`` and
    ``_sparse.h5``. The geometry and the phase (lattice and space-group number) come from ``pars.json``: the
    DataSet's parfile, else ``pars/pars.json`` beside ``PROCESSED_DATA``, or ``parfile``. Pixels are corrected for
    detector distortion as ImageD11 corrects its peaks (:func:`anri.io.imaged11.read_spatial`). Options as
    :func:`stage_histogram`.

    Returns
    -------
    dict
        "ds" (the DataSet, :func:`anri.io.imaged11.read_dataset`), "dsfile", "sparsefile", "parfile", "geo" (ImageD11
        parameters), "phase", "lattice", "space_group", "ops" (Laue-group rotations), "geom" (:func:`_geometry`),
        "scan" (the histogram's rows and omega bins), "groups" (the sparse file's scans), "chunk" (pixels per chunk),
        "frame_dty" (each frame's dty, or None), "spatial_correction" (where the distortion maps came from), and
        "stream": ``stream(groups)`` yields the pixels of those scans a chunk at a time, read ahead in a thread
    """
    import anri.io

    from .data import coarsen_scan

    dsname = f"{sample}_{dataset}"
    dsdir = os.path.join(analysisroot, sample, dsname)
    dsfile = os.path.join(dsdir, f"{dsname}_dataset.h5")
    sparsefile = os.path.join(dsdir, f"{dsname}_sparse.h5")
    ds = anri.io.imaged11.read_dataset(dsfile)
    if ds["sparsefile"] and os.path.exists(ds["sparsefile"]):  # where the DataSet says, else the standard place
        sparsefile = ds["sparsefile"]
    parfile = parfile or ds["parfile"]
    if not parfile or not os.path.exists(parfile):  # e.g. processed elsewhere: pars/ beside PROCESSED_DATA
        root = os.path.abspath(analysisroot)
        while os.path.basename(root) != "PROCESSED_DATA" and root != os.path.dirname(root):
            root = os.path.dirname(root)
        parfile = os.path.join(os.path.dirname(root), "pars", "pars.json")
    geo, phase, cell = anri.io.imaged11.read_pars_json(parfile, phase)
    lpars = np.array([cell[k] for k in _LATTICE_KEYS])
    sg = cell["cell_lattice_[P,A,B,C,I,F,R]"]
    if not isinstance(sg, float):
        msg = f"{phase}: cell_lattice_[P,A,B,C,I,F,R] = {sg} is a centring letter; a space-group number is needed"
        raise ValueError(msg)  # noqa: TRY004 (a wrong value in a file, not a wrong argument type)
    sg = int(sg)

    ybin, yedge, oedge = ds["ybincens"], ds["ybinedges"], ds["obinedges"]
    ystep0 = float(np.median(np.diff(ybin)))
    # the DataSet's rows summed in groups of gridstep, voxels of one (coarse) row
    rows = coarsen_scan({"dty0": float(ybin[0]), "ystep": ystep0, "n_rows": len(ybin), "voxel": ystep0,
                         "width_beam": 0.0}, gridstep)  # fmt: skip
    om0, ostep = float(oedge[0]), float(np.median(np.diff(oedge)))
    n_e, n_o = round(360 / B_E), round(float(oedge[-1] - oedge[0]) / B_O)
    n_o -= n_o % R_O
    scan = {"dty0": rows["dty0"], "ystep": rows["ystep"], "n_rows": rows["n_rows"], "om0": om0, "voxel": rows["voxel"],
            "width_beam": rows["width_beam"], "n_e": n_e, "n_o": n_o, "omega_step": ostep}  # fmt: skip
    y0_ds = ds["y0"]
    geom = _geometry(geo, 0.0 if y0_ds is None else y0_ds, rows["ystep"])
    _, _, ops = _lattice(lpars, sg)

    with h5py.File(sparsefile, "r") as h:
        groups = list(h.keys())
        n_max = max(int(h[g]["nnz"][()].sum()) for g in groups)
    chunk = int(min(1 << 24, 1 << max(10, int(np.ceil(np.log2(max(n_max, 1)))))))
    spatial = anri.io.imaged11.read_spatial(ds)  # the DataSet's detector distortion maps, if it names any
    spatial_src = ds["detectorh5"] or (f"{ds['e2dxfile']}, {ds['e2dyfile']}" if spatial is not None else "")
    monitor_ref = None
    if monitor:  # one reference for every scan, as ImageD11 (the mean)
        mon = np.concatenate(
            list(anri.io.imaged11.read_monitor(sparsefile, groups, monitor, ds["masterfile"]).values())
        )
        monitor_ref = float(np.mean(mon))
        log(f"monitor {monitor}: mean {monitor_ref:.4g}, min / max {mon.min() / monitor_ref:.3f} / "
            f"{mon.max() / monitor_ref:.3f} of the mean; intensities normalised to the mean")  # fmt: skip
    have_frames = ds["dty"] is not None and ds["scans"] is not None and ds["omega"] is not None
    if dty_source == "frames" and not have_frames:
        msg = "dty_source='frames' needs the DataSet's dty, omega and scans"
        raise ValueError(msg)
    # each frame's dty (fly, helical scans: the DataSet's row dty plus the motion within the row)
    fdty = anri.io.imaged11.read_frame_dty(ds, sparsefile) if have_frames else None
    bin_dty = fdty if dty_source == "frames" else ds["dty"]  # what bins the frames into rows

    def stream(groups_: list) -> Iterator[tuple]:  # read the next chunk while this one is binned
        return anri.io.imaged11.prefetch(anri.io.imaged11.stream_sparse(
            sparsefile, yedge, ds["omegamotor"], ds["dtymotor"], chunk, groups_, gridstep, bin_dty, ds["scans"],
            monitor, monitor_ref, ds["masterfile"], ds["omega"], spatial,
        ))  # fmt: skip

    log(f"dataset {dsfile}; sparse pixels {sparsefile}")
    log(f"pars    {parfile}: phase {phase}, lattice {', '.join(f'{v:g}' for v in lpars)}, space group {sg}, "
        f"{len(ops)} Laue-group rotations")  # fmt: skip
    log(f"geometry: wavelength {geo['wavelength']:.5f}, distance {geo['distance']:g}; y0 (DataSet) {y0_ds}; dty "
        f"{scan['dty0']:.6g} + {scan['n_rows']} x {scan['ystep']:.6g}"
        f"{'' if gridstep == 1 else f' (rows summed in groups of {gridstep})'}; omega {om0:.4g} .. {oedge[-1]:.4g} in "
        f"{len(oedge) - 1} frames of {ostep:.4g}")  # fmt: skip
    log(f"spatial correction: {spatial_src or 'none (the DataSet names no e2dx/e2dy or detector file)'}")
    return {"ds": ds, "dsfile": dsfile, "sparsefile": sparsefile, "parfile": parfile, "geo": geo, "phase": phase,
            "lattice": lpars, "space_group": sg, "ops": ops, "geom": geom, "scan": scan, "groups": groups,
            "chunk": chunk, "frame_dty": fdty, "dty_source": dty_source, "gridstep": gridstep,
            "spatial_correction": spatial_src, "stream": stream}  # fmt: skip


def sample_groups(groups: list, n: int = 9) -> list:
    """Return n scans spread evenly over a dataset's scans (the ring profile is measured on these)."""
    return [groups[i] for i in np.unique(np.linspace(0, len(groups) - 1, min(len(groups), n)).round().astype(int))]


def select_by_likelihood(fitted: np.ndarray, lr: np.ndarray, min_lr: float, keep: int) -> np.ndarray:
    """Return the orientations kept by likelihood ratio: above min_lr, the most likely first, at most keep.

    Parameters
    ----------
    fitted, lr
        Grid indices of the orientations fitted and their likelihood ratios (a filter stage file)
    min_lr, keep
        See :func:`stage_filter`
    """
    above = np.flatnonzero(lr > min_lr)
    return fitted[above[np.argsort(lr[above])[::-1]][:keep]]


def select_occupied(tot: np.ndarray, mask: np.ndarray, occupied: float) -> np.ndarray:
    """Return the voxels in the mask above occupied x the 99th percentile of the total occupancy in the mask.

    Parameters
    ----------
    tot
        [Nv] each voxel's total occupancy
    mask
        [Nv] the voxels fitted
    occupied
        See :func:`stage_populations`
    """
    return mask & (tot > occupied * np.percentile(tot[mask], 99))


def select_populations(tot: np.ndarray, mask: np.ndarray, frac: np.ndarray, occupied: float,
                       min_frac: float) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Return the occupied voxels and the populations reported in them.

    Parameters
    ----------
    tot
        [Nv] each voxel's total occupancy
    mask
        [Nv] the voxels fitted
    frac
        [Nv, P] each population's fraction of its voxel (:func:`anri.index.populations.group`)
    occupied, min_frac
        See :func:`stage_populations`

    Returns
    -------
    occupied, present: np.ndarray
        [Nv] voxels in the mask above occupied x the 99th percentile of the total occupancy in the mask; [Nv, P]
        populations holding at least min_frac in occupied voxels (the main one always)
    """
    occ = select_occupied(tot, mask, occupied)
    present = (frac >= min_frac) & occ[:, None]
    present[:, 0] = occ  # the main population always
    return occ, present


# --------------------------------------------------------------------------------------------------------- 1. histogram
def stage_histogram(
    analysisroot: str, sample: str, dataset: str, out: str | None = None, *, phase: str | None = None,
    parfile: str | None = None, rings: int = 6, cif: str | None = None, tth_tol: float | None = None,
    monitor: str | None = None, dty_source: str = "dataset", frame_dty: bool = True, gridstep: int = 1,
    check: bool = False, command: str | None = None, log: Callable = print,
) -> str:  # fmt: skip
    """Bin an ImageD11 dataset's sparse pixels into the indexer's histograms (see :func:`open_dataset`).

    Parameters
    ----------
    analysisroot, sample, dataset
        The dataset
    out
        Output file (default ``<sample>_<dataset>_histogram.h5`` in the working directory)
    phase
        Phase name in pars.json (default: the only one)
    parfile
        pars.json (default: the DataSet's, else pars/pars.json beside PROCESSED_DATA)
    rings
        Number of rings used
    cif
        CIF of the phase, for structure factors (default: ``|F|^2 = 1``)
    tth_tol
        2theta tolerance of the rings (deg; default: measured per ring)
    monitor
        Normalise intensities by this counter in each scan's measurement, frame by frame (e.g. fpico6)
    dty_source
        Which dty bins the frames into rows: "dataset" (each row's) or "frames" (each frame's own)
    frame_dty
        Follow dty's motion within the rows (fly and helical scans) in the model
    gridstep
        Voxel = gridstep x dty step; rows are summed in groups of gridstep
    check
        Only resolve and log the setup, write nothing, return ""
    command
        The command line, for the provenance
    log
        Progress messages

    Returns
    -------
    str
        The file written
    """
    import jax.numpy as jnp

    from .data import dty_offsets, histogram_pixels, ring_profile, ring_table

    options = {"analysisroot": analysisroot, "sample": sample, "dataset": dataset, "phase": phase, "parfile": parfile,
               "rings": rings, "cif": cif, "tth_tol": tth_tol, "monitor": monitor, "dty_source": dty_source,
               "frame_dty": frame_dty, "gridstep": gridstep}  # fmt: skip
    d = open_dataset(analysisroot, sample, dataset, phase=phase, parfile=parfile, monitor=monitor,
                     dty_source=dty_source, gridstep=gridstep, log=log)  # fmt: skip
    sc, geom, stream, ds = d["scan"], d["geom"], d["stream"], d["ds"]
    structure = None
    if cif:
        import Dans_Diffraction

        structure = Dans_Diffraction.Crystal(cif)
    ring_t = ring_table(d["lattice"], d["space_group"], d["geo"]["wavelength"], rings, structure)
    n_e, n_o, NK = sc["n_e"], sc["n_o"], sc["n_rows"]
    log(f"{len(ring_t['hkls'])} hkls in {rings} rings at 2theta {', '.join(f'{v:.2f}' for v in ring_t['tth'])}")
    log(f"memory: data {rings * (n_e // R_E) * (n_o // R_O) * NK * 4 / 1e9:.2f} GB (x ~4 in MLEM)")
    if check:
        return ""

    t1 = time.perf_counter()
    groups, chunk = d["groups"], d["chunk"]
    sample_ = sample_groups(groups)
    ring_off, ring_hw = ring_profile(stream(sample_), geom, ring_t["tth"], chunk)
    ring_t["hw"] = ring_hw
    tol = np.abs(ring_off) + ring_hw if tth_tol is None else np.full(rings, tth_tol)
    ring_t["tth_tol"] = tol
    log(f"ring widths from {len(sample_)} groups ({time.perf_counter() - t1:.0f} s): offset / half-width (95%) / "
        "tolerance, deg: " + "; ".join(f"{o:+.3f} / {w:.3f} / {t:.3f}" for o, w, t in zip(ring_off, ring_hw, tol)))  # fmt: skip

    t1 = time.perf_counter()
    H_lit, H = histogram_pixels(  # one pass: the lit histogram (rows summed, fine bins), and the fit data (per row)
        stream(groups), geom, ring_t["tth"], tol, sc["om0"],
        [((B_E, B_O, n_e, n_o), 1), ((B_E * R_E, B_O * R_O, n_e // R_E, n_o // R_O), NK)], chunk,
    )  # fmt: skip
    log(f"histograms: {time.perf_counter() - t1:.0f} s; {H.size / 1e6:.0f}M bins, {float(jnp.mean(H > 0)) * 100:.1f}% "
        "non-empty")  # fmt: skip

    ddty = expo = None
    fdty = d["frame_dty"]
    if frame_dty and fdty is not None:  # fly and helical scans: where the beam really was, per row and omega bin
        ddty, expo = dty_offsets(fdty, ds["omega"], ds["ybinedges"], gridstep, sc["om0"], B_O * R_O, n_o // R_O,
                                 sc["dty0"], sc["ystep"], fdty if dty_source == "frames" else None,
                                 return_exposure=True)  # fmt: skip
        if np.abs(ddty).max() > 1e-3 * sc["ystep"]:
            log(f"dty moves within the rows: up to {np.abs(ddty).max() / sc['ystep']:.2f} dty steps from each row's "
                "dty; the model follows it (frame_dty=False to ignore it)")  # fmt: skip
        else:
            ddty = None
        uneven = np.mean(np.abs(expo[expo > 0] - 1) > 0.01)
        if uneven > 0 or np.any(expo == 0):
            log(f"exposure per (row, omega bin) uneven: {np.mean(expo == 0) * 100:.1f}% of bins without frames, "
                f"{uneven * 100:.1f}% with more or fewer than usual; the model scales by it")  # fmt: skip
        else:
            expo = None

    out = out or f"{sample}_{dataset}_histogram.h5"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    resolved = {"dataset_file": d["dsfile"], "sparsefile": d["sparsefile"], "parfile": d["parfile"],
                "phase": d["phase"], "lattice": list(d["lattice"]), "space_group": d["space_group"],
                "wavelength": d["geo"]["wavelength"], "y0_dataset": ds["y0"], "voxel_size": sc["ystep"],
                "omega_step": sc["omega_step"], "rings_tth_deg": list(ring_t["tth"]), "tth_tol_deg": list(tol),
                "n_hkls": len(ring_t["hkls"]), "spatial_correction": d["spatial_correction"]}  # fmt: skip
    with h5py.File(out, "w") as h:
        write_provenance(h, "histogram", None, options, resolved, command)
        write_attrs(h.create_group("geometry"), d["geo"])
        g = h.create_group("rings")
        for k, v in ring_t.items():
            g[k] = np.asarray(v)
        g = h.create_group("scan")
        write_attrs(g, sc)
        if ddty is not None:
            g["ddty"] = np.asarray(ddty, np.float32)
        if expo is not None:
            g["exposure"] = np.asarray(expo, np.float32)
        h.create_dataset("H", data=np.asarray(H), compression="lzf", chunks=True)
        h.create_dataset("H_lit", data=np.asarray(H_lit), compression="lzf", chunks=True)
    write_results(out, {"bins": int(H.size), "nonempty_fraction": float(jnp.mean(H > 0))})
    log(f"-> {out}")
    return out


# ------------------------------------------------------------------------------------------------------------ 2. search
def stage_search(
    histogram_file: str, out: str | None = None, *, grid: float | None = None, max_chance: float = 0.5,
    lit: float = 1.0, etacut: float = 0.2, command: str | None = None, log: Callable = print,
) -> str:  # fmt: skip
    """Score every orientation of a grid over the fundamental zone by its completeness in the lit map.

    The cut on completeness is the filter's (:func:`stage_filter`), chosen from what this stage writes.

    Parameters
    ----------
    histogram_file
        From :func:`stage_histogram`
    out
        Output file (default: the input's name with ``_search``)
    grid
        Orientation grid step, deg (default: the coarsest of 3, 2.5, 2, 1.5, 1 whose chance completeness is at most
        max_chance)
    max_chance
        Chance completeness allowed by the automatic grid
    lit
        Lit threshold, x the median non-empty bin
    etacut
        Use reflections with ``|sin eta|`` above this (here and in the later stages)
    command, log
        See :func:`stage_histogram`

    Returns
    -------
    str
        The file written
    """
    import anri.crystal

    from .search import choose_grid, completeness_levels, completeness_of

    options = {"grid": grid, "max_chance": max_chance, "lit": lit, "etacut": etacut}
    d = load_histogram(histogram_file)
    lm = lit_map(d["H_lit"], d["scan"], d["n_rings"], lit, etacut)
    step = grid if grid is not None else choose_grid(d["ops"], d["B"], d["rings"], d["geom"], lm, max_chance, log=log)
    U_grid, delta = anri.crystal.orientation.orientation_grid(step, d["ops"])
    t1 = time.perf_counter()
    comp = completeness_of(U_grid, delta, d["B"], d["rings"], d["geom"], lm)
    chance, halfway = completeness_levels(comp)
    log(f"grid {step} deg{'' if grid is not None else ' (auto)'}: {len(U_grid)} orientations, up to {delta:.2f} deg "
        f"from the truth; completeness {time.perf_counter() - t1:.1f} s: median (chance) {chance:.2f}, 99th "
        f"{np.percentile(comp, 99):.2f}, max {comp.max():.2f}, halfway {halfway:.2f}")  # fmt: skip
    out = stage_path(histogram_file, "search", out)
    with h5py.File(out, "w") as h:
        write_provenance(h, "search", histogram_file, options, {"grid_step": step, "grid_auto": grid is None,
                                                                 "grid_worst_case_deg": delta}, command)  # fmt: skip
        h["U_grid"] = np.asarray(U_grid)
        h["completeness"] = np.asarray(comp)
    write_results(out, {"orientations": len(U_grid), "chance_completeness": chance, "completeness_halfway": halfway})
    log(f"-> {out}")
    return out


# ------------------------------------------------------------------------------------------------------------ 3. filter
def stage_filter(
    search_file: str, out: str | None = None, *, prune: str = "likelihood", min_comp: float | None = None,
    min_lr: float = 25.0, keep: int = 100000, censor: float = 0.0, command: str | None = None, log: Callable = print,
) -> str:  # fmt: skip
    """Choose the orientations for the fit.

    Writes the orientations kept, and for prune="likelihood" every orientation fitted with its likelihood ratio, so
    that :func:`select_by_likelihood` can redo the choice for other min_lr and keep without fitting again.

    Parameters
    ----------
    search_file
        From :func:`stage_search`
    out
        Output file (default: the input's name with ``_filter``)
    prune
        "likelihood": every orientation above min_comp, kept by the likelihood ratio of a global orientation fit to
        the row-summed data (intensity-aware); or "completeness": every orientation at or above min_comp
    min_comp
        Completeness needed. Default: for prune="likelihood" the chance level (the grid's median completeness;
        raising it drops decoys, and small grains, whose completeness is lower); for prune="completeness" halfway from
        the chance level to the grid's maximum
    min_lr
        Likelihood ratio needed to keep an orientation (about 5 sigma at 25)
    keep
        At most this many orientations, the most likely (or most complete) first
    censor
        Counts per histogram bin below which an empty bin agrees with the model (0: empty bins are zeros)
    command, log
        See :func:`stage_histogram`

    Returns
    -------
    str
        The file written
    """
    from . import orientations
    from .search import select_by_completeness

    options = {"prune": prune, "min_comp": min_comp, "min_lr": min_lr, "keep": keep, "censor": censor}
    s_prov = read_provenance(search_file)
    hist_file = upstream(search_file, "histogram")
    d = load_histogram(hist_file)
    with h5py.File(search_file, "r") as h:
        U_grid, comp = h["U_grid"][()], h["completeness"][()]
    so, sr = s_prov["options"], s_prov["results"]
    fitted = lr = None
    if prune == "likelihood":  # everything above min_comp, judged by a global orientation fit to the row-summed data
        t1 = time.perf_counter()
        min_comp_used = sr["chance_completeness"] if min_comp is None else min_comp
        fitted = np.flatnonzero(comp > min_comp_used)
        sc = d["scan"]
        data = d["H"].reshape(-1, sc["n_rows"]).sum(1)
        bins_o = (B_E * R_E, B_O * R_O, sc["n_e"] // R_E, sc["n_o"] // R_O, sc["om0"])
        _, lr = orientations.fit(data, U_grid[fitted], d["B"], d["rings"], d["geom"], bins_o, so["etacut"], log=log,
                                 censor=censor)  # fmt: skip
        kept = select_by_likelihood(fitted, lr, min_lr, keep)
        n_above = int(np.sum(lr > min_lr))
        log(f"orientation fit of the {len(fitted)} above chance ({time.perf_counter() - t1:.0f} s): {n_above} with "
            f"likelihood ratio > {min_lr:g}" + (f", the top {keep} kept" if n_above > keep else ", all kept"))  # fmt: skip
    elif prune == "completeness":
        min_comp_used = sr["completeness_halfway"] if min_comp is None else min_comp
        kept = select_by_completeness(comp, min_comp_used, keep)
        log(f"{int(np.sum(comp >= min_comp_used))} orientations at completeness >= {min_comp_used:.2f}; "
            f"{len(kept)} kept")  # fmt: skip
    else:
        msg = f"prune must be 'likelihood' or 'completeness', not {prune!r}"
        raise ValueError(msg)
    out = stage_path(search_file, "filter", out)
    with h5py.File(out, "w") as h:
        write_provenance(h, "filter", search_file, options, {"min_comp_used": min_comp_used}, command)
        h["kept"] = np.asarray(kept)
        h["U"] = np.asarray(U_grid[kept])
        h["completeness"] = np.asarray(comp[kept])
        if lr is not None:
            h["fitted"] = np.asarray(fitted)  # grid indices of every orientation fitted, and their likelihood ratios
            h["likelihood_ratio"] = np.asarray(lr)
    write_results(out, {"orientations_kept": len(kept)})
    log(f"-> {out}")
    return out


# --------------------------------------------------------------------------------------------------------------- 4. fit
def stage_fit(
    filter_file: str, out: str | None = None, *, y0: float | None = None, cand: int = 64, n_iter: int = 10,
    coarse: int = 1, beam: float = 0.0, mask: str | None = "disc", block_gb: float = 1.0, censor: float = 0.0,
    save_model: bool = False, command: str | None = None, log: Callable = print, callback: Callable | None = None,
) -> str:  # fmt: skip
    """Fit each voxel's occupancy of the kept orientations.

    Parameters
    ----------
    filter_file
        From :func:`stage_filter`
    out
        Output file (default: the input's name with ``_fit``)
    y0
        dty where the rotation axis is in the beam (default: the DataSet's)
    cand
        Candidate orientations per voxel
    n_iter
        MLEM iterations, SQUAREM-accelerated (3 MLEM updates each)
    coarse
        First fit voxels this many times larger, then give each voxel the candidates of its coarse neighbourhood
    beam
        FWHM of the beam across dty, as dty: each voxel is spread over the rows by this Gaussian profile integrated
        over the voxel. 0: linearly over the 2 nearest rows (close to a beam of FWHM = the dty step)
    mask
        Fit only the voxels in a sample mask: "disc" (the voxels the beam sees at every omega, :func:`~anri.index.mask.disc_mask`),
        "auto" (in the disc, the Otsu threshold, largest region and convex hull of a quick reconstruction), or a .npy
        file of [voxels, voxels] booleans in reconstruction order; None fits every voxel of the square grid, also those
        the data hardly constrain
    block_gb
        Memory for one block of voxels' system entries (GB)
    censor
        See :func:`stage_filter`
    save_model
        Also save the fitted histogram ("model", as large as the data), e.g. to compare it with the data bin by bin
    command, log
        See :func:`stage_histogram`
    callback
        Called after each MLEM iteration, see :func:`anri.index.occupancy.fit_occupancy` (not recorded in the provenance)

    Returns
    -------
    str
        The file written: occupancies ("f", "cand") of every voxel, the mask, the voxels' positions, and per dty row
        the data and fitted intensity
    """
    import anri.geom
    from anri.maths.utils import fwhm_to_sigma

    from .mask import disc_mask, reconstruct, threshold_mask
    from .occupancy import beam_rows, fit_occupancy
    from .search import predictions

    options = {"y0": y0, "cand": cand, "n_iter": n_iter, "coarse": coarse, "beam": beam, "mask": mask,
               "block_gb": block_gb, "censor": censor, "save_model": save_model}  # fmt: skip
    search_file = upstream(filter_file, "search")
    hist_file = upstream(search_file, "histogram")
    d = load_histogram(hist_file)
    so = read_provenance(search_file)["options"]
    with h5py.File(filter_file, "r") as h:
        U_kept, comp_kept = h["U"][()], h["completeness"][()]
    r = d["prov"]["resolved"]
    Y0 = r.get("y0_dataset") if y0 is None else y0
    if Y0 is None:
        msg = f"{r['dataset_file']} has no y0: give it with y0"
        raise ValueError(msg)
    sc, rings, geom, B = d["scan"], d["rings"], d["geom"], d["B"]
    YSTEP, NR = sc["ystep"], voxel_grid(sc, Y0)
    NV = NR * NR
    log(f"y0 {Y0:.6g}: voxels {NR} x {NR} of {YSTEP:g}; occupancies {NV} voxels x {cand} candidates = "
        f"{NV * cand * 8 / 1e9:.2f} GB; blocks of ~{block_gb:g} GB; {len(U_kept)} orientations")  # fmt: skip

    pred = predictions(U_kept, B, rings, geom, so["etacut"])
    pos = np.asarray(anri.geom.scan.recon_positions(NR, YSTEP), np.float32)
    scan = {"y0": Y0, **{k: sc[k] for k in ("dty0", "ystep", "n_rows", "om0", "k_in_lab", "wedge", "chi")}}
    for k in ("ddty", "exposure"):
        if k in sc:
            scan[k] = sc[k]
    dims = (B_E * R_E, B_O * R_O, sc["n_e"] // R_E, sc["n_o"] // R_O)
    if beam > 0:  # the beam's profile across dty, over voxels of YSTEP; summed rows widen its flat top
        scan.update({"sig_beam": fwhm_to_sigma(beam), "width_beam": sc["width_beam"], "voxel": sc["voxel"]})
        dims = (*dims, beam_rows(scan))
        log(f"beam: FWHM {beam:g} across dty, each voxel spread over {dims[4]} rows")
    sel, rec = np.ones(NV, bool), None
    if mask:
        if mask in ("disc", "auto"):
            sel = disc_mask(sc, Y0, NR).ravel()
            if not sel.any():
                msg = f"y0 {Y0:.6g} is outside the scanned dty range: no voxel is in the beam at every omega"
                raise ValueError(msg)
        if mask == "auto":
            rec = reconstruct(d["H"], d["n_rings"], sc["n_e"] // R_E, sc["n_o"] // R_O, scan, B_O * R_O, NR)
            sel &= threshold_mask(rec).ravel()
        elif mask != "disc":
            sel = np.load(mask).astype(bool).ravel()
            if sel.size != NV:
                msg = f"mask {mask}: {sel.size} voxels, the grid has {NR} x {NR}"
                raise ValueError(msg)
        log(f"mask ({mask}): {sel.sum()} of {NV} voxels fitted")
    f_m, cand_m, model = fit_occupancy(d["H"], pred, rings["ring_j"], pos[sel], scan, dims, cand, n_iter, coarse,
                                       block_gb * 1e9, log=log, return_model=True, censor=censor,
                                       callback=callback)  # fmt: skip
    f = np.zeros((NV, f_m.shape[1]), f_m.dtype)
    cands = np.zeros((NV, cand_m.shape[1]), cand_m.dtype)
    f[sel], cands[sel] = f_m, cand_m
    d_row, m_row = row_intensities(d["H"], model, d["n_rings"], sc, so["etacut"])
    lit_rows = m_row > 0.05 * m_row.max()
    row_ratio = np.where(lit_rows, d_row / np.maximum(m_row, 1e-30), np.nan)
    dev = row_ratio[lit_rows] - 1
    log(f"measured / fitted intensity per dty row ({lit_rows.sum()} rows with intensity): rms {np.sqrt(np.mean(dev**2)):.3f}, "
        f"min {np.nanmin(row_ratio):.3f}, max {np.nanmax(row_ratio):.3f}; row-to-row rms {np.sqrt(np.nanmean(np.diff(row_ratio) ** 2)):.3f}")  # fmt: skip

    out = stage_path(filter_file, "fit", out)
    results = {"f": f, "cand": cands, "U": U_kept, "comp": comp_kept, "pos": pos, "mask": sel, "row_ratio": row_ratio,
               "row_data": d_row, "row_model": m_row}  # fmt: skip
    if rec is not None:
        results["recon"] = rec  # the reconstruction the automatic mask was thresholded from
    if save_model:
        results["model"] = np.asarray(model)
    with h5py.File(out, "w") as h:
        write_provenance(h, "fit", filter_file, options, {"y0": Y0, "voxels": NR, "voxel_size": YSTEP}, command)
        for k, v in results.items():
            h.create_dataset(k, data=np.asarray(v), compression="lzf" if np.asarray(v).size > 1000 else None)
    write_results(out, {"voxels_in_mask": int(sel.sum()), "orientations": len(U_kept),
                        "row_ratio_rms": float(np.sqrt(np.mean(dev**2)))})  # fmt: skip
    log(f"-> {out}")
    return out


def row_intensities(H: np.ndarray, model: np.ndarray, n_rings: int, scan: dict, etacut: float) -> tuple:
    """Return the data's and the model's intensity per dty row, over the eta bins the fit models.

    Reflections at ``|sin eta|`` <= etacut are not predicted, so their data would bias every row's ratio up. A row
    that is consistently off (e.g. flux varying between the rows' scans) makes ring artefacts centred on the rotation
    axis.
    """
    n_e = scan["n_e"] // R_E
    eta_c = -180.0 + (np.arange(n_e) + 0.5) * B_E * R_E
    use_e = np.abs(np.sin(np.radians(eta_c))) > etacut
    d_row = np.asarray(H).reshape(n_rings, n_e, -1, scan["n_rows"]).sum(2)[:, use_e].sum((0, 1))
    m_row = np.asarray(model).reshape(n_rings, n_e, -1, scan["n_rows"]).sum(2)[:, use_e].sum((0, 1))
    return d_row, m_row


# ------------------------------------------------------------------------------------------------------- 5. populations
def stage_populations(
    fit_file: str, out: str | None = None, *, occupied: float = 0.2, min_frac: float = 0.1,
    command: str | None = None, log: Callable = print,
) -> str:  # fmt: skip
    """Group each voxel's occupancy into orientation populations, and write the maps.

    Writes ``<out>`` (populations: fraction, mean orientation, spread, completeness; occupied voxels), and beside it
    ``<stem>_entries.npz`` (every population as anri map entries, for the renderer, with the voxel size, lattice,
    space group and phase name) and, with ImageD11,
    ``<stem>_tmap.h5`` (a TensorMap of the main population, with maps of the number of populations, their fraction,
    spread and completeness).

    Parameters
    ----------
    fit_file
        From :func:`stage_fit`
    out
        Output file (default: the input's name with ``_populations``)
    occupied
        Voxels count as occupied (in the TensorMap and entries) above this x the 99th percentile of the total
        occupancy in the mask (the raw occupancy is always saved)
    min_frac
        Report populations holding at least this fraction of a voxel's occupancy
    command, log
        See :func:`stage_histogram`

    Returns
    -------
    str
        The file written
    """
    import anri.crystal
    import anri.geom
    import anri.io
    from anri.maths.utils import float64

    from . import populations
    from .search import completeness_of

    options = {"occupied": occupied, "min_frac": min_frac}
    search_file = upstream(fit_file, "search")
    d = load_histogram(upstream(search_file, "histogram"))
    s_prov = read_provenance(search_file)
    so, sres = s_prov["options"], s_prov["resolved"]
    fr = read_provenance(fit_file)["resolved"]
    with h5py.File(fit_file, "r") as h:
        f, cands, U_kept, sel = h["f"][()], h["cand"][()], h["U"][()], h["mask"][()]
    NR, YSTEP = fr["voxels"], fr["voxel_size"]
    NV = NR * NR
    step, delta = sres["grid_step"], sres["grid_worst_case_deg"]
    tot = f.sum(1)
    t1 = time.perf_counter()
    frac, U_pop, spread, n_pop = populations.group(f, cands, U_kept, d["ops"], 1.8 * step, p=N_POP)
    occ, present = select_populations(tot, sel, frac, occupied, min_frac)
    comp_pop = np.zeros(frac.shape, np.float32)
    lm = lit_map(d["H_lit"], d["scan"], d["n_rings"], so["lit"], so["etacut"])
    comp_pop[present] = completeness_of(U_pop[present], delta, d["B"], d["rings"], d["geom"], lm)
    n_occ = present.sum(1)
    log(f"{occ.sum()} of {NV} voxels occupied; populations ({time.perf_counter() - t1:.1f} s): per occupied voxel "
        + ", ".join(f"{k}: {np.mean(n_occ[occ] == k) * 100:.1f}%" for k in range(1, N_POP + 1))
        + f"; main fraction median {np.median(frac[occ, 0]):.2f}; spread median {np.median(spread[present]):.2f} deg "
        f"(includes the grid); completeness median {np.median(comp_pop[present]):.2f}")  # fmt: skip

    out = stage_path(fit_file, "populations", out)
    results = {"frac": frac, "U_pop": U_pop, "spread": spread, "n": n_pop, "comp_pop": comp_pop, "occupied": occ,
               "present": present}  # fmt: skip
    with h5py.File(out, "w") as h:
        write_provenance(h, "populations", fit_file, options, {"grid_step": step, "grid_worst_case_deg": delta},
                         command)  # fmt: skip
        for k, v in results.items():
            h.create_dataset(k, data=np.asarray(v), compression="lzf" if np.asarray(v).size > 1000 else None)
    write_results(out, {"voxels_occupied": int(occ.sum())})
    r = d["prov"]["resolved"]
    pos = np.asarray(anri.geom.scan.recon_positions(NR, YSTEP))
    stem = os.path.splitext(out)[0]
    v, q = np.nonzero(present)
    with float64():
        ubi_pop = np.asarray(anri.crystal.lattice.U_B_to_UBI(U_pop[v, q], d["B"]))
        ubi_main = np.asarray(anri.crystal.lattice.U_B_to_UBI(U_pop[:, 0], d["B"]))
    np.savez(f"{stem}_entries.npz", ubi=ubi_pop, pos=pos[v].astype(np.float32), density=tot[v] * frac[v, q],
             sig_rot=np.radians(spread[v, q]), voxel=v, population=q, completeness=comp_pop[v, q], voxel_size=YSTEP,
             lattice=np.asarray(r["lattice"], float), space_group=r["space_group"], phase=r["phase"])  # fmt: skip
    log(f"-> {out}, {stem}_entries.npz")
    try:
        maps = {
            "UBI": np.where(occ[:, None, None], ubi_main, np.nan),
            "phase_ids": np.where(occ, 0, -1),
            "occupancy": tot,
            "n_populations": present.sum(1),
            "fraction": frac[:, 0],
            "spread": spread[:, 0],
            "completeness": comp_pop[:, 0],
        }
        maps = {k: m.reshape(NR, NR, *m.shape[1:]) for k, m in maps.items()}  # reconstruction order
        tmap = anri.io.imaged11.tensormap_from_recon(
            maps, np.asarray(r["lattice"]), r["space_group"], r["phase"], YSTEP
        )
        try:
            tmap.get_ipf_maps()  # ipf_x, ipf_y, ipf_z
        except ImportError:
            log("orix is not installed: no IPF maps")
        _ = tmap.euler  # computed and kept in the maps. No strain: these UBIs are rotations of the nominal lattice.
        tpath = f"{stem}_tmap.h5"
        if os.path.exists(tpath):
            os.remove(tpath)
        tmap.to_h5(tpath)
        tmap.to_paraview(tpath)
        log(f"-> {tpath} (and .xdmf for ParaView)")
    except ImportError:
        log("ImageD11 is not installed: no TensorMap written")
    return out
