"""The indexer as four stages, each writing one HDF5 file that the next one reads, with its provenance.

1. :func:`stage_histogram`: an ImageD11 dataset binned into the fit histogram ``H[ring, eta, omega, row]`` and a
   row-summed lit histogram, with the measured ring widths and the scan resolved from the data.
2. :func:`stage_search`: the completeness of every orientation of a grid over the fundamental zone.
3. :func:`stage_filter`: the orientations kept for the map, by a global likelihood fit or by completeness.
4. :func:`stage_map`: occupancies per voxel, orientation populations, a TensorMap and map entries for the renderer.

Each file records how it was made in a ``provenance`` group: the stage, a unique id, the file it was made from and
that file's id (checked when it is read), the anri version and git commit, the date and command, and as attributes of
``provenance/options``, ``provenance/resolved`` and ``provenance/results`` every option given, every value worked out
from the data, and a summary of the results. :func:`show` prints the whole chain. Variants (e.g. maps with different
options from one filter) are separate files, each pointing back to its input.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator

import h5py
import numpy as np

import anri

STAGES = ("histogram", "search", "filter", "map")
B_E, B_O = 0.5, 0.25  # lit histogram bins (deg): eta, omega
R_E, R_O = 2, 4  # fit histogram bins = lit bins x these (1 x 1 deg)
N_POP = 4  # populations per voxel at most
_LATTICE_KEYS = ("cell__a", "cell__b", "cell__c", "cell_alpha", "cell_beta", "cell_gamma")


# ------------------------------------------------------------------------------------------------------- provenance
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


def _set_attrs(group: h5py.Group, values: dict) -> None:
    """Store a flat dict as attributes; None (which HDF5 cannot hold) is left out, lists become arrays."""
    for k, v in values.items():
        if v is None:
            continue
        group.attrs[k] = np.asarray(v) if isinstance(v, (list, tuple)) else v


def _get_attrs(group: h5py.Group) -> dict:
    """Read attributes back as Python values (str, bool, int, float, lists)."""
    out = {}
    for k, v in group.attrs.items():
        if isinstance(v, bytes):
            v = v.decode()
        out[k] = v.tolist() if hasattr(v, "tolist") else v
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
        One of :data:`STAGES`
    source
        The file this stage read (None for the first stage), whose id is recorded too
    options, resolved
        Every option of this stage as given, and every value worked out from the data
    command
        The command line, if run from one
    """
    file_id = uuid.uuid4().hex
    p = h.create_group("provenance")
    _set_attrs(p, {"stage": stage, "id": file_id, "anri_version": anri.VERSION, **_git_commit(),
                   "date": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
                   "cwd": os.getcwd(), "command": command})  # fmt: skip
    if source is not None:
        p.attrs["source"] = os.path.abspath(source)
        p.attrs["source_id"] = read_provenance(source)["id"]
    _set_attrs(p.create_group("options"), options)
    _set_attrs(p.create_group("resolved"), resolved)
    p.create_group("results")
    return file_id


def write_results(path: str, results: dict) -> None:
    """Add a summary of a stage's results to its provenance (``provenance/results``)."""
    with h5py.File(path, "a") as h:
        _set_attrs(h["provenance/results"], results)


def read_provenance(path: str) -> dict:
    """Read a stage file's provenance: its attributes, with "options", "resolved" and "results" as dicts."""
    with h5py.File(path, "r") as h:
        p = h["provenance"]
        out = _get_attrs(p)
        for g in ("options", "resolved", "results"):
            out[g] = _get_attrs(p[g])
    return out


def _source_path(path: str, prov: dict) -> str:
    """Return the file a stage file was made from: its recorded path, else the same name beside it (moved together)."""
    src = prov["source"]
    if os.path.exists(src):
        return src
    beside = os.path.join(os.path.dirname(os.path.abspath(path)), os.path.basename(src))
    if os.path.exists(beside):
        return beside
    msg = f"{path} was made from {src}, which is not there (nor beside {path})"
    raise FileNotFoundError(msg)


def upstream(path: str, stage: str) -> str:
    """Follow a stage file's sources back to the file of an earlier stage, checking each file is the one recorded.

    Parameters
    ----------
    path
        A stage file
    stage
        The stage wanted: this file's own or an earlier one

    Returns
    -------
    str
        That stage's file
    """
    prov = read_provenance(path)
    while prov["stage"] != stage:
        if "source" not in prov:
            msg = f"{path} ({prov['stage']}) has no {stage} stage upstream"
            raise ValueError(msg)
        src = _source_path(path, prov)
        src_prov = read_provenance(src)
        if src_prov["id"] != prov["source_id"]:
            msg = (f"{src} is not the file {path} was made from (its id is {src_prov['id']}, {path} records "
                   f"{prov['source_id']}): it was remade or replaced since")  # fmt: skip
            raise ValueError(msg)
        path, prov = src, src_prov
    return path


def show(path: str) -> str:
    """Return how a stage file was made, all the way back to the dataset, as text: one section per stage, first first.

    Parameters
    ----------
    path
        Any stage file
    """
    chain = [path]
    prov = read_provenance(path)
    while "source" in prov:
        upstream(chain[-1], STAGES[STAGES.index(prov["stage"]) - 1])  # checks the ids
        chain.append(_source_path(chain[-1], prov))
        prov = read_provenance(chain[-1])
    lines = []
    for p in reversed(chain):
        prov = read_provenance(p)
        lines.append(f"[{prov['stage']}]")
        lines.append(f"file = {os.path.abspath(p)!r}")
        for k, v in prov.items():
            if k not in ("stage", "options", "resolved", "results"):
                lines.append(f"{k} = {v!r}")
        for g in ("options", "resolved", "results"):
            if prov[g]:
                lines.append(f"\n[{prov['stage']}.{g}]")
                lines += [f"{k} = {v!r}" for k, v in prov[g].items()]
        lines.append("")
    return "\n".join(lines)


def _out(source: str, stage: str, out: str | None) -> str:
    """Return a stage's default output: the source's name with its stage replaced, beside it."""
    if out is not None:
        return out
    prev = STAGES[STAGES.index(stage) - 1]
    base = os.path.basename(source)
    stem = base[: -len(f"_{prev}.h5")] if base.endswith(f"_{prev}.h5") else os.path.splitext(base)[0]
    return os.path.join(os.path.dirname(os.path.abspath(source)), f"{stem}_{stage}.h5")


# ----------------------------------------------------------------------------------------- shared setup (one source)
def _geometry(geo: dict, y0: float, ystep: float) -> dict:
    """Build the indexer's geometry dict (float32 JAX scalars and arrays) from ImageD11 parameters."""
    import jax.numpy as jnp

    import anri.io
    from anri.maths.utils import fwhm_to_sigma

    geom = anri.io.geom_from_pars(
        geo, y0, fwhm_to_sigma(geo["wavelength"] * 2e-3), 1.5e-4, 1.5e-4, sig_beam=fwhm_to_sigma(ystep),
        voxel_size=ystep,
    )  # fmt: skip
    return {k: jnp.asarray(v, jnp.float32) if np.issubdtype(np.asarray(v).dtype, np.floating) else v
            for k, v in geom.items()}  # fmt: skip


def _lattice(lattice: np.ndarray, sg: int) -> tuple:
    """(B float32, B float64, Laue-group rotations) of a phase."""
    import anri.crystal

    B64 = anri.crystal.B_matrix(lattice)
    return B64.astype(np.float32), B64, anri.crystal.laue_rotations(anri.crystal.symmetry_matrices(sg), B64)


def _lit(H_lit: np.ndarray, n_rings: int, n_e: int, n_o: int, om0: float, ostep: float, lit: float, etacut: float,
         log: Callable) -> dict:  # fmt: skip
    """Build the lit map for completeness: bins above lit x the median non-empty bin of the row-summed histogram."""
    import jax.numpy as jnp

    from .data import lit_table

    Hs = jnp.asarray(H_lit).reshape(n_rings, n_e, n_o)
    med = float(jnp.median(Hs[Hs > 0]))
    for kk in (0.0, 1.0, 3.0, 10.0):
        log(f"  lit fraction at {kk:g} x median: {float(jnp.mean(Hs > kk * med)) * 100:.1f}%")
    return {"table": lit_table(Hs > lit * med), "om0": om0, "bins": (B_E, B_O, n_e, n_o), "frame_step": ostep,
            "etacut": etacut}  # fmt: skip


def load_histogram(path: str) -> dict:
    """Read a histogram stage file back: setup, rings, scan and both histograms, ready for the later stages."""
    import jax.numpy as jnp

    prov = read_provenance(path)
    r = prov["resolved"]
    with h5py.File(path, "r") as h:
        geo = _get_attrs(h["geometry"])
        rings = {k: h["rings"][k][()] for k in h["rings"]}
        scan = _get_attrs(h["scan"])
        for k in ("ddty", "exposure"):
            if k in h["scan"]:
                scan[k] = jnp.asarray(h["scan"][k][()], jnp.float32)
        H = jnp.asarray(h["H"][()])
        H_lit = jnp.asarray(h["H_lit"][()])
    geom = _geometry(geo, r["y0"], r["voxel_size"])
    scan.update({"k_in_lab": geom["k_in_lab"], "wedge": geom["wedge"], "chi": geom["chi"]})
    B, B64, ops = _lattice(np.asarray(r["lattice"]), r["space_group"])
    return {"prov": prov, "geom": geom, "rings": rings, "scan": scan, "H": H, "H_lit": H_lit, "B": B, "B64": B64,
            "ops": ops}  # fmt: skip


# --------------------------------------------------------------------------------------------------------- 1. histogram
def stage_histogram(
    analysisroot: str, sample: str, dataset: str, out: str | None = None, *, phase: str | None = None,
    parfile: str | None = None, rings: int = 6, cif: str | None = None, tth_tol: float | None = None,
    monitor: str | None = None, y0: float | None = None, dty_source: str = "dataset", frame_dty: bool = True,
    gridstep: int = 1, check: bool = False, command: str | None = None, log: Callable = print,
) -> str:  # fmt: skip
    """Bin an ImageD11 dataset's sparse pixels into the indexer's histograms.

    Paths follow ImageD11's layout: ``{analysisroot}/{sample}/{sample}_{dataset}/{sample}_{dataset}_dataset.h5`` and
    ``_sparse.h5``. The geometry and the phase (lattice and space-group number) come from ``pars.json``: the
    DataSet's parfile, else ``pars/pars.json`` beside ``PROCESSED_DATA``, or ``parfile``. Pixels are corrected for
    detector distortion as ImageD11 corrects its peaks (:func:`anri.io.read_spatial`).

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
    y0
        dty where the rotation axis is in the beam (default: the DataSet's)
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

    import anri.geom
    import anri.io

    from .data import coarsen_scan, dty_offsets, histogram_pixels, ring_profile, ring_table

    options = {"analysisroot": analysisroot, "sample": sample, "dataset": dataset, "phase": phase, "parfile": parfile,
               "rings": rings, "cif": cif, "tth_tol": tth_tol, "monitor": monitor, "y0": y0, "dty_source": dty_source,
               "frame_dty": frame_dty, "gridstep": gridstep}  # fmt: skip
    dsname = f"{sample}_{dataset}"
    dsdir = os.path.join(analysisroot, sample, dsname)
    dsfile = os.path.join(dsdir, f"{dsname}_dataset.h5")
    sparsefile = os.path.join(dsdir, f"{dsname}_sparse.h5")
    ds = anri.io.read_dataset(dsfile)
    if ds["sparsefile"] and os.path.exists(ds["sparsefile"]):  # where the DataSet says, else the standard place
        sparsefile = ds["sparsefile"]
    if y0 is None and ds["y0"] is None:
        msg = f"{dsfile} has no y0: give it with y0"
        raise ValueError(msg)
    Y0 = ds["y0"] if y0 is None else y0
    parfile = parfile or ds["parfile"]
    if not parfile or not os.path.exists(parfile):  # e.g. processed elsewhere: pars/ beside PROCESSED_DATA
        root = os.path.abspath(analysisroot)
        while os.path.basename(root) != "PROCESSED_DATA" and root != os.path.dirname(root):
            root = os.path.dirname(root)
        parfile = os.path.join(os.path.dirname(root), "pars", "pars.json")
    geo, phase, cell = anri.io.read_pars_json(parfile, phase)
    lpars = np.array([cell[k] for k in _LATTICE_KEYS])
    sg = cell["cell_lattice_[P,A,B,C,I,F,R]"]
    if not isinstance(sg, float):
        msg = f"{phase}: cell_lattice_[P,A,B,C,I,F,R] = {sg} is a centring letter; a space-group number is needed"
        raise ValueError(msg)  # noqa: TRY004 (a wrong value in a file, not a wrong argument type)
    sg = int(sg)

    ybin, yedge, oedge = ds["ybincens"], ds["ybinedges"], ds["obinedges"]
    G = gridstep
    ystep0 = float(np.median(np.diff(ybin)))
    # the DataSet's rows summed in groups of G, voxels of one (coarse) row
    rows = coarsen_scan({"dty0": float(ybin[0]), "ystep": ystep0, "n_rows": len(ybin), "voxel": ystep0,
                         "width_beam": 0.0}, G)  # fmt: skip
    YSTEP, DTY0, NK = rows["ystep"], rows["dty0"], rows["n_rows"]
    OM0, OSTEP = float(oedge[0]), float(np.median(np.diff(oedge)))
    N_E, N_O = round(360 / B_E), round(float(oedge[-1] - oedge[0]) / B_O)
    N_O -= N_O % R_O
    WL = geo["wavelength"]
    geom = _geometry(geo, Y0, YSTEP)
    _, _, ops = _lattice(lpars, sg)
    if cif:
        import Dans_Diffraction

        structure = Dans_Diffraction.Crystal(cif)
    else:
        structure = None
    ring_t = ring_table(lpars, sg, WL, rings, structure)
    _, PAD = anri.geom.sino_shift_and_pad(Y0, NK, DTY0, YSTEP)
    NR = NK + PAD  # recon grid NR x NR, centred on the rotation axis, as ImageD11 pads its reconstructions
    n_cells = rings * (N_E // R_E) * (N_O // R_O) * NK

    log(f"dataset {dsfile}; sparse pixels {sparsefile}")
    log(f"pars    {parfile}: phase {phase}, lattice {', '.join(f'{v:g}' for v in lpars)}, space group {sg}, "
        f"{len(ops)} Laue-group rotations")  # fmt: skip
    log(f"geometry: wavelength {WL:.5f}, distance {geo['distance']:g}; y0 {Y0:.6g}; dty {DTY0:.6g} + {NK} x {YSTEP:.6g}"
        f"{'' if G == 1 else f' (rows summed in groups of {G})'}; omega {OM0:.4g} .. {oedge[-1]:.4g} in "
        f"{len(oedge) - 1} frames of {OSTEP:.4g}")  # fmt: skip
    log(f"{len(ring_t['hkls'])} hkls in {rings} rings at 2theta {', '.join(f'{v:.2f}' for v in ring_t['tth'])}; "
        f"voxels {NR} x {NR} ({NK} dty bins + pad {PAD})")  # fmt: skip
    log(f"memory: data {n_cells * 4 / 1e9:.2f} GB (x ~4 in MLEM)")
    if check:
        return ""

    with h5py.File(sparsefile, "r") as h:
        groups = list(h.keys())
        n_max = max(int(h[g]["nnz"][()].sum()) for g in groups)
    chunk = int(min(1 << 24, 1 << max(10, int(np.ceil(np.log2(max(n_max, 1)))))))

    spatial = anri.io.read_spatial(ds)  # the DataSet's detector distortion maps, if it names any
    spatial_src = ds["detectorh5"] or (f"{ds['e2dxfile']}, {ds['e2dyfile']}" if spatial is not None else "")
    log(f"spatial correction: {spatial_src or 'none (the DataSet names no e2dx/e2dy or detector file)'}")
    monitor_ref = None
    if monitor:  # one reference for every scan, as ImageD11 (the mean)
        mon = np.concatenate(list(anri.io.read_monitor(sparsefile, groups, monitor, ds["masterfile"]).values()))
        monitor_ref = float(np.mean(mon))
        log(f"monitor {monitor}: mean {monitor_ref:.4g}, min / max {mon.min() / monitor_ref:.3f} / "
            f"{mon.max() / monitor_ref:.3f} of the mean; intensities normalised to the mean")  # fmt: skip

    have_frames = ds["dty"] is not None and ds["scans"] is not None and ds["omega"] is not None
    if dty_source == "frames" and not have_frames:
        msg = "dty_source='frames' needs the DataSet's dty, omega and scans"
        raise ValueError(msg)
    # each frame's dty (fly, helical scans: the DataSet's row dty plus the motion within the row)
    fdty = anri.io.read_frame_dty(ds, sparsefile) if have_frames else None
    bin_dty = fdty if dty_source == "frames" else ds["dty"]  # what bins the frames into rows

    def stream(groups_: list) -> Iterator[tuple]:  # read the next chunk while this one is binned
        return anri.io.prefetch(anri.io.stream_sparse(
            sparsefile, yedge, ds["omegamotor"], ds["dtymotor"], chunk, groups_, G, bin_dty, ds["scans"], monitor,
            monitor_ref, ds["masterfile"], ds["omega"], spatial,
        ))  # fmt: skip

    t1 = time.perf_counter()
    sample_groups = [
        groups[i] for i in np.unique(np.linspace(0, len(groups) - 1, min(len(groups), 9)).round().astype(int))
    ]
    ring_off, ring_hw = ring_profile(stream(sample_groups), geom, ring_t["tth"], chunk)
    ring_t["hw"] = ring_hw
    tol = np.abs(ring_off) + ring_hw if tth_tol is None else np.full(rings, tth_tol)
    ring_t["tth_tol"] = tol
    log(f"ring widths from {len(sample_groups)} groups ({time.perf_counter() - t1:.0f} s): offset / half-width (95%) / "
        "tolerance, deg: " + "; ".join(f"{o:+.3f} / {w:.3f} / {t:.3f}" for o, w, t in zip(ring_off, ring_hw, tol)))  # fmt: skip

    t1 = time.perf_counter()
    H_lit, H = histogram_pixels(  # one pass: the lit histogram (rows summed, fine bins), and the fit data (per row)
        stream(groups), geom, ring_t["tth"], tol, OM0,
        [((B_E, B_O, N_E, N_O), 1), ((B_E * R_E, B_O * R_O, N_E // R_E, N_O // R_O), NK)], chunk,
    )  # fmt: skip
    log(f"histograms: {time.perf_counter() - t1:.0f} s; {H.size / 1e6:.0f}M bins, {float(jnp.mean(H > 0)) * 100:.1f}% "
        "non-empty")  # fmt: skip

    scan = {"y0": Y0, "dty0": DTY0, "ystep": YSTEP, "n_rows": NK, "om0": OM0, "voxel": rows["voxel"],
            "width_beam": rows["width_beam"]}  # fmt: skip
    ddty = expo = None
    if frame_dty and fdty is not None:  # fly and helical scans: where the beam really was, per row and omega bin
        ddty, expo = dty_offsets(fdty, ds["omega"], yedge, G, OM0, B_O * R_O, N_O // R_O, DTY0, YSTEP,
                                 fdty if dty_source == "frames" else None, return_exposure=True)  # fmt: skip
        if np.abs(ddty).max() > 1e-3 * YSTEP:
            log(f"dty moves within the rows: up to {np.abs(ddty).max() / YSTEP:.2f} dty steps from each row's dty; "
                "the model follows it (frame_dty=False to ignore it)")  # fmt: skip
        else:
            ddty = None
        uneven = np.mean(np.abs(expo[expo > 0] - 1) > 0.01)
        if uneven > 0 or np.any(expo == 0):
            log(f"exposure per (row, omega bin) uneven: {np.mean(expo == 0) * 100:.1f}% of bins without frames, "
                f"{uneven * 100:.1f}% with more or fewer than usual; the model scales by it")  # fmt: skip
        else:
            expo = None

    out = out or f"{dsname}_histogram.h5"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    resolved = {"dataset_file": dsfile, "sparsefile": sparsefile, "parfile": parfile, "phase": phase,
                "lattice": list(lpars), "space_group": sg, "wavelength": WL, "y0": Y0, "voxel_size": YSTEP,
                "voxels": NR, "omega_step": OSTEP, "rings_tth_deg": list(ring_t["tth"]), "tth_tol_deg": list(tol),
                "n_hkls": len(ring_t["hkls"]), "spatial_correction": spatial_src}  # fmt: skip
    with h5py.File(out, "w") as h:
        write_provenance(h, "histogram", None, options, resolved, command)
        _set_attrs(h.create_group("geometry"), geo)
        g = h.create_group("rings")
        for k, v in ring_t.items():
            g[k] = np.asarray(v)
        g = h.create_group("scan")
        _set_attrs(g, {**scan, "n_e": N_E, "n_o": N_O, "omega_step": OSTEP})
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
    lit: float = 1.0, etacut: float = 0.2, min_comp: float | None = None, keep: int = 100000,
    command: str | None = None, log: Callable = print,
) -> str:  # fmt: skip
    """Score every orientation of a grid over the fundamental zone by its completeness in the lit map.

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
    min_comp
        Completeness needed: default halfway between the grid's median (the chance level) and its maximum (for
        prune="completeness"); the likelihood filter fits everything above it, default the chance level
    keep
        At most this many orientations, the most complete first
    command, log
        See :func:`stage_histogram`

    Returns
    -------
    str
        The file written
    """
    import anri.crystal

    from .predict import choose_grid, prune

    options = {"grid": grid, "max_chance": max_chance, "lit": lit, "etacut": etacut, "min_comp": min_comp, "keep": keep}
    d = load_histogram(histogram_file)
    sc = d["scan"]
    lit_map = _lit(d["H_lit"], len(d["rings"]["tth"]), sc["n_e"], sc["n_o"], sc["om0"], sc["omega_step"], lit, etacut,
                   log)  # fmt: skip
    step = grid if grid is not None else choose_grid(d["ops"], d["B"], d["rings"], d["geom"], lit_map, max_chance,
                                                     log=log)  # fmt: skip
    U_grid, delta = anri.crystal.orientation_grid(step, d["ops"])
    t1 = time.perf_counter()
    kept, comp, info = prune(U_grid, delta, d["B"], d["rings"], d["geom"], lit_map, min_comp, keep)
    log(f"grid {step} deg{'' if grid is not None else ' (auto)'}: {len(U_grid)} orientations, up to {delta:.2f} deg "
        f"from the truth; completeness {time.perf_counter() - t1:.1f} s: median (chance) {info['chance']:.2f}, 99th "
        f"{np.percentile(comp, 99):.2f}, max {comp.max():.2f}; {info['n_above']} at >= {info['min_comp']:.2f}")  # fmt: skip
    out = _out(histogram_file, "search", out)
    with h5py.File(out, "w") as h:
        write_provenance(h, "search", histogram_file, options, {"grid_step": step, "grid_auto": grid is None,
                                                                 "grid_worst_case_deg": delta}, command)  # fmt: skip
        h["U_grid"] = np.asarray(U_grid)
        h["completeness"] = np.asarray(comp)
        h["kept"] = np.asarray(kept)  # by completeness: above min_comp, the most complete first, at most keep
    write_results(out, {"orientations": len(U_grid), "chance_completeness": info["chance"],
                        "completeness_cut": info["min_comp"], "n_above": info["n_above"], "capped": info["capped"]})  # fmt: skip
    log(f"-> {out}")
    return out


# ------------------------------------------------------------------------------------------------------------ 3. filter
def stage_filter(
    search_file: str, out: str | None = None, *, prune: str = "likelihood", min_lr: float = 25.0, keep: int = 100000,
    censor: float = 0.0, command: str | None = None, log: Callable = print,
) -> str:  # fmt: skip
    """Choose the orientations for the map.

    Parameters
    ----------
    search_file
        From :func:`stage_search`
    out
        Output file (default: the input's name with ``_filter``)
    prune
        "likelihood": every orientation above the search's min_comp (default: chance), kept by the likelihood ratio of
        a global orientation fit to the row-summed data (intensity-aware); or "completeness": the search's own
        selection
    min_lr
        Likelihood ratio needed to keep an orientation (about 5 sigma at 25)
    keep
        At most this many orientations, the most likely first
    censor
        Counts per histogram bin below which an empty bin agrees with the model (0: empty bins are zeros)
    command, log
        See :func:`stage_histogram`

    Returns
    -------
    str
        The file written
    """
    from .orientations import orientation_mlem

    options = {"prune": prune, "min_lr": min_lr, "keep": keep, "censor": censor}
    s_prov = read_provenance(search_file)
    hist_file = upstream(search_file, "histogram")
    d = load_histogram(hist_file)
    with h5py.File(search_file, "r") as h:
        U_grid, comp, kept_c = h["U_grid"][()], h["completeness"][()], h["kept"][()]
    so, sr = s_prov["options"], s_prov["results"]
    min_comp_used = sr["completeness_cut"]
    lr = None
    if prune == "likelihood":  # everything above chance, judged by a global orientation fit to the row-summed data
        t1 = time.perf_counter()
        min_comp_used = so.get("min_comp", sr["chance_completeness"])
        pre = np.flatnonzero(comp > min_comp_used)
        sc = d["scan"]
        data = d["H"].reshape(-1, sc["n_rows"]).sum(1)
        bins_o = (B_E * R_E, B_O * R_O, sc["n_e"] // R_E, sc["n_o"] // R_O, sc["om0"])
        _, lr_pre = orientation_mlem(data, U_grid[pre], d["B"], d["rings"], d["geom"], bins_o, so["etacut"], log=log,
                                     censor=censor)  # fmt: skip
        above = np.flatnonzero(lr_pre > min_lr)
        order = above[np.argsort(lr_pre[above])[::-1]][:keep]
        kept, lr = pre[order], lr_pre[order]
        log(f"orientation fit of the {len(pre)} above chance ({time.perf_counter() - t1:.0f} s): {len(above)} with "
            f"likelihood ratio > {min_lr:g}" + (f", the top {keep} kept" if len(above) > keep else ", all kept"))  # fmt: skip
    elif prune == "completeness":
        kept = kept_c
    else:
        msg = f"prune must be 'likelihood' or 'completeness', not {prune!r}"
        raise ValueError(msg)
    out = _out(search_file, "filter", out)
    with h5py.File(out, "w") as h:
        write_provenance(h, "filter", search_file, options, {"min_comp_used": min_comp_used}, command)
        h["kept"] = np.asarray(kept)
        h["U"] = np.asarray(U_grid[kept])
        h["completeness"] = np.asarray(comp[kept])
        if lr is not None:
            h["likelihood_ratio"] = np.asarray(lr)
    write_results(out, {"orientations_kept": len(kept)})
    log(f"-> {out}")
    return out


# --------------------------------------------------------------------------------------------------------------- 4. map
def stage_map(
    filter_file: str, out: str | None = None, *, cand: int = 64, n_iter: int = 10, coarse: int = 1, beam: float = 0.0,
    mask: str | None = None, block_gb: float = 1.0, censor: float = 0.0, occupied: float = 0.2, min_frac: float = 0.1,
    command: str | None = None, log: Callable = print,
) -> str:  # fmt: skip
    """Fit each voxel's occupancy of the kept orientations, and group it into orientation populations.

    Writes ``<out>`` (occupancies, populations and fit diagnostics), and beside it ``<stem>_entries.npz`` (every
    population as anri map entries, for the renderer) and, with ImageD11, ``<stem>_tmap.h5`` (a TensorMap of the main
    population, with maps of the number of populations, their fraction, spread and completeness).

    Parameters
    ----------
    filter_file
        From :func:`stage_filter`
    out
        Output file (default: the input's name with ``_map``)
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
        Fit only the voxels in a sample mask: "auto" (Otsu threshold and convex hull of a quick reconstruction), or a
        .npy file of [voxels, voxels] booleans in reconstruction order (default: every voxel)
    block_gb
        Memory for one block of voxels' system entries (GB)
    censor
        See :func:`stage_filter`
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
    from anri.maths.utils import float64, fwhm_to_sigma

    from .mask import reconstruct, threshold_mask
    from .occupancy import beam_rows, fit_occupancy
    from .populations import populations
    from .predict import completeness_of, predictions

    options = {"cand": cand, "n_iter": n_iter, "coarse": coarse, "beam": beam, "mask": mask, "block_gb": block_gb,
               "censor": censor, "occupied": occupied, "min_frac": min_frac}  # fmt: skip
    search_file = upstream(filter_file, "search")
    hist_file = upstream(search_file, "histogram")
    d = load_histogram(hist_file)
    s_prov = read_provenance(search_file)
    so, sres = s_prov["options"], s_prov["resolved"]
    with h5py.File(filter_file, "r") as h:
        U_kept, comp_kept = h["U"][()], h["completeness"][()]
    with h5py.File(search_file, "r") as h:
        n_grid = len(h["completeness"])
    r = d["prov"]["resolved"]
    sc, rings, geom, B = d["scan"], d["rings"], d["geom"], d["B"]
    n_rings, NK, NR, YSTEP, Y0 = len(rings["tth"]), sc["n_rows"], r["voxels"], r["voxel_size"], r["y0"]
    NV = NR * NR
    step, delta = sres["grid_step"], sres["grid_worst_case_deg"]
    log(f"occupancies: {NV} voxels x {cand} candidates = {NV * cand * 8 / 1e9:.2f} GB; blocks of ~{block_gb:g} GB; "
        f"{len(U_kept)} of {n_grid} grid orientations")  # fmt: skip

    pred = predictions(U_kept, B, rings, geom, so["etacut"])
    pos = np.asarray(anri.geom.recon_positions(NR, YSTEP), np.float32)
    scan = {k: sc[k] for k in ("y0", "dty0", "ystep", "n_rows", "om0", "k_in_lab", "wedge", "chi")}
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
        if mask == "auto":
            rec = reconstruct(d["H"], n_rings, sc["n_e"] // R_E, sc["n_o"] // R_O, scan, B_O * R_O, NR)
            sel = threshold_mask(rec).ravel()
        else:
            sel = np.load(mask).astype(bool).ravel()
            if sel.size != NV:
                msg = f"mask {mask}: {sel.size} voxels, the grid has {NR} x {NR}"
                raise ValueError(msg)
        log(f"mask ({mask}): {sel.sum()} of {NV} voxels fitted")
    f_m, cand_m, model = fit_occupancy(d["H"], pred, rings["ring_j"], pos[sel], scan, dims, cand, n_iter, coarse,
                                       block_gb * 1e9, log=log, return_model=True, censor=censor)  # fmt: skip
    f = np.zeros((NV, f_m.shape[1]), f_m.dtype)
    cands = np.zeros((NV, cand_m.shape[1]), cand_m.dtype)
    f[sel], cands[sel] = f_m, cand_m
    # measured / fitted intensity per dty row: a row that is consistently off (e.g. flux varying between the rows'
    # scans) makes ring artefacts centred on the rotation axis. Only the eta bins the fit models: reflections at
    # |sin eta| <= etacut are not predicted, so their data would bias every row's ratio up
    n_e = sc["n_e"] // R_E
    eta_c = -180.0 + (np.arange(n_e) + 0.5) * B_E * R_E
    use_e = np.abs(np.sin(np.radians(eta_c))) > so["etacut"]
    d_row = np.asarray(d["H"]).reshape(n_rings, n_e, -1, NK).sum(2)[:, use_e].sum((0, 1))
    m_row = np.asarray(model).reshape(n_rings, n_e, -1, NK).sum(2)[:, use_e].sum((0, 1))
    lit_rows = m_row > 0.05 * m_row.max()
    row_ratio = np.where(lit_rows, d_row / np.maximum(m_row, 1e-30), np.nan)
    dev = row_ratio[lit_rows] - 1
    log(f"measured / fitted intensity per dty row ({lit_rows.sum()} rows with intensity): rms {np.sqrt(np.mean(dev**2)):.3f}, "
        f"min {np.nanmin(row_ratio):.3f}, max {np.nanmax(row_ratio):.3f}; row-to-row rms {np.sqrt(np.nanmean(np.diff(row_ratio) ** 2)):.3f}")  # fmt: skip

    tot = f.sum(1)
    occ = sel & (tot > occupied * np.percentile(tot[sel], 99))
    t1 = time.perf_counter()
    frac, U_pop, spread, n_pop = populations(f, cands, U_kept, d["ops"], 1.8 * step, p=N_POP)
    present = (frac >= min_frac) & occ[:, None]
    present[:, 0] = occ  # the main population always
    comp_pop = np.zeros(frac.shape, np.float32)
    lit_map = _lit(d["H_lit"], n_rings, sc["n_e"], sc["n_o"], sc["om0"], sc["omega_step"], so["lit"], so["etacut"],
                   lambda m: None)  # fmt: skip
    comp_pop[present] = completeness_of(U_pop[present], delta, B, rings, geom, lit_map)
    n_occ = present.sum(1)
    log(f"{occ.sum()} of {NV} voxels occupied; populations ({time.perf_counter() - t1:.1f} s): per occupied voxel "
        + ", ".join(f"{k}: {np.mean(n_occ[occ] == k) * 100:.1f}%" for k in range(1, N_POP + 1))
        + f"; main fraction median {np.median(frac[occ, 0]):.2f}; spread median {np.median(spread[present]):.2f} deg "
        f"(includes the grid); completeness median {np.median(comp_pop[present]):.2f}")  # fmt: skip

    out = _out(filter_file, "map", out)
    results = {"f": f, "cand": cands, "U": U_kept, "comp": comp_kept, "frac": frac, "U_pop": U_pop, "spread": spread,
               "n": n_pop, "comp_pop": comp_pop, "occupied": occ, "present": present, "pos": pos,
               "row_ratio": row_ratio, "row_data": d_row, "row_model": m_row, "mask": sel}  # fmt: skip
    if rec is not None:
        results["recon"] = rec  # the reconstruction the automatic mask was thresholded from
    with h5py.File(out, "w") as h:
        write_provenance(h, "map", filter_file, options, {"voxels": NR, "voxel_size": YSTEP, "y0": Y0,
                                                          "grid_step": step, "grid_worst_case_deg": delta}, command)  # fmt: skip
        for k, v in results.items():
            h.create_dataset(k, data=np.asarray(v), compression="lzf" if np.asarray(v).size > 1000 else None)
    write_results(out, {"voxels_in_mask": int(sel.sum()), "voxels_occupied": int(occ.sum()),
                        "orientations": len(U_kept)})  # fmt: skip
    stem = os.path.splitext(out)[0]
    v, q = np.nonzero(present)
    with float64():
        ubi_pop = np.asarray(anri.crystal.U_B_to_UBI(U_pop[v, q], B))
        ubi_main = np.asarray(anri.crystal.U_B_to_UBI(U_pop[:, 0], B))
    np.savez(f"{stem}_entries.npz", ubi=ubi_pop, pos=pos[v], density=tot[v] * frac[v, q],
             sig_rot=np.radians(spread[v, q]), voxel=v, population=q, completeness=comp_pop[v, q])  # fmt: skip
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
        tmap = anri.io.tensormap_from_recon(maps, np.asarray(r["lattice"]), r["space_group"], r["phase"], YSTEP)
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
