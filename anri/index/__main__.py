"""Index an ImageD11 scanning-3DXRD dataset from scratch: orientation populations per voxel.

    python -m anri.index <analysisroot> <sample> <dataset> [--phase NAME] [--parfile pars.json] [--check] ...

Paths follow ImageD11's layout: ``{analysisroot}/{sample}/{sample}_{dataset}/{sample}_{dataset}_dataset.h5`` and
``_sparse.h5``. The geometry and the phase (lattice and space-group number) come from ``pars.json``: the DataSet's
parfile, else ``pars/pars.json`` beside ``PROCESSED_DATA``, or ``--parfile``. The scan comes from the DataSet. Lengths
are in the units of the DataSet's dty and the geometry file, which must agree. No spatial distortion correction yet;
F^2 = 1.

Writes ``<tag>.npz`` (occupancies and populations), ``<tag>_entries.npz`` (every population as anri map entries, for
the renderer and :mod:`anri.refine`) and, if ImageD11 is installed, ``<tag>_tmap.h5`` (a TensorMap of the main
population, with maps of the number of populations, their fraction, spread and completeness).
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterator

T0 = time.perf_counter()


def log(msg: str) -> None:
    """Print a timestamped progress message."""
    print(f"[{time.perf_counter() - T0:7.1f} s] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(prog="python -m anri.index", description=__doc__.splitlines()[0])
    p.add_argument("analysisroot")
    p.add_argument("sample")
    p.add_argument("dataset")
    p.add_argument("--phase", help="phase name in pars.json (default: the only one)")
    p.add_argument(
        "--parfile", help="pars.json (default: the DataSet's parfile, else pars/pars.json beside PROCESSED_DATA)"
    )
    p.add_argument("--rings", type=int, default=6, help="number of rings used (default 6)")
    p.add_argument("--grid", type=float, help="orientation grid step, deg (default: the coarsest of 3, 2.5, 2, 1.5, 1 "
                   "whose chance completeness is at most --max-chance)")  # fmt: skip
    p.add_argument(
        "--max-chance", type=float, default=0.5, help="chance completeness allowed by the automatic grid (default 0.5)"
    )
    p.add_argument(
        "--keep", type=int, default=100000, help="at most this many orientations for the occupancy fit (default 100000)"
    )
    p.add_argument("--min-comp", type=float, help="completeness needed: with --prune completeness, default halfway "
                   "between the grid's median (the chance level) and its maximum; with --prune likelihood, default the "
                   "chance level (raising it drops decoys, and small grains, which have lower completeness)")  # fmt: skip
    p.add_argument("--prune", choices=("likelihood", "completeness"), default="likelihood",
                   help="keep orientations by the likelihood ratio of a global orientation fit (default; intensity-aware), "
                   "or by completeness alone")  # fmt: skip
    p.add_argument("--min-lr", type=float, default=25.0, help="likelihood ratio needed to keep an orientation (default 25, "
                   "about 5 sigma)")  # fmt: skip
    p.add_argument("--cif", help="CIF of the phase, for structure factors (default: |F|^2 = 1)")
    p.add_argument("--lit", type=float, default=1.0, help="lit threshold, x the median non-empty bin (default 1)")
    p.add_argument("--etacut", type=float, default=0.2, help="use reflections with |sin eta| above this (default 0.2)")
    p.add_argument("--tth-tol", type=float, help="2theta tolerance of the rings (deg; default: measured per ring)")
    p.add_argument("--iter", type=int, default=10, help="MLEM iterations (default 10)")
    p.add_argument("--cand", type=int, default=64, help="candidate orientations per voxel (default 64)")
    p.add_argument("--coarse", type=int, default=1, help="first fit voxels this many times larger, then give each voxel "
                   "the candidates of its coarse neighbourhood (default 1: off; 4 is ~16x cheaper on large maps)")  # fmt: skip
    p.add_argument("--occupied", type=float, default=0.2, help="voxels count as occupied (in the TensorMap and entries) "
                   "above this x the 99th percentile of the total occupancy (default 0.2; the raw occupancy is always saved)")  # fmt: skip
    p.add_argument("--min-frac", type=float, default=0.1, help="report populations holding at least this fraction of a "
                   "voxel's occupancy (default 0.1)")  # fmt: skip
    p.add_argument(
        "--block-gb", type=float, default=1.0, help="memory for one block of voxels' system entries (default 1 GB)"
    )
    p.add_argument("--y0", type=float, help="dty where the rotation axis is in the beam (default: the DataSet's y0)")
    p.add_argument("--gridstep", type=int, default=1, help="voxel = gridstep x dty step; data rows are summed in groups "
                   "of gridstep to match (default 1)")  # fmt: skip
    p.add_argument("--outdir", default=".")
    p.add_argument("--n-cpu", type=int, default=4, help="CPU devices for JAX (default 4)")
    p.add_argument("--check", action="store_true", help="print the resolved paths and parameters, then stop")
    return p.parse_args()


def main() -> None:
    """Run the indexer on the dataset named on the command line."""
    args = parse_args()
    import anri.utils

    anri.utils.setup(n_cpu=args.n_cpu)
    import jax.numpy as jnp
    import numpy as np

    import anri.crystal
    import anri.geom
    import anri.index as ix
    import anri.io

    B_E, B_O = 0.5, 0.25  # lit-map bins (deg): eta, omega
    R_E, R_O = 2, 4  # histogram bins = lit bins x these (1 x 1 deg)
    N_POP = 4

    # ------------------------------------------------------------------------------------------------- inputs
    dsname = f"{args.sample}_{args.dataset}"
    dsdir = os.path.join(args.analysisroot, args.sample, dsname)
    dsfile = os.path.join(dsdir, f"{dsname}_dataset.h5")
    sparsefile = os.path.join(dsdir, f"{dsname}_sparse.h5")
    ds = anri.io.read_dataset(dsfile)
    if ds["sparsefile"] and os.path.exists(ds["sparsefile"]):  # where the DataSet says, else the standard place
        sparsefile = ds["sparsefile"]
    if args.y0 is None and ds["y0"] is None:
        raise SystemExit(f"{dsfile} has no y0: give it with --y0")
    Y0 = ds["y0"] if args.y0 is None else args.y0
    parfile = args.parfile or ds["parfile"]
    if not parfile or not os.path.exists(parfile):  # e.g. processed elsewhere: pars/ beside PROCESSED_DATA
        root = os.path.abspath(args.analysisroot)
        while os.path.basename(root) != "PROCESSED_DATA" and root != os.path.dirname(root):
            root = os.path.dirname(root)
        parfile = os.path.join(os.path.dirname(root), "pars", "pars.json")
    geo, phase, cell = anri.io.read_pars_json(parfile, args.phase)
    lpars = [cell[k] for k in ("cell__a", "cell__b", "cell__c", "cell_alpha", "cell_beta", "cell_gamma")]
    sg = cell["cell_lattice_[P,A,B,C,I,F,R]"]
    if not isinstance(sg, float):
        raise SystemExit(
            f"{phase}: cell_lattice_[P,A,B,C,I,F,R] = {sg} is a centring letter; a space-group number is needed"
        )
    sg = int(sg)

    ybin, yedge, oedge = ds["ybincens"], ds["ybinedges"], ds["obinedges"]
    G = args.gridstep
    ystep0, nk0 = float(np.median(np.diff(ybin))), len(ybin)
    YSTEP, DTY0, NK = G * ystep0, float(ybin[0]) + 0.5 * (G - 1) * ystep0, -(-nk0 // G)
    OM0, OSTEP = float(oedge[0]), float(np.median(np.diff(oedge)))
    N_E, N_O = round(360 / B_E), round(float(oedge[-1] - oedge[0]) / B_O)
    N_O -= N_O % R_O
    WL = geo["wavelength"]
    geom = anri.io.geom_from_pars(geo, Y0, WL * 2e-3 / 2.355, 1.5e-4, 1.5e-4, sig_beam=YSTEP / 2.355, voxel_size=YSTEP)
    geom = {
        k: jnp.asarray(v, jnp.float32) if np.issubdtype(np.asarray(v).dtype, np.floating) else v
        for k, v in geom.items()
    }

    crystal = anri.crystal.Crystal(
        anri.crystal.UnitCell.from_lpars(jnp.asarray(lpars)), anri.crystal.Symmetry.from_number(sg)
    )
    B = np.asarray(crystal.B, np.float32)
    ops = anri.crystal.laue_rotations(np.asarray(crystal.sym_ops), B)
    structure = anri.crystal.Structure.from_cif(args.cif) if args.cif else None
    rings = ix.ring_table(crystal, WL, args.rings, structure)
    _, PAD = anri.geom.sino_shift_and_pad(Y0, NK, DTY0, YSTEP)
    NR = NK + PAD  # recon grid NR x NR, centred on the rotation axis, as ImageD11 pads its reconstructions
    NV = NR * NR
    n_cells = args.rings * (N_E // R_E) * (N_O // R_O) * NK

    log(f"dataset {dsfile}; sparse pixels {sparsefile}")
    log(f"pars    {parfile}: phase {phase}, lattice {', '.join(f'{v:g}' for v in lpars)}, space group {sg} "
        f"({crystal.sgname}), {len(ops)} Laue-group rotations")  # fmt: skip
    log(f"geometry: wavelength {WL:.5f}, distance {geo['distance']:g}; y0 {Y0:.6g}; dty {DTY0:.6g} + {NK} x {YSTEP:.6g}"
        f"{'' if G == 1 else f' (rows summed in groups of {G})'}; omega {OM0:.4g} .. {oedge[-1]:.4g} in "
        f"{len(oedge) - 1} frames of {OSTEP:.4g}")  # fmt: skip
    log(f"{len(rings['hkls'])} hkls in {args.rings} rings at 2theta {', '.join(f'{v:.2f}' for v in rings['tth'])}; "
        f"voxels {NR} x {NR} ({NK} dty bins + pad {PAD})")  # fmt: skip
    log(f"memory: data {n_cells * 4 / 1e9:.2f} GB (x ~4 in MLEM); occupancies {NV} voxels x {args.cand} candidates = "
        f"{NV * args.cand * 8 / 1e9:.2f} GB; blocks of ~{args.block_gb:g} GB")  # fmt: skip
    if args.check:
        return

    # ------------------------------------------------------------------------------------------------- 1. data
    with __import__("h5py").File(sparsefile, "r") as h:
        groups = list(h.keys())
        n_max = max(int(h[g]["nnz"][()].sum()) for g in groups)
    chunk = int(min(1 << 24, 1 << max(10, int(np.ceil(np.log2(max(n_max, 1)))))))

    def stream(groups_: list) -> Iterator[tuple]:  # read the next chunk while this one is binned
        return anri.io.prefetch(
            anri.io.stream_sparse(
                sparsefile, yedge, ds["omegamotor"], ds["dtymotor"], chunk, groups_, G, ds["dty"], ds["scans"]
            )
        )

    t1 = time.perf_counter()
    sample = [groups[i] for i in np.unique(np.linspace(0, len(groups) - 1, min(len(groups), 9)).round().astype(int))]
    ring_off, ring_hw = ix.ring_profile(stream(sample), geom, rings["tth"], chunk)
    rings["hw"] = ring_hw
    tth_tol = np.abs(ring_off) + ring_hw if args.tth_tol is None else np.full(args.rings, args.tth_tol)
    log(f"ring widths from {len(sample)} groups ({time.perf_counter() - t1:.0f} s): offset / half-width (95%) / tolerance, "
        "deg: " + "; ".join(f"{o:+.3f} / {w:.3f} / {t:.3f}" for o, w, t in zip(ring_off, ring_hw, tth_tol)))  # fmt: skip

    t1 = time.perf_counter()
    H_lit, H = ix.histogram_pixels(  # one pass: the lit map (rows summed, fine bins), and the fit data (per row)
        stream(groups), geom, rings["tth"], tth_tol, OM0,
        [((B_E, B_O, N_E, N_O), 1), ((B_E * R_E, B_O * R_O, N_E // R_E, N_O // R_O), NK)], chunk,
    )  # fmt: skip
    log(
        f"histograms: {time.perf_counter() - t1:.0f} s; {H.size / 1e6:.0f}M bins, {float(jnp.mean(H > 0)) * 100:.1f}% non-empty"
    )

    # ------------------------------------------------------------------------------------------------- 2. pruning
    Hs = H_lit.reshape(args.rings, N_E, N_O)
    med = float(jnp.median(Hs[Hs > 0]))
    for kk in (0.0, 1.0, 3.0, 10.0):
        log(f"  lit fraction at {kk:g} x median: {float(jnp.mean(Hs > kk * med)) * 100:.1f}%")
    lit = {"table": ix.lit_table(Hs > args.lit * med), "om0": OM0, "bins": (B_E, B_O, N_E, N_O), "frame_step": OSTEP,
           "etacut": args.etacut}  # fmt: skip
    step = args.grid if args.grid is not None else ix.choose_grid(ops, B, rings, geom, lit, args.max_chance, log=log)
    U_grid, delta = anri.crystal.orientation_grid(step, ops)
    t1 = time.perf_counter()
    kept, comp, info = ix.prune(U_grid, delta, B, rings, geom, lit, args.min_comp, args.keep)
    log(f"grid {step} deg{'' if args.grid is not None else ' (auto)'}: {len(U_grid)} orientations, up to {delta:.2f} deg "
        f"from the truth; completeness {time.perf_counter() - t1:.1f} s: median (chance) {info['chance']:.2f}, 99th "
        f"{np.percentile(comp, 99):.2f}, max {comp.max():.2f}; {info['n_above']} at >= {info['min_comp']:.2f}"
        + ("" if args.prune == "likelihood" else f", the top {args.keep} kept (--keep)" if info["capped"] else ", all kept"))  # fmt: skip
    if args.prune == "likelihood":  # everything above chance, judged by a global orientation fit to the row-summed data
        t1 = time.perf_counter()
        pre = np.flatnonzero(comp > (info["chance"] if args.min_comp is None else args.min_comp))
        d = H.reshape(-1, NK).sum(1)
        bins_o = (B_E * R_E, B_O * R_O, N_E // R_E, N_O // R_O, OM0)
        _, lr = ix.orientation_mlem(d, U_grid[pre], B, rings, geom, bins_o, args.etacut, log=log)
        above = np.flatnonzero(lr > args.min_lr)
        kept = pre[above[np.argsort(lr[above])[::-1]][: args.keep]]
        log(f"orientation fit of the {len(pre)} above chance ({time.perf_counter() - t1:.0f} s): {len(above)} with "
            f"likelihood ratio > {args.min_lr:g}" + (f", the top {args.keep} kept (--keep)" if len(above) > args.keep else ", all kept"))  # fmt: skip
    U_kept = U_grid[kept]

    # ------------------------------------------------------------------------------------------------- 3. occupancy
    pred = ix.predictions(U_kept, B, rings, geom, args.etacut)
    pos = np.asarray(anri.geom.recon_positions(NR, YSTEP), np.float32)
    scan = {"y0": Y0, "dty0": DTY0, "ystep": YSTEP, "n_rows": NK, "om0": OM0}
    dims = (B_E * R_E, B_O * R_O, N_E // R_E, N_O // R_O)
    f, cand, model = ix.fit_occupancy(H, pred, rings["ring_j"], pos, scan, dims, args.cand, args.iter, args.coarse,
                                     args.block_gb * 1e9, log=log, return_model=True)  # fmt: skip
    # measured / fitted intensity per dty row: a row that is consistently off (e.g. flux varying between the rows'
    # scans) makes ring artefacts centred on the rotation axis
    d_row, m_row = np.asarray(H).reshape(-1, NK).sum(0), model.reshape(-1, NK).sum(0)
    lit_rows = m_row > 0.05 * m_row.max()
    row_ratio = np.where(lit_rows, d_row / np.maximum(m_row, 1e-30), np.nan)
    dev = row_ratio[lit_rows] - 1
    log(f"measured / fitted intensity per dty row ({lit_rows.sum()} rows with intensity): rms {np.sqrt(np.mean(dev**2)):.3f}, "
        f"min {np.nanmin(row_ratio):.3f}, max {np.nanmax(row_ratio):.3f}; row-to-row rms {np.sqrt(np.nanmean(np.diff(row_ratio) ** 2)):.3f}")  # fmt: skip

    # ------------------------------------------------------------------------------------------------- 4. populations
    tot = f.sum(1)
    occupied = tot > args.occupied * np.percentile(tot, 99)
    t1 = time.perf_counter()
    frac, U_pop, spread, n_pop = ix.populations(f, cand, U_kept, ops, 1.8 * step, p=N_POP)
    present = (frac >= args.min_frac) & occupied[:, None]
    present[:, 0] = occupied  # the main population always
    comp_pop = np.zeros(frac.shape, np.float32)
    comp_pop[present] = ix.completeness_of(U_pop[present], delta, B, rings, geom, lit)
    n_occ = present.sum(1)
    log(f"{occupied.sum()} of {NV} voxels occupied; populations ({time.perf_counter() - t1:.1f} s): per occupied voxel "
        + ", ".join(f"{k}: {np.mean(n_occ[occupied] == k) * 100:.1f}%" for k in range(1, N_POP + 1))
        + f"; main fraction median {np.median(frac[occupied, 0]):.2f}; spread median {np.median(spread[present]):.2f} deg "
        f"(includes the grid); completeness median {np.median(comp_pop[present]):.2f}")  # fmt: skip

    # ------------------------------------------------------------------------------------------------- outputs
    os.makedirs(args.outdir, exist_ok=True)
    tag = os.path.join(args.outdir, f"{dsname}_index")
    np.savez(f"{tag}.npz", f=f, cand=cand, U=U_kept, comp=comp[kept], frac=frac, U_pop=U_pop, spread=spread, n=n_pop,
             comp_pop=comp_pop, occupied=occupied, present=present, pos=pos, grid_step=step, delta=delta,
             row_ratio=row_ratio, row_data=d_row, row_model=m_row)  # fmt: skip
    v, q = np.nonzero(present)
    np.savez(f"{tag}_entries.npz", ubi=np.linalg.inv(U_pop[v, q] @ B), pos=pos[v], density=tot[v] * frac[v, q],
             sig_rot=np.radians(spread[v, q]), voxel=v, population=q, completeness=comp_pop[v, q])  # fmt: skip
    try:
        maps = {
            "UBI": np.where(occupied[:, None, None], np.linalg.inv(U_pop[:, 0] @ B), np.nan).reshape(NR, NR, 3, 3),
            "phase_ids": np.where(occupied, 0, -1).reshape(NR, NR),
            "occupancy": tot.reshape(NR, NR),
            "n_populations": n_occ.reshape(NR, NR),
            "fraction": frac[:, 0].reshape(NR, NR),
            "spread": spread[:, 0].reshape(NR, NR),
            "completeness": comp_pop[:, 0].reshape(NR, NR),
        }
        tmap = anri.io.tensormap_from_recon(maps, np.asarray(lpars), sg, phase, YSTEP)
        try:
            tmap.get_ipf_maps()  # ipf_x, ipf_y, ipf_z
        except ImportError:
            log("orix is not installed: no IPF maps")
        _ = tmap.euler  # computed and kept in the maps. No strain: these UBIs are rotations of the nominal lattice.
        out = f"{tag}_tmap.h5"
        if os.path.exists(out):
            os.remove(out)
        tmap.to_h5(out)
        tmap.to_paraview(out)
        log(f"-> {out} (and .xdmf for ParaView)")
    except ImportError:
        log("ImageD11 is not installed: no TensorMap written")
    log(f"-> {tag}.npz, {tag}_entries.npz")


if __name__ == "__main__":
    main()
