"""Proofs for docs/source/theory/indexing.rst: the bilinear sharing of a prediction over (eta, omega) bins."""

import jax.numpy as jnp
import numpy as np
import sympy as sp

from tests.theory.common import (
    EPS,
    agree,
    beam_frame,
    cayley,
    ev,
    evaluate,
    is_zero,
    rng,
)


def test_bilinear_keeps_weight_and_position():
    """idx-bilinear: with x = (a - a_0) / b - 1/2 and t = x - floor(x), the weights 1 - t, t of bins floor(x) and
    floor(x) + 1 (centres a_0 + (i + 1/2) b) sum to 1 and their weighted centre is a: the shared prediction keeps its
    weight and its position."""
    a, a0, b, i0 = sp.symbols("a a_0 b i_0", real=True)
    t = (a - a0) / b - sp.Rational(1, 2) - i0
    centre = [a0 + (i0 + k + sp.Rational(1, 2)) * b for k in (0, 1)]
    assert is_zero((1 - t) + t - 1)
    assert is_zero((1 - t) * centre[0] + t * centre[1] - a)


def test_bilinear_code():
    """anri's eta_omega_corners are idx-bilinear's bins and weights."""
    from anri.index.data import eta_omega_corners

    a, a0, b = sp.symbols("a a_0 b", real=True)
    x = (a - a0) / b - sp.Rational(1, 2)
    eta, om = np.array([37.3, -151.85]), np.array([12.6, 301.2])
    b_e, b_o, n_e, n_o, om0 = 1.0, 0.5, 360, 720, 0.0
    te = [evaluate(x - sp.floor(x), {a: e, a0: -180, b: b_e}) for e in eta]
    to = [evaluate(x - sp.floor(x), {a: o, a0: om0, b: b_o}) for o in om]
    want = []
    for ke in (0, 1):
        for ko in (0, 1):
            want.append([[t if ke else 1 - t for t in te], [t if ko else 1 - t for t in to]])
    # x is ((a - a_0) / b - 1/2): ~3 roundings of numbers up to 720, so t is within 4 ulps of 720, absolutely
    tol = 4 * EPS * 720

    def weights(*z: object) -> np.ndarray:  # [4 corners, eta / omega, predictions]
        return np.array([[np.asarray(c[2]), np.asarray(c[3])] for c in eta_omega_corners(*z)])

    agree(np.array(want, float), weights, eta, om, om0, b_e, b_o, n_e, n_o, rtol=0, atol=tol)


def test_grid_covers_the_scan():
    """idx-grid: the axis is s = floor(n/2) - (y0 - y_first) / step rows from the middle of the n scanned rows, so the
    first row is floor(n/2) - s steps from it and the last n - 1 - floor(n/2) + s. A grid of N >= n + 2|s| + 1 voxels
    centred on the axis reaches (N - 1) / 2 >= floor(n/2) + |s| steps either way: past both, for even and odd n."""
    m = sp.symbols("m", integer=True, nonnegative=True)
    s, A = sp.symbols("s A", real=True)  # A = |s| >= s, -s
    for n in (2 * m, 2 * m + 1):
        half = (n + 2 * A + 1 - 1) / 2
        first, last = m - s, n - 1 - m + s  # floor(n / 2) = m for both
        assert is_zero(sp.expand(half - first - (A + s + (n - 2 * m) / 2)))  # >= 0 as A >= -s
        assert is_zero(sp.expand(half - last - (A - s + 1 - (n - 2 * m) / 2)))  # >= 0 as A >= s


def test_grid_code():
    """recon_positions and voxel_grid are idx-grid, exactly (integers, and multiples of the step)."""
    from anri.geom.scan import recon_positions
    from anri.index.mask import voxel_grid

    for n_rows, dty0, ystep, y0, pad in (
        (11, -5.0, 1.0, 0.0, 0.0),
        (10, -3.0, 0.5, 0.3, 0.0),
        (11, -5.0, 1.0, 2.2, 1.5),
    ):
        shift = sp.floor(sp.Rational(n_rows, 2)) - (sp.nsimplify(y0) - sp.nsimplify(dty0)) / sp.nsimplify(ystep)
        p = int(sp.ceiling(2 * abs(shift))) + 1
        p += (n_rows + p + 1) % 2
        want = n_rows + p + 2 * int(sp.ceiling(sp.nsimplify(pad) / sp.nsimplify(ystep)))
        assert voxel_grid({"n_rows": n_rows, "dty0": dty0, "ystep": ystep}, y0, pad) == want
    n, step = 5, 0.75
    want = np.array([[(i - n // 2) * step, (n // 2 - j) * step, 0.0] for i in range(n) for j in range(n)])
    agree(want, recon_positions, n, step, rtol=0, atol=0)  # small integers times step: exact


def test_pixel_angles_code():
    """pixel_angles: (2 theta, eta) of a pixel's centre d0 + s S + f F seen from the lab origin (frm-eta), and its
    omega; a tilted beam and detector."""
    from anri.index.data import pixel_angles
    from anri.maths.utils import float64

    psi, tilt = 0.05, -0.02
    k, e_h, e_v = beam_frame(sp.Float(psi, 30), sp.Float(tilt, 30))
    S, F, d0 = np.array([0.01, 0.2, -75.0]), np.array([0.3, -75.0, 0.1]), np.array([150e3, 80e3, 70e3])
    sl, fa, om = np.array([10.5, 1500.2]), np.array([2000.7, 40.1]), np.array([12.0, -33.0])
    geom = {"det_origin_lab": d0, "s_step_lab": S, "f_step_lab": F, "k_in_lab": 3 * ev(k, {})}
    want = []
    for a, b, o in zip(sl, fa, om):
        p = sp.Matrix([sp.Float(x, 30) for x in d0 + a * S + b * F])  # the pixel's position rounds once in the code
        u = p / sp.sqrt(p.dot(p))
        want.append(
            [float(sp.N(sp.deg(sp.acos(u.dot(k))), 30)), float(sp.N(sp.deg(sp.atan2(-u.dot(e_h), u.dot(e_v))), 30)), o]
        )
    with float64():
        got = np.asarray(
            pixel_angles(
                jnp.asarray(sl), jnp.asarray(fa), jnp.asarray(om), {k_: jnp.asarray(v) for k_, v in geom.items()}
            )
        )
    # the position: 16 ulps of |d0|, an angle of 16 eps |d0| / |p| <= 16 eps radians; then test_eta_code's arccos
    # (condition 1 / sin 2theta) and atan2 (1), in degrees
    tth = np.radians(np.array(want)[:, 0])
    t = 16 * EPS * 180 / np.pi * (2 + 1 / np.abs(np.sin(tth)))
    np.testing.assert_allclose(got[:, 0], np.array(want)[:, 0], rtol=0, atol=t.max())
    np.testing.assert_allclose(got[:, 1], np.array(want)[:, 1], rtol=0, atol=t.max())
    np.testing.assert_array_equal(got[:, 2], om)


def test_histogram_code():
    """idx-histogram: a pixel goes to its nearest ring if within that ring's tolerance, to eta bin
    floor((eta + 180) / b_e) mod n_e and omega bin floor((omega - om0) / b_o) if on the scan, and its row; the bins sum
    their pixels' values. Exactly (small integers)."""
    from anri.index.data import histogram

    ring_tth, tol_ = np.array([5.0, 6.0, 8.0]), np.array([0.1, 0.2, 0.1])
    n_e, n_o, n_k, b_e, b_o, om0 = 8, 6, 3, 45.0, 2.0, 10.0
    g = rng(34)
    x = np.stack([g.choice([4.95, 5.05, 5.3, 5.85, 6.15, 7.0, 8.05, 9.0], 40), g.uniform(-179, 179, 40),
                  g.uniform(9, 23, 40)], 1)  # fmt: skip
    val = g.integers(1, 9, 40).astype(float)
    row = g.integers(-1, 3, 40)
    want = np.zeros((3, n_e, n_o, n_k))
    for (t, e, o), v, k in zip(x, val, row):
        r = int(np.argmin(np.abs(ring_tth - t)))
        ie, io = int(np.floor((e + 180) / b_e)) % n_e, int(np.floor((o - om0) / b_o))
        if abs(t - ring_tth[r]) < tol_[r] and 0 <= io < n_o and 0 <= k < n_k:
            want[r, ie, io, k] += v
    got = histogram(jnp.asarray(x), jnp.asarray(val), jnp.asarray(row, jnp.int32), jnp.asarray(ring_tth),
                    jnp.asarray(tol_), om0, b_e, b_o, 3, n_e, n_o, n_k)  # fmt: skip
    np.testing.assert_array_equal(np.asarray(got), want.ravel())


def test_dty_offsets_code():
    """dty_offsets: per (row, omega bin), the mean over its frames of the frame's dty less the row's nominal dty
    (rows from each frame's own dty, binned by the edges); bins without frames 0; the exposure each bin's frames over
    the median bin's. Exactly (multiples of 1/8)."""
    from anri.index.data import dty_offsets

    g = rng(35)
    n_rows, n_fr = 4, 12
    om = np.tile((np.arange(n_fr) + 0.5) * 30.0, (n_rows, 1))
    dty = np.arange(n_rows)[:, None] + g.integers(-3, 4, (n_rows, n_fr)) / 8
    edges = np.arange(n_rows + 1) - 0.5
    off, ex = dty_offsets(dty, om, edges, 1, 0.0, 60.0, 6, 0.0, 1.0, bin_dty=dty, return_exposure=True)
    s, c = np.zeros((n_rows, 6)), np.zeros((n_rows, 6))
    for r in range(n_rows):
        for f in range(n_fr):
            k = int(np.searchsorted(edges, dty[r, f], side="right") - 1)
            o = int(np.floor(om[r, f] / 60.0))
            if 0 <= k < n_rows:
                s[k, o] += dty[r, f] - k
                c[k, o] += 1
    np.testing.assert_array_equal(off, np.where(c > 0, s / np.maximum(c, 1), 0.0))
    np.testing.assert_array_equal(ex, c / np.median(c[c > 0]))


def _system_case() -> tuple:
    """One voxel off the axis, one orientation, three predictions, a beam along x, the histogram's bins."""
    eta = np.asarray([[[37.3, -120.6, 151.2]]])  # NumPy: a JAX array made outside float64() would be float32
    om = np.asarray([[[12.6, 101.3, 260.9]]])
    use = np.ones((1, 1, 3), bool)
    w = np.asarray([[[1.3, 0.7, 2.1]]])
    ring_j = np.asarray([0, 1, 1], np.int32)
    pos = np.asarray([[3.3, -2.1, 0.0]])
    scan = {"y0": 0.4, "dty0": -6.0, "ystep": 1.0, "n_rows": 13, "om0": 0.0}
    return eta, om, use, w, ring_j, pos, scan, (1.0, 1.0, 360, 360)


def _system_expected(eta, om, w, ring_j, pos, scan, dims, rows) -> dict:
    """idx-weight's sharing: idx-bilinear over (eta, omega) bins, and the rows' weights rows(fk) at fk = (dty -
    dty0) / ystep, dty from dif-dty (beam along x, no tilts: y0 - (x sin omega + y cos omega))."""
    b_e, b_o, n_e, n_o = dims
    out = {}
    for j in range(eta.shape[-1]):
        e, o, wj = (float(np.asarray(a)[0, 0, j]) for a in (eta, om, w))
        x, y = float(pos[0, 0]), float(pos[0, 1])
        dty = sp.Float(scan["y0"], 30) - (x * sp.sin(sp.rad(sp.Float(o, 30))) + y * sp.cos(sp.rad(sp.Float(o, 30))))
        fk = (dty - scan["dty0"]) / scan["ystep"]
        xe, xo = (sp.Float(e, 30) + 180) / b_e - sp.Rational(1, 2), sp.Float(o, 30) / b_o - sp.Rational(1, 2)
        for ie, we in ((sp.floor(xe), 1 - (xe - sp.floor(xe))), (sp.floor(xe) + 1, xe - sp.floor(xe))):
            for io, wo in ((sp.floor(xo), 1 - (xo - sp.floor(xo))), (sp.floor(xo) + 1, xo - sp.floor(xo))):
                for k, wk in rows(fk, o):
                    if 0 <= k < scan["n_rows"] and 0 <= io < n_o:
                        cell = ((int(ring_j[j]) * n_e + int(ie) % n_e) * n_o + int(io)) * scan["n_rows"] + k
                        out[cell] = out.get(cell, 0) + wj * we * wo * wk
    return out


def test_system_code():
    """system: each prediction's weight shared bilinearly over (eta, omega) bins and over rows, linearly between the
    two nearest (2-row model) or by the beam profile integrated over the voxel's chord in each row, ystep / v^2
    W((k - fk) ystep, omega) (int-chord; integrates to 1 over rows); against _system_expected, in float64."""
    from anri.index.occupancy import beam_rows, system
    from anri.maths.utils import float64
    from tests.theory.test_intensity import _chord_expr, _chord_tol

    eta, om, use, w, ring_j, pos, scan, dims = _system_case()
    two = lambda fk, o: ((int(sp.floor(fk)), 1 - (fk - sp.floor(fk))), (int(sp.floor(fk)) + 1, fk - sp.floor(fk)))
    prof = {**scan, "sig_beam": 0.6, "width_beam": 0.0, "voxel": 1.0}
    nb = beam_rows(prof)

    def beam(fk: sp.Expr, o: float) -> list:
        k0 = int(sp.floor(fk + sp.Rational(1, 2))) - nb // 2
        return [(k0 + d, prof["ystep"] / prof["voxel"] ** 2
                 * _chord_expr((k0 + d - fk) * prof["ystep"], o, sp.Integer(1), sp.Float(0.6, 30), 0.0)) for d in range(nb)]  # fmt: skip

    for sc, rows, n_beam in ((scan, two, 0), (prof, beam, nb)):
        want = _system_expected(eta, om, w, ring_j, pos, sc, dims, rows)
        with float64():
            idx, wt = (np.asarray(a).ravel() for a in system(*(jnp.asarray(a) for a in (eta, om, use, w, ring_j, pos)),
                                                              sc, *dims, n_beam))  # fmt: skip
        got = {}
        for i, v in zip(idx, wt):
            if i >= 0:
                got[int(i)] = got.get(int(i), 0.0) + float(v)
        assert set(got) == {c for c, v in want.items() if float(v) != 0} | (set(got) & set(want))
        for c, v in want.items():
            # dty and the bins' fractions round over ~10 operations of values <= 400 (16 ulps of that each, moving a
            # weight by at most its slope: 1 per bin, max p = phi(0) / sigma per row); the chord weight its own
            # rounding; the products a few ulps
            t = 16 * EPS * 400 * (3 + 1 / 0.6) * max(float(w.max()), 1) + (
                _chord_tol(0.5, 30.0, 1.0, 0.6, 0.0) if n_beam else 0
            )
            np.testing.assert_allclose(got.get(c, 0.0), float(v), rtol=0, atol=t)


def test_forward_backward_code():
    """idx-linear: forward is A f (each cell the sum of its entries' weight times the voxel's occupancy of that
    orientation), backward is A^T r (each (voxel, candidate) the sum of its entries' weight times r at their cells),
    A's entries from system; so backward is forward's transpose. In float64, against sums of the entries."""
    from anri.index.occupancy import backward, forward, system
    from anri.maths.utils import float64

    eta, om, use, w, ring_j, _, scan, dims = _system_case()
    pred = (np.concatenate([eta, eta + 40]), np.concatenate([om, om + 25]), np.concatenate([use, use]),
            np.concatenate([w, 2 * w]))  # two orientations, [Q = 2, Nj]  # fmt: skip
    vox = np.asarray([[3.3, -2.1, 0.0], [-1.0, 0.5, 0.0]])
    cand = np.asarray([[0, 1], [1, 0]], np.int32)
    f = np.asarray([[0.6, 1.4], [2.0, 0.25]])
    n_cells = 2 * 360 * 360 * 13
    r = rng(36).random(n_cells)
    with float64():
        pred = tuple(jnp.asarray(p[:, 0]) for p in pred)
        Af = np.asarray(forward(jnp.asarray(f), jnp.asarray(cand), pred, jnp.asarray(ring_j), jnp.asarray(vox), scan,
                                dims, n_cells, 1))  # fmt: skip
        ATr = np.asarray(backward(jnp.asarray(r), jnp.asarray(cand), pred, jnp.asarray(ring_j), jnp.asarray(vox), scan,
                                  dims, 1))  # fmt: skip
        A = [[system(*(p[cand[v, q]][None, None] for p in pred), jnp.asarray(ring_j), jnp.asarray(vox[v : v + 1]), scan, *dims)
              for q in range(2)] for v in range(2)]  # fmt: skip
    want_f, want_b = np.zeros(n_cells), np.zeros((2, 2))
    rv = r
    for v in range(2):
        for q in range(2):
            idx, wt = (np.asarray(a).ravel() for a in A[v][q])
            ok = idx >= 0
            np.add.at(want_f, idx[ok], float(f[v, q]) * wt[ok])
            want_b[v, q] = np.sum(wt[ok] * rv[idx[ok]])
    # sums of at most 2 x 3 x 8 entries each: that many ulps of their magnitudes
    np.testing.assert_allclose(Af, want_f, rtol=0, atol=48 * EPS * np.abs(want_f).max())
    np.testing.assert_allclose(ATr, want_b, rtol=0, atol=48 * EPS * np.abs(want_b).max())


def test_rows_and_blocks_code():
    """beam_rows: 2 ceil(reach / ystep) + 1 rows, reach = 3 sigma + w / 2 + v / sqrt(2) (the voxel's half diagonal);
    block_voxels: the largest power of 2 of voxels whose entries (32 bytes per corner) fit the budget, at least 1."""
    from anri.index.occupancy import beam_rows, block_voxels

    for sig, w, v, ys in ((0.8, 0.0, 1.0, 1.0), (0.5 / 2.3548, 0.0, 0.2, 0.2), (0.3, 1.5, 0.5, 0.25)):
        reach = 3 * sp.nsimplify(sig) + sp.nsimplify(w) / 2 + sp.nsimplify(v) / sp.sqrt(2)
        assert (
            beam_rows({"sig_beam": sig, "width_beam": w, "voxel": v, "ystep": ys})
            == 2 * int(sp.ceiling(reach / sp.nsimplify(ys))) + 1
        )
    for q, nj, budget, nc in ((64, 40, 2**30, 8), (16, 400, 1e6, 28), (512, 512, 1e3, 8)):
        n = max(0, int(sp.floor(sp.log(sp.nsimplify(budget) / (32 * nc * q * nj), 2))))
        assert block_voxels(q, nj, budget, nc) == 2**n


def test_predictions_code():
    """predictions: predict's (eta, omega) for each orientation (padded to whole chunks), used where it diffracts,
    |sin eta| > etacut and the orientation is real, its weight the Lorentz-polarisation factor times |F|^2 there and
    0 elsewhere (idx-weight, coverage 1); exactly predict's outputs, combined. predict is called on the same padded
    batch of 4: a batch of another shape compiles differently, and float32 results can then differ by an ulp."""
    from anri.crystal.lattice import lpars_to_B
    from anri.index.search import predict, predictions

    B = np.asarray(lpars_to_B(np.array([2.87] * 3 + [90.0] * 3)), np.float32)
    U = np.asarray([evaluate(cayley(sp.Matrix([sp.Rational(a, 10), sp.Rational(b, 10), sp.Rational(c, 10)])), {})
                    for a, b, c in ((1, 2, 3), (-3, 1, 2), (2, -2, 1))], np.float32)  # fmt: skip
    rings = {"hkls": np.array([[1, 1, 0], [2, 0, 0], [2, 1, 1]], float), "F2": np.array([1.2, 0.8, 1.0])}
    geom = {"wavelength": 0.3, "k_in_lab": jnp.array([1.0, 0, 0]), "wedge": 0.0, "chi": 0.0, "pol_factor": 0.9}
    e, o, u, w = (np.asarray(a) for a in predictions(U, B, rings, geom, 0.2, qc=4))
    pe, po, pok, plp = (
        np.asarray(a)[:3]
        for a in predict(jnp.asarray(np.concatenate([U, U[:1]])), jnp.asarray(B), jnp.asarray(rings["hkls"]), geom)
    )
    assert e.shape[0] == 4  # padded to a whole chunk of 4
    use = pok & (np.abs(np.sin(np.radians(pe))) > 0.2)
    np.testing.assert_array_equal(u[:3], use)
    assert not u[3].any()
    np.testing.assert_array_equal(w[:3], np.where(use, plp * np.repeat(rings["F2"], 2).astype(np.float32), 0.0))
    np.testing.assert_array_equal(e[:3], pe)
    np.testing.assert_array_equal(o[:3], po)


def test_ring_widths_code():
    """ring_widths: each ring's window is half the gap to its neighbours (at most max_hw at the ends); the background
    the median of the window's outer fifth; the offset the net centroid less the ring; the half-width the distance
    from the centroid holding frac of the net intensity, to the farther edge of its bin (half a bin added). Exactly,
    in rationals, on a profile of small integers."""
    from fractions import Fraction as Fr

    from anri.index.data import ring_widths

    g = rng(40)
    lo, step, n = 4.5, 0.03125, 96  # powers of 2: bin centres exact
    ring_tth = np.array([5.0, 5.7, 6.4])
    prof = g.integers(0, 4, n).astype(float)
    for t, a, w in ((5.0, 60, 0.05), (5.7, 40, 0.08), (6.4, 30, 0.04)):
        x = lo + (np.arange(n) + 0.5) * step
        prof += np.round(a * np.exp(-0.5 * ((x - t - 0.01) / w) ** 2))
    off, hw = ring_widths(prof, lo, step, ring_tth, 0.95, 0.5)
    X = [Fr(lo) + (Fr(i) + Fr(1, 2)) * Fr(step) for i in range(n)]
    T = [Fr(str(t)) for t in ring_tth]
    edge = 2 * Fr(1, 2)  # beyond the end rings: twice max_hw
    gaps = [edge] + [T[i + 1] - T[i] for i in range(len(T) - 1)] + [edge]
    for r, t in enumerate(T):
        half = min(gaps[r], gaps[r + 1]) / 2  # the gaps below and above ring r
        win = [(x, Fr(int(p))) for x, p in zip(X, prof) if abs(x - t) < half]
        outer = sorted(p for x, p in win if abs(x - t) > Fr(8, 10) * half)
        k = len(outer)
        bg = (outer[(k - 1) // 2] + outer[k // 2]) / 2 if k else Fr(0)
        net = [(x, max(p - bg, Fr(0))) for x, p in win]
        tot = sum(v for _, v in net)
        c = sum(x * v for x, v in net) / tot
        by = sorted(net, key=lambda xv: abs(xv[0] - c))
        cum, want_hw = Fr(0), None
        for x, v in by:
            cum += v
            if cum >= Fr(95, 100) * tot:
                want_hw = abs(x - c) + Fr(step) / 2
                break
        # sums of ~30 small integers and positions, exact in float64 but for the centroid's one quotient
        assert abs(off[r] - float(c - t)) <= 16 * EPS * 7 and abs(hw[r] - float(want_hw)) <= 16 * EPS * 7, r


def test_coverage_at_prediction():
    """idx-coverage: on a bin [0, 1] x [0, 1] (eta across, 2theta along the ring) cut by a gap's edge, measured where
    eta < c (a gap crossing the ring) or where 2theta < c (a gap running along it), the bin's coverage, the measured
    area's share, is c, and it is the mean of m over a prediction placed uniformly in the bin. A prediction is at one
    place, where m is 0 or 1, so it differs from c by c or 1 - c, at most max(c, 1 - c)."""
    c = sp.Symbol("c", positive=True)
    x, y = sp.symbols("x y", real=True)
    for m in (sp.Piecewise((1, x < c), (0, True)), sp.Piecewise((1, y < c), (0, True))):
        area = sp.integrate(sp.integrate(m, (x, 0, 1)), (y, 0, 1))
        assert is_zero(sp.simplify(area.subs(c, sp.Rational(3, 10)) - sp.Rational(3, 10)))
        assert is_zero(sp.simplify(area.subs(c, sp.Rational(7, 10)) - sp.Rational(7, 10)))
    for cv in (sp.Rational(1, 10), sp.Rational(1, 2), sp.Rational(4, 5)):
        assert max(abs(1 - cv), abs(0 - cv)) == max(cv, 1 - cv)
