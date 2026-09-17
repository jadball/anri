"""A small phantom for render tests: a few cubic grains on a voxel grid, a small detector."""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np

import anri.fwd
import anri.geom
from anri.render import Frames, PeakSource

A = 5.43  # cubic cell, Angstrom
WAVELENGTH = 0.5
DET = (160, 160)


def hkls(dsmax: float = 0.75) -> np.ndarray:
    out = []
    r = int(np.ceil(dsmax * A))
    for h, k, l in itertools.product(range(-r, r + 1), repeat=3):
        par = {h % 2, k % 2, l % 2}
        if (h, k, l) == (0, 0, 0) or len(par) != 1:
            continue
        if np.sqrt(h * h + k * k + l * l) / A <= dsmax:
            out.append((h, k, l))
    return np.array(out, float)


def rotation(seed: int) -> np.ndarray:
    q = np.random.default_rng(seed).normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def geometry(scanning: bool = True) -> dict:
    # detector 160 x 160 px of 250 um at 40 mm: rings out to ~2theta 55 deg
    det = anri.geom.detector_transforms(80.0, 250.0, 0.0, 80.0, 250.0, 0.0, 0.0, 40e3, 1, 0, 0, -1)
    sc, fc, norm = anri.geom.detector_basis_vectors_lab(*det)
    g = {
        "wavelength": WAVELENGTH,
        "k_in_lab": np.array([1.0, 0.0, 0.0]),
        "ky": 0.0,
        "kz": 0.0,
        "wedge": 0.0,
        "chi": 0.0,
        "sc_lab": np.asarray(sc),
        "fc_lab": np.asarray(fc),
        "norm_lab": np.asarray(norm),
    }
    if scanning:
        g["y0"] = 0.0
    return g


def scan_source(n: int = 6, ystep: float = 50.0, extra_var=(0.1, 0.1, 0.05, 0.0)) -> PeakSource:
    """n x n voxels, two grains split down the middle, one voxel per dty step."""
    ij = (np.arange(n) - (n - 1) / 2) * ystep
    x, y = np.meshgrid(ij, ij, indexing="ij")
    grain = (x > 0).astype(int).ravel()
    u = np.stack([rotation(1), rotation(2)])[grain]
    ubi = A * np.transpose(u, (0, 2, 1))
    origin = np.column_stack([x.ravel(), y.ravel(), np.zeros(n * n)])
    h = hkls()
    cov_in = anri.fwd.get_cov_in(jnp.array([5.0, 5.0, 5.0]), WAVELENGTH * 1e-3, 1e-4, 1e-4)
    return PeakSource(
        ubi,
        origin,
        np.ones(n * n),
        h,
        np.ones(len(h)),
        geometry(True),
        np.asarray(cov_in),
        extra_var=np.array(extra_var),
        dtype=jnp.float64,
    )


def scan_frames(n_dty: int = 9, ystep: float = 50.0, ostep: float = 1.0, interlaced: bool = True) -> Frames:
    obincens = np.arange(0.0, 360.0, ostep)
    ybincens = (np.arange(n_dty) - n_dty // 2) * ystep
    omega = np.tile(obincens, (n_dty, 1))
    if interlaced:
        omega[1::2] = omega[1::2, ::-1]  # odd scans go backwards
    dty = np.repeat(ybincens[:, None], obincens.size, axis=1)
    return Frames.from_arrays(DET, omega, obincens, dty, ybincens)


def box_source(n_grains: int = 3) -> PeakSource:
    u = np.stack([rotation(10 + i) for i in range(n_grains)])
    ubi = A * np.transpose(u, (0, 2, 1))
    h = hkls()
    cov_in = anri.fwd.get_cov_in(jnp.array([5.0, 5.0, 5.0]), WAVELENGTH * 1e-3, 1e-4, 1e-4)
    return PeakSource(
        ubi,
        np.zeros((n_grains, 3)),
        np.ones(n_grains),
        h,
        np.ones(len(h)),
        geometry(False),
        np.asarray(cov_in),
        extra_var=np.array([0.1, 0.1, 0.05]),
        dtype=jnp.float64,
    )


def box_frames(ostep: float = 1.0) -> Frames:
    obincens = np.arange(0.0, 360.0, ostep)
    return Frames.from_arrays(DET, obincens, obincens)
