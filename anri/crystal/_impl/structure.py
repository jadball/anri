"""Crystal structures, their reflections, rings and structure factors, as plain functions.

A structure with atoms is a ``Dans_Diffraction.Crystal``, which also reads CIFs: ``Dans_Diffraction.Crystal(path)``.
Everything else needs only the lattice parameters and the space-group number.

These run once, on the host, so they return NumPy arrays in float64. The lattice definitions they use (B matrix,
metric tensors) are the JAX functions of :mod:`anri.crystal`, evaluated in float64 here (:func:`float64`), so that
there is a single definition of each and it can still be jitted and differentiated elsewhere.
"""

from __future__ import annotations

import warnings
from contextlib import AbstractContextManager

import jax
import numpy as np
from Dans_Diffraction.classes_crystal import Crystal as DansCrystal
from Dans_Diffraction.classes_crystal import Symmetry as DansSymmetry
from Dans_Diffraction.functions_crystallography import find_spacegroup
from jax.typing import ArrayLike

from .utils import allowed_hkls, lpars_to_B


def float64() -> AbstractContextManager:
    """Evaluate JAX in float64 inside a ``with`` block, without changing the global setting.

    Returns
    -------
    AbstractContextManager
        ``jax.enable_x64(True)``, or ``jax.experimental.enable_x64()`` on older JAX
    """
    if hasattr(jax, "enable_x64"):
        return jax.enable_x64(True)
    import importlib

    # only in older JAX (e.g. 0.4.30); getattr keeps the type checker, run with a newer JAX, from flagging it
    return getattr(importlib.import_module("jax.experimental"), "enable_x64")()  # noqa: B009


def B_matrix(lattice_parameters: ArrayLike) -> np.ndarray:
    """Compute the B matrix of a lattice in float64 (from :func:`anri.crystal.lpars_to_B`, no 2 pi).

    Parameters
    ----------
    lattice_parameters
        [6] a, b, c (angstrom), alpha, beta, gamma (degrees)

    Returns
    -------
    np.ndarray
        [3, 3] B, with ``g = U B h``
    """
    with float64():
        return np.asarray(lpars_to_B(np.asarray(lattice_parameters, float)), float)


def lattice_parameters(structure: DansCrystal) -> np.ndarray:
    """Return the lattice parameters of a ``Dans_Diffraction.Crystal``.

    Parameters
    ----------
    structure
        e.g. ``Dans_Diffraction.Crystal("Fe.cif")``

    Returns
    -------
    np.ndarray
        [6] a, b, c (angstrom), alpha, beta, gamma (degrees)
    """
    return np.asarray(structure.Cell.lp(), float)


def space_group(structure_or_name: DansCrystal | str) -> int:
    """Return the space-group number of a ``Dans_Diffraction.Crystal``, or of a Hermann-Mauguin symbol.

    Parameters
    ----------
    structure_or_name
        A ``Dans_Diffraction.Crystal``, or a symbol such as "Fm-3m"

    Returns
    -------
    int
        Space-group number, 1 to 230
    """
    if isinstance(structure_or_name, str):
        return int(find_spacegroup(structure_or_name)["space group number"])
    return int(structure_or_name.Symmetry.spacegroup_number)


def symmetry_matrices(space_group_number: int) -> np.ndarray:
    """Return a space group's operations as 4 x 4 matrices acting on fractional coordinates.

    An operation ``(R, t)`` takes a position ``x`` to ``R x + t``; reflections transform as ``h -> R^T h``.

    Parameters
    ----------
    space_group_number
        1 to 230

    Returns
    -------
    np.ndarray
        [M, 4, 4] matrices ``[[R, t], [0, 1]]``, every operation including the centring translations
    """
    sym = DansSymmetry()
    sym.load_spacegroup(sg_number=int(space_group_number))
    return np.asarray(sym.symmetry_matrices, float)


def reflections(
    lattice_parameters: ArrayLike, space_group_number: int, wavelength: float, dsmax: float
) -> dict[str, np.ndarray]:
    """List every reflection of a crystal up to a d* (and that diffracts at this wavelength), without absences.

    Systematic absences of the space group (lattice centring, screw axes, glide planes) are removed with
    :func:`anri.crystal.allowed_hkls`. Reflections with no intensity for other reasons (atoms on special positions)
    remain; :func:`structure_factors` finds those.

    Parameters
    ----------
    lattice_parameters
        [6] a, b, c (angstrom), alpha, beta, gamma (degrees)
    space_group_number
        1 to 230
    wavelength
        Angstrom
    dsmax
        Largest d* = 1 / d (1 / angstrom); capped at 2 / wavelength (2theta = 180 degrees)

    Returns
    -------
    dict
        "hkl" [N, 3] int, "ds" [N] (1 / angstrom) and "tth" [N] (degrees), sorted by d* (then h, k, l)
    """
    lp = np.asarray(lattice_parameters, float)
    B = B_matrix(lp)
    dsmax = min(float(dsmax), 2.0 / float(wavelength))
    # |h| = |g . a| <= d* |a|, so each index is bounded by d*max times the length of its direct axis
    n = np.floor(dsmax * lp[:3] + 1e-9).astype(int)
    h, k, l = np.meshgrid(*(np.arange(-m, m + 1) for m in n), indexing="ij")
    hkl = np.stack([h.ravel(), k.ravel(), l.ravel()], 1)
    hkl = hkl[np.any(hkl != 0, 1)]
    ds = np.linalg.norm(hkl @ B.T, axis=1)
    keep = ds <= dsmax * (1 + 1e-12)
    hkl, ds = hkl[keep], ds[keep]
    keep = allowed_hkls(hkl, symmetry_matrices(space_group_number))
    hkl, ds = hkl[keep], ds[keep]
    order = np.lexsort((hkl[:, 2], hkl[:, 1], hkl[:, 0], np.round(ds, 10)))
    hkl, ds = hkl[order], ds[order]
    tth = np.degrees(2 * np.arcsin(np.clip(ds * wavelength / 2, 0.0, 1.0)))
    return {"hkl": hkl, "ds": ds, "tth": tth}


def rings(ds: ArrayLike, tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Group reflections into rings of equal d*.

    A ring starts at the smallest d* not yet in a ring and takes every reflection within ``tol`` of it, so rings
    never chain into one another.

    Parameters
    ----------
    ds
        [N] d* of each reflection, sorted ascending (as :func:`reflections` returns them)
    tol
        Largest d* difference within a ring (1 / angstrom)

    Returns
    -------
    ring: np.ndarray
        [N] ring index of each reflection, 0 for the lowest d*
    ring_ds: np.ndarray
        [Nr] d* of each ring (its first reflection's)
    """
    ds = np.asarray(ds, float)
    if np.any(np.diff(ds) < -1e-9):
        raise ValueError("ds must be sorted ascending")
    ring = np.empty(len(ds), int)
    ring_ds = []
    for i, d in enumerate(ds):
        if not ring_ds or d - ring_ds[-1] > tol:
            ring_ds.append(d)
        ring[i] = len(ring_ds) - 1
    return ring, np.asarray(ring_ds)


def structure_factors(structure: DansCrystal, hkl: ArrayLike, wavelength: float) -> np.ndarray:
    """Compute ``|F|^2`` of reflections of a structure, with Debye-Waller factors and anomalous dispersion.

    Uses Dans_Diffraction's X-ray scattering with dispersion (f', f'') at this wavelength. Warns if the structure has
    no isotropic thermal factors, as there is then no Debye-Waller attenuation.

    Parameters
    ----------
    structure
        e.g. ``Dans_Diffraction.Crystal("Fe.cif")``
    hkl
        [N, 3] reflections
    wavelength
        Angstrom

    Returns
    -------
    np.ndarray
        [N] ``|F|^2``, in Dans_Diffraction's units (relative)
    """
    cif = getattr(structure, "cif", None)
    if not cif:  # built in code, not read from a CIF (Dans_Diffraction sets cif = {})
        has_thermal = bool((structure.Atoms.uiso > 0).any())
    else:
        has_thermal = "_atom_site_U_iso_or_equiv" in cif or "_atom_site_B_iso_or_equiv" in cif
    if not has_thermal:
        warnings.warn(
            f"No isotropic thermal factors (U_iso or B_iso) for {structure.name}: "
            "intensities have no Debye-Waller attenuation.",
            stacklevel=2,
        )
    structure.Scatter.setup_scatter(wavelength_a=wavelength, output=False)
    return np.asarray(
        structure.Scatter.intensity(np.asarray(hkl), scattering_type="xray dispersion", int_hkl=True), float
    )
