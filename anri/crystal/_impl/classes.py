"""Crystallography classes."""

import Dans_Diffraction as dif
import Dans_Diffraction.functions_lattice
import jax
import jax.numpy as jnp
import polars as pd
from Dans_Diffraction.classes_crystal import Cell as dd_Cell
from Dans_Diffraction.classes_crystal import Crystal as dd_Crystal
from Dans_Diffraction.classes_crystal import Symmetry as dd_Symm
from Dans_Diffraction.classes_scattering import Scattering as dd_Scatter

from anri.diffract import ds_to_tth, q_to_ds

from .utils import (
    B_to_F,
    F_to_O,
    O_to_A,
    UB_and_B_to_U,
    UBI_to_mt,
    UBI_to_UB,
    hkl_B_to_q_crystal,
    lpars_to_mt,
    mt_to_lpars,
    mt_to_rmt,
    rmt_to_B,
    volume_direct,
)


class UnitCell:
    """Class to hold and manipulate a unit cell.

    Extends :func:`Dans_Diffraction.classes_crystal.Cell`.
    """

    def __init__(self, dd_ucell: dd_Cell) -> None:
        self._uc = dd_ucell

    @property
    def lattice_parameters(self) -> jax.Array:
        """Get the lattice parameters as an array.

        Returns
        -------
        jax.Array
            [6] Lattice parameters (a,b,c,alpha,beta,gamma) with angles in degrees
        """
        return self._uc.lp()

    @property
    def mt(self) -> jax.Array:
        """Get the metric tensor as an array.

        Returns
        -------
        jax.Array
            [3,3] metric tensor
        """
        return lpars_to_mt(self.lattice_parameters)

    @property
    def rmt(self) -> jax.Array:
        """Get the reciprocal metric tensor as an array.

        Returns
        -------
        jax.Array
            [3,3] reciprocal metric tensor
        """
        return mt_to_rmt(self.mt)

    @property
    def B(self) -> jax.Array:
        """Get the B matrix as an array.

        Returns
        -------
        jax.Array
            [3,3] Reciprocal space orthogonalization matrix
        """
        return rmt_to_B(self.rmt)

    @property
    def F(self) -> jax.Array:
        """Get the F matrix as an array.

        Returns
        -------
        jax.Array
            [3,3] Real space fractionalization matrix
        """
        return B_to_F(self.B)

    @property
    def O(self) -> jax.Array:
        """Get the O matrix as an array.

        Returns
        -------
        jax.Array
            [3,3] Real space orthogonalization matrix
        """
        return F_to_O(self.F)

    @property
    def A(self) -> jax.Array:
        """Get the A matrix as an array.

        Returns
        -------
        A: jax.Array
            [3,3] Reciprocal space fractionalization matrix
        """
        return O_to_A(self.O)

    @property
    def volume(self) -> jax.Array:
        """Get the real space unit cell volume as an array.

        Returns
        -------
        jax.Array
            [1] Volume of direct space unit cell
        """
        return volume_direct(self.mt)


class Symmetry:
    """Class to hold and manipulate symmetry operators.

    Extends :func:`Dans_Diffraction.classes_crystal.Symmetry`.
    """

    def __init__(self, dd_sym: dd_Symm) -> None:
        self._sym = dd_sym

    @property
    def sgname(self) -> str:
        """Return the name of the spacegroup."""
        return self._sym.spacegroup_name()

    @property
    def sgno(self) -> int:
        """Return the number of the spacegroup."""
        return int(self._sym.spacegroup_number)

    @property
    def sym_ops(self) -> jax.Array:
        """Return the symmetry operators."""
        return jnp.array(self._sym.symmetry_matrices)[:, :3, :3]


class Crystal(UnitCell, Symmetry):
    """Class to hold and manipulate a crystal, which we think of as a :func:`UnitCell` with a :func:`Symmetry`."""

    def __init__(self, dd_ucell: dd_Cell, dd_sym: dd_Symm) -> None:
        self._uc = dd_ucell
        self._sym = dd_sym

        Symmetry.__init__(self, self._sym)
        UnitCell.__init__(self, self._uc)

        # Dataframe of scattering info
        # such as h, k, l, ds, tth, intensity
        self._scatter_table = None

    def make_hkls(self, dsmax: float, wavelength: float, expand_to_p1: bool = True, tol: float = 0.001) -> None:
        """Generate integer ring HKLs, two-theta and d* values within a given max d* range and a supplied wavelength.

        Parameters
        ----------
        dsmax
            Maximum d-star value considered.
        wavelength
            Beam wavelength in angstroms.
        expand_to_p1
            Generate all symmetry-equivalent hkls, optional
        tol
            Tolerance in d-star to consider two hkls as belong to the same ring, optional

        Notes
        -----
        We use `Dans Diffraction <https://github.com/DanPorter/Dans_Diffraction>`_ to compute HKLs.
        In that package, all HKLs are considered "valid" - we only filter by the symmetry of the system *after*
        computing the expected intensities, then we discard peaks with very low predicted intensity.
        """
        # make all hkls within a given dsmax
        # want to give max angle in d-star

        tth_max = ds_to_tth(dsmax, wavelength)

        hkl = self._uc.all_hkl(wavelength_a=wavelength, max_angle=tth_max)

        if not expand_to_p1:
            hkl = self._sym.remove_symmetric_reflections(hkl)
        hkl = self._uc.sort_hkl(hkl)
        hkl_B_to_q_crystal_vec = jax.vmap(hkl_B_to_q_crystal, in_axes=[0, None])
        q_to_ds_vec = jax.vmap(q_to_ds)
        q_crystal = hkl_B_to_q_crystal_vec(hkl, self.B)
        ds = q_to_ds_vec(q_crystal)
        tth = ds_to_tth(ds, wavelength)

        inrange = jnp.all(jnp.array([tth < tth_max, tth > 0.0]), axis=0)
        hkl = hkl[inrange, :]
        tth = tth[inrange]
        ds = ds[inrange]

        hkl = jax.device_get(hkl).astype(int)
        ds = jax.device_get(ds)
        tth = jax.device_get(tth)

        self._scatter_table = pd.DataFrame({"h": hkl[:, 0], "k": hkl[:, 1], "l": hkl[:, 2], "tth": tth, "ds": ds})

        self._ring_ds_tol = tol

    @property
    def allhkls(self) -> jax.Array:
        """Get an array of all generated (h, k, l). Must call :func:`make_hkls` first.

        Returns
        -------
        jax.Array
            [npks, 3] Array of (h, k, l) within `dsmax`, including those with no intensities!

        Raises
        ------
        AttributeError
            If the scattering table hasn't been generated yet via :func:`make_hkls`.
        """
        # all hkls
        if self._scatter_table is None:
            raise AttributeError("Must compute first with self.makerings(dsmax, wavelength)")
        return jnp.array(self._scatter_table[["h", "k", "l"]].to_numpy())

    @property
    def alltth(self) -> jax.Array:
        """Get an array of all generated two-theta values. Must call :func:`make_hkls` first.

        Returns
        -------
        jax.Array
            [npks, 3] Array of (h, k, l) within `dsmax`, including those with no intensities!

        Raises
        ------
        AttributeError
            If the scattering table hasn't been generated yet via :func:`make_hkls`.
        """
        if self._scatter_table is None:
            raise AttributeError("Must compute first with self.makerings(dsmax, wavelength)")
        return jnp.array(self._scatter_table["tth"].to_numpy())

    @property
    def allds(self) -> jax.Array:
        """Get an array of all generated d* values. Must call :func:`make_hkls` first.

        Returns
        -------
        jax.Array
            [npks,] Array of all d* within `dsmax`, including those with no intensities!

        Raises
        ------
        AttributeError
            If the scattering table hasn't been generated yet via :func:`make_hkls`.
        """
        if self._scatter_table is None:
            raise AttributeError("Must compute first with self.makerings(dsmax, wavelength)")
        return jnp.array(self._scatter_table["ds"].to_numpy())


class Grain(UnitCell):
    """An oriented :func:`UnitCell` with no symmetry."""

    def __init__(self, UBI: jax.Array) -> None:
        # UBI represents basis vectors in cartesian lab frame
        # we can directly get the lattice parameters from those
        self.UBI = UBI
        mt = UBI_to_mt(UBI)
        lpars = mt_to_lpars(mt)
        self._uc = dd_Cell(*lpars)
        super().__init__(self._uc)

        self._uc.orientation.set_u(self.U)

    @property
    def UB(self) -> jax.Array:
        """Invert (U.B)^-1 to get U.B matrix.

        Returns
        -------
        UB: jax.Array
            [3,3] U.B matrix
        """
        return UBI_to_UB(self.UBI)

    @property
    def U(self) -> jax.Array:
        """Separate U matrix from U.B.

        Returns
        -------
        U: jax.Array
            [3,3] U matrix
        """
        return UB_and_B_to_U(self.UB, self.B)


def groupby_isclose(series: pd.Series, atol: float = 0, rtol: float = 0) -> pd.Series:
    """Return series of unique ids so we can split up the series by values with a tolerance.

    Parameters
    ----------
    series
        Series to look at for values.
    atol
        Absolute tolerance between values (optional)
    rtol
        Relative tolerance between values (optional)

    Returns
    -------
        Series of unique integer IDs that we can group by.
    """
    # Sort values to make sure values are monotonically increasing:
    s = series.sort()

    # Calculate tolerance value:
    tolerance = atol + rtol * s

    # Calculate a monotonically increasing index that increase when the
    # difference between current and previous value changes:
    by = s.diff().fill_null(0).gt(tolerance).cum_sum()

    return by


class Structure(Crystal):
    """A :func:`Crystal` with some :func:`Symmetry` and some atoms."""

    def __init__(self, dd_crystal: dd_Crystal) -> None:
        self._struc = dd_crystal
        self._uc = self._struc.Cell
        self._sym = self._struc.Symmetry
        self._scatterer = dd_Scatter(self._struc)

        super().__init__(self._uc, self._sym)

        self._rings_dict = None

    @property
    def rings_dict(self) -> dict[int, pd.DataFrame]:
        """Get a dictionary of dataframes of rings, grouped by ring ID.

        Computed once on-demand if self.allhkls is preset. Intensities are included!

        Returns
        -------
        dict[int, pd.DataFrame]
            {nrings} Dict of {ring_id: dataframe}

        Raises
        ------
        AttributeError
            If the scattering table hasn't been generated yet via :func:`make_hkls`.
        """
        if self._scatter_table is None:
            raise AttributeError("Must compute all reflections first with self.make_hkls(dsmax, wavelength)")
        else:
            if self._rings_dict is None:
                self._scatter_table = self._scatter_table.with_columns(
                    intensity=self._scatterer.intensity(self.allhkls, scattering_type="xray", int_hkl=True)
                )
                # get hkls with meaningful intensities
                real_hkls = self._scatter_table.filter(pd.col("intensity") > 0.01)

                ring_ids = groupby_isclose(real_hkls["ds"], atol=self._ring_ds_tol)
                real_hkls = real_hkls.with_columns(ring_id=ring_ids)

                # group by and split into dicts
                # group by d-star with a tolerance
                self._rings_dict = {
                    ring_id: ring
                    for ring_id, (refl_id, ring) in enumerate(
                        list(real_hkls.group_by(by=pd.col(name="ring_id"), maintain_order=True))
                    )
                }
            return self._rings_dict

    @property
    def ringhkls(self) -> dict[float, jax.Array]:
        """Get a dictionary of ring HKLs, keyed by d*, just like ImageD11.

        Returns
        -------
        dict[float, jax.Array]
            {nrings} Dict of (npks,3) arrays of HKL grouped into rings

        Raises
        ------
        AttributeError
            If the scattering table hasn't been generated yet via :func:`make_hkls`.
        """
        # all hkls
        if self._scatter_table is None:
            raise AttributeError("Must compute first with self.make_hkls(dsmax, wavelength)")
        return {ring["ds"][0]: jnp.array(ring[["h", "k", "l"]].to_numpy()) for ring in self.rings_dict.values()}

    @property
    def ringhkls_arr(self) -> jax.Array:
        """Get all hkls with meaningful intensities as a single array.

        Returns
        -------
        jax.Array
            [npks,3] array of hkls with meaningful intensities

        Raises
        ------
        AttributeError
            If the scattering table hasn't been generated yet via :func:`make_hkls`.
        """
        if self._scatter_table is None:
            raise AttributeError("Must compute first with self.make_hkls(dsmax, wavelength)")
        return jnp.concatenate(list(self.ringhkls.values()))

    @property
    def ringtth(self) -> jax.Array:
        """Get an array of per-ring two-theta values.

        Returns
        -------
        jax.Array
            [nrings] array of per-ring two-theta values.

        Raises
        ------
        AttributeError
            If the scattering table hasn't been generated yet via :func:`make_hkls`.
        """
        # all hkls
        if self._scatter_table is None:
            raise AttributeError("Must compute first with self.make_hkls(dsmax, wavelength)")
        return jnp.array([ring["tth"][0] for ring in self.rings_dict.values()])

    @property
    def ringds(self) -> jax.Array:
        """Get an array of per-ring d* values.

        Returns
        -------
        jax.Array
            [nrings] array of per-ring d* values.

        Raises
        ------
        AttributeError
            If the scattering table hasn't been generated yet via :func:`make_hkls`.
        """
        if self._scatter_table is None:
            raise AttributeError("Must compute first with self.make_hkls(dsmax, wavelength)")
        return jnp.array(list(self.ringhkls.keys()))

    @property
    def ringmult(self) -> jax.Array:
        """Get an array of per-ring multiplicities.

        Returns
        -------
        jax.Array
            [nrings] array of per-ring multiplicities.

        Raises
        ------
        AttributeError
            If the scattering table hasn't been generated yet via :func:`make_hkls`.
        """
        if self._scatter_table is None:
            raise AttributeError("Must compute first with self.make_hkls(dsmax, wavelength)")
        return jnp.array([hkls.shape[0] for hkls in self.ringhkls.values()])

    @classmethod
    def from_cif(cls, filename: str) -> "Structure":
        """Load a :func:`Structure` from a CIF file.

        Parameters
        ----------
        filename
            The path to the CIF file.

        Returns
        -------
        Structure
            A :func:`Structure` instance.
        """
        dd_crystal = dif.Crystal(filename)
        return cls(dd_crystal)
