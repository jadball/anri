"""Crystallography experimentation with cctbx."""

import cctbx.crystal_orientation
import cctbx.miller
import cctbx.xray.structure
import cctbx.xray.structure_factors
import numpy as np
from scitbx import matrix


class UnitCell:
    """cctbx unitcell interface class."""

    def __init__(self, cctbx_ucell):
        self._uc = cctbx_ucell

    @property
    def lattice_parameters(self):
        return self._uc.parameters()

    @property
    def O(self):
        return matrix.sqr(self._uc.orthogonalization_matrix()).as_numpy_array()

    @property
    def F(self):
        return matrix.sqr(self._uc.fractionalization_matrix()).as_numpy_array()

    @property
    def A(self):
        return self.O.T

    @property
    def B(self):
        return self.F.T

    @property
    def mt(self):
        return matrix.sym(sym_mat3=self._uc.metrical_matrix()).as_numpy_array()

    @property
    def rmt(self):
        return matrix.sym(sym_mat3=self._uc.reciprocal_metrical_matrix()).as_numpy_array()

    @property
    def volume(self):
        return self._uc.volume()


class Symmetry:
    """cctbx symmetry interface class."""

    def __init__(self, cctbx_sym):
        self._sym = cctbx_sym
        self._ringhkls = None
        self._d_min = None
        self._is_expanded = False

    @property
    def sgname(self):
        """Spacegroup name."""
        return self._sym.space_group().info().__str__()

    @property
    def sym_ops(self):
        return np.array([np.array(op.r().as_double()).reshape((3, 3)) for op in self._sym.space_group().all_ops()])

    def makerings(self, dsmax, anomalous=True, expand_to_p1=True):
        self._d_min = 1 / dsmax

        if expand_to_p1:
            self._ringhkls = (
                cctbx.miller.build_set(self._sym, anomalous_flag=anomalous, d_min=self._d_min).expand_to_p1().sort()
            )
            self._is_expanded = True
        else:
            self._ringhkls = cctbx.miller.build_set(self._sym, anomalous_flag=anomalous, d_min=self._d_min).sort()

    @property
    def ringhkls(self):
        if self._ringhkls is None:
            raise AttributeError("Must compute first with self.makerings(dsmax)")
        return np.array(self._ringhkls.indices(), int)

    @property
    def ringds(self):
        if self._ringhkls is None:
            raise AttributeError("Must compute first with self.makerings(dsmax)")
        return np.sqrt(self._ringhkls.d_star_sq().data().as_numpy_array())

    def ringtth(self, wavelength):
        if self._ringhkls is None:
            raise AttributeError("Must compute first with self.makerings(dsmax)")
        return self._ringhkls.two_theta(wavelength=wavelength, deg=True).data().as_numpy_array()

    @property
    def ringmult(self):
        if self._ringhkls is None:
            raise AttributeError("Must compute first with self.makerings(dsmax)")
        return self._ringhkls.multiplicities().data().as_numpy_array()

    @property
    def stol_sq(self):
        return self._ringhkls.sin_theta_over_lambda_sq().data().as_numpy_array()


class Grain(UnitCell):
    """cctbx grain interface class - just an oriented B matrix (no symmetry)."""

    def __init__(self, UBI):
        """A grain in cctbx is defined from a UBI matrix"""
        self.UBI = UBI
        UBI_mat3 = matrix.sqr(UBI.ravel()).as_mat3()
        self._orien = cctbx.crystal_orientation.crystal_orientation(UBI_mat3, False)
        self._uc = self._orien.unit_cell()
        super().__init__(self._uc)

    @property
    def UB(self):
        return np.linalg.inv(self.UBI)

    @property
    def U(self):
        return self._orien.get_U_as_sqr().as_numpy_array()

    # alias for U
    @property
    def u(self):
        return self.u

    # alias for UB
    @property
    def ub(self):
        return self.UB

    # alias for UBI
    @property
    def ubi(self):
        return self.UBI


class Structure(UnitCell, Symmetry):
    """cctbx structure interface class - unitcell + symmetry + atoms."""

    def __init__(self, cctbx_struc):
        self._struc = cctbx_struc
        self._uc = self._struc.unit_cell()
        self._sym = self._struc.crystal_symmetry()

        # multiple inheritance!
        # we get access to both symmetry and unitcell class methods
        Symmetry.__init__(self, self._sym)
        UnitCell.__init__(self, self._uc)

    @property
    def mod_f_sq(self):
        if self._is_expanded:
            return (
                cctbx.xray.structure_factors.from_scatterers(
                    crystal_symmetry=self._ringhkls.crystal_symmetry(), d_min=self._d_min
                )(xray_structure=self._struc.expand_to_p1(), miller_set=self._ringhkls)
                .f_calc()
                .intensities()
                .data()
                .as_numpy_array()
            )
        else:
            return (
                cctbx.xray.structure_factors.from_scatterers(
                    crystal_symmetry=self._ringhkls.crystal_symmetry(), d_min=self._d_min
                )(xray_structure=self._struc, miller_set=self._ringhkls)
                .f_calc()
                .intensities()
                .data()
                .as_numpy_array()
            )

    def I(self, wavelength, B=0.3):
        """Compute approximate diffracted intensities for each ring in self.ringtth - multiplied by multiplicities, for powder data as an example
        Probably we want to incorporate the LPs and exp_neg_2B_stol_sq but not m * I
        Thanks to https://github.com/jcstroud/radialx/blob/master/radialx/powderx.py#L349 for implementation
        """
        # get (sin(theta)/lambda)^2
        # compute exponent for B
        exp_neg_2B_stol_sq = np.exp(-2 * B * self.stol_sq)
        tth_deg = self.ringtth(wavelength)
        tth_rad = np.radians(tth_deg)
        th_rad = tth_rad / 2
        LPs = (1 + np.cos(tth_rad) ** 2) / ((np.sin(th_rad) ** 2) * np.cos(th_rad))
        m_times_i = self.mod_f_sq * self.ringmult
        Is = (LPs * m_times_i) / (self.volume**2)
        Is = Is * exp_neg_2B_stol_sq
        return Is * 100 / Is.max()

    @classmethod
    def from_cif(cls, filename):
        struc = list(cctbx.xray.structure.from_cif(file_path=filename).values())[0]
        return cls(struc)
