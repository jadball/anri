"""Crystallography experimentation with Dan's Diffraction."""


import numpy as np
import Dans_Diffraction as dif

import pandas as pd

class unitcell:
    def __init__(self, dd_ucell):
        self._uc = dd_ucell
    
    @property
    def lattice_parameters(self):
        return self._uc.lp()
    
    @property
    def B(self):
        return self._uc.Bmatrix()
    
    @property
    def F(self):
        return self.B.T
    
    @property
    def O(self):
        return np.linalg.inv(self.F)
    
    @property
    def A(self):
        return self.O.T
    
    @property
    def mt(self):
        return np.linalg.inv(self.B) @ np.linalg.inv(self.B.T)
    
    @property
    def rmt(self):
        return np.linalg.inv(self.mt)
    
    @property
    def volume(self):
        return self._uc.volume()

class symmetry:
    def __init__(self, dd_sym):
        self._sym = dd_sym

    @property
    def sgname(self):
        return self._sym.spacegroup_name()
    
    @property
    def sym_ops(self):
        return np.array(self._sym.symmetry_matrices)[:, :3, :3]


class crystal(unitcell, symmetry):
    """a unitcell with symmetry - needed to compute ringhkls"""
    def __init__(self, dd_ucell, dd_sym):
        self._uc = dd_ucell
        self._sym = dd_sym
        
        symmetry.__init__(self, self._sym)
        unitcell.__init__(self, self._uc)

        # pandas dataframe of scattering info
        # such as h, k, l, ds, tth, intensity
        self._scatter_table = None
        
    def make_hkls(self, dsmax, wavelength, expand_to_p1=True, tol=0.001):
        # make all hkls within a given dsmax
        # want to give max angle in d-star
        q_max = dif.functions_crystallography.dspace2q(1/dsmax)
        energy_kev = dif.functions_crystallography.wave2energy(wavelength)
        tth_max = dif.functions_crystallography.q2units(q_max, 'tth', energy_kev=energy_kev)
        
        hkl = self._uc.all_hkl(wavelength_a=wavelength, maxq=q_max)
        
        if not expand_to_p1:
            hkl = self._sym.remove_symmetric_reflections(hkl)
        hkl = self._uc.sort_hkl(hkl)
        tth = self._uc.tth(hkl, wavelength_a=wavelength)
        
        inrange = np.all([tth < tth_max, tth > 0.0], axis=0)
        hkl = hkl[inrange, :]
        tth = tth[inrange]

        self._scatter_table = pd.DataFrame()
        self._scatter_table['h'] = hkl[:, 0]
        self._scatter_table['k'] = hkl[:, 1]
        self._scatter_table['l'] = hkl[:, 2]
        # self._scatter_table['len'] = np.sum(np.power(hkl, 2), axis=1)
        self._scatter_table['tth'] = tth
        self._scatter_table['ds'] = 1/dif.functions_crystallography.caldspace(tth, wavelength_a=wavelength)
        # self._scatter_table['ring_id'] = np.unique(tth, return_inverse=True)[1]
        self._ring_ds_tol = tol

    @property
    def allhkls(self):
        # all hkls
        if self._scatter_table is None:
            raise AttributeError("Must compute first with self.makerings(dsmax, wavelength)")
        return self._scatter_table[['h', 'k', 'l']].to_numpy()

    @property
    def alltth(self):
        if self._scatter_table is None:
            raise AttributeError("Must compute first with self.makerings(dsmax, wavelength)")
        return self._scatter_table['tth'].to_numpy()
    
    @property
    def allds(self):
        if self._scatter_table is None:
            raise AttributeError("Must compute first with self.makerings(dsmax, wavelength)")
        return self._scatter_table['ds'].to_numpy()


class grain(unitcell):
    """dans_diffraction grain interface class - an oriented unitcell with no symmetry"""
    def __init__(self, UBI):
        # UBI represents basis vectors in cartesian lab frame
        # we can directly get the lattice parameters from those
        self.UBI = UBI
        lpars = dif.functions_lattice.basis2latpar(UBI)
        self._uc = dif.classes_crystal.Cell(*lpars)
        super().__init__(self._uc)
        
        self._uc.orientation.set_u(self.U)
    
    @property
    def UB(self):
        return np.linalg.inv(self.UBI)
    
    @property
    def U(self):
        return self.UB @ np.linalg.inv(self.B)

def groupby_isclose(series, atol=0, rtol=0):
    # Sort values to make sure values are monotonically increasing:
    s = series.sort_values()

    # Calculate tolerance value:
    tolerance = atol + rtol * s

    # Calculate a monotonically increasing index that increase when the
    # differnce between current and previous value changes:
    by = s.diff().fillna(0).gt(tolerance).cumsum()
    # s_old = s.shift().fillna(s)
    # by = ((s - s_old).abs() > tolerance).cumsum().sort_index()

    return by

class structure(crystal):
    """dans_diffraction structure interface class - a crystal (unitcell + cymmetry) and atoms"""
    def __init__(self, dd_crystal):
        self._struc = dd_crystal
        self._uc = self._struc.Cell
        self._sym = self._struc.Symmetry
        self._scatterer = dif.classes_scattering.Scattering(self._struc)
        
        super().__init__(self._uc, self._sym)

        self._rings_dict = None

    @property
    def rings_dict(self):
        """Dictionary of ring dataframes. Computed once on-demand if self.allhkls is preset"""
        if self._scatter_table is None:
            raise AttributeError("Must compute all reflections first with self.make_hkls(dsmax, wavelength)")
        else:
            if self._rings_dict is None:
                self._scatter_table['intensity'] = self._scatterer.intensity(self.allhkls, scattering_type='xray', int_hkl=True)
                # get hkls with meaningful intensities
                real_hkls = self._scatter_table[self._scatter_table.intensity > 0.01]
                # group by and split into dicts
                # group by d-star with a tolerance of 0.0001
                self._rings_dict = {ring_id:ring for ring_id, (refl_id, ring) in enumerate(list(real_hkls.groupby(by=groupby_isclose(real_hkls.ds, atol=self._ring_ds_tol))))}
                # by = groupby_isclose(df.y, rtol=0.1)
                # self._rings_dict = {ring_id:ring for ring_id, (refl_id, ring) in enumerate(list(real_hkls.groupby('ring_id')))}
            return self._rings_dict

    @property
    def ringhkls(self):
        # all hkls
        if self._scatter_table is None:
            raise AttributeError("Must compute first with self.make_hkls(dsmax, wavelength)")
        return {ring['ds'].iat[0]:ring[['h','k','l']].to_numpy() for ring in self.rings_dict.values()}

    @property
    def ringtth(self):
        # all hkls
        if self._scatter_table is None:
            raise AttributeError("Must compute first with self.make_hkls(dsmax, wavelength)")
        return np.array([ring['tth'].iat[0] for ring in self.rings_dict.values()])
    
    @property
    def ringds(self):
        if self._scatter_table is None:
            raise AttributeError("Must compute first with self.make_hkls(dsmax, wavelength)")
        return np.array(list(self.ringhkls.keys()))
    
    @property
    def ringmult(self):
        if self._scatter_table is None:
            raise AttributeError("Must compute first with self.make_hkls(dsmax, wavelength)")
        return np.array([hkls.shape[0] for hkls in self.ringhkls.values()])
    

    @classmethod
    def from_cif(cls, filename):
        dd_crystal = dif.Crystal(filename)
        return cls(dd_crystal)
