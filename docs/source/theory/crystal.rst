.. _theory_crystal:

The crystal
===========

This chapter turns a crystal structure (a CIF) into the list of reflections Anri predicts and their structure
factors.

Lattice and orientation
-----------------------

A crystal enters as its lattice parameters, which give the :math:`\matr{B}` matrix (:eq:`frm-B`), and its orientation
:math:`\matr{U}`; together, as the UBI, they put a reflection :math:`\vec{h}` at :math:`\vec{g}_s` in the sample frame
(:eq:`frm-g`). Both are defined in :doc:`frames`.

Reflections and rings
---------------------

Every :math:`\vec{h}` with :math:`d^* = |\matr{B}\vec{h}| \le d^*_{\max}` is listed, capped at :math:`2/\lambda`
(:math:`2\theta = 180°`). The space group's systematic absences are removed: :math:`\vec{h}` is absent if some operation
:math:`(\matr{R}, \vec{t})` of the space group has :math:`\vec{h}\matr{R} = \vec{h}` and :math:`\vec{h} \cdot \vec{t}`
not an integer, since then :math:`F(\vec{h}) = e^{2\pi i\, \vec{h} \cdot \vec{t}} F(\vec{h}) = 0`. That covers lattice
centring, screw axes and glide planes. Reflections absent for other reasons (atoms on special positions) stay, and their
structure factor is zero.

**Code:** :func:`anri.crystal.structure.reflections`, :func:`anri.crystal.lattice.allowed_hkls`.

Reflections closer in :math:`d^*` than a tolerance (default :math:`10^{-4}` Å⁻¹) form one ring: rings the data cannot
separate should be merged, not split.

**Code:** :func:`anri.crystal.structure.rings`, :func:`anri.index.data.ring_table`.

Laue group
----------

Two orientations :math:`\matr{U}` and :math:`\matr{U}\matr{S}` give the same reflections when :math:`\matr{S}` is a proper rotation of the
crystal's Laue group: the point group's rotations, plus its improper operations times the inversion (Friedel's law
makes every diffraction pattern centrosymmetric). In the Cartesian crystal frame, a space-group operation :math:`\matr{R}` on
fractional coordinates acts on reflections as :math:`\vec{h} \to \matr{R}^T\vec{h}`, so

.. math::
    :label: cry-laue

    \matr{S} = \matr{B}\, \matr{R}^T \matr{B}^{-1}, \qquad \matr{R} \to \matr{R} \det \matr{R}.

That gives 24 rotations for m-3m, down to 1 for -1.

**Code:** :func:`anri.crystal.orientation.laue_rotations`.

Misorientations and averages of orientations are always taken over these (:doc:`populations`).

Structure factors
-----------------

The intensity of a reflection is proportional to :math:`|F(\vec{h})|^2`,

.. math::
    :label: cry-F

    F(\vec{h}) = \sum_{\text{atoms } a} o_a \left(f_a(s) + f'_a + i f''_a\right)
    e^{-B_a s^2}\, e^{2\pi i\, \vec{h} \cdot \vec{x}_a}, \qquad s = \frac{\sin\theta}{\lambda} = \frac{d^*}{2},

with occupancy :math:`o_a`, atomic form factor :math:`f_a`, anomalous dispersion :math:`f'_a, f''_a` at this wavelength,
isotropic displacement :math:`B_a = 8\pi^2 U_{\text{iso}}` and fractional position :math:`\vec{x}_a`. Anri does not
compute it itself: Dans Diffraction [DansDiffraction]_ reads the CIF and evaluates it. Without displacement parameters
in the CIF there is no Debye-Waller fall-off with :math:`s`, and Anri warns.

**Code:** :func:`anri.crystal.structure.structure_factors`.

The indexer scales :math:`|F|^2` to a mean of 1 over the rings used, so occupancies come out in units of the data.

Attenuation
-----------

A ray loses intensity as :math:`e^{-\mu L}` along a path :math:`L` in the sample. The linear attenuation coefficient
:math:`\mu` comes from the structure's composition and density and each element's total mass attenuation (photoelectric,
Compton and Rayleigh), from Dans Diffraction's tables, which reproduce NIST XCOM [XCOM]_ (iron at 50 keV: 1.958 cm²/g).
Anri returns it per micrometre, the unit of sample positions.

**Code:** :func:`anri.crystal.structure.attenuation`.

Where it is used: :ref:`theory_absorption`.

What is left out
----------------

- **Extinction.** Primary and secondary extinction weaken strong reflections of perfect crystals. Anri ignores it:
  intensities are kinematic.
- **Anisotropic displacement.** Only isotropic displacement parameters enter.
- **Strain in the reflection list.** Reflections and rings come from the reference lattice; a strained voxel still
  diffracts at its own :math:`\text{UBI}^{-1}\vec{h}`, but the list of which :math:`\vec{h}` exist does not change.
