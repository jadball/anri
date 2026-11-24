.. _theory_conventions:

Conventions
===========

Units, the names of things, and the symbols the theory pages use. The frames these are written in, from Miller
indices to detector pixels, are in :doc:`frames`.

Reciprocal space and units
--------------------------

- **No factor of 2π.** A reciprocal-lattice vector has length :math:`|\vec{g}| = d^* = 1/d`, and a wave-vector
  :math:`|\vec{k}| = 1/\lambda` (the crystallographer's convention, as ImageD11).
- **Lengths.** Wavelength and lattice parameters in Å, :math:`d^*` in Å⁻¹. Sample positions, dty, voxel sizes and beam
  widths share one unit, which in Anri is the micrometre. Detector distance and pixel sizes share one unit (any).
- **Angles** are in degrees at every public function, unless the name or the docstring says radians (beam divergence
  and orientation spreads are radians, being small rotations).
- **UBI.** A crystal's state is its :math:`(\matr{U}\matr{B})^{-1}` matrix, "UBI", whose rows are its real-space lattice vectors
  in the sample frame (ImageD11's ``ubi``). It holds orientation and strain together.

Distortion and strain
---------------------

- **Real space.** A distortion :math:`\matr{\beta}` acts on real-space lattice vectors, :math:`\vec{a} \to (\matr{I} +
  \matr{\beta})\vec{a}`, :math:`\matr{\beta} = \varepsilon + [\vec{w}]_\times`; :math:`\varepsilon` is the lattice
  strain, positive in tension, and :math:`\vec{w}` the lattice rotation. For a compatible field :math:`\beta_{ij} =
  \partial_j u_i`, and the Nye tensor is :math:`\alpha_{ij} = \epsilon_{jkl}\partial_k\beta_{il}`.
- **Reciprocal space.** Reciprocal-lattice vectors move by the inverse transpose, :math:`\vec{g} \to (\matr{I} +
  \matr{\beta})^{-T}\vec{g} \approx (\matr{I} - \matr{\beta}^T)\vec{g}` (:eq:`lim-reciprocal`). So a strain
  :math:`\varepsilon` along :math:`\vec{g}` moves a peak by :math:`\Delta 2\theta = -2\varepsilon\tan\theta`.
- Statements about dislocations and compatibility are about :math:`\matr{\beta}`, never about the reciprocal
  deformation.

Two branches per reflection
---------------------------

Each reflection diffracts at two omegas per turn, one either side of the beam (:doc:`diffraction`). Anri calls them
branches: ``etasign = +1`` (ImageD11's ``omega1``) and ``-1`` (``omega2``). A reflection :math:`h` gives predictions
:math:`j = 2h + \text{branch}`. For a beam along :math:`x` and an untilted rotation axis, ``etasign`` is the sign of
:math:`\eta` (:doc:`diffraction`, proved there).

Symbols used throughout
-----------------------

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Symbol
      - Meaning
    * - :math:`\lambda`, :math:`\sigma_\lambda`
      - wavelength and its spread (standard deviation)
    * - :math:`\vec{h}`
      - Miller indices :math:`(h, k, l)`
    * - :math:`\vec{g}`, :math:`\vec{Q}`
      - reciprocal-lattice vector and scattering vector; at diffraction :math:`\vec{Q} = \vec{g}`
    * - :math:`\vec{k}_{\text{in}}`, :math:`\vec{k}_{\text{out}}`
      - incoming and diffracted wave-vectors, length :math:`1/\lambda`
    * - :math:`\omega`, :math:`2\theta`, :math:`\eta`
      - rotation angle, scattering angle, azimuth (degrees)
    * - :math:`f_{vq}`
      - occupancy of orientation :math:`q` in voxel :math:`v` (:doc:`indexing`)
    * - :math:`d_b`, :math:`\mu_b`
      - measured and modelled intensity in histogram bin :math:`b`
