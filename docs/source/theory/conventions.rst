.. _theory_conventions:

Conventions
===========

Anri follows ImageD11's conventions [ImageD11]_, which follow FABLE's [FABLEgeom]_. Where it differs, this page
says so.

Frames
------

**Lab frame.** :math:`x` is along the beam (when the beam is not tilted), :math:`z` is up, along the untilted omega axis, and :math:`y`
completes a right-handed set: looking downstream, :math:`y` points to the left.

**Sample frame.** Fixed to the sample, which sits on a stack of omega (rotation about :math:`z`), chi (about :math:`x`), wedge
(about :math:`y`) and the dty translation (along :math:`y`). A point :math:`\vec{v}_s` in the sample frame is in the lab at

.. math::
    :label: conv-stack

    \vec{v}_{\text{lab}} = (0,\ \text{dty} - y_0,\ 0) + \matr{W} \matr{C} \matr{R}(\omega)\, \vec{v}_s,

with :math:`\matr{W}`, :math:`\matr{C}`, :math:`\matr{R}` right-handed rotations by wedge, chi and omega, and :math:`y_0` the dty at which the
rotation axis crosses the beam. **Code:** :func:`anri.geom.gonio.sample_to_lab`, :func:`anri.geom.gonio.lab_to_sample`.
ImageD11 signs wedge the other way; Anri's wedge is a positive rotation about :math:`y`.

**Crystal frame.** Cartesian, fixed to the lattice by the :math:`\matr{B}` matrix (:doc:`crystal`). An orientation
:math:`\matr{U}` turns the crystal frame into the sample frame.

**Beam frame.** For a beam along :math:`\hat{k}` (not vertical), the directions across it are horizontal,
:math:`\hat{e}_h = \hat{z} \times \hat{k} / |\hat{z} \times \hat{k}|`, and vertical, :math:`\hat{e}_v = \hat{k} \times \hat{e}_h`. For
a beam along :math:`x` they are lab :math:`y` and :math:`z`. Divergence, polarisation and the beam's profile are defined along these,
so they follow the beam wherever it points. **Code:** :func:`anri.geom.beam.beam_basis`.

**Detector frame.** Pixels are indexed (slow, fast). The detector's distance, tilts, beam centre, pixel sizes and
readout flips :math:`o_{11} \ldots o_{22}` are ImageD11's parameters [FABLEgeom]_. **Code:**
:func:`anri.geom.detector.detector_transforms`, :func:`anri.geom.detector.detector_basis_vectors_lab`.

Reciprocal space and units
--------------------------

- **No factor of :math:`2\pi`.** A reciprocal-lattice vector has length :math:`|\vec{g}| = d^* = 1/d`, and a wave-vector
  :math:`|\vec{k}| = 1/\lambda` (the crystallographer's convention, as ImageD11).
- **Lengths.** Wavelength and lattice parameters in Å, :math:`d^*` in Å⁻¹. Sample positions, dty, voxel sizes and beam
  widths share one unit, which in Anri is the micrometre. Detector distance and pixel sizes share one unit (any).
- **Angles** are in degrees at every public function, unless the name or the docstring says radians (beam divergence
  and orientation spreads are radians, being small rotations).
- **UBI.** A crystal's state is its :math:`(\matr{U}\matr{B})^{-1}` matrix, "UBI", whose rows are its real-space lattice vectors
  in the sample frame (ImageD11's ``ubi``). It holds orientation and strain together.

Angles of a diffracted ray
--------------------------

The scattering angle is :math:`2\theta`, between :math:`\vec{k}_{\text{in}}` and :math:`\vec{k}_{\text{out}}`. The azimuth :math:`\eta` goes
round the beam, from vertical:

.. math::
    :label: conv-eta

    \eta = \operatorname{atan2}\left(-\hat{k}_{\text{out}} \cdot \hat{e}_h,\ \hat{k}_{\text{out}} \cdot \hat{e}_v\right),

so for a beam along :math:`x`, :math:`\eta = \operatorname{atan2}(-k_y, k_z)`: :math:`\eta = 0` is straight up (12 o'clock on the
detector, looking downstream), :math:`\eta = 90°` is at 3 o'clock (towards :math:`-y`) and :math:`\pm 180°` is straight down.
**Code:** :func:`anri.diffract.scatter.k_out_to_tth_eta`.

Each reflection diffracts at two omegas per turn, one either side of the beam (:doc:`diffraction`). Anri calls them
branches: ``etasign = +1`` (ImageD11's ``omega1``) and ``-1`` (``omega2``). A reflection :math:`h` gives predictions
:math:`j = 2h + \text{branch}`.

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
