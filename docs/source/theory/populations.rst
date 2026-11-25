.. _theory_populations:

From occupancies to populations
===============================

The voxel fit (:doc:`likelihood`, :doc:`localsearch`) leaves each voxel with occupancies of up to :math:`K`
orientations, refined off the search grid. A single real orientation between grid points shares
itself among its neighbours, and a population with a spread covers several. This chapter groups them back into a few orientation populations per voxel, each with a fraction, a
mean and a spread, and describes which voxels are fitted at all.

Grouping
--------

All of a voxel's candidates (or its :math:`m` most occupied), above a fraction :math:`\epsilon` (0.02) of its most
occupied one, are grouped greedily: the most occupied candidate not yet assigned seeds a population, and every
unassigned candidate within a radius of it joins (the larger of one bin of the fit's histogram, 1°, and 1.8 times
the spacing of the fit's last orientations: 4.5° for a plain fit of a 2.5° grid, 1° after the levels), the distance being the misorientation over the
Laue group (:eq:`cry-laue`). Up to :math:`p` (4) populations per voxel, ordered by fraction, largest first: a population
of several candidates can outweigh the one the most occupied candidate seeded. The cut is relative to the most occupied
candidate, not the total, so an occupied voxel always has a population, even when its occupancy is spread thinly over
many orientations: it then comes out as one population with a large spread. A population's fraction is its share of
the voxel's whole occupancy, so every candidate must be grouped: the fit's later levels spread a grain over many children,
and grouping only the 16 most occupied (as before) under-reported the main population (median 0.87 instead of 0.94 on
a Ni superalloy).

A member's symmetric equivalent closest to the seed is the :math:`\matr{U}\matr{S}` of largest
:math:`\operatorname{tr}(\matr{U}_{\text{seed}}^T\matr{U}\matr{S}) = 1 + 2\cos\theta`; the disorientation is the least
such angle, and an orientation's fundamental-zone equivalent its own least (``tests/theory/test_populations.py::test_closest_equivalent_code``).

**Code:** :func:`anri.index.populations.group`, :func:`anri.crystal.orientation.closest_equivalent`,
:func:`anri.crystal.orientation.disorientation`, :func:`anri.crystal.orientation.to_fundamental_zone`.

Each population gets

- its **fraction**: its share of the voxel's total occupancy;
- its **mean orientation**: the members, each turned to its symmetric equivalent closest to the seed, averaged with
  their occupancies as weights, and projected back onto the rotations: the rotation closest (in the Frobenius norm) to
  the weighted sum :math:`\sum_i w_i \matr{U}_i`, from its singular value decomposition :math:`\matr{M} = \matr{V}\matr{S}\matr{W}^T`,

  .. math::
      :label: pop-mean

      \bar{\matr{U}} = \matr{V}\, \operatorname{diag}\!\left(1,\ 1,\ \det \matr{V}\matr{W}^T\right) \matr{W}^T

  [Moakher2002]_.

  **Cited:** [Moakher2002]_. **Proof** for rotations about one axis, where the sum is :math:`\matr{R}_z(\phi)
  \operatorname{diag}(\rho, \rho, \sum w_i)` with :math:`\rho\cos\phi = \sum w_i\cos a_i`, :math:`\rho\sin\phi = \sum
  w_i \sin a_i`, so the mean is :math:`\matr{R}_z(\phi)`: ``tests/theory/test_populations.py::test_mean_of_coaxial_rotations``,
  ``tests/theory/test_populations.py::test_mean_rotation_code``. **Proof** for any rotation: pairs
  :math:`\matr{R}\matr{V}_j` and :math:`\matr{R}\matr{V}_j^T` (each :math:`\matr{V}_j` by less than 90°) sum to
  :math:`\matr{R}\matr{P}` with :math:`\matr{P}` symmetric positive definite, so the mean is :math:`\matr{R}`:
  ``tests/theory/test_populations.py::test_mean_of_symmetric_pairs``,
  ``tests/theory/test_populations.py::test_mean_rotation_general_code``.

  **Code:** :func:`anri.maths.rotation.mean_rotation`;

- its **spread**: the occupancy-weighted rms misorientation of the members from that mean. A rotation's angle is read
  from its trace and its axial part together, :math:`\theta = \operatorname{atan2}(|\vec{a}|, (\operatorname{tr}\matr{R} - 1)/2)`
  with :math:`[\vec{a}]_\times = (\matr{R} - \matr{R}^T)/2`, accurate at every angle (``tests/theory/test_populations.py::test_rotation_angle_of_cayley``,
  ``tests/theory/test_populations.py::test_rotation_angle_code``).

  **Code:** :func:`anri.maths.rotation.rotation_angle`.

  The forward model takes a spread as the standard deviation :math:`\sigma` of each component of a small rotation
  vector (``sig_rot``). For an isotropic spread the rms angle is :math:`\sqrt{3}\,\sigma`
  (``tests/theory/test_populations.py::test_isotropic_rms_angle``), so the populations stage exports ``sig_rot`` as the
  spread divided by :math:`\sqrt{3}`.

**Anri's choice:** populations closer than the radius come out as one, with a larger spread. What the data cannot
separate is reported as a mean and a spread rather than as invented structure. The radius is the larger of what the
data resolve and what the orientations' spacing smears: populations closer than a 1° bin share bins in every
reflection, and one orientation between the fitted ones is shared among neighbours up to a cell diagonal,
:math:`\sqrt{3}` steps, apart (``tests/theory/test_populations.py::test_radius_quoted``). The spread includes the spacing of the orientations fitted (one orientation between them shares
itself among its neighbours), so it is an upper bound on the sample's: the spacing of the fit's last level.

What is reported
----------------

A voxel counts as **occupied** when its total occupancy is above ``occupied`` (0.2) times the 99th percentile of the
total occupancy in the mask (below). In occupied voxels, the populations holding at least ``min_frac`` (0.1) of the
voxel's occupancy are reported, the largest always. These choose what goes into the map entries and the TensorMap; the
raw occupancies are kept in the stage file whatever they are.

**Code:** :func:`anri.index.populations.select_populations`, :func:`anri.index.populations.select_occupied`,
:func:`anri.index.stages.stage_populations`.

Which voxels are fitted
-----------------------

Voxels in air have nothing to fit, and voxels the beam sees at only some omegas are poorly constrained: MLEM can give
them large occupancies that explain almost nothing. The fit is restricted to a mask (:mod:`anri.index.mask`):

- **disc** (default): the voxels the scan sees from every direction. A voxel at distance :math:`r` from the
  axis is in the beam at :math:`\text{dty} = y_0 - r\cos(\omega - \phi)`, so it stays in range at every omega when
  :math:`r` is at most the distance from :math:`y_0` to the nearer end of the scan. Over a full turn a voxel within the
  distance to the farther end is in a scanned row for at least half the turn, every direction once, so the disc
  reaches the farther end (``tests/theory/test_populations.py::test_disc_half_turn``): a half-sample scan (dty from one edge of the sample to just past the axis) is reconstructed
  whole, where the nearer end would keep only a few um round the axis. It can be grown by ``pad`` for a
  sample a little larger than the disc: a voxel at :math:`r` is in the beam over a fraction
  :math:`\tfrac{2}{\pi}\arcsin(r_{\text{disc}}/r)` of the rotation
  (``tests/theory/test_populations.py::test_disc_fraction``).

  ``tests/theory/test_populations.py::test_disc_mask_code`` checks the radius rule.

  **Code:** :func:`anri.index.mask.disc_mask`.

- **auto**: the sample's support in the projections. Each omega's projection (the histogram summed over rings and
  :math:`\eta`, per row) is scaled to its 99th-percentile row and median-filtered along omega; a voxel is sample if its
  row stays above a level (3 %) at every omega the scan covers it, since a voxel in air has some omega where the beam
  through it misses the sample. Then the largest region and its convex hull.

  **Code:** :func:`anri.index.mask.support_mask`, :func:`anri.index.mask.support`.

- a **thresholded reconstruction**, as ImageD11's: filtered back-projection of log intensities with a Hamming-windowed
  ramp filter [KakSlaney1988]_, :math:`|f|(0.54 + 0.46\cos 2\pi f)` on each projection zero-padded to a power of 2 at
  least twice its length (``tests/theory/test_populations.py::test_ramp_filter``,
  ``tests/theory/test_populations.py::test_ramp_filter_code``), thresholded by Otsu's method [Otsu1979]_, then the
  convex hull.

  **Code:** :func:`anri.index.mask.reconstruct`, :func:`anri.index.mask.ramp_filter`, :func:`anri.index.mask.otsu`,
  :func:`anri.index.mask.threshold_mask`.

- a mask from a file, or drawn by hand.

  **Code:** :func:`anri.index.mask.load_mask`, :func:`anri.index.mask.draw_mask`.

**Filter:** the auto and reconstructed masks are built from filtered data (a median along omega; a Hamming-windowed
ramp). They choose which voxels are fitted, never the values fitted there.

With absorption, the mask is also the sample's shape for the paths (:ref:`theory_absorption`): a disc is the wrong
shape unless the sample fills it.

What is left out
----------------

- **Anisotropic spreads**: one rms angle per population, not a 3 × 3 covariance in rotation.
- **Strain**: populations carry orientation only. Strain needs the pixel-level refinement that will follow indexing.
