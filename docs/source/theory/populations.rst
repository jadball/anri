.. _theory_populations:

From occupancies to populations
===============================

The voxel fit (:doc:`likelihood`) leaves each voxel with occupancies of up to :math:`K` grid orientations, and the local
search (:doc:`localsearch`) with occupancies of their children. A single real orientation between grid points shares
itself among its neighbours, and a population with a spread covers several. This chapter groups them back into a few orientation populations per voxel, each with a fraction, a
mean and a spread, and describes which voxels are fitted at all.

Grouping
--------

A voxel's :math:`m` most occupied candidates (16 by default), above a fraction :math:`\epsilon` (0.02) of its most
occupied one, are grouped greedily: the most occupied candidate not yet assigned seeds a population, and every
unassigned candidate within a radius (1.8 steps of the search grid) of it joins, the distance being the misorientation over the
Laue group (:eq:`cry-laue`). Up to :math:`p` (4) populations per voxel, ordered by fraction, largest first: a population
of several candidates can outweigh the one the most occupied candidate seeded. The cut is relative to the most occupied
candidate, not the total, so an occupied voxel always has a population, even when its occupancy is spread thinly over
many orientations: it then comes out as one population with a large spread.

**Code:** :func:`anri.index.populations.group`, :func:`anri.crystal.orientation.closest_equivalent`.

Each population gets

- its **fraction**: its share of the voxel's total occupancy;
- its **mean orientation**: the members, each turned to its symmetric equivalent closest to the seed, averaged with
  their occupancies as weights, and projected back onto the rotations: the rotation closest (in the Frobenius norm) to
  the weighted sum :math:`\sum_i w_i \matr{U}_i`, from its singular value decomposition :math:`\matr{M} = \matr{V}\matr{S}\matr{W}^T`,

  .. math::
      :label: pop-mean

      \bar{\matr{U}} = \matr{V}\, \operatorname{diag}\!\left(1,\ 1,\ \det \matr{V}\matr{W}^T\right) \matr{W}^T

  [Moakher2002]_.

  **Code:** :func:`anri.maths.rotation.mean_rotation`;

- its **spread**: the occupancy-weighted rms misorientation of the members from that mean.

  **Code:** :func:`anri.maths.rotation.rotation_angle`.

  The forward model takes a spread as the standard deviation :math:`\sigma` of each component of a small rotation
  vector (``sig_rot``). For an isotropic spread the rms angle is :math:`\sqrt{3}\,\sigma`, so the populations
  stage exports ``sig_rot`` as the spread divided by :math:`\sqrt{3}`.

**Anri's choice:** populations closer than the radius come out as one, with a larger spread. What the data cannot
separate is reported as a mean and a spread rather than as invented structure. The radius is set by the search grid's
step even after a local search: its children, a ninth of that step apart after two levels, are grouped within the same
radius, so the local search sharpens each population's mean and spread but does not separate populations closer than
the search grid could. The spread includes the spacing of the orientations fitted (one orientation between them shares
itself among its neighbours), so it is an upper bound on the sample's: the search grid's spacing, or the local search's
finer one.

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

- **disc** (default): the voxels within the scanned dty range at every omega. A voxel at distance :math:`r` from the
  axis is in the beam at :math:`\text{dty} = y_0 - r\cos(\omega - \phi)`, so it stays in range at every omega when
  :math:`r` is at most the distance from :math:`y_0` to the nearer end of the scan. It can be grown by ``pad`` for a
  sample a little larger than the disc: a voxel at :math:`r` is in the beam over a fraction
  :math:`\tfrac{2}{\pi}\arcsin(r_{\text{disc}}/r)` of the rotation.

  **Code:** :func:`anri.index.mask.disc_mask`.

- **auto**: the sample's support in the projections. Each omega's projection (the histogram summed over rings and
  :math:`\eta`, per row) is scaled to its 99th-percentile row and median-filtered along omega; a voxel is sample if its
  row stays above a level (3 %) at every omega the scan covers it, since a voxel in air has some omega where the beam
  through it misses the sample. Then the largest region and its convex hull.

  **Code:** :func:`anri.index.mask.support_mask`, :func:`anri.index.mask.support`.

- a **thresholded reconstruction**, as ImageD11's: filtered back-projection of log intensities with a Hamming-windowed
  ramp filter [KakSlaney1988]_, thresholded by Otsu's method [Otsu1979]_, then the convex hull.

  **Code:** :func:`anri.index.mask.reconstruct`, :func:`anri.index.mask.ramp_filter`, :func:`anri.index.mask.otsu`,
  :func:`anri.index.mask.threshold_mask`.

- a mask from a file, or drawn by hand.

  **Code:** :func:`anri.index.mask.load_mask`, :func:`anri.index.mask.draw_mask`.

With absorption, the mask is also the sample's shape for the paths (:ref:`theory_absorption`): a disc is the wrong
shape unless the sample fills it.

What is left out
----------------

- **Anisotropic spreads**: one rms angle per population, not a 3 × 3 covariance in rotation.
- **Strain**: populations carry orientation only. Strain needs the pixel-level refinement that will follow indexing.
