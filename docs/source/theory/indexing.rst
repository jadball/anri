.. _theory_indexing:

Indexing as tomography
======================

The indexer finds, from the data alone, which orientations each voxel of a scanning 3DXRD map holds and how much of
each. This chapter sets up the model it fits; :doc:`search` chooses the orientations it may use, :doc:`likelihood`
fits it, and :doc:`populations` reads the result.

What a voxel is
---------------

At the scales that matter (voxels of 0.1-1 µm, rays tens to hundreds of µm long), a voxel's orientation distribution is
a small mixture of populations: dislocation cells, twin domains, martensite variants, each nearly constant with a small
spread [Jakobsen2006]_. Anri therefore represents each voxel :math:`v` by occupancies :math:`f_{vq} \ge 0` of a list of
orientations :math:`q`, and fits every voxel independently of its neighbours' values.

**Anri's choice:** no coupling between voxels: no smoothness, total variation or basis-function prior on orientation.
Such priors suppress exactly the local variation Anri exists to measure, as averaging each voxel over the rays through
it does. What the data cannot resolve is reported as a mean and a spread (:doc:`populations`), not smoothed away.

The linear model
----------------

A pencil beam at one (dty, omega) lights a line of voxels. The detector sums what they diffract. So the data are
linear in the sample's density over position and orientation [Hayashi2015]_ [Poulsen2004]_:

.. math::
    :label: idx-linear

    \mu_b = \sum_{v, q} A_{b, vq}\, f_{vq},

where :math:`\mu_b` is the expected intensity in a data bin :math:`b` and :math:`A_{b, vq}` is how much of a unit occupancy of
orientation :math:`q` in voxel :math:`v` lands in it. Fitting all voxels jointly is a tomography: one image never says which voxel
along the ray made which part of a peak; only consistency across rows (dty) and across omega places it.

The histogram
-------------

Fitting at pixel level for every trial orientation would be too slow, so the sparse pixels are reduced once to a
histogram over rings, :math:`\eta`, omega and dty rows:

.. math::
    :label: idx-histogram

    H[r, e, o, k] = \sum_{\text{pixels in bin}} I_{\text{pixel}},

at the scale of the orientation grid: bins of 1° in :math:`\eta` and 1° in omega for the fit, and a finer, row-summed "lit"
histogram (0.5° × 0.25°) for the search. Each pixel's :math:`(2\theta, \eta)` is computed from the lab origin (no parallax),
and the pixel goes to the nearest ring if within that ring's tolerance. **Code:** :func:`anri.index.data.pixel_angles`,
:func:`anri.index.data.histogram`, :func:`anri.index.data.histogram_pixels`.

**Ring tolerances** are measured from the data, not assumed: each ring's :math:`2\theta` profile is taken within half the gap
to its neighbours, its background (the median of the window's outer fifth) removed, and the tolerance is the distance
from the ring's position to its centroid plus the half-width that holds 95 % of its net intensity. This sums
everything that widens a ring: parallax (sample size over distance), strain, and detector distortion. **Code:**
:func:`anri.index.data.ring_profile`, :func:`anri.index.data.ring_widths`.

**Anri's choice:** a histogram loses the shape of each peak within a bin, and parallax. It keeps the information the
search and the occupancy fit need at the scale of a 1-3° orientation grid, and makes them cheap enough to run on every
voxel of a 3000 × 3000 map.

The system matrix
-----------------

For voxel :math:`v` at sample position :math:`\vec{v}_s` and orientation :math:`q`, every prediction :math:`j` (reflection and branch,
:doc:`diffraction`) gives an :math:`(\eta_j, \omega_j)` and a weight

.. math::
    :label: idx-weight

    w_j = L_j\, P_j\, |F_j|^2\; c_{r(j), e}\; T_{vj},

the Lorentz and polarisation factors (:eq:`int-lorentz`, :eq:`int-pol`), the structure factor scaled to a mean of 1,
the measured share :math:`c` of that ring's detector area in that :math:`\eta` bin (0 on module gaps, masked pixels and off the
detector; :func:`anri.index.data.detector_coverage`), and with absorption the transmission along the incoming and
diffracted paths (:eq:`int-absorption`). The prediction's weight is shared:

- over the 2 × 2 nearest (:math:`\eta`, omega) bins, bilinearly;
- over dty rows by where the voxel is at that omega (:eq:`dif-dty`): either linearly between the two nearest rows,
  or, given the beam's width, by the beam profile integrated over the voxel's chord in each row
  (:eq:`int-profile`, the same function the renderer uses), each row weighted by its exposure.

**Code:** :func:`anri.index.occupancy.system`, :func:`anri.index.occupancy.beam_rows`,
:func:`anri.index.search.predictions`.

**Fly and helical scans.** When dty moves during the rotation, the frames of one omega bin were taken up to half a step
away from their row's nominal dty. The real dty per (row, omega bin) is measured from the frames' own motor readings and
used in place of the nominal one; bins with more or fewer frames than usual get their exposure. **Code:**
:func:`anri.index.data.dty_offsets`.

The model is never stored. Each pass over the data recomputes :math:`A`'s entries for a block of voxels and accumulates
:math:`A f` (a segment sum) or :math:`A^T r` (a gather), so memory is set by the block size, not by voxels × orientations.
**Code:** :func:`anri.index.occupancy.forward`, :func:`anri.index.occupancy.backward`,
:func:`anri.index.occupancy.block_voxels`.

The voxel grid
--------------

The voxels are an :math:`N \times N` grid of the dty step, centred on the rotation axis and sized from :math:`y_0` and the scanned
range, as ImageD11 pads its reconstructions. The histogram stage fixes it, so a :math:`y_0` chosen later moves no voxel.
**Code:** :func:`anri.geom.scan.recon_positions`, :func:`anri.index.stages.voxel_grid`.

What is left out
----------------

- **Parallax** in the histogram: a voxel off the axis puts its peaks at a slightly different :math:`2\theta` and :math:`\eta`
  than the axis would. The ring tolerance absorbs the :math:`2\theta` part; the :math:`\eta` part is below the 1° bins for
  samples small next to the detector distance.
- **Peak shape within a bin**, and strain: the histogram sees orientation only.
- **Reflections near the rotation axis**, :math:`|\sin\eta| \le` ``etacut`` (default 0.2): smeared over many omega bins
  (:ref:`theory_rotation_motion`), they are left out of the search and the fit.
