.. _theory_localsearch:

Coarse to fine in orientation and space
=======================================

The search grid (:doc:`search`) puts each orientation up to :math:`\Delta_{\max}` from the truth
(:eq:`srch-cubic-delta`): 0.87° for a 1° cubic grid, 2.6° for 3°. The fit stage refines the filter's orientations
before and while it fits the voxels (:doc:`likelihood`), without a finer grid over the whole fundamental zone, and
sizes each level's voxels to the orientations' error.

Why grid orientations are not enough
------------------------------------

**Wrong bins.** The voxel fit predicts point spots into 1° × 1° bins in :math:`(\eta, \omega)`. An orientation error
:math:`\delta` moves a spot by about :math:`\delta` in :math:`\eta` and :math:`\delta / |\sin\eta|` in
:math:`\omega`, so a grid orientation puts many of its spots in the wrong bins altogether. MLEM then shares a grain
among several grid neighbours, differently from voxel to voxel. On a synthetic phantom with a 3° grid this left a
checkerboard in every grain (61% of the phantom's voxels occupied, full scan), and on a half-sample scan it emptied the
voxels within a few µm of the rotation axis, which the standard route then turned into streaks along the rays measured
twice. With the phantom's exact orientations the same half scan indexed as cleanly as a full scan, so the forward model
was right and the orientations were not.

**Ghost twins.** A twin shares some reflections with its parent. If the grid point nearest the exact twin of a grain
lands closer to it than the grain's own grid point lands to the grain, that twin point explains the shared reflections
better, and takes occupancy where there is no twin.

**The tomography.** A small rotation :math:`\vec{\delta}` of the lattice moves a reflection in :math:`\omega` by
:math:`\Delta\omega = -\delta_z - \delta_y \cot\eta`. The voxel at distance :math:`r` from the rotation axis, at angle
:math:`\phi`, is in the beam at :math:`y = r \sin(\omega - \phi)`, so the predicted row moves by

.. math::
    :label: ls-row-shift

    \Delta y = r \cos(\omega - \phi)\, \Delta\omega,

up to :math:`R\,e` for orientations off by up to :math:`e` in a sample of radius :math:`R`. Voxels much narrower than
that cannot be fitted consistently with those orientations.

Refining on the row-summed data
-------------------------------

First, without any tomography: each orientation :math:`\matr{U}` from a grid of step :math:`s` is replaced by the 27
rotations of a :math:`3 \times 3 \times 3` grid around it,

.. math::
    :label: ls-children

    \matr{U}_{abc} = \matr{R}(\vec{r}_{abc})\, \matr{U}, \qquad \vec{r}_{abc} = \frac{s}{3}\,(a, b, c),
    \quad a, b, c \in \{-1, 0, 1\},

with :math:`\matr{R}(\vec{r})` the exact rotation by the rotation vector :math:`\vec{r}` about sample axes (the
middle child is :math:`\matr{U}` itself). The children are :math:`s/3` apart and their cells (approximately) tile the
parent's. All of them are fitted to the histogram summed over the rows by the filter's orientation fit
(:doc:`search`), and those with a likelihood ratio above the filter's ``min_lr`` are kept (at most ``keep``). Each round
divides the worst case by 3; they stop at ``refine_to`` (0.3°, a third of the bins). On the phantom, a 3° grid took 2
rounds and 6 s, and every true orientation was then within 0.06-0.27°.

**Code:** :func:`anri.index.local.children`, :func:`anri.index.local.refine`.

Levels in orientation and space
-------------------------------

Level 0 fits voxels :math:`g_0` rows wide with every refined orientation as a candidate (:eq:`lik-candidates`), and
each of the ``levels`` after it subdivides the orientations by 3 again, with voxels of

.. math::
    :label: ls-level-rows

    g_k = \max\!\left(1, \left\lceil \frac{R\, e / 3^k}{\Delta y_{\mathrm{step}}} \right\rceil \right)

rows (:eq:`ls-row-shift`; :math:`e` the worst case after refining), the last level always on the dty step. On DTL1z45
of ma4752 (:math:`R` = 141 µm, 0.25 µm steps): 0.29° on 0.75 µm voxels, then 0.096° on 0.25 µm. A coarser level sums
the histogram's rows in groups (:func:`~anri.index.data.coarsen_rows`, :func:`~anri.index.data.coarsen_scan`).

From one level to the next, each voxel keeps its orientations holding at least ``min_share`` (1%) of its occupancy,
at most ``per_voxel`` (16), and replaces them by their 27 children, each starting at a 27th of its parent's
occupancy. Each finer voxel fits (MLEM with SQUAREM, :eq:`lik-squarem`) the children its 3 × 3 coarser neighbourhood
held, an orientation held by several neighbours at its largest occupancy there, at most ``cand`` of them.

**Code:** :func:`anri.index.local.level_rows`, :func:`anri.index.local.kept`, :func:`anri.index.local.inherit`,
:func:`anri.index.occupancy.inherit_candidates`, :func:`anri.index.occupancy.coarse_positions`,
:func:`anri.index.occupancy.fit_occupancy`, :func:`anri.index.stages.stage_fit`.

**Anri's choice:** a share, not a cover. Keeping the orientations that together hold 95% of a voxel drops a thin twin
lamella that is a small share of it (more so of a coarse voxel); a share of 1% keeps it. The phantom's twin voxels came
out 98.4% right, against 89.7% for the grid fit followed by a cover-based refinement. Starting the children from their
parents' occupancy matters: started equal, 10 iterations left each grain spread over many children.

``refine_to = 0, levels = 0`` fits the grid orientations as they are, on the dty step: the map a calibration needs
(:doc:`calibration`).

What is left out
----------------

- **Orientations the filter missed**: the refinement and the levels only move what was kept, by up to :math:`s/3` per
  round. A grain the filter dropped stays missing.
- **Orientation spread** within a voxel: a spread wider than the final step is represented by several children, which
  :doc:`populations` groups back into one population with its spread.
- **Why the half scan starves at coarse orientations**: the mechanism is not derived. The refinement removes it; the
  doubled rays themselves stay self-consistent under an orientation error.
