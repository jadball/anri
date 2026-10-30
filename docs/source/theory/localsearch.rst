.. _theory_localsearch:

Sharpening orientations off the grid
====================================

The voxel fit (:doc:`likelihood`) uses the orientations of the search grid (:doc:`search`), so each is up to
:math:`\Delta_{\max}` from the truth (:eq:`srch-cubic-delta`): 0.87° for a 1° cubic grid. This chapter refines them, voxel
by voxel, without a finer grid over the whole fundamental zone.

Why the grid alone is not enough
--------------------------------

Where peaks are sharper than the grid, the fit explains a grain's peaks with several grid neighbours, and with
orientations that share some of its reflections. A twin shares some reflections with its parent. If the grid point
nearest the exact twin of a grain lands closer to it than the grain's own grid point lands to the grain, that twin point
explains the shared reflections better than the grain does, and takes occupancy where there is no twin: a ghost twin. A
finer grid everywhere would remove it, at a cost growing as the cube of the refinement.

Subdividing around what each voxel uses
---------------------------------------

Each level of the local search replaces an orientation :math:`\matr{U}` from a grid of step :math:`s` by the 27
rotations of a :math:`3 \times 3 \times 3` grid around it,

.. math::
    :label: ls-children

    \matr{U}_{abc} = \matr{R}(\vec{r}_{abc})\, \matr{U}, \qquad \vec{r}_{abc} = \frac{s}{3}\,(a, b, c),
    \quad a, b, c \in \{-1, 0, 1\},

with :math:`\matr{R}(\vec{r})` the exact rotation by the rotation vector :math:`\vec{r}` about sample axes (the
middle child is :math:`\matr{U}` itself). The children are :math:`s/3` apart and their cells (approximately) tile the
parent's, so after :math:`L` levels the step is :math:`s/3^L`: two levels take a 1° grid to 0.11°.

**Code:** :func:`anri.index.local.children`.

Only the orientations some voxel uses are subdivided: each voxel's most occupied ones that together hold ``cover``
(0.95) of its occupancy, at most ``per_voxel`` (8). Each voxel is then fitted again (MLEM with SQUAREM, as the voxel
fit, :eq:`lik-squarem`) over the children of its *own* orientations only, at most :math:`27 \times` ``per_voxel``,
each child starting at a 27th of its parent's occupancy.

**Code:** :func:`anri.index.local.used`, :func:`anri.index.local.inherit`,
:func:`anri.index.occupancy.fit_occupancy`, :func:`anri.index.stages.stage_localsearch`.

**Anri's choice:** occupancy only moves within each voxel's own orientations. Ranking every child of every orientation
for every voxel, as the voxel fit ranks its candidates (:eq:`lik-candidates`), lets a grain's many children crowd out a
thin twin's, and the twin lamella is lost. Each level uses the voxel fit's options (iterations, censoring, beam profile,
absorption, mask).

What is left out
----------------

- **Orientations the voxel fit missed**: the local search refines what each voxel already uses, and adds nothing
  further than :math:`s/3` per level from it. A grain the voxel fit gave no occupancy stays missing.
- **Orientation spread** within a voxel: a spread wider than the final step is still represented by several children,
  which :doc:`populations` groups back into one population with its spread.
