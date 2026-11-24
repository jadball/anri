.. _theory_localsearch:

Coarse to fine in orientation and space
=======================================

The search grid (:doc:`search`) puts each orientation up to :math:`\Delta_{\max}` from the truth
(:eq:`srch-cubic-delta`): 0.87° for a 1° cubic grid, 2.6° for 3°
(``tests/theory/test_localsearch.py::test_worst_cases_quoted``). The fit stage refines the filter's orientations
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
:math:`\Delta\omega = -\delta_z - \delta_y \cot\eta` (:eq:`dif-rotation-motion`), up to :math:`e/|\sin\eta|` for an
error up to :math:`e`. The voxel at :math:`(x_s, y_s) = r(\cos\phi, \sin\phi)` is in the beam at
:math:`\text{dty} - y_0 = -r \sin(\omega + \phi)` (:eq:`dif-dty`, beam along :math:`x`), so the predicted row moves by

.. math::
    :label: ls-row-shift

    \Delta y = -r \cos(\omega + \phi)\, \Delta\omega,

**Proof:** ``tests/theory/test_localsearch.py::test_row_shift`` (to first order in :math:`\Delta\omega`).

at most :math:`r|\Delta\omega|`: up to :math:`R\,e` in a sample of radius :math:`R` for a reflection at
:math:`|\sin\eta| = 1`, and :math:`R\,e/|\sin\eta|` for the others (up to 5 times more at ``etacut`` 0.2). Voxels much
narrower than that cannot be fitted consistently with those orientations.

Refining on the row-summed data
-------------------------------

First, without any tomography: each orientation :math:`\matr{U}` from a grid of step :math:`s` is replaced by the 27
rotations of a :math:`3 \times 3 \times 3` grid around it,

.. math::
    :label: ls-children

    \matr{U}_{abc} = \matr{R}(\vec{r}_{abc})\, \matr{U}, \qquad \vec{r}_{abc} = \frac{s}{3}\,(a, b, c),
    \quad a, b, c \in \{-1, 0, 1\},

with :math:`\matr{R}(\vec{r})` the exact rotation by the rotation vector :math:`\vec{r}` about sample axes (the
middle child is :math:`\matr{U}` itself).

**Definition:** the children; ``tests/theory/test_localsearch.py::test_children_code`` checks the code against it. The children are :math:`s/3` apart and their cells (approximately) tile the
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

rows (:eq:`ls-row-shift`; :math:`e` the worst case after refining), the last level always on the dty step.

**Definition:** Anri's voxel widths, sized for reflections at :math:`|\sin\eta| = 1`;
``tests/theory/test_localsearch.py::test_level_rows_code`` checks the code against it.

On DTL1z45 of ma4752 (:math:`R` = 141 µm, 0.25 µm steps), refined to 0.29° (a 1° grid once or a 3° grid twice):
0.29° on 0.75 µm voxels, then 0.096° on 0.25 µm (``tests/theory/test_localsearch.py::test_level_rows_quoted``). A coarser level sums
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

How coarse to fine couples voxels
---------------------------------

No voxel's value is tied to its neighbours' by a prior (:doc:`indexing`). The coarse-to-fine levels still couple them,
in three ways, through what the last level may fit and where it starts. Each is a filter; none is wanted (AGENTS.md:
no coupling between voxels), and each stays documented here until it is removed.

**Filter: coarse levels bin space.** Level :math:`k` fits one set of occupancies per voxel :math:`g_k` rows wide: the
constraint that each :math:`g_k \times g_k` block is uniform. It reaches the last level only through the next two.

**Filter: inherited candidates restrict the support.** A fine voxel may hold only the children of orientations its
3 × 3 coarse neighbourhood held, each at least ``min_share`` of a coarse voxel. A feature filling :math:`n` of a
coarse voxel's :math:`g^2` fine voxels, with nothing of it in the neighbours, holds :math:`n/g^2` of that voxel's
volume at most, so

.. math::
    :label: ls-support

    \text{it can reach the last level only if}\quad \frac{n}{g^2} \ge \texttt{min\_share}
    \qquad (n \ge 1 \text{ for } g \le 10 \text{ at the default } 0.01),

if the coarse fit puts its occupancy where it is. A lamella one fine voxel thick in a coarse voxel 3 rows wide is
:math:`1/3` of it; one fine voxel alone, :math:`1/9`.

**Definition:** the inherited support (a volume fraction; ``tests/theory/test_localsearch.py::test_support_share``
checks the numbers).

**Filter: the start persists where the data do not decide.** MLEM run to convergence (:eq:`lik-converged`) reaches the
maximum likelihood, which (for plain Poisson bins) is unique when its Hessian :math:`-A^T\operatorname{diag}(d/\mu^2)A` is negative definite:
when the columns of :math:`A` over the recorded bins are linearly independent. Where they are not, the likelihood is
flat along a direction, and MLEM keeps its start along it. Two candidates with the same column receive the same factor
at every update, so

.. math::
    :label: ls-start

    A_{:,1} = A_{:,2} \;\Rightarrow\; \frac{f_1^{(n)}}{f_2^{(n)}} = \frac{f_1^{(0)}}{f_2^{(0)}} \quad \text{for every } n:

their split is the start's. Children started from their parents' occupancy keep the coarse voxel's split wherever the
data cannot tell them apart.

**Proof:** ``tests/theory/test_localsearch.py::test_unique_optimum`` (the Hessian),
``tests/theory/test_localsearch.py::test_start_persists`` (two equal columns, every update).

What would remove them, for the maintainer to choose:

- the last level's candidates chosen per fine voxel from every orientation at the last level's resolution
  (:eq:`lik-candidates`), with no inheritance: no binning, no inherited support. Its cost is a candidate pass over
  every fine voxel and every refined orientation, and it brings back the crowding of near-duplicate orientations
  (refined globally to 0.1°, a grain's duplicates filled each voxel's candidates), which needs its own derivation;
- the start's split made visible instead of hidden: where the Fisher information of a voxel's candidates is singular,
  report the occupancies along its null directions as not determined, not the start's values.

What is left out
----------------

- **The couplings above** (binning, inherited support, the start along null directions).

- **Orientations the filter missed**: the refinement and the levels only move what was kept, by up to :math:`s/3` per
  round. A grain the filter dropped stays missing.
- **Orientation spread** within a voxel: a spread wider than the final step is represented by several children, which
  :doc:`populations` groups back into one population with its spread.
- **Why the half scan starves at coarse orientations**: the mechanism is not derived. The refinement removes it; the
  doubled rays themselves stay self-consistent under an orientation error.
