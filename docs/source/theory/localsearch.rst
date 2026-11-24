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
no coupling between voxels). Pricing (below) checks and removes the effect of the first two on the likelihood; the
third remains, and each stays documented here.

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

Is coarse to fine sound?
------------------------

The levels are a way to reach one well-defined answer: the maximum likelihood over the last level's orientations
:math:`\mathcal{D}` (every child of every orientation kept anywhere at the level before), each voxel free to hold any
of them. Whether a fit reached it can be checked, and where it did not, mended, without trusting the path.

**The likelihood is concave.** A recorded bin's term :math:`d\log\mu - \mu` is concave in :math:`\mu`. An empty bin's,
:math:`\log P(n \le c)`, and a saturated bin's, :math:`\log P(n \ge d)`, are logarithms of a Poisson tail, which as
functions of :math:`\mu` are a tail and a cumulative distribution of a gamma variable with a log-concave density
(shape :math:`c + 1 \ge 1`, :math:`d \ge 1`), and so log-concave:

.. math::
    :label: ls-concave

    \frac{\partial^2}{\partial\mu^2}\log Q(c + 1, \mu) \le 0 \iff Q(c + 1, \mu)\left(1 - \frac{c}{\mu}\right) \le p_c(\mu),
    \qquad
    \Gamma(c + 1, \mu) \le \frac{\mu^c e^{-\mu}}{1 - c/\mu} \quad (\mu > c),

with :math:`p_c = \mu^c e^{-\mu}/\Gamma(c + 1)`; the bound follows from :math:`c\log u \le c\log\mu + c(u - \mu)/\mu`
under the integral, and the saturated term likewise. :math:`\mu = Af` is linear, so :math:`\ell = -D/2` is concave in
:math:`f`.

**Proof:** ``tests/theory/test_localsearch.py::test_tails_log_concave``

**The certificate.** With the score of a column :math:`j` (a voxel and an orientation) at a fit,
:math:`S_j = [A^T\rho]_j/[A^T\mathbf{1}]_j` (:math:`\rho` the bins' EM ratios, :eq:`lik-censored`), the gradient is
:math:`\partial\ell/\partial f_j = [A^T\mathbf{1}]_j(S_j - 1)`. A concave function on :math:`f \ge 0` is at its maximum
exactly where its first-order conditions hold, so a fit :math:`\hat f` over the candidates, converged on them, is the
maximum over all of :math:`\mathcal{D}` if and only if

.. math::
    :label: ls-kkt

    S_j(\hat f) \le 1 \quad \text{for every } j \in \mathcal{D} \text{ not a candidate},
    \qquad
    \ell(f) \le \ell(\hat f) + \nabla\ell(\hat f)\cdot(f - \hat f) \le \ell(\hat f) \ \ \forall f \ge 0 .

A recorded bin that the fit leaves empty (:math:`\hat\mu_b = 0 < d_b`) makes the gradient of every column reaching it
infinite: such a column always violates. (MLEM's own ratio is 0 there, since no candidate reaches the bin; pricing
must not use it.) Then the levels, the inherited candidates and the binning have changed nothing in the likelihood: the answer is the
one the full problem has. Only the split among columns the data cannot tell apart (:eq:`ls-start`) still depends on
the path.

**Proof:** ``tests/theory/test_localsearch.py::test_certificate`` (the first-order inequality of a concave function;
both conditions on :math:`f \ge 0`).

**What a missing column would gain.** Adding column :math:`j` alone at :math:`\hat f` and moving it by :math:`t \ge 0`
changes :math:`\ell` by :math:`\phi(t)`, concave, with :math:`\phi'(0) = g_j = [A^T\mathbf{1}]_j(S_j - 1)`; it puts
:math:`N_j = t[A^T\mathbf{1}]_j` expected photons in the data. Concavity bounds the gain from above, and on recorded
bins and plain empty bins (:math:`c = 0`) its curvature, :math:`\phi''(t) = -\sum_b d_b a_{bj}^2/(\hat\mu_b + t
a_{bj})^2 \ge -h_j`, :math:`h_j = \sum_b d_b a_{bj}^2/\hat\mu_b^2`, bounds it from below:

.. math::
    :label: ls-gain

    \Delta D_j(N_j) = 2\phi(t) \le 2(S_j - 1)\,N_j, \qquad
    \max_{t \ge 0} 2\phi(t) \ge \frac{g_j^2}{h_j} \quad (g_j > 0), \qquad
    a_{:,j} = a_{:,i},\ \hat f_i > 0 \;\Rightarrow\; S_j = S_i = 1,\ \Delta D_j = 0 .

So :math:`S_j - 1` is a column's gain per photon, at most: :math:`\max_{j \notin \text{candidates}} S_j - 1 \le \epsilon`
says no orientation outside the candidates could lower the deviance by more than :math:`2\epsilon` per photon it
explains, a bound that holds for any shape of the gain. A duplicate of a column the fit already uses gains nothing,
and a near-duplicate gains to second order in its difference from it; a column that explains light no candidate
explains has :math:`S_j` far above 1 (infinite where the fit leaves a recorded bin empty). Ranking by :math:`S_j` at
the fit, not by the first update from a uniform start (:eq:`lik-candidates`), is what keeps near-duplicates of what is
already fitted out of the slots. The lower bound :math:`g_j^2/h_j` is weak where the fit is far below the data
(:math:`\hat\mu_b \ll d_b` makes :math:`h_j` large at :math:`t = 0` only), which is why the ranking does not use it.

**Proof:** ``tests/theory/test_localsearch.py::test_gain_bound`` (concavity's upper bound; the curvature bound, the
quadratic's maximum; an exact duplicate's score).

**How far from the optimum, at most.** For plain Poisson bins, any :math:`\tilde\rho \ge 0` with
:math:`A^T\tilde\rho \le A^T\mathbf{1}` bounds the maximum, since :math:`d\log\mu \le d\log(d/\tilde\rho) - d +
\tilde\rho\mu` (from :math:`\log y \le y - 1`). With :math:`\tilde\rho = \rho/s_{\max}`, :math:`s_{\max} = \max(1,
\max_{j \in \mathcal{D}} S_j)`, the gap is

.. math::
    :label: ls-gap

    D(\hat f) - D^* \le 2\Big[\Big(\sum_b d_b\Big)\log s_{\max} + \sum_b \hat\mu_b - \sum_b d_b\Big],

exactly 0 when the certificate holds, and rigorous whatever the fit. It is loose (the total counts times
:math:`s_{\max} - 1`), so it bounds the gap where the gains of :eq:`ls-gain` only estimate it.

**Proof:** ``tests/theory/test_localsearch.py::test_gap_bound`` (the inequality per bin; the dual point's
feasibility).

**Code:** :func:`anri.index.occupancy.price`, :func:`anri.index.occupancy.fit_occupancy` (``price_rounds``).

**Pricing.** At the last level, after the fit converges, one pass over every fine voxel and every orientation of
:math:`\mathcal{D}` computes each column's :math:`S_j`: the cost of a candidate pass (:eq:`lik-candidates`), with the
fit's ratio in place of the first update's. Each voxel's non-candidates with :math:`S_j > 1 + \epsilon`
(``price_eps``, default :math:`10^{-4}`), the highest first, take free slots: each round first adds empty slots, so
a candidate the fit still uses is never pushed out, and a slot holding under :math:`10^{-4}` of its voxel is free
(its column is priced again in the next pass). Each new column starts at a tenth of its voxel's mean occupancy (MLEM
cannot move an occupancy from 0; the likelihood being concave, the start changes how fast the fit gets there, not
where). The fit is resumed and priced again, up to ``price_rounds`` times, and the slots are cut back to what the
fullest voxel holds. The log reports the largest :math:`S_j - 1` left outside the candidates, the voxels still
violating, those out of slots, and, for plain Poisson bins, :eq:`ls-gap`. A last pass with no violator is the
certificate :eq:`ls-kkt`, to :math:`2\epsilon` per photon.

Near-duplicates are not all kept out: where the fit leaves light unexplained, every orientation reaching it scores
far above 1, a missing grain's near-duplicate children with it, and a round may spend a voxel's new slots on them.
They split that grain's occupancy along a near-null direction (:eq:`ls-start`), which :doc:`populations` merges back;
the next round's scores, at the fit, no longer favour them.

What remains: :math:`\mathcal{D}` itself is inherited (an orientation below ``min_share`` of every coarse voxel at the
level before has no children in it), and the certificate is relative to :math:`\mathcal{D}`.

What is left out
----------------

- **The couplings above** (binning, inherited support, the start along null directions): pricing makes the
  likelihood independent of the first two, relative to :math:`\mathcal{D}`; the third remains.

- **Orientations the filter missed**: the refinement and the levels only move what was kept, by up to :math:`s/3` per
  round. A grain the filter dropped stays missing.
- **Orientation spread** within a voxel: a spread wider than the final step is represented by several children, which
  :doc:`populations` groups back into one population with its spread.
- **Why the half scan starves at coarse orientations**: the mechanism is not derived. The refinement removes it; the
  doubled rays themselves stay self-consistent under an orientation error.
