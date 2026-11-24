.. _theory_likelihood:

Fitting occupancies
===================

This chapter fits the occupancies :math:`f` of the linear model (:eq:`idx-linear`) to the histogram, first globally (one
occupancy per orientation, to choose the orientations), then per voxel.

Poisson likelihood
------------------

Each histogram bin is taken as Poisson with mean :math:`\mu_b = (A f)_b`. Maximising the likelihood is minimising the
deviance

.. math::
    :label: lik-deviance

    D(f) = 2 \sum_b \left[ d_b \log\frac{d_b}{\mu_b} - d_b + \mu_b \right],

which is zero for a perfect fit. In each bin its gradient is :math:`2(1 - d_b/\mu_b)`, zero at :math:`\mu_b = d_b`, and
its curvature :math:`2 d_b/\mu_b^2 > 0`.

**Definition:** twice the Poisson log-likelihood ratio to a perfect fit. **Proof** of its gradient:
``tests/theory/test_likelihood.py::test_deviance_gradient``, ``tests/theory/test_likelihood.py::test_deviance_code``.

**Code:** :func:`anri.index.occupancy.deviance`.

**Anri's choice:** the bins hold sums of pixel intensities, normalised to the monitor, not raw photon counts, so the
Poisson model is an approximation: it weights each bin by roughly the inverse of its intensity, which is what makes
weak spots count. Occupancies are non-negative, and Poisson-likelihood fitting keeps them so without constraints.

MLEM
----

The expectation-maximisation algorithm for this model [Dempster1977]_ is the MLEM update of emission tomography
[SheppVardi1982]_ [LangeCarson1984]_:

.. math::
    :label: lik-mlem

    f_{vq} \leftarrow f_{vq}\, \frac{\left[A^T (d / A f)\right]_{vq}}{\left[A^T \mathbf{1}\right]_{vq}}.

The update moves :math:`f_{vq}` by :math:`f_{vq}\, \partial\ell/\partial f_{vq} / [A^T\mathbf{1}]_{vq}`, with :math:`\ell`
the Poisson log-likelihood: its fixed points are the Karush-Kuhn-Tucker points of the maximum likelihood under
:math:`f \ge 0` (:math:`f_{vq}\,\partial\ell/\partial f_{vq} = 0`). After every update the model's total equals the
data's, :math:`\sum_b (Af)_b = \sum_b d_b`. Each update keeps :math:`f \ge 0`, never decreases the likelihood, and needs
one forward projection :math:`A f` and one back projection per iteration, which is all Anri can afford on billions of
peaks.

**Proof:** ``tests/theory/test_likelihood.py::test_mlem_fixed_point_is_kkt``,
``tests/theory/test_likelihood.py::test_mlem_keeps_total``. **Cited:** that it never decreases the likelihood
[Dempster1977]_ [LangeCarson1984]_.

**Code:** :func:`anri.index.occupancy.mlem`, :func:`anri.index.occupancy.forward`,
:func:`anri.index.occupancy.backward`.

Accelerating MLEM: SQUAREM
--------------------------

MLEM is slow where it matters: thin features (twins a voxel or two thick) and decoys that borrow intensity from real
grains converge long after the deviance flattens. Anri accelerates it with SQUAREM [VaradhanRoland2008]_: from :math:`f`, two
MLEM updates :math:`f_1` and :math:`f_2`, then the extrapolated point

.. math::
    :label: lik-squarem

    f' = f - 2a\,\vec{r} + a^2\,\vec{v}, \qquad
    \vec{r} = f_1 - f, \quad \vec{v} = f_2 - 2f_1 + f, \quad a = -\frac{\lVert \vec{r} \rVert}{\lVert \vec{v} \rVert},

and one more MLEM update from :math:`f'`.

**Cited:** [VaradhanRoland2008]_.

**Anri's choice** of safeguards: :math:`a \le -1`, so a step is never shorter than plain MLEM (at :math:`a = -1`,
:math:`f' = f_2`); :math:`f'` is kept above :math:`10^{-3} f_2`, since MLEM multiplies and an occupancy at zero could
never come back; and if :math:`f'` fits worse than :math:`f_1`, :math:`f_2` is kept instead of the update from
:math:`f'`. Either way the next iterate's deviance is at most :math:`D(f_1) \le D(f)`, since an MLEM update never raises
it. After a refused step, :math:`-a` is capped at a quarter of the refused step, the cap growing fourfold after a step
taken at it: uncapped, near convergence :math:`-a` grew to 50-60 and every step was refused. One iteration costs three
MLEM updates and goes about as far as ten or more.

**Proof** of what Anri adds: ``tests/theory/test_likelihood.py::test_squarem_unit_step_is_two_updates``,
``tests/theory/test_likelihood.py::test_squarem_code``.

**Code:** :func:`anri.index.occupancy.squarem`.

Censored and saturated bins
---------------------------

The data are thresholded when they are segmented: only pixels above a cut are kept. A bin with no data is therefore
not a zero: every pixel in it was below the cut. Treating it as zero pulls the model down wherever faint parts of
peaks (the ends of an arc, tails) were cut. Anri takes an empty bin as the observation that its count was at most a
censoring level :math:`c` (counts per bin), and a bin holding a saturated pixel as the observation that its count was
at least its data :math:`d_b`. Their likelihoods are those of a Poisson count so censored, and MLEM's ratio becomes
the count's conditional expectation over the model (the EM step for censored data [Dempster1977]_):

.. math::
    :label: lik-censored

    \frac{d_b}{\mu_b} \;\to\;
    \begin{cases}
    d_b / \mu_b & \text{recorded,} \\
    \operatorname{E}[n \mid n \le c] / \mu_b = 1 - \dfrac{P(n = c)}{P(n \le c)} & \text{empty,} \\[2ex]
    \operatorname{E}[n \mid n \ge d_b] / \mu_b = 1 + \dfrac{P(n = d_b - 1)}{P(n \ge d_b)} & \text{saturated,}
    \end{cases}
    \qquad
    D_b = \begin{cases} -2 \log P(n \le c) & \text{empty,} \\ -2 \log P(n \ge d_b) & \text{saturated,} \end{cases}

with :math:`n \sim \text{Poisson}(\mu_b)`. Each ratio is :math:`1 - \tfrac{1}{2}\, \partial D_b / \partial \mu_b`, the
same relation as a recorded bin's, so the update and the deviance are one objective's: SQUAREM's safeguard compares
the function MLEM climbs. :math:`c = 0` gives a recorded zero (ratio 0, :math:`D_b = 2\mu_b`): plain Poisson MLEM, the
default.

**Proof:** ``tests/theory/test_likelihood.py::test_censored_em``, ``tests/theory/test_likelihood.py::test_censored_ratio_code``

**Code:** :func:`anri.index.occupancy.censored_ratio`, :func:`anri.index.occupancy.deviance`.

The censoring level is a count per bin measured from the data (:func:`anri.index.occupancy.censor_level`), not an
integer, so the probabilities are continued to real levels by the regularised incomplete gamma functions:

.. math::
    :label: lik-censored-gamma

    P(n \le c) = Q(c + 1, \mu), \qquad P(n \ge d) = P(d, \mu), \qquad
    \frac{\partial Q(a, x)}{\partial x} = -\frac{x^{a-1} e^{-x}}{\Gamma(a)},

exact for integer levels, so the hazard :math:`P(n = c)/P(n \le c)` keeps its form with
:math:`P(n = c) = \mu^c e^{-\mu}/\Gamma(c + 1)`.

**Proof:** ``tests/theory/test_likelihood.py::test_censored_gamma``

**Code:** :func:`anri.index.occupancy.censored_below`, :func:`anri.index.occupancy.saturated_above`.

Deep in a tail the regularised gammas underflow (below 1.2e-38 in float32, a model of a few hundred counts in an empty
bin). There, substituting :math:`u = x + t` in :math:`\Gamma(a, x) = \int_x^\infty u^{a-1}e^{-u}\,du` and expanding
:math:`(1 + t/x)^{a-1}` by Taylor's theorem,

.. math::
    :label: lik-censored-tail

    Q(c + 1, \mu) = P(n = c)\, S, \qquad S = \sum_{k \ge 0} \frac{c (c - 1) \cdots (c - k + 1)}{\mu^k},

where :math:`S` is a finite sum (exact) for an integer :math:`c`. For a real :math:`c` it is asymptotic: after
:math:`n \ge c` terms, Taylor's remainder :math:`\binom{c}{n}(1+\xi)^{c-n} (t/x)^n` has :math:`(1 + \xi)^{c - n} \in
(0, 1]`, so the error is the next term times a factor in :math:`(0, 1]`, and the code sums until the terms are below
float precision. For an integer :math:`c` the code uses the exact :math:`\log \sum_{j \le c} \mu^j / j!` (as
:math:`\log P(n \le c) + \mu`, which keeps the large :math:`-\mu` out of every difference between two models: the
likelihood ratio below).

**Proof:** ``tests/theory/test_likelihood.py::test_censored_tail_series``

For a saturated bin below :math:`\mu = d + 10\sqrt{d}`, the regularised lower gamma is computed from its convergent
series

.. math::
    :label: lik-saturated-series

    P(d, \mu) = P(n = d) \sum_{k \ge 0} \frac{\mu^k}{(d + 1) \cdots (d + k)},
    \qquad
    \frac{P(n = d - 1)}{P(n \ge d)} = \frac{d}{\mu\, T},

with :math:`T` the sum (each partial sum's derivative telescopes to the integrand :math:`x^{a-1}e^{-x}/\Gamma(a)` less
its last term). Its terms are positive and, past :math:`k = \mu - d`, fall by ratios :math:`r_k = \mu/(d + k + 1) < 1`
that keep falling, so what is left after a term :math:`t_k` is at most :math:`t_k\, r_k/(1 - r_k)`: the code sums
(compensated) until that is below float precision. The ratio then has no cancellation at all, and :math:`\log P`
only that of :math:`\log P(n = d)` (:eq:`lik-loader`) and of the terms' product recurrence (about one rounding per
term). Above :math:`d + 10\sqrt{d}`, :math:`P` is within :math:`10^{-23}` of 1 and is taken from the regularised gamma.

**Proof:** ``tests/theory/test_likelihood.py::test_saturated_series``, ``tests/theory/test_likelihood.py::test_geometric_remainder``

A saturated bin's :math:`d_b` can be :math:`10^5` counts, where :math:`\log P(n = d - 1) = (d-1)\log\mu - \mu -
\log\Gamma(d)` subtracts numbers of size :math:`10^6` to leave one of order 1: in float32 that ratio was 4.5 % wrong.
Anri computes it in a form without that cancellation,

.. math::
    :label: lik-loader

    \log P(n = k) = -\operatorname{bd0}(k, \mu) - \tfrac{1}{2}\log(2\pi k) - \operatorname{stirlerr}(k),
    \qquad
    \operatorname{bd0} = k \log\frac{k}{\mu} + \mu - k = (k - \mu)\,v + 2k \sum_{j \ge 1} \frac{v^{2j+1}}{2j + 1},

with :math:`v = (k - \mu)/(k + \mu)` (the series when :math:`|v| < 0.1`), and :math:`\operatorname{stirlerr}(k) =
\log\Gamma(k+1) - (k + \tfrac{1}{2})\log k + k - \tfrac{1}{2}\log 2\pi` from Stirling's series [DLMF2026]_ (5.11.1),
whose remainder for a real positive argument is bounded by the first omitted term [DLMF2026]_ (5.11(ii)): below
2.3e-16 for :math:`k \ge 15` with five terms.

**Proof:** ``tests/theory/test_likelihood.py::test_log_pmf_pieces``, ``tests/theory/test_likelihood.py::test_censored_code``

**Anri's choice:** the censoring level per bin stands in for the threshold, which acts per pixel: a bin's hidden counts
are at most :math:`c` only if no spot below the group rule's size hid more (:doc:`recording` has the per-pixel model).
The ratio and the deviance are exact for that observation; the level itself is the approximation.

.. _theory_global_fit:

The global orientation fit and the likelihood ratio
---------------------------------------------------

Before any voxel is fitted, the dty rows are summed and every orientation above chance completeness
(:doc:`search`) gets one global occupancy :math:`g_q`, by the same MLEM with SQUAREM (20 iterations). Like the voxel fit it
is intensity-aware and explains overlapping spots jointly. An orientation's occupancy grows with its grain's volume,
so orientations are kept not by how much there is of them but by how much the fit needs them: the increase in
deviance if that orientation alone were removed, the others fixed,

.. math::
    :label: lik-lr

    \Lambda_q = 2\sum_b \left[ d_b \log\frac{\mu_b}{\mu_b - g_q a_{qb}} - g_q a_{qb} \right],

over observed bins, with :math:`a_{qb}` the orientation's system entries (empty bins add their change in
:math:`D_b` of :eq:`lik-censored`). For an orientation that is not there, :math:`\Lambda_q` is about :math:`\chi^2` with one
degree of freedom [Wilks1938]_, so ``min_lr = 25`` is about 5σ. A small grain with clean spots scores well; a decoy that
only borrows intensity from real grains does not.

**Proof:** ``tests/theory/test_likelihood.py::test_likelihood_ratio_is_deviance_change`` (:math:`\Lambda_q =
D(\mu - g_q a_q) - D(\mu)`), ``tests/theory/test_likelihood.py::test_likelihood_ratio_code``. The code sums each
orientation's system entries (:eq:`idx-bilinear`'s corners of each prediction) per bin first, so entries that share a
bin give one term with :math:`a_{qb}` their sum (the check uses bins coarse enough that they do), and keeps
:math:`\mu_b - g_q a_{qb}` above :math:`10^{-6}\mu_b`, the drop in each term being the clipped model's.

**Code:** :func:`anri.index.orientations.fit`, :func:`anri.index.orientations.select_by_likelihood`.

Two caveats on reading 25 as 5σ. An occupancy cannot be negative, so the null sits on the boundary and its
distribution is half :math:`\chi^2_1`, half zero [Chernoff1954]_: the tail is smaller than assumed. Holding the other
orientations fixed overstates :math:`\Lambda_q` where orientations share spots (refitted, the neighbours would take over some
of the intensity). Treat the threshold as a ranking with a sensible scale, not an exact significance.

**Anri's choice:** the ratio is computed with :math:`\log(1 + x)` rather than :math:`\log` of a ratio near 1: in single precision
the ratio for a weak orientation rounds to :math:`1 +` a few ulps, and times counts of :math:`10^6` that was noise of order 1 that
changed from run to run.

The voxel fit
-------------

The voxel fit has too many unknowns (voxels × kept orientations) to hold. Each voxel keeps :math:`K` candidates (64 by
default): those with the largest first MLEM update from unit occupancy,

.. math::
    :label: lik-candidates

    f^{(1)}_{vq} = \frac{\left[A^T\left(d / A\mathbf{1}\right)\right]_{vq}}{\left[A^T\mathbf{1}\right]_{vq}},

the MLEM update :eq:`lik-mlem` from :math:`f = \mathbf{1}`. It scores each orientation by how well its predicted spots
are lit, each bin down-weighted by how many (voxel, orientation) pairs could explain it.

**Proof:** ``tests/theory/test_likelihood.py::test_candidates_are_first_update``. Then MLEM with SQUAREM runs on the candidates of every voxel jointly.

**Code:** :func:`anri.index.occupancy.candidates`, :func:`anri.index.occupancy.fit_occupancy`.

**Coarse to fine**: the fit stage runs this fit level by level, the first on voxels sized to the orientations' error
and with every orientation as a candidate, the later ones on finer voxels and orientations, each voxel over what its
coarser neighbourhood held (:doc:`localsearch`).

A coarse row is the sum of :math:`g` rows: its dty their mean, its voxels :math:`g` times wider, the beam's flat top
:math:`(g - 1)` steps wider, its exposure the group's mean and its offset the group's exposure-weighted mean real dty
less its own (``tests/theory/test_likelihood.py::test_coarsen_rows_code``,
``tests/theory/test_likelihood.py::test_coarsen_scan_code``).

**Code:** :func:`anri.index.occupancy.inherit_candidates`, :func:`anri.index.data.coarsen_rows`,
:func:`anri.index.data.coarsen_scan`.

When to stop
------------

MLEM's update is the log-likelihood's gradient in a metric. With :math:`\ell = -D/2` (recorded, empty and saturated
bins alike, since every ratio of :eq:`lik-censored` is :math:`1 - \tfrac{1}{2}\partial D_b/\partial\mu_b`),
:math:`\partial\ell/\partial f_j = [A^T(\rho - 1)]_j` and the update moves :math:`f` by :math:`\Delta f =
\matr{M}^{-1}\nabla\ell`, :math:`\matr{M} = \operatorname{diag}([A^T\mathbf{1}]_j / f_j)`. So the deviance's
first-order decrease along one update is

.. math::
    :label: lik-converged

    \delta = -\nabla D \cdot \Delta f = 2\sum_j [A^T\mathbf{1}]_j\, \frac{\Delta f_j^2}{f_j} \ge 0,
    \qquad
    D - D^* = \frac{\delta}{2(1 - \rho)}, \quad \frac{\delta'}{\delta} = \rho^2
    \quad\text{(one mode)},
    \qquad
    D - D^* \le \frac{\sum_m \delta_m}{2(1 - \rho_{\max})}.

Near the optimum the deviance is quadratic, :math:`D - D^* = \vec{e}^T\matr{H}\vec{e}`, and MLEM contracts each mode
(an eigenvector of :math:`\matr{M}^{-1}\matr{H}`) by :math:`\rho`. Then :math:`\delta` is the deviance a mode has
left times :math:`2(1 - \rho)`, and two consecutive updates give :math:`\rho` from the ratio of their
:math:`\delta`. :math:`\delta` is a sum of non-negative terms: unlike the difference of two deviances (sums of
:math:`10^6` bins in float32), it has no cancellation.

**Proof:** ``tests/theory/test_likelihood.py::test_mlem_is_metric_gradient`` (the update is :math:`\matr{M}^{-1}\nabla\ell`
for the censored deviance; :math:`\delta` is the directional derivative), ``tests/theory/test_likelihood.py::test_remaining_deviance``
(one mode exactly; several modes bounded by the slowest), ``tests/theory/test_likelihood.py::test_remaining_deviance_code``.

**Code:** :func:`anri.index.occupancy.step_decrease`, :func:`anri.index.occupancy.remaining_deviance`,
:func:`anri.index.occupancy.squarem`.

Each SQUAREM iteration makes two plain updates, :math:`f \to f_1 \to f_2`, so each gives :math:`\rho^2 =
\delta_1/\delta_0` and the remaining deviance :math:`\hat{G} = \delta_1 / (2(1 - \rho))` at :math:`f_1`, which the
iteration's result improves on. The fit stops once :math:`\hat{G} \le` ``tol`` and the deviance itself fell by at most
``tol`` (or eight float32 ulps of it, where that is larger) over the iteration, on two iterations running, by
default 1: a log-likelihood
ratio of 1/2 to the optimum, which no test can tell from it (:eq:`lik-lr`'s threshold is 25). The cost is one more
reduction per update, brought to the host with the deviance in one transfer.

**Anri's choice:** :math:`\hat{G}` is an estimate, not a bound. With several modes the measured :math:`\rho^2` is
the modes' mean weighted by their :math:`\delta`, below the slowest's, so the slowest mode's remainder is
underestimated, by up to the factor :math:`(1 - \rho)/(1 - \rho_{\max})`. It is worst just after a start, when the
fast modes dominate :math:`\delta`, and along near-null directions (near-duplicate orientations, :eq:`ls-start`),
where :math:`\rho_{\max} \to 1`. On a half-sample phantom it swung from 6 to 251 between successive iterations while
the deviance fell by tens to hundreds per iteration. Hence the deviance's own fall as a second condition, two
iterations running, and, after pricing, the rigorous bound :eq:`ls-gap` beside it. SQUAREM's extrapolation changes the
contraction from iteration to iteration, but the two plain updates that measure it are plain MLEM. The deviance
counts in the histogram's units (monitor-normalised sums of counts: the Poisson approximation above).

**Filter:** a fit that reaches its iteration cap (``n_iter``) before :math:`\hat{G} \le` ``tol`` has stopped early. Early
stopping keeps the well-determined part of the occupancies and leaves the rest near its start: an implicit spectral
filter, as in emission tomography, where noise grows with iterations. Each fit logs whether it converged, in how
many iterations and :math:`\hat{G}`; the log is in the stage's provenance. The cap is a guard on cost, not a choice
of regularisation: raise it, never lower it, until every level converges.

What is left out
----------------

- **Regularisation**: none by design (:doc:`indexing`); an iteration cap reached before convergence is a filter
  (above).
- **Uncertainties** of the occupancies: not computed.
- **Voxels outside the mask** (:doc:`populations`) are not fitted; their intensity has to go somewhere, and a mask that
  is too small piles it onto its edge.
- **Saturated bins' precision.** The ratio is exact to a few roundings at any count; :math:`\log P(n \ge d)` to about
  one rounding per series term (in float32 at most 1e-5 relative, at :math:`d = 300`, :math:`\mu = 1.3 d`), which
  only the deviance SQUAREM compares sees.
