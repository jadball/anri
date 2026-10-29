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

which is zero for a perfect fit. **Code:** :func:`anri.index.occupancy.deviance`.

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

Each update keeps :math:`f \ge 0`, never decreases the likelihood, and needs one forward projection :math:`A f` and one back
projection per iteration, which is all Anri can afford on billions of peaks. **Code:**
:func:`anri.index.occupancy.mlem`, :func:`anri.index.occupancy.forward`, :func:`anri.index.occupancy.backward`.

Accelerating MLEM: SQUAREM
--------------------------

MLEM is slow where it matters: thin features (twins a voxel or two thick) and decoys that borrow intensity from real
grains converge long after the deviance flattens. Anri accelerates it with SQUAREM [VaradhanRoland2008]_: from :math:`f`, two
MLEM updates :math:`f_1` and :math:`f_2`, then the extrapolated point

.. math::
    :label: lik-squarem

    f' = f - 2a\,\vec{r} + a^2\,\vec{v}, \qquad
    \vec{r} = f_1 - f, \quad \vec{v} = f_2 - 2f_1 + f, \quad a = -\frac{\lVert \vec{r} \rVert}{\lVert \vec{v} \rVert},

and one more MLEM update from :math:`f'`. **Anri's choice** of safeguards: :math:`a \le -1`, so a step is never shorter than plain
MLEM; :math:`f'` is kept above :math:`10^{-3} f_2`, since MLEM multiplies and an occupancy at zero could never come back; and if
the result fits worse than :math:`f_2`, :math:`f_2` is kept, so the deviance never rises. One iteration costs three MLEM updates
and goes about as far as ten or more. **Code:** :func:`anri.index.occupancy.squarem`.

Censored and saturated bins
---------------------------

The data are thresholded when they are segmented: only pixels above a cut are kept. A bin with no data is therefore
not a zero: every pixel in it was below the cut. Treating it as zero pulls the model down wherever faint parts of
peaks (the ends of an arc, tails) were cut. With a censoring level :math:`c` (counts per bin), an empty bin

- agrees with the model where the model predicts less than :math:`c` (MLEM ratio 1, no deviance);
- pulls the model towards :math:`c/2` where it predicts more (ratio :math:`c / 2\mu_b`), with deviance :math:`2(\mu_b - c)`.

Likewise a bin holding a saturated pixel recorded at most its true intensity: where the model predicts at least the
data there it agrees; below, the data pull it up as usual.

.. math::
    :label: lik-censored

    \frac{d_b}{\mu_b} \;\to\;
    \begin{cases}
    d_b / \mu_b & d_b > 0 \text{ (and not saturated with } \mu_b \ge d_b) \\
    1 & d_b = 0,\ \mu_b < c, \text{ or saturated with } \mu_b \ge d_b \\
    c / (2\mu_b) & d_b = 0,\ \mu_b \ge c
    \end{cases}

**Anri's choice:** this is a simple stand-in for the likelihood of a censored observation, not that likelihood
itself. :math:`c = 0` gives plain Poisson MLEM, the default. **Code:** :func:`anri.index.occupancy.censored_ratio`,
:func:`anri.index.occupancy.deviance`.

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
:math:`\max(\mu_b - c, 0)`). For an orientation that is not there, :math:`\Lambda_q` is about :math:`\chi^2` with one degree of freedom
[Wilks1938]_, so ``min_lr = 25`` is about 5σ. A small grain with clean spots scores well; a decoy that only borrows
intensity from real grains does not. **Code:** :func:`anri.index.orientations.fit`,
:func:`anri.index.stages.select_by_likelihood`.

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

which scores each orientation by how well its predicted spots are lit, each bin down-weighted by how many
(voxel, orientation) pairs could explain it. Then MLEM with SQUAREM runs on the candidates of every voxel jointly.
**Code:** :func:`anri.index.occupancy.candidates`, :func:`anri.index.occupancy.fit_occupancy`.

**Coarse to fine** (``coarse = G``): the candidate pass and a fit run first on voxels :math:`G` times larger (rows summed in
groups of :math:`G`), and each fine voxel then scores only the :math:`2K` orientations its 3 × 3 coarse neighbourhood occupied
most: about :math:`G^2` times cheaper, at the risk of missing a grain the coarse fit missed. **Code:**
:func:`anri.index.occupancy.inherit_candidates`, :func:`anri.index.occupancy.candidates_from`,
:func:`anri.index.data.coarsen_rows`, :func:`anri.index.data.coarsen_scan`.

What is left out
----------------

- **Regularisation**: none, by design (:doc:`indexing`). MLEM's early stopping (a fixed number of iterations) is the
  only implicit one: noise grows with iterations, as in emission tomography.
- **Uncertainties** of the occupancies: not computed.
- **Voxels outside the mask** (:doc:`populations`) are not fitted; their intensity has to go somewhere, and a mask that
  is too small piles it onto its edge.
