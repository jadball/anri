.. _theory_refiner:

A refiner that needs no spot shape
==================================

The refiner's task is each population's orientation and strain, from the recorded cells (:doc:`recording`), with
no model of any spot's shape. This chapter derives one: its observables are exactly linear in shape-free unknowns.
What noise, the cut, the group rule, the pixels and the instrument do to it is proved, term by term, as an identity
or a bound computed from the geometry and the data. Where no bound can exist, the chapter proves that too.

*Cell* means, as in :doc:`recording`, one pixel in one frame of one dty row.

Assumptions
-----------

Nothing is assumed about the shape of any spot, about a population's orientation or strain distribution, or about
the shape of the detector's point spread beyond (A5). The assumptions are:

- **(A1) Counting.** Counts are Poisson and independent over cells given the light (:eq:`rec-poisson`); they are
  recorded by the cut and the group rule (:eq:`rec-recorded`).
- **(A2) Kinematic scattering.** Every photon of a sub-volume in reflection :math:`h` has the scattering vector
  :math:`\vec{g} = \matr{F}\vec{g}_0` of that sub-volume. Here :math:`\vec{g}_0 = \matr{U}_k\matr{B}\vec{h}` is
  fixed per population :math:`k` (the indexer's orientation), and :math:`\matr{F}` is the sub-volume's
  reciprocal-lattice deformation: any 3 × 3 matrix, its rotation and its strain together.
- **(A3) Integrated intensity.** Over a complete rotation through the reflection, a sub-volume of volume
  :math:`dV` gives :math:`a_{bk}\,|F_h|^2 A_{kh}\, L(\vec{g}) P(\vec{g})\, dV` photons. :math:`a_{bk}` is the beam's
  weight of the population's voxel in the frames of dty row :math:`b`, :math:`|F_h|^2` the structure factor and
  :math:`A_{kh}` the transmission (:doc:`intensity`). :math:`L` and :math:`P` are the Lorentz and polarisation
  factors at the sub-volume's own diffraction (:eq:`int-lorentz`, :eq:`int-pol`).
- **(A4) The instrument.** The incoming wavevector's direction and the wavelength vary from photon to photon,
  independently of the sample. The detector moves each photon by a displacement independent of the sample (the
  point spread; the sensor's depth, up to its mean, which the geometry holds).
- **(A5) The point spread is a factor.** Every spot is the point spread convolved with something non-negative and
  otherwise arbitrary: the sample, the beam and the sensor add spread, never remove it. Only its characteristic
  function at :math:`2\pi k` per pixel is used (:eq:`rf-alias-bound`).

The geometry (detector, beam profile across dty, wavelength) is known from calibration (:doc:`calibration`).

The estimand
------------

A population :math:`k` is a distribution :math:`\mu_k` of volume over :math:`\matr{F}` (total :math:`V_k`). Its
**deformation moments** are

.. math::
    :label: rf-moments

    Y_k^{(0)} = V_k, \qquad
    \matr{Y}_k^{(1)} = \int \matr{F}\, d\mu_k, \qquad
    \matr{Y}_k^{C} = \int \matr{F}^T\matr{F}\, d\mu_k, \qquad
    \matr{Y}_k^{(2)} = \int \matr{F} \otimes \matr{F}\, d\mu_k,

**Definition:** the deformation moments.

By (A2), for every reflection and whatever :math:`\mu_k`,

.. math::

    \int \vec{g}\, d\mu_k = \matr{Y}_k^{(1)}\vec{g}_0, \qquad
    \int |\vec{g}|^2\, d\mu_k = \vec{g}_0^T\matr{Y}_k^{C}\vec{g}_0, \qquad
    \int \vec{g}\vec{g}^T d\mu_k = \matr{Y}_k^{(2)}[\vec{g}_0, \vec{g}_0].

**Proof:** ``tests/theory/test_refiner.py::test_photon_moments_linear`` (linearity of the integral; any number of
sub-volumes, symbolic).

These are the quantities a shape-free refiner determines: 1 + 9 + 6 numbers per population (45 more with the second
moments). They are defined by the sample alone. The mean deformation is :math:`\matr{Y}^{(1)}/V`. The mean of
:math:`\matr{F}^T\matr{F}`, :math:`\matr{Y}^C/V`, is the population's mean metric: its strain, exactly, whatever its
rotations (:eq:`rf-invariant`). The spread is :math:`\matr{Y}^{(2)}/V - (\matr{Y}^{(1)}/V)^{\otimes 2}`. Turning
these into a mean rotation and a mean strain is a separate, exact step with its own second-order terms
(:eq:`ref-spread-mean`); it is not part of the fit.

The statistics
--------------

For a region :math:`R` of cells (below) and a feature :math:`f`, the statistic is the recorded values summed with each
cell's own Lorentz and polarisation factor divided out:

.. math::
    :label: rf-statistic

    T_R[f] = \sum_{p \in R} \frac{r_p\, f(\hat{\vec{g}}_p)}{L(\hat{\vec{g}}_p)\, P(\hat{\vec{g}}_p)},
    \qquad f \in \{1,\ \vec{g},\ |\vec{g}|^2,\ \vec{g}\vec{g}^T\}.

Here :math:`\hat{\vec{g}}_p` is the scattering vector, in the sample frame, of the cell's centre at its frame's
centre :math:`\omega_j`, computed from the lab origin with the nominal wavelength and beam.

**Definition:** the refiner's statistics.

Why the factors are divided out per cell. A population's photons in a reflection are weighted by the Lorentz factor
of each sub-volume (A3). That factor varies across the population with the sub-volume's rotation :math:`\vec{d}`,
:math:`L \propto 1 + \vec{l}^T\vec{d}`, so the light's mean is not the volume's:

.. math::
    :label: rf-weight

    \operatorname{E}\left[(1 + \vec{l}^T\vec{d})(\matr{I} + [\vec{d}]_\times)\right]
    = \matr{I} + [\matr{S}\vec{l}]_\times + O(d^3), \qquad \matr{S} = \operatorname{Cov}(\vec{d}),

with :math:`|\vec{l}| \approx |\cot\eta|`. That is a turn of :math:`|\matr{S}\vec{l}| \sim \sigma^2|\cot\eta|`:
:math:`1.49 \times 10^{-3}` rad for a 1° spread at :math:`|\sin\eta| = 0.2`, against orientation contrasts
of :math:`10^{-4}` rad. Each photon's factor depends only on where it diffracts, so dividing every cell by its own
:math:`L P` makes every reflection weight a population's sub-volumes by their volume alone. The cell's position
stands in for the photon's, which leaves a second-order remainder (the factors' gradient times the cell's width).

**Proof:** ``tests/theory/test_refiner.py::test_weighted_mean``

A weight that depends on the sub-volume but cannot be divided out per cell (the transmission's variation with the
diffracted direction, which depends on the voxel) enters the same way: as a known coefficient times
:math:`\matr{Y}^{(2)}`. So the model stays linear.

Rotation invariance
-------------------

For every rotation :math:`\matr{R}` and every :math:`\matr{F} = \matr{R}\matr{U}`,

.. math::
    :label: rf-invariant

    |\matr{R}\matr{U}\vec{g}_0|^2 = \vec{g}_0^T\matr{U}^T\matr{U}\vec{g}_0, \qquad
    \hat{\vec{g}}_p = \matr{R}_z(\omega_j - \omega)\,\vec{g}\ \Rightarrow\ |\hat{\vec{g}}_p| = |\vec{g}|.

The sum of :math:`|\vec{g}|^2` therefore sees a population's strain and nothing of its orientation distribution. The
frames' binning in omega changes only a rotation about the axis, so it changes :math:`|\vec{g}|` not at all. The
strain observable :math:`T_R[|\vec{g}|^2]` is exact through any orientation spread and any frame width.

**Proof:** ``tests/theory/test_refiner.py::test_rotation_invariant``

The instrument
--------------

By (A4), a photon's measured vector is :math:`\hat{\vec{g}} = s(\vec{g} + \vec{D})`. The scale :math:`s =
\lambda/\lambda_0` (the wavelength) and the shift :math:`\vec{D}` (the incoming wavevector less its mean) are
independent of :math:`\vec{g}`, with :math:`\operatorname{E}[\vec{D}] = 0`. Then, exactly,

.. math::
    :label: rf-instrument

    \operatorname{E}[\hat{\vec{g}}] = \operatorname{E}[s]\operatorname{E}[\vec{g}], \qquad
    \operatorname{E}|\hat{\vec{g}}|^2 = \operatorname{E}[s^2]\left(\operatorname{E}|\vec{g}|^2 + \operatorname{E}|\vec{D}|^2\right).

Each is a factor and an offset common to every population: a scale fixed by calibration, or left free as one number
for the whole map (:eq:`lim-common`: it biases only region means, like :math:`d_0`).

**Proof:** ``tests/theory/test_refiner.py::test_instrument_independent``

The independence holds because every sub-volume rotates through the whole spectrum and every incoming direction:
the frames integrate the rotation, so a region holding all of a population's frames in a reflection samples them
all. The detector's displacement enters through :math:`\hat{\vec{g}}(\vec{x}_p)` at the displaced position. Its mean
is zero (the point spread, by definition) or calibrated (the sensor's), and its second moments give a term
:math:`\tfrac{1}{2}\operatorname{tr}(\matr{H}_f \matr{\Sigma}_e)`, with :math:`\matr{H}_f` the feature's Hessian
on the detector. It is the same for every population at one place on the detector, and smooth across it.

**Parallax.** :math:`\hat{\vec{g}}_p` is computed from the lab origin. A photon from a voxel at :math:`\vec{x}_v`
reaches the cell that the origin's ray :math:`\Pi_v(\vec{g})` would. :math:`\Pi_v` is a known smooth map, the
identity plus :math:`O(|\vec{x}_v|/L)`, with :math:`L` the detector's distance. Expanded about
:math:`\vec{g}_0`, :math:`\operatorname{E}[f(\Pi_v(\vec{g}))]` is linear in :math:`\matr{Y}^{(1)}` and
:math:`\matr{Y}^{(2)}` with known coefficients. The remainder is the third order: :math:`|\vec{x}_v|/L` times the
spread squared.

The linear model and its estimator
----------------------------------

Regions and features are stacked into one vector :math:`\vec{T}`, and the populations' moments (:eq:`rf-moments`)
into :math:`\vec{Y}`. By (A1)-(A5) and :eq:`mom-raysum`, the expected statistic of a region holding all of its
populations' light is a ray sum:

.. math::
    :label: rf-linear

    \vec{T} = \matr{G}\vec{Y} + \vec{\Delta} + \vec{\xi}, \quad \operatorname{E}[\vec{\xi}] = 0;
    \qquad
    \hat{\vec{Y}} = \matr{A}\vec{T},\ \matr{A}\matr{G} = \matr{I}
    \ \Rightarrow\
    \operatorname{E}[\hat{\vec{Y}}] - \vec{Y} = \matr{A}\vec{\Delta},\quad
    |(\matr{A}\vec{\Delta})_i| \le \sum_j |A_{ij}|\,\bar\Delta_j .

:math:`\matr{G}` is built from geometry alone: :math:`a_{bk}|F_h|^2 A_{kh}`, the powers of :math:`\vec{g}_0` and
the parallax coefficients. :math:`\vec{\Delta}` collects everything (A1)-(A5) do not make linear: the cut, the group
rule, the pixels' binning and light outside the region, each bounded below by :math:`\bar\Delta_j`. The identity
holds for any distribution of the noise :math:`\vec{\xi}`: the noise biases nothing. :math:`\matr{A}` is any left
inverse, typically weighted least squares; its weights change efficiency, never the identity (:eq:`ref-bias`). The
residuals :math:`\vec{e} = (\matr{I} - \matr{H})\vec{T}`, :math:`\matr{H} = \matr{G}\matr{A}`, have
:math:`\operatorname{E}[\vec{e}\vec{e}^T] = (\matr{I} - \matr{H})(\matr{\Sigma} + \vec{\Delta}\vec{\Delta}^T)
(\matr{I} - \matr{H})^T`.

**Proof:** ``tests/theory/test_refiner.py::test_linear_estimator``

The model is linear in the unknowns; no linearisation in the deformation is made. What stays non-linear is which
cells form each region, a discrete function of where the populations' spots are. It is fixed for an iteration, as
the window origins are (:doc:`peak_shape`), and re-made from the new estimates.

**Regions.** A population's box in a reflection is the set of cells within half-widths :math:`(t_s, t_f, t_\omega)`
of its predicted spot centre (from :math:`\matr{Y}^{(1)}`). The half-widths are a size, fixed from the population's
measured second moments, never a shape. A region is a connected union of boxes, so every population whose box meets
it is in it, with its whole box. Separated populations have regions of their own; merged ones share a region and
enter it as a ray sum. Which voxel along a ray holds which part of the light is never decided within a region. It is
decided by the tomography: by the same populations' regions in other rows and other reflections.

What the regions estimate. The estimand is each population's moments of its light *within its box*. For light
symmetric about the box's centre it is the moment of all the light. In general it differs by the light beyond the
box. For a spot of variance :math:`\sigma^2` that difference is at most :math:`\sigma^2/t` in the first moment, per
axis, for any shape (:eq:`ref-tail`); otherwise it depends on the shape, and no estimator can know it. Light beyond
every box that lies above the cut is recorded in cells outside every region; it is reported, never fitted.

The pixels
----------

The data are binned, so :math:`\hat{\vec{g}}_p` is the cell's centre, not the photon's position: a rounding
:math:`u = \operatorname{round}(x) - x` per photon and axis. Under (A5), a spot's characteristic function is
:math:`\chi = \chi_{\text{psf}}\,\chi_\nu` with :math:`|\chi_\nu| \le 1`, so for any spot,

.. math::
    :label: rf-alias-bound

    |\operatorname{E}[u]| \le \min\left(\tfrac{1}{2},\ \sum_{k \ge 1} \frac{|\chi_{\text{psf}}(2\pi k)|}{\pi k}\right),
    \qquad |\operatorname{E}[\sin 2\pi k x]| \le |\chi(2\pi k)|,

from the sawtooth's series (:eq:`det-pixel-moments`), symmetric or not. The bound is 0.0676 px for the measured point
spread (a core of 0.25 px holding 72 %, a wing of 1.3 px). A feature's first-order binning error is its gradient times
:math:`\operatorname{E}[u]`, and :math:`\bar\Delta` takes it per region with the gradient's largest value there.

**Proof:** ``tests/theory/test_refiner.py::test_alias_bound_any_spot`` (the bound; asymmetric spots, the point spread
convolved with exponentials of either sign, against their cells' masses).

In omega there is no factor like (A5) unless the beam's angular spread is measured as one: a peak inside one frame
is placed within it only to :math:`\pm h/2`. This is the bound of :eq:`det-crlb`: without a shape, a cell's content
says nothing of where in the cell its light fell. It does not touch :math:`|\vec{g}|^2` (:eq:`rf-invariant`), only the
orientation about the rotation axis.

The cut
-------

No estimator can remove the cut's bias:

.. math::
    :label: rf-no-unbiased

    c \ge 1:\quad \nexists\, h \text{ with } \operatorname{E}\big[h(r_1, r_2, \dots)\big] = \sum_p f_p \lambda_p
    \text{ for all } \lambda \ge 0, \text{ unless every } f_p = 0.

In :math:`e^{\sum\lambda}\operatorname{E}[h]`, the coefficient of :math:`\lambda_p` alone comes from a count of one in
cell :math:`p`, which is recorded as 0. So it equals the constant term, :math:`h(0, 0, \dots)`. The constant term must
be 0, and the coefficient of :math:`\lambda_p` must be :math:`f_p`.

**Proof:** ``tests/theory/test_refiner.py::test_no_unbiased_estimator`` (one cell, :math:`c = 1, 2, 3`; two cells,
:math:`c = 1`).

So a region's recorded sum misses the light the cut removes, and the miss can only be bounded. Per cell it is the
light below the cut, :math:`\lambda P(n \le c - 1)` (:eq:`mom-deficit`), and whatever :math:`\lambda`,

.. math::
    :label: rf-cut-bound

    0 \le \lambda P(n \le c - 1) \le \kappa_c, \qquad \kappa_1 = e^{-1}, \quad \kappa_2 = \phi^3 e^{-\phi} = 0.840,

(:math:`\phi` the golden ratio). The bias of a region's feature about a population's mean :math:`m` is therefore at
most :math:`\kappa_c \sum_{p \in R} |f_p - m| / (L P)_p`. That depends only on the region's cells, not on the light.
For a box of 7 × 7 pixels and 3 frames (with :math:`LP` constant over it, and :math:`N` counted in its units),
:math:`\sum |x - m| = 252` cell-pixels along one axis. At :math:`c = 1` a first moment of :math:`N` photons is off by at most :math:`92.7/N` px: 0.0093 px at :math:`10^4`. Its noise is
:math:`\sigma/\sqrt{N}`, so the bound is below the noise once :math:`N \ge (92.7/\sigma)^2`: 8600 photons for a
spot of :math:`\sigma = 1` px.

**Proof:** ``tests/theory/test_refiner.py::test_cut_deficit_bound``

The bound falls as :math:`1/N` and the noise as :math:`1/\sqrt{N}`. Faint spots are therefore the ones whose cut bias
can exceed their noise, by a factor this bound gives. The bound is tight in the worst case: light of one photon per
cell over the whole box. Smaller boxes lower it and raise the light beyond the box (:eq:`ref-tail`); the half-widths
trade the two.

The group rule
--------------

Every kept neighbour of a kept pixel is in its group, so :math:`|G_p| \ge 1 +` (its kept neighbours) on every
pattern. The probability of a small group then falls in each neighbour's :math:`a_q`. So if every neighbour of
:math:`p` is kept with probability at least :math:`a_*`,

.. math::
    :label: rf-group

    1 - w_p = P(|G_p| < m \mid p \text{ kept}) \le \sum_{j \le m - 2} \binom{8}{j} a_*^j (1 - a_*)^{8 - j}
    \quad (m = 3:\ (1 - a_*)^8 + 8 a_* (1 - a_*)^7),

:math:`8.2 \times 10^{-5}` when each neighbour holds 3 photons (:math:`a_* = P(n > 1)` at :math:`\lambda = 3`). The
group rule's loss is then :math:`e_p(1 - w_p)` per cell (:eq:`mom-deficit`).

**Proof:** ``tests/theory/test_refiner.py::test_group_rule_bound`` (the pattern inequality on every pattern checked,
through :func:`anri.io.imaged11.segment`; the derivative; the number; and the lone pair).

Without a bound on the neighbours' light there is none on the loss: two adjacent bright pixels with dark
neighbours are removed whole at :math:`m = 3`, however bright. (A5) rules such spots out only as far as the point
spread puts light in the neighbours: a pixel whose neighbours each get at least a share :math:`q` of its light meets
:eq:`rf-group`'s condition once it holds :math:`3/q` photons. The share :math:`q` is the point spread's, and is not
derived here. Spots fainter than that, and compact, are the regime of :doc:`recording`'s table of compact spots.
There the group rule's loss depends on the light's position within its pixel, and this refiner can only report the
bound, not correct it.

Noise and error bars
--------------------

Each recorded value is a non-decreasing function of every cell's count: a kept pixel stays kept with more counts, and
more kept pixels only join groups. The counts are independent, so by Harris' inequality (the one-variable step,
:math:`2\operatorname{Cov}(f, g) = \operatorname{E}[(f(X) - f(Y))(g(X) - g(Y))] \ge 0`, applied one count at a time
by conditioning)

.. math::
    :label: rf-harris

    \operatorname{Cov}(r_p, r_q) \ge 0 \quad \text{for all } p, q,

and :math:`\operatorname{Cov}(r_p, r_q) = 0` beyond the :math:`(4m - 3) \times (4m - 3)` stencil (:doc:`recording`).
A sum with non-negative weights therefore has a variance at least the sum of its cells' variances: treating cells, or
regions closer than :math:`2(m - 1)` pixels in a frame, as independent understates every error bar, never overstates
it.

**Proof:** ``tests/theory/test_refiner.py::test_recorded_values_increasing`` (monotonicity through
:func:`anri.io.imaged11.segment`, every pattern of counts 0-3 on a strip, :math:`m = 1, 2, 3`; the one-variable
identity).

Error bars that need no noise model come from the residuals (:eq:`rf-linear`). Group the regions into blocks
independent of each other: connected components of the graph joining regions of one frame and dty row that are
within :math:`2(m - 1)` pixels, so different frames, rows and far regions are independent exactly, and
:math:`\matr{\Sigma}` is block-diagonal. The estimate's covariance is then :math:`\sum_b
\matr{A}_{:b}\matr{\Sigma}_{bb}\matr{A}_{:b}^T`. The block sandwich :math:`\sum_b
\matr{A}_{:b}\vec{e}_b\vec{e}_b^T\matr{A}_{:b}^T` puts the block of :eq:`rf-linear`'s residual covariance,
:math:`[(\matr{I} - \matr{H})(\matr{\Sigma} + \vec{\Delta}\vec{\Delta}^T)(\matr{I} - \matr{H})^T]_{bb}`, in place
of :math:`\matr{\Sigma}_{bb}`. So it errs large by the bias and small by the leverage :math:`\matr{H}`, whose share
falls as the number of unknowns over the number of blocks. It is consistent as the blocks per unknown grow, with no
model of the noise.

The guarantee
-------------

Under (A1)-(A5), with regions fixed independently of the noise of the data they are applied to:

1. **Noise** biases nothing (:eq:`rf-linear`). Error bars come from blocks that are independent exactly
   (:eq:`rf-harris`), with no noise model.
2. **Spot shape** enters nowhere in :math:`\matr{G}`. It enters only through the bounded terms 3-5 and the
   estimand's definition (light within each box).
3. **The cut** biases each region's statistics by at most :math:`\kappa_c \sum_R |f_p - m|/(LP)_p`
   (:eq:`rf-cut-bound`), a number of the geometry. No estimator can do better than bound it (:eq:`rf-no-unbiased`).
4. **The group rule** biases by at most :eq:`rf-group` where every lit cell's neighbours hold enough light, and by an
   unbounded amount otherwise: the faint compact spots of :doc:`recording`.
5. **The pixels** bias a feature by at most its gradient times :eq:`rf-alias-bound`. Omega binning biases nothing in
   the strain (:eq:`rf-invariant`) and up to half a frame in the rotation about the axis.

The refiner reports, per population, :math:`\hat{\vec{Y}}`, its block-sandwich covariance, and the bias bound
:math:`\sum_j |A_{ij}|\bar\Delta_j`. That bound is computed from the geometry, the regions and the point spread's
:math:`\chi(2\pi k)`, not from any spot shape. Where the bound exceeds the error bar, the estimate is limited by what
the data cannot say, and the report says so.

What it costs, against :doc:`limits`:

- Information within a cell (the pixels' sub-cell information for spots narrower than about 0.3 px,
  :eq:`det-crlb`) is unused, since only a shape could use it.
- Merged populations are determined only through the tomography of their ray sums. That is the Fourier slice's limit,
  with its weak combinations (:eq:`lim-seen-weak`); the bias terms reach the map along those combinations amplified as
  :doc:`limits` says (per-voxel errors by up to :math:`1/\sin^2\theta`).

What is left out
----------------

- **Second-order remainders** of the per-cell :math:`LP` division, the detector displacement's common term and
  parallax: stated in size, not bounded rigorously.
- **The box half-widths**: their trade between :eq:`rf-cut-bound` and :eq:`ref-tail` is not optimised here.
- **Absorption's variation within a spot** enters through :math:`\matr{Y}^{(2)}` (above); its coefficient is not
  written out.
- **A bound in omega** needs the beam's angular spread as a factor of every spot's omega profile, as (A5) is for
  the detector; it is not established.
- **Multi-phase overlaps**: populations of different phases sharing a region add their ray sums like any others;
  nothing more is needed, but nothing is checked.
- **Code**: none of the refiner is in anri yet.
