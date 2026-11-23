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
cell's own Lorentz and polarisation factor divided out, and each frame's monitor ratio :math:`M_j / M_{\text{ref}}`
(after the recording, which acts on raw counts; :eq:`rec-monitor`):

.. math::
    :label: rf-statistic

    T_R[f] = \sum_{p \in R} \frac{M_{\text{ref}}}{M_j}\, \frac{r_p\, f(\hat{\vec{g}}_p)}{L(\hat{\vec{g}}_p)\, P(\hat{\vec{g}}_p)},
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

**Cited:** the linear model of a ray's mean diffraction vector in the nine components of :math:`\matr{U}\matr{B}`,
and its strain counterpart in the six of :math:`\varepsilon`, and their factorisation into scalar tomographic
projections followed by diagonal weights (the Miller indices, or the strain's directional cosines, over the path
length), so that the operator and its adjoint are never assembled [Henningsson2023]_. :eq:`rf-linear` is that model
with three changes. Its unknowns are volume integrals, fitted from unnormalised sums of recorded values together with
the counts, not volume averages over a path length; so it stays linear where density, beam or absorption vary along
the ray. Its strain is read from :math:`|\vec{g}|^2` per cell (:eq:`rf-invariant`), not from the centroid by a Taylor
expansion that needs the mosaicity to be moderate (:eq:`ref-spread-mean` gives the bias). And its cells are divided by
their own Lorentz and polarisation factors (:eq:`rf-weight`). The factorisation carries over unchanged: :math:`\matr{G}`
is a sum of scalar projections, one per component of each moment, weighted by powers of :math:`\vec{g}_0`. Their
demonstration regularises with a smooth radial basis, which removes the voxel grid's aliases (:eq:`rf-discrete`), and
generates its data with the model itself, so that no misfit reaches the weak directions (:eq:`rf-amplify`); anri does
neither (AGENTS.md rules out basis-function priors), which is why it needs :eq:`rf-project`.

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

The weak directions
-------------------

Two months of refiners (tomography, pixel-wise EM, first moments, second moments, Gaussian populations) have
shared one failure: misfits between model and data are absorbed by the weak combinations of :doc:`limits` and by
free variables such as spreads, which run to unphysical values. This section derives why, for any statistic, and
what estimate avoids it.

**No statistic sees them.** A reflection carries the in-plane spatial frequency :math:`\vec{k}` only if its ray runs
along :math:`\vec{\xi} = \hat{\vec{z}} \times \hat{\vec{k}}` (the Fourier slice, :eq:`lim-slice`). Its
:math:`\vec{g}_0` is then :math:`a\hat{\vec{k}} + c\hat{\vec{z}} + s\hat{\vec{\xi}}`, with :math:`s` of the order
of :math:`\sin\theta` (:eq:`lim-ray`). Each of its photons has :math:`\vec{g} = \matr{F}\vec{g}_0`, so for every
function :math:`f`,

.. math::
    :label: rf-weak-any

    f\big((\matr{F} + \vec{v}\hat{\vec{\xi}}^T)\,\vec{g}_0\big) = f(\matr{F}\vec{g}_0) \quad \text{at } s = 0,
    \qquad
    \vec{g}_0^T\matr{C}\vec{g}_0 = a^2 C_{kk} + c^2 C_{zz} + 2ac\, C_{kz} + 2as\, C_{k\xi} + 2cs\, C_{z\xi} + s^2 C_{\xi\xi}.

So the data at frequency :math:`\vec{k}` see each population's :math:`\matr{F}` acting on :math:`\hat{\vec{k}}`
and :math:`\hat{\vec{z}}`, and its action on :math:`\hat{\vec{\xi}}` only through :math:`s`. This holds whatever is
computed from them: pixels, histograms or moments of any order, the cut and the group rule included, since all of
these are functions of the photons' :math:`\vec{g}`. The metric's :math:`C_{\xi\xi}` is seen through :math:`s^2`. The
weak combinations are a property of single-axis scanning, not of a method. The second moments
:math:`\matr{Y}^{(2)}[\vec{g}_0, \vec{g}_0]` carry the same weights, so a population's spread has weak components too:
its spread of :math:`\matr{F}\hat{\vec{\xi}}`.

**Proof:** ``tests/theory/test_refiner.py::test_weak_for_any_statistic``

**How a misfit reaches them.** With :math:`\matr{G} = \matr{U}\matr{S}\matr{V}^T` (whitened by the weights), the
least-squares estimate's bias along the right singular vector :math:`\vec{v}_i` is
:math:`\langle\vec{u}_i, \vec{\Delta}\rangle / s_i` (:eq:`rf-linear`). A weak combination has :math:`s_i \propto
\sin\theta`, or :math:`\sin^2\theta` for :math:`C_{\xi\xi}`, so any misfit with any overlap on its data pattern
:math:`\vec{u}_i` is amplified by :math:`1/s_i`. For Ni (111 at 70 keV, 111 and 420 at 43 keV)
:math:`\sin\theta` = 0.0435, 0.0709 and 0.183: amplification by 5.5-23, and by 30-530 for :math:`C_{\xi\xi}`. A
nuisance fitted alongside (a free spread, a width) can only lower the effective :math:`s_i`, never raise it: its
Schur complement :math:`\vec{g}^T\matr{P}\vec{g} \le \vec{g}^T\vec{g}`. A misfit that a free spread could absorb is
absorbed there, at whatever value of the spread it takes.

.. math::
    :label: rf-amplify

    \vec{b} = \sum_i \frac{\langle\vec{u}_i, \vec{\Delta}\rangle}{s_i}\,\vec{v}_i, \qquad
    s_{\text{weak}} \propto \sin\theta, \qquad
    \vec{g}_w^T\matr{P}_\eta\vec{g}_w \le \vec{g}_w^T\vec{g}_w .

**Proof:** ``tests/theory/test_refiner.py::test_weak_amplification``, ``tests/theory/test_refiner.py::test_sin_theta_numbers``

**On a voxel grid they are seen through the model's geometry.** :eq:`rf-weak-any` is exact for continuous fields.
A field constant on voxels of step :math:`h`, :math:`\matr{\beta}(\vec{x}) = \sum_v \matr{B}_v\chi(\vec{x} -
\vec{x}_v)`, has the transform :math:`\hat\chi(\vec{k})\matr{B}(\vec{k})`, with :math:`\matr{B}(\vec{k}) = \sum_v
\matr{B}_v e^{-i\vec{k}\cdot\vec{x}_v}` periodic on the reciprocal lattice. The reflections that carry an alias
:math:`\vec{k}_n = \vec{k} + 2\pi\vec{n}/h` see :math:`\matr{B}(\vec{k})` applied to :math:`\hat{\vec{k}}_n` and
:math:`\hat{\vec{z}}`, and :math:`\hat{\vec{k}}_n` has a component along :math:`\hat{\vec{\xi}}(\vec{k})`:

.. math::
    :label: rf-discrete

    \matr{B}(\vec{k} + 2\pi\vec{n}/h) = \matr{B}(\vec{k}), \qquad
    \matr{B}\hat{\vec{k}}_n = (\hat{\vec{k}}_n\cdot\hat{\vec{k}})\,\matr{B}\hat{\vec{k}}
    + \sin(\psi_n - \psi)\,\matr{B}\hat{\vec{\xi}},

with :math:`\psi`, :math:`\psi_n` the directions of :math:`\vec{k}` and :math:`\vec{k}_n`. So a voxel model sees its
weak combination :math:`\matr{B}(\vec{k})\hat{\vec{\xi}}(\vec{k})` with the coefficient :math:`\sin(\psi_n -
\psi)\,\hat\chi(\vec{k}_n)\,T(\vec{k}_n)`: through the voxel's assumed shape :math:`\hat\chi` and the beam's
transfer :math:`T` at frequencies beyond the beam's resolution. At the first alias of a beam whose FWHM is the voxel
step, :math:`T = e^{-\pi^2/(4\ln 2)} = 0.028` (:eq:`lim-transfer`, :math:`\Lambda = 1`), before :math:`\hat\chi`'s
factor: of the same order as the combination's physical visibility, :math:`\sin\theta = 0.04`-0.18 times its
coefficient. A voxel model therefore sees its weak combinations about as much through its own discretisation as
through the physics.

**Proof:** ``tests/theory/test_refiner.py::test_discrete_weak_directions`` (the periodicity; the decomposition; the
number).

The real sample is not constant on voxels: its content at :math:`\vec{k}_n` is its own structure at that frequency,
unrelated to :math:`\matr{B}(\vec{k})`. So a voxel model fits its weak combinations from the high-frequency tails of
its beam profile and voxel shape, which are exactly what a wrong beam profile or an unmodelled runout makes wrong.
That is the mechanism of the failures: the weak combinations are set by the model's errors, not by the data.

**Errors in the geometry.** An error :math:`\delta\matr{G}` (beam weights, a runout of about 1 µm, a beam profile
known only roughly) gives a misfit :math:`\delta\matr{G}\,\vec{Y}`, and :math:`\vec{Y}` is dominated by the lattice
itself: :math:`\matr{Y}^{(1)} \approx V\matr{I}`. Features centred on a reference per region, :math:`\vec{g} -
\vec{g}_R` and :math:`|\vec{g}|^2 - |\vec{g}_R|^2` (with the counts fitted alongside), turn the misfit into

.. math::
    :label: rf-centre

    \delta\matr{G}\,(\vec{Y} - \vec{Y}_R) = \sum_k \delta a_k V_k\, (\vec{m}_k - \vec{g}_R),

proportional to each population's deviation from the reference, about :math:`10^{-3}` of the lattice, instead of to
the lattice. The counts' own misfit, :math:`\sum \delta a_k V_k`, falls on the densities, which are not sought. The
deformation then inherits only the geometry's relative error, :math:`\delta a / a` times its own contrast. A runout
that is common to every voxel and periodic in omega is also of the drift's form (:mod:`anri.index.drift`, measured
from Friedel pairs), and can be fitted as one function of omega for the whole map.

**Proof:** ``tests/theory/test_refiner.py::test_centred_features``

**What to estimate.** The combinations whose :math:`s_i` is above a level :math:`s_{\min}` are estimated, with
bias at most :math:`|\vec{\Delta}|/s_{\min}`. The rest are reported as not determined, with their covariance. This is
not a smoothness prior. It does not change the data's fit along the seen directions, and it reports what the data
cannot resolve (AGENTS.md) instead of filling it with model error.

.. math::
    :label: rf-project

    \hat{\vec{Y}}_{\text{seen}} = \sum_{s_i \ge s_{\min}} \frac{\langle\vec{u}_i, \vec{T}\rangle}{s_i}\,\vec{v}_i,
    \qquad
    |\operatorname{E}[\hat{\vec{Y}}_{\text{seen}}] - P_{\text{seen}}\vec{Y}| \le \frac{|\vec{\Delta}|}{s_{\min}}.

**Proof:** ``tests/theory/test_refiner.py::test_projected_estimator``

The alternative, where the maintainer allows it within a merged domain, is the constraint
:math:`(\operatorname{curl}\matr{\beta})_{\cdot z} = 0` (:doc:`limits`). It removes exactly the weak combinations of
the first moments, and its bias is the dislocation density threading the slice that it assumes away. Neither choice
recovers the weak combinations; each makes their error known. Separated populations are not affected: alone in their
regions, they are seen in every reflection with all nine numbers (:doc:`limits`), with no tomography to weaken them.

One linear solve, and its split
-------------------------------

**Boxes from the indexer are complete at once.** The regions depend on where the spots are, and so on
:math:`\vec{Y}`. Iterating them is unnecessary if the first boxes already hold the light. The indexer's orientation
is within a rotation :math:`|\vec{\delta}| \le e` of a population's mean (:doc:`localsearch` gives :math:`e` per
level). A rotation moves a spot by :math:`\Delta\omega = \vec{j}_\omega\cdot\vec{\delta}` and :math:`\Delta\eta =
\vec{j}_\eta\cdot\vec{\delta}` (:eq:`dif-rotation-motion`), and a strain :math:`|\varepsilon| \le
\varepsilon_{\max}` moves it in :math:`2\theta` only. So, by Cauchy-Schwarz,

.. math::
    :label: rf-box-margin

    |\Delta\omega| \le \frac{e}{|\sin\eta|}, \qquad
    |\Delta\eta| \le e\sqrt{1 + \frac{\tan^2\theta}{\sin^2\eta}}, \qquad
    |\Delta 2\theta| \le 2\varepsilon_{\max}\tan\theta,

each attained for :math:`\vec{\delta}` along its :math:`\vec{j}`. A box whose half-widths exceed these margins plus
the light's own extent :math:`t` holds all of the population's light within :math:`t` of its true centre, wherever
that is.

**Proof:** ``tests/theory/test_refiner.py::test_box_margin`` (the lengths of :math:`\vec{j}`; Cauchy-Schwarz as
:math:`|\vec{j}|^2|\vec{\delta}|^2 - (\vec{j}\cdot\vec{\delta})^2 = |\vec{j}\times\vec{\delta}|^2`; Bragg's law
differentiated).

So the regions are fixed once, from the indexer's populations. The estimand is the light within these fixed boxes,
a fixed linear functional of the light, and :eq:`rf-linear` is one linear system, solved once. The price is box size:
the omega margin grows as :math:`1/|\sin\eta|`, and with it the cut's bound (:eq:`rf-cut-bound`). This is a second
reason, beyond :eq:`dif-rotation-motion`'s smearing, for leaving out reflections near the rotation axis.

**What is determined per population.** Write a population's first moment absolutely, :math:`\matr{Z}_k = \int
\matr{F}\matr{U}_k\matr{B}\, d\mu_k`, so that it adds :math:`\matr{Z}_k\vec{h}` to every region it is in, whatever
its reference orientation. Two populations of one voxel in the same region of a row then have equal columns there:
only :math:`\matr{Z}_k + \matr{Z}_{k'}` is seen. Their difference is seen only in the rows where they are in
different regions, through :math:`(\matr{Z}_k - \matr{Z}_{k'})\vec{h}` for those reflections, and

.. math::
    :label: rf-identify

    \matr{Z} = [\matr{Z}\vec{h}_1, \matr{Z}\vec{h}_2, \matr{Z}\vec{h}_3]\,\matr{H}^{-1}
    \ \text{if } \det\matr{H} \ne 0, \qquad
    (\matr{Z} + \vec{v}\vec{n}^T)\vec{h} = \matr{Z}\vec{h}\ \ \forall \vec{h} \perp \vec{n}.

The difference of two populations' first moments is therefore determined if and only if the reflections that
separate them span three dimensions. Their metrics' difference needs reflections whose :math:`\vec{h}\vec{h}^T` span
the symmetric matrices, six at least (100, 010, 001, 110, 101, 011 do; any five of them do not). Otherwise the two
are one population as far as the data can tell. The refiner reports their sum, with its spread, as AGENTS.md asks.
This is decided from the regions alone, before any fit.

**Proof:** ``tests/theory/test_refiner.py::test_identify_populations``

**The split, for the first moments.** Within a merged domain, the field :math:`\matr{Z}(\vec{x})` (zero outside the
domain :math:`\Omega`, whose other populations sit in other regions) is seen by its Fourier slices. At an in-plane
:math:`\vec{k}`, with :math:`\hat{\vec{\xi}} = \hat{\vec{z}}\times\hat{\vec{k}}`, each row :math:`i` of
:math:`\hat{\matr{Z}}` has the in-plane columns :math:`(\hat Z_{ix}, \hat Z_{iy})`. These split as

.. math::
    :label: rf-split

    \hat{\vec{k}}\hat{\vec{k}}^T + \hat{\vec{\xi}}\hat{\vec{\xi}}^T = \matr{I}_2, \qquad
    |\vec{k}|\,\hat{\vec{\xi}}\cdot(\hat Z_{ix}, \hat Z_{iy}) = k_x\hat Z_{iy} - k_y\hat Z_{ix}
    = -i\,\widehat{(\operatorname{curl}\matr{Z})}_{iz}, \qquad
    L^2(\Omega)^2 = \nabla H^1(\Omega) \oplus \{\vec{w} : \nabla\cdot\vec{w} = 0,\ \vec{w}\cdot\vec{n}|_{\partial\Omega} = 0\}.

The weak part is each row's solenoidal in-plane field, and the seen part its gradient. The third column
:math:`\matr{Z}\hat{\vec{z}}` and the mean are seen whole. On the domain, the two parts are the Helmholtz
decomposition: the seen part is :math:`\nabla\phi_i`, with :math:`\phi_i` from the Neumann problem
:math:`\Delta\phi_i = \nabla\cdot\vec{v}_i` in :math:`\Omega`, :math:`\partial_n\phi_i = \vec{v}_i\cdot\vec{n}` on
its boundary. Orthogonality is :math:`\nabla\cdot(\phi\vec{w}) = \nabla\phi\cdot\vec{w} + \phi\nabla\cdot\vec{w}` and
the divergence theorem. It is three scalar Poisson problems per domain: sparse, local, and computable at any scale.
The curl along :math:`z` is the Nye tensor's column of :eq:`lim-nye`.

**Proof:** ``tests/theory/test_refiner.py::test_split_first_moments``

**The split, for the metric and the spreads.** :math:`|\vec{g}|^2` sees :math:`\vec{g}_0^T\matr{C}\vec{g}_0`
(:eq:`rf-weak-any`). The out-of-plane parts follow the same pattern: :math:`C_{zz}` is seen whole, and the vector
:math:`(C_{xz}, C_{yz})` splits as in :eq:`rf-split`, its solenoidal part seen through :math:`s`. The in-plane block
splits in the orthonormal basis :math:`\hat{\vec{k}}\hat{\vec{k}}^T`, :math:`(\hat{\vec{k}}\hat{\vec{\xi}}^T +
\hat{\vec{\xi}}\hat{\vec{k}}^T)/\sqrt{2}`, :math:`\hat{\vec{\xi}}\hat{\vec{\xi}}^T`, seen through :math:`s^0`,
:math:`s^1`, :math:`s^2`, and

.. math::
    :label: rf-split-metric

    |\vec{k}|^2\,\hat{\vec{\xi}}^T\hat{\matr{C}}\hat{\vec{\xi}} = k_y^2\hat C_{xx} + k_x^2\hat C_{yy} - 2k_xk_y\hat C_{xy}
    = -\widehat{(\operatorname{inc}\matr{C})}_{zz},

Saint-Venant's incompatibility. The doubly weak part is the field's incompatibility: zero for a compatible field
:math:`\operatorname{sym}\nabla\vec{u}`, and all of an Airy field :math:`(\psi_{yy}, \psi_{xx}, -\psi_{xy})`. The seen
part is the Hessians :math:`\nabla\nabla\phi`, which have only a :math:`\hat{\vec{k}}\hat{\vec{k}}^T` part. The second
moments :math:`\matr{Y}^{(2)}[\vec{g}_0, \vec{g}_0]` carry the same weights, so a population's spread splits the same
way: its weak parts are the spread of :math:`\matr{F}\hat{\vec{\xi}}`. These are the spreads that ran away in earlier
refiners.

**Proof:** ``tests/theory/test_refiner.py::test_split_metric``

**The solve.** In each merged domain the in-plane columns of :math:`\matr{Z}` are written as :math:`\nabla\phi_i +
\vec{w}_i`, with the third column, the mean, and the populations of separated light as they are. Two estimates follow,
as the two outputs:

- **(A)** :math:`\vec{w}_i` fitted too, reported with its covariance and flagged as seen only through
  :math:`\sin\theta` (or not determined, below :math:`s_{\min}`, :eq:`rf-project`);
- **(B)** :math:`\vec{w}_i = 0`, the curl-free constraint, whose bias is the true :math:`\vec{w}_i`: the dislocation
  density threading the slice.

Because each reflection's :math:`\vec{g}_0` holds :math:`s\,\hat{\vec{\xi}}`, the seen and weak columns overlap in the
data at order :math:`s`. So (A) and (B) differ in their seen parts by order :math:`s` times the weak part. That
difference is itself a diagnostic: where it exceeds the error bars, the constraint matters. The same split applies to
the metric and the spreads, with the Hessian in place of the gradient.

What the split does not remove is the beam's own weak family. A seen component at frequency :math:`\vec{k}` is
attenuated by the beam's transfer :math:`T(\vec{k})` (:eq:`lim-transfer`), so its noise and bias are amplified by
:math:`1/T`, by more than 4 at periods below 1.6 beam FWHM
(``tests/theory/test_refiner.py::test_beam_quarter_period``). That is the map's resolution. It is reported as a
resolution, not filtered, since filtering would be a smoothness prior.

**The split on the voxel grid.** On a grid the split needs no boundary conditions of its own. Let :math:`\matr{D}` be
the discrete gradient (for the first moments' in-plane columns) or the discrete Hessian (for the metric's in-plane
block), restricted to the domain's voxels, with its gauge removed: a constant for the gradient, an affine function
for the Hessian. Then :math:`\matr{D}` has full column rank, and

.. math::
    :label: rf-split-grid

    \matr{P} = \matr{D}(\matr{D}^T\matr{D})^{-1}\matr{D}^T, \qquad
    \matr{P} = \matr{P}^T = \matr{P}^2, \qquad \matr{P}\matr{D} = \matr{D}

is the orthogonal projector onto the seen fields; the weak part is :math:`(\matr{I} - \matr{P})`. The normal
equations :math:`\matr{D}^T\matr{D}\vec{\phi} = \matr{D}^T\vec{v}` carry the boundary conditions implicitly (in the
continuum limit: Neumann for the gradient; for the Hessian, those of a free plate edge, not derived here). They are
sparse, symmetric and positive definite: conjugate gradients per domain. In one dimension every field is a gradient and nothing is weak, as the slice argument says.

**Proof:** ``tests/theory/test_refiner.py::test_discrete_projector``

**The continuum limit.** The projections are least-squares problems, and their natural boundary conditions follow by
integrating the first variation by parts. For the gradient, :math:`\min \tfrac{1}{2}\int_\Omega |\nabla\phi -
\vec{v}|^2` gives Neumann's problem (:eq:`rf-split`). For the Hessian, with :math:`\matr{M} = \nabla\nabla\phi -
\matr{C}`, :math:`n` the boundary's normal, :math:`s` its arc length and :math:`Q_n = (\partial_i M_{ij}) n_j`,

.. math::
    :label: rf-plate

    \matr{M} : \nabla\nabla\psi = \partial_i(M_{ij}\partial_j\psi) - \partial_j\big((\partial_i M_{ij})\psi\big)
    + (\partial_i\partial_j M_{ij})\psi
    \ \Rightarrow\
    \partial_i\partial_j M_{ij} = 0 \text{ in } \Omega, \qquad
    M_{nn} = 0, \quad Q_n + \partial_s M_{nt} = 0 \text{ on } \partial\Omega,

with :math:`M_{nt}` continuous at corners. Inside, it is the biharmonic :math:`\Delta^2\phi = \partial_i\partial_j
C_{ij}`. On the boundary the normal moment and Kirchhoff's effective shear vanish: a free plate edge. The boundary
term :math:`n_i M_{ij}\partial_j\psi - (\partial_i M_{ij}) n_j\psi` becomes :math:`M_{nn}\partial_n\psi - (Q_n +
\partial_s M_{nt})\psi` after the tangential derivative is moved off :math:`\psi`, a total derivative along the
boundary. Then :math:`\psi` and :math:`\partial_n\psi` are free there.

**Proof:** ``tests/theory/test_refiner.py::test_free_plate_edge`` (the pointwise identity; the boundary term on an
edge, up to that total derivative).

The level :math:`s_{\min}`
--------------------------

Whitened by the noise covariance, a unit data pattern :math:`\vec{u}_i` has noise of variance
:math:`\vec{u}_i^T\vec{u}_i = 1`. Its bias, with each misfit component bounded by :math:`\bar\Delta_j`
(:eq:`rf-linear`), is at most :math:`\sum_j |u_{ij}|\bar\Delta_j` by the triangle inequality. So the estimate along
:math:`\vec{v}_i` errs, at :math:`z` standard deviations, by at most :math:`(z + \sum_j |u_{ij}|\bar\Delta_j)/s_i`, and

.. math::
    :label: rf-smin

    \text{keep } \vec{v}_i \iff s_i \ge s_{\min,i} = \frac{z + \sum_j |u_{ij}|\bar\Delta_j}{\epsilon},
    \qquad
    \frac{s_{\text{weak}}^2}{s_{\text{seen}}^2} = \frac{\sum_h s_h^2 a_h^2}{\sum_h a_h^2},

with :math:`\epsilon` the precision the map must reach. A weak column's coefficient in reflection :math:`h` is
:math:`s_h a_h`, where the seen column's is :math:`a_h`. So its gain relative to the seen one is the rms of the
reflections' :math:`s_h \sim \sin\theta_h`, weighted by how much each reflection sees.

**Proof:** ``tests/theory/test_refiner.py::test_smin``

The level is per direction and comes from the bounds of :eq:`rf-cut-bound`, :eq:`rf-group` and
:eq:`rf-alias-bound`, not from a choice. Where the bias bound is small, the noise sets it, and where the bias bound is
large it sets it. Only :math:`\epsilon` and :math:`z` are choices: the precision sought, and the confidence.

Histograms
----------

The moments keep a few numbers per region. The arc's profile, how its light is spread along the ring and in omega,
carries each voxel's whole orientation distribution: its populations, not only their mean and spread (AGENTS.md).
The same linearity holds for it. For any bin :math:`B` of cells (in rotation space, along the arc), the expected
recorded sum over :math:`B`, each cell divided by its :math:`LP`, is

.. math::
    :label: rf-hist

    \sum_k a_{bk}|F_h|^2 A_{kh} \int K_B(\matr{F})\, d\mu_k(\matr{F}),

with :math:`K_B(\matr{F})` the probability that a photon of a sub-volume :math:`\matr{F}` is counted in :math:`B`:
linear in each population's distribution, for any bins. The moments are linear functionals of the finest histogram,
the cells themselves. So the histogram keeps everything the moments keep and more. Its tomography is the indexer's
linear model (:eq:`idx-linear`) with the exact kernel and the bounds of this page, at bins as fine as the kernel
allows.

**Proof:** ``tests/theory/test_refiner.py::test_histogram_linear``

The kernel's shape is the instrument's: the point spread (trusted, (A5)), the bandwidth and the beam's divergence
(A4). Those last two are known only roughly. Bins much wider than the instrument need only its variance
:math:`\sigma^2`. A photon at distance :math:`d` from its bin's edge leaves the bin only if the instrument moves it by
at least :math:`d`, so for any instrument shape

.. math::
    :label: rf-bin-edge

    |K_B(\matr{F}) - \mathbb{1}_B(\matr{F}\vec{g}_0)| \le \frac{\sigma^2}{d^2}.

**Proof:** ``tests/theory/test_refiner.py::test_bin_edge`` (:math:`\mathbb{1}(|e| \ge d) \le |e|^2/d^2`).

Light within a few :math:`\sigma` of a bin's edge is the bins' misfit, bounded like the box's tails. Bins of a few
instrument widths are the finest that need no instrument shape. Finer bins need the instrument's kernel, as
deconvolution does. The weak combinations are the same for histograms (:eq:`rf-weak-any`): bins resolve a voxel's
populations, not what single-axis data cannot see.

Two more bounds
---------------

**The mean rotation at large spreads.** The fit gives :math:`\int\matr{F}` exactly at any spread. Its polar factor is
the rotations' chordal mean [Moakher2002]_ (:eq:`pop-mean`) only if strain does not enter. Write :math:`\matr{F} =
\matr{R}(\matr{I} + \varepsilon)`. The polar factor :math:`\matr{Q}` of :math:`\matr{M} = \matr{Q}\matr{S}` moves,
to first order in a change :math:`d\matr{M}`, by :math:`\matr{Q}\matr{\Omega}`, with

.. math::
    :label: rf-polar

    \matr{S}\matr{\Omega} + \matr{\Omega}\matr{S} = \matr{Q}^T d\matr{M} - d\matr{M}^T\matr{Q}, \qquad
    \Omega_{ij} = \frac{W_{ij}}{s_i + s_j}\ (\matr{S} \text{ diagonal}), \qquad
    \operatorname{skew}\operatorname{E}[\matr{R}\varepsilon] = \operatorname{skew}\operatorname{E}\big[[\vec{d}]_\times\varepsilon\big]
    \ (\matr{R} = \matr{I} + [\vec{d}]_\times).

A symmetric strain does not turn the mean. Only the correlation of a sub-volume's rotation :math:`\vec{d}` with its
strain does, and :math:`|\matr{\Omega}| \lesssim |\operatorname{E}[[\vec{d}]_\times\varepsilon]|`: a spread of
:math:`10^{-2}` rad fully correlated with a strain spread of :math:`10^{-3}` turns it by :math:`10^{-5}`. Elastic
bending is fully correlated (:doc:`limits`), so this is the case to bound.

**Proof:** ``tests/theory/test_refiner.py::test_polar_mean``

**Thermal diffuse tails.** One-phonon diffuse scattering about a Bragg point falls as :math:`A(\hat{\vec{q}})/|\vec{q}|^2`
with :math:`A` even. Its variance grows without bound with the region, so :eq:`ref-tail` does not apply. Within a ball
of radius :math:`t` about the Bragg point it adds

.. math::
    :label: rf-tds

    t\oint A\, d\Omega \ \text{(counts)}, \qquad 0 \ \text{(first moment)}, \qquad
    \frac{t^3}{3}\oint A\,\hat{\vec{q}}\hat{\vec{q}}^T d\Omega \ \text{(second moments)};
    \qquad
    -t\Big(\oint A\,\hat{\vec{q}}\hat{\vec{q}}^T d\Omega\Big)\vec{\delta} \ \text{(first moment, off centre by } \vec{\delta}).

It biases no first moment in a box symmetric about the Bragg point. Off centre, it pulls a centred first moment
towards the box's centre by its share of the light times :math:`(\oint A\hat{\vec{q}}\hat{\vec{q}}^T / \oint
A)\,\vec{\delta}`: :math:`\vec{\delta}/3` for isotropic :math:`A`. The box's centre is the indexer's prediction, so
this is a pull towards the prior, bounded by the margin of :eq:`rf-box-margin`. It biases the second moments by an
amount that grows as :math:`t^3`. A spread read from second moments therefore includes diffuse scattering in
proportion to the box's volume, and must be read with that term subtracted or bounded. This is the tail seen on ID11's
Si peaks over 5-9 frames.

**Proof:** ``tests/theory/test_refiner.py::test_tds_tails`` (spherical integrals for :math:`A = 1` and :math:`A = 1 +
a\hat q_z^2`; the off-centre term by the divergence theorem, and its volume side for :math:`A = 1`).

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
- **The box margins** are first order in :math:`e`, as :eq:`dif-rotation-motion` is.
- **The precision** :math:`\epsilon` and confidence :math:`z` of :eq:`rf-smin` are the maintainer's to set.
- **Histograms at scale**: their unknowns are each voxel's density over a rotation grid, many more than the moments'
  16; how the two are combined (histograms for populations, moments for their means) is not derived.
- **Boxes for curved arcs.** A box in :math:`(s, f, \omega)` around an arc of several degrees holds many cells the
  arc misses, and :eq:`rf-cut-bound` grows with them: boxes should follow the arc, defined in rotation space.
- **The beam profile and runout** enter :math:`\matr{G}`; :eq:`rf-centre` bounds how they reach the deformation,
  but their own sizes are estimates (runout about 1 µm).
- **Multi-phase overlaps**: populations of different phases sharing a region add their ray sums like any others;
  nothing more is needed, but nothing is checked.
- **Code**: none of the refiner is in anri yet.
