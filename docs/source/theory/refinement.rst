.. _theory_refinement:

What a refiner can be robust to
===============================

:doc:`limits` asks what the data determine; this chapter asks which statistics of the recorded cells
(:doc:`recording`) a local refiner of orientation and strain can fit so that an error in the peak shapes and the
counting noise leave it unbiased, and what each choice costs. The shapes are never known exactly: the point spread's
wing, the sensor's depth kernel, the beam's convergence, thermal diffuse tails and the sample's own spread are
measured to some precision, and the renderer draws each source of spread as a first-order Gaussian
(:doc:`peak_shape`).

The results, in brief:

- **Weights never bias a fit; shape errors do, by a projection** (:eq:`ref-bias`). An error that looks exactly like a
  change of a fitted nuisance (a spread, a width) is absorbed by it and biases nothing else (:eq:`ref-profile`).
- **Symmetry protects a location** (:eq:`ref-symmetric`): where the light of one population is centrally symmetric
  about a point of the cell lattice, any odd estimating function (a centroid over a region symmetric about it, the
  Poisson score of any symmetric model) is unbiased whatever the spot's width and tails, through the cut and the
  group rule.
- **The pixels break that symmetry by aliasing** (:eq:`ref-alias`): the bias of a wrongly shaped model oscillates with
  the spot's position in its cell, with an amplitude set by the characteristic functions at :math:`2\pi`. For the
  measured point spread drawn as one Gaussian of the right variance it is 0.067 px.
- **A spread moves a mean at second order** (:eq:`ref-spread-mean`, :eq:`ref-ring`): an orientation spread of 1° reads
  as a strain of order :math:`10^{-4}` in any first-order model, and moves the mean direction by as much where the
  spread is anisotropic. :math:`|\vec{g}|` itself is exact at any spread.
- **Splitting merged light by shape is bias-dominated** (:eq:`ref-split`): two populations 3-6 own widths apart,
  fitted with a slightly wrong shape, have a separation whose bias exceeds its noise above 50 to 30000
  photons. Light that cannot be split robustly must be fitted as a ray sum (:eq:`mom-raysum`), not divided.

The bias of a fitted statistic
------------------------------

Let a statistic :math:`\vec{t}` of the recorded cells (region counts and moments, :doc:`moments`; or the score of a
likelihood) have the model expectation :math:`\vec{\tau}(\vec{\theta})`, with Jacobian :math:`\matr{G}` and noise
covariance :math:`\matr{C}`, and let the truth be :math:`\operatorname{E}[\vec{t}] = \vec{\tau}(\vec{\theta}_0) +
\vec{\Delta}`: :math:`\vec{\Delta}` is what the model gets wrong (a shape, a tail, a window's missing light). Fitted by
weighted least squares with any symmetric positive definite :math:`\matr{W}`, to first order about
:math:`\vec{\theta}_0`:

.. math::
    :label: ref-bias

    \operatorname{E}[\hat{\vec{\theta}}] - \vec{\theta}_0 = (\matr{G}^T\matr{W}\matr{G})^{-1}\matr{G}^T\matr{W}\,\vec{\Delta},
    \qquad
    \operatorname{Cov}[\hat{\vec{\theta}}] = (\matr{G}^T\matr{W}\matr{G})^{-1}\matr{G}^T\matr{W}\matr{C}\matr{W}\matr{G}
    (\matr{G}^T\matr{W}\matr{G})^{-1},

which is :math:`(\matr{G}^T\matr{C}^{-1}\matr{G})^{-1}` at :math:`\matr{W} = \matr{C}^{-1}`.

**Proof:** ``tests/theory/test_refinement.py::test_fitted_statistic_bias`` (exact for a linear model; Gauss-Newton
about :math:`\vec{\theta}_0` otherwise).

So a wrong weight matrix never biases a fit: with :math:`\vec{\Delta} = 0` every :math:`\matr{W}` gives an unbiased
estimate, and only its efficiency and its stated error bars depend on :math:`\matr{W}`. The covariances of the group
rule (:eq:`rec-cov-pair`) therefore belong in the error bars (the sandwich) and, for efficiency, in the weights; two
regions of one frame within :math:`2(m - 1)` pixels of each other (4 px for :math:`m = 3`) are correlated, and
treating them as independent misstates their errors without biasing them.

With nuisance parameters :math:`\vec{\eta}` fitted too (a population's spread, a width), :math:`\matr{G} =
[\matr{G}_\theta, \matr{G}_\eta]` and the bias on :math:`\vec{\theta}` is

.. math::
    :label: ref-profile

    \vec{b}_\theta = (\matr{G}_\theta^T\matr{W}\matr{P}\matr{G}_\theta)^{-1}\matr{G}_\theta^T\matr{W}\matr{P}\,\vec{\Delta},
    \qquad
    \matr{P} = \matr{I} - \matr{G}_\eta(\matr{G}_\eta^T\matr{W}\matr{G}_\eta)^{-1}\matr{G}_\eta^T\matr{W},

so the part of :math:`\vec{\Delta}` that a change of :math:`\vec{\eta}` could make (:math:`\matr{P}\matr{G}_\eta =
0`) biases nothing but :math:`\vec{\eta}`.

**Proof:** ``tests/theory/test_refinement.py::test_profiled_bias``

A free spread therefore absorbs a wrong *width*; what it cannot absorb is a wrong *shape* at the same width (a wing, a
skew, the aliasing below), nor any error in light shared with another population.

Symmetry
--------

Let :math:`\sigma(\vec{x}) = 2\vec{c} - \vec{x}` be a point reflection that maps the cell lattice to itself: in each of
(frame, row, column), :math:`c` on a cell's centre or on the edge between two. If a population's light is centrally
symmetric about :math:`\vec{c}` (any centred Gaussian, whatever its covariance; any symmetric mixture), so are its
cells' :math:`\lambda_p`; the cut acts on each cell alone, and the group rule commutes with :math:`\sigma`, which maps
8-neighbours to 8-neighbours and so each group of :math:`p` onto one of :math:`\sigma(p)` with its boundary. Then the
expected recorded values are symmetric, and for any estimating function odd under :math:`\sigma`,

.. math::
    :label: ref-symmetric

    \operatorname{E}[r_{\sigma(p)}] = \operatorname{E}[r_p]
    \quad\Rightarrow\quad
    \operatorname{E}\Big[\sum_p r_p\, \psi_p(\vec{c})\Big] = 0
    \quad \text{for every } \psi \text{ with } \psi_{\sigma(p)}(\vec{c}) = -\psi_p(\vec{c}),

for example the centroid :math:`\psi_p = \vec{x}_p - \vec{c}` over a region symmetric about :math:`\vec{c}`, and the
score of a Poisson or censored likelihood (:eq:`rec-likelihood`, :eq:`lik-censored`) whose model is symmetric about
:math:`\vec{c}`, whatever its width and tails. So a separated population's location is consistent (unbiased as the
counts grow) whatever is wrong with the width, the tails, the window (if it is symmetric too), the cut or the group
rule, at the positions where :math:`\sigma` exists.

**Proof:** ``tests/theory/test_refinement.py::test_group_rule_is_symmetric`` (the group rule's animals and their
boundaries map one to one under the reflection, every pixel of two detectors, groups of up to 4 pixels),
``tests/theory/test_refinement.py::test_symmetric_estimating_equation`` (the centroid and the score of any even model,
any function of the model's light per cell, on centres and on edges).

What symmetry does not protect: light that is not symmetric (a curved arc, :eq:`ref-spread-mean`; the sensor's
exponential kernel, :eq:`det-sensor`, whose shift is common to every voxel and so only of region means,
:eq:`lim-common`), light shared with another population (:eq:`ref-split`), and positions between the lattice's symmetry
points (:eq:`ref-alias`). The group rule's loss of compact spots (:doc:`recording`) is one such position-dependent
effect: symmetric about centres and corners, not between.

Pixels: the aliasing a shape error leaves
-----------------------------------------

For a spot of any shape symmetric about its mean :math:`\mu` (characteristic function :math:`\chi`, real and even), in
cells of unit width centred on the integers, the mean cell index is

.. math::
    :label: ref-alias

    \operatorname{E}[i] = \mu + A(\mu), \qquad
    A(\mu) = \sum_{k \ge 1} \frac{(-1)^k}{\pi k}\, \chi(2\pi k)\, \sin 2\pi k\mu,
    \qquad
    |A_{\text{true}} - A_{\text{model}}| \le \sum_{k \ge 1} \frac{|\chi_{\text{true}}(2\pi k) - \chi_{\text{model}}(2\pi k)|}{\pi k},

from the sawtooth's series (:eq:`det-pixel-moments`) and :math:`\operatorname{E}[\sin 2\pi k x] = \chi(2\pi k)\sin
2\pi k\mu`. A model whose binned mean matches the data's is wrong in :math:`\mu` by :math:`A_{\text{true}} -
A_{\text{model}}` (over :math:`1 + A'`): zero on the symmetry points (:eq:`ref-symmetric`), largest between. Over a
uniform sub-cell position it has mean zero and mean square :math:`\sum_k (\Delta\chi_k)^2 / (2\pi^2 k^2)`.

**Proof:** ``tests/theory/test_refinement.py::test_alias_any_symmetric_spot`` (a core-and-wing mixture and a Laplace
spot against their cells' masses at 20 digits; the sines' orthogonality).

The amplitude is set by :math:`\chi(2\pi)`, which the narrowest part of a spot dominates, not by its variance. The
measured point spread (core 0.25 px holding 72 %, wing 1.3 px, AGENTS.md) has :math:`\chi(2\pi) = 0.210`; drawn as one
Gaussian of its variance (sd 0.720 px, :math:`\chi(2\pi) = 3.6 \times 10^{-5}`) its binned mean is wrong by up to
0.0667 px (near a quarter pixel), 0.0472 px rms over the position, within the bound's 0.0676. At 180 mm a 75 µm
pixel's 0.067 is :math:`2.8 \times 10^{-5}` rad, a strain of :math:`1.7 \times 10^{-4}` at :math:`\tan\theta = 0.08`.

**Proof:** ``tests/theory/test_refinement.py::test_alias_numbers``

The same holds in omega in units of the frame width: a peak whose core is narrower than a frame (ID11's, AGENTS.md)
has :math:`\chi(2\pi)` near 1. A spot of sd 0.5 cell or more has :math:`\chi(2\pi) \le 0.0072` and negligible aliasing
whatever its shape. Across reflections the sub-cell positions are effectively independent, so the aliasing of a
voxel's mean over :math:`M` reflections falls as :math:`1/\sqrt{M}`; but neighbouring voxels see a reflection at
nearly the same position, so in a map it is a spatially correlated error, not noise.

Orientation spread: what a mean keeps
-------------------------------------

A population's lattice turned by :math:`\exp([\vec{d}]_\times)`, :math:`\vec{d}` of mean 0 and covariance
:math:`\matr{S}` (any distribution, symmetric for the third order to vanish), moves its scattering vectors on average
to

.. math::
    :label: ref-spread-mean

    \operatorname{E}[\matr{R}] = \matr{I} + \tfrac{1}{2}(\matr{S} - \operatorname{tr}\matr{S}\,\matr{I}) + O(\matr{S}^2),
    \qquad
    \frac{\hat{\vec{g}}\cdot\operatorname{E}[\matr{R}]\vec{g}}{|\vec{g}|} - 1
    = -\tfrac{1}{2}\left(\operatorname{tr}\matr{S} - \hat{\vec{g}}^T\matr{S}\hat{\vec{g}}\right),
    \qquad
    \text{across: } \tfrac{1}{2}(\matr{I} - \hat{\vec{g}}\hat{\vec{g}}^T)\matr{S}\hat{\vec{g}},

while :math:`|\matr{R}\vec{g}| = |\vec{g}|` exactly.

**Proof:** ``tests/theory/test_refinement.py::test_spread_mean`` (Rodrigues' formula expanded to second order, its
expectation with only the second moments of :math:`\vec{d}`).

So the centroid of a population's light in Cartesian :math:`\vec{g}` reads an isotropic spread :math:`\sigma` as a
strain :math:`-\sigma^2`: :math:`-7.62 \times 10^{-5}` at 0.5°, :math:`-3.05 \times 10^{-4}` at 1°,
:math:`-1.22 \times 10^{-3}` at 2°, against strain contrasts of :math:`10^{-4}`; the mean of each cell's
:math:`|\vec{g}|` (its 2θ) is exact at any spread. An anisotropic spread also moves the mean direction: about one axis
at 45° to :math:`\vec{g}`, by :math:`\sigma^2/4`, :math:`7.62 \times 10^{-5}` rad at 1°.

**Proof:** ``tests/theory/test_refinement.py::test_spread_numbers``

A first-order model makes the opposite error. The renderer spreads a peak along a straight line in :math:`(s, f)`
(:eq:`ps-propagate`), tangent to the ring at the peak; a ring of radius :math:`\rho` pixels curves away from it, and a
spread of sd :math:`\sigma_t = \rho\sigma_\eta` along the tangent puts its cells' mean radius at

.. math::
    :label: ref-ring

    \bar\rho = \rho + \frac{\sigma_t^2}{2\rho}, \qquad
    \Delta 2\theta = \tfrac{1}{2}\sigma_\eta^2 \sin 2\theta\cos 2\theta, \qquad
    \varepsilon_{\text{read}} = -\frac{\Delta 2\theta}{2\tan\theta} = -\tfrac{1}{2}\sigma_\eta^2\cos^2\theta\cos 2\theta

(a detector normal to the beam, :math:`\rho = L\tan 2\theta`): :math:`-1.52 \times 10^{-4}` at
:math:`\sigma_\eta = 1°`.

**Proof:** ``tests/theory/test_refinement.py::test_tangent_spot_on_a_ring``

So a strain refiner whose model draws spreads at first order carries a bias of order :math:`\sigma_\eta^2/2` per
reflection, a function of each population's spread, which varies from voxel to voxel: not a common offset
(:eq:`lim-common`), so it reaches the map. Below about 0.3° it is under :math:`1.4 \times 10^{-5}`; above, the
model's mean must carry the second-order terms of :eq:`ref-spread-mean` (with :math:`\matr{S}` from the second
moments, :doc:`moments`), or render exact rotations (:doc:`peak_shape`).

Light beyond a region
---------------------

A region that leaves out some of a population's light changes its first moment by the moment of what is left out.
The face bounds of the renderer (:eq:`ps-faces`, :eq:`ps-face-reach`) give that for the model's own Gaussians; for the
truth, whose shape is known only to its variance :math:`\sigma^2`, the light beyond a face at :math:`t > 0` from the
mean carries at most

.. math::
    :label: ref-tail

    \operatorname{E}\left[(x - \mu)\, \mathbb{1}(x - \mu \ge t)\right] \le \frac{\sigma^2}{t}

of the first moment (per photon), since :math:`x\,\mathbb{1}(x \ge t) \le x^2/t` everywhere; a spot of two points
(mass :math:`p` at :math:`t`) attains it as :math:`p \to 0`.

**Proof:** ``tests/theory/test_refinement.py::test_tail_first_moment``

At :math:`t = 4\sigma` that is :math:`\sigma/4`, against :math:`1.3 \times 10^{-4}\sigma` for a Gaussian: with the
shape unknown beyond its variance, a region cut on one side is not robust at any practical margin. A region symmetric
about a separated spot needs no bound (:eq:`ref-symmetric`); a region holding several populations needs every face
far out in all of them.

Splitting merged light
----------------------

Where two populations' light overlaps, the share of each cell that belongs to each is a matter of shape. Take two
spots of one shape, equal light, :math:`d` apart in units of their sd, fitted by the Poisson likelihood of a
continuous record with two Gaussians of a free common width :math:`s` (the nuisance of :eq:`ref-profile`). As the
photons grow, the fit tends to the root of the expected score (the Poisson record's score, :math:`\int \nabla\log
\lambda_{\text{model}}\, dN - \nabla \int \lambda_{\text{model}}`, whose second term vanishes for a normalised
model):

.. math::
    :label: ref-split

    \int \lambda_{\text{true}}(x)\, \nabla_{a, s} \log\lambda_{\text{model}}(x; a, s)\, dx = 0,
    \qquad
    \lambda_{\text{model}} = \tfrac{1}{2}\left[\varphi_s(x - a/2) + \varphi_s(x + a/2)\right],

its noise the sandwich of :eq:`ref-bias` (:math:`\matr{H}^{-1}\matr{B}\matr{H}^{-1}/N`). With a true shape of a core
and a wing (unit variance), the separation's bias :math:`a - d`, its standard deviation times :math:`\sqrt{N}`, and
the photons :math:`N` of the pair at which the two are equal:

.. list-table::
    :header-rows: 1

    * - true shape
      - :math:`d = 3`
      - :math:`d = 4`
      - :math:`d = 6`
    * - wing 5.2 × the core, 28 % (the measured point spread's)
      - 0.210, 1.53, 54
      - 0.129, 1.70, 170
      - 0.0383, 1.88, 2400
    * - wing 2 × the core, 10 %
      - 0.0333, 2.02, 3700
      - 0.0318, 1.94, 3700
      - 0.0113, 1.97, 30000
    * - Gaussian
      - 0
      - 0
      - 0 (noise 2.00: each spot's centroid alone)

**Definition:** the fit's limit under a wrong shape, as :eq:`ref-bias`'s with the score as the statistic.
**Proof:**
``tests/theory/test_refinement.py::test_split_bias`` (the table, Newton on the expected score with its exact Jacobian),
``tests/theory/test_refinement.py::test_log_pair_derivs``.

Below :math:`d \approx 2` the fit merges the two into one wider spot (:math:`a = 0`), as for any two Gaussians of a
free width. Between, every photon beyond a few hundred to a few thousand per pair makes the split more precise but no
more accurate: real spots hold :math:`10^3`-:math:`10^6`. So splitting merged light by its shape is bias-dominated for
any plausible shape error unless the shape is known to the precision of the counts, which the point spread's wing
(:doc:`detector`) and a population's own spread are not. The robust alternative keeps the light whole: its counts
and moments over a region holding all of it are the beam- and density-weighted ray sums of every contributor's own
moments (:eq:`mom-raysum`), exact for any shape, and they determine the merged contributors as a tomography
(:doc:`limits`), with no information on their separation.

What follows for a refiner
--------------------------

Every statistic above that is robust either keeps one population's light alone or a set of populations' light
whole: none divides light by its shape. With no spot shape trusted at all, :doc:`refiner` builds the refiner on that:
deformation moments as an exactly linear tomography of region sums, and the noise, the cut, the group rule and the
pixels each bounded.

What is left out
----------------

The questions this page left open (the cut's deficit in merged light, splitting in two and three dimensions, the
likelihood's aliasing, the separation threshold) belong to methods that divide merged light by its shape. The refiner
of :doc:`refiner` never does: it bounds the cut per cell (:eq:`rf-cut-bound`) and tells populations apart only where
the regions separate them (:eq:`rf-identify`). They stay open only for a shape-based method.
