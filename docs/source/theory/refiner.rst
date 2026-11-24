.. _theory_refiner:

A refiner that needs no spot shape
==================================

The refiner's task is each population's orientation and strain, from the recorded cells (:doc:`recording`), with
no model of any spot's shape. This chapter derives one: its observables are exactly linear in shape-free unknowns.
What noise, the cut, the group rule, the pixels and the instrument do to it is proved, term by term, as an identity
or a bound computed from the geometry and the data. Where no bound can exist, the chapter proves that too.

*Cell* means, as in :doc:`recording`, one pixel in one frame of one dty row.

It builds on :doc:`robustness`, whose lemmas it uses: a wrong weight never biases a fit (:eq:`ref-bias`), symmetry
protects a separated location (:eq:`ref-symmetric`), a spread moves a mean at second order (:eq:`ref-spread-mean`),
light beyond a region is bounded by its variance (:eq:`ref-tail`), and splitting merged light by shape is
bias-dominated (:eq:`ref-split`), which is why this refiner never does it.

Assumptions
-----------

Nothing is assumed about the shape of any spot, about a population's orientation or strain distribution, or about
the shape of the detector's point spread beyond (A5). The assumptions are:

- **(A1) Counting.** Counts are Poisson and independent over cells given the light (:eq:`rec-poisson`); they are
  recorded by the cut and the group rule (:eq:`rec-recorded`).
- **(A2) Kinematic scattering.** Every photon of a sub-volume in reflection :math:`h` has the scattering vector
  :math:`\vec{g} = \matr{F}\vec{g}_0` of that sub-volume. Here :math:`\vec{g}_0 = \matr{U}_k\matr{B}\vec{h}` is
  fixed per population :math:`k` (the indexer's orientation), and :math:`\matr{F}` is the sub-volume's
  reciprocal-lattice deformation: any 3 × 3 matrix, its rotation and its strain together. For a real-space
  distortion :math:`\matr{\beta}`, :math:`\matr{F} = (\matr{I} + \matr{\beta})^{-T} \approx \matr{I} - \matr{\beta}^T`
  (:eq:`lim-reciprocal`): statements about dislocations or compatibility are about :math:`\matr{\beta}`, not
  :math:`\matr{F}`.
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
  function at :math:`2\pi k` per pixel is used (:eq:`rf-alias-bound`). In words: no spot is ever sharper than the
  detector's own blur, so a spot's centre is pinned well within a pixel.
- **(A6) The instrument's omega kernel is a factor.** Every peak's omega profile is the instrument's omega kernel
  convolved with something non-negative and otherwise arbitrary. The kernel is a Gaussian whose variance is
  :eq:`rf-omega-kernel` with three widths (horizontal and vertical convergence, bandwidth), measured on a single
  rotation of an undeformed Si crystal and taken at the low end of their uncertainty. It is an operating condition of
  the refinement: such a Si scan, with reflections on at least two rings, is measured with every experiment
  (:eq:`rf-omega-identify`). The kernel is never fitted to the sample.

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

    \operatorname{E}\left[(1 + \vec{l}^T\vec{d})\exp([\vec{d}]_\times)\right]
    = \matr{I} + \tfrac{1}{2}(\matr{S} - \operatorname{tr}\matr{S}\,\matr{I}) + [\matr{S}\vec{l}]_\times + O(d^3),
    \qquad \matr{S} = \operatorname{Cov}(\vec{d}),

the unweighted mean (:eq:`ref-spread-mean`) plus a turn, with :math:`|\vec{l}| \approx |\cot\eta|`. That is a turn of :math:`|\matr{S}\vec{l}| \sim \sigma^2|\cot\eta|`:
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
independent of :math:`\vec{g}`, with :math:`\operatorname{E}[\vec{D}] = 0`. Then, exactly for this model of the
measurement,

.. math::
    :label: rf-instrument

    \operatorname{E}[\hat{\vec{g}}] = \operatorname{E}[s]\operatorname{E}[\vec{g}], \qquad
    \operatorname{E}|\hat{\vec{g}}|^2 = \operatorname{E}[s^2]\left(\operatorname{E}|\vec{g}|^2 + \operatorname{E}|\vec{D}|^2\right).

The model leaves one thing out: a photon from another incoming direction or wavelength diffracts at a slightly
different omega, while the statistic turns every photon back by its frame's centre, a rotation about the axis
correlated with :math:`\vec{D}`. Its mean is zero to first order; its second order is not bounded here.
Each is a factor and an offset common to every population: a scale fixed by calibration, or left free as one number
for the whole map (:eq:`lim-common`: it biases only region means, like :math:`d_0`).

**Proof:** ``tests/theory/test_refiner.py::test_instrument_independent``

The independence holds because every sub-volume rotates through the whole spectrum and every incoming direction:
the frames integrate the rotation, so a region holding all of a population's frames in a reflection samples them
all. The detector's displacement enters through :math:`\hat{\vec{g}}(\vec{x}_p)` at the displaced position. Its mean
is zero (the point spread, by definition) or calibrated (the sensor's), and its second moments give a term
:math:`\tfrac{1}{2}\operatorname{tr}(\matr{H}_f \matr{\Sigma}_e)`, with :math:`\matr{H}_f` the feature's Hessian
on the detector. For a feature of the detector radius :math:`\rho` alone, such as :math:`|\vec{g}|^2 =
(2/\lambda^2)(1 - L/\sqrt{L^2 + \rho^2})` on a detector normal to the beam at distance :math:`L`, the Hessian's
eigenvalues are :math:`f''` (radial) and :math:`f'/\rho` (tangential), so

.. math::
    :label: rf-displacement

    \tfrac{1}{2}\left(f''\sigma_r^2 + \frac{f'}{\rho}\sigma_t^2\right) = f\,\frac{\sigma_r^2 + \sigma_t^2}{\rho^2}\,
    \big(1 + O(\rho^2/L^2)\big):

:math:`5.7 \times 10^{-6}` of :math:`|\vec{g}|^2` for a point spread of 0.5 px² per axis at :math:`\rho = 420` px.
It is the same for every population at one place on the detector, so it goes into the common scale
(:eq:`lim-common`).

**Proof:** ``tests/theory/test_refiner.py::test_displacement_common_term``

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
neither (AGENTS.md rules out basis-function priors and coupling between voxels), so it reports each voxel's error bar
and bias bound instead (:eq:`rf-linear`).

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

from the sawtooth's series (:eq:`det-pixel-moments`), symmetric or not. It is the general form of :eq:`ref-alias`, which
is the error of one wrongly shaped model. The bound is 0.0676 px for the measured point
spread (a core of 0.25 px holding 72 %, a wing of 1.3 px). A feature's first-order binning error is its gradient times
:math:`\operatorname{E}[u]`, and :math:`\bar\Delta` takes it per region with the gradient's largest value there.

**Proof:** ``tests/theory/test_refiner.py::test_alias_bound_any_spot`` (the bound; asymmetric spots, the point spread
convolved with exponentials of either sign, against their cells' masses).

**Omega.** Frames bin omega as pixels bin the detector, and (A6) does for omega what (A5) does for the pixels. With
no lattice rotation, :eq:`ps-omega` gives a reflection's omega shift as three independent terms, so the instrument's
omega kernel has variance

.. math::
    :label: rf-omega-kernel

    v(\eta, \theta) = \sigma_h^2 + \sigma_v^2\cot^2\eta + \sigma_\lambda^2\,\frac{\tan^2\theta}{\sin^2\eta}
    \quad (\text{radians}^2), \qquad
    N(0, v) = N(0, v_{\text{lo}}) * N(0, v - v_{\text{lo}}),

with :math:`\sigma_h`, :math:`\sigma_v` the horizontal and vertical convergence and :math:`\sigma_\lambda` the
bandwidth. A lower variance :math:`v_{\text{lo}} \le v` is still a factor of the kernel, so the low end of the
measured widths can be used safely.

**Proof:** ``tests/theory/test_refiner.py::test_omega_kernel_variance``

With :math:`r = \sqrt{v_{\text{lo}}}/h` the floor's sd in frames, :eq:`rf-alias-bound` in frames gives

.. math::
    :label: rf-omega-bound

    |\operatorname{E}[\omega_j - \omega]| \le h\,\min\Big(\tfrac{1}{2},\ \sum_{k \ge 1} \frac{e^{-2\pi^2k^2r^2}}{\pi k}\Big):
    \quad 0.5,\ 0.355,\ 0.151,\ 0.054,\ 0.0023 \text{ frames at } r = 0.05,\ 0.1,\ 0.2,\ 0.3,\ 0.5.

**Proof:** ``tests/theory/test_refiner.py::test_omega_floor_table``

The guarantee is useful once the instrument alone spreads every peak over 0.2-0.3 of a frame. At ID11 with 0.05°
frames, the bandwidth gives 0.012 frames at :math:`|\sin\eta| = 1` and 0.058 at 0.2 (3e-4 FWHM, :math:`\tan\theta =
0.08`), and horizontal convergence of :math:`8 \times 10^{-5}` rad gives 0.092. So the bound pins omega near
:math:`\eta = 0` and 180°, where vertical convergence times :math:`|\cot\eta|` widens the kernel. Elsewhere sharp
peaks keep a worst case of about a third of a frame. Finer frames shrink it in proportion. The bound concerns only the
rotation about the axis: it does not touch :math:`|\vec{g}|^2` (:eq:`rf-invariant`), nor the rotations that move a peak
along or across the ring. Over :math:`M` reflections whose sub-frame positions are independent and uniform, the errors
have mean 0 and rms at most :math:`h/\sqrt{12}`, so a voxel's rotation about the axis is off by :math:`h/\sqrt{12M}` in
rms. The guarantee stays the worst case, because the positions are set by the geometry, not drawn.

**The three widths are determined by one Si rotation.** A peak narrower than a frame cannot show its own width in its
own frames. What shows the kernel is many reflections whose predicted omegas fall at different places in their
frames. Si's orientation is known after calibration far better than a frame, so each reflection's predicted position
:math:`\mu` within its frame is known. Two facts then decide it.

.. math::
    :label: rf-omega-identify

    \frac{\partial}{\partial v}\Big[\Phi\big(\tfrac{a}{\sqrt{v}}\big) - \Phi\big(\tfrac{b}{\sqrt{v}}\big)\Big]
    = -\frac{a\,\varphi(a/\sqrt{v}) + |b|\,\varphi(b/\sqrt{v})}{2v^{3/2}} < 0, \qquad
    \det\begin{bmatrix} 1 & c_1 & t_1(1 + c_1) \\ 1 & c_2 & t_1(1 + c_2) \\ 1 & c_1 & t_2(1 + c_1) \end{bmatrix}
    = (c_2 - c_1)(t_2 - t_1)(1 + c_1),

with :math:`a = \tfrac{1}{2} - \mu > 0 > b = -\tfrac{1}{2} - \mu` (in frames), :math:`c = \cot^2\eta` and
:math:`t = \tan^2\theta`. First, the share of a reflection's light in its own frame falls strictly with its kernel's
variance, so each reflection's frames determine its :math:`v`. That needs only some light outside the frame: zero
information comes only from a peak wholly inside it. Second, :math:`v = \sigma_h^2 + \sigma_v^2 c +
\sigma_\lambda^2 t(1 + c)` is linear in the three variances, and three reflections on two rings at two values of
:math:`\eta` determine them (the determinant). On one ring the third column is :math:`t` times the sum of the first
two, and the rank is 2 for any number of reflections. Bandwidth and convergence then cannot be told apart.

**Proof:** ``tests/theory/test_refiner.py::test_omega_identifiable`` (the derivative; the determinant; one ring's rank).

So 500 or more Si reflections on two or more rings determine the kernel; the condition is the rings, not the count.
Only reflections within a few kernel widths of a frame's edge carry information. With a kernel of sd :math:`\sigma`
frames that is about a fraction :math:`6\sigma` of them, about 35 of 500 at :math:`\sigma = 0.012`: still many more
than three unknowns.

**How precisely.** A Si reflection's counts in its frames are Poisson with means :math:`N P_j(\mu, v)`. Its unknown
intensity :math:`N` costs nothing: the cross information with :math:`v` is :math:`\sum_j \partial_v P_j = 0`. Through
:math:`v_r = \vec{x}_r\cdot\vec{\vartheta}`, with :math:`\vec{\vartheta}` the three squared widths and :math:`\vec{x}_r =
(1, \cot^2\eta_r, \tan^2\theta_r/\sin^2\eta_r)/h^2`, the Fisher information on the widths is

.. math::
    :label: rf-si-fisher

    \matr{I}(\vec{\vartheta}) = \sum_r I_r\, \vec{x}_r\vec{x}_r^T, \qquad
    I_r = N_r \sum_j \frac{(\partial_v P_j)^2}{P_j},

exactly, from each reflection's own predicted position. Its inverse bounds the widths' covariance.

**Proof:** ``tests/theory/test_refiner.py::test_si_fisher``

For a kernel of sd :math:`s` frames, much narrower than a frame, only the nearer edge matters. At distance :math:`d`
from it, :math:`q = \Phi(-d/s)` spills over, :math:`\partial_v q = d\,\varphi(d/s)/(2s^3)`, and over reflections spread
evenly in :math:`d`,

.. math::
    :label: rf-si-narrow

    \overline{I_r} = \frac{N K}{2 s^3}, \quad K = \int_0^\infty \frac{z^2\varphi(z)^2}{\Phi(z)\Phi(-z)}\,dz = 1.10,
    \qquad
    \frac{\operatorname{sd}(s)}{s} = \frac{1}{\sqrt{2 M N K s}} \quad (M \text{ reflections}).

So the narrower the kernel, the more information each reflection gives on its variance, but the fewer reflections give
any: only those within a few :math:`s` of an edge, a fraction of about :math:`s`. That is why :math:`s` stays in the
relative precision. For :math:`M = 500` reflections of :math:`N = 10^4` photons it is 0.27 % at :math:`s = 0.012`
frames (the bandwidth's) and 0.099 % at 0.092 (the horizontal convergence's); 0.87 % at :math:`N = 10^3`. The limit
holds while the far edge is negligible, :math:`s \lesssim 0.15`. Beyond that, :eq:`rf-si-fisher` is summed in full.

**Proof:** ``tests/theory/test_refiner.py::test_si_fisher_narrow`` (the derivative; the change of variables
:math:`d = sz`; :math:`K`; the numbers).

The positions must be fitted with the widths, never taken as given. The frame masses are :math:`G(j - \mu, v)` with
:math:`G` even in its first argument, so :math:`G_x` is odd and :math:`G_v` even. The cross information between a
reflection's position and :math:`v` is therefore odd in :math:`\mu`. Over reflections spread symmetrically within their
frames it vanishes, so the Si orientation fitted jointly costs the widths nothing on average. Left unfitted, a position
error of sd :math:`\tau` adds :math:`\tau^2` to every :math:`v` (:eq:`rf-omega-kernel`: Gaussians convolve). That
raises the floor, the one unsafe direction for (A6). Si's orientation error after calibration, about the pixel
angle over :math:`\sqrt{M}`, is of the order of 0.02 frames at ID11. That is larger than the bandwidth's 0.012, so
fitting the positions jointly is necessary, not optional.

**Proof:** ``tests/theory/test_refiner.py::test_si_position_orthogonal``,
``tests/theory/test_refiner.py::test_si_position_number``

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
trade the two. In a box of half-width :math:`t` cells along each of three axes, a first moment along one axis sums
:math:`|x|` over :math:`(2t + 1)^2 t(t + 1) \approx 4t^4` cells. With the light beyond the faces known only through
its variance :math:`\sigma^2`,

.. math::
    :label: rf-box-width

    B(t) = \frac{4\kappa_c t^4}{N} + \frac{\sigma^2}{t}, \qquad
    t_* = \Big(\frac{\sigma^2 N}{16\kappa_c}\Big)^{1/5}, \qquad B(t_*) = \frac{5}{4}\frac{\sigma^2}{t_*}:

4.43 cells and 0.282 cells for :math:`\sigma = 1` cell, :math:`N = 10^4`, :math:`c = 1`. The worst case falls only
as :math:`N^{-1/5}`, because a tail known only by its variance can hide anywhere. This sets the half-widths before the
fit, from the indexer's spread and the instrument's width, never from the data being fitted. After the fit, the
bound on the light beyond the faces is the measured one of :eq:`rf-shift`, usually much smaller.

**Proof:** ``tests/theory/test_refiner.py::test_box_half_width``

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
neighbours are removed whole at :math:`m = 3`, however bright. (A5) rules such spots out, with one more property of
the point spread: that it is radially non-increasing, as a mixture of centred Gaussians is. Two facts follow.

**Leak inside a cell.** For a photon arriving anywhere in cell :math:`p`, each edge neighbour receives at least

.. math::
    :label: rf-psf-leak

    \ell = \sum_c f_c\left[\Phi(2/s_c) - \Phi(1/s_c)\right]\left[\Phi(1/s_c) - \tfrac{1}{2}\right]

of it (components of share :math:`f_c` and sd :math:`s_c`, in pixels). Each component's mass factorises; the mass
across the edge grows as the photon nears it; the mass along the edge falls as the photon leaves the middle. So the
least is at the far corner. For the measured point spread, :math:`\ell = 0.0124`.

**Reflection outside a cell.** A photon arriving beyond an edge of :math:`p` puts at least as much light in the cell
across that edge as in :math:`p`. Reflecting across the edge's line maps :math:`p` onto that cell and brings every
point of :math:`p` closer to the photon.

**Proof:** ``tests/theory/test_refiner.py::test_psf_leak_number`` (the monotonicities and the number),
``tests/theory/test_refiner.py::test_reflection_lemma``

Together, whatever the light arriving on the detector, the four edge neighbours of a cell hold at least
:math:`\ell\lambda_p`, and those of a pair of cells (six) at least :math:`\min(\ell, \tfrac{1}{2})` of the pair's
light. The largest of :math:`n` cells holds at least their mean, and :math:`P(n \le c)` falls with the light. So a cell
is in a group smaller than 3 with probability

.. math::
    :label: rf-group-psf

    1 - w_p \le 9\, P\big(n \le c \mid \ell\lambda_p/6\big), \qquad
    e_p(1 - w_p) \le 9\lambda_p P\big(n \le c \mid \ell\lambda_p/6\big) \le \frac{54\,\kappa_{c+1}}{\ell},

for any spot. The first term is the cell alone, the other eight the cell with one kept neighbour. The constant is
:math:`3.6 \times 10^3` photons per cell for the measured point spread, reached near :math:`\lambda_p \approx
800`.

**Proof:** ``tests/theory/test_refiner.py::test_group_rule_psf_bound``

The bound is rigorous and weak. It takes every photon at the worst place in its cell (the corner), every neighbour
at its least, and a union over eight pairs. A spot centred on a pixel leaks about three times :math:`\ell` into each
neighbour, and :doc:`recording`'s table shows such spots kept above about 100 photons. Only a model of where in the
cell the light falls would close the gap, and that is a shape. The page's own conclusion stands: in faint compact
spots the group rule's loss is bounded, not known. :eq:`rf-group-psf` puts it in :math:`\bar\Delta` as the
worst case; :eq:`rf-group` is the sharper bound wherever the neighbours' recorded light shows them bright.

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
and :math:`\hat{\vec{z}}`, and its action on :math:`\hat{\vec{\xi}}` only through :math:`s`. To first order
:math:`\matr{F}\hat{\vec{\xi}} - \hat{\vec{\xi}} = -\matr{\beta}^T\hat{\vec{\xi}}`, the row :math:`\hat{\vec{\xi}}` of the
real-space distortion (:eq:`lim-slice`, :eq:`lim-weak-parts`). This holds whatever is
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

**Proof:** ``tests/theory/test_refiner.py::test_weak_amplification``,
``tests/theory/test_refiner.py::test_sin_theta_numbers``

**On a voxel grid they are seen through the model's geometry.** :eq:`rf-weak-any` is exact for continuous fields.
A reciprocal deformation constant on voxels of step :math:`h`, :math:`\matr{\beta}^*(\vec{x}) = \sum_v \matr{B}_v\chi(\vec{x} -
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

**What to estimate.** Every voxel's numbers are estimated by the left inverse of :eq:`rf-linear`, each with its own
covariance and its own bias bound :math:`\sum_j |A_{ij}|\bar\Delta_j`, and reported as not determined where the bound
or the error bar exceeds the precision sought. Nothing is removed from the estimate and no voxel's value depends on a
prior over its neighbours'; the singular directions (:eq:`rf-smin`) are a diagnostic of which combinations are weak,
not a step of the estimate.

**Anri's rule:** no truncation. Keeping only the singular directions with :math:`s_i \ge s_{\min}` would estimate
:math:`P_{\text{seen}}\vec{Y}` (:eq:`rf-project`), not :math:`\vec{Y}`: the right singular vectors span many voxels,
so each voxel's value would become a combination of its neighbours' (the resolution matrix :math:`P_{\text{seen}} =
\sum_{s_i \ge s_{\min}}\vec{v}_i\vec{v}_i^T`), and the directions dropped include those the beam's transfer
attenuates (:eq:`lim-transfer`): a coupling between voxels and a low-pass filter. Its bias bound, below, is stated so
that the cost of what Anri does not do is known.

.. math::
    :label: rf-project

    \hat{\vec{Y}}_{\text{seen}} = \sum_{s_i \ge s_{\min}} \frac{\langle\vec{u}_i, \vec{T}\rangle}{s_i}\,\vec{v}_i,
    \qquad
    |\operatorname{E}[\hat{\vec{Y}}_{\text{seen}}] - P_{\text{seen}}\vec{Y}| \le \frac{|\vec{\Delta}|}{s_{\min}}.

**Proof:** ``tests/theory/test_refiner.py::test_projected_estimator``

No constraint between voxels fills the weak combinations (:doc:`limits`, Anri's rule): a no-dislocation
constraint would fill two of the three and couple neighbouring voxels. Separated populations are not affected: alone
in their regions, they are seen in every reflection with all nine numbers (:doc:`limits`), with no tomography to
weaken them.

One linear solve, and its split
-------------------------------

**Boxes from the indexer are complete at once.** The regions depend on where the spots are, and so on
:math:`\vec{Y}`. Iterating them is unnecessary if the first boxes already hold the light. The indexer's orientation
is within a rotation :math:`|\vec{\delta}| \le e` of a population's mean (:doc:`localsearch` gives :math:`e` per
level). A rotation moves a spot by :math:`\Delta\omega = \vec{j}_\omega\cdot\vec{\delta}` and :math:`\Delta\eta =
\vec{j}_\eta\cdot\vec{\delta}` (:eq:`dif-rotation-motion`). A strain :math:`|\varepsilon| \le
\varepsilon_{\max}` changes :math:`2\theta`, and also turns :math:`\vec{g}` by up to about :math:`\varepsilon_{\max}`
(:eq:`rf-box-margin-exact`), which adds to :math:`e`. So, by Cauchy-Schwarz,

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

To all orders, along the rotation :math:`\exp(t[\vec{\delta}]_\times)`, :math:`t \in [0, 1]`, the spot moves at
:eq:`dif-rotation-motion`'s rates evaluated where it is. With :math:`s_0 = |\sin\eta|` at the start and :math:`s_*`
the larger root of :math:`s = s_0 - e(1 + \tan\theta/s)`, :math:`|\sin\eta|` stays above :math:`s_*` along the
whole path. At a first touch it would be at least :math:`s_0 - e\sqrt{1 + \tan^2\theta/s_*^2} > s_*`. So, exactly,

.. math::
    :label: rf-box-margin-exact

    |\Delta\omega| \le \frac{e}{s_*} = \frac{e}{s_0} + \frac{e^2(s_0 + \tan\theta)}{s_0^3} + O(e^3), \qquad
    |\Delta\eta| \le e\sqrt{1 + \frac{\tan^2\theta}{s_*^2}};
    \qquad
    \sin\angle\big((\matr{I} + \varepsilon)\vec{u}, \vec{u}\big) \le \frac{\|\varepsilon\|}{1 - \|\varepsilon\|},
    \quad \frac{|(\matr{I} + \varepsilon)\vec{u}|}{|\vec{u}|} \in [1 - \|\varepsilon\|, 1 + \|\varepsilon\|],

for real :math:`s_*` (:math:`(s_0 - e)^2 \ge 4e\tan\theta`). These margins hold for rotations of any size.
Reflections too close to the axis for a real :math:`s_*` are left out. A strain, of operator norm
:math:`\|\varepsilon\|`, turns :math:`\vec{g}` by the angle on the right (since :math:`(\matr{I} +
\varepsilon)\vec{u}\times\vec{u} = \varepsilon\vec{u}\times\vec{u}`), which adds to :math:`e`, and changes
:math:`|\vec{g}|` within the factors given. This bounds :math:`2\theta` exactly through Bragg's law.

**Proof:** ``tests/theory/test_refiner.py::test_box_margin_exact`` (the root and its expansion; :math:`\sqrt{1 +
T^2/s^2} < 1 + T/s`; the strain's cross product).

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

**The split, for the first moments.** :math:`\matr{Z}` maps Miller indices to the sample frame, so its columns are
not sample directions. With a reference :math:`\matr{U}_0\matr{B}` for a merged domain, :math:`\matr{M} =
\matr{Z}(\matr{U}_0\matr{B})^{-1}` is the domain's first moment of :math:`\matr{F}\matr{U}_k\matr{U}_0^{-1}`, a map of
the sample frame to itself, and :math:`\matr{Z}\vec{h} = \matr{M}\vec{g}_0` with :math:`\vec{g}_0 =
\matr{U}_0\matr{B}\vec{h}`. Within the domain, the field :math:`\matr{M}(\vec{x})` (zero outside the domain
:math:`\Omega`, whose other populations sit in other regions) is seen by its Fourier slices. At an in-plane
:math:`\vec{k}`, with :math:`\hat{\vec{\xi}} = \hat{\vec{z}}\times\hat{\vec{k}}`, each row :math:`i` of
:math:`\hat{\matr{M}}` has the in-plane columns :math:`(\hat M_{ix}, \hat M_{iy})`. These split as

.. math::
    :label: rf-split

    \hat{\vec{k}}\hat{\vec{k}}^T + \hat{\vec{\xi}}\hat{\vec{\xi}}^T = \matr{I}_2, \qquad
    |\vec{k}|\,\hat{\vec{\xi}}\cdot(\hat M_{ix}, \hat M_{iy}) = k_x\hat M_{iy} - k_y\hat M_{ix}
    = -i\,\widehat{(\operatorname{curl}\matr{M})}_{iz}, \qquad
    L^2(\Omega)^2 = \nabla H^1(\Omega) \oplus \{\vec{w} : \nabla\cdot\vec{w} = 0,\ \vec{w}\cdot\vec{n}|_{\partial\Omega} = 0\}.

The weak part is each row's solenoidal in-plane field, and the seen part its gradient. The third column
:math:`\matr{M}\hat{\vec{z}}` and the mean are seen whole. On the domain, the two parts are the Helmholtz
decomposition: the seen part is :math:`\nabla\phi_i`, with :math:`\phi_i` from the Neumann problem
:math:`\Delta\phi_i = \nabla\cdot\vec{v}_i` in :math:`\Omega`, :math:`\partial_n\phi_i = \vec{v}_i\cdot\vec{n}` on
its boundary. Orthogonality is :math:`\nabla\cdot(\phi\vec{w}) = \nabla\phi\cdot\vec{w} + \phi\nabla\cdot\vec{w}` and
the divergence theorem. It is three scalar Poisson problems per domain: sparse, local, and computable at any scale.

**Proof:** ``tests/theory/test_refiner.py::test_split_first_moments``

The curl of :math:`\matr{M}`'s rows is not the dislocation density. To first order :math:`\matr{M} - V\matr{I} = V
\matr{\beta}^* = -V\matr{\beta}^T` (per unit volume :math:`V`), and with :math:`\matr{\beta} = \varepsilon +
[\vec{w}]_\times`,

.. math::
    :label: rf-curl-reciprocal

    \epsilon_{jkl}\,\partial_k\beta^*_{il} = \alpha_{ij} - 2\,\epsilon_{jkl}\,\partial_k\varepsilon_{il},

the Nye tensor less twice the curl of the strain. So the weak (solenoidal) parts hold the dislocation density and
strain gradients together. Which entries the dislocations fix is :eq:`lim-weak-parts`: in :math:`\matr{M}`, the
solenoidal part of row :math:`z` (:math:`\beta_{\xi z}`, :math:`\alpha_{\xi\xi}`) and the entry
:math:`\hat{\vec{\xi}}^T\hat{\matr{M}}\hat{\vec{\xi}}` of the in-plane rows' solenoidal parts (:math:`\varepsilon_{\xi\xi}`,
:math:`\alpha_{\xi z}`), while the solenoidal parts of rows :math:`x` and :math:`y` along :math:`\hat{\vec{k}}`
(:math:`\hat{\vec{k}}^T\hat{\matr{M}}\hat{\vec{\xi}} = -\hat\beta_{\xi k}`) are fixed by none.

**Proof:** ``tests/theory/test_refiner.py::test_curl_reciprocal`` (fields :math:`\varepsilon(\vec{x})`,
:math:`\vec{w}(\vec{x})`, by differentiation).

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

**The solve.** In each merged domain the in-plane columns of :math:`\matr{M}` are written as :math:`\nabla\phi_i +
\vec{w}_i`, with the third column, the mean, and the populations of separated light as they are. This is an invertible
change of variables (:eq:`rf-split-grid`: the seen part and its complement span every field), so the least-squares
estimate in it is the per-voxel estimate exactly: it couples no voxels. It separates the weak part for its error bars
and for the iteration's conditioning (:eq:`rf-cost`). One estimate follows:

- :math:`\vec{w}_i` fitted with everything else, per voxel, reported with its covariance and flagged as seen only
  through :math:`\sin\theta` (or as not determined where its bias bound or error bar exceeds the precision sought).
  The same split applies to the metric and the spreads, with the Hessian in place of the gradient; there the doubly
  weak part is the strain's incompatibility (:eq:`rf-split-metric`), the same entry as :math:`\varepsilon_{\xi\xi}`
  above.

What the split does not remove is the beam's own weak family. A seen component at frequency :math:`\vec{k}` is
attenuated by the beam's transfer :math:`T(\vec{k})` (:eq:`lim-transfer`), so its noise and bias are amplified by
:math:`1/T`, by more than 4 at periods below 1.6 beam FWHM
(``tests/theory/test_refiner.py::test_beam_quarter_period``). That is the map's resolution. It is reported as a
resolution, through each voxel's error bar, never filtered: filtering would couple voxels.

**The split on the voxel grid.** On a grid the split needs no boundary conditions of its own. Let :math:`\matr{D}` be
the discrete gradient (for the first moments' in-plane columns) or the discrete Hessian (for the metric's in-plane
block), restricted to the domain's voxels, with its gauge removed: a constant for the gradient, an affine function
for the Hessian. Then :math:`\matr{D}` has full column rank, and

.. math::
    :label: rf-split-grid

    \matr{P} = \matr{D}(\matr{D}^T\matr{D})^{-1}\matr{D}^T, \qquad
    \matr{P} = \matr{P}^T = \matr{P}^2, \qquad \matr{P}\matr{D} = \matr{D}

is the orthogonal projector onto the seen fields; the weak part is :math:`(\matr{I} - \matr{P})`. The normal equations
:math:`\matr{D}^T\matr{D}\vec{\phi} = \matr{D}^T\vec{v}` carry the boundary conditions implicitly (in the continuum
limit: Neumann for the gradient; for the Hessian, those of a free plate edge, :eq:`rf-plate`). They are sparse,
symmetric and positive definite: conjugate gradients per domain. In one dimension every field is a gradient and nothing
is weak, as the slice argument says.

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

    \vec{v}_i \text{ determined} \iff s_i \ge s_{\min,i} = \frac{z + \sum_j |u_{ij}|\bar\Delta_j}{\epsilon},
    \qquad
    \frac{s_{\text{weak}}^2}{s_{\text{seen}}^2} = \frac{\sum_h s_h^2 a_h^2}{\sum_h a_h^2},

with :math:`\epsilon` the precision the map must reach. A weak column's coefficient in reflection :math:`h` is
:math:`s_h a_h`, where the seen column's is :math:`a_h`. So its gain relative to the seen one is the rms of the
reflections' :math:`s_h \sim \sin\theta_h`, weighted by how much each reflection sees.

**Proof:** ``tests/theory/test_refiner.py::test_smin``

The level is per direction and comes from the bounds of :eq:`rf-cut-bound`, :eq:`rf-group` and
:eq:`rf-alias-bound`, not from a choice. Where the bias bound is small, the noise sets it, and where the bias bound is
large it sets it. Only :math:`\epsilon` and :math:`z` are choices: the precision sought, and the confidence. The level
says which combinations the map determines; it removes none of them from the estimate (no truncation, above).

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

**Light that is not the population's.** Diffuse scattering about a Bragg point (thermal, from dislocation walls) is
part of the light no spot model is trusted with. Its variance can grow without bound with the box (one-phonon
scattering falls as :math:`1/|\vec{q}|^2`), so :eq:`ref-tail` does not bound it. Two properties do, with no model of
its shape. Light centrosymmetric about the Bragg point adds nothing to a first moment over a box symmetric about that
point (:eq:`ref-symmetric`). A box off centre by :math:`\delta` differs from the centred one by slabs of width
:math:`\delta` at its faces, and for any light :math:`\rho` along an axis

.. math::
    :label: rf-shift

    \int_{-t+\delta}^{t+\delta} x\rho\, dx - \int_{-t}^{t} x\rho\, dx
    = \int_{t}^{t+\delta} x\rho\, dx - \int_{-t}^{-t+\delta} x\rho\, dx,
    \qquad |\cdot| \le (t + \delta)\,\rho(\text{slabs}).

The box's centre is the indexer's prediction, within the margin of :eq:`rf-box-margin`. So such light biases a centred
first moment by at most the box's half-width times the light in the slabs at its faces. That light is in the data:
the recorded light of the faces' cells, plus at most :math:`\kappa_c` per cell (:eq:`rf-cut-bound`). Second moments
are another matter: any light in the box adds its own second moment, and nothing tells it from the population's
spread. A spread read from second moments is therefore an upper bound on the population's own, including whatever
diffuse light the box holds.

**Proof:** ``tests/theory/test_refiner.py::test_box_shift``

**Boxes that follow arcs.** A population's box in a reflection is a set of directions of :math:`\vec{g}`, not a
rectangle in :math:`(s, f, \omega)`. A rotation :math:`\exp([\vec{d}]_\times)` turns :math:`\hat{\vec{g}}` by
:math:`\vec{d}\times\hat{\vec{g}} = -[\hat{\vec{g}}]_\times\vec{d}` to first order. A linear map :math:`\matr{L}` takes
the ellipsoid :math:`\vec{d}^T\matr{S}^{-1}\vec{d} \le k^2` onto the ellipse :math:`\vec{y}^T(\matr{L}\matr{S}\matr{L}^T)^{-1}\vec{y}
\le k^2`: both have the support function :math:`k\sqrt{\vec{w}^T\matr{L}\matr{S}\matr{L}^T\vec{w}}`. So a population
whose rotations lie within :math:`k` of its mean, in the metric of its spread :math:`\matr{S}`, has its light's
directions within

.. math::
    :label: rf-arc-box

    \vec{y}^T\left([\hat{\vec{g}}]_\times\matr{S}[\hat{\vec{g}}]_\times^T\right)^{+}\vec{y} \le k^2, \qquad
    \left|\tfrac{1}{2}\vec{d}_\perp(\vec{d}\cdot\hat{\vec{g}})\right| \le \tfrac{1}{4}|\vec{d}|^2,

in the tangent plane at the mean :math:`\hat{\vec{g}}`. The ellipse is its first-order image (:math:`^{+}` the
pseudo-inverse in the plane). The second term bounds the second order, the arc's curvature, by which the box must be
widened. The box is that ellipse, widened by the indexer's error :math:`e` (:eq:`rf-box-margin`) and by the curvature
term, crossed with the :math:`2\theta` interval, and mapped to the cells whose :math:`\hat{\vec{g}}_p` fall in it.

**Proof:** ``tests/theory/test_refiner.py::test_arc_box`` (the image's covariance; the support function attained
and, by Cauchy-Schwarz in the metric :math:`\matr{S}`, not exceeded; the second-order term).

Most bananas need no such box. A banana on the detector is the sum, in each frame, of the spots of every voxel the
beam lights along its ray over the frame's omega range. Their slightly different orientations (dislocation cells, a
random walk of misorientation) spread it along the arc. After the indexer it is many populations with small spreads. Their region is
the union of their small boxes, and it follows the arc by construction. Only a single population with a large
anisotropic spread (bending about one axis, AGENTS.md) needs the ellipse. Its :math:`\matr{S}` comes from the
indexer, since boxes are fixed once (:eq:`rf-box-margin`).

Runout
------

The TDXRD stage moves the sample across the beam by up to about 1 µm, periodically in omega. A voxel at
:math:`(x, y)` is then in the beam at :math:`\text{dty} = y_0 - (x\sin\omega + y\cos\omega) + r(\omega)`
(:eq:`dif-dty`). Two facts decide what can be done with :math:`r`:

.. math::
    :label: rf-runout

    \text{translation by } (s_x, s_y):\ \Delta\text{dty} = -(s_x\sin\omega + s_y\cos\omega), \qquad
    \text{dty}(\omega + \pi) - \big(2y_0 - \text{dty}(\omega)\big) = r(\omega) + r(\omega + \pi).

The runout's first harmonic is exactly a translation of the whole sample, so it moves the map's origin and is not
a measurement: nothing needs it. Friedel pairs (:mod:`anri.index.drift`) see :math:`r(\omega) + r(\omega + \pi)`,
which holds its mean (:math:`y_0`) and its even harmonics, and none of its odd ones. So the second harmonic, a
stage's ellipticity, is measured by the pairs the indexer already fits. The odd harmonics from the third up are not
seen by the pairs. They enter :math:`\matr{G}` as a function of omega common to every voxel, a few global unknowns,
bilinear with :math:`\vec{Y}`. They are fitted by alternating with the linear solve, or bounded, and centring
(:eq:`rf-centre`) limits what they do to the deformation.

**Proof:** ``tests/theory/test_refiner.py::test_runout_harmonics`` (harmonics 1-4).

The odd harmonics from the third up are determined by the populations themselves. A population's counts against dty,
at the omega of each of its reflections, are the projection of its voxels. Their centroid is :math:`y_0 -
(\bar{x}\sin\omega + \bar{y}\cos\omega) + r(\omega)`, with :math:`(\bar{x}, \bar{y})` the centre of its light: a
constant and a first harmonic from the population, plus the runout shared by all. Over a full turn the harmonics are
orthogonal. So the runout's harmonics from the second up are fixed by the populations' centroid curves, less their
own first harmonics, once each population's reflections fall at enough distinct omegas (:math:`2n + 1` for
harmonics up to :math:`n`). Fitting them by alternating with the linear solve never raises the misfit. Near the
solution it contracts the error, per sweep, by the squared cosine of the angle between the runout's data pattern and
the map's:

.. math::
    :label: rf-runout-fit

    \text{centroid}(\omega) = y_0 - \bar{x}\sin\omega - \bar{y}\cos\omega + r(\omega), \qquad
    r - r_* \ \to\ \frac{(\vec{a}\cdot\vec{b})^2}{|\vec{a}|^2|\vec{b}|^2}\,(r - r_*) \ \text{per sweep}.

It stalls (cosine 1) only on a pattern the map can make itself, the first harmonic, which is fixed as the map's origin
instead.

**Proof:** ``tests/theory/test_refiner.py::test_runout_identifiable`` (the centroid; the harmonics' orthogonality),
``tests/theory/test_refiner.py::test_alternation_rate`` (one column per block).

Reflections at the ends of the omega range are complete only if their box lies within the scanned range. One whose
box (:eq:`rf-box-margin`) reaches past an end is left out: (A3) needs the whole rotation through the reflection.

Histograms and moments together
-------------------------------

The two statistics combine without a new model. Partition each reflection's directions into bins a few instrument
widths wide (:eq:`rf-bin-edge`), and take within each bin the counts and moments of :eq:`rf-statistic`. Then each
(population, bin) pair contributes, linearly (:eq:`rf-hist`),

.. math::
    :label: rf-bin-moments

    \sum_k a_{bk}|F_h|^2 A_{kh} \int_{\matr{F}\vec{g}_0 \in B} f(\matr{F}\vec{g}_0)\, d\mu_k(\matr{F})
    + O\!\left(\frac{\sigma^2}{d^2}\right)\sup|f|,

the moments of the part of the population whose light falls in the bin, up to the light within reach of the bin's
edges. A voxel's orientation distribution is thus represented by sub-populations, one per bin it lights, each with
its own moments. The bins give the profile along the arc; the moments give each bin's mean and spread below the
bin's width.

**Definition:** the combined statistic; its linearity is :eq:`rf-hist`'s and its edge term :eq:`rf-bin-edge`'s.

The populations of :doc:`populations` are such bins, grown to what the indexer resolves. Sub-populations of one voxel
whose bins are disjoint in a reflection are separated there by construction, so :eq:`rf-identify` applies to them:
they are told apart where the reflections that separate them span three dimensions. A rotation about
:math:`\vec{g}` maps a bin onto itself, so those reflections are never all of them. Along a ray, the same bin of
different voxels is merged and is a tomography. So the one linear system of :eq:`rf-linear` has a block of unknowns
per (voxel, occupied bin). How many are occupied, and so the system's size, is the indexer's to say.

Remainders
----------

Three effects were stated in size only: the per-cell :math:`LP` division (evaluated at the cell, not at the photon),
absorption's variation within a spot, and parallax. Each is now bounded.

**A weight known to a factor.** If a photon's weight is used as :math:`w e^{a}` instead of :math:`w`, with
:math:`|a| \le G r` (the log-weight's gradient :math:`G` times the distance :math:`r` between where the weight is
evaluated and where it should be), then each photon's weight is off by :math:`e^a - 1`, and

.. math::
    :label: rf-weight-bound

    |e^a - 1| \le e^{|a|} - 1 \ \Rightarrow\
    \Big|\sum_p \lambda_p (e^{a_p} - 1)(x_p - m)\Big| \le (e^{Gr} - 1)\sum_p \lambda_p |x_p - m|:

a first moment about the population's mean is off by at most :math:`e^{Gr} - 1` of the light's mean absolute
deviation, for any light. For :math:`LP`, :math:`r` is the cell's half-diagonal and :math:`G` the gradient of
:math:`\log(LP)` in pixels: with :math:`\log L = -\log\sin 2\theta - \log|\sin\eta|`, it is :math:`|\cot\eta|`
per radian of :math:`\eta` and :math:`2|\cot 2\theta|` per radian of :math:`2\theta`, times the radians a pixel
spans. For absorption, :math:`r` is the spot's angular extent and :math:`G` the derivative of :math:`\mu\ell` with
the diffracted direction. That one is known per voxel, and can enter linearly through :math:`\matr{Y}^{(2)}` instead.

**Proof:** ``tests/theory/test_refiner.py::test_weight_lemma`` (:math:`e^{-a} - 1 - (1 - e^{a}) = 2(\cosh a - 1)`).

**Parallax.** A photon from :math:`\vec{x}_v` reaching the detector at :math:`\vec{X}` is seen from the origin
turned by

.. math::
    :label: rf-parallax

    \sin\alpha = \frac{|\vec{x}_v \times \vec{X}|}{|\vec{X} - \vec{x}_v|\,|\vec{X}|}
    \le \frac{|\vec{x}_v|}{|\vec{X}| - |\vec{x}_v|},

since :math:`(\vec{X} - \vec{x}_v)\times\vec{X} = -\vec{x}_v\times\vec{X}`: :math:`5.6 \times 10^{-4}` for 100 µm at
180 mm.

**Proof:** ``tests/theory/test_refiner.py::test_parallax_angle``

The map itself is exact and simple. A photon leaving :math:`\vec{x}_v` along the unit :math:`\vec{u}` meets the
detector plane :math:`\hat{\vec{n}}\cdot\vec{X} = D` at :math:`\vec{X} = (D/\hat{\vec{n}}\cdot\vec{u})\,(\vec{u} +
\hat{\vec{n}}\times(\vec{x}_v\times\vec{u})/D)`. So the direction the origin sees is a linear map, normalised:

.. math::
    :label: rf-parallax-exact

    \Pi_v(\vec{u}) = \frac{(\matr{I} + \epsilon\matr{A})\vec{u}}{|(\matr{I} + \epsilon\matr{A})\vec{u}|}
    = \vec{u} + \epsilon\,\vec{T}(\vec{u}) + O(\epsilon^2), \quad
    \matr{A} = [\hat{\vec{n}}]_\times[\hat{\vec{e}}]_\times,\ \epsilon = \frac{|\vec{x}_v|}{D},\
    \vec{T}(\vec{u}) = (\matr{I} - \vec{u}\vec{u}^T)\matr{A}\vec{u}, \qquad
    \vec{T}(\vec{u}_0 + \vec{\delta}) = \vec{T}(\vec{u}_0) + D\vec{T}\,\vec{\delta}
    + \tfrac{1}{2}D^2\vec{T}[\vec{\delta}, \vec{\delta}] - \vec{\delta}\,(\vec{\delta}^T\matr{A}\vec{\delta}),

with :math:`\hat{\vec{e}} = \vec{x}_v/|\vec{x}_v|` and :math:`\|\matr{A}\| \le 1`. :math:`\vec{T}` is a cubic, so its
expansion ends at the written third-order term. A first-moment feature :math:`\vec{\ell}\cdot\hat{\vec{g}}`, after its
expansion to second order (with :math:`\matr{Y}^{(1)}` and :math:`\matr{Y}^{(2)}`), keeps exactly the remainder
:math:`-\epsilon|\vec{k}|\,(\vec{\ell}\cdot\vec{\delta})(\vec{\delta}^T\matr{A}\vec{\delta})`. That is at most
:math:`\epsilon|\vec{k}||\vec{\ell}||\vec{\delta}|^3`, and its mean is a third moment of the spread, zero for a
centrosymmetric one. :math:`|\hat{\vec{g}}|^2` gives the same with :math:`\vec{\ell} = -2|\vec{k}|\hat{\vec{k}}_{\text{in}}`,
because :math:`\vec{u}\cdot\vec{T}(\vec{u}) = (\vec{u}^T\matr{A}\vec{u})(1 - |\vec{u}|^2) = 0` on the sphere.
:math:`\epsilon|\vec{\delta}|^3` is :math:`3.0 \times 10^{-9}` for 100 µm at 180 mm and a 1° spread.

**Proof:** ``tests/theory/test_refiner.py::test_parallax_exact`` (the plane's intersection; the first order; the
cubic's third-order term; :math:`\vec{u}\cdot\vec{T}`; :math:`\|[\hat{\vec{n}}]_\times\| = 1`; the number).

Background
----------

Light that belongs to no population (air scatter, fluorescence, a diffuse floor) adds to every cell. Over a region,
a background :math:`b(\vec{x}) = b_0 + \vec{b}_1\cdot(\vec{x} - \vec{x}_R)` adds to the counts, first and second
moments :math:`b_0(S_0, \vec{S}_1, \matr{S}_2) + \vec{b}_1\cdot(\vec{S}_1, \matr{S}_2, S_3)`, with :math:`S_n` the
region's own moments (cells weighted by :math:`1/LP`): a linear term in :eq:`rf-linear`. Which background model is
allowed is decided by what it can mimic. In a box symmetric about its centre the odd :math:`S_n` vanish, and then

.. math::
    :label: rf-background

    \big(N,\ N\vec{m},\ N(\matr{\Sigma} + \vec{m}\vec{m}^T)\big)
    = \big(N,\ \vec{0},\ N(\matr{\Sigma} + \vec{m}\vec{m}^T)\big) + \vec{b}_1\cdot(\vec{S}_1, \matr{S}_2, S_3)
    \quad\text{at } \vec{b}_1 = N\matr{S}_2^{-1}\vec{m}, \qquad
    b_0\vec{S}_1 = \vec{0}.

A population shifted by :math:`\vec{m}` is, in all three statistics, exactly an unshifted one with a wider spread plus
a background slope. So a background slope free in each region removes that region's information on position: the
deformation. A constant background per region leaves the centred first moment alone and adds :math:`b_0\matr{S}_2` to
the second moments, where it is confounded with the spread, which is reported as an upper bound already
(:eq:`rf-shift`).

**Proof:** ``tests/theory/test_refiner.py::test_background_confound`` (a symmetric 2D box, symbolic weights).

So background is never a free slope per region. It is estimated from the cells outside every box: those boxes are
fixed by the indexer (:eq:`rf-box-margin`), so the complement is not chosen by the data and is unbiased
(:eq:`mom-select`). Its model is smooth across the detector over scales much larger than a box. At worst it varies
with omega, for a stress rig whose small-angle scattering reaches the detector at some angles. Its prediction in each
region is subtracted. A residual slope :math:`\delta\vec{b}_1` across a box then moves a centred
first moment by :math:`\matr{S}_2\,\delta\vec{b}_1/N` (:eq:`rf-background`), which enters :math:`\bar\Delta`. A
residual constant moves the counts and second moments only. Background below the cut is in the cut's bound already,
which holds per cell for any light (:eq:`rf-cut-bound`).

The cost of the solve
---------------------

One iteration applies :math:`\matr{G}` and :math:`\matr{G}^T` once each. By the factorisation into scalar
projections [Henningsson2023]_ that is :math:`2F` projections for :math:`F` moment components (16 with the metric, 61
with the second moments), each touching every voxel once per reflection it is in. For example, 3000 × 3000 voxels
seen in 300 reflections each with :math:`F = 16` is :math:`8.6 \times 10^{10}` voxel-ray weights per iteration, and
:math:`1.7 \times 10^{12}` for 20.

The number of iterations follows from the condition number. For a symmetric positive definite operator with
eigenvalues in :math:`[\ell, L]`, :math:`\kappa = L/\ell`, the step :math:`2/(\ell + L)` contracts the error by
:math:`(\kappa - 1)/(\kappa + 1)` per iteration. That reaches a tolerance :math:`\tau` within :math:`(\kappa/2)\ln(1/\tau)`
iterations, since :math:`\ln\frac{\kappa - 1}{\kappa + 1} = -2\operatorname{atanh}(1/\kappa) \le -2/\kappa`. Conjugate
gradients minimise the same error over a Krylov space that holds that iterate, so they need no more. The ray sums'
normal operator is a convolution:

.. math::
    :label: rf-cost

    \matr{R}^T\matr{R} f(\vec{x}) = \int_0^\pi\!\!\int f(\vec{x} + t\hat{\vec{\theta}})\, dt\, d\theta
    = \Big(f * \frac{1}{|\vec{z}|}\Big)(\vec{x}), \qquad
    \widehat{\tfrac{1}{|\vec{z}|}} = \frac{2\pi}{|\vec{k}|}, \qquad
    \kappa = \frac{1}{T(k_{\max})^2}\ \text{after a ramp } |\vec{k}|.

With the beam's profile on both sides it falls as :math:`T(\vec{k})^2/|\vec{k}|`. On an :math:`n`-voxel-wide grid
:math:`|\vec{k}|` spans a factor :math:`n/2` (from the grid's size to its Nyquist), so :math:`\kappa \gtrsim n/2`. A ramp preconditioner :math:`|\vec{k}|` along dty, per reflection (as in filtered
back-projection), leaves :math:`T(\vec{k})^2`. Then :math:`\kappa = 1/T(k_{\max})^2`: 5.9 when the beam's FWHM is the
dty step (the Nyquist period is 2 FWHM, :math:`T = 0.41`), and 20 iterations reach :math:`\tau = 10^{-3}`. With the
FWHM two steps wide, :math:`T = 0.028` and :math:`\kappa = 1.2 \times 10^3`: the bound is then 4100 iterations, and
conjugate gradients need no more. The weak combinations, estimated in place, would multiply :math:`\kappa` by
:math:`1/\sin^2\theta` (30-530). Solving in the split parameterisation (:eq:`rf-split-grid`) with the weak block
scaled by its gain (:eq:`rf-smin`) is a change of units, not a filter; the condition number of the scaled system is
not derived here.

**Filter:** an iteration stopped before it converges keeps the well-conditioned directions and leaves the rest near
their starting values: a spectral filter, like :eq:`rf-project`'s. The solve runs to the tolerance :math:`\tau` on the
normal equations' residual, never to a fixed count chosen to stop early.

**Proof:** ``tests/theory/test_refiner.py::test_richardson_bound`` (the contraction; the logarithm's bound; the
Krylov space), ``tests/theory/test_refiner.py::test_ray_normal_symbol`` (the transform; the numbers),
``tests/theory/test_refiner.py::test_cost_numbers``.

The cost per iteration is set by :math:`F` and the reflections per voxel. Sub-populations per bin
(:eq:`rf-bin-moments`) multiply :math:`F` by the bins each voxel lights. The iterations are set by the beam's
sampling, not by the grid's size, once preconditioned.

Several slices
--------------

Everything above treats one slice. A stack of slices has frequencies along :math:`z` too. The rays are horizontal,
so a frequency :math:`\vec{k} = \vec{k}_h + k_z\hat{\vec{z}}` is carried by :math:`\hat{\vec{\xi}} =
\hat{\vec{z}}\times\hat{\vec{k}}_h` whatever :math:`k_z`, and the weak part is still the row
:math:`\hat{\vec{\xi}}^T\hat{\matr{\beta}}` (:eq:`lim-slice`). With :math:`\hat{\vec{n}} =
\hat{\vec{k}}\times\hat{\vec{\xi}}`, the Nye tensor of a plane wave is :math:`\hat{\matr{\alpha}}\vec{m} =
i\hat{\matr{\beta}}(\vec{m}\times\vec{k})` for any :math:`\vec{m}`, and

.. math::
    :label: rf-3d

    \hat{\vec{n}}\times\vec{k} = |\vec{k}|\,\hat{\vec{\xi}}, \qquad
    \hat{\vec{\xi}}\times\vec{k} = -|\vec{k}|\,\hat{\vec{n}}, \qquad
    \hat\beta_{\xi\xi} = \frac{\hat\alpha_{\xi n}}{i|\vec{k}|}, \qquad
    \hat\beta_{\xi n} = -\frac{\hat\alpha_{\xi\xi}}{i|\vec{k}|}, \qquad
    \frac{\partial\hat{\matr{\alpha}}}{\partial\hat\beta_{\xi k}} = 0:

:eq:`lim-weak-parts` with :math:`\hat{\vec{z}}` replaced by :math:`\hat{\vec{n}}`, normal to :math:`\vec{k}`, which
leans from :math:`\hat{\vec{z}}` towards :math:`\vec{k}_h` as :math:`\vec{k}` tilts. Two weak numbers are dislocation
content with Burgers vector along the ray; the third, :math:`\beta_{\xi k}`, is the gradient along :math:`\vec{k}` of the
displacement along the ray, which no dislocation density fixes. A frequency along :math:`z` itself (:math:`\vec{k}_h =
0`) is carried by every horizontal ray, whose :math:`\vec{g}_0` span all directions: it is seen whole. So a stack adds
no weak directions beyond the slice's.

**Proof:** ``tests/theory/test_refiner.py::test_three_dimensional_weak``

The guarantee
-------------

Under (A1)-(A5), with regions fixed independently of the noise of the data they are applied to. Boxes taken from an
indexer fitted to the same data do not meet that condition: the selection bias of :eq:`mom-select` then enters, and
is not bounded here. The boxes must come from data independent of those they are applied to (for example populations
indexed on alternate frames, applied to the others), a choice for the maintainer. The indexer's error :math:`e` in
:eq:`rf-box-margin` is a statistical bound, not a hard one: the completeness of the boxes holds with its confidence.

1. **Noise** biases nothing (:eq:`rf-linear`). Error bars come from blocks that are independent exactly
   (:eq:`rf-harris`), with no noise model.
2. **Spot shape** enters nowhere in :math:`\matr{G}`. It enters only through the bounded terms 3-5 and the
   estimand's definition (light within each box).
3. **The cut** biases each region's statistics by at most :math:`\kappa_c \sum_R |f_p - m|/(LP)_p`
   (:eq:`rf-cut-bound`), a number of the geometry. No estimator can do better than bound it (:eq:`rf-no-unbiased`).
4. **The group rule** biases by at most :eq:`rf-group` where every lit cell's neighbours hold enough light, and by an
   unbounded amount otherwise: the faint compact spots of :doc:`recording`.
5. **The pixels** bias a feature by at most its gradient times :eq:`rf-alias-bound`. Omega binning biases nothing in
   the strain (:eq:`rf-invariant`) and, in the rotation about the axis, at most :eq:`rf-omega-bound` under (A6).

The refiner reports, per population, :math:`\hat{\vec{Y}}`, its block-sandwich covariance, and the bias bound
:math:`\sum_j |A_{ij}|\bar\Delta_j`. That bound is computed from the geometry, the regions and the point spread's
:math:`\chi(2\pi k)`, not from any spot shape. Where the bound exceeds the error bar, the estimate is limited by what
the data cannot say, and the report says so.

What it costs, against :doc:`limits`:

- Information within a cell (the pixels' sub-cell information for spots narrower than about 0.3 px,
  :eq:`det-crlb`) is unused, since only a shape could use it.
- Merged populations are determined only through the tomography of their ray sums. That is the Fourier slice's limit,
  with its weak combinations (:eq:`lim-seen-weak`); the bias terms reach the map along those combinations amplified by
  up to :math:`1/\sin\theta` for the first moments and :math:`1/\sin^2\theta` for the metric's :math:`C_{\xi\xi}`
  (:eq:`rf-amplify`).

What is left out
----------------

- **The instrument's omega shift** correlated with :math:`\vec{D}` (:eq:`rf-instrument`), at second order.
- **Parallax at second order in** :math:`\epsilon`: :eq:`rf-parallax-exact` is exact, but its expansion is written to
  first order only.
- **Anisotropic boxes**: the indexer gives only an isotropic spread per population (:doc:`populations`), so
  :eq:`rf-arc-box` needs an anisotropic :math:`\matr{S}` the indexer does not yet report.
- **Absorption's variation within a spot** is bounded by :eq:`rf-weight-bound`; its exact coefficient through
  :math:`\matr{Y}^{(2)}` is not written out.
- **The omega kernel's Gaussian form** (:eq:`rf-omega-kernel`) is the instrument's, as the point spread's mixture
  is; a non-Gaussian bandwidth or convergence profile would change :eq:`rf-omega-bound`'s numbers, not its form.
- **The Si fit's three-width covariance** for a given set of reflections is :eq:`rf-si-fisher` inverted; its
  conditioning depends on the rings and :math:`\eta` spread (:eq:`rf-omega-identify`), not written out for a typical
  scan.
- **The group rule's bound** (:eq:`rf-group-psf`) is rigorous but weak; a sharper one needs where in a cell the light
  falls.
- **The precision** :math:`\epsilon` and confidence :math:`z` of :eq:`rf-smin` are the maintainer's to set.
- **The system's size** with sub-populations per bin (:eq:`rf-bin-moments`) is set by how many bins each voxel
  lights (:eq:`rf-cost` gives the cost per bin); the count needs a phantom.
- **The background's smooth model** (its form across the detector, and in omega where it must) is not chosen here;
  :eq:`rf-background` only rules out a free slope per region.
- **The beam profile and runout** enter :math:`\matr{G}`; :eq:`rf-centre` bounds how they reach the deformation,
  but their own sizes are estimates (runout about 1 µm).
- **Multi-phase overlaps**: populations of different phases sharing a region add their ray sums like any others;
  nothing more is needed, but nothing is checked.
- **Code**: none of the refiner is in anri yet.
