.. _theory_limits:

What the data determine
=======================

Before choosing a method, this page asks what scanning 3DXRD with one rotation axis can determine at all, per
voxel: orientation and strain, at which spatial scale, at which precision, and which part survives an error in the
peak shapes. Every method must be judged against these limits: none can beat them, and a method that falls short of
them throws information away.

The results, in brief:

- **Separated light is well determined.** Where a population's light is its own in a reflection, that reflection
  sees it alone: all nine numbers of its deformation at counting precision, and its position to
  :math:`\sigma_{\text{beam}}/\sqrt{N}`, finer than the beam.
- **Merged light is a tomography with three weak combinations per spatial frequency**: the dislocation density with
  lines along the rotation axis. Six combinations are resolved to about 1.5-2 beam widths; region means of all nine
  are robust.
- **Errors common to every voxel bias only region means**, never the variation across a map.
- **Second moments carry no information on the means through their expectation** once each voxel's own spread is
  free; they can only lower the means' noise through their correlation with the first moments (:doc:`moments`).

Words: a *population* is one entry of a voxel's orientation distribution (:doc:`populations`); its *spot* in a
reflection is the light it alone would give; its *own width* :math:`W` is that spot's width in rotation (instrument
plus its own spread). Two populations on one ray are *separated* in a reflection when their spots do not overlap,
*merged* when they do.

The linear model
----------------

Write a population's deformation as :math:`\matr{\beta}` (3 × 3: rotation and strain together), so that a
reciprocal-lattice vector moves from :math:`\vec{g}_0` to :math:`(\matr{I} + \matr{\beta})\,\vec{g}_0`. A
reflection is seen at one omega (:doc:`diffraction`), so each reflection is one projection direction :math:`\vec{\xi}`
(the incident beam in the sample frame). The mean scattering vector of a blob is a ray sum over the voxels the beam
lights (:doc:`moments`):

.. math::
    :label: lim-model

    N_b\, \vec{m}_b = \sum_v a_{bv}\, N_v \left(\vec{g}_0 + \matr{\beta}_v \vec{g}_0\right),
    \qquad
    N_b = \sum_v a_{bv} N_v,

with :math:`a_{bv}` the beam's weight of voxel :math:`v` in the blob's row and :math:`N_v` its counts.

**Definition:** the blob's first moment is exactly this ray sum for any spot shape (:doc:`moments`); this page
linearises it in :math:`\matr{\beta}`.

The ray is almost perpendicular to every :math:`\vec{g}` it sees:

.. math::
    :label: lim-ray

    \frac{\vec{g}\cdot\vec{\xi}}{|\vec{g}|} = -\sin\theta .

**Proof:** ``tests/theory/test_limits.py::test_ray_component``

The Fourier slice
-----------------

Take the voxels' :math:`\matr{\beta}(\vec{x})` in one slice and its Fourier coefficient
:math:`\hat{\matr{\beta}}(\vec{k})` at an in-plane frequency :math:`\vec{k}`. A ray sum along :math:`\vec{\xi}`
carries only frequencies perpendicular to :math:`\vec{\xi}`, so :math:`\vec{k}` is carried by rays along
:math:`\vec{\xi} = \hat{\vec{z}} \times \hat{\vec{k}}`, whose :math:`\vec{g}` lie in the plane of :math:`\vec{k}` and
:math:`\vec{z}` up to :math:`\sin\theta` (:eq:`lim-ray`). For :math:`\vec{g} = a\hat{\vec{k}} + c\hat{\vec{z}}`:

.. math::
    :label: lim-slice

    \hat{\matr{\beta}}\,\vec{g} = a\,\hat{\matr{\beta}}\hat{\vec{k}} + c\,\hat{\matr{\beta}}\hat{\vec{z}} .

**Proof:** ``tests/theory/test_limits.py::test_fourier_slice``

So at each :math:`\vec{k} \neq 0` the data see :math:`\hat{\matr{\beta}}` applied to :math:`\hat{\vec{k}}` and
:math:`\hat{\vec{z}}` (six numbers) directly, and :math:`\hat{\matr{\beta}}\hat{\vec{\xi}}` (three numbers) only with
weight :math:`\sin\theta`. At :math:`\vec{k} = 0` (means over regions) all nine are seen. Split into strain
:math:`\varepsilon` and rotation :math:`w`, with :math:`\vec{k}` along :math:`x` and :math:`\vec{\xi}` along :math:`y`:

.. math::
    :label: lim-seen-weak

    \text{seen:}\quad \varepsilon_{xx},\ \varepsilon_{zz},\ \varepsilon_{xz},\ w_y,\
    \varepsilon_{xy} + w_z,\ \varepsilon_{yz} - w_x;
    \qquad
    \text{weak:}\quad \varepsilon_{yy},\ \varepsilon_{xy} - w_z,\ \varepsilon_{yz} + w_x .

**Proof:** ``tests/theory/test_limits.py::test_seen_and_weak``

Orientation and strain are entangled in two pairs only: the rotation about :math:`z` with the in-plane shear, and the
rotation about :math:`\vec{k}` with :math:`\varepsilon_{\xi z}`. An orientation map at full resolution therefore carries
an error the size of the unresolved shear (1e-4 to 1e-3 rad) unless something below resolves it.

The weak part is the dislocation density
----------------------------------------

The three weak numbers are the Nye tensor's components with line direction :math:`z`, the rotation axis:

.. math::
    :label: lim-nye

    i\,\hat{\matr{\beta}}(\vec{k})\,\vec{\xi} = \widehat{(\operatorname{curl}\matr{\beta})}_{\cdot z},
    \qquad
    \vec{\xi} = \hat{\vec{z}} \times \vec{k},
    \qquad
    (\operatorname{curl}\matr{\beta})_{iz} = \partial_x \beta_{iy} - \partial_y \beta_{ix} .

**Proof:** ``tests/theory/test_limits.py::test_nye``

A compatible field, the gradient of a displacement (:math:`\matr{\beta} = \nabla\vec{u}`), has no weak part at all:

.. math::
    :label: lim-compatible

    \hat{\matr{\beta}} = i\,\vec{u}\,\vec{k}^T \;\Rightarrow\; \hat{\matr{\beta}}\,\vec{\xi} = 0 .

**Proof:** ``tests/theory/test_limits.py::test_compatible_no_weak``

So a single slice determines everything about a deformation except where geometrically necessary dislocations thread
the slice, and those only through :math:`\sin\theta`.

**Anri's choice** (maintainer, 2026-11-23): inside a merged domain (populations whose orientations agree within their
own widths), :math:`(\operatorname{curl}\matr{\beta})_{\cdot z} = 0` may be imposed. It is a physical constraint, not
a smoothness prior: dislocation walls with a misorientation larger than the spot width separate their light and are
resolved by separation, so only walls below the spot width are assumed away, and that is the size of the bias if the
assumption is wrong. It fills all three weak combinations at every :math:`\vec{k}` from one slice; equilibrium fills
only two and needs a stack of slices and the elastic constants.

The beam
--------

The dty profile of the beam multiplies the projections' Fourier coefficients by its transform. For a Gaussian beam
of FWHM 1 at a period :math:`\Lambda` (in beam FWHMs):

.. math::
    :label: lim-transfer

    T(\Lambda) = \exp\!\left(-\frac{\pi^2}{4 \ln 2\, \Lambda^2}\right).

**Proof:** ``tests/theory/test_limits.py::test_beam_transfer``

Noise is amplified by :math:`1/T`: 35, 4.9 and 2.4 at :math:`\Lambda` = 1, 1.5 and 2 FWHM. In a merged domain the
seen combinations resolve to about 1.5-2 beam FWHM, set by the counts, not by the dty step.

Information in merged light
---------------------------

Relative to a lone voxel (information :math:`M N_v / W^2` from :math:`M` reflections), a pattern of period :math:`L`
steps in a domain whose voxels share their spots over a length of :math:`D` voxels along the rays is seen with

.. math::
    :label: lim-mu

    \mu = \frac{L}{\pi D}\, T^2 .

**Proof:** ``tests/theory/test_limits.py::test_merged_information``

Assumptions: constant merged length :math:`D`; counting noise only, equal counts per spot; linear; aliasing (the
Poisson-summation terms of the dty sampling) dropped, which needs :math:`L \gtrsim 4`; and enough reflections that
every direction of :math:`\vec{k}` has a coherent view, :math:`M \gg 2\pi D / L`; with fewer, some directions get none
(a true null). The weak combinations get at most :math:`\mu \sin^2\theta`.

Separation
----------

A small rotation :math:`\vec{\delta}` moves :math:`\vec{g}` by :math:`\vec{\delta} \times \vec{g}`:

.. math::
    :label: lim-separation

    |\vec{\delta} \times \vec{g}|^2 = |\vec{\delta}|^2 |\vec{g}|^2 - (\vec{\delta}\cdot\vec{g})^2 ,

**Proof:** ``tests/theory/test_limits.py::test_rotation_about_g_invisible``

so two populations misoriented by :math:`\vec{\delta}` are separated in every reflection whose :math:`\vec{g}` is not
close to parallel with :math:`\vec{\delta}`, once :math:`|\vec{\delta}|` exceeds a few own widths. A reflection that
separates a population sees it alone: no ray sum, no Fourier slice, all nine numbers at counting precision from its
reflections. Strain rarely separates light on its own: a strain :math:`\varepsilon` moves a spot by
:math:`\Delta 2\theta = 2\varepsilon\tan\theta` (:doc:`diffraction`), and one 75 µm pixel at 180 mm (4e-4 rad) is
2.5e-3 in strain at :math:`\tan\theta = 0.08`, more than most strain contrasts. Strain is resolved where orientation
separates.

In a cell structure whose misorientation accumulates as a random walk, :math:`\theta_c \sqrt{r/c}` at distance
:math:`r` (cells of size :math:`c` and misorientation :math:`\theta_c`, voxels of size :math:`d`), populations separate
once that exceeds :math:`n` own widths :math:`W` (:math:`n \approx` 2-3), so the merged length is

.. math::
    :label: lim-merged

    D = \frac{n^2 W^2 c}{d\,\theta_c^2} ,

and at least :math:`c/d` (voxels inside one cell merge).

**Proof:** ``tests/theory/test_limits.py::test_merged_length``

When a voxel's own spread dominates (:math:`W = \theta_c\sqrt{d/c}`, many cells per voxel), :math:`D = n^2`: 4-9
voxels whatever the misorientation. When the instrument dominates, :math:`D = (n W/\theta_c)^2\, c/d`. Deformed
metals are mostly separated; annealed grains are merged over the grain.

Localisation
------------

A separated population's counts versus dty follow the beam profile centred on its position across the ray. With
:math:`N` photons, a Gaussian beam of width :math:`\sigma_{\text{beam}}`, sampled continuously:

.. math::
    :label: lim-localise

    \operatorname{sd}(x_0) = \frac{\sigma_{\text{beam}}}{\sqrt{N}} .

**Proof:** ``tests/theory/test_limits.py::test_localisation``

Sampled in rows (the dty steps), the Cramér-Rao bound depends on where the population sits between rows; it is
stationary, by symmetry, midway between rows and on a row. At a step of FWHM/2 both are within 2e-5 of
:math:`\sigma_{\text{beam}}/\sqrt{N}`; at one FWHM, 0.84 (midway) and 1.27 (on a row) times it; at 1.5 FWHM, 0.57 and
4.5 times. Reflections at different omegas give the two in-plane coordinates. This uses only the
counts of the population's own light and the beam profile, a shared instrument function, so it does not depend on
spot shapes. The TDXRD stage's radial runout (~1 µm, periodic in omega) is a position error of that size unless it is
modelled.

**Anri's choice** (maintainer, 2026-11-23): volume fractions are *not* constrained to sum to a density (multiphase
samples and voids must be handled). So the centroid of a population localises beyond the beam, but its shape is
resolved only to the beam (:eq:`lim-transfer`).

Second moments
--------------

The data are linear in each voxel's density over :math:`\matr{\beta}`. A blob's counts, first and raw second
moments are ray sums of :math:`N_v`, :math:`N_v \matr{\beta}_v` and
:math:`N_v (\matr{\Sigma}_v + \matr{\beta}_v\matr{\beta}_v^T)`, where :math:`\matr{\Sigma}_v` is the voxel's own
spread (projected per reflection). With :math:`\matr{\Sigma}_v` free, the expectations of the second moments determine only the sum, and carry no
information on :math:`\matr{\beta}_v`. Their noise is correlated with the first moments' (the same counts), so where a
voxel's spread is seen by more second-moment sums than it has free numbers, they lower the noise of the means as a
control variate (:eq:`mom-info`); that gain is proved to exist but not yet sized. More precisely, the central second moment of a
blob changes with a voxel's :math:`x_v` (any projection of :math:`\matr{\beta}_v \vec{g}_0`) as

.. math::
    :label: lim-mom2-jac

    \frac{\partial C_b}{\partial x_v} = \frac{2 w_v}{\sum w}\,(x_v - m_b),

zero at a flat state, and per blob a new row is worth :math:`(|\vec{a}|^2/s^2)(1 + \cos^2)` of a first-moment row
(Gaussian moment noise, :math:`\vec{a}` the voxel's contrast against the blob mean, :math:`s` the blob width).

**Proof:** ``tests/theory/test_limits.py::test_second_moment_contrast_only``

How many of a voxel's nine numbers the second moments see beyond a free spread, for a rotation or a strain contrast:

.. math::
    :label: lim-mom2-rank

    \begin{array}{lcc}
    \text{spread} & \text{rotation contrast} & \text{strain contrast} \\
    \text{known} & 9 & 9 \\
    \text{free rotation spread} & 6 & 9 \\
    \text{+ free strain spread, no cross-covariance} & 6 & 3 \\
    \text{free } 9 \times 9 & 0 & 0
    \end{array}

**Proof:** ``tests/theory/test_limits.py::test_second_moment_absorption``

Every non-zero entry with a free spread rests on there being no covariance between rotation and strain inside a
voxel (elastic bending makes them fully correlated; dislocation cells make both from the same dislocations). If that
is wrong, the strain read through a rotation contrast :math:`c` is biased by about
:math:`\rho\,\sigma_\varepsilon \sigma_w / c`, not reduced by counts. So second moments are not used for the means;
they are used for spreads, and for the second-order term of large spreads (a curved arc's mean is not the rotated
mean).

Model error
-----------

An error in the model biases the solution along each singular direction by the projection of the whitened error on
that direction's data pattern: an error in the range of the model is absorbed as a wrong parameter, not amplified;
along a direction of information :math:`\mu` the bias is at worst :math:`1/\mu` times the error. Which errors fall
where:

- **Common to every voxel** (instrument shape, the energy, the geometry): a shift :math:`s` shared by every voxel's
  spot in a reflection shifts every blob mean by :math:`s`, whatever the ray:

  .. math::
      :label: lim-common

      \frac{\sum_v w_v (x_v + s)}{\sum_v w_v} = m_b + s .

  **Proof:** ``tests/theory/test_limits.py::test_common_offset_is_k0``

  The data change is constant over rays, so it lives at :math:`\vec{k} = 0`: it biases region means (like a wrong
  d0) and never the variation across a map.
- **Per voxel** (a spot's shape where it is cut by the threshold, a region boundary, a window): spatially varying,
  so amplified along the weak directions, by up to :math:`1/\sin^2\theta` (~60-1000) for the dislocation combinations.

What a method must keep
-----------------------

The limits above follow from the exact model of the recorded cells (:doc:`recording`), whatever statistic a method
computes from them. Any method is judged by two derived numbers: the information it keeps, against these limits, and
the bias a wrong spot shape gives it, projected as in :eq:`lim-common`. From the results on this page:

- The information that makes the problem well-conditioned is separation: which part of a banana came from which
  population. A statistic that adds up all the light of a blob (its total counts and moments, :doc:`moments`) is
  exact for any spot shape but keeps none of it: every voxel lighting the blob is merged.
- The full likelihood of the recorded cells keeps all of it, but needs every spot's shape, and a wrong shape biases
  every voxel along a ray through the light the voxels share.
- Second moments with free spreads add to the means only through their noise correlation (above), not yet sized.

Between these lie statistics over parts of a blob. :doc:`robustness` derives their shape bias: a separated
population's location is protected by symmetry (:eq:`ref-symmetric`) up to the pixels' aliasing (:eq:`ref-alias`),
while splitting merged light by its shape is bias-dominated above a few hundred to a few thousand photons for
plausible shape errors (:eq:`ref-split`). So the statistics a robust refiner keeps are moments over regions complete
for their contributors: separated light alone in its region, merged light as a ray sum.

What is left out
----------------

- **Unresolved structure.** A general orientation distribution per voxel is underdetermined: by counting, it needs
  about (voxels across the grain) × (spread / own width) reflections, thousands for a deformed grain against the
  100-500 measured. The population representation (few compact components per voxel) is therefore necessary.
- **The full Schur complement** of the weak combinations over a ring set is not written out. Their gain relative to
  the seen ones is the reflections' rms :math:`\sin\theta` (:eq:`rf-smin`), and a fitted nuisance can only lower it
  (:eq:`rf-amplify`).
- **Several slices** are treated in :eq:`rf-3d`: a stack adds no weak directions beyond the slice's; the weak part
  at a tilted frequency is the Nye component along :math:`\hat{\vec{k}}\times\hat{\vec{\xi}}`.
- **Large rotations.** Everything here is linearised; a population's own arc is curved at spreads of degrees. The
  refiner's moments are exact at any spread (:eq:`rf-moments`), and its box margins to all orders
  (:eq:`rf-box-margin-exact`).
