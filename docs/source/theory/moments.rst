.. _theory_moments:

Moments of recorded light
=========================

The exact model of the data is the likelihood of the recorded cells (:doc:`recording`). This chapter states
statistics of that recorded light, counts and moments over a set of cells, and proves their properties: what they
equal in expectation, what they can and cannot see, and what breaks them. It does not choose a method; the choice
comes from comparing these properties with what the data determine (:doc:`limits`).

Counts and moments over a region
--------------------------------

For a set of cells :math:`R` (a region) and a function :math:`f` of a cell's coordinates :math:`\vec{x}_p` (its
frame's omega and its row and column, or the scattering vector of its centre), the statistic is

.. math::
    :label: mom-sum

    T_R[f] = \sum_{p \in R} r_p\, f(\vec{x}_p),

with :math:`f = 1` the counts, :math:`f = \vec{x}` the first moments and :math:`f = \vec{x}\vec{x}^T` the raw second
moments. Means and covariances are ratios of these sums.

**Definition:** the region statistics.

If :math:`R` is fixed without looking at the data, the expectation passes through the sum, and the covariance
includes every pair of correlated cells (:eq:`rec-cov`):

.. math::
    :label: mom-expect

    \operatorname{E}\big[T_R[f]\big] = \sum_{p \in R} \operatorname{E}[r_p]\, f(\vec{x}_p), \qquad
    \operatorname{Cov}\big[T_R[f], T_R[g]\big] = \sum_{p, q \in R} \operatorname{Cov}(r_p, r_q)\, f(\vec{x}_p)\, g(\vec{x}_q).

**Proof:** ``tests/theory/test_moments.py::test_selection_bias`` (the fixed region), with
:math:`\operatorname{E}[r_p]` from :eq:`rec-expected`.

Sums over a region of only the diagonal :math:`\operatorname{Var}(r_p)` understate the covariance wherever a
region holds faint, correlated cells (:doc:`recording`, the group rule).

Exact ray sums, for any spot shape
----------------------------------

A cell's light is the sum over the contributors that light it (voxels, or populations within voxels :math:`v`),
each weighted by the beam on it :math:`b_v` and its density :math:`\rho_v`, times its own spot's light in that cell
:math:`s_{vp}`, of any shape: :math:`\lambda_p = \sum_v b_v \rho_v s_{vp}`. Then

.. math::
    :label: mom-raysum

    \sum_{p \in R} \lambda_p f(\vec{x}_p) = \sum_v b_v\, \rho_v\, m^R_v[f], \qquad
    m^R_v[f] = \sum_{p \in R} s_{vp} f(\vec{x}_p):

the moments of the light are beam- and density-weighted sums of each contributor's own (binned) spot moments. No
light is divided between contributors, and no spot shape is assumed. A contributor's shape enters only through its
own moments over :math:`R` (all its moments when :math:`R` holds all its light).

**Proof:** ``tests/theory/test_moments.py::test_ray_sum``

The recorded light falls short of the light by the cut's and the group rule's deficit
(:eq:`rec-trunc`, :eq:`rec-expected`):

.. math::
    :label: mom-deficit

    \operatorname{E}\big[T_R[f]\big] = \sum_v b_v\, \rho_v\, m^R_v[f] - \sum_{p \in R} D_p f(\vec{x}_p), \qquad
    D_p = \lambda_p - \operatorname{E}[r_p] = \lambda_p P(n_p \le c - 1) + e_p (1 - w_p).

**Proof:** ``tests/theory/test_moments.py::test_cut_deficit``

So the ray-sum property is exact above the cut, where :math:`D_p` vanishes (for :math:`c = 1` its first part is
:math:`\lambda_p e^{-\lambda_p}`, and :math:`1 - w_p` falls as the neighbours brighten). Where :math:`D_p` matters,
at the faint edges of spots, the expectation is still exact (:eq:`rec-expected`), but it is a non-linear function of
the sum of the contributors' light in each cell: there the tails of the spot shapes enter.

What raw second moments see
---------------------------

The data are linear in each contributor's density over the deformation :math:`\matr{\beta}`
(:doc:`limits`). Over :math:`\matr{\beta}` (any distribution, mean :math:`\matr{\beta}_v`, covariance
:math:`\matr{\Sigma}_v`) its counts, first and raw second moments are :math:`N_v`, :math:`N_v\matr{\beta}_v` and

.. math::
    :label: mom-q

    N_v \matr{Q}_v, \qquad \matr{Q}_v = \matr{\Sigma}_v + \matr{\beta}_v \matr{\beta}_v^T.

:math:`(\matr{\beta}_v, \matr{\Sigma}_v) \mapsto (\matr{\beta}_v, \matr{Q}_v)` is one to one, so with
:math:`\matr{\Sigma}_v` free the expected second moments depend on :math:`\matr{Q}_v` alone: a change
:math:`\delta\matr{\beta}` with :math:`\delta\matr{\Sigma} = -(\delta\matr{\beta}\,\matr{\beta}^T +
\matr{\beta}\,\delta\matr{\beta}^T)` leaves them unchanged. Their means carry no information on
:math:`\matr{\beta}_v`.

**Proof:** ``tests/theory/test_moments.py::test_second_moments_see_q``

They can still add some through their noise, which is correlated with the first moments' (the same counts). With
first-moment rows :math:`\matr{J}_1` and noise covariance :math:`\matr{C}`, when the second-moment sums are as many
as the free :math:`\matr{Q}` numbers (their block of the Jacobian square and invertible), the information on
:math:`\matr{\beta}` with :math:`\matr{Q}` profiled out is the first moments' alone, whatever :math:`\matr{C}`:

.. math::
    :label: mom-info

    \matr{I}_{\beta \mid Q} = \matr{J}_1^T\, \matr{C}_{11}^{-1}\, \matr{J}_1 .

**Proof:** ``tests/theory/test_moments.py::test_second_moments_information``

When more sums see the same :math:`\matr{Q}_v` (several blobs per voxel) and their noise is correlated with the first
moments', the profiled information is larger than :eq:`mom-info` (the same test, exact example). That gain comes
from the noise alone (third moments of the light about the region's centre); it is not sized here.

Regions defined by the model and by the data
--------------------------------------------

A region defined from the data, such as the recorded pixels dilated, always holds every recorded pixel. So the data
summed over it have the mean of the data summed over every cell, while the model summed over the same (random)
region has a smaller mean:

.. math::
    :label: mom-select

    \operatorname{E}\Big[\sum_{p \in R} r_p - \sum_{p \in R} \operatorname{E}[r_p]\Big]
    = \sum_p \operatorname{E}[r_p]\,\big(1 - P(p \in R)\big) \ge 0, \qquad
    \operatorname{E}[r_p \mid p \in R] = \frac{\operatorname{E}[r_p]}{P(p \in R)}.

**Proof:** ``tests/theory/test_moments.py::test_selection_bias`` (exact enumeration on a strip, :math:`m = 1` and
:math:`2`; a region fixed in advance shows no bias).

The bias is the expected recorded light of the cells the region leaves out, and it exists for :math:`m = 1` too:
it does not need the group rule, only that recorded pixels are always inside their region. A region fixed by the
model (e.g. cells whose model light is above a level, fixed before the data are read) is unbiased for any level
(:eq:`mom-expect`); the level trades information, not accuracy. Recorded pixels outside every such region are
compared with the model's expected light there, separately.

**A region inside one frame.** Every cell of it has the same omega :math:`\omega_j` (:doc:`recording`), so its
omega moments are fixed by its counts: :math:`T[\omega] = \omega_j T[1]`, :math:`T[\omega^2] = \omega_j^2 T[1]`,
:math:`T[\omega s] = \omega_j T[s]`, :math:`T[\omega f] = \omega_j T[f]`, in data and model alike. Of the ten sums
up to second order,

.. math::
    :label: mom-rank

    \operatorname{rank} \operatorname{Cov}\big[T[1], T[\vec{x}], T[\vec{x}\vec{x}^T]\big] = 6.

**Proof:** ``tests/theory/test_moments.py::test_single_frame_rank``

Light beyond a window
---------------------

The forward model renders each peak over a fixed window of cells and reports the fraction it captured
(:doc:`peak_shape`, windows). The light a window leaves out does three things to the recorded light:

1. it is missing from :math:`\lambda_p` of the cells beyond the window's faces;
2. through the group rule it changes :math:`\operatorname{E}[r_p]` of every cell of the same frame within
   :math:`m - 1` pixels of those cells (:eq:`rec-reach`), and of no other cell:

   .. math::
       :label: mom-stencil

       \frac{\partial w_p}{\partial a_q} \neq 0 \iff q \text{ in the frame of } p,\ \max(|s_q - s_p|, |f_q - f_p|) \le m - 1;

   **Proof:** ``tests/theory/test_moments.py::test_exclusion_stencil``

3. it can be recorded outside any region where the model expects light.

**Anri's rule:** cells where a model may be incomplete never enter a comparison of data with model. The cells
beyond each window face where the left-out light may exceed a tolerance, dilated in-frame by :math:`m - 1` pixels
(:eq:`mom-stencil`), are left out for data and model alike. The set is fixed by the model, so leaving it out biases
nothing (:eq:`mom-expect`); a set too large costs information, never accuracy.

How far beyond a face the light can reach is bounded where the cells' light falls log-concavely. For a Gaussian in
cells of width :math:`h` (any mean, sd :math:`\sigma`), the cell masses satisfy

.. math::
    :label: mom-logconcave

    b_{k-1}\, b_{k+1} \le \rho\, b_k^2, \qquad \rho = \frac{1 + e^{-2h^2/\sigma^2}}{2} < 1,

since :math:`\varphi(x - h)\varphi(y + h) = \varphi(x)\varphi(y)\,e^{-(h(y - x) + h^2)/\sigma^2}` and, symmetrised
over :math:`x \leftrightarrow y` in one cell, the factor is :math:`\cosh(h(y - x)/\sigma^2)\,e^{-h^2/\sigma^2} \le
\rho`. So the ratio of successive cells falls outward, and with :math:`b` the light on a face and :math:`r` its ratio
to the cell inside it, the light :math:`n` cells beyond is at most :math:`b\, r^n`.

**Proof:** ``tests/theory/test_moments.py::test_binned_gaussian_log_concave``

The bound needs log-concavity. A mixture of Gaussians is not log-concave: the detector's point spread with a core
and a wing (``geom["psf_mix"]``), or the thick sensor's depth kernel drawn as several Gaussians. For a core of sd
0.25 px holding 72% and a wing of sd 1.3 px holding 28% (AGENTS.md, 43.6 keV), in 1 px cells about the centre,

.. math::
    :label: mom-mixture

    \frac{b_0\, b_2}{b_1^2} = 3.30 > 1, \qquad \frac{b_3}{b_1 r^2} = 7.8 \quad (r = b_1 / b_0):

a face one pixel out from the core would bound the light two cells further out 7.8 times too low.

**Proof:** ``tests/theory/test_moments.py::test_mixture_not_log_concave``

With such spots the exclusion must be bounded per mixture term, each a Gaussian.

Absolute intensities and their gauge
------------------------------------

Through the cut, :math:`\operatorname{E}[r_p]` depends on each cell's absolute light, so the light's scale is a
parameter. If the light of component :math:`k` (a population of a voxel) on ring :math:`h` is
:math:`w_k s_h G_{kh}`, with a weight per component, a scale per ring and :math:`G` known, the data depend on
:math:`(w, s)` only through the products. In logarithms the Jacobian's rows are
:math:`\vec{e}_k + \vec{e}_{K + h}` over the observed pairs :math:`(k, h)`:

.. math::
    :label: mom-gauge

    \operatorname{rank} = K + H - P, \qquad \text{null vectors } (\mathbf{1}_{k \in \mathcal{P}}, -\mathbf{1}_{h \in \mathcal{P}})
    \text{ for each connected part } \mathcal{P},

with :math:`P` the number of connected parts of the graph linking each component to the rings it reflects on. One
gauge, :math:`w \to a w,\ s \to s / a`, when every component shares rings with others; a component alone on its
rings has its own. The gauge holds exactly through the cut and the group rule, since they see only the products.

**Proof:** ``tests/theory/test_moments.py::test_gauge``

A scale per ring stands for :math:`|F|^2` and everything else common to a ring; it assumes every reflection of the
ring has the same :math:`|F|^2` and the same absorption. Absorption along each reflection's paths differs within a
ring (:doc:`intensity`), so it must be modelled, not absorbed by the ring scale.

What is left out
----------------

- **The noise contribution of second moments** (after :eq:`mom-info`) is shown to exist, not sized.
- **Faint edges.** Where :math:`D_p` of :eq:`mom-deficit` matters, the region statistics depend on spot tails; how
  much is not sized here.
- **Mixture spots** have no window bound until one is written per mixture term.
- **Within-ring differences** of :math:`|F|^2` (several families on one ring) break the per-ring scale.
- Code: none of this is in anri yet (sandbox: ``anri/sandbox/refine/cutexact.py``, ``render_fast.py``).
