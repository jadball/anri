.. _theory_recording:

What the detector records
=========================

This chapter is the exact model of the data: the probability of what is written to the sparse file, given the
expected light in every cell. The light itself comes from the forward model (:doc:`intensity`, :doc:`peak_shape`).
Everything a method compares with the data, whatever the method, is a function of this model, and must follow from
it.

Cells
-----

The data are counts in discrete cells :math:`p = (j, s, f)`: frame :math:`j`, detector row :math:`s` and column
:math:`f`, in one dty row of the scan. A cell's expected light :math:`\lambda_p` (photons) is the intensity
integrated over the pixel's area, over the frame's omega range :math:`[\omega_j - h/2, \omega_j + h/2)` (:math:`h`
the omega step), and over the dty positions the sample passes through during the frame:

.. math::
    :label: rec-cell

    \lambda_p = \frac{M_j}{M_{\text{ref}}} \int_{\omega_j - h/2}^{\omega_j + h/2} \int_{\text{pixel } (s, f)}
    I\big(\vec{x}, \omega, \text{dty}(\omega)\big)\, d\vec{x}\, d\omega,

with :math:`I` the intensity per unit area and omega at the monitor's reference flux, and :math:`M_j` the monitor
of frame :math:`j`.

**Definition:** the expected light of a cell.

**Anri's rule:** data and model live on the same cells. The model's light is integrated over each cell
(:eq:`rec-cell`), never evaluated at the cell's centre, and every function of the data (counts, moments, a
likelihood) is computed on cells, for data and model alike. Then pixel and frame binning enter both identically,
and nothing about the binning is left to correct.

**Anri's rule:** a cell's omega is its frame's centre :math:`\omega_j`, in data and model alike (and its dty is
its frame's dty reading). A moment over cells of any function :math:`g` of omega is then

.. math::
    :label: rec-staircase

    \sum_j g(\omega_j)\, \Lambda_j = \int g\big(\bar\omega(\omega)\big)\, \varphi(\omega)\, d\omega,

with :math:`\varphi` the light per unit omega, :math:`\Lambda_j` its integral over frame :math:`j`, and
:math:`\bar\omega(\omega)` the centre of the frame holding :math:`\omega`: a moment of the binned light, not of the
continuous one.

**Definition:** the frame-binned moment.

The two differ by the familiar Sheppard terms only on average. With the frame grid at a uniform random offset,
:math:`\bar\omega - \omega` is uniform on :math:`[-h/2, h/2)` and independent of :math:`\omega`, so with
:math:`\sigma^2`, :math:`\mu_3`, :math:`\mu_4` the central moments of the continuous light:

.. math::
    :label: rec-sheppard

    \operatorname{E}[\bar\omega] = \operatorname{E}[\omega], \qquad
    \operatorname{Var}[\bar\omega] = \sigma^2 + \frac{h^2}{12}, \qquad
    \mu_3[\bar\omega] = \mu_3, \qquad
    \mu_4[\bar\omega] = \mu_4 + \frac{\sigma^2 h^2}{2} + \frac{h^4}{80}.

**Proof:** ``tests/theory/test_recording.py::test_sheppard_offset_average``

For one spot at one offset these are wrong: a spot inside one frame has binned omega variance zero, not
:math:`h^2/12`, and its binned mean is the frame centre. Because the model bins exactly as the data
(:eq:`rec-staircase`), no correction is ever applied; the terms of :eq:`rec-sheppard` matter only to someone
comparing binned data with a continuous model, which Anri does not do. A region lying in one frame has omega moments
fixed by its counts (:doc:`moments`).

Photon counts
-------------

The Eiger counts photons. Given the map, each cell's count is Poisson and the cells are independent:

.. math::
    :label: rec-poisson

    n_p \sim \operatorname{Poisson}(\lambda_p), \quad \text{independent over cells (pixels and frames)}.

**Definition:** the counting model (an assumption, below).

A photon counted in another pixel than where it arrived (escaped fluorescence, the detector's point spread) keeps
this true: a Poisson source whose photons go to pixel 1 with probability :math:`q` and to pixel 2 otherwise gives
independent Poisson counts, since the joint generating function factorises,

.. math::
    :label: rec-displace

    \operatorname{E}\left[s^{n_1} t^{n_2}\right] = e^{\lambda(q s + (1 - q) t - 1)}
    = e^{\lambda q (s - 1)}\, e^{\lambda (1 - q)(t - 1)}.

**Proof:** ``tests/theory/test_recording.py::test_displaced_counts_poisson``

So the detector's point spread moves light between cells of :eq:`rec-cell` and does not correlate them, as long as
each photon is counted once. A photon counted twice (its fluorescence above the lower threshold in a neighbour *and*
its remaining energy above it in its own pixel) would correlate neighbours; whether the Eiger's thresholds allow that
is a question for the maintainer.

Monitor normalisation
---------------------

The sparse file holds raw counts. When intensities are normalised to the monitor (the indexer's ``--monitor``, as
ImageD11's ``DataSet.set_monitor``), each recorded value :math:`r_p` becomes

.. math::
    :label: rec-monitor

    d_p = \frac{M_{\text{ref}}}{M_j}\, r_p, \qquad
    \operatorname{E}[d_p] = \frac{M_{\text{ref}}}{M_j}\operatorname{E}[r_p], \qquad
    \operatorname{Var}[d_p] = \left(\frac{M_{\text{ref}}}{M_j}\right)^2 \operatorname{Var}[r_p].

**Definition:** the normalisation, applied after segmentation.

**Code:** :func:`anri.io.imaged11.stream_sparse` (its ``monitor`` option).

The cut and the group rule below act on the raw counts. A model must therefore compute the expected recorded value
from the raw light, :math:`\lambda_p` with the factor :math:`M_j / M_{\text{ref}}` of :eq:`rec-cell`, and
normalise after; cutting a normalised model at :math:`c` would be a different cut on every frame.

The cut and the group rule
--------------------------

ImageD11's segmentation (``lima_segmenter``) records a frame's pixels in two steps:

1. a pixel is *kept* when its count is strictly above the cut, :math:`n_p > c` (masked pixels never are);
2. a kept pixel is *recorded* when its 8-connected group of kept pixels in that frame, :math:`G_p`, has at least
   :math:`m` pixels (``pixels_in_spot``; ImageD11's defaults on the Eiger: :math:`c = 1`, :math:`m = 3`).

.. math::
    :label: rec-recorded

    r_p = n_p\, \mathbb{1}(n_p > c)\, \mathbb{1}(|G_p| \ge m).

**Definition:** the recorded value; :math:`r_p = 0` means "not recorded", not "zero photons".

**Code:** :func:`anri.io.imaged11.write_scan` (the cut, for simulated data; the group rule is not in anri yet).

For a Poisson count (sums over :math:`k > c` done in closed form):

.. math::
    :label: rec-trunc

    a_p = P(n_p > c), \qquad
    e_p = \operatorname{E}[n_p\, \mathbb{1}(n_p > c)] = \lambda_p\, P(n_p \ge c), \qquad
    \operatorname{E}[n_p^2\, \mathbb{1}(n_p > c)] = \lambda_p^2\, P(n_p \ge c - 1) + \lambda_p\, P(n_p \ge c).

**Proof:** ``tests/theory/test_recording.py::test_truncated_poisson_moments``

The cut is not linear in the light: :math:`\lambda P(n \ge c)` of a sum is not the sum over its parts. So the cut
must be applied to each cell's integrated light :math:`\lambda_p`, never to a continuous spot, nor to each of several
contributors to a cell separately (``test_cut_not_linear`` in the same file).

**The group.** Given that :math:`p` is kept, its group is the connected set :math:`S \ni p` of kept pixels with
every 8-neighbour outside :math:`S` (its boundary :math:`\partial S`) not kept. The pixels are independent, so

.. math::
    :label: rec-animal

    P_k = P(|G_p| = k \mid p \text{ kept})
    = \sum_{\substack{S \ni p \text{ connected} \\ |S| = k}}\
    \prod_{q \in S \setminus p} a_q \prod_{q \in \partial S} (1 - a_q),

with :math:`a_q = 0` off the detector and on masked pixels.

**Proof:** ``tests/theory/test_recording.py::test_group_probabilities``

For :math:`k = 1, 2` (:math:`N(p)` the 8 neighbours of
:math:`p`):

.. math::
    :label: rec-p12

    P_1 = \prod_{q \in N(p)} (1 - a_q), \qquad
    P_2 = \sum_{q \in N(p)} a_q \prod_{r \in N(p) \cup N(q) \setminus \{p, q\}} (1 - a_r).

**Proof:** ``tests/theory/test_recording.py::test_group_probabilities`` (every kept pattern of a 4 × 4 detector:
both sides are multilinear in the :math:`a_q`, so agreeing on every 0/1 pattern is equality).

A group of :math:`k` pixels holding :math:`p`, with its boundary, reaches :math:`k` pixels from :math:`p`
(Chebyshev distance), and some reach exactly that. The rule "at least :math:`m`" needs :math:`P_1 \dots P_{m-1}`, so

.. math::
    :label: rec-reach

    \text{the recorded value of } p \text{ depends on the } (2m - 1) \times (2m - 1) \text{ cells around it in its frame}

(5 × 5 for :math:`m = 3`; 7 × 7 for :math:`m = 4`, where :math:`P_3` is needed too).

**Proof:** ``tests/theory/test_recording.py::test_group_reach``

**Expected value and variance.** Given that :math:`p` is kept, its group depends only on the other pixels, and
:math:`n_p` is independent of them, so with :math:`w_p = 1 - \sum_{k < m} P_k`:

.. math::
    :label: rec-expected

    \operatorname{E}[r_p] = e_p\, w_p, \qquad
    \operatorname{Var}[r_p] = \operatorname{E}[n_p^2 \mathbb{1}(n_p > c)]\, w_p - e_p^2 w_p^2, \qquad
    P(r_p > 0) = a_p\, w_p.

**Proof:** ``tests/theory/test_recording.py::test_group_probabilities`` (the identity
:math:`P(|G_p| \ge m \mid p \text{ kept}) = 1 - \sum_{k<m} P_k`; the factor :math:`e_p` is the independence of
:math:`n_p`).

Exact for any :math:`m` once :math:`P_1 \dots P_{m-1}` are summed (:eq:`rec-animal`); :eq:`rec-p12` makes it closed
for :math:`m \le 3`.

**Neighbours are correlated.** The group rule couples recorded values: two pixels whose windows
(:eq:`rec-reach`) overlap are not independent. For an isolated pair with :math:`m = 2`, and for the ends of a row of
three with :math:`m = 3`:

.. math::
    :label: rec-cov

    \operatorname{Cov}(r_p, r_q) = e_p e_q (1 - a_p a_q), \qquad
    \operatorname{Cov}(r_1, r_3) = e_1 e_3 a_2 (1 - a_1 a_2 a_3).

**Proof:** ``tests/theory/test_recording.py::test_recorded_covariance``

At :math:`\lambda = 2`, :math:`c = 1` the pair's correlation is 0.82: at the faint edges of spots the recorded
values are far from independent, and the variance of any sum over cells must include these covariances (for
:math:`m = 1` they vanish).

The likelihood of a frame
-------------------------

Let :math:`K` be the set of recorded pixels of a frame, :math:`\partial K` the pixels next to :math:`K` (not in it)
and :math:`U` all the others. :math:`K` is recorded exactly when its pixels are kept, :math:`\partial K` is not kept
(else it would join a recorded group), and the kept pixels of :math:`U` form only groups smaller than :math:`m`.
With the recorded counts :math:`y_p > c`:

.. math::
    :label: rec-likelihood

    P(K, \vec{y}) = \prod_{p \in K} \frac{\lambda_p^{y_p} e^{-\lambda_p}}{y_p!}
    \prod_{q \in \partial K} P(n_q \le c)\; Z_m(U),
    \qquad
    Z_m(U) = P\big(\text{every kept group in } U \text{ has fewer than } m \text{ pixels}\big).

**Proof:** ``tests/theory/test_recording.py::test_frame_likelihood`` (every recorded set of a 2 × 3 detector,
:math:`m = 1, 2, 3`; the counts factor by independence).

Frames and dty rows are independent, so the likelihood of the data is the product of :eq:`rec-likelihood` over
frames. For :math:`m = 1`, :math:`Z_1(U) = \prod_{q \in U} P(n_q \le c)`: every unrecorded pixel contributes
:math:`P(n_q \le c)`, the censored Poisson likelihood. For :math:`m \ge 2`, :math:`Z_m(U)` is the probability of a
pattern of small groups, a sum over patterns with no closed form; it is close to :math:`\prod_U P(n_q \le c)` where
the :math:`a_q` are small.

**Anri's rule:** pixels not recorded are censored, never zero. An unrecorded cell says only "below the cut, or in a
group too small" (:eq:`rec-likelihood`); its likelihood is never :math:`e^{-\lambda_p}`. The indexer's histogram
bins use the exact likelihood of a censored count per bin, with a censoring level per bin in place of the per-pixel
threshold (:doc:`likelihood`, censored bins).

**What ImageD11 can also do, and the model must know.** A per-frame cap ``howmany`` (100000 pixels) raises the cut
on a frame with more kept pixels, a cut that depends on the data: check that it never triggered. A background file
(``bgfile``) is subtracted before the cut, so the cut acts on :math:`n_p - \text{bg}_p` and the recorded values are
not Poisson.

What is left out
----------------

- **Background.** Diffuse scattering, fluorescence of the sample and air scatter add to :math:`\lambda_p`. They are
  not in the forward model; near the cut they change every :math:`a_q` and so every :math:`w_p`.
- **Counting losses.** The Eiger's count-rate limit (pile-up at high rates) and pixel-to-pixel gain differences are
  not modelled; nor is a photon counted twice (above).
- **The cap and background subtraction** of the segmentation (above).
- :math:`Z_m(U)` of :eq:`rec-likelihood` is not computed by anything in anri.
- Code: :eq:`rec-trunc` to :eq:`rec-expected` are not yet in anri (sandbox: ``anri/sandbox/refine/cutexact.py``,
  which agrees with them, and drops the covariances of :eq:`rec-cov`).
