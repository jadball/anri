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

**Code:** :func:`anri.simulate.stages.record` (simulated data: each count drawn, see `Simulated data`_).

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

.. _theory_shared_pixels:

Shared pixels
-------------

The Eiger2's pixels along its chips' edges are 75 × 150 µm, twice the others, and the readout divides each one's count
between two pixels of the output image, one of them over the gap between the chips (a *virtual pixel*). pyFAI's
detector definitions (``src/pyFAI/detectors/_dectris.py``, upstream commit ``aae0b3a``, read 2026-11-26) give the
layout, not the rule. An Eiger module of 2 × 4 chips of 256 × 256 pixels is 514 × 1030 pixels, "two missing pixels are
interpolated at each sub-module boundary"; an Eiger2 module is 512 × 1028. Module gaps are 37 × 10 pixels (Eiger) and
38 × 12 (Eiger2), masked. The Eiger2 CdTe also masks a 2-pixel column in the middle of each module. pyFAI places every
output pixel on one regular 75 µm grid ("images are re-built to be contiguous"), and models neither the large pixels'
area nor how their counts are divided. So the Eiger2's 512 rows hold no extra pixels at its chips' horizontal
boundary, and which boundaries carry virtual pixels is not settled by pyFAI.

How the readout divides an integer count (evenly with a rule for the odd photon, at random, or as non-integer values
after a correction) is not stated here. Everything below holds for any rule that conserves the count. The rule can be
measured from the data: wherever both outputs of a shared pixel are recorded, halving keeps their counts within one of
each other, with the odd photon always on the same side, while a random split spreads them binomially.

Take a physical pixel :math:`P`, its light :math:`\lambda_P` integrated over its whole area (:eq:`rec-cell`), its count
:math:`n \sim \operatorname{Poisson}(\lambda_P)`, and its output pixels :math:`o \in O(P)` with counts :math:`n_o`
given by a rule that, given :math:`n`, depends on nothing else and keeps :math:`\sum_o n_o = n`. Then

.. math::
    :label: rec-shared

    \sum_{o \in O(P)} n_o = n \sim \operatorname{Poisson}(\lambda_P), \qquad
    P\big(n_{o_1} = 2,\ n_{o_2} = 0\big) = 0 < P(n_{o_1} = 2)\, P(n_{o_2} = 0) \quad \text{(halving)}:

merged, the outputs are one cell of :eq:`rec-poisson`, with the physical pixel's area, whatever the rule. Apart they
are not independent cells: under halving (:math:`\lceil n/2 \rceil` to one, :math:`\lfloor n/2 \rfloor` to the other)
their counts differ by at most one.

**Proof:** ``tests/theory/test_recording.py::test_shared_pixel_merge`` (the sum, for halving into two and three; the
halving's joint law against its marginals, exactly).

Which pixels are shared is known from the detector: its layout of chips and modules (pyFAI's definitions, above).
Only the dividing rule is not. Anri does not correct for shared pixels yet, and treats their outputs as ordinary
cells. That is a small effect, and it is local. On pyFAI's Eiger2 layout the module's four extra columns and their
partners are 8 of its 1028, about 1 % of the pixels, if the rows carry none. The correction, when it is wanted, is the
merge above: the outputs of a shared pixel summed into one cell with the physical pixel's area and centre, for data
and model alike.

The segmentation runs on the output image, so the cut and the group rule act on the divided counts, not on
:math:`n`. With the group rule left aside (:math:`m = 1`), the merged cell's recorded value and the light it loses are

.. math::
    :label: rec-shared-recorded

    r_P = \sum_{o \in O(P)} n_o\, \mathbb{1}(n_o > c), \qquad
    \lambda_P - \operatorname{E}[r_P] = \sum_n P(n \mid \lambda_P)\, \ell(n), \qquad
    \ell(n) = \sum_{o \in O(P)} n_o\, \mathbb{1}(n_o \le c).

Under halving at :math:`c = 1`, :math:`\ell = 0, 1, 2, 1, 0` for :math:`n = 0, \dots, 4`, and 0 beyond (both halves
exceed :math:`c` once :math:`n \ge 2(c + 1)`). So the loss is :math:`e^{-\lambda}(\lambda + \lambda^2 +
\lambda^3/6)`, largest at :math:`\lambda = 1.882`, the root of :math:`\lambda^3 + 3\lambda^2 - 6\lambda - 6`, where it
is 0.995 photons: against :math:`e^{-1} = 0.368` for an ordinary pixel and 0.736 for two pixels of half the light each.
At :math:`c = 2` it is 2.156. The cut on a halved pixel removes a count of :math:`2c + 1` in part and fewer whole,
like a cut near :math:`2c + 1` on the physical pixel. Each half is non-decreasing in :math:`n`, so the recorded
values stay non-decreasing functions of the physical counts, and :eq:`rf-harris` holds.

**Proof:** ``tests/theory/test_recording.py::test_shared_pixel_cut`` (the losses at :math:`c = 1, 2`; the deficit, its
derivative and maximum; the monotonicity of each half).

The group rule sees the outputs: a shared pixel's two outputs are 8-neighbours, kept or not by one count, so their kept
events are dependent, and both kept count as two pixels of a group. :eq:`rec-animal` holds with the outputs of each
shared pixel taken as one node, whose kept pattern is a function of its :math:`n`. A lone bright shared pixel is a
group of two by itself.

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

Checked in ImageD11's source (upstream master, 2026-11-23; the same in release 1.17.0): ``select`` keeps a pixel when
its value times the mask is above ``cut``; ``clean`` labels the kept pixels of one frame with
``cImageD11.sparse_connectedpixels`` (a pixel joins the one before it in its row and the three at columns
:math:`j - 1, j, j + 1` in the row above: 8-connected), counts each label's pixels (``np.bincount``) and keeps those
in labels of at least ``pixels_in_spot`` (for ``pixels_in_spot`` :math:`\le 1` it keeps every pixel).

.. math::
    :label: rec-recorded

    r_p = n_p\, \mathbb{1}(n_p > c)\, \mathbb{1}(|G_p| \ge m).

**Definition:** the recorded value; :math:`r_p = 0` means "not recorded", not "zero photons".

**Code:** :func:`anri.io.imaged11.segment`.

**Proof:** ``tests/theory/test_recording.py::test_segment_is_the_group_rule`` (every kept pattern of a 4 × 4
detector, :math:`m = 1 \dots 5`); ``tests/reference/test_imaged11.py::TestSegment`` checks it against ImageD11's own
``lima_segmenter`` on Poisson frames.

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
:math:`n_p`); ``tests/theory/test_recording.py::test_recorded_moments_code`` (the code, with :eq:`rec-trunc` and
:eq:`rec-p12`, against the animal sums at 30 digits, :math:`c = 0, 1, 2`, :math:`m = 1, 2, 3`).

**Code:** :func:`anri.io.imaged11.recorded_moments` (:math:`m \le 3`, differentiable in the light), on the 5 × 5
cells :func:`anri.io.imaged11.neighbourhoods` gathers.

Exact for any :math:`m` once :math:`P_1 \dots P_{m-1}` are summed (:eq:`rec-animal`); :eq:`rec-p12` makes it closed
for :math:`m \le 3`.

**Compact spots.** A spot whose light is mostly in one or two pixels forms a group of fewer than :math:`m` pixels
unless its neighbours are bright enough to be kept, so the group rule removes it however bright that pixel is. The
expected share of a Gaussian spot's light recorded at :math:`c = 1`, without the group rule (:math:`m = 1`) and with
ImageD11's (:math:`m = 3`), for a spot centred on a pixel or on a pixel's corner, and with the measured point spread's
wing (28 % of the light widened by 1.3 px, AGENTS.md), :math:`\sum_p \operatorname{E}[r_p] / \sum_p \lambda_p`
for a spot of :math:`N` photons, as :math:`m = 1` / :math:`m = 3`:

.. list-table::
    :header-rows: 1

    * - spot (sd, where)
      - :math:`N = 10`
      - :math:`N = 30`
      - :math:`N = 100`
    * - 0.25 px, on a centre
      - 0.928 / 0.003
      - 0.953 / 0.102
      - 0.988 / 0.860
    * - 0.25 px, on a corner
      - 0.918 / 0.734
      - 0.999 / 0.999
      - 1.000 / 1.000
    * - 0.5 px, on a centre
      - 0.766 / 0.375
      - 0.930 / 0.926
      - 0.987 / 0.987
    * - 0.25 px with the wing, on a centre
      - 0.733 / 0.022
      - 0.806 / 0.383
      - 0.904 / 0.897

**Proof:** ``tests/theory/test_recording.py::test_compact_spot_numbers`` (:eq:`rec-expected` by
:func:`anri.io.imaged11.recorded_moments`, on 15 × 15 cells).

So the group rule matters only for spots of fewer than about 100 photons, but there it depends on where the spot
falls within its pixel: a sharp spot of 30 photons keeps 10 % of its light centred on a pixel and all of it on a
corner. A method that ignores the rule loses faint compact spots by their sub-pixel position, a bias in whatever is
fitted to them, not only a loss of counts; and the indexer's censoring level per bin (:doc:`likelihood`) cannot
stand for it, since an empty bin may hide a removed group of any brightness.

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

**Any two pixels.** Given the kept pattern, the counts of kept pixels are independent of each other and of the pattern,
with :math:`\operatorname{E}[n_p \mid p \text{ kept}] = e_p / a_p`; and the kept indicators are independent, so
conditioning on :math:`p` and :math:`q` kept sets :math:`a_p = a_q = 1` in any probability of the pattern. With
:math:`A_p` the event :math:`|G_p| \ge m` and :math:`s_p = \sum_{k<m} P_k = 1 - w_p` (:eq:`rec-animal`), for
:math:`p \ne q` in one frame:

.. math::
    :label: rec-cov-pair

    \operatorname{Cov}(r_p, r_q) = e_p e_q \left(\rho_{pq} - w_p w_q\right), \qquad
    \rho_{pq} = P(A_p \cap A_q \mid p, q \text{ kept}) = 1 - s_p^{(q)} - s_q^{(p)} + J_{pq},

**Proof:** ``tests/theory/test_recording.py::test_recorded_covariance_any_pair`` (below).

with :math:`s_p^{(q)}` the polynomial :math:`s_p` at :math:`a_q = 1`, and :math:`J_{pq} = P(\text{neither}\ A_p
\text{ nor } A_q \mid p, q \text{ kept})`, a sum over the groups the two pixels can have: one group of fewer than
:math:`m` pixels holding both, or two disjoint groups :math:`S \ni p`, :math:`T \ni q` of fewer than :math:`m`
each, neither touching the other (:math:`S \cap \partial T = T \cap \partial S = \emptyset`),

.. math::
    :label: rec-joint

    J_{pq} = \sum_{\substack{S \ni p, q \\ |S| < m}} \prod_{S \setminus \{p, q\}} a \prod_{\partial S} (1 - a)
    \; + \sum_{\substack{S \ni p,\ T \ni q \\ |S|, |T| < m}} \prod_{(S \cup T) \setminus \{p, q\}} a
    \prod_{\partial S \cup \partial T} (1 - a).

**Proof:** ``tests/theory/test_recording.py::test_recorded_covariance_any_pair`` (every pair of a 2 × 3 and of a
1 × 5 detector, :math:`m = 1, 2, 3`, against the enumerated covariance, symbolically; and :eq:`rec-joint` on every kept
pattern of a 3 × 4 detector, every pair, :math:`m = 2, 3, 4`: both sides are multilinear in the :math:`a`, as in
:eq:`rec-p12`).

:eq:`rec-cov` is its two simplest cases. Pixels in different frames are independent, and so are pixels of one frame
more than :math:`2(m - 1)` apart (Chebyshev distance): their windows (:eq:`rec-reach`) are disjoint. At exactly
:math:`2(m - 1)` some are correlated (the ends of a row of five, :math:`m = 3`:
:math:`\operatorname{Cov}(r_1, r_5) = e_1 e_5\, a_2 a_3 a_4 (1 - a_3)`, in the same test). So a pixel is correlated
with the pixels of its frame in a :math:`(4m - 3) \times (4m - 3)` stencil: 5 × 5 for :math:`m = 2`, 9 × 9 for
:math:`m = 3`. The covariance of the sums of :math:`r_p f_p` over two sets of cells :math:`R`, :math:`R'` (a moment's
features :math:`f_p`, :doc:`moments`) is then :math:`\sum_{p \in R,\, q \in R'} \operatorname{Cov}(r_p, r_q)\,
f_p f_q^T` over the pairs in one frame within the stencil, :math:`\operatorname{Var}(r_p)` (:eq:`rec-expected`) where
:math:`p = q`; it vanishes between sets more than :math:`2(m - 1)` apart in every frame, and not otherwise.

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

Simulated data
--------------

**Anri's rule:** simulated data are a realisation of this model, never its expectation. The expected recorded value
:math:`\operatorname{E}[r_p] = e_p w_p` (:eq:`rec-expected`) is not an integer and cannot be written as ImageD11's
sparse counts; nor can the noise and the correlations of :eq:`rec-cov` be had without drawing the counts.
:func:`anri.simulate.stages.record` draws each rendered cell's count from :math:`\operatorname{Poisson}(\lambda_p)`
(:eq:`rec-poisson`), then keeps and records pixels by :eq:`rec-recorded` (:func:`anri.io.imaged11.segment`), with
:math:`c = 1`, :math:`m = 3` by default, as real data. The draws are reproducible: row :math:`i` of a render with
seed :math:`s` uses numpy's default generator seeded with :math:`(s, i)`, the same whichever rows a job renders, and
the seed is in the render stage's provenance.

For tests that need a fixed file, ``noise="round"`` writes :math:`\operatorname{round}(\lambda_p)` through the same
cut and group rule. That is neither a realisation nor :math:`\operatorname{E}[r_p]`: a cell of :math:`\lambda = 1.4`
is never recorded at :math:`c = 1`, though :math:`P(n > 1) = 0.41`. Never judge a method on rounded data.

What the simulation assumes, beyond the model above: each photon counted once and no pile-up (both being checked),
no background, no mask, no ``howmany`` cap and no background file. The renderer drops cells whose light is below
:math:`10^{-3}` photons (its ``min_value``); such a cell would be kept with probability
:math:`P(n \ge c + 1) \le \lambda^{c + 1} / (c + 1)! \le 5 \times 10^{-7}` at :math:`c = 1` (each term
:math:`\lambda^{c + 1 + i} / (c + 1 + i)!` of the tail is at most :math:`\lambda^{c + 1} / (c + 1)!` times
:math:`\lambda^i / i!`, and with the factor :math:`e^{-\lambda}` those sum to 1).

What is left out
----------------

- **Background.** Diffuse scattering, fluorescence of the sample and air scatter add to :math:`\lambda_p`. They are
  not in the forward model; near the cut they change every :math:`a_q` and so every :math:`w_p`.
- **Counting losses.** The Eiger's count-rate limit (pile-up at high rates) and pixel-to-pixel gain differences are
  not modelled; nor is a photon counted twice (above).
- **The cap and background subtraction** of the segmentation (above).
- **Shared pixels** (:eq:`rec-shared`) are not corrected for (about 1 % of the pixels). The readout's rule for
  dividing a count, and whether a flatfield or
  count-rate correction makes the outputs non-integer before segmentation (then they are not Poisson at all), are to
  be confirmed. :func:`anri.io.imaged11.recorded_moments` treats every output as an independent pixel; the group
  rule's shared node is not computed by anything in anri, and nothing in :mod:`anri.io` merges shared outputs yet.
- :math:`Z_m(U)` of :eq:`rec-likelihood` is not computed by anything in anri.
- Code: :eq:`rec-expected` is :func:`anri.io.imaged11.recorded_moments` for :math:`m \le 3`; the covariances
  :eq:`rec-cov-pair` and :eq:`rec-joint` are not yet in anri (sandbox: ``anri/sandbox/refine/cutexact.py``, which
  agrees with them for :math:`m \le 3`).
