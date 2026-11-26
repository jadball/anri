.. _theory_indexing:

Indexing as tomography
======================

The indexer finds, from the data alone, which orientations each voxel of a scanning 3DXRD map holds and how much of
each. This chapter sets up the model it fits; :doc:`search` chooses the orientations it may use, :doc:`likelihood`
fits it, :doc:`localsearch` refines the orientations off the grid and fits coarse to fine, and :doc:`populations` reads the result.

What a voxel is
---------------

At the scales that matter (voxels of 0.1-1 µm, rays tens to hundreds of µm long), a voxel's orientation distribution is
a small mixture of populations: dislocation cells, twin domains, martensite variants, each nearly constant with a small
spread. In copper strained by 3-4 % in tension, for example, nearly dislocation-free subgrains of 1-3 µm give sharp
peaks, misoriented by hundredths of a degree, on a diffuse cloud from the dislocation walls between them; the
subgrains resolved held at least 30 % of the volume [Jakobsen2006]_. Those sharp sub-spots were seen at an angular
resolution of 0.004° with a 14 µm beam; at ordinary resolution they blur into one smeared peak.

Anri therefore represents each voxel :math:`v` by occupancies :math:`f_{vq} \ge 0` of a list of orientations :math:`q`,
and fits every voxel independently of its neighbours' values.

**Anri's choice:** no coupling between voxels: no smoothness, total variation or basis-function prior on orientation.
Such priors suppress exactly the local variation Anri exists to measure, as averaging each voxel over the rays through
it does. What the data cannot resolve is reported as a mean and a spread (:doc:`populations`), not smoothed away.
The coarse-to-fine levels still couple voxels through the support and the start of the last level's fit; those
couplings are written out, as filters, in :doc:`localsearch`.

The linear model
----------------

A pencil beam at one (dty, omega) lights a line of voxels. The detector sums what they diffract. So the data are
linear in the sample's density over position and orientation [Hayashi2015]_ [Poulsen2004]_:

.. math::
    :label: idx-linear

    \mu_b = \sum_{v, q} A_{b, vq}\, f_{vq},

where :math:`\mu_b` is the expected intensity in a data bin :math:`b` and :math:`A_{b, vq}` is how much of a unit occupancy of
orientation :math:`q` in voxel :math:`v` lands in it. Fitting all voxels jointly is a tomography: one image never says which voxel
along the ray made which part of a peak; only consistency across rows (dty) and across omega places it.

**Cited:** [Hayashi2015]_ [Poulsen2004]_; :math:`A` is Anri's, below.

The histogram
-------------

Fitting at pixel level for every trial orientation would be too slow, so the sparse pixels are reduced once to a
histogram over rings, :math:`\eta`, omega and dty rows:

.. math::
    :label: idx-histogram

    H[r, e, o, k] = \sum_{\text{pixels in bin}} I_{\text{pixel}},

at the scale of the orientation grid: bins of 1° in :math:`\eta` and 1° in omega for the fit, and a finer, row-summed
"lit" histogram (0.5° × 0.25°) for the search. Each pixel's :math:`(2\theta, \eta)` is computed from the lab origin (no
parallax), and the pixel goes to the nearest ring if within that ring's tolerance.

**Definition:** the histogram the indexer fits, in place of the pixels. Its bins: :math:`\eta` bin
:math:`\lfloor(\eta + 180°)/b_\eta\rfloor \bmod n_\eta`, omega bin :math:`\lfloor(\omega - \omega_0)/b_\omega\rfloor` on the scan
(``tests/theory/test_indexing.py::test_pixel_angles_code``, ``tests/theory/test_indexing.py::test_histogram_code``).

**Code:** :func:`anri.index.data.pixel_angles`, :func:`anri.index.data.histogram`,
:func:`anri.index.data.histogram_pixels`.

**Ring tolerances** are measured from the data, not assumed: each ring's :math:`2\theta` profile is taken within half
the gap to its neighbours (for the end rings at most ``max_hw``), its background (the median of the
window's outer fifth) removed, and the tolerance is the distance from the ring's position to its centroid plus the
half-width that holds 95 % of its net intensity, plus half a profile bin (a bin's intensity may sit anywhere within it).
This sums everything that widens a ring: parallax (sample size over distance), strain, and detector distortion. The
rule is checked exactly, in rationals (``tests/theory/test_indexing.py::test_ring_widths_code``).

**Code:** :func:`anri.index.data.ring_profile`, :func:`anri.index.data.ring_widths`.

**Anri's choice:** a histogram loses the shape of each peak within a bin, and parallax. It keeps the information the
search and the occupancy fit need at the scale of a 1-3° orientation grid, and makes them cheap enough to run on every
voxel of a 3000 × 3000 map.

The system matrix
-----------------

For voxel :math:`v` at sample position :math:`\vec{v}_s` and orientation :math:`q`, every prediction :math:`j` (reflection and branch,
:doc:`diffraction`) gives an :math:`(\eta_j, \omega_j)` and a weight

.. math::
    :label: idx-weight

    w_j = L_j\, P_j\, |F_j|^2\; c_{r(j), e}\; T_{vj},

the Lorentz and polarisation factors (:eq:`int-lorentz`, :eq:`int-pol`), the structure factor scaled to a mean of 1,
the measured share :math:`c` of that ring's detector area in that :math:`\eta` bin (0 on module gaps, masked pixels and off the
detector; :func:`anri.index.data.detector_coverage`), and with absorption the transmission along the incoming and
diffracted paths (:eq:`int-absorption`). The prediction's weight is shared:

- over the 2 × 2 nearest (:math:`\eta`, omega) bins, bilinearly (:eq:`idx-bilinear`);
- over dty rows by where the voxel is at that omega (:eq:`dif-dty`): either linearly between the two nearest rows,
  or, given the beam's width, by the beam profile integrated over the voxel's chord in each row
  (:eq:`int-profile`, the same function the renderer uses), each row weighted by its exposure.

With a beam profile a row :math:`k` takes :math:`(\Delta y / v^2)\, W((k - f_k)\Delta y, \omega - \psi)` (:eq:`int-chord`),
:math:`f_k` the voxel's fractional row from :eq:`dif-dty`, over the :math:`2\lceil(3\sigma + w/2 + v/\sqrt{2})/\Delta y\rceil + 1`
rows it can reach (``tests/theory/test_indexing.py::test_system_code``,
``tests/theory/test_indexing.py::test_rows_and_blocks_code``, ``tests/theory/test_indexing.py::test_predictions_code``).

**Code:** :func:`anri.index.occupancy.system`, :func:`anri.index.occupancy.beam_rows`,
:func:`anri.index.search.predictions`.

**Definition:** the weight of a point-like prediction; its factors are derived in :doc:`intensity` and :doc:`crystal`.

Along each of :math:`\eta` and omega, a prediction at angle :math:`a`, in bins of width :math:`b` from :math:`a_0` (bin
:math:`i` centred on :math:`a_0 + (i + \tfrac{1}{2})b`), goes to bins :math:`\lfloor x \rfloor` and
:math:`\lfloor x \rfloor + 1` with weights

.. math::
    :label: idx-bilinear

    1 - t \quad\text{and}\quad t, \qquad x = \frac{a - a_0}{b} - \frac{1}{2}, \quad t = x - \lfloor x \rfloor,

which sum to 1 and whose weighted bin centre is :math:`a`: the prediction keeps its weight and its position.

**Proof:** ``tests/theory/test_indexing.py::test_bilinear_keeps_weight_and_position``,
``tests/theory/test_indexing.py::test_bilinear_code``.

**Code:** :func:`anri.index.data.eta_omega_corners`.

**Coverage at the prediction.** A point-like prediction's light lands where it is predicted, at :math:`(2\theta_j,
\eta_j)`, and is recorded only if that point is measured. Its expected measured share is :math:`m(2\theta_j, \eta_j)`,
with :math:`m = 1` on measured detector area and 0 elsewhere. :eq:`idx-weight` uses instead its bin's coverage
:math:`c`, the measured share of the (ring, :math:`\eta` bin)'s area (:func:`anri.index.data.detector_coverage`
samples each pixel 4 × 4 within the ring's :math:`2\theta` tolerance). The two agree only on average over where in the
bin the prediction falls:

.. math::
    :label: idx-coverage

    c = \frac{1}{|B|}\int_B m\, dA = \operatorname{E}_{\vec{x} \sim U(B)}\big[m(\vec{x})\big], \qquad
    \big|m(2\theta_j, \eta_j) - c\big| \le \max(c, 1 - c),

with :math:`B` the bin's area on the detector. For one orientation the prediction is at one place. On a bin half
covered, a spot on its measured half is predicted at half its light, and a spot on its masked half at half when none is
recorded. The fit then moves occupancy to make up a misfit of up to :math:`\max(c, 1 - c)` of the reflection's light.
The coverage is a shape assumption: light spread evenly over the bin. The misfit arises only on partly covered bins:
the edges of each gap's crossing of a ring, a bin or two at 0.5°, and wherever a gap runs along a ring. On an Ag2S scan
the horizontal module gaps did that over :math:`\eta = \pm(65`-86)°, where :math:`c` lay between 0 and 1 because the gap
covered part of the ring's :math:`2\theta` window, and a sharp prediction there is either measured or not.

**Proof:** ``tests/theory/test_indexing.py::test_coverage_at_prediction`` (the average over a bin cut by a gap's edge
across :math:`\eta` and along :math:`2\theta`; the misfit's bound for :math:`m \in \{0, 1\}`).

So each prediction is scaled by :math:`m` where it lands, on the ring's centre line, instead of by its bins' average.
The histogram stage records :math:`m` at 16 points per 0.5° :math:`\eta` bin, 0.031°, half a pixel at 900 pixels from
the beam, by tracing each point to the detector and, with a distortion, to the pixel whose corrected position it is
(``tests/theory/test_indexing.py::test_point_coverage_code``). Every bin is then predicted with the light that lands
on its measured pixels, up to the prediction's own width. Unmeasured cells need nothing more: they contribute nothing
to the likelihood, as :doc:`recording` requires of cells with no data. They are not censored, which would say "below
the cut". Histogram files from before carry only the bins' average, which the later stages use as before, now taken
once at each prediction's own :math:`\eta`.

**Code:** :func:`anri.index.data.point_coverage`, :func:`anri.index.data.coverage_at`.

**Fly and helical scans.** When dty moves during the rotation, the frames of one omega bin were taken up to half a step
away from their row's nominal dty. The real dty per (row, omega bin) is measured from the frames' own motor readings and
used in place of the nominal one: the mean over the bin's frames of their dty less the row's. Bins with more or fewer
frames than usual get their exposure, frames over the median bin's
(``tests/theory/test_indexing.py::test_dty_offsets_code``).

**Code:** :func:`anri.index.data.dty_offsets`.

The model is never stored. Each pass over the data recomputes :math:`A`'s entries for a block of voxels and accumulates
:math:`A f` (a segment sum) or :math:`A^T r` (a gather), so memory is set by the block size, not by voxels ×
orientations: blocks of the largest power of 2 of voxels whose entries, 32 bytes per corner, fit the budget
(``tests/theory/test_indexing.py::test_forward_backward_code``, ``tests/theory/test_indexing.py::test_rows_and_blocks_code``).

**Code:** :func:`anri.index.occupancy.forward`, :func:`anri.index.occupancy.backward`,
:func:`anri.index.occupancy.block_voxels`.

The voxel grid
--------------

The voxels are an :math:`N \times N` grid of the dty step, centred on the rotation axis and sized from :math:`y_0` and
the scanned range, as ImageD11 pads its reconstructions. The histogram stage fixes it, so a :math:`y_0` chosen later
moves no voxel. It can be widened on every side (``grid_pad``) for a sample reaching past the scanned range, whose outer
part would otherwise pile its intensity onto the grid's edge. With ``gridstep`` :math:`G`, the dty rows are summed in
groups of :math:`G` and the voxels are :math:`G` dty steps wide. With :math:`n` rows of step :math:`\Delta` from
:math:`y_{\text{first}}`, the axis :math:`s` rows from their middle, and a pad of :math:`g`, voxel :math:`(i, j)` is at

.. math::
    :label: idx-grid

    x = (i - \lfloor N/2 \rfloor)\Delta, \quad y = (\lfloor N/2 \rfloor - j)\Delta, \qquad
    N = n + p + 2\lceil g/\Delta \rceil, \quad p = \lceil 2|s| \rceil + 1\ (+1 \text{ if } n + p \text{ is even}),
    \quad s = \lfloor n/2 \rfloor - \frac{y_0 - y_{\text{first}}}{\Delta}.

**Definition:** ImageD11's reconstruction order and padding. **Proof:** ``tests/theory/test_indexing.py::test_grid_covers_the_scan``
(the grid reaches past both ends of the scan), ``tests/theory/test_indexing.py::test_grid_code``.

**Code:** :func:`anri.geom.scan.recon_positions`, :func:`anri.index.mask.voxel_grid`.

What is left out
----------------

- **Parallax** in the histogram: a voxel off the axis puts its peaks at a slightly different :math:`2\theta` and :math:`\eta`
  than the axis would. The ring tolerance absorbs the :math:`2\theta` part; the :math:`\eta` part is below the 1° bins for
  samples small next to the detector distance.
- **Peak shape within a bin**, and strain: the histogram sees orientation only.
- **Coverage across a prediction's width** (:eq:`idx-coverage`): a prediction is scaled by the coverage at its centre,
  so a spot whose own width straddles a gap's edge is counted as wholly in or wholly out.
- **Reflections near the rotation axis**, :math:`|\sin\eta| \le` ``etacut`` (default 0.2): smeared over many omega bins
  (:ref:`theory_rotation_motion`), they are left out of the search and the fit.
