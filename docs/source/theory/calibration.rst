.. _theory_calibration:

Calibrating the geometry
========================

The forward model and the indexer need the detector's distance, beam centre and tilts, and the wavelength. The
motors and the monochromator give a first guess, rarely good enough: a ring 0.05° off its prediction falls outside the
indexer's ring windows. Anri calibrates in two steps, as ImageD11's notebooks do [ImageD11]_: powder rings, from that
first guess to a geometry good enough to index; then the indexed scan's peaks, which separate the wavelength from the
distance and model where each peak scatters from. The parameters are ImageD11's [FABLEgeom]_ (:doc:`frames`).

Powder rings
------------

The data are a detector image: a dense powder image (e.g. a calibrant's), or a scan's sparse pixels summed over every
frame, whose spots fill the rings well enough. Each pixel above a threshold :math:`c` keeps the weight
:math:`w_p = I_p - c` and its distortion-corrected position, and its :math:`(2\theta_p, \eta_p)` is seen from the lab
origin (:eq:`dif-raytrace` inverted, :eq:`frm-eta`), as ImageD11 computes it: there is no parallax.

**Code:** :func:`anri.calib.rings.sum_frames`, :func:`anri.calib.rings.lit_pixels`,
:func:`anri.calib.rings.pixel_tth_eta`.

A phase's rings are its reflections' :math:`d^*` (:doc:`crystal`), those within ``merge`` of each other taken as one,
at :math:`2\theta_r = 2\arcsin(\lambda d^*_r / 2)` (:eq:`dif-bragg`).

**Code:** :func:`anri.calib.rings.ring_ds`.

The residuals
^^^^^^^^^^^^^

Each pixel goes to the nearest ring within a tolerance :math:`t` (degrees of :math:`2\theta`), and to one of
:math:`n_\eta` sectors in :math:`\eta`. Rings closer than :math:`2\max t` to a neighbour are left out: their pixels
cannot be told apart, and each ring's profile pulls the other's mean. Each (ring, sector) :math:`(r, j)` gives one
residual, the weighted mean :math:`2\theta` of its pixels minus its ring's:

.. math::
    :label: cal-ring-residual

    \rho_{rj} = \frac{\sum_{p \in (r, j)} w_p\, 2\theta_p}{\sum_{p \in (r, j)} w_p} - 2\theta_r(\lambda).

**Code:** :func:`anri.calib.rings.assign`, :func:`anri.calib.rings.ring_offsets`.

**Anri's choice:** every sector counts once, as each peak of ImageD11's sector-by-sector peak search does, so a bright
spot does not outweigh the rest of its ring. The mean is over each pixel's exact :math:`2\theta`, so the arc's
curvature within a sector does not bias it. Since a sector's mean is over the pixels within :math:`t`, the last
tolerance must still span a few pixels (one Eiger pixel at 150 mm is 0.03°).

The fit
^^^^^^^

The free parameters :math:`\vec{x}` (by default the distance, the beam centre and tilt_y, tilt_z) minimise
:math:`\sum \rho_{rj}^2` by Levenberg-Marquardt: the Gauss-Newton step from :math:`\matr{J}^T\matr{J}`, with
:math:`\matr{J} = \partial\vec{\rho} / \partial\vec{x}` from JAX's forward-mode differentiation [JAX2018]_, damped by
:math:`\mu\,\operatorname{diag}(\matr{J}^T\matr{J})`, :math:`\mu` raised tenfold until a step lowers the cost and
lowered after it. The pixels are re-assigned before every step, so each ring's window follows the ring. The fit runs in
rounds of decreasing tolerance; from the second, sectors further than ``clip`` robust standard deviations
(:math:`1.4826` times the median absolute residual) from their ring are left out (another phase, a spot on a ring's
tail). A round stops when every step is below a tenth of its parameter's error. The covariance of the free parameters
is :math:`(\matr{J}^T\matr{J})^{-1}\, \chi^2 / (N - n)`, with :math:`N` sectors and :math:`n` free parameters.

**Code:** :func:`anri.calib.rings.fit_rings`, :func:`anri.calib.stages.stage_rings`.

**Anri's choice:** tilt_x turns the detector in its own plane, about the beam: it moves no ring, so rings cannot fit
it. The wavelength stays as given. A ring of radius :math:`R` on a detector at distance :math:`L` has
:math:`\tan 2\theta = R/L`, and :math:`2\theta \approx \lambda d^*` at small angles, so :math:`R \approx L\lambda d^*`:
rings constrain the product :math:`L\lambda`, and the distance absorbs a wrong wavelength (1.8 % in the calibration
guide's phantom, +2 % in the distance). Both are left for the scan.

Searching before fitting
^^^^^^^^^^^^^^^^^^^^^^^^

Assigning pixels to rings only works from a start within about one ring gap, and the motors' centre can be 100 pixels
off. So two searches come first, neither of which assigns rings.

**The centre.** About the true centre every ring is a circle (to the small ellipticity the tilts give it), so the
histogram :math:`H_k(\vec{c})` of the pixels' radii about a trial centre :math:`\vec{c}`, weighted by
:math:`\sqrt{w_p}`, is most peaked there, whatever the distance and the rings:

.. math::
    :label: cal-centre

    \hat{\vec{c}} = \arg\max_{\vec{c}} \sum_k H_k(\vec{c})^2.

It is searched on grids of 8, 2 and 0.5 pixels (bins of half the grid step), each around the best of the last, the
first reaching ``max_shift`` pixels from the start.

**Code:** :func:`anri.calib.rings.find_centre`.

**The distance.** With the centre found, a pixel at :math:`2\theta_p` seen from the starting distance :math:`L_0`
is at :math:`u_p = \tan 2\theta_p` in units of :math:`L_0`; from a distance :math:`L`, ring :math:`r` falls at
:math:`u = \tan(2\theta_r)\,L/L_0`. Each trial :math:`L` scores the weight its ring windows catch, less what a flat
background of the same total would put in them:

.. math::
    :label: cal-distance

    s(L) = \sum_r \left[\sum_{p:\ u_p \in W_r(L)} \sqrt{w_p} \;-\; \bar{w}\, |W_r(L)| \right], \qquad
    W_r(L) = \frac{L}{L_0}\left[\tan(2\theta_r - t),\ \tan(2\theta_r + t)\right],

over the rings whose windows lie within the pixels' range of :math:`u`, with :math:`\bar{w}` the summed
:math:`\sqrt{w_p}` per unit :math:`u` over that range. :math:`L/L_0` runs over ``span`` in logarithmic steps that move
the outermost ring by about a quarter of :math:`t`.

**Code:** :func:`anri.calib.rings.find_distance`.

**Anri's choice:** the square root of the intensity, in both searches, so that a few strong spots do not decide. Two
peaks of :math:`s(L)` of similar height mean the rings could be matched one ring off.

The scan's peaks
----------------

With the sample indexed (the fit stage's map, :doc:`likelihood`), each peak can be matched to an orientation and a
reflection, and the geometry fitted to the peaks' :math:`(s_c, f_c, \omega)`. The peaks are ImageD11's 2D peaks,
merged into one per 4D label and dty row: their intensity-weighted :math:`(s_c, f_c, \omega, \text{dty})`. Merging
within a row averages over the energy band (each frame of a rocking peak diffracts a different part of it) and keeps
each row's scattering origin.

**Code:** :func:`anri.io.imaged11.read_peaks`.

Matching, with parallax
^^^^^^^^^^^^^^^^^^^^^^^

In a scan the beam crosses a large grain at a different place in every row, so a peak's scattering origin, and with it
its :math:`\vec{g}`, depends on where along the beam it came from: the parallax a single-grain fit (ImageD11's
``makemap``, ``refine_em``) cannot model. The beam's centre line at the peak's dty and :math:`\omega` is sampled at
every voxel it crosses, :math:`\vec{v}_t` in the sample frame (:eq:`frm-stack` inverted). From each, the peak's
pixel gives :math:`\vec{k}_{\text{out}}` and so :math:`\vec{g}_t = \vec{k}_{\text{out}} - \vec{k}_{\text{in}}`
(:eq:`dif-laue`), and each orientation :math:`o` the voxel holds (occupancies above ``min_occ`` times the map's
largest) gives the peak's Miller indices :math:`\vec{h}_{to} = \text{UBI}_o\, \vec{g}_t`. The match is

.. math::
    :label: cal-match

    (\hat{t}, \hat{o}) = \arg\min_{t, o} \left|\vec{h}_{to} - \operatorname{round}(\vec{h}_{to})\right|,

the reflection :math:`\operatorname{round}(\vec{h}_{\hat{t}\hat{o}})`, and the origin the occupancy-weighted mean of the
points along the line where :math:`\hat{o}` is held. Of each round's matches, those within its tolerance are kept, one
peak per (orientation, reflection, branch, dty row), the best matched; the branch is the one whose predicted
:math:`\omega` is nearer.

**Code:** :func:`anri.calib.fit.match`, :func:`anri.calib.fit.calibrate`.

The fit
^^^^^^^

Each kept peak gives three residuals, observed minus predicted (:func:`anri.fwd.peak.get_centroid_both`, the forward
model's own centroid, :doc:`diffraction`), in units of the round's robust standard deviations
:math:`\vec{\sigma} = 1.4826 \operatorname{MAD}` of each:

.. math::
    :label: cal-scan-residual

    \vec{\rho}_i = \frac{(s_c, f_c, \omega)_i - \vec{c}\left(s\,\text{UBI}_{o}\, \matr{R}(\vec{a}_o)^T,\ \vec{v}_i,
    \vec{h}_i;\ \vec{x}\right)}{\vec{\sigma}},

with :math:`\vec{x}` the free geometry, :math:`s` the lattice scale, and :math:`\matr{R}(\vec{a}_o)` a small rotation
of orientation :math:`o` (a Rodrigues vector, refitted with the geometry). Peaks beyond ``clip`` :math:`\sigma` are
left out. The Gauss-Newton normal equations have a block per orientation, coupled only through :math:`\vec{x}`:
each orientation's rotation moves only its own peaks. Each orientation's :math:`3 \times 3` block is eliminated (a
Schur complement), leaving a system the size of :math:`\vec{x}`:

.. math::
    :label: cal-schur

    \matr{S} = \matr{H}_{xx} - \sum_o \matr{H}_{xo} \matr{H}_{oo}^{-1} \matr{H}_{ox}, \qquad
    \matr{S}\, \Delta\vec{x} = -\vec{b}_x + \sum_o \matr{H}_{xo} \matr{H}_{oo}^{-1} \vec{b}_o, \qquad
    \Delta\vec{a}_o = -\matr{H}_{oo}^{-1}\left(\vec{b}_o + \matr{H}_{ox} \Delta\vec{x}\right),

the same step as solving for everything together, damped as above (Levenberg-Marquardt). The covariance of the free
parameters is :math:`\matr{S}^{-1}\, \chi^2 / (3N - n - 3G)`, with :math:`N` peaks and :math:`G` orientations.

**Code:** :func:`anri.calib.fit.fit_geometry`.

**What the scan separates.** Bragg's law and the Ewald condition depend on the wavelength and the lattice only through
:math:`\lambda / d`: scaling :math:`\vec{g}` and :math:`\vec{k}` together leaves :eq:`dif-ewald` unchanged. So the
wavelength and the lattice scale cannot both be free: a calibrant (a certified lattice) frees the wavelength, any other
sample (a known energy) the lattice scale. The distance is no longer tied to the wavelength as it was by the rings: a
peak's :math:`\omega` fixes its :math:`\theta` through the Ewald condition, whatever the distance, and its radius on
the detector then fixes the distance.

**Rounds.** Each round matches every peak again with the latest geometry and orientations, with its own tolerance, so
the tolerances can shrink from round to round. After the listed ones the last is repeated (at most ``extra_rounds``
times) while a round still moves a free parameter by more than its standard deviation; if it still does, the quoted
errors are too small, and the stage warns.

**Code:** :func:`anri.calib.fit.calibrate`, :func:`anri.calib.stages.stage_fit`.

**Anri's choice:** for calibrants and little-deformed samples only. Each peak goes to the orientation that predicts it
best at the current geometry, so where a voxel holds orientations closer together than the geometry's error (cells,
subgrains, a local search's children, :doc:`localsearch`), the matching absorbs part of that error and each round
closes only part of the gap: on the 316L phantom's cells the distance was still :math:`6 \times 10^{-4}` out after ten
rounds. The stage refuses a local search's map.

The strain left
^^^^^^^^^^^^^^^

With the geometry fixed, each orientation with at least ``min_peaks`` peaks gets its lattice freed: six strains as well
as its rotation, :math:`\text{UBI} \to \text{UBI}\,\matr{R}(\vec{a})^T(\matr{I} + \matr{E})` with :math:`\matr{E}`
symmetric. Its strain from the reference lattice :math:`\matr{B}` (scaled by :math:`s`) is then read from
:math:`\matr{K} = \matr{B}\,\text{UBI} = \matr{U}^T(\matr{I} + \matr{E})`:

.. math::
    :label: cal-strain

    \matr{I} + \matr{E} = \left(\matr{K}^T\matr{K}\right)^{1/2}, \qquad
    \varepsilon_{\text{h}} = \tfrac{1}{3}\operatorname{tr}\matr{E}, \qquad
    \matr{E}_{\text{dev}} = \matr{E} - \varepsilon_{\text{h}}\matr{I},

in the sample frame. On a strain-free calibrant a right geometry leaves only noise, as ImageD11's silicon notebook
estimates its precision. A deviatoric strain shared by every orientation (its peak-weighted mean) points to the tilts or
the centre; a hydrostatic one to the wavelength or the distance against the reference lattice. The ratio of
:math:`\chi^2` with free lattices to that without tells how much of the misfit the strains explain.

**Code:** :func:`anri.calib.fit.strain_left`, :func:`anri.calib.fit.lattice_strain`,
:func:`anri.calib.fit.split_strain`.

Remaking a calibration
----------------------

``python -m anri.calib pars`` writes a rings or calibration file as ImageD11 parameter files, and its ``pars.json``
names the file it came from. The indexer's histogram stage, given that ``pars.json``, records the file as its source,
so the provenance of a calibration runs back through the indexing and the rings to the dataset, and the commands that
remake it can be read off its chain.

**Code:** :func:`anri.calib.stages.write_pars`, :func:`anri.calib.stages.commands`.

What is left out
----------------

- **Wedge, chi and y0** stay as they were indexed. A trend of the residuals against :math:`\omega` or dty points to
  them.
- **Detector distortion** is not fitted: the pixels and peaks are corrected by ImageD11's spatial correction first.
- **Parallax in the rings**: each pixel's :math:`2\theta` is seen from the lab origin. A large sample widens the
  rings, and the scan's fit, which models where each peak scatters from, then corrects what is left.
- **The map's layers**: the beam's centre line is sampled in one layer (a 2D map of columns).
- **Each peak's own precision**: one standard deviation per component and round for every peak, not a weight per
  peak from its intensity or width.
- **Deformed samples**: their geometry is refined with their strains, which this calibration does not do.
