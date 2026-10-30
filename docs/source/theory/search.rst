.. _theory_search:

Which orientations to try
=========================

The occupancy fit (:doc:`likelihood`) needs a finite list of orientations. Anri covers the fundamental zone with a
grid, keeps the grid orientations whose predicted spots land on intensity, and then keeps those a global,
intensity-aware fit says are needed.

Grids over the fundamental zone
-------------------------------

Because of the Laue group (:eq:`cry-laue`), only one fundamental zone of orientation space needs covering.

**Cubic** (24 rotations): a regular grid of Rodrigues vectors :math:`\vec{r} = \tan(\theta/2)\,\hat{n}`, inside the cubic
fundamental zone :math:`|r_i| \le \tan(\pi/8)`, :math:`|r_1| + |r_2| + |r_3| \le 1` [Frank1988]_: a cube truncated by
eight planes, with corners at :math:`(\sqrt{2} - 1, \sqrt{2} - 1, 3 - 2\sqrt{2})` and their permutations and sign changes
([Rosca2014]_, section 4.4). The grid is coarsest at the
origin, so the largest misorientation from any orientation to its nearest grid point is that of half a cell diagonal
there:

.. math::
    :label: srch-cubic-delta

    \Delta_{\max} = 2\arctan\left(\frac{\sqrt{3}}{2}\tan\frac{s}{2}\right)

for a grid step :math:`s`.

**Code:** :func:`anri.crystal.orientation.orientation_grid`, :func:`anri.crystal.orientation.cubic_grid_misorientation`.

For cubic crystals it needs about half the orientations of a cubochoric grid for the same worst case.

**Other Laue groups**: the cubochoric grid of all rotations [Rosca2014]_, a regular grid in a cube of edge
:math:`\pi^{2/3}` mapped onto the homochoric ball, preserving volume (section 2.1 and appendix A there), and from the
ball to the quaternions by the homochoric vector :math:`\vec{h} = \hat{n}\,[\tfrac{3}{4}(\omega - \sin\omega)]^{1/3}`
([Rosca2014]_, section 3.3; [Rowenhorst2015]_, equation 13), so its points are spread evenly over orientation space.
:math:`N = 131.97 / (s - 0.037)` points per semi-edge give a mean nearest-neighbour disorientation of about :math:`s`:
the fit of [SinghDeGraef2016]_ (equation 9), which holds for every rotation group (their table 3). The grid is reduced
to the fundamental zone around the identity, plus a shell half a step wide outside it (without the shell, points just
outside the zone leave gaps along its boundary, as their symmetric images are not grid points). The worst case is larger
than the mean spacing.

**Anri's choice:** measured for every Laue group from 1.5 to 12°, it is at most :math:`1.15\,s`, and Anri uses
:math:`1.2\,s`. The same grid and fit are in orix's ``cubochoric_sampling`` [Johnstone2020]_.

**Code:** :func:`anri.crystal.orientation.cubochoric_quaternions`.

Completeness
------------

An orientation's completeness is the fraction of its predicted reflections that land on intensity [Lauridsen2001]_
[Schmidt2014]_. Anri measures it against a lit map: the row-summed (ring, :math:`\eta`, omega) histogram, thresholded at
``lit`` times its median non-empty bin. A prediction counts if any lit bin lies within its own tolerance box; a
summed-area table makes each box a constant-time lookup.

**Code:** :func:`anri.index.search.completeness`, :func:`anri.index.data.lit_table`, :func:`anri.index.stages.lit_map`.

Predictions on bins less than half measured (:eq:`idx-weight`'s coverage :math:`c < 0.5`) are not counted: they could not
be lit. Nor are those with :math:`|\sin\eta| \le` ``etacut``.

Tolerances from the grid
------------------------

The truth is up to :math:`\Delta_{\max}` from its nearest grid point, so each prediction's box must hold the spot of any
orientation that close. From :eq:`dif-rotation-motion`, a rotation by :math:`\Delta` in any direction moves a reflection by
at most :math:`\Delta\sqrt{1 + \tan^2\theta/\sin^2\eta}` in :math:`\eta` and :math:`\Delta/|\sin\eta|` in omega. Anri uses the slightly
larger bounds

.. math::
    :label: srch-tolerance

    t_\eta = \frac{\Delta_{\max}}{\cos\theta}\left(1 + \tan\theta\,|\cot\eta|\right) + \frac{h_r}{\sin 2\theta\,\cos 2\theta},
    \qquad
    t_\omega = \frac{\Delta_{\max}}{\cos\theta\,|\sin\eta|} + \frac{\Delta\omega_{\text{frame}}}{2},

with the ring's measured half-width :math:`h_r` (:func:`anri.index.data.ring_widths`), taken as a displacement on the
detector across the ring, and half a frame. Checked numerically against the exact Ewald solution (random orientations,
:math:`|\sin\eta| > 0.3`): the largest spot movement reaches 0.999 of the bound and never exceeds it. Peak widths are
not added: a broad peak lights a broad region of the lit map.

**Code:** :func:`anri.index.search.match_tolerances`, :func:`anri.index.search.predict`.

Choosing the grid step
----------------------

A coarser grid has larger tolerances, so a wrong orientation is more likely to look complete by chance. Most grid
orientations are wrong, so the median completeness over a random sample of the grid is its *chance completeness*. Anri
takes the coarsest step (3, 2.5, 2, 1.5, 1°) whose chance completeness is at most ``max_chance`` (0.5).

**Code:** :func:`anri.index.search.choose_grid`, :func:`anri.index.search.grid_chance`.

**Anri's choice:** the threshold is set by the data's own background of chance matches, not by a fixed completeness,
since a crowded lit map (many grains, broad peaks) raises every orientation's completeness.

From completeness to the fit
----------------------------

Completeness ignores intensity, so a decoy whose spots all fall on other grains' peaks can be complete. The search
therefore only removes the hopeless: every grid orientation above chance completeness goes on to a global orientation
fit, which keeps those that measurably improve the likelihood of the row-summed data (:ref:`theory_global_fit`). The
older route, keeping orientations above halfway from chance to the grid's best completeness, remains as
``prune="completeness"``.

**Code:** :func:`anri.index.search.prune`, :func:`anri.index.stages.stage_filter`.

What is left out
----------------

- **Orientations off the grid** are represented by their neighbours on it: a single orientation between grid points
  shares its occupancy among them. :doc:`localsearch` refines them, and :doc:`populations` merges them back.
- **Strain** in the search: predictions use the reference lattice.
