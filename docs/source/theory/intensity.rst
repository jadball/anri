.. _theory_intensity:

How bright a peak is
====================

The integrated intensity of one peak, from one map entry (a voxel's orientation population with density :math:`\rho`), in
one frame, is the product

.. math::
    :label: int-amp

    I = \rho\; |F(\vec{h})|^2\; L\; P\; w_{\text{beam}}\; T,

of the structure factor (:eq:`cry-F`), the Lorentz factor :math:`L`, the polarisation factor :math:`P`, the share of the
voxel the beam lights :math:`w_{\text{beam}}`, and a transmission :math:`T`. The renderer then spreads :math:`I` over
pixels and frames (:doc:`peak_shape`).

**Code:** :func:`anri.fwd.render.render_peaks`.

The indexer uses the same factors in its system matrix (:doc:`indexing`). Densities are relative: there is no absolute
scale (flux, detector gain, exposure time).

Lorentz factor
--------------

A reflection crosses the Ewald sphere at a rate set by the geometry, so the time it spends diffracting, and with it
the integrated intensity of a rotation scan, varies from reflection to reflection. For rotation about an axis
:math:`\hat{n}`, the Lorentz factor is [Kabsch1988]_ (p. 922, after Zachariasen and [MilchMinor1974]_)

.. math::
    :label: int-lorentz

    L = \frac{1}{\left|\hat{n} \cdot (\hat{k}_{\text{in}} \times \hat{k}_{\text{out}})\right|}
    = \frac{1}{\sin 2\theta\, |\sin\eta|} \quad \text{(vertical axis)},

where the second form is ImageD11's ``refinegrains.lf`` [ImageD11]_. Kabsch writes the reciprocal as
:math:`|\vec{S} \cdot (\vec{u} \times \vec{S}_0)| / (|\vec{S}||\vec{S}_0|)`, with :math:`\vec{u}` the axis and
:math:`\vec{S}_0`, :math:`\vec{S}` the incoming and diffracted wave-vectors: the same triple product. :math:`\hat{n}` is
the omega axis tilted by wedge and chi. :math:`L` diverges as :math:`\sin\eta \to 0`: reflections near the rotation axis
pass slowly through the Ewald sphere, which is also why their peaks are wide in omega (:ref:`theory_rotation_motion`).

**Code:** :func:`anri.diffract.factors.lorentz`.

Polarisation factor
-------------------

A synchrotron beam is polarised in the horizontal plane. With a degree of horizontal polarisation :math:`p` (1: fully
horizontal, 0: unpolarised),

.. math::
    :label: int-pol

    P = \tfrac{1}{2}(1 + p)\left(1 - (\hat{k}_{\text{out}} \cdot \hat{e}_h)^2\right)
      + \tfrac{1}{2}(1 - p)\left(1 - (\hat{k}_{\text{out}} \cdot \hat{e}_v)^2\right).

For a beam along :math:`x` this is ImageD11's ``refinegrains.polarization`` with ``eta0 = 0``. For :math:`p = 0` it is
the familiar :math:`(1 + \cos^2 2\theta)/2`. It is Kabsch's vector form [Kabsch1988]_ (p. 922), whose field lies along
the polarisation plane's normal :math:`\vec{n} = \hat{e}_h` with probability :math:`1 - p_K`, so
:math:`p_K = (1 - p)/2`; and Kahn et al.'s
:math:`P = \tfrac{1}{2}(1 + \cos^2 2\theta) - \tfrac{1}{2}\mathcal{J}' \cos 2\rho \sin^2 2\theta` [Kahn1982]_ (eqs.
13-15, after Azaroff), with :math:`\mathcal{J}' = p` and :math:`\rho = \eta + 90°` measured from the horizontal.

**Code:** :func:`anri.diffract.factors.polarisation`, :func:`anri.diffract.factors.lorentz_polarisation`.

.. _theory_beam_profile:

The beam's profile over a voxel
-------------------------------

The beam's profile across it, horizontally and vertically, is a flat top of width :math:`w` blurred by a Gaussian of
standard deviation :math:`\sigma` (a pure Gaussian when :math:`w = 0`), normalised to integrate to 1, so a wider beam spreads the
same flux:

.. math::
    :label: int-profile

    p(u) = \frac{1}{w}\left[\Phi\!\left(\frac{u + w/2}{\sigma}\right) - \Phi\!\left(\frac{u - w/2}{\sigma}\right)\right],

with :math:`\Phi` the standard normal cumulative distribution. A pencil beam is narrow both ways, a line beam wide one
way, a box beam wide both ways.

**Anri's choice:** this family covers the ID11 beams (Gaussian focused beams, slit-defined flat tops) with two numbers
per direction; the beam's width is measured per experiment.

How much of a voxel the beam lights is :math:`p` integrated over the voxel. Horizontally, a square voxel of side :math:`a` turned
by :math:`\omega` has a chord along the beam that is a trapezoid in the across-beam coordinate :math:`u`, with half-widths
:math:`\tfrac{a}{2}(|\cos\omega| + |\sin\omega|)` at its base and :math:`\tfrac{a}{2}\,\big||\cos\omega| - |\sin\omega|\big|` at its
top, and area :math:`a^2`. The integral of :math:`p` against it is closed-form, from the first and second antiderivatives of the
Gaussian cumulative distribution:

.. math::
    :label: int-ramp

    \int \Phi\!\left(\frac{x}{\sigma}\right) dx = x\,\Phi\!\left(\frac{x}{\sigma}\right) + \sigma\,\varphi\!\left(\frac{x}{\sigma}\right),
    \qquad
    \iint \Phi\!\left(\frac{x}{\sigma}\right) = \frac{x^2 + \sigma^2}{2}\,\Phi\!\left(\frac{x}{\sigma}\right) + \frac{x\sigma}{2}\,\varphi\!\left(\frac{x}{\sigma}\right).

**Code:** :func:`anri.geom.beam.chord_weight`, :func:`anri.maths.utils.ramp_integral`,
:func:`anri.maths.utils.ramp_integral2`.

Vertically, a column voxel (2D map) takes the whole profile, which integrates to 1; a beam tilted by :math:`\alpha` out
of the horizontal plane travels :math:`1/\cos\alpha` further through it. A cube (3D map) integrates the vertical profile
over its height.

**Code:** :func:`anri.geom.beam.beam_weight`, :func:`anri.geom.beam.beam_offsets`.

The weight is evaluated at each frame's own dty and at the mean omega of the peak's intensity in that frame: in a fly
scan dty moves during the rotation, and a voxel off the axis moves across the beam as omega turns, so a peak spread over
many frames is lit differently in each.

**Code:** :func:`anri.fwd.render.render_peaks`.

.. _theory_absorption:

Absorption
----------

The incoming beam reaches the voxel through a path :math:`L_{\text{in}}` of sample, and the diffracted ray leaves it
through :math:`L_{\text{out}}` (Beer-Lambert):

.. math::
    :label: int-absorption

    T = \exp\left(-\mu\,(L_{\text{in}} + L_{\text{out}})\right),

with :math:`\mu` from :func:`anri.crystal.structure.attenuation`.

**Code:** :func:`anri.geom.paths.transmission`.

The paths come from a map of the fraction of each voxel the sample fills, in the slice, taken to be the same above and
below (a prism). Then the path from a voxel to the sample's edge along a direction depends only on the voxel and the
direction's angle :math:`\phi` in the slice plane: :math:`D(v, \phi)`, computed once for a set of angles by resampling
the map along each direction and summing it backwards. A ray leaving the plane at an elevation :math:`\epsilon` travels
:math:`D/\cos\epsilon`. One table serves the incoming beam (looking back towards the source) and every diffracted ray.

**Code:** :func:`anri.geom.paths.edge_distances`, :func:`anri.geom.paths.path_lengths`,
:func:`anri.geom.paths.ray_directions`.

**Anri's choice:** boundaries are resolved to about a voxel (bilinear resampling), :math:`\phi` to 1° by default, and a
nearly vertical ray is capped at :math:`1/0.05` times its in-plane path, since a real sample is not an endless prism.

The indexer applies :math:`T` per voxel and per predicted reflection (:doc:`indexing`), so the fitted occupancies are free
of absorption. Its sample is the fit's mask, of one phase.

The renderer uses the same paths, in optical depths: each map entry gets :math:`\int \mu\,\mathrm{d}l` to the
sample's edge in every in-plane direction, and each frame of a peak's window is scaled by
:math:`\exp(-(\tau_{\text{in}} + \tau_{\text{out}}))` at that frame's omega. Its sample is either the map itself, every
voxel holding entries full of material with its phases' :math:`\mu` weighted by their densities (so a multiphase map
absorbs phase by phase, and a 3D map gets one slice per layer), or a given map of :math:`\mu`, e.g. from an absorption
tomography, which also holds what the map does not (pores, unindexed phases).

**Code:** :func:`anri.fwd.render.add_absorption`, :func:`anri.geom.paths.attenuation_map`,
:func:`anri.fwd.render.render_row`.

A transmission per frame (e.g. an attenuator) can be given as well.

**Code:** :func:`anri.fwd.render.make_row`.

Monitor normalisation
---------------------

The beam's flux drifts. When a monitor counter is given, every frame of the data is divided by its monitor reading (over
a reference value), as ImageD11's ``DataSet.set_monitor``.

**Code:** :func:`anri.io.imaged11.read_monitor`, :func:`anri.io.imaged11.stream_sparse`.

What is left out
----------------

- **Extinction** (:doc:`crystal`), and multiple scattering.
- **Absorption in 3D**: the sample is a prism of its slice (of each layer, in the renderer); paths through the layers
  above and below are not followed. A sample built from a map is full wherever the map has entries, and empty elsewhere.
- **Absorption in the global orientation fit** of the filter stage, which has no voxel positions
  (:ref:`theory_global_fit`).
- **The detector's efficiency** versus angle of incidence and energy, and its gain: absorbed in the relative scale.
