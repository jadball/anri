.. _theory_diffraction:

Where a reflection diffracts
============================

This chapter takes one voxel, with its UBI and position, and one reflection :math:`\vec{h}`, and finds where its peak lands:
the omega at which it diffracts, the direction of the diffracted ray, and the detector pixel it hits. Together these
make the peak's centroid :math:`(s_c, f_c, \omega)` (and, for scanning 3DXRD, the dty at which it is seen).

The diffraction condition
-------------------------

Elastic scattering conserves the length of the wave-vector, and a crystal scatters only where the change of
wave-vector is a reciprocal-lattice vector (the Laue condition):

.. math::
    :label: dif-laue

    \vec{Q} = \vec{k}_{\text{out}} - \vec{k}_{\text{in}} = \vec{g},
    \qquad |\vec{k}_{\text{out}}| = |\vec{k}_{\text{in}}| = \frac{1}{\lambda}.

**Definition:** elastic, kinematic scattering.

**Code:** :func:`anri.diffract.scatter.k_to_q_lab`, :func:`anri.diffract.scatter.q_lab_to_k_out`.

Squaring :math:`\vec{k}_{\text{out}} = \vec{g} + \vec{k}_{\text{in}}` and using the equal lengths gives the Ewald
condition

.. math::
    :label: dif-ewald

    \vec{g} \cdot \vec{k}_{\text{in}} = -\frac{|\vec{g}|^2}{2}.

**Proof:** ``tests/theory/test_diffraction.py::test_ewald``.

The scattering angle follows from the length alone, Bragg's law in reciprocal form:

.. math::
    :label: dif-bragg

    \sin\theta = \frac{\lambda\, d^*}{2}.

**Proof:** ``tests/theory/test_diffraction.py::test_bragg``, ``tests/theory/test_diffraction.py::test_bragg_code``.

**Code:** :func:`anri.diffract.scatter.ds_to_tth`.

Solving for omega
-----------------

The sample turns about its omega axis, so :math:`\vec{g}` sweeps a cone and meets the Ewald condition at up to two omegas
per turn. Work in the frame that turns with omega, in which the rotation axis is :math:`\hat{z}` and the beam is
:math:`\tilde{\vec{k}} = (\matr{W}\matr{C})^T \vec{k}_{\text{in}}` (:eq:`frm-stack`). Split :math:`\vec{g}_s` (:eq:`frm-g`) into
parts along and across the axis, :math:`\vec{g}_\parallel = (\vec{g}_s \cdot \hat{z})\hat{z}` and
:math:`\vec{g}_\perp = \vec{g}_s - \vec{g}_\parallel`. Rotating by :math:`\omega` turns only :math:`\vec{g}_\perp`, so
:eq:`dif-ewald` becomes

.. math::
    :label: dif-omega-harmonic

    \alpha\cos\omega + \beta\sin\omega = \delta,
    \qquad
    \alpha = \tilde{\vec{k}} \cdot \vec{g}_\perp,\quad
    \beta = \tilde{\vec{k}} \cdot (\hat{z} \times \vec{g}_\perp),\quad
    \delta = -\frac{|\vec{g}_s|^2}{2} - \tilde{\vec{k}} \cdot \vec{g}_\parallel.

**Proof:** ``tests/theory/test_diffraction.py::test_omega_harmonic``.

With :math:`R = \sqrt{\alpha^2 + \beta^2}` and :math:`\phi = \operatorname{atan2}(\alpha, \beta)` this is
:math:`R\sin(\omega + \phi) = \delta`, with the two solutions

.. math::
    :label: dif-omega

    \omega_1 = \arcsin\frac{\delta}{R} - \phi, \qquad \omega_2 = -\arcsin\frac{\delta}{R} - \phi - \pi,

which exist when :math:`|\delta / R| \le 1` [MilchMinor1974]_ [BusingLevy1967]_. :math:`\omega_1` is branch
``etasign = +1`` and :math:`\omega_2` is ``-1`` (:doc:`conventions`).

**Cited:** [MilchMinor1974]_. **Proof:** ``tests/theory/test_diffraction.py::test_omega_solutions`` (both satisfy
:eq:`dif-omega-harmonic`), ``tests/theory/test_diffraction.py::test_omega_code``.

The name ``etasign`` is the sign of :math:`\eta`, for a beam along :math:`x` and an untilted axis. The left side of
:eq:`dif-omega-harmonic` has derivative :math:`R\cos(\omega + \phi)`, which is :math:`+R\sqrt{1 - \delta^2/R^2}` at
:math:`\omega_1` and its negative at :math:`\omega_2`. That derivative is
:math:`\tilde{\vec{k}} \cdot (\hat{z} \times \vec{g}) = \vec{g}_{\text{lab}} \cdot (\vec{k}_{\text{in}} \times \hat{z})`,
which at diffraction is :math:`\sin 2\theta \sin\eta / \lambda^2` (:eq:`dif-kout`). So :math:`\omega_1` diffracts with
:math:`\eta > 0` and :math:`\omega_2` with :math:`\eta < 0`. With tilts, the branch is the side of the plane of beam and
rotation axis on which :math:`\vec{g}_{\text{lab}}` lies (proof: ``tests/theory/test_diffraction.py::test_branch_sign``,
``tests/theory/test_diffraction.py::test_branch_code``).

**Code:** :func:`anri.diffract.scatter.omega_solns_core`, :func:`anri.diffract.scatter.omega_from_core`,
:func:`anri.diffract.scatter.omega_solns`.

**Anri's choice:** the arcsine's argument is clipped to :math:`\pm(1 - 10^{-6})` and :math:`R^2` floored at
:math:`10^{-7}`, so that the gradient stays finite at the edge of the Ewald sphere and along the rotation axis. The
price is at that edge: where :math:`|\delta/R| > 1 - 10^{-6}` the omega returned is off the solution by up to
:math:`\arccos(1 - 10^{-6}) = 0.081°` (proof: ``tests/theory/test_diffraction.py::test_clip_cost``), 1.6 frames of
0.05°. The edge is where the two branches meet, :math:`\sin\eta \to 0`, so the reflections it touches are those
``etacut`` leaves out; a method that keeps reflections near the axis must not use the clipped omega. The two
branches share everything but the last line: a vmap over :math:`\pm 1` gives both.

**Code:** :func:`anri.fwd.peak.hkl_to_k_omega`.

The diffracted ray, 2θ and η
----------------------------

At the solution, :math:`\vec{k}_{\text{out}} = \matr{W}\matr{C}\matr{R}(\omega)\vec{g}_s + \vec{k}_{\text{in}}` in the
lab.

**Code:** :func:`anri.fwd.peak.hkl_to_k_omega`.

Its angles are :math:`2\theta = \arccos(\hat{k}_{\text{out}} \cdot \hat{k})` and :math:`\eta` from :eq:`frm-eta`.
Conversely [Poulsen2017]_ (eq. 39),

.. math::
    :label: dif-kout

    \hat{k}_{\text{out}} = \cos 2\theta\, \hat{k} + \sin 2\theta \left(-\sin\eta\, \hat{e}_h + \cos\eta\, \hat{e}_v\right).

**Cited:** [Poulsen2017]_ (for a beam along :math:`x`). **Proof:** ``tests/theory/test_diffraction.py::test_kout`` (for
any beam it is a unit vector at :math:`2\theta` from :math:`\hat{k}` whose :eq:`frm-eta` is :math:`\eta`),
``tests/theory/test_diffraction.py::test_kout_code``.

**Code:** :func:`anri.diffract.scatter.k_out_to_tth_eta`, :func:`anri.diffract.scatter.tth_eta_to_k_out`.

Beam divergence enters here: an incoming ray tilted by small angles :math:`(k_y, k_z)` (radians) across the beam is
:math:`\vec{k}_{\text{in}} \propto \hat{k} + k_y\hat{e}_h + k_z\hat{e}_v`, normalised to :math:`1/\lambda`. The forward model
evaluates the centroid at :math:`k_y = k_z = 0` and uses the tilts only through their derivatives (:doc:`peak_shape`).

Where the voxel scatters from
-----------------------------

The ray starts at the voxel, not at the rotation axis, so its position matters (parallax). At the solution's omega,
and the frame's dty, the voxel centre is at :eq:`frm-stack`. Anri has two kinds of voxel:

- a **column** (2D maps): the voxel extends through the whole beam height and scatters at the height where the beam
  crosses it. For a beam tilted out of the horizontal plane that height rises with distance along the beam.
- a **cube** (3D maps): it scatters from its centre.

For a column, the beam (direction :math:`\hat{k}`, tilted by :math:`a` out of the horizontal plane, at azimuth
:math:`\psi`) is taken to cross the rotation axis at the voxel's own height, so the voxel at :math:`\vec{v}_{\text{lab}}`
of :eq:`frm-stack` scatters from

.. math::
    :label: dif-column

    \vec{o} = \vec{v}_{\text{lab}} + \hat{z}\, \tan a\, \left(x_{\text{lab}} \cos\psi + y_{\text{lab}} \sin\psi\right),

the beam's rise over the voxel's horizontal distance along it.

**Proof:** ``tests/theory/test_diffraction.py::test_column_height``, ``tests/theory/test_diffraction.py::test_column_code``.

**Code:** :func:`anri.geom.gonio.origin_lab`.

**Scanning 3DXRD.** A pencil beam lights a voxel only when dty brings it onto the beam's centre line. The beam's
horizontal line is :math:`y = (k_y/k_x)\,x` through the lab origin, and dty moves the sample along :math:`y`, so

.. math::
    :label: dif-dty

    \text{dty} = y_0 + \frac{k_y}{k_x}\, x_{\text{lab}} - y_{\text{lab}},
    \qquad (x_{\text{lab}}, y_{\text{lab}}, \cdot) = \matr{W}\matr{C}\matr{R}(\omega)\, \vec{v}_s.

For a beam along :math:`x` and no tilts that is :math:`\text{dty} = y_0 - (x_s \sin\omega + y_s \cos\omega)`: each voxel
traces a sinusoid in (omega, dty), the sinogram.

**Proof:** ``tests/theory/test_diffraction.py::test_dty``, ``tests/theory/test_diffraction.py::test_dty_code``,
``tests/theory/test_diffraction.py::test_dty_and_origin_code`` (with :eq:`dif-column`'s origin at that dty).

**Code:** :func:`anri.geom.gonio.find_dty_for_beam_xy`, :func:`anri.geom.gonio.dty_and_origin_lab`,
:func:`anri.fwd.scan.get_centroid_scan`.

In a box beam the whole sample is lit at dty = :math:`y_0`.

**Code:** :func:`anri.fwd.box.get_centroid_box`.

The renderer instead puts the voxel at each frame's real dty and weights it by how much of it the beam lights there
(:ref:`theory_beam_profile`).

On the detector
---------------

The ray :math:`\vec{o} + t\,\hat{k}_{\text{out}}` from origin :math:`\vec{o}` meets the detector plane, spanned from pixel (0, 0)
at :math:`\vec{d}_0` by one-pixel steps :math:`\vec{s}` (slow) and :math:`\vec{f}` (fast), at

.. math::
    :label: dif-raytrace

    s_c = \frac{\hat{k}_{\text{out}} \cdot \left(\vec{f} \times (\vec{d}_0 - \vec{o})\right)}{\hat{k}_{\text{out}} \cdot (\vec{s} \times \vec{f})},
    \qquad
    f_c = \frac{\hat{k}_{\text{out}} \cdot \left((\vec{d}_0 - \vec{o}) \times \vec{s}\right)}{\hat{k}_{\text{out}} \cdot (\vec{s} \times \vec{f})}

(Cramer's rule) [Thomas1992]_. :math:`\vec{s}`, :math:`\vec{f}` and :math:`\vec{d}_0` come from the detector's ImageD11
parameters [FABLEgeom]_. The solution does not depend on the length of :math:`\hat{k}_{\text{out}}`.

**Cited:** [Thomas1992]_. **Proof:** ``tests/theory/test_diffraction.py::test_raytrace`` (it solves
:math:`\vec{o} + t\hat{k}_{\text{out}} = \vec{d}_0 + s_c\vec{s} + f_c\vec{f}`),
``tests/theory/test_diffraction.py::test_raytrace_code``.

**Code:** :func:`anri.geom.detector.raytrace_to_det`, :func:`anri.geom.detector.detector_basis_vectors_lab`.

Chaining :eq:`frm-g`, :eq:`dif-omega`, the origin and :eq:`dif-raytrace` gives the forward model's one centroid
function, which the renderer, the box-beam and the scanning models all call.

**Code:** :func:`anri.fwd.peak.get_centroid`.

.. _theory_rotation_motion:

How a small rotation moves a peak
---------------------------------

Sub-grain orientation shifts are what Anri measures, so it matters how they show in the data. A small lattice rotation
:math:`\vec{\delta} = (\delta_x, \delta_y, \delta_z)` (a rotation vector in the lab frame at the diffracting omega, beam
along :math:`x`, axis along :math:`z`) keeps :math:`|\vec{g}|`, so it leaves :math:`2\theta` unchanged and moves the peak by

.. math::
    :label: dif-rotation-motion

    \Delta\omega = -\delta_z - \delta_y \cot\eta,
    \qquad
    \Delta\eta = \delta_x - \delta_y \frac{\tan\theta}{\sin\eta},
    \qquad
    \Delta 2\theta = 0,

to first order: it is the exact derivative of :eq:`dif-omega` and :eq:`frm-eta`. With :math:`\vec{g}_0 =
\vec{k}_{\text{out}} - \vec{k}_{\text{in}}` the diffracting vector, the rotated vector at a further turn :math:`\omega` is
:math:`\matr{R}_z(\omega)(\matr{I} + [\vec{\delta}]_\times)\vec{g}_0`; the implicit function theorem on the Ewald
condition :eq:`dif-ewald` gives :math:`\Delta\omega`, and the change of :math:`\vec{k}_{\text{out}}` it implies gives
:math:`\Delta\eta` and :math:`\Delta 2\theta`.

**Proof:** ``tests/theory/test_diffraction.py::test_rotation_motion``,
``tests/theory/test_diffraction.py::test_rotation_motion_code`` (the derivatives of
:func:`anri.diffract.scatter.omega_solns` and :func:`anri.diffract.scatter.k_out_to_tth_eta`).

Read term by term:

- about the rotation axis (:math:`z`), a peak moves only in omega, not on the detector;
- about the beam (:math:`x`), only in :math:`\eta`, along the Debye-Scherrer ring, by the same angle;
- about :math:`y`, in both, magnified near :math:`\eta = 0°` and :math:`180°`, where omega and :math:`\eta` couple into a tilted streak.

So orientation moves intensity *along* the ring and over omega, and strain moves it *across* the ring
(:math:`\Delta 2\theta = -2\varepsilon\tan\theta` for a lattice strain :math:`\varepsilon` along :math:`\vec{g}`, :doc:`conventions`). An isotropic orientation spread :math:`\sigma` widens a peak by
:math:`\sigma/|\sin\eta|` in omega and :math:`\sigma\sqrt{1 + \tan^2\theta/\sin^2\eta}` in :math:`\eta`: peaks near the rotation axis
(:math:`|\sin\eta|` small) are smeared most, which is why the indexer leaves out reflections with
:math:`|\sin\eta| \le` ``etacut``. A rotation about :math:`\vec{g}` itself moves its peak not at all: each reflection sees only
part of a rotation, and only many reflections together determine it (proof of these three statements:
``tests/theory/test_diffraction.py::test_rotation_motion_consequences``).

What is left out
----------------

- **Refraction** at the sample surface, and any deviation from kinematic (single-scattering) diffraction.
- **The voxel's own extent** in the centroid: a voxel is a point at its centre here. Its size enters the intensity
  through the beam profile (:doc:`intensity`) and, in the indexer, through the dty rows it is spread over.
- **Detector distortion** here: the centroid is on the ideal detector; the renderer maps it to raw pixels (:doc:`detector`).
