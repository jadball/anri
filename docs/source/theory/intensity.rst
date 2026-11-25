.. _theory_intensity:

How bright a peak is
====================

The integrated intensity of one peak, from one map entry (a voxel's orientation population with density :math:`\rho`), in
one frame, is the product

.. math::
    :label: int-amp

    I = \rho\; |F(\vec{h})|^2\; L\; P\; w_{\text{beam}}\; T,

of the structure factor (:eq:`cry-F`), the Lorentz factor :math:`L`, the polarisation factor :math:`P` (with the
extinction factor folded in when a phase has one, :ref:`theory_extinction`), the share of the voxel the beam lights
:math:`w_{\text{beam}}`, and a transmission :math:`T`. The renderer then spreads :math:`I` over
pixels and frames (:doc:`peak_shape`).

**Definition:** the kinematic integrated intensity of a rotation scan; each factor is derived below.

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

**Cited:** [Kabsch1988]_. **Proof:** ``tests/theory/test_intensity.py::test_lorentz_vertical`` (the triple product is
:math:`-\sin 2\theta\sin\eta`), ``tests/theory/test_intensity.py::test_lorentz_code``.

**Code:** :func:`anri.diffract.factors.lorentz`.

Polarisation factor
-------------------

A synchrotron beam is polarised in the horizontal plane. With a degree of horizontal polarisation :math:`p` (1: fully
horizontal, 0: unpolarised),

.. math::
    :label: int-pol

    P = \tfrac{1}{2}(1 + p)\left(1 - (\hat{k}_{\text{out}} \cdot \hat{e}_h)^2\right)
      + \tfrac{1}{2}(1 - p)\left(1 - (\hat{k}_{\text{out}} \cdot \hat{e}_v)^2\right).

**Cited:** [Kabsch1988]_ [Kahn1982]_. **Proof:** ``tests/theory/test_intensity.py::test_polarisation_forms`` (the
unpolarised form and Kahn et al.'s), ``tests/theory/test_intensity.py::test_polarisation_code``.

For a beam along :math:`x` this is ImageD11's ``refinegrains.polarization`` with ``eta0 = 0``. For :math:`p = 0` it is
the familiar :math:`(1 + \cos^2 2\theta)/2`. It is Kabsch's vector form [Kabsch1988]_ (p. 922), whose field lies along
the polarisation plane's normal :math:`\vec{n} = \hat{e}_h` with probability :math:`1 - p_K`, so
:math:`p_K = (1 - p)/2`; and Kahn et al.'s
:math:`P = \tfrac{1}{2}(1 + \cos^2 2\theta) - \tfrac{1}{2}\mathcal{J}' \cos 2\rho \sin^2 2\theta` [Kahn1982]_ (eqs.
13-15, after Azaroff), with :math:`\mathcal{J}' = p` and :math:`\rho = \eta + 90°` measured from the horizontal.

**Code:** :func:`anri.diffract.factors.polarisation`, :func:`anri.diffract.factors.lorentz_polarisation`
(``tests/theory/test_intensity.py::test_lorentz_polarisation_code``).

.. _theory_beam_profile:

The beam's profile over a voxel
-------------------------------

The beam's profile across it, horizontally and vertically, is a flat top of width :math:`w` blurred by a Gaussian of
standard deviation :math:`\sigma` (a pure Gaussian when :math:`w = 0`), normalised to integrate to 1, so a wider beam spreads the
same flux:

.. math::
    :label: int-profile

    p(u) = \frac{1}{w}\left[\Phi\!\left(\frac{u + w/2}{\sigma}\right) - \Phi\!\left(\frac{u - w/2}{\sigma}\right)\right]
    = \frac{1}{w}\int_{-w/2}^{w/2} \frac{1}{\sigma}\varphi\!\left(\frac{u - c}{\sigma}\right) dc,

with :math:`\Phi` the standard normal cumulative distribution. A pencil beam is narrow both ways, a line beam wide one
way, a box beam wide both ways.

**Definition:** the family of beam profiles. **Proof:** ``tests/theory/test_intensity.py::test_profile`` (the two forms
agree, and the profile integrates to 1).

**Anri's choice:** this family covers the ID11 beams (Gaussian focused beams, slit-defined flat tops) with two numbers
per direction; the beam's width is measured per experiment.

**Anri's choice:** a flat top narrower than :math:`0.01\sigma` is taken as the Gaussian, where the difference quotient
of :math:`P_2` would lose digits. The flat top is the Gaussian averaged over its centres, :math:`p_w = p_0 +
(w^2/24)\,p_0'' + O(w^4)`, and :math:`\int|\varphi''| = 4\varphi(1)`, so the weight changes by at most
:math:`(w^2/24)\, 4\varphi(1)\, h/\sigma^2` to leading order: :math:`4.0 \times 10^{-6}\, h` at the switch.
**Proof:** ``tests/theory/test_intensity.py::test_flat_top_small_width``,
``tests/theory/test_intensity.py::test_flat_top_switch_code``.

A Gaussian beam is often given by its full width at half maximum:

.. math::
    :label: int-fwhm

    \sigma = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}.

**Proof:** ``tests/theory/test_intensity.py::test_fwhm``, ``tests/theory/test_intensity.py::test_fwhm_code``.

**Code:** :func:`anri.maths.utils.fwhm_to_sigma`, :func:`anri.maths.utils.sigma_to_fwhm`.

How much of a voxel the beam lights is :math:`p` integrated over the voxel. It needs the first and second
antiderivatives of the Gaussian cumulative distribution, both vanishing at :math:`-\infty`:

.. math::
    :label: int-ramp

    R_1(x) = \int_{-\infty}^x \Phi\!\left(\frac{t}{\sigma}\right) dt = x\,\Phi\!\left(\frac{x}{\sigma}\right) + \sigma\,\varphi\!\left(\frac{x}{\sigma}\right),
    \qquad
    R_2(x) = \int_{-\infty}^x R_1 = \frac{x^2 + \sigma^2}{2}\,\Phi\!\left(\frac{x}{\sigma}\right) + \frac{x\sigma}{2}\,\varphi\!\left(\frac{x}{\sigma}\right).

**Proof:** ``tests/theory/test_intensity.py::test_ramp_integrals``, ``tests/theory/test_intensity.py::test_ramp_integrals_code``.

**Code:** :func:`anri.maths.utils.ramp_integral`, :func:`anri.maths.utils.ramp_integral2`.

So the profile's cumulative distribution is :math:`P_1(u) = [R_1(u + w/2) - R_1(u - w/2)]/w` and its integral
:math:`P_2(u) = [R_2(u + w/2) - R_2(u - w/2)]/w` (for a Gaussian, :math:`\Phi(u/\sigma)` and :math:`R_1(u)`).

Horizontally, a square voxel of side :math:`a` turned by :math:`\omega`, centred :math:`u` across the beam, has a chord
along the beam that is a trapezoid :math:`t(y)` in the across-beam offset :math:`y`, with half-widths
:math:`A = \tfrac{a}{2}(|\cos\omega| + |\sin\omega|)` at its base and :math:`B = \tfrac{a}{2}\,\big||\cos\omega| - |\sin\omega|\big|`
at its top, plateau :math:`h = a^2/(A + B)` and area :math:`a^2`. The trapezoid is a sum of four ramps
:math:`\rho(y) = \max(y, 0)`, and a ramp convolved with the profile is :math:`P_2`, so

.. math::
    :label: int-chord

    W(u, \omega) = \int t(y)\, p(u + y)\, dy
    = \frac{h}{A - B}\Big[P_2(u + A) - P_2(u + B) - P_2(u - B) + P_2(u - A)\Big],
    \qquad
    W = h\,\big[P_1(u + A) - P_1(u - A)\big] \ \ (A = B).

**Proof:** ``tests/theory/test_intensity.py::test_chord_trapezoid`` (the chord's shape and area; the four ramps, region
by region), ``tests/theory/test_intensity.py::test_chord_convolution`` (a ramp convolved with the Gaussian is :math:`R_1`,
and averaged over the flat top :math:`P_2`), ``tests/theory/test_intensity.py::test_chord_weight_code``.

**Code:** :func:`anri.geom.beam.chord_weight`.

The profile is even, and so are the trapezoid and the cube's height, so

.. math::
    :label: int-even

    P_1(x) + P_1(-x) = 1, \qquad P_2(x) - P_2(-x) = x, \qquad W(-u, \omega) = W(u, \omega).

**Proof:** ``tests/theory/test_intensity.py::test_profile_even``.

**Anri's rule:** :math:`P_1` and :math:`P_2` are evaluated at :math:`-|x|`, :math:`W` at :math:`-|u|` and a cube's share at
:math:`-|\Delta z|`, each through a select whose two branches are both exact, so that every derivative is right at 0
(:math:`-|x|` itself has a second derivative of 0 there). Far out, :math:`P_1 \to 1` and :math:`P_2` grows like
:math:`x`, and :eq:`int-chord`'s four terms cancel to the tail's light: evaluated directly at :math:`u = 8\sigma`, they
lost every digit in float32 and two in float64.

The four-term difference also cancels near :math:`\omega = 0°, 90°, \ldots`, where :math:`A - B \to 0`. Since
:math:`\partial_t P_2(u \pm t) = \pm P_1(u \pm t)`, :eq:`int-chord` is the mean over :math:`t \in [B, A]` of the
profile's mass over :math:`[u - t, u + t]`, and an :math:`n`-point Gauss-Legendre mean (nodes :math:`x_i`, weights
:math:`w_i` on :math:`[-1, 1]`, midpoint :math:`m = (A + B)/2`, :math:`L = A - B`) is close to it. With
:math:`E(m, L) = \int_{-1}^1 f(m + Lx/2)\, d\mu(x)` its error (:math:`\mu` the uniform mean minus the rule),
:math:`M_k = \max|f^{(k)}|` and :math:`K_n = (n!)^4 / ((2n+1)((2n)!)^3)`:

.. math::
    :label: int-chord-mean

    W = \frac{h}{A - B}\int_B^A f(t)\, dt, \quad f(t) = P_1(u + t) - P_1(u - t); \qquad
    \Big|\partial_m^i \partial_L^j E\Big| \le K_n \sum_{k=0}^{j} \binom{2n}{k} \frac{j!}{(j-k)!}\, 2^{k-j}\,
    L^{2n-k} M_{2n+i+j-k}, \quad M_k \le \frac{2 \max|\varphi^{(k-1)}|}{\sigma^k},

the error itself at :math:`i = j = 0`, :math:`K_n L^{2n} M_{2n}`. A Gauss rule integrates the Hermite interpolant of a
function at its nodes exactly, and the function minus it is its :math:`2n`-th derivative over :math:`(2n)!` times
:math:`\prod (x - x_i)^2`, of one sign, whose mean is that of the monic Legendre polynomial squared;
:math:`\partial_m^i \partial_L^j E` is the same functional applied to :math:`(x/2)^j f^{(i+j)}(m + Lx/2)`, whose
:math:`2n`-th derivative Leibniz's rule gives; and a flat top is an average of Gaussians, so its derivatives are
bounded by the Gaussian's. For :math:`n = 6` the error is at most :math:`4.37 \times 10^{-13}\,(L/\sigma)^{12}` of
:math:`h`.

**Proof:** ``tests/theory/test_intensity.py::test_chord_mean`` (the integral; the rule's error constant from the monic
Legendre polynomial; Leibniz's rule for the derivatives; :math:`\max|\varphi^{(k)}|` at a root of
:math:`\mathrm{He}_{k+1}`), ``tests/theory/test_intensity.py::test_chord_weight_near_box_code``.

**Anri's rule:** below :math:`L = 2.5\sigma` (float32) or :math:`0.5\sigma` (float64) the weight is the 6-point mean;
above, the four-term form, whose rounding of :math:`P_2` (tens of operations, values up to :math:`A + \sigma`) is
amplified by :math:`1/L`, and by a further :math:`1/L` for each omega derivative. The switch balances the second
derivatives (the beam fold, :eq:`int-fold`, takes two): at it the mean's error and its first two :math:`L`-derivatives
are at most :math:`2.6 \times 10^{-8}`, :math:`1.7 \times 10^{-7}/\sigma` and :math:`1.1 \times 10^{-6}/\sigma^2` of
:math:`h` in float32, :math:`1.1 \times 10^{-16}`, :math:`2.8 \times 10^{-15}/\sigma` and
:math:`6.6 \times 10^{-14}/\sigma^2` in float64, at or below the four-term form's rounding there. The rectangle's form
:math:`h\,[P_1(u + A) - P_1(u - A)]` is :math:`f(m + L/2)`: wrong at first order in :math:`L`, and its omega
derivative at zeroth order (:math:`\partial_\omega A \approx \pm a/2` near :math:`90°`, where
:math:`\partial_\omega W \to 0`).

**Measured:** against :eq:`int-chord` at 40 digits, for :math:`a = 0.7`, :math:`\sigma = 0.45`, Gaussian and flat-top
(:math:`w = 1.5`) profiles, :math:`u` up to :math:`8\sigma` and :math:`\omega` from :math:`10^{-7}` to 30° off 90°. Within
:math:`4\sigma`: relative errors of :math:`W` below :math:`6 \times 10^{-15}` (float64) and :math:`7 \times 10^{-7}`
(float32), its first and second omega derivatives within :math:`2 \times 10^{-12}` and :math:`3 \times 10^{-12}`
(float64), :math:`5 \times 10^{-4}` and :math:`4 \times 10^{-4}` (float32) of their largest value. At
:math:`6`-:math:`8\sigma`, where the ramp integrals themselves cancel at large :math:`-x`: :math:`2 \times 10^{-11}` and
:math:`3 \times 10^{-5}` for :math:`W`, up to :math:`6 \times 10^{-9}` and :math:`7 \times 10^{-5}` for the derivatives.
The previous rectangle switch (at :math:`A - B < 10^{-6} a`, :math:`3 \times 10^{-3} a` in float32) had errors of
:math:`6 \times 10^{-7}` (float64) and :math:`2 \times 10^{-3}` (float32) in :math:`W` near the beam, and omega
derivatives wrong by :math:`10^2` times their scale.

Vertically, a column voxel (2D map) takes the whole profile, which integrates to 1. A cube (3D map) of side :math:`a`,
its centre :math:`\Delta z` above the beam's centre line, takes its height's share. A beam tilted by :math:`\alpha` out of
the horizontal plane travels :math:`1/\cos\alpha` further through a voxel, and its vertical profile is measured
vertically:

.. math::
    :label: int-beam-weight

    w_{\text{beam}} = \frac{W(u, \omega - \psi)}{\cos\alpha} \times
    \begin{cases} 1 & \text{column} \\ P_{1,v}(\Delta z + a/2) - P_{1,v}(\Delta z - a/2) & \text{cube} \end{cases},

with :math:`\psi` the beam's azimuth and :math:`P_{1,v}` the vertical profile's cumulative distribution, its width and
:math:`\sigma` divided by :math:`\cos\alpha`.

**Definition:** the beam's weight of a voxel. For a tilted beam through a cube it treats the horizontal and vertical
directions separately, which is exact only for :math:`\alpha = 0`. Checked against the code, for horizontal and tilted
beams, Gaussian and flat-top, by ``tests/theory/test_intensity.py::test_beam_weight_code``. The offsets: :math:`u =
\vec{p}\cdot\hat{e}_h`, and :math:`\Delta z` above the centre line, which has risen :math:`\tan\alpha` per unit of
horizontal distance along the beam (``tests/theory/test_intensity.py::test_beam_offsets``,
``tests/theory/test_intensity.py::test_beam_offsets_code``).

**Code:** :func:`anri.geom.beam.beam_weight`, :func:`anri.geom.beam.beam_offsets`.

The beam's weight within a peak
-------------------------------

A voxel off the axis moves across the beam as omega turns, so within a peak's omega profile (often narrower than a
frame) the beam's weight changes. The renderer expands :math:`\log w_{\text{beam}}` to second order in omega about the
centroid, :math:`q(\omega) = \exp(\ell + \beta d + \tfrac{1}{2}\gamma d^2)` with :math:`d = \omega - \mu_\omega` (by automatic
differentiation, any profile, along the scan's path: at each omega the dty the scan had there, :eq:`int-dty-motion`), and
multiplies the peak's Gaussian by it:

.. math::
    :label: int-fold

    \int N(\omega; \mu_\omega, \Sigma_{\omega\omega})\, q(\omega)\, d\omega
    = e^{\ell + \beta^2 \Sigma'_{\omega\omega}/2} \sqrt{\frac{\Sigma'_{\omega\omega}}{\Sigma_{\omega\omega}}},
    \qquad
    \Sigma'_{\omega\omega} = \frac{1}{1/\Sigma_{\omega\omega} - \gamma},
    \qquad
    \mu'_\omega = \mu_\omega + \Sigma'_{\omega\omega}\beta.

Slow and fast keep their regression on omega (slopes :math:`\kappa_s`, :math:`\kappa_f`), so
:math:`\Sigma'_{s\omega} = \kappa_s \Sigma'_{\omega\omega}`, :math:`\Sigma'_{ss} = \Sigma_{ss} - \kappa_s^2(\Sigma_{\omega\omega} - \Sigma'_{\omega\omega})`
(likewise for fast, and :math:`\Sigma'_{sf} = \Sigma_{sf} - \kappa_s\kappa_f(\Sigma_{\omega\omega} - \Sigma'_{\omega\omega})`), and
the centroid moves by :math:`\kappa_s (\mu'_\omega - \mu_\omega)` on the detector.

**Proof:** ``tests/theory/test_intensity.py::test_beam_fold`` (the Gaussian integral, the new mean and variance, the
conditional kept), ``tests/theory/test_intensity.py::test_beam_fold_code`` (the code's fold against :eq:`int-fold`, from its
own :math:`\ell, \beta, \gamma`).

**Code:** :func:`anri.fwd.render.render_peaks`.

It is exact for a log-quadratic weight: a Gaussian beam, a voxel small compared with it, and motion across the beam linear
over the peak. The renderer clamps :math:`\gamma \le 0` (the fold never widens a peak: at a square voxel's kinks the point
curvature can come out large and positive) and :math:`|\beta| \le 3/\sqrt{\Sigma'_{\omega\omega}}`.

Each frame :math:`f` then keeps the residual of the true weight against the expansion, at the mean omega
:math:`\bar\omega_f` of the peak's mass in it and at the dty the scan had there, :math:`y(\bar\omega_f)`
(:eq:`int-dty-motion`):

.. math::
    :label: int-frame-weight

    I_f = \rho\,|F|^2 L P\; e^{\ell + \beta^2\Sigma'_{\omega\omega}/2}\sqrt{\Sigma'_{\omega\omega}/\Sigma_{\omega\omega}}\;
    \frac{w_{\text{beam}}(\bar\omega_f, y(\bar\omega_f))}{q(\bar\omega_f)}\; T_f\; P'(\text{frame } f),

with :math:`P'` the folded Gaussian's mass in the frame, before it is spread over pixels, and :math:`T_f` the transmission at the frame's omega. This
is :eq:`int-amp` with its beam weight and transmission taken per frame: the per-peak amplitude :math:`\rho|F|^2LP`
holds neither, so the beam's weight enters once, through the fold and the residual.

**Definition:** Anri's per-frame weight. It is exact when :math:`w_{\text{beam}}/q` is constant over each frame's share of
the peak (a weight log-quadratic along the scan's path), and right to first order in the residual's variation within a
frame otherwise.

**Code:** :func:`anri.fwd.render.render_peaks`.

A frame integrates over time, and in a fly scan dty moves while omega turns. Both move uniformly in time, so within a
frame dty is linear in omega: with frame :math:`f`'s omega and dty readings :math:`\omega_f, y_f` taken at its centre
and :math:`r_f` the rate of dty with omega,

.. math::
    :label: int-dty-motion

    \omega = \omega_f + \Omega t, \quad y = y_f + V t
    \;\Rightarrow\;
    y(\omega) = y_f + r_f\,(\omega - \omega_f), \quad r_f = V/\Omega,
    \qquad
    \int g\big(\omega(t), y(t)\big)\, dt = \frac{1}{\Omega}\int g\big(\omega, y(\omega)\big)\, d\omega,

so the light a peak puts in a frame is :math:`\int_{\text{frame}} N(\omega)\, w_{\text{beam}}(\omega, y(\omega))\, d\omega`: the
beam's weight taken along the path, where the voxel is at the moment the peak's light arrives. :eq:`int-fold` and
:eq:`int-frame-weight` compute this integral, exact for a weight log-quadratic along the path. It is not the profile
averaged over the frame's whole motion :math:`\Delta = r_f\,\delta\omega` (a box, adding :math:`\Delta^2/12` to the
profile's variance): that is the special case of a peak whose light is spread evenly over the frame. A peak narrower
than a frame is lit where the voxel is when the peak crosses, at :math:`y(\bar\omega_f)`, not over the whole motion.

**Proof:** ``tests/theory/test_intensity.py::test_dty_path`` (the change of variables; the even-light case is the box
average), ``tests/theory/test_intensity.py::test_dty_motion`` (the box's mean and :math:`\Delta^2/12`),
``tests/theory/test_intensity.py::test_dty_path_code`` (:func:`anri.fwd.render.make_row`'s path within each frame).

The scan records one dty per frame, so the rate comes from the readings: central differences in omega,

.. math::
    :label: int-dty-rate

    r_f = \frac{h_-^2\, (y_{f+1} - y_f) + h_+^2\, (y_f - y_{f-1})}{h_+ h_- (h_+ + h_-)},
    \qquad h_\pm = |\omega_{f\pm1} - \omega_f|,

one-sided at the row's ends: exact for readings linear (and, inside the row, quadratic) in omega, and zero where dty
is constant over a row (a step scan; the differences are then exactly 0), which then gives the weights it gave before.

**Proof:** ``tests/theory/test_intensity.py::test_dty_rate`` (exact on a quadratic), ``tests/theory/test_intensity.py::test_dty_path_code``.

**Code:** :func:`anri.fwd.render.make_row` ("dty_rate_sorted"), through ``_dty_at`` and ``_frame_weights``.

**Anri's choice:** the readings are taken as each frame's centre, as omega's are, and the motion as uniform within a
frame; a scan whose dty steps between frames, or whose readings are a frame's start, is not modelled. Ask how a
dataset's dty was recorded.

.. _theory_absorption:

Absorption
----------

The incoming beam reaches the voxel through a path :math:`L_{\text{in}}` of sample, and the diffracted ray leaves it
through :math:`L_{\text{out}}` (Beer-Lambert):

.. math::
    :label: int-absorption

    T = \exp\left(-\mu\,(L_{\text{in}} + L_{\text{out}})\right),
    \qquad
    L = \frac{D(v, \phi)}{\max(\cos\epsilon,\ 0.05)},

with :math:`\mu` from :func:`anri.crystal.structure.attenuation`, and each path :math:`L` from the voxel's in-plane
distance :math:`D(v, \phi)` to the sample's edge and the ray's elevation :math:`\epsilon` (below).

**Definition:** Beer-Lambert attenuation along the two paths, through a prism. Checked against the code by
``tests/theory/test_intensity.py::test_path_code``.

**Code:** :func:`anri.geom.paths.transmission`.

The paths come from a map of the fraction of each voxel the sample fills, in the slice, taken to be the same above and
below (a prism). Then the path from a voxel to the sample's edge along a direction depends only on the voxel and the
direction's angle :math:`\phi` in the slice plane: :math:`D(v, \phi)`, computed once for a set of angles by resampling
the map along each direction and summing it backwards. A ray leaving the plane at an elevation :math:`\epsilon` travels
:math:`D/\cos\epsilon`. One table serves the incoming beam (looking back towards the source) and every diffracted ray.

The backward sum is the trapezoid rule, exact for the map's linear interpolant whenever the ray's samples fall on the
map's nodes (``tests/theory/test_intensity.py::test_edge_trapezoid``,
``tests/theory/test_intensity.py::test_edge_distances_code``); the two directions are frm-stack inverted
(``tests/theory/test_intensity.py::test_ray_directions_code``).

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
tomography, which also holds what the map does not (pores, unindexed phases). A voxel's coefficient is
:math:`\sum w_i \mu_i / \sum w_i` over its entries, 0 without any
(``tests/theory/test_intensity.py::test_attenuation_map_code``).

``tests/theory/test_intensity.py::test_add_absorption_code`` checks the optical depths of a two-phase, two-layer map.

**Code:** :func:`anri.fwd.render.add_absorption`, :func:`anri.geom.paths.attenuation_map`,
:func:`anri.fwd.render.render_row`.

A transmission per frame (e.g. an attenuator) can be given as well.

**Code:** :func:`anri.fwd.render.make_row`.

.. _theory_extinction:

Extinction
----------

In a perfect crystal the diffracted beam is diffracted back into the incident direction, and the incident beam is
weakened by what the domains it has already crossed diffracted: strong reflections come out weaker than the
kinematical :math:`|F|^2` predicts, in the limit only as strong as :math:`|F|`. The integrated intensity is the
kinematical one times an extinction factor :math:`y \le 1` [Zachariasen1967]_. Anri follows Becker and Coppens'
treatment of Darwin's mosaic crystal [BeckerCoppens1974]_: perfect spherical domains of radius :math:`r`, misaligned by a
Gaussian or Lorentzian distribution of width parameter :math:`g`, make up a spherical crystal of radius :math:`R`.

With the diffracting power per unit length ([BeckerCoppens1974]_, equation 5)

.. math::
    :label: int-ext-q

    Q = \left(\frac{r_e\,|F|\,K}{V}\right)^2 \frac{\lambda^3}{\sin 2\theta},

**Cited:** [BeckerCoppens1974]_ (equation 5).

where :math:`r_e` is the classical electron radius, :math:`V` the cell volume and :math:`K = 1` for the field normal to
the scattering plane, :math:`\cos 2\theta` in it, and :math:`\bar{a} = 3 r \sin 2\theta / (2\lambda)`, the primary
(within a domain) and secondary (between domains) extinction parameters are (their equations 35, 40b, 41b and 42)

.. math::
    :label: int-ext-x

    x = \tfrac{2}{3}\, Q\, \bar{a}\, \tfrac{3r}{2}, \qquad
    X = \tfrac{2}{3}\, Q\, \alpha\, \tfrac{3R}{2}, \qquad
    \alpha = \frac{\bar{a}}{\sqrt{1 + \bar{a}^2 / 2g^2}} \;\text{(Gaussian)}, \quad
    \frac{\bar{a}}{1 + 2\bar{a}/3g} \;\text{(Lorentzian)},

**Cited:** [BeckerCoppens1974]_ (equations 35, 40b, 41b, 42).

and each factor has the form of their equation 37,

.. math::
    :label: int-ext-y

    y_p = \left(1 + 2x + \frac{A x^2}{1 + B x}\right)^{-1/2}, \qquad y = y_p\, y_s,

**Cited:** [BeckerCoppens1974]_ (equations 37, 38, 43, 43a). **Proof:** ``tests/theory/test_intensity.py::test_extinction_limits``
(:math:`y = 1` at :math:`x = 0`, :math:`y` falls with :math:`x`, and :math:`y^2 x \to B/(A + kB)`, :math:`k` the factor on :math:`x`: a perfect crystal's intensity grows as
:math:`|F|`).

with :math:`A, B` functions of :math:`\cos 2\theta` fitted to their numerical solutions of the Darwin transfer
equations (equation 38 for :math:`y_p`; 43 and 43a for :math:`y_s`, in :math:`X`, with :math:`2.12 X` for :math:`2X`
for a Gaussian distribution). With :math:`c = \cos 2\theta`, the coefficients as Anri uses them (the factor on :math:`x` first):

.. math::
    :label: int-ext-ab

    y_p:\ 2,\ 0.20 + 0.45c,\ 0.22 - 0.12(0.5 - c)^2; \quad
    y_s^{\text{G}}:\ 2.12,\ 0.58 + 0.48c + 0.24c^2,\ 0.02 - 0.025c; \quad
    y_s^{\text{L}}:\ 2,\ 0.025 + 0.285c,\ \begin{cases} 0.15 - 0.2(0.75 - c)^2 & c > 0 \\ -0.45c & c \le 0 \end{cases}.

**Cited:** [BeckerCoppens1974]_ (equations 38, 43, 43a); ``tests/reference/test_tables.py::TestExtinction`` checks that
they reproduce the paper's tables. **Proof:** ``tests/theory/test_intensity.py::test_extinction_code``.

The two field components are extinguished separately (their equations 45 and 46). If a
fraction :math:`w` of the incident intensity has its field normal to the scattering plane, so that
:math:`P = w + (1 - w)\cos^2 2\theta`, the polarisation factor becomes

.. math::
    :label: int-ext-pol

    P\,y = w\; y(x, X) + (1 - w)\cos^2 2\theta\; y(x\cos^2 2\theta,\, X\cos^2 2\theta),
    \qquad
    w = \tfrac{1}{2}(1 + p)(\hat{e}_h \cdot \hat{n})^2 + \tfrac{1}{2}(1 - p)(\hat{e}_v \cdot \hat{n})^2,

with :math:`\hat{n}` the scattering plane's normal.

**Cited:** [BeckerCoppens1974]_ (equations 45, 46). **Proof:** ``tests/theory/test_intensity.py::test_extinction_polarisation``
(this :math:`w` makes :math:`w + (1 - w)\cos^2 2\theta` equal :eq:`int-pol`),
``tests/theory/test_intensity.py::test_extinction_polarisation_code`` (no extinction gives :eq:`int-pol`).

With absorption, the mean path through the crystal :math:`3R/2` becomes the absorption-weighted mean path
:math:`\bar{T}_\mu` (their equations 52 and 53b), which is computed for a sphere by quadrature; :math:`x` is unchanged.
Becker and Coppens find this first-order correction good for :math:`\mu R` up to about 1 at small Bragg angles (their
equation 54). The quadrature's mean path is divided by the same quadrature's unweighted mean, not :math:`3R/2` itself,
so the ratio is exactly 1 without absorption; checked with :eq:`int-ext-q` and :eq:`int-ext-x` by
``tests/theory/test_intensity.py::test_extinction_parameters_code``.

**Measured:** for :math:`\mu R \le 1` the 12-point rule's ratio is within :math:`7 \times 10^{-4}` of a 64-point rule's
(the paths' square roots at the sphere's surface limit it).

**Code:** :func:`anri.diffract.factors.extinction_parameters`, :func:`anri.diffract.factors.extinction`,
:func:`anri.diffract.factors.polarisation_extinction`; per phase in the renderer through the entries' "extinction"
(:func:`anri.fwd.render.render_row`).

**Anri's choice:** one set of parameters (:math:`r`, :math:`R`, :math:`g`, the distribution) per phase, and none unless
given: in deformed or fine-grained samples extinction is negligible, but a perfect calibrant crystal (an undeformed Si
cube) is the textbook case. :math:`x` and :math:`X` need the absolute :math:`|F|^2` (electrons per cell), so they are
computed per reflection where the structure is known; the renderer's :math:`|F|^2` may be relative. The smooth
:math:`\theta` dependence of :math:`A` and :math:`B` is Becker and Coppens' own fit, within about 4% of their tables for
:math:`\sin\theta \le 0.5`, and 12% at :math:`\sin\theta = 0.9`, :math:`X = 10`.

Monitor normalisation
---------------------

The beam's flux drifts. When a monitor counter is given, every frame of the data is divided by its monitor reading (over
a reference value), as ImageD11's ``DataSet.set_monitor``.

**Code:** :func:`anri.io.imaged11.read_monitor`, :func:`anri.io.imaged11.stream_sparse`.

What is left out
----------------

- **Multiple scattering**, and extinction beyond spherical domains and crystals: other shapes and anisotropic extinction
  (:ref:`theory_extinction`).
- **Absorption in 3D**: the sample is a prism of its slice (of each layer, in the renderer); paths through the layers
  above and below are not followed. A sample built from a map is full wherever the map has entries, and empty elsewhere.
- **Absorption in the global orientation fit** of the filter stage, which has no voxel positions
  (:ref:`theory_global_fit`).
- **The detector's efficiency** versus angle of incidence and energy, and its gain: absorbed in the relative scale
  (a thick sensor's depth is in :doc:`detector`).
- **A tilted beam through a cube** (:eq:`int-beam-weight`): horizontal and vertical treated separately.
