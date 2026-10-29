.. _theory_peak_shape:

How wide a peak is
==================

A peak is not a point. The beam has a spread of wavelengths and directions, the voxel has a spread of orientations,
the detector blurs, and every frame integrates over its whole omega step. This chapter gives each peak a Gaussian in
:math:`(s, f, \omega)` and spreads its intensity (:eq:`int-amp`) over the pixels and frames it covers.

Covariance by propagation
-------------------------

The centroid :math:`\vec{c} = (s_c, f_c, \omega)` is a smooth function of the inputs that spread: the wavelength :math:`\lambda`,
the beam's tilts :math:`(k_y, k_z)` across it, and a small rotation :math:`\vec{\delta}` of the lattice. Each is taken as an
independent Gaussian about its nominal value, and pushed through the centroid to first order (the delta method)
[GUM2008]_:

.. math::
    :label: ps-propagate

    \matr{\Sigma} = \matr{J}\, \matr{\Sigma}_{\text{in}}\, \matr{J}^T
    = \sum_k \sigma_k^2\, \vec{J}_{:,k}\, \vec{J}_{:,k}^T,
    \qquad
    \matr{J} = \frac{\partial \vec{c}}{\partial (\lambda, k_y, k_z, \vec{\delta})}.

With a diagonal :math:`\matr{\Sigma}_{\text{in}}` the sum needs one Jacobian column at a time, and JAX's forward-mode
differentiation computes each as one Jacobian-vector product of the centroid function [JAX2018]_: no derivative is
written by hand, so the covariance always matches the centroid it comes from.

**Code:** :func:`anri.fwd.peak.make_propagator`, :func:`anri.fwd.peak.propagate_cov`, :func:`anri.fwd.peak.get_cov_in`.

The inputs:

- **Wavelength spread** :math:`\sigma_\lambda` moves a peak across the ring (radially), the only term that grows with
  :math:`2\theta`: on the detector at :math:`\eta = 45°` the slow-fast covariance is as large as the variances, so the peak is a
  tilted streak and the off-diagonal terms are kept.
- **Divergence** :math:`\sigma_{k_y}`, :math:`\sigma_{k_z}` (radians): tilting the incoming beam is, for the Ewald condition, a
  rotation of the crystal the other way, so horizontal divergence (about :math:`z`) widens every peak equally in omega, and
  vertical divergence (about :math:`y`) by :math:`|\cot\eta|` (:eq:`dif-rotation-motion`).
- **Orientation spread** of a map entry, :math:`\sigma_{\text{rot}}` (radians, isotropic):
  :math:`\text{UBI} \to \text{UBI}\,(\matr{I} + [\vec{\delta}]_\times)^T`, exact to first order. It widens peaks by
  :math:`\sigma/|\sin\eta|` in omega and along the ring in :math:`\eta`.

  **Code:** :func:`anri.maths.rotation.skew`.

Added to the propagated covariance: the detector's point spread :math:`\sigma_{\text{psf}}^2` (pixels²) on the slow and
fast variances, an optional extra omega variance, and small floors so that a peak with no broadening at all still
renders.

**Code:** :func:`anri.fwd.render.render_peaks`.

**Anri's choice:** first order is right for spreads up to a few degrees; beyond that a peak is a curved arc
(:ref:`theory_rotation_motion`) and a Gaussian is the wrong shape. AGENTS.md asks for exact rotations at large spreads;
the renderer does not do that yet.

Integrating a Gaussian over frames and pixels
---------------------------------------------

Omega turns continuously, so a frame records everything between its omega edges: a model must integrate each frame
over its omega range, never evaluate it at one omega. Likewise a pixel integrates over its area. The intensity in a
(frame, row, column) cell is the Gaussian's mass in that box. A correlated 3D Gaussian has no closed-form box mass,
so Anri conditions, one axis at a time:

.. math::
    :label: ps-factor

    P(\text{cell}) \approx P(\omega \in \text{frame})\;
    P(s \in \text{row} \mid \omega \in \text{frame})\;
    P(f \in \text{column} \mid s \in \text{row},\ \omega \in \text{frame}).

Each factor is a 1D Gaussian mass, taken with the mean and variance of the previous variables *within* their cell:
the moments of a truncated normal, for :math:`x \sim N(\mu, \sigma^2)` restricted to :math:`[l, u)`, :math:`a = (l - \mu)/\sigma`,
:math:`b = (u - \mu)/\sigma`, :math:`Z = \Phi(b) - \Phi(a)` [JohnsonKotz1994]_:

.. math::
    :label: ps-truncated

    \operatorname{E}[x] = \mu + \sigma\,\frac{\varphi(a) - \varphi(b)}{Z},
    \qquad
    \operatorname{Var}[x] = \sigma^2 \left[1 + \frac{a\varphi(a) - b\varphi(b)}{Z}
    - \left(\frac{\varphi(a) - \varphi(b)}{Z}\right)^2\right].

**Code:** :func:`anri.maths.utils.truncated_moments`, :func:`anri.maths.utils.bin_fractions`.

For slow given omega in a frame whose omega has truncated mean :math:`m_\omega` and variance :math:`v_\omega`, by the
Gaussian's regression and the law of total variance:

.. math::
    :label: ps-condition

    \mu_{s \mid \omega} = \mu_s + \frac{\Sigma_{s\omega}}{\Sigma_{\omega\omega}}(m_\omega - \mu_\omega),
    \qquad
    \sigma^2_{s \mid \omega} = \Sigma_{ss} - \frac{\Sigma_{s\omega}^2}{\Sigma_{\omega\omega}}
    + \left(\frac{\Sigma_{s\omega}}{\Sigma_{\omega\omega}}\right)^2 v_\omega,

and fast given (slow, omega) the same way, with omega integrated out given slow within the frame.

**Anri's choice:** this keeps the slow-fast and detector-omega correlations, and the true position of a peak narrower
than a frame or a pixel (most peaks are narrower than a frame: conditioning on the within-frame mean, not the frame
centre, matters). It is exact for a Gaussian along each axis separately, and an approximation for the joint box mass. In
the upper tail (:math:`a > 0`) the mass is computed as :math:`\Phi(-a) - \Phi(-b)`, not :math:`\Phi(b) - \Phi(a)`: in
single precision the difference of two numbers close to 1 left model pixels far from bright peaks wrong, even negative.

**Code:** :func:`anri.maths.utils.bin_fractions`.

Windows
-------

Each peak is evaluated over a fixed window of cells around its centroid, (frames, rows, columns), each odd. Each frame's
row and column window is centred on the peak's mean position *in that frame*, so a peak that moves across the detector
with omega stays inside its window. The fraction of the Gaussian inside the window is reported per peak as ``captured``,
so a window that is too small shows up. Peaks broad in omega can be given more frames (``max_frames``). Fixed window
sizes keep array shapes fixed, so the renderer compiles once per size.

**Code:** :func:`anri.fwd.render.render_peaks`, :func:`anri.fwd.render.window_origins`,
:func:`anri.fwd.render.render_row`.

Which pixels a peak touches is not differentiable (windows are placed from rounded centroids); the pixel values are,
with respect to the map. Fixed window origins (``origins``) keep values smooth in the parameters for a refiner.

What is left out
----------------

- **Non-Gaussian peaks.** Each source of spread is Gaussian and linearised. Real sub-grain structure gives curved,
  banana-shaped peaks (several map entries per voxel, each with its own spread, approximate them as a mixture).
- **Anisotropic orientation spread.** One width per map entry; AGENTS.md asks for a full 3 × 3 covariance in rotation.
- **Strain spread** within an entry (it would widen peaks in :math:`2\theta` as :math:`2\varepsilon\tan\theta`).
- **Thermal diffuse tails**, seen on measured Si peaks over 5-9 frames.
