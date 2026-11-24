.. _theory_peak_shape:

How wide a peak is
==================

A peak is not a point. The beam has a spread of wavelengths and directions, the voxel has a spread of orientations,
and every frame integrates over its whole omega step. This chapter gives each peak a Gaussian in :math:`(s, f, \omega)`
and spreads its intensity (:eq:`int-amp`) over the frames and pixels it covers, inside a fixed window. What the
detector adds between the Gaussian and the pixel values (point spread, sensor depth, distortion, the pixels
themselves) is in :doc:`detector`.

Covariance by propagation
-------------------------

The centroid :math:`\vec{c} = (s_c, f_c, \omega)` is a smooth function of the inputs that spread: the wavelength :math:`\lambda`,
the beam's tilts :math:`(k_y, k_z)` across it, and a small rotation :math:`\vec{\delta}` of the lattice. Each is taken as a
Gaussian about its nominal value, and pushed through the centroid to first order (the delta method) [GUM2008]_:

.. math::
    :label: ps-propagate

    \matr{\Sigma} = \matr{J}\, \matr{\Sigma}_{\text{in}}\, \matr{J}^T
    = \sum_k \sigma_k^2\, \vec{J}_{:,k}\, \vec{J}_{:,k}^T \quad (\matr{\Sigma}_{\text{in}} \text{ diagonal}),
    \qquad
    \matr{J} = \frac{\partial \vec{c}}{\partial (\lambda, k_y, k_z, \vec{\delta})}.

**Cited:** [GUM2008]_. **Proof:** ``tests/theory/test_peak_shape.py::test_propagate_diagonal`` (the sum of columns),
``tests/theory/test_peak_shape.py::test_propagate_code`` (the propagator against :math:`\matr{J}\matr{\Sigma}\matr{J}^T`,
diagonal and full).

With a diagonal :math:`\matr{\Sigma}_{\text{in}}` the sum needs one Jacobian column at a time, and JAX's forward-mode
differentiation computes each as one Jacobian-vector product of the centroid function [JAX2018]_: no derivative is
written by hand, so the covariance always matches the centroid it comes from. A full :math:`\matr{\Sigma}_{\text{in}}`
takes the whole Jacobian, and the result is linear in it.

**Code:** :func:`anri.fwd.peak.make_propagator`, :func:`anri.fwd.peak.propagate_cov`, :func:`anri.fwd.peak.get_cov_in`.

The tutorial :doc:`../tutorials/forward_model_covariance` propagates a position error, a wavelength spread and the
divergence through the centroid.

The inputs, independent of each other:

- **Wavelength spread** :math:`\sigma_\lambda` moves a peak across the ring (radially): on the detector at
  :math:`\eta = 45°` the slow-fast covariance is as large as the variances, so the peak is a tilted streak and the
  off-diagonal terms are kept.
- **Divergence** :math:`\sigma_{k_y}`, :math:`\sigma_{k_z}` (radians): tilts of the incoming beam along the horizontal and
  vertical across it (:func:`anri.geom.beam.beam_basis`).
- **Orientation spread** of a map entry: a covariance :math:`\matr{S}` (radians², [3, 3], sample frame) of a small rotation
  vector :math:`\vec{\delta}`, or one standard deviation :math:`\sigma_{\text{rot}}` for :math:`\matr{S} = \sigma_{\text{rot}}^2 \matr{I}`.
  The lattice turns as

  .. math::
      :label: ps-rotation

      \matr{U}\matr{B} \to \matr{R}\,\matr{U}\matr{B}, \quad
      \text{UBI} \to \text{UBI}\,\matr{R}^T, \qquad
      \matr{R} = \exp([\vec{\delta}]_\times) = \matr{I} + [\vec{\delta}]_\times + O(\delta^2),

  **Proof:** ``tests/theory/test_peak_shape.py::test_rotation_first_order`` (Rodrigues' formula differentiated at 0;
  :math:`\matr{R}^{-1} = \matr{R}^T`; and :func:`anri.maths.rotation.skew`).

  which is exact to first order, all that :eq:`ps-propagate` uses. Its contribution is :math:`\matr{J}_\delta \matr{S} \matr{J}_\delta^T`,
  linear in :math:`\matr{S}`: an anisotropic spread (a dominant rotation axis) gives each reflection its own streak.

**Code:** :func:`anri.maths.rotation.skew`, :func:`anri.fwd.render.render_peaks` (an entry's "sig_rot", [N] or [N, 3, 3]).

For a horizontal beam and an untilted axis (wedge = chi = 0), differentiating the Ewald condition
:math:`2\vec{k}_{\text{in}} \cdot \vec{g} + |\vec{g}|^2 = 0` (:eq:`dif-ewald`) gives each input's omega shift, and so the
omega variance in closed form:

.. math::
    :label: ps-omega

    \Delta\omega = -\delta_z - \delta_y\cot\eta
    - \frac{\Delta\lambda}{\lambda}\,\frac{\tan\theta}{\sin\eta} + k_y - k_z\cot\eta,
    \qquad
    \sigma_\omega^2 = \vec{j}\,\matr{R}_\omega \matr{S} \matr{R}_\omega^T \vec{j}^T
    + \left(\frac{\sigma_\lambda}{\lambda}\right)^2 \frac{\tan^2\theta}{\sin^2\eta}
    + \sigma_{k_y}^2 + \sigma_{k_z}^2 \cot^2\eta,

with :math:`\vec{\delta}` in the lab frame at the diffracting omega, :math:`\vec{j} = (0, -\cot\eta, -1)`, and
:math:`\matr{R}_\omega` the rotation by omega from sample to lab. An isotropic spread gives
:math:`\sigma^2(1 + \cot^2\eta) = \sigma^2/\sin^2\eta`.

**Proof:** ``tests/theory/test_peak_shape.py::test_omega_motion`` (each shift from the linearised Ewald condition),
``tests/theory/test_peak_shape.py::test_omega_variance_code`` (the renderer's omega variance for every spread together,
anisotropic :math:`\matr{S}` included, against :eq:`ps-omega` at six peaks).

Added to the propagated covariance:

.. math::
    :label: ps-total

    \matr{\Sigma}_{(s, f, \omega)} = \matr{J}\matr{\Sigma}_{\text{in}}\matr{J}^T
    + \operatorname{diag}\left(\sigma_{\text{psf}}^2 + 10^{-4},\ \sigma_{\text{psf}}^2 + 10^{-4},\ \sigma_{\omega,\text{extra}}^2 + 10^{-8}\right),

the detector's point spread (pixels², :doc:`detector`), an optional extra omega variance (degrees²), and small floors
so that a peak with no broadening at all still renders.

**Definition:** what the renderer adds to the propagated covariance; the point spread's place is derived in
:ref:`theory_detector_psf`.

**Code:** :func:`anri.fwd.render.render_peaks`.

**Anri's choice:** first order is right for the shape of spreads up to a few degrees; beyond that a peak is a curved
arc (:ref:`theory_rotation_motion`) and a Gaussian is the wrong shape. AGENTS.md asks for exact rotations at large
spreads; the renderer does not do that yet. For a peak's *mean* first order is not enough at the precision a strain
refiner needs: a straight spot of spread :math:`\sigma_\eta` along a curved ring reads as a strain of about
:math:`-\sigma_\eta^2/2`, :math:`1.5 \times 10^{-4}` at 1° (:eq:`ref-ring`, :eq:`ref-spread-mean`).

Frames
------

Omega turns continuously, so a frame records everything between its omega edges. A scan gives one omega per frame (its
centre); the edges are the midpoints between neighbouring centres once sorted, and the first and last frames are as wide
as their neighbours:

.. math::
    :label: ps-edges

    e_0 = \omega_0 - \tfrac{1}{2}(\omega_1 - \omega_0),\quad
    e_j = \tfrac{1}{2}(\omega_{j-1} + \omega_j),\quad
    e_n = \omega_{n-1} + \tfrac{1}{2}(\omega_{n-1} - \omega_{n-2}).

**Definition:** the frames tile the scan with no gaps (no dead time between frames). Checked against the code by
``tests/theory/test_peak_shape.py::test_make_row_edges``.

**Code:** :func:`anri.fwd.render.make_row`.

Integrating a Gaussian over frames and pixels
---------------------------------------------

**Anri's rule:** a model value is the peak's mass in a (frame, row, column) cell, never its density at the cell's
centre. A frame integrates over its whole omega range and a pixel over its area; what this does to a peak's moments,
and to the information a narrow peak carries, is derived in :ref:`theory_detector_pixels`, and holds for frames in
omega in units of the frame width.

A correlated 3D Gaussian has no closed-form box mass, so Anri conditions, one axis at a time:

.. math::
    :label: ps-factor

    P(\text{cell}) \approx P(\omega \in \text{frame})\;
    P(s \in \text{row} \mid \omega \in \text{frame})\;
    P(f \in \text{column} \mid s \in \text{row},\ \omega \in \text{frame}),

each factor a 1D Gaussian mass with the mean and variance of that conditional.

**Definition:** Anri's approximation of the box mass; what it gets exactly, and its error, are derived below. The code
computes exactly this: ``tests/theory/test_peak_shape.py::test_window_cells_code`` (every cell of a correlated peak's
window, against :eq:`ps-factor` with :eq:`ps-truncated`, :eq:`ps-condition` and :eq:`ps-condition-f` evaluated at 30
digits, to 1000 ulps absolute).

The moments come from the truncated normal: for :math:`x \sim N(\mu, \sigma^2)` restricted to :math:`[l, u)`,
:math:`a = (l - \mu)/\sigma`, :math:`b = (u - \mu)/\sigma`, :math:`Z = \Phi(b) - \Phi(a)`,

.. math::
    :label: ps-truncated

    \operatorname{E}[x] = \mu + \sigma\,\frac{\varphi(a) - \varphi(b)}{Z},
    \qquad
    \operatorname{Var}[x] = \sigma^2 \left[1 + \frac{a\varphi(a) - b\varphi(b)}{Z}
    - \left(\frac{\varphi(a) - \varphi(b)}{Z}\right)^2\right].

**Cited:** [JohnsonKotz1994]_. **Proof:** ``tests/theory/test_peak_shape.py::test_truncated_moments`` (the integrals of
:math:`z\varphi` and :math:`z^2\varphi`), ``tests/theory/test_peak_shape.py::test_upper_tail_mass``
(:math:`\Phi(-a) - \Phi(-b) = \Phi(b) - \Phi(a)`), ``tests/theory/test_peak_shape.py::test_truncated_moments_code``
(both tails, bins narrow and wide, each to a tolerance derived from its cancellation).

**Code:** :func:`anri.maths.utils.truncated_moments`, :func:`anri.maths.utils.bin_fractions`.

In the upper tail (:math:`a > 0`) the mass is computed as :math:`\Phi(-a) - \Phi(-b)`: the difference of two numbers close
to 1 loses everything in single precision, and left model pixels far from bright peaks wrong, even negative.

Slow given omega in a frame whose omega has truncated mean :math:`m_\omega` and variance :math:`v_\omega`: slow given
omega is Gaussian, with regression slope :math:`\kappa = \Sigma_{s\omega}/\Sigma_{\omega\omega}`, so by the law of total
variance

.. math::
    :label: ps-condition

    \mu_{s \mid F} = \mu_s + \kappa\,(m_\omega - \mu_\omega),
    \qquad
    \sigma^2_{s \mid F} = \Sigma_{ss} - \frac{\Sigma_{s\omega}^2}{\Sigma_{\omega\omega}} + \kappa^2 v_\omega.

**Proof:** ``tests/theory/test_peak_shape.py::test_condition_slow`` (the Schur complement and the total variance).

Fast given (slow in a row, omega in a frame): fast regresses on (slow, omega) with
:math:`(g_s, g_\omega) = (\Sigma_{fs}, \Sigma_{f\omega})\,\matr{\Sigma}_{(s,\omega)}^{-1}` and residual variance
:math:`V_f = \Sigma_{ff} - g_s\Sigma_{fs} - g_\omega\Sigma_{f\omega}`. Taking omega within the frame as
:math:`N(m_\omega, v_\omega)`, slow and omega are jointly Gaussian there, omega regresses on slow with
:math:`k = \kappa v_\omega / \sigma^2_{s \mid F}`, and with the row's truncated slow moments :math:`m_s, v_s`:

.. math::
    :label: ps-condition-f

    \mu_{f \mid R, F} = \mu_f + g_s (m_s - \mu_s) + g_\omega \left(m_\omega + k (m_s - \mu_{s \mid F}) - \mu_\omega\right),
    \qquad
    \sigma^2_{f \mid R, F} = V_f + g_\omega^2\, v_\omega (1 - k\kappa) + (g_s + g_\omega k)^2\, v_s.

**Proof:** ``tests/theory/test_peak_shape.py::test_condition_fast`` (the regressions, the residual variance as a Schur
complement of the 3 × 3 covariance, and the within-frame conditionals);
``tests/theory/test_peak_shape.py::test_window_cells_code``.

**Code:** :func:`anri.fwd.render.render_peaks`.

What the factorisation gets exactly, and what it does not:

- :math:`P(\omega \in \text{frame})` is exact.
- The slow factor has exactly the first two moments of slow given omega in the frame (:eq:`ps-condition`). The
  conditional itself is :math:`\tau Z + \kappa\,\omega_F` (:math:`\tau^2 = \Sigma_{ss} - \Sigma_{s\omega}^2/\Sigma_{\omega\omega}`,
  :math:`\omega_F` omega truncated to the frame), not a Gaussian: its cumulants beyond the second are
  :math:`\kappa^n \kappa_n(\omega_F)`.
- The fast factor's moments are exact if :math:`\omega_F` is Gaussian, and slow within the row too; otherwise they carry
  the same kind of error, through :math:`g_\omega` and :math:`g_s`.
- So the factorisation is exact when every conditioning slope is 0 (no correlation), or when the cells are wide compared
  with the peak (no truncation: :math:`v_\omega = \Sigma_{\omega\omega}`). It keeps the slow-fast and detector-omega
  correlations, and the position of a peak narrower than a frame within that frame (conditioning on the within-frame mean,
  not the frame centre).

The error of a factor, to leading order in the conditional's standardised cumulants
:math:`\gamma_n = \kappa^n \kappa_n(\omega_F)/\sigma^n_{s \mid F}` (the Edgeworth expansion about the Gaussian of the
same mean and variance), over a cell of standardised edges :math:`a, b`:

.. math::
    :label: ps-edgeworth

    P_{\text{true}} - P_{\text{Gauss}} \approx
    -\frac{\gamma_3}{6}\Big[\varphi(z)\,\mathrm{He}_2(z)\Big]_a^b
    -\frac{\gamma_4}{24}\Big[\varphi(z)\,\mathrm{He}_3(z)\Big]_a^b,
    \qquad
    \gamma_4 = -\tfrac{6}{5}\, r^2 \text{ for a uniform } \omega_F,\quad r = \frac{\kappa^2 v_\omega}{\sigma^2_{s \mid F}}.

**Proof:** ``tests/theory/test_peak_shape.py::test_cumulants_add`` (cumulants of :math:`\tau Z + \kappa\omega_F`; a
uniform's excess kurtosis :math:`-6/5`), ``tests/theory/test_peak_shape.py::test_edgeworth_step`` (each term's integral
over the cell), ``tests/theory/test_peak_shape.py::test_uniform_limit``.

:math:`r` is the share of the conditional variance that comes from where omega is within the frame. It is small unless a
peak moves across several pixels within one frame while being sharp on the detector (large :math:`\kappa`, small
:math:`\tau`): a peak near :math:`\eta = 0` with a large orientation spread. The limit :math:`r = 1` (a streak uniform
over its row) is the worst case: rendered as the Gaussian of the same variance, the row of its own width gets
:math:`2\Phi(\sqrt{3}) - 1 = 0.917` of it, not all.

**Anri's choice:** the factorisation, for its fixed cost per cell and its exact handling of the narrow-peak case that
matters most (peaks narrower than a frame). A peak streaking across pixels within a frame should be given a wider frame
window and checked with :eq:`ps-edgeworth`.

Windows
-------

Each peak is evaluated over a fixed window of cells around its centroid, (frames, rows, columns), each odd. Each frame's
row and column window is centred on the peak's mean position *in that frame* (:eq:`ps-condition`), so a peak that moves
across the detector with omega stays inside its window. Peaks broad in omega can be given more frames (``max_frames``).
Fixed window sizes keep array shapes fixed, so the renderer compiles once per size.

**Code:** :func:`anri.fwd.render.render_peaks`, :func:`anri.fwd.render.window_origins`,
:func:`anri.fwd.render.render_row`.

A window holds only part of a peak. What it holds is reported per peak as ``captured``, the sum of its cells' masses
over the frames inside the scan (``in_range`` is the share of the omega profile inside the scan's omega range). The
rest is beyond the window's faces. A window moves from frame to frame, so its faces are defined on the cells the
factorisation (:eq:`ps-factor`) cuts the Gaussian into, each cell outside the window belonging to exactly one face. With
the window's frames :math:`j_o \ldots j_o + n_o - 1` (sorted omega order) and frame :math:`j`'s rows
:math:`i_j \ldots i_j + n_s - 1` and columns :math:`c_j \ldots c_j + n_f - 1`:

- faces 0 and 1: the scan's frames before the window's first frame, and after its last (every row and column);
- faces 2 and 3: in each window frame, the rows before its row window, and after it (every column);
- faces 4 and 5: in each window frame, and each row of its row window, the columns before its column window, and
  after it.

The cells beyond a face telescope (:eq:`det-pixel`), so the light beyond each is a sum of the conditionals' tails. With
the frame masses :math:`P_j` and the conditionals :math:`N(\mu_{s|j}, \sigma^2_{s|j})` (:eq:`ps-condition`), the rows'
slow masses :math:`P_{i|j}` and :math:`N(\mu_{f|ij}, \sigma^2_{f|ij})` (:eq:`ps-condition-f`), summed over the window's
frames inside the scan, and :math:`t(x) = (x - \mu_\omega)/\sigma_\omega`:

.. math::
    :label: ps-faces

    \begin{aligned}
    L_0 &= \Phi(t(e_{j_o})) - \Phi(t(e_0)), &
    L_1 &= \Phi(t(e_n)) - \Phi(t(e_{j_o + n_o})), \\
    L_2 &= \sum_j P_j\, \Phi\!\left(\frac{i_j - \tfrac{1}{2} - \mu_{s|j}}{\sigma_{s|j}}\right), &
    L_3 &= \sum_j P_j\, \Phi\!\left(\frac{\mu_{s|j} - i_j - n_s + \tfrac{1}{2}}{\sigma_{s|j}}\right), \\
    L_4 &= \sum_j P_j \sum_i P_{i|j}\, \Phi\!\left(\frac{c_j - \tfrac{1}{2} - \mu_{f|ij}}{\sigma_{f|ij}}\right), &
    L_5 &= \sum_j P_j \sum_i P_{i|j}\, \Phi\!\left(\frac{\mu_{f|ij} - c_j - n_f + \tfrac{1}{2}}{\sigma_{f|ij}}\right),
    \end{aligned}

with the frame indices :math:`j_o` and :math:`j_o + n_o` clipped to the scan's :math:`[0, n]` (a window reaching past
the scan's end has no frames beyond that face).

**Definition:** the light of the factorised Gaussian's cells beyond each face, summed by the telescoping of
:eq:`det-pixel`; proved, with the sum below, by ``tests/theory/test_peak_shape.py::test_faces_partition``.

They make up the rest of the scan's light:

.. math::
    :label: ps-face-sum

    \text{captured} + \sum_{k=0}^{5} L_k = \Phi(t(e_n)) - \Phi(t(e_0)) = \text{in\_range}.

**Proof:** ``tests/theory/test_peak_shape.py::test_faces_partition`` (each face's cells telescope to its tail, and with
the window's cells they add up to the scan's light), ``tests/theory/test_peak_shape.py::test_faces_code`` (each
:math:`L_k` of a correlated peak, against :eq:`ps-faces` at 30 digits, and :eq:`ps-face-sum`),
``tests/theory/test_peak_shape.py::test_captured_code``.

The faces are exact for the factorised Gaussian: they are the light the renderer would put in the cells beyond them.
They are in the units of ``captured`` (fractions of the peak's Gaussian, after the beam's fold :eq:`int-fold`, before
the beam's and the frames' weights). A mixture rendered over one window (a thick sensor's slices,
:eq:`det-sensor-mix`; a mixed point spread, :eq:`det-psf-mix`, whose components share every window origin, since the
origins depend only on the centroid and the omega profile) has the faces of its components summed by their weights:
the faces are linear in the light, so no shape is assumed.

How far beyond a face the light reaches follows from the same tails. Let :math:`T_g(m)` be the light of one Gaussian
:math:`g` of a face beyond the face moved :math:`m` cells out (frames of equal width). Its cell masses are
log-concave (:eq:`mom-logconcave`), so the ratios of its tails fall and

.. math::
    :label: ps-face-reach

    \sum_g T_g(m) \le L\,\rho^m, \qquad \rho = \max_g \frac{T_g(1)}{T_g(0)}, \qquad L = \sum_g T_g(0):

:math:`T(m + 1)/T(m)` is the mean of the cell ratios :math:`b_{k+1}/b_k` over the cells :math:`k \ge m`, weighted by
:math:`b_k`; going from :math:`m` to :math:`m - 1` adds the cell :math:`m - 1`, whose ratio is at least every later one,
and adding a value at least the mean raises the mean. A face holds one Gaussian per frame (faces 2, 3), per frame and
row (faces 4, 5), per mixture component, and :math:`\sum_g T_g(0)\rho_g^m \le (\sum_g T_g(0)) \max_g \rho_g^m`.

**Proof:** ``tests/theory/test_peak_shape.py::test_face_reach`` (the mean raised by a value above it; the ratios of a
binned Gaussian's tails falling; the mixture's bound).

**Code:** :func:`anri.fwd.render.render_row` (stats "face_light", the :math:`L_k`, and "face_decay", the :math:`\rho`,
[n_peaks, 6] in face order), through ``_faces``.

A spot that is not a mixture of Gaussians over one window has no such closed form, and a log-concave extrapolation from
the cells inside the window does not bound a mixture (:eq:`mom-mixture`). For any spot of variance :math:`\sigma^2` along the face's normal, the light beyond a face at distance :math:`t` from its mean
is bounded (Cantelli's inequality):

.. math::
    :label: ps-cantelli

    P(x - \mu \ge t) \le \frac{\sigma^2}{\sigma^2 + t^2}.

**Proof:** ``tests/theory/test_peak_shape.py::test_cantelli`` (Markov's inequality on :math:`(x - \mu + u)^2`, at its best
:math:`u = \sigma^2/t`).

At :math:`t = 3.5\sigma` this is 0.0755, against :math:`\Phi(-3.5) = 0.000233` for a Gaussian: a window sized for a
Gaussian is not safe for a spot of unknown shape, whose missing light is known only to this bound.

**Anri's rule:** a window's missing light must be known, per face (:eq:`ps-faces`, :eq:`ps-face-reach`). A model pixel is complete only where every peak
that can put light there covers it with its window; next to a face, the model lacks at most that face's light, and a
comparison with the data there must account for it or leave the cell out. A missing light that is not bounded biases
whatever is fitted towards the parameters that move light into the window. The measurement side, which cells a fit
compares, is in :doc:`likelihood`.

**Anri's rule:** derivatives are taken with the window origins fixed. Which cells a peak covers jumps when its rounded
centroid moves by a pixel or a frame; the value of a cell then jumps by the light of the face that is swapped, so the
model is piecewise smooth in the parameters, and its derivative is the derivative of the piece with fixed origins. For a
refiner, ``origins`` (:func:`anri.fwd.render.window_origins`) fixes them over an iteration.

**Code:** :func:`anri.fwd.render.window_origins`.

What is left out
----------------

- **Non-Gaussian spots of one entry.** Each map entry's spot is a Gaussian, every source of spread linearised. The
  banana-shaped peak seen on the detector is not one entry's spot: it is the sum, in each frame, of the spots of every
  voxel the beam lights along its ray, over the omega range the frame integrates (as the sample turns, the voxels on
  the ray change). Their slightly different orientations put their spots at different places along the arc. The
  renderer forms that sum exactly; what it leaves out is only each entry's own shape, which is wrong for an entry
  whose own spread is large enough to curve (:ref:`theory_rotation_motion`, :eq:`ref-ring`).
- **Exact rotations** for spreads beyond a few degrees (:eq:`ps-rotation` is first order).
- **Strain spread** within an entry (it would widen peaks in :math:`2\theta` as :math:`2\varepsilon\tan\theta`).
- **Dead time** between frames: the frames tile the scan (:eq:`ps-edges`).
- **The factorisation's higher cumulants** (:eq:`ps-edgeworth`), for peaks streaking across pixels within one frame.
- **Thermal diffuse tails**, seen on measured Si peaks over 5-9 frames.
