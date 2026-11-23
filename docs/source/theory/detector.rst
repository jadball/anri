.. _theory_detector:

The detector
============

:doc:`peak_shape` gives each peak a Gaussian in :math:`(s, f, \omega)` on an ideal detector. This chapter is what lies
between that Gaussian and a pixel's value: where a photon stops in a thick sensor, how far its signal spreads, how the
detector's pixels are distorted, and the pixels themselves. In the order a photon meets them:

1. it reaches the sensor's surface at a point of the ideal detector (the Gaussian);
2. it is absorbed at some depth along its ray, so further out than where it arrived (:ref:`theory_detector_sensor`);
3. its signal is spread, by fluorescence and by the charge cloud (:ref:`theory_detector_psf`);
4. it is counted in the pixel where it ends up, a pixel whose place on the ideal detector is given by the distortion
   (:ref:`theory_detector_spatial`, :ref:`theory_detector_pixels`).

Steps 2 and 3 move the photon continuously; step 4 is the only discrete one.

.. _theory_detector_pixels:

Pixels
------

A pixel counts every photon that ends up within its area. Its value is the mass of the spot's density over the pixel's
cell, :math:`[i - \tfrac{1}{2}, i + \tfrac{1}{2})` in pixel coordinates (pixel centres at the integers), and likewise a
frame's is the mass over its omega range:

.. math::
    :label: det-pixel

    I_i = I \int_{i - 1/2}^{i + 1/2} p(x)\, dx
    = I \left[\Phi\!\left(\frac{i + \tfrac{1}{2} - \mu}{\sigma}\right) - \Phi\!\left(\frac{i - \tfrac{1}{2} - \mu}{\sigma}\right)\right]
    \ \text{(Gaussian)},
    \qquad
    \sum_{i=-n}^{n} I_i = I\left[\Phi\!\left(\frac{n + \tfrac{1}{2} - \mu}{\sigma}\right) - \Phi\!\left(\frac{-n - \tfrac{1}{2} - \mu}{\sigma}\right)\right].

**Definition:** a pixel's value. **Proof:** ``tests/theory/test_detector.py::test_box_masses_sum`` (the masses telescope,
so a row of pixels holds exactly the spot's mass over its extent, whatever the spot's width).

**Anri's rule:** a spot is integrated over each pixel's area and each frame's omega range, never sampled at their
centres. A spot narrower than a pixel sampled at pixel centres has its intensity wrong by any factor (the density at
the nearest centre times the area), and its sum over pixels is not its intensity.

**Code:** :func:`anri.maths.utils.bin_fractions` (through :func:`anri.fwd.render.render_peaks`).

What the pixels do to a spot's moments follows from writing the pixel index as :math:`\operatorname{round}(x) = x + s(x)`,
with :math:`s` the sawtooth :math:`-x` on :math:`[-\tfrac{1}{2}, \tfrac{1}{2})`, of period 1. Its Fourier series, those of
:math:`s^2`, the Gaussian's characteristic function and Stein's identity give, for :math:`x \sim N(\mu, \sigma^2)` and
:math:`\rho_k = e^{-2\pi^2 k^2 \sigma^2}`, exactly:

.. math::
    :label: det-pixel-moments

    \operatorname{E}[i] = \mu + \sum_{k \ge 1} \frac{(-1)^k}{\pi k}\, \rho_k \sin 2\pi k\mu,
    \qquad
    \operatorname{Var}[i] = \sigma^2 + \frac{1}{12}
    + \sum_{k \ge 1} (-1)^k \rho_k \cos 2\pi k\mu \left(4\sigma^2 + \frac{1}{\pi^2 k^2}\right)
    - \left(\operatorname{E}[i] - \mu\right)^2.

**Proof:** ``tests/theory/test_detector.py::test_sawtooth_series`` (the coefficients of :math:`s` and :math:`s^2`),
``tests/theory/test_detector.py::test_gaussian_fourier`` (the characteristic function, and
:math:`\operatorname{E}[(x - \mu) e^{i\omega x}] = i\omega\sigma^2 \operatorname{E}[e^{i\omega x}]`, which gives
:math:`\operatorname{Cov}(x, s(x))` term by term), ``tests/theory/test_detector.py::test_pixel_moments_code`` (the moments
of :func:`anri.maths.utils.bin_fractions`' masses against the series, summed until :math:`\rho_k < 10^{-30}`).

So a spot's pixel variance is its own plus :math:`1/12` (a uniform over one pixel), and its pixel centroid is its true
centroid, up to terms in :math:`\rho_1 = e^{-2\pi^2\sigma^2}` that oscillate with the spot's position within a pixel. The
centroid's bias has amplitude :math:`\rho_1/\pi` to leading order: 0.145 pixel at :math:`\sigma = 0.2`, 0.0539 at 0.3, and
0.00229 at 0.5 (``tests/theory/test_detector.py::test_pixel_bias_numbers``). Model and data are integrated alike, so a fit
of pixel values carries no such bias; a fit of centroids does.

How well can a spot's position be known from its pixel counts? Each count is Poisson with mean
:math:`\lambda_i = N p_i(\mu)` (:math:`N` photons, :math:`p_i` the box mass), so the Fisher information about :math:`\mu` and the
Cramér-Rao bound on any unbiased estimate are

.. math::
    :label: det-crlb

    \mathcal{I}(\mu) = N \sum_i \frac{\left(\partial p_i/\partial\mu\right)^2}{p_i},
    \qquad
    \frac{\partial p_i}{\partial\mu} = \frac{\varphi(a_i) - \varphi(b_i)}{\sigma},
    \qquad
    \sigma_\mu \ge \mathcal{I}^{-1/2},

with :math:`a_i, b_i` the cell's standardised edges. With no pixels (infinitely small ones) the information is
:math:`N/\sigma^2`.

**Proof:** ``tests/theory/test_detector.py::test_poisson_fisher`` (a Poisson count's information
:math:`\lambda'^2/\lambda`; the mass's derivative; :math:`\int \varphi'^2/\varphi = 1`).

The bound, as :math:`\sqrt{N}\,\sigma_\mu` in pixels, for a spot centred on a pixel and on the edge between two, against
:math:`\sqrt{\sigma^2 + 1/12}`:

.. list-table::
    :header-rows: 1

    * - :math:`\sigma` (pixels)
      - centred
      - on an edge
      - :math:`\sqrt{\sigma^2 + 1/12}`
    * - 0.1
      - 25.5
      - 0.125
      - 0.306
    * - 0.2
      - 0.636
      - 0.251
      - 0.351
    * - 0.3
      - 0.466
      - 0.374
      - 0.416
    * - 0.5
      - 0.579
      - 0.575
      - 0.577
    * - 1
      - 1.04
      - 1.04
      - 1.04

(``tests/theory/test_detector.py::test_crlb_numbers``, evaluated at 30 digits.) From half a pixel up, the pixels cost what
a uniform within a pixel adds, :math:`1/12`. Below about 0.3 pixel the bound depends on where the spot falls: centred
in a pixel, nearly all its light is in that pixel, which says almost nothing of where within it; on an edge, the split
between two pixels places it well. A spot much narrower than a pixel keeps its position only where it straddles pixels.
The same holds in omega, in units of the frame width: a peak much narrower than a frame is placed within its frame only
by its neighbouring frames: an ID11 peak whose core is narrower than one frame says little about its own omega width.

.. _theory_detector_psf:

Point spread
------------

The point spread is a continuous kernel: a photon absorbed at :math:`x` is counted as if it had arrived at
:math:`x + \epsilon`, with :math:`\epsilon` from the kernel. A Gaussian spot convolved with a Gaussian kernel is Gaussian with
the variances added:

.. math::
    :label: det-psf

    \int \frac{1}{\sigma_1}\varphi\!\left(\frac{x - y}{\sigma_1}\right) \frac{1}{\sigma_2}\varphi\!\left(\frac{y}{\sigma_2}\right) dy
    = \frac{1}{\sigma}\varphi\!\left(\frac{x}{\sigma}\right),
    \qquad \sigma^2 = \sigma_1^2 + \sigma_2^2.

**Proof:** ``tests/theory/test_detector.py::test_gaussian_convolution``,
``tests/theory/test_detector.py::test_psf_code`` (the renderer adds :math:`\sigma_{\text{psf}}^2` to the slow and fast
variances, and nothing else).

**Anri's rule:** kernels act on the continuous spot, before the pixels integrate it, and the pixels integrate it once.
Integrating first and then convolving the pixel values with a pixel-integrated kernel counts the pixel's :math:`1/12`
twice:

.. math::
    :label: det-box-once

    \operatorname{Var} = \sigma^2 + \sigma_k^2 + \tfrac{1}{12} \ \text{(kernel, then pixels)},
    \qquad
    (\sigma^2 + \tfrac{1}{12}) + (\sigma_k^2 + \tfrac{1}{12}) \ \text{(pixels, then a pixel-integrated kernel)}.

**Proof:** ``tests/theory/test_detector.py::test_box_once``.

The renderer follows the rule: the point spread is added to the Gaussian's covariance (:eq:`ps-total`), and the sensor's
kernel is a mixture of Gaussians, each integrated over the pixels once (:eq:`det-sensor-mix`).

**Code:** :func:`anri.fwd.render.render_peaks`.

A photon-counting CdTe detector's point spread is not one Gaussian: a sharp core, and a wing from fluorescence (a Cd or
Te K photon that escapes and is absorbed a pixel or two away, and counted there when the lower threshold is below it).
The renderer takes a mixture, rendering is linear in intensity, so each component is rendered with its share:

.. math::
    :label: det-psf-mix

    I_{\text{pixel}} = \sum_k f_k\, I_{\text{pixel}}(\sigma_k), \qquad \sum_k f_k = 1.

**Definition:** the point spread as a mixture of Gaussians, by linearity of the convolution.

**Measured:** a core of ~0.2-0.3 pixel and a wing of ~1.1-1.5 pixel holding ~28% of the counts at 43.6 keV, ~8% at
72 keV (thresholds above the K lines), with ``python -m anri.calib detector`` (AGENTS.md). Measure it per energy and
threshold: a wrong one is absorbed by whatever else is fitted.

**Code:** :func:`anri.fwd.render.render_row` (geom "psf_mix").

**Anri's choice:** the charge cloud of a photon counted with its threshold at half its energy goes, whole, to the pixel
holding most of its charge, nearly the pixel holding its centre: a box integration of where it stopped, not a blur. The
core :math:`\sigma` stands for what is measured (convergence, residual charge sharing), not for the cloud's size.

.. _theory_detector_sensor:

A thick sensor
--------------

A photon reaching the sensor at an angle :math:`\phi` to its normal travels a path :math:`\ell` before it is absorbed,
:math:`\ell \sim` Exp(:math:`\mu`), or passes through, uncounted, beyond :math:`T/\cos\phi` (:math:`T` the thickness). It is counted
where it stops, :math:`\ell\sin\phi` further along the in-plane direction of its ray. In pixels, with :math:`\hat{k}` the ray
from the sample (the lab origin) to the spot, :math:`\hat{n}` the sensor's normal, and :math:`\matr{A} = [\vec{s}, \vec{f}]` the
pixel steps:

.. math::
    :label: det-sensor

    x = L\,\ell \ \text{along } \hat{u}, \quad \ell \sim \mu e^{-\mu\ell} \text{ on } [0, T/\cos\phi],
    \qquad
    L\hat{u} = (\matr{A}^T\matr{A})^{-1}\matr{A}^T\left(\hat{k} - (\hat{n}\cdot\hat{k})\hat{n}\right),
    \quad \cos\phi = |\hat{n}\cdot\hat{k}|,

so the displacement in pixels is a truncated exponential of scale :math:`L/\mu` and cut :math:`LT/\cos\phi`, along
:math:`\hat{u}`.

**Definition:** Beer-Lambert absorption along the ray, counted where it stops. **Proof:**
``tests/theory/test_detector.py::test_in_plane_solve`` (the written-out 2 × 2 solve recovers an in-plane vector's
pixel coordinates exactly), ``tests/theory/test_detector.py::test_sensor_kernel_code`` (direction, scale, cut and
slices against :eq:`det-sensor` and :eq:`det-slices`).

**Code:** :func:`anri.fwd.render.render_row` (geom "sensor"), through ``_sensor_kernel``.

The exponential is not Gaussian, so the renderer cuts it into :math:`n` slices of equal probability (:math:`n = 4`) and
puts a Gaussian of each slice's mean and variance in its place. With :math:`P = 1 - e^{-R/b}` (scale :math:`b`, cut :math:`R`),

.. math::
    :label: det-slices

    e_k = -b\log\!\left(1 - \frac{k}{n}P\right),
    \quad
    m_k = b + \frac{e_k e^{-e_k/b} - e_{k+1} e^{-e_{k+1}/b}}{Z_k},
    \quad
    \overline{x^2}_k = \frac{\left[e^{-x/b}(x^2 + 2xb + 2b^2)\right]_{e_{k+1}}^{e_k}}{Z_k},
    \quad
    Z_k = e^{-e_k/b} - e^{-e_{k+1}/b} = \frac{P}{n},

with :math:`e_n = R`. Each slice holds :math:`1/n` of the truncated distribution, so the mixture of the slices' Gaussians,
each of weight :math:`1/n`, has exactly the truncated exponential's mean and variance (the law of total variance), and
tends to its shape as :math:`n` grows.

**Proof:** ``tests/theory/test_detector.py::test_exponential_slices`` (the quantiles, each slice's mass, mean and second
moment, and the mixture's mean), ``tests/theory/test_detector.py::test_exponential_slices_code``.

**Code:** :func:`anri.maths.utils.exponential_slices`.

The spot is then rendered as the mixture, each slice's Gaussian shifted along :math:`\hat{u}` by its mean :math:`r_k`
(the slices' mean removed) and widened along it by its variance :math:`v_k`:

.. math::
    :label: det-sensor-mix

    \frac{1}{n}\sum_k N\!\left(\vec{\mu} + r_k\hat{u},\ \matr{\Sigma} + v_k\hat{u}\hat{u}^T\right),
    \qquad
    \frac{1}{n}\sum_k (v_k + r_k^2) = \operatorname{Var}(x),

all over one window, placed for the whole kernel's covariance :math:`\matr{\Sigma} + \operatorname{Var}(x)\,\hat{u}\hat{u}^T`.

**Definition:** the renderer's sensor kernel (omega is not changed). The window's variance is the mixture's, by the
law of total variance (:eq:`det-slices`).

**Code:** :func:`anri.fwd.render.render_peaks`.

**Anri's choice:** the kernel's mean shift is left out: a geometry calibrated on spots or rings measured with the same
sensor holds it. It is not exactly a moved detector. Untruncated, a photon at angle :math:`\phi` is shifted by
:math:`\sin\phi/\mu` on average, while a detector normal to the beam moved back by :math:`\Delta` shifts it by
:math:`\Delta\tan\phi`; the best :math:`\Delta = 1/\mu` leaves

.. math::
    :label: det-sensor-shift

    \frac{\sin\phi - \tan\phi}{\mu} = -\frac{\phi^3}{2\mu} + O(\phi^5),

a radial distortion of third order in :math:`2\theta` (:math:`\phi = 2\theta`): -0.297 µm at :math:`2\theta = 10°` for
:math:`1/\mu = 111` µm (CdTe at 43.6 keV), a few thousandths of a 75 µm pixel. A calibration that fits a radial
distortion absorbs it too.

**Proof:** ``tests/theory/test_detector.py::test_sensor_mean_shift`` (the series; the exponential's mean :math:`1/\mu`;
the number).

.. _theory_detector_spatial:

Spatial distortion
------------------

A measured detector's pixels are not on the ideal grid. A distortion map (dx, dy) [n_slow, n_fast], sampled at the raw
pixels' centres, says where each raw pixel :math:`\vec{x}` (row, column) is on the ideal detector:
:math:`\vec{c}(\vec{x}) = \vec{x} + \vec{d}(\vec{x})`, with :math:`\vec{d}` interpolated bilinearly between centres.

**Anri's rule:** a raw pixel's value is the spot's mass over the image of its cell on the ideal detector (its corners
mapped through :math:`\vec{c}`), equivalently the spot's density pulled back to raw coordinates, times
:math:`|\det \partial\vec{c}/\partial\vec{x}|`, integrated over the raw cell. Never the density at its mapped centre, and
never pixel values resampled from one grid to the other.

Where :math:`\vec{c}` is affine over a spot, with Jacobian :math:`\matr{A}`, a Gaussian :math:`N(\vec{\mu}_c, \matr{S})` on the
ideal detector is, on the raw pixels,

.. math::
    :label: det-spatial

    N\!\left(\vec{x}_0,\ \matr{A}^{-1}\matr{S}\matr{A}^{-T}\right), \quad \vec{c}(\vec{x}_0) = \vec{\mu}_c,
    \qquad
    \operatorname{Cov}(\vec{x}, \omega) = \matr{A}^{-1}\operatorname{Cov}(\vec{c}, \omega),
    \qquad
    \vec{x}_{j+1} = \vec{x}_j - \matr{A}(\vec{x}_j)^{-1}\left(\vec{c}(\vec{x}_j) - \vec{\mu}_c\right),

with the same mass (the density transforms with :math:`|\det\matr{A}|`), so a raw pixel covering more of the ideal detector
gets more light. :math:`\vec{x}_0` is found by Newton's method, exact in one step where :math:`\vec{c}` is affine.

**Proof:** ``tests/theory/test_detector.py::test_pullback`` (the quadratic form and the determinant),
``tests/theory/test_detector.py::test_newton_affine``, ``tests/theory/test_detector.py::test_map_shape_code``,
``tests/theory/test_detector.py::test_undistort_code`` (an affine map, and a bilinear map with a cross term, against the
exact root and Jacobian).

**Code:** :func:`anri.fwd.render.render_row` (geom "spatial"), through ``_undistort`` and ``_map_shape``.

The sensor kernel's direction maps the same way, :math:`\hat{u} \to \matr{A}^{-1}\hat{u}` (no longer of unit length;
:math:`r_k\hat{u}` and :math:`v_k\hat{u}\hat{u}^T` stay right).

The renderer follows the rule to first order in the map's variation across a spot: it takes :math:`\matr{A}` at
:math:`\vec{x}_0` and integrates the pulled-back Gaussian over raw cells. Where the map curves, the true cell edges move
from the affine ones by at most :math:`\tfrac{1}{2}K r^2` pixels at distance :math:`r` from :math:`\vec{x}_0`, with :math:`K` the
largest second derivative of :math:`\vec{c}`. A bilinear interpolation is not affine within a cell (it has a cross term,
so its Jacobian varies there and jumps between cells); Newton's method still converges, quadratically
(``test_undistort_code``).

.. _theory_detector_splat:

Splatting a grain's voxels
--------------------------

For a grain map, ``splat`` renders a spot (a grain's reflection) as one kernel, a single peak's window cells for a point
at an integer pixel, convolved with each voxel's weight shared between the four pixels around the voxel's own centroid.
Along each axis, a voxel at fractional offset :math:`d` gives :math:`1 - d` to one pixel and :math:`d` to the next:

.. math::
    :label: det-splat

    \operatorname{E} = d, \qquad \operatorname{Var} = d(1 - d) \le \tfrac{1}{4}.

**Proof:** ``tests/theory/test_detector.py::test_splat_bilinear``.

**Code:** :func:`anri.fwd.render.render_row` (``splat``).

**Where the code falls short:** this breaks the rule of :eq:`det-pixel`. The kernel is integrated over the pixels once,
for a point at an integer pixel; sharing it between two integer positions is not the pixel integration of the spot at
:math:`d`. Each voxel's intensity and centroid are kept, but its variance grows by :math:`d(1 - d)` per axis, up to
:math:`1/4` pixel², by an amount that depends on where the voxel falls within a pixel. That is small against a spot of
a few pixels, and of the size of the effects above for a spot narrower than a pixel. Splatting also takes the beam's
weight per frame at the frame's mean omega, without :eq:`int-fold`.

What an exact splat needs. Rendering the kernel with :math:`\matr{\Sigma}` and sharing a voxel's weight between grid
points is the pixel integration, once, of the continuous mixture :math:`\sum_j s_j N(x - j, \matr{\Sigma})`: the rule of
:eq:`det-box-once` is kept when that mixture has the voxel's spot's mean and variance. Bilinear sharing's variance
:math:`d(1 - d)` changes with :math:`d`, so no shared kernel can make up for it. Quadratic B-spline weights on a grid of
step :math:`1/n` pixel, :math:`w_{\mp} = \tfrac{1}{2}(\tfrac{1}{2} \mp e)^2`, :math:`w_0 = \tfrac{3}{4} - e^2` (:math:`e` the
offset from the nearest grid point, in grid steps), have the offset as mean and a variance that does not depend on
it:

.. math::
    :label: det-splat-exact

    \sum_k w_k\, k = e, \qquad \sum_k w_k\, k^2 - e^2 = \tfrac{1}{4}\ \text{grid step}^2 = \frac{1}{4n^2}\ \text{pixel}^2,

so a kernel rendered with :math:`\tfrac{1}{4n^2}` taken off each detector conditional variance (slow given the frame,
fast given the row and frame, :eq:`ps-condition`, :eq:`ps-condition-f`) gives every voxel its spot's first two
moments exactly, for any subpixel position. It needs those conditional variances above :math:`\tfrac{1}{4n^2}`: a
point spread of 0.25 pixel needs :math:`n = 2`, of 0.2 pixel :math:`n = 3`, and each phase of the finer grid is its own
kernel and convolution, :math:`n^2` of each per spot. The beam's fold differs from voxel to voxel (each crosses the
beam at its own place): its weight per frame and its shift of the frame's mean are per voxel and fit the sharing,
but its narrowing of the profile needs a margin in that variance too.

**Proof:** ``tests/theory/test_detector.py::test_splat_bspline``.

The model's threshold
---------------------

The renderer drops a model pixel whose value, summed over every peak of the row, is below ``min_value``; a peak's own
share of a pixel is never cut, since many weak voxels can light one pixel above it. The data's threshold, the
segmentation's cut, is the measurement's: pixels below it are censored, not zero (:eq:`lik-censored`).

**Code:** :func:`anri.fwd.render.render_row`.

What is left out
----------------

- **Counting as a discrete process**: charge sharing between pixels, the thresholds' effect on a split cloud, pile-up and
  dead time; the measured point spread stands for them.
- **The sensor kernel's mean shift** (:eq:`det-sensor-shift`), and the slices' shape beyond their first two moments.
- **The distortion's curvature** across a spot (first order in the map).
- **Splatting's extra variance** (:eq:`det-splat`).
- **The detector's efficiency** and gain versus angle and energy: in the relative scale.
