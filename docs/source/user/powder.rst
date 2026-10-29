Powder pattern export
=====================

``python -m anri.index powder`` sums a scanning-3DXRD dataset's sparse pixels over every frame into a powder pattern:
the whole sample, every rotation and every dty row. It is useful for phase identification and for refining a phase's
cell with a Rietveld or Le Bail program.

.. code-block:: bash

    python -m anri.index powder <analysisroot> <sample> <dataset> --monitor fpico6 --detector-mask mask.edf

The pixels are read as the indexer reads them (spatial distortion corrected, monitor normalised), binned by their
centre in (2theta, eta), and divided by the solid angle of the measured detector pixels in each bin. The result is the
intensity per steradian per frame, the azimuthal average of the 2D pattern, with no Lorentz or polarisation
correction. It writes ``<sample>_<dataset>_powder.h5`` and a three-column ``.xye`` (2theta, intensity, esd) beside it.

The HDF5 file holds:

- ``tth``, ``intensity``, ``sigma`` (esd from the spread over eta) and ``sigma_counting``;
- ``coverage``: the measured share of each ring's solid angle (module gaps, masked pixels and the detector's edges);
- ``lit_fraction``: the share of the measured pixel-frames that hold a data pixel;
- ``cake``: the (2theta, eta) sums behind them, for other choices of normalisation;
- a ``provenance`` group, as every stage file (``python -m anri.index show``).

Two properties of the data matter more than in ordinary powder diffraction:

- **Grain statistics.** A few hundred grains act as a coarse powder: peak intensities scatter far beyond counting
  statistics. Each eta bin is an independent estimate of the ring's average, so ``sigma`` is the spread over eta.
  It is typically tens of times the counting esd. Cells and line positions are reliable; intensities, phase
  fractions and displacement parameters much less so.
- **Censored valleys.** Sparse data keep only pixels above the segmentation cut, so between peaks, where nearly
  every pixel was below it, the intensity is censored, not zero. A ``lit_fraction`` far below its median marks such
  bins; exclude them from a fit.

If you have rietx
-----------------

`rietx <https://rietx.org>`_ (Python 3.11 or later; not an anri dependency) can refine a known structure against
the exported pattern. It ships its own instructions for agents
(``rietx/data/skill/rietx/SKILL.md`` in the installed package); follow them. Points specific to these data:

- rietx's anomalous-scattering table stops at 70 keV. Above that, give f' and f'' yourself
  (``Dispersion(overrides=...)``), or turn the correction off (``instrument.source.dispersion = None``).
- The eta average of a horizontally polarised beam over the whole ring is a polarisation fraction of 0.5.
- Compute muR from the sample (radius and linear attenuation, ``anri.crystal.structure.attenuation``); do not
  refine it.
- A CIF with more than one data block (e.g. from Jana) must be cut to the structure's block.
- For indexing, give the indexer the measured zero shift (``SearchSpec(shift_allowance_deg=...)``): its default
  allowance, a line width, is far too generous at short wavelengths, where every wrong cell then fits.

.. code-block:: python

    import h5py
    import numpy as np
    import rietx as rx
    from rietx.schemas.instrument import Dispersion

    with h5py.File("Ag2S_3_za0_powder.h5") as h:
        tth, y, s, lit, cov = (h[k][()] for k in ("tth", "intensity", "sigma", "lit_fraction", "coverage"))
        wavelength = h["provenance/resolved"].attrs["wavelength"]
    limits = (2.7, 15.0)
    keep = cov > 0.3
    fit = keep & (tth > limits[0]) & (tth < limits[1])
    censored = fit & (lit < 0.1 * np.median(lit[fit]))  # chosen by looking at lit_fraction
    edges = np.flatnonzero(np.diff(np.r_[0, censored.astype(int), 0]))
    step = tth[1] - tth[0]
    excluded = [(tth[a] - 1.5 * step, tth[b - 1] + 1.5 * step) for a, b in zip(edges[::2], edges[1::2])]

    data = rx.PatternData(two_theta=tth[keep].tolist(), intensity=y[keep].tolist(), sigma=s[keep].tolist(),
                          excluded_regions=excluded)
    structure = rx.Structure.from_cif("phase.cif")
    ins = rx.Instrument.debye_scherrer(wavelength=wavelength, polarization=0.5, mu_r=0.65)
    ins.source.dispersion = Dispersion(overrides={"Ag": (0.10, 0.59), "S": (-0.006, 0.006)})  # above 70 keV
    ins.background = rx.auto_background(data, kind="chebyshev", two_theta_limits=limits, seed=True).background
    ref = rx.Refinement(structure, ins)
    result = ref.fit(data, plan="mccusker_structural", two_theta_limits=limits)
    print(ref.summary(deliverable="cell"))
