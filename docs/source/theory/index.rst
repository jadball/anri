.. _theory:

======
Theory
======

These pages state, in order, the physics and statistics behind Anri: each equation Anri computes, where it comes
from, how it is proved, and the function that computes it. They are the single source of truth for Anri's maths: code
follows them, and new maths is written here, with its proof, before the code that uses it.

- Every numbered equation has a status: **Proof** (a test in ``tests/theory`` that proves it with sympy, or checks
  that the function computing it agrees with the proved expression), **Definition**, **Cited** or **Measured**.
  ``tests/theory/test_pages.py`` fails if one has none, or names a proof that does not exist.
- Paragraphs starting **Anri's rule** are requirements: code that breaks one is a bug.
- Every displayed equation Anri computes is followed by a **Code** line naming the function that computes it. A test
  (``tests/unit/test_theory_docs.py``) imports every name these pages give, so a renamed function fails CI instead of
  leaving a dead link here.
- Where an equation comes from the literature, it cites the paper (:ref:`theory_references`). Where it is Anri's own
  choice or approximation, the page says so in a paragraph starting **Anri's choice**, with the reason.
- Each chapter ends with **What is left out**: the approximations made, so that you know when a result can't be
  trusted.

The forward model (:mod:`anri.fwd` and the layers below it), the detector and what it records, the measurements a
refiner is built on and the limits of what the data determine, the indexer (:mod:`anri.index`) and the calibration
of the geometry (:mod:`anri.calib`). Where the maths is ahead of the code (the threshold's model, region moments), the
pages say so. The :doc:`tutorials <../tutorials/index>` compute the same equations with Anri's functions
and check them.

Forward model
=============

1. :doc:`conventions`: units, branches and symbols.
2. :doc:`frames`: the reference frames, from Miller indices to detector pixels, with diagrams.
3. :doc:`crystal`: from a crystal structure to reciprocal-lattice vectors and their intensities.
4. :doc:`diffraction`: where a reflection diffracts: the Ewald condition, omega, 2θ and η, and the detector.
5. :doc:`intensity`: how bright a peak is: Lorentz, polarisation, the beam's profile over a voxel, absorption.
6. :doc:`peak_shape`: how wide a peak is and how it is spread over pixels and frames; windows.
7. :doc:`detector`: the sensor's depth, the point spread, the distortion and the pixels.
8. :doc:`recording`: what the detector records: counting noise, frames, the monitor and the threshold.

Measurement and limits
======================

9. :doc:`moments`: statistics of recorded light (counts and moments over a region) and what each keeps: exact ray
   sums for any spot shape, what a region's choice and a window's edge do to them.
10. :doc:`limits`: what the data determine: the Fourier slice, the dislocations it misses, the beam, separation,
    localisation, second moments, model error, and what a method must keep to reach the limits.
11. :doc:`refinement`: what a refiner's statistics are robust to: weights and shape errors, symmetry, the pixels'
    aliasing, a spread's second-order mean, light beyond a region, and the cost of splitting merged light.

Indexer
=======

12. :doc:`indexing`: the data as a linear tomography of orientation occupancies, and the histogram it is fitted to.
13. :doc:`search`: which orientations to try: grids over the fundamental zone, and completeness.
14. :doc:`likelihood`: fitting occupancies: Poisson likelihood, MLEM, SQUAREM, censoring and the likelihood ratio.
15. :doc:`localsearch`: refining the orientations off the search grid, and fitting the voxels coarse to fine.
16. :doc:`populations`: from occupancies to orientation populations per voxel, and the sample mask.

Calibration
===========

17. :doc:`calibration`: the detector and beam geometry from powder rings, then from the indexed scan's peaks.

:doc:`references` lists every work cited.

.. toctree::
    :hidden:
    :numbered:

    conventions
    frames
    crystal
    diffraction
    intensity
    peak_shape
    detector
    recording
    moments
    limits
    refinement
    indexing
    search
    likelihood
    localsearch
    populations
    calibration

.. toctree::
    :hidden:

    references
