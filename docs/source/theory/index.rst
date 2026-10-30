.. _theory:

======
Theory
======

These pages state, in order, the physics and statistics behind Anri's forward model, indexer and calibration: each
equation Anri computes, where it comes from, and the function that computes it.

- Every displayed equation is followed by a **Code** line naming the function that computes it. A test
  (``tests/unit/test_theory_docs.py``) imports every name these pages give, so a renamed function fails CI instead of
  leaving a dead link here.
- Where an equation comes from the literature, it cites the paper (:ref:`theory_references`). Where it is Anri's own
  choice or approximation, the page says so in a paragraph starting **Anri's choice**, with the reason.
- Each chapter ends with **What is left out**: the approximations made, so that you know when a result can't be
  trusted.

Only the settled parts of Anri are covered: the forward model (:mod:`anri.fwd` and the layers below it), the
indexer (:mod:`anri.index`) and the calibration of the geometry (:mod:`anri.calib`). The :doc:`tutorials <../tutorials/index>` compute the same equations with Anri's functions
and check them.

Forward model
=============

1. :doc:`conventions`: units, branches and symbols.
2. :doc:`frames`: the reference frames, from Miller indices to detector pixels, with diagrams.
3. :doc:`crystal`: from a crystal structure to reciprocal-lattice vectors and their intensities.
4. :doc:`diffraction`: where a reflection diffracts: the Ewald condition, omega, 2θ and η, and the detector.
5. :doc:`intensity`: how bright a peak is: Lorentz, polarisation, the beam's profile over a voxel, absorption.
6. :doc:`peak_shape`: how wide a peak is and how it is spread over pixels and frames.

Indexer
=======

7. :doc:`indexing`: the data as a linear tomography of orientation occupancies, and the histogram it is fitted to.
8. :doc:`search`: which orientations to try: grids over the fundamental zone, and completeness.
9. :doc:`likelihood`: fitting occupancies: Poisson likelihood, MLEM, SQUAREM, censoring and the likelihood ratio.
10. :doc:`localsearch`: sharpening each voxel's orientations off the search grid.
11. :doc:`populations`: from occupancies to orientation populations per voxel, and the sample mask.

Calibration
===========

12. :doc:`calibration`: the detector and beam geometry from powder rings, then from the indexed scan's peaks.

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
    indexing
    search
    likelihood
    localsearch
    populations
    calibration

.. toctree::
    :hidden:

    references
