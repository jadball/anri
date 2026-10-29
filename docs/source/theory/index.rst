.. _theory:

======
Theory
======

These pages state, in order, the physics and statistics behind Anri's forward model and indexer: each equation Anri
computes, where it comes from, and the function that computes it.

- Every displayed equation is followed by a **Code** line naming the function that computes it. A test
  (``tests/unit/test_theory_docs.py``) imports every name these pages give, so a renamed function fails CI instead of
  leaving a dead link here.
- Where an equation comes from the literature, it cites the paper (:ref:`theory_references`). Where it is Anri's own
  choice or approximation, the page says so:

  **Anri's choice**, with the reason.

- Each chapter ends with **What is left out**: the approximations made, so that you know when a result can't be
  trusted.

Only the settled parts of Anri are covered: the forward model (:mod:`anri.fwd` and the layers below it) and the
indexer (:mod:`anri.index`). The tutorials (:ref:`tut_geom` and the notebooks) show the same things with code.

Forward model
=============

1. :doc:`conventions`: frames, units and angles.
2. :doc:`crystal`: from a crystal structure to reciprocal-lattice vectors and their intensities.
3. :doc:`diffraction`: where a reflection diffracts: the Ewald condition, omega, 2θ and η, and the detector.
4. :doc:`intensity`: how bright a peak is: Lorentz, polarisation, the beam's profile over a voxel, absorption.
5. :doc:`peak_shape`: how wide a peak is and how it is spread over pixels and frames.

Indexer
=======

6. :doc:`indexing`: the data as a linear tomography of orientation occupancies, and the histogram it is fitted to.
7. :doc:`search`: which orientations to try: grids over the fundamental zone, and completeness.
8. :doc:`likelihood`: fitting occupancies: Poisson likelihood, MLEM, SQUAREM, censoring and the likelihood ratio.
9. :doc:`populations`: from occupancies to orientation populations per voxel, and the sample mask.

:doc:`references` lists every work cited.

.. toctree::
    :hidden:
    :numbered:

    conventions
    crystal
    diffraction
    intensity
    peak_shape
    indexing
    search
    likelihood
    populations

.. toctree::
    :hidden:

    references
