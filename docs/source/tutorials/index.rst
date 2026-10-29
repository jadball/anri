=========
Tutorials
=========

The tutorials teach Anri by running it, in order: each builds on the ones before. Each goes with a page of
:ref:`theory`, which states the equations; the tutorial computes them with Anri's functions and checks the results, so
the two never disagree for long. To do one task with your own data, see the :doc:`../guides/index` instead.

Crystallography
===============

From a CIF to the reflections a crystal can give: the lattice and its :math:`\matr{B}` matrix, the reflections and
their rings, structure factors and the Laue group. With :doc:`../theory/crystal`.

.. nbgallery::

    crystallography

.. _tut_geom:

Geometry
========

The goniometer takes a point or a direction from the sample frame to the lab, and the detector takes a diffracted ray
to a pixel. Both follow :ref:`theory_frames`, which defines every frame, from Miller indices to detector pixels, with
diagrams, and lists where Anri differs from the FABLE geometry (the
:download:`FABLE geometry document<./Fable_Geometry_version_1.0.8.pdf>`). The detector tutorial ends with FABLE's own
test case.

.. nbgallery::

    goniometer_geometry
    detector_geometry

Forward modelling
=================

Where a grain's reflections land, then how wide each peak is, then whole scans: the renderer simulates a scanning
3DXRD dataset pixel by pixel, and a DCT scan with it (stored run: it needs a GPU, so it is not re-executed when the
docs are built). With :doc:`../theory/diffraction`, :doc:`../theory/intensity` and :doc:`../theory/peak_shape`.

.. nbgallery::

    forward_model_simple
    forward_model_covariance
    renderer
    dct

Indexing
========

Indexing finds each voxel's orientations from the data alone, by fitting every voxel's orientation occupancies jointly
with the forward model. The first notebook makes a phantom microstructure (grains, misoriented cells and twins), the
second simulates a scan of it and indexes it from scratch. Both are stored runs, not re-executed when the docs are
built. With :doc:`../theory/indexing` and the chapters after it.

.. nbgallery::

    phantom
    indexing
