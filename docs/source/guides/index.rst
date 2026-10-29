======
Guides
======

Each guide does one task with your own data, from start to finish. The notebooks among them are stored runs (they need
a GPU or your own data), so they are not re-executed when the docs are built; run them yourself, change the parameters
in their first cell, and each prints the command line that does the same.

- :doc:`rendering_parameters`: run the stages of ``python -m anri.simulate`` on a map, with a plot for each option, and
  write an ImageD11 dataset.
- :doc:`indexing_parameters`: run the stages of ``python -m anri.index`` on an ImageD11 dataset, with a diagnostic
  plot and, where it is cheap, a slider for each option.
- :doc:`calibration`: run ``python -m anri.calib fit`` on an indexed map and the scan's peaks, with plots to choose
  its options before the fit and to judge the geometry after it (residuals, and the strain it leaves), and write
  ImageD11 parameter files.
- :doc:`compare_frames`: compare a measured dataset's frames, row by row and frame by frame, with a map's forward
  model.
- :doc:`powder`: sum a dataset's pixels into a powder pattern, for phase identification and for refining a cell with a
  Rietveld or Le Bail program.

.. toctree::
    :hidden:

    rendering_parameters
    indexing_parameters
    calibration
    compare_frames
    powder
