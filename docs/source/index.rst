.. _anri_documentation:

Anri: GPU-accelerated Diffraction Microstructure Imaging analysis
=================================================================

|repo| |docs| |license| |tests| |codecov| |style| |platform| |jax|

.. |repo| image:: https://img.shields.io/badge/repo-GitHub-lightgrey?logo=github
   :target: https://github.com/jadball/anri

.. |platform| image:: https://img.shields.io/badge/platform-cross--platform-brightgreen.svg
   :target: https://www.python.org/

.. |jax| image:: https://img.shields.io/badge/code-JAX-blue.svg
   :target: https://github.com/jax-ml/jax

.. |tests| image:: https://img.shields.io/github/actions/workflow/status/jadball/anri/main.yml
   :target: https://github.com/jadball/anri/actions/workflows/main.yml

.. |codecov| image:: https://codecov.io/gh/jadball/anri/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/jadball/anri

.. |style| image:: https://img.shields.io/badge/code%20style-Ruff-000000.svg
   :target: https://github.com/astral-sh/ruff

.. |docs| image:: https://img.shields.io/badge/docs-sphinx-blue?logo=sphinx.svg
   :target: https://jadball.github.io/anri/

.. |license| image:: https://img.shields.io/github/license/jadball/anri
   :target: https://github.com/jadball/anri/blob/main/LICENSE
   :alt: GitHub License

What is Anri?
=============

Anri is a Python package for the GPU-accelerated analysis of Diffraction Microstructure Imaging data, such as (Scanning) Three-Dimensional X-Ray Diffraction \[(s)3DXRD\].

How does it work?
=================

The core concept of Anri is as follows:

- **JAX-native code**

(almost) all of Anri is implemented in `JAX <https://docs.jax.dev/en/latest/>`_, a Python library for high-performance (e.g. GPU-accelerated) numerical computing.  
The benefit is that Anri will work on any major x86 (and some ARM) CPUs and any recent CUDA-compatible GPU.

- **Intensity-aware forward model**

At the core of Anri is a high-performance forward model that goes from sample space (e.g. a grid of UBI matrices) to detector space (e.g. \[slow, fast\] coordinates). This can be used to investigate the performance of existing (s)3DXRD analysis packages such as `ImageD11 <https://github.com/FABLE-3DXRD/ImageD11>`_ by comparing forward-projected data to the raw data that you measured. Intensities are computed using the structure factors thanks to `Dan's Diffraction <https://github.com/DanPorter/Dans_Diffraction>`_ and accumulate in detector pixels.

- **Differentiability**

Great effort has been undertaken to ensure that JAX-native Anri functions are differentiable, using the powerful auto-diff capabilities of JAX. This has two obvious use-cases:

- **Peak shapes**
By expressing instrumental parameters such as incident beam divergence and energy spread as Gaussian distributions, Anri can use the per-peak Jacobians produced by JAX to propagate these parameters into detector space as a covariance matrix in output space, thereby rendering fairly realistic peak shapes that are not just simple detector point spread functions. Therefore, a spread in beam energy (for example) manifests as a radial distribution on the detector. 

- **Gradient-aware optimisation (in progress)**
Anri will take advantage of the differentiable, intensity-aware forward model to perform iterative refinement of grain maps produced by `ImageD11 <https://github.com/FABLE-3DXRD/ImageD11>`_ (and perhaps other programs in the future) to yield refined maps of orientation gradients and strains.


.. toctree::
    :maxdepth: 2
    :hidden:

    user/index
    reference/index
    CONTRIBUTING
    CHANGELOG

Installation
============

From Conda
----------

Coming soon!

From source (for developers)
----------------------------

Anri may (eventually) rely on packages from both `conda` and `pip`.  
For ease of installation, it is recommended to use `unidep <https://github.com/basnijholt/unidep>`_ which can install packages from both sources.

Clone the repository
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone git@github.com:jadball/anri.git anri
    cd anri

Set up a Conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    conda create -n <env-name>
    conda activate <env-name>

Ensure pip is running from the Conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    which pip  # should yield something inside the environment <env-name>

Install build dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install --upgrade pip unidep

Install conda, then pip deps, then the package itself (with `dev` optional deps) as editable:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    unidep install .[dev] -e