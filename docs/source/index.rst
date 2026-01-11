.. _anri_documentation:

Anri documentation
===================

|repo| |platform| |jax| |tests| |codecov| |style| |docs|

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

Anri is a Python library for processing polycrystalline diffraction data with `JAX <https://docs.jax.dev/en/latest/>`_.

`GitHub Repository <https://github.com/jadball/anri>`_  
`License <https://raw.githubusercontent.com/jadball/anri/main/LICENSE>`_

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

    unidep install .[dev]