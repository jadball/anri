.. _anri_documentation:

Anri documentation
===================

Anri is a Python library for processing polycrystalline diffraction data with `JAX <https://docs.jax.dev/en/latest/>`_.

`GitHub Repository <https://github.com/jadball/anri>`_  
`License <https://raw.githubusercontent.com/jadball/anri/main/LICENSE>`_

.. toctree::
    :maxdepth: 2
    :hidden:

    user/index
    reference/index
    CHANGELOG
    CONTRIBUTING

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