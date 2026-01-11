# anri
Anri is a Python package for analysing Three-Dimensional X-Ray Diffraction data.  
Anri is in the very early stages of development and is not yet ready to use for experimental analysis.

[![cross-platform](https://img.shields.io/badge/platform-cross--platform-brightgreen.svg)](https://www.python.org/)
[![JAX Python](https://img.shields.io/badge/code-JAX-blue.svg)](https://github.com/jax-ml/jax)
[![tests (windows, ubuntu, mac os)](https://img.shields.io/github/actions/workflow/status/jadball/anri/main.yml)](https://github.com/jadball/anri/actions/workflows/main.yml)
[![code style ruff](https://img.shields.io/badge/code%20style-Ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Sphinx documentation](https://img.shields.io/badge/docs-sphinx-blue?logo=sphinx.svg)](https://jadball.github.io/anri/)

# Dependencies
We currently target all stable releases of Python. Today this is `3.9 - 3.14` on Windows, ubuntu and OSX (ARM and x86).

# Installation
## From Conda
Coming soon!
## From source (for developers)
Anri may (eventually) rely on packages from both `conda` and `pip`.  
For ease of installation, it is recommended to use [unidep](https://github.com/basnijholt/unidep) which can install packages from both sources.
### Clone the repository
```bash
git clone git@github.com:jadball/anri.git anri
cd anri
```
### Set up a Conda environment
```bash
conda create -n <env-name>
conda activate <env-name>
```
### Ensure pip is running from the Conda environment
```bash
which pip  # should yield something inside the environment <env-name>
```
### Install build dependencies
```bash
pip install --upgrade pip unidep
```
### Install conda, then pip deps, then the package itself (with `dev` optional deps) as editable:
```bash
unidep install .[dev]
```

# Development
## Repository layout
This GitHub repository is based on the python package template by @allenai: [python-package-template](https://github.com/allenai/python-package-template).
## IDE
[Visual Studio Code](https://code.visualstudio.com/) is recommended for development.  
## Linting, formatting and type checking
`anri` uses `ruff` to lint and format, and `ty` for type-checking.  
All Python functions and files (outside of `anri/sandbox`) must conform for the GitHub CI tests to pass.  
With `Visual Studio Code` you have easy access to automatic lint-on-save and format-on-save via extensions.  
Inside `.vscode` you have a `settings.sample.json` which, if you're happy with, you can rename to `settings.json` to apply my recommended per-project settings for this repository.  
You also have `extensions.json` containing my recommended extensions (including `ruff`) which `Visual Studio Code` should prompt you to install automatically.