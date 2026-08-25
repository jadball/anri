# GPU-accelerated Diffraction Microstructure Imaging analysis

[![GitHub Repo](https://img.shields.io/badge/repo-GitHub-lightgrey?logo=github)](https://github.com/jadball/anri)
[![Docs](https://img.shields.io/badge/docs-sphinx-blue?logo=sphinx.svg)](https://jadball.github.io/anri/)
[![License](https://img.shields.io/github/license/jadball/anri)](https://github.com/jadball/anri/blob/main/LICENSE)
[![tests (windows, ubuntu, mac os)](https://img.shields.io/github/actions/workflow/status/jadball/anri/main.yml)](https://github.com/jadball/anri/actions/workflows/main.yml)
[![CodeCov](https://codecov.io/gh/jadball/anri/branch/main/graph/badge.svg)](https://codecov.io/gh/jadball/anri)
[![code style ruff](https://img.shields.io/badge/code%20style-Ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![cross-platform](https://img.shields.io/badge/platform-cross--platform-brightgreen.svg)](https://www.python.org/)
[![JAX Python](https://img.shields.io/badge/code-JAX-blue.svg)](https://github.com/jax-ml/jax)

# What is Anri?
Anri is a Python package for the GPU-accelerated analysis of Diffraction Microstructure Imaging data, such as (Scanning) Three-Dimensional X-Ray Diffraction \[(s)3DXRD\].

The core concept of Anri is as follows:

## JAX-native code
(almost) all of Anri is implemented in [JAX](https://github.com/jax-ml/jax), a Python library for high-performance (e.g. GPU-accelerated) numerical computing.  
The benefit is that Anri will work on any major x86 (and some ARM) CPUs and any recent CUDA-compatible GPU.

## Intensity-aware forward model
At the core of Anri is a high-performance forward model that goes from sample space (e.g. a grid of UBI matrices) to detector space (e.g. \[slow, fast\] coordinates). This can be used to investigate the performance of existing (s)3DXRD analysis packages such as [ImageD11](https://github.com/FABLE-3DXRD/ImageD11) by comparing forward-projected data to the raw data that you measured. Intensities are computed using the structure factors thanks to [Dan's Diffraction](https://github.com/DanPorter/Dans_Diffraction) and accumulate in detector pixels.

## Differentiability
Great effort has been undertaken to ensure that JAX-native Anri functions are differentiable, using the powerful auto-diff capabilities of JAX. This has two obvious use-cases:

### Peak shapes
By expressing instrumental parameters such as incident beam divergence and energy spread as Gaussian distributions, Anri can use the per-peak Jacobians produced by JAX to propagate these parameters into detector space as a covariance matrix in output space, thereby rendering fairly realistic peak shapes that are not just simple detector point spread functions. Therefore, a spread in beam energy (for example) manifests as a radial distribution on the detector. 

### Gradient-aware optimisation (in progress)
Anri will take advantage of the differentiable, intensity-aware forward model to perform iterative refinement of grain maps produced by [ImageD11](https://github.com/FABLE-3DXRD/ImageD11) (and perhaps other programs in the future) to yield refined maps of orientation gradients and strains.

# What can Anri do today?

Anri is in the early stages of development and is not yet ready to use for experimental analysis.  
However, it can already be used today to perform interesting scientific analysis on simulated data.
See the [Tutorials](https://jadball.github.io/anri/tutorials/index.html) and [Examples](https://jadball.github.io/anri/examples/index.html) section of the Documentation for some examples of what Anri can do.

# Dependencies
We currently target all stable releases of Python. Today this is `3.9 - 3.14` on Windows, ubuntu and OSX (ARM and x86).

# Installation at the ESRF
## From source (for developers)
### Clone the repository
```bash
git clone git@github.com:jadball/anri.git anri
cd anri
```
### Set up a mamba environment
```bash
module load mamba
mamba create --prefix=./.conda -c conda-forge python pip setuptools
mamba activate ./.conda
```
### Install build dependencies
```bash
python -m pip install --upgrade pip unidep
```
### Install conda, then pip deps, then the package itself (with `dev` optional deps) as editable.
This gives you CUDA-enabled JAX.
```bash
unidep install .[dev,cuda12] -e
```

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
python -m pip install --upgrade pip unidep
```
### Install conda, then pip deps, then the package itself (with `dev` optional deps) as editable:
```bash
unidep install .[dev] -e
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

# Citing Anri

A paper describing and using Anri is under development. In the meantime, please cite this repository directly, updating the version number as required:
```
@software{anri2026github,
  author = {James A. D. Ball},
  title = {{Anri}: {GPU}-accelerated {D}iffraction {M}icrostructure {I}maging analysis},
  url = {https://github.com/jadball/anri},
  version = {0.1.0},
  year = {2026},
}
```

# Papers Citing Anri
If you use this codebase in your publication, feel free to open a Pull Request to add your work here!

* Ball, J. A. D., Andreasen, J. W., Angelis, S. D., Wright, J. P., & Detlefs, C. (2026, July 9). Multi-Beam 3DXRD. IOP Conference Series: Materials Science and Engineering. 46th Risø International Symposium on Materials Science: Characterization of evolving microstructures in metals, DTU Risø Campus, Roskilde, Denmark. Accepted for publication.

# Credits
Anri is currently primarily developed by James A. D. Ball. Many sections of Anri are based on [ImageD11](https://github.com/FABLE-3DXRD/ImageD11) - I recommend you check it out!

# Acknowledgements
We are grateful to Carsten Detlefs, Axel Henningsson, and Jon P. Wright for their invaluable advice during the development of Anri.

