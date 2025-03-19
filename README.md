# anri
Anri is a Python package for analysing Three-Dimensional X-Ray Diffraction data.

# Dependencies
We currently target all stable releases of Python. Today this is `3.9 - 3.13` on Windows, Mac and OSX (ARM and x86)

# Installation
## From Conda
Coming soon!
## From source (for developers)
Anri will (eventually) rely on packages from both `conda` and `pip`.  
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
## IDE
[Visual Studio Code](https://code.visualstudio.com/) is recommended for development.  
## Linting, formatting and type checking
`anri` uses `ruff` to lint and format, and `mypy` for type-checking.  
All Python functions and files (outside of `anri/sandbox`) must conform for the GitHub CI tests to pass.  
With `Visual Studio Code` you have easy access to automatic lint-on-save and format-on-save via extensions.  
Inside `.vscode` you have a `settings.sample.json` which, if you're happy with, you can rename to `settings.json` to apply my recommended per-project settings for this repository.  
You also have `extensions.json` containing my recommended extensions (including `ruff`) which `Visual Studio Code` should prompt you to install automatically.