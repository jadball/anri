# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import logging
import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

sys.path.insert(0, os.path.abspath("../../"))

from anri import VERSION, VERSION_SHORT  # noqa: E402

# -- Project information -----------------------------------------------------

project = "anri"
copyright = f"{datetime.today().year}, James A. D. Ball"
author = "James A. D. Ball"
version = VERSION_SHORT
release = VERSION

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    # "sphinx_gallery.gen_gallery",
    "nbsphinx",
]

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "ImageD11": ("https://imaged11.readthedocs.io/en/latest/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["../_templates"]

# Myst-parser options
# Tell myst-parser to assign header anchors for h1-h3.
myst_heading_anchors = 4
suppress_warnings = ["myst.header"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

# HTML stuff
html_theme = "pydata_sphinx_theme"
html_title = f"anri v{VERSION}"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["css/custom.css"]

html_favicon = "_static/favicon.ico"

# html_theme_options = {
#     "footer_icons": [
#         {
#             "name": "GitHub",
#             "url": "https://github.com/jadball/anri",
#             "html": """
#                 <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
#                     <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
#                 </svg>
#             """,  # noqa: E501
#             "class": "",
#         },
#     ],
# }

# -- sphinx.ext.autodoc
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
# By default, sort documented members by type within classes and modules.
autosummary_ignore_module_all = False
autosummary_imported_members = True
autodoc_typehints_format = "short"
autodoc_default_options = {
    "show-inheritance": True,
}
autosummary_generate = True

# sphinx_autodoc_typehints
# Include default values when documenting parameter types.
typehints_defaults = "comma"

# -- sphinx.ext.mathjax
mathjax3_config = {
    'loader': {'load': ['[tex]/ams']},
    'tex': {
        'macros': {
            'bm': [r'\boldsymbol{#1}',1],
            'vec': [r'\mathbf{#1}', 1],
            'matr': [r'\bm{\mathit{#1}}', 1],
            'tens': [r'\bm{#1}', 1],
            'abs': [r'\lvert #1 \rvert', 1],
        }
    }
}

# -- nbsphinx
# https://nbsphinx.readthedocs.io/en/0.8.0/never-execute.html
nbsphinx_execute = "always"  # auto, always, never
nbsphinx_allow_errors = True
nbsphinx_execute_arguments = [
    "--InlineBackend.rc=figure.facecolor='w'",
    "--InlineBackend.rc=font.size=15",
]

# -- sphinx.ext.napoleon
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_preprocess_types = True
typehints_use_rtype = False

# -- Hack to get rid of stupid warnings from sphinx_autodoc_typehints --------
class ShutupSphinxAutodocTypehintsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if "Cannot resolve forward reference" in record.msg:
            return False
        return True
logging.getLogger("sphinx.sphinx_autodoc_typehints").addFilter(ShutupSphinxAutodocTypehintsFilter())


import docutils.nodes

# Define a "do nothing" visitor for the unknown node
def skip_node(self, node):
    pass

# Patch MathDollarReplacer to ignore GalleryToc
from sphinx_math_dollar.extension import MathDollarReplacer
MathDollarReplacer.visit_GalleryToc = skip_node


# import sphinx.ext.apidoc

# def run_apidoc(_):
#     # This is exactly where conf.py lives: Anri/docs/source/
#     conf_dir = os.path.abspath(os.path.dirname(__file__))
    
#     # Force the output into Anri/docs/source/reference/
#     api_output_dir = os.path.join(conf_dir, "reference")
    
#     # Project root is two levels up from docs/source: Anri/
#     project_root = os.path.abspath(os.path.join(conf_dir, "../../"))
    
#     # The actual code folder: Anri/anri/
#     package_dir = os.path.join(project_root, "anri")
    
#     # The sandbox to ignore: Anri/anri/sandbox/
#     sandbox_dir = os.path.join(package_dir, "sandbox")

#     argv = [
#         "--force",
#         # "--module-first",
#         "-o", api_output_dir,
#         package_dir,
#         sandbox_dir,
#     ]
    
#     sphinx.ext.apidoc.main(argv)

# def setup(app):
#     app.connect('builder-inited', run_apidoc)

# source_suffix = [".rst", ".md"]
# autodoc_member_order = "groupwise"