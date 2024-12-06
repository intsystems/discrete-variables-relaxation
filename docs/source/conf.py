# Configuration file for the Sphinx documentation builder.

import os
import sys

from relaxit import __version__

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------

project = "Just Relax It"
copyright = "2024, Daniil Dorin, Igor Ignashin, Nikita Kiselev, Andrey Veprikov"
author = "Daniil Dorin, Igor Ignashin, Nikita Kiselev, Andrey Veprikov"

version = __version__
master_doc = "index"

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'myst_parser'
]

highlight_language = 'python'

autodoc_mock_imports = ["numpy", "scipy", "sklearn"]

templates_path = ["_templates"]
exclude_patterns = []
html_extra_path = []

html_context = {
    "display_github": True,
    "github_user": "Intelligent-Systems-Phystech",
    "github_repo": "discrete-variables-relaxation",
    "github_version": "main",
    "conf_py_path": "./doc/",
}

# -- Options for HTML output -------------------------------------------------

html_logo = "_static/img/logo-small.png"

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 3,
    "logo_only": True,
}

html_static_path = ["_static"]
html_css_files = [
    'css/relaxit.css',
]

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}
