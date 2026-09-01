import os
import sys

conf_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(conf_dir, "..", "..", "src")))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MotifCompendium'
author = 'SalilDeshpande, ChangMYun'
release = '1.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",                  # Notebook + Markdown support
    "sphinx.ext.autodoc",       # Generate API docs from docstrings
    "sphinx.ext.autosummary",   # Generate summary pages and stubs
    "sphinx.ext.napoleon",      # Google/NumPy style docstrings
    "sphinx.ext.viewcode",      # Link to source code
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence"
]
nb_execution_mode = "auto"

exclude_patterns = [
    "_build",                    # ignore build folder
    "**.ipynb_checkpoints",      # ignore notebook checkpoints
    "**/__pycache__",            # ignore Python cache
]

templates_path = ['_templates']
html_static_path = ['_static']

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
add_module_names = False

# Preserve the order of members as they appear in source files
# Options: 'alphabetical' (default), 'bysource', 'groupwise'
autodoc_member_order = 'bysource'

# Automatically generate autosummary pages (one-page-per-module stubs)
autosummary_generate = True
autosummary_imported_members = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
