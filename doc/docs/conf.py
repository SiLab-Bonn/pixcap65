# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# will need the source code
print(os.path.abspath('../../pixcap65'))
sys.path.insert(0, os.path.abspath('../../pixcap65'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    "myst_parser",
]

# The master toctree document.
master_doc = 'index'

project = 'Pixcap65'
# noinspection PyShadowingBuiltins
copyright = '2021, SiLAB'
author = 'Hans Krüger, Evelyn Kimmerle, Dominik Fischer'
release = '[0.1.0]'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# options for latex output?

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
