# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fglib2'
copyright = '2023, Tom Schierenbeck, Abdelrhman Bassiouny, Alessandro Santonicola'
author = 'Tom Schierenbeck, Abdelrhman Bassiouny, Alessandro Santonicola'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'autoapi.extension',
    'nbsphinx',
    'sphinx_gallery.load_style',
]
templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

# -- AutoAPI configuration ---------------------------------------------------
autoapi_dirs = ['../src']
