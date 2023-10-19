# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fglib2'
copyright = '2023, Tom Schierenbeck, Abdelrhman Bassiouny'
author = 'Tom Schierenbeck, Abdelrhman Bassiouny'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['autoapi.extension',
              'sphinx_rtd_theme',
              "nbsphinx",
              "sphinx_gallery.load_style"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

# -- AutoAPI configuration ---------------------------------------------------
autoapi_dirs = ['../src']
