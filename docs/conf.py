# Configuration file for the Sphinx documentation builder.
# Full list of built-in configuration values: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import sphinx_rtd_theme

# Add the project root directory to sys.path to make autodoc work correctly.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "GANsForVirtualEye"
copyright = "2024, Shailendra Bhandari"
author = "Shailendra Bhandari"
release = "3/12/2024"  # Version or release date

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.imgmath",
    "sphinx.ext.napoleon",
]

# Specify the root document for Sphinx
master_doc = "index"  # Root document name (typically 'index')

# Specify templates and source file types
templates_path = ["_templates"]
source_suffix = ".rst"  # Only use .rst files

# Exclude build and OS-specific files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Language configuration
language = "en"
keep_warnings = True  # Show warnings as system messages in output


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = []  # Path for static files (e.g., CSS)

# -- Options for HTML Help output --------------------------------------------
htmlhelp_basename = "GANsForVirtualEye"  # Base name for HTML help builder

# -- Options for Epub output -------------------------------------------------
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ["search.html"]

# -- Options for Intersphinx -------------------------------------------------
# Configure intersphinx to reference the Python standard library documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None)
}