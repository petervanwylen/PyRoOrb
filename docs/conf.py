"""Sphinx configuration for the PyRoOrb API reference.

Build: ``pip install -e ".[docs]"`` then ``sphinx-build -b html docs docs/_build/html``.
REBOUND and ASSIST are mocked so the reference builds without compiling them; autodoc only
reads docstrings and signatures, it never runs a fit.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pyroorb  # noqa: E402

project = "PyRoOrb"
author = "Peter VanWylen"
copyright = "2026, Peter VanWylen"
release = pyroorb.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
]

autodoc_mock_imports = ["rebound", "assist"]
autodoc_member_order = "bysource"
numpydoc_show_class_members = False      # api.rst lists the supported members explicitly
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
}

exclude_patterns = ["_build", "*.md"]
html_theme = "furo"
html_title = f"PyRoOrb {release}"
html_logo = "logo.png"
html_static_path = []
