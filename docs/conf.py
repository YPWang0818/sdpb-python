"""Sphinx configuration for sdpb-python."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "sdpb-python"
author = "Yi-Ping Wang"
copyright = "2026, Yi-Ping Wang"
release = "0.2.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
autoclass_content = "class"
napoleon_google_docstring = True
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "mpmath": ("https://mpmath.org/doc/current", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "API_DESIGN.md", "BUILDING.md"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

html_theme = "furo"
html_title = "sdpb-python"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/YPWang0818/sdpb-python",
    "source_branch": "main",
    "source_directory": "docs/",
}
