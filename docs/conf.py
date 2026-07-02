"""Sphinx configuration for the pgd2 documentation."""

from __future__ import annotations

import importlib.metadata

# -- Project information -----------------------------------------------------

project = "pgd2"
author = "Brandon Lukas"
copyright = "2026, Brandon Lukas"

try:
    release = importlib.metadata.version("pgd2")
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

myst_enable_extensions = ["colon_fence", "deflist"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------

# Swiss / International Typographic Style: grotesque sans, flush-left,
# white + near-black + a single red accent, square geometry, no gradients.
html_theme = "furo"
html_title = f"pgd2 {version}"
html_static_path = ["_static"]
html_css_files = ["css/swiss.css"]

_SWISS_RED = "#e2001a"
_SANS = (
    '"Inter", "Helvetica Neue", Helvetica, Arial, '
    '"Liberation Sans", sans-serif'
)
_MONO = (
    '"IBM Plex Mono", "SFMono-Regular", ui-monospace, '
    '"Cascadia Mono", Menlo, Consolas, monospace'
)

html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "font-stack": _SANS,
        "font-stack--monospace": _MONO,
        "color-brand-primary": _SWISS_RED,
        "color-brand-content": _SWISS_RED,
        "color-brand-visited": _SWISS_RED,
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#ffffff",
        "color-foreground-primary": "#0a0a0a",
        "color-foreground-secondary": "#4a4a4a",
        "color-foreground-muted": "#6a6a6a",
        "color-code-background": "#f4f4f4",
        "color-code-foreground": "#0a0a0a",
        "color-sidebar-background": "#ffffff",
        "color-sidebar-background-border": "#0a0a0a",
        "color-sidebar-search-border": "#0a0a0a",
        "color-highlight-on-target": "#fdeaec",
        "color-api-background": "#f4f4f4",
    },
    "dark_css_variables": {
        "font-stack": _SANS,
        "font-stack--monospace": _MONO,
        "color-brand-primary": "#ff3b2e",
        "color-brand-content": "#ff3b2e",
        "color-brand-visited": "#ff3b2e",
        "color-background-primary": "#0a0a0a",
        "color-background-secondary": "#0a0a0a",
        "color-foreground-primary": "#f2f2f2",
        "color-foreground-secondary": "#b8b8b8",
        "color-foreground-muted": "#8a8a8a",
        "color-code-background": "#161616",
        "color-code-foreground": "#f2f2f2",
        "color-sidebar-background": "#0a0a0a",
        "color-sidebar-background-border": "#f2f2f2",
        "color-sidebar-search-border": "#f2f2f2",
        "color-highlight-on-target": "#2a1416",
        "color-api-background": "#161616",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/brandonlukas/pgd2",
            "html": "GitHub",
            "class": "swiss-footer-link",
        },
    ],
}
