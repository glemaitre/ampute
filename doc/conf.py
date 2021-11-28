# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("sphinxext"))
from github_link import make_linkcode_resolve  # noqa: E402


# -- Project information -----------------------------------------------------

project = "ampute"
copyright = "2021, G. Lemaitre"
author = "G. Lemaitre"

# The full version, including alpha/beta/rc tags
from ampute import __version__  # noqa: E402

version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "numpydoc",
    "sphinx_issues",
    "sphinx_gallery.gen_gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for math equations -----------------------------------------------

imgmath_image_format = "svg"
# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get("NO_MATHJAX"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
    mathjax_path = ""
else:
    extensions.append("sphinx.ext.mathjax")
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"
html_title = f"Version {version}"
# html_favicon = "_static/img/favicon.ico"
# html_logo = "_static/img/logo_wide.png"
html_style = "css/ampute.css"
html_css_files = [
    "css/ampute.css",
]
html_sidebars = {
    "changelog": [],
}

html_theme_options = {
    "external_links": [],
    "github_url": "https://github.com/glemaitre/ampute",
    "use_edit_page_button": True,
    "show_toc_level": 1,
}

html_context = {
    "github_user": "glemaitre",
    "github_repo": "ampute",
    "github_version": "main",
    "doc_path": "doc",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = "amputedoc"

# -- Options for autodoc ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
}

# generate autosummary even if no references
autosummary_generate = True

# -- Options for numpydoc -----------------------------------------------------

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False

# -- Options for intersphinx --------------------------------------------------

# intersphinx configuration
intersphinx_mapping = {
    "python": (
        "https://docs.python.org/{.major}".format(sys.version_info),
        None,
    ),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
}

# -- Options for sphinx-gallery -----------------------------------------------

# Generate the plot for the gallery
plot_gallery = True

# sphinx-gallery configuration
sphinx_gallery_conf = {
    "doc_module": "ampute",
    "backreferences_dir": os.path.join("references/generated"),
    "show_memory": True,
    "reference_url": {"ampute": None},
}

# -- Options for github link for what's new -----------------------------------

# Config for sphinx_issues
issues_uri = "https://github.com/glemaitre/ampute/issues/{issue}"
issues_github_path = "glemaitre/ampute"
issues_user_uri = "https://github.com/{user}"

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "ampute",
    "https://github.com/glemaitre/"
    "ampute/blob/{revision}/"
    "{package}/{path}#L{lineno}",
)

# -- Additional temporary hacks -----------------------------------------------

# Temporary work-around for spacing problem between parameter and parameter
# type in the doc, see https://github.com/numpy/numpydoc/issues/215. The bug
# has been fixed in sphinx (https://github.com/sphinx-doc/sphinx/pull/5976) but
# through a change in sphinx basic.css except rtd_theme does not use basic.css.
# In an ideal world, this would get fixed in this PR:
# https://github.com/readthedocs/sphinx_rtd_theme/pull/747/files


def setup(app):
    app.add_js_file("js/copybutton.js")
    app.add_css_file("basic.css")
