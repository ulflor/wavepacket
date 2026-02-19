# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Wavepacket'
copyright = '2024-%Y, Ulf Lorenz'
author = 'Ulf Lorenz'
release = '0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.imgmath",
    "sphinx.ext.inheritance_diagram",
    "autoapi.extension",
    "numpydoc",
    "myst_nb"
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# ------ AutoApi configuration
autoapi_dirs = ["../src/wavepacket"]

autoapi_options = ['members', 'undoc-members', 'imported-members',
                   'show-inheritance', 'show-module-summary']


# Unless we prefix _every_ module by "_", we get every symbol twice: Once in the
# exposing package, once in the defining module. This here skips all module docs.
def skip_modules(app, what, name, obj, skip, options):
    if what == "module":
        skip = True

    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_modules)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'


# -- Options for MyST-NB

# Individual calculations take relatively long to finish. The good solution is to keep the demos
# simple, but this is not always simple to do.
nb_execution_timeout = 90

# We use $ and $$ in markdown notebooks, because after conversion to raw (.ipynb)
# Jupyter notebooks, standard Jupyter setups seem to understand that these are
# equations and render them. There seems to be a (MathJax?) problem at least in some
# Jupyter setups that does not handle linebreaks unless we use amsmath environments, so we do that.
myst_enable_extensions = ["dollarmath", "amsmath"]
