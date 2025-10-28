# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Wavepacket'
copyright = '2024-%Y, Ulf Lorenz'
author = 'Ulf Lorenz'
release = '0.2'

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

