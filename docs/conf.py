import os
import sys

# make the optiml package importable for autodoc
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'OptiML'
author = 'Donato Meoli'
copyright = '2021, Donato Meoli'

release = '1.7'
version = '1.7'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',       # pull the docstrings out of the source
    'sphinx.ext.autosummary',   # generate summary tables for the API
    'sphinx.ext.napoleon',      # understand both the NumPy-style (ml) and Google docstrings
    'sphinx.ext.viewcode',      # add links to the highlighted source
    'sphinx.ext.mathjax',       # render the LaTeX in the docstrings
    'sphinx.ext.intersphinx',   # cross-reference numpy/scipy/sklearn objects
]

autosummary_generate = True
autodoc_member_order = 'bysource'
# render both the class docstring (the algorithm description) and the __init__
# docstring (the :param list of the constructor) on each class page
autoclass_content = 'both'
autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'show-inheritance': True,
}

# the optimizers use the reStructuredText `:param:` style while the sklearn-compatible
# ml estimators use the NumPy docstring style; napoleon lets both render uniformly
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
