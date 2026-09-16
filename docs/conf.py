"""Sphinx configuration for madng-tpsa."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))

project = 'madng-tpsa'
copyright = '2026, MAD-NG TPSA contributors'  # noqa: A001

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented_params'
autoclass_content = 'both'
autodoc_member_order = 'bysource'
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_preprocess_types = False
html_theme = 'sphinx_rtd_theme'
