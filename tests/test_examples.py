"""Tests that the repository examples run as standalone scripts."""

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

EXAMPLES = sorted((ROOT / 'examples').glob('[0-9][0-9]_*.py'))


@pytest.mark.parametrize(
    'example',
    EXAMPLES,
    ids=lambda path: path.name,
)
def test_example_runs(example):
    # 01 demonstrates the optional Jupyter-oriented math formatter.
    if example.name == '01_tpsa_basics.py' and find_spec('IPython') is None:
        pytest.skip('01_tpsa_basics.py math formatting requires IPython')

    subprocess.run(  # noqa: S603
        [sys.executable, str(example)],
        cwd=ROOT,
        check=True,
    )
