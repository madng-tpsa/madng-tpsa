"""Locate the packaged MAD-NG TPSA shared library and public headers.

Use :func:`core_library` and :func:`include_dir` when embedding the native TPSA
library or compiling code against its C API. The build installs these artifacts
alongside the Python package.
"""

from __future__ import annotations

import sys
from pathlib import Path

LIB_BASENAME = 'madng_tpsa'
_PKG = Path(__file__).resolve().parent
_LIB = _PKG / 'lib'
_INCLUDE = _PKG / 'include'


def _library_name() -> str:
    """Return the canonical name of the shared object file for the current platform."""
    match sys.platform:
        case 'darwin':
            return f'lib{LIB_BASENAME}.dylib'
        case 'linux':
            return f'lib{LIB_BASENAME}.so'

    raise RuntimeError(f'Platform {sys.platform} not supported for MAD-NG TPSA')


def core_library() -> str:
    """Absolute path to the ``madng_tpsa`` shared library."""
    path_to_so = Path(lib_dir()) / _library_name()
    return str(path_to_so)


def lib_dir() -> str:
    """Absolute path to the directory containing the ``madng_tpsa`` shared library."""
    path_to_so = _LIB / _library_name()

    if not path_to_so.exists():
        raise RuntimeError(
            f'GTPSA library {path_to_so} does not exist. Was the package built correctly?',
        )

    return str(_LIB)


def include_dir() -> str:
    """Directory holding the public MAD-NG GTPSA headers."""
    test_file = 'mad_tpsa.hpp'

    if not (_INCLUDE / test_file).exists():
        raise RuntimeError(
            f'GTPSA include headers not found in {_INCLUDE}. Was the package built correctly?',
        )

    return str(_INCLUDE)
