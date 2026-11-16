"""Tool to generate the Madng CFFI wrappers. Can be run without a full madng-tpsa import."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src' / 'madng_tpsa'
NATIVE = ROOT / 'native'

# Import the generator without importing madng_tpsa itself.
# The shared library might not exist yet!
sys.path.insert(0, str(PACKAGE))

from madng_definitions import DEFINITIONS  # noqa: E402
from madng_function import MadngFunction, generate_headers  # noqa: E402


def main():
    """Generate the Madng CFFI wrappers."""
    functions = [
        MadngFunction(declaration, **options) for declaration, options in DEFINITIONS.values()
    ]

    cdef, header, source = generate_headers(functions)

    (NATIVE / 'madng_panic_generated.h').write_text(
        '/* Automatically generated. Do not edit. */\n' + header + '\n'
    )

    (NATIVE / 'madng_panic_generated.inc').write_text(
        '/* Automatically generated. Do not edit. */\n' + source + '\n'
    )

    (PACKAGE / '_generated_cdef.py').write_text(
        '# Automatically generated. Do not edit.\n# ruff: noqa\n\nCDEF = """\n' + cdef + '\n"""\n'
    )


if __name__ == '__main__':
    main()
