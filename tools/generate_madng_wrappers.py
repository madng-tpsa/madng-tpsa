"""Generate native MAD-NG panic wrappers without importing the built library."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src' / 'madng_tpsa'

sys.path.insert(0, str(PACKAGE))

from madng_definitions import FUNCTIONS  # noqa: E402
from madng_function import unique_signatures  # noqa: E402


def generate_headers(functions):
    """Generate the native header and source from distinct signatures."""
    signatures = unique_signatures(functions)
    header = '\n\n'.join(sig.native_header() for sig in signatures)
    source = '\n\n'.join(sig.native_source() for sig in signatures)
    return header, source


def main() -> None:
    """Write generated C files to the requested build directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    header, source = generate_headers(FUNCTIONS)
    warning = '/* Automatically generated. Do not edit. */\n'
    (output / 'madng_panic_generated.h').write_text(warning + header + '\n', encoding='utf-8')
    (output / 'madng_panic_generated.inc').write_text(warning + source + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
