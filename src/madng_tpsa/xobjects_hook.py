"""Expose MAD-NG TPSA headers to Xobjects through an entrypoint hook."""

from madng_tpsa.paths import include_dir


def get_build_info():
    """Return headers required to compile TPSA kernels."""
    return {
        'include_dirs': [include_dir()],
    }
