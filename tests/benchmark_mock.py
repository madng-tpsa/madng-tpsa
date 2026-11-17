"""Illustrative Python overhead comparison using the compiled mock C library.

Run: python benchmark_mock.py
Requires pytest, cffi and a C compiler. No MAD-NG installation is needed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from time import perf_counter

from test_madng_function import HEADERS, Descriptor, TpsaTensor, build_native, make_series

from madng_tpsa.madng_function import MadngFunction


def measure(fn, repetitions=30_000):
    """Measure the average execution time of a function in microseconds."""
    for _ in range(2_000):
        fn()
    start = perf_counter()
    for _ in range(repetitions):
        fn()
    return (perf_counter() - start) * 1e6 / repetitions


def _run_benchmark(tmp: Path):
    ffi, lib, fs = build_native(tmp)
    descriptor = Descriptor(ffi, lib)
    runtime = (ffi, lib, fs)
    left = TpsaTensor(make_series(runtime, [10, 20], descriptor=descriptor), ffi)
    right = TpsaTensor(make_series(runtime, [3, 4], descriptor=descriptor), ffi)
    output = TpsaTensor(make_series(runtime, [0, 0], descriptor=descriptor), ffi)
    fn = fs['mad_tpsa_liebra']
    assert fn._native_real is not None
    scalar_fn, protected = fn._native_real
    scalar_fn, protected = fn._native_real
    generic = MadngFunction(HEADERS[1], ffi=ffi, lib=lib)

    print('Mock CFFI benchmark (µs per call; lower is faster)')  # noqa: T201
    for name, callable_ in (
        (
            'Raw protected CFFI',
            lambda: protected(
                scalar_fn, 2, left.pointer_array, right.pointer_array, output.pointer_array
            ),
        ),
        ('Specialised/preallocated', lambda: fn.fast_call(left, right, output)),
        ('Generic/preallocated', lambda: generic(left, right, output)),
        ('Specialised/auto-output', lambda: fn.fast_call(left, right)),
    ):
        print(f'{name:<27} {measure(callable_):.2f}')  # noqa: T201


def main():
    """Run the mock CFFI benchmark."""
    with tempfile.TemporaryDirectory(prefix='compiled') as tmp:
        _run_benchmark(Path(tmp))


if __name__ == '__main__':
    main()
