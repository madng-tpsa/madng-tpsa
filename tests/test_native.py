"""Native protected-call recovery and isolation, without bypassing Python API validation.


Set MADNG_TPSA_NATIVE_STRESS=1 to run the long allocation stress test.
The RSS assertion is intentionally generous: malloc arenas and native caches vary.

This complements test_native_recovery_stress.py. It deliberately uses known
error-producing inputs rather than undefined pointer tricks, and keeps all
TPSA objects backing CFFI pointers alive. The opt-in RSS test runs in child
processes so a native crash or leak cannot corrupt the normal pytest worker.

These are *early* native failure paths, not evidence that arbitrary native
allocations are cleaned up when longjmp bypasses native cleanup code.
"""

from __future__ import annotations

import gc
import importlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from pathlib import Path
from threading import Barrier
from time import perf_counter
from typing import TYPE_CHECKING

import pytest

from madng_tpsa import Descriptor, TpsaError, TpsaMap
from madng_tpsa._cffi import ffi, lib

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Any

    from madng_tpsa import Tpsa


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHILD_FLAG = '--native-recovery-child'


def _pointers(series: Sequence[Tpsa]) -> Any:
    """Callers must retain references to every TPSA owning the native pointers."""
    return ffi.new('void*[]', [component.ptr for component in series])


def _native_inverse_case() -> tuple[list[Tpsa], list[Tpsa], list[Tpsa], Any, Any, Any]:
    """Return native arrays *and* the TPSAs owning their C pointers."""
    descriptor = Descriptor(variables=('stress_q', 'stress_p'), order=4)
    q, p = descriptor.vars()
    failure = [q, descriptor.zero()]
    success = [q + p**2, p]
    output = [descriptor.zero(), descriptor.zero()]
    return failure, success, output, _pointers(failure), _pointers(success), _pointers(output)


def _inverse_call(inputs: Any, outputs: Any) -> int:
    return int(lib.madng_tpsa_protected_map_inverse_call(lib.mad_tpsa_minv, 2, inputs, 2, outputs))


def _ptrs(series: Iterable[Tpsa]) -> Any:
    return ffi.new('void*[]', [item.ptr for item in series])


def _rss_bytes() -> int:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak * 1024)


def test_thousands_of_protected_native_failures_preserve_next_operation() -> None:
    iterations = int(os.environ.get('MADNG_TPSA_NATIVE_ITERATIONS', '250000'))
    d = Descriptor(variables=('rec_q', 'rec_p'), order=4)
    q, p = d.vars()
    singular = [q, d.zero()]
    good = [q + p**2, p]
    outputs = [d.zero(), d.zero()]
    singular_ptrs, good_ptrs, output_ptrs = map(_ptrs, (singular, good, outputs))
    gc.collect()
    before = _rss_bytes()
    for i in range(iterations):
        status = lib.madng_tpsa_protected_map_inverse_call(
            lib.mad_tpsa_minv, 2, singular_ptrs, 2, output_ptrs
        )
        assert status != 0
        if i % 100 == 0:
            message = ffi.string(lib.madng_tpsa_last_error_message()).decode()
            assert message
            # A valid native operation must still succeed after repeated longjmps.
            assert (
                lib.madng_tpsa_protected_map_inverse_call(
                    lib.mad_tpsa_minv, 2, good_ptrs, 2, output_ptrs
                )
                == 0
            )
            assert outputs[0] == q - p**2
    gc.collect()
    # Advisory leak bound rather than exact leak detection; OS allocators retain arenas.
    assert _rss_bytes() - before < 96 * 1024 * 1024


def test_simultaneous_native_errors_keep_thread_local_messages() -> None:
    barrier = Barrier(2)

    def singular_worker() -> list[str]:
        d = Descriptor(variables=('thread_sq', 'thread_sp'), order=3)
        q, _ = d.vars()
        inputs = (q, d.zero())
        inp = _ptrs(inputs)
        outputs = (d.zero(), d.zero())
        out = _ptrs(outputs)
        barrier.wait(timeout=10)
        messages = []
        for _ in range(30):
            status = lib.madng_tpsa_protected_map_inverse_call(lib.mad_tpsa_minv, 2, inp, 2, out)
            assert status != 0
            messages.append(ffi.string(lib.madng_tpsa_last_error_message()).decode())
        return messages

    def incompatible_worker() -> list[str]:
        d1 = Descriptor(variables=('thread_a', 'thread_b'), order=3)
        d2 = Descriptor(variables=('thread_c', 'thread_d'), order=5)
        a = d1.var(1)
        b = d2.var(1)
        out = d1.zero()
        barrier.wait(timeout=10)
        messages = []
        for _ in range(30):
            status = lib.madng_tpsa_protected_binary_call(lib.mad_tpsa_add, a.ptr, b.ptr, out.ptr)
            assert status != 0
            messages.append(ffi.string(lib.madng_tpsa_last_error_message()).decode())
        return messages

    with ThreadPoolExecutor(max_workers=2) as pool:
        singular_future = pool.submit(singular_worker)
        incompatible_future = pool.submit(incompatible_worker)
        singular_messages = singular_future.result(timeout=30)
        incompatible_messages = incompatible_future.result(timeout=30)
    assert all(singular_messages)
    assert all('incompatib' in message.lower() for message in incompatible_messages)
    assert not any('incompatib' in message.lower() for message in singular_messages)

    # Verify Python's public error conversion is still functional.
    d = Descriptor(variables=('post_q', 'post_p'), order=3)
    q, p = d.vars()
    with pytest.raises(ValueError):
        TpsaMap([q, d.zero()]).inverse()
    assert TpsaMap([q + p, p]).inverse().evaluate((2, 3))[0] == pytest.approx(-1)

    # Regular check
    mapping = TpsaMap([q + p**2, p])
    inverse = mapping.inverse()
    assert inverse.post_q == q - p**2
    assert inverse.post_p == p


def test_distinct_native_failures_restore_valid_operations() -> None:
    """Exercise map, unary and binary protected paths, not just one wrapper."""
    descriptor = Descriptor(variables=('recovery_q', 'recovery_p'), order=4)
    q, p = descriptor.vars()
    zero = descriptor.zero()
    scratch = descriptor.zero()

    bad_map = [q, zero]
    output = [descriptor.zero(), descriptor.zero()]
    assert _inverse_call(_pointers(bad_map), _pointers(output)) != 0
    assert 'rank-deficient' in ffi.string(lib.madng_tpsa_last_error_message()).decode()

    assert lib.madng_tpsa_protected_unary_call(lib.mad_tpsa_sqrt, zero.ptr, scratch.ptr) != 0
    assert 'sqrt' in ffi.string(lib.madng_tpsa_last_error_message()).decode()

    assert lib.madng_tpsa_protected_binary_call(lib.mad_tpsa_div, q.ptr, zero.ptr, scratch.ptr) != 0
    assert 'div' in ffi.string(lib.madng_tpsa_last_error_message()).decode()

    with pytest.raises(TpsaError, match='sqrt'):
        zero.sqrt()

    good_map = [q + p**2, p]
    good_output = [descriptor.zero(), descriptor.zero()]
    assert _inverse_call(_pointers(good_map), _pointers(good_output)) == 0
    assert good_output[0] == q - p**2
    assert good_output[1] == p
    assert (1 + q).sqrt().const_part == pytest.approx(1)


def _thread_case(function: str, repetitions: int, barrier: Barrier) -> int:
    """Test thread-local error text and successful calls on the same thread."""
    descriptor = Descriptor(variables=(f'{function}_q', f'{function}_p'), order=3)
    bad = descriptor.zero()
    good = descriptor.constant(1)
    output = descriptor.zero()
    bad_function = lib.mad_tpsa_sqrt if function == 'sqrt' else lib.mad_tpsa_log
    barrier.wait(timeout=15)
    for i in range(repetitions):
        if i % 31 == 0:
            barrier.wait(timeout=15)
        assert lib.madng_tpsa_protected_unary_call(bad_function, bad.ptr, output.ptr) != 0
        message = ffi.string(lib.madng_tpsa_last_error_message()).decode()
        assert function in message, (function, i, message)
        if i % 31 == 0:
            assert lib.madng_tpsa_protected_unary_call(bad_function, good.ptr, output.ptr) == 0
            assert output.const_part == pytest.approx(1 if function == 'sqrt' else 0)
    return repetitions


def _exercise_two_threads(repetitions: int) -> dict[str, int]:
    """Synchronise distinct native error streams in an isolated process."""
    barrier = Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(_thread_case, 'sqrt', repetitions, barrier)
        right = pool.submit(_thread_case, 'log', repetitions, barrier)
        return {'sqrt': left.result(timeout=30), 'log': right.result(timeout=30)}


def test_two_threads_keep_error_messages_independent() -> None:
    """Exercise two native threads in a child process to contain crashes."""
    assert _run_child('threads', 2000) == {'sqrt': 2000, 'log': 2000}


def _sample_rss() -> int:
    """Current RSS, not ru_maxrss, to distinguish transient and persistent memory."""
    psutil = importlib.import_module('psutil')
    return int(psutil.Process().memory_info().rss)


def _memory_child(mode: str, iterations: int) -> dict[str, object]:
    """Run an allocation/control experiment in a disposable Python process."""
    failure, success, output, bad_ptrs, good_ptrs, out_ptrs = _native_inverse_case()
    # Keep Python owners alive until every C call is complete.
    assert len(failure) == len(success) == len(output) == 2
    if mode not in {'fail', 'success'}:
        raise ValueError(f'Unknown memory test mode: {mode}')
    expected_good = (success[0] - 2 * success[1] ** 2, success[1])
    chosen = bad_ptrs if mode == 'fail' else good_ptrs
    expected_status = 1 if mode == 'fail' else 0

    def one(i: int) -> None:
        status = _inverse_call(chosen, out_ptrs)
        if status != expected_status:
            raise AssertionError(f'{mode}: iteration {i}, unexpected native status {status}')
        if mode == 'fail' and i % 1024 == 0:
            message = ffi.string(lib.madng_tpsa_last_error_message()).decode()
            if 'rank-deficient' not in message:
                raise AssertionError(f'wrong error message at {i}: {message}')
        if i % 1024 == 0:
            # Immediately follow failed/successful calls by a valid operation.
            if _inverse_call(good_ptrs, out_ptrs) != 0:
                raise AssertionError(f'valid inverse failed after {mode} iteration {i}')
            if output[0] != expected_good[0] or output[1] != expected_good[1]:
                raise AssertionError(f'inverse coefficients corrupted after {mode} iteration {i}')

    # Warm up lazy native allocations, LAPACK, CFFI, and Python's allocator.
    for i in range(8192):
        one(i)
    gc.collect()
    samples = [_sample_rss()]
    start = perf_counter()
    blocks = 8
    for block in range(blocks):
        for i in range(block * iterations // blocks, (block + 1) * iterations // blocks):
            one(i)
        gc.collect()
        samples.append(_sample_rss())
        # Stop a runaway native leak before it consumes excessive memory.
        if samples[-1] - samples[0] > 128 * 1024 * 1024:
            raise MemoryError(f'{mode}: RSS growth exceeded 128 MiB: {samples}')
    return {
        'mode': mode,
        'iterations': iterations,
        'rss_bytes': samples,
        'seconds': perf_counter() - start,
    }


def _run_child(mode: str, iterations: int) -> dict[str, Any]:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(Path(__file__).resolve()), _CHILD_FLAG, mode, str(iterations)],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (result.stderr, result.stdout)
    return json.loads(result.stdout)


def test_native_rss_growth_against_successful_control() -> None:
    """RSS trend, benchmarked against equally repeated valid native calls."""
    if find_spec('psutil') is None:
        pytest.skip('Install optional psutil to measure current native-process RSS')
    iterations = int(os.environ.get('MADNG_TPSA_NATIVE_ITERATIONS', '250000'))
    if iterations < 1000:
        error = 'MADNG_TPSA_NATIVE_ITERATIONS must be at least 1000'
        raise ValueError(error)

    control = _run_child('success', iterations)
    failing = _run_child('fail', iterations)
    rss_control = control['rss_bytes']
    rss_failing = failing['rss_bytes']
    assert isinstance(rss_control, list) and isinstance(rss_failing, list)
    assert len(rss_control) == len(rss_failing) == 9
    growth_control = rss_control[-1] - rss_control[0]
    growth_failing = rss_failing[-1] - rss_failing[0]
    print(  # noqa: T201
        f'native RSS: {iterations:,} calls per child; '
        f'control {growth_control / 1048576:.2f} MiB, '
        f'failures {growth_failing / 1048576:.2f} MiB; '
        f'failure samples {[round(v / 1048576, 2) for v in rss_failing]}'
    )
    # Guardrail, not a proof of leak freedom. Avoid false positives from the
    # native allocator retaining pools (especially on macOS).
    assert growth_failing <= max(16 * 1024 * 1024, growth_control + 16 * 1024 * 1024)
    late_fail = rss_failing[-1] - rss_failing[4]
    late_control = rss_control[-1] - rss_control[4]
    assert late_fail <= max(8 * 1024 * 1024, late_control + 8 * 1024 * 1024)


if __name__ == '__main__' and len(sys.argv) == 4 and sys.argv[1] == _CHILD_FLAG:
    # Exactly one machine-readable line; do not allow pytest fixture setup to
    # pollute measurements or output in the disposable child process.
    if sys.argv[2] == 'threads':
        report = _exercise_two_threads(int(sys.argv[3]))
    else:
        report = _memory_child(sys.argv[2], int(sys.argv[3]))
    print(json.dumps(report))  # noqa: T201


def _ptrs(series):
    return ffi.new('void*[]', [item.ptr for item in series])


def _zeros(descriptor, count):
    return [descriptor.zero() for _ in range(count)]


def test_map_compose():
    descriptor = Descriptor(2, 4)
    x, y = descriptor.vars()
    # A(x, y) = (x + y^2, y)
    left = [x + y**2, y]
    # B(x, y) = (x, y + x^2)
    right = [x, y + x**2]
    output = _zeros(descriptor, 2)
    status = lib.madng_tpsa_protected_map_binary_call(
        lib.mad_tpsa_compose,
        2,
        _ptrs(left),
        2,
        _ptrs(right),
        _ptrs(output),
    )
    assert status == 0
    # compose(A, B) = A(B(x, y))
    expected = [x + (y + x**2) ** 2, y + x**2]
    assert output[0] == expected[0]
    assert output[1] == expected[1]


def test_map_inverse():
    descriptor = Descriptor(2, 4)
    x, y = descriptor.vars()
    map_ = [x + y**2, y]
    output = _zeros(descriptor, 2)
    status = lib.madng_tpsa_protected_map_inverse_call(
        lib.mad_tpsa_minv,
        2,
        _ptrs(map_),
        2,
        _ptrs(output),
    )
    assert status == 0
    assert output[0] == x - y**2
    assert output[1] == y


def test_map_inverse_composes_to_identity():
    descriptor = Descriptor(2, 4)
    x, y = descriptor.vars()
    map_ = [x + y**2, y + x**3]
    inverse = _zeros(descriptor, 2)
    status = lib.madng_tpsa_protected_map_inverse_call(
        lib.mad_tpsa_minv,
        2,
        _ptrs(map_),
        2,
        _ptrs(inverse),
    )
    assert status == 0
    composed = _zeros(descriptor, 2)
    status = lib.madng_tpsa_protected_map_binary_call(
        lib.mad_tpsa_compose,
        2,
        _ptrs(map_),
        2,
        _ptrs(inverse),
        _ptrs(composed),
    )
    assert status == 0
    assert composed[0] == x
    assert composed[1] == y


def test_vector_to_field():
    descriptor = Descriptor(2, 3)
    q, p = descriptor.vars()
    generator = (q**2 + p**2) / 2
    field = _zeros(descriptor, 2)
    status = lib.madng_tpsa_protected_scalar_to_map_call(
        lib.mad_tpsa_vec2fld,
        2,
        generator.ptr,
        _ptrs(field),
    )
    assert status == 0
    assert field[0] == -p
    assert field[1] == q


def test_field_to_vector_inverts_vector_to_field():
    descriptor = Descriptor(2, 4)
    q, p = descriptor.vars()
    generator = q**3 / 3 + q * p**2 + p**4 / 4
    field = _zeros(descriptor, 2)
    status = lib.madng_tpsa_protected_scalar_to_map_call(
        lib.mad_tpsa_vec2fld,
        2,
        generator.ptr,
        _ptrs(field),
    )
    assert status == 0
    recovered = descriptor.zero()
    status = lib.madng_tpsa_protected_map_to_scalar_call(
        lib.mad_tpsa_fld2vec,
        2,
        _ptrs(field),
        recovered.ptr,
    )
    assert status == 0
    assert recovered == generator


def test_lie_bracket_with_itself_is_zero():
    descriptor = Descriptor(2, 3)
    q, p = descriptor.vars()
    field = [q * p, q**2 - p**2]
    output = _zeros(descriptor, 2)
    status = lib.madng_tpsa_protected_map_pair_call(
        lib.mad_tpsa_liebra,
        2,
        _ptrs(field),
        _ptrs(field),
        _ptrs(output),
    )
    assert status == 0
    assert output[0].is_zero()
    assert output[1].is_zero()


def test_exp_poisson_zero_field_leaves_map_unchanged():
    descriptor = Descriptor(2, 4)
    x, y = descriptor.vars()
    field = _zeros(descriptor, 2)
    map_ = [x + y**2, y + x**3]
    output = _zeros(descriptor, 2)
    status = lib.madng_tpsa_protected_map_binary_call(
        lib.mad_tpsa_exppb,
        2,
        _ptrs(field),
        2,
        _ptrs(map_),
        _ptrs(output),
    )
    assert status == 0
    assert output[0] == map_[0]
    assert output[1] == map_[1]


def test_map_translate_identity():
    descriptor = Descriptor(2, 3)
    x, y = descriptor.vars()
    map_ = [x, y]
    output = _zeros(descriptor, 2)
    offsets = ffi.new('double[]', [1.5, -2.0])
    status = lib.madng_tpsa_protected_map_translate_call(
        lib.mad_tpsa_translate,
        2,
        _ptrs(map_),
        2,
        offsets,
        _ptrs(output),
    )
    assert status == 0
    assert output[0] == x + 1.5
    assert output[1] == y - 2.0


def test_map_eval():
    descriptor = Descriptor(2, 3)
    x, y = descriptor.vars()
    map_ = [x**2 + y, x * y]
    point = ffi.new('double[]', [2.0, 3.0])
    output = ffi.new('double[]', 2)
    status = lib.madng_tpsa_protected_map_eval_call(
        lib.mad_tpsa_eval,
        2,
        _ptrs(map_),
        2,
        point,
        output,
    )
    assert status == 0
    assert output[0] == pytest.approx(7.0)
    assert output[1] == pytest.approx(6.0)


def test_map_inverse_failure_is_caught():
    descriptor = Descriptor(2, 3)
    x, _ = descriptor.vars()

    # Singular linear part:
    # M(x, y) = (x, 0)
    map_ = [
        x,
        descriptor.zero(),
    ]
    output = _zeros(descriptor, 2)
    status = lib.madng_tpsa_protected_map_inverse_call(
        lib.mad_tpsa_minv,
        2,
        _ptrs(map_),
        2,
        _ptrs(output),
    )
    assert status != 0
    message = ffi.string(lib.madng_tpsa_last_error_message()).decode()
    assert message
