import pytest

import madng_tpsa
from madng_tpsa._cffi import ffi, lib


def _ptrs(series):
    return ffi.new('void*[]', [item.ptr for item in series])


def _zeros(descriptor, count):
    return [descriptor.zero() for _ in range(count)]


def test_map_compose():
    descriptor = madng_tpsa.Descriptor(2, 4)
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
    descriptor = madng_tpsa.Descriptor(2, 4)
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
    descriptor = madng_tpsa.Descriptor(2, 4)
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
    descriptor = madng_tpsa.Descriptor(2, 3)
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
    descriptor = madng_tpsa.Descriptor(2, 4)
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
    descriptor = madng_tpsa.Descriptor(2, 3)
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
    descriptor = madng_tpsa.Descriptor(2, 4)
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
    descriptor = madng_tpsa.Descriptor(2, 3)
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
    descriptor = madng_tpsa.Descriptor(2, 3)
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
    descriptor = madng_tpsa.Descriptor(2, 3)
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
