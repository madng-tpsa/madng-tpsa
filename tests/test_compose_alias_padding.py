"""Regression tests for the native MAD-NG composition alias-check workaround."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import madng_tpsa.tensor as tensor_module
from madng_tpsa import Descriptor, TpsaMap, TpsaTensor
from madng_tpsa._cffi import ffi


@pytest.mark.parametrize('operation', ['compose', 'substitute'])
@pytest.mark.parametrize('use_complex', [False, True])
@pytest.mark.parametrize('size', [2, 6])
def test_native_compose_alias_padding(
    monkeypatch: pytest.MonkeyPatch, operation: str, *, use_complex: bool, size: int
) -> None:
    d = Descriptor(var_names=('x', 'px'), order=4, param_names=('k',))
    x, px = d.vars()
    k = d.param('k')
    components = [x + k, px, x * px, x**2, px**2, x + px]
    expected = [2 * x + k, px, 2 * x * px, 4 * x**2, px**2, 2 * x + px]
    shape = (2,) if size == 2 else (2, 3)
    tensor = TpsaTensor(components[:size], shape=shape)
    if use_complex:
        tensor = tensor.to_complex()

    original_lib = tensor_module.lib
    calls: list[str] = []
    real_call = 'madng_tpsa_protected_map_binary_call'
    complex_call = 'madng_ctpsa_protected_map_binary_call'

    class CheckedLib:
        def __getattr__(self, name: str) -> Any:
            func = getattr(original_lib, name)
            if name not in (real_call, complex_call):
                return func

            def checked(native_func, na, ma, nb, mb, mc):
                assert na == -size
                assert nb == d.monomial_length
                assert len(mb) == max(abs(na), nb)
                assert all(mb[i] != ffi.NULL for i in range(nb))
                assert all(mb[i] == ffi.NULL for i in range(nb, len(mb)))
                calls.append(name)
                return func(native_func, na, ma, nb, mb, mc)

            return checked

    monkeypatch.setattr(tensor_module, 'lib', CheckedLib())
    if operation == 'compose':
        result = tensor.compose(TpsaMap([2 * x, px]))
    else:
        result = tensor.substitute(x=2 * x)

    assert calls == [complex_call if use_complex else real_call]
    assert result.shape == shape
    assert result.is_complex == use_complex
    expected_tensor = TpsaTensor(expected[:size], shape=shape)
    if use_complex:
        expected_tensor = expected_tensor.to_complex()
    assert result.flat == expected_tensor.flat
    np.testing.assert_allclose(
        result.evaluate([2, 3], param_values=[1]),
        np.asarray([5, 3, 12, 16, 9, 7][:size]).reshape(shape),
    )
