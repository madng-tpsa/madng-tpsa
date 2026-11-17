"""Compile and execute the generated C wrappers through CFFI.

These tests use a deliberately small fake TPSA native library; they do not
exercise the MAD-NG algorithms themselves.
"""
# ruff: noqa

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import cffi
import pytest

from madng_tpsa.madng_function import (
    MadngFunction,
    NativeSize,
    generate_cffi_header,
    unique_signatures,
)


def generate_headers(functions):
    signatures = unique_signatures(functions)
    return (
        generate_cffi_header(functions),
        '\n\n'.join(sig.native_header() for sig in signatures),
        '\n\n'.join(sig.native_source() for sig in signatures),
    )


HEADERS = [
    'void mad_tpsa_logxdy(const tpsa_t *x, const tpsa_t *y, tpsa_t *r);',
    'void mad_tpsa_liebra(ssz_t na, const tpsa_t *ma[], const tpsa_t *mb[], tpsa_t *mc[]);',
    'void mad_tpsa_compose(ssz_t na, const tpsa_t *ma[], ssz_t nb, const tpsa_t *mb[], tpsa_t *mc[]);',
    'void mad_tpsa_translate(ssz_t na, const tpsa_t *ma[], ssz_t nb, const num_t tb[], tpsa_t *mc[]);',
    'void mad_tpsa_eval(ssz_t na, const tpsa_t *ma[], ssz_t nb, const num_t tb[], num_t tc[]);',
    'ord_t mad_tpsa_mord(ssz_t na, const tpsa_t *ma[], log_t hi);',
    'void mad_tpsa_axpsqrtbpcx2(const tpsa_t *x, num_t a, num_t b, num_t c, tpsa_t *r);',
    'void mad_tpsa_minv(ssz_t na, const tpsa_t *ma[], ssz_t nb, tpsa_t *mc[]);',
]


EXTRA_HEADERS = [
    'void mad_tpsa_three(const tpsa_t *x, const tpsa_t *y, const tpsa_t *z, tpsa_t *r);',
    'void mad_tpsa_scale(num_t a, tpsa_t *x);',
    'void mad_tpsa_init(num_t a, tpsa_t *r);',
    'void mad_mock_axpy(ssz_t n, const num_t x[], num_t y[]);',
    'num_t mad_mock_sum(ssz_t n, const num_t x[], idx_t *count);',
    'num_t mad_mock_scalar(const num_t *x);',
    'void mad_mock_fill(ssz_t na, ssz_t nb, num_t out[]);',
    'void mad_mock_keyword(num_t from, num_t *class);',
]


def build_native(tmp: Path):
    prototypes = [MadngFunction(s) for s in HEADERS + EXTRA_HEADERS]
    cdef, header, source = generate_headers(prototypes)
    ffi = cffi.FFI()
    ffi.cdef(
        cdef
        + """
        void *make_tpsa(double v);
        void *make_ctpsa(double v);
        double get_tpsa(const void *p);
        void free_tpsa(void *p);
        const char *madng_tpsa_last_error_location(void);
        const char *madng_tpsa_last_error_message(void);
    """
    )
    code = textwrap.dedent(r"""
        #include <setjmp.h>
        #include <stdlib.h>
        typedef int ssz_t;
        typedef int idx_t;
        typedef _Bool log_t;
        typedef unsigned char ord_t;
        typedef double num_t;
        typedef struct { double value; } tpsa_t;
        typedef struct { double value; } ctpsa_t;
        typedef void(*handler_t)(const char*,const char*);
        static _Thread_local handler_t current_handler;
        static _Thread_local handler_t madng_panic_previous_error_handler;
        static _Thread_local jmp_buf madng_panic_environment;
        static handler_t madng_tpsa_set_error_handler(handler_t h) {
            handler_t old=current_handler; current_handler=h; return old;
        }
        static void madng_panic_restore_error_handler(void) {
            madng_tpsa_set_error_handler(madng_panic_previous_error_handler);
        }
        static void madng_panic_longjmp(const char *location, const char *message) {
            (void)location; (void)message;
            longjmp(madng_panic_environment,1);
        }
        const char *madng_tpsa_last_error_location(void) {return "test";}
        const char *madng_tpsa_last_error_message(void) {return "deliberately failed";}
        void *make_tpsa(double v) {tpsa_t *p=malloc(sizeof(*p));p->value=v;return p;}
        void *make_ctpsa(double v) {ctpsa_t *p=malloc(sizeof(*p));p->value=v;return p;}
        double get_tpsa(const void *p) {return ((const tpsa_t *)p)->value;}
        void free_tpsa(void *p) {free(p);}
        #define MADNG_PROTECTED_CALL(call)                                                           \
        do {                                                                                         \
            /* Always restore the caller's handler, whether MAD-NG returns or panics. */             \
            madng_panic_previous_error_handler = madng_tpsa_set_error_handler(madng_panic_longjmp);  \
            if (setjmp(madng_panic_environment) == 0) {                                              \
                call;                                                                                \
                madng_panic_restore_error_handler();                                                 \
                return 0;                                                                            \
            }                                                                                        \
            madng_panic_restore_error_handler();                                                     \
            return 1;                                                                                \
        } while (0)
    """)
    code += (
        '\n'
        + header
        + '\n'
        + textwrap.dedent("""
        void mad_tpsa_logxdy(const tpsa_t *x, const tpsa_t *y, tpsa_t *r) {
            r->value = x->value - y->value;
        }
        void mad_ctpsa_logxdy(const ctpsa_t *x, const ctpsa_t *y, ctpsa_t *r) {
            r->value = 100 + x->value - y->value;
        }
        void mad_tpsa_liebra(ssz_t n, const tpsa_t *a[], const tpsa_t *b[], tpsa_t *c[]) {
            for(int i=0;i<n;i++) c[i]->value=a[i]->value-b[i]->value;
        }
        void mad_ctpsa_liebra(ssz_t n, const ctpsa_t *a[], const ctpsa_t *b[], ctpsa_t *c[]) {
            for(int i=0;i<n;i++) c[i]->value=100+a[i]->value-b[i]->value;
        }
        void mad_tpsa_compose(ssz_t na, const tpsa_t *a[], ssz_t nb, const tpsa_t *b[], tpsa_t *c[]) {
            for(int i=0;i<abs(na);i++) c[i]->value=a[i]->value+b[i%nb]->value;
        }
        void mad_ctpsa_compose(ssz_t na, const ctpsa_t *a[], ssz_t nb, const ctpsa_t *b[], ctpsa_t *c[]) {
            for(int i=0;i<abs(na);i++) c[i]->value=100+a[i]->value+b[i%nb]->value;
        }
        void mad_tpsa_translate(ssz_t na, const tpsa_t *a[], ssz_t nb, const num_t tb[], tpsa_t *c[]) {
            for(int i=0;i<na;i++) c[i]->value=a[i]->value+tb[i%nb];
        }
        void mad_ctpsa_translate(ssz_t na, const ctpsa_t *a[], ssz_t nb, const num_t tb[], ctpsa_t *c[]) {
            for(int i=0;i<na;i++) c[i]->value=100+a[i]->value+tb[i%nb];
        }
        void mad_tpsa_eval(ssz_t na, const tpsa_t *a[], ssz_t nb, const num_t tb[], num_t tc[]) {
            for(int i=0;i<na;i++) tc[i]=a[i]->value+tb[i%nb];
        }
        void mad_ctpsa_eval(ssz_t na, const ctpsa_t *a[], ssz_t nb, const num_t tb[], num_t tc[]) {
            for(int i=0;i<na;i++) tc[i]=100+a[i]->value+tb[i%nb];
        }
        ord_t mad_tpsa_mord(ssz_t na, const tpsa_t *a[], log_t hi) { (void)a; return na+hi; }
        ord_t mad_ctpsa_mord(ssz_t na, const ctpsa_t *a[], log_t hi) { (void)a; return na+hi+100; }
        void mad_tpsa_axpsqrtbpcx2(const tpsa_t *x, num_t a, num_t b, num_t c, tpsa_t *r) {
            if(a<0) current_handler("here","bad input");
            r->value=x->value+a+b+c;
        }
        void mad_ctpsa_axpsqrtbpcx2(const ctpsa_t *x, num_t a, num_t b, num_t c, ctpsa_t *r) {
            if(a<0) current_handler("here","bad input");
            r->value=100+x->value+a+b+c;
        }
        void mad_tpsa_minv(ssz_t na, const tpsa_t *a[], ssz_t nb, tpsa_t *c[]) {
            for(int i=0;i<nb;i++) c[i]->value=-a[i%na]->value;
            for(int i=nb;i<na;i++) {
                if(a[i]->value!=1 || c[i]->value!=1) current_handler("minv","bad parameter tail");
                c[i]->value=42;
            }
        }
        void mad_ctpsa_minv(ssz_t na, const ctpsa_t *a[], ssz_t nb, ctpsa_t *c[]) {
            for(int i=0;i<nb;i++) c[i]->value=-100-a[i%na]->value;
            for(int i=nb;i<na;i++) {
                if(a[i]->value!=1 || c[i]->value!=1) current_handler("minv","bad parameter tail");
                c[i]->value=42;
            }
        }
    """)
    )
    code += textwrap.dedent("""
        void mad_tpsa_three(const tpsa_t *x, const tpsa_t *y, const tpsa_t *z, tpsa_t *r) {
            r->value=x->value+y->value+z->value;
        }
        void mad_ctpsa_three(const ctpsa_t *x, const ctpsa_t *y, const ctpsa_t *z, ctpsa_t *r) {
            r->value=100+x->value+y->value+z->value;
        }
        void mad_tpsa_scale(num_t a, tpsa_t *x) {x->value*=a;}
        void mad_ctpsa_scale(num_t a, ctpsa_t *x) {x->value=100+a*x->value;}
        void mad_tpsa_init(num_t a, tpsa_t *r) {r->value=a;}
        void mad_ctpsa_init(num_t a, ctpsa_t *r) {r->value=100+a;}
        void mad_mock_axpy(ssz_t n, const num_t x[], num_t y[]) {
            for(int i=0;i<n;i++) y[i]+=x[i];
        }
        num_t mad_mock_sum(ssz_t n, const num_t x[], idx_t *count) {
            num_t sum=0; for(int i=0;i<n;i++) sum+=x[i]; *count=n; return sum;
        }
        num_t mad_mock_scalar(const num_t *x) {return *x;}
        void mad_mock_fill(ssz_t na, ssz_t nb, num_t out[]) {
            for(int i=0;i<nb;i++) out[i]=na+i;
        }
        void mad_mock_keyword(num_t from, num_t *class) {*class=from+1;}
    """)
    code += '\n' + source + '\n'
    source_path = Path(tmp, 'test.c')
    source_path.write_text(code)
    library_path = Path(tmp, 'libtest.so')
    subprocess.run(
        [
            'cc',
            '-std=c11',
            '-O2',
            '-Wall',
            '-Werror',
            '-shared',
            '-fPIC',
            str(source_path),
            '-o',
            str(library_path),
        ],
        check=True,
    )
    lib = ffi.dlopen(str(library_path))
    functions = {
        MadngFunction(s).name: MadngFunction(
            s,
            ffi=ffi,
            lib=lib,
            aligner=mock_aligner,
            array_sizes={'mc': 'na'}
            if 'compose' in s
            else (
                {'mc': 'na'}
                if 'translate' in s
                else ({'tc': 'na'} if 'mad_tpsa_eval' in s else None)
            ),
        )
        for s in HEADERS
    }
    return ffi, lib, functions


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    return build_native(tmp_path_factory.mktemp('compiled'))


class Descriptor:
    def __init__(self, ffi, lib, *, variables=('x', 'px'), parameters=(), order=4):
        self.ffi, self.lib = ffi, lib
        self.variables, self.parameters, self.order = variables, parameters, order
        self.union_calls = 0

    @property
    def num_params(self):
        return len(self.parameters)

    def params(self):
        return [
            Tpsa(self.ffi.gc(self.lib.make_tpsa(1), self.lib.free_tpsa), self)
            for _ in self.parameters
        ]

    def zero(self, *, order=None):
        assert order is None or order <= self.order
        return Tpsa(self.ffi.gc(self.lib.make_tpsa(0), self.lib.free_tpsa), self)

    def complex_zero(self, *, order=None):
        assert order is None or order <= self.order
        return ComplexTpsa(self.ffi.gc(self.lib.make_ctpsa(0), self.lib.free_tpsa), self)

    def union(self, other):
        self.union_calls += 1
        params = tuple(dict.fromkeys(self.parameters + other.parameters))
        return Descriptor(
            self.ffi,
            self.lib,
            variables=self.variables,
            parameters=params,
            order=max(self.order, other.order),
        )


class Tpsa:
    def __init__(self, ptr, descriptor):
        self.ptr = ptr
        self.descriptor = descriptor

    def change_descriptor(self, descriptor):
        d = descriptor
        value = self.descriptor.lib.get_tpsa(self.ptr)
        make = d.lib.make_ctpsa if isinstance(self, ComplexTpsa) else d.lib.make_tpsa
        return type(self)(d.ffi.gc(make(value), d.lib.free_tpsa), d)

    def to_complex(self):
        d = self.descriptor
        value = d.lib.get_tpsa(self.ptr)
        return ComplexTpsa(d.ffi.gc(d.lib.make_ctpsa(value), d.lib.free_tpsa), d)


class ComplexTpsa(Tpsa):
    def to_complex(self):
        return self


class TpsaTensor:
    def __init__(self, values, ffi):
        self.flat = tuple(values)
        self.descriptor = values[0].descriptor
        self.size = len(values)
        self.is_complex = isinstance(values[0], ComplexTpsa)
        self._pointer_array = ffi.new('void*[]', [x.ptr for x in values])
        self.ffi = ffi
        self.pointer_reads = 0
        self._params = self.descriptor.params()
        if self.is_complex:
            self._params = [p.to_complex() for p in self._params]

    @property
    def pointer_array_with_params(self):
        return self.ffi.new('void*[]', [t.ptr for t in (*self.flat, *self._params)])

    def extend_descriptor(self, descriptor):
        return self.change_descriptor(descriptor)

    def _new_like(self, values):
        return TpsaTensor(values, self.ffi)

    @property
    def pointer_array(self):
        self.pointer_reads += 1
        return self._pointer_array

    def change_descriptor(self, descriptor):
        return TpsaTensor([x.change_descriptor(descriptor) for x in self.flat], self.ffi)

    def to_complex(self):
        values = [x if isinstance(x, ComplexTpsa) else self._complex_from(x) for x in self.flat]
        return TpsaTensor(values, self.ffi)

    @staticmethod
    def _complex_from(x):
        return x.to_complex()


def mock_aligner(left, right, *, extend_left, extend_right, reorder_variables):
    """Align mock objects; the real package aligner only accepts real package types."""
    if left.descriptor.variables != right.descriptor.variables:
        if not reorder_variables or set(left.descriptor.variables) != set(
            right.descriptor.variables
        ):
            raise ValueError('Incompatible descriptor variables')
    common = left.descriptor
    if common is not right.descriptor:
        common = common.union(right.descriptor)
    if left.descriptor is not common:
        left = (
            left.extend_descriptor(common)
            if extend_left and hasattr(left, 'extend_descriptor')
            else left.change_descriptor(common)
        )
    if right.descriptor is not common:
        right = (
            right.extend_descriptor(common)
            if extend_right and hasattr(right, 'extend_descriptor')
            else right.change_descriptor(common)
        )

    def is_complex(obj):
        return isinstance(obj, ComplexTpsa) or getattr(obj, 'is_complex', False)

    complex_needed = is_complex(left) or is_complex(right)
    if complex_needed:
        if not is_complex(left):
            left = left.to_complex()
        if not is_complex(right):
            right = right.to_complex()
    return left, right, complex_needed


def make_series(native, vals, *, descriptor=None, _complex=False):
    ffi, lib, _ = native
    if descriptor is None:
        descriptor = Descriptor(ffi, lib)
    make = lib.make_ctpsa if _complex else lib.make_tpsa
    cls = ComplexTpsa if _complex else Tpsa
    return [cls(ffi.gc(make(v), lib.free_tpsa), descriptor) for v in vals]


def get_values(native, series):
    _, lib, _ = native
    return [lib.get_tpsa(x.ptr) for x in series]


def test_generated_c_deduplicates_signatures():
    fs = [MadngFunction(x) for x in HEADERS]
    cffi_header, native_header, native_source = generate_headers(fs)
    assert cffi_header.count('mad_ctpsa_logxdy(') == 1
    # logxdy and a dozen similar binary TPSA functions share 1 typedef and wrapper.
    test = MadngFunction('void mad_tpsa_other(const tpsa_t *x, const tpsa_t *y, tpsa_t *r);')
    _, header2, source2 = generate_headers([fs[0], test])
    assert source2.count('int madng_protected_sig_') == 2  # real and complex
    assert header2.count('typedef') == 2
    assert 'MADNG_PROTECTED_CALL' in native_source
    ffi = cffi.FFI()
    ffi.cdef(cffi_header)


def test_scalar_real_complex_dispatch(native):
    ffi, lib, fs = native
    a, b = make_series(native, [5, 2])
    real = fs['mad_tpsa_logxdy'](a, b)
    assert type(real) is Tpsa
    assert get_values(native, [real]) == [3]
    ca, cb = make_series(native, [5, 2], _complex=True)
    complex_result = fs['mad_tpsa_logxdy'](ca, cb)
    assert type(complex_result) is ComplexTpsa
    assert get_values(native, [complex_result]) == [103]


def test_tensor_fast_path_and_allocation(native):
    ffi, lib, fs = native
    d = Descriptor(ffi, lib)
    left = TpsaTensor(make_series(native, [10, 20], descriptor=d), ffi)
    right = TpsaTensor(make_series(native, [1, 2], descriptor=d), ffi)
    result = fs['mad_tpsa_liebra'](left, right)
    assert len(result) == 2
    assert get_values(native, result) == [9, 18]
    assert d.union_calls == 0
    assert left.pointer_reads == 1 and right.pointer_reads == 1
    wrong = TpsaTensor(make_series(native, [1], descriptor=d), ffi)
    with pytest.raises(ValueError, match='length'):
        fs['mad_tpsa_liebra'](left, wrong)


def test_descriptor_union_and_complex_promotion(native):
    ffi, lib, fs = native
    left_d = Descriptor(ffi, lib, parameters=('k',))
    right_d = Descriptor(ffi, lib, parameters=('q',))
    left = TpsaTensor(make_series(native, [3, 4], descriptor=left_d), ffi)
    right = TpsaTensor(make_series(native, [1, 2], descriptor=right_d, _complex=True), ffi)
    result = fs['mad_tpsa_liebra'](left, right)
    assert len(result) == 2
    assert all(type(x) is ComplexTpsa for x in result)
    assert result[0].descriptor.parameters == ('k', 'q')
    assert get_values(native, result) == [102, 102]
    assert left_d.union_calls == 1
    assert left.is_complex is False  # Original inputs not mutated
    assert right.is_complex is True


def test_compose_two_different_dimensions(native):
    ffi, lib, fs = native
    d = Descriptor(ffi, lib)
    ma = TpsaTensor(make_series(native, [10, 20, 30], descriptor=d), ffi)
    mb = TpsaTensor(make_series(native, [1, 2], descriptor=d), ffi)
    result = fs['mad_tpsa_compose'](ma, mb)
    assert len(result) == 3
    assert get_values(native, result) == [11, 22, 31]
    invalid = make_series(native, [0, 0], descriptor=d)
    with pytest.raises(ValueError, match='length'):
        fs['mad_tpsa_compose'](ma, mb, invalid)


def test_shared_dimension_output_and_numeric(native):
    ffi, lib, fs = native
    d = Descriptor(ffi, lib)
    ma = TpsaTensor(make_series(native, [10, 20, 30], descriptor=d), ffi)
    out = fs['mad_tpsa_eval'](ma, [0.5, 1.5])
    assert out == [10.5, 21.5, 30.5]
    np = pytest.importorskip('numpy')
    tb = np.array([0.5, 1.5], dtype=np.float64)
    tc = np.zeros(3, dtype=np.float64)
    res = fs['mad_tpsa_eval'](ma, tb, tc)
    assert res is tc
    np.testing.assert_array_equal(tc, [10.5, 21.5, 30.5])


def test_nonvoid_and_error(native):
    ffi, lib, fs = native
    d = Descriptor(ffi, lib)
    ma = TpsaTensor(make_series(native, [1, 2, 3], descriptor=d), ffi)
    assert fs['mad_tpsa_mord'](ma, True) == 4
    x = make_series(native, [3], descriptor=d)[0]
    fn = fs['mad_tpsa_axpsqrtbpcx2']
    with pytest.raises(RuntimeError, match='deliberately failed'):
        fn(x, -1, 0, 0)
    assert get_values(native, [fn(x, 1, 2, 3)]) == [9]


def test_override_size_and_output_types(native):
    ffi, lib, fs = native
    d = Descriptor(ffi, lib)
    ma = TpsaTensor(make_series(native, [5, 6], descriptor=d), ffi)
    fn = MadngFunction(HEADERS[-1], ffi=ffi, lib=lib, size_defaults={'nb': 'na'})
    out = fn(ma)
    assert get_values(native, out) == [-5, -6]
    fn2 = fs['mad_tpsa_liebra']
    complex_right = TpsaTensor(make_series(native, [1, 2], descriptor=d, _complex=True), ffi)
    wrong_output = make_series(native, [0, 0], descriptor=d)
    with pytest.raises(TypeError, match='wrong real/complex'):
        fn2(ma, complex_right, wrong_output)


def test_parses_original_examples():
    more = """
        void mad_tpsa_vec2fld(ssz_t na, const tpsa_t *a, tpsa_t *mc[]);
        void mad_tpsa_fld2vec(ssz_t na, const tpsa_t *ma[], tpsa_t *c);
        void mad_tpsa_exppb(ssz_t na, const tpsa_t *ma[], ssz_t nb, const tpsa_t *mb[], tpsa_t *mc[]);
        void mad_tpsa_pminv(ssz_t na, const tpsa_t *ma[], ssz_t nb, tpsa_t *mc[], idx_t select[]);
        void mad_tpsa_mconv(ssz_t na, const tpsa_t *ma[], ssz_t nc, tpsa_t *mc[], ssz_t n, idx_t t2r_[], int pb);
    """
    for declaration in more.split(';'):
        if declaration.strip():
            MadngFunction(
                declaration + ';',
                directions={'select': 'in'}
                if 'select[]' in declaration
                else ({'t2r_': 'in'} if 't2r_' in declaration else None),
            )


def test_mixed_flavours_with_same_descriptor(native):
    ffi, lib, fs = native
    descriptor = Descriptor(ffi, lib)
    real = TpsaTensor(make_series(native, [7, 8], descriptor=descriptor), ffi)
    complex_tensor = TpsaTensor(
        make_series(native, [1, 2], descriptor=descriptor, _complex=True), ffi
    )
    out = fs['mad_tpsa_liebra'](real, complex_tensor)
    assert all(isinstance(item, ComplexTpsa) for item in out)
    assert all(item.descriptor is descriptor for item in out)
    assert not real.is_complex


def test_forced_complex_promotes_even_when_both_inputs_real(native):
    ffi, lib, fs = native
    descriptor = Descriptor(ffi, lib)
    left = TpsaTensor(make_series(native, [1, 2], descriptor=descriptor), ffi)
    right = TpsaTensor(make_series(native, [1, 2], descriptor=descriptor), ffi)
    result = fs['mad_tpsa_liebra'](left, right, complex_needed=True)
    assert all(isinstance(item, ComplexTpsa) for item in result)


def test_preallocated_output_descriptor_must_match(native):
    ffi, lib, fs = native
    first = Descriptor(ffi, lib)
    other = Descriptor(ffi, lib)
    left = TpsaTensor(make_series(native, [1, 2], descriptor=first), ffi)
    right = TpsaTensor(make_series(native, [1, 2], descriptor=first), ffi)
    wrong = TpsaTensor(make_series(native, [0, 0], descriptor=other), ffi)
    with pytest.raises(ValueError, match='wrong descriptor'):
        fs['mad_tpsa_liebra'](left, right, wrong)


def test_explicit_binding(native):
    ffi, lib, _ = native
    descriptor = Descriptor(ffi, lib)
    left = TpsaTensor(make_series(native, [2, 3], descriptor=descriptor), ffi)
    right = TpsaTensor(make_series(native, [1, 1], descriptor=descriptor), ffi)
    fn = MadngFunction(HEADERS[1])
    with pytest.raises(RuntimeError, match='bind'):
        fn(left, right)
    fn.bind(ffi, lib)
    assert fn.is_specialised
    result = fn(left, right)
    assert get_values(native, result) == [1, 2]


@pytest.mark.parametrize('complex_needed', [False, True])
def test_parameter_aware_layouts(native, complex_needed):
    ffi, lib, _ = native
    d = Descriptor(ffi, lib, parameters=('k',))
    a = TpsaTensor(make_series(native, [10, 20, 30], descriptor=d, _complex=complex_needed), ffi)
    b = TpsaTensor(make_series(native, [5, 7], descriptor=d, _complex=complex_needed), ffi)
    compose = MadngFunction(
        HEADERS[2],
        ffi=ffi,
        lib=lib,
        array_sizes={'mc': 'na'},
        native_sizes={'na': NativeSize(sign=-1), 'nb': NativeSize(params_from='mb')},
        parameter_arrays=('mb',),
        return_like='ma',
    )
    result = compose(a, b)
    extra = 100 if complex_needed else 0
    assert get_values(native, result.flat) == [15 + extra, 27 + extra, 31 + extra]
    assert result.size == 3
    minv = MadngFunction(
        HEADERS[-1],
        ffi=ffi,
        lib=lib,
        size_aliases={'nb': 'na'},
        native_sizes={'na': NativeSize(params_from='ma')},
        parameter_arrays=('ma', 'mc'),
        return_like='ma',
    )
    result = minv(b)
    assert result.size == 2
    assert get_values(native, result.flat) == [-5 - extra, -7 - extra]
    assert lib.get_tpsa(b._params[0].ptr) == 1  # input identities were not overwritten
    with pytest.raises(ValueError, match='nb must equal na'):
        minv(b, nb=1)


def test_wrapper_tags_are_stable():
    fn = MadngFunction(HEADERS[-1])
    assert fn.real.tag == 'v_s_krpa_s_rpa'
    assert fn.complex is not None
    assert fn.complex.tag == 'v_s_kcpa_s_cpa'


def test_three_operand_alignment(native):
    ffi, lib, _ = native
    ds = [Descriptor(ffi, lib, parameters=(name,)) for name in ('k', 'q', 'r')]
    x, y, z = [
        make_series(native, [i + 1], descriptor=d, _complex=i == 1)[0] for i, d in enumerate(ds)
    ]
    fn = MadngFunction(EXTRA_HEADERS[0], ffi=ffi, lib=lib)
    result = fn(x, y, z)
    assert type(result) is ComplexTpsa
    assert result.descriptor.parameters == ('k', 'q', 'r')
    assert get_values(native, [result]) == [106]
    assert x.descriptor is ds[0] and z.descriptor is ds[2]
    with pytest.raises(ValueError, match='complex'):
        fn(x, y, z, complex_needed=False)


def test_output_factory(native):
    ffi, lib, _ = native
    d = Descriptor(ffi, lib)
    x = make_series(native, [3], descriptor=d)[0]
    calls = []

    def factory(size, context):
        assert size is None
        assert context == {'x': x, 'a': 1, 'b': 2, 'c': 3}
        calls.append(context)
        return d.zero(order=2)

    fn = MadngFunction(HEADERS[6], ffi=ffi, lib=lib, output_factory={'r': factory})
    assert get_values(native, [fn(x, 1, 2, 3)]) == [9]
    out = d.zero()
    assert fn(x, 1, 2, 3, out) is out
    assert len(calls) == 1


def test_series_inout_and_output_only(native):
    ffi, lib, _ = native
    d = Descriptor(ffi, lib)
    x = make_series(native, [3], descriptor=d)[0]
    scale = MadngFunction(EXTRA_HEADERS[1], ffi=ffi, lib=lib, directions={'x': 'inout'})
    assert scale(2, x) is x
    assert get_values(native, [x]) == [6]
    cx = x.to_complex()
    assert scale(2, cx) is cx
    assert get_values(native, [cx]) == [112]
    init = MadngFunction(
        EXTRA_HEADERS[2], ffi=ffi, lib=lib, output_factory={'r': lambda n, ctx: d.zero()}
    )
    assert get_values(native, [init(7)]) == [7]
    out = d.complex_zero()
    assert init(8, out, complex_needed=True) is out
    assert get_values(native, [out]) == [108]


def test_numeric_only_inout(native):
    ffi, lib, _ = native
    fn = MadngFunction(EXTRA_HEADERS[3], ffi=ffi, lib=lib, directions={'y': 'inout'})
    assert fn([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0]
    np = pytest.importorskip('numpy')
    out = np.array([3.0, 4.0])
    assert fn([1.0, 2.0], out) is out
    np.testing.assert_array_equal(out, [4.0, 6.0])
    with pytest.raises(TypeError, match='writable contiguous'):
        fn([1.0, 2.0], np.zeros(4)[::2])
    out.flags.writeable = False
    with pytest.raises(TypeError, match='writable contiguous'):
        fn([1.0, 2.0], out)


def test_scalar_pointers_and_return_value(native):
    ffi, lib, _ = native
    fn = MadngFunction(EXTRA_HEADERS[4], ffi=ffi, lib=lib)
    assert fn([1.0, 2.0, 3.0]) == (6.0, 3)
    count = ffi.new('int *')
    assert fn([1.0, 2.0], count) == (3.0, 2)
    assert count[0] == 2
    scalar = MadngFunction(EXTRA_HEADERS[5], ffi=ffi, lib=lib)
    assert scalar(3.5) == 3.5
    assert scalar(ffi.new('double *', 2.5)) == 2.5


def test_size_resolution_and_python_aliases(native):
    ffi, lib, _ = native
    fn = MadngFunction(EXTRA_HEADERS[6], ffi=ffi, lib=lib, size_defaults={'na': 'nb'})
    assert fn(nb=3) == [3.0, 4.0, 5.0]
    assert fn([0.0, 0.0]) == [2.0, 3.0]
    assert fn(na=1, nb=3) == [1.0, 2.0, 3.0]
    alias = MadngFunction(EXTRA_HEADERS[6], ffi=ffi, lib=lib, size_aliases={'na': 'nb'})
    assert alias(na=3) == [3.0, 4.0, 5.0]
    with pytest.raises(ValueError, match='must equal'):
        alias(na=3, nb=2)
    with pytest.raises(TypeError, match='integer'):
        alias(na=2.0, nb=2)
    with pytest.raises(TypeError, match='integer'):
        alias(na=True, nb=1)
    keyword = MadngFunction(EXTRA_HEADERS[7], ffi=ffi, lib=lib)
    assert keyword(from_arg=3) == 4.0


def test_explicit_raw_complex_pointers(native):
    ffi, lib, _ = native
    d = Descriptor(ffi, lib)
    a, b = make_series(native, [4, 1], descriptor=d, _complex=True)
    out = d.complex_zero()
    fn = MadngFunction(HEADERS[0], ffi=ffi, lib=lib, align=False)
    assert fn(a.ptr, b.ptr, out, complex_needed=True) is out
    assert get_values(native, [out]) == [103]
    ta = TpsaTensor([a, b], ffi)
    tb = TpsaTensor([b, a], ffi)
    tc = TpsaTensor([d.complex_zero(), d.complex_zero()], ffi)
    liebra = MadngFunction(HEADERS[1], ffi=ffi, lib=lib, align=False)
    assert (
        liebra(
            ffi.cast('void **', ta.pointer_array), tb.pointer_array, tc, na=2, complex_needed=True
        )
        is tc
    )
    assert get_values(native, tc.flat) == [103, 97]
