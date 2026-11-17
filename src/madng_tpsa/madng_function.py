"""Signature-driven real/complex MAD-NG calls with generated protected wrappers.

Designed to live in the madng_tpsa package.  Parsing, ABI choice, native symbol
resolution, and argument indexes are computed at construction, never in call().

Generated implementations belong at the end of native/madng_panic.c, in the
same translation unit as its static, thread-local setjmp helpers.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass
from operator import index
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


_C_TYPES = {
    'void': 'void',
    'ssz_t': 'int',
    'idx_t': 'int',
    'int': 'int',
    'num_t': 'double',
    'cpx_t': 'double _Complex',
    'ord_t': 'unsigned char',
    'log_t': '_Bool',
    'tpsa_t': 'void',
    'ctpsa_t': 'void',
}
_INT = {'ssz_t', 'idx_t', 'int', 'ord_t', 'log_t'}
_SERIES = {'tpsa_t', 'ctpsa_t'}
_TYPE_TAGS = {
    'void': 'v',
    'ssz_t': 's',
    'idx_t': 'i',
    'int': 'n',
    'num_t': 'd',
    'cpx_t': 'z',
    'ord_t': 'o',
    'log_t': 'b',
    'tpsa_t': 'r',
    'ctpsa_t': 'c',
}
_MISSING = object()
_IDENT = r'[A-Za-z_]\w*'
_DECL = re.compile(
    rf'^\s*(?P<const>const\s+)?(?P<base>{_IDENT})\s*'
    rf'(?P<stars>\*+)?\s*(?P<name>{_IDENT})\s*(?P<array>\[\s*\])?\s*$'
)
_FUNC = re.compile(
    rf'^\s*(?P<return>{_IDENT})\s+(?P<name>{_IDENT})\s*\((?P<args>.*?)\)\s*;\s*$',
    re.S,
)


def _strip_comments(text: str) -> str:
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    return re.sub(r'//[^\n]*', '', text)


@dataclass(frozen=True, slots=True)
class Argument:
    """Represents a single argument in a C function signature."""

    name: str
    base: str
    const: bool
    pointer: bool
    array: bool
    direction: Literal['in', 'out', 'inout']

    @property
    def is_size(self) -> bool:
        """Return True if the argument represents a size parameter (ssz_t)."""
        return self.base == 'ssz_t' and not self.pointer and not self.array

    @property
    def is_series(self) -> bool:
        """Return True if the argument represents a TPSA series (tpsa_t or ctpsa_t)."""
        return self.base in _SERIES

    def declaration(self, *, cffi: bool, name: str | None = None) -> str:
        """Return the C declaration for this argument, optionally using CFFI types."""
        base = _C_TYPES[self.base] if cffi else self.base
        prefix = 'const ' if self.const else ''
        name = self.name if name is None else name
        if self.array:
            # A TPSA array is an array of pointers, not an opaque void array.
            return f'{prefix}{base} {"*" if self.is_series else ""}{name}[]'
        if self.pointer:
            return f'{prefix}{base} *{name}'
        return f'{base} {name}'

    def abi_key(self) -> str:
        """Return a string representing the ABI-relevant aspects of this argument."""
        # Names and Python-side directions are NOT part of the C ABI.
        return (
            ('k' if self.const else '')
            + _TYPE_TAGS[self.base]
            + ('p' if self.pointer else '')
            + ('a' if self.array else '')
        )


def _parse(declaration: str, directions: Mapping[str, str] | None = None) -> Signature:
    match = _FUNC.fullmatch(_strip_comments(declaration).strip())
    if match is None:
        err_mess = 'Expected one simple C prototype ending in a semicolon'
        raise ValueError(err_mess)
    result, name = match['return'], match['name']
    if result not in _C_TYPES or result in _SERIES:
        err_mess = f'Unsupported return type {result!r}'
        raise ValueError(err_mess)
    args_text = match['args'].strip()
    args: list[Argument] = []
    for part in [] if args_text in ('', 'void') else args_text.split(','):
        spec = _DECL.fullmatch(part)
        if spec is None:
            err_mess = f'Unsupported C argument {part!r}'
            raise ValueError(err_mess)
        base, arg_name = spec['base'], spec['name']
        stars, array = len(spec['stars'] or ''), bool(spec['array'])
        if base not in _C_TYPES or base == 'void':
            err_mess = f'Unsupported C type {base!r}'
            raise ValueError(err_mess)
        if stars > 1 or (base in _SERIES and stars != 1):
            err_mess = f'Unsupported pointer depth in {part!r}'
            raise ValueError(err_mess)
        if array and base in _SERIES and stars != 1:
            err_mess = f'Invalid series pointer array {part!r}'
            raise ValueError(err_mess)
        direction = 'in' if (spec['const'] or not (stars or array)) else 'out'
        if directions and arg_name in directions:
            direction = directions[arg_name]
        if direction not in ('in', 'out', 'inout'):
            err_mess = f'Invalid direction {direction!r} for {arg_name}'
            raise ValueError(err_mess)
        if direction != 'in' and not (stars or array):
            err_mess = f'By-value argument {arg_name} cannot be an output'
            raise ValueError(err_mess)
        args.append(Argument(arg_name, base, bool(spec['const']), bool(stars), array, direction))
    names = [arg.name for arg in args]
    if len(names) != len(set(names)):
        err_mess = 'Duplicate argument name'
        raise ValueError(err_mess)
    if directions and set(directions) - set(names):
        err_mess = 'Unknown direction override'
        raise ValueError(err_mess)
    return Signature(name, result, tuple(args))


@dataclass(frozen=True, slots=True)
class Signature:
    """Represents a C function signature."""

    name: str
    return_type: str
    args: tuple[Argument, ...]

    @property
    def tag(self) -> str:
        """Return a unique tag representing the ABI of this signature."""
        return '_'.join((_TYPE_TAGS[self.return_type], *(a.abi_key() for a in self.args)))

    @property
    def function_type(self) -> str:
        """Return the C type name for the function pointer corresponding to this signature."""
        return f'madng_sig_{self.tag}_fn'

    @property
    def wrapper_name(self) -> str:
        """Return the name of the generated protected wrapper function."""
        return f'madng_protected_sig_{self.tag}'

    def params(self, *, cffi: bool, generic: bool = False) -> str:
        """Return the C parameter list for this signature, optionally using CFFI types."""
        return (
            ', '.join(
                a.declaration(cffi=cffi, name=f'p{i}' if generic else None)
                for i, a in enumerate(self.args)
            )
            or 'void'
        )

    def protected_params(self, *, cffi: bool) -> str:
        """Return the C parameter list for the protected wrapper function."""
        params = [f'{self.function_type} function']
        params.extend(a.declaration(cffi=cffi, name=f'p{i}') for i, a in enumerate(self.args))
        if self.return_type != 'void':
            rt = _C_TYPES[self.return_type] if cffi else self.return_type
            params.append(f'{rt} *result_')
        return ', '.join(params)

    def cffi_symbol(self) -> str:
        """Return the CFFI symbol declaration for this signature."""
        return f'{_C_TYPES[self.return_type]} {self.name}({self.params(cffi=True)});'

    def cffi_wrapper(self) -> str:
        """Return the CFFI wrapper function declaration for this signature."""
        result = _C_TYPES[self.return_type]
        return (
            f'typedef {result} (*{self.function_type})'
            f'({self.params(cffi=True, generic=True)});\n'
            f'int {self.wrapper_name}({self.protected_params(cffi=True)});'
        )

    def native_header(self) -> str:
        """Return the native C header declaration for this signature."""
        return (
            f'typedef {self.return_type} (*{self.function_type})'
            f'({self.params(cffi=False, generic=True)});\n'
            f'int {self.wrapper_name}({self.protected_params(cffi=False)});'
        )

    def native_source(self) -> str:
        """Return the native C source definition for the protected wrapper function."""
        arguments = ', '.join(f'p{i}' for i in range(len(self.args)))
        invocation = f'function({arguments})'
        if self.return_type != 'void':
            invocation = '*result_ = ' + invocation
        return (
            f'int {self.wrapper_name}'
            f'({self.protected_params(cffi=False)}) {{\n'
            f'    MADNG_PROTECTED_CALL({invocation});\n'
            '}'
        )


def _complex_signature(real: Signature, *, name: str | None) -> Signature:
    derived = name or real.name.replace('mad_tpsa_', 'mad_ctpsa_', 1)
    if derived == real.name and name is None:
        err_mess = 'Cannot derive complex name; specify complex_name'
        raise ValueError(err_mess)
    return Signature(
        derived,
        real.return_type,
        tuple(
            Argument(
                a.name,
                'ctpsa_t' if a.base == 'tpsa_t' else a.base,
                a.const,
                a.pointer,
                a.array,
                a.direction,
            )
            for a in real.args
        ),
    )


def _series_is_complex(obj: Any) -> bool:
    flag = getattr(obj, 'is_complex', None)
    if flag is not None:
        return bool(flag)
    # Direct scalar type; the real package has a ComplexTpsa class.
    return type(obj).__name__ == 'ComplexTpsa'


def _promote(obj: Any) -> Any:
    method = getattr(obj, 'to_complex', None)
    if method is not None:
        return method()
    # Scalars in the real package are converted with a class method.
    from madng_tpsa.complex_tpsa import ComplexTpsa

    return ComplexTpsa.from_tpsa(obj)


def _pointer(obj: Any) -> Any:
    try:
        return obj.ptr
    except AttributeError:
        return getattr(obj, 'ref', obj)


def _array_length(obj: Any) -> int | None:
    if obj is None:
        return None
    # Important: len(tensor) can be the first axis, rather than total size.
    try:
        return obj.size
    except AttributeError:
        try:
            return len(obj)
        except TypeError:
            return None


def _size_int(value: Any) -> int:
    """Accept integral sizes without silently truncating floats."""
    if isinstance(value, bool):
        err_mess = 'Size must be an integer, not bool'
        raise TypeError(err_mess)
    return index(value)


def _array_result(obj: Any, converted: Any) -> Any:
    """Convert temporary numeric output buffers; retain borrowed raw pointers."""
    if isinstance(obj, (list, tuple)) or obj is converted:
        n = _array_length(obj)
        return obj if n is None else [converted[j] for j in range(n)]
    return obj


@dataclass(frozen=True, slots=True)
class NativeSize:
    """Convert a logical non-negative length to the native dimension argument."""

    sign: Literal[1, -1] = 1
    params_from: str | None = None

    def __post_init__(self) -> None:
        if self.sign not in (-1, 1):
            err_mess = 'NativeSize.sign must be 1 or -1'
            raise ValueError(err_mess)


class MadngFunction:
    """Parse once and call both real and complex MAD-NG functions.

    The common invocation is f(left_tensor, right_tensor): descriptors and
    scalar types align automatically; dimensions derive from array lengths;
    missing outputs are allocated with the aligned descriptor.

    For uncommon APIs, configure array_sizes, size_defaults, directions,
    output_factory, or complex_declaration explicitly at construction time.
    """

    def __init__(
        self,
        declaration: str,
        *,
        ffi: Any = None,
        lib: Any = None,
        complex_name: str | None = None,
        complex_declaration: str | None = None,
        auto_complex: bool = True,
        array_sizes: Mapping[str, str] | None = None,
        size_defaults: Mapping[str, str | int] | None = None,
        directions: Mapping[str, Literal['in', 'out', 'inout']] | None = None,
        output_factory: Mapping[str, Callable[[int | None, dict[str, Any]], Any]] | None = None,
        output_order: int | None = None,
        align: bool = True,
        extend_vectors: bool = True,
        extend_left: bool | None = None,
        extend_right: bool | None = None,
        reorder_variables: bool = False,
        aligner: Callable[..., Any] | None = None,
        native_sizes: Mapping[str, NativeSize] | None = None,
        size_aliases: Mapping[str, str] | None = None,
        parameter_arrays: tuple[str, ...] = (),
        return_like: str | None = None,
        check_lengths: bool = True,
    ) -> None:
        real = _parse(declaration, directions)
        if any(arg.base == 'ctpsa_t' for arg in real.args):
            err_mess = 'Construct from the real tpsa_t signature'
            raise ValueError(err_mess)
        if complex_declaration is not None:
            other = _parse(complex_declaration, directions)
        elif auto_complex and any(a.base == 'tpsa_t' for a in real.args):
            other = _complex_signature(real, name=complex_name)
        else:
            other = None
        if other is not None:
            if len(other.args) != len(real.args) or (
                tuple(a.name for a in other.args) != tuple(a.name for a in real.args)
            ):
                err_mess = 'Real and complex signatures must have matching arguments'
                raise ValueError(err_mess)
            for a, b in zip(real.args, other.args, strict=True):
                expected = 'ctpsa_t' if a.base == 'tpsa_t' else a.base
                # Complex numeric arrays require an explicit complex prototype.
                if a.base == 'num_t' and a.array and b.base == 'cpx_t':
                    expected = 'cpx_t'
                if (b.base, b.const, b.pointer, b.array, b.direction) != (
                    expected,
                    a.const,
                    a.pointer,
                    a.array,
                    a.direction,
                ):
                    err_mess = f'Incompatible complex argument: {a.name}'
                    raise ValueError(err_mess)
            if real.return_type != other.return_type:
                err_mess = 'Different real/complex return conventions require an adapter'
                raise ValueError(err_mess)

        self.name = real.name
        self.real = real
        self.complex = other
        self.arguments = real.args
        self.ffi = ffi
        self.lib = lib
        self.output_order = output_order
        self.align = align
        self.extend_vectors = extend_vectors
        self.extend_left = extend_vectors if extend_left is None else extend_left
        self.extend_right = extend_vectors if extend_right is None else extend_right
        self.reorder_variables = reorder_variables
        self.check_lengths = check_lengths
        self.output_factory = dict(output_factory or {})
        self._pair_aligner = aligner
        self.return_like = return_like
        self._names = {a.name: i for i, a in enumerate(self.arguments)}
        self._size_indexes = tuple(i for i, a in enumerate(real.args) if a.is_size)
        self._input_series = tuple(
            i for i, a in enumerate(real.args) if a.is_series and a.direction != 'out'
        )
        self._outputs = tuple(i for i, a in enumerate(real.args) if a.direction != 'in')
        self._parameter_arrays: set[int] = set()
        if parameter_arrays and not check_lengths:
            err_mess = 'Parameter-inclusive arrays require length validation'
            raise ValueError(err_mess)
        for name in parameter_arrays:
            i = self._names.get(name)
            if i is None or not (real.args[i].array and real.args[i].is_series):
                err_mess = f'{name!r} is not a TPSA array'
                raise ValueError(err_mess)
            if real.args[i].direction == 'inout':
                err_mess = 'Parameter-inclusive inout arrays are not supported'
                raise ValueError(err_mess)
            self._parameter_arrays.add(i)
        self._native_size_rules: dict[int, tuple[int, int | None]] = {}
        for name, rule in (native_sizes or {}).items():
            i = self._names.get(name)
            if i is None or i not in self._size_indexes:
                err_mess = f'{name!r} is not a size argument'
                raise ValueError(err_mess)
            source = None if rule.params_from is None else self._names.get(rule.params_from)
            if rule.params_from is not None and source not in self._input_series:
                err_mess = 'params_from must identify a TPSA input'
                raise ValueError(err_mess)
            self._native_size_rules[i] = (rule.sign, source)
        if return_like is not None:
            ref = self._names.get(return_like)
            if (
                ref is None
                or ref not in self._input_series
                or not real.args[ref].array
                or len(self._outputs) != 1
                or real.return_type != 'void'
                or not real.args[self._outputs[0]].is_series
                or not real.args[self._outputs[0]].array
                or real.args[self._outputs[0]].direction != 'out'
            ):
                err_mess = 'return_like requires one TPSA array output and an array input'
                raise ValueError(err_mess)
        inputs = [i for i, a in enumerate(real.args) if not a.is_size and a.direction != 'out']
        self._positionals = tuple(inputs + [i for i in self._outputs if i not in inputs])

        size_lookup = {a.name: i for i, a in enumerate(real.args) if a.is_size}
        mapped = {}
        preceding_size = None
        for arg in real.args:
            if arg.is_size:
                preceding_size = arg.name
            if arg.array and preceding_size is not None:
                mapped[arg.name] = preceding_size
        mapped.update(array_sizes or {})
        for array_name, size_name in mapped.items():
            if array_name not in self._names or not real.args[self._names[array_name]].array:
                err_mess = f'{array_name!r} is not an array argument'
                raise ValueError(err_mess)
            if size_name not in size_lookup:
                err_mess = f'{size_name!r} is not a size argument'
                raise ValueError(err_mess)
        if any(real.args[i].name not in mapped for i in self._parameter_arrays):
            err_mess = 'Parameter-inclusive arrays require a logical array size'
            raise ValueError(err_mess)
        self.array_sizes = mapped
        self._array_size_pairs = tuple(
            (self._names[arr], size_lookup[dim]) for arr, dim in mapped.items()
        )
        defaults = dict(size_defaults or {})
        aliases = dict(size_aliases or {})
        if set(aliases) & set(defaults):
            err_mess = 'A size cannot have both a default and an alias'
            raise ValueError(err_mess)
        defaults.update(aliases)
        if set(aliases) - set(size_lookup) or set(aliases.values()) - set(size_lookup):
            err_mess = 'Unknown size alias'
            raise ValueError(err_mess)
        self._size_aliases = tuple((size_lookup[a], size_lookup[b]) for a, b in aliases.items())
        if set(defaults) - set(size_lookup):
            err_mess = 'Unknown size_defaults key'
            raise ValueError(err_mess)
        for value in defaults.values():
            if isinstance(value, str) and value not in size_lookup:
                err_mess = f'Unknown size-default reference {value!r}'
                raise ValueError(err_mess)
            if not isinstance(value, (str, int)) or isinstance(value, bool):
                err_mess = 'size_defaults values must be size names or integers'
                raise TypeError(err_mess)
        self._size_defaults = tuple(
            (
                size_lookup[name],
                ('ref', size_lookup[value]) if isinstance(value, str) else ('value', value),
            )
            for name, value in defaults.items()
        )
        if set(self.output_factory) - {real.args[i].name for i in self._outputs}:
            err_mess = 'Factory names must be output arguments'
            raise ValueError(err_mess)

        self._native_real: tuple[Any, Any] | None = None
        self._native_complex: tuple[Any, Any] | None = None
        self.fast_call: Callable[..., Any] = self._unbound
        self.is_specialised = False
        self.call_source: str | None = None
        if ffi is not None and lib is not None:
            self.bind(ffi, lib)

    def bind(self, ffi: Any, lib: Any) -> None:
        """Resolve symbols and compile the call plan once the library is loaded."""
        self.ffi, self.lib = ffi, lib
        self._native_real = self._resolve(self.real)
        self._native_complex = self._resolve(self.complex) if self.complex else None
        self.fast_call = self._build_call()
        self.is_specialised = True

    def _unbound(self, *_args: Any, **_kwargs: Any) -> Any:
        err_mess = f'{self.name}: bind the native library before calling'
        raise RuntimeError(err_mess)

    def _resolve(self, sig: Signature) -> tuple[Any, Any] | None:
        try:
            function = self.ffi.addressof(self.lib, sig.name)
            protected = getattr(self.lib, sig.wrapper_name)
        except AttributeError:
            return None  # Gives an informative error only if this flavour is called.
        return function, protected

    def cffi_header(self) -> str:
        """Return the CFFI header for this function."""
        return generate_cffi_header([self])

    def native_header(self) -> str:
        """Return the native C header for this function."""
        return '\n\n'.join(sig.native_header() for sig in unique_signatures([self]))

    def native_source(self) -> str:
        """Return the native C source for this function."""
        return '\n\n'.join(sig.native_source() for sig in unique_signatures([self]))

    @property
    def wrapper_name(self) -> str:
        """Return the name of the wrapper function for this function."""
        return self.real.wrapper_name

    def _align_inputs(self, values: list[Any], *, forced: bool | None) -> bool:
        indexes = self._input_series
        if not indexes:
            return bool(forced)
        series = [values[i] for i in indexes]
        known = [obj for obj in series if hasattr(obj, 'descriptor')]
        inferred_complex = any(_series_is_complex(obj) for obj in known)
        if forced is False and inferred_complex:
            err_mess = 'complex_needed=False is incompatible with complex inputs'
            raise ValueError(err_mess)
        complex_needed = inferred_complex if forced is None else forced
        if not self.align:
            if complex_needed and any(not _series_is_complex(x) for x in known):
                err_mess = 'align=False cannot pass real TPSAs to a complex routine'
                raise TypeError(err_mess)
            return complex_needed
        if len(known) <= 1:
            if complex_needed and known and not _series_is_complex(known[0]):
                for i in indexes:
                    if hasattr(values[i], 'descriptor'):
                        values[i] = _promote(values[i])
            return complex_needed

        first = known[0].descriptor
        descriptors_equal = all(obj.descriptor is first for obj in known[1:])
        if descriptors_equal and not (
            complex_needed and any(not _series_is_complex(obj) for obj in known)
        ):
            return complex_needed  # Hot path: no descriptor union, no TPSA copies.

        if (
            len(indexes) == len(known) == 2
            and self._pair_aligner is None
            and __package__ == 'madng_tpsa'
        ):
            # Only import the alignment function if we're in the main package context.
            # This allows to run MadngFunction in standalone to generate the C source.
            from .tensor import _align_descriptors

            self._pair_aligner = _align_descriptors

        if len(indexes) == 2 and len(known) == 2 and self._pair_aligner is not None:
            left, right, already_complex = self._pair_aligner(
                values[indexes[0]],
                values[indexes[1]],
                extend_left=self.extend_left,
                extend_right=self.extend_right,
                reorder_variables=self.reorder_variables,
            )
            if complex_needed and not already_complex:
                if not _series_is_complex(left):
                    left = _promote(left)
                if not _series_is_complex(right):
                    right = _promote(right)
            values[indexes[0]] = left
            values[indexes[1]] = right
            return complex_needed

        common = first
        for obj in known[1:]:
            if obj.descriptor is not common:
                variables = obj.descriptor.variables
                compatible = (
                    set(variables) == set(common.variables)
                    if self.reorder_variables
                    else variables == common.variables
                )
                if not compatible:
                    err_mess = 'Incompatible descriptor variables or variable order'
                    raise ValueError(err_mess)
                common = common.union(obj.descriptor)
        for pos, i in enumerate(indexes):
            obj = values[i]
            if not hasattr(obj, 'descriptor'):
                continue
            if obj.descriptor is not common:
                extend = (
                    (self.extend_left, self.extend_right)[pos] if pos < 2 else self.extend_vectors
                )
                if extend and hasattr(obj, 'extend_descriptor'):
                    obj = obj.extend_descriptor(common)
                else:
                    obj = obj.change_descriptor(common)
            if complex_needed and not _series_is_complex(obj):
                obj = _promote(obj)
            values[i] = obj
        return complex_needed

    def _convert_array(self, arg: Argument, obj: Any, *, complex_needed: bool = False) -> Any:
        if arg.is_series:
            try:
                return obj.pointer_array  # Cached and zero-copy for TpsaTensor.
            except AttributeError:
                if not isinstance(obj, (tuple, list)):
                    return obj  # Explicit CFFI array/pointer.
                return self.ffi.new('void*[]', [_pointer(item) for item in obj])
        ctype = _C_TYPES[arg.base]
        if complex_needed and self.complex is not None:
            ctype = _C_TYPES[self.complex.args[self._names[arg.name]].base]
        if not isinstance(obj, (tuple, list)):
            try:
                typ = self.ffi.typeof(obj)
                if typ.kind in ('pointer', 'array'):
                    return obj
            except TypeError:
                pass
            fmt = {
                'double': 'd',
                'double _Complex': 'Zd',
                'int': 'i',
                'unsigned char': 'B',
                '_Bool': '?',
            }[ctype]
            try:
                view = memoryview(obj)
            except TypeError:
                view = None
            if (
                view is not None
                and view.ndim == 1
                and view.c_contiguous
                and view.format == fmt
                and (arg.direction == 'in' or not view.readonly)
            ):
                const = 'const ' if arg.const else ''
                return self.ffi.from_buffer(
                    f'{const}{ctype}[]',
                    obj,
                    require_writable=arg.direction != 'in',
                )
            if view is not None and arg.direction != 'in':
                err_mess = (
                    f'Output {arg.name} requires a writable contiguous 1D buffer '
                    'of the correct type'
                )
                raise TypeError(err_mess)
        return self.ffi.new(f'{ctype}[]', list(obj))

    def _output_with_params(
        self, arg: Argument, obj: Any, reference: Any, *, complex_needed: bool
    ) -> tuple[Any, list[Any]]:
        """Prepare writable parameter identities and retain them until the native call."""
        params = list(reference.descriptor.params())
        if complex_needed:
            params = [_promote(p) for p in params]
        if not params:
            return self._convert_array(arg, obj), params
        components = obj.flat if hasattr(obj, 'flat') else obj
        if not isinstance(components, (tuple, list)):
            err_mess = 'Parameter-inclusive outputs require a tensor or list of TPSAs'
            raise TypeError(err_mess)
        return self.ffi.new('void*[]', [_pointer(t) for t in (*components, *params)]), params

    def _convert_pointer(self, arg: Argument, obj: Any) -> Any:
        """Convert a non-TPSA scalar pointer, retaining CFFI buffers."""
        try:
            self.ffi.typeof(obj)
        except TypeError:
            return self.ffi.new(f'{_C_TYPES[arg.base]} *', obj)
        return obj

    def _check_output_type(
        self, index: int, obj: Any, *, complex_needed: bool, descriptor: Any = None
    ) -> None:
        arg = self.arguments[index]
        if not arg.is_series:
            return
        if arg.array and isinstance(obj, (list, tuple)):
            for sample in obj:
                if not hasattr(sample, 'descriptor'):
                    err_mess = f'Output {arg.name} must contain TPSAs'
                    raise TypeError(err_mess)
                if _series_is_complex(sample) != complex_needed:
                    err_mess = f'Output {arg.name} has the wrong real/complex TPSA type'
                    raise TypeError(err_mess)
                if descriptor is not None and sample.descriptor is not descriptor:
                    err_mess = f'Output {arg.name} uses the wrong descriptor'
                    raise ValueError(err_mess)
            return
        if not hasattr(obj, 'descriptor'):
            return  # Raw CFFI pointers: responsibility of caller.
        if _series_is_complex(obj) != complex_needed:
            err_mess = f'Output {arg.name} has the wrong real/complex TPSA type'
            raise TypeError(err_mess)
        if descriptor is not None and obj.descriptor is not descriptor:
            err_mess = f'Output {arg.name} uses the wrong descriptor'
            raise ValueError(err_mess)

    def _raise_native_error(self, first: Any) -> None:
        method = getattr(first, '_raise_mad_error', None)
        if method is None and hasattr(first, 'flat'):
            method = getattr(first.flat[0], '_raise_mad_error', None)
        if method is not None:
            method()
        ffi, lib = self.ffi, self.lib
        loc = ffi.string(lib.madng_tpsa_last_error_location()).decode(errors='replace')
        msg = ffi.string(lib.madng_tpsa_last_error_message()).decode(errors='replace')
        err_mess = f'{self.name}: MAD-NG error at {loc}: {msg}'
        raise RuntimeError(err_mess)

    def _build_call(self) -> Callable[..., Any]:
        """Emit one Python call plan; only the stages needed by this signature run."""
        args = self.arguments
        c_names = [a.name for a in args]
        # Preserve C names where possible. Keywords and names which would shadow
        # the generated implementation receive a unique Python argument name.
        reserved = {
            'complex_needed',
            'range',
            'getattr',
            'isinstance',
            'len',
            'type',
            'bool',
            'int',
            'float',
            'list',
            'tuple',
            'ValueError',
            'TypeError',
            'RuntimeError',
        }
        occupied = set(c_names) | reserved
        names = c_names.copy()
        for i, name in enumerate(c_names):
            if (
                not name.isidentifier()
                or keyword.iskeyword(name)
                or name.startswith('_')
                or name in reserved
            ):
                base = name.lstrip('_')
                alias = (base if base.isidentifier() else f'arg{i}') + '_arg'
                while alias in occupied or keyword.iskeyword(alias):
                    alias += '_'
                names[i] = alias
                occupied.add(alias)
        inputs = [names[i] for i in self._positionals if args[i].direction != 'out']
        outputs = [names[i] for i in self._positionals if args[i].direction == 'out']
        sizes = [names[i] for i in self._size_indexes]
        signature = ', '.join(
            inputs
            + [f'{n}=None' for n in outputs]
            + ['*']
            + [f'{n}=None' for n in sizes]
            + ['complex_needed=None']
        )
        lines = [f'def _call({signature}):']

        def put(line: str, indent: int = 1) -> None:
            lines.append('    ' * indent + line)

        series = [names[i] for i in self._input_series]
        first = series[0] if series else 'None'

        def align_inputs(indent: int) -> None:
            put(f'_vals = [_MISSING] * {len(args)}', indent)
            for i in self._input_series:
                put(f'_vals[{i}] = {names[i]}', indent)
            put('_complex = _self._align_inputs(_vals, forced=complex_needed)', indent)
            for i in self._input_series:
                put(f'{names[i]} = _vals[{i}]', indent)

        # Keep the common unary/binary path straight-line. The same aligner is
        # used for slow paths and for functions with more than two TPSA inputs.
        if len(series) == 2:
            left, right = series
            put(f'_lc = _is_complex({left})')
            put(f'_rc = _is_complex({right})')
            put(f'_d = getattr({left}, "descriptor", None)')
            put(f'_rd = getattr({right}, "descriptor", None)')
            put(
                'if _self.align and (_d is not _rd or _lc != _rc or '
                '(complex_needed is True and not (_lc and _rc))):'
            )
            align_inputs(2)
            put(f'_d = getattr({left}, "descriptor", None)', 2)
            put(f'if _d is None: _d = getattr({right}, "descriptor", None)', 2)
            put('else:')
            put('_complex = (_lc or _rc) if complex_needed is None else complex_needed', 2)
            put('if _complex is False and (_lc or _rc):', 2)
            put("raise ValueError('complex_needed=False conflicts with complex inputs')", 3)
            put(
                'if not _self.align and _complex and ('
                '(_d is not None and not _lc) or (_rd is not None and not _rc)):',
                2,
            )
            put("raise TypeError('align=False cannot pass real TPSAs to a complex routine')", 3)
            put('if _d is None: _d = _rd', 2)
        elif len(series) == 1:
            put(f'_detected = _is_complex({first})')
            put(f'_d = getattr({first}, "descriptor", None)')
            put('_complex = _detected if complex_needed is None else complex_needed')
            put('if _detected and not _complex:')
            put("raise ValueError('complex_needed=False conflicts with complex inputs')", 2)
            put('if _complex and not _detected and _d is not None:')
            put('if not _self.align:', 2)
            put("raise TypeError('align=False cannot pass real TPSAs to a complex routine')", 3)
            align_inputs(2)
            put(f'_d = getattr({first}, "descriptor", None)', 2)
        elif series:
            align_inputs(1)
            put('_d = None')
            for name in series:
                put(f'if _d is None: _d = getattr({name}, "descriptor", None)')
        else:
            put('_complex = bool(complex_needed)')
            put('_d = None')
        if self.complex is None:
            put('if _complex:')
            put("raise TypeError('No complex variant configured')", 2)
        put('_native = _self._native_complex if _complex else _self._native_real')
        put('if _native is None:')
        put("raise RuntimeError('Complex or real native function/wrapper is unavailable')", 2)
        put('_fn, _protected = _native')

        # Infer from supplied arrays first, then apply defaults. Alias groups
        # are resolved at generation time, so a native count is never confused
        # with the logical length of its Python array.
        for i in self._size_indexes:
            size = names[i]
            candidate_indexes = sorted(
                (ai for ai, si in self._array_size_pairs if si == i),
                key=lambda ai: args[ai].direction == 'out',
            )
            for ai in candidate_indexes:
                candidate = names[ai]
                put(f'if {size} is None and {candidate} is not None:')
                put(f'{size} = _length({candidate})', 2)
            put(f'if {size} is not None and type({size}) is not int:')
            put(f'{size} = _size_int({size})', 2)
        groups = [[i] for i in self._size_indexes]
        for ai, bi in self._size_aliases:
            a = next(g for g in groups if ai in g)
            b = next(g for g in groups if bi in g)
            if a is not b:
                a.extend(b)
                groups.remove(b)
        alias_targets = {a for a, _ in self._size_aliases}
        defaults = {i: d for i, d in self._size_defaults if i not in alias_targets}
        done: set[int] = set()
        visiting: set[int] = set()

        def emit_size_group(group: list[int]) -> None:
            key = group[0]
            if key in done:
                return
            if key in visiting:
                err_mess = 'Cyclic size defaults'
                raise ValueError(err_mess)
            visiting.add(key)
            size = names[key]
            for i in group[1:]:
                other = names[i]
                put(f'if {size} is None: {size} = {other}')
                put(f'elif {other} is not None and {size} != {other}:')
                put(f"raise ValueError('{size} must equal {other}')", 2)
            for i in group:
                if i not in defaults:
                    continue
                kind, value = defaults[i]
                if kind == 'ref':
                    emit_size_group(next(g for g in groups if value in g))
                    value_expr = names[value]
                else:
                    value_expr = repr(value)
                put(f'if {size} is None: {size} = {value_expr}')
            put(f'if {size} is None:')
            put(f"raise TypeError('Cannot infer {size}; pass it explicitly')", 2)
            put(f'if {size} < 0:')
            put(f"raise ValueError('{size} must be non-negative')", 2)
            for i in group[1:]:
                put(f'{names[i]} = {size}')
            visiting.remove(key)
            done.add(key)

        for group in groups:
            emit_size_group(group)

        def check_length(i: int, size: str) -> None:
            if not self.check_lengths:
                return
            name = names[i]
            put(f'_n = _length({name})')
            # Unbounded CFFI pointers have no Python length. They are the
            # caller's responsibility; parameter-inclusive arrays are not raw.
            condition = (
                f'_n != {size}'
                if i in self._parameter_arrays
                else (f'_n is not None and _n != {size}')
            )
            put(f'if {condition}:')
            put(f"raise ValueError('{name} has a length inconsistent with {size}')", 2)

        for ai, si in self._array_size_pairs:
            if args[ai].direction != 'out':
                check_length(ai, names[si])

        if self.return_like is not None:
            put(f'_allocate = {names[self._outputs[0]]} is None')
        for i in self._outputs:
            arg = args[i]
            name = names[i]
            c_name = arg.name
            dim = self.array_sizes.get(c_name)
            size = names[self._names[dim]] if dim is not None else None
            if arg.direction == 'out':
                put(f'if {name} is None:')
                if c_name in self.output_factory:
                    # Rare custom allocations may inspect all supplied inputs,
                    # resolved dimensions, and already allocated/supplied outputs.
                    context = ', '.join(
                        f'{a.name!r}: {n}' for a, n in zip(args, names, strict=True)
                    )
                    put(f'_context = {{{context}}}', 2)
                    for j in self._outputs:
                        if args[j].direction == 'out':
                            put(f'if {names[j]} is None: _context.pop({args[j].name!r})', 2)
                    put(
                        f'{name} = _self.output_factory[{c_name!r}]({size or "None"}, _context)',
                        2,
                    )
                elif arg.array and size is None:
                    put(
                        f"raise TypeError('Cannot infer length of output {name}; "
                        "pass it explicitly')",
                        2,
                    )
                elif arg.is_series:
                    put('if _d is None:', 2)
                    put(f"raise TypeError('Cannot infer descriptor for output {name}')", 3)
                    put('_zero = _d.complex_zero if _complex else _d.zero', 2)
                    put(
                        '_order = _d.order if _self.output_order is None else _self.output_order',
                        2,
                    )
                    value = (
                        f'[_zero(order=_order) for _ in range({size})]'
                        if arg.array
                        else '_zero(order=_order)'
                    )
                    put(f'{name} = {value}', 2)
                else:
                    ctype = _C_TYPES[arg.base]
                    complex_ctype = (
                        _C_TYPES[self.complex.args[i].base] if self.complex is not None else ctype
                    )
                    suffix = '[]' if arg.array else ' *'
                    native_type = (
                        repr(ctype + suffix)
                        if complex_ctype == ctype
                        else f'({(complex_ctype + suffix)!r} if _complex else {(ctype + suffix)!r})'
                    )
                    size_arg = f', {size}' if arg.array else ''
                    put(f'{name} = _ffi.new({native_type}{size_arg})', 2)
                if arg.is_series and c_name not in self.output_factory:
                    put('else:')
                    put(
                        f'_self._check_output_type({i}, {name}, complex_needed=_complex, '
                        'descriptor=_d)',
                        2,
                    )
            if arg.is_series and (arg.direction == 'inout' or c_name in self.output_factory):
                put(
                    f'_self._check_output_type({i}, {name}, complex_needed=_complex, descriptor=_d)'
                )
            if arg.array and size is not None and arg.direction == 'out':
                check_length(i, size)

        for i, arg in enumerate(args):
            name = names[i]
            if i in self._native_size_rules:
                sign, source = self._native_size_rules[i]
                expr = name if source is None else f'{name} + {names[source]}.descriptor.num_params'
                put(f'_c_{name} = {sign} * ({expr})')
            elif i in self._parameter_arrays:
                if arg.direction == 'out':
                    put(
                        f'_c_{name}, _owners_{name} = _self._output_with_params('
                        f'_by_name[{arg.name!r}], {name}, {first}, complex_needed=_complex)'
                    )
                else:
                    put(f'_c_{name} = {name}.pointer_array_with_params')
            elif arg.array:
                put(
                    f'_c_{name} = _self._convert_array('
                    f'_by_name[{arg.name!r}], {name}, complex_needed=_complex)'
                )
            elif arg.is_series:
                put(f'_c_{name} = _pointer({name})')
            elif arg.pointer:
                put(f'_c_{name} = _self._convert_pointer(_by_name[{arg.name!r}], {name})')
            elif arg.base in _INT:
                put(f'_c_{name} = {name} if type({name}) is int else int({name})')
            else:
                put(f'_c_{name} = {name} if type({name}) is float else float({name})')
        params = ['_fn', *('_c_' + n for n in names)]
        if self.real.return_type != 'void':
            put(f"_return = _ffi.new('{_C_TYPES[self.real.return_type]} *')")
            params.append('_return')
        put(f'if _protected({", ".join(params)}):')
        put(f'_self._raise_native_error({first})', 2)

        results = [] if self.real.return_type == 'void' else ['_return[0]']
        for i in self._outputs:
            arg = args[i]
            name = names[i]
            if not arg.is_series and arg.array:
                results.append(f'_array_result({name}, _c_{name})')
            elif not arg.is_series and arg.pointer:
                results.append(f'_c_{name}[0]')
            else:
                results.append(name)
        if self.return_like is not None:
            put('if _allocate:')
            reference = names[self._names[self.return_like]]
            put(f'return {reference}._new_like({names[self._outputs[0]]})', 2)
        put('return ' + (', '.join(results) if results else 'None'))

        namespace: dict[str, Any] = {
            '_self': self,
            '_ffi': self.ffi,
            '_pointer': _pointer,
            '_is_complex': _series_is_complex,
            '_length': _array_length,
            '_size_int': _size_int,
            '_array_result': _array_result,
            '_MISSING': _MISSING,
            '_by_name': {a.name: a for a in args},
        }
        src = '\n'.join(lines)
        self.call_source = src
        exec(compile(src, f'<MAD-NG call {self.name}>', 'exec'), namespace)
        hot = namespace['_call']
        hot.__doc__ = f'Specialised {self.name} call; compiled once from its C signature.'
        return hot

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the precompiled call plan."""
        return self.fast_call(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the precompiled call plan."""
        return self.fast_call(*args, **kwargs)


def unique_signatures(functions: Sequence[MadngFunction]) -> tuple[Signature, ...]:
    """Return the distinct native signatures, in definition order."""
    wrappers: dict[str, Signature] = {}
    for func in functions:
        for sig in (func.real, func.complex):
            if sig is not None:
                wrappers[sig.tag] = sig
    return tuple(wrappers.values())


def generate_cffi_header(functions: Sequence[MadngFunction]) -> str:
    """Generate native symbols and deduplicated protected CFFI declarations."""
    symbols = {
        sig.name: sig.cffi_symbol()
        for func in functions
        for sig in (func.real, func.complex)
        if sig is not None
    }
    wrappers = (sig.cffi_wrapper() for sig in unique_signatures(functions))
    return '\n'.join((*symbols.values(), '', *wrappers))
