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
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


_C_TYPES = {
    'void': 'void',
    'ssz_t': 'int',
    'idx_t': 'int',
    'int': 'int',
    'num_t': 'double',
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
    # Current TpsaMap branch uses .ptr; old integrations used .ref.
    try:
        return obj.ptr
    except AttributeError:
        return obj.ref


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
        specialise: bool = True,
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
        self._specialise = specialise
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
        self.fast_call: Callable[..., Any] = self._call_generic
        self.is_specialised = False
        if ffi is not None and lib is not None:
            self.bind(ffi, lib)

    def bind(self, ffi: Any, lib: Any) -> None:
        """Resolve symbols and compile the call plan once the library is loaded."""
        self.ffi, self.lib = ffi, lib
        self._native_real = self._resolve(self.real)
        self._native_complex = self._resolve(self.complex) if self.complex else None
        hot = self._build_specialised_call() if self._specialise else None
        self.fast_call = hot if hot is not None else self._call_generic
        self.is_specialised = hot is not None

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

        if self._pair_aligner is None and __package__ == 'madng_tpsa':
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

    def _convert_array(self, arg: Argument, obj: Any) -> Any:
        if arg.is_series:
            try:
                return obj.pointer_array  # Cached and zero-copy for TpsaTensor.
            except AttributeError:
                if not isinstance(obj, (tuple, list)):
                    return obj  # Explicit CFFI array/pointer.
                return self.ffi.new('void*[]', [_pointer(item) for item in obj])
        if not isinstance(obj, (tuple, list)):
            try:
                typ = self.ffi.typeof(obj)
                if typ.kind in ('pointer', 'array'):
                    return obj
            except TypeError:
                pass
            fmt = {'double': 'd', 'int': 'i', 'unsigned char': 'B', '_Bool': '?'}[
                _C_TYPES[arg.base]
            ]
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
                    f'{const}{_C_TYPES[arg.base]}[]',
                    obj,
                    require_writable=arg.direction != 'in',
                )
        return self.ffi.new(f'{_C_TYPES[arg.base]}[]', list(obj))

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

    def _convert(self, arg: Argument, obj: Any) -> Any:
        if arg.array:
            return self._convert_array(arg, obj)
        if arg.is_series:
            return _pointer(obj)
        if arg.pointer:
            try:
                self.ffi.typeof(obj)
                return obj
            except TypeError:
                return self.ffi.new(f'{_C_TYPES[arg.base]} *', None if obj is None else obj)
        if arg.base in _INT:
            return obj if type(obj) is int else int(obj)
        if arg.base == 'num_t':
            return obj if type(obj) is float else float(obj)
        err_mess = f'Unsupported value for {arg.name}'
        raise TypeError(err_mess)

    def _allocate_output(self, index: int, values: list[Any], *, complex_needed: bool) -> Any:
        arg = self.arguments[index]
        factory = self.output_factory.get(arg.name)
        if factory is not None:
            sizes = {
                self.arguments[i].name: values[i]
                for i in self._size_indexes
                if values[i] is not _MISSING
            }
            context = {
                a.name: v for a, v in zip(self.arguments, values, strict=True) if v is not _MISSING
            }
            dim = self.array_sizes.get(arg.name)
            return factory(sizes.get(dim), context)
        if not arg.is_series:
            if arg.array:
                dim = self.array_sizes.get(arg.name)
                if dim is None or values[self._names[dim]] is _MISSING:
                    err_mess = f'Cannot infer length of output {arg.name}; pass it explicitly'
                    raise TypeError(err_mess)
                return self.ffi.new(f'{_C_TYPES[arg.base]}[]', values[self._names[dim]])
            if arg.pointer:
                return self.ffi.new(f'{_C_TYPES[arg.base]} *')
            err_mess = f'Cannot allocate {arg.name}'
            raise TypeError(err_mess)
        series = next(
            (values[i] for i in self._input_series if hasattr(values[i], 'descriptor')), None
        )
        if series is None:
            err_mess = f'Cannot infer descriptor for output {arg.name}'
            raise TypeError(err_mess)
        descriptor = series.descriptor
        zero = descriptor.complex_zero if complex_needed else descriptor.zero
        order = descriptor.order if self.output_order is None else self.output_order
        if not arg.array:
            return zero(order=order)
        dim_name = self.array_sizes.get(arg.name)
        if dim_name is None or values[self._names[dim_name]] is _MISSING:
            err_mess = f'Cannot infer length of output {arg.name}; pass it explicitly'
            raise TypeError(err_mess)
        n = values[self._names[dim_name]]
        return [zero(order=order) for _ in range(n)]

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

    def _build_specialised_call(self) -> Callable[..., Any] | None:
        """Compile a Python call plan once, with a natural Python signature.

        This is an optimisation, not another handwritten wrapper. The emitted
        function is specialised to its C prototype. Unsupported signatures
        retain the fully checked generic call().
        """
        args = self.arguments
        if (
            not self._input_series
            or len(self._input_series) > 2
            or any(a.direction == 'inout' for a in args)
            or any(a.pointer and not a.is_series for a in args)
            or set(self.output_factory)
        ):
            return None
        all_names = [a.name for a in args]
        if any(keyword.iskeyword(name) or name == 'complex_needed' for name in all_names):
            return None
        inputs = [args[i].name for i in self._positionals if args[i].direction != 'out']
        outputs = [args[i].name for i in self._positionals if args[i].direction == 'out']
        sizes = [args[i].name for i in self._size_indexes]
        positional = inputs + [name + '=None' for name in outputs]
        keywords = [name + '=None' for name in sizes] + ['complex_needed=None']
        signature = ', '.join(positional + (['*'] + keywords if keywords else []))
        lines = [f'def _hot({signature}):']

        def put(line: str, indent: int = 1) -> None:
            lines.append('    ' * indent + line)

        names = [args[i].name for i in self._input_series]
        first = names[0]
        if len(names) == 2:
            left, right = names
            put(f'_lc = _is_complex({left})')
            put(f'_rc = _is_complex({right})')
            put(
                f'if _self.align and ({left}.descriptor is not {right}.descriptor or '
                f'_lc != _rc or (complex_needed is True and not (_lc and _rc))):'
            )
            put(f'_vals = [_MISSING] * {len(args)}', 2)
            put(f'_vals[{self._input_series[0]}] = {left}', 2)
            put(f'_vals[{self._input_series[1]}] = {right}', 2)
            put('_complex = _self._align_inputs(_vals, forced=complex_needed)', 2)
            put(f'{left} = _vals[{self._input_series[0]}]', 2)
            put(f'{right} = _vals[{self._input_series[1]}]', 2)
            put('else:')
            put('_complex = (_lc or _rc) if complex_needed is None else complex_needed', 2)
            put('if _complex is False and (_lc or _rc):', 2)
            put("raise ValueError('complex_needed=False conflicts with complex inputs')", 3)
            put('if not _self.align and _complex and not (_lc and _rc):', 2)
            put("raise TypeError('align=False cannot pass real TPSAs to a complex routine')", 3)
        else:
            put(f'_detected = _is_complex({first})')
            put('_complex = _detected if complex_needed is None else complex_needed')
            put('if _detected and not _complex:')
            put("raise ValueError('complex_needed=False conflicts with complex inputs')", 2)
            put('if _complex and not _detected:')
            put('if not _self.align:', 2)
            put("raise TypeError('align=False cannot pass real TPSAs to a complex routine')", 3)
            put(f'_vals = [_MISSING] * {len(args)}', 2)
            put(f'_vals[{self._input_series[0]}] = {first}', 2)
            put('_complex = _self._align_inputs(_vals, forced=complex_needed)', 2)
            put(f'{first} = _vals[{self._input_series[0]}]', 2)
        put('_native = _self._native_complex if _complex else _self._native_real')
        put('if _native is None:')
        put("raise RuntimeError('Complex or real native function/wrapper is unavailable')", 2)
        put('_fn, _protected = _native')

        # Dimensions are derived from input arrays preferentially. Shared sizes
        # incur one length check per additional array, unless disabled.
        for index in self._size_indexes:
            size = args[index].name
            candidates = [
                args[ia].name
                for ia, idim in self._array_size_pairs
                if idim == index and args[ia].direction != 'out'
            ]
            for candidate in candidates:
                put(f'if {size} is None:')
                put(f'{size} = _length({candidate})', 2)
                break
            default = dict(self._size_defaults).get(index)
            if default is not None:
                kind, value = default
                value_expr = args[value].name if kind == 'ref' else repr(value)
                put(f'if {size} is None:')
                put(f'{size} = {value_expr}', 2)
            output_candidates = [
                args[ia].name
                for ia, idim in self._array_size_pairs
                if idim == index and args[ia].direction == 'out'
            ]
            if output_candidates:
                put(f'if {size} is None and {output_candidates[0]} is not None:')
                put(f'{size} = _length({output_candidates[0]})', 2)
            put(f'if {size} is None:')
            put(f"raise TypeError('Cannot infer {size}; pass it explicitly')", 2)
            put(f'if {size} < 0:')
            put(f"raise ValueError('{size} must be non-negative')", 2)
        for ai, bi in self._size_aliases:
            an, bn = args[ai].name, args[bi].name
            put(f'if {an} != {bn}:')
            put(f"raise ValueError('{an} must equal {bn}')", 2)
        if self.check_lengths:
            for ai, si in self._array_size_pairs:
                arg = args[ai]
                if arg.direction == 'out':
                    continue
                put(f'if _length({arg.name}) != {args[si].name}:')
                put(
                    f"raise ValueError('{arg.name} has a length inconsistent with "
                    f"{args[si].name}')",
                    2,
                )

        if self.return_like is not None:
            put(f'_allocate = {args[self._outputs[0]].name} is None')
        put(f'_d = {first}.descriptor')
        put('_order = _d.order if _self.output_order is None else _self.output_order')
        put('_zero = _d.complex_zero if _complex else _d.zero')
        for i in self._outputs:
            arg = args[i]
            name = arg.name
            put(f'if {name} is None:')
            if arg.is_series:
                if arg.array:
                    size = self.array_sizes.get(name)
                    if size is None:
                        return None  # Need per-output shape policy.
                    put(f'{name} = [_zero(order=_order) for _ in range({size})]', 2)
                else:
                    put(f'{name} = _zero(order=_order)', 2)
            elif arg.array:
                size = self.array_sizes.get(name)
                if size is None:
                    return None
                put(f"{name} = _ffi.new('{_C_TYPES[arg.base]}[]', {size})", 2)
            else:
                return None
            if arg.is_series:
                put('else:')
                put(
                    f'_self._check_output_type({i}, {name}, complex_needed=_complex, '
                    f'descriptor=_d)',
                    2,
                )
            if self.check_lengths and arg.array:
                put(f'if _length({name}) != {self.array_sizes[name]}:')
                put(f"raise ValueError('Output {name} has inconsistent length')", 2)

        # Convert in native argument order. Tensor.pointer_array is reused
        # without making an intermediate list or copying pointers.
        for i, arg in enumerate(args):
            name = arg.name
            if i in self._native_size_rules:
                sign, source = self._native_size_rules[i]
                expr = (
                    name
                    if source is None
                    else f'{name} + {args[source].name}.descriptor.num_params'
                )
                put(f'_c_{name} = {sign} * ({expr})')
            elif i in self._parameter_arrays:
                if arg.direction == 'out':
                    put(
                        f'_c_{name}, _owners_{name} = _self._output_with_params('
                        f'_by_name[{name!r}], {name}, {first}, complex_needed=_complex)'
                    )
                else:
                    put(f'_c_{name} = {name}.pointer_array_with_params')
            elif arg.is_series:
                if arg.array:
                    put(f'_c_{name} = _self._convert_array(_by_name[{name!r}], {name})')
                else:
                    put(f'_c_{name} = _pointer({name})')
            elif arg.array:
                put(f'_c_{name} = _self._convert_array(_by_name[{name!r}], {name})')
            else:
                put(f'_c_{name} = {name}')
        if self.real.return_type != 'void':
            put(f"_return = _ffi.new('{_C_TYPES[self.real.return_type]} *')")
        params = ', '.join('_c_' + a.name for a in args)
        if self.real.return_type != 'void':
            params += ', _return' if params else '_return'
        put(f'_status = _protected(_fn{", " if params else ""}{params})')
        put('if _status:')
        put(f'_self._raise_native_error({first})', 2)
        result_exprs = []
        if self.real.return_type != 'void':
            result_exprs.append('_return[0]')
        for i in self._outputs:
            arg = args[i]
            name = arg.name
            if arg.array and not arg.is_series:
                put(f'if {name} is _c_{name} or isinstance({name}, (list,tuple)):')
                put(f'_res_{name} = [_c_{name}[j] for j in range(_length({name}))]', 2)
                put('else:')
                put(f'_res_{name} = {name}', 2)
                result_exprs.append('_res_' + name)
            else:
                result_exprs.append(name)
        if self.return_like is not None:
            put('if _allocate:')
            put(f'return {self.return_like}._new_like({args[self._outputs[0]].name})', 2)
        if not result_exprs:
            put('return None')
        elif len(result_exprs) == 1:
            put(f'return {result_exprs[0]}')
        else:
            put('return (' + ', '.join(result_exprs) + ')')

        namespace: dict[str, Any] = {
            '_self': self,
            '_ffi': self.ffi,
            '_pointer': _pointer,
            '_is_complex': _series_is_complex,
            '_length': _array_length,
            '_MISSING': _MISSING,
            '_by_name': {a.name: a for a in args},
        }
        src = '\n'.join(lines)
        exec(compile(src, f'<MAD-NG call {self.name}>', 'exec'), namespace)
        hot = namespace['_hot']
        hot.__doc__ = f'Specialised {self.name} call; compiled once from its C signature.'
        return hot

    def _call_generic(self, *args: Any, complex_needed: bool | None = None, **kwargs: Any) -> Any:
        """Dispatch a generic call, used for signatures we cannot specialise.

        Positional arguments omit ssz_t dimensions. All input arguments come
        first; optional output arguments follow. Outputs may be omitted.
        """
        if self.ffi is None or self.lib is None:
            err_mess = 'ffi and lib are required for runtime calls'
            raise RuntimeError(err_mess)
        if len(args) > len(self._positionals):
            err_mess = f'{self.name}: too many arguments'
            raise TypeError(err_mess)
        values: list[Any] = [_MISSING] * len(self.arguments)
        for pos, value in zip(self._positionals, args, strict=False):
            values[pos] = value
        if kwargs:
            for key, value in kwargs.items():
                idx = self._names.get(key)
                if idx is None:
                    err_mess = f'{self.name}: unknown argument {key}'
                    raise TypeError(err_mess)
                if values[idx] is not _MISSING:
                    err_mess = f'{self.name}: duplicate argument {key}'
                    raise TypeError(err_mess)
                values[idx] = value
        for i, arg in enumerate(self.arguments):
            if arg.direction != 'out' and not arg.is_size and values[i] is _MISSING:
                err_mess = f'{self.name}: missing input {arg.name}'
                raise TypeError(err_mess)

        need_complex = self._align_inputs(values, forced=complex_needed)
        if need_complex and self.complex is None:
            err_mess = f'{self.name}: no complex variant configured'
            raise TypeError(err_mess)
        native = self._native_complex if need_complex else self._native_real
        if native is None:
            variant = self.complex.name if need_complex and self.complex else self.real.name
            err_mess = f'Native symbol or protected wrapper for {variant} is unavailable'
            raise RuntimeError(err_mess)

        # Infer and validate dimensions before allocating outputs. Pair order is
        # compiled in the constructor, so there is no name lookup on the hot path.
        for array_idx, size_idx in self._array_size_pairs:
            value = values[array_idx]
            if value is _MISSING:
                continue
            length = _array_length(value)
            if length is None:
                continue  # Raw CFFI pointer with unknown length.
            current = values[size_idx]
            if current is not _MISSING and current != length:
                err_mess = f'{self.arguments[array_idx].name} has length {length}, '
                err_mess += f'but {self.arguments[size_idx].name}={current}'
                raise ValueError(err_mess)
            values[size_idx] = length
        for i, (kind, default) in self._size_defaults:
            if values[i] is _MISSING:
                values[i] = values[default] if kind == 'ref' else default
        for i in self._size_indexes:
            size = values[i]
            if size is _MISSING:
                err_mess = f'{self.name}: cannot infer {self.arguments[i].name}'
                raise TypeError(err_mess)
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                err_mess = f'{self.arguments[i].name} must be a non-negative integer'
                raise TypeError(err_mess)
        for ai, bi in self._size_aliases:
            if values[ai] != values[bi]:
                err_mess = f'{self.arguments[ai].name} must equal {self.arguments[bi].name}'
                raise ValueError(err_mess)
        primary_descriptor = next(
            (values[i].descriptor for i in self._input_series if hasattr(values[i], 'descriptor')),
            None,
        )
        allocate_like = self.return_like is not None and (
            values[self._outputs[0]] is _MISSING or values[self._outputs[0]] is None
        )
        for i in self._outputs:
            if values[i] is _MISSING or values[i] is None:
                values[i] = self._allocate_output(i, values, complex_needed=need_complex)
            self._check_output_type(
                i, values[i], complex_needed=need_complex, descriptor=primary_descriptor
            )
        for array_idx, size_idx in self._array_size_pairs:
            obj = values[array_idx]
            length = _array_length(obj)
            if length is not None and length != values[size_idx]:
                err_mess = f'{self.arguments[array_idx].name} has length {length}, '
                err_mess += f'expected {values[size_idx]}'
                raise ValueError(err_mess)

        ffi = self.ffi
        converted = []
        owners = []
        for i, (arg, val) in enumerate(zip(self.arguments, values, strict=True)):
            if i in self._native_size_rules:
                sign, source = self._native_size_rules[i]
                count = val if source is None else val + values[source].descriptor.num_params
                converted.append(sign * count)
            elif i in self._parameter_arrays:
                if arg.direction == 'out':
                    pointer, params = self._output_with_params(
                        arg, val, values[self._input_series[0]], complex_needed=need_complex
                    )
                    owners.append(params)
                    converted.append(pointer)
                else:
                    converted.append(val.pointer_array_with_params)
            else:
                converted.append(self._convert(arg, val))
        ret = None
        if self.real.return_type != 'void':
            ret = ffi.new(f'{_C_TYPES[self.real.return_type]} *')
            converted.append(ret)
        function, protected = native
        status = protected(function, *converted)
        if status:
            first = values[self._input_series[0]] if self._input_series else None
            self._raise_native_error(first)

        if self.return_like is not None and allocate_like:
            return values[self._names[self.return_like]]._new_like(values[self._outputs[0]])
        results = [] if ret is None else [ret[0]]
        for i in self._outputs:
            arg = self.arguments[i]
            obj = values[i]
            if not arg.is_series and arg.array:
                c_obj = converted[i]
                if isinstance(obj, (list, tuple)) or obj is c_obj:
                    # For a Python list, CFFI operates on a temporary buffer;
                    # auto-allocated native arrays also need Python conversion.
                    n = _array_length(obj)
                    if n is None:
                        err_mess = f'Cannot determine length of output {arg.name}'
                        raise TypeError(err_mess)
                    results.append([c_obj[j] for j in range(n)])
                else:
                    results.append(obj)
            elif not arg.is_series and arg.pointer:
                results.append(converted[i][0])
            else:
                results.append(obj)
        if not results:
            return None
        return results[0] if len(results) == 1 else tuple(results)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the precompiled (or generic fallback) call plan."""
        return self.fast_call(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the precompiled (or generic fallback) call plan."""
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
