"""MAD-NG declarations and Python call policies (also imported at build time).

Adding a wrapper
----------------
Define one ``MadngFunction`` from the real MAD-NG C prototype. It is
registered automatically in ``FUNCTIONS``, which is used to generate the
native protected wrappers during the build and to bind the Python callables
when the library is loaded. No separate C or CFFI declaration is needed.

By default, the ``mad_ctpsa_*`` counterpart is derived from ``mad_tpsa_*``
by replacing ``tpsa_t`` with ``ctpsa_t``. Pass ``auto_complex=False`` when
there is no complex counterpart, ``complex_name`` to override its symbol,
or ``complex_declaration`` to provide an explicit counterpart. The latter
must currently have the same argument structure and return convention;
complex numeric arrays can use cpx_t in an explicit complex declaration;
complex numeric scalars and return values still need additional support.

Calling a wrapper
-----------------
Inputs are positional or keyword arguments, followed by optional output
arguments. Omit ``ssz_t`` dimensions from the positional arguments; they
are inferred or supplied by name. For example, ``LIEBRA(F, G)`` allocates
an output, while ``LIEBRA(F, G, out)`` writes into a supplied output.
``VEC2FLD(f, na=f.descriptor.num_vars)`` needs an explicit dimension because
its input is a scalar TPSA. ``fast_call`` accesses the same generated plan
without the extra ``MadngFunction.__call__`` method dispatch. Inspect
``function.call_source`` after binding to see the generated Python code.
C names conflicting with Python keywords or generated locals receive a
unique safe ``_arg`` suffix in the generated Python signature.

Const pointers/arrays and by-value arguments are inputs; non-const pointers
and arrays default to outputs. Use ``directions={'select': 'in'}`` for a
non-const C input, or ``'inout'`` for a required input which is also returned.
Directions are Python-side policies, not C type qualifiers. An `idx_t *`
input, such as PMINV's parameter-inclusive selection buffer, is a
pointer rather than a sized array and must be supplied as CFFI storage.
A non-void C return value precedes the output arguments in the result. A
single result is returned directly; multiple results form a tuple, and a
void function with no outputs returns None. Numeric array outputs supplied
as writable, contiguous buffers of the correct type are updated in place.
Python list outputs use temporary native storage and return a new list.

Dimensions and pointer arrays
-----------------------------
A logical size is the number of Python TPSA components. A native size is the
integer actually passed to C. A pointer-buffer length is the number of
allocated pointers. These are usually equal, but need not be, for example
when a map has six phase-space variables and two parameters.

``array_sizes={'mc': 'na'}``
    Associate array ``mc`` with the logical size ``na``. By default each
    array uses the nearest preceding ``ssz_t`` argument. Override only the
    exceptions. For compose, the output has the size of the left tensor,
    not the size of the substitution map.
``size_aliases={'nb': 'na'}``
    Require the two logical sizes to be equal. Either can determine the
    other, and inconsistent supplied values or array lengths are rejected.
    This is the rule for full inversion; it does not equate native sizes.
``size_defaults={'nb': 'na'}`` or ``size_defaults={'nb': 6}``
    Supply a fallback logical dimension, used only when no explicit or
    inferable size is available. Unlike an alias, a default can be overridden.
    References to other sizes are allowed, including forward references;
    cyclic defaults are invalid.
``native_sizes={'na': NativeSize(sign=-1)}``
    Transform a logical size only when passing it to C. ``NativeSize`` takes
    ``sign`` (+1 by default, or -1) and optional ``params_from``, the C name
    of an input TPSA or tensor whose aligned descriptor supplies the parameter
    count. The native value is ``sign * (logical_size + num_params)`` (zero
    parameters if ``params_from`` is omitted). For a logical size of six and
    two parameters in ``ma``, ``NativeSize()`` gives 6,
    ``NativeSize(sign=-1)`` gives -6, ``NativeSize(params_from='ma')`` gives 8,
    and combining both gives -8. Logical allocation and length checks are
    unchanged. A negative ``na`` is the MAD-NG convention that allows compose
    to act on an arbitrary number of components.
``parameter_arrays=('ma', 'mc')``
    Include parameter identities in the native pointer buffers of these
    TPSA arrays. An input uses the map's ``pointer_array_with_params``. For
    an output, fresh writable parameter identities are allocated, kept
    alive throughout the call, and excluded from the returned components.
    This does not change the logical length or automatically add parameters
    to a native size; configure ``native_sizes`` separately as needed.
    Parameter-inclusive arrays require logical size checks.
``check_lengths=True``
    Validate known array lengths against their logical dimensions. Disable
    only for a trusted calling convention. Raw CFFI pointers do not expose
    their descriptors or necessarily their lengths; the caller must supply
    the correct sizes, real/complex flavour, descriptor compatibility, and
    live storage. Use Python TPSAs/tensors for automatic alignment.

Descriptor alignment and real/complex dispatch
----------------------------------------------
The common path reuses TPSAs that already have the same descriptor and
flavour. Otherwise the inputs are aligned and real operands are promoted
when a complex operand is present. At a call site, ``complex_needed=True``
can explicitly request complex promotion (also needed for complex raw
pointers or an output-only complex operation). False rejects known complex
inputs. Omitting it selects the flavour from the inputs.

``align=True``
    Enable descriptor alignment and promotion. With False, the caller must
    supply compatible inputs; known real operands cannot be sent to a complex
    routine.
``extend_vectors=True``
    Default for the descriptor-alignment extension policy. A vector can use
    ``extend_descriptor`` rather than just ``change_descriptor``. The former
    preserves the vector's coordinate semantics (identity components for
    maps, zeros for vector fields when extending a coordinate set).
``extend_left`` and ``extend_right``
    Override ``extend_vectors`` for the first and second TPSA inputs.
    The default (None) inherits the common policy. Compose disables extension
    of the left tensor and enables it for the right substitution map.
``reorder_variables=False``
    Require matching variable order by default. True permits matching sets
    of variable labels in different orders, as needed by compose. The current
    pair aligner does not accept arbitrary different variable sets; these
    options do not override descriptor compatibility checks.
``aligner``
    Optional custom binary alignment function, mainly useful for tests.
    Normally the package's ``_align_descriptors`` is used.

Output allocation
-----------------
``output_order``
    Truncation order of automatically allocated TPSAs. None (the default)
    uses the aligned input descriptor's order.
``output_factory={'r': factory}``
    Override allocation for the named output. ``factory(size, context)``
    receives its logical length (or None for a scalar/unmapped array) and
    a mapping of aligned inputs, resolved sizes and available outputs, keyed
    by the original C argument names. Useful when the descriptor cannot be
    inferred from an input or the output has unusual allocation requirements.
``return_like='ma'``
    For a void function with exactly one TPSA array output, reconstruct an
    automatically allocated result via the named *aligned* input's
    ``_new_like`` method, preserving container type/shape and the resulting
    descriptor. Without it, return the list of TPSAs. A supplied output is
    returned as-is, rather than reconstructed. Inout operands that must be
    aligned/promoted can be replaced by compatible objects; use the returned
    values rather than assuming an original incompatible object was mutated.


The optional ``ffi`` and ``lib`` constructor arguments bind a standalone
function immediately. Registry definitions leave them unset; the package
binds all functions once, at import. All supported signatures use the same
specialised call generator, not a separate generic dispatcher.

Inspecting autogenerated call signatures
----------------------------------------
The following example shows how to inspect the autogenerated call signature
for the COMPOSE function:

```
from madng_tpsa.madng_definitions import COMPOSE
import inspect
print(inspect.signature(COMPOSE.fast_call))
print(COMPOSE.call_source)
```
"""
# ruff: noqa: E501

if __package__:
    from .madng_function import MadngFunction, NativeSize
else:
    from madng_function import MadngFunction, NativeSize  # ty: ignore[unresolved-import]

# Logical dimensions are always positive; native_sizes changes only C arguments.
# fmt: off
MNRM = MadngFunction('num_t mad_tpsa_mnrm(ssz_t na, const tpsa_t *ma[]);')
MORD = MadngFunction('ord_t mad_tpsa_mord(ssz_t na, const tpsa_t *ma[], log_t hi);')
FLD2VEC = MadngFunction('void mad_tpsa_fld2vec(ssz_t na, const tpsa_t *ma[], tpsa_t *c);')
VEC2FLD = MadngFunction('void mad_tpsa_vec2fld(ssz_t na, const tpsa_t *a, tpsa_t *mc[]);')
MINV = MadngFunction(
    'void mad_tpsa_minv(ssz_t na, const tpsa_t *ma[], ssz_t nb, tpsa_t *mc[]);',
    size_aliases={'nb': 'na'},
    parameter_arrays=('ma', 'mc'),
    native_sizes={'na': NativeSize(params_from='ma')},
    return_like='ma',
)
PMINV = MadngFunction(
    # `idx_t *select` is ABI-equivalent to `select[]`, but leaves the
    # parameter-inclusive selection buffer outside logical size inference.
    'void mad_tpsa_pminv(ssz_t na, const tpsa_t *ma[], ssz_t nb, tpsa_t *mc[], idx_t *select);',
    directions={'select': 'in'},
    size_aliases={'nb': 'na'},
    native_sizes={'na': NativeSize(params_from='ma')},
    parameter_arrays=('ma', 'mc'),
    return_like='ma',
)
TRANSLATE = MadngFunction(
    'void mad_tpsa_translate(ssz_t na, const tpsa_t *ma[], ssz_t nb, const num_t tb[], tpsa_t *mc[]);',
    complex_declaration='void mad_ctpsa_translate(ssz_t na, const ctpsa_t *ma[], ssz_t nb, const cpx_t tb[], ctpsa_t *mc[]);',
    array_sizes={'mc': 'na'},
    return_like='ma',
)
LIEBRA = MadngFunction(
    'void mad_tpsa_liebra(ssz_t na, const tpsa_t *ma[], const tpsa_t *mb[], tpsa_t *mc[]);',
    return_like='ma',
)
FGRAD = MadngFunction(
    'void mad_tpsa_fgrad(ssz_t na, const tpsa_t *ma[], const tpsa_t *b, tpsa_t *c);',
    extend_left=True,
    extend_right=False,
)
COMPOSE = MadngFunction(
    'void mad_tpsa_compose(ssz_t na, const tpsa_t *ma[], ssz_t nb, const tpsa_t *mb[], tpsa_t *mc[]);',
    array_sizes={'mc': 'na'},
    native_sizes={'na': NativeSize(sign=-1), 'nb': NativeSize(params_from='mb')},
    parameter_arrays=('mb',),
    extend_left=False,
    extend_right=True,
    reorder_variables=True,
    return_like='ma',
)
EVAL = MadngFunction(
    'void mad_tpsa_eval(ssz_t na, const tpsa_t *ma[], ssz_t nb, const num_t tb[], num_t tc[]);',
    complex_declaration='void mad_ctpsa_eval(ssz_t na, const ctpsa_t *ma[], ssz_t nb, const cpx_t tb[], cpx_t tc[]);',
    array_sizes={'tc': 'na'},
)
EXPPB = MadngFunction(
    'void mad_tpsa_exppb(ssz_t na, const tpsa_t *ma[], ssz_t nb, const tpsa_t *mb[], tpsa_t *mc[]);',
    array_sizes={'mc': 'nb'},
    return_like='mb',
)
LOGPB = MadngFunction(
    'void mad_tpsa_logpb(ssz_t na, const tpsa_t *ma[], const tpsa_t *mb[], tpsa_t *mc[]);',
)
# fmt: on

# Each declaration is registered automatically; no second list of names.
FUNCTIONS = tuple(value for value in globals().values() if isinstance(value, MadngFunction))
