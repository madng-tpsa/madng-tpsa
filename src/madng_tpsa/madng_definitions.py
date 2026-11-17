"""MAD-NG declarations and Python call policies (also imported at build time)."""
# ruff: noqa: E501

if __package__:
    from .madng_function import MadngFunction, NativeSize
else:
    from madng_function import MadngFunction, NativeSize  # ty: ignore[unresolved-import]

# Logical dimensions are always positive; native_sizes changes only C arguments.
# fmt: off
MINV = MadngFunction(
    'void mad_tpsa_minv(ssz_t na, const tpsa_t *ma[], ssz_t nb, tpsa_t *mc[]);',
    size_aliases={'nb': 'na'},
    parameter_arrays=('ma', 'mc'),
    native_sizes={'na': NativeSize(params_from='ma')},
    return_like='ma',
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
LIEBRA = MadngFunction(
    'void mad_tpsa_liebra(ssz_t na, const tpsa_t *ma[], const tpsa_t *mb[], tpsa_t *mc[]);',
    return_like='ma',
)
VEC2FLD = MadngFunction('void mad_tpsa_vec2fld(ssz_t na, const tpsa_t *a, tpsa_t *mc[]);')
# fmt: on

# Each declaration is registered automatically; no second list of names.
FUNCTIONS = tuple(value for value in globals().values() if isinstance(value, MadngFunction))
