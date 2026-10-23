"""Survey of descriptor, TPSA tensor, and map APIs not covered in 01-05.

Each operation uses a small real or complex series and avoids relying on
transient CFFI pointers beyond the lifetime of their owning TPSAs.
"""

import numpy as np

from madng_tpsa import ComplexTpsa, Descriptor, TpsaMap, TpsaTensor

d = Descriptor(variables=('q', 'p'), parameters=('k',), order=5, param_order=2)
q, p = d.vars()
k = d.param('k')

# Descriptor construction, validation, and serialization.
assert d.num_vars == 2 and d.num_params == 1
assert d.monomial_length == 3
assert d.is_valid_monomial((1, 0, 1))
assert d.monomial_index((1, 0, 0)) >= 0
assert Descriptor.from_dict(d.to_dict()) is d
assert d.union(d) is d
print('Variables:', d.variables, 'parameters:', d.parameters)
print('Variable index:', d.var_or_param_index('q'))
print('Parameter series:', d.params())
print('Descriptor native pointer:', d.ptr)
print('Descriptor pointer lookup:', Descriptor.from_ptr(d.ptr, d.variables, d.parameters) is d)
print('Descriptor order limits:', d.order, d.param_order, d.max_orders)
print('Descriptor formatting:', d.format_polynomial(q))
print('Constant TPSA:', d.constant(2))
print('Zero scalar factories:', d.zero(), d.complex_zero())
print('Named variable factory:', d.var('q'))
print('Factory polynomial:', d.polynomial({(1, 0, 0): 2.0}))

# Scalar TPSA inspection, order manipulations and numerical helpers.
f = 2 + 0.1 * q + 0.05 * p
print('Scalar native pointer:', f.ptr)
print('Scalar pointer lookup:', type(f).from_ptr(f.ptr, descriptor=d) is f)
print('Scalar copy and equality:', f.equals(f.copy()))
print('Scalar format:', f.format('code'))
print('Scalar coefficient:', f.get((1, 0, 0)))
print('Nonzero coefficients:', f.monomial_coeffs())
print('Degree and allocated order:', f.degree, f.truncation_order)
print('Norm:', f.norm(), 'unit:', f.unit())
print('First-order variable/parameter coefficients:', f.grad(), f.param_grad())
print('Derivative and antiderivative:', f.derivative('q'), f.integrate('q'))
print('Order transformations:', f.homogeneous(1), f.truncate(2), f.clear_order(1))
print('Zero and constant?', f.is_zero(), f.is_constant())
print('Scalar JSON round-trip:', type(f).from_dict(f.to_dict()))
modified_f = f.copy()
modified_f.set((0, 2, 0), 3.0)
modified_f.const_part = 1.5
print('Modified copy:', modified_f)
modified_f.clear()
print('Cleared copy:', modified_f.is_zero())
print('Scalar absolute value:', abs(f))
print('Converted scalar:', f.change_descriptor(d))

# Supported mathematical functions. Choose a small argument with constant
# part in the domains of real log/sqrt/asin/acos/atanh.
small = d.constant(0.25) + 0.01 * q
for function_name in (
    'sqrt',
    'exp',
    'log',
    'sin',
    'cos',
    'tan',
    'sinc',
    'asin',
    'acos',
    'atan',
    'sinh',
    'cosh',
    'tanh',
    'sinhc',
    'asinh',
    'acosh',
    'atanh',
    'erf',
    'erfc',
    'erfcx',
    'erfi',
):
    # acosh is real only for arguments >= 1.
    argument = small + 2 if function_name == 'acosh' else small
    result = getattr(argument, function_name)()
    print(f'{function_name}:', result.const_part)
print('Binary special functions:', small.atan2(f), small.hypot(f), small.hypot3(f, f))
print('Trigonometric pair:', small.sincos())
print('Additional sine/cosine pairs:', small.sincosq(), small.sincosmq())
print('Hyperbolic pair:', small.sincosh())
print('Additional hyperbolic pairs:', small.sincoshq(), small.sincoshmq())
print('Complex error function from real series:', small.wofz())
print('Real-valued wofz variant:', small.wofz_real())
print('Poisson bracket:', q.poisson_bracket(p))

# Complex-specific operations and real/imaginary projections.
z = ComplexTpsa.from_tpsa(q, p)
print('Complex projections:', z.real(), z.imag(), z.conjugate())
print('Complex series norm:', z.norm())
print('Complex pointer lookup:', ComplexTpsa.from_ptr(z.ptr, descriptor=d) is z)
print('Complex equality:', z.equals(z.copy()))
print('Complex Poisson bracket:', z.poisson_bracket(z))
complex_small = z * 0.01 + (0.25 + 0.10j)
for name in (
    'sqrt',
    'exp',
    'log',
    'sin',
    'cos',
    'tan',
    'sinh',
    'cosh',
    'tanh',
    'asin',
    'acos',
    'atan',
    'asinh',
    'acosh',
    'atanh',
    'erf',
    'erfc',
    'erfcx',
    'erfi',
    'wofz',
):
    print('Complex', name, getattr(complex_small, name)().const_part)
print('Complex JSON:', ComplexTpsa.from_dict(z.to_dict()))

# Rank-two tensors, views and structural operations.
tensor = TpsaTensor([[q + k, p], [q * p, k + p**2]])
row = tensor[0]
print('Tensor rank/size/shape:', tensor.rank, tensor.size, tensor.shape)
print('View shape and independent copy:', row.shape, row.copy().shape)
print('Tensor truncation orders:', tensor.truncation_orders)
print('Tensor native pointer array:', len(tensor.pointer_array))
print(
    'Tensor metadata:',
    tensor.num_vars,
    tensor.num_params,
    tensor.degree,
    tensor.truncation_order,
    len(tensor.flat),
)
print(
    'Tensor order operations:',
    tensor.homogeneous(1).shape,
    tensor.truncate(2).shape,
    tensor.clear_order(1).shape,
)
modifiable_tensor = tensor.copy()
modifiable_tensor[0, 0] = q
modifiable_tensor[1] = TpsaTensor([p, q])
modifiable_tensor.const_part = 2.0
print('Tensor view assignment:', modifiable_tensor.const_part)
print('Tensor complex promotion:', tensor.to_complex().is_complex)
print('Tensor serialization:', TpsaTensor.from_dict(tensor.to_dict()).shape)
print('Tensor descriptor conversion:', tensor.change_descriptor(d).shape)
print('Tensor evaluation:', tensor.evaluate((1, 2), parameters=(3,)))
print('Tensor derivative:', tensor.partial_derivative(('q', 'k')).shape)
print('Tensor Hessian shape:', tensor.derivative_tensor(2).shape)
print('Tensor composition:', tensor.compose(TpsaMap(d)).shape)
print('Tensor arithmetic:', (tensor + tensor).shape, (tensor * 2).shape)

# Map construction in named/positional forms and parameter sensitivity.
identity = TpsaMap(d)
map_ = TpsaMap(p=p + k, q=q + p**2)
print('Named map order:', map_.component_names)
print('Map named coordinates:', map_.named_components, map_.components)
print('Map pointer array with parameter identities:', len(map_.pointer_array_with_params))
print('Map coefficient:', map_.coefficient('q', (0, 2, 0)))
print('Map constants:', map_.const_part)
print('Jacobian:', map_.evaluate_jacobian())
print('Hessian shape:', map_.hessian(wrt='all').shape)
print('Map arithmetic/norm:', (map_ - identity).shape, map_.norm())
print(
    'Map copy, JSON, complex:',
    map_.copy().shape,
    TpsaMap.from_dict(map_.to_dict()).shape,
    map_.to_complex().is_complex,
)
print('Map descriptor conversion:', map_.change_descriptor(d).shape)
print('Map translation:', map_.translate((1, 0)).evaluate((2, 3), parameters=(4,)))
print('Map pullback:', map_.pullback(q**2).get((0, 4, 0)))
print('Map composition:', (map_ @ identity))

# Map extension adds a new identity output in descriptor order.
d_extended = Descriptor(variables=('q', 'p', 'r'), parameters=('k',), order=5)
assert map_.extend_descriptor(d_extended).shape == (3,)

# First-order setters
modifiable = identity.copy()
modifiable.set_coefficient('q', (2, 0, 0), 0.5)
modifiable.const_part = [0.25, 0.0]
print('Mutated map:', modifiable)

# Hamiltonian conversion and Lie conventions.
generator = k * q**3 / 3
field = TpsaMap.from_generator(generator)
recovered = field.to_generator()
lie_map = identity.exp_poisson(generator)
np.testing.assert_allclose(
    lie_map.evaluate((0.1, 0.2), parameters=(0.5,)),
    identity.exp_poisson(-field).evaluate((0.1, 0.2), parameters=(0.5,)),
)
print('Generator round-trip:', recovered)
print('Vector-field Lie bracket:', field.lie_bracket(field))
print('Logarithmic field and scalar:', lie_map.log_poisson(), lie_map.log_generator())
invertible = TpsaMap([q + p**2, p])
invertible.const_part = 0
print('Formal inverse:', invertible.inverse())
print('Partial inverse:', invertible.partial_inverse((True, False)))
