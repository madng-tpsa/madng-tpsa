# GTPSA MAD-NG API notes

This folder is automatically populated by the build process (`pip install -e path/to/madng_tpsa`)
and contains the patched GTPSA public headers from MAD-NG.

These notes collect practical information about the subset of the MAD-NG GTPSA API
that madng_tpsa uses and exposes.

## Descriptor

A `desc_t` descriptor defines the algebraic space in which TPSA objects live. It is
not itself a polynomial. Instead, it describes which monomials can exist and how
they are indexed.

At the public C API level, descriptors are opaque:

```c
typedef struct desc_ desc_t;
```

A descriptor is created with functions such as:

```c
const desc_t* mad_desc_newv(int nv, ord_t mo);
const desc_t* mad_desc_newvp(int nv, ord_t mo, int np_, ord_t po_);
const desc_t* mad_desc_newvpo(int nv, ord_t mo, int np_, ord_t po_, const ord_t no_[]);
```

Conceptually, a descriptor holds:

- `nv`: number of ordinary variables.
- `np`: number of parameters.
- $nn = nv + np$: total monomial length.
- `mo`: maximum polynomial order.
- `po`: maximum combined parameter order.
- `no_`: optional per-variable/per-parameter maximum orders.
- monomial tables and lookup structures used to map monomials to coefficient
  indexes.

Parameters are appended after ordinary variables in monomial tuples. For example,
a descriptor with `nv = 2` and `np = 1` uses monomials of length 3:

$$(x_\mathrm{order}, y_\mathrm{order}, p_\mathrm{order})$$

The descriptor owns the indexing rules. Functions such as `mad_desc_idxm`,
`mad_desc_mono`, and `mad_desc_isvalidm` convert between monomial descriptions
and internal coefficient indexes, or check whether a monomial is representable in
the descriptor.

The descriptor constructors differ only in how much structure they specify:

| Function | Description |
| --- | --- |
| `const desc_t* mad_desc_newv(int nv, ord_t mo)` | Create a descriptor with `nv` variables and maximum total order `mo`. All variables may appear up to order `mo`. |
| `const desc_t* mad_desc_newvp(int nv, ord_t mo, int np_, ord_t po_)` | Create a descriptor with `nv` variables, `np_` parameters, maximum total order `mo`, and maximum combined parameter order `po_`. Parameters are appended after variables. |
| `const desc_t* mad_desc_newvpo(int nv, ord_t mo, int np_, ord_t po_, const ord_t no_[])` | Create a descriptor with explicit per-variable/per-parameter maximum orders. The `no_` array has length `nv + np_`; variable limits come first, then parameter limits. MAD-NG raises `mo` to at least the maximum entry in `no_`, and raises `po_` to at least the maximum parameter entry. |

MAD-NG reuses equivalent descriptors internally. A TPSA object keeps a pointer to
its descriptor, and TPSA objects should only be combined when their descriptors
are compatible.

## TPSA

A `tpsa_t` is one truncated power series in the algebraic space described by a
`desc_t`.

At the public C API level, TPSA objects are also opaque:

```c
typedef struct tpsa_ tpsa_t;
```

A TPSA is created on a descriptor:

```c
tpsa_t* mad_tpsa_newd(const desc_t *d, ord_t mo);
```

The `mo` argument is the maximum order for this particular series. Passing
`mad_tpsa_dflt` asks MAD-NG to use the descriptor's maximum order.

Conceptually, a TPSA holds:

- a pointer to its descriptor;
- its own active maximum order for which memory has been allocated`ao`;
- its current trunctation order `mo` (with `mo` $\leq$`ao`)
- the range of orders that currently contain non-zero terms `lo` -> `hi`;
- optional metadata such as a user id and name;
- the coefficient array.

Coefficient index `0` is the scalar, or constant, coefficient. It is not a
variable. Higher coefficient indexes correspond to monomials defined by the
descriptor tables.

The main access patterns are:

- `mad_tpsa_setvar`: initialize an identity variable series.
- `mad_tpsa_setprm`: initialize an identity parameter series.
- `mad_tpsa_geti` / `mad_tpsa_seti`: get or update by internal coefficient index.
- `mad_tpsa_getm` / `mad_tpsa_setm`: get or update by monomial tuple.
- `mad_tpsa_cycle`: iterate over stored coefficients.

The setters `mad_tpsa_seti` and `mad_tpsa_setm` use the update rule:

$$\mathrm{new\_value} = a\,\mathrm{old\_value} + b$$

For example, setting the constant coefficient to `5` uses index `0` with
$a = 0$ and $b = 5$.

## TPSA construction, shape, and lifetime

| Function | Description |
| --- | --- |
| `tpsa_t* mad_tpsa_newd(const desc_t *d, ord_t mo)` | Allocate a TPSA on descriptor `d`. If `mo` is `mad_tpsa_dflt`, MAD-NG uses the descriptor's maximum order. |
| `tpsa_t* mad_tpsa_new(const tpsa_t *t, ord_t mo)` | Allocate a TPSA compatible with an existing TPSA. |
| `void mad_tpsa_del(const tpsa_t *t)` | Free a TPSA. |
| `const desc_t* mad_tpsa_desc(const tpsa_t *t)` | Return the descriptor that defines the TPSA's algebraic space. |
| `ord_t mad_tpsa_ord(const tpsa_t *t, log_t hi_)` | Return the active maximum order, or the highest currently non-zero order when `hi_` is true. |
| `ord_t mad_tpsa_mo(tpsa_t *t, ord_t mo)` | Change the truncation order to `mo` (which has to be smaller than `t->ao`). Terms of higher degree will be truncated.|
| `log_t mad_tpsa_isnul(const tpsa_t *t)` | Test whether the series is zero. |
| `log_t mad_tpsa_isval(const tpsa_t *t)` | Test whether the series contains only a constant coefficient. |
| `void mad_tpsa_clear(tpsa_t *t)` | Clear all coefficients. madng_tpsa uses this before loading coefficients from a dictionary. |
| `void mad_tpsa_copy(const tpsa_t *t, tpsa_t *r)` | Copy one TPSA into another compatible TPSA. |

## Coefficients and monomials

MAD-NG uses 1-based indices for variables and parameters in calls such as
`mad_tpsa_setvar`, `mad_tpsa_setprm`, `mad_tpsa_integ`, and `mad_tpsa_deriv`.
Monomial arrays are 0-based C arrays whose length is $nv + np$.

| Function | Description                                                                                                                                                            |
| --- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `void mad_tpsa_setvar(tpsa_t *t, num_t v, idx_t iv, num_t scl_)` | Initialize `t` as identity variable `iv`, expanded around constant value `v`. The variable index is 1-based.                                                           |
| `void mad_tpsa_setprm(tpsa_t *t, num_t v, idx_t ip)` | Initialize `t` as identity parameter `ip`, expanded around value `v`. Parameters are appended after variables in monomial tuples.                                      |
| `void mad_tpsa_setval(tpsa_t *t, num_t v)` | Set `t` to the constant series `v`.                                                                                                                                    |
| `idx_t mad_tpsa_cycle(const tpsa_t *t, idx_t i, ssz_t n, ord_t m_[], num_t *v_)` | Iterate over stored coefficients. Start with `i = -1`; each call returns the next coefficient index, fills `m_` with the monomial orders, and fills `v_` with the coefficient. |
| `num_t mad_tpsa_geti(const tpsa_t *t, idx_t i)` | Return coefficient by internal coefficient index. Index `0` is the constant coefficient.                                                                               |
| `num_t mad_tpsa_getm(const tpsa_t *t, ssz_t n, const ord_t m[])` | Return coefficient by full monomial tuple.                                                                                                                             |
| `void mad_tpsa_seti(tpsa_t *t, idx_t i, num_t a, num_t b)` | Update coefficient index `i` with $\mathrm{new\_value} = a\cdot\mathrm{old\_value} + b$.                                                                               |
| `void mad_tpsa_setm(tpsa_t *t, ssz_t n, const ord_t m[], num_t a, num_t b)` | Update a coefficient selected by full monomial tuple with $\mathrm{new\_value} = a\cdot\mathrm{old\_value} + b$.                                                       |
| `idx_t mad_tpsa_idxm(const tpsa_t *t, ssz_t n, const ord_t m[])` | Convert a valid monomial tuple to MAD-NG's internal coefficient index. Returns `-1` if invalid.                                                                        |
| `ord_t mad_tpsa_mono(const tpsa_t *t, idx_t i, ssz_t n, ord_t m_[], ord_t *p_)` | Convert an internal coefficient index back to monomial orders.                                                                                                         |

## Arithmetic operators

All binary TPSA operations require compatible descriptors. MAD-NG normally enforces
this with `ensure(...)`, which terminates the process; madng_tpsa checks compatibility
first at the Python layer where practical.

| Function | Description |
| --- | --- |
| `log_t mad_tpsa_equ(const tpsa_t *a, const tpsa_t *b, num_t tol_)` | Compare two compatible TPSAs coefficient-by-coefficient, with tolerance `tol_`. |
| `void mad_tpsa_add(const tpsa_t *a, const tpsa_t *b, tpsa_t *c)` | Compute $c = a + b$. |
| `void mad_tpsa_sub(const tpsa_t *a, const tpsa_t *b, tpsa_t *c)` | Compute $c = a - b$. |
| `void mad_tpsa_mul(const tpsa_t *a, const tpsa_t *b, tpsa_t *c)` | Compute $c = ab$, truncated to `c`'s order. |
| `void mad_tpsa_div(const tpsa_t *a, const tpsa_t *b, tpsa_t *c)` | Compute $c = a / b$. |
| `void mad_tpsa_pow(const tpsa_t *a, const tpsa_t *b, tpsa_t *c)` | Compute TPSA exponentiation $c = a^b$. |
| `void mad_tpsa_powi(const tpsa_t *a, int n, tpsa_t *c)` | Compute integer power $c = a^n$. |
| `void mad_tpsa_pown(const tpsa_t *a, num_t v, tpsa_t *c)` | Compute scalar real power $c = a^v$. |
| `void mad_tpsa_axpb(num_t a, const tpsa_t *x, num_t b, tpsa_t *r)` | Compute the affine scalar/TPSA expression $r = ax + b$. madng_tpsa uses this for scalar addition and subtraction. |

## Scalar helper operations

These are scalar/TPSA operations in the C API. In Python they are implementation
details of arithmetic operators, not currently separate public methods.

| Function | Description |
| --- | --- |
| `void mad_tpsa_acc(const tpsa_t *a, num_t v, tpsa_t *c)` | Accumulate $c \leftarrow c + va$. Aliasing is supported. |
| `void mad_tpsa_scl(const tpsa_t *a, num_t v, tpsa_t *c)` | Compute $c = va$. |
| `void mad_tpsa_divn(const tpsa_t *a, num_t v, tpsa_t *c)` | Compute $c = a/v$. |
| `void mad_tpsa_inv(const tpsa_t *a, num_t v, tpsa_t *c)` | Compute $c = v/a$. madng_tpsa uses this for reflected scalar division. |
| `void mad_tpsa_invsqrt(const tpsa_t *a, num_t v, tpsa_t *c)` | Compute $c = v/\sqrt{a}$. |

## Elementary math functions

Most unary functions evaluate a scalar function on a TPSA by expanding around the
constant part and propagating the truncated series terms.

| Function | Description |
| --- | --- |
| `num_t mad_tpsa_nrm(const tpsa_t *a)` | Return the sum of absolute values of stored coefficients. This is a coefficient norm, not a Euclidean norm. |
| `void mad_tpsa_unit(const tpsa_t *a, tpsa_t *c)` | Normalize by the magnitude/sign of the constant coefficient. madng_tpsa rejects zero constant part before calling this. |
| `void mad_tpsa_abs(const tpsa_t *a, tpsa_t *c)` | Absolute value branch determined by the constant coefficient. |
| `void mad_tpsa_sqrt(const tpsa_t *a, tpsa_t *c)` | Square root of a TPSA. |
| `void mad_tpsa_exp(const tpsa_t *a, tpsa_t *c)` | Exponential. |
| `void mad_tpsa_log(const tpsa_t *a, tpsa_t *c)` | Natural logarithm. |
| `void mad_tpsa_sin(const tpsa_t *a, tpsa_t *c)` | Sine. |
| `void mad_tpsa_cos(const tpsa_t *a, tpsa_t *c)` | Cosine. |
| `void mad_tpsa_tan(const tpsa_t *a, tpsa_t *c)` | Tangent. |
| `void mad_tpsa_sinc(const tpsa_t *a, tpsa_t *c)` | Unnormalized sinc, $\sin(x)/x$, with MAD-NG's regularization at zero. This differs from NumPy's normalized $\mathrm{sinc}(x) = \sin(\pi x)/(\pi x)$. |
| `void mad_tpsa_sinh(const tpsa_t *a, tpsa_t *c)` | Hyperbolic sine. |
| `void mad_tpsa_cosh(const tpsa_t *a, tpsa_t *c)` | Hyperbolic cosine. |
| `void mad_tpsa_tanh(const tpsa_t *a, tpsa_t *c)` | Hyperbolic tangent. |
| `void mad_tpsa_sinhc(const tpsa_t *a, tpsa_t *c)` | Unnormalized $\sinh(x)/x$, with MAD-NG's regularization at zero. |
| `void mad_tpsa_asin(const tpsa_t *a, tpsa_t *c)` | Inverse sine. |
| `void mad_tpsa_acos(const tpsa_t *a, tpsa_t *c)` | Inverse cosine. |
| `void mad_tpsa_atan(const tpsa_t *a, tpsa_t *c)` | Inverse tangent. |
| `void mad_tpsa_asinh(const tpsa_t *a, tpsa_t *c)` | Inverse hyperbolic sine. |
| `void mad_tpsa_acosh(const tpsa_t *a, tpsa_t *c)` | Inverse hyperbolic cosine. |
| `void mad_tpsa_atanh(const tpsa_t *a, tpsa_t *c)` | Inverse hyperbolic tangent. |

## Paired trigonometric and hyperbolic functions

These functions compute two related outputs in one call. We checked the
implementation, not only the comments, for the `sincos*` conventions.

| Function | Description |
| --- | --- |
| `void mad_tpsa_sincos(const tpsa_t *a, tpsa_t *s, tpsa_t *c)` | Compute $(s, c) = (\sin(a), \cos(a))$. |
| `void mad_tpsa_sincosq(const tpsa_t *a, tpsa_t *s, tpsa_t *c)` | Compute $(s, c) = (\mathrm{sinc}(\sqrt{a}), \cos(\sqrt{a}))$. |
| `void mad_tpsa_sincosmq(const tpsa_t *a, tpsa_t *s, tpsa_t *c)` | Compute $(s, c) = ((\mathrm{sinc}(\sqrt{a}) - 1)/a, (\cos(\sqrt{a}) - 1)/a)$. |
| `void mad_tpsa_sincosh(const tpsa_t *a, tpsa_t *s, tpsa_t *c)` | Compute $(s, c) = (\sinh(a), \cosh(a))$. |
| `void mad_tpsa_sincoshq(const tpsa_t *a, tpsa_t *s, tpsa_t *c)` | Compute $(s, c) = (\mathrm{sinhc}(\sqrt{a}), \cosh(\sqrt{a}))$. |
| `void mad_tpsa_sincoshmq(const tpsa_t *a, tpsa_t *s, tpsa_t *c)` | Compute $(s, c) = ((\mathrm{sinhc}(\sqrt{a}) - 1)/a, (\cosh(\sqrt{a}) - 1)/a)$. |

## Error and Faddeeva functions

MAD-NG includes Faddeeva-based functions from `mad_erfw`. For real TPSAs, complex
intermediate results are projected back to a real TPSA.

| Function | Description |
| --- | --- |
| `void mad_tpsa_erf(const tpsa_t *a, tpsa_t *c)` | Error function. |
| `void mad_tpsa_erfc(const tpsa_t *a, tpsa_t *c)` | Complementary error function. |
| `void mad_tpsa_erfcx(const tpsa_t *a, tpsa_t *c)` | Scaled complementary error function, related to $\mathrm{wf}(ix)$. |
| `void mad_tpsa_erfi(const tpsa_t *a, tpsa_t *c)` | Imaginary error function, implemented through a complex intermediate. |
| `void mad_tpsa_wf(const tpsa_t *a, tpsa_t *c)` | Faddeeva function $w(z) = \exp(-z^2)\mathrm{erfc}(-iz)$. |

## Binary and ternary math functions

| Function | Description |
| --- | --- |
| `void mad_tpsa_atan2(const tpsa_t *y, const tpsa_t *x, tpsa_t *r)` | Compute $r = \mathrm{atan2}(y, x)$. madng_tpsa lifts scalar operands to constant TPSAs before calling this. |
| `void mad_tpsa_hypot(const tpsa_t *x, const tpsa_t *y, tpsa_t *r)` | Compute $r = \sqrt{x^2 + y^2}$. |
| `void mad_tpsa_hypot3(const tpsa_t *x, const tpsa_t *y, const tpsa_t *z, tpsa_t *r)` | Compute $r = \sqrt{x^2 + y^2 + z^2}$. |

## Differential algebra functions

These functions operate on the formal power series structure. Indices are 1-based
and may refer to ordinary variables or appended parameters unless otherwise noted.

| Function | Description |
| --- | --- |
| `void mad_tpsa_integ(const tpsa_t *a, tpsa_t *c, idx_t iv)` | Formal indefinite integral of `a` with respect to variable/parameter index `iv`. The integration constant is zero. |
| `void mad_tpsa_deriv(const tpsa_t *a, tpsa_t *c, idx_t iv)` | First partial derivative with respect to variable/parameter index `iv`. |
| `void mad_tpsa_derivm(const tpsa_t *a, tpsa_t *c, ssz_t n, const ord_t m[])` | Higher or mixed partial derivative. The monomial `m` gives derivative orders; for example $(2, 1)$ means $\partial^3/(\partial x^2\,\partial y)$ in a two-variable descriptor. |
| `void mad_tpsa_poisbra(const tpsa_t *a, const tpsa_t *b, tpsa_t *c, int nv)` | Poisson bracket $\llbracket a,b \rrbracket = \sum_i \partial a/\partial q_i\,\partial b/\partial p_i - \partial a/\partial p_i\,\partial b/\partial q_i$, using canonical variable pairs $(q_1,p_1), (q_2,p_2), \ldots$ stored as indices $(1,2), (3,4), \ldots$. Passing $nv = 0$ uses all available variable pairs; otherwise MAD-NG uses $nv/2$ pairs. madng_tpsa exposes this as `num_pairs`, validates $0 < 2\,\mathrm{num\_pairs} \le \mathrm{num\_vars}$, and passes $2\,\mathrm{num\_pairs}$ to C. |
| `void mad_tpsa_taylor(const tpsa_t *a, ssz_t n, const num_t coef[], tpsa_t *c)` | Evaluate $\sum_i \mathrm{coef}[i]\,(a - a_0)^i$, where $a_0$ is the constant part. Coefficients are already Taylor coefficients, so for a scalar function $f$ pass $f(a_0)$, $f'(a_0)$, $f''(a_0)/2!$, and so on. |
| `void mad_tpsa_taylor_h(const tpsa_t *a, ssz_t n, const num_t coef[], tpsa_t *c)` | Same mathematical result as `mad_tpsa_taylor`, evaluated with Horner's method. MAD-NG's implementation notes say this can be slower because multiplication is always full order. |

Vector field operations

| Function | Description |
| --- | --- |
| `void mad_tpsa_vec2fld(ssz_t na, const tpsa_t *a,tpsa_t *mc[])` | Change a scalar field `a` ($f$, a generator) into a vector field `c` ($F$). This follows the sign convemtion $F = -S \nabla f $.|
| `void mad_tpsa_fld2vec(ssz_t na, const tpsa_t *ma[], tpsa_t *c)` | Retrieve a scalar field `c` ($f$) from a vector field `a` ($F$): $f(x) = \int\limits_0^x S F(x';0) dx'$|
| `void mad_tpsa_fgrad(ssz_t na, const tpsa_t *ma[], const tpsa_t * b  , tpsa_t *c)` | Take the scalar product between a vector field `a` ($K$) and the derivative of the scalar `b` ($f$) and store in `c` ($g$): $g = K\cdot \nabla f$ |
| `void mad_tpsa_liebra(ssz_t na, const tpsa_t *ma[], const tpsa_t *mb[], tpsa_t *mc[])` | Take the component-wise Lie bracket between vector fields `a` ($F$) and `b` ($G$) and store in `c` ($K$): $K^i = F\cdot\nabla G^i - G\cdot\nabla F^i$ |
| `void mad_tpsa_exppb(ssz_t na, const tpsa_t *ma[], ssz_t nb, const tpsa_t *mb[], tpsa_t *mc[])` | Take the component-wise exponentiated Lie bracket between vector fields `a` ($F$) and `b` ($G$) and store in `c` ($K$): $K^i = e^{F^j\partial_j} G^i = G^i + F^j\partial_j(G^i) + \frac{1}{2} F^j\partial_j(F^k\partial_k(G^i)) + \cdots$ (the series is cut by convergence when the component norm reaches <1e-9 or after 100 terms).
| `void mad_tpsa_logpb(ssz_t na, const tpsa_t *ma[], const tpsa_t *mb[], tpsa_t *mc[])` | exp(log(:F:))K |

Map operations:

| Function | Description |
| --- | --- |
| `ord_t mad_tpsa_mord(ssz_t na, const tpsa_t *ma[], log_t hi)` | Maximum order of the map |
| `num_t mad_tpsa_mnrm(ssz_t na, const tpsa_t *ma[])` | Map norm: sum over all components of the absolute values of all monomial coefficients |
| `void mad_tpsa_minv(ssz_t na, const tpsa_t *ma[], ssz_t nb, tpsa_t *mc[])` | Inverse of the map |
| `void mad_tpsa_pminv(ssz_t na, const tpsa_t *ma[], ssz_t nb, tpsa_t *mc[], idx_t select[])` | Partial inverse of the map |
| `void mad_tpsa_compose(ssz_t na, const tpsa_t *ma[], ssz_t nb, const tpsa_t *mb[], tpsa_t *mc[])` | Compose `a` with `b`: $a \circ b$ |
| `void mad_tpsa_translate(ssz_t na, const tpsa_t *ma[], ssz_t nb, const num_t tb[], tpsa_t *mc[])` | Translate `a` by a 6D point `b` |
| `void mad_tpsa_eval(ssz_t na, const tpsa_t *ma[], ssz_t nb, const num_t tb[], num_t   tc[])` | Evaluate `a` in the point `b` (joined list of coordinates and parameters) |
| `void mad_tpsa_mconv(ssz_t na, const tpsa_t *ma[], ssz_t nc, tpsa_t *mc[], ssz_t n, idx_t t2r_[], int pb)` | Convert maps to another maps using tpsa conversion |

## High-level combined expressions

These functions fuse common expressions into one C call. madng_tpsa currently uses
only a small subset directly. Careful, as there are conditions on the arguments or
the function fails (like `x->lo` $\leq$ `y->lo` in `mad_tpsa_axpbypc`).

| Function | Description |
| --- | --- |
| `void mad_tpsa_axpb(num_t a, const tpsa_t *x, num_t b, tpsa_t *r)` | Compute $r = ax + b$. |
| `void mad_tpsa_axpbypc(num_t a, const tpsa_t *x, num_t b, const tpsa_t *y, num_t c, tpsa_t *r)` | Compute $r = ax + by + c$. |
| `void mad_tpsa_axypb(num_t a, const tpsa_t *x, const tpsa_t *y, num_t b, tpsa_t *r)` | Compute $r = axy + b$. |
| `void mad_tpsa_axypbzpc(num_t a, const tpsa_t *x, const tpsa_t *y, num_t b, const tpsa_t *z, num_t c, tpsa_t *r)` | Compute $r = axy + bz + c$. |

## madng_tpsa extensions

The `madng_tpsa.h` and `madng_log.h` headers contain madng_tpsa-owned additions to the
MAD-NG API.
These helpers are compiled into the packaged `libmadng_tpsa` shared library, but
are not upstream MAD-NG functions.

| Function | Description |
| --- | --- |
| `int madng_tpsa_check_tpsa_compatibility(const tpsa_t *left, const tpsa_t *right)` | Return non-zero when two TPSA objects are compatible according to MAD-NG's descriptor compatibility rule. |
| `int madng_tpsa_tpsa_single_monomial(const tpsa_t *series, int monomial_length, ord_t monomial_orders[])` | Copy the monomial orders from a TPSA containing exactly one non-constant coefficient equal to `1`. Return its underlying coefficient index on success, or `-1` if the series has a constant part, no non-constant monomial, multiple non-constant monomials, a non-unit coefficient, or an incompatible output length. |

madng_tpsa uses these helpers to validate inputs in Python before calling MAD-NG
functions that would otherwise terminate through `ensure(...)`.

## Complex TPSA (CTPSA) C API

`mad_ctpsa.h` provides the complex counterpart to `mad_tpsa.h`. It is included by
`madng_tpsa.h`, so consumers normally need only:

```c
#include "madng_tpsa.h"
```

`ctpsa_t` is opaque and uses the same `desc_t` descriptors and monomial conventions
as `tpsa_t`. `cpx_t` is MAD-NG's complex scalar type. Function names ending in `_r`
avoid passing `cpx_t` by value: complex scalars are passed as adjacent real and
imaginary `num_t` arguments instead. This is useful for foreign-function interfaces.

Unless stated otherwise, input and output CTPSAs must have compatible descriptors;
results are written to the final CTPSA/TPSA argument. The tables below are a complete
index of the functions declared in `mad_ctpsa.h`.

### Construction, inspection, and mutation

| Function | Description |
| --- | --- |
| `ctpsa_t* mad_ctpsa_newd(const desc_t *d, ord_t mo)` | Allocate a CTPSA on a descriptor, with the requested maximum order. |
| `ctpsa_t* mad_ctpsa_new(const ctpsa_t *t, ord_t mo)` | Allocate a CTPSA compatible with another CTPSA (or a cast `tpsa_t`). |
| `void mad_ctpsa_del(const ctpsa_t *t)` | Free a CTPSA. |
| `const desc_t* mad_ctpsa_desc(const ctpsa_t *t)` | Return the CTPSA descriptor. |
| `ord_t mad_ctpsa_mo(ctpsa_t *t, ord_t mo)` | Get/set the active maximum order. |
| `int32_t mad_ctpsa_uid(ctpsa_t *t, int32_t uid_)` | Get/set the user id; a non-zero optional argument sets it. |
| `str_t mad_ctpsa_nam(ctpsa_t *t, str_t nam_)` | Get/set the name; a non-null optional argument sets it. |
| `ssz_t mad_ctpsa_len(const ctpsa_t *t, log_t hi_)` | Return active length, or the length through the highest non-zero order. |
| `ord_t mad_ctpsa_ord(const ctpsa_t *t, log_t hi_)` | Return active order, or the highest non-zero order. |
| `log_t mad_ctpsa_isnul(const ctpsa_t *t)` | Test whether all coefficients are zero. |
| `log_t mad_ctpsa_isval(const ctpsa_t *t)` | Test whether the series is constant. |
| `log_t mad_ctpsa_isvalid(const ctpsa_t *t)` | Test internal validity. |
| `num_t mad_ctpsa_density(const ctpsa_t *t, num_t stat_[2], log_t reset)` | Return coefficient density; optionally collect/reset density statistics. |
| `void mad_ctpsa_copy(const ctpsa_t *t, ctpsa_t *r)` | Copy a CTPSA. |
| `void mad_ctpsa_convert(const ctpsa_t *t, ctpsa_t *r, ssz_t n, idx_t t2r_[], int pb)` | Convert/reindex a CTPSA with an index map. |
| `idx_t mad_ctpsa_maxord(const ctpsa_t *t, ssz_t n, idx_t idx_[])` | Find coefficient indexes at the maximum populated order. |
| `void mad_ctpsa_sclord(const ctpsa_t *t, ctpsa_t *r, log_t inv, log_t prm)` | Scale by monomial order, optionally inversely or for parameters. |
| `void mad_ctpsa_getord(const ctpsa_t *t, ctpsa_t *r, ord_t ord)` | Extract one homogeneous order. |
| `void mad_ctpsa_cutord(const ctpsa_t *t, ctpsa_t *r, int ord)` | Clear orders from `ord` upward, or through `-ord` for a negative order. |
| `void mad_ctpsa_clrord(ctpsa_t *t, ord_t ord)` | Clear one homogeneous order in place. |
| `void mad_ctpsa_setvar(ctpsa_t *t, cpx_t v, idx_t iv, cpx_t scl_)` | Initialize an identity variable about a complex value. |
| `void mad_ctpsa_setprm(ctpsa_t *t, cpx_t v, idx_t ip)` | Initialize an identity parameter about a complex value. |
| `void mad_ctpsa_setval(ctpsa_t *t, cpx_t v)` | Set a constant complex series. |
| `void mad_ctpsa_update(ctpsa_t *t)` | Recompute cached range/order metadata after direct changes. |
| `void mad_ctpsa_clear(ctpsa_t *t)` | Clear all coefficients. |
| `void mad_ctpsa_setvar_r(ctpsa_t *t, num_t v_re, num_t v_im, idx_t iv, num_t scl_re_, num_t scl_im_)` | `_r` form of `mad_ctpsa_setvar`. |
| `void mad_ctpsa_setprm_r(ctpsa_t *t, num_t v_re, num_t v_im, idx_t ip)` | `_r` form of `mad_ctpsa_setprm`. |
| `void mad_ctpsa_setval_r(ctpsa_t *t, num_t v_re, num_t v_im)` | `_r` form of `mad_ctpsa_setval`. |

### Real/complex conversion and coefficient access

| Function | Description |
| --- | --- |
| `void mad_ctpsa_cplx(const tpsa_t *re_, const tpsa_t *im_, ctpsa_t *r)` | Form a CTPSA from optional real and imaginary TPSAs. |
| `void mad_ctpsa_real(const ctpsa_t *t, tpsa_t *r)` | Extract the real part to a TPSA. |
| `void mad_ctpsa_imag(const ctpsa_t *t, tpsa_t *r)` | Extract the imaginary part to a TPSA. |
| `void mad_ctpsa_cabs(const ctpsa_t *t, tpsa_t *r)` | Extract the complex magnitude to a TPSA. |
| `void mad_ctpsa_carg(const ctpsa_t *t, tpsa_t *r)` | Extract the complex phase to a TPSA. |
| `void mad_ctpsa_rect(const ctpsa_t *t, ctpsa_t *r)` | Convert polar representation to rectangular representation. |
| `void mad_ctpsa_polar(const ctpsa_t *t, ctpsa_t *r)` | Convert rectangular representation to polar representation. |
| `ord_t mad_ctpsa_mono(const ctpsa_t *t, idx_t i, ssz_t n, ord_t m_[], ord_t *p_)` | Convert a coefficient index to its monomial and order. |
| `idx_t mad_ctpsa_idxs(const ctpsa_t *t, ssz_t n, str_t s)` | Convert a digit-string monomial to an index. |
| `idx_t mad_ctpsa_idxm(const ctpsa_t *t, ssz_t n, const ord_t m[])` | Convert a dense monomial-order array to an index. |
| `idx_t mad_ctpsa_idxsm(const ctpsa_t *t, ssz_t n, const idx_t m[])` | Convert sparse `(index, order)` pairs to an index. |
| `idx_t mad_ctpsa_cycle(const ctpsa_t *t, idx_t i, ssz_t n, ord_t m_[], cpx_t *v_)` | Iterate stored coefficients, returning the next index and monomial. |
| `cpx_t mad_ctpsa_geti(const ctpsa_t *t, idx_t i)` | Get a coefficient by internal index. |
| `cpx_t mad_ctpsa_gets(const ctpsa_t *t, ssz_t n, str_t s)` | Get a coefficient by digit-string monomial. |
| `cpx_t mad_ctpsa_getm(const ctpsa_t *t, ssz_t n, const ord_t m[])` | Get a coefficient by dense monomial orders. |
| `cpx_t mad_ctpsa_getsm(const ctpsa_t *t, ssz_t n, const idx_t m[])` | Get a coefficient by sparse monomial orders. |
| `void mad_ctpsa_seti(ctpsa_t *t, idx_t i, cpx_t a, cpx_t b)` | Update an indexed coefficient as `a * old + b`. |
| `void mad_ctpsa_sets(ctpsa_t *t, ssz_t n, str_t s , cpx_t a, cpx_t b)` | Update a digit-string-selected coefficient as `a * old + b`. |
| `void mad_ctpsa_setm(ctpsa_t *t, ssz_t n, const ord_t m[], cpx_t a, cpx_t b)` | Update a dense-monomial-selected coefficient as `a * old + b`. |
| `void mad_ctpsa_setsm(ctpsa_t *t, ssz_t n, const idx_t m[], cpx_t a, cpx_t b)` | Update a sparse-monomial-selected coefficient as `a * old + b`. |
| `void mad_ctpsa_cpyi(const ctpsa_t *t, ctpsa_t *r, idx_t i)` | Copy one indexed coefficient to another CTPSA. |
| `void mad_ctpsa_cpys(const ctpsa_t *t, ctpsa_t *r, ssz_t n, str_t s)` | Copy one digit-string-selected coefficient. |
| `void mad_ctpsa_cpym(const ctpsa_t *t, ctpsa_t *r, ssz_t n, const ord_t m[])` | Copy one dense-monomial-selected coefficient. |
| `void mad_ctpsa_cpysm(const ctpsa_t *t, ctpsa_t *r, ssz_t n, const idx_t m[])` | Copy one sparse-monomial-selected coefficient. |
| `void mad_ctpsa_geti_r(const ctpsa_t *t, idx_t i, cpx_t *r)` | Store an indexed coefficient through a `cpx_t *` output. |
| `void mad_ctpsa_gets_r(const ctpsa_t *t, ssz_t n, str_t s , cpx_t *r)` | Store a digit-string-selected coefficient through a `cpx_t *` output. |
| `void mad_ctpsa_getm_r(const ctpsa_t *t, ssz_t n, const ord_t m[], cpx_t *r)` | Store a dense-monomial-selected coefficient through a `cpx_t *` output. |
| `void mad_ctpsa_getsm_r(const ctpsa_t *t, ssz_t n, const idx_t m[], cpx_t *r)` | Store a sparse-monomial-selected coefficient through a `cpx_t *` output. |
| `void mad_ctpsa_seti_r(ctpsa_t *t, idx_t i, num_t a_re, num_t a_im, num_t b_re, num_t b_im)` | `_r` form of `mad_ctpsa_seti`. |
| `void mad_ctpsa_sets_r(ctpsa_t *t, ssz_t n, str_t s , num_t a_re, num_t a_im, num_t b_re, num_t b_im)` | `_r` form of `mad_ctpsa_sets`. |
| `void mad_ctpsa_setm_r(ctpsa_t *t, ssz_t n, const ord_t m[], num_t a_re, num_t a_im, num_t b_re, num_t b_im)` | `_r` form of `mad_ctpsa_setm`. |
| `void mad_ctpsa_setsm_r(ctpsa_t *t, ssz_t n, const idx_t m[], num_t a_re, num_t a_im, num_t b_re, num_t b_im)` | `_r` form of `mad_ctpsa_setsm`. |
| `void mad_ctpsa_getv(const ctpsa_t *t, idx_t i, ssz_t n, cpx_t v[])` | Copy a consecutive coefficient range to a complex array. |
| `void mad_ctpsa_setv(ctpsa_t *t, idx_t i, ssz_t n, const cpx_t v[])` | Copy a complex array into a consecutive coefficient range. |

### Arithmetic and mixed real/complex arithmetic

| Function | Description |
| --- | --- |
| `log_t mad_ctpsa_equ(const ctpsa_t *a, const ctpsa_t *b, num_t tol_)` | Compare two CTPSAs with a tolerance. |
| `void mad_ctpsa_dif(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c)` | Compute $c_i = (a_i-b_i)/\max(|a_i|,1)$ coefficient-wise. |
| `void mad_ctpsa_add(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c)` | Compute $c = a + b$. |
| `void mad_ctpsa_sub(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c)` | Compute $c = a - b$. |
| `void mad_ctpsa_mul(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c)` | Compute $c = ab$, truncated to `c`'s order. |
| `void mad_ctpsa_div(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c)` | Compute $c = a / b$. |
| `log_t mad_ctpsa_divc(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c, num_t tol_)` | Compute $c = a / b$ by cancellation; return whether it succeeded. |
| `void mad_ctpsa_pow(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c)` | Compute CTPSA exponentiation $c = a^b$. |
| `void mad_ctpsa_powi(const ctpsa_t *a, int n, ctpsa_t *c)` | Compute integer power $c = a^n$. |
| `void mad_ctpsa_pown(const ctpsa_t *a, cpx_t v, ctpsa_t *c)` | Compute complex scalar power $c = a^v$. |
| `void mad_ctpsa_pown_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c)` | `_r` form of `mad_ctpsa_pown`. |
| `log_t mad_ctpsa_equt(const ctpsa_t *a, const tpsa_t *b, num_t tol)` | Compare a CTPSA with a TPSA after real-to-complex conversion. |
| `void mad_ctpsa_dift(const ctpsa_t *a, const tpsa_t *b, ctpsa_t *c)` | Compute $c_i = (a_i-b_i)/\max(|a_i|,1)$ after promoting `b`. |
| `void mad_ctpsa_tdif(const tpsa_t *a, const ctpsa_t *b, ctpsa_t *c)` | Compute $c_i = (a_i-b_i)/\max(|a_i|,1)$ after promoting `a`. |
| `void mad_ctpsa_addt(const ctpsa_t *a, const tpsa_t *b, ctpsa_t *c)` | Compute $c = a + b$ after promoting `b` to CTPSA. |
| `void mad_ctpsa_subt(const ctpsa_t *a, const tpsa_t *b, ctpsa_t *c)` | Compute $c = a - b$ after promoting `b` to CTPSA. |
| `void mad_ctpsa_tsub(const tpsa_t *a, const ctpsa_t *b, ctpsa_t *c)` | Compute $c = a - b$ after promoting `a` to CTPSA. |
| `void mad_ctpsa_mult(const ctpsa_t *a, const tpsa_t *b, ctpsa_t *c)` | Compute $c = ab$ after promoting `b` to CTPSA. |
| `void mad_ctpsa_divt(const ctpsa_t *a, const tpsa_t *b, ctpsa_t *c)` | Compute $c = a / b$ after promoting `b` to CTPSA. |
| `void mad_ctpsa_tdiv(const tpsa_t *a, const ctpsa_t *b, ctpsa_t *c)` | Compute $c = a / b$ after promoting `a` to CTPSA. |
| `void mad_ctpsa_powt(const ctpsa_t *a, const tpsa_t *b, ctpsa_t *c)` | Compute $c = a^b$ after promoting `b` to CTPSA. |
| `void mad_ctpsa_tpow(const tpsa_t *a, const ctpsa_t *b, ctpsa_t *c)` | Compute $c = a^b$ after promoting `a` to CTPSA. |

### Complex elementary and scalar functions

| Function | Description |
| --- | --- |
| `num_t mad_ctpsa_nrm(const ctpsa_t *a)` | Return the coefficient norm. |
| `void mad_ctpsa_unit(const ctpsa_t *a, ctpsa_t *c)` | Normalize by the complex constant part. |
| `void mad_ctpsa_conj(const ctpsa_t *a, ctpsa_t *c)` | Complex conjugate. |
| `void mad_ctpsa_sqrt(const ctpsa_t *a, ctpsa_t *c)` | Square root. |
| `void mad_ctpsa_exp(const ctpsa_t *a, ctpsa_t *c)` | Exponential. |
| `void mad_ctpsa_log(const ctpsa_t *a, ctpsa_t *c)` | Natural logarithm. |
| `void mad_ctpsa_sincos(const ctpsa_t *a, ctpsa_t *s, ctpsa_t *c)` | Compute sine and cosine together. |
| `void mad_ctpsa_sincosq(const ctpsa_t *a, ctpsa_t *s, ctpsa_t *c)` | Compute `sinc(sqrt(a))` and `cos(sqrt(a))`. |
| `void mad_ctpsa_sincosmq(const ctpsa_t *a, ctpsa_t *s, ctpsa_t *c)` | Compute the corresponding regularized quotient pair. |
| `void mad_ctpsa_sin(const ctpsa_t *a, ctpsa_t *c)` | Sine. |
| `void mad_ctpsa_cos(const ctpsa_t *a, ctpsa_t *c)` | Cosine. |
| `void mad_ctpsa_tan(const ctpsa_t *a, ctpsa_t *c)` | Tangent. |
| `void mad_ctpsa_cot(const ctpsa_t *a, ctpsa_t *c)` | Cotangent. |
| `void mad_ctpsa_sinc(const ctpsa_t *a, ctpsa_t *c)` | Unnormalized sinc. |
| `void mad_ctpsa_sincosh(const ctpsa_t *a, ctpsa_t *s, ctpsa_t *c)` | Compute hyperbolic sine and cosine together. |
| `void mad_ctpsa_sincoshq(const ctpsa_t *a, ctpsa_t *s, ctpsa_t *c)` | Compute `sinhc(sqrt(a))` and `cosh(sqrt(a))`. |
| `void mad_ctpsa_sincoshmq(const ctpsa_t*a, ctpsa_t *s, ctpsa_t *c)` | Compute the corresponding regularized quotient pair. |
| `void mad_ctpsa_sinh(const ctpsa_t *a, ctpsa_t *c)` | Hyperbolic sine. |
| `void mad_ctpsa_cosh(const ctpsa_t *a, ctpsa_t *c)` | Hyperbolic cosine. |
| `void mad_ctpsa_tanh(const ctpsa_t *a, ctpsa_t *c)` | Hyperbolic tangent. |
| `void mad_ctpsa_coth(const ctpsa_t *a, ctpsa_t *c)` | Hyperbolic cotangent. |
| `void mad_ctpsa_sinhc(const ctpsa_t *a, ctpsa_t *c)` | Unnormalized hyperbolic sinc. |
| `void mad_ctpsa_asin(const ctpsa_t *a, ctpsa_t *c)` | Inverse sine. |
| `void mad_ctpsa_acos(const ctpsa_t *a, ctpsa_t *c)` | Inverse cosine. |
| `void mad_ctpsa_atan(const ctpsa_t *a, ctpsa_t *c)` | Inverse tangent. |
| `void mad_ctpsa_acot(const ctpsa_t *a, ctpsa_t *c)` | Inverse cotangent. |
| `void mad_ctpsa_asinc(const ctpsa_t *a, ctpsa_t *c)` | Inverse unnormalized sinc. |
| `void mad_ctpsa_asinh(const ctpsa_t *a, ctpsa_t *c)` | Inverse hyperbolic sine. |
| `void mad_ctpsa_acosh(const ctpsa_t *a, ctpsa_t *c)` | Inverse hyperbolic cosine. |
| `void mad_ctpsa_atanh(const ctpsa_t *a, ctpsa_t *c)` | Inverse hyperbolic tangent. |
| `void mad_ctpsa_acoth(const ctpsa_t *a, ctpsa_t *c)` | Inverse hyperbolic cotangent. |
| `void mad_ctpsa_asinhc(const ctpsa_t *a, ctpsa_t *c)` | Inverse unnormalized hyperbolic sinc. |
| `void mad_ctpsa_erf(const ctpsa_t *a, ctpsa_t *c)` | Error function. |
| `void mad_ctpsa_erfc(const ctpsa_t *a, ctpsa_t *c)` | Complementary error function. |
| `void mad_ctpsa_erfcx(const ctpsa_t *a, ctpsa_t *c)` | Scaled complementary error function. |
| `void mad_ctpsa_erfi(const ctpsa_t *a, ctpsa_t *c)` | Imaginary error function. |
| `void mad_ctpsa_wf(const ctpsa_t *a, ctpsa_t *c)` | Faddeeva function. |
| `void mad_ctpsa_acc(const ctpsa_t *a, cpx_t v, ctpsa_t *c)` | Accumulate `c += v * a`. |
| `void mad_ctpsa_scl(const ctpsa_t *a, cpx_t v, ctpsa_t *c)` | Compute `c = v * a`. |
| `void mad_ctpsa_divn(const ctpsa_t *a, cpx_t v, ctpsa_t *c)` | Compute `c = a / v`. |
| `void mad_ctpsa_inv(const ctpsa_t *a, cpx_t v, ctpsa_t *c)` | Compute `c = v / a`. |
| `void mad_ctpsa_invsqrt(const ctpsa_t *a, cpx_t v, ctpsa_t *c)` | Compute `c = v / sqrt(a)`. |
| `void mad_ctpsa_hypot(const ctpsa_t *x, const ctpsa_t *y, ctpsa_t *r)` | Compute the two-argument complex hypotenuse. |
| `void mad_ctpsa_hypot3(const ctpsa_t *x, const ctpsa_t *y, const ctpsa_t *z, ctpsa_t *r)` | Compute the three-argument complex hypotenuse. |
| `void mad_ctpsa_acc_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c)` | `_r` form of `mad_ctpsa_acc`. |
| `void mad_ctpsa_scl_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c)` | `_r` form of `mad_ctpsa_scl`. |
| `void mad_ctpsa_divn_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c)` | `_r` form of `mad_ctpsa_divn`. |
| `void mad_ctpsa_inv_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c)` | `_r` form of `mad_ctpsa_inv`. |
| `void mad_ctpsa_invsqrt_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c)` | `_r` form of `mad_ctpsa_invsqrt`. |

### Differential algebra and composed expressions

| Function | Description |
| --- | --- |
| `void mad_ctpsa_integ(const ctpsa_t *a, ctpsa_t *c, idx_t iv)` | Integrate with respect to a 1-based variable/parameter index. |
| `void mad_ctpsa_deriv(const ctpsa_t *a, ctpsa_t *c, idx_t iv)` | Differentiate with respect to a 1-based variable/parameter index. |
| `void mad_ctpsa_derivm(const ctpsa_t *a, ctpsa_t *c, ssz_t n, const ord_t m[])` | Differentiate by a dense monomial-order array. |
| `void mad_ctpsa_poisbra(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c, int nv)` | Compute a Poisson bracket of two CTPSAs. |
| `void mad_ctpsa_taylor(const ctpsa_t *a, ssz_t n, const cpx_t coef[], ctpsa_t *c)` | Evaluate a Taylor expansion from complex coefficients. |
| `void mad_ctpsa_taylor_h(const ctpsa_t *a, ssz_t n, const cpx_t coef[], ctpsa_t *c)` | Evaluate that Taylor expansion with Horner's method. |
| `void mad_ctpsa_poisbrat(const ctpsa_t *a, const tpsa_t *b, ctpsa_t *c, int nv)` | Poisson bracket of a CTPSA and TPSA. |
| `void mad_ctpsa_tpoisbra(const tpsa_t *a, const ctpsa_t *b, ctpsa_t *c, int nv)` | Poisson bracket of a TPSA and CTPSA. |
| `void mad_ctpsa_axpb(cpx_t a, const ctpsa_t *x, cpx_t b, ctpsa_t *r)` | Compute `a*x + b`. |
| `void mad_ctpsa_axpbypc(cpx_t a, const ctpsa_t *x, cpx_t b, const ctpsa_t *y, cpx_t c, ctpsa_t *r)` | Compute `a*x + b*y + c`. |
| `void mad_ctpsa_axypb(cpx_t a, const ctpsa_t *x, const ctpsa_t *y, cpx_t b, ctpsa_t *r)` | Compute `a*x*y + b`. |
| `void mad_ctpsa_axypbzpc(cpx_t a, const ctpsa_t *x, const ctpsa_t *y, cpx_t b, const ctpsa_t *z, cpx_t c, ctpsa_t *r)` | Compute `a*x*y + b*z + c`. |
| `void mad_ctpsa_axypbvwpc(cpx_t a, const ctpsa_t *x, const ctpsa_t *y, cpx_t b, const ctpsa_t *v, const ctpsa_t *w, cpx_t c, ctpsa_t *r)` | Compute `a*x*y + b*v*w + c`. |
| `void mad_ctpsa_ax2pby2pcz2(cpx_t a, const ctpsa_t *x, cpx_t b, const ctpsa_t *y, cpx_t c, const ctpsa_t *z, ctpsa_t *r)` | Compute `a*x^2 + b*y^2 + c*z^2`. |
| `void mad_ctpsa_axpsqrtbpcx2(const ctpsa_t *x, cpx_t a, cpx_t b, cpx_t c, ctpsa_t *r)` | Compute `a*x + sqrt(b + c*x^2)`. |
| `void mad_ctpsa_logaxpsqrtbpcx2(const ctpsa_t *x, cpx_t a, cpx_t b, cpx_t c, ctpsa_t *r)` | Compute `log(a*x + sqrt(b + c*x^2))`. |
| `void mad_ctpsa_logxdy(const ctpsa_t *x, const ctpsa_t *y, ctpsa_t *r)` | Compute `log(x) / y`. |
| `void mad_ctpsa_axpb_r(num_t a_re, num_t a_im, const ctpsa_t *x, num_t b_re, num_t b_im, ctpsa_t *r)` | `_r` form of `mad_ctpsa_axpb`. |
| `void mad_ctpsa_axpbypc_r(num_t a_re, num_t a_im, const ctpsa_t *x, num_t b_re, num_t b_im, const ctpsa_t *y, num_t c_re, num_t c_im, ctpsa_t *r)` | `_r` form of `mad_ctpsa_axpbypc`. |
| `void mad_ctpsa_axypb_r(num_t a_re, num_t a_im, const ctpsa_t *x, const ctpsa_t *y, num_t b_re, num_t b_im, ctpsa_t *r)` | `_r` form of `mad_ctpsa_axypb`. |
| `void mad_ctpsa_axypbzpc_r(num_t a_re, num_t a_im, const ctpsa_t *x, const ctpsa_t *y, num_t b_re, num_t b_im, const ctpsa_t *z, num_t c_re, num_t c_im, ctpsa_t *r)` | `_r` form of `mad_ctpsa_axypbzpc`. |
| `void mad_ctpsa_axypbvwpc_r(num_t a_re, num_t a_im, const ctpsa_t *x, const ctpsa_t *y, num_t b_re, num_t b_im, const ctpsa_t *v, const ctpsa_t *w, num_t c_re, num_t c_im, ctpsa_t *r)` | `_r` form of `mad_ctpsa_axypbvwpc`. |
| `void mad_ctpsa_ax2pby2pcz2_r(num_t a_re, num_t a_im, const ctpsa_t *x, num_t b_re, num_t b_im, const ctpsa_t *y, num_t c_re, num_t c_im, const ctpsa_t *z, ctpsa_t *r)` | `_r` form of `mad_ctpsa_ax2pby2pcz2`. |
| `void mad_ctpsa_axpsqrtbpcx2_r(const ctpsa_t *x, num_t a_re, num_t a_im, num_t b_re, num_t b_im, num_t c_re, num_t c_im, ctpsa_t *r)` | `_r` form of `mad_ctpsa_axpsqrtbpcx2`. |
| `void mad_ctpsa_logaxpsqrtbpcx2_r(const ctpsa_t *x, num_t a_re, num_t a_im, num_t b_re, num_t b_im, num_t c_re, num_t c_im, ctpsa_t *r)` | `_r` form of `mad_ctpsa_logaxpsqrtbpcx2`. |

### Maps, I/O, and diagnostic functions

| Function | Description |
| --- | --- |
| `void mad_ctpsa_vec2fld(ssz_t na, const ctpsa_t *a , ctpsa_t *mc[])` | Convert a vector map to a field representation. |
| `void mad_ctpsa_fld2vec(ssz_t na, const ctpsa_t *ma[], ctpsa_t *c)` | Convert a field representation to a vector map. |
| `void mad_ctpsa_fgrad(ssz_t na, const ctpsa_t *ma[], const ctpsa_t * b , ctpsa_t *c)` | Apply a field gradient. |
| `void mad_ctpsa_liebra(ssz_t na, const ctpsa_t *ma[], const ctpsa_t *mb[], ctpsa_t *mc[])` | Compute a Lie bracket of maps. |
| `void mad_ctpsa_exppb(ssz_t na, const ctpsa_t *ma[], ssz_t nb, const ctpsa_t *mb[], ctpsa_t *mc[])` | Apply `exp(:F:)` to a map. |
| `void mad_ctpsa_logpb(ssz_t na, const ctpsa_t *ma[], const ctpsa_t *mb[], ctpsa_t *mc[])` | Compute the logarithmic Poisson-bracket map representation. |
| `ord_t mad_ctpsa_mord(ssz_t na, const ctpsa_t *ma[], log_t hi)` | Return the maximum map order. |
| `num_t mad_ctpsa_mnrm(ssz_t na, const ctpsa_t *ma[])` | Return a map coefficient norm. |
| `void mad_ctpsa_minv(ssz_t na, const ctpsa_t *ma[], ssz_t nb, ctpsa_t *mc[])` | Invert a map. |
| `void mad_ctpsa_pminv(ssz_t na, const ctpsa_t *ma[], ssz_t nb, ctpsa_t *mc[], idx_t select[])` | Partially invert a map using selected coordinates. |
| `void mad_ctpsa_compose(ssz_t na, const ctpsa_t *ma[], ssz_t nb, const ctpsa_t *mb[], ctpsa_t *mc[])` | Compose two maps. |
| `void mad_ctpsa_translate(ssz_t na, const ctpsa_t *ma[], ssz_t nb, const cpx_t tb[], ctpsa_t *mc[])` | Translate a map by complex values. |
| `void mad_ctpsa_eval(ssz_t na, const ctpsa_t *ma[], ssz_t nb, const cpx_t tb[], cpx_t tc[])` | Evaluate a map at complex values. |
| `void mad_ctpsa_mconv(ssz_t na, const ctpsa_t *ma[], ssz_t nc, ctpsa_t *mc[], ssz_t n, idx_t t2r_[], int pb)` | Convert/reindex every component of a map. |
| `void mad_ctpsa_print(const ctpsa_t *t, str_t name_, num_t eps_, int nohdr_, FILE *stream_)` | Print a CTPSA to a stream. |
| `ctpsa_t* mad_ctpsa_scan(FILE *stream_)` | Read a CTPSA from a stream. |
| `const desc_t* mad_ctpsa_scan_hdr(int *kind_, char name_[NAMSZ], FILE *stream_)` | Read a serialized CTPSA header from a stream. |
| `void mad_ctpsa_scan_coef(ctpsa_t *t, FILE *stream_)` | Read serialized coefficients into an existing CTPSA. |
| `ctpsa_t* mad_ctpsa_init(ctpsa_t *t, const desc_t *d, ord_t mo)` | Initialize supplied storage; unsafe if its allocation does not match `mo`. |
| `int mad_ctpsa_debug(const ctpsa_t *t, str_t name_, str_t fnam_, int line_, FILE *stream_)` | Print CTPSA debugging information. |
| `void mad_ctpsa_divc_clrcnt(void)` | Reset `mad_ctpsa_divc` diagnostic counters. |
| `void mad_ctpsa_divc_getcnt(ssz_t *cnt, ssz_t *fail)` | Retrieve `mad_ctpsa_divc` call and failure counters. |
