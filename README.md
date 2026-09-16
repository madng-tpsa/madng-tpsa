# madng-tpsa - GTPSA (truncated power series) for Python

This repository exposes the features of MAD-NG's GTPSA (Generalised Truncated Power Series
Algebra) through a Python interface. The package builds `libmadng_tpsa.so` on Linux or
`libmadng_tpsa.dylib` on macOS, and uses CFFI ABI mode to access the native functionality.

```python
import numpy as np
import scipy

from madng_tpsa import Descriptor

d = Descriptor(variables=['x', 'y'], order=3)
x, y = d.vars()
f = x * x + 2.0 * y

# Inspect!
f  # => Tpsa({(0, 1): 2.0, (2, 0): 1.0})
f.format()  # => '2 * y + x**2'
f.format('table')  # => a Rich table showing all coefficients

# Basic operations
c = d.constant(42)
c.const_part  # => 42
f.const_part  # => 0

f.monomial_coeffs()  # => {(0, 1): 2.0, (2, 0): 1.0}
f.derivative('y')  # => Tpsa({(0, 0): 2.0})
f.grad()  # => [0, 2]

# Compatible with Numpy protocol
np.sin(f)  # => Tpsa({(0, 1): 2.0, (2, 0): 1.0, (0, 3): -1.3333333333333333})
scipy.special.wofz(f)  # => ComplexTpsa({(0, 0): (1+0j), (0, 1): 2.256...j, (2, 0): 1.128...j, (0, 2): (-4+0j), (2, 1): (-4+0j), (0, 3): -6.018...j})
```

Through the `xobjects` entry point declared in `pyproject.toml`, the MAD-NG TPSA C API is
automatically made available to Xobjects, so projects built on Xobjects can reuse it without
much extra wiring, relying on `madng-tpsa` abstractions in the Python-side API.

## Build

The project uses scikit-build-core and CMake:

```bash
git submodule update --init --recursive
pip install .
pip install -e .
python -m build
```

`pip install -e .` runs CMake and copies the generated shared library into
`src/madng_tpsa/lib/`, matching the installed wheel layout. MAD-NG headers copied into
`src/madng_tpsa/include/` and platform shared libraries in `src/madng_tpsa/lib/` are generated
artefacts and ignored by Git.

By default, CMake builds from the bundled `madng/src` checkout. To use another MAD-NG
source tree:

```bash
pip install . --config-settings=cmake.define.MADNG_SRC=/path/to/madng/src
```

Requirements: a C compiler, CMake, BLAS/LAPACK on Linux, and Accelerate on macOS.

## Development

Install the package together with the development tools:

```bash
python -m pip install --group dev -e .
```

Enable the pre-commit hooks:

```bash
pre-commit install
```

The hooks run Ruff formatting/linting and Ty type checking. To run the same checks
manually:

```bash
pre-commit run --all-files
```

## Tests

```bash
pytest tests/
```

The tests exercise the standalone engine bindings; xtrack is not imported.

## Documentation

Build the API documentation with:

```bash
python -m pip install --group docs -e .
make -C docs html
```

[madng]: https://github.com/MethodicalAcceleratorDesign/MAD
