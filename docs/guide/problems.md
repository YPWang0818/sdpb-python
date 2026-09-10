# Describing problems

## Polynomials

{class}`~sdpb_python.Polynomial` holds coefficients in increasing degree:

```python
from sdpb_python import Polynomial
p = Polynomial([1, 0, "1/12"])      # 1 + x^2/12
p.degree                            # 2
p(2)                                # evaluates with Python arithmetic
```

Anywhere a polynomial is expected you may pass a bare coefficient list, or a
single number for a constant. With sympy installed:

```python
import sympy
x = sympy.Symbol("x")
Polynomial.from_sympy(x**4/12 + x**2 + 1)      # rationals stay exact (Fraction)
Polynomial.from_sympy(sympy.sqrt(2) * x, digits=300)
```

## Polynomial matrices

A {class}`~sdpb_python.PolynomialMatrix` is one positivity constraint. Its
`polynomials` argument is a `dim x dim` nested list; each entry is the list of
`N + 1` polynomials multiplying $z_0, \dots, z_N$. The matrix must be
symmetric (entries `[r][s]` and `[s][r]` equal).

```python
from sdpb_python import PolynomialMatrix, DampedRational

# 2x2 matrix, N = 1 (two polynomials per entry)
m = PolynomialMatrix(
    [[[[1, 0, 1], [0, 1]],   [[0, 1], [0]]],
     [[[0, 1], [0]],         [[2], [1, 0, 0, 1]]]],
    prefactor=DampedRational.exp_minus_x(poles=[-1, -2]),
)
m.dim, m.num_vectors, m.max_degree     # 2, 2, 3
```

### Prefactors and sampling

SDPB samples each matrix at `max_degree + 1` points chosen to minimise
interpolation error for the prefactor. You can override any part of that, using
the same optional fields as `pmp.json`:

| Field | Meaning |
|---|---|
| `prefactor` | {class}`~sdpb_python.DampedRational` $c\, b^x / \prod (x - p_k)$; default $e^{-x}$ |
| `reduced_prefactor` | Prefactor with fewer poles, used to choose the sample points |
| `max_num_poles` | Keep at most this many poles when deriving `reduced_prefactor` |
| `sample_points` | Explicit points $x_k \ge 0$ |
| `sample_scalings`, `reduced_sample_scalings` | Explicit prefactor values at the points |
| `bilinear_basis` | Explicit `(even, odd)` lists of basis polynomials |

Inspect what SDPB derived:

```python
s = m.sampled()                  # SampledMatrix
s.sample_points, s.sample_scalings, s.bilinear_basis, s.bilinear_bases
```

Sample points must be non-negative; sdpb-python checks this because SDPB
itself would crash on a negative point.

## Polynomial matrix programs

```python
from sdpb_python import PMP
pmp = PMP(objective=[a0, a1, ..., aN], normalization=[n0, ..., nN], matrices=[m1, m2, ...])
pmp.num_variables       # N
```

Validation at construction: non-empty objective, normalization of the same
length, every matrix carrying `N + 1` polynomials per entry, symmetric
matrices. Errors are `ValueError`s raised before anything reaches C++.

`normalization=None` means $z_0 = 1$ (SDPB's trivial normalization); then
`Solution.z` is `None` and `Solution.y` is $z_1, \dots, z_N$.

### The SDP behind a PMP

`pmp.to_sdp(precision=None, max_num_poles=None)` returns an
{class}`~sdpb_python.SDPData`: the objective constant $b_0$, the vector $b$,
the normalization, and per block `dim`, `num_points`, the sampled bases, `c`
and `B`. This is exactly what SDPB's `pmp2sdp` writes to an `sdp/` directory,
and is what the test-suite compares against SDPB's reference data.
`max_num_poles` plays the role of `pmp2sdp --maxNumPoles` and combines with
each matrix's own `max_num_poles` by taking the minimum.

## Linear matrix inequalities

{class}`~sdpb_python.LMI` takes `b` (length `N`), `f` (default 0) and
`blocks`, a list whose entries are the sequences $(M_0, M_1, \dots, M_N)$ of
symmetric matrices for each block. Matrices may be nested lists, numpy arrays,
or `mpmath.matrix`; symmetry and equal sizes are checked.

```python
from sdpb_python import LMI
lmi = LMI(b=[1, 1], f=3, blocks=[(M0, M1, M2)])
```

To minimise, negate `b` and `f`. An LMI is equivalent to a PMP whose
polynomials are constants with a constant prefactor; the test-suite checks the
two agree.
