# Numbers and precision

## Inputs

Wherever a number is expected, sdpb-python accepts

| Type | Interpretation |
|---|---|
| `int`, `bool` | exact |
| `str` | decimal (`"1e-30"`, `"0.1"`) or a fraction (`"1/12"`); passed to SDPB verbatim |
| `fractions.Fraction` | exact, rendered at the working precision |
| `float` | by its shortest round-trip decimal: `0.1` means one tenth, not the binary double |
| `mpmath.mpf` | exact |
| anything with a decimal `str()` (sympy numbers, numpy scalars, `Decimal`) | via `mpmath.mpf(str(value))` |

The conversion happens in {mod}`sdpb_python.numbers`. Values cross into C++
as decimal strings and are parsed by GMP at the fixed precision.

## Outputs

Every number in a {class}`~sdpb_python.Solution`, {class}`~sdpb_python.SDPData`
or {class}`~sdpb_python.SampledMatrix` is an `mpmath.mpf` (matrices are
`mpmath.matrix`). They are created with exactly the solver's precision
(SDPB prints `ceil(bits * log10(2)) + 1` digits, which identifies the binary
value uniquely), independent of mpmath's global precision at the time.

mpmath's global context is left alone. To continue computing at full
precision:

```python
import mpmath
mpmath.mp.prec = 768                 # or: with mpmath.workprec(768): ...
gap = abs(solution.primal_objective - solution.dual_objective)
```

Arithmetic on the results at a lower `mpmath.mp.prec` silently rounds, as with
any mpmath computation.

## One precision per process

See {ref}`precision`. In short: call `sdpb_python.set_precision(bits)` at the
start of a script or notebook; a different precision needs a new process.

GMP rounds the requested precision up to whole limbs, so
`sdpb_python.precision()` and `Solution.precision` report the value you asked
for and the actual bits respectively; the actual value is never smaller.
