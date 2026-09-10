# Quickstart

This walks through the one-dimensional example from the SDPB manual:

$$
\text{maximize } -y \quad \text{such that} \quad 1 + x^4 + y\left(\tfrac{x^4}{12} + x^2\right) \ge 0 \ \text{ for all } x \ge 0 .
$$

## 1. Fix the precision

SDPB works at a fixed number of bits, and (because of a restriction in the
Elemental library it uses) that number can be set only **once per process**:

```python
import sdpb_python as sdpb

sdpb.set_precision(768)
```

If you skip this, the first solve fixes it (default 400 bits). See
{ref}`precision`.

## 2. Describe the problem

The constraint is a $1 \times 1$ polynomial matrix whose single entry holds
two polynomials, one multiplying $z_0$ and one multiplying $z_1$:

```python
P0 = sdpb.Polynomial([1, 0, 0, 0, 1])            # 1 + x^4
P1 = sdpb.Polynomial([0, 0, 1, 0, "1/12"])       # x^2 + x^4/12
matrix = sdpb.PolynomialMatrix([[[P0, P1]]])      # polynomials[row][col] = [P0, P1]
```

A {class}`~sdpb_python.PMP` maximises `a . z` subject to `n . z = 1` and the
positivity of `sum_i z_i M_i(x)` for every matrix. With `a = (0, -1)` and
`n = (1, 0)` we have $z_0 = 1$, $z_1 = y$, and the objective $-y$:

```python
pmp = sdpb.PMP(objective=[0, -1], normalization=[1, 0], matrices=[matrix])
```

Coefficients can be `int`, `float`, `str` (decimal or, as above, a fraction),
`fractions.Fraction`, or `mpmath.mpf`; see {doc}`guide/numbers`.

## 3. Solve

```python
solution = pmp.solve(duality_gap_threshold="1e-30",
                     primal_error_threshold="1e-30", dual_error_threshold="1e-30")
print(solution.status)            # TerminateReason.PRIMAL_DUAL_OPTIMAL
print(solution.optimal)           # True
print(solution.dual_objective)    # 1.8402657631320492...  (the maximum of -y)
print(solution.y)                 # [mpf('-1.8402657631320492...')]
print(solution.z)                 # [mpf('1.0'), mpf('-1.8402657631320492...')]
```

Every keyword is one of SDPB's options in snake_case; the defaults are SDPB's
own. `solution.y` are the dual variables, `solution.z` the PMP variables
including the normalised component, so here $z = (1, y)$ and the objective
$a \cdot z = -y \approx 1.84$. Numbers are `mpmath.mpf` carrying the solver's
full precision, but mpmath *prints* them at its current working precision:
set `mpmath.mp.prec = 768` to see all digits and before doing further
arithmetic with them.

## 4. Ask for more of the solution

```python
solution = pmp.solve(want=("x", "y", "z", "X", "Y", "c_minus_By"), output_dir="out")
solution.x[0]           # primal variables of block 0
solution.Y[0][0]        # even-parity PSD block of Y for block 0 (an mpmath.matrix)
solution.c_minus_By[0]  # the extremal functional c - B.y on block 0
```

With `output_dir` set, SDPB also writes `iterations.json` and
`c_minus_By/c_minus_By.json` there, exactly as the `sdpb` executable does.

## 5. Linear matrix inequalities

For a problem without polynomials, use {class}`~sdpb_python.LMI`:
maximise $f + b \cdot y$ subject to $M_0 + \sum_n y_n M_n \succeq 0$.

```python
lmi = sdpb.LMI(b=[1], blocks=[([[1, 0], [0, 1]], [[0, 1], [1, 0]])])
print(lmi.solve().y)    # [mpf('1.0')]  since [[1, y], [y, 1]] >= 0  <=>  |y| <= 1
```

## 6. Keep the solver alive

To run in stages, warm-start, or checkpoint, hold a
{class}`~sdpb_python.Solver`:

```python
with pmp.solver() as solver:
    coarse = solver.run(max_iterations=20)
    fine = solver.run(duality_gap_threshold="1e-60")
    solver.save_checkpoint("ck")
```

Pressing Ctrl-C during `run()` stops SDPB at the end of the current iteration
and raises {class}`~sdpb_python.SolverInterrupted`, whose `solution` attribute
holds the state reached so far. See {doc}`guide/solver_handle`.

## 7. Files

Problems written for the SDPB command line tools load directly:

```python
pmp = sdpb.read_pmp_json("pmp.json")        # also .nsv lists of json files
sdpb.write_pmp_json(pmp, "copy.json")
sdp = pmp.to_sdp()                          # the sdp/ directory contents, in memory
```
