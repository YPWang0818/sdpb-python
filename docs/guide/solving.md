# Solving

## `solve()`

Both {class}`~sdpb_python.PMP` and {class}`~sdpb_python.LMI` have

```python
solution = problem.solve(options=None, **overrides)
```

`options` is a {class}`~sdpb_python.SolverOptions`; keyword overrides use the
same names. Defaults are SDPB's own, so `problem.solve()` behaves like
`sdpb` with no options except that no checkpoint or output directory is
written.

```python
from sdpb_python import SolverOptions
opts = SolverOptions(precision=768, max_iterations=1000,
                     duality_gap_threshold="1e-30", primal_error_threshold="1e-30",
                     dual_error_threshold="1e-30")
solution = pmp.solve(opts, want=("y", "z", "x"))
```

The full list of fields, with SDPB's defaults and the corresponding command
line flags, is in the {class}`~sdpb_python.SolverOptions` reference. The ones
you will touch most:

- `precision`: bits; see below.
- `duality_gap_threshold`, `primal_error_threshold`, `dual_error_threshold`:
  the optimality test. Pass tiny numbers as strings (`"1e-30"`), not floats.
- `max_iterations`, `max_runtime`: budgets.
- `find_primal_feasible`, `find_dual_feasible`, `detect_*_feasible_jump`:
  early stops used in feasibility (bisection) searches, where the objective is
  irrelevant.
- `want`: which solution parts to return; `("y", "z")` by default.
- `checkpoint_dir`, `checkpoint_interval`, `output_dir`: files, see below.
- `verbosity`: `"none"`, `"regular"`, `"debug"`, `"trace"`. SDPB prints to the
  C++ standard output, which is fine in a terminal; in Jupyter it goes to the
  kernel's console.

(precision)=
## Precision

SDPB computes with GMP floats of a fixed number of bits and its Elemental
library allows that number to be set **once per process**. sdpb-python
therefore:

- fixes the precision on the first call that needs it, from
  `SolverOptions.precision` (SDPB's default 400) or an explicit
  {func}`sdpb_python.set_precision`;
- treats `precision=None` as "the fixed precision";
- raises {class}`~sdpb_python.SDPBError` (`"precision is already fixed at N
  bits ..."`) when a later call asks for a different value.

```python
import sdpb_python as sdpb
sdpb.set_precision(768)       # do this first in scripts and notebooks
sdpb.precision()              # 768
```

To work at several precisions, use separate processes
(`multiprocessing` with the `"spawn"` start method, or `subprocess`).

Results come back as `mpmath.mpf` created at the solver's precision. mpmath's
global context is not changed; set `mpmath.mp.prec` (or use
`mpmath.workprec`) before doing further high-precision arithmetic with them.

## Results

{class}`~sdpb_python.Solution` fields:

| Field | Content |
|---|---|
| `status`, `optimal` | {class}`~sdpb_python.TerminateReason`; `optimal` is `status is PRIMAL_DUAL_OPTIMAL` |
| `primal_objective`, `dual_objective`, `duality_gap`, `primal_error`, `dual_error` | `mpf` |
| `y` | list of `mpf`, length `N` |
| `z` | list of `mpf`, length `N + 1`, or `None` (no normalization / not requested) |
| `x` | per block, list of `mpf` (requested with `"x"`) |
| `X`, `Y` | per block, `(even, odd)` pair of `mpmath.matrix` (requested with `"X"`, `"Y"`) |
| `c_minus_By` | per block, list of `mpf` (requested with `"c_minus_By"`) |
| `iterations`, `runtime_seconds`, `precision`, `blocks` | bookkeeping |

Termination reasons, in the priority SDPB checks them:

| `TerminateReason` | SDPB text | Cause |
|---|---|---|
| `PRIMAL_DUAL_OPTIMAL` | found primal-dual optimal solution | gap and both errors below thresholds |
| `DUAL_FEASIBLE` | found dual feasible solution | `find_dual_feasible` |
| `PRIMAL_FEASIBLE` | found primal feasible solution | `find_primal_feasible` |
| `DUAL_FEASIBLE_JUMP_DETECTED` | dual feasible jump detected | `detect_dual_feasible_jump` |
| `PRIMAL_FEASIBLE_JUMP_DETECTED` | primal feasible jump detected | `detect_primal_feasible_jump` |
| `MAX_ITERATIONS_EXCEEDED` | maxIterations exceeded | `max_iterations` |
| `MAX_RUNTIME_EXCEEDED` | maxRuntime exceeded | `max_runtime` |
| `PRIMAL_STEP_TOO_SMALL`, `DUAL_STEP_TOO_SMALL` | primal/dual step too small | `min_primal_step`, `min_dual_step` |
| `MAX_COMPLEMENTARITY_EXCEEDED` | maxComplementarity exceeded | numerical blow-up |
| `SIGTERM_RECEIVED` | SIGTERM signal received | SIGTERM, or Ctrl-C through a {class}`~sdpb_python.Solver` |

## Files SDPB writes

Nothing is written by default. With `output_dir="out"`, SDPB writes
`out/iterations.json` (one record per iteration with objectives, errors, step
lengths and condition numbers) and `out/c_minus_By/c_minus_By.json`, exactly
like `sdpb --outDir out`. Existing files are rotated to `iterations.0.json`
and so on.

With `checkpoint_dir="ck"`, SDPB loads a checkpoint from `ck` if one exists,
writes one every `checkpoint_interval` seconds, and writes a final one when it
stops. Checkpoints are binary and only usable at the same precision. Restarting
from a checkpoint continues the iteration where it left off, so a run
interrupted after $k$ iterations and restarted reaches the optimum in $n - k$
more iterations, where $n$ is the uninterrupted count.

## Errors

Problems with the Python-side description raise `ValueError` or `TypeError`.
Anything that goes wrong inside SDPB raises {class}`~sdpb_python.SDPBError`,
whose message is the first line of SDPB's message and whose `details`
attribute holds the full text including SDPB's stack trace.

sdpb-python supports a single MPI rank. Under `mpirun -n 2` the solve raises
an `SDPBError` explaining this instead of hanging.
