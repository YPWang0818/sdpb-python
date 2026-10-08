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

To work at several precisions, use separate processes: a
{class}`~sdpb_python.SolverPool` per precision (see below), or
`multiprocessing` with the `"spawn"` start method.

Results come back as `mpmath.mpf` created at the solver's precision. mpmath's
global context is not changed; set `mpmath.mp.prec` (or use
`mpmath.workprec`) before doing further high-precision arithmetic with them.

## Many solves in parallel

A process runs one solve at a time. {class}`~sdpb_python.SolverPool` runs
independent solves in worker processes, one per physical core by default
(capped by the available memory), each with one BLAS thread:

```python
from sdpb_python import SolverPool

with SolverPool(workers=4, precision=768) as pool:
    futures = [pool.submit(pmp, fast_step_length=True) for pmp in problems]
    solutions = [f.result() for f in futures]      # or: pool.map(problems, ...)
```

`submit` returns a {class}`concurrent.futures.Future`; its result is the
{class}`~sdpb_python.Solution`, or it raises the {class}`~sdpb_python.SDPBError`
of the solve. A pooled solve gives bit for bit the result of `pmp.solve(...)`
with the same options, whatever the number of workers. The precision is fixed
per pool; `precision=` in the options must match it.

`pool.cancel(future)` drops a queued solve, or stops a running one at the end
of its current iteration (its future then raises
{class}`~sdpb_python.SolveCancelled`); a solve that finishes before the request
arrives keeps its result. If a worker dies (a crash, `MPI_Abort`, an OOM kill)
its solve raises {class}`~sdpb_python.SDPBWorkerError` and the pool starts a new
worker. Leaving the `with` block waits for the queued solves; an exception
inside it (e.g. Ctrl-C) kills the workers instead.

The workers are started with the `"spawn"` method, which re-imports the main
module: in a script, create the pool under `if __name__ == "__main__":`.
`solve_many(problems, workers=..., **options)` is the one-call form.

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

## Profiling

With `profile_path="solve.json"`, SDPB's timer tree of the solve is written to
that file (valid JSON; the format is documented in SDPB's `docs/Usage.md`,
section "Profiling"). The root is `solve_pmp` with the children `parse_spec`
(the Python strings to `BigFloat`), `pmp2sdp` (sampling and bilinear bases:
`convert`, then `block_info` and `sdp_ctor`) and `solver` (`ctor`, SDPB's own
`run` subtree with one `iter` node per iteration and the per-phase timers, and
`collect` for converting the results back to strings). `meta` carries the
problem shapes, the solver parameters (`meta.parameters`), and your labels:

```python
sol = pmp.solve(profile_path="solve.json", profile_meta={"case": "e7", "node": "3"})
print(sol.profile_path)
```

Without `profile_path` SDPB records no timers at all (building the timer tree
costs a few percent of a small solve). A `Solver` handle created without a
profile starts recording at the first `run(profile_path=...)`.

`profile_detail=0` keeps only the coarse phase timers (a few thousand entries),
`1` (default) adds the per-block kernel timers with their matrix shapes (about
300 entries per iteration), `2` adds MPI barrier probes. `profile_autosave=True`
also rewrites the file every 10 iterations, so a killed solve leaves data. The
file is written when the solve raises an `SDPBError` too. For a `Solver` handle
the profile is cumulative: `handle_build` and one `handle_run` (attribute `k`)
per `run()`, rewritten after every run. The analysis tools in SDPB's
`scripts/profile/` read these files.

## Errors

Problems with the Python-side description raise `ValueError` or `TypeError`.
This includes a variable that multiplies only zero polynomials in every
matrix: its column of the SDP would vanish and SDPB would abort deep inside
the solver, so the {class}`~sdpb_python.PMP` constructor rejects it.

Anything that goes wrong inside SDPB raises {class}`~sdpb_python.SDPBError`.
`str(error)` is SDPB's message with the throwing source location in brackets,
for example

```
Error when computing Cholesky decomposition of block_0: A was not numerically HPD: 2 3 -2.8e-115 [../src/sdp_solve/.../compute_Q.cxx:36 (initialize_schur_off_diagonal())]
```

`error.details` holds the full text SDPB produced, including its stack trace,
and `error.location` the `file:line (function)` alone.

## Unbounded and infeasible programs

SDPB has no explicit "unbounded" or "infeasible" status. An unbounded program
(for example maximise `y` subject to `1 + y x >= 0`) ends either with
{attr}`~sdpb_python.TerminateReason.MAX_COMPLEMENTARITY_EXCEEDED` or with a
Cholesky failure raised as an `SDPBError` ("not numerically HPD"), depending
on precision and the iteration path. Code that searches over parameters
should handle both. If only feasibility matters, `find_dual_feasible=True`
or `find_primal_feasible=True` stop cleanly with a `*_FEASIBLE` status as soon
as a feasible point is found, which is the usual way to run bisection
searches. Thresholds far below what the precision can resolve (for example
`1e-150` at 400 bits) also end in the Cholesky failure.

sdpb-python supports a single MPI rank. Under `mpirun -n 2` the solve raises
an `SDPBError` explaining this instead of hanging.
