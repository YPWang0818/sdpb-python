# The solver handle

`solve()` builds SDPB's solver, runs it, and throws it away. A
{class}`~sdpb_python.Solver` keeps it alive:

```python
solver = pmp.solver(options=None, **overrides)     # also lmi.solver(...)
```

Construction converts the problem to an SDP and allocates the initial iterate
($x = y = 0$, $X$ and $Y$ multiples of the identity, or a checkpoint from
`checkpoint_dir`). Use it as a context manager, or call `close()`, to free the
C++ objects; a closed handle raises `SDPBError` on use.

## Repeated runs

```python
with pmp.solver(duality_gap_threshold="1e-30") as solver:
    first = solver.run(max_iterations=20)      # MAX_ITERATIONS_EXCEEDED
    final = solver.run()                       # continues from iteration 21
    solver.total_iterations                    # 20 + final.iterations
```

`run()` continues the interior-point iteration from the current
$(x, X, y, Y)$ with the options given (the handle's options, overridden per
call). The trajectory is the same as an uninterrupted run, so splitting a solve
into stages costs nothing. Typical uses: tightening thresholds after a coarse
solve, extending the iteration budget, or changing `want` between stages.

`state()` returns the current iterate as a {class}`~sdpb_python.Solution`
without iterating (its `status` is that of the last run).

## Warm starts

```python
reference = pmp.solve(want=("y", "X", "Y"))
with pmp.solver() as solver:
    solver.warm_start(y=reference.y, X=reference.X, Y=reference.Y)
    solution = solver.run()          # converges in a few iterations
```

`warm_start` overwrites any of `y` (length `N`), `X` and `Y` (per block a pair
`(even, odd)` of matrices in the shapes SDPB uses, as returned in a
`Solution`). Shapes are checked. Note that a good warm start needs $X$ and
$Y$ as well as $y$: setting only `y` with the default $X, Y$ is not faster.

## Checkpoints

```python
solver.save_checkpoint("ck")                    # binary checkpoint_<g>_<rank> + checkpoint.json
later = pmp.solver(checkpoint_dir="ck")         # loads it
```

This is the same format the `sdpb` executable writes; the two are
interchangeable at equal precision and rank count.

## Interrupting

Pressing Ctrl-C (SIGINT) while `run()` is executing does not kill the process.
sdpb-python asks SDPB to stop, which it does at the end of the current
iteration, and `run()` raises {class}`~sdpb_python.SolverInterrupted`, a
`KeyboardInterrupt` subclass:

```python
try:
    solution = solver.run()
except sdpb.SolverInterrupted as e:
    partial = e.solution           # state at the interruption, status SIGTERM_RECEIVED
    solver.save_checkpoint("ck")   # the handle is still usable
```

Because SDPB checks for the request once per iteration, a large problem can
take one full iteration (seconds to minutes) to react. Python's own SIGINT
handler is restored after every run.

Without a handle (plain `solve()`), Ctrl-C likewise stops SDPB cleanly; the
returned `Solution` then has status `SIGTERM_RECEIVED`.
