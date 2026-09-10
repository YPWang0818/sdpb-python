# API reference

Everything below is importable from `sdpb_python`.

## Problems

```{eval-rst}
.. autoclass:: sdpb_python.PMP
.. autoclass:: sdpb_python.PolynomialMatrix
.. autoclass:: sdpb_python.Polynomial
.. autoclass:: sdpb_python.DampedRational
.. autoclass:: sdpb_python.LMI
.. autoclass:: sdpb_python.SampledMatrix
.. autoclass:: sdpb_python.SDPData
.. autoclass:: sdpb_python.problem.SDPBlock
```

## Solving

```{eval-rst}
.. autoclass:: sdpb_python.SolverOptions
   :exclude-members: replace
.. autoclass:: sdpb_python.Solution
.. autoclass:: sdpb_python.TerminateReason
   :undoc-members:
.. autoclass:: sdpb_python.BlockInfo
.. autoclass:: sdpb_python.Solver
.. autoexception:: sdpb_python.SolverInterrupted
.. autoexception:: sdpb_python.SDPBError
```

## Precision

```{eval-rst}
.. autofunction:: sdpb_python.set_precision
.. autofunction:: sdpb_python.precision
```

## Files

```{eval-rst}
.. autofunction:: sdpb_python.read_pmp_json
.. autofunction:: sdpb_python.write_pmp_json
.. autofunction:: sdpb_python.io.read_nsv
.. autofunction:: sdpb_python.io.pmp_to_json_dict
```

## Numbers

```{eval-rst}
.. automodule:: sdpb_python.numbers
   :members: to_str, from_str, digits_for
```

## Legacy passthrough

```{eval-rst}
.. autofunction:: sdpb_python.solve_dir
.. autoclass:: sdpb_python.SDPBResult
.. autofunction:: sdpb_python.sdpb_version
```
