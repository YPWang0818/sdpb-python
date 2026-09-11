# sdpb-python

Python bindings for [SDPB](https://github.com/davidsd/sdpb), the arbitrary
precision semidefinite program solver used in the conformal bootstrap, built
against [a fork](https://github.com/YPWang0818/sdpb) that exposes SDPB's
solver as a library.

With sdpb-python you describe a *polynomial matrix program* (or a plain linear
matrix inequality) with Python objects, solve it in-process without writing
SDPB's input directories, and get the solution back as `mpmath` numbers.
On Linux x86_64 it installs from the prebuilt wheels of a release:

```
pip install sdpb-python --find-links https://github.com/YPWang0818/sdpb-python/releases/expanded_assets/v0.2.0
```


```python
import sdpb_python as sdpb

pmp = sdpb.PMP(
    objective=[0, -1], normalization=[1, 0],
    matrices=[sdpb.PolynomialMatrix([[[[1, 0, 0, 0, 1], [0, 0, 1, 0, "1/12"]]]])],
)
solution = pmp.solve(precision=768)
print(solution.status.value, solution.dual_objective)
```

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
background
```

```{toctree}
:maxdepth: 2
:caption: User guide

guide/problems
guide/solving
guide/solver_handle
guide/io
guide/numbers
guide/development
guide/testing
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
building
design
```

## Scope

- Single-process solving of PMPs and LMIs at arbitrary precision.
- Full access to SDPB's solver options, termination reasons, and solution
  parts (`x`, `y`, `z`, `X`, `Y`, `c - B·y`).
- A solver handle for repeated runs, warm starts, checkpoints, and clean
  interruption with Ctrl-C.
- `pmp.json` reading and writing for interoperability with SDPB's command line
  tools (`pmp2sdp`, `sdpb`, `spectrum`).

Not covered (yet): multi-rank MPI runs, the `spectrum` and `approx_objective`
tools, and SDPB's Mathematica and XML input formats.

## License

MIT, like SDPB itself. See the `LICENSE` file and the README for the
licenses of the linked libraries.
