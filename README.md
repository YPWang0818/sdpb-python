# sdpb-python

Cython bindings for [a fork of SDPB](https://github.com/YPWang0818/sdpb), the
semidefinite program solver used in the conformal bootstrap.

## Layout

```
├── .gitmodules
├── pyproject.toml            # package metadata, build requirements
├── setup.py                  # Cython extension build (links SDPB static libs)
├── c-src/sdpb/               # SDPB fork, git submodule
├── src/sdpb_python/          # importable package (hyphens are not allowed in module names)
│   ├── __init__.py
│   ├── solver.py             # high-level solve() / SDPBResult
│   ├── _sdpb.pyx             # Cython module
│   ├── sdpb_wrapper.pxd      # extern declarations
│   └── cpp/sdpb_wrapper.*    # C++ shim over SDPB
├── scripts/build_sdpb.sh
└── tests/
```

## Build

1. Clone with submodules:
   ```
   git clone --recurse-submodules <this repo>
   ```
2. Install SDPB's dependencies (see `docs/BUILDING.md` for a tested Ubuntu 24.04
   recipe, or `c-src/sdpb/Install.md`): MPI, Elemental, Boost, GMP, MPFR, FLINT,
   libarchive, libxml2, RapidJSON, MPSolve, CBLAS.
3. Build SDPB with waf and install the package:
   ```
   scripts/build_sdpb.sh [waf configure options, e.g. --elemental-dir=...]
   ```
   The script configures waf with `-fPIC` (needed to link SDPB's static
   libraries into a Python extension) and `setup.py` reads the include/library
   flags waf discovered from `c-src/sdpb/build/c4che/_cache.py`, so pass any
   extra dependency paths to waf via the script's arguments.

## Usage

Numbers are exchanged as `mpmath.mpf`; inputs accept `int`, `float`, `str`,
`Fraction` or `mpf`.

```python
import mpmath
import sdpb_python as sdpb

sdpb.set_precision(768)   # once per process (SDPB's Elemental fixes it on first use)

# maximize -y  s.t.  1 + x^4 + y (x^4/12 + x^2) >= 0 for x >= 0   (SDPB manual, 1d)
pmp = sdpb.PMP(
    objective=[0, -1], normalization=[1, 0],
    matrices=[sdpb.PolynomialMatrix([[[sdpb.Polynomial([1, 0, 0, 0, 1]),
                                       sdpb.Polynomial([0, 0, 1, 0, mpmath.mpf(1) / 12])]]])],
)
sol = pmp.solve(duality_gap_threshold="1e-30", want=("y", "z", "x"))
print(sol.status, sol.primal_objective, sol.y)

# a linear matrix inequality: maximize y s.t. [[1, y], [y, 1]] >= 0
sol = sdpb.LMI(b=[1], blocks=[([[1, 0], [0, 1]], [[0, 1], [1, 0]])]).solve()

# pmp.json interoperability with the SDPB command line tools
pmp = sdpb.read_pmp_json("pmp.json"); sdpb.write_pmp_json(pmp, "copy.json")
sdp = pmp.to_sdp()        # what pmp2sdp would write (objectives, per-block c, B, bases)
```

Keep the solver state between runs with a handle (warm starts, checkpoints,
tighter thresholds; Ctrl-C raises `SolverInterrupted` with the partial solution):

```python
with pmp.solver(precision=768) as solver:
    first = solver.run(max_iterations=50)
    final = solver.run(duality_gap_threshold="1e-60")
    solver.save_checkpoint("ck")
```

`SolverOptions` (or keyword overrides) mirror `sdpb`'s options in snake_case;
`Solution` carries objectives, errors, `y`, `z`, and optionally `x`, `X`, `Y`,
`c_minus_By`. See `docs/API_DESIGN.md`.

The legacy CLI passthrough remains as `sdpb.solve_dir(sdp_dir, out_dir, **cli_options)`.

## Tests

```
pytest
```

Tests that need the compiled extension are skipped when it is not built. The
long end-to-end datasets (SingletScalar) run with `pytest --run-slow`.
