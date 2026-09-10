# sdpb-python

Cython bindings for [a fork of SDPB](https://github.com/YPWang0818/sdpb), the
semidefinite program solver used in the conformal bootstrap.

## Layout

```
├── .gitmodules               # submodule c-src/sdpb tracks the fork's python-api branch
├── pyproject.toml            # package metadata, build requirements (setuptools + Cython)
├── setup.py                  # builds the extension against SDPB's static libraries
├── c-src/sdpb/               # SDPB fork (git submodule), built with waf
├── src/sdpb_python/          # the package (hyphens are not allowed in module names)
│   ├── problem.py            # PMP, PolynomialMatrix, Polynomial, DampedRational
│   ├── lmi.py                # LMI
│   ├── handle.py             # Solver handle (repeated runs, warm starts, checkpoints)
│   ├── options.py            # SolverOptions
│   ├── solution.py           # Solution, TerminateReason
│   ├── io.py                 # read_pmp_json / write_pmp_json
│   ├── numbers.py            # mpmath <-> decimal strings
│   ├── solver.py             # legacy CLI passthrough (solve_dir)
│   ├── _sdpb.pyx, sdpb_wrapper.pxd   # Cython layer
│   └── cpp/sdpb_wrapper.*    # C++ shim over SDPB
├── scripts/build_sdpb.sh     # waf configure + build + pip install -e .
├── docs/                     # BUILDING.md (dependency recipe), API_DESIGN.md
└── tests/                    # pytest suite mirroring SDPB's own tests
```

## Clone and build

The SDPB sources are a git submodule pinned to the fork's `python-api` branch,
so clone with submodules:

```
git clone --recurse-submodules git@github.com:YPWang0818/sdpb-python.git
cd sdpb-python
# if you already cloned without --recurse-submodules:
git submodule update --init --recursive
```

### 1. Dependencies

SDPB needs a C++17 compiler, MPI, Boost, GMP, MPFR, FLINT, libarchive,
libxml2, RapidJSON, a CBLAS, plus the
[bootstrap-collaboration fork of Elemental](https://gitlab.com/bootstrapcollaboration/elemental)
and [MPSolve](https://github.com/robol/MPSolve), which are usually built from
source. `docs/BUILDING.md` is a tested step-by-step recipe for Ubuntu 24.04
that installs the source-built libraries to `~/install`; SDPB's own
`c-src/sdpb/Install.md` covers other systems.

### 2. Python environment

```
python3 -m venv .venv && source .venv/bin/activate
pip install setuptools wheel Cython mpmath pytest      # numpy, sympy optional
```

### 3. Build SDPB and the extension

```
scripts/build_sdpb.sh
```

The script runs `waf configure` with `-fPIC` (SDPB's static libraries are
linked into a Python shared object), `waf build`, and then
`pip install --no-build-isolation -e .`. It looks for Elemental and MPSolve
under `$DEPS_PREFIX` (default `~/install`); extra arguments go to
`waf configure`, and `JOBS` sets the build parallelism:

```
DEPS_PREFIX=/opt/sdpb-deps JOBS=4 scripts/build_sdpb.sh --flint-dir=/opt/flint
```

`setup.py` reads the include and library flags waf discovered from
`c-src/sdpb/build/c4che/_cache.py`, so the extension links exactly what SDPB
was configured with. Set `CC=mpicxx CXX=mpicxx` if you run `pip install`
yourself.

### 4. Check

```
python -c "import sdpb_python; print(sdpb_python.sdpb_version())"
pytest                              # about a minute; SDPB's own suite: c-src/sdpb/test/run_all_tests.sh
```

If MPI programs print "Authorization required, but no authorization protocol
specified", that is X11 noise from a set `DISPLAY`; `unset DISPLAY` silences it.

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
