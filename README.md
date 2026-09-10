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

```python
import sdpb_python

result = sdpb_python.solve("path/to/sdp", "path/to/out", precision=768)
print(result.terminate_reason, result.primal_objective)
```

Keyword arguments map onto `sdpb` command-line options.

## Tests

```
pytest
```

Tests that need the compiled extension are skipped when it is not built.
