# Installation

sdpb-python compiles SDPB and links it into a Python extension, so it needs
SDPB's full C++ tool chain. There are no binary wheels.

## Requirements

- Python 3.10 or newer, with `mpmath`.
- A C++17 compiler and an MPI implementation (`mpicxx`), Boost, GMP with C++
  bindings, MPFR, FLINT (2.8 or newer), libarchive, libxml2, RapidJSON, and a
  CBLAS such as OpenBLAS.
- The [bootstrap-collaboration fork of Elemental](https://gitlab.com/bootstrapcollaboration/elemental)
  and [MPSolve](https://github.com/robol/MPSolve), normally built from source.

{doc}`building` is a tested, step-by-step recipe for Ubuntu 24.04; SDPB's own
`Install.md` in the submodule covers other systems and HPC sites.

## Steps

1. Clone with the SDPB submodule (it follows the fork's `python-api` branch):

   ```
   git clone --recurse-submodules git@github.com:YPWang0818/sdpb-python.git
   cd sdpb-python
   ```

2. Create an environment with the build tools:

   ```
   python3 -m venv .venv && source .venv/bin/activate
   pip install setuptools wheel Cython mpmath pytest
   ```

3. Build SDPB and install the package in editable mode:

   ```
   scripts/build_sdpb.sh
   ```

   The script configures SDPB's waf build with `-fPIC`, builds it, and runs
   `pip install --no-build-isolation -e .`. It expects Elemental and MPSolve
   under `$DEPS_PREFIX` (default `~/install`); extra arguments are passed to
   `waf configure`, `JOBS` sets the parallelism.

4. Check:

   ```
   python -c "import sdpb_python; print(sdpb_python.sdpb_version())"
   pytest
   ```

## Optional extras

- `numpy`: matrices for {class}`~sdpb_python.LMI` can be arrays.
- `sympy`: {meth}`Polynomial.from_sympy <sdpb_python.Polynomial.from_sympy>`.
- Documentation: `pip install sphinx myst-parser furo sphinx-copybutton`, then
  `sphinx-build -b html docs docs/_build/html`.

## Notes

- If MPI programs print `Authorization required, but no authorization protocol
  specified`, that is X11 noise caused by a set `DISPLAY`; `unset DISPLAY`
  silences it.
- The build links SDPB statically into the extension; the Elemental and
  MPSolve shared libraries are found through an rpath, so they must stay where
  they were at build time.
