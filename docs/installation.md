# Installation

## Binary wheels (Linux x86_64)

Each [release](https://github.com/YPWang0818/sdpb-python/releases) carries
self-contained wheels for CPython 3.10 to 3.13 on Linux x86_64 (glibc 2.28 or
newer, i.e. any mainstream distribution from 2019 on). They bundle SDPB and
every library it needs, so nothing has to be compiled. Let pip choose the wheel
for your Python from the release page:

```
pip install sdpb-python --find-links https://github.com/YPWang0818/sdpb-python/releases/expanded_assets/v0.2.1
```

or install one file directly:

```
pip install https://github.com/YPWang0818/sdpb-python/releases/download/v0.2.1/sdpb_python-0.2.1-cp312-cp312-manylinux_2_28_x86_64.whl
```

Then check:

```
python -c "import sdpb_python; print(sdpb_python.sdpb_version())"
```

Two things to know about the wheels:

- They contain their own MPI library (MPICH), used only to initialise SDPB's
  single-process solver. Do not load another MPI in the same process, for
  example `mpi4py` built against OpenMPI.
- They are built with a portable OpenBLAS and generic x86_64 code; a native
  build tuned for your CPU can be somewhat faster.

For other platforms, or to work on the C++ side, build from source as below.

## Building from source

sdpb-python compiles SDPB and links it into a Python extension, so it needs
SDPB's full C++ tool chain.

### Requirements

- Python 3.10 or newer, with `mpmath`.
- A C++17 compiler and an MPI implementation (`mpicxx`), Boost, GMP with C++
  bindings, MPFR, FLINT (2.8 or newer), libarchive, libxml2, RapidJSON, and a
  CBLAS such as OpenBLAS.
- The [bootstrap-collaboration fork of Elemental](https://gitlab.com/bootstrapcollaboration/elemental)
  and [MPSolve](https://github.com/robol/MPSolve), normally built from source.

{doc}`building` is a tested, step-by-step recipe for Ubuntu 24.04; SDPB's own
`Install.md` in the submodule covers other systems and HPC sites.

### Steps

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

### Optional extras

- `numpy`: matrices for {class}`~sdpb_python.LMI` can be arrays.
- `sympy`: {meth}`Polynomial.from_sympy <sdpb_python.Polynomial.from_sympy>`.
- Documentation: `pip install sphinx myst-parser furo sphinx-copybutton`, then
  `sphinx-build -b html docs docs/_build/html`.

### Notes

- If MPI programs print `Authorization required, but no authorization protocol
  specified`, that is X11 noise caused by a set `DISPLAY`; `unset DISPLAY`
  silences it.
- The build links SDPB statically into the extension; the Elemental and
  MPSolve shared libraries are found through an rpath, so they must stay where
  they were at build time.
- The package runs on one MPI rank. Launching several processes with an
  `mpirun` or `srun` from a *different* MPI than the package links (the wheels
  bundle MPICH) does not create a multi-rank job: each process initialises MPI
  alone, and they would silently duplicate the work and overwrite each other's
  output. That is refused with an error naming the launcher; set
  `SDPB_PYTHON_ALLOW_MULTI_PROCESS=1` when the processes really are meant to be
  independent, for example a sweep in which each writes to its own directory.
- OpenBLAS selects its kernels at run time from the CPU family, and for an
  AMD family-15/17 CPU it takes Opteron kernels that use the 3DNow!
  instruction `femms`. QEMU/KVM guests with the default CPU model report such
  a family without 3DNow!, so the first `solve()` died with `Illegal
  instruction` inside `libopenblas` (v0.2.1 wheels and source builds alike).
  The package now sets `OPENBLAS_CORETYPE` itself on such CPUs before
  OpenBLAS loads (module `sdpb_python._cpu`; an existing setting is kept),
  and the wheels' OpenBLAS is built without the Opteron targets. On a source
  build, `OPENBLAS_CORETYPE=NEHALEM` in the environment is the manual
  equivalent.

## How the wheels are made

`.github/workflows/deps-image.yml` builds a `manylinux_2_28` Docker image with
MPICH, GMP, MPFR, FLINT, Boost, OpenBLAS, the Elemental fork and MPSolve
installed under `/opt/deps` (`docker/deps.Dockerfile`) and pushes it to
`ghcr.io/ypwang0818/sdpb-python-deps`. `.github/workflows/wheels.yml` then runs
`cibuildwheel` in that image on every `v*` tag: it builds SDPB's static
libraries once, builds the extension for each Python version, repairs the
wheels so they bundle every shared library, runs the test-suite inside each,
checks with `scripts/check_wheel_isa.sh` that no bundled library has AVX code
baked in (the `manylinux` tag fixes glibc, not the CPU), and attaches them to
the GitHub release. The same image can be pulled locally to reproduce a build.
