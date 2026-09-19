# Building from scratch (Ubuntu 24.04 / Pop!_OS 24.04)

These are the exact steps used to build SDPB and the Python extension on a
fresh Ubuntu 24.04 machine.

## 1. System packages

```
sudo apt-get install -y openmpi-bin libopenmpi-dev libgmp-dev libmpfr-dev \
    libboost-dev libboost-program-options-dev libboost-serialization-dev \
    libboost-stacktrace-dev libopenblas-dev rapidjson-dev \
    libmetis-dev libflint-dev pkg-config cmake g++
```

FLINT 3.0.1 from apt satisfies SDPB's minimum (2.8.0), so it is not built from source.

This is what the Python extension needs. SDPB's command-line tools and its own
test-suite (section 4) additionally need `libboost-all-dev libxml2-dev
libarchive-dev bison flex` and MPSolve; none of that is used by the package.

## 2. Elemental (bootstrap-collaboration fork), installed to `~/install`

```
git clone --depth=1 https://gitlab.com/bootstrapcollaboration/elemental.git
mkdir elemental/build && cd elemental/build
CC=mpicc CXX=mpicxx cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/install -DCMAKE_BUILD_TYPE=Release
make -j4 && make install
```

## 3. SDPB and the Python package

```
git clone --recurse-submodules git@github.com:YPWang0818/sdpb-python.git
cd sdpb-python
python3 -m venv .venv && source .venv/bin/activate
pip install setuptools wheel Cython mpmath pytest
scripts/build_sdpb.sh
```

The submodule `c-src/sdpb` follows the fork's `python-api` branch (a few
patches over upstream master, see `API_DESIGN.md` §6).

The script configures waf with `--libs-only` (only the four static libraries
the extension links: no tools, no MPSolve, libxml2 or libarchive) and
`CXXFLAGS=-fPIC` (required to link them into a Python shared object), builds
them, then runs `pip install -e .`.

## 4. Tests

The Python package tests:

```
pytest
```

SDPB's own suite is only of interest when changing the fork. It needs the full
SDPB build, hence the extra packages named in section 1 and
[MPSolve](https://github.com/robol/MPSolve) under `~/install`. The test script
expects it in `build/`, which the package's `--libs-only` build also uses, so
rebuild the latter afterwards (unit tests need 6 MPI ranks, integration tests
take ~8 min):

```
cd c-src/sdpb
CC=mpicc CXX=mpicxx CXXFLAGS=-fPIC python3 ./waf configure \
    --elemental-dir=$HOME/install --mpsolve-dir=$HOME/install
python3 ./waf build && ./test/run_all_tests.sh
cd ../.. && scripts/rebuild.sh --force    # back to the --libs-only build
```

## Notes

- If `DISPLAY` is set without X authorisation, MPI programs print
  "Authorization required, but no authorization protocol specified" on stderr.
  It is harmless; `unset DISPLAY` silences it.
- The integration test `pmp2sdp / filesystem errors / invalid_nsv` has been
  seen to fail intermittently because the captured stderr of an aborted MPI
  run came back empty. Rerunning it passes.
- OpenBLAS selects its kernels at run time from the CPU family, and for an
  AMD family-15/17 CPU it takes Opteron kernels that use the 3DNow!
  instruction `femms`. QEMU/KVM guests with the default CPU model report such
  a family without 3DNow!, so the first `solve()` died with `Illegal
  instruction` inside `libopenblas` (wheel builds before this fix and source
  builds against a system OpenBLAS alike).
  The package now sets `OPENBLAS_CORETYPE` itself on such CPUs before
  OpenBLAS loads (module `sdpb_python._cpu`; an existing setting is kept),
  and the wheels' OpenBLAS is built without the Opteron targets. On a source
  build, `OPENBLAS_CORETYPE=NEHALEM` in the environment is the manual
  equivalent.
