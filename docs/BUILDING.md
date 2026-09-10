# Building from scratch (Ubuntu 24.04 / Pop!_OS 24.04)

These are the exact steps used to build SDPB and the Python extension on a
fresh Ubuntu 24.04 machine.

## 1. System packages

```
sudo apt-get install -y openmpi-bin libopenmpi-dev libgmp-dev libmpfr-dev \
    libboost-all-dev libopenblas-dev libxml2-dev libarchive-dev rapidjson-dev \
    libmetis-dev libflint-dev pkg-config bison flex cmake g++
```

FLINT 3.0.1 from apt satisfies SDPB's minimum (2.8.0), so it is not built from source.

## 2. Elemental (bootstrap-collaboration fork), installed to `~/install`

```
git clone --depth=1 https://gitlab.com/bootstrapcollaboration/elemental.git
mkdir elemental/build && cd elemental/build
CC=mpicc CXX=mpicxx cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/install -DCMAKE_BUILD_TYPE=Release
make -j4 && make install
```

## 3. MPSolve, installed to `~/install`

```
git clone --depth=1 https://github.com/robol/MPSolve.git
cd MPSolve && ./autogen.sh
CC=mpicc CXX=mpicxx ./configure --prefix=$HOME/install --disable-dependency-tracking \
    --disable-examples --disable-ui --disable-graphical-debugger --disable-documentation
make -j4 && make install
```

## 4. SDPB and the Python package

```
git clone --recurse-submodules git@github.com:YPWang0818/sdpb-python.git
cd sdpb-python
python3 -m venv .venv && source .venv/bin/activate
pip install setuptools wheel Cython mpmath pytest
scripts/build_sdpb.sh
```

The submodule `c-src/sdpb` follows the fork's `python-api` branch (a few
patches over upstream master, see `API_DESIGN.md` §6).

The script configures waf with `CXXFLAGS=-fPIC` (required to link the static
libraries into a Python shared object), builds SDPB, then runs
`pip install -e .`.

## 5. Tests

SDPB's own suite (unit tests need 6 MPI ranks, integration tests take ~8 min):

```
cd c-src/sdpb && ./test/run_all_tests.sh
```

The Python package tests:

```
pytest
```

## Notes

- If `DISPLAY` is set without X authorisation, MPI programs print
  "Authorization required, but no authorization protocol specified" on stderr.
  It is harmless; `unset DISPLAY` silences it.
- The integration test `pmp2sdp / filesystem errors / invalid_nsv` has been
  seen to fail intermittently because the captured stderr of an aborted MPI
  run came back empty. Rerunning it passes.
