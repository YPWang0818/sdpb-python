#!/usr/bin/env bash
# cibuildwheel "before-all": build SDPB's static libraries (with -fPIC) inside
# the dependency image, once per container, before the per-Python wheel builds.
set -euxo pipefail

DEPS="${DEPS_PREFIX:-/opt/deps}"
export PATH="$DEPS/bin:$PATH"
# The project is bind-mounted and owned by another uid: let git describe work.
git config --global --add safe.directory '*'

PY=/opt/python/cp312-cp312/bin/python
cd "${1:-/project}/c-src/sdpb"
CC=mpicc CXX=mpicxx CXXFLAGS="-fPIC" "$PY" ./waf configure \
  --elemental-incdir="$DEPS/include" --elemental-libdir="$DEPS/lib64" \
  --mpsolve-dir="$DEPS" --flint-dir="$DEPS" \
  --gmpxx-dir="$DEPS" --mpfr-dir="$DEPS" --boost-dir="$DEPS" \
  --cblas-dir="$DEPS" --rapidjson-dir="$DEPS" \
  || { cat build/config.log; exit 1; }
"$PY" ./waf build -j"$(nproc)" --targets=sdp_solve,pmp2sdp_lib,pmp,sdpb_util
