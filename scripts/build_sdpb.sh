#!/usr/bin/env bash
# Build the SDPB static libraries with waf, then build the Python extension.
#
# Requires: MPI (mpicxx), Elemental (bootstrap-collaboration fork), Boost
# (headers, program_options, serialization), GMP, MPFR, FLINT, RapidJSON, CBLAS.
# SDPB is configured with --libs-only: only the four static libraries the
# extension links are built, so MPSolve, libxml2 and libarchive are not needed.
# See docs/BUILDING.md for the full setup.
#
# Environment variables:
#   DEPS_PREFIX   where Elemental is installed (default: $HOME/install)
#   JOBS          parallel build jobs (default: nproc)
# Extra arguments are passed to `waf configure`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDPB="$ROOT/c-src/sdpb"
DEPS_PREFIX="${DEPS_PREFIX:-$HOME/install}"
JOBS="${JOBS:-$(nproc)}"

if [ ! -f "$SDPB/wscript" ]; then
  echo "Submodule missing; run: git submodule update --init --recursive" >&2
  exit 1
fi

cd "$SDPB"
# -fPIC is required because the static libraries are linked into a Python
# shared extension.  The default waf build (used for the sdpb executables)
# does not add it.
CC=mpicc CXX=mpicxx CXXFLAGS="-fPIC" python3 ./waf configure --libs-only \
  --elemental-dir="$DEPS_PREFIX" "$@"
python3 ./waf build -j"$JOBS"

cd "$ROOT"
CC=mpicxx CXX=mpicxx pip install --no-build-isolation -e ".[dev]"
