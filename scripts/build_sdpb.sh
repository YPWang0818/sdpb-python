#!/usr/bin/env bash
# Build the SDPB static libraries with waf, then build the Python extension.
# Requires: MPI (mpicxx), Elemental, Boost, GMP, MPFR, FLINT, libarchive,
# libxml2, RapidJSON, MPSolve, CBLAS.  See c-src/sdpb/Install.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDPB="$ROOT/c-src/sdpb"

if [ ! -f "$SDPB/wscript" ]; then
  echo "Submodule missing; run: git submodule update --init --recursive" >&2
  exit 1
fi

cd "$SDPB"
python3 ./waf configure "$@"
python3 ./waf build -j"$(nproc)"

cd "$ROOT"
CC=mpicxx CXX=mpicxx pip install -e ".[dev]"
