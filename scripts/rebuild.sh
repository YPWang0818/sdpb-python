#!/usr/bin/env bash
# Rebuild only what changed since the last build, for editable installs.
#
#   scripts/rebuild.sh            rebuild what is out of date
#   scripts/rebuild.sh --pull     git pull (with submodule) first
#   scripts/rebuild.sh --force    rebuild SDPB and the extension unconditionally
#
# Decides in three tiers:
#   1. SDPB static libraries  (c-src/sdpb changed, or never built)   ~6 min
#   2. the Cython extension   (src/sdpb_python/cpp, *.pyx, *.pxd, setup.py,
#                              or tier 1 rebuilt)                    ~3 min
#   3. nothing                (pure-Python changes are live already)
# Run it from the environment the package is installed into (e.g. your
# project's venv); it uses that environment's `pip`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDPB="$ROOT/c-src/sdpb"
BUILD="$SDPB/build"
STAMP="$BUILD/.sdpb-python-built"     # submodule commit the libraries were built from
DEPS_PREFIX="${DEPS_PREFIX:-$HOME/install}"
JOBS="${JOBS:-$(nproc)}"
PKG="$ROOT/src/sdpb_python"

force=0
for arg in "$@"; do
  case "$arg" in
    --force) force=1 ;;
    --pull) (cd "$ROOT" && git pull && git submodule update --init --recursive) ;;
    -h|--help) sed -n 2,14p "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ ! -f "$SDPB/wscript" ]; then
  echo "SDPB submodule missing; run: git submodule update --init --recursive" >&2
  exit 1
fi
if ! command -v pip >/dev/null; then
  echo "pip not found; activate the environment sdpb-python is installed into" >&2
  exit 1
fi

# ---- tier 1: SDPB static libraries -----------------------------------------
sdpb_head="$(git -C "$SDPB" rev-parse HEAD)"
sdpb_dirty="$(git -C "$SDPB" status --porcelain -- src wscript waf-tools | head -c1)"
rebuild_sdpb=0
if [ "$force" = 1 ]; then
  rebuild_sdpb=1; why_sdpb="--force"
elif [ ! -f "$BUILD/libsdp_solve.a" ] || [ ! -f "$BUILD/libpmp.a" ]; then
  rebuild_sdpb=1; why_sdpb="libraries not built"
elif [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$sdpb_head" ]; then
  rebuild_sdpb=1; why_sdpb="submodule moved to $(git -C "$SDPB" rev-parse --short HEAD)"
elif [ -n "$sdpb_dirty" ]; then
  rebuild_sdpb=1; why_sdpb="uncommitted changes in c-src/sdpb"
fi

if [ "$rebuild_sdpb" = 1 ]; then
  echo "== SDPB libraries: rebuilding ($why_sdpb)"
  cd "$SDPB"
  if [ ! -f "$BUILD/c4che/_cache.py" ] || [ "$force" = 1 ]; then
    CC=mpicc CXX=mpicxx CXXFLAGS="-fPIC" python3 ./waf configure \
      --elemental-dir="$DEPS_PREFIX" --mpsolve-dir="$DEPS_PREFIX"
  fi
  python3 ./waf build -j"$JOBS"
  echo "$sdpb_head" > "$STAMP"
  cd "$ROOT"
else
  echo "== SDPB libraries: up to date"
fi

# ---- tier 2: the Cython extension ------------------------------------------
so="$(ls "$PKG"/_sdpb*.so 2>/dev/null | head -1 || true)"
rebuild_ext=0
if [ "$rebuild_sdpb" = 1 ]; then
  rebuild_ext=1; why_ext="SDPB libraries rebuilt"
elif [ -z "$so" ]; then
  rebuild_ext=1; why_ext="extension not built (first install?)"
elif [ -n "$(find "$PKG/cpp" "$PKG"/_sdpb.pyx "$PKG"/sdpb_wrapper.pxd "$ROOT/setup.py" "$ROOT/pyproject.toml" -newer "$so" -type f 2>/dev/null | head -1)" ]; then
  rebuild_ext=1; why_ext="extension sources changed"
elif ! pip show -q sdpb-python 2>/dev/null; then
  rebuild_ext=1; why_ext="not installed in this environment"
fi

if [ "$rebuild_ext" = 1 ]; then
  echo "== extension: rebuilding and installing editable ($why_ext)"
  cd "$ROOT"
  CC=mpicxx CXX=mpicxx pip install --no-build-isolation -e .
else
  echo "== extension: up to date (Python changes need no rebuild)"
fi
echo "== done: $(python -c 'import sdpb_python as s; print(s.__version__, s.sdpb_version())' 2>/dev/null || echo 'import failed?')"
