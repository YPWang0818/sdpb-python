#!/usr/bin/env bash
# Fail if a built wheel bundles a library with AVX instructions baked in.
#
#   scripts/check_wheel_isa.sh wheelhouse/*.whl
#
# The manylinux tag only fixes the glibc version, not the CPU: a library
# compiled with -march=native on an AVX2 build host crashes with SIGILL on
# older CPUs and on QEMU/KVM guests using the default CPU model (this happened
# to libflint in v0.2.0).  Libraries that select their kernels at run time
# (OpenBLAS, OpenSSL, libgfortran's ifunc matmul) legitimately contain AVX code
# and are skipped.
set -euo pipefail

skip_re='^lib(openblas|crypto|ssl|gfortran)'
# VEX/EVEX-encoded mnemonics start with "v"; the letters after "v" rule out
# the SSE "vmovd"-style legacy names, which do not exist.
avx_re='^\s+[0-9a-f]+:\s+([0-9a-f]{2} )+\s*v[a-z0-9]+\s'

rc=0
for whl in "$@"; do
  tmp="$(mktemp -d)"
  unzip -qo "$whl" -d "$tmp"
  while IFS= read -r so; do
    base="$(basename "$so")"
    if [[ "$base" =~ $skip_re ]]; then continue; fi
    n="$(objdump -d "$so" | grep -c -E "$avx_re" || true)"
    if [ "$n" -gt 0 ]; then
      printf 'FAIL %s: %s has %d AVX instructions (e.g. %s)\n' "$(basename "$whl")" "$base" "$n" \
        "$(objdump -d "$so" | grep -m1 -E "$avx_re" | awk '{print $NF" "$(NF-1)}' | head -c 40)"
      rc=1
    fi
  done < <(find "$tmp" -name '*.so*' -type f)
  rm -rf "$tmp"
  [ "$rc" = 0 ] && echo "ok   $(basename "$whl"): no baked-in AVX in bundled libraries"
done
exit "$rc"
