"""Work around OpenBLAS choosing kernels the CPU cannot run.

OpenBLAS selects its kernels at run time from the CPU vendor and family.  For
an AMD family-15 or family-17 CPU it takes the OPTERON targets, whose GEMM
copy kernels begin with the 3DNow! instruction ``femms``.  Real CPUs of those
families have 3DNow!, but QEMU/KVM guests with the default CPU model report
such a family without it, and the first ``solve()`` then dies with SIGILL
inside libopenblas (bug report against v0.2.1).

OpenBLAS honours ``OPENBLAS_CORETYPE`` if it is set before the library loads,
so :func:`apply` sets it in that one situation, to the newest kernel set the
reported flags support.  It never overrides a value the user has set, and it
does nothing on other CPUs.  The wheels' own OpenBLAS is built without the
OPTERON targets, so this matters mostly for source builds against a system
OpenBLAS.
"""

from __future__ import annotations

import os
import platform
import sys

# cpu families (as /proc/cpuinfo reports them) that OpenBLAS maps to OPTERON
_OPTERON_FAMILIES = {15, 17}


def openblas_coretype_override(cpuinfo: str) -> str | None:
    """Core type to force for the CPU described by ``cpuinfo``, or ``None``."""
    vendor = family = None
    flags: set[str] = set()
    for line in cpuinfo.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "vendor_id" and vendor is None:
            vendor = value
        elif key == "cpu family" and family is None:
            try:
                family = int(value)
            except ValueError:
                return None
        elif key == "flags" and not flags:
            flags = set(value.split())
        if vendor is not None and family is not None and flags:
            break
    if vendor != "AuthenticAMD" or family not in _OPTERON_FAMILIES or "3dnow" in flags:
        return None
    if "sse4_2" in flags:
        return "NEHALEM"
    if "ssse3" in flags:
        return "CORE2"
    return "PRESCOTT"


def apply() -> str | None:
    """Set ``OPENBLAS_CORETYPE`` if this CPU needs it; return what was set."""
    if os.environ.get("OPENBLAS_CORETYPE") or sys.platform != "linux" \
            or platform.machine() != "x86_64":
        return None
    try:
        with open("/proc/cpuinfo", encoding="ascii", errors="replace") as f:
            cpuinfo = f.read()
    except OSError:
        return None
    coretype = openblas_coretype_override(cpuinfo)
    if coretype:
        os.environ["OPENBLAS_CORETYPE"] = coretype
    return coretype
