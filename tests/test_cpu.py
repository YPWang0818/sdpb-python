"""OpenBLAS core-type override for CPUs misreported as 3DNow!-less Opterons (v0.2.1 bug report)."""

import os

import pytest

from sdpb_python import _cpu

QEMU64 = """processor\t: 0
vendor_id\t: AuthenticAMD
cpu family\t: 15
model\t\t: 107
model name\t: QEMU Virtual CPU version 2.5+
flags\t\t: fpu de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 syscall nx lm nopl cpuid tsc_known_freq pni ssse3 cx16 sse4_1 sse4_2 x2apic popcnt aes hypervisor lahf_lm cmp_legacy 3dnowprefetch vmmcall
processor\t: 1
vendor_id\t: AuthenticAMD
cpu family\t: 15
flags\t\t: fpu sse sse2 pni ssse3 sse4_1 sse4_2 3dnowprefetch
"""


def test_qemu64_gets_nehalem():
    assert _cpu.openblas_coretype_override(QEMU64) == "NEHALEM"


def test_older_flags_step_down():
    assert _cpu.openblas_coretype_override(QEMU64.replace(" sse4_1 sse4_2", "")) == "CORE2"
    assert _cpu.openblas_coretype_override(QEMU64.replace(" ssse3", "").replace(" sse4_1 sse4_2", "")) == "PRESCOTT"


@pytest.mark.parametrize("variant", [
    QEMU64.replace("3dnowprefetch", "3dnow 3dnowext 3dnowprefetch"),   # real K8: has 3DNow!
    QEMU64.replace("cpu family\t: 15", "cpu family\t: 25"),           # Zen 3: no OPTERON kernels
    QEMU64.replace("AuthenticAMD", "GenuineIntel"),                    # Intel path is fine
    "",                                                                 # unreadable
])
def test_untouched(variant):
    assert _cpu.openblas_coretype_override(variant) is None


def test_apply_respects_user_setting(monkeypatch):
    monkeypatch.setenv("OPENBLAS_CORETYPE", "HASWELL")
    assert _cpu.apply() is None
    assert os.environ["OPENBLAS_CORETYPE"] == "HASWELL"
