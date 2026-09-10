"""Run a check function in a fresh interpreter at a given precision.

SDPB's Elemental fixes the precision once per process, so checks that must run
at a precision other than tests.util.datasets.TEST_PRECISION are executed in a
subprocess.  ``func_path`` is ``"module.function"``; the function performs its
own assertions and its (picklable) arguments are passed positionally.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def run_at_precision(bits: int, func_path: str, *args, timeout: int = 3600) -> str:
    module, func = func_path.rsplit(".", 1)
    code = (
        "import sdpb_python\n"
        f"sdpb_python.set_precision({int(bits)})\n"
        f"import {module} as m\n"
        f"m.{func}(*{args!r})\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO),
                                                                      os.environ.get("PYTHONPATH", "")]))}
    env.pop("DISPLAY", None)
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                          timeout=timeout, env=env)
    if proc.returncode != 0:
        raise AssertionError(f"subprocess at {bits} bits failed (exit {proc.returncode}):\n"
                             f"{proc.stderr[-6000:]}")
    return proc.stdout
