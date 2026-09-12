"""T7: process-level behaviour (MPI ranks, Ctrl-C, leaks, sympy)."""

import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import time

import pytest

from tests.util.datasets import DATA, TEST_PRECISION, STANDARD_ARGS
from tests.util.subproc import REPO

ONE_D = DATA / "1d" / "input" / "pmp.json"
ENV = {**os.environ, "PYTHONPATH": f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
ENV.pop("DISPLAY", None)


def _linked_mpi_family(path: str) -> str | None:
    """``"mpich"`` or ``"openmpi"`` for the MPI an ELF file links, ``None`` if unknown."""
    out = subprocess.run(["ldd", path], capture_output=True, text=True).stdout
    match = re.search(r"libmpi[-\w.]*\.so\.(\d+)", out)
    if match is None:
        return None
    # MPICH and the implementations sharing its ABI (Intel MPI, MVAPICH) are soname 12.
    return "mpich" if match.group(1) == "12" else "openmpi"


def _mpirun_family() -> str | None:
    """Which MPI the ``mpirun`` on PATH belongs to, or ``None`` if unrecognised."""
    proc = subprocess.run(["mpirun", "--version"], capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    if "Open MPI" in text or "OpenRTE" in text:
        return "openmpi"
    if "HYDRA" in text or "MPICH" in text or "Intel(R) MPI" in text:
        return "mpich"
    return None


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not available")
def test_two_ranks_rejected(sdpb_ext):
    # The wheels bundle their own MPI (MPICH).  Launched by a foreign mpirun, each
    # process joins no job and runs as an independent singleton, so it solves instead
    # of seeing a second rank; that is the documented limitation, not a failure.
    linked, launcher = _linked_mpi_family(sdpb_ext.__file__), _mpirun_family()
    if linked and launcher and linked != launcher:
        pytest.skip(f"package links {linked}, mpirun is {launcher}: "
                    "ranks would run as independent singletons")
    code = (
        "import sdpb_python\n"
        f"pmp = sdpb_python.read_pmp_json({str(ONE_D)!r})\n"
        "try:\n"
        f"    pmp.solve(precision={TEST_PRECISION}, max_iterations=1)\n"
        "except sdpb_python.SDPBError as e:\n"
        "    print('REJECTED:', e); raise SystemExit(3)\n"
        "print('SOLVED')\n"
    )
    version = subprocess.run(["mpirun", "--version"], capture_output=True, text=True).stdout
    oversubscribe = ["--oversubscribe"] if "Open MPI" in version else []
    proc = subprocess.run(["mpirun", *oversubscribe, "-n", "2", sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=300, env=ENV, cwd=REPO)
    assert "REJECTED: sdpb_python supports a single MPI rank" in proc.stdout, proc.stdout + proc.stderr[-2000:]
    assert "SOLVED" not in proc.stdout


def test_sigint_stops_cleanly(sdpb_ext):
    """Ctrl-C during a run -> SolverInterrupted with the partial solution; the solver can continue."""
    slow = DATA / "SingletScalar_cT_test_nmax6" / "primal_dual_optimal" / "input" / "pmp.nsv"
    code = (
        "import sdpb_python, sys, time\n"
        f"pmp = sdpb_python.read_pmp_json({str(slow)!r})\n"
        f"solver = pmp.solver(precision={TEST_PRECISION}, **{STANDARD_ARGS!r})\n"
        "print('READY', flush=True)\n"
        "try:\n"
        "    solver.run()\n"
        "except sdpb_python.SolverInterrupted as e:\n"
        "    print('INTERRUPTED', e.solution.status.value, e.solution.iterations, flush=True)\n"
        "    cont = solver.run(max_iterations=1)\n"
        "    print('CONTINUED', cont.status.value, solver.total_iterations, flush=True)\n"
        "    raise SystemExit(0)\n"
        "print('NOT INTERRUPTED'); raise SystemExit(2)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=ENV, cwd=REPO)
    assert proc.stdout.readline().strip() == "READY"
    time.sleep(15)  # let a few iterations run
    proc.send_signal(signal.SIGINT)
    out, err = proc.communicate(timeout=600)
    assert proc.returncode == 0, out + err[-3000:]
    lines = [l for l in out.splitlines() if l.startswith(("INTERRUPTED", "CONTINUED"))]
    assert lines[0].startswith("INTERRUPTED SIGTERM signal received"), out
    assert lines[1].startswith("CONTINUED maxIterations exceeded"), out


def test_no_leak_over_repeated_solves(sdpb_ext):
    from sdpb_python import read_pmp_json

    pmp = read_pmp_json(ONE_D)
    args = dict(STANDARD_ARGS, want=("x", "y", "z", "X", "Y", "c_minus_By"))
    for _ in range(10):
        pmp.solve(**args)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(60):
        pmp.solve(**args)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert after - before < 30 * 1024, f"RSS grew by {(after - before) / 1024:.1f} MB over 60 solves"


def test_polynomial_from_sympy():
    sympy = pytest.importorskip("sympy")
    from fractions import Fraction

    from sdpb_python import Polynomial

    x = sympy.Symbol("x")
    p = Polynomial.from_sympy(sympy.Rational(1, 12) * x**4 + x**2 + 3)
    assert p.coeffs == (Fraction(3), Fraction(0), Fraction(1), Fraction(0), Fraction(1, 12))
    q = Polynomial.from_sympy((x + 1) ** 2)
    assert q.coeffs == (1, 2, 1)
    r = Polynomial.from_sympy(sympy.sqrt(2) * x)
    assert r.coeffs[0] == 0 and str(r.coeffs[1]).startswith("1.41421356")
    with pytest.raises(ValueError):
        Polynomial.from_sympy(x * sympy.Symbol("y"))
