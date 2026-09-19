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


def _own_listening_sockets() -> list[str]:
    """TCP sockets this process holds in LISTEN state, from /proc (no external tools)."""
    inodes = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            match = re.match(r"socket:\[(\d+)\]", os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            continue
        if match:
            inodes.add(match.group(1))
    found = []
    for table in ("tcp", "tcp6"):
        try:
            with open(f"/proc/net/{table}") as f:
                rows = f.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            if fields[3] == "0A" and fields[9] in inodes:  # 0A = TCP_LISTEN
                address, port = fields[1].rsplit(":", 1)
                found.append(f"{table} {address}:{int(port, 16)}")
    return found


def _bundles_its_mpi(path: str) -> bool:
    """True for a wheel, whose MPI lives in ``sdpb_python.libs`` next to the package."""
    out = subprocess.run(["ldd", path], capture_output=True, text=True).stdout
    return any("libmpi" in line and "sdpb_python.libs" in line for line in out.splitlines())


def test_wheel_opens_no_network_listener(sdpb_ext):
    """The bundled MPI must not listen on the network (bug report against v0.2.1).

    Built with its default tcp network module, MPICH opens a listener on
    0.0.0.0 in MPI_Init even for one process, and aborts the process when
    anything that is not MPICH connects to it, e.g. a port scan on a shared
    server.  The wheels therefore bundle an MPICH without a network module.
    A source build links the system's MPI, which this package cannot control.
    """
    if not _bundles_its_mpi(sdpb_ext.__file__):
        pytest.skip("source build against a system MPI; only the wheels' bundled MPI is checked")
    sdpb_ext.initialize()
    from sdpb_python import LMI

    LMI(b=[1], blocks=[([[1, 0], [0, 1]], [[0, 1], [1, 0]])]).solve()
    assert _own_listening_sockets() == []


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
