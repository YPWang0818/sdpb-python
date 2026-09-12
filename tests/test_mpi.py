"""T8: a foreign MPI launcher is rejected before the work is duplicated.

The bindings run on one rank; the C++ side rejects a larger world.  That check
cannot fire when the launcher belongs to a different MPI than the package links
(the wheels bundle MPICH), because each process then initialises MPI alone and
sees a world of one.  Verified on a qemu64 guest: four Open MPI ranks running
the MPICH wheel each solved the same problem and left the shared output
directory with unparseable iterations files.
"""

import os
import subprocess
import sys

import pytest

from sdpb_python import SDPBError, _mpi
from tests.util.datasets import DATA, TEST_PRECISION
from tests.util.subproc import REPO

ONE_D = DATA / "1d" / "input" / "pmp.json"


# -- detection ---------------------------------------------------------------

@pytest.mark.parametrize("var,launcher", [
    ("OMPI_COMM_WORLD_SIZE", "Open MPI"),
    ("PMI_SIZE", "MPICH, Hydra or Intel MPI"),
    ("MV2_COMM_WORLD_SIZE", "MVAPICH"),
    ("SLURM_NTASKS", "Slurm"),
    ("SLURM_STEP_NUM_TASKS", "Slurm"),
])
def test_each_launcher_is_recognised(var, launcher):
    assert _mpi.launcher_job({var: "4"}) == (4, var, launcher)
    message = _mpi.multi_process_message(1, {var: "4"})
    assert launcher in message and var in message


@pytest.mark.parametrize("env", [
    {},                                  # no launcher
    {"OMPI_COMM_WORLD_SIZE": "1"},       # a one-process job is fine
    {"OMPI_COMM_WORLD_SIZE": "oops"},    # unusable value: do not guess
    {"PMIX_RANK": "1"},                  # a rank, not a size
])
def test_no_launcher_no_complaint(env):
    assert _mpi.launcher_job(env) is None
    assert _mpi.multi_process_message(1, env) is None


def test_real_world_is_left_to_the_cpp_check():
    """With a matching launcher the ranks see each other; SDPB reports that itself."""
    assert _mpi.multi_process_message(2, {"OMPI_COMM_WORLD_SIZE": "2"}) is None


def test_opt_out():
    env = {"OMPI_COMM_WORLD_SIZE": "4", _mpi.OPT_OUT: "1"}
    assert _mpi.multi_process_message(1, env) is None


def test_check_raises_sdpb_error():
    with pytest.raises(SDPBError, match="single MPI rank"):
        _mpi.check_single_process(1, {"OMPI_COMM_WORLD_SIZE": "2"})
    _mpi.check_single_process(1, {})  # no launcher: returns quietly


# -- the solve paths honour it ------------------------------------------------

SOLVE_SNIPPETS = {
    "pmp": f"sdpb_python.read_pmp_json({str(ONE_D)!r}).solve(precision={TEST_PRECISION}, max_iterations=1)",
    "solver_handle": f"sdpb_python.read_pmp_json({str(ONE_D)!r}).solver(precision={TEST_PRECISION})",
    "lmi": ("sdpb_python.LMI(b=[1, 1], f=3, blocks=[([[1, 0], [0, 1]], [[-1, 0], [0, 1]], "
            f"[[0, 0], [0, -1]])]).solve(precision={TEST_PRECISION}, max_iterations=1)"),
}


def _run(snippet: str, extra_env: dict) -> subprocess.CompletedProcess:
    code = ("import sdpb_python\n"
            "try:\n"
            f"    {snippet}\n"
            "except sdpb_python.SDPBError as e:\n"
            "    print('REJECTED:', e); raise SystemExit(3)\n"
            "print('RAN')\n")
    env = {**os.environ, "PYTHONPATH": f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}", **extra_env}
    env.pop("DISPLAY", None)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=600, env=env, cwd=REPO)


@pytest.mark.parametrize("name", sorted(SOLVE_SNIPPETS))
def test_foreign_launcher_is_rejected(sdpb_ext, name):
    """A launcher advertising several processes while MPI sees one must not run."""
    proc = _run(SOLVE_SNIPPETS[name], {"OMPI_COMM_WORLD_SIZE": "4"})
    assert "REJECTED: sdpb_python supports a single MPI rank" in proc.stdout, \
        proc.stdout + proc.stderr[-2000:]
    assert "Open MPI started 4 processes" in proc.stdout
    assert "RAN" not in proc.stdout


def test_opt_out_lets_it_run(sdpb_ext):
    proc = _run(SOLVE_SNIPPETS["pmp"], {"OMPI_COMM_WORLD_SIZE": "4", _mpi.OPT_OUT: "1"})
    assert "RAN" in proc.stdout, proc.stdout + proc.stderr[-2000:]


def test_plain_run_is_unaffected(sdpb_ext):
    proc = _run(SOLVE_SNIPPETS["pmp"], {})
    assert "RAN" in proc.stdout, proc.stdout + proc.stderr[-2000:]
