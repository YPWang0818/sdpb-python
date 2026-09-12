"""Catch an MPI launcher that this process's own MPI did not join.

The bindings run SDPB on a single rank and reject a larger world (the C++ side
raises when ``MPI_Comm_size`` is above one).  That check only works when the
launcher and the linked MPI are the same implementation.  The wheels bundle
MPICH, so ``mpirun`` from Open MPI (or ``srun`` against a different MPI) starts
several processes that each initialise MPI alone: every one sees a world of
one, solves the whole problem, and writes over the others' output files.

The launcher still describes the job in the environment, whatever MPI it is,
so :func:`check_single_process` compares that with the world the extension
reports and raises before any work is duplicated.  Set
``SDPB_PYTHON_ALLOW_MULTI_PROCESS=1`` to allow it deliberately, e.g. a
parameter sweep where each process writes somewhere of its own.
"""

from __future__ import annotations

import os
from typing import Mapping

from .errors import SDPBError

OPT_OUT = "SDPB_PYTHON_ALLOW_MULTI_PROCESS"

# Size of the job as each launcher advertises it, in the order they are checked.
_SIZE_VARS = (
    ("OMPI_COMM_WORLD_SIZE", "Open MPI"),
    ("PMI_SIZE", "MPICH, Hydra or Intel MPI"),
    ("MV2_COMM_WORLD_SIZE", "MVAPICH"),
    ("SLURM_STEP_NUM_TASKS", "Slurm"),
    ("SLURM_NTASKS", "Slurm"),
)


def launcher_job(env: Mapping[str, str] | None = None) -> tuple[int, str, str] | None:
    """``(processes, variable, launcher)`` for a job of more than one process.

    ``None`` when no launcher advertises one, or when the value is unusable.
    """
    environ = os.environ if env is None else env
    for var, launcher in _SIZE_VARS:
        value = environ.get(var)
        if value is None:
            continue
        try:
            size = int(value)
        except ValueError:
            continue
        if size > 1:
            return size, var, launcher
    return None


def multi_process_message(mpi_size: int, env: Mapping[str, str] | None = None) -> str | None:
    """The error to raise for a launcher this process's MPI ignored, else ``None``."""
    environ = os.environ if env is None else env
    if environ.get(OPT_OUT):
        return None
    # A world larger than one is the C++ side's to reject; it has the real ranks.
    if mpi_size != 1:
        return None
    job = launcher_job(environ)
    if job is None:
        return None
    size, var, launcher = job
    return (
        f"sdpb_python supports a single MPI rank. This process sees a world of one, "
        f"but {launcher} started {size} processes ({var}={size}), so the launcher and "
        f"the MPI this package links are different implementations and none of the "
        f"processes can see the others. Each would solve the whole problem and they "
        f"would overwrite each other's output. Run a single process, use a launcher "
        f"from the same MPI the package links, or set {OPT_OUT}=1 if the processes "
        f"are meant to be independent."
    )


def check_single_process(mpi_size: int, env: Mapping[str, str] | None = None) -> None:
    """Raise :class:`SDPBError` when a foreign launcher started several processes."""
    message = multi_process_message(mpi_size, env)
    if message is not None:
        raise SDPBError(message)
