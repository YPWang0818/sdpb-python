# distutils: language = c++
"""Low-level Cython bindings to SDPB.  Use :mod:`sdpb_python` instead."""

from libcpp.string cimport string
from libcpp.vector cimport vector

from . cimport sdpb_wrapper as cw


def version() -> str:
    """Return the git version string of the bundled SDPB build."""
    return cw.version().decode()


def initialize() -> None:
    """Initialise MPI/Elemental.  Idempotent; called automatically by ``run``."""
    cw.initialize()


def finalize() -> None:
    """Shut down MPI/Elemental.  Call at most once, at the end of the program."""
    cw.finalize()


def mpi_rank() -> int:
    return cw.mpi_rank()


def mpi_size() -> int:
    return cw.mpi_size()


def run(args) -> None:
    """Run the SDPB solver with ``sdpb``-style command line arguments.

    ``args`` is an iterable of strings, e.g. ``["--sdpDir", "sdp", "--outDir", "out"]``.
    """
    cdef vector[string] cargs
    for a in args:
        cargs.push_back(str(a).encode())
    cw.run(cargs)
