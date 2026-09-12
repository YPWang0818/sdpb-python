"""Python bindings for a fork of SDPB (https://github.com/YPWang0818/sdpb)."""

from . import _cpu

# Must run before the extension (and with it libopenblas) is first loaded.
_cpu.apply()

from .errors import SDPBError  # noqa: E402
from .handle import Solver, SolverInterrupted
from .io import read_pmp_json, write_pmp_json
from .lmi import LMI
from .options import SolverOptions
from .problem import PMP, DampedRational, Polynomial, PolynomialMatrix, SDPData, SampledMatrix
from .solution import BlockInfo, Solution, TerminateReason
from .solver import SDPBResult, sdpb_version
from .solver import solve as solve_dir


def set_precision(bits: int) -> None:
    """Fix the working precision (bits) for this process.

    SDPB's Elemental allows setting it once per process; calling again with a
    different value raises :class:`SDPBError`.  When not called, the first solve
    fixes it (to its ``precision`` option, default 400).
    """
    from . import _sdpb

    try:
        _sdpb.set_precision(int(bits))
    except RuntimeError as exc:
        raise SDPBError(str(exc)) from None


def precision() -> int:
    """Precision (bits) fixed for this process, or 0 if not fixed yet."""
    from . import _sdpb

    return _sdpb.requested_precision()

__version__ = "0.2.1"
__all__ = [
    "PMP", "PolynomialMatrix", "Polynomial", "DampedRational", "LMI",
    "SolverOptions", "Solution", "TerminateReason", "BlockInfo", "SDPData", "SampledMatrix",
    "SDPBError", "Solver", "SolverInterrupted", "read_pmp_json", "write_pmp_json",
    "sdpb_version", "solve_dir", "SDPBResult", "set_precision", "precision", "__version__",
]
