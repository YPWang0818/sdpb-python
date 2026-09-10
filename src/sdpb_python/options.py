"""Solver options mirroring SDPB's Solver_Parameters."""

from __future__ import annotations

import dataclasses
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterable

from .numbers import to_str

VERBOSITY = {"none": 0, "regular": 1, "debug": 2, "trace": 3}
SOLUTION_PARTS = ("x", "y", "z", "X", "Y", "c_minus_By")
_INT64_MAX = 2**63 - 1


@dataclass
class SolverOptions:
    """Solver options.  Defaults are SDPB's own.

    Names follow ``sdpb``'s command line options in snake_case, e.g.
    ``duality_gap_threshold`` for ``--dualityGapThreshold``.  Numeric
    thresholds accept anything :mod:`sdpb_python.numbers` converts
    (``int``, ``float``, ``str``, ``Fraction``, ``mpf``); strings such as
    ``"1e-30"`` are the safest way to write tiny numbers.

    Attributes:
        precision: Working precision in bits (``--precision``).  ``None`` means
            the precision already fixed for this process, or SDPB's default
            400 if none is fixed yet.  See :ref:`precision`.
        max_iterations: Stop after this many iterations (``--maxIterations``, 500).
        max_runtime: Stop after this many seconds (``--maxRuntime``, unlimited).
        duality_gap_threshold: Optimality test on the normalised duality gap
            ``|primal - dual| / max(|primal| + |dual|, 1)`` (1e-30).
        primal_error_threshold: Optimality test on the primal residues (1e-30).
        dual_error_threshold: Optimality test on the dual residues (1e-30).
        initial_matrix_scale_primal: ``X`` starts as this multiple of the
            identity (``--initialMatrixScalePrimal``, 1e20).
        initial_matrix_scale_dual: Same for ``Y`` (1e20).
        feasible_centering_parameter: Centering parameter ``beta`` used once
            primal and dual are feasible (0.1).
        infeasible_centering_parameter: Centering parameter while infeasible (0.3).
        step_length_reduction: Fraction of the maximal step taken (0.7).
        max_complementarity: Abort when ``Tr(XY)/dim`` exceeds this (1e100).
        min_primal_step: Stop when the primal step length falls below this (0).
        min_dual_step: Stop when the dual step length falls below this (0).
        find_primal_feasible: Stop as soon as a primal feasible point is found.
        find_dual_feasible: Stop as soon as a dual feasible point is found.
        detect_primal_feasible_jump: Stop when the primal step length reaches 1.
        detect_dual_feasible_jump: Stop when the dual step length reaches 1.
        max_shared_memory_bytes: Memory budget for the exact ``Q`` computation
            (``--maxSharedMemory``); ``0`` lets SDPB choose about half of the
            free RAM.  Accepts ints or strings such as ``"100K"`` / ``"64G"``.
        checkpoint_dir: Directory for SDPB's binary checkpoints.  When set, a
            checkpoint found there is loaded before solving and one is written
            at the end and every ``checkpoint_interval`` seconds.  ``None``
            disables checkpointing.
        checkpoint_interval: Seconds between checkpoints (3600).
        output_dir: If set, SDPB writes ``iterations.json`` (one record per
            iteration) and ``c_minus_By/c_minus_By.json`` there, as the
            ``sdpb`` executable does in its ``--outDir``.
        verbosity: ``"none"`` (default), ``"regular"``, ``"debug"`` or
            ``"trace"`` (or 0-3).  Anything above ``"none"`` prints SDPB's
            iteration table to the C++ standard output.
        want: Which parts of the solution to return besides the objectives and
            errors: any of ``"x"``, ``"y"``, ``"z"``, ``"X"``, ``"Y"``,
            ``"c_minus_By"``.  Default ``("y", "z")``.  Large problems have
            large ``X``/``Y``; ask only for what you need.
    """

    precision: int | None = None  # None: the process precision if fixed, else SDPB's 400
    max_iterations: int = 500
    max_runtime: int = _INT64_MAX
    duality_gap_threshold: Any = "1e-30"
    primal_error_threshold: Any = "1e-30"
    dual_error_threshold: Any = "1e-30"
    initial_matrix_scale_primal: Any = "1e20"
    initial_matrix_scale_dual: Any = "1e20"
    feasible_centering_parameter: Any = "0.1"
    infeasible_centering_parameter: Any = "0.3"
    step_length_reduction: Any = "0.7"
    max_complementarity: Any = "1e100"
    min_primal_step: Any = "0"
    min_dual_step: Any = "0"
    find_primal_feasible: bool = False
    find_dual_feasible: bool = False
    detect_primal_feasible_jump: bool = False
    detect_dual_feasible_jump: bool = False
    max_shared_memory_bytes: int = 0
    checkpoint_dir: str | os.PathLike | None = None
    checkpoint_interval: int = 3600
    output_dir: str | os.PathLike | None = None
    verbosity: str | int = "none"
    want: Iterable[str] = ("y", "z")

    def replace(self, **overrides: Any) -> "SolverOptions":
        unknown = set(overrides) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise TypeError(f"unknown solver option(s): {sorted(unknown)}")
        return dataclasses.replace(self, **overrides)

    def _to_dict(self) -> dict:
        """The dict consumed by the Cython layer."""
        bits = effective_precision(self.precision)
        if isinstance(self.verbosity, str):
            verbosity = VERBOSITY[self.verbosity]
        else:
            verbosity = int(self.verbosity)
        want = set(self.want)
        unknown = want - set(SOLUTION_PARTS)
        if unknown:
            raise ValueError(f"unknown solution part(s) {sorted(unknown)}; choose from {SOLUTION_PARTS}")
        if self.checkpoint_dir is None:
            # SDPB reads checkpoints from the CWD when checkpoint_in is empty,
            # so point it at a directory that does not exist.
            checkpoint_in = os.path.join(tempfile.gettempdir(), "sdpb_python_no_checkpoint")
            checkpoint_out = ""
        else:
            checkpoint_in = checkpoint_out = os.fspath(self.checkpoint_dir)
        return {
            "max_iterations": int(self.max_iterations),
            "max_runtime": int(self.max_runtime),
            "checkpoint_interval": int(self.checkpoint_interval),
            "max_shared_memory_bytes": _parse_bytes(self.max_shared_memory_bytes),
            "find_primal_feasible": bool(self.find_primal_feasible),
            "find_dual_feasible": bool(self.find_dual_feasible),
            "detect_primal_feasible_jump": bool(self.detect_primal_feasible_jump),
            "detect_dual_feasible_jump": bool(self.detect_dual_feasible_jump),
            "precision": bits,
            "duality_gap_threshold": to_str(self.duality_gap_threshold, bits),
            "primal_error_threshold": to_str(self.primal_error_threshold, bits),
            "dual_error_threshold": to_str(self.dual_error_threshold, bits),
            "initial_matrix_scale_primal": to_str(self.initial_matrix_scale_primal, bits),
            "initial_matrix_scale_dual": to_str(self.initial_matrix_scale_dual, bits),
            "feasible_centering_parameter": to_str(self.feasible_centering_parameter, bits),
            "infeasible_centering_parameter": to_str(self.infeasible_centering_parameter, bits),
            "step_length_reduction": to_str(self.step_length_reduction, bits),
            "max_complementarity": to_str(self.max_complementarity, bits),
            "min_primal_step": to_str(self.min_primal_step, bits),
            "min_dual_step": to_str(self.min_dual_step, bits),
            "checkpoint_in": checkpoint_in,
            "checkpoint_out": checkpoint_out,
            "output_dir": "" if self.output_dir is None else os.fspath(self.output_dir),
            "verbosity": verbosity,
            "want_x": "x" in want,
            "want_z": "z" in want,
            "want_X": "X" in want,
            "want_Y": "Y" in want,
            "want_c_minus_By": "c_minus_By" in want,
        }


_UNITS = {"": 1, "B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def _parse_bytes(value: Any) -> int:
    """Accept ints or SDPB-style strings such as ``"100.1K"`` or ``"64G"``."""
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().upper().rstrip("B")
    for unit, factor in _UNITS.items():
        if unit and s.endswith(unit):
            return int(float(s[: -len(unit)]) * factor)
    return int(float(s))


def effective_precision(precision: int | None) -> int:
    """SDPB's Elemental fixes the precision once per process (see docs/API_DESIGN.md)."""
    if precision is not None:
        return int(precision)
    from . import _sdpb  # noqa: WPS433

    fixed = _sdpb.requested_precision()
    return fixed if fixed else 400


def resolve(options: SolverOptions | None, overrides: dict) -> SolverOptions:
    base = SolverOptions() if options is None else options
    return base.replace(**overrides) if overrides else base
