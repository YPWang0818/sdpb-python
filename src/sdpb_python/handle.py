"""A solver handle: keep SDPB's state alive between runs."""

from __future__ import annotations

import os
from typing import Any, Sequence

import mpmath

from . import numbers as _num
from .errors import SDPBError, wrap_cpp_error
from .options import SolverOptions, effective_precision, resolve
from .solution import Solution, TerminateReason


class SolverInterrupted(KeyboardInterrupt):
    """Ctrl-C during a run: SDPB stopped cleanly at the end of an iteration.

    ``solution`` holds the state at that point; the :class:`Solver` can be run
    again to continue.
    """

    def __init__(self, solution: Solution):
        super().__init__("SDPB run interrupted")
        self.solution = solution


def _ext():
    from . import _sdpb

    return _sdpb


class Solver:
    """Owns SDPB's `Block_Info`, `SDP` and `SDP_Solver` for one problem.

    Created by :meth:`PMP.solver` or :meth:`LMI.solver`.  :meth:`run` continues
    the interior-point iteration from the current ``x, X, y, Y`` and may be
    called repeatedly, e.g. with tighter thresholds.  Use as a context manager
    or call :meth:`close` to release the C++ objects.
    """

    def __init__(self, kind: str, spec: dict, options: SolverOptions):
        self._options = options
        self._precision = effective_precision(options.precision)
        try:
            self._impl = _ext().Solver(kind, spec, options._to_dict())
        except Exception as exc:
            raise wrap_cpp_error(exc) from None
        self.blocks_dims = self._impl.dims()
        self.blocks_num_points = self._impl.num_points()
        # status of the last run; "not converged" until run() is called
        self._last_status = TerminateReason.MAX_ITERATIONS_EXCEEDED

    # -- lifecycle
    def close(self) -> None:
        if getattr(self, "_impl", None) is not None:
            self._impl.close()

    @property
    def closed(self) -> bool:
        return self._impl.closed

    def __enter__(self) -> "Solver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- running
    @property
    def options(self) -> SolverOptions:
        return self._options

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def num_variables(self) -> int:
        return self._impl.num_variables()

    @property
    def total_iterations(self) -> int:
        """Iterations completed over all calls of :meth:`run`."""
        return self._impl.total_iterations()

    def run(self, options: SolverOptions | None = None, **overrides: Any) -> Solution:
        """Continue iterating.  Raises :class:`SolverInterrupted` on Ctrl-C."""
        opts = resolve(self._options if options is None else options, overrides)
        try:
            d = self._impl.run(opts._to_dict())
        except Exception as exc:
            raise wrap_cpp_error(exc) from None
        solution = Solution._from_dict(d)
        self._last_status = solution.status
        if solution.status is TerminateReason.SIGTERM_RECEIVED and _ext().last_run_interrupted():
            raise SolverInterrupted(solution)
        return solution

    def state(self, want: Sequence[str] = ("y", "z", "x", "X", "Y")) -> Solution:
        """The current iterate without running (objectives are those of the last run)."""
        opts = self._options.replace(want=tuple(want))
        try:
            d = self._impl.state(opts._to_dict())
        except Exception as exc:
            raise wrap_cpp_error(exc) from None
        d["terminate_reason"] = self._last_status.value
        return Solution._from_dict(d)

    # -- warm start
    def warm_start(self, y: Sequence[Any] | None = None, X: Sequence[Any] | None = None,
                   Y: Sequence[Any] | None = None) -> "Solver":
        """Overwrite the iterate before :meth:`run`.

        ``y`` has length N; ``X`` / ``Y`` are per-block pairs ``(even, odd)`` of
        matrices in the shapes SDPB uses (see ``Solution.X``).
        """
        bits = self._precision
        try:
            if y is not None:
                self._impl.set_y(_num.strs(y, bits))
            if X is not None:
                self._impl.set_X(_flatten_blocks(X, bits))
            if Y is not None:
                self._impl.set_Y(_flatten_blocks(Y, bits))
        except Exception as exc:
            raise wrap_cpp_error(exc) from None
        return self

    def save_checkpoint(self, directory: str | os.PathLike) -> None:
        """Write SDPB's binary checkpoint (``checkpoint_<g>_<rank>``, ``checkpoint.json``)."""
        try:
            self._impl.save_checkpoint(os.fspath(directory))
        except Exception as exc:
            raise wrap_cpp_error(exc) from None


def _flatten_blocks(blocks: Sequence[Any], bits: int) -> list[list[list[str]]]:
    out = []
    for pair in blocks:
        even, odd = pair
        out.append(_num.matrix_strs(even, bits))
        out.append(_num.matrix_strs(odd, bits))
    return out
