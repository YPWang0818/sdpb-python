"""Results of a solve."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import mpmath

from .numbers import from_str, matrix_mpfs, mpfs


class TerminateReason(enum.Enum):
    """Why SDPB stopped (``SDP_Solver_Terminate_Reason``).

    The value of each member is the exact text SDPB writes to ``out.txt``.
    Only :attr:`PRIMAL_DUAL_OPTIMAL` means the problem was solved to the
    requested thresholds; the ``*_FEASIBLE`` and ``*_JUMP_DETECTED`` reasons
    are early stops requested through :class:`~sdpb_python.SolverOptions`.
    """

    PRIMAL_DUAL_OPTIMAL = "found primal-dual optimal solution"
    PRIMAL_FEASIBLE = "found primal feasible solution"
    DUAL_FEASIBLE = "found dual feasible solution"
    PRIMAL_FEASIBLE_JUMP_DETECTED = "primal feasible jump detected"
    DUAL_FEASIBLE_JUMP_DETECTED = "dual feasible jump detected"
    MAX_COMPLEMENTARITY_EXCEEDED = "maxComplementarity exceeded"
    MAX_ITERATIONS_EXCEEDED = "maxIterations exceeded"
    MAX_RUNTIME_EXCEEDED = "maxRuntime exceeded"
    PRIMAL_STEP_TOO_SMALL = "primal step too small"
    DUAL_STEP_TOO_SMALL = "dual step too small"
    SIGTERM_RECEIVED = "SIGTERM signal received"

    @classmethod
    def from_sdpb(cls, text: str) -> "TerminateReason":
        for member in cls:
            if member.value == text:
                return member
        raise ValueError(f"unknown terminate reason {text!r}")


@dataclass
class BlockInfo:
    """Size of one SDP block.

    Attributes:
        dim: Matrix dimension of the constraint (``m_j``).
        num_points: Number of sample points (``d_j + 1``); 1 for an LMI block.
    """

    dim: int
    num_points: int

    @property
    def schur_size(self) -> int:
        return self.num_points * self.dim * (self.dim + 1) // 2


@dataclass
class Solution:
    """Solver output.  Numbers are ``mpmath.mpf`` at ``precision`` bits.

    See :doc:`/background` for the meaning of the variables.

    Attributes:
        status: Why the solver stopped; :attr:`optimal` is the usual check.
        primal_objective: ``f + c . x``.
        dual_objective: ``f + b . y``; for a PMP this is the optimal value of
            the objective ``a . z``.
        duality_gap: ``|primal - dual| / max(|primal| + |dual|, 1)``.
        primal_error: Largest primal residue.
        dual_error: Largest dual residue.
        y: The dual variables, length ``N``.
        z: The PMP variables of Manual eq. (3.1), length ``N + 1``, when the
            problem has a normalization and ``"z"`` was requested; else ``None``.
        x: Per block, the primal variables (length ``num_points * dim * (dim+1) / 2``);
            ``None`` unless ``"x"`` was requested.
        X: Per block, the pair ``(even, odd)`` of positive semidefinite primal
            matrices; ``None`` unless ``"X"`` was requested.
        Y: Per block, the pair ``(even, odd)`` of positive semidefinite dual
            matrices; for an LMI block ``Y[j][0]`` is ``M_0 + sum_n y_n M_n``.
        c_minus_By: Per block, ``c - B y``: the extremal functional evaluated on
            the sampled constraints (what ``spectrum`` consumes).
        iterations: Iterations completed by this run.
        runtime_seconds: Wall time of the run.
        precision: Bits of precision used.
        blocks: :class:`BlockInfo` per block.
    """

    status: TerminateReason
    primal_objective: mpmath.mpf
    dual_objective: mpmath.mpf
    duality_gap: mpmath.mpf
    primal_error: mpmath.mpf
    dual_error: mpmath.mpf
    y: list[mpmath.mpf]
    z: list[mpmath.mpf] | None
    x: list[list[mpmath.mpf]] | None
    X: list[tuple[mpmath.matrix, mpmath.matrix]] | None
    Y: list[tuple[mpmath.matrix, mpmath.matrix]] | None
    c_minus_By: list[list[mpmath.mpf]] | None
    iterations: int
    runtime_seconds: float
    precision: int
    blocks: list[BlockInfo] = field(default_factory=list)

    @property
    def optimal(self) -> bool:
        return self.status is TerminateReason.PRIMAL_DUAL_OPTIMAL

    @classmethod
    def _from_dict(cls, d: dict) -> "Solution":
        bits = d["precision"]
        num_blocks = len(d["dims"])

        def pairs(mats):
            if not mats:
                return None
            return [(matrix_mpfs(mats[2 * j], bits), matrix_mpfs(mats[2 * j + 1], bits))
                    for j in range(num_blocks)]

        return cls(
            status=TerminateReason.from_sdpb(d["terminate_reason"]),
            primal_objective=from_str(d["primal_objective"], bits),
            dual_objective=from_str(d["dual_objective"], bits),
            duality_gap=from_str(d["duality_gap"], bits),
            primal_error=from_str(d["primal_error"], bits),
            dual_error=from_str(d["dual_error"], bits),
            y=mpfs(d["y"], bits),
            z=mpfs(d["z"], bits) if d["z"] is not None else None,
            x=[mpfs(block, bits) for block in d["x"]] if d["x"] else None,
            X=pairs(d["X"]),
            Y=pairs(d["Y"]),
            c_minus_By=[mpfs(block, bits) for block in d["c_minus_By"]] if d["c_minus_By"] else None,
            iterations=d["iterations"],
            runtime_seconds=d["runtime_ms"] / 1000.0,
            precision=bits,
            blocks=[BlockInfo(dim, n) for dim, n in zip(d["dims"], d["num_points"])],
        )
