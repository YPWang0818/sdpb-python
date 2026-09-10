"""Results of a solve."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import mpmath

from .numbers import from_str, matrix_mpfs, mpfs


class TerminateReason(enum.Enum):
    """Why SDPB stopped (``SDP_Solver_Terminate_Reason``)."""

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
    SIGTERM_RECEIVED = "SIGTERM received"

    @classmethod
    def from_sdpb(cls, text: str) -> "TerminateReason":
        for member in cls:
            if member.value == text:
                return member
        raise ValueError(f"unknown terminate reason {text!r}")


@dataclass
class BlockInfo:
    dim: int
    num_points: int

    @property
    def schur_size(self) -> int:
        return self.num_points * self.dim * (self.dim + 1) // 2


@dataclass
class Solution:
    """Solver output.  Numbers are ``mpmath.mpf`` at ``precision`` bits."""

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
