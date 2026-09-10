"""Linear matrix inequalities: ``maximize f + b.y  s.t.  M_0 + sum_n y_n M_n >= 0``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from . import numbers as _num
from .errors import wrap_cpp_error
from .options import SolverOptions, effective_precision, resolve
from .solution import Solution


def _ext():
    from . import _sdpb

    return _sdpb


@dataclass
class LMI:
    """Block-diagonal linear matrix inequality.

    ``blocks[j]`` is the sequence ``(M_0, M_1, ..., M_N)`` of symmetric matrices for
    block ``j`` (nested lists, numpy arrays or ``mpmath.matrix``).  To minimise,
    negate ``b`` and ``f``.
    """

    b: Sequence[Any]
    blocks: Sequence[Sequence[Any]]
    f: Any = 0

    def __post_init__(self):
        self.b = list(self.b)
        self.blocks = [list(block) for block in self.blocks]
        if not self.blocks:
            raise ValueError("LMI needs at least one block")
        n = len(self.b) + 1
        for j, block in enumerate(self.blocks):
            if len(block) != n:
                raise ValueError(f"blocks[{j}] has {len(block)} matrices, expected N+1 = {n}")

    def _to_spec(self, bits: int) -> dict:
        blocks = []
        for j, block in enumerate(self.blocks):
            mats = [_num.matrix_strs(m, bits) for m in block]
            d = len(mats[0])
            for n, m in enumerate(mats):
                if len(m) != d or any(len(row) != d for row in m):
                    raise ValueError(f"blocks[{j}][{n}] must be {d}x{d}")
                for r in range(d):
                    for s in range(r + 1, d):
                        if m[r][s] != m[s][r]:
                            raise ValueError(f"blocks[{j}][{n}] is not symmetric at ({r},{s})")
            blocks.append(mats)
        return {"f": _num.to_str(self.f, bits), "b": _num.strs(self.b, bits), "blocks": blocks}

    def solve(self, options: SolverOptions | None = None, **overrides: Any) -> Solution:
        opts = resolve(options, overrides)
        try:
            d = _ext().solve_lmi(self._to_spec(effective_precision(opts.precision)), opts._to_dict())
        except Exception as exc:
            raise wrap_cpp_error(exc) from None
        return Solution._from_dict(d)
