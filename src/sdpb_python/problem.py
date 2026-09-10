"""Polynomial matrix programs (PMP), the input format of SDPB.

Manual eq. (3.1): maximize ``a . z`` over ``z`` such that ``n . z = 1`` and
``sum_n z_n M_n^j(x) >= 0`` for all ``x >= 0`` and every block ``j``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import mpmath

from . import numbers as _num
from .errors import wrap_cpp_error
from .options import SolverOptions, effective_precision, resolve
from .solution import Solution

NumberLike = Any


def _ext():
    from . import _sdpb  # noqa: WPS433 (lazy: extension may be missing)

    return _sdpb


# ------------------------------------------------------------------ pieces


@dataclass
class DampedRational:
    """``constant * base**x / prod_i (x - poles[i])``.

    SDPB's default prefactor is ``exp(-x)``: ``DampedRational(base=math.exp(-1))``.
    """

    constant: NumberLike = 1
    base: NumberLike = 1
    poles: Sequence[NumberLike] = ()

    @classmethod
    def exp_minus_x(cls, poles: Sequence[NumberLike] = ()) -> "DampedRational":
        # Full-precision e^-1 is computed on the C++ side when omitted; when
        # given explicitly we render it at 400+ bits via mpmath.
        with mpmath.workprec(4096):
            return cls(1, mpmath.exp(-1), poles)

    def _to_spec(self, bits: int) -> dict:
        return {
            "constant": _num.to_str(self.constant, bits),
            "base": _num.to_str(self.base, bits),
            "poles": _num.strs(self.poles, bits),
        }

    @classmethod
    def _from_spec(cls, d: dict, bits: int) -> "DampedRational":
        return cls(_num.from_str(d["constant"], bits), _num.from_str(d["base"], bits),
                   _num.mpfs(d["poles"], bits))


@dataclass
class Polynomial:
    """``a_0 + a_1 x + ... + a_n x^n`` given by ``coeffs = [a_0, ..., a_n]``."""

    coeffs: Sequence[NumberLike] = (0,)

    def __post_init__(self):
        if isinstance(self.coeffs, Polynomial):
            self.coeffs = self.coeffs.coeffs
        elif isinstance(self.coeffs, (str, bytes)) or not hasattr(self.coeffs, "__iter__"):
            self.coeffs = (self.coeffs,)  # a constant
        self.coeffs = tuple(self.coeffs) or (0,)

    @property
    def degree(self) -> int:
        return len(self.coeffs) - 1

    def __call__(self, x):
        result = 0
        for a in reversed(self.coeffs):
            result = result * x + a
        return result

    def _to_spec(self, bits: int) -> list[str]:
        return _num.strs(self.coeffs, bits)


def _is_negative(x: Any) -> bool:
    try:
        return mpmath.mpf(str(x).strip()) < 0
    except Exception:
        return False


def _as_polynomial(p: Any) -> Polynomial:
    return p if isinstance(p, Polynomial) else Polynomial(p)


@dataclass
class PolynomialMatrix:
    """One positivity constraint: ``sum_n z_n P^{rs}_n(x)`` is PSD for ``x >= 0``.

    ``polynomials[r][s]`` is the list of ``N+1`` polynomials ``P^{rs}_0..P^{rs}_N``
    (each a :class:`Polynomial` or a coefficient list); the matrix must be symmetric.
    Optional fields mirror ``pmp.json`` (see ``docs/SDPB_input_format.md`` in SDPB);
    anything omitted is computed by SDPB.
    """

    polynomials: Sequence[Sequence[Sequence[Any]]]
    prefactor: DampedRational | None = None
    reduced_prefactor: DampedRational | None = None
    max_num_poles: int | None = None
    sample_points: Sequence[NumberLike] | None = None
    sample_scalings: Sequence[NumberLike] | None = None
    reduced_sample_scalings: Sequence[NumberLike] | None = None
    bilinear_basis: tuple[Sequence[Any], Sequence[Any]] | None = None
    label: str = ""

    def __post_init__(self):
        rows = [[[_as_polynomial(p) for p in entry] for entry in row] for row in self.polynomials]
        dim = len(rows)
        if dim == 0 or any(len(row) != dim for row in rows):
            raise ValueError("polynomials must be a square dim x dim matrix")
        lengths = {len(entry) for row in rows for entry in row}
        if len(lengths) != 1:
            raise ValueError(f"every entry must hold the same number of polynomials, got {sorted(lengths)}")
        for r in range(dim):
            for s in range(r + 1, dim):
                if [p.coeffs for p in rows[r][s]] != [p.coeffs for p in rows[s][r]]:
                    raise ValueError(f"polynomials must be symmetric, entries ({r},{s}) and ({s},{r}) differ")
        self.polynomials = rows
        if self.sample_points is not None:
            self.sample_points = list(self.sample_points)
            for i, x in enumerate(self.sample_points):
                if _is_negative(x):
                    raise ValueError(f"sample_points[{i}] = {x} is negative; SDPB samples x >= 0 only")
        for name in ("sample_scalings", "reduced_sample_scalings"):
            values = getattr(self, name)
            if values is not None:
                setattr(self, name, list(values))
                if self.sample_points is not None and len(values) != len(self.sample_points):
                    raise ValueError(f"{name} has {len(values)} entries but there are "
                                     f"{len(self.sample_points)} sample points")
        if self.bilinear_basis is not None:
            even, odd = self.bilinear_basis
            self.bilinear_basis = ([_as_polynomial(p) for p in even], [_as_polynomial(p) for p in odd])

    @property
    def dim(self) -> int:
        return len(self.polynomials)

    @property
    def num_vectors(self) -> int:
        """``N + 1``: number of polynomials per entry."""
        return len(self.polynomials[0][0])

    @property
    def max_degree(self) -> int:
        return max(p.degree for row in self.polynomials for entry in row for p in entry)

    def _to_spec(self, bits: int, max_num_poles: int | None = None) -> dict:
        num_poles = _combine_max_num_poles(self.max_num_poles, max_num_poles)
        return {
            "dim": self.dim,
            "polynomials": [[[p._to_spec(bits) for p in entry] for entry in row] for row in self.polynomials],
            "prefactor": None if self.prefactor is None else self.prefactor._to_spec(bits),
            "reduced_prefactor": None if self.reduced_prefactor is None else self.reduced_prefactor._to_spec(bits),
            "max_num_poles": num_poles,
            "sample_points": None if self.sample_points is None else _num.strs(self.sample_points, bits),
            "sample_scalings": None if self.sample_scalings is None else _num.strs(self.sample_scalings, bits),
            "reduced_sample_scalings": (None if self.reduced_sample_scalings is None
                                        else _num.strs(self.reduced_sample_scalings, bits)),
            "bilinear_basis": (None if self.bilinear_basis is None else
                               ([p._to_spec(bits) for p in self.bilinear_basis[0]],
                                [p._to_spec(bits) for p in self.bilinear_basis[1]])),
        }

    def sampled(self, precision: int | None = None, max_num_poles: int | None = None) -> "SampledMatrix":
        """The sampling data SDPB derives for this block (points, scalings, bases)."""
        precision = effective_precision(precision)
        try:
            d = _ext().sample_matrix(self._to_spec(precision, max_num_poles), precision)
        except Exception as exc:
            raise wrap_cpp_error(exc) from None
        bits = precision
        return SampledMatrix(
            prefactor=DampedRational._from_spec(d["prefactor"], bits),
            reduced_prefactor=DampedRational._from_spec(d["reduced_prefactor"], bits),
            sample_points=_num.mpfs(d["sample_points"], bits),
            sample_scalings=_num.mpfs(d["sample_scalings"], bits),
            reduced_sample_scalings=_num.mpfs(d["reduced_sample_scalings"], bits),
            bilinear_basis=([Polynomial(_num.mpfs(c, bits)) for c in d["bilinear_basis"][0]],
                            [Polynomial(_num.mpfs(c, bits)) for c in d["bilinear_basis"][1]]),
            bilinear_bases=(_num.matrix_mpfs(d["bilinear_bases"][0], bits),
                            _num.matrix_mpfs(d["bilinear_bases"][1], bits)),
        )


def _combine_max_num_poles(local: int | None, global_: int | None) -> int | None:
    """As SDPB's JSON reader: min of the per-matrix and the global limit."""
    values = [v for v in (local, global_) if v is not None and v >= 0]
    if not values:
        return None if local is None and global_ is None else -1
    return min(values)


@dataclass
class SampledMatrix:
    prefactor: DampedRational
    reduced_prefactor: DampedRational
    sample_points: list[mpmath.mpf]
    sample_scalings: list[mpmath.mpf]
    reduced_sample_scalings: list[mpmath.mpf]
    bilinear_basis: tuple[list[Polynomial], list[Polynomial]]
    bilinear_bases: tuple[mpmath.matrix, mpmath.matrix]

    @property
    def num_points(self) -> int:
        return len(self.sample_points)


# ------------------------------------------------------------------ SDP data


@dataclass
class SDPBlock:
    block_index: int
    dim: int
    num_points: int
    bilinear_bases: tuple[mpmath.matrix, mpmath.matrix]
    c: list[mpmath.mpf]
    B: mpmath.matrix


@dataclass
class SDPData:
    """What ``pmp2sdp`` writes to an ``sdp/`` directory (Manual eq. 2.2)."""

    objective_const: mpmath.mpf
    b: list[mpmath.mpf]
    normalization: list[mpmath.mpf] | None
    blocks: list[SDPBlock]
    precision: int

    @classmethod
    def _from_dict(cls, d: dict, bits: int) -> "SDPData":
        return cls(
            objective_const=_num.from_str(d["objective_const"], bits),
            b=_num.mpfs(d["b"], bits),
            normalization=None if d["normalization"] is None else _num.mpfs(d["normalization"], bits),
            blocks=[SDPBlock(
                block_index=blk["block_index"], dim=blk["dim"], num_points=blk["num_points"],
                bilinear_bases=(_num.matrix_mpfs(blk["bilinear_bases"][0], bits),
                                _num.matrix_mpfs(blk["bilinear_bases"][1], bits)),
                c=_num.mpfs(blk["c"], bits),
                B=_num.matrix_mpfs(blk["B"], bits),
            ) for blk in d["blocks"]],
            precision=bits,
        )


# ------------------------------------------------------------------ PMP


@dataclass
class PMP:
    """A polynomial matrix program.

    ``objective`` is ``a_0..a_N``; ``normalization`` is ``n_0..n_N`` or ``None``;
    each matrix holds ``N+1`` polynomials per entry.
    """

    objective: Sequence[NumberLike]
    normalization: Sequence[NumberLike] | None = None
    matrices: Sequence[PolynomialMatrix] = field(default_factory=list)

    def __post_init__(self):
        self.objective = list(self.objective)
        if not self.objective:
            raise ValueError("objective must not be empty")
        if self.normalization is not None:
            self.normalization = list(self.normalization)
            if len(self.normalization) != len(self.objective):
                raise ValueError("normalization must have the same length as objective")
        self.matrices = list(self.matrices)
        if not self.matrices:
            raise ValueError("PMP needs at least one matrix")
        n = len(self.objective)
        for j, m in enumerate(self.matrices):
            if not isinstance(m, PolynomialMatrix):
                raise TypeError(f"matrices[{j}] is not a PolynomialMatrix")
            if m.num_vectors != n:
                raise ValueError(f"matrices[{j}] has {m.num_vectors} polynomials per entry, "
                                 f"but objective has length {n}")

    @property
    def num_variables(self) -> int:
        """``N``: number of dual variables ``y``."""
        return len(self.objective) - 1

    def _to_spec(self, bits: int, max_num_poles: int | None = None) -> dict:
        return {
            "objective": _num.strs(self.objective, bits),
            "normalization": None if self.normalization is None else _num.strs(self.normalization, bits),
            "matrices": [m._to_spec(bits, max_num_poles) for m in self.matrices],
        }

    def to_sdp(self, precision: int | None = None, max_num_poles: int | None = None) -> SDPData:
        """Convert to SDP data, as ``pmp2sdp --precision`` would."""
        precision = effective_precision(precision)
        try:
            d = _ext().pmp_to_sdp(self._to_spec(precision, max_num_poles), precision)
        except Exception as exc:
            raise wrap_cpp_error(exc) from None
        return SDPData._from_dict(d, precision)

    def solve(self, options: SolverOptions | None = None, *, max_num_poles: int | None = None,
              **overrides: Any) -> Solution:
        """Solve with SDPB.  Keyword arguments override :class:`SolverOptions` fields."""
        opts = resolve(options, overrides)
        try:
            d = _ext().solve_pmp(self._to_spec(effective_precision(opts.precision), max_num_poles),
                                 opts._to_dict())
        except Exception as exc:
            raise wrap_cpp_error(exc) from None
        return Solution._from_dict(d)
