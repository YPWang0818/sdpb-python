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
    """A damped rational prefactor ``constant * base**x / prod_i (x - poles[i])``.

    Multiplying a polynomial matrix by a positive prefactor does not change the
    positivity constraint, but it tells SDPB where to place the sample points
    and how to scale them.  SDPB's default is ``exp(-x)``;
    :meth:`exp_minus_x` builds it (with optional poles) at full precision.
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

    @classmethod
    def from_sympy(cls, expr: Any, x: Any = None, digits: int = 300) -> "Polynomial":
        """From a sympy expression polynomial in ``x`` (the only symbol if omitted).

        Exact rationals stay exact (as ``Fraction``); other numbers are evaluated
        to ``digits`` significant digits.
        """
        import sympy
        from fractions import Fraction

        expr = sympy.sympify(expr)
        if x is None:
            symbols = sorted(expr.free_symbols, key=str)
            if len(symbols) > 1:
                raise ValueError(f"expression has several symbols {symbols}; pass x explicitly")
            x = symbols[0] if symbols else sympy.Symbol("x")
        poly = sympy.Poly(sympy.expand(expr), x)
        coeffs = []
        for c in reversed(poly.all_coeffs()):  # low degree first
            if c.is_Rational:
                coeffs.append(Fraction(int(c.p), int(c.q)))
            else:
                coeffs.append(str(sympy.N(c, digits)))
        return cls(coeffs)

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
        return mpmath.mpf(_num.to_str(x, 128)) < 0
    except Exception:
        return False


def _is_zero(x: Any) -> bool:
    try:
        return mpmath.mpf(_num.to_str(x, 128)) == 0
    except Exception:
        return False


def _as_polynomial(p: Any) -> Polynomial:
    return p if isinstance(p, Polynomial) else Polynomial(p)


@dataclass
class PolynomialMatrix:
    """One positivity constraint: ``sum_n z_n P^{rs}_n(x)`` is PSD for ``x >= 0``.

    ``polynomials[r][s]`` is the list of ``N+1`` polynomials ``P^{rs}_0..P^{rs}_N``
    (each a :class:`Polynomial` or a coefficient list); the matrix must be
    symmetric.  The optional fields mirror the keys of ``pmp.json``; anything
    omitted is computed by SDPB (see :meth:`sampled`).

    Attributes:
        polynomials: ``dim x dim`` nested lists of ``N+1`` polynomials each.
        prefactor: :class:`DampedRational`; default ``exp(-x)``.
        reduced_prefactor: Prefactor with fewer poles used for sampling;
            default derived from ``prefactor`` and ``max_num_poles``.
        max_num_poles: Keep at most this many poles of the prefactor when
            sampling (negative: no limit).
        sample_points: Explicit sample points ``x_k >= 0``.
        sample_scalings: Explicit ``prefactor(x_k)``.
        reduced_sample_scalings: Explicit ``reduced_prefactor(x_k)``.
        bilinear_basis: Explicit ``(even, odd)`` lists of basis polynomials.
        label: Free text, e.g. the source file; not used by the solver.
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
    """Sampling data SDPB derived for one :class:`PolynomialMatrix`.

    Attributes:
        prefactor: The prefactor actually used (default ``exp(-x)``).
        reduced_prefactor: The prefactor after ``max_num_poles`` trimming.
        sample_points: The ``num_points`` points ``x_k >= 0``.
        sample_scalings: ``prefactor(x_k)``.
        reduced_sample_scalings: ``reduced_prefactor(x_k)``.
        bilinear_basis: Even and odd orthogonal polynomials ``q_m(x)``.
        bilinear_bases: The bases sampled at the points (as stored in
            ``block_data`` files): even and odd matrices.
    """

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
    """One block of an :class:`SDPData` (the content of ``block_data_<j>``).

    Attributes:
        block_index: Global block index ``j``.
        dim: Matrix dimension.
        num_points: Number of sample points.
        bilinear_bases: Even and odd sampled bases.
        c: Constraint constants, length ``num_points * dim * (dim+1) / 2``.
        B: Constraint matrix, ``len(c)`` by ``N``.
    """

    block_index: int
    dim: int
    num_points: int
    bilinear_bases: tuple[mpmath.matrix, mpmath.matrix]
    c: list[mpmath.mpf]
    B: mpmath.matrix


@dataclass
class SDPData:
    """What ``pmp2sdp`` writes to an ``sdp/`` directory (Manual eq. 2.2).

    The normalization of the PMP has been eliminated: ``objective_const`` is
    ``b_0`` and ``b`` is ``b_1..b_N``.

    Attributes:
        objective_const: The constant term ``b_0``.
        b: The dual objective vector.
        normalization: The PMP's normalization, kept for recovering ``z``.
        blocks: One :class:`SDPBlock` per constraint.
        precision: Bits used for the conversion.
    """

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
    """A polynomial matrix program (SDPB manual eq. 3.1):

    maximize ``a . z`` over ``z`` in ``R^{N+1}`` such that ``n . z = 1`` and, for
    every matrix ``j``, ``sum_i z_i M^j_i(x)`` is positive semidefinite for all
    ``x >= 0``.

    ``objective`` is ``a_0..a_N``; ``normalization`` is ``n_0..n_N`` (``None``
    means ``(1, 0, ..., 0)``, i.e. ``z_0 = 1`` and ``y = z_1..z_N``); each
    matrix holds ``N+1`` polynomials per entry.
    """

    objective: Sequence[NumberLike]
    normalization: Sequence[NumberLike] | None = None
    matrices: Sequence[PolynomialMatrix] = field(default_factory=list)

    def __post_init__(self):
        self.objective = list(self.objective)
        if not self.objective:
            raise ValueError("objective must not be empty")
        if len(self.objective) == 1:
            # The normalization n . z = 1 fixes the single component, leaving no
            # free variable (N = 0); SDPB then fails an internal assertion
            # (BigInt_Shared_Memory_Syrk_Context: input_window_split_factor > 0).
            raise ValueError("objective needs at least two components: the normalization "
                             "n . z = 1 fixes one of them, so a single component leaves "
                             "no free variable to optimize over")
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
        self._check_no_degenerate_variables()

    def _check_no_degenerate_variables(self) -> None:
        """Reject variables whose column of the SDP's B matrix would vanish.

        SDPB eliminates the normalization through the component ``k`` with the
        largest ``|n_k|``; the column of ``z_i`` (``i != k``) is built from
        ``P_i - (n_i / n_k) P_k``.  If that is identically zero in every matrix,
        the Schur complement ``Q`` is singular and SDPB aborts deep inside the
        solver (``check_normalized_Q_diagonal``).  Catch it here instead.
        """
        n = len(self.objective)
        nonzero = [False] * n
        for m in self.matrices:
            for row in m.polynomials:
                for entry in row:
                    for i, poly in enumerate(entry):
                        if not nonzero[i] and any(not _is_zero(c) for c in poly.coeffs):
                            nonzero[i] = True
        if self.normalization is None:
            k = 0
            norm = [1] + [0] * (n - 1)
        else:
            norm = self.normalization
            k = max(range(n), key=lambda i: abs(mpmath.mpf(_num.to_str(norm[i], 128))))
        for i in range(n):
            if i == k or nonzero[i]:
                continue
            if _is_zero(norm[i]) or not nonzero[k]:
                raise ValueError(
                    f"variable z_{i} multiplies only zero polynomials in every matrix, so its column "
                    "of the SDP would vanish and SDPB would fail; drop the variable or give it a "
                    "nonzero polynomial")

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

    def solver(self, options: SolverOptions | None = None, *, max_num_poles: int | None = None,
               **overrides: Any):
        """A :class:`~sdpb_python.handle.Solver` holding this problem's state."""
        from .handle import Solver

        opts = resolve(options, overrides)
        return Solver("pmp", self._to_spec(effective_precision(opts.precision), max_num_poles), opts)

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
