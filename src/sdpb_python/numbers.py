"""Conversions between Python numbers and the decimal strings SDPB exchanges.

Strings are printed with ``max_digits10`` digits for the working precision, so
``mpf -> str -> BigFloat -> str -> mpf`` is lossless.
"""

from __future__ import annotations

import math
from fractions import Fraction
from numbers import Integral, Rational, Real
from typing import Any, Iterable

import mpmath

NumberLike = Any  # int, float, str, Fraction, mpmath.mpf, ...


def digits_for(bits: int) -> int:
    """Decimal digits that uniquely identify a binary float of ``bits`` bits."""
    return math.ceil(bits * math.log10(2)) + 1


def to_str(value: NumberLike, bits: int) -> str:
    """Render ``value`` as a decimal string exact to ``bits`` of precision."""
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise ValueError("empty number string")
        return s
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Fraction) or (isinstance(value, Rational) and not isinstance(value, Real)):
        with mpmath.workprec(bits):
            return mpmath.nstr(mpmath.mpf(value.numerator) / value.denominator, digits_for(bits))
    if isinstance(value, float):
        # Shortest decimal that round-trips the double: 0.1 means "0.1",
        # not 0.1000000000000000055511151231257827.
        return repr(value)
    if isinstance(value, mpmath.mpf):
        with mpmath.workprec(max(bits, value.context.prec)):
            return mpmath.nstr(value, digits_for(max(bits, value.context.prec)))
    if hasattr(value, "_mpf_"):
        return to_str(mpmath.mpf(value._mpf_), bits)
    try:  # sympy numbers, numpy scalars, decimal.Decimal, ...
        return to_str(mpmath.mpf(str(value)), bits)
    except Exception as exc:
        raise TypeError(f"cannot convert {type(value).__name__} to a number") from exc


def from_str(s: str, bits: int) -> mpmath.mpf:
    """Parse a decimal string without rounding below ``bits`` of precision."""
    with mpmath.workprec(max(bits, mpmath.mp.prec)):
        return mpmath.mpf(s)


def strs(values: Iterable[NumberLike], bits: int) -> list[str]:
    return [to_str(v, bits) for v in values]


def mpfs(values: Iterable[str], bits: int) -> list[mpmath.mpf]:
    return [from_str(v, bits) for v in values]


def matrix_strs(rows: Any, bits: int) -> list[list[str]]:
    """Convert a nested sequence / numpy array / mpmath.matrix to string rows."""
    if isinstance(rows, mpmath.matrix):
        return [[to_str(rows[i, j], bits) for j in range(rows.cols)] for i in range(rows.rows)]
    if hasattr(rows, "tolist"):
        rows = rows.tolist()
    return [[to_str(v, bits) for v in row] for row in rows]


def matrix_mpfs(data: dict, bits: int) -> mpmath.matrix:
    """From the Cython layer's {"shape": (h, w), "rows": [[str]]}."""
    h, w = data["shape"]
    m = mpmath.matrix(h, w)
    for i, row in enumerate(data["rows"]):
        for j, v in enumerate(row):
            m[i, j] = from_str(v, bits)
    return m
