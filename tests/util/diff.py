"""Comparison helpers mirroring SDPB's test/src/test_util/diff.hxx.

``assert_close(a, b, bits)`` requires ``|a - b| < 2^-bits * (|a| + |b|)``,
exactly as the fork's ``diff(El::BigFloat, El::BigFloat)`` does.
"""

from __future__ import annotations

import contextlib
from typing import Any

import mpmath


def assert_close(a: Any, b: Any, bits: int, what: str = "") -> None:
    """Relative binary-precision comparison; ``bits < 0`` means exact."""
    if isinstance(a, mpmath.matrix) or isinstance(b, mpmath.matrix):
        a = mpmath.matrix(a)
        b = mpmath.matrix(b)
        assert (a.rows, a.cols) == (b.rows, b.cols), f"{what}: shape {a.rows}x{a.cols} != {b.rows}x{b.cols}"
        for i in range(a.rows):
            for j in range(a.cols):
                assert_close(a[i, j], b[i, j], bits, f"{what}[{i},{j}]")
        return
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        assert len(a) == len(b), f"{what}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_close(x, y, bits, f"{what}[{i}]")
        return
    with mpmath.workprec(max(mpmath.mp.prec, 2 * abs(bits) + 64)):
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        if a == b:
            return
        if bits < 0:
            raise AssertionError(f"{what}: {a} != {b} (exact comparison)")
        eps = mpmath.mpf(2) ** (-bits)
        assert abs(a - b) < eps * (abs(a) + abs(b)), (
            f"{what}: |{a} - {b}| = {abs(a - b)} not < 2^-{bits} * (|a|+|b|) = {eps * (abs(a) + abs(b))}")


@contextlib.contextmanager
def precision(bits: int):
    """Temporarily set mpmath's working precision (Float_Binary_Precision analogue)."""
    with mpmath.workprec(bits):
        yield
