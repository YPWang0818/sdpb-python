"""T5: linear matrix inequalities (analytic checks; no fork analogue)."""

import mpmath
import pytest

from sdpb_python import LMI, PMP, Polynomial, PolynomialMatrix, DampedRational
from tests.util.datasets import TEST_PRECISION
from tests.util.diff import assert_close

OPTS = dict(duality_gap_threshold="1e-40", primal_error_threshold="1e-40", dual_error_threshold="1e-40")


def test_two_by_two(sdpb_ext):
    """maximize y  s.t.  [[1, y], [y, 1]] >= 0   ->  y = 1."""
    M0 = [[1, 0], [0, 1]]
    M1 = [[0, 1], [1, 0]]
    sol = LMI(b=[1], blocks=[(M0, M1)]).solve(want=("y", "Y", "X", "x"), **OPTS)
    assert sol.optimal
    with mpmath.workprec(TEST_PRECISION):
        assert_close(sol.y, [1], 120, "y")
        assert_close(sol.dual_objective, 1, 120, "dual objective")
        # Y = M0 + y M1 is the PSD block (even parity); the odd block is empty
        Y_even, Y_odd = sol.Y[0]
        assert_close(Y_even, mpmath.matrix([[1, 1], [1, 1]]), 120, "Y")
        assert (Y_odd.rows, Y_odd.cols) == (0, 0)
    assert sol.blocks[0].dim == 2 and sol.blocks[0].num_points == 1


def test_two_variables_and_offset(sdpb_ext):
    """maximize 3 + y1 + y2  s.t.  diag(1 - y1, 1 - y2, 1 + y1 + y2) >= 0  ->  y1 = y2 = 1, value 5."""
    d = lambda a, b, c: [[a, 0, 0], [0, b, 0], [0, 0, c]]
    sol = LMI(b=[1, 1], f=3, blocks=[(d(1, 1, 1), d(-1, 0, 1), d(0, -1, 1))]).solve(**OPTS)
    assert sol.optimal
    with mpmath.workprec(TEST_PRECISION):
        assert_close(sol.y, [1, 1], 120, "y")
        assert_close(sol.dual_objective, 5, 120, "objective")


def test_minimisation_by_negation(sdpb_ext):
    """minimize y  s.t.  [[1, y], [y, 1]] >= 0  ->  y = -1  (negate b)."""
    sol = LMI(b=[-1], blocks=[([[1, 0], [0, 1]], [[0, 1], [1, 0]])]).solve(**OPTS)
    with mpmath.workprec(TEST_PRECISION):
        assert_close(sol.y, [-1], 120)


def test_lmi_equals_constant_pmp(sdpb_ext):
    """An LMI is a PMP with constant polynomials and a constant prefactor (one sample point)."""
    M0 = [[2, 0], [0, 2]]
    M1 = [[0, 1], [1, 0]]
    lmi = LMI(b=[1], blocks=[(M0, M1)]).solve(**OPTS)
    pmp = PMP(objective=[0, 1], normalization=None,
              matrices=[PolynomialMatrix([[[Polynomial([M0[r][s]]), Polynomial([M1[r][s]])] for s in range(2)]
                                          for r in range(2)],
                                         prefactor=DampedRational(1, 1, []))]).solve(**OPTS)
    assert lmi.optimal and pmp.optimal
    with mpmath.workprec(TEST_PRECISION):
        assert_close(lmi.y, pmp.y, 99, "y")
        assert_close(lmi.dual_objective, pmp.dual_objective, 99, "objective")
        assert_close(lmi.y, [2], 120, "y = 2")


def test_validation():
    with pytest.raises(ValueError, match="N\\+1"):
        LMI(b=[1], blocks=[([[1]],)])
    with pytest.raises(ValueError, match="symmetric"):
        LMI(b=[1], blocks=[([[1, 2], [3, 1]], [[0, 0], [0, 0]])])._to_spec(400)
