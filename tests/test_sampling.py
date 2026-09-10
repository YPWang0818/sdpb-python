"""T2: analogue of unit_tests/cases/pmp_sampling.test.cxx."""

import math

import mpmath
import pytest

from sdpb_python import DampedRational, Polynomial, PolynomialMatrix
from tests.util.diff import assert_close

DIFF_BITS = 16  # "Truncated to 5 decimal digits, i.e. precision = 16"


def matrix_of_degree(degree: int) -> PolynomialMatrix:
    """1x1 matrix with one polynomial of the given degree (so num_points = degree + 1)."""
    return PolynomialMatrix([[[Polynomial([1] * (degree + 1))]]])


def rows_match_up_to_sign(got: mpmath.matrix, expected, bits):
    """The fork flips row signs 'when necessary' (bases are defined up to sign)."""
    expected = mpmath.matrix(expected) if expected else mpmath.matrix(0, 0)
    assert (got.rows, got.cols) == (expected.rows, expected.cols)
    for i in range(got.rows):
        row = [got[i, j] for j in range(got.cols)]
        exp = [expected[i, j] for j in range(got.cols)]
        try:
            assert_close(row, exp, bits)
        except AssertionError:
            assert_close([-v for v in row], exp, bits, f"row {i} (negated)")


def test_exp_minus_x_degree_4(sdpb_ext):
    prefactor = DampedRational(1, math.exp(-1), [])
    m = PolynomialMatrix([[[Polynomial([1, 1, 1, 1, 1])]]], prefactor=DampedRational.exp_minus_x())
    s = m.sampled(precision=768)
    assert_close(s.sample_points, [0.061812, 0.56588, 1.6319, 3.4239, 6.4864], DIFF_BITS)
    assert_close(s.sample_scalings, [0.94006, 0.56786, 0.19555, 0.032586, 0.0015240], DIFF_BITS)
    rows_match_up_to_sign(s.bilinear_bases[0],
                          [[0.735538, 0.571674, 0.335472, 0.136945, 0.0296154],
                           [-0.454533, 0.0809192, 0.586352, 0.609108, 0.268386],
                           [0.339114, -0.329580, -0.394268, 0.369777, 0.695842]], DIFF_BITS)
    rows_match_up_to_sign(s.bilinear_bases[1],
                          [[0.266195, 0.625991, 0.623829, 0.368860, 0.109794],
                           [-0.314913, -0.462694, 0.124527, 0.655667, 0.491262]], DIFF_BITS)
    # default prefactor is exp(-x): omitting it gives the same sampling
    s2 = matrix_of_degree(4).sampled(precision=768)
    assert_close(s2.sample_points, s.sample_points, 700)


def test_exp_minus_x_over_x_x_plus_1_degree_4(sdpb_ext):
    with mpmath.workprec(1024):
        prefactor = DampedRational(1, mpmath.exp(-1), [0, -1])
    m = PolynomialMatrix([[[Polynomial([1, 1, 1, 1, 1])]]], prefactor=prefactor)
    s = m.sampled(precision=768)
    assert_close(s.sample_points, [0, 0.0501905, 0.490170, 1.56871, 3.82960], DIFF_BITS)
    assert_close(s.sample_scalings, ["1.0000e16", 18.0432, 0.838570, 0.0516961, 0.00117426], DIFF_BITS)
    rows_match_up_to_sign(s.bilinear_bases[0],
                          [[1.00000, 4.247724e-8, 9.157346e-9, 2.273678e-9, 3.426744e-10],
                           [-2.241430e-8, 0.3407876, 0.7175002, 0.5701354, 0.2097686],
                           [1.771584e-8, -0.3631303, -0.3850530, 0.4332262, 0.7295105]], DIFF_BITS)
    rows_match_up_to_sign(s.bilinear_bases[1],
                          [[0, 0.8036324, 0.5414187, 0.2404866, 0.05663024],
                           [0, -0.4294390, 0.2667574, 0.7239642, 0.4693597]], DIFF_BITS)


def test_constant_prefactor_degree_0(sdpb_ext):
    m = PolynomialMatrix([[[Polynomial([1])]]], prefactor=DampedRational(1, 1, []))
    s = m.sampled(precision=768)
    assert_close(s.sample_points, [0], -1)
    assert_close(s.sample_scalings, [1], -1)
    assert_close(s.bilinear_bases[0], [[1]], -1)
    assert (s.bilinear_bases[1].rows, s.bilinear_bases[1].cols) == (0, 1)


@pytest.mark.parametrize("poles", [[], [0], [0, 0], [-1], [0, -1], [0, -1, -2]], ids=str)
@pytest.mark.parametrize("degree", [0, 1, 2, 10])
def test_crash_tests(sdpb_ext, poles, degree):
    """Sampling must not fail for these prefactor / degree combinations."""
    with mpmath.workprec(1024):
        prefactor = DampedRational(1, mpmath.exp(-1), poles)
    m = PolynomialMatrix([[[Polynomial([1] * (degree + 1))]]], prefactor=prefactor)
    s = m.sampled(precision=768)
    assert s.num_points >= 1
    assert len(s.sample_scalings) == s.num_points
