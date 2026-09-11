"""SDPBError surfaces SDPB's message, not only the location header (bug report 2026-09-11)."""

import mpmath
import pytest

from sdpb_python import PMP, Polynomial, PolynomialMatrix, SDPBError
from sdpb_python.errors import summarize


def test_sdpb_macro_message_is_surfaced():
    details = ("in initialize_schur_off_diagonal() at ../src/x/compute_Q.cxx:36: \n"
               "  Error when computing Cholesky decomposition of block_0: A was not numerically HPD\n"
               "Stacktrace:\n 0# 0x1234 in libfoo.so\n")
    e = SDPBError(details)
    assert str(e).startswith("Error when computing Cholesky decomposition of block_0")
    assert "compute_Q.cxx:36" in str(e)
    assert "Stacktrace" not in str(e)
    assert e.details == details
    assert e.location == "../src/x/compute_Q.cxx:36 (initialize_schur_off_diagonal())"


def test_assert_macro_multiline_message():
    details = ("in check_normalized_Q_diagonal() at ../src/x/compute_Q.cxx:85: \n"
               "  Assertion 'diff < eps' failed:\n"
               "    Normalized Q should have ones on diagonal. For i = 0: Q_ii = 0\n"
               "Stacktrace:\n 0# ...\n")
    assert str(SDPBError(details)).startswith("Assertion 'diff < eps' failed: Normalized Q")


def test_single_line_wrapper_error_unchanged():
    e = SDPBError("verbosity must be 0..3")
    assert str(e) == "verbosity must be 0..3"
    assert e.location is None


def test_header_without_message():
    assert summarize("in f() at a.cxx:1: \nStacktrace:\n 0# x") == "SDPB error at a.cxx:1 (f())"


def test_empty():
    assert str(SDPBError("")) == "SDPB error"
    assert str(SDPBError("  \n ")) == "SDPB error"


def test_degenerate_variable_rejected():
    """A variable multiplying only zero polynomials is caught before reaching SDPB."""
    with pytest.raises(ValueError, match="z_1 multiplies only zero polynomials"):
        PMP(objective=[0, 1], normalization=[1, 0],
            matrices=[PolynomialMatrix([[[Polynomial([-1, -1]), Polynomial([0])]]])])
    # with a nontrivial normalization the eliminated component may carry the polynomial
    ok = PMP(objective=[0, 1], normalization=[1, 1],
             matrices=[PolynomialMatrix([[[Polynomial([1, 2]), Polynomial([0])]]])])
    assert ok.num_variables == 1
    # ...but not when its column vanishes too
    with pytest.raises(ValueError, match="z_0 multiplies only zero polynomials"):
        PMP(objective=[1, 0], normalization=[0, 1],
            matrices=[PolynomialMatrix([[[Polynomial([0]), Polynomial([1, 1])]]])])


def test_real_cholesky_failure_message(sdpb_ext):
    """Over-tight thresholds make SDPB's Cholesky fail; the message must reach str(e)."""
    from tests.util.datasets import TEST_PRECISION

    pmp = PMP(objective=[0, 1], normalization=[1, 0],
              matrices=[PolynomialMatrix([[[Polynomial(["3/2", -2, 1]), Polynomial([-1])]]])])
    with pytest.raises(SDPBError) as info:
        pmp.solve(precision=TEST_PRECISION, duality_gap_threshold="1e-150", primal_error_threshold="1e-150",
                  dual_error_threshold="1e-150", max_iterations=2000)
    e = info.value
    assert "not numerically HPD" in str(e), str(e)
    assert "compute_Q.cxx" in str(e)
    assert e.location is not None and "initialize_schur_off_diagonal" in e.location
    assert "Stacktrace" in e.details
