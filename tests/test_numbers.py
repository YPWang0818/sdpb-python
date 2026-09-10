"""T1: number round trips (Boost_Float / json test analogues)."""

from fractions import Fraction

import mpmath
import pytest

from sdpb_python import numbers as num
from tests.util.reference import read_out_txt
from tests.util.datasets import DATA


@pytest.mark.parametrize("bits", [400, 664, 768, 1024])
def test_python_roundtrip(bits):
    with mpmath.workprec(bits):
        values = [mpmath.mpf("1") / 3, mpmath.mpf("1.84026576313204924668804017173"),
                  mpmath.mpf("2.844993327070838830130404255408e-309"), mpmath.mpf(10) ** 300,
                  -mpmath.mpf("7.7716e-305"), mpmath.mpf(0), mpmath.mpf(-1)]
        for v in values:
            s = num.to_str(v, bits)
            back = num.from_str(s, bits)
            assert back == v, (v, s, back)


def test_inputs_accepted():
    bits = 400
    assert num.to_str(1, bits) == "1"
    assert num.to_str(True, bits) == "1"
    assert num.to_str("  1e-30 ", bits) == "1e-30"
    with mpmath.workprec(bits):
        assert num.from_str(num.to_str("1/12", bits), bits) == mpmath.mpf(1) / 12
    with mpmath.workprec(bits):
        assert num.from_str(num.to_str(Fraction(1, 3), bits), bits) == mpmath.mpf(1) / 3
        assert num.from_str(num.to_str(0.1, bits), bits) == mpmath.mpf("0.1")  # decimal, not binary 0.1
    with pytest.raises(ValueError):
        num.to_str("", bits)
    with pytest.raises(TypeError):
        num.to_str(object(), bits)


def test_cpp_roundtrip(sdpb_ext):
    """mpf -> str -> BigFloat -> str -> mpf is the identity (through sample_matrix)."""
    from tests.util.datasets import TEST_PRECISION

    for bits in (TEST_PRECISION,):
        with mpmath.workprec(bits):
            values = [mpmath.mpf("1") / 3, mpmath.mpf("1.84026576313204924668804017173"),
                      mpmath.mpf("2.8449933e-309"), mpmath.mpf(10) ** 300, mpmath.mpf(-1) / 7]
            # constant polynomial v with prefactor 1: SDPB samples at x=0 with scaling 1,
            # so the SDP's c vector is exactly [v]
            for v in values:
                spec = {"objective": ["0"], "normalization": None,
                        "matrices": [{"dim": 1, "polynomials": [[[[num.to_str(v, bits)]]]],
                                      "prefactor": {"constant": "1", "base": "1", "poles": []}}]}
                out = sdpb_ext.pmp_to_sdp(spec, bits)
                assert num.from_str(out["blocks"][0]["c"][0], bits) == v


def test_reference_out_txt_values_roundtrip():
    out = read_out_txt(DATA / "1d" / "output" / "out" / "out.txt")
    for key, s in out.values.items():
        if key == "Solver runtime":
            continue
        v = num.from_str(s, 768)
        assert num.from_str(num.to_str(v, 768), 768) == v
