"""Precision is fixed once per process (SDPB's Elemental); native-precision runs use subprocesses."""

import pytest

import sdpb_python
from sdpb_python import SDPBError
from tests.util.datasets import DATA, TEST_PRECISION
from tests.util.reference import read_iterations_json, read_out_txt


def test_precision_lock(sdpb_ext):
    assert sdpb_python.precision() == TEST_PRECISION
    sdpb_python.set_precision(TEST_PRECISION)  # same value is fine
    with pytest.raises(SDPBError, match="already fixed"):
        sdpb_python.set_precision(TEST_PRECISION + 64)
    pmp = sdpb_python.read_pmp_json(DATA / "1d" / "input" / "pmp.json")
    with pytest.raises(SDPBError, match="already fixed"):
        pmp.to_sdp(precision=TEST_PRECISION + 64)
    with pytest.raises(SDPBError, match="already fixed"):
        pmp.solve(precision=TEST_PRECISION + 64, max_iterations=1)
    # None picks the fixed precision
    assert pmp.to_sdp().precision >= TEST_PRECISION


def check_1d_native(precision: int, diff_bits: int):
    """Runs in a subprocess at 664 bits: the 1d reference was produced there (fork: precision=664)."""
    import mpmath

    from tests.util.diff import assert_close

    dataset_dir = DATA / "1d"
    pmp = sdpb_python.read_pmp_json(dataset_dir / "input" / "pmp.json")
    sol = pmp.solve(precision=precision, want=("y",))
    assert sdpb_python.precision() == precision
    ref = read_out_txt(dataset_dir / "output" / "out" / "out.txt")
    assert sol.status.value == ref.terminate_reason
    with mpmath.workprec(precision):
        assert_close(sol.primal_objective, ref.values["primalObjective"], diff_bits, "primalObjective")
        assert_close(sol.dual_objective, ref.values["dualObjective"], diff_bits, "dualObjective")
    assert sol.iterations == len(read_iterations_json(dataset_dir / "output" / "out" / "iterations.json"))


def test_1d_at_native_precision(sdpb_ext):
    from tests.util.subproc import run_at_precision

    run_at_precision(664, "tests.test_precision.check_1d_native", 664, 99)
