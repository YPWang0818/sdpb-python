"""End-to-end test: solve a small SDP from SDPB's test data through the Python API."""

from decimal import Decimal
from pathlib import Path

import pytest

SDPB_TEST_DATA = Path(__file__).resolve().parents[1] / "c-src" / "sdpb" / "test" / "data"
ONE_D = SDPB_TEST_DATA / "end-to-end_tests" / "1d" / "output"

# Same solver settings as SDPB's own end-to-end integration test for "1d".
SOLVER_OPTIONS = dict(
    precision=768,
    checkpointInterval=3600,
    maxRuntime=1340,
    dualityGapThreshold="1.0e-30",
    primalErrorThreshold="1.0e-30",
    dualErrorThreshold="1.0e-30",
    initialMatrixScalePrimal="1.0e20",
    initialMatrixScaleDual="1.0e20",
    feasibleCenteringParameter=0.1,
    infeasibleCenteringParameter=0.3,
    stepLengthReduction=0.7,
    maxComplementarity="1.0e100",
    maxIterations=1000,
    verbosity=0,
    writeSolution="x,y",
    noFinalCheckpoint=True,
)


@pytest.mark.skipif(not ONE_D.exists(), reason="SDPB submodule test data not checked out")
def test_solve_1d(sdpb_ext, tmp_path):
    import sdpb_python
    from sdpb_python import SDPBResult

    expected = SDPBResult.from_dir(ONE_D / "out")
    out_dir = tmp_path / "out"

    result = sdpb_python.solve_dir(ONE_D / "sdp", out_dir, **SOLVER_OPTIONS)

    assert result.optimal
    assert result.terminate_reason == expected.terminate_reason
    assert (out_dir / "out.txt").exists()
    assert (out_dir / "y.txt").exists()
    # The integration test compares to ~99 bits; be a little looser here.
    tol = Decimal("1e-25")
    assert abs(Decimal(result.primal_objective) - Decimal(expected.primal_objective)) < tol
    assert abs(Decimal(result.dual_objective) - Decimal(expected.dual_objective)) < tol
    assert Decimal(result.duality_gap) < Decimal("1e-29")
