"""T6: solver handle, checkpoints, warm starts (analogue of sdpb/io_tests)."""

import os
import stat

import mpmath
import pytest

from sdpb_python import LMI, SDPBError, Solver, TerminateReason, read_pmp_json
from tests.util.datasets import DATA, TEST_PRECISION, STANDARD_ARGS
from tests.util.diff import assert_close

ONE_D = DATA / "1d" / "input" / "pmp.json"
ARGS = dict(STANDARD_ARGS)


@pytest.fixture
def pmp(sdpb_ext):
    return read_pmp_json(ONE_D)


def test_run_continues(pmp):
    """run() after MaxIterationsExceeded continues the same trajectory."""
    reference = pmp.solve(**ARGS)
    assert reference.optimal
    with pmp.solver(**ARGS) as solver:
        first = solver.run(max_iterations=5)
        assert first.status is TerminateReason.MAX_ITERATIONS_EXCEEDED
        assert first.iterations == 5
        second = solver.run()
        assert second.optimal
        assert solver.total_iterations == 5 + second.iterations == reference.iterations
        with mpmath.workprec(TEST_PRECISION):
            assert_close(second.primal_objective, reference.primal_objective, 99, "primal")
            assert_close(second.y, reference.y, 99, "y")
        assert solver.state().y == second.y
    assert solver.closed
    with pytest.raises(SDPBError):
        solver.run()


def test_checkpoint_restart(pmp, tmp_path):
    """Analogue of sdpb/io_tests/checkpoint_read: stop after 1 iteration, restart from checkpoint."""
    reference = pmp.solve(**ARGS)
    ck = tmp_path / "ck"
    partial = pmp.solve(max_iterations=1, checkpoint_dir=ck, **{k: v for k, v in ARGS.items() if k != "max_iterations"})
    assert partial.status is TerminateReason.MAX_ITERATIONS_EXCEEDED
    assert (ck / "checkpoint.json").exists()
    restarted = pmp.solve(checkpoint_dir=ck, **ARGS)
    assert restarted.optimal
    assert restarted.iterations == reference.iterations - 1
    with mpmath.workprec(TEST_PRECISION):
        assert_close(restarted.primal_objective, reference.primal_objective, 99)
        assert_close(restarted.y, reference.y, 99)
    # the handle can save a checkpoint too
    with pmp.solver(**ARGS) as solver:
        solver.run(max_iterations=2)
        solver.save_checkpoint(tmp_path / "ck2")
    assert (tmp_path / "ck2" / "checkpoint.json").exists()
    again = pmp.solve(checkpoint_dir=tmp_path / "ck2", **ARGS)
    assert again.iterations == reference.iterations - 2


def test_corrupted_checkpoint(pmp, tmp_path):
    """Analogue of sdpb/io_tests/checkpoint_corrupt and checkpoint_read (unreadable)."""
    ck = tmp_path / "ck"
    pmp.solve(max_iterations=1, checkpoint_dir=ck, **{k: v for k, v in ARGS.items() if k != "max_iterations"})
    files = sorted(p for p in ck.iterdir() if p.name.startswith("checkpoint_"))
    assert files
    data = files[0].read_bytes()
    files[0].write_bytes(data[: len(data) // 2])
    with pytest.raises(SDPBError):
        pmp.solve(checkpoint_dir=ck, **ARGS)
    files[0].write_bytes(data)
    if os.geteuid() != 0:
        files[0].chmod(0)
        try:
            with pytest.raises(SDPBError, match="checkpoint"):
                pmp.solve(checkpoint_dir=ck, **ARGS)
        finally:
            files[0].chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_warm_start(pmp):
    """Starting from the optimum finds it again in very few iterations."""
    reference = pmp.solve(want=("y", "X", "Y"), **ARGS)
    with pmp.solver(**ARGS) as solver:
        solver.warm_start(y=reference.y, X=reference.X, Y=reference.Y)
        state = solver.state()
        with mpmath.workprec(TEST_PRECISION):
            assert_close(state.y, reference.y, -1, "y set exactly")
            assert_close(state.X[0][0], reference.X[0][0], -1, "X set exactly")
        warm = solver.run()
        assert warm.optimal
        assert warm.iterations < reference.iterations // 2
    with pmp.solver(**ARGS) as solver, pytest.raises(SDPBError, match="length"):
        solver.warm_start(y=[1, 2, 3])


def test_lmi_handle(sdpb_ext):
    lmi = LMI(b=[1], blocks=[([[1, 0], [0, 1]], [[0, 1], [1, 0]])])
    with lmi.solver(duality_gap_threshold="1e-40") as solver:
        sol = solver.run()
        assert sol.optimal and sol.z is None
        with mpmath.workprec(TEST_PRECISION):
            assert_close(sol.y, [1], 120)
        assert solver.num_variables == 1
