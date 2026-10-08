"""SolverPool / solve_many: parallel solves in worker processes."""

import os
import signal
import time

import pytest

from sdpb_python import PMP, Polynomial, PolynomialMatrix, SDPBError, read_pmp_json
from sdpb_python.numbers import to_str
from sdpb_python.pool import SDPBWorkerError, SolveCancelled, SolverPool, solve_many
from tests.util.datasets import DATA, STANDARD_ARGS, TEST_PRECISION

ONE_D = DATA / "1d" / "input" / "pmp.json"
ARGS = dict(STANDARD_ARGS)


@pytest.fixture
def pmp(sdpb_ext):
    return read_pmp_json(ONE_D)


@pytest.fixture(scope="module")
def pool():
    pytest.importorskip("sdpb_python._sdpb", reason="sdpb_python._sdpb extension not built")
    with SolverPool(2, precision=TEST_PRECISION) as p:
        yield p


def variants(pmp, n):
    """n different problems: the 1d PMP with shifted objectives."""
    return [PMP(objective=[pmp.objective[0], f"-{k + 2}/{k + 3}"], normalization=pmp.normalization,
                matrices=pmp.matrices) for k in range(n)]


def long_problem(pmp, copies=40):
    """The 1d matrices repeated: a solve of many seconds, ~160 iterations."""
    return PMP(objective=pmp.objective, normalization=pmp.normalization, matrices=list(pmp.matrices) * copies)


def failing_problem():
    """Cholesky failure in SDPB (see test_profile.test_profile_on_error)."""
    return PMP(objective=[0, 1], normalization=[1, 0],
               matrices=[PolynomialMatrix([[[Polynomial(["3/2", -2, 1]), Polynomial([-1])]]])])


def fingerprint(sol):
    return (sol.status, sol.iterations, to_str(sol.primal_objective, TEST_PRECISION),
            to_str(sol.dual_objective, TEST_PRECISION), [to_str(v, TEST_PRECISION) for v in sol.y])


def wait_running(future, timeout=60):
    t0 = time.monotonic()
    while not future.running():
        assert time.monotonic() - t0 < timeout, "the solve did not start"
        time.sleep(0.05)


def test_pool_results_identical_to_sequential(pmp, pool):
    problems = variants(pmp, 4)
    expected = [fingerprint(p.solve(**ARGS)) for p in problems]
    futures = [pool.submit(p, **ARGS) for p in problems]
    assert [fingerprint(f.result()) for f in futures] == expected


def test_map_keeps_input_order(pmp, pool):
    problems = variants(pmp, 5)
    solutions = pool.map(problems, **ARGS)
    assert [fingerprint(s) for s in solutions] == [fingerprint(p.solve(**ARGS)) for p in problems]


def test_solve_many(pmp):
    problems = variants(pmp, 3)
    solutions = solve_many(problems, workers=2, precision=TEST_PRECISION, **ARGS)
    assert [s.optimal for s in solutions] == [True] * 3


def test_error_is_raised_from_the_future(sdpb_ext, pool):
    future = pool.submit(failing_problem(), duality_gap_threshold="1e-150", primal_error_threshold="1e-150",
                         dual_error_threshold="1e-150", max_iterations=2000)
    with pytest.raises(SDPBError) as info:
        future.result()
    assert "Cholesky" in info.value.details
    # the pool still works
    assert pool.submit(read_pmp_json(ONE_D), **ARGS).result().optimal


def test_precision_must_match(pmp, pool):
    with pytest.raises(ValueError, match="bits"):
        pool.submit(pmp, precision=TEST_PRECISION - 256, **ARGS)


def test_cancel_queued(pmp):
    with SolverPool(1, precision=TEST_PRECISION) as pool:
        first = pool.submit(long_problem(pmp, copies=3), **ARGS)
        queued = pool.submit(pmp, **ARGS)
        assert pool.cancel(queued)
        assert queued.cancelled()
        assert first.result().optimal


def test_cancel_running(pmp):
    with SolverPool(1, precision=TEST_PRECISION) as pool:
        future = pool.submit(long_problem(pmp), **ARGS)
        wait_running(future)
        time.sleep(1.0)
        t0 = time.monotonic()
        assert pool.cancel(future)
        with pytest.raises(SolveCancelled) as info:
            future.result(timeout=60)
        assert time.monotonic() - t0 < 30
        partial = info.value.solution
        assert partial is not None and 0 < partial.iterations < 160 and partial.runtime_seconds > 0
        # the worker is reused for the next solve
        assert pool.submit(pmp, **ARGS).result().optimal


def test_dead_worker_is_replaced(pmp):
    with SolverPool(1, precision=TEST_PRECISION) as pool:
        future = pool.submit(long_problem(pmp), **ARGS)
        wait_running(future)
        time.sleep(0.5)
        (pid,) = pool.pids
        os.kill(pid, signal.SIGKILL)
        with pytest.raises(SDPBWorkerError, match="exited"):
            future.result(timeout=60)
        assert pool.submit(pmp, **ARGS).result().optimal
        assert pool.pids != [pid]


def test_workers_use_spawn_and_one_blas_thread(pool):
    assert pool._context.get_start_method() == "spawn"
    assert pool.env["OPENBLAS_NUM_THREADS"] == "1"


def test_refuses_to_run_under_a_foreign_launcher(sdpb_ext, monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    monkeypatch.delenv("SDPB_PYTHON_ALLOW_MULTI_PROCESS", raising=False)
    with pytest.raises(SDPBError, match="SolverPool"):
        SolverPool(1, precision=TEST_PRECISION)


def test_closed_pool_rejects_work(pmp):
    pool = SolverPool(1, precision=TEST_PRECISION)
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.submit(pmp, **ARGS)
