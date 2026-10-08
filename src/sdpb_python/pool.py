"""Solve independent problems in parallel on a pool of worker processes.

SDPB's state is process-global (MPI, the working precision, signal handlers),
so one process runs one solve at a time and parallel solves need processes.
:class:`SolverPool` starts ``workers`` processes with the ``"spawn"`` method
(never ``fork``: the parent may already have initialised MPI). Each worker is
an independent single-rank SDPB with one BLAS thread and the pool's precision
fixed once; a problem is sent as the same strings :meth:`PMP.solve` passes to
SDPB, so a solve in the pool gives exactly the result of a solve in this
process.

The parent never initialises MPI or loads SDPB for the pool: problems are
converted to strings here and parsed by the workers.

Each worker opens its own MPI listener socket (MPICH ch3:nemesis builds do so
even for one process), so N workers mean N listeners.
"""

from __future__ import annotations

import collections
import concurrent.futures as cf
import itertools
import multiprocessing
import os
import signal
import threading
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from typing import Any, Iterable, Mapping, Sequence

from . import _mpi
from .errors import SDPBError
from .options import SolverOptions, effective_precision, resolve
from .solution import Solution

# Environment of every worker unless the caller overrides it: one BLAS thread
# per solve (the pool parallelises across solves).
DEFAULT_WORKER_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

# Variables through which an MPI launcher hands a process its rank. A worker
# must start as a singleton MPI process, so they are removed from its
# environment (a parent started by `mpirun -n 1` would pass them on).
_LAUNCHER_PREFIXES = ("OMPI_", "PMIX_", "PMI_", "MV2_COMM_WORLD_")

# Rough peak memory of a worker solving the small problems this pool is for
# (Python + SDPB + one solve), used to cap the default number of workers.
WORKER_MEMORY_BYTES = 250 * 1024**2


class SolveCancelled(cf.CancelledError):
    """The solve was cancelled with :meth:`SolverPool.cancel` while it ran.

    ``solution`` is the state at the end of the last iteration (status
    ``SIGTERM_RECEIVED``, with its ``iterations`` and ``runtime_seconds``), or
    ``None`` when the worker was stopped another way.
    """

    def __init__(self, message: str, solution: Solution | None = None):
        super().__init__(message)
        self.solution = solution


class SDPBWorkerError(SDPBError):
    """A worker process died (crash, ``MPI_Abort``, killed) during a solve."""


def physical_cores() -> int:
    """Physical cores available to this process (hyper-threads counted once)."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - not Linux
        return os.cpu_count() or 1
    cores = set()
    for cpu in cpus:
        base = f"/sys/devices/system/cpu/cpu{cpu}/topology/"
        try:
            with open(base + "physical_package_id") as f:
                package = f.read().strip()
            with open(base + "core_id") as f:
                core = f.read().strip()
        except OSError:
            return len(cpus)
        cores.add((package, core))
    return max(len(cores), 1)


def available_memory() -> int | None:
    """``MemAvailable`` in bytes, or ``None`` when unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def default_workers() -> int:
    """min(physical cores, available memory / WORKER_MEMORY_BYTES), at least 1."""
    n = physical_cores()
    mem = available_memory()
    if mem is not None:
        n = min(n, max(int(mem // WORKER_MEMORY_BYTES), 1))
    return max(n, 1)


# ---------------------------------------------------------------------------
# Worker process


def _set_parent_death_signal(sig: int) -> None:
    """Linux: deliver `sig` to this process when the parent dies."""
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, int(sig), 0, 0, 0)  # PR_SET_PDEATHSIG
    except Exception:  # pragma: no cover - best effort
        pass


def _worker_main(conn: Connection, precision: int, env: Mapping[str, str]) -> None:
    for key in [k for k in os.environ if k.startswith(_LAUNCHER_PREFIXES)]:
        del os.environ[key]
    os.environ.update(env)
    # Own process group: Ctrl-C in the terminal reaches only the parent, which
    # decides what to cancel. Idle workers ignore SIGINT; during a solve SDPB's
    # handler turns it into "stop after this iteration" (see SolverPool.cancel).
    os.setpgrp()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # If the parent dies, SIGTERM makes SDPB stop at the next iteration and
    # the closed pipe then ends this loop.
    _set_parent_death_signal(signal.SIGTERM)
    try:
        import sdpb_python
        from sdpb_python.errors import wrap_cpp_error
        from sdpb_python.solver import _ext

        sdpb_python.set_precision(precision)
        ext = _ext()
        conn.send(("ready", os.getpid(), sdpb_python.sdpb_version()))
    except BaseException as exc:  # noqa: BLE001 - report anything to the parent
        conn.send(("fatal", f"{type(exc).__name__}: {exc}"))
        return
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        if message[0] == "stop":
            return
        _, task_id, kind, spec, options = message
        try:
            result = getattr(ext, "solve_" + kind)(spec, options)
            if ext.last_run_interrupted():
                conn.send(("cancelled", task_id, result))
            else:
                conn.send(("ok", task_id, result))
        except BaseException as exc:  # noqa: BLE001
            error = wrap_cpp_error(exc)
            try:
                conn.send(("error", task_id, error))
            except Exception:  # unpicklable exception
                conn.send(("error", task_id, SDPBError(f"{type(error).__name__}: {error}")))


# ---------------------------------------------------------------------------
# Parent side


@dataclass
class _Task:
    task_id: int
    kind: str
    spec: dict
    options: dict
    future: cf.Future


@dataclass
class _Worker:
    process: Any
    conn: Connection
    ready: bool = False
    pid: int | None = None
    task: _Task | None = None
    cancel_requested: bool = False
    eof: bool = False  # the pipe is closed; only the exit is left to handle
    starts: int = 1


@dataclass
class _State:
    pending: collections.deque = field(default_factory=collections.deque)
    workers: list = field(default_factory=list)
    closing: bool = False
    broken: BaseException | None = None


class SolverPool:
    """A pool of worker processes that solve :class:`PMP` / :class:`LMI` problems.

    ::

        with SolverPool(workers=4, precision=768) as pool:
            futures = [pool.submit(pmp, fast_step_length=True) for pmp in problems]
            solutions = [f.result() for f in futures]

    Args:
        workers: Number of processes; default :func:`default_workers` (physical
            cores, capped by available memory).
        precision: Working precision of every worker, in bits. ``None`` means
            the precision fixed in this process, or SDPB's default 400.
        env: Extra environment variables for the workers (on top of
            :data:`DEFAULT_WORKER_ENV`), e.g. ``{"LD_PRELOAD": ...}``.

    :meth:`submit` returns a :class:`concurrent.futures.Future` whose result is
    the :class:`Solution` (or whose exception is the :class:`SDPBError` the solve
    raised). Results do not depend on the number of workers or the order in
    which solves run.
    """

    def __init__(self, workers: int | None = None, *, precision: int | None = None,
                 env: Mapping[str, str] | None = None):
        job = _mpi.launcher_job()
        if job is not None and not os.environ.get(_mpi.OPT_OUT):
            size, var, launcher = job
            raise SDPBError(
                f"SolverPool runs its own worker processes and must not itself be one of "
                f"{size} processes started by {launcher} ({var}={size}). Start the "
                f"parent as a single process, or set {_mpi.OPT_OUT}=1.")
        self.workers = default_workers() if workers is None else int(workers)
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        self.precision = effective_precision(precision)
        self.env = dict(DEFAULT_WORKER_ENV)
        if env:
            self.env.update({str(k): str(v) for k, v in env.items()})
        self._context = multiprocessing.get_context("spawn")
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._state = _State()
        self._wake_recv, self._wake_send = self._context.Pipe(duplex=False)
        for _ in range(self.workers):
            self._state.workers.append(self._start_worker())
        self._thread = threading.Thread(target=self._dispatch, name="SolverPool", daemon=True)
        self._thread.start()

    # -- public API

    def submit(self, problem, options: SolverOptions | None = None, *,
               max_num_poles: int | None = None, **overrides: Any) -> cf.Future:
        """Queue one solve; keyword arguments override :class:`SolverOptions` fields."""
        from .lmi import LMI
        from .problem import PMP

        opts = resolve(options, overrides)
        if opts.precision is None:
            opts = opts.replace(precision=self.precision)
        elif int(opts.precision) != self.precision:
            raise ValueError(f"the pool solves at {self.precision} bits; got precision={opts.precision} "
                             f"(the precision is fixed once per worker process)")
        if isinstance(problem, PMP):
            kind, spec = "pmp", problem._to_spec(self.precision, max_num_poles)
        elif isinstance(problem, LMI):
            if max_num_poles is not None:
                raise TypeError("max_num_poles applies to PMP problems only")
            kind, spec = "lmi", problem._to_spec(self.precision)
        else:
            raise TypeError(f"cannot solve a {type(problem).__name__}; expected PMP or LMI")
        options_dict = opts._to_dict()
        future: cf.Future = cf.Future()
        task = _Task(next(self._ids), kind, spec, options_dict, future)
        with self._lock:
            if self._state.broken is not None:
                raise SDPBWorkerError(f"the pool is broken: {self._state.broken}")
            if self._state.closing:
                raise RuntimeError("cannot submit to a closed SolverPool")
            self._state.pending.append(task)
        self._wake()
        return future

    def map(self, problems: Iterable, options: SolverOptions | None = None, **overrides: Any) -> list:
        """Solve every problem; the solutions in input order (raises the first error)."""
        futures = [self.submit(p, options, **overrides) for p in problems]
        try:
            return [f.result() for f in futures]
        except BaseException:
            for f in futures:
                self.cancel(f)
            raise

    def cancel(self, future: cf.Future) -> bool:
        """Cancel a solve: drop it if it is still queued, or stop it at the end of
        its current iteration if it runs (its future then raises
        :class:`SolveCancelled`). Returns False when the solve already finished.
        A running solve that completes before the request arrives keeps its result.
        """
        if future.done():
            return False
        with self._lock:
            for task in self._state.pending:
                if task.future is future:
                    self._state.pending.remove(task)
                    future.cancel()
                    return True
            for worker in self._state.workers:
                if worker.task is not None and worker.task.future is future:
                    if not worker.cancel_requested:
                        worker.cancel_requested = True
                        try:
                            os.kill(worker.pid, signal.SIGINT)
                        except (OSError, TypeError):
                            pass
                    return True
        return future.cancel()

    def cancel_all(self) -> None:
        """Cancel every queued and running solve."""
        with self._lock:
            futures = [t.future for t in self._state.pending]
            futures += [w.task.future for w in self._state.workers if w.task is not None]
        for f in futures:
            self.cancel(f)

    def close(self, *, cancel: bool = False) -> None:
        """Finish the queued solves (or cancel them), then stop the workers."""
        if cancel:
            self.cancel_all()
        with self._lock:
            self._state.closing = True
        self._wake()
        self._thread.join()

    def terminate(self) -> None:
        """Kill the workers now. Queued solves are cancelled; running ones raise
        :class:`SolveCancelled`."""
        with self._lock:
            self._state.closing = True
            tasks = list(self._state.pending)
            self._state.pending.clear()
            workers = list(self._state.workers)
        for task in tasks:
            task.future.cancel()
        for worker in workers:
            if worker.process.is_alive():
                worker.process.kill()
        self._wake()
        self._thread.join()

    def __enter__(self) -> "SolverPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.terminate()

    @property
    def pids(self) -> list[int]:
        """Process ids of the workers that have started."""
        with self._lock:
            return [w.pid for w in self._state.workers if w.pid is not None]

    # -- internals

    def _start_worker(self) -> _Worker:
        parent_conn, child_conn = self._context.Pipe(duplex=True)
        process = self._context.Process(target=_worker_main, args=(child_conn, self.precision, self.env),
                                        name="sdpb_python-worker", daemon=True)
        process.start()
        child_conn.close()
        return _Worker(process=process, conn=parent_conn)

    def _wake(self) -> None:
        try:
            self._wake_send.send_bytes(b"x")
        except OSError:  # pragma: no cover - pool already shut down
            pass

    def _dispatch(self) -> None:
        try:
            self._dispatch_loop()
        except BaseException as exc:  # pragma: no cover - a bug in the pool
            with self._lock:
                self._state.broken = exc
            self._fail_all(SDPBWorkerError(f"SolverPool dispatcher failed: {exc!r}"))
        finally:
            for worker in self._state.workers:
                try:
                    worker.conn.send(("stop",))
                except OSError:
                    pass
            for worker in self._state.workers:
                worker.process.join(timeout=10)
                if worker.process.is_alive():
                    worker.process.kill()
                    worker.process.join()
                worker.conn.close()
            self._wake_recv.close()
            self._wake_send.close()

    def _fail_all(self, error: BaseException) -> None:
        with self._lock:
            tasks = list(self._state.pending)
            self._state.pending.clear()
            tasks += [w.task for w in self._state.workers if w.task is not None]
            for w in self._state.workers:
                w.task = None
        for task in tasks:
            if not task.future.done():
                task.future.set_exception(error)

    def _dispatch_loop(self) -> None:
        state = self._state
        while True:
            with self._lock:
                workers = list(state.workers)
            objects: list = [self._wake_recv]
            for w in workers:
                objects += [w.process.sentinel] if w.eof else [w.conn, w.process.sentinel]
            for ready in wait(objects):
                if ready is self._wake_recv:
                    while self._wake_recv.poll():
                        self._wake_recv.recv_bytes()
                    continue
                for w in workers:
                    if ready is w.conn:
                        self._receive(w)
                    elif ready is w.process.sentinel:
                        self._handle_exit(w)
            self._assign()
            with self._lock:
                idle = all(w.task is None for w in state.workers)
                if (state.closing and not state.pending and idle) or state.broken is not None:
                    break

    def _receive(self, worker: _Worker) -> None:
        try:
            message = worker.conn.recv()
        except (EOFError, OSError):
            worker.eof = True  # the sentinel reports the exit
            return
        self._handle_message(worker, message)

    def _handle_message(self, worker: _Worker, message: tuple) -> None:
        kind = message[0]
        if kind == "ready":
            worker.ready, worker.pid = True, message[1]
            return
        if kind == "fatal":
            error = SDPBWorkerError(f"a worker failed to start: {message[1]}")
            with self._lock:
                self._state.broken = error
            self._fail_all(error)
            return
        with self._lock:
            task, worker.task = worker.task, None
            worker.cancel_requested = False
        if task is None or task.task_id != message[1]:  # pragma: no cover - protocol bug
            return
        if kind == "ok":
            try:
                task.future.set_result(Solution._from_dict(message[2]))
            except Exception as exc:  # noqa: BLE001
                task.future.set_exception(exc)
        elif kind == "cancelled":
            try:
                partial = Solution._from_dict(message[2])
            except Exception:  # noqa: BLE001
                partial = None
            task.future.set_exception(SolveCancelled("solve cancelled while running", partial))
        else:
            task.future.set_exception(message[2])

    def _handle_exit(self, worker: _Worker) -> None:
        # Messages sent just before the exit are still in the pipe.
        while not worker.eof:
            try:
                if not worker.conn.poll():
                    break
                message = worker.conn.recv()
            except (EOFError, OSError):
                break
            self._handle_message(worker, message)
            if self._state.broken is not None:
                return
        worker.process.join()
        code = worker.process.exitcode
        with self._lock:
            task, worker.task = worker.task, None
            closing = self._state.closing
            index = self._state.workers.index(worker)
        if task is not None and not task.future.done():
            if worker.cancel_requested or closing:
                task.future.set_exception(SolveCancelled("worker stopped during the solve"))
            else:
                task.future.set_exception(SDPBWorkerError(
                    f"worker process {worker.pid} exited with code {code} during the solve"))
        if not worker.ready:
            error = SDPBWorkerError(f"a worker process exited with code {code} before it was ready")
            with self._lock:
                self._state.broken = error
            self._fail_all(error)
            return
        worker.conn.close()
        if closing:
            with self._lock:
                self._state.workers.remove(worker)
            return
        replacement = self._start_worker()
        replacement.starts = worker.starts + 1
        with self._lock:
            self._state.workers[index] = replacement

    def _assign(self) -> None:
        with self._lock:
            for worker in self._state.workers:
                if not self._state.pending:
                    return
                if not worker.ready or worker.task is not None or not worker.process.is_alive():
                    continue
                while self._state.pending:
                    task = self._state.pending.popleft()
                    if task.future.set_running_or_notify_cancel():
                        break
                else:
                    return
                worker.task = task
                worker.cancel_requested = False
                try:
                    worker.conn.send(("solve", task.task_id, task.kind, task.spec, task.options))
                except OSError:
                    # The worker died; _handle_exit fails the task and restarts it.
                    pass


def solve_many(problems: Sequence, options: SolverOptions | None = None, *, workers: int | None = None,
               precision: int | None = None, env: Mapping[str, str] | None = None,
               **overrides: Any) -> list:
    """Solve independent problems on a temporary :class:`SolverPool`; the
    solutions in input order."""
    problems = list(problems)
    n = min(default_workers() if workers is None else workers, max(len(problems), 1))
    with SolverPool(n, precision=precision, env=env) as pool:
        return pool.map(problems, options, **overrides)
