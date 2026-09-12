# sdpb-python binding API — design and implementation plan

Status: proposal (2026-09-10). Sources: `c-src/sdpb/architecture.md`,
`c-src/sdpb/api-doc.md`, and the headers under `c-src/sdpb/src/`.

## 1. Context and decisions

The current package only shells SDPB's `main()` through CLI-style arguments and
parses `out.txt`. The goal is a real Python API for building and solving
problems in memory.

Decisions already taken:

| Topic | Decision |
|---|---|
| Inputs | In-memory polynomial matrix programs (PMP) and in-memory linear matrix inequalities (LMI). File inputs (`pmp.json`, `sdp/`) and post-processing tools (`spectrum`, `approx_objective`) are out of scope for now. |
| Fork | Patches to `YPWang0818/sdpb` are allowed (branch `python-api`, kept upstream-mergeable). |
| Numbers | `mpmath.mpf` at the Python boundary. |
| Parallelism | Single process, single MPI rank; the package starts no processes of its own. MPI is initialised because SDPB requires it, and `mpirun -n>1` is rejected. See §2.1. |

Constraints from the C++ side that shape the design (api-doc §2):

- One `Environment` per process; every SDPB object holding communicators must
  die before it. Precision is a process-global set by `Environment::set_precision`
  and must be set before any `BigFloat` (including `Solver_Parameters` defaults)
  is created.
- `Solver_Parameters` has no in-class defaults; defaults live in `options()`.
- An empty `checkpoint_in` makes `SDP_Solver` look for checkpoints in the CWD.
- `SDP` has no route from a `Polynomial_Matrix_Program` without a disk round trip
  (api-doc §3.5); the in-memory `SDP` constructor only supports `num_points == 1`.
- Errors are `std::runtime_error` with a stack trace in the message. In a single
  process, catching them is safe.
- `Environment` installs its own SIGTERM handler; `run()` polls it each iteration.
- **The bootstrap fork of Elemental allows `SetPrecision` exactly once per
  process** (`src/core/imports/gmp.cpp` throws "Not allowed to call
  SetPrecision twice"). Precision is therefore fixed by the first call that
  needs it and cannot change afterwards; a different precision needs a new
  process. (api-doc §2 suggests re-calling `set_precision`; that does not hold
  for this Elemental.)

## 2. Architecture

```
Python (src/sdpb_python/)
  problem.py   DampedRational, Polynomial, PolynomialMatrix, PMP     (pure Python)
  lmi.py       LMI                                                    (pure Python)
  options.py   SolverOptions                                          (pure Python)
  solution.py  Solution, TerminateReason, BlockInfo                   (pure Python)
  numbers.py   to_str / from_str  <->  mpmath.mpf                     (pure Python)
  io.py        read_pmp_json / write_pmp_json                         (pure Python)
  _sdpb.pyx    thin Cython over the shim: lists of str in, lists of str out
Cython shim (src/sdpb_python/cpp/)
  sdpb_wrapper.{hxx,cxx}   Session (Environment + precision), solve_pmp, solve_lmi,
                           Solver handle, plain-type structs
Fork (c-src/sdpb, branch python-api)
  Block_Info(env, dims, num_points, verbosity)
  SDP(f, b, groups, normalization, block_info, grid)
  Environment::request_termination()
```

Design rule: everything crossing the Cython boundary is `std::string`,
`std::vector`, `int64_t`, `bool`, or a struct of those. No Elemental, Boost or
GMP types are visible to Cython. This keeps `.pxd` files trivial and lets the C++
shim be unit-tested on its own.

### 2.1 Parallelism: what the package does and does not do

The package adds **no process-level parallelism**. A solve runs in the calling
process, on the calling thread. There is no `multiprocessing`, no
`concurrent.futures`, no `fork`, no worker subprocess, and no MPI rank beyond
the one the caller is already running in. Starting several solves at once is
the caller's business, not the library's.

MPI is inherited plumbing, not a feature of this API. SDPB is written as an MPI
program: it spreads SDP blocks over ranks, uses Elemental for the distributed
linear algebra, and `Environment::initialize()` builds a shared-memory
communicator and broadcasts to work out how many nodes the job spans. None of
that links or runs without an MPI library, so the shim keeps one `Environment`
alive for the life of the module (Elemental may be initialised only once per
process) and every entry point calls `require_single_rank()` first. The
collectives still execute, over a group of one, which costs almost nothing and
keeps the code identical to the cluster build. Results are copied out through
`DistMatrix<STAR,STAR>` for the same reason.

What does run in parallel sits below the API: the bundled OpenBLAS is a
pthread build (`libopenblasp`), so the double-precision GEMMs inside a solve
use every core, under `OPENBLAS_NUM_THREADS`. Users who want several solves at
once run independent Python processes, each with its own `output_dir`.

Two consequences follow, both implemented:

- **A second MPI in the process is unsafe.** The wheels bundle their own MPICH;
  loading another MPI, e.g. `mpi4py` built against Open MPI, in the same
  process is unsupported.
- **A foreign launcher cannot be caught by `require_single_rank()`.** Under an
  `mpirun` from a different MPI than the package links, each process
  initialises MPI alone and sees a world of one, so the C++ check never fires
  and the processes silently duplicate the solve and overwrite each other's
  output (verified: four Open MPI ranks running the MPICH wheel left every
  `iterations.json` in a shared directory unparseable). `_mpi.py` therefore
  compares the job size the launcher advertises in the environment
  (`OMPI_COMM_WORLD_SIZE`, `PMI_SIZE`, `MV2_COMM_WORLD_SIZE`, `SLURM_NTASKS`,
  `SLURM_STEP_NUM_TASKS`) with the world the extension reports, and raises
  before any work starts. `SDPB_PYTHON_ALLOW_MULTI_PROCESS=1` opts out, for
  deliberately independent processes such as a sweep. A real multi-rank world
  stays the C++ check's to reject.

If the single-rank restriction is ever lifted, this section is the list of
things that have to change: the `require_single_rank()` calls, the launcher
check, and the assumption that every block lives on the caller's rank.

## 3. Numbers at the boundary

Decimal strings, lossless in both directions:

- C++ → Python: `El::BigFloat` printed with `set_stream_precision`, i.e.
  `ceil(prec·log10 2) + 1` digits (`max_digits10`), which uniquely identifies the
  binary value. Python parses with `mpmath.workprec(bits)` so nothing is rounded.
- Python → C++: any of `int`, `float`, `str`, `fractions.Fraction`, `mpmath.mpf`
  is converted with `mpmath.mpf` at the solver precision and printed with
  `mpmath.nstr(x, dps + 2)` (or `Fraction`/`int` printed exactly); the shim
  parses with `El::BigFloat(std::string)` after `set_precision`.

`Solution` values are `mpmath.mpf` from the global `mpmath.mp` context. Users
who continue in high precision set `mpmath.mp.prec` themselves; the docstring
says so and `Solution.precision` records the bits used.

**Precision is per process.** `sdpb_python.set_precision(bits)` may be called
once; otherwise the first `solve()`, `to_sdp()` or `sampled()` fixes it (from
its `precision` argument, default 400). `SolverOptions.precision=None` means
"the fixed precision". A later request for a different value raises
`SDPBError("precision is already fixed at N bits ...")`. Consequences for the
test-suite are in §9.

## 4. Public Python API

### 4.1 Problem description (`problem.py`)

```python
DampedRational(constant=1, base=exp(-1), poles=())      # constant * base**x / prod(x - p)
Polynomial(coeffs)                                       # a0 + a1 x + ...; also .from_sympy()
PolynomialMatrix(polynomials, *, prefactor=None, reduced_prefactor=None,
                 max_num_poles=None, sample_points=None, sample_scalings=None,
                 reduced_sample_scalings=None, bilinear_basis=None)
    # polynomials[r][s] is a sequence of N+1 Polynomial (or coefficient lists);
    # dim x dim, symmetric. Optional fields mirror pmp.json exactly (api-doc §5.2).
PMP(objective, normalization=None, matrices=())          # Manual eq. (3.1)
PMP.solve(options=None, **overrides) -> Solution
```

Validation in Python before crossing: symmetry of `polynomials`, every vector
has length `N+1`, `normalization` length `N+1`, sample-field lengths consistent
with each other. C++ `validate()` remains the final authority.

Introspection (needed by the tests in §9, useful for users too):

```python
PolynomialMatrix.sampled(precision) -> SampledMatrix
    # what the C++ constructor filled in: prefactor, reduced_prefactor, sample_points,
    # sample_scalings, reduced_sample_scalings, bilinear_basis (even/odd polynomials)
PMP.to_sdp(precision, max_num_poles=None) -> SDPData
    # the Output_SDP result: objective_const (b_0), b, normalization, and per block
    # dim, num_points, bilinear_bases[even|odd] (sampled matrices), c, B
    # — exactly the content of an sdp/ directory written by pmp2sdp.
```

### 4.2 Linear matrix inequalities (`lmi.py`)

```python
LMI(b, blocks, f=0)      # maximize f + b.y  s.t.  M0 + sum_n y_n M_n >= 0 per block
    # blocks: list of (M_0, M_1, ..., M_N) symmetric matrices (nested lists / numpy / mpmath.matrix)
LMI.solve(options=None, **overrides) -> Solution
```

Maps to the existing in-memory `SDP` constructor (api-doc §3.4):
`c_p = M_0[r,s]`, `B[p,n] = −M_{n+1}[r,s]`, rows ordered `s` outer, `r ≤ s`
inner. Minimisation is `LMI(-b, ...)`. `Solution.Y[j]` is then
`M_0 + Σ y_n M_n` for block `j`.

### 4.3 Options (`options.py`)

`SolverOptions` is a dataclass with SDPB's defaults (api-doc §8.7) plus:

| Field | Default | Maps to |
|---|---|---|
| `precision` | 400 | `Environment::set_precision`, `Solver_Parameters::precision` |
| `max_iterations`, `max_runtime`, `duality_gap_threshold`, `primal_error_threshold`, `dual_error_threshold`, `initial_matrix_scale_primal/dual`, `feasible/infeasible_centering_parameter`, `step_length_reduction`, `max_complementarity`, `min_primal_step`, `min_dual_step`, `find_primal/dual_feasible`, `detect_primal/dual_feasible_jump`, `max_shared_memory_bytes` | SDPB defaults | same-named `Solver_Parameters` fields |
| `checkpoint_dir` | `None` | `checkpoint_in`/`checkpoint_out`; `None` → a fresh temp dir for `checkpoint_in` and `""` (disabled) for `checkpoint_out` |
| `checkpoint_interval` | 3600 s | `checkpoint_interval` |
| `verbosity` | `"none"` | `Verbosity` |
| `output_dir` | `None` | if set, `iterations.json` and `c_minus_By/` are written there (as `sdpb` does) |
| `want` | `("y",)` | which solution parts to return: any of `"x"`, `"y"`, `"z"`, `"X"`, `"Y"`, `"c_minus_By"` |

Keyword overrides on `solve(**overrides)` are the same names.

### 4.4 Results (`solution.py`)

```python
class TerminateReason(Enum): PRIMAL_DUAL_OPTIMAL, PRIMAL_FEASIBLE, DUAL_FEASIBLE,
    PRIMAL_FEASIBLE_JUMP_DETECTED, DUAL_FEASIBLE_JUMP_DETECTED,
    MAX_COMPLEMENTARITY_EXCEEDED, MAX_ITERATIONS_EXCEEDED, MAX_RUNTIME_EXCEEDED,
    PRIMAL_STEP_TOO_SMALL, DUAL_STEP_TOO_SMALL, SIGTERM_RECEIVED

@dataclass
class Solution:
    status: TerminateReason;  optimal: bool
    primal_objective, dual_objective, duality_gap, primal_error, dual_error: mpf
    y: list[mpf]                     # length N
    z: list[mpf] | None              # length N+1, only when the PMP had a normalization
    x: list[list[mpf]] | None        # per block, when requested
    X, Y: list[list[mpmath.matrix]] | None   # per block: [even, odd] PSD blocks, when requested
    c_minus_By: list[list[mpf]] | None
    iterations: int;  runtime_seconds: float;  precision: int
    blocks: list[BlockInfo]          # dim, num_points, schur size per block
```

`z` is computed in the shim with `fill_weights` + `max_normalization_index`
(api-doc §3.7). `x`, `X`, `Y` are copied out of the solver's block containers
(local matrices in a single process; the shim still goes through
`DistMatrix<STAR,STAR>` so it stays MPI-correct).

### 4.5 Solver handle (`handle.py`)

```python
solver = PMP.solver(options)   # builds Block_Info, SDP, SDP_Solver; holds them (also LMI.solver)
solver.run(**overrides) -> Solution     # continues the iteration; may be called repeatedly
solver.state(want=...) -> Solution      # current iterate without running
solver.warm_start(y=..., X=..., Y=...)  # writes into solver.y / X / Y before run()
solver.save_checkpoint(dir); solver.total_iterations; solver.close()  # or a `with` block
```

Backed by `SDP_Solver::run()` being re-callable (api-doc §8.5). The handle
records its precision and refuses `run()` if the global precision has since
changed. `close()` (and `__del__`) frees the C++ objects; the session's
`atexit` finalizer refuses to finalize MPI while handles are alive and warns.

### 4.6 Errors and signals

- `SDPBError(RuntimeError)`: `.message` is the first line of the C++ message;
  `.details` keeps the full text including SDPB's stack trace.
- Ctrl-C during a solve: the shim installs a SIGINT handler for the duration of
  `run()` that calls `Environment::request_termination()` (fork patch P3); the
  solver returns `SIGTERM_RECEIVED` at the end of the current iteration and
  Python raises `SolverInterrupted` (a `KeyboardInterrupt`) carrying the partial
  `Solution`; the handle can `run()` again afterwards (patch P5).
- The GIL is released during `run()`; a module-level lock serialises solves
  because SDPB is not reentrant.
- MPI size > 1 raises at import of the session with a clear message.

## 5. C++ shim (`src/sdpb_python/cpp/`)

Plain-type structs mirroring §4:

```cpp
namespace sdpb_python {
struct Damped_Rational_Spec { std::string constant, base; std::vector<std::string> poles; };
struct Polynomial_Matrix_Spec {
  size_t dim; std::vector<std::vector<std::vector<std::vector<std::string>>>> polynomials; // [r][s][n][k]
  std::optional<Damped_Rational_Spec> prefactor, reduced_prefactor;
  std::optional<int64_t> max_num_poles;
  std::optional<std::vector<std::string>> sample_points, sample_scalings, reduced_sample_scalings;
  std::optional<std::array<std::vector<std::vector<std::string>>, 2>> bilinear_basis;
};
struct PMP_Spec { std::vector<std::string> objective; std::optional<std::vector<std::string>> normalization;
                  std::vector<Polynomial_Matrix_Spec> matrices; };
struct LMI_Spec { std::string f; std::vector<std::string> b;
                  std::vector<std::vector<std::vector<std::vector<std::string>>>> blocks; }; // [j][n][r][s]
struct Solver_Options { /* every Solver_Parameters field as int64/bool/string */
                        std::string verbosity, checkpoint_in, checkpoint_out, output_dir; unsigned want; };
struct Solution_Data { std::string terminate_reason, primal_objective, dual_objective, duality_gap,
                       primal_error, dual_error; std::vector<std::string> y, z;
                       std::vector<std::vector<std::string>> x, c_minus_By;
                       std::vector<std::vector<Matrix_Data>> X, Y;  // Matrix_Data = {h, w, column-major strings}
                       int64_t iterations, runtime_ms; size_t precision;
                       std::vector<size_t> dims, num_points; };

void set_precision(size_t bits); size_t precision();
Solution_Data solve_pmp(const PMP_Spec &, const Solver_Options &);
Solution_Data solve_lmi(const LMI_Spec &, const Solver_Options &);
class Solver { /* milestone 3: owns Block_Info, El::Grid, SDP, SDP_Solver, Timers */ };
}
```

Internals of `solve_pmp` (single process, all blocks local):

1. `set_precision(bits)`; build `Solver_Parameters` via the §3.6 helper
   (`default_solver_parameters()` lives in the shim), then override from options.
2. Build `Polynomial_Vector_Matrix` for each spec (constructor fills sample
   points, scalings, bases), then `Polynomial_Matrix_Program` with
   `matrix_index_local_to_global = 0..J-1` and label paths `in-memory/block_j`.
3. `Output_SDP out(pmp, {"sdpb_python"}, timers)` — applies the normalization
   and produces `Dual_Constraint_Group`s.
4. Fork patch P1: `Block_Info block_info(env, dims, num_points, verbosity)`.
5. Fork patch P2: `SDP sdp(out.objective_const, out.dual_objective_b,
   out.dual_constraint_groups, out.normalization, block_info, grid)`.
6. `SDP_Solver solver(params, verbosity, false, block_info, grid, N)`;
   `solver.run(env, params, verbosity, to_property_tree(params), block_info, sdp,
   grid, start, iterations_json_path, timers, block_timings_ms)`.
7. Extract results; convert to strings with `set_stream_precision`.

`solve_lmi` skips steps 2–5 and uses the existing in-memory `SDP` constructor
with `yp_to_y = I`, `primal_c_scale = 1`, `normalization = (1, 0, …, 0)` and
`Block_Info(env, dims, verbosity)`.

The existing `run(args)` passthrough (`sdpb` CLI in-process) stays as
`sdpb_python.legacy.solve_dir` so the current end-to-end test keeps working.

## 6. Fork patches (branch `python-api` of `YPWang0818/sdpb`)

Each patch is small, self-contained, and comes with a Catch2 unit test in
`test/src/unit_tests/` so it can be offered upstream.

**P1 — `Block_Info` in-memory constructor with `num_points`** (`src/sdp_solve/Block_Info.hxx`, `Block_Info/Block_Info.cxx`)

```cpp
Block_Info(const Environment &env, const std::vector<size_t> &dimensions,
           const std::vector<size_t> &num_points, const size_t &proc_granularity,
           const Verbosity &verbosity);
```
Costs `get_schur_block_size(b)²` as the existing dims-only constructor does
(which becomes a delegation with `num_points = 1`).

**P2 — `SDP` from `Dual_Constraint_Group`s** (`src/sdp_solve/SDP.hxx`, `SDP/SDP.cxx`)

```cpp
SDP(const El::BigFloat &objective_const, const std::vector<El::BigFloat> &dual_objective_b,
    const std::vector<Dual_Constraint_Group> &groups,              // one per block_info.block_indices, same order
    const std::optional<std::vector<El::BigFloat>> &normalization,
    const Block_Info &block_info, const El::Grid &grid);
```
Implemented by reusing the existing scatter path: convert each group to an
`SDP_Block_Data` (it has the same fields; `bases_blocks` via
`set_bilinear_bases_block_local`) and call the logic of `set_sdp_from_root`
(`SDP/read_block_data/read_block_data.cxx:26`), which is moved out of its
anonymous namespace into a header under `SDP/read_block_data/`. Finish with
`validate(block_info)`. Unit test: build the SDP both ways from
`test/data/end-to-end_tests/1d` (via `write_sdp` to a temp dir and via the new
constructor) and compare every block matrix.

**P3 — `Environment::request_termination()`** (`src/sdpb_util/Environment.{hxx,cxx}`)

Static; sets the existing `sigterm_flag`. Lets a host stop the solver cleanly.

**P4 — `SDP_Solver::num_iterations`** (`src/sdp_solve/SDP_Solver.hxx`, `run/run.cxx`)

Public counter of completed iterations of the last `run()` (`Timers` cannot be
enumerated, and `iterations.json` is optional).

**P5 — `Environment::clear_termination_request()`**

The SIGTERM flag is a file-static that nothing resets; without this, a solver
stopped once could never run again in the same process.

Not patched: `Solver_Parameters` defaults (helper lives in the shim), the
`write_control_json` declaration mismatch (unused here).

## 7. Build changes

- `setup.py`: add `pmp` to the static libraries; link order
  `sdp_solve, pmp2sdp_lib, pmp, sdpb_util` (api-doc §1.1). Add the new shim
  sources. Keep the four `src/sdpb/*.cxx` sources for the legacy passthrough.
- `pyproject.toml`: runtime dependency `mpmath>=1.3`; optional `numpy`,
  `sympy` extras for input conveniences.
- `.gitmodules` / submodule pointer: track branch `python-api`.

## 8. Milestones

Status (2026-09-11): M1–M3 implemented on fork branch `python-api`
(patches P1–P5) and in `src/sdpb_python/`; tiers T1–T7 pass (T4's slow
datasets take ~17 minutes with `--run-slow`).

**M1 — in-memory PMP → `Solution`** (core)
Fork P1–P3 with their unit tests (§9.4); shim `solve_pmp`; Python
`problem.py` (including `sampled()` and `to_sdp()`), `options.py`,
`solution.py`, `numbers.py`; `io.read_pmp_json`; test utilities (§9.2).
Tests: tiers T1–T4 of §9.3 (the slow datasets may land after the fast ones).

**M2 — LMI**
`lmi.py`, shim `solve_lmi`. Tests: tier T5.

**M3 — solver handle and extras**
`Solver` handle with `run()` re-entry, warm start, checkpoints; `X`, `Y`
extraction; `io.write_pmp_json` for interoperability with the CLI tools
(`pmp2sdp`, `spectrum`); SIGINT → clean stop; `Polynomial.from_sympy`.
Tests: tiers T6 and T7.

**Later / out of scope now**: MPI (`mpirun -n N python`), `spectrum` and
`approx_objective` wrappers, file readers through `pmp_read`.

## 9. Testing

The Python suite mirrors the fork's own suite (`c-src/sdpb/test/`) stage by
stage and reuses its reference data, so a Python result is checked against the
same numbers the C++ binaries are checked against.

### 9.1 What the fork does

| Fork test | Mechanism | Tolerance |
|---|---|---|
| `unit_tests/pmp_sampling` | sample points, scalings, bilinear bases vs Mathematica values for `exp(-x)`, `exp(-x)/x/(x+1)` (degree 4) and `prefactor=1, degree=0`; crash tests over poles `{}, {0}, {0,0}, {-1}, {0,-1}, {0,-1,-2}` × degree `0,1,2,10` | 16 bits |
| `integration_tests/end-to-end` | per dataset: `pmp2sdp` → `diff_sdp` vs `output/sdp` (objectives, normalization, `pmp_info`, per block dim/num_points/bases/c/B) → `sdpb` → `check_c_minus_By` (recompute `c − B·y` from the sdp and `y.txt`) → `diff_sdpb_output_dir` (`out.txt` keys terminateReason/primalObjective/dualObjective, `y.txt`, `x_*.txt`, `z.txt`, `iterations.json` fields except timings and `block_name`, `c_minus_By.json`) → `spectrum` | solve at 768 bits (664 for `1d`), compare at 99 bits, i.e. `|a−b| < 2^-99 (|a|+|b|)`; small errors (`P-err`, `D-err`, …) skipped when both are below `2^-49` |
| `integration_tests/pmp2sdp` | json and Mathematica inputs vs `sdp_orig`; output formats; filesystem error cases; duplicate/conflicting objectives across NSV files | 392 of 512 bits |
| `integration_tests/sdbp` | profiling output; corrupted input archive; write checkpoint with `maxIterations=1` then restart from it; unreadable and corrupted checkpoints must fail with specific messages | — |
| other unit tests (block mapping, BLAS scheduling, bigint syrk, serialization, JSON) | internal to the solver | not applicable to the binding |

### 9.2 Test utilities (`tests/util/`, the analogue of `test/src/test_util`)

- `diff.py`: `assert_close(a, b, bits)` implementing the fork's relative
  binary-precision comparison for `mpf`, vectors and matrices; `Precision`
  context manager setting `mpmath.mp.prec` and the comparison bits together
  (analogue of `Float_Binary_Precision`).
- `reference.py`: pure-Python readers for the fork's reference files:
  `out.txt`, `y.txt`/`x_*.txt`/`z.txt` (text matrices: `"h w"` then rows),
  `iterations.json`, `c_minus_By/c_minus_By.json`, and an SDP directory
  (`control.json`, `objectives.json`, `normalization.json`, `pmp_info.json`,
  `block_info_*.json`, `block_data_*.json`). Every reference SDP in the fork
  uses JSON block data, so the Boost binary format is never needed.
- `datasets.py`: one record per fork dataset (name, input files, precision,
  solver arguments copied from `end-to-end.test.cxx`, keys to compare, marks).
  NSV lists are expanded in Python; only JSON inputs are supported, so datasets
  whose only input is `.xml` or `.m` are listed as skipped with the reason.

### 9.3 Test tiers

**T1 — number and I/O round trips** (analogue of `Boost_Float`/`json` tests)
- `mpf → str → BigFloat → str → mpf` is the identity at 400, 664, 768, 1024 bits,
  including extreme exponents and the values in `1d/output/out/out.txt`.
- `read_pmp_json` on every `input/*.json` in the fork's data, then
  `write_pmp_json` and read back: identical PMP objects.

**T2 — sampling** (analogue of `pmp_sampling`)
- `PolynomialMatrix.sampled()` reproduces the three Mathematica cases from
  `pmp_sampling.test.cxx` at 16 bits, including the row-sign convention the
  fork applies to the bases.
- Crash tests: the same poles × degree grid must construct without error.

**T3 — PMP → SDP conversion** (analogue of the `pmp2sdp` stage and `diff_sdp`)
- For every dataset with `output/sdp`: `PMP.to_sdp(precision)` equals the
  reference directory at 99 bits (objective constant, `b`, normalization when
  present, per block dim, num_points, even/odd bases, `c`, `B`).
  `pmp2sdp/json` is compared at 392 of 512 bits like the fork does.
- `1d` variants (`pmp.json`, `pmp-no-optional-fields.json`,
  `pmp-sample-points.json`, `pmp-all-sampling-fields.json`) all produce the same
  SDP: this exercises every optional field of `PolynomialMatrix`.
- `SingletScalar_cT_test_nmax6/primal_dual_optimal_reduced` covers per-block
  `reduced_prefactor`, and `…_max_num_poles_14` covers `max_num_poles=14`
  passed at conversion time.
- Error cases (analogue of pmp2sdp's `objectives are different` and validation):
  asymmetric matrix, polynomial vectors of unequal length, normalization of the
  wrong length, and two input files with conflicting objectives raise
  `ValueError`/`SDPBError` with a clean first line.

**T4 — end-to-end solve** (analogue of `end-to-end_tests`), parametrised over
the datasets, each run with the dataset's own solver arguments:

| Dataset | Input | Precision | Compared | Notes |
|---|---|---|---|---|
| `1d` (4 variants) | json | 664 | out.txt keys, `y`, `x_0`, `c−B·y` | fast; default suite |
| `1d-old-sampling` | pmp.json | 768 | same | explicit sampling data |
| `1d-duplicate-poles` | json | 768 | out.txt keys, `y` only | fork notes `x` may differ |
| `1d-isolated-zeros` | pmp.nsv → 7 json files | 768 | out.txt, `y`, `x_0..x_6`, `c−B·y` | multi-file NSV; single-process in the fork too |
| `SingletScalar_cT/primal_dual_optimal` | nsv → 13 json | 768 | out.txt, `y`, `z`, `x_0..x_10`, `c−B·y`, `iterations.json` fields | `slow` mark; also run twice via the `Solver` handle to mirror `run_sdpb_twice` |
| `…/primal_dual_optimal_reduced`, `…_max_num_poles_14` | nsv | 768 | same | `slow` |
| `SingletScalarAllowed/primal_feasible_jump` | nsv | 768 | terminateReason, primal/dual objective, gap, dualError; `y`, `z` | `detect_*_feasible_jump=True`, `max_shared_memory_bytes=100.1K` as in the fork; `slow` |
| `SingletScalarAllowed/dual_feasible_jump` | nsv | 768 | as above with primalError | `slow` |
| `1d-constraints`, `dfibo-…` | xml only | — | skipped | no XML reader in scope; revisit if `pmp_read` is wrapped |

Checks per dataset, in the fork's order: `to_sdp()` vs `output/sdp` (T3 reuse),
`solve(want=("x","y","z","c_minus_By"), output_dir=tmp)`, `c − B·y` recomputed
in Python from `to_sdp()` and `Solution.y` equals `Solution.c_minus_By` and the
reference `c_minus_By.json`, `Solution` fields vs `out.txt`, `y`/`x`/`z` vs the
text files, and `iterations.json` written to `output_dir` vs the reference with
the fork's skip rules. The `spectrum` stage is out of scope and not compared.

**T5 — LMI** (no fork analogue; analytic)
- `max y s.t. [[1, y],[y, 1]] ⪰ 0` → `y = 1`, `Y` has a zero eigenvalue.
- A two-variable problem with a closed-form optimum.
- LMI equals a PMP whose polynomials are constants (the fork's
  `prefactor=1, degree=0` special case has a single sample point at 0 and
  bases `[[1]]`, `0×1`), compared at 99 bits.

**T6 — solver handle and checkpoints** (analogue of `sdpb/io_tests`), on `1d`
at 1024 bits as the fork uses:
- `Solver.run(max_iterations=1)` with `checkpoint_dir`, then a fresh `Solver`
  restarted from it, equals an uninterrupted run at 99 bits.
- Restart from a checkpoint whose `X_matrix_0.txt` is truncated after two lines
  raises `SDPBError` containing `Corrupted data in file`; an unreadable file
  raises one containing `Unable to open checkpoint file`.
- `run()` called twice on one handle with a tighter `duality_gap_threshold`
  continues rather than restarts (`iterations` grows, objective unchanged).
- SIGINT during a long run returns `SIGTERM_RECEIVED` and raises
  `KeyboardInterrupt` with the partial solution.

**T7 — process-level**
- `mpirun -n 2 python -c "import sdpb_python"` exits with the single-process
  message rather than hanging (subprocess test).
- Leak check: 100 solves of `1d` in one process keep RSS flat.
- The legacy passthrough test (`tests/test_solve.py`) keeps passing.

### 9.4 Fork-side tests for the patches (Catch2, in `test/src/unit_tests/cases/`)

- `sdp_in_memory.test.cxx` (P1 + P2): read `1d/input/pmp.json` with
  `read_polynomial_matrix_program`, build `Output_SDP`, write it with
  `write_sdp` to a temp dir, then construct `SDP` from the directory and from
  the groups; `DIFF` every block matrix, `dual_objective_b`, `objective_const`
  and `normalization` at `diff_precision = -1` (exact). Runs on rank 0 only,
  like `pmp_sampling`.
- `environment.test.cxx` (P3): `request_termination()` makes
  `sigterm_received()` true.
- `./test/run_all_tests.sh` stays green on the `python-api` branch.

### 9.5 Running

- The pytest process runs at one precision, `TEST_PRECISION = 768` (as the
  fork's `unit_tests` binary does). References made at other precisions are
  compared at their `diff_precision`; where that is not enough because SDPB's
  sample points converge only to a precision-dependent accuracy
  (`pmp2sdp/json` at 512 bits) or to reproduce a dataset's native run
  (`1d` at 664 bits), the check runs in a subprocess via
  `tests/util/subproc.run_at_precision`.
- `pytest` runs T1–T3, T5, T6 and the fast T4 datasets (about a minute).
- `pytest --run-slow` adds the SingletScalar datasets; they take minutes each
  in a single process (the fork gives them 6 ranks), so CI runs them nightly.
- The fork's data is read from the submodule; no copies are kept in this repo.
