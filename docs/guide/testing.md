# Testing

The test-suite mirrors SDPB's own (`c-src/sdpb/test/`) stage by stage and
reuses its reference data, so a Python result is checked against the same
numbers the C++ binaries are checked against, with the same tolerances.

```
pytest                 # about a minute; skips extension tests if it is not built
pytest --run-slow      # adds the SingletScalar datasets, about 17 minutes
```

| Tests | Mirrors | What is checked |
|---|---|---|
| `test_numbers.py`, `test_io.py` | `Boost_Float`/`json` unit tests | number round trips through C++; `pmp.json` read/write over every reference input; error cases |
| `test_sampling.py` | `pmp_sampling` unit test | sample points, scalings and bases against the fork's Mathematica values (16 bits); crash tests over poles and degrees |
| `test_pmp_to_sdp.py` | `pmp2sdp` stage, `diff_sdp` | `PMP.to_sdp()` against reference `sdp/` directories at 99 of 768 bits |
| `test_end_to_end.py` | `end-to-end_tests` | solve each dataset with the fork's own solver arguments; compare `out.txt` keys, `y`, `x`, `z`, `c - B·y` (recomputed and against the reference), and `iterations.json` field by field |
| `test_lmi.py` | (analytic) | closed-form LMI optima; LMI equals a constant PMP |
| `test_solver_handle.py` | `sdpb/io_tests` | checkpoint restart and corruption, run continuation, warm start |
| `test_process.py`, `test_precision.py` | (process level) | precision lock, Ctrl-C, `mpirun -n 2` rejection, leak check, sympy |

Comparisons use SDPB's rule $|a - b| < 2^{-p}(|a| + |b|)$ from
`tests/util/diff.py`. The pytest process runs at 768 bits; checks that must
reproduce a reference made at another precision run in a subprocess
(`tests/util/subproc.py`).

The fork's own suite is run with `c-src/sdpb/test/run_all_tests.sh` (unit
tests under 6 MPI ranks plus integration tests, about 8 minutes); it includes
unit tests for the patches the binding relies on.
