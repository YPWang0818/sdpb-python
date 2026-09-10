# Files and interoperability

## `pmp.json`

SDPB's JSON input format is read and written directly:

```python
from sdpb_python import read_pmp_json, write_pmp_json
pmp = read_pmp_json("pmp.json")
pmp = read_pmp_json("a.json", "b.json")      # several files, matrices concatenated
pmp = read_pmp_json("files.nsv")             # NUL-separated list of files, expanded recursively
write_pmp_json(pmp, "out.json", precision=768)
```

As in SDPB, every file may carry `objective` and `normalization`; they must
agree across files. The optional per-matrix keys (`prefactor` or the legacy
`DampedRational`, `reducedPrefactor`, `maxNumPoles`, `samplePoints`,
`sampleScalings`, `reducedSampleScalings`, `bilinearBasis`,
`bilinearBasis_0`, `bilinearBasis_1`) map onto the fields of
{class}`~sdpb_python.PolynomialMatrix`.

Mathematica (`.m`) and XML inputs are not supported; convert them with SDPB's
`pmp2sdp` or write JSON from Mathematica with `SDPB.m`.

## Working with SDPB's command line tools

- **`pmp2sdp`**: `pmp.to_sdp()` gives the same data in memory. To feed a
  problem built in Python to the command line tools, `write_pmp_json` it and
  run `pmp2sdp` on the file.
- **`sdpb`**: `pmp.solve(output_dir="out", checkpoint_dir="ck")` produces the
  same `iterations.json`, `c_minus_By/` and checkpoints as
  `sdpb --outDir out --checkpointDir ck`. To run the executable itself from
  Python on an existing `sdp/` directory, the legacy passthrough
  {func}`sdpb_python.solve_dir` calls SDPB's `main` logic in-process and
  parses its `out.txt`.
- **`spectrum`**: needs `pmp_info.json` from an `sdp/` directory produced by
  `pmp2sdp`, plus `c_minus_By.json` and `x_<j>.txt` from an `sdpb` output
  directory. `Solution.c_minus_By` and `Solution.x` hold the same numbers;
  writing them in SDPB's text formats is left to the caller for now.

## Numbers in files

All numbers are written as decimal strings with enough digits to reproduce the
binary value at the given precision (`write_pmp_json(..., precision=768)`).
Reading never rounds: strings are converted at the solver's precision when the
problem is solved.
