"""The fork's end-to-end datasets, with the solver arguments used in
test/src/integration_tests/cases/end-to-end.test.cxx."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SDPB = Path(__file__).resolve().parents[2] / "c-src" / "sdpb"
DATA = SDPB / "test" / "data" / "end-to-end_tests"

# SDPB's Elemental fixes the precision once per process, so the whole pytest
# process runs at one precision (as the fork's unit_tests binary does, 768).
# Datasets whose reference was made at another precision are still compared
# at their diff_precision; test_precision.py re-solves 1d at its native 664
# bits in a subprocess.
TEST_PRECISION = 768

# --checkpointInterval 3600 --maxRuntime 1340 --dualityGapThreshold 1.0e-30 ...
STANDARD_ARGS = dict(
    checkpoint_interval=3600, max_runtime=1340,
    duality_gap_threshold="1.0e-30", primal_error_threshold="1.0e-30", dual_error_threshold="1.0e-30",
    initial_matrix_scale_primal="1.0e20", initial_matrix_scale_dual="1.0e20",
    feasible_centering_parameter="0.1", infeasible_centering_parameter="0.3",
    step_length_reduction="0.7", max_complementarity="1.0e100", max_iterations=1000,
)

# sdpb defaults used when the fork passes no explicit args (Solver_Parameters defaults)
DEFAULT_ARGS = dict(max_iterations=500, duality_gap_threshold="1e-30",
                    primal_error_threshold="1e-30", dual_error_threshold="1e-30")

DFIBO_ARGS = dict(
    find_dual_feasible=True, find_primal_feasible=True,
    initial_matrix_scale_primal="1e10", initial_matrix_scale_dual="1e10",
    max_complementarity="1e30", dual_error_threshold="1e-10", primal_error_threshold="1e-153",
    max_runtime=259200, checkpoint_interval=3600, max_iterations=1000,
    feasible_centering_parameter="0.1", infeasible_centering_parameter="0.3",
    step_length_reduction="0.7", max_shared_memory_bytes="100K",
)

ALLOWED_ARGS = dict(
    checkpoint_interval=3600, max_runtime=1341,
    duality_gap_threshold="1.0e-30", primal_error_threshold="1.0e-200", dual_error_threshold="1.0e-200",
    initial_matrix_scale_primal="1.0e20", initial_matrix_scale_dual="1.0e20",
    feasible_centering_parameter="0.1", infeasible_centering_parameter="0.3",
    step_length_reduction="0.7", max_complementarity="1.0e100", max_iterations=1000,
    detect_primal_feasible_jump=True, detect_dual_feasible_jump=True,
    max_shared_memory_bytes="100.1K",
)

ALL_OUT_KEYS = ("terminateReason", "primalObjective", "dualObjective")


@dataclass
class Dataset:
    name: str                      # directory under end-to-end_tests/
    inputs: list[str]              # input files relative to <name>/input; one solve per entry
    precision: int = 768
    diff_precision: int = 99
    args: dict = field(default_factory=lambda: dict(DEFAULT_ARGS))
    max_num_poles: int | None = None
    out_keys: tuple[str, ...] = ALL_OUT_KEYS
    check_x: bool = True           # 1d-duplicate-poles: "The vector x may differ"
    check_sdp: bool = True         # output/sdp exists and is compared
    slow: bool = False
    skip: str = ""                 # reason to skip

    @property
    def dir(self) -> Path:
        return DATA / self.name

    @property
    def sdp_dir(self) -> Path:
        return self.dir / "output" / "sdp"

    @property
    def out_dir(self) -> Path:
        return self.dir / "output" / "out"


DATASETS = [
    Dataset("1d", ["pmp.json", "pmp-no-optional-fields.json", "pmp-sample-points.json",
                   "pmp-all-sampling-fields.json"], precision=664),
    Dataset("1d-old-sampling", ["pmp.json"]),
    Dataset("1d-duplicate-poles", ["pmp.json"], check_x=False),
    Dataset("1d-constraints", ["pmp.xml"], skip="XML input is not supported"),
    Dataset("1d-isolated-zeros", ["pmp.nsv"], args=dict(STANDARD_ARGS), check_sdp=False),
    Dataset("dfibo-0-0-j=3-c=3.0000-d=3-s=6", ["pmp.xml"], args=DFIBO_ARGS, skip="XML input is not supported"),
    Dataset("SingletScalar_cT_test_nmax6/primal_dual_optimal", ["pmp.nsv"], args=dict(STANDARD_ARGS), slow=True),
    Dataset("SingletScalar_cT_test_nmax6/primal_dual_optimal_reduced",
            ["pmp_reduced_prefactor.nsv", "pmp_max_num_poles.nsv"], args=dict(STANDARD_ARGS), slow=True),
    Dataset("SingletScalar_cT_test_nmax6/primal_dual_optimal_reduced_max_num_poles_14",
            ["pmp_reduced_prefactor.nsv", "pmp_max_num_poles.nsv"], args=dict(STANDARD_ARGS),
            max_num_poles=14, slow=True),
    Dataset("SingletScalarAllowed_test_nmax6/primal_feasible_jump", ["pmp.nsv"], args=ALLOWED_ARGS,
            out_keys=("terminateReason", "primalObjective", "dualObjective", "dualityGap", "dualError"),
            slow=True),
    Dataset("SingletScalarAllowed_test_nmax6/dual_feasible_jump", ["pmp.nsv"], args=ALLOWED_ARGS,
            out_keys=("terminateReason", "primalObjective", "dualObjective", "dualityGap", "primalError"),
            slow=True),
]


def dataset_params():
    """pytest params: one per (dataset, input file)."""
    import pytest

    params = []
    for ds in DATASETS:
        for inp in ds.inputs:
            marks = []
            if ds.skip:
                marks.append(pytest.mark.skip(reason=ds.skip))
            if ds.slow:
                marks.append(pytest.mark.slow)
            params.append(pytest.param(ds, inp, id=f"{ds.name}/{inp}", marks=marks))
    return params
