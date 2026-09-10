"""T4: analogue of integration_tests/cases/end-to-end.test.cxx (pmp2sdp + sdpb stages)."""

import mpmath
import pytest

from sdpb_python import read_pmp_json
from tests.util.datasets import TEST_PRECISION, Dataset, dataset_params
from tests.util.diff import assert_close
from tests.util.reference import (read_c_minus_By_json, read_iterations_json, read_matrix_txt,
                                  read_out_txt, read_vector_txt, to_matrix)

OUT_TXT_FIELDS = {"primalObjective": "primal_objective", "dualObjective": "dual_objective",
                  "dualityGap": "duality_gap", "primalError": "primal_error", "dualError": "dual_error"}
SMALL_ERROR_KEYS = {"P-err", "p-err", "D-err", "R-err"}
IGNORED_ITERATION_KEYS = {"total_time", "iter_time", "block_name"}


def diff_iterations(got: list[dict], ref: list[dict], diff_bits: int):
    """As diff_iterations_json in diff_sdpb_out.cxx."""
    assert len(got) == len(ref), f"{len(got)} iterations, reference has {len(ref)}"
    abs_eps = mpmath.mpf(2) ** (-(diff_bits // 2))
    for n, (a, b) in enumerate(zip(got, ref), start=1):
        assert list(a) == list(b), f"iteration {n}: keys differ"
        for key in a:
            if key in IGNORED_ITERATION_KEYS:
                continue
            av, bv = mpmath.mpf(str(a[key])), mpmath.mpf(str(b[key]))
            if key in SMALL_ERROR_KEYS and abs(av) + abs(bv) < abs_eps:
                continue
            assert_close(av, bv, diff_bits, f"iteration {n} {key}")


@pytest.mark.parametrize("dataset,input_name", dataset_params())
def test_end_to_end(sdpb_ext, dataset: Dataset, input_name: str, tmp_path):
    pmp = read_pmp_json(dataset.dir / "input" / input_name)
    bits, diff_bits = TEST_PRECISION, dataset.diff_precision
    out_dir = tmp_path / "out"
    solution = pmp.solve(precision=bits, want=("x", "y", "z", "c_minus_By"), output_dir=out_dir,
                         max_num_poles=dataset.max_num_poles, **dataset.args)

    ref_out = read_out_txt(dataset.out_dir / "out.txt")
    with mpmath.workprec(bits):
        # out.txt
        assert solution.status.value == ref_out.terminate_reason
        for key in dataset.out_keys:
            if key == "terminateReason":
                continue
            assert_close(getattr(solution, OUT_TXT_FIELDS[key]), ref_out.values[key], diff_bits, key)
        # y.txt, z.txt, x_<j>.txt
        assert_close(solution.y, read_vector_txt(dataset.out_dir / "y.txt"), diff_bits, "y")
        if (dataset.out_dir / "z.txt").exists():
            assert solution.z is not None
            assert_close(solution.z, read_vector_txt(dataset.out_dir / "z.txt"), diff_bits, "z")
        if dataset.check_x:
            for j in range(len(solution.blocks)):
                path = dataset.out_dir / f"x_{j}.txt"
                if path.exists():
                    assert_close(solution.x[j], read_vector_txt(path), diff_bits, f"x_{j}")
        # c - B.y: recomputed from the SDP data (check_c_minus_By) and from the reference file
        sdp = pmp.to_sdp(bits, max_num_poles=dataset.max_num_poles)
        y = mpmath.matrix(solution.y)
        for j, block in enumerate(sdp.blocks):
            recomputed = mpmath.matrix(block.c) - block.B * y
            assert_close(solution.c_minus_By[j], [recomputed[i] for i in range(recomputed.rows)],
                         diff_bits, f"c_minus_By[{j}] vs recomputed")
        ref_cmb = read_c_minus_By_json(dataset.out_dir / "c_minus_By" / "c_minus_By.json")
        assert_close(solution.c_minus_By, ref_cmb, diff_bits, "c_minus_By vs reference")
        # iterations.json: the reference directory may also hold iterations.<k>.json
        # from a 2-iteration timing run (several MPI ranks) and, for datasets the
        # fork runs twice to test checkpoint loading, a trivial 0-iteration final
        # run; the real solve is the longest of them.
        got = read_iterations_json(out_dir / "iterations.json")
        assert solution.iterations == len(got)
        reference = max((read_iterations_json(p) for p in dataset.out_dir.glob("iterations*.json")), key=len)
        diff_iterations(got, reference, diff_bits)
