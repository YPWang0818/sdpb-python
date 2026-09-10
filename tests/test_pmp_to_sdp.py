"""T3: analogue of the pmp2sdp stage and diff_sdp()."""

from pathlib import Path

import mpmath
import pytest

from sdpb_python import PMP, read_pmp_json
from tests.util.datasets import DATA, SDPB, TEST_PRECISION, Dataset, dataset_params
from tests.util.diff import assert_close
from tests.util.reference import RefSDP, read_sdp_dir, to_matrix


def diff_sdp(pmp: PMP, ref: RefSDP, precision: int, diff_bits: int, max_num_poles=None,
             check_normalization=True):
    sdp = pmp.to_sdp(precision, max_num_poles=max_num_poles)
    assert len(sdp.blocks) == ref.num_blocks
    with mpmath.workprec(precision):
        assert_close(sdp.objective_const, ref.objective_const, diff_bits, "objective_const")
        assert_close(sdp.b, ref.b, diff_bits, "b")
        if check_normalization:
            assert (sdp.normalization is None) == (ref.normalization is None)
            if ref.normalization is not None:
                assert_close(sdp.normalization, ref.normalization, diff_bits, "normalization")
        # pmp_info: per-block dim, prefactors, sample points and scalings
        assert len(ref.pmp_info) == ref.num_blocks
        for j, (block, info) in enumerate(zip(sdp.blocks, ref.pmp_info)):
            assert block.block_index == j == info["index"]
            assert block.dim == info["dim"]
            sampled = pmp.matrices[j].sampled(precision, max_num_poles=max_num_poles)
            for key, got in (("prefactor", sampled.prefactor), ("reducedPrefactor", sampled.reduced_prefactor)):
                exp = info[key]
                assert_close(got.constant, exp["constant"], diff_bits, f"block {j} {key}.constant")
                assert_close(got.base, exp["base"], diff_bits, f"block {j} {key}.base")
                assert_close(list(got.poles), exp["poles"], diff_bits, f"block {j} {key}.poles")
            for key, attr in (("samplePoints", "sample_points"), ("sampleScalings", "sample_scalings"),
                              ("reducedSampleScalings", "reduced_sample_scalings")):
                assert_close(getattr(sampled, attr), info[key], diff_bits, f"block {j} {key}")
        for j, (block, rb) in enumerate(zip(sdp.blocks, ref.blocks)):
            assert (block.dim, block.num_points) == (rb.dim, rb.num_points), f"block {j}"
            assert_close(block.bilinear_bases[0], to_matrix(rb.bilinear_bases_even), diff_bits, f"block {j} even")
            assert_close(block.bilinear_bases[1], to_matrix(rb.bilinear_bases_odd), diff_bits, f"block {j} odd")
            assert_close(block.c, rb.c, diff_bits, f"block {j} c")
            assert_close(block.B, to_matrix(rb.B), diff_bits, f"block {j} B")


@pytest.mark.parametrize("dataset,input_name", dataset_params())
def test_end_to_end_sdp(sdpb_ext, dataset: Dataset, input_name: str):
    if not dataset.check_sdp:
        pytest.skip("dataset has no reference sdp/")
    pmp = read_pmp_json(dataset.dir / "input" / input_name)
    ref = read_sdp_dir(dataset.sdp_dir)
    # 1d: normalization.json is absent for pmp-no-optional-fields.json (fork skips it too)
    diff_sdp(pmp, ref, TEST_PRECISION, dataset.diff_precision, max_num_poles=dataset.max_num_poles,
             check_normalization=dataset.name != "1d")


def check_pmp2sdp_json_reference(precision: int, diff_bits: int):
    """Runs in a subprocess at ``precision`` (see test_pmp2sdp_json_reference)."""
    data_dir = SDPB / "test" / "data" / "pmp2sdp" / "json"
    for name in ("pmp.json", "file_list.nsv"):
        pmp = read_pmp_json(data_dir / name)
        diff_sdp(pmp, read_sdp_dir(data_dir / "sdp_orig"), precision, diff_bits)


def test_pmp2sdp_json_reference(sdpb_ext):
    """pmp2sdp/json: precision 512, compared at 392 bits as in pmp2sdp.test.cxx.

    The sample points converge to a precision-dependent accuracy, so the
    reference must be reproduced at its own precision (in a subprocess).
    """
    from tests.util.subproc import run_at_precision

    run_at_precision(512, "tests.test_pmp_to_sdp.check_pmp2sdp_json_reference", 512, 392)


def test_1d_variants_agree(sdpb_ext):
    """All four 1d inputs describe the same SDP."""
    inputs = ["pmp.json", "pmp-no-optional-fields.json", "pmp-sample-points.json", "pmp-all-sampling-fields.json"]
    sdps = [read_pmp_json(DATA / "1d" / "input" / n).to_sdp(TEST_PRECISION) for n in inputs]
    with mpmath.workprec(TEST_PRECISION):
        for other in sdps[1:]:
            assert_close(sdps[0].b, other.b, 99)
            for a, b in zip(sdps[0].blocks, other.blocks):
                assert_close(a.c, b.c, 99)
                assert_close(a.B, b.B, 99)
                assert_close(a.bilinear_bases[0], b.bilinear_bases[0], 99)
