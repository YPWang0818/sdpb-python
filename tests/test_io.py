"""T1: pmp.json reader/writer round trip over every JSON input in the fork's data."""

import json
from pathlib import Path

import pytest

from sdpb_python import PMP, read_pmp_json, write_pmp_json
from sdpb_python.io import pmp_to_json_dict, read_nsv
from tests.util.datasets import DATA, SDPB

JSON_INPUTS = sorted(p for p in DATA.rglob("input/**/*.json"))


@pytest.mark.parametrize("path", JSON_INPUTS, ids=lambda p: str(p.relative_to(DATA)))
def test_read_write_roundtrip(path, tmp_path):
    with open(path) as fh:
        raw = json.load(fh)
    if "objective" not in raw or not raw.get("PositiveMatrixWithPrefactorArray"):
        pytest.skip("file is part of an .nsv set (objective or matrices live elsewhere)")
    pmp = read_pmp_json(path)
    assert len(pmp.matrices) == len(raw["PositiveMatrixWithPrefactorArray"])
    out = tmp_path / "pmp.json"
    write_pmp_json(pmp, out)
    again = read_pmp_json(out)
    assert pmp_to_json_dict(again) == pmp_to_json_dict(pmp)


def test_nsv_expansion():
    nsv = DATA / "1d-isolated-zeros" / "input" / "pmp.nsv"
    files = read_nsv(nsv)
    assert len(files) == 7 and all(f.suffix == ".json" for f in files)
    pmp = read_pmp_json(nsv)
    assert len(pmp.matrices) == 7
    assert pmp.normalization is not None


def test_conflicting_objectives_rejected(tmp_path):
    """Analogue of pmp2sdp 'objectives are different'."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    src = DATA / "1d" / "input" / "pmp.json"
    with open(src) as fh:
        d = json.load(fh)
    a.write_text(json.dumps(d))
    d["objective"] = ["1", "1"]
    b.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="objective"):
        read_pmp_json(a, b)


def test_duplicate_objectives_accepted(tmp_path):
    """Analogue of pmp2sdp 'duplicate objectives': same objective in two files is fine."""
    src = DATA / "1d" / "input" / "pmp.json"
    pmp = read_pmp_json(src, src)
    assert len(pmp.matrices) == 2


def test_validation_errors():
    with pytest.raises(ValueError, match="symmetric"):
        from sdpb_python import PolynomialMatrix
        PolynomialMatrix([[[[1], [2]], [[1], [3]]], [[[1], [2]], [[1], [3]]]])
    with pytest.raises(ValueError, match="same number"):
        from sdpb_python import PolynomialMatrix
        PolynomialMatrix([[[[1], [2]]], ])  # ok
        PolynomialMatrix([[[[1], [2]], [[1]]], [[[1]], [[1], [2]]]])
    with pytest.raises(ValueError, match="normalization"):
        from sdpb_python import PolynomialMatrix
        PMP([0, 1], [1], [PolynomialMatrix([[[[1], [2]]]])])
    with pytest.raises(ValueError, match="negative"):
        from sdpb_python import PolynomialMatrix
        PolynomialMatrix([[[[1], [2]]]], sample_points=[-1])
    with pytest.raises(ValueError, match="polynomials per entry"):
        from sdpb_python import PolynomialMatrix
        PMP([0, 1, 2], None, [PolynomialMatrix([[[[1], [2]]]])])
