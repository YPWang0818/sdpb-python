"""Per-solve profiles (SolverOptions.profile_path / profile_detail / profile_meta)."""

import json

import pytest

from sdpb_python import PMP, Polynomial, PolynomialMatrix, SDPBError, SolverOptions, read_pmp_json
from tests.util.datasets import DATA, TEST_PRECISION, STANDARD_ARGS

ONE_D = DATA / "1d" / "input" / "pmp.json"
ARGS = dict(STANDARD_ARGS)


@pytest.fixture
def pmp(sdpb_ext):
    return read_pmp_json(ONE_D)


def load(path):
    with open(path) as f:
        return json.load(f)


def names(doc):
    return {t["name"] for t in doc["timers"]}


def parent_name(doc, name):
    by_id = {t["id"]: t for t in doc["timers"]}
    node = next(t for t in doc["timers"] if t["name"] == name)
    return by_id[node["parent"]]["name"] if node["parent"] >= 0 else None


def test_options_to_dict():
    d = SolverOptions(profile_path="p.json", profile_detail=2, profile_autosave=True,
                      profile_meta={"case": "x", "node": 3})._to_dict()
    assert d["profile_path"] == "p.json"
    assert d["profile_detail"] == 2
    assert d["profile_autosave"] is True
    assert d["profile_meta"] == [("case", "x"), ("node", "3")]
    assert SolverOptions()._to_dict()["profile_path"] == ""


def test_profile_file_valid_json(pmp, tmp_path):
    path = tmp_path / "p.json"
    sol = pmp.solve(profile_path=path, profile_meta={"case": "t", "node": "3"}, **ARGS)
    assert sol.optimal
    assert sol.profile_path == str(path)
    doc = load(path)
    assert doc["schema_version"] == 1
    assert doc["precision_bits"] == TEST_PRECISION
    assert doc["num_ranks"] == 1
    meta = doc["meta"]
    assert meta["run_kind"] == "sdpb_python"
    assert meta["labels"] == {"case": "t", "node": "3"}
    assert isinstance(meta["parameters"], dict) and "precision" in meta["parameters"]
    assert "blocks" in meta and "N" in meta and "local_blocks" in meta
    assert {"solve_pmp", "parse_spec", "pmp2sdp", "convert", "solver", "ctor", "run", "iter",
            "collect"} <= names(doc)
    assert parent_name(doc, "solve_pmp") is None
    assert parent_name(doc, "parse_spec") == "solve_pmp"
    assert parent_name(doc, "convert") == "pmp2sdp"
    assert parent_name(doc, "pmp2sdp") == "solve_pmp"
    assert parent_name(doc, "run") == "solver"
    assert parent_name(doc, "collect") == "solver"
    elapsed = {t["name"]: t["elapsed"] for t in doc["timers"]}
    assert elapsed["solve_pmp"] >= elapsed["parse_spec"] + elapsed["pmp2sdp"] + elapsed["solver"]
    assert sum(1 for t in doc["timers"] if t["name"] == "iter") == sol.iterations + 1  # + terminating iteration


def test_profile_default_off(pmp, tmp_path):
    sol = pmp.solve(**ARGS)
    assert sol.profile_path is None
    assert list(tmp_path.iterdir()) == []


def test_profile_on_error(sdpb_ext, tmp_path):
    """The profile is written even when the solve raises (Cholesky failure)."""
    failing = PMP(objective=[0, 1], normalization=[1, 0],
                  matrices=[PolynomialMatrix([[[Polynomial(["3/2", -2, 1]), Polynomial([-1])]]])])
    path = tmp_path / "err.json"
    with pytest.raises(SDPBError):
        failing.solve(precision=TEST_PRECISION, duality_gap_threshold="1e-150",
                      primal_error_threshold="1e-150", dual_error_threshold="1e-150",
                      max_iterations=2000, profile_path=path)
    doc = load(path)
    assert "run" in names(doc)
    assert "iter" in names(doc)


def test_profile_detail_and_autosave(pmp, tmp_path):
    coarse = pmp.solve(profile_path=tmp_path / "d0.json", profile_detail=0, **ARGS)
    fine = pmp.solve(profile_path=tmp_path / "d1.json", profile_detail=1, **ARGS)
    assert len(load(coarse.profile_path)["timers"]) < len(load(fine.profile_path)["timers"])
    kinds = {t["attrs"].get("kind") for t in load(fine.profile_path)["timers"]}
    assert "cholesky" in kinds and "eig" in kinds
    # autosave: the file exists even for a run that stops after one iteration
    args = {k: v for k, v in ARGS.items() if k != "max_iterations"}
    sol = pmp.solve(profile_path=tmp_path / "auto.json", profile_autosave=True, max_iterations=1, **args)
    assert sol.iterations == 1
    assert (tmp_path / "auto.json").exists()
    assert not (tmp_path / "auto.json.tmp").exists()


def test_handle_profile_requested_at_run(pmp, tmp_path):
    """A handle built without a profile records nothing until a run asks for one
    (the timers are off by default), then profiles from that run on."""
    path = tmp_path / "late.json"
    with pmp.solver(**ARGS) as solver:
        first = solver.run(max_iterations=2)
        assert first.profile_path is None
        second = solver.run(profile_path=path)
        assert second.optimal
        assert second.profile_path == str(path)
    doc = load(path)
    runs = [t for t in doc["timers"] if t["name"] == "handle_run"]
    assert [r["attrs"]["k"] for r in runs] == ["2"]
    assert "handle_build" not in names(doc)
    assert sum(1 for t in doc["timers"] if t["name"] == "iter") == second.iterations + 1


def test_handle_profile_cumulative(pmp, tmp_path):
    path = tmp_path / "h.json"
    with pmp.solver(profile_path=path, **ARGS) as solver:
        first = solver.run(max_iterations=2)
        assert first.profile_path == str(path)
        second = solver.run()
        assert second.optimal
    doc = load(path)
    runs = [t for t in doc["timers"] if t["name"] == "handle_run"]
    assert [r["attrs"]["k"] for r in runs] == ["1", "2"]
    assert "handle_build" in names(doc)
    assert sum(1 for t in doc["timers"] if t["name"] == "iter") == first.iterations + second.iterations + 2  # one terminating iteration per run
