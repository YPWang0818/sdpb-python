import sdpb_python
from sdpb_python.solver import SDPBResult, _to_cli

SAMPLE_OUT = """\
terminateReason = "found primal-dual optimal solution";
primalObjective = 1.840261440728371962751281851062;
dualObjective   = 1.840261440728371962751281851062;
dualityGap      = 5.5e-31;
primalError     = 1.2e-40;
dualError       = 3.4e-40;
Solver runtime  = 7;
"""


def test_version_string():
    assert sdpb_python.__version__


def test_parse_out_txt():
    r = SDPBResult.from_text(SAMPLE_OUT)
    assert r.optimal
    assert r.terminate_reason == "found primal-dual optimal solution"
    assert r.primal_objective.startswith("1.8402614")
    assert r.solver_runtime == 7
    assert r.extra == {}


def test_cli_conversion():
    args = _to_cli({"precision": 768, "noFinalCheckpoint": True, "skip": None, "off": False})
    assert args == ["--precision", "768", "--noFinalCheckpoint"]


def test_extension_reports_version(sdpb_ext):
    assert isinstance(sdpb_ext.version(), str)
    assert sdpb_ext.mpi_size() >= 1
