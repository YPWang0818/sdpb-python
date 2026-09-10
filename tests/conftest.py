import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="run the slow end-to-end datasets (minutes each)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="slow dataset; use --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def sdpb_ext():
    """The compiled extension (precision fixed at TEST_PRECISION), or skip if not built."""
    pytest.importorskip("sdpb_python._sdpb", reason="sdpb_python._sdpb extension not built")
    import sdpb_python
    from sdpb_python import _sdpb
    from tests.util.datasets import TEST_PRECISION

    sdpb_python.set_precision(TEST_PRECISION)
    return _sdpb
