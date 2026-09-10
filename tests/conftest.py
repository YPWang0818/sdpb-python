import pytest


@pytest.fixture
def sdpb_ext():
    """The compiled extension, or skip the test when it has not been built."""
    pytest.importorskip("sdpb_python._sdpb", reason="sdpb_python._sdpb extension not built")
    from sdpb_python import _sdpb

    return _sdpb
