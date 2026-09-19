"""Access to the compiled ``_sdpb`` extension and its version."""

from __future__ import annotations


def _ext():
    """Import the compiled extension lazily with a helpful error message."""
    try:
        from . import _sdpb
    except ImportError as exc:  # pragma: no cover - only hit when not built
        raise ImportError(
            "The sdpb_python._sdpb extension is not built. Build SDPB "
            "(scripts/build_sdpb.sh) and reinstall with `pip install -e .`."
        ) from exc
    return _sdpb


def sdpb_version() -> str:
    """Git version of the SDPB sources compiled into the extension."""
    return _ext().version()
