"""Exception types."""

from __future__ import annotations

import re

# SDPB's THROW/ASSERT macros (src/sdpb_util/assert.hxx) format every message as
#   in <function>() at <file>:<line>: \n  <message lines>\nStacktrace:\n 0# ...
_LOCATION = re.compile(r"^in (?P<func>\S+) at (?P<file>.+?):(?P<line>\d+): ?$")


class SDPBError(RuntimeError):
    """An error raised inside SDPB's C++ code.

    ``str(error)`` is SDPB's message, with the source location appended in
    brackets when SDPB reported one; ``details`` keeps the whole text as
    SDPB produced it, including its stack trace.
    """

    def __init__(self, details: str):
        self.details = details
        super().__init__(summarize(details))

    @property
    def location(self) -> str | None:
        """``file:line (function)`` of the SDPB source that threw, if known."""
        header = _header(self.details)
        return f"{header['file']}:{header['line']} ({header['func']})" if header else None


def _header(details: str) -> re.Match | None:
    lines = details.strip().splitlines()
    return _LOCATION.match(lines[0].strip()) if lines else None


def summarize(details: str) -> str:
    """The message part of an SDPB error text, without the stack trace."""
    lines = details.strip().splitlines()
    if not lines:
        return "SDPB error"
    body: list[str] = []
    for line in lines:
        if line.strip() == "Stacktrace:":
            break
        body.append(line)
    header = _LOCATION.match(body[0].strip()) if body else None
    if header is None:
        return " ".join(line.strip() for line in body if line.strip()) or lines[0]
    message = " ".join(line.strip() for line in body[1:] if line.strip())
    location = f"{header['file']}:{header['line']} ({header['func']})"
    return f"{message} [{location}]" if message else f"SDPB error at {location}"


def wrap_cpp_error(exc: BaseException) -> BaseException:
    """Translate an exception from the Cython layer into SDPBError."""
    if isinstance(exc, (RuntimeError, ValueError)) and not isinstance(exc, SDPBError):
        return SDPBError(str(exc))
    return exc
