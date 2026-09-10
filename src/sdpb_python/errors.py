"""Exception types."""


class SDPBError(RuntimeError):
    """An error raised inside SDPB's C++ code.

    ``str(error)`` is the first line of SDPB's message; ``details`` keeps the
    whole text, including SDPB's stack trace.
    """

    def __init__(self, details: str):
        self.details = details
        first = details.strip().splitlines()[0] if details.strip() else "SDPB error"
        super().__init__(first)


def wrap_cpp_error(exc: BaseException) -> BaseException:
    """Translate an exception from the Cython layer into SDPBError."""
    if isinstance(exc, (RuntimeError, ValueError)) and not isinstance(exc, SDPBError):
        return SDPBError(str(exc))
    return exc
