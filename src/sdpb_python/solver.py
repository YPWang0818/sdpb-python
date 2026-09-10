"""High-level solver interface built on the ``_sdpb`` Cython extension."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


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


@dataclass
class SDPBResult:
    """Contents of SDPB's ``out.txt``."""

    terminate_reason: str
    primal_objective: str
    dual_objective: str
    duality_gap: str
    primal_error: str
    dual_error: str
    solver_runtime: int
    out_dir: Path | None = None
    extra: dict[str, str] = field(default_factory=dict)

    _KEYS = {
        "terminateReason": "terminate_reason",
        "primalObjective": "primal_objective",
        "dualObjective": "dual_objective",
        "dualityGap": "duality_gap",
        "primalError": "primal_error",
        "dualError": "dual_error",
        "Solver runtime": "solver_runtime",
    }

    @classmethod
    def from_text(cls, text: str, out_dir: Path | None = None) -> "SDPBResult":
        """Parse the ``key = value;`` lines of an ``out.txt`` file."""
        values: dict[str, str] = {}
        for line in text.splitlines():
            m = re.match(r"\s*(.+?)\s*=\s*(.*?);\s*$", line)
            if m:
                values[m.group(1)] = m.group(2).strip().strip('"')
        missing = [k for k in cls._KEYS if k not in values]
        if missing:
            raise ValueError(f"out.txt is missing fields: {missing}")
        kwargs: dict[str, Any] = {attr: values.pop(key) for key, attr in cls._KEYS.items()}
        kwargs["solver_runtime"] = int(kwargs["solver_runtime"])
        return cls(out_dir=out_dir, extra=values, **kwargs)

    @classmethod
    def from_dir(cls, out_dir: str | os.PathLike) -> "SDPBResult":
        out_dir = Path(out_dir)
        return cls.from_text((out_dir / "out.txt").read_text(), out_dir=out_dir)

    @property
    def optimal(self) -> bool:
        return "optimal" in self.terminate_reason


def _to_cli(options: Mapping[str, Any]) -> list[str]:
    """Convert ``{"precision": 768, "noFinalCheckpoint": True}`` to argv."""
    args: list[str] = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        args.append(f"--{key}")
        if value is not True:
            args.append(str(value))
    return args


def solve(
    sdp_dir: str | os.PathLike,
    out_dir: str | os.PathLike,
    **options: Any,
) -> SDPBResult:
    """Run SDPB on ``sdp_dir`` and return the parsed ``out.txt``.

    Keyword arguments map one-to-one onto ``sdpb`` command-line options,
    e.g. ``solve("sdp", "out", precision=768, maxIterations=500)``.
    Boolean ``True`` becomes a bare flag such as ``--noFinalCheckpoint``.
    """
    args = ["--sdpDir", os.fspath(sdp_dir), "--outDir", os.fspath(out_dir)]
    args += _to_cli(options)
    _ext().run(args)
    return SDPBResult.from_dir(out_dir)
