"""Readers for SDPB's reference files (test/src/integration_tests/util/*.cxx analogues)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import mpmath


def read_matrix_txt(path: Path) -> list[list[str]]:
    """``y.txt`` / ``x_<j>.txt`` / ``X_matrix_<i>.txt``: ``"h w"`` then rows."""
    tokens = path.read_text().split()
    h, w = int(tokens[0]), int(tokens[1])
    values = tokens[2:]
    assert len(values) == h * w, f"{path}: expected {h*w} values, got {len(values)}"
    return [values[i * w:(i + 1) * w] for i in range(h)]


def read_vector_txt(path: Path) -> list[str]:
    rows = read_matrix_txt(path)
    assert all(len(r) == 1 for r in rows), f"{path}: not a column vector"
    return [r[0] for r in rows]


@dataclass
class OutTxt:
    terminate_reason: str
    values: dict[str, str]  # primalObjective, dualObjective, dualityGap, primalError, dualError, Solver runtime


def read_out_txt(path: Path) -> OutTxt:
    reason = None
    values = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(.+?)\s*=\s*(.*?);\s*$", line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if key == "terminateReason":
            reason = value.strip('"')
        else:
            values[key] = value
    assert reason is not None, f"{path}: no terminateReason"
    return OutTxt(reason, values)


def read_iterations_json(path: Path) -> list[dict]:
    with open(path) as fh:
        return json.load(fh)


def read_c_minus_By_json(path: Path) -> list[list[str]]:
    with open(path) as fh:
        return json.load(fh)["c_minus_By"]


@dataclass
class RefBlock:
    dim: int
    num_points: int
    bilinear_bases_even: list[list[str]]
    bilinear_bases_odd: list[list[str]]
    c: list[str]
    B: list[list[str]]


@dataclass
class RefSDP:
    num_blocks: int
    objective_const: str
    b: list[str]
    normalization: list[str] | None
    pmp_info: list[dict]
    blocks: list[RefBlock]


def read_sdp_dir(path: Path) -> RefSDP:
    """Parse an ``sdp/`` directory with JSON block data (all fork references use JSON)."""
    with open(path / "control.json") as fh:
        control = json.load(fh)
    with open(path / "objectives.json") as fh:
        objectives = json.load(fh)
    normalization = None
    if (path / "normalization.json").exists():
        with open(path / "normalization.json") as fh:
            normalization = json.load(fh)["normalization"]
    with open(path / "pmp_info.json") as fh:
        pmp_info = json.load(fh)
    blocks = []
    for j in range(control["num_blocks"]):
        with open(path / f"block_info_{j}.json") as fh:
            info = json.load(fh)
        data_path = path / f"block_data_{j}.json"
        assert data_path.exists(), f"{data_path} missing (binary block data is not supported)"
        with open(data_path) as fh:
            data = json.load(fh)
        blocks.append(RefBlock(
            dim=info["dim"], num_points=info["num_points"],
            bilinear_bases_even=data["bilinear_bases_even"],
            bilinear_bases_odd=data["bilinear_bases_odd"],
            c=data["c"], B=data["B"],
        ))
    return RefSDP(control["num_blocks"], objectives["constant"], objectives["b"], normalization, pmp_info, blocks)


def to_matrix(rows: list[list[str]]) -> mpmath.matrix:
    if not rows:
        return mpmath.matrix(0, 0)
    if rows and not rows[0]:
        return mpmath.matrix(len(rows), 0)
    return mpmath.matrix([[mpmath.mpf(v) for v in row] for row in rows])
