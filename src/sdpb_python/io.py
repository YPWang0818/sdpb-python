"""Read and write SDPB's ``pmp.json`` format (and ``.nsv`` file lists)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from .problem import PMP, DampedRational, Polynomial, PolynomialMatrix


def _damped_rational(d: dict) -> DampedRational:
    return DampedRational(constant=d["constant"], base=d["base"], poles=list(d.get("poles", [])))


def _matrix_from_json(d: dict, label: str = "") -> PolynomialMatrix:
    kwargs = {}
    # SDPB accepts both "prefactor" and the legacy "DampedRational" key.
    prefactor = d.get("prefactor", d.get("DampedRational"))
    if prefactor is not None:
        kwargs["prefactor"] = _damped_rational(prefactor)
    if d.get("reducedPrefactor") is not None:
        kwargs["reduced_prefactor"] = _damped_rational(d["reducedPrefactor"])
    if d.get("maxNumPoles") is not None:
        kwargs["max_num_poles"] = int(d["maxNumPoles"])
    for json_key, attr in (("samplePoints", "sample_points"), ("sampleScalings", "sample_scalings"),
                           ("reducedSampleScalings", "reduced_sample_scalings")):
        if d.get(json_key) is not None:
            kwargs[attr] = list(d[json_key])
    if d.get("bilinearBasis") is not None:
        # single basis: SDPB uses it for both parities
        basis = [Polynomial(c) for c in d["bilinearBasis"]]
        kwargs["bilinear_basis"] = (basis, basis)
    if d.get("bilinearBasis_0") is not None or d.get("bilinearBasis_1") is not None:
        kwargs["bilinear_basis"] = ([Polynomial(c) for c in d.get("bilinearBasis_0", [])],
                                    [Polynomial(c) for c in d.get("bilinearBasis_1", [])])
    polynomials = [[[Polynomial(c) for c in entry] for entry in row] for row in d["polynomials"]]
    return PolynomialMatrix(polynomials, label=label, **kwargs)


def read_nsv(path: str | os.PathLike) -> list[Path]:
    """Expand an ``.nsv`` file (NUL-separated list of paths, recursively)."""
    path = Path(path)
    out: list[Path] = []
    for entry in path.read_bytes().split(b"\0"):
        name = entry.decode().strip()
        if not name:
            continue
        p = (path.parent / name).resolve()
        if p.suffix == ".nsv":
            out.extend(read_nsv(p))
        else:
            out.append(p)
    return out


def read_pmp_json(*paths: str | os.PathLike) -> PMP:
    """Load one or more ``pmp.json`` files (or ``.nsv`` lists) into a :class:`PMP`.

    As in SDPB, every file may carry the objective and normalization; they must
    agree, and matrices are concatenated in file order.
    """
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files.extend(read_nsv(p) if p.suffix == ".nsv" else [p])
    if not files:
        raise ValueError("no input files")
    objective = normalization = None
    matrices: list[PolynomialMatrix] = []
    for f in files:
        if f.suffix != ".json":
            raise ValueError(f"unsupported PMP format {f.suffix!r} ({f}); only .json is supported")
        with open(f) as fh:
            d = json.load(fh)
        if "objective" in d:
            if objective is not None and list(d["objective"]) != objective:
                raise ValueError(f"objective in {f} differs from earlier files")
            objective = list(d["objective"])
        if "normalization" in d:
            if normalization is not None and list(d["normalization"]) != normalization:
                raise ValueError(f"normalization in {f} differs from earlier files")
            normalization = list(d["normalization"])
        for i, m in enumerate(d.get("PositiveMatrixWithPrefactorArray", [])):
            matrices.append(_matrix_from_json(m, label=f"{f}[{i}]"))
    if objective is None:
        raise ValueError("no objective found in input files")
    return PMP(objective=objective, normalization=normalization, matrices=matrices)


def _num(value, bits: int) -> str:
    from .numbers import to_str

    return to_str(value, bits)


def pmp_to_json_dict(pmp: PMP, precision: int = 768) -> dict:
    """The ``pmp.json`` representation (numbers as strings)."""
    def dr(d: DampedRational) -> dict:
        return {"constant": _num(d.constant, precision), "base": _num(d.base, precision),
                "poles": [_num(p, precision) for p in d.poles]}

    out: dict = {"objective": [_num(a, precision) for a in pmp.objective]}
    if pmp.normalization is not None:
        out["normalization"] = [_num(n, precision) for n in pmp.normalization]
    arr = []
    for m in pmp.matrices:
        entry: dict = {}
        if m.prefactor is not None:
            entry["prefactor"] = dr(m.prefactor)
        if m.reduced_prefactor is not None:
            entry["reducedPrefactor"] = dr(m.reduced_prefactor)
        if m.max_num_poles is not None:
            entry["maxNumPoles"] = m.max_num_poles
        if m.sample_points is not None:
            entry["samplePoints"] = [_num(v, precision) for v in m.sample_points]
        if m.sample_scalings is not None:
            entry["sampleScalings"] = [_num(v, precision) for v in m.sample_scalings]
        if m.reduced_sample_scalings is not None:
            entry["reducedSampleScalings"] = [_num(v, precision) for v in m.reduced_sample_scalings]
        if m.bilinear_basis is not None:
            entry["bilinearBasis_0"] = [[_num(c, precision) for c in p.coeffs] for p in m.bilinear_basis[0]]
            entry["bilinearBasis_1"] = [[_num(c, precision) for c in p.coeffs] for p in m.bilinear_basis[1]]
        entry["polynomials"] = [[[[_num(c, precision) for c in p.coeffs] for p in e] for e in row]
                                for row in m.polynomials]
        arr.append(entry)
    out["PositiveMatrixWithPrefactorArray"] = arr
    return out


def write_pmp_json(pmp: PMP, path: str | os.PathLike, precision: int = 768) -> None:
    with open(path, "w") as fh:
        json.dump(pmp_to_json_dict(pmp, precision), fh, indent=1)
