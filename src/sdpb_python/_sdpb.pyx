# distutils: language = c++
"""Low-level Cython bindings to SDPB.

Everything here works with plain Python containers of decimal strings.
Use :mod:`sdpb_python` for the public API.
"""

from libc.stdint cimport int64_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from . cimport sdpb_wrapper as cw


# ---------------------------------------------------------------- helpers

cdef string _s(object value) except *:
    return str(value).encode()


cdef vector[string] _strs(object values) except *:
    cdef vector[string] out
    for v in values:
        out.push_back(_s(v))
    return out


cdef list _pystrs(const vector[string]& values):
    return [values[i].decode() for i in range(values.size())]


cdef cw.Damped_Rational_Spec _damped_rational(dict d) except *:
    cdef cw.Damped_Rational_Spec spec
    spec.constant = _s(d["constant"])
    spec.base = _s(d["base"])
    spec.poles = _strs(d["poles"])
    return spec


cdef dict _py_damped_rational(const cw.Damped_Rational_Spec& spec):
    return {"constant": spec.constant.decode(), "base": spec.base.decode(),
            "poles": _pystrs(spec.poles)}


cdef cw.Matrix_Data _matrix(object rows) except *:
    """rows: list of lists of number strings (row-major)."""
    cdef cw.Matrix_Data m
    m.height = len(rows)
    m.width = len(rows[0]) if m.height else 0
    for row in rows:
        if len(row) != m.width:
            raise ValueError("ragged matrix")
        for v in row:
            m.elements.push_back(_s(v))
    return m


cdef dict _py_matrix(const cw.Matrix_Data& m):
    """{"shape": (h, w), "rows": [[str]]} -- shape kept even for empty matrices."""
    cdef size_t i, j
    return {"shape": (m.height, m.width),
            "rows": [[m.elements[i * m.width + j].decode() for j in range(m.width)]
                     for i in range(m.height)]}


cdef cw.Polynomial_Matrix_Spec _matrix_spec(dict d) except *:
    """d: dict with keys as produced by problem.PolynomialMatrix._to_spec()."""
    cdef cw.Polynomial_Matrix_Spec spec
    cdef vector[vector[vector[string]]] row
    cdef vector[vector[string]] entry
    spec.dim = d["dim"]
    for r in d["polynomials"]:
        row.clear()
        for s in r:
            entry.clear()
            for coeffs in s:
                entry.push_back(_strs(coeffs))
            row.push_back(entry)
        spec.polynomials.push_back(row)
    if d.get("prefactor") is not None:
        spec.has_prefactor = True
        spec.prefactor = _damped_rational(d["prefactor"])
    if d.get("reduced_prefactor") is not None:
        spec.has_reduced_prefactor = True
        spec.reduced_prefactor = _damped_rational(d["reduced_prefactor"])
    if d.get("max_num_poles") is not None:
        spec.has_max_num_poles = True
        spec.max_num_poles = d["max_num_poles"]
    if d.get("sample_points") is not None:
        spec.has_sample_points = True
        spec.sample_points = _strs(d["sample_points"])
    if d.get("sample_scalings") is not None:
        spec.has_sample_scalings = True
        spec.sample_scalings = _strs(d["sample_scalings"])
    if d.get("reduced_sample_scalings") is not None:
        spec.has_reduced_sample_scalings = True
        spec.reduced_sample_scalings = _strs(d["reduced_sample_scalings"])
    if d.get("bilinear_basis") is not None:
        spec.has_bilinear_basis = True
        even, odd = d["bilinear_basis"]
        for coeffs in even:
            spec.bilinear_basis_even.push_back(_strs(coeffs))
        for coeffs in odd:
            spec.bilinear_basis_odd.push_back(_strs(coeffs))
    return spec


cdef cw.PMP_Spec _pmp_spec(dict d) except *:
    cdef cw.PMP_Spec spec
    spec.objective = _strs(d["objective"])
    if d.get("normalization") is not None:
        spec.has_normalization = True
        spec.normalization = _strs(d["normalization"])
    for m in d["matrices"]:
        spec.matrices.push_back(_matrix_spec(m))
    return spec


cdef cw.Solver_Options _options(dict d) except *:
    cdef cw.Solver_Options o = cw.default_solver_options()
    o.max_iterations = d["max_iterations"]
    o.max_runtime = d["max_runtime"]
    o.checkpoint_interval = d["checkpoint_interval"]
    o.max_shared_memory_bytes = d["max_shared_memory_bytes"]
    o.find_primal_feasible = d["find_primal_feasible"]
    o.find_dual_feasible = d["find_dual_feasible"]
    o.detect_primal_feasible_jump = d["detect_primal_feasible_jump"]
    o.detect_dual_feasible_jump = d["detect_dual_feasible_jump"]
    o.precision = d["precision"]
    o.duality_gap_threshold = _s(d["duality_gap_threshold"])
    o.primal_error_threshold = _s(d["primal_error_threshold"])
    o.dual_error_threshold = _s(d["dual_error_threshold"])
    o.initial_matrix_scale_primal = _s(d["initial_matrix_scale_primal"])
    o.initial_matrix_scale_dual = _s(d["initial_matrix_scale_dual"])
    o.feasible_centering_parameter = _s(d["feasible_centering_parameter"])
    o.infeasible_centering_parameter = _s(d["infeasible_centering_parameter"])
    o.step_length_reduction = _s(d["step_length_reduction"])
    o.max_complementarity = _s(d["max_complementarity"])
    o.min_primal_step = _s(d["min_primal_step"])
    o.min_dual_step = _s(d["min_dual_step"])
    o.checkpoint_in = _s(d["checkpoint_in"])
    o.checkpoint_out = _s(d["checkpoint_out"])
    o.output_dir = _s(d["output_dir"])
    o.verbosity = d["verbosity"]
    o.want_x = d["want_x"]
    o.want_z = d["want_z"]
    o.want_X = d["want_X"]
    o.want_Y = d["want_Y"]
    o.want_c_minus_By = d["want_c_minus_By"]
    return o


cdef dict _py_solution(const cw.Solution_Data& s):
    cdef size_t i
    return {
        "terminate_reason": s.terminate_reason.decode(),
        "primal_objective": s.primal_objective.decode(),
        "dual_objective": s.dual_objective.decode(),
        "duality_gap": s.duality_gap.decode(),
        "primal_error": s.primal_error.decode(),
        "dual_error": s.dual_error.decode(),
        "y": _pystrs(s.y),
        "z": _pystrs(s.z) if s.has_z else None,
        "x": [_pystrs(s.x[i]) for i in range(s.x.size())],
        "c_minus_By": [_pystrs(s.c_minus_By[i]) for i in range(s.c_minus_By.size())],
        "X": [_py_matrix(s.X[i]) for i in range(s.X.size())],
        "Y": [_py_matrix(s.Y[i]) for i in range(s.Y.size())],
        "iterations": s.iterations,
        "runtime_ms": s.runtime_ms,
        "precision": s.precision,
        "dims": [s.dims[i] for i in range(s.dims.size())],
        "num_points": [s.num_points[i] for i in range(s.num_points.size())],
    }


# ---------------------------------------------------------------- session

def version() -> str:
    """Return the git version string of the bundled SDPB build."""
    return cw.version().decode()


def initialize() -> None:
    """Initialise MPI/Elemental.  Idempotent; called automatically."""
    cw.initialize()


def finalize() -> None:
    """Shut down MPI/Elemental.  Call at most once, at the end of the program."""
    cw.finalize()


def mpi_rank() -> int:
    return cw.mpi_rank()


def mpi_size() -> int:
    return cw.mpi_size()


def set_precision(bits: int) -> None:
    cw.set_precision(bits)


def precision() -> int:
    """Actual precision (bits) fixed for this process, or 0 if not yet fixed."""
    return cw.precision()


def requested_precision() -> int:
    """The precision requested by the first set_precision() call, or 0."""
    return cw.requested_precision()


def max_digits10() -> int:
    return cw.max_digits10()


def run(args) -> None:
    """Run the SDPB solver with ``sdpb``-style command line arguments."""
    cdef vector[string] cargs = _strs(args)
    with nogil:
        cw.run(cargs)


# ---------------------------------------------------------------- problems

def default_solver_options() -> dict:
    cdef cw.Solver_Options o = cw.default_solver_options()
    return {
        "max_iterations": o.max_iterations,
        "max_runtime": o.max_runtime,
        "checkpoint_interval": o.checkpoint_interval,
        "max_shared_memory_bytes": o.max_shared_memory_bytes,
        "find_primal_feasible": o.find_primal_feasible,
        "find_dual_feasible": o.find_dual_feasible,
        "detect_primal_feasible_jump": o.detect_primal_feasible_jump,
        "detect_dual_feasible_jump": o.detect_dual_feasible_jump,
        "precision": o.precision,
        "duality_gap_threshold": o.duality_gap_threshold.decode(),
        "primal_error_threshold": o.primal_error_threshold.decode(),
        "dual_error_threshold": o.dual_error_threshold.decode(),
        "initial_matrix_scale_primal": o.initial_matrix_scale_primal.decode(),
        "initial_matrix_scale_dual": o.initial_matrix_scale_dual.decode(),
        "feasible_centering_parameter": o.feasible_centering_parameter.decode(),
        "infeasible_centering_parameter": o.infeasible_centering_parameter.decode(),
        "step_length_reduction": o.step_length_reduction.decode(),
        "max_complementarity": o.max_complementarity.decode(),
        "min_primal_step": o.min_primal_step.decode(),
        "min_dual_step": o.min_dual_step.decode(),
    }


def sample_matrix(dict matrix_spec, size_t precision_bits) -> dict:
    """Sampling data (points, scalings, bases) the C++ constructor fills in."""
    cdef cw.Polynomial_Matrix_Spec spec = _matrix_spec(matrix_spec)
    cdef cw.Sampled_Matrix_Data s
    with nogil:
        s = cw.sample_matrix(spec, precision_bits)
    cdef size_t i
    return {
        "prefactor": _py_damped_rational(s.prefactor),
        "reduced_prefactor": _py_damped_rational(s.reduced_prefactor),
        "sample_points": _pystrs(s.sample_points),
        "sample_scalings": _pystrs(s.sample_scalings),
        "reduced_sample_scalings": _pystrs(s.reduced_sample_scalings),
        "bilinear_basis": (
            [_pystrs(s.bilinear_basis_even[i]) for i in range(s.bilinear_basis_even.size())],
            [_pystrs(s.bilinear_basis_odd[i]) for i in range(s.bilinear_basis_odd.size())],
        ),
        "bilinear_bases": (_py_matrix(s.bilinear_bases_even), _py_matrix(s.bilinear_bases_odd)),
    }


def pmp_to_sdp(dict pmp_spec, size_t precision_bits) -> dict:
    """Convert a PMP to SDP data (what pmp2sdp writes to an sdp/ directory)."""
    cdef cw.PMP_Spec spec = _pmp_spec(pmp_spec)
    cdef cw.SDP_Data s
    with nogil:
        s = cw.pmp_to_sdp(spec, precision_bits)
    cdef size_t i
    blocks = []
    for i in range(s.blocks.size()):
        blocks.append({
            "block_index": s.blocks[i].block_index,
            "dim": s.blocks[i].dim,
            "num_points": s.blocks[i].num_points,
            "bilinear_bases": (_py_matrix(s.blocks[i].bilinear_bases_even),
                               _py_matrix(s.blocks[i].bilinear_bases_odd)),
            "c": _pystrs(s.blocks[i].c),
            "B": _py_matrix(s.blocks[i].B),
        })
    return {
        "objective_const": s.objective_const.decode(),
        "b": _pystrs(s.b),
        "normalization": _pystrs(s.normalization) if s.has_normalization else None,
        "blocks": blocks,
    }


def solve_pmp(dict pmp_spec, dict options) -> dict:
    cdef cw.PMP_Spec spec = _pmp_spec(pmp_spec)
    cdef cw.Solver_Options o = _options(options)
    cdef cw.Solution_Data s
    with nogil:
        s = cw.solve_pmp(spec, o)
    return _py_solution(s)


cdef cw.LMI_Spec _lmi_spec(dict lmi_spec) except *:
    cdef cw.LMI_Spec spec
    cdef vector[cw.Matrix_Data] block
    spec.f = _s(lmi_spec["f"])
    spec.b = _strs(lmi_spec["b"])
    for matrices in lmi_spec["blocks"]:
        block.clear()
        for m in matrices:
            block.push_back(_matrix(m))
        spec.blocks.push_back(block)
    return spec


def solve_lmi(dict lmi_spec, dict options) -> dict:
    """lmi_spec: {"f": str, "b": [str], "blocks": [[rows...] * (N+1)] per block}"""
    cdef cw.LMI_Spec spec = _lmi_spec(lmi_spec)
    cdef cw.Solver_Options o = _options(options)
    cdef cw.Solution_Data s
    with nogil:
        s = cw.solve_lmi(spec, o)
    return _py_solution(s)


def last_run_interrupted() -> bool:
    """True if the last run stopped because SIGINT (Ctrl-C) was received."""
    return cw.last_run_interrupted()


cdef class Solver:
    """Handle owning SDPB's solver state (see sdpb_python.handle.Solver)."""
    cdef unique_ptr[cw.Solver] ptr

    def __cinit__(self, str kind, dict spec, dict options):
        cdef cw.PMP_Spec pmp
        cdef cw.LMI_Spec lmi
        cdef cw.Solver_Options o = _options(options)
        if kind == "pmp":
            pmp = _pmp_spec(spec)
            with nogil:
                self.ptr.reset(new cw.Solver(pmp, o))
        elif kind == "lmi":
            lmi = _lmi_spec(spec)
            with nogil:
                self.ptr.reset(new cw.Solver(lmi, o))
        else:
            raise ValueError("kind must be 'pmp' or 'lmi'")

    cdef cw.Solver* _get(self) except NULL:
        if not self.ptr:
            raise RuntimeError("solver is closed")
        return self.ptr.get()

    def close(self):
        self.ptr.reset()

    @property
    def closed(self) -> bool:
        return not self.ptr

    def run(self, dict options) -> dict:
        cdef cw.Solver* solver = self._get()
        cdef cw.Solver_Options o = _options(options)
        cdef cw.Solution_Data s
        with nogil:
            s = solver.run(o)
        return _py_solution(s)

    def state(self, dict options) -> dict:
        cdef cw.Solver* solver = self._get()
        cdef cw.Solver_Options o = _options(options)
        return _py_solution(solver.state(o))

    def set_y(self, y):
        self._get().set_y(_strs(y))

    def set_X(self, blocks):
        cdef vector[cw.Matrix_Data] mats
        for m in blocks:
            mats.push_back(_matrix(m))
        self._get().set_X(mats)

    def set_Y(self, blocks):
        cdef vector[cw.Matrix_Data] mats
        for m in blocks:
            mats.push_back(_matrix(m))
        self._get().set_Y(mats)

    def save_checkpoint(self, directory):
        self._get().save_checkpoint(_s(directory))

    def dims(self) -> list:
        cdef vector[size_t] d = self._get().dims()
        return [d[i] for i in range(d.size())]

    def num_points(self) -> list:
        cdef vector[size_t] d = self._get().num_points()
        return [d[i] for i in range(d.size())]

    def num_variables(self) -> int:
        return self._get().num_variables()

    def total_iterations(self) -> int:
        return self._get().total_iterations()
