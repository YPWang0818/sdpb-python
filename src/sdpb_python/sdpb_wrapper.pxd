# Cython declarations for the C++ shim in cpp/sdpb_wrapper.hxx.
from libc.stdint cimport int64_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "sdpb_wrapper.hxx" namespace "sdpb_python" nogil:
    string version() except +
    void initialize() except +
    void finalize() except +
    int mpi_rank() except +
    int mpi_size() except +
    void set_precision(size_t bits) except +
    size_t precision() except +
    size_t requested_precision() except +
    size_t max_digits10() except +
    void run(const vector[string]& args) except +

    cdef cppclass Damped_Rational_Spec:
        string constant
        string base
        vector[string] poles

    cdef cppclass Polynomial_Matrix_Spec:
        size_t dim
        vector[vector[vector[vector[string]]]] polynomials
        bool has_prefactor
        Damped_Rational_Spec prefactor
        bool has_reduced_prefactor
        Damped_Rational_Spec reduced_prefactor
        bool has_max_num_poles
        int64_t max_num_poles
        bool has_sample_points
        vector[string] sample_points
        bool has_sample_scalings
        vector[string] sample_scalings
        bool has_reduced_sample_scalings
        vector[string] reduced_sample_scalings
        bool has_bilinear_basis
        vector[vector[string]] bilinear_basis_even
        vector[vector[string]] bilinear_basis_odd

    cdef cppclass PMP_Spec:
        vector[string] objective
        bool has_normalization
        vector[string] normalization
        vector[Polynomial_Matrix_Spec] matrices

    cdef cppclass Matrix_Data:
        size_t height
        size_t width
        vector[string] elements

    cdef cppclass LMI_Spec:
        string f
        vector[string] b
        vector[vector[Matrix_Data]] blocks

    cdef cppclass Sampled_Matrix_Data:
        Damped_Rational_Spec prefactor
        Damped_Rational_Spec reduced_prefactor
        vector[string] sample_points
        vector[string] sample_scalings
        vector[string] reduced_sample_scalings
        vector[vector[string]] bilinear_basis_even
        vector[vector[string]] bilinear_basis_odd
        Matrix_Data bilinear_bases_even
        Matrix_Data bilinear_bases_odd

    Sampled_Matrix_Data sample_matrix(const Polynomial_Matrix_Spec& spec,
                                      size_t precision_bits) except +

    cdef cppclass Block_Data:
        size_t block_index
        size_t dim
        size_t num_points
        Matrix_Data bilinear_bases_even
        Matrix_Data bilinear_bases_odd
        vector[string] c
        Matrix_Data B

    cdef cppclass SDP_Data:
        string objective_const
        vector[string] b
        bool has_normalization
        vector[string] normalization
        vector[Block_Data] blocks

    SDP_Data pmp_to_sdp(const PMP_Spec& spec, size_t precision_bits) except +

    cdef cppclass Solver_Options:
        int64_t max_iterations
        int64_t max_runtime
        int64_t checkpoint_interval
        size_t max_shared_memory_bytes
        bool find_primal_feasible
        bool find_dual_feasible
        bool detect_primal_feasible_jump
        bool detect_dual_feasible_jump
        size_t precision
        string duality_gap_threshold
        string primal_error_threshold
        string dual_error_threshold
        string initial_matrix_scale_primal
        string initial_matrix_scale_dual
        string feasible_centering_parameter
        string infeasible_centering_parameter
        string step_length_reduction
        string max_complementarity
        string min_primal_step
        string min_dual_step
        string checkpoint_in
        string checkpoint_out
        string output_dir
        int verbosity
        bool want_x
        bool want_z
        bool want_X
        bool want_Y
        bool want_c_minus_By

    Solver_Options default_solver_options() except +

    cdef cppclass Solution_Data:
        string terminate_reason
        string primal_objective
        string dual_objective
        string duality_gap
        string primal_error
        string dual_error
        vector[string] y
        bool has_z
        vector[string] z
        vector[vector[string]] x
        vector[vector[string]] c_minus_By
        vector[Matrix_Data] X
        vector[Matrix_Data] Y
        int64_t iterations
        int64_t runtime_ms
        size_t precision
        vector[size_t] dims
        vector[size_t] num_points

    Solution_Data solve_pmp(const PMP_Spec& spec, const Solver_Options& options) except +
    Solution_Data solve_lmi(const LMI_Spec& spec, const Solver_Options& options) except +
