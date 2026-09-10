// Thin C++ API over SDPB, kept free of Elemental/Boost/GMP types so that the
// Cython layer only has to know about std::string / std::vector / integers.
//
// Numbers cross this boundary as decimal strings printed with enough digits
// (max_digits10 for the current precision) to be lossless.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sdpb_python
{
  // ---- session -----------------------------------------------------------

  // SDPB git version string baked in at compile time.
  std::string version();
  // Initialise MPI / Elemental (idempotent).  Called automatically.
  void initialize();
  // Finalise MPI / Elemental.  Safe to call more than once.
  void finalize();
  int mpi_rank();
  int mpi_size();
  // Set the global GMP/MPFR precision (bits).  SDPB's Elemental allows this
  // once per process: the first call fixes the precision; later calls must
  // request the same value or throw.
  void set_precision(size_t bits);
  // Actual precision in bits (GMP rounds up), or 0 if not yet fixed.
  size_t precision();
  // The value passed to the first successful set_precision(), or 0.
  size_t requested_precision();
  // Number of decimal digits that identify a value at the current precision.
  size_t max_digits10();

  // Legacy: run the `sdpb` executable's main() logic with its argv.
  void run(const std::vector<std::string> &args);

  // ---- problem specification ---------------------------------------------

  struct Damped_Rational_Spec
  {
    std::string constant = "1", base = "1";
    std::vector<std::string> poles;
  };

  // One positivity constraint = one SDP block. Mirrors pmp.json fields.
  struct Polynomial_Matrix_Spec
  {
    size_t dim = 0;
    // polynomials[r][s][n] = coefficients a_0, a_1, ... of P^{rs}_n(x)
    std::vector<std::vector<std::vector<std::vector<std::string>>>> polynomials;
    bool has_prefactor = false;
    Damped_Rational_Spec prefactor;
    bool has_reduced_prefactor = false;
    Damped_Rational_Spec reduced_prefactor;
    bool has_max_num_poles = false;
    int64_t max_num_poles = -1;
    bool has_sample_points = false;
    std::vector<std::string> sample_points;
    bool has_sample_scalings = false;
    std::vector<std::string> sample_scalings;
    bool has_reduced_sample_scalings = false;
    std::vector<std::string> reduced_sample_scalings;
    bool has_bilinear_basis = false;
    // [m] = coefficients of the m-th basis polynomial
    std::vector<std::vector<std::string>> bilinear_basis_even,
      bilinear_basis_odd;
  };

  struct PMP_Spec
  {
    std::vector<std::string> objective; // a_0..a_N
    bool has_normalization = false;
    std::vector<std::string> normalization; // n_0..n_N
    std::vector<Polynomial_Matrix_Spec> matrices;
  };

  // Dense matrix, row-major.
  struct Matrix_Data
  {
    size_t height = 0, width = 0;
    std::vector<std::string> elements;
  };

  // Linear matrix inequality: maximize f + b.y s.t. M_0 + sum_n y_n M_n >= 0
  struct LMI_Spec
  {
    std::string f = "0";
    std::vector<std::string> b;
    // blocks[j][n] = block j of M_n, symmetric dim_j x dim_j, n = 0..N
    std::vector<std::vector<Matrix_Data>> blocks;
  };

  // ---- introspection -----------------------------------------------------

  // Fields filled in by the Polynomial_Vector_Matrix constructor.
  struct Sampled_Matrix_Data
  {
    Damped_Rational_Spec prefactor, reduced_prefactor;
    std::vector<std::string> sample_points, sample_scalings,
      reduced_sample_scalings;
    std::vector<std::vector<std::string>> bilinear_basis_even,
      bilinear_basis_odd;
    // sampled bases as written to block_data files
    Matrix_Data bilinear_bases_even, bilinear_bases_odd;
  };
  Sampled_Matrix_Data sample_matrix(const Polynomial_Matrix_Spec &spec,
                                    size_t precision_bits);

  // The content of an sdp/ directory written by pmp2sdp (Output_SDP).
  struct Block_Data
  {
    size_t block_index = 0, dim = 0, num_points = 0;
    Matrix_Data bilinear_bases_even, bilinear_bases_odd;
    std::vector<std::string> c;
    Matrix_Data B;
  };
  struct SDP_Data
  {
    std::string objective_const;
    std::vector<std::string> b;
    bool has_normalization = false;
    std::vector<std::string> normalization;
    std::vector<Block_Data> blocks;
  };
  SDP_Data pmp_to_sdp(const PMP_Spec &spec, size_t precision_bits);

  // ---- solving -----------------------------------------------------------

  struct Solver_Options
  {
    int64_t max_iterations = 500;
    int64_t max_runtime = INT64_MAX;
    int64_t checkpoint_interval = 3600;
    size_t max_shared_memory_bytes = 0;
    bool find_primal_feasible = false, find_dual_feasible = false,
         detect_primal_feasible_jump = false,
         detect_dual_feasible_jump = false;
    size_t precision = 400;
    std::string duality_gap_threshold = "1e-30",
                primal_error_threshold = "1e-30",
                dual_error_threshold = "1e-30",
                initial_matrix_scale_primal = "1e20",
                initial_matrix_scale_dual = "1e20",
                feasible_centering_parameter = "0.1",
                infeasible_centering_parameter = "0.3",
                step_length_reduction = "0.7", max_complementarity = "1e100",
                min_primal_step = "0", min_dual_step = "0";
    // checkpoint_in must not be empty (an empty path means CWD in SDPB).
    std::string checkpoint_in, checkpoint_out;
    // If non-empty: iterations.json and c_minus_By/ are written there.
    std::string output_dir;
    int verbosity = 0; // 0 none, 1 regular, 2 debug, 3 trace
    bool want_x = false, want_z = true, want_X = false, want_Y = false,
         want_c_minus_By = false;
  };
  // Defaults from Solver_Parameters::options() (call after set_precision).
  Solver_Options default_solver_options();

  struct Solution_Data
  {
    std::string terminate_reason;
    std::string primal_objective, dual_objective, duality_gap, primal_error,
      dual_error;
    std::vector<std::string> y;
    bool has_z = false;
    std::vector<std::string> z;
    // indexed by global block; empty unless requested
    std::vector<std::vector<std::string>> x, c_minus_By;
    std::vector<Matrix_Data> X, Y; // 2 per block: [even, odd]
    int64_t iterations = 0;
    int64_t runtime_ms = 0;
    size_t precision = 0;
    std::vector<size_t> dims, num_points;
  };

  Solution_Data solve_pmp(const PMP_Spec &spec, const Solver_Options &options);
  Solution_Data solve_lmi(const LMI_Spec &spec, const Solver_Options &options);
}
