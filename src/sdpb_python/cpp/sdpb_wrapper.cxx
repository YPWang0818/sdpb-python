#include "sdpb_wrapper.hxx"

#include "pmp/Polynomial_Matrix_Program.hxx"
#include "pmp/max_normalization_index.hxx"
#include "pmp2sdp/Dual_Constraint_Group.hxx"
#include "pmp2sdp/Output_SDP/Output_SDP.hxx"
#include "sdp_solve/sdp_solve.hxx"
#include "sdpb/SDPB_Parameters.hxx"
#include "sdpb_util/Environment.hxx"
#include "sdpb_util/Timers/Timers.hxx"
#include "sdpb_util/copy_matrix.hxx"
#include "sdpb_util/fill_weights.hxx"
#include "sdpb_util/ostream/set_stream_precision.hxx"

#include <El.hpp>

#include <boost/program_options.hpp>

#include <chrono>
#include <cmath>
#include <csignal>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

// Defined in c-src/sdpb/src/sdpb/solve.cxx and write_timing.cxx.
Timers solve(const Block_Info &block_info, const SDPB_Parameters &parameters,
             const Environment &env,
             const std::chrono::time_point<std::chrono::high_resolution_clock>
               &start_time,
             El::Matrix<int32_t> &block_timings_ms);

void write_block_timings(const fs::path &checkpoint_out,
                         const Block_Info &block_info,
                         const El::Matrix<int32_t> &block_timings_ms,
                         Verbosity verbosity);

namespace sdpb_python
{
  // ======================================================================
  // session
  // ======================================================================
  namespace
  {
    // Elemental/MPI may only be initialised once per process, so keep a
    // single Environment alive for the lifetime of the module.
    std::unique_ptr<Environment> global_env;
    // SDPB is not reentrant.
    std::mutex solve_mutex;
    // Precision requested by the first set_precision() call (0 = none yet).
    size_t requested_precision_bits = 0;
    // Set by the SIGINT handler installed during run().
    volatile std::sig_atomic_t sigint_flag = 0;
    bool last_interrupted = false;

    void handle_sigint(int)
    {
      sigint_flag = 1;
      Environment::request_termination();
    }

    // Installs a SIGINT handler that asks SDPB to stop at the next
    // iteration; restores the previous handler (Python's) afterwards.
    struct Sigint_Guard
    {
      void (*previous)(int);
      Sigint_Guard()
      {
        sigint_flag = 0;
        Environment::clear_termination_request();
        previous = std::signal(SIGINT, handle_sigint);
      }
      ~Sigint_Guard()
      {
        std::signal(SIGINT, previous);
        last_interrupted = sigint_flag != 0;
        Environment::clear_termination_request();
      }
    };

    void require_single_rank()
    {
      if(El::mpi::Size() > 1)
        throw std::runtime_error(
          "sdpb_python supports a single MPI rank; run without mpirun "
          "(or with -n 1). Got "
          + std::to_string(El::mpi::Size()) + " ranks.");
    }

    Environment &env()
    {
      initialize();
      return *global_env;
    }

    // ---- number conversion ----
    std::string to_str(const El::BigFloat &x)
    {
      std::ostringstream ss;
      set_stream_precision(ss);
      ss << x;
      return ss.str();
    }
    std::string to_str(const Boost_Float &x)
    {
      std::ostringstream ss;
      set_stream_precision(ss);
      ss << x;
      return ss.str();
    }
    El::BigFloat to_bigfloat(const std::string &s)
    {
      if(s.empty())
        throw std::invalid_argument("empty number string");
      return El::BigFloat(s);
    }
    Boost_Float to_boost_float(const std::string &s)
    {
      if(s.empty())
        throw std::invalid_argument("empty number string");
      return Boost_Float(s);
    }
    std::vector<El::BigFloat> to_bigfloats(const std::vector<std::string> &v)
    {
      std::vector<El::BigFloat> out;
      out.reserve(v.size());
      for(const auto &s : v)
        out.push_back(to_bigfloat(s));
      return out;
    }
    std::vector<std::string> to_strs(const std::vector<El::BigFloat> &v)
    {
      std::vector<std::string> out;
      out.reserve(v.size());
      for(const auto &x : v)
        out.push_back(to_str(x));
      return out;
    }
    std::vector<std::string> to_strs(const std::vector<Boost_Float> &v)
    {
      std::vector<std::string> out;
      out.reserve(v.size());
      for(const auto &x : v)
        out.push_back(to_str(x));
      return out;
    }
    Matrix_Data to_matrix_data(const El::Matrix<El::BigFloat> &m)
    {
      Matrix_Data out;
      out.height = m.Height();
      out.width = m.Width();
      out.elements.reserve(out.height * out.width);
      for(size_t i = 0; i < out.height; ++i)
        for(size_t j = 0; j < out.width; ++j)
          out.elements.push_back(to_str(m.Get(i, j)));
      return out;
    }
    El::Matrix<El::BigFloat> to_local(const El::DistMatrix<El::BigFloat> &d)
    {
      El::DistMatrix<El::BigFloat, El::STAR, El::STAR> star(d);
      El::Matrix<El::BigFloat> local;
      copy_matrix(star, local);
      return local;
    }
    std::vector<std::string> column_to_strs(const El::Matrix<El::BigFloat> &m)
    {
      std::vector<std::string> out;
      out.reserve(m.Height());
      for(int i = 0; i < m.Height(); ++i)
        out.push_back(to_str(m.Get(i, 0)));
      return out;
    }

    Damped_Rational to_damped_rational(const Damped_Rational_Spec &spec)
    {
      Damped_Rational d;
      d.constant = to_boost_float(spec.constant);
      d.base = to_boost_float(spec.base);
      for(const auto &p : spec.poles)
        d.poles.push_back(to_boost_float(p));
      return d;
    }
    Damped_Rational_Spec to_spec(const Damped_Rational &d)
    {
      Damped_Rational_Spec spec;
      spec.constant = to_str(d.constant);
      spec.base = to_str(d.base);
      spec.poles = to_strs(d.poles);
      return spec;
    }
    Polynomial_Vector
    to_polynomial_vector(const std::vector<std::vector<std::string>> &polys)
    {
      Polynomial_Vector out;
      for(const auto &coeffs : polys)
        {
          Polynomial p;
          p.coefficients = to_bigfloats(coeffs);
          if(p.coefficients.empty())
            p.coefficients.push_back(El::BigFloat(0));
          out.push_back(std::move(p));
        }
      return out;
    }
    std::vector<std::vector<std::string>>
    to_coefficient_lists(const Polynomial_Vector &polys)
    {
      std::vector<std::vector<std::string>> out;
      for(const auto &p : polys)
        out.push_back(to_strs(p.coefficients));
      return out;
    }

    Verbosity to_verbosity(const int v)
    {
      switch(v)
        {
        case 0: return Verbosity::none;
        case 1: return Verbosity::regular;
        case 2: return Verbosity::debug;
        case 3: return Verbosity::trace;
        default: throw std::invalid_argument("verbosity must be 0..3");
        }
    }
  }

  std::string version() { return SDPB_VERSION_STRING; }

  void initialize()
  {
    if(!global_env)
      global_env = std::make_unique<Environment>();
  }

  void finalize() { global_env.reset(); }

  int mpi_rank()
  {
    initialize();
    return El::mpi::Rank();
  }

  int mpi_size()
  {
    initialize();
    return El::mpi::Size();
  }

  void set_precision(const size_t bits)
  {
    initialize();
    if(bits < 64)
      throw std::invalid_argument("precision must be at least 64 bits");
    if(requested_precision_bits == 0)
      {
        Environment::set_precision(bits);
        requested_precision_bits = bits;
        return;
      }
    if(bits == requested_precision_bits || bits == El::gmp::Precision())
      return;
    throw std::runtime_error(
      "precision is already fixed at " + std::to_string(requested_precision_bits)
      + " bits for this process (SDPB's Elemental allows setting it once); "
        "cannot switch to "
      + std::to_string(bits) + " bits. Use a separate process.");
  }

  size_t precision()
  {
    initialize();
    return requested_precision_bits == 0 ? 0 : El::gmp::Precision();
  }

  size_t requested_precision() { return requested_precision_bits; }

  size_t max_digits10()
  {
    initialize();
    return std::ceil(El::gmp::Precision() * std::log10(2.0)) + 1;
  }

  // ======================================================================
  // legacy: sdpb main() with argv
  // ======================================================================
  void run(const std::vector<std::string> &args)
  {
    std::lock_guard<std::mutex> lock(solve_mutex);
    const Environment &environment = env();

    std::vector<std::string> argv_storage;
    argv_storage.reserve(args.size() + 1);
    argv_storage.emplace_back("sdpb");
    argv_storage.insert(argv_storage.end(), args.begin(), args.end());
    std::vector<char *> argv;
    for(auto &s : argv_storage)
      argv.push_back(s.data());

    SDPB_Parameters parameters(static_cast<int>(argv.size()), argv.data());
    if(!parameters.is_valid())
      throw std::runtime_error("sdpb: invalid or missing parameters");

    set_precision(parameters.solver.precision);
    const auto start_time = std::chrono::high_resolution_clock::now();

    Block_Info block_info(environment, parameters.sdp_path,
                          parameters.solver.checkpoint_in,
                          parameters.proc_granularity, parameters.verbosity);

    if(El::mpi::Size(El::mpi::COMM_WORLD) > 1
       && block_info.block_timings_filename.empty()
       && !exists(parameters.solver.checkpoint_in / "checkpoint.0"))
      {
        SDPB_Parameters timing_parameters(parameters);
        timing_parameters.solver.max_iterations = 2;
        timing_parameters.no_final_checkpoint = true;
        timing_parameters.solver.checkpoint_interval
          = std::numeric_limits<int64_t>::max();
        timing_parameters.solver.max_runtime
          = std::numeric_limits<int64_t>::max();
        timing_parameters.solver.duality_gap_threshold = 0;
        timing_parameters.solver.primal_error_threshold = 0;
        timing_parameters.solver.dual_error_threshold = 0;
        timing_parameters.solver.min_primal_step = 0;
        timing_parameters.solver.min_dual_step = 0;
        if(timing_parameters.verbosity < Verbosity::debug)
          timing_parameters.verbosity = Verbosity::none;

        El::Matrix<int32_t> block_timings_ms;
        solve(block_info, timing_parameters, environment, start_time,
              block_timings_ms);
        if(block_timings_ms.Height() == 0 && block_timings_ms.Width() == 0)
          throw std::runtime_error(
            "block_timings vector is empty: timing run exited before "
            "completing two solver iterations.");

        write_block_timings(timing_parameters.solver.checkpoint_out,
                            block_info, block_timings_ms,
                            timing_parameters.verbosity);
        El::mpi::Barrier(El::mpi::COMM_WORLD);
        Block_Info new_info(environment, parameters.sdp_path,
                            block_timings_ms, parameters.proc_granularity,
                            parameters.verbosity);
        swap(block_info, new_info);

        const auto elapsed_seconds
          = std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::high_resolution_clock::now() - start_time)
              .count();
        parameters.solver.max_runtime -= elapsed_seconds;
      }
    else if(!block_info.block_timings_filename.empty()
            && block_info.block_timings_filename
                 != (parameters.solver.checkpoint_out / "block_timings"))
      {
        if(El::mpi::Rank() == 0)
          {
            create_directories(parameters.solver.checkpoint_out);
            copy_file(block_info.block_timings_filename,
                      parameters.solver.checkpoint_out / "block_timings",
                      fs::copy_options::overwrite_existing);
          }
      }

    El::Matrix<int32_t> block_timings_ms;
    solve(block_info, parameters, environment, start_time, block_timings_ms);
  }

  // ======================================================================
  // PMP construction
  // ======================================================================
  namespace
  {
    Polynomial_Vector_Matrix to_pvm(const Polynomial_Matrix_Spec &spec)
    {
      const size_t dim = spec.dim;
      if(spec.polynomials.size() != dim)
        throw std::invalid_argument("polynomials must have dim rows");
      Simple_Matrix<Polynomial_Vector> polynomials(dim, dim);
      for(size_t r = 0; r < dim; ++r)
        {
          if(spec.polynomials[r].size() != dim)
            throw std::invalid_argument("polynomials must have dim columns");
          for(size_t s = 0; s < dim; ++s)
            polynomials(r, s) = to_polynomial_vector(spec.polynomials[r][s]);
        }

      std::optional<Damped_Rational> prefactor, reduced_prefactor;
      if(spec.has_prefactor)
        prefactor = to_damped_rational(spec.prefactor);
      if(spec.has_reduced_prefactor)
        reduced_prefactor = to_damped_rational(spec.reduced_prefactor);
      std::optional<int64_t> max_num_poles;
      if(spec.has_max_num_poles)
        max_num_poles = spec.max_num_poles;
      std::optional<std::vector<El::BigFloat>> sample_points, sample_scalings,
        reduced_sample_scalings;
      if(spec.has_sample_points)
        sample_points = to_bigfloats(spec.sample_points);
      if(spec.has_sample_scalings)
        sample_scalings = to_bigfloats(spec.sample_scalings);
      if(spec.has_reduced_sample_scalings)
        reduced_sample_scalings = to_bigfloats(spec.reduced_sample_scalings);
      std::optional<std::array<Polynomial_Vector, 2>> bilinear_basis;
      if(spec.has_bilinear_basis)
        bilinear_basis
          = std::array<Polynomial_Vector, 2>{
            to_polynomial_vector(spec.bilinear_basis_even),
            to_polynomial_vector(spec.bilinear_basis_odd)};

      return Polynomial_Vector_Matrix(polynomials, prefactor,
                                      reduced_prefactor, max_num_poles,
                                      sample_points, sample_scalings,
                                      reduced_sample_scalings, bilinear_basis);
    }

    // All matrices are local (single process).
    Polynomial_Matrix_Program to_pmp(const PMP_Spec &spec)
    {
      if(spec.matrices.empty())
        throw std::invalid_argument("PMP has no matrices");
      std::vector<Polynomial_Vector_Matrix> matrices;
      std::vector<size_t> local_to_global;
      std::vector<fs::path> block_paths;
      for(size_t j = 0; j < spec.matrices.size(); ++j)
        {
          matrices.push_back(to_pvm(spec.matrices[j]));
          local_to_global.push_back(j);
          block_paths.emplace_back("in-memory/block_" + std::to_string(j));
        }
      std::optional<std::vector<El::BigFloat>> normalization;
      if(spec.has_normalization)
        normalization = to_bigfloats(spec.normalization);
      return Polynomial_Matrix_Program(
        to_bigfloats(spec.objective), normalization, spec.matrices.size(),
        std::move(matrices), std::move(local_to_global),
        std::move(block_paths));
    }
  }

  Sampled_Matrix_Data sample_matrix(const Polynomial_Matrix_Spec &spec,
                                    const size_t precision_bits)
  {
    std::lock_guard<std::mutex> lock(solve_mutex);
    set_precision(precision_bits);
    const Polynomial_Vector_Matrix pvm = to_pvm(spec);
    Sampled_Matrix_Data out;
    out.prefactor = to_spec(pvm.prefactor);
    out.reduced_prefactor = to_spec(pvm.reduced_prefactor);
    out.sample_points = to_strs(pvm.sample_points);
    out.sample_scalings = to_strs(pvm.sample_scalings);
    out.reduced_sample_scalings = to_strs(pvm.reduced_sample_scalings);
    out.bilinear_basis_even = to_coefficient_lists(pvm.bilinear_basis[0]);
    out.bilinear_basis_odd = to_coefficient_lists(pvm.bilinear_basis[1]);
    const Dual_Constraint_Group group(0, pvm);
    out.bilinear_bases_even = to_matrix_data(group.bilinear_bases[0]);
    out.bilinear_bases_odd = to_matrix_data(group.bilinear_bases[1]);
    return out;
  }

  SDP_Data pmp_to_sdp(const PMP_Spec &spec, const size_t precision_bits)
  {
    std::lock_guard<std::mutex> lock(solve_mutex);
    set_precision(precision_bits);
    Timers timers(env(), Verbosity::none);
    const Polynomial_Matrix_Program pmp = to_pmp(spec);
    const Output_SDP sdp(pmp, {"sdpb_python"}, timers);

    SDP_Data out;
    out.objective_const = to_str(sdp.objective_const);
    out.b = to_strs(sdp.dual_objective_b);
    out.has_normalization = sdp.normalization.has_value();
    if(out.has_normalization)
      out.normalization = to_strs(sdp.normalization.value());
    for(const auto &group : sdp.dual_constraint_groups)
      {
        Block_Data block;
        block.block_index = group.block_index;
        block.dim = group.dim;
        block.num_points = group.num_points;
        block.bilinear_bases_even = to_matrix_data(group.bilinear_bases[0]);
        block.bilinear_bases_odd = to_matrix_data(group.bilinear_bases[1]);
        block.c = to_strs(group.constraint_constants);
        block.B = to_matrix_data(group.constraint_matrix);
        out.blocks.push_back(std::move(block));
      }
    return out;
  }

  // ======================================================================
  // solver
  // ======================================================================
  namespace
  {
    Solver_Parameters make_solver_parameters(const Solver_Options &o)
    {
      // Start from SDPB's own defaults so that nothing stays uninitialised.
      Solver_Parameters p;
      auto description = p.options();
      boost::program_options::variables_map vm;
      const char *argv[] = {"sdpb"};
      boost::program_options::store(
        boost::program_options::parse_command_line(1, argv, description), vm);
      boost::program_options::notify(vm);

      p.max_iterations = o.max_iterations;
      p.max_runtime = o.max_runtime;
      p.checkpoint_interval = o.checkpoint_interval;
      p.max_shared_memory_bytes = o.max_shared_memory_bytes;
      p.find_primal_feasible = o.find_primal_feasible;
      p.find_dual_feasible = o.find_dual_feasible;
      p.detect_primal_feasible_jump = o.detect_primal_feasible_jump;
      p.detect_dual_feasible_jump = o.detect_dual_feasible_jump;
      p.precision = o.precision;
      p.duality_gap_threshold = to_bigfloat(o.duality_gap_threshold);
      p.primal_error_threshold = to_bigfloat(o.primal_error_threshold);
      p.dual_error_threshold = to_bigfloat(o.dual_error_threshold);
      p.initial_matrix_scale_primal
        = to_bigfloat(o.initial_matrix_scale_primal);
      p.initial_matrix_scale_dual = to_bigfloat(o.initial_matrix_scale_dual);
      p.feasible_centering_parameter
        = to_bigfloat(o.feasible_centering_parameter);
      p.infeasible_centering_parameter
        = to_bigfloat(o.infeasible_centering_parameter);
      p.step_length_reduction = to_bigfloat(o.step_length_reduction);
      p.max_complementarity = to_bigfloat(o.max_complementarity);
      p.min_primal_step = to_bigfloat(o.min_primal_step);
      p.min_dual_step = to_bigfloat(o.min_dual_step);
      if(o.checkpoint_in.empty())
        throw std::invalid_argument(
          "checkpoint_in must not be empty (SDPB would read the CWD)");
      p.checkpoint_in = o.checkpoint_in;
      p.checkpoint_out = o.checkpoint_out;
      return p;
    }

    std::string to_str(const SDP_Solver_Terminate_Reason &reason)
    {
      std::ostringstream ss;
      ss << reason;
      return ss.str();
    }

    // Gathers results from a solver (after run(), or its current state).
    Solution_Data
    collect(const SDP_Solver &solver, const SDP &sdp,
            const Block_Info &block_info, const Solver_Options &o,
            const std::string &terminate_reason, const int64_t runtime_ms)
    {
      const size_t N = sdp.dual_objective_b.Height();
      Solution_Data out;
      out.terminate_reason = terminate_reason;
      out.runtime_ms = runtime_ms;
      out.primal_objective = to_str(solver.primal_objective);
      out.dual_objective = to_str(solver.dual_objective);
      out.duality_gap = to_str(solver.duality_gap);
      out.primal_error = to_str(solver.primal_error());
      out.dual_error = to_str(solver.dual_error);
      out.iterations = solver.num_iterations;
      out.precision = El::gmp::Precision();
      out.dims = block_info.dimensions;
      out.num_points = block_info.num_points;

      // y is duplicated on every local block
      El::Matrix<El::BigFloat> y_local;
      if(!solver.y.blocks.empty())
        y_local = to_local(solver.y.blocks.at(0));
      else
        y_local.Resize(N, 1);
      out.y = column_to_strs(y_local);

      if(o.want_z && sdp.normalization.has_value())
        {
          const auto &normalization = sdp.normalization.value();
          std::vector<El::BigFloat> z(normalization.size());
          fill_weights(y_local, max_normalization_index(normalization),
                       normalization, z);
          out.has_z = true;
          out.z = to_strs(z);
        }

      const size_t num_blocks = block_info.dimensions.size();
      const size_t num_local = block_info.block_indices.size();
      if(o.want_x)
        out.x.resize(num_blocks);
      if(o.want_c_minus_By)
        out.c_minus_By.resize(num_blocks);
      if(o.want_X)
        out.X.resize(2 * num_blocks);
      if(o.want_Y)
        out.Y.resize(2 * num_blocks);
      for(size_t i = 0; i < num_local; ++i)
        {
          const size_t j = block_info.block_indices.at(i);
          if(o.want_x)
            out.x.at(j) = column_to_strs(to_local(solver.x.blocks.at(i)));
          if(o.want_c_minus_By)
            {
              El::Matrix<El::BigFloat> c
                = to_local(sdp.primal_objective_c.blocks.at(i));
              const El::Matrix<El::BigFloat> B
                = to_local(sdp.free_var_matrix.blocks.at(i));
              // c := c - B y
              El::Gemv(El::NORMAL, El::BigFloat(-1), B, y_local,
                       El::BigFloat(1), c);
              out.c_minus_By.at(j) = column_to_strs(c);
            }
          for(const size_t parity : {0, 1})
            {
              if(o.want_X)
                out.X.at(2 * j + parity)
                  = to_matrix_data(to_local(solver.X.blocks.at(2 * i + parity)));
              if(o.want_Y)
                out.Y.at(2 * j + parity)
                  = to_matrix_data(to_local(solver.Y.blocks.at(2 * i + parity)));
            }
        }
      return out;
    }

    // Runs the interior-point iteration on an existing solver.
    Solution_Data run_existing(SDP_Solver &solver, const SDP &sdp,
                               const Block_Info &block_info,
                               const El::Grid &grid, const Solver_Options &o,
                               Timers &timers)
    {
      const Environment &environment = env();
      const Verbosity verbosity = to_verbosity(o.verbosity);
      const Solver_Parameters parameters = make_solver_parameters(o);
      const auto start_time = std::chrono::high_resolution_clock::now();

      fs::path iterations_json_path;
      if(!o.output_dir.empty())
        {
          fs::create_directories(o.output_dir);
          iterations_json_path = fs::path(o.output_dir) / "iterations.json";
        }
      El::Matrix<int32_t> block_timings_ms;
      SDP_Solver_Terminate_Reason reason;
      {
        Sigint_Guard guard;
        reason = solver.run(environment, parameters, verbosity,
                            to_property_tree(parameters), block_info, sdp,
                            grid, start_time, iterations_json_path, timers,
                            block_timings_ms);
      }
      const auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time);

      if(!parameters.checkpoint_out.empty())
        solver.save_checkpoint(parameters.checkpoint_out, verbosity,
                               to_property_tree(parameters));

      return collect(solver, sdp, block_info, o, to_str(reason),
                     runtime.count());
    }

    // Builds a fresh solver, runs it, gathers the results.
    Solution_Data run_solver(const SDP &sdp, const Block_Info &block_info,
                             const El::Grid &grid, const Solver_Options &o,
                             Timers &timers)
    {
      const Verbosity verbosity = to_verbosity(o.verbosity);
      const Solver_Parameters parameters = make_solver_parameters(o);
      const size_t N = sdp.dual_objective_b.Height();
      SDP_Solver solver(parameters, verbosity, false, block_info, grid, N);
      return run_existing(solver, sdp, block_info, grid, o, timers);
    }
  }

  bool last_run_interrupted() { return last_interrupted; }

  Solver_Options default_solver_options()
  {
    initialize();
    Solver_Parameters p;
    auto description = p.options();
    boost::program_options::variables_map vm;
    const char *argv[] = {"sdpb"};
    boost::program_options::store(
      boost::program_options::parse_command_line(1, argv, description), vm);
    boost::program_options::notify(vm);

    Solver_Options o;
    o.max_iterations = p.max_iterations;
    o.max_runtime = p.max_runtime;
    o.checkpoint_interval = p.checkpoint_interval;
    o.max_shared_memory_bytes = p.max_shared_memory_bytes;
    o.find_primal_feasible = p.find_primal_feasible;
    o.find_dual_feasible = p.find_dual_feasible;
    o.detect_primal_feasible_jump = p.detect_primal_feasible_jump;
    o.detect_dual_feasible_jump = p.detect_dual_feasible_jump;
    o.precision = p.precision;
    o.duality_gap_threshold = to_str(p.duality_gap_threshold);
    o.primal_error_threshold = to_str(p.primal_error_threshold);
    o.dual_error_threshold = to_str(p.dual_error_threshold);
    o.initial_matrix_scale_primal = to_str(p.initial_matrix_scale_primal);
    o.initial_matrix_scale_dual = to_str(p.initial_matrix_scale_dual);
    o.feasible_centering_parameter = to_str(p.feasible_centering_parameter);
    o.infeasible_centering_parameter
      = to_str(p.infeasible_centering_parameter);
    o.step_length_reduction = to_str(p.step_length_reduction);
    o.max_complementarity = to_str(p.max_complementarity);
    o.min_primal_step = to_str(p.min_primal_step);
    o.min_dual_step = to_str(p.min_dual_step);
    return o;
  }

  // ---- problem = Block_Info + Grid + SDP (destroyed in reverse order) ----
  namespace
  {
    struct Problem
    {
      std::unique_ptr<Block_Info> block_info;
      std::unique_ptr<El::Grid> grid;
      std::unique_ptr<SDP> sdp;
      Problem() = default;
      Problem(Problem &&) = default;
      Problem &operator=(Problem &&) = default;
      ~Problem()
      {
        sdp.reset();
        grid.reset();
        block_info.reset();
      }
    };

    Problem build_pmp_problem(const PMP_Spec &spec, const Verbosity verbosity,
                              Timers &timers)
    {
      const Polynomial_Matrix_Program pmp = to_pmp(spec);
      const Output_SDP output_sdp(pmp, {"sdpb_python"}, timers);

      std::vector<size_t> dims, num_points;
      for(const auto &m : pmp.matrices)
        {
          dims.push_back(m.polynomials.Height());
          num_points.push_back(m.sample_points.size());
        }
      Problem p;
      p.block_info
        = std::make_unique<Block_Info>(env(), dims, num_points, 1, verbosity);
      p.grid = std::make_unique<El::Grid>(p.block_info->mpi_comm.value);

      // groups in the order of block_info.block_indices
      std::vector<Dual_Constraint_Group> groups;
      for(const size_t j : p.block_info->block_indices)
        {
          const auto &g = output_sdp.dual_constraint_groups;
          const auto it
            = std::find_if(g.begin(), g.end(), [j](const auto &group) {
                return group.block_index == j;
              });
          if(it == g.end())
            throw std::logic_error("missing Dual_Constraint_Group");
          groups.push_back(*it);
        }
      p.sdp = std::make_unique<SDP>(
        output_sdp.objective_const, output_sdp.dual_objective_b, groups,
        output_sdp.normalization, *p.block_info, *p.grid);
      return p;
    }

    Problem build_lmi_problem(const LMI_Spec &spec, const Verbosity verbosity)
    {
      const size_t N = spec.b.size();
      if(spec.blocks.empty())
        throw std::invalid_argument("LMI has no blocks");
      std::vector<size_t> dims;
      std::vector<std::vector<El::BigFloat>> c;
      std::vector<El::Matrix<El::BigFloat>> B;
      for(size_t j = 0; j < spec.blocks.size(); ++j)
        {
          const auto &M = spec.blocks[j];
          if(M.size() != N + 1)
            throw std::invalid_argument("each block needs N+1 matrices");
          const size_t d = M[0].height;
          for(const auto &m : M)
            if(m.height != d || m.width != d || m.elements.size() != d * d)
              throw std::invalid_argument("block matrices must be square and "
                                          "of equal size");
          dims.push_back(d);
          c.emplace_back();
          B.emplace_back(d * (d + 1) / 2, N);
          size_t p = 0;
          for(size_t s = 0; s < d; ++s)
            for(size_t r = 0; r <= s; ++r, ++p)
              {
                c[j].push_back(to_bigfloat(M[0].elements[r * d + s]));
                for(size_t n = 0; n < N; ++n)
                  B[j](p, n) = -to_bigfloat(M[n + 1].elements[r * d + s]);
              }
        }

      Problem p;
      p.block_info = std::make_unique<Block_Info>(env(), dims, verbosity);
      p.grid = std::make_unique<El::Grid>(p.block_info->mpi_comm.value);
      const El::Grid global_grid;
      El::DistMatrix<El::BigFloat, El::STAR, El::STAR> yp_to_y(N, N,
                                                               global_grid),
        b_star(N, 1, global_grid);
      El::Identity(yp_to_y, N, N);
      for(size_t n = 0; n < N; ++n)
        b_star.Set(n, 0, to_bigfloat(spec.b[n]));
      std::vector<El::BigFloat> normalization(N + 1, El::BigFloat(0));
      normalization[0] = 1;
      p.sdp = std::make_unique<SDP>(to_bigfloat(spec.f), c, B, yp_to_y, b_star,
                                    normalization, El::BigFloat(1),
                                    *p.block_info, *p.grid);
      return p;
    }
  }

  Solution_Data solve_pmp(const PMP_Spec &spec, const Solver_Options &o)
  {
    std::lock_guard<std::mutex> lock(solve_mutex);
    require_single_rank();
    set_precision(o.precision);
    const Verbosity verbosity = to_verbosity(o.verbosity);
    Timers timers(env(), verbosity);
    const Problem p = build_pmp_problem(spec, verbosity, timers);
    return run_solver(*p.sdp, *p.block_info, *p.grid, o, timers);
  }

  Solution_Data solve_lmi(const LMI_Spec &spec, const Solver_Options &o)
  {
    std::lock_guard<std::mutex> lock(solve_mutex);
    require_single_rank();
    set_precision(o.precision);
    const Verbosity verbosity = to_verbosity(o.verbosity);
    Timers timers(env(), verbosity);
    const Problem p = build_lmi_problem(spec, verbosity);
    Solver_Options options = o;
    options.want_z = false;
    return run_solver(*p.sdp, *p.block_info, *p.grid, options, timers);
  }

  // ---- Solver handle -----------------------------------------------------
  struct Solver::Impl
  {
    Verbosity verbosity;
    Timers timers;
    Problem problem;
    std::unique_ptr<SDP_Solver> solver;
    Impl(const Verbosity v) : verbosity(v), timers(env(), v) {}
    ~Impl() { solver.reset(); }
  };

  namespace
  {
    std::unique_ptr<SDP_Solver>
    make_sdp_solver(const Problem &p, const Solver_Options &o,
                    const Verbosity verbosity)
    {
      const Solver_Parameters parameters = make_solver_parameters(o);
      const size_t N = p.sdp->dual_objective_b.Height();
      return std::make_unique<SDP_Solver>(parameters, verbosity, false,
                                          *p.block_info, *p.grid, N);
    }
  }

  Solver::Solver(const PMP_Spec &spec, const Solver_Options &o)
  {
    std::lock_guard<std::mutex> lock(solve_mutex);
    require_single_rank();
    set_precision(o.precision);
    const Verbosity verbosity = to_verbosity(o.verbosity);
    impl = std::make_unique<Impl>(verbosity);
    impl->problem = build_pmp_problem(spec, verbosity, impl->timers);
    impl->solver = make_sdp_solver(impl->problem, o, verbosity);
    has_normalization_ = impl->problem.sdp->normalization.has_value();
  }

  Solver::Solver(const LMI_Spec &spec, const Solver_Options &o)
  {
    std::lock_guard<std::mutex> lock(solve_mutex);
    require_single_rank();
    set_precision(o.precision);
    const Verbosity verbosity = to_verbosity(o.verbosity);
    impl = std::make_unique<Impl>(verbosity);
    impl->problem = build_lmi_problem(spec, verbosity);
    impl->solver = make_sdp_solver(impl->problem, o, verbosity);
    has_normalization_ = false;
  }

  Solver::~Solver() = default;

  Solution_Data Solver::run(const Solver_Options &o)
  {
    std::lock_guard<std::mutex> lock(solve_mutex);
    set_precision(o.precision);
    Solver_Options options = o;
    if(!has_normalization_)
      options.want_z = false;
    Solution_Data out
      = run_existing(*impl->solver, *impl->problem.sdp, *impl->problem.block_info,
                     *impl->problem.grid, options, impl->timers);
    total_iterations_ += out.iterations;
    return out;
  }

  Solution_Data Solver::state(const Solver_Options &o) const
  {
    std::lock_guard<std::mutex> lock(solve_mutex);
    Solver_Options options = o;
    if(!has_normalization_)
      options.want_z = false;
    return collect(*impl->solver, *impl->problem.sdp, *impl->problem.block_info,
                   options, "", 0);
  }

  void Solver::set_y(const std::vector<std::string> &y)
  {
    std::lock_guard<std::mutex> lock(solve_mutex);
    const size_t N = impl->problem.sdp->dual_objective_b.Height();
    if(y.size() != N)
      throw std::invalid_argument("y must have length " + std::to_string(N));
    El::Matrix<El::BigFloat> y_local(N, 1);
    for(size_t i = 0; i < N; ++i)
      y_local(i, 0) = to_bigfloat(y[i]);
    for(auto &block : impl->solver->y.blocks)
      copy_matrix(y_local, block);
  }

  namespace
  {
    void set_block_diagonal(const Block_Info &block_info,
                            const std::vector<Matrix_Data> &blocks,
                            Block_Diagonal_Matrix &target, const char *name)
    {
      const size_t num_blocks = block_info.dimensions.size();
      if(blocks.size() != 2 * num_blocks)
        throw std::invalid_argument(std::string(name) + " needs 2 matrices per block");
      for(size_t i = 0; i < block_info.block_indices.size(); ++i)
        {
          const size_t j = block_info.block_indices.at(i);
          for(const size_t parity : {0, 1})
            {
              const auto &m = blocks.at(2 * j + parity);
              auto &dist = target.blocks.at(2 * i + parity);
              if(m.height != size_t(dist.Height())
                 || m.width != size_t(dist.Width()))
                throw std::invalid_argument(
                  std::string(name) + " block " + std::to_string(j) + " parity "
                  + std::to_string(parity) + " must be "
                  + std::to_string(dist.Height()) + "x"
                  + std::to_string(dist.Width()));
              El::Matrix<El::BigFloat> local(m.height, m.width);
              for(size_t r = 0; r < m.height; ++r)
                for(size_t c = 0; c < m.width; ++c)
                  local(r, c) = to_bigfloat(m.elements[r * m.width + c]);
              copy_matrix(local, dist);
            }
        }
    }
  }

  void Solver::set_X(const std::vector<Matrix_Data> &blocks)
  {
    std::lock_guard<std::mutex> lock(solve_mutex);
    set_block_diagonal(*impl->problem.block_info, blocks, impl->solver->X, "X");
  }

  void Solver::set_Y(const std::vector<Matrix_Data> &blocks)
  {
    std::lock_guard<std::mutex> lock(solve_mutex);
    set_block_diagonal(*impl->problem.block_info, blocks, impl->solver->Y, "Y");
  }

  void Solver::save_checkpoint(const std::string &directory) const
  {
    std::lock_guard<std::mutex> lock(solve_mutex);
    if(directory.empty())
      throw std::invalid_argument("checkpoint directory must not be empty");
    Solver_Options o = default_solver_options();
    o.checkpoint_in = directory;
    o.checkpoint_out = directory;
    const Solver_Parameters parameters = make_solver_parameters(o);
    impl->solver->save_checkpoint(directory, impl->verbosity,
                                  to_property_tree(parameters));
  }

  std::vector<size_t> Solver::dims() const
  {
    return impl->problem.block_info->dimensions;
  }
  std::vector<size_t> Solver::num_points() const
  {
    return impl->problem.block_info->num_points;
  }
  size_t Solver::num_variables() const
  {
    return impl->problem.sdp->dual_objective_b.Height();
  }
}
