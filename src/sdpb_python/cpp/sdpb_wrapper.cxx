#include "sdpb_wrapper.hxx"

#include "sdpb/SDPB_Parameters.hxx"
#include "sdpb_util/Environment.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <El.hpp>

#include <chrono>
#include <limits>
#include <memory>
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
  namespace
  {
    // Elemental/MPI may only be initialised once per process, so keep a
    // single Environment alive for the lifetime of the module.
    std::unique_ptr<Environment> global_env;
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

  void run(const std::vector<std::string> &args)
  {
    initialize();
    const Environment &env = *global_env;

    // SDPB_Parameters parses argv with boost::program_options.
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

    Environment::set_precision(parameters.solver.precision);
    const auto start_time = std::chrono::high_resolution_clock::now();

    Block_Info block_info(env, parameters.sdp_path,
                          parameters.solver.checkpoint_in,
                          parameters.proc_granularity, parameters.verbosity);

    // Mirror sdpb/main.cxx: do a short timing run when running in parallel
    // without existing block timings or a checkpoint.
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
        solve(block_info, timing_parameters, env, start_time,
              block_timings_ms);
        if(block_timings_ms.Height() == 0 && block_timings_ms.Width() == 0)
          throw std::runtime_error(
            "block_timings vector is empty: timing run exited before "
            "completing two solver iterations.");

        write_block_timings(timing_parameters.solver.checkpoint_out,
                            block_info, block_timings_ms,
                            timing_parameters.verbosity);
        El::mpi::Barrier(El::mpi::COMM_WORLD);
        Block_Info new_info(env, parameters.sdp_path, block_timings_ms,
                            parameters.proc_granularity,
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
    solve(block_info, parameters, env, start_time, block_timings_ms);
  }
}
