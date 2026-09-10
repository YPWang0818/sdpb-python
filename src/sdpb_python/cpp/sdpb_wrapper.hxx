// Thin C++ API over SDPB, kept free of Elemental/Boost types so that the
// Cython layer only has to know about std::string / std::vector / int.
#pragma once

#include <string>
#include <vector>

namespace sdpb_python
{
  // SDPB git version string baked in at compile time.
  std::string version();

  // Initialise MPI / Elemental (idempotent).  Called automatically by run().
  void initialize();
  // Finalise MPI / Elemental.  Safe to call more than once.
  void finalize();

  int mpi_rank();
  int mpi_size();

  // Run the SDPB solver exactly as the `sdpb` executable would, given its
  // command-line arguments (without argv[0]).  Results are written to the
  // directory passed via --outDir; the Python layer parses `out.txt`.
  // Throws std::runtime_error on failure.
  void run(const std::vector<std::string> &args);
}
