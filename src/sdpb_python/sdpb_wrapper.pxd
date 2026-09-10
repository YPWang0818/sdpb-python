# Cython declarations for the C++ shim in cpp/sdpb_wrapper.hxx.
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "sdpb_wrapper.hxx" namespace "sdpb_python":
    string version() except +
    void initialize() except +
    void finalize() except +
    int mpi_rank() except +
    int mpi_size() except +
    void run(const vector[string]& args) except +
