# manylinux_2_28 image with SDPB's dependencies installed under /opt/deps.
# Used by cibuildwheel (see [tool.cibuildwheel] in pyproject.toml) to build
# self-contained Linux wheels.  Rebuild only when a dependency changes.
#
# Only what the Python extension needs is here.  SDPB is configured with
# `--libs-only` (scripts/cibw_before_all.sh), so MPSolve, libxml2, libarchive
# and most compiled Boost libraries, which only SDPB's command-line tools use,
# are absent.
#
# One build stage per dependency, all installing into the same prefix
# (/opt/deps: MPICH's compiler wrappers, pkg-config and CMake files bake it in).
# A stage copies in the stages it depends on, so BuildKit builds independent
# stages in parallel and, with the registry cache used by
# .github/workflows/deps-image.yml, rebuilds only the stages downstream of a
# change:
#
#     gmp -> mpfr -> flint
#     mpich, openblas, mpfr -> elemental
#     boost
#
# The final stage merges the leaf stages; files shared between them (GMP and
# MPFR) are identical copies.

# ---------------------------------------------------------------- base
FROM quay.io/pypa/manylinux_2_28_x86_64 AS base

ARG JOBS=4
# An ENV, unlike an ARG, is inherited by the stages built FROM this one.
ENV JOBS=$JOBS
ENV DEPS=/opt/deps
ENV PATH=$DEPS/bin:$PATH
ENV LD_LIBRARY_PATH=$DEPS/lib:$DEPS/lib64

# Distribution packages (AlmaLinux 8 + EPEL).  GMP, MPFR and Boost from the
# distribution are too old for SDPB and are built from source below.  The
# image's own cmake is 4.x, which refuses Elemental's old
# cmake_minimum_required, so the distribution's 3.26 is used.  METIS is a
# dependency of Elemental; gfortran builds OpenBLAS's LAPACK.  OpenBLAS is
# deliberately not taken from the distribution: see below.
RUN dnf -y install epel-release && dnf config-manager --set-enabled powertools \
    && dnf -y install metis-devel cmake xz bzip2 gcc-toolset-14-gcc-gfortran \
    && dnf clean all
ENV CMAKE=/usr/bin/cmake

WORKDIR /usr/local/src

# ---------------------------------------------------------------- gmp
FROM base AS gmp
# GMP 6.3 (SDPB needs >= 6.2.1, with C++ bindings)
RUN curl --retry 5 -fsSL https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz | tar xJ \
    && cd gmp-6.3.0 && ./configure --prefix=$DEPS --enable-cxx --enable-fat >/dev/null \
    && make -j$JOBS >/dev/null && make install >/dev/null && cd .. && rm -rf gmp-6.3.0

# ---------------------------------------------------------------- mpfr
FROM gmp AS mpfr
# MPFR 4.2
RUN curl --retry 5 -fsSL https://ftp.gnu.org/gnu/mpfr/mpfr-4.2.1.tar.xz | tar xJ \
    && cd mpfr-4.2.1 && ./configure --prefix=$DEPS --with-gmp=$DEPS >/dev/null \
    && make -j$JOBS >/dev/null && make install >/dev/null && cd .. && rm -rf mpfr-4.2.1

# ---------------------------------------------------------------- mpich
FROM base AS mpich
# MPICH (single library, works as a singleton without mpiexec; bundled into wheels).
# The nemesis channel is built with the `none` network module instead of the
# default `tcp`: with tcp, MPI_Init opens a TCP listener on 0.0.0.0 even for a
# single process, and MPICH's connection state machine (socksm.c) *asserts*
# when the first packet on an incoming connection is not an MPICH handshake,
# so any port scan or stray connection aborted a running solve (v0.2.1 and
# earlier).  The package runs on one rank and never talks to another node;
# shared-memory communication between ranks on one host (mpirun -n 2 in the
# tests) does not use the network module.  Only MPI_Open_port/MPI_Comm_spawn
# need it, and SDPB uses neither.
# The sed works around a bug in MPICH 4.2.3's `none` module: it leaves the
# process's business card empty, the PMI `put` then carries an empty value, and
# hydra dies with "unable to parse PMI command" for any mpirun -n>1.  A dummy
# entry keeps multi-rank launches working, so that they reach this package's
# own "supports a single MPI rank" error.
# --disable-romio: nothing here uses MPI-IO (no MPI_File_* reference in Elemental or SDPB).
RUN curl --retry 5 -fsSL https://www.mpich.org/static/downloads/4.2.3/mpich-4.2.3.tar.gz | tar xz \
    && cd mpich-4.2.3 \
    && NONE=src/mpid/ch3/channels/nemesis/netmod/none/none.c \
    && sed -i '0,/^    return MPI_SUCCESS;$/s//    MPL_str_add_string_arg(bc_val_p, val_max_sz_p, "netmod", "none");\n    return MPI_SUCCESS;/' $NONE \
    && grep -A3 '^static int nm_init' $NONE | grep -q MPL_str_add_string_arg \
    && ./configure --prefix=$DEPS --disable-fortran --with-device=ch3:nemesis:none \
       --enable-shared --disable-static --disable-romio >/dev/null \
    && make -j$JOBS >/dev/null && make install >/dev/null && cd .. && rm -rf mpich-4.2.3

# ---------------------------------------------------------------- boost
FROM base AS boost
# Boost 1.86: all headers, and the three compiled libraries the extension uses
# (Solver_Parameters, checkpoints, stack traces in error messages).
RUN curl --retry 5 -fsSL https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.bz2 | tar xj \
    && cd boost_1_86_0 \
    && ./bootstrap.sh --prefix=$DEPS \
       --with-libraries=program_options,serialization,stacktrace >/dev/null \
    && ./b2 -j$JOBS link=shared variant=release install >/dev/null && cd .. && rm -rf boost_1_86_0

# ---------------------------------------------------------------- flint
FROM mpfr AS flint
# FLINT 3.1.  Its configure defaults to --enable-arch=native, i.e. -march=native,
# which bakes the build host's ISA (AVX2 on GitHub runners) into libflint with
# no run-time dispatch; the wheel then dies with SIGILL on CPUs without AVX.
# --disable-arch keeps the compiler's baseline x86-64 code generation (AVX2 and
# AVX-512 are already off by default).  GMP (--enable-fat) and OpenBLAS pick
# their kernels at run time, so they are safe to tune.
RUN curl --retry 5 -fsSL https://github.com/flintlib/flint/releases/download/v3.1.3/flint-3.1.3.tar.gz | tar xz \
    && cd flint-3.1.3 \
    && ./configure --prefix=$DEPS --with-gmp=$DEPS --with-mpfr=$DEPS --disable-static \
       --disable-arch --disable-avx2 --disable-avx512 >/dev/null \
    && make -j$JOBS >/dev/null && make install >/dev/null && cd .. && rm -rf flint-3.1.3

# ---------------------------------------------------------------- openblas
FROM base AS openblas
# OpenBLAS with run-time kernel selection (DYNAMIC_ARCH) restricted to targets
# without 3DNow! code.  The distribution's OpenBLAS also dispatches at run time,
# but its list includes the OPTERON targets, whose GEMM copy kernels start with
# the 3DNow! instruction `femms`; OpenBLAS picks them for any AMD family-15/17
# CPU, and QEMU's default CPU model is exactly that yet has no 3DNow!, so the
# v0.2.1 wheel died with SIGILL there.  With the OPTERON targets left out of
# DYNAMIC_LIST such CPUs get the NEHALEM kernels (verified on a qemu64 guest;
# the always-built plain-SSE3 PRESCOTT kernels are the last resort).
# NUM_THREADS bounds the thread pool; NO_AFFINITY keeps OpenBLAS from pinning
# threads in a library that runs inside Python.
RUN curl --retry 5 -fsSL https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.34/OpenBLAS-0.3.34.tar.gz | tar xz \
    && cd OpenBLAS-0.3.34 \
    && make -j$JOBS DYNAMIC_ARCH=1 DYNAMIC_LIST="NEHALEM SANDYBRIDGE HASWELL SKYLAKEX ZEN" \
       NO_AFFINITY=1 USE_OPENMP=0 NUM_THREADS=64 >/dev/null \
    && make PREFIX=$DEPS DYNAMIC_ARCH=1 DYNAMIC_LIST="NEHALEM SANDYBRIDGE HASWELL SKYLAKEX ZEN" \
       NO_AFFINITY=1 USE_OPENMP=0 NUM_THREADS=64 install >/dev/null \
    && cd .. && rm -rf OpenBLAS-0.3.34

# ---------------------------------------------------------------- elemental
FROM mpfr AS elemental
COPY --from=mpich $DEPS $DEPS
COPY --from=openblas $DEPS $DEPS
# Elemental (bootstrap-collaboration fork), linked against the OpenBLAS above.
# Pinned: a moving branch would make the image unreproducible.
ARG ELEMENTAL_COMMIT=4e03e7a66c0ba08d04edbcfdd951ed72b37655f2
RUN mkdir elemental && cd elemental && git init -q \
    && git fetch -q --depth=1 https://gitlab.com/bootstrapcollaboration/elemental.git $ELEMENTAL_COMMIT \
    && git checkout -q FETCH_HEAD \
    && mkdir build && cd build \
    && CC=mpicc CXX=mpicxx $CMAKE .. -DCMAKE_INSTALL_PREFIX=$DEPS -DCMAKE_BUILD_TYPE=Release \
       -DMATH_LIBS="-L$DEPS/lib -lopenblas" \
       -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DGMP_INCLUDES=$DEPS/include -DGMP_LIBRARIES=$DEPS/lib/libgmp.so \
       -DMPFR_INCLUDES=$DEPS/include -DMPFR_LIBRARIES=$DEPS/lib/libmpfr.so \
       -DGMPXX_INCLUDES=$DEPS/include -DGMPXX_LIBRARIES=$DEPS/lib/libgmpxx.so >/dev/null \
    && make -j$JOBS >/dev/null && make install >/dev/null && cd ../.. && rm -rf elemental

# ---------------------------------------------------------------- final image
FROM base
COPY --from=elemental $DEPS $DEPS
COPY --from=flint $DEPS $DEPS
COPY --from=boost $DEPS $DEPS

# RapidJSON: the distribution's 1.1.0 (2016) does not compile with GCC 14,
# so take the header-only library from upstream master. There is no release
# after 1.1.0, so pin a commit: a moving branch would make the image
# unreproducible.
ARG RAPIDJSON_COMMIT=24b5e7a8b27f42fa16b96fc70aade9106cf7102f
RUN mkdir rapidjson && cd rapidjson && git init -q \
    && git fetch -q --depth=1 https://github.com/Tencent/rapidjson.git $RAPIDJSON_COMMIT \
    && git checkout -q FETCH_HEAD \
    && cp -r include/rapidjson $DEPS/include/rapidjson && cd .. && rm -rf rapidjson

WORKDIR /
