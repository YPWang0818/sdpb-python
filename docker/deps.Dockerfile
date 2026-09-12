# manylinux_2_28 image with SDPB's dependencies installed under /opt/deps.
# Used by cibuildwheel (see [tool.cibuildwheel] in pyproject.toml) to build
# self-contained Linux wheels.  Rebuild only when a dependency changes.
FROM quay.io/pypa/manylinux_2_28_x86_64

ARG JOBS=4
ENV DEPS=/opt/deps
ENV PATH=$DEPS/bin:$PATH
ENV LD_LIBRARY_PATH=$DEPS/lib:$DEPS/lib64

# Distribution packages that are recent enough (AlmaLinux 8 + EPEL).
# GMP, MPFR and Boost from the distribution are too old for SDPB and are
# built from source below.  The image's own cmake is 4.x, which refuses
# Elemental's old cmake_minimum_required, so the distribution's 3.26 is used.
# OpenBLAS is deliberately not taken from the distribution: see below.
RUN dnf -y install epel-release && dnf config-manager --set-enabled powertools \
    && dnf -y install libarchive-devel libxml2-devel metis-devel \
       rapidjson-devel cmake autoconf automake libtool bison flex xz bzip2 \
       gcc-toolset-14-gcc-gfortran \
    && dnf clean all
ENV CMAKE=/usr/bin/cmake

WORKDIR /usr/local/src

# GMP 6.3 (SDPB needs >= 6.2.1, with C++ bindings)
RUN curl --retry 5 -fsSL https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz | tar xJ \
    && cd gmp-6.3.0 && ./configure --prefix=$DEPS --enable-cxx --enable-fat >/dev/null \
    && make -j$JOBS >/dev/null && make install >/dev/null && cd .. && rm -rf gmp-6.3.0

# MPFR 4.2
RUN curl --retry 5 -fsSL https://ftp.gnu.org/gnu/mpfr/mpfr-4.2.1.tar.xz | tar xJ \
    && cd mpfr-4.2.1 && ./configure --prefix=$DEPS --with-gmp=$DEPS >/dev/null \
    && make -j$JOBS >/dev/null && make install >/dev/null && cd .. && rm -rf mpfr-4.2.1

# MPICH (single library, works as a singleton without mpiexec; bundled into wheels)
RUN curl --retry 5 -fsSL https://www.mpich.org/static/downloads/4.2.3/mpich-4.2.3.tar.gz | tar xz \
    && cd mpich-4.2.3 \
    && ./configure --prefix=$DEPS --disable-fortran --with-device=ch3:nemesis \
       --enable-shared --disable-static >/dev/null \
    && make -j$JOBS >/dev/null && make install >/dev/null && cd .. && rm -rf mpich-4.2.3

# Boost 1.86 (the libraries SDPB links)
RUN curl --retry 5 -fsSL https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.bz2 | tar xj \
    && cd boost_1_86_0 \
    && ./bootstrap.sh --prefix=$DEPS \
       --with-libraries=date_time,filesystem,program_options,iostreams,serialization,system,stacktrace,process >/dev/null \
    && ./b2 -j$JOBS link=shared variant=release install >/dev/null && cd .. && rm -rf boost_1_86_0

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

# Elemental (bootstrap-collaboration fork), linked against the OpenBLAS above
RUN git clone --depth=1 https://gitlab.com/bootstrapcollaboration/elemental.git \
    && mkdir elemental/build && cd elemental/build \
    && CC=mpicc CXX=mpicxx $CMAKE .. -DCMAKE_INSTALL_PREFIX=$DEPS -DCMAKE_BUILD_TYPE=Release \
       -DMATH_LIBS="-L$DEPS/lib -lopenblas" \
       -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DGMP_INCLUDES=$DEPS/include -DGMP_LIBRARIES=$DEPS/lib/libgmp.so \
       -DMPFR_INCLUDES=$DEPS/include -DMPFR_LIBRARIES=$DEPS/lib/libmpfr.so \
       -DGMPXX_INCLUDES=$DEPS/include -DGMPXX_LIBRARIES=$DEPS/lib/libgmpxx.so >/dev/null \
    && make -j$JOBS >/dev/null && make install >/dev/null && cd ../.. && rm -rf elemental

# MPSolve: required by SDPB's waf configure (used only by the spectrum tool,
# which the Python extension does not include or link).  Its headers define
# `false`/`true` as enum members, so it must not be compiled as C23.
RUN git clone --depth=1 https://github.com/robol/MPSolve.git \
    && cd MPSolve && ./autogen.sh >/dev/null \
    && CC=mpicc CXX=mpicxx CFLAGS="-std=gnu11 -O2" CPPFLAGS="-I$DEPS/include" LDFLAGS="-L$DEPS/lib" \
       ./configure --prefix=$DEPS --disable-dependency-tracking \
       --disable-examples --disable-ui --disable-graphical-debugger --disable-documentation >/dev/null \
    && make -j$JOBS >/dev/null && make install >/dev/null && cd .. && rm -rf MPSolve

# RapidJSON: the distribution's 1.1.0 (2016) does not compile with GCC 14,
# so take the header-only library from upstream master.
RUN git clone --depth=1 https://github.com/Tencent/rapidjson.git \
    && cp -r rapidjson/include/rapidjson $DEPS/include/rapidjson && rm -rf rapidjson

WORKDIR /
