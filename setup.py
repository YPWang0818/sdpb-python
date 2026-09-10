"""Build script for the sdpb-python Cython extension.

The extension links against static libraries produced by SDPB's own waf build
(``c-src/sdpb/build``).  Build SDPB first, e.g. with ``scripts/build_sdpb.sh``,
then ``pip install -e .``.

Environment variables:
    SDPB_BUILD_DIR   Path to the waf build directory (default: c-src/sdpb/build).
    SDPB_SRC_DIR     Path to the SDPB source tree (default: c-src/sdpb).
    CXX / CC         Compiler; defaults to ``mpicxx`` because SDPB requires MPI.
"""

import os
import re
import subprocess
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent
SDPB_SRC = Path(os.environ.get("SDPB_SRC_DIR", ROOT / "c-src" / "sdpb")).resolve()
SDPB_BUILD = Path(os.environ.get("SDPB_BUILD_DIR", SDPB_SRC / "build")).resolve()
PKG = ROOT / "src" / "sdpb_python"

# SDPB is compiled with mpicxx; the extension must be too.
os.environ.setdefault("CXX", "mpicxx")
os.environ.setdefault("CC", "mpicxx")


def sdpb_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=SDPB_SRC, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def waf_cache() -> dict:
    """Read the flags waf discovered at configure time so we link the same way."""
    cache = SDPB_BUILD / "c4che" / "_cache.py"
    if not cache.exists():
        return {}
    env: dict = {}
    exec(cache.read_text(), env)  # waf writes plain python assignments
    return {k: v for k, v in env.items() if not k.startswith("__")}


def collect(env: dict, prefix: str) -> list:
    out: list = []
    for key, value in env.items():
        if key.startswith(prefix + "_") and isinstance(value, list):
            for item in value:
                if item not in out:
                    out.append(item)
    return out


env = waf_cache()

include_dirs = [
    str(PKG / "cpp"),
    str(SDPB_SRC / "src"),
    str(SDPB_SRC / "external"),
    *collect(env, "INCLUDES"),
]
library_dirs = [str(SDPB_BUILD), *collect(env, "LIBPATH"), *collect(env, "STLIBPATH")]

# Static libs built by waf (order matters for static linking) plus external libs.
sdpb_static_libs = ["sdp_solve", "pmp2sdp_lib", "sdpb_util"]
libraries = sdpb_static_libs + collect(env, "LIB") + collect(env, "STLIB")
if not env:
    # Reasonable defaults when the waf cache is unavailable.
    libraries += ["El", "boost_filesystem", "boost_system", "boost_program_options",
                  "boost_date_time", "boost_serialization", "boost_iostreams",
                  "gmpxx", "gmp", "mpfr", "flint", "archive", "xml2", "mps"]

# SDPB's ``sdpb`` executable sources (minus main.cxx) are compiled straight into the
# extension, since waf only builds them into the binary, not into a library.
sdpb_program_sources = [
    str(SDPB_SRC / "src" / "sdpb" / f)
    for f in ("solve.cxx", "write_timing.cxx", "SDPB_Parameters.cxx", "save_solution.cxx")
]

extensions = [
    Extension(
        "sdpb_python._sdpb",
        sources=[
            str(PKG / "_sdpb.pyx"),
            str(PKG / "cpp" / "sdpb_wrapper.cxx"),
            *sdpb_program_sources,
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-Wall"],
        define_macros=[
            ("OMPI_SKIP_MPICXX", None),
            ("SDPB_VERSION_STRING", f'"{sdpb_version()}"'),
        ],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={"embedsignature": True},
    ),
)
