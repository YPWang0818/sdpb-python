import subprocess

import sdpb_python


def test_version_string():
    assert sdpb_python.__version__


def test_public_names_exist():
    for name in sdpb_python.__all__:
        assert hasattr(sdpb_python, name), name


def test_cli_passthrough_is_gone():
    # Removed in 0.3.0 together with the libraries only it needed (libarchive).
    assert not hasattr(sdpb_python, "solve_dir")
    assert not hasattr(sdpb_python, "SDPBResult")


def test_extension_reports_version(sdpb_ext):
    assert isinstance(sdpb_ext.version(), str)
    assert sdpb_ext.mpi_size() >= 1


def test_extension_links_no_tool_only_libraries(sdpb_ext):
    """Libraries that only SDPB's command-line tools need must not come back.

    libarchive (sdp.zip) drags OpenSSL, zstd, lz4, bz2, acl and libxml2 into the
    wheels; MPSolve is GPL-3.  The extension contains no code that uses them.
    """
    out = subprocess.run(["ldd", sdpb_ext.__file__], capture_output=True, text=True).stdout
    assert "libEl" in out  # ldd worked
    for lib in ("libarchive", "libxml2", "libmps", "libcrypto", "libboost_iostreams",
                "libboost_filesystem", "libboost_date_time", "libboost_process"):
        assert lib not in out, lib
