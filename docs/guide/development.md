# Development install

Use this when you want fixes to the library to reach your own project as soon
as they are committed, or when you work on sdpb-python itself. The release
wheels are frozen binaries; the alternative is an *editable* install of a
clone, where Python imports the package straight from the clone's `src/`
directory.

## Prerequisites

An editable install builds from source, so SDPB's C++ dependencies must be on
the machine: MPI, Boost, GMP, MPFR, FLINT, libarchive, libxml2, RapidJSON, a
CBLAS, plus the Elemental fork and MPSolve (by default under `~/install`).
{doc}`../building` is the tested recipe for Ubuntu 24.04. Without the
dependencies, use the Docker variant at the end of this page.

## Installing into your project's environment

1. Clone the library once, with its SDPB submodule:

   ```
   git clone --recurse-submodules git@github.com:YPWang0818/sdpb-python.git ~/dev/sdpb-python
   ```

2. Activate the environment your project uses and make sure the build tools
   are in it:

   ```
   source /path/to/your-project/.venv/bin/activate
   pip install setuptools wheel Cython mpmath
   ```

3. Build SDPB and install the package editable into that environment:

   ```
   cd ~/dev/sdpb-python
   scripts/build_sdpb.sh
   ```

   The script configures SDPB's waf build with `-fPIC`, builds it, and ends
   with `pip install --no-build-isolation -e .`, which lands in whichever
   environment is active.

4. Check from your project that the import resolves into the clone:

   ```
   python -c "import sdpb_python; print(sdpb_python.__file__)"
   # ~/dev/sdpb-python/src/sdpb_python/__init__.py
   ```

Several projects can share one clone: repeat steps 2 and 3 in each
environment (step 3 reduces to the one-line `pip install` once SDPB is built).

## Picking up fixes

```
cd ~/dev/sdpb-python
scripts/rebuild.sh --pull
```

`rebuild.sh` pulls (with `--pull`), then rebuilds only what the pull touched:

| Changed | Action taken | Time |
|---|---|---|
| Python files in `src/sdpb_python/` | nothing; the next interpreter start sees them | 0 |
| `src/sdpb_python/cpp/`, `_sdpb.pyx`, `sdpb_wrapper.pxd`, `setup.py`, `pyproject.toml` | rebuild the extension, reinstall editable | 2 to 3 min |
| the `c-src/sdpb` submodule commit, or uncommitted edits to its sources | rebuild SDPB's libraries, then the extension | about 8 min |

`--force` rebuilds everything; `DEPS_PREFIX` and `JOBS` are honoured as in
`build_sdpb.sh`. Run it from the environment the package is installed into,
because it uses that environment's `pip`.

Two reminders:

- Python only loads a module once per interpreter. Restart the kernel in a
  notebook, or the process in a long-running job, after pulling.
- If the release wheel was installed in the same environment earlier, the
  editable install replaces it; `pip show sdpb-python` tells you which is
  active ("Editable project location" appears for the clone).

## Working on the library

Edit, then run the tests:

```
scripts/rebuild.sh          # after C++/Cython edits; a no-op for Python edits
pytest                      # about a minute; pytest --run-slow for the long datasets
```

For changes to SDPB itself, commit them in `c-src/sdpb` on the `python-api`
branch; `rebuild.sh` notices the new commit (or uncommitted edits) and
rebuilds the libraries. The fork's own suite is
`c-src/sdpb/test/run_all_tests.sh`.

## Without installing the dependencies: Docker

The image used to build the release wheels is public and contains every
dependency under `/opt/deps`:

```
docker run -it --rm -v "$PWD":/project -w /project \
    ghcr.io/ypwang0818/sdpb-python-deps:manylinux_2_28 bash
# inside the container
bash scripts/cibw_before_all.sh /project                    # builds SDPB against /opt/deps
/opt/python/cp312-cp312/bin/python -m venv .venv-docker && source .venv-docker/bin/activate
pip install setuptools wheel Cython mpmath pytest
CC=mpicxx CXX=mpicxx pip install --no-build-isolation -e .
pytest
```

Edits made on the host are visible inside the container immediately; the same
rebuild rules apply. This environment is only usable from inside the
container, so it suits working on the library rather than on a project that
depends on it.
