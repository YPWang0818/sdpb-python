"""Features chosen when the package is built.

``PROFILING``: the per-solve profiling options (``SolverOptions.profile_path``,
``profile_detail``, ``profile_autosave``, ``profile_meta`` and
``Solution.profile_path``). Release wheels leave them out. Preview builds of the
``profiling`` branch, and development checkouts that want them, include them by
adding an empty module ``_profiling_enabled.py`` next to this file
(``touch src/sdpb_python/_profiling_enabled.py``; it is ignored by git).
"""

from __future__ import annotations

try:
    from . import _profiling_enabled  # noqa: F401
except ImportError:
    PROFILING = False
else:
    PROFILING = True
