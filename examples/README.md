# Examples

- `tour.py`: an end-to-end tour of the API against an installed package; every
  step asserts its result and the script ends with `ALL CHECKS PASSED`. Useful
  as a smoke test of a fresh installation:

  ```
  python -m venv .venv && source .venv/bin/activate
  pip install sdpb-python --find-links https://github.com/YPWang0818/sdpb-python/releases/expanded_assets/v0.2.1
  python examples/tour.py
  ```

  It runs in under a minute and needs no data files.
