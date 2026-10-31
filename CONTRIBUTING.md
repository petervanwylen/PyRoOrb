# Contributing to PyRoOrb

Thank you for your interest.

**Issues are welcome**: bug reports, fits that look wrong, questions, ideas. A good report
names the object (designation, or the astrometry itself), what you ran, what you expected and
what you got; for a fit, the RMS and the observation counts from the summary help most.

**Pull requests are by invitation.** Please open an issue first and describe the change; if it
fits the direction of the project you will be invited to send it. This keeps unsolicited
work from going unmerged.

If you do send code:

- `pytest -m "not slow"` and `pytest -n 4` pass (the default run is offline);
  `tests/test_regression.py` must not move a fit;
- a new behaviour comes with a test, and every object you fix is added to the regression
  fixtures (`tests/build_fixtures.py`);
- comments state the rule and its measured reason (`tests/test_public_code_hygiene.py`);
- anything read or ported from another code is recorded in `PROVENANCE.md`; PyRoOrb ports
  find_orb's logic, never its source lines.

The engine's rules are in [`AGENTS.md`](AGENTS.md).

By contributing you agree that your work is released under the project's licence (GPL-3.0).
