"""The search kernel has two execution paths -- interpreted, and numba
-JIT-compiled -- built from a single source (:mod:`pydigest2._search`).
These tests hold them to producing *identical* results, so a change that
only happens to work under one of them can't slip through.
"""
import math
from pathlib import Path

import pytest

from pyroorb.digest2 import ranging
from pyroorb.digest2.constants import EPART, HPART, IPART, QPART
from pyroorb.digest2.engine import Digest2Engine
from pyroorb.digest2.observations import Observation
from pyroorb.digest2.ranging import score_tracklet

DATA = Path(__import__("pyroorb.digest2", fromlist=["_"]).__file__).parent / "data"

# A real two-observation NEOCP tracklet. Deliberately short: the pure-Python
# kernel is ~30x slower than the compiled one, so this keeps the comparison
# to a couple of seconds while still exercising the whole search.
_OBS = [
    Observation(mjd=58693.45388, ra=0.46637088386644815, dec=0.21503571456036694,
                vmag=19.68, obscode="I41"),
    Observation(mjd=58693.47553, ra=0.4665403262479959, dec=0.21520152083930638,
                vmag=19.57, obscode="I41"),
]


@pytest.fixture(scope="module")
def engine():
    return Digest2Engine.load(model_path=str(DATA / "digest2.model.csv"),
                              obscodes_path=str(DATA / "digest2.obscodes"))


def _score(engine):
    return score_tracklet(
        _OBS, vmag=19.625, is_ades=False, class_indices=list(range(15)),
        model=engine.model, site_table=engine.site_table,
        default_obserr_rad=math.radians(1.0 / 3600.0), no_threshold=False,
        repeatable=True,
    )


def test_backend_is_reported():
    assert ranging.KERNEL_BACKEND in ("numba", "python")


@pytest.mark.skipif(ranging.KERNEL_BACKEND != "numba",
                    reason="numba not installed; only one kernel path exists here")
def test_compiled_and_interpreted_kernels_agree_exactly(engine, monkeypatch):
    """The JIT and the interpreter must produce bit-identical scores.

    Both run the same function out of pydigest2._search; the only
    difference is compilation and whether the scratch buffers are NumPy
    arrays or plain lists. Any divergence here would mean the compiled
    path is not doing what the readable source says.
    """
    compiled = _score(engine)

    # Force the interpreted path by swapping in exactly what the
    # no-numba import branch of pydigest2.ranging would have set up.
    monkeypatch.setattr(ranging, "_search_all", ranging._search_all_py)
    monkeypatch.setattr(ranging, "_NUMBA", False)
    monkeypatch.setattr(ranging, "_QPART", QPART)
    monkeypatch.setattr(ranging, "_EPART", EPART)
    monkeypatch.setattr(ranging, "_IPART", IPART)
    monkeypatch.setattr(ranging, "_HPART", HPART)
    interpreted = _score(engine)

    assert interpreted.rms == compiled.rms
    for c in range(15):
        assert interpreted.raw_scores[c] == compiled.raw_scores[c], f"raw class {c}"
        assert interpreted.noid_scores[c] == compiled.noid_scores[c], f"noid class {c}"


def test_scores_are_deterministic_in_repeatable_mode(engine):
    first = _score(engine)
    second = _score(engine)
    assert first.raw_scores == second.raw_scores
    assert first.noid_scores == second.noid_scores


def _score_subset(engine, classes):
    return score_tracklet(
        _OBS, vmag=19.625, is_ades=False, class_indices=classes,
        model=engine.model, site_table=engine.site_table,
        default_obserr_rad=math.radians(1.0 / 3600.0), no_threshold=False,
        repeatable=True,
    )


def test_restricting_classes_changes_the_search_not_just_the_output(engine):
    """Restricting the class set genuinely changes the scores, and that is
    correct behaviour, not a bug.

    ``tagAngle`` reports "found a new bin" only with respect to the
    *configured* classes, and that boolean is what drives the angle
    search's recursion (and therefore how many draws it takes from the
    LCG). So a narrower class set explores a different set of orbits.
    The reference implementation behaves the same way -- it is why
    OPERATION.md notes the program "runs considerably faster" when
    classes are listed -- and this port is validated against the C
    engine in *both* configurations (see
    tests/test_integration.py::test_class_restricted_scoring_matches_c_reference).

    Pinning it here means a future "optimization" that decoupled the
    filter from the search -- which would look harmless, and would even
    make the numbers prettier -- gets caught.
    """
    full = _score_subset(engine, list(range(15)))
    subset = _score_subset(engine, [1, 7])

    assert subset.raw_scores[1] != full.raw_scores[1]
    # ... but the same restricted request is itself reproducible
    assert _score_subset(engine, [1, 7]).raw_scores == subset.raw_scores
    # and both remain in the same ballpark: this is search noise on a
    # 2-observation arc, not a different algorithm
    assert subset.raw_scores[1] == pytest.approx(full.raw_scores[1], abs=5.0)
