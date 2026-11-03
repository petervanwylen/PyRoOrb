"""The fixture JSONs on disk must carry the flags `build_fixtures.OBJECTS` declares for them.

`build_fixtures.py` is the DECLARATION and `fixtures/*.json` is the ARTEFACT, and everything that
reads a flag at test time reads the artefact: `test_regression._check` takes `benchmark`, `quality`
and `force` off the fixture, never off the recipe. So the two can disagree silently, and did --
`quality: true` was declared for 2008_NA in 83e4e13 and lost four commits later when the fixture was
rebuilt with fresh astrometry, because the rebuild wrote a hand-picked subset of the declared keys.
The result was the one object with a fit-quality guard running speed-only for two days while
build_fixtures still asserted it was guarded.

This is the guard for that whole class: offline, no fitting, just the two files compared.
"""
import json
import os

import pytest

import build_fixtures as bf
from _fixtures import FIX


def _declared():
    """{label: opts} from build_fixtures.OBJECTS -- the declaration side."""
    return {e[0]: (e[2] if len(e) > 2 else {}) for e in bf.OBJECTS}


def _fixture_path(label):
    return os.path.join(FIX, label + ".json")


@pytest.mark.parametrize("label", sorted(_declared()))
def test_fixture_carries_its_declared_flags(label):
    path = _fixture_path(label)
    if not os.path.exists(path):
        pytest.skip(f"{label} declared but not built")
    fx = json.load(open(path, encoding="utf-8"))
    for key, want in _declared()[label].items():
        assert fx.get(key) == want, (
            f"{label}: build_fixtures declares {key}={want!r} but the fixture on disk has "
            f"{fx.get(key)!r}. A rebuild dropped it -- see build_fixtures' payload construction.")


# Fixtures that legitimately have no build_fixtures entry. RANGEAMBIG_3NIGHT_MBA is hand-made for
# test_constraint's range-ambiguity case -- its three "designations" are raw numeric tracklet ids,
# not something fetch_observations can look up -- so it is re-baselineable (update_goldens globs
# the directory) but not re-buildable. Recorded rather than silently tolerated: anything ADDED to
# this set is a fixture nobody can rebuild from source astrometry, which should be a deliberate
# choice and not an oversight.
# PAIR_2012GC2 is 2012 GC2 = CERV9W2 exactly as a web user fitted it (2026-09-11): the 16 NEOCP
# observations still labelled CERV9W2. The MPC has since republished that tracklet under 2012 GC2,
# so a live fetch now dedups CERV9W2 away and returns a single-designation fit -- the case is
# reconstructable only by relabelling, never rebuildable by build_fixtures.
UNDECLARED_OK = {"RANGEAMBIG_3NIGHT_MBA", "PAIR_2012GC2"}


def test_every_fixture_is_declared():
    """The reverse direction: a fixture nobody declares cannot be REBUILT (build_fixtures is where
    its desigs and flags live), even though update_goldens can still re-baseline it."""
    declared = set(_declared()) | UNDECLARED_OK
    on_disk = {os.path.splitext(f)[0] for f in os.listdir(FIX) if f.endswith(".json")}
    assert not (on_disk - declared), \
        f"fixtures with no build_fixtures entry: {sorted(on_disk - declared)}"
