"""The random-object endpoint behind the top bar's 🎲 button (engine_api.random_object).

The full catalogue is a download; these tests draw from a small tracked sample of it
(tests/mpcorb_sample/: up to 300 objects per family, every field the index holds), so
they run offline and never depend on the catalogue of the day.
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MVP = os.path.dirname(os.path.dirname(_HERE))

from pyroorb.web import engine_api  # noqa: E402
from pyroorb import mpcorb  # noqa: E402


from pathlib import Path  # noqa: E402

_SAMPLE = Path(_MVP) / "tests" / "mpcorb_sample" / "MPCORB_index.npz"


@pytest.fixture
def sample_catalogue(monkeypatch):
    """The random draw reads the tracked sample index, never a download."""
    monkeypatch.setattr(mpcorb, "MPCORB_NPZ", _SAMPLE)
    monkeypatch.setattr(mpcorb, "build_index", lambda verbose=True: _SAMPLE)


needs_catalogue = pytest.mark.usefixtures("sample_catalogue")


def test_every_advertised_family_has_a_filter():
    """The dropdown is built from random_families(); a family it offers with no filter
    behind it would 400 on click."""
    keys = [f["key"] for f in engine_api.RANDOM_FAMILIES]
    assert set(keys) == set(engine_api._RANDOM_FILTERS)
    assert len(keys) == len(set(keys))
    assert keys[0] == "any"                       # the default draw
    for f in engine_api.RANDOM_FAMILIES:
        assert f["label"] and f["hint"]


def test_random_families_payload():
    p = engine_api.random_families()
    assert p["ok"] and p["max_oppositions"] == engine_api.RANDOM_MAX_OPPOSITIONS
    assert p["families"] is engine_api.RANDOM_FAMILIES


def test_unknown_family_is_an_error_not_a_draw():
    p = engine_api.random_object("nonesuch")
    assert p["ok"] is False and "nonesuch" in p["error"]


@needs_catalogue
@pytest.mark.parametrize("family", [f["key"] for f in engine_api.RANDOM_FAMILIES])
def test_draw_respects_the_family_box_and_the_opposition_cap(family):
    p = engine_api.random_object(family)
    assert p["ok"], p.get("error")
    assert p["designation"] and p["family"] == family
    # The cap is the whole point of the button: no 30-opposition megafit on one click.
    assert 1 <= p["oppositions"] <= engine_api.RANDOM_MAX_OPPOSITIONS
    lo_a, hi_a = engine_api._RANDOM_FILTERS[family].get("a", (None, None))
    if lo_a is not None:
        assert p["a"] >= lo_a
    if hi_a is not None:
        assert p["a"] <= hi_a
    q = engine_api._RANDOM_FILTERS[family].get("q")
    if q:
        lo, hi = q
        peri = p["a"] * (1.0 - p["e"])
        assert (lo is None or peri >= lo) and (hi is None or peri <= hi)


@needs_catalogue
def test_draws_are_not_all_the_same_object():
    """Unseeded: clicking again gives you something else (the button is useless if not)."""
    got = {engine_api.random_object("any")["designation"] for _ in range(8)}
    assert len(got) > 1
