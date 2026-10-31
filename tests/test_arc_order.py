"""Regression: the arc-length display must be the true chronological span, even when
the observation list is not in time order.

Every normal entry path (parse_obs / parse_ades_file / fetch_combined) sorts obs by
jd_utc, so `arc_length_days()` -- which returns ti[-1] - ti[0] -- gets a sorted list.
The one path that did NOT was the web UI "Add to current fit" paste: loader.add_to_fit
built `combined = list(base_obs) + added`, appending the pasted obs at the end without
re-sorting. When those added obs fell inside or before the existing arc, the last
element of the list was no longer the chronologically latest obs, so the header showed
a much-too-short (occasionally negative) arc while the opposition count -- which sorts
internally -- stayed correct.

Two fixes, both pinned here:
  1. loader.add_to_fit sorts the merged list by jd_utc (restores the invariant).
  2. FittedOrbit.arc_length_days sorts defensively (correct regardless of obs order).
"""
import pytest
import numpy as np
import pyroorb.fit_object as fo

# refits a fixture in several obs orders
pytestmark = pytest.mark.slow


def _bare_fit(jds, mask=None):
    """A FittedOrbit shell carrying only what arc_length_days/_incl_obs read, so the
    span logic can be exercised without running a real fit."""
    f = fo.FittedOrbit.__new__(fo.FittedOrbit)
    f._obs = [{"jd_utc": j} for j in jds]
    f._mask = np.ones(len(jds), dtype=bool) if mask is None else np.asarray(mask, bool)
    return f


def test_arc_length_true_span_when_obs_unordered():
    """ti[-1]-ti[0] on an UNORDERED list would understate the span; sorted() fixes it.
    Base arc spans 100 d; an added night lands in the middle, appended at the end."""
    base = [2459500.0 + i for i in range(0, 101, 10)]   # 2459500 .. 2459600, sorted
    added = [2459550.5]                                 # mid-arc, appended last
    f = _bare_fit(base + added)                         # NOT re-sorted, as the old bug left it
    assert f.arc_length_days() == 100.0, f.arc_length_days()


def test_arc_length_not_negative_with_precovery_appended():
    """Precovery obs (earlier than the whole base arc) appended last used to make the
    old ti[-1]-ti[0] go negative. The true span must be first-to-last chronological."""
    base = [2459500.0, 2459550.0, 2459600.0]
    added = [2459400.0, 2459410.0]                      # precovery, earlier than everything
    f = _bare_fit(base + added)
    assert f.arc_length_days() == 200.0, f.arc_length_days()   # 2459600 - 2459400


def test_arc_length_respects_inclusion_mask():
    """The span is over INCLUDED obs only, still order-independent."""
    jds = [2459600.0, 2459500.0, 2459555.0, 2459520.0]  # scrambled
    mask = [True, True, False, True]                    # drop the 2459555 obs
    f = _bare_fit(jds, mask)
    assert f.arc_length_days() == 100.0, f.arc_length_days()   # 2459600 - 2459500


def test_add_to_fit_sorts_merged_obs():
    """loader.add_to_fit must hand fit_obs_list a chronologically-sorted union, so the
    arc display is right no matter where the pasted obs fall. We stub the obs source and
    the fitter to capture the `combined` list without a real fetch/fit."""
    from pyroorb.web import loader; from pyroorb.web import engine_api

    base_obs = [
        {"jd_utc": 2459500.0, "ra_deg": 10.0, "dec_deg": 5.0, "obscode": "F51", "designation": "2023 QJ3"},
        {"jd_utc": 2459600.0, "ra_deg": 11.0, "dec_deg": 5.1, "obscode": "F51", "designation": "2023 QJ3"},
    ]
    # Pasted obs that fall BEFORE the base arc (precovery) -- the case that produced a
    # short/negative arc under the old append-without-sort. Parsing is stubbed (this test
    # pins the merge/sort, not the OBS80 reader, which has its own tests).
    added_obs = [
        {"jd_utc": 2459450.0, "ra_deg": 9.0, "dec_deg": 4.9, "obscode": "F51", "designation": "2023 QJ3"},
        {"jd_utc": 2459460.0, "ra_deg": 9.1, "dec_deg": 4.9, "obscode": "F51", "designation": "2023 QJ3"},
    ]
    captured = {}

    def fake_base_obs_for(session=None, fit_key=None):
        return list(base_obs), None

    def fake_parse(text, obscodes, **kw):
        # **kw, not a fixed signature: loader._parse has grown a keyword since (radar_out),
        # and a stub that can't absorb it raises TypeError inside add_to_fit -- which
        # turned a stale stub into a plausible "could not parse pasted obs" payload and
        # this test into a confusing "fit_obs_list was never called". Absorb whatever the
        # caller passes. (add_to_fit no longer misreports that: it catches only
        # obs_sources.ObsParseError, so an internal TypeError propagates as itself --
        # see web/tests/test_loader_errors.py. **kw still keeps the stub future-proof.)
        return list(added_obs), "OBS80"

    def fake_fit_obs_list(combined, meta, obscodes, force=False, meta_extra=None):
        captured["combined"] = combined
        return {"ok": True}

    orig_base, orig_fit, orig_parse = engine_api.base_obs_for, engine_api.fit_obs_list, loader._parse
    engine_api.base_obs_for = fake_base_obs_for
    engine_api.fit_obs_list = fake_fit_obs_list
    loader._parse = fake_parse
    try:
        result = loader.add_to_fit("dummy blob", engine_api._obscodes(), fit_key="dummy")
    finally:
        engine_api.base_obs_for, engine_api.fit_obs_list = orig_base, orig_fit
        loader._parse = orig_parse

    # Check the RETURN first: add_to_fit reports every early-out as {ok: False, error},
    # so asserting on it names the actual cause. Asserting only on the captured call
    # reports "fit_obs_list was never called" for any of a dozen different reasons.
    assert result.get("ok", True), f"add_to_fit bailed before fitting: {result.get('error')}"
    combined = captured.get("combined")
    assert combined is not None, "fit_obs_list was never called"
    jds = [o["jd_utc"] for o in combined]
    assert jds == sorted(jds), f"merged obs not time-sorted: {jds}"
    # the pasted 2021 precovery must sort to the FRONT, ahead of the 2459500 base obs
    assert jds[0] < 2459500.0, jds
