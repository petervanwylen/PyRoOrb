"""compact_seed_window: the shortest N-consecutive-night seed in any apparition."""

import sys
from pathlib import Path


from pyroorb.itf.harness import compact_seed_window


def _app(night_jds, per_night=2):
    return [{"jd_utc": jd + 0.01 * k} for jd in night_jds for k in range(per_night)]


def test_picks_shortest_window_across_apparitions():
    apps = [_app([100, 110, 120, 130]),          # 3-night windows span 20 d
            _app([500, 501, 502.5, 540])]         # 500..502.5 spans 2.5 d
    i, head, before, after = compact_seed_window(apps, 3, min_obs=6)
    assert i == 1
    assert head[0]["jd_utc"] == 500
    assert before == [] and [o["jd_utc"] for o in after] == [540, 540.01]


def test_middle_window_leaves_both_sides_plantable():
    apps = [_app([10, 30, 31, 32, 60])]
    i, head, before, after = compact_seed_window(apps, 3, min_obs=6)
    assert {round(o["jd_utc"]) for o in head} == {30, 31, 32}
    assert {round(o["jd_utc"]) for o in before} == {10}
    assert {round(o["jd_utc"]) for o in after} == {60}
    assert len(head) + len(before) + len(after) == len(apps[0])


def test_obs_floor_prefers_richer_window_then_relaxes():
    thin = _app([1, 2, 3], per_night=1)           # compact but only 3 obs
    rich = _app([50, 55, 60], per_night=2)
    assert compact_seed_window([thin, rich], 3, min_obs=6)[0] == 1
    assert compact_seed_window([thin], 3, min_obs=6)[0] == 0


def test_no_apparition_has_enough_nights():
    assert compact_seed_window([_app([1, 2])], 3) is None
