"""_carried_pool: a tracklet this level re-swept is ranked by its fresh score."""
from types import SimpleNamespace

from pyroorb.itf.search import _carried_pool


def _ctx(cap):
    return SimpleNamespace(cfg=SimpleNamespace(carried_max=cap))


def _rank(d):
    return 1.0 - d.get("ml_p", 0.0)


def test_fresh_copy_beats_its_stripped_inherited_copy():
    truth = frozenset({"t1", "t2"})
    # inherited from the level above: stripped of its score, as _carried_pool returns it
    inherited = [{"_ids": truth, "rows": ["old"], "_feat": None}]
    # this level swept it again and scored it well, among weaker distractors
    fresh = [{"_ids": truth, "rows": ["new"], "ml_p": 0.216, "traj_category": "likely"}]
    fresh += [{"_ids": frozenset({f"d{i}"}), "rows": [], "ml_p": 0.01,
               "traj_category": "maybe"} for i in range(5)]
    out = _carried_pool(_ctx(3), inherited, fresh, _rank)
    assert out[0]["_ids"] == truth            # ranked first, not cut by the cap
    assert out[0]["rows"] == ["new"]


def test_inherited_only_tracklets_still_carry():
    old = frozenset({"o"})
    out = _carried_pool(_ctx(10), [{"_ids": old, "rows": [], "_feat": None}],
                        [{"_ids": frozenset({"n"}), "rows": [], "ml_p": 0.5,
                          "traj_category": "likely"}], _rank)
    assert {t["_ids"] for t in out} == {old, frozenset({"n"})}


def test_unlikely_fresh_tracklets_are_not_carried():
    out = _carried_pool(_ctx(10), [], [{"_ids": frozenset({"u"}), "rows": [],
                                        "ml_p": 0.0, "traj_category": "unlikely"}], _rank)
    assert out is None
