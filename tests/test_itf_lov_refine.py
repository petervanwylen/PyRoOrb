"""Offline unit tests for the continuous LOV offset refinement (itf/lov_refine.py).

The expensive half (LOV re-correction + propagation) is exercised live by
tests/itf_lov_refine_ab.py and the recovery suite; what is pinned here is the
decision logic that a live run would only ever show indirectly.
"""
import math

from pyroorb.itf.config import ItfConfig
from pyroorb.itf.lov_refine import node_grid, _windows, LovRefineProfile, StreamingLovRefine, _jd
from pyroorb.itf.lov_profile import LovProfileSet


def _cfg(**kw):
    return ItfConfig(**kw)


def test_node_grid_covers_every_variant_neighbourhood():
    cfg = _cfg(lov_refine_step=0.01, lov_refine_halfwidth=0.05)
    lov_sigmas, step = node_grid([0.0, 1.0, -2.5], cfg)
    assert step == 0.01
    for v in (0.0, 1.0, -2.5):
        near = [k for k in lov_sigmas if abs(k - v) <= 0.05 + 1e-9]
        # the variant itself plus five nodes either side
        assert len(near) == 11, (v, near)


def test_node_grid_coarsens_the_step_rather_than_dropping_the_tail():
    """Over budget the SPAN must survive: truncating it would silently
    un-refine exactly the far-out variants the fan was widened to reach."""
    variants = [i * 0.5 for i in range(-10, 11)]
    cfg = _cfg(lov_refine_step=0.001, lov_refine_halfwidth=0.15, lov_refine_max_nodes=200)
    lov_sigmas, step = node_grid(variants, cfg)
    assert len(lov_sigmas) <= 200
    assert step > 0.001                       # coarsened
    assert min(lov_sigmas) <= -5.0 and max(lov_sigmas) >= 5.0  # both tails still covered


def test_windows_need_distinct_nights_not_just_members():
    """Four tracklets of ONE night agreeing on k is one measurement, not
    four -- the cheapest coincidence there is."""
    one_night = [(0.0 + i * 1e-4, 100, i) for i in range(4)]
    assert _windows(one_night, 0.005, 3) == []
    three_nights = [(0.0, 100, "a"), (0.001, 101, "b"), (0.002, 102, "c")]
    got = _windows(three_nights, 0.005, 3)
    assert got and got[0][0] == 3


def test_windows_are_bounded_width_not_single_linkage():
    """A dense trial queue of candidates 0.001 apart must NOT chain into one
    cluster spanning the whole axis -- that is the failure a tolerance
    exists to prevent."""
    dense = [(i * 0.001, i, i) for i in range(400)]
    for n_nights, grp in _windows(dense, 0.005, 3):
        span = max(g[0] for g in grp) - min(g[0] for g in grp)
        assert span <= 0.010 + 1e-9, span


class _NoLovFit:
    """A fit whose LOV is unavailable -- a nongrav-augmented state, say."""
    def lov_refit(self):
        return None, None, None


def test_prepare_declines_without_a_lov():
    fine = LovRefineProfile(_NoLovFit(), _cfg(), [0.0, 1.0])
    fine.set_origin(2451545.0)
    assert fine.prepare({0, 1, 2}) is False
    assert fine.ready is False


class _Window:
    def __init__(self, a, b):
        self.unix_start, self.unix_end = a, b


def test_grid_cells_come_from_the_windows_and_bracket_them():
    """The propagation is sized before the sweep, off the windows -- the
    candidates are not known yet, and that ordering is what lets it happen
    while there is still memory for it."""
    cfg = _cfg(lov_refine_grid_days=4.0)
    fine = LovRefineProfile(_NoLovFit(), cfg, [0.0])
    day = 86400.0
    fine.set_origin(_jd(0.0))
    cells = fine.cells_for_windows([_Window(0.0, 10 * day)])
    assert cells == {1, 2, 3, 4}          # 10 days spans 3 cells, +1 partner
    # a window's own epochs must be interpolatable: cell and cell+1 both present
    for w_sec in (0.0, 5 * day, 10 * day):
        c = fine.cell_of(_jd(w_sec))
        assert c in cells and c + 1 in cells


def test_retention_is_bounded_by_config_not_by_the_field():
    """The memory of the pass is a choice: 30k candidates cost what 300 do."""
    cfg = _cfg(lov_refine_keep=50, lov_refine_resid_max=1e9)
    s = StreamingLovRefine.__new__(StreamingLovRefine)
    s.cfg, s.log = cfg, lambda *a: None
    s._pending, s._kept, s._nominal_hashes = [], [], set()
    s.n_seen, s.t_measure = 0, 0.0

    class _F:
        ready = True
        def measure(self, items, oc):
            return [(0.001 * i, float(i)) for i in range(len(items))]
    s.fine = _F()
    s.obscodes = None
    for batch in range(6):
        # (k, rows, night, served_by_nominal, first_end, last_end) -- rows are
        # kept as a reference and only turned into an identity at flush, for
        # the few that clear the cut
        s._pending = [(0.0, [{"obs80": f"{batch}-{i}-{j}" + " " * 80,
                              "trksub": "t", "stn": "F51"} for j in (0, 1)],
                       i, False, (0, "F51", 0, 0), (0, "F51", 0, 0))
                      for i in range(100)]
        s.flush()
    assert len(s._kept) <= 2 * cfg.lov_refine_keep
    assert s._kept[0].resid <= s._kept[-1].resid or len(s._kept) <= cfg.lov_refine_keep


def test_a_profile_whose_rows_carry_no_position_forgoes_the_refinement():
    """The refinement must never be the reason a candidate is dropped: a row
    the measure cannot use leaves ``ends`` None and the fan's k* stands."""
    lov_profiles = LovProfileSet(_cfg(), [0.0, 1.0])
    tr = {"_ids": frozenset({"x"}), "qual": 1.0, "dist": 10.0,
          "traj_category": "likely",
          "rows": [{"epoch_sec": 1.0}, {"epoch_sec": 2.0}]}
    lov_profiles.add(0.0, tr)
    p = lov_profiles.profiles[tr["_ids"]]
    assert p.ends is None
    assert math.isinf(p.lov_refine_resid)
