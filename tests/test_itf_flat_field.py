"""Flat-field trial selection: when the ranker rates no candidate a plausible
match, the singles window widens (and a barren take may backtrack at any depth)."""
from pyroorb.itf.config import ItfConfig
from pyroorb.itf.search import _build_trials, flat_field, _NULL_TRACER


def _cands(ps):
    return [{"ml_p": p, "_ids": frozenset({f"t{i}"})} for i, p in enumerate(ps)]


def test_flat_when_no_likely_candidate_is_plausible():
    cfg = ItfConfig()
    assert flat_field(cfg, _cands([0.049, 0.024, 0.009]), 3)
    assert not flat_field(cfg, _cands([0.6, 0.024]), 2)          # one plausible
    assert not flat_field(cfg, _cands([0.01]), 1)                 # a lone candidate
    cfg.flat_field_ml_p = 0.0
    assert not flat_field(cfg, _cands([0.049, 0.024]), 2)         # disabled


def test_flat_field_widens_the_singles_window():
    cfg = ItfConfig()
    cfg.recurse_second_singles = 0
    cands = _cands([0.05 - 0.001 * i for i in range(40)])
    ranked = list(range(40))
    normal = _build_trials(cfg, cands, 40, ranked, None, False, _NULL_TRACER, "")
    flat = _build_trials(cfg, cands, 40, ranked, None, False, _NULL_TRACER, "", flat=True)
    singles = lambda ts: [t[0] for t in ts if len(t) == 1]
    assert singles(normal) == ranked[:cfg.recurse_singles]
    assert singles(flat) == ranked[:cfg.flat_field_singles]
