"""Unit tests for the apparition-wise comet extension's pure pieces (fit_object
_comet_extend_apparitions is exercised end-to-end only by the seeded 21P fixture):

- the evaporation guard's `orig_n`: a mask that is a deliberate SUBSET of the evaluator
  (one apparition of a comet on the full record) must be measured against its own entry
  count, not the evaluator's -- against ev.n the guard fired on every ramp step and the
  4197-of-7434 apparition kept every obs at 1.84";
- the accumulative model sequence choice (_ng_comet_choice_accum): an unjustified comet1 is
  skipped, not a wall, so comet2 is judged straight against gravity on dp=2 -- while the
  strict model sequence (_ng_comet_choice) still breaks there, byte for byte as before.
"""
import numpy as np
import pytest

import pyroorb.fit_object as fo
from pyroorb.arc_extend import auto_reject


class _StubEv:
    def __init__(self, n):
        self.times_tdb = np.linspace(0.0, 100.0, n)
        self.n = n


def test_guard_measures_a_subset_mask_against_its_own_count():
    ev = _StubEv(100)
    mask = np.zeros(100, dtype=bool)
    mask[60:100] = True                          # 40 of 100: under 2/3 of ev.n already
    rs = np.full(100, 0.5)
    rs[:60] = 1e4                                # the excluded apparitions: nowhere near redeemable
    rs[60] = 50.0                                # one blunder inside the subset
    # default (historical): measured against ev.n -> guard fires, nothing rejected
    m1 = mask.copy()
    assert auto_reject(ev, m1, None, 3.0, 3.3, rs=rs) == -1
    assert m1[60]
    # orig_n = the entry count -> the blunder goes
    m2 = mask.copy()
    assert auto_reject(ev, m2, None, 3.0, 3.3, rs=rs, orig_n=int(mask.sum())) == 1
    assert not m2[60] and m2.sum() == 39


def test_guard_default_is_unchanged_on_a_whole_arc():
    ev = _StubEv(30)
    mask = np.ones(30, dtype=bool)
    rs = np.full(30, 0.5); rs[3] = 50.0
    m1, m2 = mask.copy(), mask.copy()
    assert auto_reject(ev, m1, None, 3.0, 3.3, rs=rs) == 1
    assert auto_reject(ev, m2, None, 3.0, 3.3, rs=rs, orig_n=30) == 1
    assert (m1 == m2).all()


def _nested_models(chi2_0, chi2_1, chi2_2, chi2_3=None, N=8000):
    models = {"comet1": {"chi2": chi2_1, "phys_ok": True},
              "comet2": {"chi2": chi2_2, "phys_ok": True}}
    if chi2_3 is not None:
        models["comet3"] = {"chi2": chi2_3, "phys_ok": True}
    out = {"chi2_0": chi2_0, "N": N, "models": models}
    return models, out


def test_accum_choice_skips_a_useless_comet1_and_takes_comet2_over_gravity():
    # comet1 buys 5 chi2 (dBIC 5 - ln 8000 < NONGRAV_BIC_MARGIN), comet2 gains 97% over gravity
    models, out = _nested_models(100000.0, 99995.0, 3000.0)
    assert fo._ng_comet_choice(models, out) is None          # the strict model sequence breaks
    assert fo._ng_comet_choice_accum(models, out) == "comet2"


def test_accum_choice_agrees_with_strict_when_every_step_earns_its_place():
    models, out = _nested_models(100000.0, 50000.0, 20000.0, 8000.0)
    assert fo._ng_comet_choice(models, out) == "comet3"
    assert fo._ng_comet_choice_accum(models, out) == "comet3"


def test_accum_choice_still_makes_the_added_terms_earn_their_place():
    # comet2 barely better than comet1, comet3 barely better than comet2: neither earns it
    models, out = _nested_models(100000.0, 20000.0, 19995.0, 19990.0)
    assert fo._ng_comet_choice(models, out) == "comet1"
    assert fo._ng_comet_choice_accum(models, out) == "comet1"


def test_accum_continue_keeps_climbing_after_an_unjustified_step():
    models, out = _nested_models(100000.0, 99995.0, 3000.0)
    assert not fo._comet_continue(out, "comet1")             # strict: stop at comet1
    assert fo._comet_continue_accum(out, "comet1")           # accum: comet2 may be the answer
    out["models"]["comet2"]["chi2"] = float("inf")           # a step that failed to fit
    assert not fo._comet_continue_accum(out, "comet2")



# ===========================================================================================
# THE ORCHESTRATION of fit_object._comet_extend_apparitions.
#
# Before these tests the seed choice, the growth order, the refusal ledger, the interior
# second look and the "nothing joined -> None" path were covered end-to-end ONLY by the 21P
# regression fixture -- which is flagged `benchmark`, i.e. deselected from `pytest`,
# `pytest -n 4` and `pytest --live`. The one object the design was derived from was the one
# object the routine suite never ran.
#
# Two layers:
#   * the PURE decisions (_ext_seed_candidates / _ExtBlocks / _ext_next_apparition /
#     _ext_join_ok / _ext_interior_refusals) -- the arithmetic lifted verbatim out of the big
#     function so each rule can be asserted without an evaluator;
#   * the FUNCTION ITSELF driven against a scripted fake evaluator/fitter, so the loop that
#     wires those decisions together is exercised in milliseconds.
# Neither layer replaces the 21P fixture as evidence that the real LM converges on a real
# comet; they are evidence that the CONTROL FLOW is what the docstring claims.
# ===========================================================================================

from pyroorb.arc_extend import OPP_GAP_DAYS


def _t_apps(sizes, gap=400.0):
    """Times for consecutive apparitions of the given sizes -- one day apart inside an
    apparition, `gap` days (> OPP_GAP_DAYS, so `_clusters` splits them) between.
    Returns (times, clusters) with clusters in the same form `_clusters` produces."""
    assert gap > OPP_GAP_DAYS
    t, clusters, now = [], [], 0.0
    for k in sizes:
        clusters.append((len(t), len(t) + k - 1))
        t.extend(now + i for i in range(k))
        now += k + gap
    return np.asarray(t, float), clusters


# --- the pure decisions --------------------------------------------------------------------

def test_seed_candidates_are_later_rich_apparitions_most_recent_first():
    t, cl = _t_apps([40, 4, 30, 12])
    base = np.zeros(len(t), bool); base[cl[0][0]:cl[0][1] + 1] = True
    got = fo._ext_seed_candidates(t, cl, 40, float(t[base].max()))
    # app1's 4 obs are below COMET_EXT_ANCHOR_FRAC * 40 = 10 and drop out; app2 (30) and app3
    # (12) qualify, MOST RECENT FIRST -- the prefer-the-recent rule.
    assert got == [cl[3], cl[2]]


def test_seed_candidates_exclude_everything_not_later_than_the_current_arc():
    t, cl = _t_apps([20, 20, 20])
    base = np.zeros(len(t), bool); base[cl[2][0]:cl[2][1] + 1] = True   # newest already in
    assert fo._ext_seed_candidates(t, cl, 20, float(t[base].max())) == []


def test_seed_candidates_respect_the_absolute_fit_min_obs_floor():
    # richest is tiny, so COMET_EXT_ANCHOR_FRAC * richest < FIT_MIN_OBS and the floor bites
    t, cl = _t_apps([4, 2, 8])
    base = np.zeros(len(t), bool); base[cl[0][0]:cl[0][1] + 1] = True
    assert fo._ext_seed_candidates(t, cl, 4, float(t[base].max())) == [cl[2]]


def _arc(t, cl, i, j=None):
    """(lo, hi) of an arc spanning apparitions cl[i]..cl[j] (j defaults to i)."""
    j = i if j is None else j
    return float(t[cl[i][0]]), float(t[cl[j][1]])


def test_blocks_a_refusal_forecloses_poorer_apparitions_on_its_own_side():
    t, cl = _t_apps([4, 1254, 465])            # 21P's shape: an old tail behind a rich refusal
    b = fo._ExtBlocks(t)
    b.add(*cl[1], lo=float(t[cl[2][0]]))       # 1254 obs refused, on the 'lo' side
    assert b.blocked(*cl[0], *_arc(t, cl, 2))  # the 4 obs beyond it
    assert not b.blocked(*cl[2], *_arc(t, cl, 2))


def test_blocks_a_richer_apparition_beyond_a_refusal_is_still_tried():
    t, cl = _t_apps([465, 19, 300])
    b = fo._ExtBlocks(t)
    b.add(*cl[1], lo=float(t[cl[2][0]]))       # 21P's refused 19-obs tail
    assert not b.blocked(*cl[0], *_arc(t, cl, 2))   # 465 > 19: the model has not been refuted here


def test_blocks_a_refusal_under_comet_min_obs_blocks_nothing():
    t, cl = _t_apps([3, fo.COMET_MIN_OBS - 1, 100])
    b = fo._ExtBlocks(t)
    b.add(*cl[1], lo=float(t[cl[2][0]]))
    assert not b.blocked(*cl[0], *_arc(t, cl, 2))


def test_blocks_are_one_sided():
    t, cl = _t_apps([50, 200, 50])
    b = fo._ExtBlocks(t)
    b.add(*cl[1], lo=float(t[cl[0][0]]))       # refused from an arc that starts before it -> 'hi'
    assert b.blocked(*cl[2], *_arc(t, cl, 0))
    assert not b.blocked(*cl[0], *_arc(t, cl, 0))


def test_blocks_a_refusal_the_arc_has_grown_around_is_stale():
    # cl[1] refused when the arc was cl[0] alone: a live 'hi' block against cl[3]'s 8 obs.
    # Once cl[2] (richer, so never blocked) has joined, the arc spans cl[1] and the same
    # entry counts for nothing -- the model was carried past it by a join that succeeded.
    t, cl = _t_apps([50, 20, 50, 8])
    b = fo._ExtBlocks(t)
    b.add(*cl[1], lo=float(t[cl[0][0]]))
    assert b.blocked(*cl[3], *_arc(t, cl, 0))
    assert not b.blocked(*cl[3], *_arc(t, cl, 0, 2))
    # and the mirror image on the 'lo' side
    t, cl = _t_apps([8, 50, 20, 50])
    b = fo._ExtBlocks(t)
    b.add(*cl[2], lo=float(t[cl[3][0]]))
    assert b.blocked(*cl[0], *_arc(t, cl, 3))
    assert not b.blocked(*cl[0], *_arc(t, cl, 1, 3))


def test_next_apparition_goes_forward_in_time_first_then_backward():
    t, cl = _t_apps([10, 10, 10, 10, 10])
    hi = float(t[cl[2][1]])
    todo = [cl[0], cl[1], cl[3], cl[4]]
    assert fo._ext_next_apparition(todo, t, hi) == cl[3]      # nearest LATER one
    todo.remove(cl[3])
    assert fo._ext_next_apparition(todo, t, hi) == cl[4]
    todo.remove(cl[4])
    assert fo._ext_next_apparition(todo, t, hi) == cl[1]      # then backward, latest-first
    todo.remove(cl[1])
    assert fo._ext_next_apparition(todo, t, hi) == cl[0]


def test_join_ok_each_clause_is_load_bearing():
    good = dict(rn=1.2, rn0=1.0, med=1.0, frac=0.9)
    assert fo._ext_join_ok(**good)
    assert not fo._ext_join_ok(**{**good, "rn": float("nan")})
    # catastrophic in absolute terms, even though the ratio bar would pass off a bad seed
    assert not fo._ext_join_ok(**{**good, "rn": fo.CATASTROPHIC_RMS_N + 0.1, "rn0": 100.0})
    assert not fo._ext_join_ok(**{**good, "rn": fo.COMET_EXT_RMS_N_FACTOR + 1e-6})
    assert not fo._ext_join_ok(**{**good, "med": fo.DISCORDANT_SIGMA + 1e-9})
    assert not fo._ext_join_ok(**{**good, "frac": fo.COMET_EXT_KEEP_FRAC - 1e-9})


def test_join_ok_rms_bar_is_floored_at_one_sigma():
    # a seed already at 0.4 sigma must not set an unreachable 0.6-sigma bar for the join
    assert fo._ext_join_ok(rn=1.4, rn0=0.4, med=1.0, frac=1.0)
    assert not fo._ext_join_ok(rn=1.6, rn0=0.4, med=1.0, frac=1.0)


def test_interior_refusals_are_strictly_inside_the_grown_arc():
    t, cl = _t_apps([10, 10, 10, 10])
    refused = [{"obs": list(c)} for c in cl]
    got = fo._ext_interior_refusals(refused, t, float(t[cl[0][0]]), float(t[cl[3][1]]))
    assert [r["obs"] for r in got] == [list(cl[1]), list(cl[2])]   # endpoints are not interior


# --- the function itself, on a scripted fake fit -------------------------------------------

class _FakeEv:
    """The evaluator surface _comet_extend_apparitions touches. `sigma` is supplied by the
    scenario as a function of (set of included apparitions, apparition index), so a scenario
    can express the thing that makes the interior second look meaningful: an apparition whose
    residuals depend on WHAT ELSE the arc holds."""

    def __init__(self, t, clusters, sigma):
        self.times_tdb = np.asarray(t, float)
        self.n = len(t)
        self._cl, self._sigma = clusters, sigma
        self.mask_now = np.ones(self.n, bool)

    def _apps(self, mask):
        return frozenset(i for i, (cs, ce) in enumerate(self._cl) if mask[cs:ce + 1].any())

    def fit(self, state, mask, **kw):
        return np.asarray(state, float), None

    def rms_norm(self, state, mask):
        mask = np.asarray(mask, bool)
        apps = self._apps(mask)
        return float(max(self._sigma(apps, i) for i in apps))

    def resid_sigmas(self, state):
        """Per-obs sigma under the LAST mask fitted -- which is what the caller wants: it
        asks for these immediately after refitting `trial`."""
        apps = self._apps(self.mask_now)
        rs = np.empty(self.n)
        for i, (cs, ce) in enumerate(self._cl):
            rs[cs:ce + 1] = self._sigma(apps, i)
        return rs


class _FakeOrbit:
    def __init__(self, ctx, mask, rn):
        self._ctx, self._mask, self._state = ctx, np.asarray(mask, bool), np.zeros(6)
        self.history, self._rn = [], float(rn)

    def rms_normalized(self):
        return self._rn

    def n_observations(self):
        return int(self._mask.sum())


class _Scenario:
    """Wires the fakes in over fit_object for one test. `sigmas` is either a per-apparition
    list of constants or a callable (included apparitions, apparition index) -> sigma.
    `reject` is the fraction of each included apparition the rejection round removes."""

    def __init__(self, monkeypatch, sizes, sigmas, base_app, reject=0.0, gap=400.0):
        self.t, self.clusters = _t_apps(sizes, gap)
        sigma = sigmas if callable(sigmas) else (lambda apps, i, _s=sigmas: _s[i])
        self.ev = _FakeEv(self.t, self.clusters, sigma)
        self.ctx = {"ev": self.ev, "epoch_tdb": 2455000.0}
        mask = np.zeros(len(self.t), bool)
        cs, ce = self.clusters[base_app]
        mask[cs:ce + 1] = True
        self.fit = _FakeOrbit(self.ctx, mask, self.ev.rms_norm(None, mask))
        self.reject = float(reject)
        self.tried = []                          # every mask the fitter was asked for

        import pyroorb.arc_extend as ae
        monkeypatch.setattr(fo, "_lift_nongrav_state", lambda f, m: np.zeros(9))
        monkeypatch.setattr(fo, "_reanchor_apparition", lambda f, s, e, v=False: None)
        monkeypatch.setattr(fo, "_build_orbit",
                            lambda ctx, st, m, result, hist: self._orbit(ctx, m))
        monkeypatch.setattr(fo, "refine_nongrav", self._refine)
        monkeypatch.setattr(ae, "auto_reject_within_arc", self._reject_round)

    def _orbit(self, ctx, mask):
        m = np.asarray(mask, bool).copy()
        self.ev.mask_now = m
        return _FakeOrbit(ctx, m, self.ev.rms_norm(None, m))

    def _refine(self, fit, model, mask=None, **kw):
        m = np.asarray(mask, bool).copy()
        self.tried.append(m)
        return self._orbit(self.ctx, m)

    def _reject_round(self, ev, mask, state, orig_span=None, orig_n=None):
        if self.reject <= 0.0:
            return 0
        for (cs, ce) in self.clusters:           # drop the tail of each included apparition
            idx = np.flatnonzero(mask[cs:ce + 1]) + cs
            k = int(round(self.reject * len(idx)))
            if k:
                mask[idx[len(idx) - k:]] = False
        self.ev.mask_now = np.asarray(mask, bool).copy()
        return 1

    def tried_app(self, i):
        cs, ce = self.clusters[i]
        return any(m[cs:ce + 1].any() for m in self.tried)


def test_extension_returns_none_when_every_apparition_is_already_included(monkeypatch):
    sc = _Scenario(monkeypatch, [20, 20], [1.0, 1.0], base_app=0)
    sc.fit._mask[:] = True
    assert fo._comet_extend_apparitions(sc.ctx, sc.fit, 40) is None


def test_extension_returns_none_when_nothing_the_model_can_hold_is_left(monkeypatch):
    # the other apparitions sit at 9 sigma: past DISCORDANT_SIGMA and past CATASTROPHIC_RMS_N
    sc = _Scenario(monkeypatch, [20, 20, 20], [9.0, 1.0, 9.0], base_app=1)
    assert fo._comet_extend_apparitions(sc.ctx, sc.fit, 40) is None


def test_extension_grows_forward_first_then_backward_and_reports_what_joined(monkeypatch):
    # app2 (200 obs) is the seed and STAYS the seed: app3/app4 carry 8 obs each, under
    # COMET_EXT_ANCHOR_FRAC x 200 = 50, so the re-seed declines them (they are still perfectly
    # joinable -- that bar is about what may ANCHOR the growth, not what may join it).
    sc = _Scenario(monkeypatch, [20, 20, 200, 8, 8], [1.0] * 5, base_app=2)
    out = fo._comet_extend_apparitions(sc.ctx, sc.fit, 40)
    assert out is not None
    ctx, fit, mask, warm, rec = out
    # forward first (app3 then app4), then backward latest-first (app1 then app0)
    assert [r["obs"][0] for r in rec["joined"]] == [sc.clusters[k][0] for k in (3, 4, 1, 0)]
    assert rec["refused"] == []
    assert mask.all() and rec["n_included"] == len(sc.t)
    # the model sequence's warm start carries BOTH the judged model and comet1 (the nested model sequence has
    # to justify comet1 over gravity before it may look at comet2)
    assert set(warm) == {fo.COMET_EXT_MODEL, "comet1"}


def test_extension_seeds_from_the_most_recent_well_observed_clean_apparition(monkeypatch):
    # the fit arrives on the RICHEST apparition (app0, 200 obs), as the catastrophic stage
    # hands it over; app2 is later and rich enough, so the growth re-seeds there instead.
    sc = _Scenario(monkeypatch, [200, 4, 60], [1.0, 1.0, 1.0], base_app=0)
    out = fo._comet_extend_apparitions(sc.ctx, sc.fit, 40)
    assert out is not None and out[4]["seed_obs"] == list(sc.clusters[2])


def test_extension_does_not_reseed_onto_a_thin_later_fragment(monkeypatch):
    # app2's 4 obs are under COMET_EXT_ANCHOR_FRAC x 200 -- 21P's "11-obs 2024 fragment" case
    sc = _Scenario(monkeypatch, [200, 60, 4], [1.0, 1.0, 1.0], base_app=0)
    out = fo._comet_extend_apparitions(sc.ctx, sc.fit, 40)
    assert out is not None and out[4]["seed_obs"] == list(sc.clusters[1])


def test_extension_lets_a_refusal_block_poorer_apparitions_behind_it(monkeypatch):
    # app0 (4 clean obs) sits behind app1 (100 obs the model cannot hold). Growing backward
    # from app2, app1 is refused and must foreclose app0: four loose old positions may not
    # extend an arc past an apparition that demonstrably refuted the model.
    sc = _Scenario(monkeypatch, [4, 100, 50], [1.0, 4.0, 1.0], base_app=2)
    assert fo._comet_extend_apparitions(sc.ctx, sc.fit, 40) is None
    assert not sc.tried_app(0)                   # never even trialled


def test_extension_still_tries_a_richer_apparition_beyond_a_refusal(monkeypatch):
    # 21P's actual shape: the 19-obs tail is refused, the 465-obs apparition beyond it is not
    # blocked by it and joins.
    sc = _Scenario(monkeypatch, [40, 12, 30], [1.0, 4.0, 1.0], base_app=2)
    out = fo._comet_extend_apparitions(sc.ctx, sc.fit, 40)
    assert out is not None
    ctx, fit, mask, warm, rec = out
    assert [r["obs"] for r in rec["joined"]] == [list(sc.clusters[0])]
    assert [r["obs"] for r in rec["refused"]] == [list(sc.clusters[1])]


def test_extension_gives_an_interior_refusal_a_second_look(monkeypatch):
    # app1 is discordant while the arc is app2 alone, and fine once app0 (which is richer, so
    # the refusal does not block it) has joined and the arc SPANS it. That is exactly the case
    # the second look exists for -- a verdict reached against a shorter arc is a different
    # question once the arc is longer, and an interior join cannot extend the arc anywhere.
    def sigma(apps, i):
        if i == 1:
            return 1.0 if 0 in apps else 4.0
        return 1.0
    sc = _Scenario(monkeypatch, [40, 12, 30], sigma, base_app=2)
    out = fo._comet_extend_apparitions(sc.ctx, sc.fit, 40)
    assert out is not None
    ctx, fit, mask, warm, rec = out
    assert rec["refused"] == []
    assert [r["obs"] for r in rec["joined"]] == [list(sc.clusters[0]), list(sc.clusters[1])]
    assert rec["joined"][1].get("retried") is True
    assert mask.all()


def test_extension_refuses_an_apparition_the_rejection_guts(monkeypatch):
    # every residual is fine, but the rejection round removes 60% of each apparition: past
    # COMET_EXT_KEEP_FRAC, so no join survives however low the survivors' median.
    sc = _Scenario(monkeypatch, [30, 30], [1.0, 1.0], base_app=0, reject=0.6)
    assert fo._comet_extend_apparitions(sc.ctx, sc.fit, 40) is None


def test_extension_adopts_a_successful_reanchor(monkeypatch):
    sc = _Scenario(monkeypatch, [20, 20], [1.0, 1.0], base_app=0)
    ctx2 = {"ev": sc.ev, "epoch_tdb": 2460000.0}
    re = _FakeOrbit(ctx2, sc.fit._mask, 0.5)
    monkeypatch.setattr(fo, "_reanchor_apparition", lambda f, s, e, v=False: re)
    out = fo._comet_extend_apparitions(sc.ctx, sc.fit, 40)
    assert out is not None
    assert out[0] is ctx2 and out[1] is re       # the re-anchored ctx/fit are what it ran on
    assert out[4]["epoch_tdb"] == 2460000.0


def test_extension_forgets_a_refusal_the_arc_has_grown_around_backward(monkeypatch):
    # Growing backward from app3: app2 (10 obs) is refused, app1 (300 obs, richer) is not
    # blocked by it and joins -- the arc now SPANS app2. app0 (8 clean obs) must still get a
    # hearing: the refusal it would have to be carried "past" is one the arc already went
    # around, on the strength of 300 obs, so it bounds nothing any more.
    sc = _Scenario(monkeypatch, [8, 300, 10, 200], [1.0, 1.0, 9.0, 1.0], base_app=3)
    out = fo._comet_extend_apparitions(sc.ctx, sc.fit, 40)
    assert out is not None
    ctx, fit, mask, warm, rec = out
    assert sc.tried_app(0)
    assert [r["obs"] for r in rec["joined"]] == [list(sc.clusters[1]), list(sc.clusters[0])]
    assert [r["obs"] for r in rec["refused"]] == [list(sc.clusters[2])]


def test_extension_forgets_a_refusal_the_arc_has_grown_around_forward(monkeypatch):
    # The same thing forward: app2 (240 obs, under COMET_EXT_ANCHOR_FRAC x 1000 so the growth
    # does not re-seed there) joins past the refused app1; app3's 8 obs beyond it are tried.
    sc = _Scenario(monkeypatch, [1000, 10, 240, 8], [1.0, 9.0, 1.0, 1.0], base_app=0)
    out = fo._comet_extend_apparitions(sc.ctx, sc.fit, 40)
    assert out is not None
    ctx, fit, mask, warm, rec = out
    assert sc.tried_app(3)
    assert [r["obs"] for r in rec["joined"]] == [list(sc.clusters[2]), list(sc.clusters[3])]


def test_extension_a_hearing_is_not_a_join_a_bad_sparse_apparition_is_still_refused(monkeypatch):
    # The hearing the stale block used to deny is still the full trial: a sparse far
    # apparition that IS a mislinkage (9 sigma) is refused on its own residuals.
    sc = _Scenario(monkeypatch, [8, 300, 10, 200], [9.0, 1.0, 9.0, 1.0], base_app=3)
    out = fo._comet_extend_apparitions(sc.ctx, sc.fit, 40)
    assert out is not None
    ctx, fit, mask, warm, rec = out
    assert sc.tried_app(0)
    assert [r["obs"] for r in rec["joined"]] == [list(sc.clusters[1])]
    assert sorted(r["obs"] for r in rec["refused"]) == [list(sc.clusters[0]), list(sc.clusters[2])]
    assert not mask[sc.clusters[0][0]:sc.clusters[0][1] + 1].any()
