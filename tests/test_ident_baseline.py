"""The PRE-EXISTING-fault baseline of ident_check (link_systematic, preexisting_exception):
an identification is judged only on what it ADDS to each body's own record.

Synthetic fits, with the solo fits put in the joint fit's cache, so nothing here runs the engine.
The engine-level guard -- known-bogus pairs keep firing -- is the slow test at the bottom.
"""
import os

import numpy as np
import pytest

from pyroorb import ident_check as ic


class _Fit:
    """Just what the systematic/exception tests read: _obs, _mask, _resid_all."""

    def __init__(self, obs, resid, mask=None):
        self._obs = obs
        self._resid_all = np.asarray(resid, float)
        self._mask = np.ones(len(obs), bool) if mask is None else np.asarray(mask, bool)


def _obs(desig, jd, code="F51", perm=None):
    return {"designation": desig, "jd_utc": jd, "obscode": code, "packed_perm": perm}


# Body A: one apparition, 4 nights x 4 obs whose RA means sit 0.9" off in alternating directions
# (tight within a night) -- a trend systematic_misfit fires on. Body B: a clean second apparition
# 400 d later, plus a third body's tracklet so the link has something to identify.
_A_MEANS = [0.9, -0.9, 0.9, -0.9]


def _body_a(scale=1.0, n_nights=4):
    obs, res = [], []
    for n in range(n_nights):
        for k in range(4):
            obs.append(_obs("2014 AA1", 2456700.0 + 3 * n + 0.01 * k))
            res.append((scale * _A_MEANS[n] + 0.02 * (-1) ** k, 0.01 * (-1) ** k))
    return obs, res


def _body_b():
    obs, res = [], []
    for n in range(3):
        for k in range(4):
            obs.append(_obs("2015 BB2", 2457100.0 + 2 * n + 0.01 * k, code="G96"))
            res.append((0.03 * (-1) ** k, 0.02 * (-1) ** (k + 1)))
    return obs, res


def _joint(scale_a=1.0):
    oa, ra = _body_a(scale_a)
    ob, rb = _body_b()
    return _Fit(oa + ob, ra + rb)


def _with_solo(fit, solos):
    fit.__dict__["_solo_fits"] = dict(solos)
    return fit


def test_trend_fires_without_a_baseline():
    fit = _with_solo(_joint(), {"2014 AA1": None})        # no solo fit -> no exemption
    desc, pre = ic.link_systematic(fit)
    assert desc and "RA" in desc and pre == []
    # same apparition, axis and drift as the plain gate; the ratio differs only because the
    # verdict's significance test also counts the station offset (SYS_SE_STATION_TAU)
    assert desc.split(", ")[0] == ic.systematic_misfit(fit).split(", ")[0]


def test_station_tau_quiets_a_harmless_significant_offset():
    # three 703 nights whose RA means sit 0.55" apart, 8 obs each scattered 0.3": the scatter
    # test does not fire (1.8x), the mean's own error does (SE 0.11", ~4.9x) -- but 703's own
    # night-to-night offset (tau ~0.37" in RA) is exactly that size, so the identification
    # verdict, which counts it, calls it noise
    from pyroorb import nightmodel
    tau = nightmodel.station_params("703")[2]
    obs, res = [], []
    for n, mu in enumerate([0.55, -0.55, 0.55]):
        for k in range(8):
            obs.append(_obs("2014 AA1", 2456700.0 + 3 * n + 0.01 * k, code="703"))
            res.append((mu + 0.3 * (-1) ** k, 0.0))
    ob, rb = _body_b()
    fit = _with_solo(_Fit(obs + ob, res + rb), {"2014 AA1": None})
    se = 0.3 * np.sqrt(8 / 7) / np.sqrt(8)
    assert 0.55 / se >= ic.SYS_SE_RATIO > 0.55 / np.hypot(se, tau), tau   # the premise
    assert ic.systematic_misfit(fit)
    assert ic.link_systematic(fit) == (None, [])


def test_trend_already_in_own_record_is_set_aside():
    oa, ra = _body_a()
    fit = _with_solo(_joint(), {"2014 AA1": _Fit(oa, ra)})  # A alone trends exactly the same
    desc, pre = ic.link_systematic(fit)
    assert desc is None
    assert len(pre) == 1 and "RA" in pre[0]


def test_trend_the_link_makes_worse_still_fires():
    oa, ra = _body_a(scale=0.5)                            # A alone: 0.45", the joint: 0.9"
    fit = _with_solo(_joint(), {"2014 AA1": _Fit(oa, ra)})
    desc, pre = ic.link_systematic(fit)
    assert desc and pre == []
    assert 0.9 > ic.BASELINE_WORSEN_FACTOR * 0.45 + ic.BASELINE_WORSEN_SLACK


def test_clean_own_record_keeps_the_trend():
    oa, _ = _body_a()
    clean = [(0.02 * (-1) ** k, 0.01) for k in range(len(oa))]
    fit = _with_solo(_joint(), {"2014 AA1": _Fit(oa, clean)})
    desc, pre = ic.link_systematic(fit)
    assert desc and pre == []


def test_mixed_apparition_is_never_exempt():
    # the trending apparition also holds the OTHER body's obs: no one record to compare with
    oa, ra = _body_a()
    oa[0] = _obs("2015 BB2", oa[0]["jd_utc"])
    ob, rb = _body_b()
    fit = _with_solo(_Fit(oa + ob, ra + rb), {"2014 AA1": _Fit(*_body_a())})
    desc, pre = ic.link_systematic(fit)
    assert desc and pre == []


def test_one_body_via_packed_perm():
    # a numbered record carries several labels; the number joins them into one body
    oa, ra = _body_a()
    for i, o in enumerate(oa):
        o["designation"] = "415989" if i % 2 else "2002 AC19"
        o["packed_perm"] = "f5989"
    ob, rb = _body_b()
    fit = _with_solo(_Fit(oa + ob, ra + rb), {"415989": _Fit(oa, ra)})   # the unpacked number
    desc, pre = ic.link_systematic(fit)
    assert desc is None and len(pre) == 1


def test_no_solo_fit_when_nothing_fires():
    ob, rb = _body_b()
    oa, _ = _body_a()
    fit = _Fit(oa + ob, [(0.02, 0.01)] * len(oa) + rb)
    assert ic.link_systematic(fit) == (None, [])
    assert "_solo_fits" not in fit.__dict__                # the fast path asked no solo fit


def test_exception_rejected_by_own_record_is_preexisting():
    oa, ra = _body_a()
    ob, rb = _body_b()
    fit = _Fit(oa + ob, ra + rb)
    tr = [0, 1, 2, 3]                                      # A's first night, excepted by the link
    solo_mask = [i not in tr for i in range(len(oa))]
    _with_solo(fit, {"2014 AA1": _Fit([dict(o) for o in oa], ra, mask=solo_mask)})
    assert ic.preexisting_exception(fit, tr)
    assert not ic.preexisting_exception(fit, [4, 5, 6, 7])   # A alone keeps that night


def test_exception_solo_keeps_it_or_no_solo():
    oa, ra = _body_a()
    ob, rb = _body_b()
    fit = _with_solo(_Fit(oa + ob, ra + rb), {"2014 AA1": _Fit(oa, ra)})
    assert not ic.preexisting_exception(fit, [0, 1, 2, 3])   # A alone keeps every obs
    fit.__dict__["_solo_fits"] = {"2014 AA1": None}
    assert not ic.preexisting_exception(fit, [0, 1, 2, 3])   # no baseline, no exemption


class _RmsFit(_Fit):
    def rms_residual(self):
        r = self._resid_all[self._mask]
        return float(np.sqrt(np.mean(r ** 2)))


def _misfit_link(own_scale=1.0, new_off=0.3, new_sig=None, a_scale=0.7):
    """Body A: a messy record (1.19" rms at the default ``a_scale``, 1.5" sigmas) kept whole; one new 4-obs tracklet
    that sits ``new_off`` off the joint orbit, with per-obs sigma ``new_sig`` (None: no sigmas)."""
    oa = [_obs("2014 AA1", 2456700.0 + 3 * n + 0.01 * k) for n in range(4) for k in range(4)]
    ra = [(a_scale * 1.8 * (-1) ** k, a_scale * 1.6 * (-1) ** n) for n in range(4) for k in range(4)]
    new = [_obs("C123456", 2457100.0 + 0.01 * k, code="T08") for k in range(4)]
    rn = [(new_off, -new_off)] * 4
    fit = _Fit(oa + new, ra + rn)
    if new_sig is not None:
        fit._sig_ra = fit._sig_dec = np.array([1.5] * len(oa) + [new_sig] * len(new))
    solo = _RmsFit(oa, [(own_scale * x, own_scale * y) for x, y in ra])
    return _with_solo(fit, {"2014 AA1": solo, "C123456": None})


def test_a_record_that_is_messy_alone_does_not_block():
    note = ic.preexisting_misfit(_misfit_link())
    assert note and "2014 AA1" in note and "alone" in note


def test_a_link_that_makes_the_record_fit_worse_is_still_a_misfit():
    assert ic.preexisting_misfit(_misfit_link(own_scale=0.5)) is None   # 0.6" alone, 1.19" joint


# RMS_CEILING bounds the baseline too: no correct record of the regression fixtures fits worse
# than 0.98", so a record at 1.5" or more alone is itself suspect and excuses nothing.
@pytest.mark.parametrize("a_scale", [1.0, 1.8])                        # 1.70" and 3.06" alone
def test_a_record_past_the_ceiling_is_no_baseline(a_scale):
    assert ic.preexisting_misfit(_misfit_link(a_scale=a_scale)) is None


def test_a_joint_fit_past_the_ceiling_is_no_baseline():
    """1.19" alone may be the record's own, but the link may not take it past 1.5" jointly even
    within the worsen allowance (1.25 x 1.19 + 0.10 = 1.59)."""
    assert ic.preexisting_misfit(_misfit_link(own_scale=1.0 / 1.3)) is not None   # 0.92 -> 1.19
    assert ic.preexisting_misfit(_misfit_link(a_scale=0.9, own_scale=1.0 / 1.2)) is None


def test_new_tracklet_off_the_orbit_is_still_a_misfit():
    assert ic.preexisting_misfit(_misfit_link(new_off=2.0)) is None     # the NEW obs do not fit


# The new tracklet is the thing being judged, so it gets the full kept-fit bar (0.6", or 2 sigma
# under 1.5") on its own residuals -- not RMS_CEILING alone, which only backstops the sigma clause.
@pytest.mark.parametrize("off", [0.7, 1.0, 1.4])
def test_tight_new_tracklet_between_the_bar_and_the_ceiling_is_a_misfit(off):
    assert ic.preexisting_misfit(_misfit_link(new_off=off, new_sig=0.1)) is None   # 7-14 sigma


def test_new_tracklet_without_sigmas_gets_the_arcsec_bar():
    assert ic.preexisting_misfit(_misfit_link(new_off=1.0)) is None


def test_loose_new_tracklet_within_its_own_sigmas_fits():
    assert ic.preexisting_misfit(_misfit_link(new_off=1.0, new_sig=1.0)) is not None   # 1 sigma
    assert ic.preexisting_misfit(_misfit_link(new_off=0.5, new_sig=0.1)) is not None   # under 0.6"


def test_a_record_whose_solo_fit_failed_gets_no_exemption():
    """A second record big enough to fit alone (B: 12 obs, 3 nights) whose solo fit FAILED has no
    baseline, so the link gets no exemption -- even though A's does and B sits under 1.5"."""
    fit = _misfit_link()
    ob, rb = _body_b()
    rb = [(0.9 * (-1) ** k, 0.9) for k in range(len(ob))]          # B at 0.9" jointly
    fit._obs = fit._obs + ob
    fit._resid_all = np.vstack([fit._resid_all, rb])
    fit._mask = np.ones(len(fit._obs), bool)
    fit.__dict__["_solo_fits"]["2015 BB2"] = _RmsFit(ob, rb)
    assert ic.preexisting_misfit(fit) is not None                  # B fits as it does alone
    fit.__dict__["_solo_fits"]["2015 BB2"] = None                  # B alone: error / deadline
    assert ic.preexisting_misfit(fit) is None


def test_preexisting_drops_marks_only_what_the_body_alone_rejects():
    oa, ra = _body_a()
    ob, rb = _body_b()
    mask = np.ones(len(oa) + len(ob), bool)
    mask[[0, 1, 2, 3, 5, len(oa)]] = False          # A's night 1 + one obs of night 2; one of B
    fit = _Fit(oa + ob, ra + rb, mask=mask)
    solo_mask = [i not in (0, 1, 2, 3) for i in range(len(oa))]       # A alone drops night 1 only
    _with_solo(fit, {"2014 AA1": _Fit(oa, ra, mask=solo_mask), "2015 BB2": _Fit(ob, rb)})
    pre = ic.preexisting_drops(fit)
    assert list(np.flatnonzero(pre)) == [0, 1, 2, 3]
    fit.__dict__["_pre_drops"] = pre
    assert list(np.flatnonzero(ic._dropped(fit))) == [5, len(oa)]      # what the LINK dropped


def _raising(exc):
    def fit_alone(fit, sub):
        raise exc
    return fit_alone


def test_a_solo_fit_out_of_its_own_time_is_unanswered(monkeypatch):
    from pyroorb import deadline
    ob, rb = _body_b()
    monkeypatch.setattr(ic, "_fit_alone",
                        _raising(deadline.DeadlineExceeded(ic.SOLO_LABEL, 120.0, 121.0)))
    fit = _Fit(ob, rb)
    assert ic.solo_fit(fit, "2015 BB2") is None
    assert "120 s" in ic.solo_unanswered(fit)["2015 BB2"]


def test_a_solo_fit_that_raises_is_unanswered(monkeypatch):
    ob, rb = _body_b()
    monkeypatch.setattr(ic, "_fit_alone", _raising(np.linalg.LinAlgError("singular")))
    fit = _Fit(ob, rb)
    assert ic.solo_fit(fit, "2015 BB2") is None
    assert ic.solo_unanswered(fit) == {"2015 BB2": "failed (LinAlgError)"}


def test_the_callers_deadline_is_not_the_solo_fits_to_absorb(monkeypatch):
    """A nested deadline takes the stricter, OUTER label: that is the whole check running
    out, and it must reach whoever set it."""
    from pyroorb import deadline
    ob, rb = _body_b()
    monkeypatch.setattr(ic, "_fit_alone",
                        _raising(deadline.DeadlineExceeded("web fit", 60.0, 61.0)))
    fit = _Fit(ob, rb)
    with pytest.raises(deadline.DeadlineExceeded):
        ic.solo_fit(fit, "2015 BB2")


def _verdict(code):
    return {"offer": code != "no", "clean": code == "y", "code": code,
            "submittable": code == "y", "reason": f"the {code} reason",
            "cautions": ["an earlier caution"], "notes": [], "flags": {}}


@pytest.mark.parametrize("code", ["no", "y2", "y-systematic", "y-examine"])
def test_an_unanswered_baseline_never_refuses_a_link(monkeypatch, code):
    """Without a body's solo fit nothing already in its record can be set aside, so a
    refusal or downgrade may be the record's and not the link's: 'unknown' -- offered for
    review, never transmitted -- saying what the check would otherwise have said."""
    monkeypatch.setattr(ic, "_linkage_verdict", lambda fit: _verdict(code))
    fit = _Fit(*_body_b())
    fit.__dict__["_solo_unanswered"] = {"2015 BB2": "ran out of its 120 s"}
    v = ic.linkage_verdict(fit)
    assert v["code"] == "unknown" and v["offer"] and not v["submittable"] and not v["clean"]
    assert "2015 BB2 (ran out of its 120 s)" in v["reason"]
    assert v["cautions"][0] == f"Without that baseline the check would have said '{code}': " \
                               f"the {code} reason"
    assert v["cautions"][1:] == ["an earlier caution"]


def test_a_clean_link_or_an_answered_baseline_keeps_its_verdict(monkeypatch):
    fit = _Fit(*_body_b())
    monkeypatch.setattr(ic, "_linkage_verdict", lambda f: _verdict("y"))
    fit.__dict__["_solo_unanswered"] = {"2015 BB2": "failed (ValueError)"}
    assert ic.linkage_verdict(fit)["code"] == "y"               # setting aside only lifts
    monkeypatch.setattr(ic, "_linkage_verdict", lambda f: _verdict("no"))
    fit.__dict__.pop("_solo_unanswered")
    assert ic.linkage_verdict(fit)["code"] == "no"              # every baseline answered


def test_a_side_is_a_body_not_a_label():
    """Numbered lines under one provisional, under another, and with none -- ADES (number in
    packed_perm beside the label) or OBS80 (labelled by the number) -- are one body."""
    ades_a = {"designation": "2002 AC19", "packed_perm": "f5989"}
    ades_b = {"designation": "2005 UV74", "packed_perm": "f5989"}
    ades_n = {"designation": "415989", "packed_perm": "f5989"}
    obs80 = {"designation": "415989", "packed": "f5989"}
    assert {ic._body_key(o) for o in (ades_a, ades_b, ades_n, obs80)} == {"415989"}
    assert ic._body_key({"designation": "2026 XY1", "packed_perm": None}) == "2026 XY1"


def _alias_link():
    """Body 415989: 3 nights under '2002 AC19' plus a 2-obs tracklet under its alias
    '2005 UV74' that the joint fit drops; body '2026 XY1' a year later, 3 nights."""
    obs, mask = [], []
    for n in range(3):
        for k in range(4):
            o = _obs("2002 AC19", 2456700.0 + 3 * n + 0.01 * k, perm="f5989")
            obs.append(o)
            mask.append(True)
    for k in range(2):
        obs.append(_obs("2005 UV74", 2456712.0 + 0.01 * k, code="691", perm="f5989"))
        mask.append(False)
    for n in range(3):
        for k in range(4):
            obs.append(_obs("2026 XY1", 2457100.0 + 2 * n + 0.01 * k, code="G96"))
            mask.append(True)
    for i, o in enumerate(obs):
        o["ra_deg"], o["dec_deg"] = 10.0 + 0.001 * i, 5.0
    return _PairFit(obs, [(0.05, 0.05)] * len(obs), mask=mask)


class _PairFit(_Fit):
    """What identify()'s gates read beyond _Fit: the observation count and the kept rms."""
    designation = "2002 AC19"

    def n_observations(self):
        return len(self._obs)

    def rms_residual(self):
        return float(np.sqrt(np.mean(self._resid_all[self._mask] ** 2)))


def test_an_alias_labels_dropped_tracklet_leaves_its_body_standing():
    """The alias's two obs are gone, but body 415989 keeps twelve: a side is still there, so
    the link is not refused as 'nothing left of that side'."""
    why = {}
    ic._identify_pass(_alias_link(), _shape=True, _why=why)
    assert "drops every observation" not in (why.get("reason") or "")
    fit = _alias_link()
    fit._mask[:12] = False                             # now the whole BODY is gone
    why = {}
    assert ic._identify_pass(fit, _shape=True, _why=why) is None
    assert "drops every observation of 415989" in why["reason"]


def test_too_few_obs_gets_no_solo_fit():
    ob, rb = _body_b()
    one = [_obs("C123456", 2457101.0 + 0.01 * k, code="T08") for k in range(3)]
    fit = _Fit(ob + one, rb + [(0.0, 0.0)] * 3)
    assert ic.solo_fit(fit, "C123456") is None             # 3 obs, one night: never fitted


# --- engine guard ----------------------------------------------------------------------------
@pytest.mark.slow
def test_243p_clipped_tracklets_are_its_own_records():
    """2025 MT356 = 243P: the joint fit clips 23 whole tracklets, past the 4-tracklet budget --
    but 243P's own record, fitted alone, rejects them too. A pre-existing concern the link does
    not cause never blocks it, so the link is an identification again (a y2: the exceptions that
    remain are still named), and the verdict says what was set aside."""
    from _fixtures import load_fixture, FIX
    import pyroorb.fit_object as fo
    from pyroorb import obs_sources
    fx = load_fixture(os.path.join(FIX, "PAIR_2025MT356.json"))
    fit = fo.fit_orbit([dict(o) for o in fx["obs"]], obs_sources.load_obscodes(),
                       desig=fx["meta"]["designation"], meta=fx["meta"])
    v = ic.linkage_verdict(fit)
    assert v["code"] == "y2", v
    assert any("already in the MPC record" in n for n in v["notes"]), v["notes"]


@pytest.mark.slow
@pytest.mark.parametrize("label", ["PAIR_2014HV328", "PAIR_2014EW228", "PAIR_2021TJ205",
                                   "PAIR_2025NF505"])
def test_known_bogus_pairs_still_trend(label):
    """Known-bogus links must keep firing once each side's own record is the baseline: their
    trend is what the JOIN does, not what either record already shows."""
    from _fixtures import load_fixture, FIX
    import pyroorb.fit_object as fo
    from pyroorb import obs_sources
    fx = load_fixture(os.path.join(FIX, label + ".json"))
    fit = fo.fit_orbit([dict(o) for o in fx["obs"]], obs_sources.load_obscodes(),
                       desig=fx["meta"]["designation"], meta=fx["meta"])
    desc, pre = ic.link_systematic(fit)
    assert desc, f"{label}: a bogus link's trend was set aside as pre-existing ({pre})"
