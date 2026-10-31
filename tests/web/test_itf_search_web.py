"""The web ITF-search endpoint wiring (engine_api.itf_search_web), offline.

Uses the ``_searcher`` test seam so no database or network is touched: a found
linkage must come back as a FRESH full fit payload (session + fit_key + the
``itf`` block the client banners from), a no-find must come back
``{ok, itf.found: false}``, and an expired session must fail gracefully.
Also checks the seed handed to the searcher honours the client's mask and
that the UI path can never trigger an MPC submission.

    python tests/web/test_itf_search_web.py
"""
import pytest
import threading
import os, sys, warnings; warnings.filterwarnings("ignore")
# the submit tests exercise the enabled path; don't depend on the dev .env
os.environ["DIRECT_SUBMIT_ALLOWED"] = "1"     # these tests drive the MPC route
_HERE = os.path.dirname(os.path.abspath(__file__))
import _fixtures as FX
import pyroorb.web.engine_api as E
from pyroorb.itf.search import ItfSearchResult
from pyroorb.itf.verdict import SearchVerdict


def _mpc_submit(session, acked_warnings=None, _submitter=None, **kw):
    """The web submit on its DIRECT-to-MPC route, with a submitter identity -- what
    these tests have always exercised (vetting + transmission through the injected
    ``_submitter``).  The queue route is covered in tests/test_web_submit_queue.py."""
    kw.setdefault("name", "T. Tester")
    kw.setdefault("email", "t@example.org")
    kw.setdefault("route", "mpc")
    return E.itf_submit_web(session, acked_warnings, _submitter, **kw)


# runs the ITF search end-to-end through the web layer
pytestmark = pytest.mark.slow

FIXTURE = "2008_TE4"

# One seed line under the object's packed designation and one linked ITF
# tracklet line -- what a real search hands back in `combined_obs80`, which the
# submit path now filters by the live mask instead of trusting a frozen doc.
_SEED_LINE = ("     " + f"{'K08T04E':<7}" + "  C"
              + f"{'2008 10 07.14066':<17}" + f"{'10 00 00.00':<12}"
              + f"{'+10 00 00.0':<12}" + " " * 9 + " 21.0G" + " " + "     " + "703")
_ITF_LINE = ("     " + f"{'PR00001':<7}" + "  C"
             + f"{'2016 01 02.34567':<17}" + f"{'10 00 00.00':<12}"
             + f"{'+10 00 00.0':<12}" + " " * 9 + " 21.0G" + " " + "     " + "F51")


@pytest.fixture(autouse=True)
def _mpc_nights_ok(request, monkeypatch):
    """These stub searches link ONE seed line to ONE tracklet -- a layout, not an arc -- so
    the MPC's night requirement (``itf.mpcrules.submittable_nights``, which ``_pending_verdict``
    asks first) would refuse every one of them before the mechanics under test are reached.
    Waived here; ``test_the_mpc_night_rule_refuses_a_pending_find`` runs the real one."""
    if request.node.name == "test_the_mpc_night_rule_refuses_a_pending_find":
        return
    import pyroorb.itf.mpcrules as R
    monkeypatch.setattr(R, "submittable_nights", lambda astrometry: (True, ""))


def test_the_mpc_night_rule_refuses_a_pending_find():
    """K26E25U: two W84 tracklets on 2019-06-03, 01:09 and 04:16 UT, beside
    a 4-night 2026 seed, were offered as submittable. Under 12 hours apart they are ONE night,
    and two apparitions of [1, 4] nights fail the MPC's 3+2 rule -- so the live re-judge refuses
    it even when the search's own record says submittable (e.g. the set was narrowed since)."""
    def line(date, stn):
        return ("     " + f"{'K26E25U':<7}" + "  C" + f"{date:<17}" + f"{'13 14 15.33':<12}"
                + f"{'-10 30 37.3':<12}" + " " * 9 + " 23.4i" + " " + "     " + stn)
    seed = [line(f"2026 03 {d}.26991", "X05") for d in (14, 15, 17, 19)]
    itf = [line("2019 06 03.04800", "W84"), line("2019 06 03.17800", "W84")]
    ok, why, possible = E._pending_verdict({}, {"submittable": True}, None, seed + itf,
                                           {"links": {"link_0": {}}})
    assert ok is False and possible is True
    assert "12 hours" in why and "[1, 4]" in why
    # ...the same pair 13 hours apart is two nights, and the search's yes stands
    itf = [line("2019 06 02.40000", "W84"), line("2019 06 02.94200", "W84")]
    ok, _, _ = E._pending_verdict({}, {"submittable": True}, None, seed + itf,
                                  {"links": {"link_0": {}}})
    assert ok is True


def _fake_line(o, i):
    # column-correct OBS80 line (date cols 16-32, station 78-80) -- the seed
    # builder and mask logic only need the layout, not real astrometry
    from datetime import datetime, timedelta
    dt = datetime(1970, 1, 1) + timedelta(days=o["jd_utc"] - 2440587.5)
    date = f"{dt.year:04d} {dt.month:02d} {dt.day + (dt.hour/24 + dt.minute/1440 + dt.second/86400):08.5f}"
    line = ("     ZZ{:05d}".format(i) + "  C" + f"{date:<17}"
            + f"{'10 00 00.00':<12}" + f"{'+10 00 00.0':<12}" + " " * 9
            + " 21.0G" + " " + "     " + f"{o['obscode']:>3}")
    assert len(line) == 80
    return line


def _session(with_lines=True):
    fx = FX.load_fixture(os.path.join(FX.FIX, FIXTURE + ".json"))
    payload = E.fit_obs_list(fx["obs"], fx["meta"], E._obscodes(),
                             force=fx.get("force", False))
    sid = payload.get("session")
    assert sid, f"fit failed: {payload.get('error')}"
    if with_lines:
        # ADES-sourced fits carry no obs80_line (the endpoint would fall back
        # to designation seeding); graft lines so the line-seed path is tested
        for i, o in enumerate(E._SESSIONS[sid]["fit"]._obs):
            o["obs80_line"] = _fake_line(o, i)
    return sid, payload


def test_found_returns_full_fit_payload():
    sid, base = _session()
    fit = E.session_fit(sid)
    seen = {}

    def searcher(seed_obs80, primaries, cfg):
        seen["seed"] = seed_obs80
        seen["primaries"] = primaries
        seen["cfg"] = cfg
        itf_line = ("     " + f"{'PR00001':<7}" + "  C"
                    + f"{'2016 01 02.34567':<17}" + f"{'10 00 00.00':<12}"
                    + f"{'+10 00 00.0':<12}" + " " * 9 + " 21.0G" + " "
                    + "     " + "F51")
        assert len(itf_line) == 80
        return ItfSearchResult(verdict=SearchVerdict.LINKED, reason="test", fit=fit,
                               itf_obs80=itf_line + "\n",
                               linkage={"links": {}},
                               telemetry={"obs_count": 2})

    p = E.itf_search_web(sid, {"rank_mode": "certainty"}, _searcher=searcher)
    assert p.get("ok"), p.get("error")
    assert p["itf"]["found"] is True
    assert p["itf"]["added_obs"] == 2
    assert p["itf"]["tracklets"] == [{"trksub": "PR00001",
                                      "date": "2016 01 02", "stn": "F51",
                                      "n_obs": 1}]
    # a fresh, renderable session payload (same shape run_fit produces)
    assert p.get("session") and p["session"] != sid
    assert p.get("fit_key")
    assert "elements" in p and "summary" in p and "residuals" in p
    # options threaded through; the seed carried the session's OBS80 lines
    assert seen["cfg"].ranking_mode == "certainty"
    assert seen["cfg"].submit is False          # the UI can never submit
    n_lines = len([l for l in seen["seed"].splitlines() if l.strip()])
    assert n_lines >= 3
    print(f"PASS found: payload session={p['session']} seed_lines={n_lines}")


def test_mask_filters_the_seed():
    sid, base = _session()
    n = len(E._SESSIONS[sid]["fit"]._obs)
    lens = {}

    def searcher(seed_obs80, primaries, cfg):
        lens["n"] = len([l for l in seed_obs80.splitlines() if l.strip()])
        return ItfSearchResult(verdict=SearchVerdict.NONE, reason="no likely matches")

    mask = [True] * n
    mask[0] = False                              # exclude one obs client-side
    E.itf_search_web(sid, {}, mask_list=mask, _searcher=searcher)
    with_mask = lens["n"]
    E.itf_search_web(sid, {}, _searcher=searcher)
    without = lens["n"]
    assert without - with_mask == 1, (with_mask, without)
    print(f"PASS mask: {without} -> {with_mask} seed lines")


def test_alias_family_seeds_primary_only():
    # 2016 UH221-class case: the session's obs carry TWO designations (an
    # existing MPC identification family). The seed/submission identity must
    # be the single family primary -- both aliases would fetch the family
    # twice (a duplicate-poisoned seed).
    sid, _ = _session()
    obs = E._SESSIONS[sid]["fit"]._obs
    half = len(obs) // 2
    for o in obs[half:]:
        # a legal designation that sorts after the primary: 'Y' is a half-month
        # letter, 'Z' is the legal 25th order letter.  'ZZ' would be neither --
        # 'Z' is not a half-month at all, so packing.pack refuses it.
        o["designation"] = "2099 YZ99"
    seen = {}

    def searcher(seed_obs80, primaries, cfg):
        seen["primaries"] = primaries
        return ItfSearchResult(verdict=SearchVerdict.NONE, reason="no likely matches")

    E.itf_search_web(sid, {}, _searcher=searcher)
    assert seen["primaries"] == [obs[0]["designation"]], seen["primaries"]
    print(f"PASS alias family -> primary only: {seen['primaries']}")


def test_user_assembled_pair_keeps_both_designations():
    # "K13M06C = K21C05V" typed into the search box: the identification the USER
    # proposed. BOTH designations must reach the search and the submission document
    # -- collapsing to the alphabetically-earliest primary (right for a one-name
    # fetch, whose extra designations are acknowledged aliases) once had the ITF
    # document naming one half of a linkage the user had already made. An
    # acknowledged alias riding along with either of them still stays out.
    sid, _ = _session()
    fit = E._SESSIONS[sid]["fit"]
    obs = fit._obs
    third = len(obs) // 3
    for o in obs[third:2 * third]:
        o["designation"] = "2021 CV5"           # the second REQUESTED designation
    for o in obs[2 * third:]:
        o["designation"] = "2099 YZ99"          # an acknowledged alias, not requested
    fit.n_requested = 2
    fit.requested_designations = [obs[0]["designation"], "2021 CV5"]
    from pyroorb.packing import pack, unpack
    for o in obs:                               # file each line under its own designation,
        line = o["obs80_line"]                  # so the document collects real ids
        o["obs80_line"] = "     " + f"{pack(o['designation']):<7}" + line[12:]
    seen = {}

    def searcher(seed_obs80, primaries, cfg):
        seen["primaries"] = primaries
        return ItfSearchResult(verdict=SearchVerdict.NONE, reason="no likely matches")

    E.itf_search_web(sid, {}, _searcher=searcher)
    assert seen["primaries"] == [obs[0]["designation"], "2021 CV5"], seen["primaries"]

    # ...and the document the "Submit linkage" button would send names both too
    from pyroorb.itf.linkage import requested_primaries
    assert requested_primaries(fit) == [obs[0]["designation"], "2021 CV5"]
    doc, err = E._itf_live_linkage(E._SESSIONS[sid])
    assert doc is not None, err
    named = doc["links"]["link_0"].get("designations") or []

    assert sorted(unpack(p) or p for p in named) == sorted(
        [obs[0]["designation"], "2021 CV5"]), named
    print(f"PASS user-assembled pair keeps both: {seen['primaries']} -> {named}")


def test_submit_uses_server_held_linkage():
    # a found search stashes the linkage on the NEW session; the submit
    # endpoint transmits THAT (client sends no document) and is one-shot
    sid, _ = _session()
    fit = E.session_fit(sid)

    def searcher(seed_obs80, primaries, cfg):
        return ItfSearchResult(verdict=SearchVerdict.LINKED, reason="ok", fit=fit,
                               itf_obs80=_ITF_LINE + "\n",
                               combined_obs80=_SEED_LINE + "\n" + _ITF_LINE + "\n",
                               linkage={"links": {"link_0": {
                                   "designations": ["K16C37L"]}}},
                               telemetry={"obs_count": 1})

    found = E.itf_search_web(sid, {}, _searcher=searcher)
    new_sid = found["session"]
    assert found["itf"]["submittable"] is True
    assert E._SESSIONS[new_sid]["itf_linkage"] is not None

    sent = {}
    def submitter(doc, cfg, log):
        sent["doc"] = doc
        sent["armed"] = cfg.submit
        log("MPC submission: 200 OK")
        return True, "200 OK", "1 identification received"

    r = _mpc_submit(new_sid, _submitter=submitter)
    assert r["ok"] and r["accepted"] and r["status"] == "200 OK"
    assert sent["armed"] is True                          # cfg.submit armed
    # the document is BUILT FROM THE LIVE SET, not the searcher's frozen one:
    # the seed's designation plus the linked tracklet
    link0 = sent["doc"]["links"]["link_0"]
    assert "K08T04E" in link0.get("designations", []), link0
    assert [t[0] for t in link0.get("trksubs", [])] == ["PR00001"], link0
    # one-shot: linkage cleared, a second submit refuses
    assert E._SESSIONS[new_sid]["itf_linkage"] is None
    r2 = _mpc_submit(new_sid, _submitter=submitter)
    assert not r2["ok"] and r2.get("expired")
    # a non-200 does NOT clear the pending linkage (retryable)
    found2 = E.itf_search_web(sid, {}, _searcher=searcher)
    sid2 = found2["session"]
    r3 = _mpc_submit(sid2, _submitter=lambda d, c, log: (True, "500 Server Error", None))
    assert r3["ok"] and not r3["accepted"]
    assert E._SESSIONS[sid2]["itf_linkage"] is not None
    print("PASS submit uses server-held linkage (one-shot, retryable on non-200)")


def test_submit_requires_warning_acks():
    # a linkage that carries quality cautions must not be submittable unless
    # the client echoes back EVERY stored warning (the browser confirm() loop
    # is not the gate -- the server is)
    sid, _ = _session()
    fit = E.session_fit(sid)
    warn = "the object's own observations fit worse after the link"

    def searcher(seed_obs80, primaries, cfg):
        r = ItfSearchResult(verdict=SearchVerdict.LINKED, reason="ok", fit=fit,
                            itf_obs80=_ITF_LINE + "\n",
                            combined_obs80=_SEED_LINE + "\n" + _ITF_LINE + "\n",
                            linkage={"links": {}},
                            telemetry={"obs_count": 0})
        r.warnings.append(warn)
        return r

    found = E.itf_search_web(sid, {}, _searcher=searcher)
    new_sid = found["session"]
    assert found["itf"]["warnings"] == [warn]

    ok_sub = lambda d, c, log: (True, "200 OK", None)
    # no acks / partial acks -> refused, linkage stays pending
    r = _mpc_submit(new_sid, _submitter=ok_sub)
    assert not r["ok"] and r["warnings"] == [warn]
    r = _mpc_submit(new_sid, acked_warnings=["something else"],
                         _submitter=ok_sub)
    assert not r["ok"]
    assert E._SESSIONS[new_sid]["itf_linkage"] is not None
    # full acks -> submits
    r = _mpc_submit(new_sid, acked_warnings=[warn], _submitter=ok_sub)
    assert r["ok"] and r["accepted"]
    print("PASS submit refuses unacknowledged quality cautions")


def test_ades_survives_the_linkage_refit():
    """A found linkage hands back a fit of ADES OBSERVATIONS, not of their
    80-column shadow.

    The search's currency is lines, so the combined arc is re-parsed from
    ``combined_obs80`` -- and that round trip used to strip the session's own
    published observations of everything ADES carries (obsID, trkSub, the
    reported rmsRA/rmsDec and the weighting sigma that follows from them), so
    after an ITF search the ORIGINAL object came back blank in the "addl info"
    pane and weighted by the station model.  Now each seed ob is restored from
    the dict the session already held (itf/obsreg.py).

    ``combined_obs80`` here is the seed arc alone: what is under test is the
    restore, and a fabricated ITF line would only make the refit fail and fall
    back to the search's own fit, testing nothing."""
    sid, _ = _session()
    fit = E.session_fit(sid)
    for i, o in enumerate(fit._obs):
        o["obs_id"] = f"OID{i:04d}"
        o["trksub"] = "PR00001"
        o["ades_rms_ra"], o["ades_rms_dec"] = 0.12, 0.14
        o["sigma_ra"], o["sigma_dec"] = 0.12, 0.14
        o["uncert_src"] = "ADES"
    combined = "\n".join(o["obs80_line"] for o in fit._obs) + "\n"

    def searcher(seed_obs80, primaries, cfg):
        return ItfSearchResult(verdict=SearchVerdict.LINKED, reason="ok", fit=fit,
                               itf_obs80=_ITF_LINE + "\n",
                               combined_obs80=combined,
                               linkage={"links": {}},
                               telemetry={"obs_count": 1})

    p = E.itf_search_web(sid, {}, _searcher=searcher)
    assert p.get("ok"), p.get("error")
    assert p["session"] != sid                    # a genuinely fresh refit
    rows = p["residuals"]
    with_id = [r for r in rows if r.get("obs_id")]
    with_rms = [r for r in rows if r.get("ades_rms_ra") is not None]
    assert len(with_id) == len(rows), f"{len(with_id)}/{len(rows)} kept obsID"
    assert len(with_rms) == len(rows), f"{len(with_rms)}/{len(rows)} kept rmsRA"
    assert all(r.get("trksub") == "PR00001" for r in rows)
    # ...and the reported sigma reached the FIT, not just the display: every
    # observation the refit was handed carries it.  Asserted on the fit's own
    # obs rather than on the residual rows' sigma, because the engine may
    # legitimately scale a residual sigma afterwards (over-observation
    # de-weighting, link relaxation) -- what is under test is the input.
    new_obs = E.session_fit(p["session"])._obs
    assert all(abs(o["sigma_ra"] - 0.12) < 1e-9 for o in new_obs)
    assert all(o.get("uncert_src") == "ADES" for o in new_obs)
    print(f"PASS ADES survives the linkage refit: {len(rows)} obs kept "
          f"obsID/trkSub/rmsRA and were weighted by it")


def test_pending_linkage_survives_a_refit_and_follows_the_mask():
    """(a) The Submit button must not vanish because you looked at the linkage.
    A Full step re-solves the SAME observations -- the pending linkage is not
    stale -- so every interactive payload carries the itf block again.

    (b) The submission follows the SCREEN.  Switch off every observation of a
    linked tracklet and that tracklet leaves the document; switch off enough
    that fewer than two ids remain and there is no linkage left to send."""
    sid, _ = _session()
    fit = E.session_fit(sid)

    def searcher(seed_obs80, primaries, cfg):
        return ItfSearchResult(verdict=SearchVerdict.LINKED, reason="ok", fit=fit,
                               itf_obs80=_ITF_LINE + "\n",
                               combined_obs80=_SEED_LINE + "\n" + _ITF_LINE + "\n",
                               linkage={"links": {"link_0": {
                                   "designations": ["K16C37L"]}}},
                               telemetry={"obs_count": 1})

    found = E.itf_search_web(sid, {}, _searcher=searcher)
    new_sid = found["session"]
    s = E._SESSIONS[new_sid]

    # (a) an interactive re-fit still advertises the linkage
    block = E._itf_pending_block(s)
    assert block and block["submittable"] and block["pending"], block
    assert [t["trksub"] for t in block["tracklets"]] == ["PR00001"], block

    # (b) switch the linked tracklet OFF -> it leaves the document, and with
    # only the seed id left there is no linkage to submit
    itf_id = s["itf_pending"]["itf_ids"][0]
    for o, keep in zip(s["fit"]._obs, s["fit"].is_included()):
        line = o.get("obs80_line")
        if line and E.obs_unique_id(line) == itf_id:
            raise AssertionError("fixture fit should not carry the ITF line")
    # simulate the user's toggle by marking it excluded in the pending filter:
    # the fit does not carry it, so add it as a KNOWN-but-excluded observation
    class _MaskedFit:
        def __init__(self, base, drop):
            self._obs = list(base._obs) + [{"obs80_line": _ITF_LINE}]
            self._inc = list(base.is_included()) + [False]
        def is_included(self):
            return list(self._inc)
    s["fit"] = _MaskedFit(fit, itf_id)
    doc, err = E._itf_live_linkage(s)
    assert doc is None and "not a linkage" in (err or ""), (doc, err)
    r = _mpc_submit(new_sid, _submitter=lambda d, c, log: (True, "200 OK", None))
    assert not r["ok"] and "no longer a linkage" in r["error"], r
    print("PASS pending linkage survives a re-fit and follows the obs mask")


def test_diagnostics_survive_a_refit_like_the_warnings_do():
    """The trend is a DIAGNOSTIC, not a gate (27723c5): it is reported everywhere and acted
    on by nothing. But `itf_pending` stored only `warnings` and `vetting`, and the rebuilt
    block emitted no `diagnostics` key at all -- so the info banner survived exactly until
    the user took a Full step, and the whole point of the diagnostic channel is that it
    stays visible. Nothing downstream can recompute it: it is the search's measurement."""
    sid, _ = _session()
    fit = E.session_fit(sid)

    def searcher(seed_obs80, primaries, cfg):
        return ItfSearchResult(verdict=SearchVerdict.LINKED, reason="ok", fit=fit,
                               itf_obs80=_ITF_LINE + "\n",
                               combined_obs80=_SEED_LINE + "\n" + _ITF_LINE + "\n",
                               linkage={"links": {"link_0": {
                                   "designations": ["K16C37L"]}}},
                               diagnostics=["residual ramp on PR00001: 7.7x the arc"],
                               telemetry={"obs_count": 1})

    found = E.itf_search_web(sid, {}, _searcher=searcher)
    assert found["itf"]["diagnostics"], found["itf"]
    # ...and a diagnostic is never a submit gate, so it must not join the warnings
    assert not found["itf"]["warnings"], found["itf"]

    s = E._SESSIONS[found["session"]]
    block = E._itf_pending_block(s)
    assert block["diagnostics"] == found["itf"]["diagnostics"], block
    assert not block["warnings"], block
    print("PASS ITF diagnostics survive a re-fit")


def test_faint_limit_override_reaches_the_config():
    """The options modal's faint cut must land on the ItfConfig the search runs
    with -- and an untyped field must leave the engine default alone, since
    mag_limit lives ON the config rather than in the itf_search signature."""
    from pyroorb.itf.config import ItfConfig
    sid, base = _session()
    seen = {}

    def searcher(seed_obs80, primaries, cfg):
        seen["cfg"] = cfg
        return ItfSearchResult(verdict=SearchVerdict.NONE, reason="none", telemetry={})

    # typed: the override rides through
    E.itf_search_web(sid, {"mag_limit": 22.8}, _searcher=searcher)
    assert seen["cfg"].mag_limit == 22.8, seen["cfg"].mag_limit
    assert seen["cfg"].mag_limit_explicit is True, "a typed cut is used exactly"

    # untyped: the engine's own default stands (the uncert_mult staleness trap),
    # free to be deepened per object
    E.itf_search_web(sid, {}, _searcher=searcher)
    assert seen["cfg"].mag_limit == ItfConfig().mag_limit
    assert seen["cfg"].mag_limit_explicit is False

    # the UI has no way to arm the dormant deeper-retry, and must not stumble
    # into one: `is False`, not `== False`, because the old float-valued knob
    # would have satisfied `== 0.0` while meaning something quite different
    assert seen["cfg"].mag_limit_fallback is False
    print(f"PASS faint limit: override 22.8, default {ItfConfig().mag_limit}")


def test_db_unreachable_is_a_friendly_error():
    # without the _searcher seam the endpoint goes through itf_search_live;
    # a connect failure must come back as a clear error, not a raw traceback
    import pyroorb.itf.search as S
    sid, _ = _session()
    orig = S.itf_search_live
    def boom(*a, **k):
        raise S.ItfDbUnreachable("connection refused")
    S.itf_search_live = boom
    try:
        p = E.itf_search_web(sid, {})
    finally:
        S.itf_search_live = orig
    assert not p["ok"] and "ITF database unreachable" in p["error"]
    print("PASS db-unreachable -> friendly error")


def test_not_found_and_expired():
    sid, _ = _session()

    def searcher(seed_obs80, primaries, cfg):
        return ItfSearchResult(verdict=SearchVerdict.NONE, reason="no likely matches")

    p = E.itf_search_web(sid, {}, _searcher=searcher)
    assert p["ok"] and p["itf"]["found"] is False
    assert p["itf"]["reason"] == "no likely matches"

    p2 = E.itf_search_web("s999999", {}, _searcher=searcher)
    assert not p2.get("ok") and p2.get("expired")
    print("PASS not-found + expired")


def test_two_results_come_back_as_two_whole_payloads():
    """A search holding a submittable linkage AND a fuller unconfirmed 0.6
    (ItfSearchResult.alternative) answers with both, each a whole fit on its own
    session, so the page can ask which to show; one result has no alternative."""
    sid, _ = _session()
    fit = E.session_fit(sid)
    itf2 = _ITF_LINE.replace("PR00001", "PR00002").replace("2016 01 02", "2017 01 02")

    def both(seed_obs80, primaries, cfg):
        linked = ItfSearchResult(verdict=SearchVerdict.LINKED, reason="test", fit=fit,
                                 itf_obs80=_ITF_LINE + "\n", linkage={"links": {}},
                                 telemetry={"obs_count": 1})
        full = ItfSearchResult(verdict=SearchVerdict.POSSIBLE, reason="1.75 opps",
                               fit=fit, itf_obs80=_ITF_LINE + "\n" + itf2 + "\n",
                               telemetry={"obs_count": 2})
        full.alternative = linked
        return full

    p = E.itf_search_web(sid, {}, _searcher=both)
    assert p["itf"]["found"] and p["itf"]["possible"] and not p["itf"]["submittable"]
    alt = p["itf"]["alternative"]
    assert alt["itf"]["found"] and alt["itf"]["submittable"]
    assert "alternative" not in alt["itf"]
    assert alt["session"] and alt["session"] not in (p["session"], sid)
    assert "elements" in alt and "residuals" in alt
    # each session holds its own result: only the linkage's can be submitted
    assert E._SESSIONS[alt["session"]]["itf_pending"]["submittable"] is True
    assert E._SESSIONS[p["session"]]["itf_pending"]["submittable"] is False
    assert [t["trksub"] for t in p["itf"]["tracklets"]] == ["PR00001", "PR00002"]

    def one(seed_obs80, primaries, cfg):
        return ItfSearchResult(verdict=SearchVerdict.LINKED, reason="test", fit=fit,
                               itf_obs80=_ITF_LINE + "\n", linkage={"links": {}},
                               telemetry={"obs_count": 1})

    assert "alternative" not in E.itf_search_web(sid, {}, _searcher=one)["itf"]


def test_payload_speaks_english_not_status_codes():
    """The engine's comparison shorthands (opps 1.5/1.75, status 0/0.6/1) are
    unreadable, so nothing the browser is handed may carry them: the block
    reports an ``outcome`` sentence, and the reason/warnings/trace are
    rewritten (src/pyroorb/web/itf_wording.py)."""
    sid, _ = _session()
    fit = E.session_fit(sid)

    def nothing(seed_obs80, primaries, cfg):
        return ItfSearchResult(verdict=SearchVerdict.NONE,
                               reason="1.5 opps without progress, abandoning")

    p = E.itf_search_web(sid, {}, _searcher=nothing)
    itf = p["itf"]
    assert itf["found"] is False and "status" not in itf
    assert itf["outcome"] == "no linkage"
    assert "1.5" not in itf["reason"], itf["reason"]
    assert "single-night" in itf["reason"], itf["reason"]

    def possible(seed_obs80, primaries, cfg):
        return ItfSearchResult(
            verdict=SearchVerdict.POSSIBLE, reason="1.75 opps -- too fragile to submit",
            fit=fit, itf_obs80=_ITF_LINE + "\n",
            warnings=["the linkage reaches 1.75 oppositions -- review by hand"],
            vetting={"opps_pre": 1, "opps_post": 1.75},
            telemetry={"obs_count": 1})

    p2 = E.itf_search_web(sid, {}, _searcher=possible)
    itf2 = p2["itf"]
    assert itf2["found"] and itf2["possible"] and "status" not in itf2
    assert itf2["outcome"] == "possible linkage (not submittable)"
    for text in [itf2["reason"], itf2["warnings"][0], itf2["vetting"]["opps_post"]]:
        assert "1.75" not in text, text
    # the warning the client acks at submit time is the one it was shown
    assert E._SESSIONS[p2["session"]]["itf_warnings"] == itf2["warnings"]
    print("PASS payload carries sentences, not 1.5/1.75/0.6")


def test_background_job_polls_out_the_payload(monkeypatch):
    """The hosted path: POST returns a job handle at once (no proxy request
    limit can reach a search that nothing is waiting on) and GET /api/itf_result
    hands the payload over when the worker thread finishes."""
    import time as _t
    # This is about the JOB TABLE, not about the box.  Under `pytest -n 4` four
    # workers hold ephemerides at once and the real capacity verdict can come
    # back `unusable`, at which point admission refuses the search and the test
    # fails for a reason it is not testing (see the note in test_itf_capacity).
    monkeypatch.setattr(E, "_itf_capacity_block", lambda: None)
    sid, _ = _session()
    gate = threading.Event()

    def searcher(seed_obs80, primaries, cfg):
        gate.wait(20)                    # hold the "search" open like a real one
        return ItfSearchResult(verdict=SearchVerdict.NONE, reason="no likely matches")

    started = E.itf_search_start(sid, {"trace_id": "trJOB1"},
                                 _searcher=searcher)
    assert started["ok"] and started["job"] == "trJOB1"
    # ...and it is genuinely still running: the POST did not block on it
    mid = E.itf_search_status("trJOB1")
    assert mid["ok"] and mid["done"] is False and mid["elapsed"] >= 0
    # a repeat start on a running job is the same job, not a second search
    assert E.itf_search_start(sid, {"trace_id": "trJOB1"},
                              _searcher=searcher)["job"] == "trJOB1"
    gate.set()
    for _ in range(200):
        got = E.itf_search_status("trJOB1")
        if got.get("done"):
            break
        _t.sleep(0.05)
    assert got["ok"] and got["done"] is True
    assert got["result"]["ok"] and got["result"]["itf"]["found"] is False
    assert got["result"]["itf"]["reason"] == "no likely matches"
    # an id the server no longer knows tells the client to stop polling
    dead = E.itf_search_status("trNOSUCH")
    assert not dead["ok"] and dead["expired"]
    print("PASS background job: handle -> poll -> payload")


def test_background_job_reports_a_crashed_search():
    """A search that raises must still ANSWER the poller -- otherwise the client
    polls a job that will never finish."""
    import time as _t
    sid, _ = _session()

    def searcher(seed_obs80, primaries, cfg):
        raise RuntimeError("boom")

    E.itf_search_start(sid, {"trace_id": "trJOB2"}, _searcher=searcher)
    for _ in range(200):
        got = E.itf_search_status("trJOB2")
        if got.get("done"):
            break
        _t.sleep(0.05)
    assert got["done"] and got["result"]["ok"] is False
    assert "boom" in got["result"]["error"]
    print("PASS background job: a crashed search still answers")


def test_eviction_takes_a_collected_result_before_an_uncollected_one():
    """Pressure on the job table must fall on payloads the client already has.
    A finished result nobody has polled out yet is a search the user waited
    minutes for and has never seen -- it goes last, after every collected one."""
    import time as _t
    sid, _ = _session()

    def searcher(seed_obs80, primaries, cfg):
        return ItfSearchResult(verdict=SearchVerdict.NONE, reason="no likely matches")

    def _run_to_completion(job, collect):
        """Finish a job, optionally polling its payload out. Waiting is done by
        INSPECTING the table, not by polling: a poll is what marks a job
        collected, which is the very thing under test here."""
        E.itf_search_start(sid, {"trace_id": job}, _searcher=searcher)
        for _ in range(200):
            if E._ITF_JOBS[job]["done"]:
                break
            _t.sleep(0.05)
        else:
            raise AssertionError(f"{job} never finished")
        if collect:
            assert E.itf_search_status(job)["done"]

    saved = dict(E._ITF_JOBS)
    try:
        E._ITF_JOBS.clear()
        # the unseen result is the OLDEST job in the table -- so a purely
        # age-ordered eviction would take it, and only the collected-first rule
        # saves it. (That ordering is what makes this test discriminating.)
        _run_to_completion("trUNCOLLECTED", collect=False)
        for n in range(E._ITF_JOBS_MAX - 2):
            _run_to_completion(f"trFILL{n}", collect=True)
        _run_to_completion("trCOLLECTED", collect=True)
        assert len(E._ITF_JOBS) == E._ITF_JOBS_MAX
        E.itf_search_start(sid, {"trace_id": "trNEW"}, _searcher=searcher)

        # the oldest COLLECTED payload is what went...
        assert E.itf_search_status("trFILL0").get("expired")
        # ...and the older, unseen one is still there, still handing over its
        # result -- the user never saw it, so it is the last thing to drop
        got = E.itf_search_status("trUNCOLLECTED")
        assert got["ok"] and got["done"]
        assert got["result"]["itf"]["reason"] == "no likely matches"
    finally:
        E._ITF_JOBS.clear()
        E._ITF_JOBS.update(saved)
    print("PASS background job: eviction spares an uncollected result")


def _pair_session(label, neocp_desig=None):
    """A web session on a two-designation PAIR fixture -- what the user gets by typing two
    designations into the search box. ``neocp_desig`` marks that side's lines as NEOCP-referenced
    (cols 73-77), which is how a real NEOCP object arrives."""
    fx = FX.load_fixture(os.path.join(FX.FIX, label + ".json"))
    payload = E.fit_obs_list(fx["obs"], fx["meta"], E._obscodes(),
                             force=fx.get("force", False))
    sid = payload.get("session")
    assert sid, f"fit failed: {payload.get('error')}"
    if neocp_desig:
        for o in E._SESSIONS[sid]["fit"]._obs:
            if o.get("designation") != neocp_desig:
                continue
            # an NEOCP tracklet is filed under a temporary trksub, not a packed designation
            line = ("     " + f"{'OOn0024':<7}" + "  C" + f"{'2026 07 19.77091':<17}"
                    + f"{'23 28 12.15':<12}" + f"{'+02 07 04.3':<12}" + " " * 9
                    + " 20.9P" + " " + "NEOCP" + "O18")
            assert len(line) == 80, len(line)
            o["obs80_line"] = line
        payload = E._apply_interactive(E._SESSIONS[sid], sid, None, single=True) or payload
    return sid, payload


def test_trace_feed_relevels_a_running_search():
    """The trace viewer's "Detail" control re-levels the search that is RUNNING,
    through the feed poll (GET /api/itf_trace&level=).  It used to be the only
    place the level could be picked at all, and it was read once at launch -- so
    moving it while watching a trace did nothing.  The STARTING level now lives
    in the search-options modal; this one must genuinely move the live
    threshold, and must clamp a junk value rather than let it reach the
    search."""
    from pyroorb.itf.trace import Tracer, L_STORY, L_DETAIL, L_HEAD, L_FIREHOSE

    ent = E._trace_register("tr-test-relevel")
    try:
        tracer = Tracer(lambda line: E._trace_append(ent, line), L_STORY)
        ent["tracer"] = tracer

        # a detail line is dropped at the level the search started on...
        tracer("trial detail", L_DETAIL)
        assert E.itf_trace_feed("tr-test-relevel")["lines"] == []

        # ...and is written once the viewer asks for more detail
        p = E.itf_trace_feed("tr-test-relevel", 0, 3)
        assert p["level"] == L_DETAIL, p
        tracer("trial detail", L_DETAIL)
        p = E.itf_trace_feed("tr-test-relevel", 0)
        assert len(p["lines"]) == 1 and "trial detail" in p["lines"][0], p

        # junk and out-of-range levels are clamped, never fatal
        for bad, want in [("nonsense", L_DETAIL), (99, L_FIREHOSE), (-4, L_HEAD)]:
            assert E.itf_trace_feed("tr-test-relevel", 0, bad)["level"] == want
    finally:
        E._TRACES.pop("tr-test-relevel", None)

def test_multi_designation_fit_offers_the_identification():
    # The point of the feature: a joint fit the USER assembled is a proposed identification, and
    # the payload must offer it without any ITF search having run.
    sid, p = _pair_session("PAIR_2002QT68")
    itf = p.get("itf")
    assert itf, "a two-designation fit must carry a linkage block"
    assert itf["kind"] == "ident" and itf["found"] and itf["submittable"], itf
    assert itf["clean"] is True, itf["reason"]
    link = itf["linkage"]["links"]["link_0"]
    n_ids = len(link.get("designations") or []) + len(link.get("trksubs") or [])
    assert n_ids >= 2, link
    assert itf["warnings"] == [], itf["warnings"]
    print("PASS multi-designation fit offers the identification")


def test_identification_the_engine_rejects_is_not_offered():
    # 2003 QC91 = 2006 QX180 links only by gutting one side's baseline (60% retention, under
    # MPC_MIN_RETENTION) -- the batch files it as a clear 'no'. The UI must not offer to submit
    # what --ident-check would throw out: same rule, one place.
    sid, p = _pair_session("PAIR_2003QC91")
    itf = p.get("itf") or {}
    assert not itf.get("found"), itf
    assert not itf.get("submittable"), itf
    assert "retention" in (itf.get("reason") or ""), itf.get("reason")
    # ...and the submit endpoint refuses it too, not just the button
    r = _mpc_submit(sid, _submitter=lambda d, c, log: (True, "200 OK", None))
    assert not r["ok"] and r.get("expired"), r
    print("PASS a rejected identification is neither offered nor submittable")


def test_neocp_identification_submits_the_server_held_document():
    # J99T43V + OOn0024, the target shape: a published designation and an NEOCP
    # object. The document must file the NEOCP id under trksubs, stamp identification_type
    # "neocp", and go out through the same vetted server-held path as an ITF find.
    fx_desigs = None
    sid, p = _pair_session("PAIR_2002QT68")
    desigs = sorted({o.get("designation") for o in E._SESSIONS[sid]["fit"]._obs})
    sid, p = _pair_session("PAIR_2002QT68", neocp_desig=desigs[-1])
    itf = p.get("itf")
    assert itf and itf["found"] and itf["neocp"] is True, itf
    link = itf["linkage"]["links"]["link_0"]
    assert [t[0] for t in link.get("trksubs") or []] == ["OOn0024"], link
    assert link.get("identification_type") == "neocp", link

    sent = {}
    def submitter(doc, cfg, log):
        sent["doc"] = doc
        sent["armed"] = cfg.submit
        return True, "200 OK", "1 identification received"

    r = _mpc_submit(sid, acked_warnings=itf["warnings"], _submitter=submitter)
    assert r["ok"] and r["accepted"], r
    assert sent["armed"] is True                      # the outward switch is armed server-side
    assert sent["doc"]["links"]["link_0"]["identification_type"] == "neocp", sent["doc"]
    # one-shot: the same identification is not offered or sent twice
    p2 = E._apply_interactive(E._SESSIONS[sid], sid, None, single=True)
    assert not (p2.get("itf") or {}).get("found"), p2.get("itf")
    r2 = _mpc_submit(sid, _submitter=submitter)
    assert not r2["ok"], r2
    print("PASS NEOCP identification submits the server-held document (one-shot)")


def test_ident_submission_requires_its_cautions_acked():
    # An ambiguous ('y-examine') identification carries its reason as a caution, and the server --
    # not the browser confirm() -- is what refuses an unacknowledged one.
    #
    # The fixture has to be one the ident rule currently calls y-examine, and that is a MOVING
    # target: this test was written on PAIR_2015XO232, which the clip-a-whole-night rule later
    # reclassified as a flat 'no' (it only joins by clipping a night 4.3" off, part of its own
    # coherent short arc), so it stopped being offered at all and the test failed for want of an
    # ambiguous pair rather than for any fault in the acking. If that happens again, do not fight
    # the verdict -- pick another y-examine pair: `tests/pair_shapes.py` prints identify() /
    # examine_or_no() for every PAIR_ fixture. (It happened a second time, 2026-08-19: PAIR_2022MT4
    # was never ambiguous at all, it is a 'y2' excepting one 3-obs B74 tracklet -- it only READ as
    # y-examine because identify() renders exceptions as verbatim OBS80 records and an ADES fixture
    # has none. Now that linkage_verdict judges the SHAPE, it classifies correctly, is no longer
    # transmissible, and carries no caution to ack.) A third time, 2026-09-13: PAIR_2013CK176 was
    # refused only because one small night lost most of its observations inside an apparition that
    # stands without them; identify() now asks whether that clipped is load-bearing, and CK176 is the
    # 'y2' its build_fixtures entry always said it was. PAIR_2025MT356 (clips 23 whole tracklets)
    # is, as of that change, the ONLY y-examine left in the PAIR set. A FOURTH time,
    # 2026-09-17 -- and this one ended the hunt, because the well is now DRY. PAIR_2025MT356
    # pairs 2025 MT356 with COMET 243P (= P/2003 S2 = P/2010 P5), and an identification naming
    # a comet is refused outright and sent to the MPC's Jira route instead
    # (itf.linkage.comet_refusal), so the block reports found=False and there is nothing to
    # ack. It was the ONLY comet among the 102 PAIR fixtures -- and the only y-examine.
    # Everything identify() now calls y-examine (2026EN44/ES68/EW42/EZ42/JW6) is CLEAN under
    # linkage_verdict, which is the function this path actually consults, and 2026GV5 is a
    # 'y2': not submittable, no cautions to ack.
    #
    # So the test no longer hunts for a fixture that happens to be ambiguous. That hunt is what
    # broke it four times, and ambiguity was never the SUBJECT -- the subject is the SERVER-SIDE
    # enforcement of the caution click-through. The verdict is therefore INJECTED and everything
    # else stays real: a real fit of a real pair, the real _ident_linkage_block assembling the
    # real document, and the real itf_submit_web deciding. If linkage_verdict's shape ever
    # changes, this fails loudly at the seam rather than silently testing nothing.
    from pyroorb import ident_check
    _orig_verdict = ident_check.linkage_verdict

    def _forced_y_examine(fit):
        v = dict(_orig_verdict(fit))
        v.update(offer=True, clean=False, code="y-examine", submittable=True,
                 reason="identification candidate: the arcs join, but only just",
                 cautions=["a caution the server must see acknowledged"])
        return v

    ident_check.linkage_verdict = _forced_y_examine
    try:
        sid, p = _pair_session("PAIR_2026EN44")
        itf = p.get("itf")
        assert itf and itf["found"] and itf["clean"] is False, itf
        assert itf["warnings"], "an ambiguous identification must carry a caution"
        r = _mpc_submit(sid, _submitter=lambda d, c, log: (True, "200 OK", None))
        assert not r["ok"] and r.get("warnings"), r
        r2 = _mpc_submit(sid, acked_warnings=itf["warnings"],
                              _submitter=lambda d, c, log: (True, "200 OK", None))
        assert r2["ok"] and r2["accepted"], r2
    finally:
        ident_check.linkage_verdict = _orig_verdict
    print("PASS an ambiguous identification needs its cautions acknowledged")


def test_a_withheld_candidate_is_not_reported_as_nothing_found():
    """A night that LINKED and was withheld by the one-night show is a
    different finding from an empty search, and the payload says so: the
    block carries ``suppressed`` and the reason names the clause that refused
    (search._refuse) instead of the old trial queue wording."""
    sid, _ = _session()

    def withheld(seed_obs80, primaries, cfg):
        return ItfSearchResult(
            verdict=SearchVerdict.POSSIBLE,
            reason=("nothing shown: the tracklet that linked is almost "
                    "certainly one of many that would fit this orbit just as "
                    "well (many similarly-ranked candidate tracklets); at a match probability of 0.60 it is "
                    "nowhere near the standout this report needs before "
                    "naming a night."),
            vetting={"show_suppressed": True})

    itf = E.itf_search_web(sid, {}, _searcher=withheld)["itf"]
    assert itf["found"] is False and itf["suppressed"] is True, itf
    assert "one of many" in itf["reason"], itf["reason"]
    assert "1.5" not in itf["reason"], itf["reason"]      # still humanized

    def nothing(seed_obs80, primaries, cfg):
        return ItfSearchResult(verdict=SearchVerdict.NONE, reason="no likely matches")

    assert E.itf_search_web(sid, {}, _searcher=nothing)["itf"]["suppressed"] is False
    print("PASS a withheld candidate is flagged, not filed as nothing found")


def test_the_web_pays_for_the_rival_count_and_nothing_else_does():
    """The one-night show's rival COUNT costs k trial fits, so the engine
    default is not to count at all (`itf batch`, the harness and the
    suite must be untouched) and the WEB search is the one surface that turns
    it on -- a person who might go digging through archival images is exactly
    who the measurement is for."""
    from pyroorb.itf.config import ItfConfig

    assert ItfConfig().show_n_rivals == 0, "the engine must not count"
    assert E.WEB_RIVAL_COUNT_K > 0

    sid, _ = _session()
    seen = {}

    def searcher(seed_obs80, primaries, cfg):
        seen["k"] = cfg.show_n_rivals
        seen["cap"] = cfg.show_rival_count_max_linkers
        return ItfSearchResult(verdict=SearchVerdict.NONE, reason="no likely matches")

    E.itf_search_web(sid, {}, _searcher=searcher)
    assert seen["k"] == E.WEB_RIVAL_COUNT_K, seen
    assert seen["cap"] == ItfConfig().show_rival_count_max_linkers, seen
    print("PASS the web turns the rival count on; the engine default is off")


if __name__ == "__main__":
    fails = 0
    for fn in (test_found_returns_full_fit_payload, test_mask_filters_the_seed,
               test_alias_family_seeds_primary_only,
               test_user_assembled_pair_keeps_both_designations,
               test_submit_uses_server_held_linkage,
               test_submit_requires_warning_acks,
               test_ades_survives_the_linkage_refit,
               test_pending_linkage_survives_a_refit_and_follows_the_mask,
               test_db_unreachable_is_a_friendly_error,
               test_not_found_and_expired,
               test_payload_speaks_english_not_status_codes,
               test_a_withheld_candidate_is_not_reported_as_nothing_found,
               test_the_web_pays_for_the_rival_count_and_nothing_else_does,
               test_trace_feed_relevels_a_running_search,
               test_multi_designation_fit_offers_the_identification,
               test_identification_the_engine_rejects_is_not_offered,
               test_neocp_identification_submits_the_server_held_document,
               test_ident_submission_requires_its_cautions_acked):
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if fails else 0)


def test_a_banned_designation_never_reaches_the_live_document():
    """A partial pickup the search condemned (itf/pickup.py) stays ON SCREEN --
    the fit, the astrometry, the tracklet list -- and leaves only the DOCUMENT.

    The live rebuild is where that has to be enforced a second time: the submit
    button does not send the document the search returned, it rebuilds one from
    whatever is included right now, and a rebuild that did not know about the
    ban would quietly re-add the designation the search removed."""
    sid, _ = _session()
    fit = E.session_fit(sid)
    pickup_line = ("     " + f"{'K17Sd6S':<7}" + "  C"
                   + f"{'2017 10 15.20501':<17}" + f"{'10 00 00.00':<12}"
                   + f"{'+10 00 00.0':<12}" + " " * 9 + " 21.0G" + " "
                   + "     " + "G96")
    itf = _ITF_LINE + "\n" + pickup_line + "\n"

    def searcher(seed_obs80, primaries, cfg):
        return ItfSearchResult(
            verdict=SearchVerdict.LINKED, reason="ok", fit=fit, itf_obs80=itf,
            combined_obs80=_SEED_LINE + "\n" + itf,
            linkage={"links": {"link_0": {"designations": ["K08T04E"],
                                          "trksubs": [["PR00001", "2016 01 02.34567",
                                                       "F51"]]}}},
            vetting={"linkage_dropped_designations": ["K17Sd6S"]},
            telemetry={"obs_count": 2})

    found = E.itf_search_web(sid, {}, _searcher=searcher)
    s = E._SESSIONS[found["session"]]
    assert s["itf_pending"]["banned_designations"] == ["K17Sd6S"]
    # the tracklet is still shown to the user...
    block = E._itf_pending_block(s)
    assert "K17Sd6S" in "\n".join(s["itf_pending"]["lines"])
    assert "K17Sd6S" in {t["trksub"] for t in block["tracklets"]}
    # ...and is nowhere in the document the button would send
    doc, err = E._itf_live_linkage(s)
    assert err is None and doc is not None, (doc, err)
    link = doc["links"]["link_0"]
    assert "K17Sd6S" not in (link.get("designations") or [])
    assert "K17Sd6S" not in [t[0] for t in link.get("trksubs") or []]
    print("PASS a banned partial pickup stays on screen and out of the document")


def test_a_memory_constrained_search_says_so_without_gating_the_submit():
    """The search FINISHED, but the box was tight and it ran less deeply than
    usual (a shallow capacity verdict up front, or a LOV profile pass the memory
    floor killed mid-run).  The user has to be told -- "no additional
    observations found" from a search that ran degraded is misleading -- and it
    must arrive on `constrained`, not in `warnings`, which gate the submit
    click-through and exist for doubts about the LINKAGE."""
    sid, _base = _session()

    def searcher(seed_obs80, primaries, cfg):
        from pyroorb.itf.search import ItfSearchResult
        return ItfSearchResult(
            verdict=SearchVerdict.NONE, reason="nothing linked",
            telemetry={"capacity": {
                "verdict": "shallow",
                "reason": "the server ran out of memory part-way through"}})

    p = E.itf_search_web(sid, {}, _searcher=searcher)
    con = p["itf"]["constrained"]
    assert con["verdict"] == "shallow" and "memory" in con["reason"]
    assert p["itf"]["incomplete"] is None       # nothing was truncated


def test_an_incomplete_sweep_is_reported_on_a_NO_FIND():
    """"No match" from a corridor that was never fully queried is not evidence
    of absence, and the user cannot tell the two apart from the verdict alone.
    Before this the not-found payload carried no warnings at all."""
    sid, _base = _session()

    def searcher(seed_obs80, primaries, cfg):
        from pyroorb.itf.search import ItfSearchResult
        r = ItfSearchResult(verdict=SearchVerdict.NONE, reason="nothing linked")
        r.incomplete_sweep = {
            "windows": 4200, "lowmem": 4200, "cancelled": 0, "budget": 0,
            "memory_floor_mb": 300, "verdict_affected": True,
            "message": "The corridor was NOT swept completely: 4200 window(s).",
        }
        return r

    p = E.itf_search_web(sid, {}, _searcher=searcher)
    inc = p["itf"]["incomplete"]
    assert inc["windows"] == 4200 and inc["memory"] is True
    assert "memory" in inc["why"]
    assert inc["verdict_affected"] is True


def test_an_incomplete_sweep_is_reported_on_a_FIND_TOO():
    """A linkage out of a truncated corridor is real but partial: the windows
    never queried could hold more tracklets of the same object.  It must NOT
    join `warnings`, which gate the submit click-through -- an incomplete
    sweep is a capacity fact, not a doubt about the astrometry, and making
    users acknowledge it before submitting trains them to click through."""
    sid, _base = _session()
    fit = E.session_fit(sid)
    note = "The corridor was NOT swept completely: 7 window(s)."

    def searcher(seed_obs80, primaries, cfg):
        itf_line = ("     " + f"{'PR00002':<7}" + "  C"
                    + f"{'2016 01 02.34567':<17}" + f"{'10 00 00.00':<12}"
                    + f"{'+10 00 00.0':<12}" + " " * 9 + " 21.0G" + " "
                    + "     " + "F51")
        from pyroorb.itf.search import ItfSearchResult
        r = ItfSearchResult(verdict=SearchVerdict.LINKED, reason="test", fit=fit,
                            combined_obs80=seed_obs80 + "\n" + itf_line,
                            itf_obs80=itf_line + "\n",
                            linkage={"links": {}}, telemetry={"obs_count": 1})
        r.warnings = [note, "the object's own observations fit worse"]
        r.incomplete_sweep = {
            "windows": 7, "lowmem": 7, "cancelled": 0, "budget": 0,
            "memory_floor_mb": 300, "verdict_affected": False,
            "message": note,
        }
        return r

    p = E.itf_search_web(sid, {}, _searcher=searcher)
    assert p["itf"]["found"] is True
    assert p["itf"]["incomplete"]["windows"] == 7
    assert p["itf"]["incomplete"]["verdict_affected"] is False
    joined = " ".join(p["itf"]["warnings"])
    assert "NOT swept completely" not in joined, \
        "the capacity note must not gate the submit click-through"
    assert "fit worse" in joined, "real doubts must still gate it"


def test_a_trend_diagnostic_is_reported_but_never_gates_the_submit():
    """The residual trend is a DIAGNOSTIC (2026-08-25): reported on the payload,
    never in `warnings`.

    Its gate was withdrawn when the full calibration measured it firing on 2 of
    58 correct tracklets against 3 of 119 wrong -- a test that rejects about as
    much real astrometry as junk must not be able to take a submission away.
    Same contract as `incomplete`: shown, never acknowledged."""
    sid, _base = _session()
    fit = E.session_fit(sid)
    trend_note = "a linked tracklet's residuals TREND: the 2021-09-13 tracklet walks -2.5 sigma"

    def searcher(seed_obs80, primaries, cfg):
        itf_line = ("     " + f"{'PR00002':<7}" + "  C"
                    + f"{'2016 01 02.34567':<17}" + f"{'10 00 00.00':<12}"
                    + f"{'+10 00 00.0':<12}" + " " * 9 + " 21.0G" + " "
                    + "     " + "F51")
        from pyroorb.itf.search import ItfSearchResult
        r = ItfSearchResult(verdict=SearchVerdict.LINKED, reason="test", fit=fit,
                            combined_obs80=seed_obs80 + "\n" + itf_line,
                            itf_obs80=itf_line + "\n",
                            linkage={"links": {}}, telemetry={"obs_count": 1})
        r.diagnostics = [trend_note]
        return r

    p = E.itf_search_web(sid, {}, _searcher=searcher)
    assert p["itf"]["found"] is True
    assert p["itf"]["submittable"] is True, "a diagnostic must not block submission"
    assert any("TREND" in d for d in p["itf"]["diagnostics"]), p["itf"]["diagnostics"]
    assert not any("TREND" in w for w in p["itf"]["warnings"]), p["itf"]["warnings"]


def test_a_complete_sweep_says_nothing_about_completeness():
    """No banner when there is nothing to report -- a capacity warning on every
    search would be noise, and noise is how real warnings get ignored."""
    sid, _base = _session()

    def searcher(seed_obs80, primaries, cfg):
        from pyroorb.itf.search import ItfSearchResult
        return ItfSearchResult(verdict=SearchVerdict.NONE, reason="nothing linked")

    p = E.itf_search_web(sid, {}, _searcher=searcher)
    assert p["itf"]["incomplete"] is None


def test_sync_search_is_admission_gated_like_the_async_path(monkeypatch):
    """A client that POSTs without async gets the blocking answer -- but through
    the same concurrency cap and unusable-box refusal as the background path.
    The raw itf_search_web used to skip both, so N blocking POSTs each started a
    live sweep unseen by the cap (the failure admission control exists to stop).
    """
    import time as _t
    # the CONCURRENCY cap is what is under test; the memory verdict is the other
    # half of admission and would refuse on a squeezed runner (see above)
    monkeypatch.setattr(E, "_itf_capacity_block", lambda: None)
    sid, _ = _session()
    saved = dict(E._ITF_JOBS)
    saved_max = E._ITF_MAX_RUNNING
    try:
        E._ITF_JOBS.clear()
        # fill every slot with a job that is stuck "running"
        E._ITF_MAX_RUNNING = 2
        now = _t.time()
        for k in ("busyA", "busyB"):
            E._ITF_JOBS[k] = {"done": False, "started": now, "ended": 0.0,
                              "collected": 0.0, "payload": None,
                              "rss_start_mb": None}
        assert E._itf_running() == 2

        def searcher(seed_obs80, primaries, cfg):
            return ItfSearchResult(verdict=SearchVerdict.NONE, reason="should not run")

        refused = E.itf_search_sync(sid, {}, _searcher=searcher)
        assert refused["ok"] is False and refused["busy"] is True
        assert refused["running"] == 2
        # ...and the refused search did NOT register a job of its own
        assert E._itf_running() == 2
        print("PASS sync search refused when the box is full")

        # free a slot: now it runs, returns the payload, and releases the slot
        E._ITF_JOBS.pop("busyB")
        ran = E.itf_search_sync(sid, {}, _searcher=searcher)
        assert ran["ok"] and ran["itf"]["found"] is False
        assert E._itf_running() == 1, "the slot must be released after the search"
    finally:
        E._ITF_MAX_RUNNING = saved_max
        E._ITF_JOBS.clear()
        E._ITF_JOBS.update(saved)


def test_sync_search_does_not_clobber_a_running_async_job():
    """The blocking path mints its own tracking key, so a client reusing a
    trace_id cannot overwrite the _ITF_JOBS entry of a live async search."""
    sid, _ = _session()
    saved = dict(E._ITF_JOBS)
    try:
        E._ITF_JOBS.clear()
        import time as _t
        E._ITF_JOBS["trSHARED"] = {"done": False, "started": _t.time(),
                                   "ended": 0.0, "collected": 0.0,
                                   "payload": "the async payload",
                                   "rss_start_mb": None}

        def searcher(seed_obs80, primaries, cfg):
            return ItfSearchResult(verdict=SearchVerdict.NONE, reason="sync ran")

        # a sync POST carrying the SAME trace_id must not touch trSHARED
        E.itf_search_sync(sid, {"trace_id": "trSHARED"}, _searcher=searcher)
        assert E._ITF_JOBS["trSHARED"]["payload"] == "the async payload"
        assert E._ITF_JOBS["trSHARED"]["done"] is False
    finally:
        E._ITF_JOBS.clear()
        E._ITF_JOBS.update(saved)


def test_incomplete_sweep_why_names_the_dominant_cause_not_a_default():
    """A mixed cancel+memory truncation must not be headlined 'a row budget was
    reached' when the budget dropped nothing -- that points the user at a remedy
    that does nothing. The banner names whichever cause dropped the most."""
    sid, _base = _session()
    fit = E.session_fit(sid)

    def searcher(seed_obs80, primaries, cfg):
        from pyroorb.itf.search import ItfSearchResult
        r = ItfSearchResult(verdict=SearchVerdict.NONE, reason="nothing linked")
        r.incomplete_sweep = {
            "windows": 10, "lowmem": 4, "cancelled": 6, "budget": 0,
            "memory_floor_mb": 300, "verdict_affected": True,
            "message": "The corridor was NOT swept completely: 10 window(s).",
        }
        return r

    p = E.itf_search_web(sid, {}, _searcher=searcher)
    why = p["itf"]["incomplete"]["why"]
    assert why.startswith("the database cancelled queries"), why
    assert "row budget" not in why, "a budget that dropped nothing must not lead"
    assert "and other limits" in why, "the memory tail should be flagged"


def test_incomplete_sweep_on_a_find_from_a_budget_truncation_is_reported():
    """A find whose sweep was cut short by a ROW BUDGET (not memory) is still
    incomplete: the unswept windows could hold more tracklets. It surfaces as a
    banner, not a submit gate."""
    sid, _base = _session()
    fit = E.session_fit(sid)

    def searcher(seed_obs80, primaries, cfg):
        itf_line = ("     " + f"{'PR00003':<7}" + "  C"
                    + f"{'2016 01 02.34567':<17}" + f"{'10 00 00.00':<12}"
                    + f"{'+10 00 00.0':<12}" + " " * 9 + " 21.0G" + " "
                    + "     " + "F51")
        from pyroorb.itf.search import ItfSearchResult
        r = ItfSearchResult(verdict=SearchVerdict.LINKED, reason="test", fit=fit,
                            combined_obs80=seed_obs80 + "\n" + itf_line,
                            itf_obs80=itf_line + "\n",
                            linkage={"links": {}}, telemetry={"obs_count": 1})
        r.incomplete_sweep = {
            "windows": 9, "lowmem": 0, "cancelled": 0, "budget": 9,
            "memory_floor_mb": 300, "verdict_affected": False,
            "message": "The corridor was NOT swept completely: 9 window(s).",
        }
        return r

    p = E.itf_search_web(sid, {}, _searcher=searcher)
    assert p["itf"]["found"] is True
    assert p["itf"]["incomplete"]["windows"] == 9
    assert p["itf"]["incomplete"]["why"] == "a row budget was reached"
    assert p["itf"]["incomplete"]["memory"] is False


def test_a_possible_stays_unsubmittable_through_a_step_until_the_set_grows(monkeypatch):
    """2021 PA97: the search links ONE tracklet and files it as a possible (status 0.6,
    ``result.linkage`` None) -- no "Submit linkage" button, correctly. A Full step then
    re-solved the SAME observations, and the live rebuild (which only knows build_linkage's
    two-id minimum: one designation + one tracklet satisfies it) reported submittable=True,
    so the button appeared on a linkage the search had just refused.

    By design, after a possible, a step may raise the button only when observations were
    ADDED or SWITCHED ON since the search -- and only if the grown set passes THE SAME gate the
    search applied (itf.search.opp_profile_acceptable), not a second web-side rule."""
    sid, _ = _session()
    fit = E.session_fit(sid)

    def searcher(seed_obs80, primaries, cfg):
        return ItfSearchResult(verdict=SearchVerdict.POSSIBLE, reason="one lone new night (1.5 opps)", fit=fit,
                               itf_obs80=_ITF_LINE + "\n",
                               combined_obs80=_SEED_LINE + "\n" + _ITF_LINE + "\n",
                               linkage=None, telemetry={"obs_count": 1})

    found = E.itf_search_web(sid, {}, _searcher=searcher)
    assert found["itf"]["found"] and found["itf"]["possible"]
    assert found["itf"]["submittable"] is False
    new_sid = found["session"]
    s = E._SESSIONS[new_sid]
    pend = s["itf_pending"]
    assert pend["submittable"] is False and pend["judged_ids"], pend
    assert set(pend["seed_gate"]) == {"pub_ids", "initial_opp_profile", "pub_arclength_days"}

    # the gate must not even be consulted while the set has not grown
    import pyroorb.itf.search as S
    calls = []
    def gate_refuses(*a, **k):
        calls.append((a, k)); return False, "profile says no"
    monkeypatch.setattr(S, "opp_profile_acceptable", gate_refuses)

    # (1) a Full step on the same observations: still a possible, still no button
    p = E._apply_interactive(s, new_sid, None, single=True)
    assert p.get("ok", True) and p["itf"]["found"], p.get("error")
    assert p["itf"]["submittable"] is False and p["itf"]["possible"] is True, p["itf"]
    assert "possible" in p["itf"]["reason"], p["itf"]["reason"]
    assert not calls, "the gate was re-run on an unchanged set"
    # ...and a hand-crafted POST is refused by the same verdict
    r = _mpc_submit(new_sid, _submitter=lambda d, c, log: (True, "200 OK", None))
    assert not r["ok"] and "not submittable" in r["error"], r

    # (2) the set GROWS: a second night in the new opposition switched on / pasted in
    _NEW_LINE = _ITF_LINE[:15] + f"{'2016 01 05.31234':<17}" + _ITF_LINE[32:]
    base = s["fit"]
    class _GrownFit:
        def __init__(self):
            self._obs = list(base._obs) + [{"obs80_line": _NEW_LINE, "jd_utc": 2457392.81,
                                            "obscode": "F51", "designation": "PR00001"}]
            self._inc = list(base.is_included()) + [True]
        def is_included(self): return list(self._inc)
        def __getattr__(self, k): return getattr(base, k)
    s["fit"] = _GrownFit()
    pend["lines"] = pend["lines"] + [_NEW_LINE]     # the screen's astrometry grew too
    block = E._itf_pending_block(s)
    assert calls, "a grown set must be judged by the search's own gate"
    (cfg, combined, trial), kw = calls[-1]
    assert _NEW_LINE in combined and set(kw) == {"pub_arclength_days", "pub_ids", "initial_opp_profile"}
    assert kw["pub_ids"] == set(pend["seed_gate"]["pub_ids"])
    assert block["submittable"] is False and "still fails" in block["reason"], block
    assert "profile says no" in block["reason"]

    # (3) the same grown set, and the gate passes: NOW the button may appear
    monkeypatch.setattr(S, "opp_profile_acceptable", lambda *a, **k: (True, ""))
    block = E._itf_pending_block(s)
    assert block["submittable"] is True and block["possible"] is False, block
    assert "added or switched on since the search" in block["reason"]

    # (4) a SWAP is not growth. The set the search JUDGED must still be
    # wholly present: switching one judged observation off while switching an unjudged
    # one on leaves `included - judged` non-empty, which used to read as growth and
    # raise the button on evidence the search's ranking and recursion never saw --
    # possibly missing the very tracklet the linkage needed. Nothing downstream
    # recovers it: opp_profile_acceptable is the OPPOSITION-PROFILE clause only (and here it passes).
    calls.clear()
    swapped = s["fit"]
    dropped = sorted(set(pend["judged_ids"]))[0]
    from pyroorb.itf.obs80_fields import obs_unique_id
    off = [i for i, o in enumerate(swapped._obs)
           if o.get("obs80_line") and obs_unique_id(o["obs80_line"]) == dropped]
    assert off, "the judged id must be findable on the fit"
    inc = swapped.is_included()
    for i in off:
        inc[i] = False
    swapped._inc = inc                        # the _GrownFit above owns the mask
    block = E._itf_pending_block(s)
    assert block["submittable"] is False and block["possible"] is True, block
    assert "no longer in the fit" in block["reason"], block["reason"]
    assert not calls, "a swapped set must not even reach the search's gate"
    print("PASS a possible stays unsubmittable through a step until the set grows")


def test_a_profile_only_possible_is_offered_after_a_step_and_not_before(monkeypatch):
    """The ONE possible the search refuses on opposition profile alone -- the
    recursion-accumulated 1.75, two new oppositions of one night each, fitting cleanly with no
    doubt raised (``vetting["promotable_on_step"]``) -- becomes submittable once the user has
    RE-SOLVED it, without the observation set growing. Not before a step: the search's own
    payload raises no button and a hand-crafted POST is refused. And a possible WITHOUT the
    flag (a doubt was raised) keeps the growth rule, whatever the step count."""
    sid, _ = _session()
    fit = E.session_fit(sid)

    def searcher(seed_obs80, primaries, cfg):
        return ItfSearchResult(verdict=SearchVerdict.POSSIBLE, reason="1.75 opps -- too fragile to submit, abandoning",
                               fit=fit, itf_obs80=_ITF_LINE + "\n",
                               combined_obs80=_SEED_LINE + "\n" + _ITF_LINE + "\n",
                               linkage=None, telemetry={"obs_count": 1},
                               warnings=["Possible linkage, not confirmed. Two of its three "
                                         "oppositions rest on a single night each."],
                               vetting={"possible_unconfirmed": True,
                                        "promotable_on_step": True})

    found = E.itf_search_web(sid, {}, _searcher=searcher)
    assert found["itf"]["found"] and found["itf"]["possible"]
    assert found["itf"]["submittable"] is False, "the search's own payload never offers it"
    new_sid = found["session"]
    s = E._SESSIONS[new_sid]
    assert not s["itf_pending"].get("stepped")

    # the opposition-profile gate is never consulted on this path: the search already judged the profile
    import pyroorb.itf.search as S
    calls = []
    monkeypatch.setattr(S, "opp_profile_acceptable",
                        lambda *a, **k: (calls.append(1), (False, "profile says no"))[1])

    # (1) before any step: refused, and the reason names what would offer it
    block = E._itf_pending_block(s)
    assert block["submittable"] is False and block["possible"] is True, block
    assert "Full step or Multi-step" in block["reason"], block["reason"]
    # (cautions acked, so the POST reaches the VERDICT rather than the ack check)
    r = _mpc_submit(new_sid, acked_warnings=found["itf"]["warnings"],
                         _submitter=lambda d, c, log: (True, "200 OK", None))
    assert not r["ok"] and "not submittable" in r["error"], r

    # (2) a Full step on the SAME observations: offered, at the user's judgement
    p = E._apply_interactive(s, new_sid, None, single=True)
    assert p.get("ok", True) and p["itf"]["found"], p.get("error")
    assert s["itf_pending"]["stepped"] is True
    assert p["itf"]["submittable"] is True and p["itf"]["possible"] is False, p["itf"]
    assert "re-solved after a step" in p["itf"]["reason"], p["itf"]["reason"]
    assert not calls, "the opposition-profile gate has no say over a profile-only promotion"
    assert p["itf"]["warnings"], "the search's caution still gates the click-through"

    # (3) control: the same profile WITHOUT the flag (a doubt was raised) stays refused by a step
    sid2, _ = _session()
    fit2 = E.session_fit(sid2)

    def doubtful(seed_obs80, primaries, cfg):
        return ItfSearchResult(verdict=SearchVerdict.POSSIBLE, reason="possible linkage, kept for review: seed",
                               fit=fit2, itf_obs80=_ITF_LINE + "\n",
                               combined_obs80=_SEED_LINE + "\n" + _ITF_LINE + "\n",
                               linkage=None, telemetry={"obs_count": 1},
                               vetting={"possible_unconfirmed": True, "seed_degraded": True})

    found2 = E.itf_search_web(sid2, {}, _searcher=doubtful)
    s2 = E._SESSIONS[found2["session"]]
    p2 = E._apply_interactive(s2, found2["session"], None, single=True)
    assert p2["itf"]["submittable"] is False and p2["itf"]["possible"] is True, p2["itf"]
    assert "re-solving the same observations does not change that" in p2["itf"]["reason"]
    print("PASS a profile-only possible is offered after a step and not before")


def test_a_leaveout_demoted_caution_is_rejudged_when_the_set_changes(monkeypatch):
    """itf/leaveout.py demotes a caution on the observation set the SEARCH judged. Submit
    the same set and it stays demoted (no refit); switch an observation off first and it is
    re-judged on what is being submitted -- one that no longer confirms comes back as a
    click-through the server enforces, one that still confirms does not."""
    from pyroorb.itf import leaveout
    sid, _ = _session()
    fit = E.session_fit(sid)
    warn = ("a tracklet of a linked tracklet (F51, 2016-01-02, 3 obs, p=0.01) averages "
            "+0.9 mag from the object's predicted brightness (possible different body)")
    note = leaveout.note(warn, {"miss": 0.2, "smaj": 0.3, "D": 0.8, "walk": 0.2})
    assert leaveout.original(note) == warn

    def searcher(seed_obs80, primaries, cfg):
        r = ItfSearchResult(verdict=SearchVerdict.LINKED, reason="ok", fit=fit,
                            itf_obs80=_ITF_LINE + "\n",
                            combined_obs80=_SEED_LINE + "\n" + _ITF_LINE + "\n",
                            linkage={"links": {}}, telemetry={"obs_count": 0})
        r.diagnostics.append(note)
        return r

    ok_sub = lambda d, c, log: (True, "200 OK", None)
    calls = []

    def judge(kept):
        def fake(warnings, diagnostics, *a, **k):
            calls.append(list(warnings))
            return (list(warnings) if kept else []), list(diagnostics), []
        monkeypatch.setattr(leaveout, "confirm_cautions", fake)

    # unchanged set: the search's judgement stands, nothing is refitted
    judge(kept=True)
    found = E.itf_search_web(sid, {}, _searcher=searcher)
    new_sid = found["session"]
    assert found["itf"]["warnings"] == [] and found["itf"]["diagnostics"] == [note]
    assert E._leaveout_rejudge(E._SESSIONS[new_sid]) == [] and calls == []

    # an observation switched off: re-judged; no longer confirmed -> must be acknowledged
    E._SESSIONS[new_sid]["fit"]._mask[0] = False
    r = _mpc_submit(new_sid, acked_warnings=[], _submitter=ok_sub)
    assert not r["ok"] and r["rejudged"] == [warn] and r["warnings"] == [warn]
    assert len(calls) == 1
    r = _mpc_submit(new_sid, acked_warnings=[warn], _submitter=ok_sub)
    assert r["ok"] and r["accepted"]
    assert len(calls) == 1                      # the retry reused the cached re-judge

    # ...and one that still confirms on the changed set needs no acknowledgement
    judge(kept=False)
    found = E.itf_search_web(sid, {}, _searcher=searcher)
    sid2 = found["session"]
    E._SESSIONS[sid2]["fit"]._mask[0] = False
    r = _mpc_submit(sid2, acked_warnings=[], _submitter=ok_sub)
    assert r["ok"] and r["accepted"]
