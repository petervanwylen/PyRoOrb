"""The single-tracklet (SR) ITF search on a synthetic two-body object: the screen and the
ranking must put the object's own later tracklet first among random distractors, and the
helpers that route a search (short arc, apparition window, overlay window) must hold."""
import math

import numpy as np
import pytest

from pyroorb.itf import srsearch as sr


@pytest.fixture(scope="module")
def obscodes():
    from pyroorb.obs_sources import load_obscodes
    return load_obscodes()


def _observe(state_pos, state_vel, t_emit_tdb, jds, codes, obscodes):
    """Topocentric (ra, dec) degrees of a two-body orbit at UTC ``jds`` from ``codes``."""
    el, ok = sr._safe_elements(*sr.universal_elements(state_pos[None, :], state_vel[None, :],
                                                      t_emit_tdb))
    assert ok[0]
    O, _ = sr.fast_observers(np.asarray(jds), codes, obscodes)
    out = []
    for j, o in zip(jds, O):
        t = j + sr.TT_UTC_DAYS
        for _ in range(3):
            p, _v = sr.conic.state_at(*el, np.array([t]))
            t = j + sr.TT_UTC_DAYS - np.linalg.norm(p[0] - o) / sr.C_AU_DAY
        d = p[0] - o
        out.append((math.degrees(math.atan2(d[1], d[0])) % 360,
                    math.degrees(math.asin(d[2] / np.linalg.norm(d)))))
    return out


@pytest.fixture(scope="module")
def synthetic(obscodes):
    rng = np.random.default_rng(7)
    jd0 = 2460400.5 + 0.30
    att0 = {"x": np.radians([150.0, 12.0, -0.30, 0.10]), "tdb0": jd0 + sr.TT_UTC_DAYS}
    R, V = sr.fast_observers(np.array([jd0]), ["F51"], obscodes)
    rho, rd = 0.6, -0.004
    pos, vel = sr.states(att0["x"], rho, rd, R[0], V[0])
    t_emit = att0["tdb0"] - rho / sr.C_AU_DAY
    seed_jds = [jd0 + k * 0.012 for k in range(4)]
    seed = [{"jd_utc": j, "ra_deg": a, "dec_deg": d, "obscode": "F51", "sigma_ra": 0.2,
             "sigma_dec": 0.2} for j, (a, d) in zip(seed_jds, _observe(pos, vel, t_emit,
                                                                      seed_jds, ["F51"] * 4,
                                                                      obscodes))]
    tj = [jd0 + 6.0 + k * 0.01 for k in range(3)]
    trk = _observe(pos, vel, t_emit, tj, ["G96"] * 3, obscodes)
    (a0, d0), (a2, d2) = trk[0], trk[-1]
    span = tj[-1] - tj[0]
    truth = {"tracklet_id": "TRUTH", "stn": "G96", "epoch_mid_sec": (tj[1] - 2440587.5) * 86400,
             "center_ra": trk[1][0], "center_dec": trk[1][1],
             "dra_cosdec_rate": (a2 - a0) * math.cos(math.radians(d0)) / span,
             "ddec_rate": (d2 - d0) / span, "sig_pos_e_arcsec": 0.4, "sig_pos_n_arcsec": 0.4,
             "sig_vel_e_degday": 0.004, "sig_vel_n_degday": 0.004, "n_obs": 3}
    rows = [truth]
    for i in range(300):
        rows.append({"tracklet_id": f"D{i}", "stn": "G96",
                     "epoch_mid_sec": (jd0 + rng.uniform(-30, 30) - 2440587.5) * 86400,
                     "center_ra": rng.uniform(100, 200), "center_dec": rng.uniform(-20, 40),
                     "dra_cosdec_rate": rng.normal(0, 0.4), "ddec_rate": rng.normal(0, 0.3),
                     "sig_pos_e_arcsec": 0.4, "sig_pos_n_arcsec": 0.4,
                     "sig_vel_e_degday": 0.004, "sig_vel_n_degday": 0.004, "n_obs": 3})
    return seed, rows


def test_the_true_tracklet_ranks_first(synthetic, obscodes):
    seed, rows = synthetic
    cfg = sr.SrConfig(n_rho=80, n_v=80, w_prior=0.0)
    att = sr.seed_attributable(seed, obscodes, cfg)
    cand = sr.candidate_arrays(rows, obscodes)
    ranked = sr.rank(att, cand, cfg, obscodes)
    assert ranked, "nothing survived the screen"
    assert cand["id"][ranked[0]["ci"]] == "TRUTH"
    assert ranked[0]["chi2"] < 6.0


@pytest.fixture(scope="module")
def two_night(obscodes):
    """A 0.1 au object seen from F51 on two nights a day apart (a NEOCP seed the short-arc
    rule sends here), its own tracklet six days later, 300 random distractors."""
    rng = np.random.default_rng(11)
    jd0 = 2460400.5 + 0.30
    x = np.radians([150.0, 12.0, -1.2, 0.4])
    R, V = sr.fast_observers(np.array([jd0]), ["F51"], obscodes)
    rho = 0.1
    pos, vel = sr.states(x, rho, -0.003, R[0], V[0])
    t_emit = jd0 + sr.TT_UTC_DAYS - rho / sr.C_AU_DAY
    sjd = [jd0 + night + k * 0.012 for night in (0.0, 1.0) for k in range(4)]
    seed = [{"jd_utc": j, "ra_deg": a, "dec_deg": d, "obscode": "F51", "sigma_ra": 0.2,
             "sigma_dec": 0.2}
            for j, (a, d) in zip(sjd, _observe(pos, vel, t_emit, sjd, ["F51"] * 8, obscodes))]
    tj = [jd0 + 6.0 + k * 0.01 for k in range(3)]
    trk = _observe(pos, vel, t_emit, tj, ["G96"] * 3, obscodes)
    (a0, d0), (a2, d2) = trk[0], trk[-1]
    rows = [{"tracklet_id": "TRUTH", "stn": "G96",
             "epoch_mid_sec": (tj[1] - sr.UNIX_JD0) * 86400,
             "center_ra": trk[1][0], "center_dec": trk[1][1],
             "dra_cosdec_rate": (a2 - a0) * math.cos(math.radians(d0)) / (tj[-1] - tj[0]),
             "ddec_rate": (d2 - d0) / (tj[-1] - tj[0]), "sig_pos_e_arcsec": 0.4,
             "sig_pos_n_arcsec": 0.4, "sig_vel_e_degday": 0.004, "sig_vel_n_degday": 0.004,
             "n_obs": 3}]
    rows += [{"tracklet_id": f"D{i}", "stn": "G96",
              "epoch_mid_sec": (jd0 + rng.uniform(-30, 30) - sr.UNIX_JD0) * 86400,
              "center_ra": rng.uniform(100, 200), "center_dec": rng.uniform(-20, 40),
              "dra_cosdec_rate": rng.normal(0, 1.2), "ddec_rate": rng.normal(0, 0.5),
              "sig_pos_e_arcsec": 0.4, "sig_pos_n_arcsec": 0.4, "sig_vel_e_degday": 0.004,
              "sig_vel_n_degday": 0.004, "n_obs": 3} for i in range(300)]
    return seed, rows


def test_a_two_night_seed_screens_its_own_tracklet(two_night, obscodes):
    """The attributable comes from one night: a line through both nights missed this object
    by ~40" under a 0.2"/day rate sigma and screened the true tracklet out (chi2 ~2500)."""
    seed, rows = two_night
    assert sr._nights([o["jd_utc"] for o in seed]) == 2
    cfg = sr.SrConfig(n_rho=80, n_v=80, w_prior=0.0)
    att = sr.seed_attributable(seed, obscodes, cfg)
    assert att["n"] == 4 and att["t0"] > seed[4]["jd_utc"] - 0.01     # the later night
    cand = sr.candidate_arrays(rows, obscodes)
    ranked = sr.rank(att, cand, cfg, obscodes)
    assert ranked and cand["id"][ranked[0]["ci"]] == "TRUTH"
    assert ranked[0]["chi2"] < 6.0
    both = sr.line_attributable(seed, obscodes, cfg)                    # the old seed
    assert "TRUTH" not in [cand["id"][r["ci"]] for r in sr.rank(both, cand, cfg, obscodes)]


def test_seed_night_is_the_busiest_station_night():
    o = lambda stn, *jds: [{"obscode": stn, "jd_utc": j} for j in jds]   # noqa: E731
    pick = lambda obs: [x["jd_utc"] for x in sr.seed_night(obs)]       # noqa: E731
    # the busiest station-night, whichever station it belongs to
    assert pick(o("F51", 1.0, 1.01, 2.0, 2.01, 2.02) + o("G96", 1.5, 1.51)) == [2.0, 2.01, 2.02]
    assert pick(o("F51", 1.0, 1.01) + o("G96", 1.5, 2.6, 2.61)) == [2.6, 2.61]
    # equal counts: the longer night, then the later one
    assert pick(o("F51", 1.0, 1.05, 2.0, 2.01)) == [1.0, 1.05]
    assert pick(o("F51", 1.0, 1.125, 2.0, 2.125)) == [2.0, 2.125]
    # a night is a 12-hour window from its first observation: 0.49 d later is the same
    # night, 0.5 d later the next one
    assert pick(o("F51", 1.0, 1.49, 3.0)) == [1.0, 1.49]
    assert pick(o("F51", 1.0, 1.5, 1.51)) == [1.5, 1.51]


def test_a_seed_with_one_obs_a_night_is_refused_not_a_crash():
    # F51 once on each of two nights + G96 once: no station-night holds two positions
    obscodes = {"F51": (0.9, 0.3, 0.9), "G96": (0.8, 0.4, 0.9), "C51": (0.0, 0.0, 0.0)}
    o = lambda stn, jd: {"obscode": stn, "jd_utc": jd, "ra_deg": 10.0, "dec_deg": 5.0}  # noqa: E731
    seed = [o("F51", 1.0), o("F51", 2.0), o("G96", 1.5)]
    assert "twice in one night" in sr.seed_night_problem(seed, obscodes)
    res = sr.sr_search(seed, source=None, obscodes=obscodes)
    assert res.link is None and "twice in one night" in res.reason
    # a less-observed station's two-position night is used rather than refused
    seed2 = seed + [o("G96", 2.6), o("G96", 2.61)]
    assert sr.seed_night_problem(seed2, obscodes) is None
    assert [x["jd_utc"] for x in sr.seed_night(seed2, obscodes)] == [2.6, 2.61]
    # a space-based station's night is never the one ranged from
    seed3 = seed + [o("C51", 3.0), o("C51", 3.01), o("C51", 3.02)]
    assert sr.seed_night_problem(seed3, obscodes) is not None
    # ...nor does it refuse a seed whose busiest station is in space: the ground night ranges
    seed4 = seed3 + [o("F51", 3.5), o("F51", 3.51)]
    assert sr.seed_night_problem(seed4, obscodes) is None
    assert [x["obscode"] for x in sr.seed_night(seed4, obscodes)] == ["F51", "F51"]


def test_only_the_seeds_own_rejections_are_pardoned():
    """A proposal must keep every observation, except one the seed's own fit rejects."""
    cfg = sr.SrConfig()
    bad = ("691", 212364893506)
    fit = {"ok": True, "count": 24, "used": 23, "chi2_dof": 0.5, "max_z": 2.2,
           "dropped": [bad]}
    assert not sr._passes(fit, cfg, 24)
    assert sr._fits_but_drops(fit, cfg, 24)
    assert sr._passes(fit, cfg, 24, frozenset({bad}))
    other = dict(fit, dropped=[("G96", 1)])                     # a different observation
    assert not sr._passes(other, cfg, 24, frozenset({bad}))
    two = dict(fit, used=22, dropped=[bad, ("G96", 1)])         # one more than pardoned
    assert not sr._passes(two, cfg, 24, frozenset({bad}))
    worse = dict(fit, chi2_dof=2.5)                             # the other tests still hold
    assert not sr._passes(worse, cfg, 24, frozenset({bad}))
    assert not sr._fits_but_drops(worse, cfg, 24)


def test_the_router_asks_whether_the_seed_fixes_a_distance(monkeypatch):
    """One night goes to SR unfitted; any other seed by its own fit's verdict, whatever its
    length (a seed no orbit fits answers False and is ranged; a fit with no answer -- out of
    time, an error -- is no evidence and leaves it to the regular search); past the routing
    arc it is never fitted."""
    one = [{"jd_utc": 2460000.5 + k * 0.01} for k in range(4)]
    two = one + [{"jd_utc": 2460001.6}]
    week = one + [{"jd_utc": 2460007.6}]
    long = one + [{"jd_utc": 2460000.5 + sr.ROUTE_MAX_FIT_ARC_DAYS + 1}]
    route = lambda obs: sr.needs_sr(obs, "text", {})[0]        # noqa: E731
    monkeypatch.setattr(sr, "seed_determined", lambda *a, **k: pytest.fail("no fit"))
    assert route(one) and not route(long) and not route([])
    for verdict, use_sr in ((True, False), (False, True), (None, False)):
        monkeypatch.setattr(sr, "seed_determined", lambda *a, v=verdict, **k: v)
        assert route(two) is use_sr and route(week) is use_sr


def test_seed_determined_tells_no_answer_from_no_orbit(monkeypatch):
    from types import SimpleNamespace
    from pyroorb.itf import fitter
    import numpy as np

    class Fit:
        def __init__(self, cov, det):
            self.cov, self.det = cov, det

        def covariance_matrix(self, kind):
            return self.cov

        def determinacy(self):
            return self.det

    def answer(tr):
        monkeypatch.setattr(fitter, "fit_astrometry", lambda *a, **k: tr)
        return sr.seed_determined("text", {})
    ok = lambda fit: SimpleNamespace(ok=True, fit=fit, timed_out=False)   # noqa: E731
    assert answer(SimpleNamespace(ok=False, fit=None, timed_out=True)) is None
    assert answer(SimpleNamespace(ok=False, fit=None, timed_out=False)) is False
    assert answer(ok(Fit(np.eye(6), None))) is True
    assert answer(ok(Fit(np.eye(6), {"level": "unconstrained"}))) is False
    # no readable covariance: no answer, never "determined"
    assert answer(ok(Fit(np.full((6, 6), np.nan), None))) is None
    monkeypatch.setattr(fitter, "fit_astrometry", lambda *a, **k: 1 / 0)
    assert sr.seed_determined("text", {}) is None


def test_apparition_config_bounds_the_sweep():
    from pyroorb.itf.config import ItfConfig
    cfg = sr.apparition_config(ItfConfig(), 2460400.5)
    assert cfg.ephem_start == "2023-10-03" and cfg.archive_start == ""
    assert cfg.ephem_steps * cfg.ephem_step_days == pytest.approx(360, abs=cfg.ephem_step_days)


def test_overlay_window_rows_admit_plants_by_time_only():
    from pyroorb.itf.candidates import OverlaySource
    from pyroorb.itf.config import ItfConfig
    feat = {k: None for k in ("tracklet_id", "stn", "provid", "source", "n_obs",
                              "center_ra", "center_dec", "dra_cosdec_rate", "ddec_rate",
                              "sig_pos_e_arcsec", "sig_pos_n_arcsec", "sig_vel_e_degday",
                              "sig_vel_n_degday", "mag_mean", "band")}
    fk = {"feat": dict(feat, tracklet_id="P1", epoch_mid_sec=1000.0),
          "obs": [{"obs80": "line-a"}, {"obs80": "line-b"}]}
    src = OverlaySource(None, [fk], ItfConfig())
    assert [r["tracklet_id"] for r in src.window_rows(0, 2000)] == ["P1"]
    assert src.window_rows(2000, 3000) == []
    assert src.tracklet_obs80(["P1"]) == {"P1": ["line-a", "line-b"]}


# ---------------------------------------------------------------------- the chimera guard


def _obs80(jd_day, stn="F51", desig="TST0001", minute=0):
    """A parseable 80-column line on 2024 Jan ``jd_day`` (+ ``minute``)."""
    frac = f"{(minute / 1440.0):.5f}"[1:]
    return (f"     {desig:<7}  C2024 01 {jd_day:02d}{frac} 10 00 00.00 +10 00 00.0"
            f"          20.0 V      {stn}")


def _tracklet(day, stn="F51", desig="TST0001"):
    return [_obs80(day, stn, desig, m) for m in (0, 15, 30)]


def _shape_problem(lines):
    return sr.caps_problem(*sr.linkage_shape("\n".join(lines)))


def test_guard_arc_caps():
    """Two nights within 4 d and three within 15 d pass; longer, or one night, refuse;
    four nights have no cap."""
    two = _tracklet(2) + _tracklet(5)
    assert sr.linkage_shape("\n".join(two))[0] == 2 and _shape_problem(two) is None
    assert "two nights" in _shape_problem(_tracklet(2) + _tracklet(8))
    assert _shape_problem(_tracklet(2) + _tracklet(9) + _tracklet(16)) is None
    assert "three nights" in _shape_problem(_tracklet(2) + _tracklet(9) + _tracklet(18))
    assert _shape_problem(_tracklet(1) + _tracklet(9) + _tracklet(18) + _tracklet(28)) is None
    assert "single night" in _shape_problem(_tracklet(2))


def test_guard_relaxed_leave_out():
    ok = sr.leaveout_confirmed
    assert ok({"smaj": 0.3, "miss": 1.4, "n_min": 3})            # under the 1.5" floor
    assert ok({"smaj": 4.0, "miss": 7.9, "n_min": 3})            # within 2 sigma
    assert not ok({"smaj": 4.0, "miss": 8.5, "n_min": 3})        # past 2 sigma
    assert not ok({"smaj": 12.0, "miss": 1.0, "n_min": 3})       # prediction says nothing
    assert not ok({"smaj": 0.3, "miss": 0.2, "n_min": 1})        # one obs: no rate
    assert not ok({"error": "leave-out fit failed"})


def test_guard_refusal_names_the_failed_test_and_a_real_date(monkeypatch):
    """A loose prediction is reported as unvouched, not as a miss; the tracklet is dated
    YYYY-MM-DD hh:mm UT, never by a raw JD."""
    assert sr._ut_label(2457415.98) == "2016-01-28 11:31 UT"
    loose = sr._leaveout_why({"smaj": 189.7, "miss": 135.8, "n_min": 3})
    assert "too uncertain" in loose and "not predicted" not in loose and "189.7" in loose
    missed = sr._leaveout_why({"smaj": 4.0, "miss": 8.5, "n_min": 3})
    assert "not predicted" in missed and "8.5" in missed
    assert "single observation" in sr._leaveout_why({"smaj": 0.3, "miss": 0.2, "n_min": 1})
    assert "fit failed" in sr._leaveout_why({"error": "leave-out fit failed"})
    own = _tracklet(1, "G96", "K23D00W") + _tracklet(4, "G96", "K23D00W")
    itf = _tracklet(8, "I41", "ZT00002")
    _measuring(monkeypatch, lambda *a: {"smaj": 189.7, "miss": 135.8, "n_min": 3})
    ok, why = sr.chimera_guard("\n".join(own + itf), "\n".join(itf), {})
    assert not ok and "JD" not in why and " UT)" in why


def _measuring(monkeypatch, fn):
    from pyroorb.itf import leaveout
    monkeypatch.setattr(leaveout, "measure", fn)


def test_guard_result_asks_only_the_added_tracklets(monkeypatch):
    from types import SimpleNamespace
    own = _tracklet(1, "G96", "K23D00W") + _tracklet(9, "G96", "K23D00W")
    itf_a, itf_b = _tracklet(18, "703", "ZT00001"), _tracklet(28, "I41", "ZT00002")
    res = SimpleNamespace(combined_obs80="\n".join(own + itf_a + itf_b),
                          itf_obs80="\n".join(itf_a + itf_b))
    seen = []

    def measure(unit, combined, oc, good=True):
        seen.append(unit[0][77:80])
        return {"smaj": 0.5, "miss": 0.4 if good else 9.0, "n_min": len(unit)}
    guard = lambda r: sr.chimera_guard(r.combined_obs80, r.itf_obs80, {})   # noqa: E731
    _measuring(monkeypatch, measure)
    assert guard(res) == (True, None)
    assert sorted(seen) == ["703", "I41"]                       # every ITF tracklet asked
    _measuring(monkeypatch, lambda u, c, o: measure(u, c, o, good=u[0][77:80] != "I41"))
    ok, why = guard(res)
    assert not ok and "I41" in why and "not predicted" in why
    # the caps refuse before any leave-out fit is made -- and the rule asks the guard of
    # a regular result's ADDED tracklets
    short = SimpleNamespace(combined_obs80="\n".join(_tracklet(1) + _tracklet(9)),
                            itf_obs80="\n".join(_tracklet(9)))
    _measuring(monkeypatch, lambda *a: pytest.fail("no leave-out"))
    (why,) = sr.chimera_warnings_result(short, {})
    assert why.startswith(sr.CHIMERA_GUARD_WARNING) and "two nights" in why


def test_guard_short_seed_asks_the_seed_too(monkeypatch):
    """A short seed vouches for nothing: its own tracklet must be predicted by the rest, so a
    chain of another object's tracklets cannot vouch for itself."""
    seed = _tracklet(1, "G96", "SEED001")
    itf = _tracklet(4, "703", "ZT00001") + _tracklet(9, "I41", "ZT00002")
    seen = []

    def measure(unit, combined, oc):
        seen.append(unit[0][77:80])
        return {"smaj": 0.5, "miss": 9.0 if unit[0][77:80] == "G96" else 0.4,
                "n_min": len(unit)}
    _measuring(monkeypatch, measure)
    (why,) = sr.chimera_warnings_short_seed("\n".join(seed), "\n".join(itf), {})
    assert "G96" in why
    assert "G96" in seen
    assert sr.chimera_warnings_short_seed("\n".join(seed), "", {}) == [
        sr.CHIMERA_GUARD_WARNING + "no ITF tracklet to judge"]


def test_a_grown_handoff_also_judges_the_sr_link_on_its_own(monkeypatch):
    """At 4+ nights the guard is skipped because the search's gates judged the tracklets --
    but the SR link rode in that search's seed, so seed + SR link is judged as a linkage of
    its own too (where, at 1-3 nights, the guard runs)."""
    seed = _tracklet(1, "G96", "SEED001")
    core = _tracklet(2, "703", "ZT00001")
    added = _tracklet(5, "I41", "ZT00002") + _tracklet(9, "F51", "ZT00003")
    judged = []

    def fake(combined, unit_text, obscodes, what="ITF tracklet", unvouched=False):
        n = sr.linkage_shape(combined)[0]
        judged.append((n, unvouched))
        return [f"warning at {n} nights"]
    monkeypatch.setattr(sr, "chimera_warnings", fake)
    out = sr.chimera_warnings_short_seed("\n".join(seed), "\n".join(core + added), {},
                                         core_text="\n".join(core))
    assert judged == [(4, False), (2, True)]          # the whole as usual; the SR link unvouched
    assert out == ["warning at 4 nights", "warning at 2 nights"]
    judged.clear()
    # nothing added beyond the SR link: one judgement, the whole -- which IS that link
    sr.chimera_warnings_short_seed("\n".join(seed), "\n".join(core), {},
                                   core_text="\n".join(core))
    assert judged == [(2, True)]
    judged.clear()
    # no SR link at all (a regular search's find): the usual rule
    sr.chimera_warnings_short_seed("\n".join(seed), "\n".join(core + added), {})
    assert judged == [(4, False)]


def test_sr_then_itf_names_the_link_unvouched_and_never_transmits(monkeypatch):
    link = sr.SrLink(trkids=["T1"], kind="pair", lines={"T1": ["LINE-T1"]}, sr_chi2=1.0,
                     dts=[1.0], fit={})
    res = sr.SrResult()
    res.link = link
    monkeypatch.setattr(sr, "sr_search", lambda *a, **k: res)
    monkeypatch.setattr(sr, "obs80_lines", lambda obs: ["SEEDLINE"])
    got = {}
    import pyroorb.itf.search as S
    monkeypatch.setattr(S, "itf_search", lambda source, **kw: got.update(kw) or "ITF")
    seed = [{"jd_utc": 2460000.5}]
    assert sr.sr_then_itf(seed, object(), {}, ["X"], object(), do_submit=False)[1] == "ITF"
    assert got["unvouched_obs80"] == "LINE-T1" and got["do_submit"] is False
    assert got["chimera_check"] is False
    with pytest.raises(ValueError):
        sr.sr_then_itf(seed, object(), {}, ["X"], object(), do_submit=True)


def test_unvouched_seed_lines_are_not_the_objects_own():
    from types import SimpleNamespace
    from pyroorb.itf import search as S
    ctx = SimpleNamespace(pub_ids={"a", "b", "c"}, unvouched_ids=frozenset({"c"}))
    assert S._own_ids(ctx) == {"a", "b"}
    assert S._own_ids(SimpleNamespace(pub_ids={"a"})) == {"a"}


def test_unacceptable_triplets_never_crowd_out_a_pair(monkeypatch):
    """A triplet over ``trip_chi2_max`` can never be picked, so it is not proposed: six of
    them must not take every engine-fit slot from a pair that passes."""
    ids = [f"T{i}" for i in range(13)]
    monkeypatch.setattr(sr, "seed_attributable", lambda *a: {"t0": 2460000.5, "tdb0": 0.0})
    monkeypatch.setattr(sr, "seed_night_problem", lambda *a: None)
    monkeypatch.setattr(sr, "candidate_arrays", lambda rows, oc: {
        "id": ids, "tdb": [float(i) for i in range(13)]})
    monkeypatch.setattr(sr, "rank", lambda *a: [{"ci": 12, "chi2": 1.0}])
    monkeypatch.setattr(sr, "triplets", lambda *a: [
        {"ci": (2 * i, 2 * i + 1), "chi2": 12.0 + i} for i in range(6)])
    fitted = []

    def fit(seed, cl, oc, max_sec):
        fitted.append(cl[0][:2])
        return {"ok": True, "count": 3 + len(cl), "used": 3 + len(cl), "chi2_dof": 0.5,
                "max_z": 1.0}
    monkeypatch.setattr(sr, "_fit_summary", fit)

    class Source:
        def window_rows(self, a, b):
            return [{}]

        def tracklet_obs80(self, keys, trksubs=None):
            return {k: [f"{k}-a", f"{k}-b"] for k in keys}
    seed = [{"jd_utc": 2460000.5 + 0.01 * i, "obs80_line": "seed"} for i in range(3)]
    res = sr.sr_search(seed, Source(), {}, cfg=sr.SrConfig())
    assert fitted == ["T1"]                    # the pair; no triplet was worth a fit
    assert res.link is not None and res.link.kind == "pair" and res.link.trkids == ["T12"]


@pytest.mark.parametrize("nights", [1, 2])
def test_the_seeds_own_rejection_is_asked_only_of_a_multi_night_seed(monkeypatch, nights):
    bad = ("691", 1)
    monkeypatch.setattr(sr, "seed_attributable", lambda *a: {"t0": 2460000.5, "tdb0": 0.0})
    monkeypatch.setattr(sr, "seed_night_problem", lambda *a: None)
    monkeypatch.setattr(sr, "candidate_arrays", lambda rows, oc: {"id": ["T0"], "tdb": [5.0]})
    monkeypatch.setattr(sr, "rank", lambda *a: [{"ci": 0, "chi2": 1.0}])
    monkeypatch.setattr(sr, "triplets", lambda *a: [])
    monkeypatch.setattr(sr, "_fit_summary", lambda seed, cl, oc, max_sec: {
        "ok": True, "count": 6, "used": 5, "chi2_dof": 0.5, "max_z": 1.0, "dropped": [bad]})
    asked = []
    monkeypatch.setattr(sr, "_seed_rejects",
                        lambda *a: asked.append(1) or frozenset({bad}))

    class Source:
        def window_rows(self, a, b):
            return [{}]

        def tracklet_obs80(self, keys, trksubs=None):
            return {k: [f"{k}-a", f"{k}-b"] for k in keys}
    seed = [{"jd_utc": 2460000.5 + (i // 2) * (nights - 1) + 0.01 * i, "obs80_line": "seed"}
            for i in range(4)]
    res = sr.sr_search(seed, Source(), {}, cfg=sr.SrConfig())
    assert (res.link is not None) == (nights == 2)
    assert asked == ([1] if nights == 2 else [])


def test_guard_short_seed_keeps_every_line_whole(monkeypatch):
    """OBS80 is read by column: the first seed line and the first ITF line keep their five
    leading blank columns, so no observation is lost from a unit or from the leave-out fit
    (a two-observation tracklet read as one is always refused)."""
    seed = _tracklet(1, "G96", "SEED001")
    itf = _tracklet(3, "703", "ZT00001")[:2]
    units, combos = [], []

    def measure(unit, combined, oc):
        units.append(list(unit))
        combos.append(combined)
        return {"smaj": 0.5, "miss": 0.4, "n_min": len(unit)}
    _measuring(monkeypatch, measure)
    ws = sr.chimera_warnings_short_seed("\n" + "\n".join(seed) + "\n",
                                        "\n".join(itf) + "\n\n", {})
    assert ws == []
    assert sorted(len(u) for u in units) == [2, 3]
    assert combos[0].splitlines() == seed + itf
    assert all(ln[:5] == "     " for u in units for ln in u)


def test_a_guard_that_cannot_run_has_not_confirmed(monkeypatch):
    """One rule for every surface: a check that crashes has not confirmed the linkage -- a
    warning saying so, told apart from an objection (``not_run``)."""
    def boom(*a, **k):
        raise RuntimeError("leave-out exploded")
    _measuring(monkeypatch, boom)
    text = "\n".join(_tracklet(1) + _tracklet(3)) + "\n"
    (why,) = sr.chimera_warnings(text, text, {})
    assert why == sr.CHECK_NOT_RUN + "the 2-3 night chimera guard (RuntimeError)"
    assert sr.not_run(why) and "could not run" in why
    assert not sr.not_run(sr.CHIMERA_GUARD_WARNING + "tracklet 2 is 9\" off")


# ------------------------------------------------- one chimera rule, by night count

def _rule(lines, measure_ok=True, scorer_says=None, unvouched=False, no_scorer=False):
    """chimera_warnings on ``lines`` with a stub leave-out and a stub link scorer; returns
    (warnings, guard asked, scorer asked)."""
    from pyroorb.itf import leaveout, linkscore
    asked = {"guard": 0, "scorer": 0}

    def measure(u, c, oc):
        asked["guard"] += 1
        return {"smaj": 0.5, "miss": 0.4 if measure_ok else 9.0, "n_min": len(u)}

    def scorer(text, oc):
        asked["scorer"] += 1
        return scorer_says
    text = "\n".join(lines) + "\n"
    mp = pytest.MonkeyPatch()
    mp.setattr(leaveout, "measure", measure)
    linkscore.set_for_tests([] if no_scorer else [scorer])
    try:
        ws = sr.chimera_warnings(text, text, {}, unvouched=unvouched)
    finally:
        mp.undo()
        linkscore.set_for_tests(())
    return ws, asked["guard"], asked["scorer"]


def test_chimera_rule_two_nights_is_the_guard_alone():
    ws, g, s = _rule(_tracklet(1) + _tracklet(3), measure_ok=False, scorer_says="never asked")
    assert g and not s
    assert len(ws) == 1 and ws[0].startswith(sr.CHIMERA_GUARD_WARNING)
    assert ws[0].startswith(sr.CHIMERA_WARNING)
    assert _rule(_tracklet(1) + _tracklet(3))[0] == []


def test_chimera_rule_three_nights_asks_the_guard_and_the_scorers():
    lines = _tracklet(1) + _tracklet(3) + _tracklet(6)
    ws, g, s = _rule(lines, scorer_says="the model objects")
    assert g and s and ws == [sr.CHIMERA_WARNING + "the model objects"]
    ws, g, s = _rule(lines, measure_ok=False, scorer_says=None)
    assert g and s and len(ws) == 1 and "2-3 night chimera guard" in ws[0]


def test_chimera_rule_four_nights_never_asks_the_guard():
    lines = _tracklet(1) + _tracklet(3) + _tracklet(6) + _tracklet(9)
    ws, g, s = _rule(lines, measure_ok=False, scorer_says=None)
    assert g == 0 and s == 1 and ws == []
    ws, _g, _s = _rule(lines, scorer_says="the model objects")
    assert ws == [sr.CHIMERA_WARNING + "the model objects"]
    # a scorer that cannot score it: still no guard on a linkage the search's gates judged
    from pyroorb.itf import linkscore
    ws, g, _s = _rule(lines, measure_ok=False, scorer_says=linkscore.NO_VERDICT)
    assert g == 0 and ws == []


def test_an_unvouched_four_night_link_no_scorer_can_score_is_reported_unjudged():
    """The SR link rode in the regular search's seed: no search gate judged it.  At 4+
    nights the scorers judge it alone (the one rule); when none can, it is reported
    unjudged -- a not-run warning, never read as clean -- and the 2-3 night guard, which
    kept 27 of 44 true 4+ night links, is never asked."""
    from pyroorb.itf import linkscore
    lines = _tracklet(1) + _tracklet(3) + _tracklet(6) + _tracklet(9)
    ws, g, s = _rule(lines, measure_ok=False, scorer_says=None, unvouched=True)
    assert g == 0 and s == 1 and ws == []                    # a verdict: the scorer alone
    ws, g, s = _rule(lines, measure_ok=False, scorer_says="the model objects", unvouched=True)
    assert g == 0 and ws == [sr.CHIMERA_WARNING + "the model objects"]
    ws, g, s = _rule(lines, measure_ok=False, scorer_says=linkscore.NO_VERDICT,
                     unvouched=True)
    assert g == 0 and s == 1 and ws == [sr.UNJUDGED] and sr.not_run(ws[0])
    ws, g, _s = _rule(lines, measure_ok=False, unvouched=True, no_scorer=True)
    assert g == 0 and ws == [sr.UNJUDGED]
    ws, g, _s = _rule(lines, scorer_says=linkscore.NO_VERDICT)   # a link a search gate judged
    assert g == 0 and ws == []


def test_judge_names_the_scorers_that_gave_a_verdict():
    from pyroorb.itf import linkscore

    def broken(t, o):
        raise RuntimeError("x")

    def cannot(t, o):
        return linkscore.NO_VERDICT

    def clean(t, o):
        return None
    got = []
    objections, not_run = linkscore.judge("x", {}, scorers=[broken, cannot, clean],
                                          verdicts=got)
    assert objections == [] and len(not_run) == 1 and got == ["test_judge_names_the_scorers_"
                                                              "that_gave_a_verdict.<locals>.clean"]


def test_a_link_scorer_that_raises_never_objects_but_has_not_confirmed():
    from pyroorb.itf import linkscore

    def broken(text, oc):
        raise RuntimeError("no model")
    objections, not_run = linkscore.judge("x", {}, scorers=[broken, lambda t, o: "real"])
    assert objections == ["real"]
    assert not_run == ["link scorer test_a_link_scorer_that_raises_never_objects_but_has_not_"
                       "confirmed.<locals>.broken failed (RuntimeError)"]
    # at 4 nights (the scorers alone) it reaches the linkage as a not-run warning, beside
    # the objection
    linkscore.set_for_tests([broken, lambda t, o: "real"])
    text = "\n".join(_tracklet(1) + _tracklet(3) + _tracklet(6) + _tracklet(9)) + "\n"
    assert sr.chimera_warnings(text, text, {}) == [sr.CHIMERA_WARNING + "real",
                                                   sr.CHECK_NOT_RUN + not_run[0]]


def test_a_link_scorer_that_does_not_load_has_not_confirmed(tmp_path, monkeypatch):
    from pyroorb.itf import linkscore
    reg = tmp_path / "scorers.txt"
    reg.write_text("no_such_module_xyz:SCORER\n", encoding="utf-8")
    monkeypatch.setattr(linkscore, "BUILTIN", ())
    monkeypatch.setenv(linkscore.FILE_ENV, str(reg))
    linkscore.set_for_tests(None)                            # discover from the file
    assert linkscore.judge("x", {}) == ([], ["link scorer no_such_module_xyz:SCORER "
                                              "did not load"])


def test_link_scorers_found_through_the_registry_file(tmp_path, monkeypatch):
    import os
    from pyroorb.itf import linkscore
    reg = tmp_path / "scorers.txt"
    reg.write_text("# comment\nos.path:join\nno_such_module_xyz:SCORER\nos.path:join\n",
                   encoding="utf-8")
    monkeypatch.setattr(linkscore, "BUILTIN", ())
    monkeypatch.setenv(linkscore.FILE_ENV, str(reg))
    said = []
    assert linkscore.discover(log=said.append) == [os.path.join]      # once; bad one skipped
    assert said and "no_such_module_xyz" in said[0]


def test_chimera_warnings_result_reuses_the_search_judgement():
    from types import SimpleNamespace
    res = SimpleNamespace(combined_obs80="x", itf_obs80="y",
                          vetting={"chimera_warnings": ["possible chimera: cached"]})
    assert sr.chimera_warnings_result(res, {}) == ["possible chimera: cached"]
    empty = SimpleNamespace(combined_obs80="", itf_obs80="", vetting={})
    assert sr.chimera_warnings_result(empty, {}) == []


def test_a_chimera_check_that_crashes_is_a_warning_not_silence(monkeypatch):
    # an unjudged result must not read as a clean one to any sender
    from pyroorb.itf.search import ItfSearchResult, _attach_chimera_warnings
    from pyroorb.itf.verdict import SearchVerdict

    def boom(result, obscodes):
        raise OSError("station table unreadable")
    monkeypatch.setattr(sr, "chimera_warnings_result", boom)
    res = ItfSearchResult(verdict=SearchVerdict.LINKED, combined_obs80="a\n", itf_obs80="b\n")
    _attach_chimera_warnings(res, {}, log=lambda *a: None)
    assert res.warnings == [sr.CHECK_NOT_RUN + "the chimera check (OSError)"]
    assert res.vetting["chimera_warnings"] == res.warnings


def test_a_real_send_is_judged_before_it_goes_and_a_500_is_not_submitted(monkeypatch):
    from types import SimpleNamespace
    from pyroorb.itf import search as S
    from pyroorb.itf.verdict import SearchVerdict
    said = []
    monkeypatch.setattr(sr, "chimera_warnings_result",
                        lambda r, oc: [sr.CHIMERA_WARNING + "the model objects"])
    monkeypatch.setattr(S, "submit_linkage", lambda doc, cfg, **k: (
        said.append("SEND") or (True, "500 Internal Server Error", None)))
    ctx = SimpleNamespace(do_submit=True, cfg=SimpleNamespace(submit=True), chimera_check=True,
                          obscodes={}, log=said.append)
    res = S.ItfSearchResult(verdict=SearchVerdict.LINKED, combined_obs80="a\n",
                            itf_obs80="b\n")
    S._transmit(ctx, res, {"links": {}}, {})
    # the warning is read before the send, and the send still goes (who asked answers for it)
    assert said == ["WARNING (sent anyway): possible chimera: the model objects", "SEND"]
    assert res.submitted is False                       # the MPC answered, but did not take it


def test_the_mpc_took_it_only_on_a_2xx():
    from pyroorb.itf.linkage import delivered
    assert delivered(True, "200 OK") and delivered(True, "201 Created")
    assert not delivered(True, "500 Internal Server Error")
    assert not delivered(True, "403 Forbidden")
    assert not delivered(False, "dry-run (submission disabled)")
    assert not delivered(True, "")


def test_the_search_attaches_the_chimera_warnings_once():
    from pyroorb.itf import linkscore
    from pyroorb.itf.search import ItfSearchResult, _attach_chimera_warnings
    from pyroorb.itf.verdict import SearchVerdict
    lines = _tracklet(1) + _tracklet(3) + _tracklet(6) + _tracklet(9)
    res = ItfSearchResult(verdict=SearchVerdict.LINKED, combined_obs80="\n".join(lines) + "\n",
                          itf_obs80="\n".join(_tracklet(9)) + "\n", warnings=["seed rms up"])
    linkscore.set_for_tests([lambda t, o: "the model objects"])
    _attach_chimera_warnings(res, {})
    _attach_chimera_warnings(res, {})
    assert res.warnings == ["seed rms up", sr.CHIMERA_WARNING + "the model objects"]
    assert res.vetting["chimera_warnings"] == [sr.CHIMERA_WARNING + "the model objects"]
    none = ItfSearchResult(verdict=SearchVerdict.NONE)
    _attach_chimera_warnings(none, {})
    assert none.warnings == []


# ------------------------------------------- contract pins (night counts, the link's lines)

def test_router_and_linkage_shape_night_boundaries(monkeypatch):
    """Nights are the tree's 12-hour windows (``orbit_common.count_nights``), in both the
    router and the guard -- never a gap rule, which reads an object followed around the
    clock for days as a single night."""
    monkeypatch.setattr(sr, "seed_determined", lambda *a, **k: True)
    o = lambda *jds: [{"jd_utc": j} for j in jds]               # noqa: E731
    assert sr.needs_sr(o(1.0, 1.2, 1.49), "", {})[0]            # one night: SR
    assert not sr.needs_sr(o(1.0, 1.51), "", {})[0]            # two nights: fitted, determined
    assert sr._nights([1.0, 1.51, 2.99]) == 3
    round_the_clock = [1.0 + 0.2 * k for k in range(13)]          # 2.4 d, no gap over 0.2 d
    assert sr._nights(round_the_clock) == 5
    assert not sr.needs_sr(o(*round_the_clock), "", {})[0]
    assert not sr.needs_sr(o(None), "", {})[0]
    one = _tracklet(1)
    assert sr.linkage_shape("\n".join(one)) == (1, pytest.approx(30 / 1440, abs=1e-5))
    assert sr.linkage_shape("\n".join(one + _tracklet(2)))[0] == 2
    sat = [ln[:14] + "s" + ln[15:] for ln in _tracklet(9)]     # position records: ignored
    assert sr.linkage_shape("\n".join(one + sat))[0] == 1
    assert sr.linkage_shape("") == (0, 0.0)


def test_sr_search_hands_back_second_lines_and_db_trksubs(monkeypatch):
    monkeypatch.setattr(sr, "seed_attributable", lambda *a: {"t0": 2460000.5, "tdb0": 0.0})
    monkeypatch.setattr(sr, "seed_night_problem", lambda *a: None)
    monkeypatch.setattr(sr, "candidate_arrays", lambda rows, oc: {
        "id": ["A", "B"], "tdb": [3.0, 5.0]})
    monkeypatch.setattr(sr, "rank", lambda *a: [{"ci": 1, "chi2": 1.0}])
    monkeypatch.setattr(sr, "triplets", lambda *a: [])
    seeds = []

    def fit(seed, cl, oc, max_sec):
        seeds.append(list(seed))
        return {"ok": True, "count": len(seed) + len(cl), "used": len(seed) + len(cl),
                "chi2_dof": 0.5, "max_z": 1.0}
    monkeypatch.setattr(sr, "_fit_summary", fit)

    class Source:
        def window_rows(self, a, b):
            return [{}]

        def tracklet_obs80(self, keys, trksubs=None):
            trksubs.update({f"{k}-a": f"sub{k}" for k in keys})
            return {k: [f"{k}-a", f"{k}-b"] for k in keys}
    seed = [{"jd_utc": 2460000.5 + 0.01 * i, "obs80_line": f"s{i}"} for i in range(2)]
    seed[1]["obs80_line2"] = "s1-second"
    res = sr.sr_search(seed, Source(), {}, cfg=sr.SrConfig())
    assert seeds == [["s0", "s1", "s1-second"]]
    assert res.link.trkids == ["B"] and res.link.lines == {"B": ["B-a", "B-b"]}
    assert res.link.trksubs == {"B-a": "subB"} and res.link.dts == [5.0]


def test_sr_search_never_links_the_seeds_own_detections(monkeypatch):
    """An accepted ITF tracklet stays in the ITF table under its trkSub while it sits on a
    NEOCP object, so a seed holding it can be offered it back: such a proposal is never
    fitted, the next one is."""
    monkeypatch.setattr(sr, "seed_attributable", lambda *a: {"t0": 2460000.5, "tdb0": 0.0})
    monkeypatch.setattr(sr, "seed_night_problem", lambda *a: None)
    monkeypatch.setattr(sr, "candidate_arrays", lambda rows, oc: {
        "id": ["OWN", "NEW"], "tdb": [3.0, 5.0]})
    monkeypatch.setattr(sr, "rank", lambda *a: [{"ci": 0, "chi2": 0.5}, {"ci": 1, "chi2": 1.0}])
    monkeypatch.setattr(sr, "triplets", lambda *a: [])
    fitted = []

    def fit(seed, cl, oc, max_sec):
        fitted.append(cl[0][5:12].strip())
        return {"ok": True, "count": len(seed) + len(cl), "used": len(seed) + len(cl),
                "chi2_dof": 0.5, "max_z": 1.0}
    monkeypatch.setattr(sr, "_fit_summary", fit)
    own, new = _tracklet(4, "703", "ZT00001"), _tracklet(6, "I41", "ZT00002")
    # the seed's copy of its accepted tracklet, re-rendered under the object's own name
    seed_own = [ln[:5] + "P22abc " + ln[12:] for ln in own]

    class Source:
        def window_rows(self, a, b):
            return [{}]

        def tracklet_obs80(self, keys, trksubs=None):
            return {"OWN": own, "NEW": new}
    seed = [{"jd_utc": 2460000.5, "obs80_line": ln} for ln in _tracklet(1, "F51", "P22abc")]
    seed += [{"jd_utc": 2460003.5, "obs80_line": ln} for ln in seed_own]
    res = sr.sr_search(seed, Source(), {}, cfg=sr.SrConfig())
    assert fitted == ["ZT00002"] and res.link.trkids == ["NEW"]
    assert res.proposals[0]["why"] == "the seed's own detections"
