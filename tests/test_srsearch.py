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


def test_short_arc_routes_to_the_single_tracklet_search():
    one = [{"jd_utc": 2460000.5 + k * 0.01} for k in range(4)]
    two_close = one + [{"jd_utc": 2460001.6}]
    two_far = one + [{"jd_utc": 2460004.6}]
    assert sr.short_arc(one) and sr.short_arc(two_close) and not sr.short_arc(two_far)


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


def test_guard_result_asks_only_the_added_tracklets():
    from types import SimpleNamespace
    own = _tracklet(1, "G96", "K23D00W") + _tracklet(9, "G96", "K23D00W")
    itf_a, itf_b = _tracklet(18, "703", "ZT00001"), _tracklet(28, "I41", "ZT00002")
    res = SimpleNamespace(combined_obs80="\n".join(own + itf_a + itf_b),
                          itf_obs80="\n".join(itf_a + itf_b))
    seen = []

    def measure(unit, combined, oc, good=True):
        seen.append(unit[0][77:80])
        return {"smaj": 0.5, "miss": 0.4 if good else 9.0, "n_min": len(unit)}
    assert sr.chimera_guard_result(res, {}, measure=measure) == (True, None)
    assert sorted(seen) == ["703", "I41"]                       # every ITF tracklet asked
    bad = lambda u, c, o: measure(u, c, o, good=u[0][77:80] != "I41")   # noqa: E731
    ok, why = sr.chimera_guard_result(res, {}, measure=bad)
    assert not ok and "I41" in why and "not predicted" in why
    # the caps refuse before any leave-out fit is made
    short = SimpleNamespace(combined_obs80="\n".join(_tracklet(1) + _tracklet(9)),
                            itf_obs80="\n".join(_tracklet(9)))
    ok, why = sr.chimera_guard_result(short, {}, measure=lambda *a: pytest.fail("no leave-out"))
    assert not ok and "two nights" in why


def test_guard_short_seed_asks_the_seed_too():
    """A short seed vouches for nothing: its own tracklet must be predicted by the rest, so a
    chain of another object's tracklets cannot vouch for itself."""
    seed = _tracklet(1, "G96", "SEED001")
    itf = _tracklet(4, "703", "ZT00001") + _tracklet(9, "I41", "ZT00002")
    seen = []

    def measure(unit, combined, oc):
        seen.append(unit[0][77:80])
        return {"smaj": 0.5, "miss": 9.0 if unit[0][77:80] == "G96" else 0.4,
                "n_min": len(unit)}
    ok, why = sr.chimera_guard_short_seed("\n".join(seed), "\n".join(itf), {}, measure=measure)
    assert not ok and "G96" in why
    assert "G96" in seen
    assert sr.chimera_guard_short_seed("\n".join(seed), "", {}, measure=measure)[0] is False
