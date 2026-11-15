"""The sky check (pyroorb.skycheck): conic propagation, catalogue parsing, the quick-pass
margin, tracklet grouping and matching, and an end-to-end check against the fitter's own
prediction path."""
import math

import numpy as np
import pytest

from pyroorb import twobody
from pyroorb.skycheck import catalogue, check, conic
import pyroorb.skycheck as skycheck


# --------------------------------------------------------------------------- conic
def test_conic_matches_the_independent_kepler_propagator_for_every_conic():
    rng = np.random.default_rng(3)
    n = 300
    e = np.concatenate([rng.uniform(0, 0.99, 100), rng.uniform(0.99, 1.01, 100),
                        rng.uniform(1.01, 5, 100)])
    q = rng.uniform(0.05, 40, n)
    T = 2461000.0 + rng.uniform(-20000, 20000, n)
    P, Q = conic.orientation(rng.uniform(0, 180, n), rng.uniform(0, 360, n),
                             rng.uniform(0, 360, n))
    t = 2461300.5
    pos, vel = conic.state_at(q, e, T, P, Q, t)
    for k in range(n):
        vp = math.sqrt(conic.MU_SUN * (1 + e[k]) / q[k])
        s0 = np.concatenate([q[k] * P[k], vp * Q[k]])
        dt = t - T[k]
        if e[k] < 1:                                   # the same whole-revolution reduction
            per = 2 * math.pi * (q[k] / (1 - e[k])) ** 1.5 / math.sqrt(conic.MU_SUN)
            dt -= per * round(dt / per)
        ref = np.asarray(twobody.kepler_propagate(s0, dt))
        assert np.linalg.norm(pos[k] - ref[:3]) / np.linalg.norm(ref[:3]) < 1e-9
        assert np.linalg.norm(vel[k] - ref[3:]) / np.linalg.norm(ref[3:]) < 1e-9


def test_elliptic_elements_and_angles_invert_state_and_orientation():
    rng = np.random.default_rng(5)
    n = 500
    e = rng.uniform(0, 0.97, n)
    q = rng.uniform(0.2, 30, n)
    T = 2461000 + rng.uniform(-5e4, 5e4, n)
    inc, node, peri = rng.uniform(1, 179, n), rng.uniform(0, 360, n), rng.uniform(0, 360, n)
    P, Q = conic.orientation(inc, node, peri)
    t = 2461200.5
    p, v = conic.state_at(q, e, T, P, Q, t)
    q2, e2, T2, P2, Q2 = conic.elliptic_elements(p, v, t)
    p1, _ = conic.state_at(q, e, T, P, Q, t + 37.0)
    p2, _ = conic.state_at(q2, e2, T2, P2, Q2, t + 37.0)
    assert np.max(np.linalg.norm(p2 - p1, axis=1) / np.linalg.norm(p1, axis=1)) < 1e-9
    i2, n2, w2 = conic.angles(P, Q)
    assert np.allclose(i2, inc, atol=1e-9)
    assert np.allclose(((n2 - node + 180) % 360) - 180, 0, atol=1e-8)
    assert np.allclose(((w2 - peri + 180) % 360) - 180, 0, atol=1e-8)


def test_a_hyperbola_is_not_given_elliptic_elements():
    q, e, T = np.array([1.0]), np.array([1.5]), np.array([2461000.0])
    P, Q = conic.orientation([10.0], [20.0], [30.0])
    p, v = conic.state_at(q, e, T, P, Q, 2461010.0)
    assert np.isnan(conic.elliptic_elements(p, v, 2461010.0)[0][0])


# --------------------------------------------------------------------------- parsing
CERES = ("00001    3.34  0.15 K2669 145.40355   73.28536   80.24900   10.58788  0.0796897  "
         "0.21424022   2.7660466  0 MPO998000  7451 128 1801-2026 0.80 M-v 30k MPCLINUX   0000"
         "      (1) Ceres              20260510")
HALE_BOPP = ("    CJ95O010  1997 03 29.0285  0.926983  0.994893  130.7397  281.8159   89.7304  "
             "20261114  -2.0  4.0  C/1995 O1 (Hale-Bopp)                                    "
             "MPC194091")
ATLAS3I = ("0003I         2025 10 29.4825  1.356507  6.139884  128.0055  322.1535  175.1129  "
           "20251121   8.0  4.0  3I/ATLAS                                                 "
           "MPEC 2026-G41")


def test_mpcorb_line_is_parsed_into_perihelion_form():
    desig, name, c = catalogue.parse_mpcorb_line(CERES)
    assert (desig, name) == ("1", "(1) Ceres")
    assert c["epoch"] == catalogue.unpack_epoch("K2669")
    assert c["q"] == pytest.approx(2.7660466 * (1 - 0.0796897))
    assert c["U"] == 0.0 and c["H"] == 3.34
    # M = 145.4 deg is past aphelion's side: the nearest perihelion is ahead of the epoch
    n = math.degrees(math.sqrt(conic.MU_SUN / 2.7660466 ** 3))
    assert c["T"] == pytest.approx(c["epoch"] + (360 - 145.40355) / n, abs=1e-6) or \
        c["T"] == pytest.approx(c["epoch"] - 145.40355 / n, abs=1e-6)


def test_comet_lines_keep_hyperbolic_orbits():
    d, name, c = catalogue.parse_comet_line(HALE_BOPP)
    assert d == "C/1995 O1" and name == "C/1995 O1 (Hale-Bopp)"
    assert c["e"] == pytest.approx(0.994893) and c["G"] == 4.0
    d, _, c = catalogue.parse_comet_line(ATLAS3I)
    assert d == "3I/ATLAS" and c["e"] > 6


def test_packed_epochs():
    assert catalogue.unpack_epoch("K26BE") == pytest.approx(2461358.5)     # 2026 Nov 14
    assert catalogue.unpack_epoch("J9611") == pytest.approx(2450083.5)     # 1996 Jan 1
    assert catalogue.unpack_epoch("bad") is None


def test_neocp_orbit_rows_are_read_with_or_without_h():
    from pyroorb.skycheck import neocp
    text = ("Object   H     G    Epoch    M         Peri.      Node       Incl.        e"
            "           n         a                     NObs NOpp   Arc    r.m.s.       Orbit ID\n"
            "P22slHJ 23.4  0.15  K26BE  28.31167  230.32279   41.85846   52.70989  0.8183129  "
            "0.51570483   1.5400515                  3   1    0 days 0.12         NEOCPNomin\n"
            "P22slHJ       0.15  K26BE   0.11961  231.41701   46.25183   79.10004  0.9959018  "
            "0.00305433  47.0463025                  3   1    0 days 0.06         NEOCPV0001\n")
    orbs = neocp.parse_orbits(text)
    assert [o["orbit_id"] for o in orbs] == ["NEOCPNomin", "NEOCPV0001"]
    assert orbs[0]["H"] == 23.4 and math.isnan(orbs[1]["H"])
    cat = neocp.as_catalogue({"P22slHJ": orbs})
    assert len(cat) == 2 and cat.designation(1) == "P22slHJ"


def test_times_and_angles_in_every_accepted_spelling():
    jd = skycheck.julian_date("2026-03-20T02:30:30.653")
    assert skycheck.julian_date("2026 03 20.104522") == pytest.approx(jd, abs=1e-6)
    assert skycheck.julian_date(jd) == jd
    assert skycheck.angle_deg("04 09 32.4", hours=True) == pytest.approx(62.385)
    assert skycheck.angle_deg("-05:06:07") == pytest.approx(-(5 + 6 / 60 + 7 / 3600))
    assert skycheck.angle_deg(12.5) == 12.5
    ra, dec = skycheck.sexagesimal(62.385, -5.1019444)
    assert ra == "04 09 32.40" and dec == "-05 06 07.0"


# --------------------------------------------------------------------------- the margin
def test_the_margin_grows_with_time_from_the_epoch_and_never_falls():
    dts = np.linspace(0, 900, 400)
    m = check.relative_margin(dts)
    assert m[0] == 0.0
    assert np.all(np.diff(m) >= 0)
    assert check.relative_margin(-120.0) == check.relative_margin(120.0)
    assert check.relative_margin(450.0) == pytest.approx(0.14)


# --------------------------------------------------------------------------- tracklets
def _rec(jd, code, desig=None, trksub=None):
    return {"jd_utc": jd, "obscode": code, "designation": desig, "trksub": trksub,
            "ra_deg": 10.0, "dec_deg": 0.0}


def test_tracklets_split_by_station_id_object_and_time():
    recs = [_rec(1.00, "G96", "A"), _rec(1.01, "G96", "A"), _rec(1.02, "G96", "B"),
            _rec(1.03, "F51", "A"), _rec(1.50, "G96", "A"),                # 12 h later
            _rec(1.04, "G96", None, "t1"), _rec(1.70, "G96", None, "t1")]  # one trkSub
    groups = skycheck.tracklets(recs)
    as_sets = sorted(sorted(g) for g in groups)
    assert as_sets == [[0, 1], [2], [3], [4], [5, 6]]


def test_one_clear_match_is_chosen_two_close_ones_are_ambiguous():
    m = lambda d, off: {"designation": d, "mean_offset": off}
    assert skycheck._choose([]) == (None, False)
    assert skycheck._choose([m("a", 0.4)]) == ("a", False)
    assert skycheck._choose([m("a", 0.4), m("b", 12.0)]) == ("a", False)
    assert skycheck._choose([m("a", 0.4), m("b", 1.0)]) == (None, True)
    assert skycheck._choose([m("a", 5.0), m("b", 9.0)]) == (None, True)


def test_the_public_wrappers_defer_to_the_engine_defaults(monkeypatch):
    import pyroorb
    seen = {}
    monkeypatch.setattr(skycheck, "identify", lambda obs, **kw: seen.update(id=kw))
    monkeypatch.setattr(skycheck, "check_field", lambda *a, **kw: seen.update(f=kw))
    pyroorb.identify_tracklets([], refresh=False)
    assert not {"radius_arcmin", "match_arcsec", "motion_arcsec"} & set(seen["id"])
    pyroorb.check_field(1.0, 2.0, 2461000.5, refresh=False)
    assert "radius_arcmin" not in seen["f"]
    pyroorb.identify_tracklets([], motion_arcsec=7.0, refresh=False)
    assert seen["id"]["motion_arcsec"] == 7.0


def test_the_web_page_takes_its_defaults_from_the_engine():
    from pathlib import Path
    from pyroorb.skycheck import webaddon
    d = webaddon.status()["defaults"]
    assert (d["radius"], d["match"], d["motion"]) == (
        skycheck.DEFAULT_RADIUS_ARCMIN, skycheck.MATCH_ARCSEC, skycheck.MOTION_ARCSEC)
    html = (Path(webaddon.__file__).parent / "static" / "skycheck.html").read_text("utf-8")
    for field in ("fRadius", "iRadius", "iMatch", "iMotion"):
        tag = html[html.index(f'id="{field}"') - 40:html.index(f'id="{field}"') + 40]
        assert "value=" not in tag, f"{field} hard-codes a default"


# --------------------------------------------------------------------------- end to end
def _one_orbit_catalogue():
    """(1) Ceres-like and a second object far away, as an in-memory catalogue."""
    rows = []
    for d, node in (("1", 80.249), ("2", 260.0)):
        a, e, M, epoch = 2.766, 0.0797, 145.4, 2461200.5
        rows.append((d, f"({d}) test", {
            "q": a * (1 - e), "e": e, "i": 10.588, "node": node, "peri": 73.285,
            "T": float(conic.perihelion_time(a, e, M, epoch)), "epoch": epoch,
            "H": 3.34, "G": 0.15, "U": 0.0, "nopp": 128.0}, catalogue.KIND_ASTEROID))
    return catalogue.Catalogue.from_rows(rows)


def _truth_radec(cat, k, t_utc, obscode):
    """The object as the FITTER predicts it (force_assist.predict_radec_batch: its own
    light-time and observer path), for station ``obscode`` at UTC JD ``t_utc``."""
    from astropy.time import Time
    from pyroorb import force_assist, obs_sources
    epoch = float(cat.cols["epoch"][k])
    p, v = conic.state_at(cat.cols["q"][k:k + 1], cat.cols["e"][k:k + 1],
                          cat.cols["T"][k:k + 1], cat.P[k:k + 1], cat.Q[k:k + 1], epoch)
    R = obs_sources.observer_positions_batch([t_utc], [obscode], obs_sources.load_obscodes())
    tdb = Time(t_utc, format="jd", scale="utc").tdb.jd
    out = force_assist.predict_radec_batch(None, epoch, np.concatenate([p[0], v[0]]),
                                           [tdb], R, asteroids=False)
    return float(out[0, 0]), float(out[0, 1])


def test_a_field_finds_the_object_where_the_fitter_puts_it(monkeypatch):
    cat = _one_orbit_catalogue()
    monkeypatch.setattr(catalogue, "load", lambda **kw: cat)
    t = 2461150.3
    ra, dec = _truth_radec(cat, 0, t, "G96")
    res = skycheck.check_field(ra, dec, t, radius_arcmin=5, obscode="G96", neocp=False,
                               refresh=False)
    assert [o["designation"] for o in res.objects] == ["1"]
    assert res.objects[0]["separation"] < 0.01                  # arcsec
    assert 6 < res.objects[0]["mag"] < 10


def test_a_tracklet_is_identified_by_all_its_positions(monkeypatch):
    cat = _one_orbit_catalogue()
    monkeypatch.setattr(catalogue, "load", lambda **kw: cat)
    times = [2461150.30, 2461150.32, 2461150.34]
    recs = []
    for t in times:
        ra, dec = _truth_radec(cat, 0, t, "F51")
        recs.append({"jd_utc": t, "ra_deg": ra, "dec_deg": dec, "obscode": "F51",
                     "designation": "trk1"})
    res = skycheck.identify(recs, neocp=False, refresh=False)
    assert len(res) == 1 and res[0].identification == "1"
    assert res[0].candidates[0]["motion_rms"] < 0.01
    # the same positions shifted in time by an hour no longer follow the object's motion
    moved = [dict(r, jd_utc=r["jd_utc"] + (k - 1) * 0.04) for k, r in enumerate(recs)]
    res = skycheck.identify(moved, neocp=False, refresh=False)
    assert res[0].identification is None


def test_a_dual_status_object_is_identified_not_ambiguous(monkeypatch):
    """(118401) = 176P/LINEAR: one orbit in MPCORB and CometEls alike; the tracklet is the
    asteroid, the comet row marked as the same body."""
    base = _one_orbit_catalogue()
    rows = []
    for d, kind in (("118401", catalogue.KIND_ASTEROID), ("176P/LINEAR", catalogue.KIND_COMET)):
        rows.append((d, d, {c: float(base.cols[c][0]) for c in catalogue.FLOAT_COLS}, kind))
    cat = catalogue.Catalogue.from_rows(rows)
    monkeypatch.setattr(catalogue, "load", lambda **kw: cat)
    recs = []
    for t in (2461150.30, 2461150.32, 2461150.34):
        ra, dec = _truth_radec(cat, 0, t, "F51")
        recs.append({"jd_utc": t, "ra_deg": ra, "dec_deg": dec, "obscode": "F51",
                     "designation": "trk1"})
    res = skycheck.identify(recs, neocp=False, refresh=False)
    assert res[0].identification == "118401" and not res[0].ambiguous
    assert [c.get("same_as") for c in res[0].candidates if c["kind"] == "comet"] == ["118401"]


def test_a_fast_movers_motion_allowance_is_a_fraction_of_its_rate():
    """60"/min over an hour: positions whose rate is 20% off match, a stationary source at the
    middle of the predicted track (rate 100% off) does not."""
    cat = _one_orbit_catalogue()
    t = 2461150.3 + np.linspace(0, 1 / 24, 25)    # dense: the old track-length allowance
                                                    # (1084") let the stationary one (1082") in
    track = 60.0 * (t - t.mean()) * 1440.0 / 3600.0               # deg along RA at dec 0
    p = check.Prediction(0, 10.0 + track, np.zeros_like(t), 10.0, 0.0, 60.0, 0.0,
                         2.0, 1.0, 10.0, 150.0, 15.0)

    def cand(ra_obs):
        qy = check.Query(t, ra_obs, np.zeros_like(t), "500", 1.0)
        qy.centre = check.unit(10.0, 0.0)
        return skycheck._tracklet_candidate(cat, p, qy, "asteroid", 30.0, 4.0)
    assert cand(10.0 + 0.8 * track)["match"]
    assert not cand(np.full_like(t, 10.0))["match"]


def test_a_throttled_neocp_refresh_keeps_the_last_good_copy(monkeypatch, tmp_path):
    from pyroorb.skycheck import neocp
    monkeypatch.setattr(neocp, "cache_path", lambda: tmp_path / "neocp.json")
    (tmp_path / "neocp.json").write_text('{"fetched": 0, "objects": {"P1": [{"x": 1}]}}')
    monkeypatch.setattr(neocp, "_get", lambda url, params=None, timeout=60: "P1\nP2\n")
    monkeypatch.setattr(neocp, "_fetch_one", lambda tid: (tid, []))   # every page fails
    assert neocp.fetch() == {"P1": [{"x": 1}]}
    assert neocp.cached() == {"P1": [{"x": 1}]}
    monkeypatch.setattr(neocp, "_fetch_one", lambda tid: (tid, [{"y": 2}]))  # recovered
    assert neocp.fetch() == {"P1": [{"y": 2}], "P2": [{"y": 2}]}
    assert neocp.cached() == {"P1": [{"y": 2}], "P2": [{"y": 2}]}


def test_without_refresh_the_neocp_is_read_from_the_cache_only(monkeypatch):
    from pyroorb.skycheck import neocp
    orb = {"tid": "P1", "H": 22.0, "G": 0.15, "epoch": 2461000.5, "M": 10.0, "peri": 30.0,
           "node": 40.0, "i": 5.0, "e": 0.3, "a": 1.4, "orbit_id": "NEOCPNomin"}
    monkeypatch.setattr(neocp, "fetch", lambda **kw: 1 / 0)          # the network: never
    monkeypatch.setattr(neocp, "cached", lambda: {"P1": [orb]})
    import time
    now = time.time() / 86400.0 + 2440587.5
    cat = skycheck._neocp_catalogue([now - 1.0], True, refresh=False)
    assert cat is not None and cat.designation(0) == "P1"
    # only when asked for, and only for recent times (a NEOCP orbit is of the last few days)
    assert skycheck._neocp_catalogue([now - 1.0], False, refresh=False) is None
    assert skycheck._neocp_catalogue([now - 1.0, now - 40.0], True, refresh=False) is None


def test_the_neocp_is_checked_only_when_asked_for():
    import inspect
    import pyroorb
    for f in (skycheck.check_field, skycheck.identify, pyroorb.check_field,
              pyroorb.identify_tracklets):
        assert inspect.signature(f).parameters["neocp"].default is False
