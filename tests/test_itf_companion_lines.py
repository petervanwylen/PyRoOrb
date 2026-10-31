"""Two-line (space-based / roving) records in the ITF search: the `s`/`v`
companion line rides with its observation but is not one.

A NEOWISE (C51) tracklet of 8 observations is 16 lines.  Counting lines read
it as 8/16 = 50% retained, failed it, and sent the linkage to the leave-out
repair, which parsed the `s` line as a position and crashed the search with
``could not convert string to float: ' +'``."""
from pyroorb.itf import leaveout
from pyroorb.itf.obs80_fields import is_companion, n_observations, obs_ids, obs_unique_id

S1 = "     K14T33J  S2014 11 04.93749521 13 55.307-23 31 24.94               L~6EngC51"
S2 = "     K14T33J  s2014 11 04.9374951 + 4982.6823 - 3872.7701 - 2751.1299        C51"
G1 = "     K14T33J  C2021 03 23.43781 14 33 55.21 -04 12 26.0          21.0 GV~4i4fG96"
G2 = "     K14T33J  C2021 03 23.44296 14 33 53.66 -04 12 26.4          20.9 GV~4i4fG96"
V2 = "     K14T33J  v2014 11 04.937495 1 284.8340 +35.12345    123                 247"

#: ground stations with parallax constants; C51 and 247 (rho 0) cannot be placed
OBSCODES = {"G96": (249.21128, 0.845111, 0.533614), "247": (0.0, 0.0, 0.0)}


def test_companion_lines_are_recognised_from_their_id():
    assert is_companion(obs_unique_id(S2))
    assert is_companion(obs_unique_id(V2))
    for line in (S1, G1, G2):
        assert not is_companion(obs_unique_id(line))


def test_a_satellite_tracklet_counts_its_observations_not_its_lines():
    ids = obs_ids("\n".join([S1, S2, G1, G2]))
    assert len(ids) == 4                  # the companion stays in the id set
    assert n_observations(ids) == 3


def test_leaveout_dates_a_companion_line_like_its_observation():
    assert abs(leaveout._jd(S2) - leaveout._jd(S1)) < 1e-6
    assert leaveout._observations([S2, G2, S1, G1]) == [S1, G1, G2]


def test_influence_of_a_satellite_tracklet_is_untestable_not_a_crash():
    m = leaveout.influence([S1, S2], None, None, OBSCODES)
    assert "C51" in m["error"]


def test_measure_of_a_satellite_tracklet_is_untestable_not_a_crash(monkeypatch):
    import pyroorb.itf.fitter; from pyroorb import itf
    ok = type("Fit", (), {"ok": True, "fit": object(), "error": None})()
    monkeypatch.setattr(itf.fitter, "fit_astrometry", lambda *a, **k: ok)
    m = leaveout.measure([S1, S2], "\n".join([S1, S2, G1, G2]), OBSCODES)
    assert m["confirmed"] is False and "C51" in m["error"]


def test_a_final_sweep_satellite_tracklet_is_one_tracklet_with_its_companions():
    """The final sweep used to take a tracklet's ids from its DB rows, where a
    two-line record is ONE 160-column row -- so only the observation line was
    registered, while the appended astrometry carries both lines.  The
    retention gate then read the companion as a tracklet of its own with zero
    observations, 0% retained, and failed it."""
    from types import SimpleNamespace
    import pyroorb.itf.search as S
    rows = [{"obs80": S1 + S2}]
    t = {"rows": rows, "astrometry": S._astro_text(rows), "trksub": "K14T33J",
         "stn": "C51", "obs_count": 1}
    ids = S._tracklet_ids(t)
    assert ids == obs_ids(S1 + "\n" + S2) and n_observations(ids) == 1
    ctx = SimpleNamespace(swept_tracklets={i: ("C51 tracklet", ids) for i in ids})
    groups = S._linked_tracklets_as_swept(ctx, t["astrometry"])
    assert groups == [("C51 tracklet", ids)], groups


def test_named_tracklets_reads_past_a_companion_line():
    text = f"photometric outlier on C51 2014-11-04"
    leaveout.named_tracklets(text, "\n".join([S1, S2]))   # must not raise
