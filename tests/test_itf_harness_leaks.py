"""A planted tracklet must look EXACTLY like the index row a real one would
have -- any feature that differs is a label every harness-trained ranker can
read (measured: n_in_corridor median 0 for plants vs 470
for real rows; weighting sigma in place of the reported rms on 68% of 112
ex-ITF tracklets)."""
from pyroorb.itf.candidates import OverlaySource
from pyroorb.itf.config import ItfConfig
from pyroorb.itf.harness import fake_from_run
from pyroorb.itf.index import IndexConfig, compute_tracklet_features
from pyroorb.itf.obs80_fields import unix_from_jd

from test_itf_search import INDEX_CFG, T0, _rows, _window, _obs80


def _pub_obs(rms):
    """Published-obs dicts (the fetch's shape) for the _rows() tracklet, whose
    WEIGHTING sigma (sigma_ra) is deliberately not its reported rms."""
    out = []
    for r in _rows():
        out.append({"jd_utc": r["es"] / 86400.0 + 2440587.5,
                    "ra_deg": r["ra"], "dec_deg": r["dec"], "obscode": "F51",
                    "obs80_line": _obs80("K20A00A", "2020 09 13.32",
                                         "10 00 00.00", "+10 00 00.0"),
                    "sigma_ra": 0.5, "sigma_dec": 0.5,
                    "ades_rms_ra": rms, "ades_rms_dec": rms,
                    "mag": r["mag"], "band": r["band"]})
    return out


def test_a_plant_carries_the_reported_rms_not_the_weighting_sigma():
    for rms in (None, 0.8):
        fk, why = fake_from_run(_pub_obs(rms), "PR00001", 0, INDEX_CFG)
        assert fk is not None, why
        rows = _rows()
        for r in rows:
            r["rms_ra"] = r["rms_dec"] = rms
        want = compute_tracklet_features(rows, INDEX_CFG)
        for k in ("sig_pos_e_arcsec", "sig_vel_e_degday", "rate_sigma",
                  "fit_chi2_dof"):
            assert abs(fk["feat"][k] - want[k]) <= 1e-9 * max(1, abs(want[k])), \
                (rms, k, fk["feat"][k], want[k])


def test_a_plant_counts_in_its_corridor_population():
    """Alone in a window a plant used to read n_in_corridor 0 -- a value no
    real candidate can have."""
    cfg = ItfConfig()
    feat = compute_tracklet_features(_rows(), INDEX_CFG)
    obs = [{"obs80": _obs80("PR00001", "2020 09 13.32", "10 00 00.00",
                            "+10 00 00.0"),
            "ra": 150.0, "dec": 10.0, "epoch_sec": T0, "trksub": "PR00001",
            "stn": "F51", "trkid": "PR00001"}]
    src = OverlaySource(None, [{"feat": feat, "obs": obs}], cfg, INDEX_CFG)
    (_w, _rows_, _chi2, feats), = src.sweep_candidates([_window()], 1.1, 0.0,
                                                       True)
    assert feats["PR00001"]["_n_in_corridor"] == 1
