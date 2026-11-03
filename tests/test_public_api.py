"""The supported Python API (``import pyroorb``): read or fetch observations, fit, check
determinacy, ephemeris, export. Offline: the inputs are tracked files and frozen fixtures."""
import os
import sys
import warnings

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyroorb  # noqa: E402
from _fixtures import fixture_obs  # noqa: E402

D2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digest2", "fixtures")
OBS80 = os.path.join(D2, "sample.obs")
PSV = os.path.join(D2, "sample.psv")


def test_read_observations_from_a_path_or_text_in_either_format():
    a = pyroorb.read_observations(OBS80)
    assert isinstance(a, pyroorb.Observations) and len(a) == 3
    assert a.meta["format"] == "OBS80"
    with open(OBS80) as fh:
        b = pyroorb.read_observations(fh.read())
    assert [o["jd_utc"] for o in b] == [o["jd_utc"] for o in a]
    p = pyroorb.read_observations(PSV)
    assert p.meta["format"] == "ADES" and len(p) == 3
    assert pyroorb.read_observations(OBS80, format="obs80").meta["format"] == "OBS80"


def test_times_accept_iso_jd_and_astropy():
    from astropy.time import Time
    jd = pyroorb._to_jd("2026-01-01T00:00:00")
    assert jd == pytest.approx(2461041.5)
    assert pyroorb._to_jd(jd) == jd
    assert pyroorb._to_jd(Time(jd, format="jd", scale="utc")) == pytest.approx(jd)


def test_an_unknown_export_format_is_refused():
    with pytest.raises(ValueError, match="mpcorb"):
        pyroorb.export(object(), "votable")


@pytest.mark.slow
def test_a_single_tracklet_is_flagged_not_refused():
    obs = pyroorb.read_observations(OBS80)
    with pytest.warns(pyroorb.UndeterminedOrbitWarning):
        fit = pyroorb.fit_orbit(obs)
    assert fit.is_determined is False
    info = fit.determinacy()
    assert info and info["headline"]
    assert {"a", "e", "i"} <= set(fit.elements())      # the nominal solution is returned


@pytest.mark.slow
def test_a_determined_orbit_fits_ephemerides_and_exports():
    records, meta = fixture_obs("2023_QJ3")
    obs = pyroorb.Observations(records, meta)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyroorb.UndeterminedOrbitWarning)
        fit = pyroorb.fit_orbit(obs, designation="2023 QJ3")
    assert fit.is_determined and fit.determinacy() is None
    assert fit.n_observations() == 20 and fit.rms_residual() < 0.5
    rows = pyroorb.ephemeris(fit, "2026-11-01", "2026-11-03", step=1.0, site="568",
                             full=False)
    assert len(rows) == 3
    for fmt in ("mpcorb", "sof", "eightline", "covar"):
        text = pyroorb.export(fit, fmt)
        assert isinstance(text, str) and text.strip(), fmt
    with pytest.raises(ValueError, match="elements"):
        fit.covariance_matrix("ecliptic")          # used to hand back the state matrix silently
    assert fit.covariance_matrix("state").shape == (6, 6)
    assert fit.covariance_matrix().shape == (11, 11)


def test_load_settings_reads_the_env_file_without_overriding(tmp_path, monkeypatch):
    from pyroorb import envfile
    (tmp_path / ".env").write_text("PYROORB_TEST_SETTING=from-file\nPYROORB_TEST_SET=from-file\n")
    monkeypatch.setenv("PYROORB_TEST_SET", "from-shell")
    monkeypatch.delenv("PYROORB_TEST_SETTING", raising=False)
    monkeypatch.delenv("PYROORB_ENV_FILE", raising=False)
    monkeypatch.setitem(envfile._LOADED, "done", False)
    monkeypatch.setattr(envfile, "candidates", lambda cwd=None: [tmp_path / ".env"])
    assert pyroorb.load_settings(tmp_path) == tmp_path / ".env"
    assert os.environ["PYROORB_TEST_SETTING"] == "from-file"
    assert os.environ["PYROORB_TEST_SET"] == "from-shell"        # a real variable wins


def test_select_cuts_by_time_and_station_and_carries_meta():
    records, meta = fixture_obs("162385")                        # optical + radar
    obs = pyroorb.Observations(records, meta)
    times = sorted(o["jd_utc"] for o in obs)
    mid = times[len(times) // 2]
    early = obs.select(stop=mid)
    assert isinstance(early, pyroorb.Observations)
    assert len(early) == sum(t < mid for t in times)
    assert all(r["jd_utc"] < mid for r in early.meta["radar"])
    assert early.meta.get("designation") == obs.meta.get("designation")
    assert obs.select(radar=False).meta["radar"] == [] and len(obs.select(radar=False)) == len(obs)
    station = obs[0]["obscode"]
    assert {o["obscode"] for o in obs.select(stations=[station])} == {station}
    assert station not in {o["obscode"] for o in obs.select(exclude_stations=[station])}
    assert len(obs.select(start="1900-01-01", stop="2200-01-01")) == len(obs)


def test_digest2_scores_need_no_fit():
    one = pyroorb.read_observations(OBS80)
    d2 = pyroorb.digest2_scores(one)
    assert d2["n_obs"] == 3 and set(d2["raw"]) == set(d2["noid"]) == set(d2["classes"])
    assert all(0.0 <= s <= 100.0 for s in d2["raw"].values())
    assert d2["classes"]["NEO"].startswith("NEO")
    per = pyroorb.digest2_scores(one, per_tracklet=True)
    assert len(per) == 1 and per[0]["indices"] == [0, 1, 2]
    assert pyroorb.digest2_scores(one[:1]) is None                # one observation: declined
    # the function's name must survive the vendored subpackage's import (pyroorb/digest2/)
    import importlib
    importlib.import_module("pyroorb.digest2.classes")
    assert callable(pyroorb.digest2_scores)


def test_fit_options_reach_the_engine_unchanged(monkeypatch):
    """nongrav / center / detect_satellite / mpc_seed are spellings of the engine's own
    arguments; the defaults must call it exactly as before they existed."""
    from pyroorb import fit_object
    calls = []

    class _Fit:
        is_determined = True

    def fake_fit(obs, obscodes, **kw):
        calls.append(kw)
        return _Fit()

    monkeypatch.setattr(fit_object, "fit_orbit", fake_fit)
    monkeypatch.setattr(fit_object, "fetch_mpc_seed", lambda obs, meta: {"seed": meta["designation"]})
    obs = pyroorb.read_observations(OBS80)
    pyroorb.fit_orbit(obs, obscodes={})
    assert calls[-1]["auto_nongrav"] is None and calls[-1]["center"] is None
    assert calls[-1]["auto_center"] is None and calls[-1]["mpc_seed"] is None
    assert calls[-1]["force"] is False
    # the brute-force linker is not public (it can glue two objects into one clean-looking
    # orbit); the command line's --force keeps it
    with pytest.raises(TypeError):
        pyroorb.fit_orbit(obs, obscodes={}, force=True)
    pyroorb.fit_orbit(obs, obscodes={}, nongrav="none", detect_satellite=False)
    assert calls[-1]["auto_nongrav"] is False and calls[-1]["auto_center"] is False
    pyroorb.fit_orbit(obs, "X", obscodes={}, center="Jupiter")
    assert calls[-1]["center"] == "jupiter"
    pyroorb.fit_orbit(obs, "X", obscodes={}, mpc_seed=True)
    assert calls[-1]["mpc_seed"] == {"seed": "X"}
    with pytest.raises(ValueError, match="nongrav"):
        pyroorb.fit_orbit(obs, obscodes={}, nongrav="outgassing")
    with pytest.raises(ValueError, match="center"):
        pyroorb.fit_orbit(obs, obscodes={}, nongrav="comet2", center="jupiter")


def test_a_forced_model_is_the_notebooks_pipeline(monkeypatch):
    """nongrav="<model>" = gravity fit -> refine_nongrav(model, kept regardless) -> the radar
    re-joined: what the Bennu and Encke notebooks did by hand before `nongrav=` existed
    (bit-identical on Bennu to 2012)."""
    from pyroorb import fit_object, radar
    seen = []

    class _Fit:
        is_determined = True
        asteroids = False

    monkeypatch.setattr(fit_object, "fit_orbit", lambda obs, oc, **kw: seen.append(("fit", kw["auto_nongrav"])) or _Fit())
    monkeypatch.setattr(fit_object, "refine_nongrav",
                        lambda f, model, revert_if_worse, asteroids: seen.append(("ng", model, revert_if_worse)) or f)
    monkeypatch.setattr(radar, "refine_radar", lambda f, rows, oc: seen.append(("radar", len(rows))) or f)
    obs = pyroorb.read_observations(OBS80)
    pyroorb.fit_orbit(obs, obscodes={}, nongrav="yarkovsky")
    assert seen == [("fit", False), ("ng", "yarkovsky", False)]
    seen.clear()
    with_radar = pyroorb.Observations(obs, {**obs.meta, "radar": [{"jd_utc": 0.0}]})
    pyroorb.fit_orbit(with_radar, obscodes={}, nongrav="comet2")
    assert seen == [("fit", False), ("ng", "comet2", False), ("radar", 1)]


def test_asteroid_perturbers_switch_restores_and_refuses_without_kernel(monkeypatch):
    import pyroorb.force_assist as fa
    monkeypatch.setattr(fa, "ASTEROIDS_PATH", None)
    monkeypatch.setattr(fa, "asteroid_kernel_path", lambda: "/kernel.bsp")
    with pyroorb.asteroid_perturbers():
        assert fa.ASTEROIDS_PATH == "/kernel.bsp"
        with pyroorb.asteroid_perturbers(False):
            assert fa.ASTEROIDS_PATH is None
        assert fa.ASTEROIDS_PATH == "/kernel.bsp"
    assert fa.ASTEROIDS_PATH is None
    pyroorb.asteroid_perturbers()                       # the call form stays on
    assert fa.ASTEROIDS_PATH == "/kernel.bsp"
    monkeypatch.setattr(fa, "asteroid_kernel_path", lambda: None)
    with pytest.raises(FileNotFoundError, match="fetch-data"):
        pyroorb.asteroid_perturbers()


def test_an_empty_sr_family_says_what_to_use_instead(monkeypatch):
    """SR is a single-night tool: on two nights it usually keeps no orbit (2024 MK, 2023 DZ2:
    0 of 400). The refusal names the count, the arc and the methods that do work there."""
    import numpy as np
    from pyroorb import ensemble, sr
    monkeypatch.setattr(sr, "statistical_ranging", lambda *a, **k: (None, None, 0))
    obs = pyroorb.read_observations(OBS80)

    class _Fit:
        _ctx = {}
        _obs = list(obs)
        _mask = np.ones(len(obs), bool)

    with pytest.raises(ValueError, match=r"kept 0 of 400 .*single-night.*'lov' or 'covariance'"):
        ensemble.sample_orbits(_Fit(), 400, method="sr")


def test_body_states_of_an_asteroid_perturber_carry_a_velocity():
    """The asteroid kernel gives positions only (its velocities read NaN); body_states fills the
    velocity in, which close_approaches needs (2022 KX28's 2013 Ceres pass read NaN without)."""
    import numpy as np
    import pyroorb.force_assist as fa
    if fa.asteroid_kernel_path() is None:
        pytest.skip("the asteroid perturber kernel is not installed")
    s = pyroorb.body_states("Ceres", ["2013-01-09", "2013-01-10"])
    assert np.all(np.isfinite(s))
    step = s[1, :3] - s[0, :3]                                  # one day of motion, au
    assert np.allclose(0.5 * (s[0, 3:] + s[1, 3:]), step, rtol=1e-4)
    assert np.allclose(pyroorb.body_states("sun", "2013-01-09"), 0.0)
    with pytest.raises(ValueError, match="unknown body"):
        pyroorb.body_states("vulcan", "2013-01-09")


@pytest.fixture(scope="module")
def qj3():
    records, meta = fixture_obs("2023_QJ3")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroorb.UndeterminedOrbitWarning)
        return pyroorb.fit_orbit(pyroorb.Observations(records, meta), designation="2023 QJ3")


@pytest.mark.slow
def test_propagate_and_elements_are_exact_at_the_epoch(qj3):
    import numpy as np
    from astropy.time import Time
    ep = Time(qj3.epoch_jd(), format="jd", scale="tdb")
    s, c = pyroorb.propagate(qj3, ep, covariance=True)
    assert np.allclose(s[0], qj3.state()[:6], rtol=0, atol=1e-14)
    assert np.allclose(c[0], qj3.covariance_matrix("state"),
                       rtol=1e-6, atol=1e-9 * abs(qj3.covariance_matrix("state")).max())
    el = pyroorb.osculating_elements(s[0], ep)
    assert el == pytest.approx(qj3.elements(), rel=1e-12, abs=1e-12)
    times = ["2026-11-01", "2020-01-01", "2026-12-01"]          # out of order, across the epoch
    together = pyroorb.propagate(qj3, times)
    apart = np.vstack([pyroorb.propagate(qj3, t) for t in times])
    assert np.allclose(together, apart, rtol=0, atol=1e-12)


@pytest.mark.slow
def test_close_approach_is_the_minimum_of_a_dense_scan(qj3):
    import numpy as np
    from pyroorb import products
    rows = pyroorb.close_approaches(qj3, "2023-01-01", "2033-01-01", body="mars", max_distance=0.5)
    assert rows, "a Mars-crosser should pass Mars within 0.5 au in ten years"
    r = rows[0]
    t = r["jd_tdb"] + np.arange(-0.05, 0.05, 1e-4)
    d = np.linalg.norm(pyroorb.propagate(qj3, products._utc_jd(t))[:, :3]
                       - products.body_states("mars", t)[:, :3], axis=1)
    assert r["distance_au"] == pytest.approx(d.min(), abs=1e-9)
    assert r["sigma_distance_au"] > 0 and r["sigma_time_days"] > 0


@pytest.mark.slow
def test_radar_residuals_and_predictions_reproduce_the_joint_fit():
    records, meta = fixture_obs("162385")
    obs = pyroorb.Observations(records, meta)
    fit = pyroorb.fit_orbit(obs, designation="162385")
    assert fit.radar["applied"]
    res = pyroorb.radar_residuals(fit, meta["radar"])
    assert [r["z"] for r in res] == pytest.approx([r["z"] for r in fit.radar["rows"]], abs=1e-3)
    for r in meta["radar"][:3]:
        p = pyroorb.radar_prediction(fit, r["jd_utc"], r["tx"], receiver=r["rx"],
                                     freq_hz=r["freq_hz"])[0]
        pred = p["delay_s"] if r["kind"] == "delay" else p["doppler_hz"]
        assert abs(r["value"] - pred) < 5 * r["sigma"]


@pytest.mark.slow
def test_the_covariance_ensemble_matches_the_linear_products(qj3):
    import numpy as np
    from pyroorb.ensemble import OrbitEnsemble
    ens = pyroorb.sample_orbits(qj3, 300)
    assert len(ens) == 300 and ens.weights.sum() == pytest.approx(1.0)
    assert ens.element_percentiles((5, 95))["a"] == pytest.approx(qj3.element_ranges()["a"], rel=1e-4)
    one = OrbitEnsemble(qj3, [qj3.state()[:6]], [1.0], qj3.epoch_tdb, "nominal")
    sky = one.sky_positions("2026-11-01", site="568")
    eph = pyroorb.ephemeris(qj3, "2026-11-01", "2026-11-01", site="568", full=False)[0]
    assert abs(sky["ra_deg"][0, 0] - eph["ra_deg"]) * 3600 < 0.01
    assert abs(sky["dec_deg"][0, 0] - eph["dec_deg"]) * 3600 < 0.01
    lov = pyroorb.sample_orbits(qj3, 11, method="lov")
    assert lov.method == "lov" and len(lov) >= 5
    with pytest.raises(ValueError, match="method"):
        pyroorb.sample_orbits(qj3, 10, method="bootstrap")


@pytest.mark.slow
def test_range_posterior_on_a_tracklet():
    obs = pyroorb.read_observations(OBS80)
    with pytest.warns(pyroorb.UndeterminedOrbitWarning):
        fit = pyroorb.fit_orbit(obs)
    flat = pyroorb.range_posterior(fit, prior=None)
    pop = pyroorb.range_posterior(fit)
    for post in (flat, pop):
        assert post["weight"].sum() == pytest.approx(1.0)
        p = post["rho_percentiles"]
        assert 0 < p[5] <= p[50] <= p[95]
    assert flat["prior"] is None and pop["prior"] is not None
    mp = pyroorb.most_probable_orbit(fit)
    if mp is not None:
        assert mp.most_probable["rho_med"] == pytest.approx(pop["rho_percentiles"][50], rel=0.2)


# --- the documented surface ---------------------------------------------------------------

_DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")


def _documented():
    """(directive, name) of every auto* entry in docs/api.rst."""
    import re
    text = open(os.path.join(_DOCS, "api.rst"), encoding="utf-8").read()
    return re.findall(r"^\.\. auto(function|class|method|attribute):: (\S+)", text, re.M)


def test_every_documented_name_exists_and_has_a_docstring():
    entries = _documented()
    assert len(entries) > 40
    for kind, name in entries:
        obj = pyroorb
        for part in name.split("."):
            obj = getattr(obj, part)
        if isinstance(obj, property):
            obj = obj.fget
        doc = getattr(obj, "__doc__", None)
        assert doc and doc.strip(), f"docs/api.rst lists {name} but it has no docstring"


def test_every_exported_name_is_documented():
    documented = {name for _, name in _documented()}
    for name in pyroorb.__all__:
        if name in ("__version__", "load_obscodes"):
            continue          # load_obscodes: the older spelling of obscodes(), kept importable
        assert name in documented, f"pyroorb.{name} is exported but not in docs/api.rst"


def test_notebooks_mark_every_internal_import():
    """notebooks/README.md: cells that reach into internal modules say so. A notebook cell that
    imports a pyroorb submodule must carry the 'internal API, may change' comment."""
    import json
    import re
    nbdir = os.path.join(os.path.dirname(_DOCS), "notebooks")
    pat = re.compile(r"^\s*(from pyroorb(\.\S+)?\s+import|import pyroorb\.)", re.M)
    for name in sorted(os.listdir(nbdir)):
        if not name.endswith(".ipynb"):
            continue
        nb = json.load(open(os.path.join(nbdir, name), encoding="utf-8"))
        for k, cell in enumerate(nb["cells"]):
            src = "".join(cell["source"])
            if cell["cell_type"] == "code" and pat.search(src):
                assert "internal API, may change" in src, f"{name} cell {k} imports an internal module unmarked"
