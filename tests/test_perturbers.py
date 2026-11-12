"""The field model (the 16 plus extra bodies), and the perturber screen that fills it.

Fast tests cover the representation: a Field is truthy, survives the force-model pin, the meta
round trip and the label. The slow ones need the extra-perturber kernel and refit fixtures:
(19) Fortuna on 2016 RM33 = 2005 SA198, and (324) Bamberga's 0.00006 au pass on (310585).
"""
import numpy as np
import pytest

import pyroorb
from pyroorb import perturbers as pt
from pyroorb.forcemodel import PinnedForce, pin, pinned_asteroids


def test_a_field_is_the_16_plus_extras_and_always_on():
    f = pt.Field([19, 29])
    assert f and pt.Field() and isinstance(f, frozenset)
    assert pt.extras(f) == {19, 29} and pt.extras(True) == frozenset()
    assert f.label() == "+(19)+(29)"
    assert pt.as_field([29, 19]) == f and pt.as_field([]) is True
    assert pt.as_field(None) is None and pt.as_field(0) is False
    assert pt.field_bodies(pt.Field([19]), self_number=4) == (set(pt.BIG16) - {4}) | {19}


def test_the_pin_keeps_a_field_and_hands_it_to_every_call():
    class _Backend:
        def ensure_asteroid_kernel(self, verbose=False):
            return "k"

        def predict_radec_batch(self, *a, asteroids=None):
            return asteroids

    b = PinnedForce(_Backend(), pt.Field([19]))
    assert pinned_asteroids(b) == {19}
    assert b.predict_radec_batch() == {19}
    assert b.predict_radec_batch(asteroids=False) is False          # an explicit ask still wins
    assert pin(b, pt.Field([19])) is b                              # same field: unchanged
    assert pinned_asteroids(pin(b, True)) is True


def test_meta_and_label_record_the_extras():
    from pyroorb import fit_object as fo
    assert fo._meta_asteroids(pt.Field([29, 19])) == [19, 29]      # JSON-safe
    assert fo._meta_asteroids(True) is True
    assert fo._asteroids_from_meta([19]) == pt.Field([19])
    assert fo._asteroids_from_meta(True) is True and fo._asteroids_from_meta(None) is False
    ctx = {"backend_name": "ASSIST(DE440+GR)", "backend": None}
    assert fo._force_model_label(ctx, pt.Field([19]), None) == "ASSIST(DE440+GR+16ast+(19))"
    assert fo._force_model_label(ctx, True, None) == "ASSIST(DE440+GR+16ast)"


def test_switches(monkeypatch):
    monkeypatch.setenv(pt.SWITCH_ENV, "off")
    assert not pt.enabled()
    monkeypatch.setenv(pt.SWITCH_ENV, "")
    assert pt.enabled()
    with pytest.raises(ValueError, match="perturbers"):
        pyroorb.fit_orbit([], perturbers="sometimes")


def test_the_screen_takes_the_object_out_of_its_own_candidates():
    class _F:
        designation = "(29) Amphitrite"
        designations = ()
    assert pt.self_number(_F()) == 29
    for name, want in (("2016 RM33", None), ("29", 29), ("29 Amphitrite", 29),
                       ("2005 SA198", None), ("1036 Ganymed", 1036)):
        _F.designation = name
        assert pt.self_number(_F()) == want, name


needs_kernel = pytest.mark.skipif(pt.kernel_path() is None,
                                  reason="extra-perturber kernel not installed")


def _refit(label):
    from _fixtures import FIX, load_fixture
    from pyroorb import fit_object as fo, obs_sources
    fx = load_fixture(f"{FIX}/{label}.json")
    return fo.fit_orbit(fx["obs"], obs_sources.load_obscodes(), desig=fx["meta"]["designation"],
                        meta=fx["meta"], verbose=False)


@needs_kernel
def test_all_is_the_whole_kernel_less_the_object_itself():
    from pyroorb import fit_object as fo
    a = pt.all_extras()
    assert len(a) == 327 and not a & set(pt.BIG16) and 19 in a
    f = pt.all_field()
    assert pt.is_all(f) and 16 + len(f) == 343
    me = pt.all_field(exclude=19)                       # fitting (19) Fortuna: never inside itself
    assert 19 not in me and pt.is_all(me) and 19 not in pt.field_bodies(me)
    assert not pt.is_all(pt.Field([19])) and not pt.is_all(True) and not pt.is_all(False)
    ctx = {"backend_name": "ASSIST(DE440+GR)", "backend": None}
    assert fo._force_model_label(ctx, f, None) == "ASSIST(DE440+GR+343ast)"   # counted, not listed
    assert fo._force_model_label(ctx, me, None) == "ASSIST(DE440+GR+342ast)"


def test_all_is_a_spelling_the_api_and_refit_accept():
    with pytest.raises(ValueError, match="'auto', 'all' or 'off'"):
        pyroorb.fit_orbit([], perturbers="every")
    from pyroorb import fit_object as fo
    with pytest.raises(ValueError, match="mode"):
        fo.refine_perturbers(type("F", (), {"_ctx": {}})(), mode="most")


@pytest.mark.slow
@needs_kernel
def test_all_fits_in_every_body_and_says_so():
    """perturbers="all": no screen, the whole kernel, kept unconditionally; Fortuna is in it, so
    the 2005 night comes back as with the screen's pick."""
    import json
    from _fixtures import FIX, load_fixture
    fx = load_fixture(f"{FIX}/PAIR_2016RM33.json")
    obs = pyroorb.Observations(fx["obs"], fx["meta"])
    fit = pyroorb.fit_orbit(obs, "2016 RM33", perturbers="all")
    assert fit.perturbers_all and fit.force_model == "ASSIST(DE440+GR+343ast)"
    assert fit.perturber_screen["mode"] == "all" and fit.perturber_screen["bodies"] == 343
    assert fit.extra_perturbers == []                    # nothing judged, nothing to name
    d = json.loads(json.dumps(fit.as_dict()))
    assert d["perturbers_all"] is True and d["asteroids"] is True
    y05 = [r for r in fit.residuals() if r["designation"] == "2005 SA198"]
    assert sum(r["included"] for r in y05) == len(y05) == 15
    # the same observations in the 16 only, and back: refit's perturbers= option
    r16 = pyroorb.refit(fit, perturbers="16")
    assert r16.asteroids is True and not r16.perturbers_all
    assert pyroorb.refit(r16, perturbers="all").perturbers_all
    assert pyroorb.refit(fit, perturbers="off").asteroids is False
    assert r16.rms_residual() > fit.rms_residual()       # Fortuna's deflection is missing there
    with pytest.raises(ValueError, match="perturbers"):
        pyroorb.refit(fit, perturbers="some")


@pytest.mark.slow
@needs_kernel
def test_all_keeps_a_non_gravitational_model():
    from pyroorb import fit_object as fo
    fit = _refit("2004_JN1")
    assert fit.nongrav_model
    g = fo.refine_perturbers(fit, mode="all")
    assert g.perturbers_all and g.nongrav_model == fit.nongrav_model
    assert "+343ast" in g.force_model and g.perturber_screen["mode"] == "all"


@pytest.mark.slow
@needs_kernel
def test_the_web_toggle_solves_in_every_body_and_back():
    from _fixtures import FIX, load_fixture
    import pyroorb.web.engine_api as E
    fx = load_fixture(f"{FIX}/PAIR_2016RM33.json")
    p = E.fit_obs_list(fx["obs"], fx["meta"], E._obscodes())
    sid = p["session"]
    assert p["perturbers_all_count"] == 343 and not p["perturbers_all"]
    a = E.set_asteroids(sid, "all")
    assert a.get("ok", True) and a["perturbers_all"] and a["asteroid_extras"] == []
    assert "+343ast" in a["force_model"]
    b = E.set_asteroids(sid, True)
    assert not b["perturbers_all"] and b["asteroids"]
    c = E.set_asteroids(sid, False)
    assert not c["asteroids"] and not c["perturbers_all"]


@pytest.mark.slow
@needs_kernel
def test_fortuna_is_picked_and_the_2005_night_comes_back():
    from pyroorb import fit_object as fo
    fit = _refit("PAIR_2016RM33")
    assert fit.asteroids is True                                    # an identification: +16
    sc = pt.screen(fit)
    assert sc.picks == [19] and sc.scores[19][0] > 800 and sc.scores[19][1] < 0.002
    assert max(v[0] for b, v in sc.scores.items() if b != 19) < 0.1
    g = fo.refine_perturbers(fit)
    assert g.force_model == "ASSIST(DE440+GR+16ast+(19))" and g.asteroids == {19}
    assert g.perturber_screen["kept"] is True
    # every surface can name what was added: the fit carries it, as_dict exports it (JSON-safe)
    (e,) = g.extra_perturbers
    assert e["number"] == 19 and e["name"] == "Fortuna" and e["closest_date"] == "2011-04-13"
    import json
    d = json.loads(json.dumps(g.as_dict()))
    assert d["asteroids"] is True and d["extra_perturbers"][0]["name"] == "Fortuna"
    assert "Fortuna" in pt.summary(g.extra_perturbers)
    y05 = [r for r in g.residuals() if r["designation"] == "2005 SA198"]
    assert sum(r["included"] for r in y05) == len(y05) == 15
    assert g.rms_residual() < fit.rms_residual()
    # products integrate in the same field, past the kernel window too (the 16 there)
    eph = pyroorb.ephemeris(g, "2026-12-01", "2026-12-02", 1.0)
    assert len(eph) == 2
    assert np.all(np.isfinite(pyroorb.propagate(g, ["2150-01-01"])[0]))
    # and the public refit keeps the field
    assert pyroorb.refit(g).asteroids == {19}


@pytest.mark.slow
@needs_kernel
def test_without_sb441_n16_the_field_comes_from_the_kernel(monkeypatch, tmp_path):
    """A default install builds only the extra-perturber kernel. The 16 must then come from it
    (sb441-n16's records verbatim inside 1900-2100): before, every asteroid field there silently
    integrated planets-only, so the screen picked Fortuna and the refit, gaining nothing,
    dropped it -- and an identification fit tried to download the 616 MB sb441-n16."""
    from pyroorb import fit_object as fo, force_assist as fa
    missing = str(tmp_path / "sb441-n16.bsp")
    monkeypatch.setattr(fa, "ASTEROIDS_DEFAULT_PATH", missing)
    monkeypatch.setattr(fa, "ASTEROIDS_PATH", None)
    monkeypatch.setattr(fa, "ASTEROIDS_OVERRIDE", None)
    monkeypatch.setattr(fa, "_EPHEM_AST", None)
    assert fa.asteroid_kernel_path() is None and fa.asteroids_available()
    assert fa.ensure_asteroid_kernel() == pt.kernel_path()          # no download
    assert pt.window(fa.get_ephem(True)) is not None               # the 16, from the kernel
    fit = _refit("PAIR_2016RM33")
    assert fit.force_model == "ASSIST(DE440+GR+16ast)"
    g = fo.refine_perturbers(fit)
    assert g.asteroids == {19} and g.perturber_screen["kept"] is True
    assert sum(r["included"] for r in g.residuals() if r["designation"] == "2005 SA198") == 15
    assert not tmp_path.joinpath("sb441-n16.bsp").exists()


@pytest.mark.slow
@needs_kernel
def test_an_itf_seed_latches_its_field_and_its_trials_run_in_it():
    """ITF trial fits are gravity by design; a seed that needs a body beyond the 16 latches that
    field once per search (itf.fitter.seed_perturbers) and every trial fit runs in it
    (fit_astrometry(perturbers=)). 2016 RM33's seed alone scores (19) Fortuna at dchi2 ~19, and
    the 2005 night past Fortuna's 2011 pass then fits in a trial -- in gravity it is rejected.
    The forced field must be what the fit is PINNED to: pinning the bare bool at fit time
    dropped Fortuna while the label still named it."""
    from _fixtures import FIX, load_fixture
    from pyroorb import fit_object as fo, force_assist as fa, obs_sources
    from pyroorb.forcemodel import pinned_asteroids
    from pyroorb.itf.fitter import seed_perturbers, trial_from_fit
    fx = load_fixture(f"{FIX}/PAIR_2016RM33.json")
    meta = {k: v for k, v in fx["meta"].items() if k not in ("designations", "n_requested")}
    meta["designation"] = "2016 RM33"
    oc = obs_sources.load_obscodes()
    seed = fo.fit_orbit([o for o in fx["obs"] if o.get("designation") != "2005 SA198"], oc,
                        desig="2016 RM33", meta=meta, verbose=False)
    field, t = seed_perturbers(trial_from_fit(seed))
    assert field == {19} and t.fit.force_model.endswith("+(19))")
    with fa.perturbers_on_for("itf", field):
        trial = fo.fit_orbit(fx["obs"], oc, desig="2016 RM33", meta=meta, verbose=False)
    assert pinned_asteroids(trial._ctx["backend"]) == {19}
    y05 = [r for r in trial.residuals() if r["designation"] == "2005 SA198"]
    assert sum(r["included"] for r in y05) == len(y05) == 15


@pytest.mark.slow
@needs_kernel
def test_the_identification_services_judge_the_refined_fit():
    """The queue's assessment and the NEOCP monitor's identification fit go through
    itf.fitter.refine_trial: the pair's TrialFit comes back in the field it needs."""
    from _fixtures import FIX, load_fixture
    from pyroorb import fit_object as fo, obs_sources
    from pyroorb.itf.fitter import refine_trial, trial_from_fit
    fx = load_fixture(f"{FIX}/PAIR_2016RM33.json")
    f = fo.fit_orbit(fx["obs"], obs_sources.load_obscodes(), desig="2016 RM33", meta=fx["meta"],
                     verbose=False)
    t = trial_from_fit(f)
    r = refine_trial(t)
    assert r.fit.asteroids == {19} and r.used > t.used


@pytest.mark.slow
@needs_kernel
def test_the_public_fit_says_when_it_added_a_body():
    from _fixtures import FIX, load_fixture
    fx = load_fixture(f"{FIX}/PAIR_2016RM33.json")
    obs = pyroorb.Observations(fx["obs"], fx["meta"])
    with pytest.warns(pyroorb.PerturbersAddedWarning, match=r"\(19\) Fortuna"):
        fit = pyroorb.fit_orbit(obs, "2016 RM33")
    assert [e["number"] for e in fit.extra_perturbers] == [19]


@pytest.mark.slow
@needs_kernel
def test_an_hour_long_pass_is_scored_by_its_impulse():
    """(324) Bamberga passed (310585) at 0.00006 au: sampling the encounter at 0.25 d scored
    it 5 while the refit gained 1670; the analytic impulse puts it where the refit does."""
    from pyroorb import fit_object as fo
    from pyroorb import force_assist as fa
    with fa.perturbers_on_for("test"):
        fit = _refit("310585")
    sc = pt.screen(fit)
    assert sc.picks and sc.picks[0] == 324
    assert sc.scores[324][0] > 300 and sc.scores[324][1] < 0.0002


@pytest.mark.slow
@needs_kernel
def test_switched_off_leaves_the_fit_alone(monkeypatch):
    from pyroorb import fit_object as fo
    monkeypatch.setenv(pt.SWITCH_ENV, "off")
    fit = _refit("PAIR_2016RM33")
    g = fo.refine_perturbers(fit)
    assert g is fit and g.perturber_screen == {"skipped": "switched off"}


@pytest.mark.slow
@needs_kernel
def test_a_screen_that_cannot_run_says_why(monkeypatch):
    """A missing SPK reader left the web on the bare 16 with nothing on the page; the reason
    must ride on the fit (the web turns it into a banner)."""
    import sys
    from pyroorb import fit_object as fo
    monkeypatch.setitem(sys.modules, "jplephem", None)
    monkeypatch.setitem(sys.modules, "jplephem.spk", None)
    monkeypatch.setattr(pt, "_TABLES", {})
    fit = _refit("PAIR_2016RM33")
    g = fo.refine_perturbers(fit)
    assert g is fit and g.perturber_screen["skipped"].startswith("screen failed")


def test_the_bundled_kernel_is_the_published_build():
    """The kernel ships inside the package (every install screens with no download), and it is
    the build KERNEL_SHA256 names, not a silently swapped file."""
    import hashlib
    from pyroorb import perturber_kernel as pk
    k = pt.BUNDLED_KERNEL
    assert k.parent.name == "data" and k.parent.parent.name == "pyroorb" and k.exists()
    assert hashlib.sha256(k.read_bytes()).hexdigest() == pk.KERNEL_SHA256


def test_the_bundled_kernel_is_preferred(monkeypatch, tmp_path):
    from pyroorb import force_assist as fa
    monkeypatch.delenv(pt.KERNEL_ENV, raising=False)
    (tmp_path / pt.KERNEL_NAME).write_bytes(b"an older rebuild")
    monkeypatch.setattr(fa, "EPHEM_DIR", str(tmp_path))
    assert pt.kernel_path() == str(pt.BUNDLED_KERNEL)
    other = tmp_path / "mine.bsp"
    other.write_bytes(b"x")
    monkeypatch.setenv(pt.KERNEL_ENV, str(other))
    assert pt.kernel_path() == str(other)                   # an explicit override still wins


def test_the_position_table_never_lands_in_the_package(monkeypatch, tmp_path):
    """Beside a writable kernel elsewhere, but for the bundled copy in the cache folder: the
    75 MB table would otherwise grow an editable checkout and every wheel built from it."""
    from pyroorb import datadirs
    monkeypatch.setattr(datadirs, "cache_dir", lambda: tmp_path / "cache")
    t = pt._Table._path(pt.BUNDLED_KERNEL)
    assert t.parent == tmp_path / "cache" / "perturbers"
    elsewhere = tmp_path / "ephem" / pt.KERNEL_NAME
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"k")
    assert pt._Table._path(elsewhere).parent == elsewhere.parent


def test_the_kernel_says_it_is_ours_and_derived_from_jpl():
    """NAIF's rules for a modified kernel (renamed, annotated, attributed to whoever modified it)
    and JPL's data-use policy (source URL, credit line, no implied endorsement), in the comment
    area that travels with the file."""
    from pyroorb import perturber_kernel as pk
    c = pk.comment_text(327, 20.0, 20.0)
    for must in ("PRODUCER: the PyRoOrb project", "NOT a JPL product",
                 "no\nendorsement by Caltech, JPL or NASA", "Courtesy NASA/JPL-Caltech",
                 pk.N373_URL, "Park, R. S.", "MODIFICATIONS:", "VALIDATION:", "TERMS:"):
        assert must in c, must
    c.encode("ascii")                                       # an SPK comment area is ASCII


def test_the_published_kernel_carries_that_comment():
    """The shipped file says it is ours, inside itself."""
    from jplephem.spk import SPK
    k = SPK.open(str(pt.BUNDLED_KERNEL))
    try:
        assert "PRODUCER: the PyRoOrb project" in k.comments()
        assert "Courtesy NASA/JPL-Caltech" in k.comments()
    finally:
        k.close()


def test_the_mass_table_is_simda_beyond_the_16_and_credits_its_source():
    t = pt.mass_table()
    assert "Kretlow" in t["citation"] and t["release"]["doi"] == "10.5281/zenodo.4039774"
    # the file credits its source itself, ahead of the data, for a copy read without the docs
    import json
    from pathlib import Path
    raw = (Path(pt.__file__).with_name("data") / "perturber_masses.json").read_text(encoding="utf-8")
    keys = list(json.loads(raw))
    assert keys[:3] == ["_about", "_credit", "_license"]
    assert "SiMDA" in raw[:400] and "NOT original to PyRoOrb" in raw[:400]
    names = (Path(pt.__file__).with_name("data") / "perturber_names.json").read_text(encoding="utf-8")
    assert names.lstrip().startswith('{\n"_about"') and "Minor Planet Center" in names[:400]
    assert pt.name(324) == "Bamberga"
    m = t["masses"]
    assert not set(m) & set(pt.BIG16)                         # the 16 keep DE440
    assert set(e["rank"] for e in m.values()) <= {"A", "B", "C"}
    assert m[324]["kg"] == pytest.approx(1.04e19, rel=0.01)   # Bamberga, rank A
    assert pt.mass_source(324).startswith("SiMDA") and pt.mass_source(1) == "DE440"


@needs_kernel
def test_the_field_integrates_with_the_table_masses():
    """The screen (gm) and the integration (ephem_for) use the same mass: SiMDA's for a body the
    table covers, DE440's for the 16 and for the rest."""
    e = pt.ephem_for(pt.field_bodies(pt.Field([324])))
    from pyroorb import force_assist as fa
    spk = fa._SpkS.from_address(e.spk_asteroids)
    mass = {spk.targets[i].code - 2000000: spk.targets[i].mass for i in range(spk.num)}
    # GMs are ~1e-15 au^3/day^2: compare ratios (pytest.approx's absolute floor would hide them)
    assert mass[324] / pt.mass_table()["masses"][324]["gm_au3_day2"] == pytest.approx(1, abs=1e-12)
    assert mass[324] / pt.gm(324) == pytest.approx(1, abs=1e-12)
    assert mass[324] / pt.de440_gm(324) == pytest.approx(1.12, abs=0.01)   # SiMDA, not DE440
    assert mass[1] / pt.de440_gm(1) == pytest.approx(1, abs=1e-12)
    assert pt.mass_kg(324) == pytest.approx(1.04e19, rel=0.01)
    d = pt.describe(324, (1400.0, 6e-5, 2453672.5))
    assert d["mass_source"].startswith("SiMDA") and "SiMDA" in pt.summary([d])


@needs_kernel
def test_one_ephemeris_per_body_set_and_its_window():
    a = pt.ephem_for(pt.field_bodies(pt.Field([19])))
    assert a is pt.ephem_for(pt.field_bodies(pt.Field([19])))     # cached
    b = pt.ephem_for(pt.field_bodies(True))
    assert a is not b
    lo, hi = pt.window(a)
    assert lo < 2415100 and hi > 2488000                            # 1900 .. 2100
    assert lo > 2414000 and hi < 2489100                            # an extra body bounds it
    lo, hi = pt.window(b)
    assert lo < 2378497 and hi > 2524593                            # the 16 alone: 1800 .. 2200
    assert a.get_particle(11 + 16, 2451545.0 - a.jd_ref).m > 0      # the 17th body has mass


def _jd(year):
    return 2451545.0 + (year - 2000) * 365.25


def _default_install(monkeypatch, tmp_path):
    """No sb441-n16: the 16 come from the extra-perturber kernel, as on a default install."""
    from pyroorb import force_assist as fa
    monkeypatch.setattr(fa, "ASTEROIDS_DEFAULT_PATH", str(tmp_path / "sb441-n16.bsp"))
    monkeypatch.setattr(fa, "ASTEROIDS_PATH", None)
    monkeypatch.setattr(fa, "ASTEROIDS_OVERRIDE", None)
    monkeypatch.setattr(fa, "_EPHEM_AST", None)
    monkeypatch.setattr(fa, "_FALLBACK_SAID", set())
    assert fa.asteroid_kernel_path() is None
    return fa


@needs_kernel
def test_a_sweep_leaving_its_window_runs_in_the_widest_field_and_says_so(monkeypatch, tmp_path):
    """Default install: a sweep reaching 1850 leaves an extra body's 1900-2100 but not the 16's
    1800-2200, so it runs in the 16 (before, planets-only, silently, under a +16 label); one
    reaching 1700 leaves both and runs planets-only. Each is said once per process."""
    import warnings
    fa = _default_install(monkeypatch, tmp_path)
    with pytest.warns(UserWarning, match="the 16 asteroid perturbers only"):
        e = fa._fallback_ephem(_jd(1850), _jd(2020))
    assert e is pt.ephem_for(pt.field_bodies(True))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fa._fallback_ephem(_jd(1850), _jd(2020)) is e            # said once
    with pytest.warns(UserWarning, match="planets only"):
        e = fa._fallback_ephem(_jd(1700), _jd(2020))
    assert e is fa.get_ephem(False) and pt.window(e) is None


@needs_kernel
def test_the_16_integrate_before_1900_on_the_default_install(monkeypatch, tmp_path):
    """A one-state sweep to 1850 in the kernel's 16 runs inside their window (no fallback, no
    warning) and is pulled by them: it differs from the planets-only sweep."""
    import warnings
    fa = _default_install(monkeypatch, tmp_path)
    s0 = np.array([2.7, 0.0, 0.0, 0.0, 0.0105, 0.001])               # a main-belt orbit
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with16 = fa.predict_state(None, _jd(2020), s0, _jd(1850), asteroids=True)
    planets = fa.predict_state(None, _jd(2020), s0, _jd(1850), asteroids=False)
    assert np.all(np.isfinite(with16))
    assert np.linalg.norm(with16[:3] - planets[:3]) > 1e-7             # au; the 16 pulled it


@needs_kernel
@pytest.mark.skipif(not __import__("pyroorb.force_assist").force_assist.asteroid_kernel_path(),
                    reason="needs sb441-n16 to compare against")
def test_the_kernels_16_are_sb441_n16_over_1800_to_2200():
    """The 16 are JPL's records verbatim over their whole span, so a fit in them integrates
    as with sb441-n16 wherever both cover it. Positions agree to centimetres, not bits: a record
    is located from its segment's start epoch, and a different cut start changes the last bits
    of the normalised time (0.07 m at most, the same as the 1900-2100 kernel)."""
    from jplephem.spk import SPK
    from pyroorb import force_assist as fa
    ours, jpl = SPK.open(pt.kernel_path()), SPK.open(fa.asteroid_kernel_path())
    t = np.random.default_rng(3).uniform(_jd(1800) + 2, _jd(2200) - 2, 400)
    want = {s.target: s for s in jpl.segments                       # JPL splits each body's
            if s.start_jd < t.min() and s.end_jd > t.max()}            # span into segments
    for s in ours.segments:
        if s.target - 2000000 in pt.BIG16:
            assert np.abs(s.compute(t) - want[s.target].compute(t)).max() < 1e-3, s.target  # km


@pytest.mark.slow
@needs_kernel
def test_concurrent_table_builds_never_share_a_temporary(monkeypatch, tmp_path):
    """Writers building the position table at once each use their own temporary file: with one
    shared name, one writer's os.replace moved another's file away and that screen failed with
    FileNotFoundError (a 3-process study pool after a kernel change)."""
    import threading
    saved, errors = [], []
    real_save = np.save

    def record(f, a):
        saved.append(str(f))
        real_save(f, a)

    monkeypatch.setattr(pt._Table, "_path", staticmethod(lambda k: tmp_path / "k.table4d.npy"))
    monkeypatch.setattr(pt.np, "save", record)

    def build():
        try:
            assert pt._Table(pt.kernel_path()).pos.shape[0] == 343
        except Exception as e:                      # pragma: no cover - the failure mode
            errors.append(e)

    ts = [threading.Thread(target=build) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    assert len(saved) == len(set(saved))
    assert not list(tmp_path.glob("*.tmp.npy"))
