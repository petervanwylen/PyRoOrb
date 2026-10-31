"""Planetocentric orbit tests (the "told-the-center" path).

Two layers:
  * Fast, fit-free (always on): the element_vector ``mu`` back-compat, the reframe round-trip
    (a known jovicentric state -> planetocentric elements recovers a/e/i exactly), and the
    unbound/fly-by guard. These pin the reporting math with no propagation.
  * ``test_synthetic_fit`` -- the end-to-end seed -> fit -> report on a synthetic Himalia-like
    jovicentric orbit: generate noisy astrometry through our own N-body model and confirm the
    ``--center jupiter`` fit recovers a/e/i within a few sigma. Heavier (a real fit); gated
    behind PC_DYNAMIC=1 so the default suite stays fast (mirrors test_keyholes' KH_DYNAMIC).

Run: PC_DYNAMIC=1 python tests/test_planetocentric.py   (or under pytest)
"""
import os, sys, math
import numpy as np
import pytest
from astropy.time import Time

import pyroorb.planetocentric as pc
from pyroorb.orbital_elements import element_vector, MU_SUN

# synthetic-satellite fits
pytestmark = pytest.mark.slow

OBL = math.radians(23.43928111)
EPOCH0 = 2460000.5


def _ecl_to_eq(v):
    c, s = math.cos(OBL), math.sin(OBL)
    return np.array([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])


def _kepler_state(a, e, i, node, argp, nu, mu):
    """Classical (ecliptic) elements -> ecliptic (r, v). Angles in radians."""
    p = a * (1 - e * e)
    r = p / (1 + e * math.cos(nu))
    r_pf = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    v_pf = math.sqrt(mu / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])

    def Rz(t):
        c, s = math.cos(t), math.sin(t); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    def Rx(t):
        c, s = math.cos(t), math.sin(t); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    Q = Rz(node) @ Rx(i) @ Rz(argp)
    return Q @ r_pf, Q @ v_pf


def _jovicentric_helio(a, e, i_deg, node_deg, argp_deg, nu_deg, epoch=EPOCH0):
    mu = pc.planet_gm("jupiter")
    r_ecl, v_ecl = _kepler_state(a, e, math.radians(i_deg), math.radians(node_deg),
                                 math.radians(argp_deg), math.radians(nu_deg), mu)
    rel_eq = np.concatenate([_ecl_to_eq(r_ecl), _ecl_to_eq(v_ecl)])
    return rel_eq + pc.planet_helio_state("jupiter", epoch)


def test_mu_backcompat():
    """element_vector default == explicit MU_SUN (the new mu param is backward-compatible)."""
    s = np.array([1.3, 0.4, 0.1, -0.006, 0.014, 0.002])
    a = element_vector(s, EPOCH0)
    b = element_vector(s, EPOCH0, mu=MU_SUN)
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b))


def test_reframe_roundtrip():
    """A known jovicentric state, embedded heliocentrically and reframed back, recovers a/e/i."""
    a_t, e_t, i_t = 0.07655, 0.162, 29.0
    helio = _jovicentric_helio(a_t, e_t, i_t, 45, 70, 120)
    d = pc.planetocentric_elements(helio, EPOCH0, "jupiter")
    assert d["bound"] and d["a"] > 0
    assert abs(d["a"] - a_t) < 1e-6, d["a"]
    assert abs(d["e"] - e_t) < 1e-6, d["e"]
    assert abs(d["i"] - i_t) < 1e-4, d["i"]
    assert "prograde satellite of Jupiter" in pc.classify(d)


def test_unbound_guard():
    """A hyperbolic planetocentric state is flagged unbound (find_orb's fly-by rule), not fabricated."""
    mu = pc.planet_gm("jupiter")
    # radius 0.05 au from Jupiter, speed well above escape -> hyperbolic
    v_esc = math.sqrt(2 * mu / 0.05)
    rel_eq = _ecl_to_eq(np.array([0.05, 0, 0])), _ecl_to_eq(np.array([0, 2.0 * v_esc, 0]))
    helio = np.concatenate(rel_eq) + pc.planet_helio_state("jupiter", EPOCH0)
    d = pc.planetocentric_elements(helio, EPOCH0, "jupiter")
    assert d["bound"] is False
    assert "unbound" in pc.classify(d)


def test_resolve_planet():
    assert pc.resolve_planet("Saturn") == "saturn"
    assert pc.resolve_planet("jovicentric") == "jupiter"
    assert pc.is_validated("neptune") and not pc.is_validated("earth")
    for bad in ("pluto", "moon", "ceres", "xyz"):
        try:
            pc.resolve_planet(bad); assert False, bad
        except ValueError:
            pass


def test_mass_contest():
    """find_orb's mass/dist^2.5 contest (auto-detect b): a point deep inside Jupiter's Hill sphere
    picks Jupiter; a Jupiter Trojan (~5 au from both Sun and Jupiter) picks the Sun."""
    # 0.05 au from Jupiter -> Jupiter dominates
    near_jup = pc.planet_helio_state("jupiter", EPOCH0)[:3] + np.array([0.05, 0, 0])
    assert pc.detect_center(np.concatenate([near_jup, [0, 0, 0]]), EPOCH0) == "jupiter"
    # Jupiter Trojan: rotate Jupiter's position 60 deg about the pole -> ~equidistant from both
    J = pc.planet_helio_state("jupiter", EPOCH0)[:3]
    th = math.radians(60); c, s = math.cos(th), math.sin(th)
    trojan = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ J
    contest = pc.mass_contest(np.concatenate([trojan, [0, 0, 0]]), EPOCH0)
    assert contest[0][0] == "sun", contest[:2]
    assert pc.detect_center(np.concatenate([trojan, [0, 0, 0]]), EPOCH0) is None


def test_dominating_planet():
    """The Hill-sphere trigger: a point inside Jupiter's Hill sphere -> jupiter; a Jupiter Trojan
    (~5 au away, far outside) and a main-belt distance -> None. This is what makes auto-detect fire
    even when a satellite's heliocentric fit is perfectly good (S/2003 J 2)."""
    J = pc.planet_helio_state("jupiter", EPOCH0)
    inside = J[:6].copy(); inside[:3] += np.array([0.1, 0, 0])      # 0.1 au from Jupiter (< Hill ~0.35)
    assert pc.dominating_planet(inside, EPOCH0) == "jupiter"
    th = math.radians(60); c, s = math.cos(th), math.sin(th)
    trojan = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ J[:3]
    assert pc.dominating_planet(np.concatenate([trojan, [0, 0, 0]]), EPOCH0) is None
    assert pc.dominating_planet(np.array([2.6, 0, 0, 0, 0, 0]), EPOCH0) is None   # main belt


def test_synthetic_fit():
    """End-to-end: synthetic Himalia-like jovicentric orbit -> noisy astrometry -> --center fit
    recovers a/e/i. Heavier (a real fit); gated behind PC_DYNAMIC=1."""
    if not os.environ.get("PC_DYNAMIC"):
        try:
            import pytest
            pytest.skip("set PC_DYNAMIC=1 to run the end-to-end synthetic satellite fit")
        except ImportError:
            return  # no pytest -> plain runner: treat as a no-op pass

    import pyroorb.force_assist as fa
    import pyroorb.fit_object as fo
    from pyroorb.obs_sources import observer_positions_batch, load_obscodes

    np.random.seed(42)
    a_t, e_t, i_t = 0.07655, 0.162, 29.0
    helio0 = _jovicentric_helio(a_t, e_t, i_t, 45, 70, 120)

    obscodes = load_obscodes()
    n, span = 16, 150.0
    jd_utc = EPOCH0 - 69.184 / 86400 + np.linspace(0, span, n)
    tdb = Time(jd_utc, format="jd", scale="utc").tdb.jd
    Robs = list(observer_positions_batch(list(jd_utc), ["500"] * n, obscodes))
    base0 = fa.build_base_sim(EPOCH0)
    pred = np.asarray(fa.predict_radec_batch(base0, EPOCH0, helio0, tdb, Robs))
    sig = 0.2 / 3600.0
    ra = pred[:, 0] + (np.random.randn(n) * sig) / np.cos(np.radians(pred[:, 1]))
    dec = pred[:, 1] + np.random.randn(n) * sig
    obs = [{"jd_utc": float(jd_utc[k]), "ra_deg": float(ra[k]), "dec_deg": float(dec[k]),
            "obscode": "500", "packed": "SYN01", "designation": "SYN01",
            "mag": 18.0, "band": "V", "mag_sigma": None} for k in range(n)]

    fit = fo.fit_orbit(obs, obscodes, desig="SYN01", meta=None, center="jupiter")
    assert getattr(fit, "center", None) == "jupiter"
    assert fit.n_observations() >= n - 1
    assert fit.rms_residual() < 0.5

    fep = fit.epoch_tdb
    truth_helio = np.asarray(fa.predict_state(base0, EPOCH0, helio0, fep), float)
    truth = pc.planetocentric_elements(truth_helio, fep, "jupiter")
    rec = pc.planetocentric_elements(fit.state_vector(), fep, "jupiter")
    assert rec["bound"]
    assert abs(rec["a"] - truth["a"]) / truth["a"] < 0.02, (rec["a"], truth["a"])
    assert abs(rec["e"] - truth["e"]) < 0.01, (rec["e"], truth["e"])
    assert abs(rec["i"] - truth["i"]) < 0.2, (rec["i"], truth["i"])

    # AUTO-DETECT ON BY DEFAULT (b): the SAME arc with NO center given is recognised as a Jupiter
    # satellite by fit_orbit itself.
    auto = fo.fit_orbit(obs, obscodes, desig="SYN01", meta=None, center=None)
    assert getattr(auto, "center", None) == "jupiter"
    assert auto.rms_residual() < 0.5
    # with auto-detect disabled, the raw heliocentric fit is (correctly) garbage and un-flagged
    raw = fo.fit_orbit(obs, obscodes, desig="SYN01", meta=None, center=None, auto_center=False)
    assert getattr(raw, "center", None) is None
    assert raw.rms_residual() > 30.0


def test_web_payload_block():
    """The web serialise_fit payload carries the planetocentric block (badge + element toggle
    data) for a detected satellite, and None for a heliocentric object. Gated (needs a fit)."""
    if not os.environ.get("PC_DYNAMIC"):
        try:
            import pytest
            pytest.skip("set PC_DYNAMIC=1 to run the web-payload check")
        except ImportError:
            return
    import pyroorb.force_assist as fa; import pyroorb.fit_object as fo; from pyroorb.web import engine_api
    from pyroorb.obs_sources import observer_positions_batch, load_obscodes
    np.random.seed(7)
    helio0 = _jovicentric_helio(0.07655, 0.162, 29.0, 45, 70, 120)
    oc = load_obscodes(); n, span = 16, 150.0
    jd = EPOCH0 - 69.184 / 86400 + np.linspace(0, span, n)
    tdb = Time(jd, format="jd", scale="utc").tdb.jd
    Robs = list(observer_positions_batch(list(jd), ["500"] * n, oc))
    base0 = fa.build_base_sim(EPOCH0)
    pred = np.asarray(fa.predict_radec_batch(base0, EPOCH0, helio0, tdb, Robs))
    ra = pred[:, 0] + (np.random.randn(n) * 0.2 / 3600) / np.cos(np.radians(pred[:, 1]))
    dec = pred[:, 1] + np.random.randn(n) * 0.2 / 3600
    obs = [{"jd_utc": float(jd[k]), "ra_deg": float(ra[k]), "dec_deg": float(dec[k]),
            "obscode": "500", "packed": "SYN", "designation": "SYN",
            "mag": 18.0, "band": "V", "mag_sigma": None} for k in range(n)]
    sat = fo.fit_orbit(obs, oc, desig="SYN", meta=None, center=None)
    block = engine_api.serialise_fit(sat)["planetocentric"]
    assert block and block["center"] == "jupiter" and block["bound"]
    avals = {r["name"]: r["value"] for r in block["elements"]}
    assert abs(avals["a"] - 0.0766) < 0.002 and abs(avals["i"] - 29.0) < 0.3
    # a heliocentric object carries no block
    raw = fo.fit_orbit(obs, oc, desig="SYN", meta=None, center=None, auto_center=False)
    assert engine_api.serialise_fit(raw)["planetocentric"] is None
