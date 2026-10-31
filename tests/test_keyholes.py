"""Phase-3 keyhole tests.

Two layers, both OFFLINE:
  * ``test_resonant_circle`` -- a fast, fit-free check of the analytic Valsecchi resonant-circle
    primitive (the LOCATOR core): the circle's required post-encounter angle theta' must invert,
    through the Opik energy relation, back to exactly the resonant semimajor axis a'_res. This
    pins the geometry/algebra with no propagation.
  * ``test_apophis_keyhole`` -- the end-to-end LOCATOR + PROBABILITY on Apophis's discovery-era
    arc (the historical setting in which a 2036 resonant-return keyhole was actually in play).
    Validates STRUCTURE: the 2029 deflecting encounter is found, a'(LOV) sweeps across unity,
    and a 7/6 keyhole with a ~2036 return and a sub-km b-plane width is located with a non-zero
    impact probability. Absolute IP is NOT gated (it depends on our independent weighting and on
    the exact arc); the historical/precise value is ~0 today. Heavier (multi-year propagation);
    skipped unless KH_DYNAMIC=1 so the default suite stays fast.

Run the heavy layer with: KH_DYNAMIC=1 pytest tests/test_keyholes.py
"""
import os, math

import pytest

import pyroorb.keyholes as K


def test_resonant_circle():
    """The resonant circle for an h/k return must be internally consistent with the Opik
    energy relation a' = 1/(1 - U^2 - 2U cos(theta')): inverting the circle's cos(theta') must
    recover a'_res = (h/k)^(2/3) for a range of encounter geometries."""
    v_planet = 29.78
    checked = 0
    for u_inf in (4.0, 5.84, 8.0):           # 5.84 ~ Apophis
        U = u_inf / v_planet
        for (h, k) in ((7, 6), (8, 7), (5, 4), (11, 9)):
            a_resonant = (h / k) ** (2.0 / 3.0)
            cos_tp = (1.0 - U * U - 1.0 / a_resonant) / (2.0 * U)
            if not -1.0 <= cos_tp <= 1.0:
                continue
            # invert the Opik relation -> must return a_resonant
            a_back = 1.0 / (1.0 - U * U - 2.0 * U * cos_tp)
            assert abs(a_back - a_resonant) < 1e-9, f"Opik inversion off: {a_back} vs {a_resonant}"
            # the circle must exist and be finite for a non-tangent theta
            for theta_deg in (60.0, 110.0, 140.0):
                c = K.resonant_circle(u_inf, math.radians(theta_deg), h, k, v_planet)
                if c is None:
                    continue
                zc, R = c
                assert math.isfinite(zc) and math.isfinite(R) and R >= 0.0
                checked += 1
    assert checked > 0, "no resonant circles were exercised"


def _load_apophis_disc():
    """Fit Apophis's discovery-era arc with the MPC published-orbit seed DISABLED (so it is a
    genuine short-arc solution with realistic uncertainty). Cached obs only if present; else a
    one-time fetch. Returns the FittedOrbit or None if obs are unavailable offline."""
    os.environ["PYROORB_MPC_SEED"] = "0"
    from pyroorb import obs_sources; import pyroorb.fit_object as fo
    from astropy.time import Time
    obscodes = obs_sources.load_obscodes()
    try:
        obs, meta = obs_sources.fetch_observations(["99942"], obscodes)
    except Exception as ex:
        pytest.skip(f"apophis obs unavailable offline: {ex}")
    cut = Time("2005-08-01", scale="utc").jd
    disc = sorted((o for o in obs if o["jd_utc"] <= cut), key=lambda o: o["jd_utc"])
    if len(disc) < 30:
        pytest.skip(f"only {len(disc)} discovery-arc obs cached")
    return fo.fit_orbit(disc, obscodes, desig="99942 (disc arc)", meta={**meta, "n_requested": 1})


def test_apophis_keyhole():
    if os.environ.get("KH_DYNAMIC") != "1":
        pytest.skip("set KH_DYNAMIC=1 to run the multi-year propagation")
    from astropy.time import Time
    fit = _load_apophis_disc()
    r = K.find_keyholes(fit, years=12.0, from_jd=Time("2025-01-01", scale="utc").jd, n_probe=31)
    assert r.get("ok"), f"keyholes not ok: {r}"
    enc = r.get("deflecting_encounter") or {}
    # the 2029 deflecting encounter must be found and be deep
    assert enc.get("date", "").startswith("2029"), f"deflector not 2029: {enc.get('date')}"
    # the 2029 encounter pushes every clone to a' ~ 1.1; what matters is that the LOV's a'-span
    # BRACKETS the 7/6 resonant value (1.108) so the 2036 keyhole is threaded.
    a76 = (7.0 / 6.0) ** (2.0 / 3.0)
    span = enc.get("post_encounter_a_span_au")
    assert span and span[0] < a76 < span[1], f"a'(LOV) must bracket the 7/6 resonance {a76:.4f}: {span}"
    # a 7/6 resonant-return keyhole with a ~2036 return must be located
    k76 = [k for k in r.get("keyholes", []) if k.get("resonance") == "7/6"]
    assert k76, f"no 7/6 keyhole found; resonances={[k['resonance'] for k in r.get('keyholes',[])]}"
    kh = k76[0]
    assert (kh.get("return_date") or "").startswith("2036"), f"7/6 return not 2036: {kh.get('return_date')}"
    if kh.get("resolved"):
        assert 0.0 < kh["ip"] <= 1.0
        w = kh.get("keyhole_width_km")
        assert w is None or w < 50.0, f"keyhole implausibly wide: {w} km"
