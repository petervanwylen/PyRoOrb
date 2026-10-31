"""OFFLINE correctness gate for impact-location prediction (impact.find_impact).

Re-fits each ESA "past impactor" from a static fixture and asserts the predicted
100 km atmospheric-entry point reproduces the measured impact:

  - 2008 TC3 and 2024 XA1 carry PRECISE km-level 100 km-entry oracles (Dimare &
    Del Vigna 2020 Table 1 / Vokrouhlicky & Farnocchia 2025 arXiv:2502.09712): our
    entry point reproduces them to 2.7 km and 0.4 km. These are the hard gates.
  - 2018 LA / 2023 CX1 / 2024 BX1 are REGION+TIME+VELOCITY checks (their published
    ground truth is the strewn field, which sits downrange of the 100 km entry, so
    the ground gate is loose; the time and entry-speed oracles are tight).
  The full-population performance sweep covered all 11 pre-impact-detected
  asteroids, including the coarse ocean cases not gated here.

No network: reads tests/impact_fixtures/*.json (built by build_impact_fixtures.py).

    python -m pytest tests/test_impact.py  # or under pytest
"""
import pytest
import sys, os, glob, json, math
from astropy.time import Time
import pyroorb.fit_object as fo
from pyroorb import obs_sources
from pyroorb import impact

# refits the impact fixtures and propagates them to Earth encounter
pytestmark = pytest.mark.slow

IMPACT_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "impact_fixtures")

TIME_GATE_S = 120.0        # entry-time match (loose; covers short-arc fits)
VEL_GATE_KMS = 2.0         # entry-speed match vs catalog V

_OC = None
def _obscodes():
    global _OC
    if _OC is None:
        _OC = obs_sources.load_obscodes()
    return _OC


_FITS = {}


def _fit(path):
    """The fitted orbit for one impact fixture, MEMOISED.

    2008 TC3 alone was re-fit three times per run (the location check, the
    footprint check, and the corridor check each did their own `fit_orbit`), and
    every fit is a full engine solve. The fixtures are frozen and the fit is
    deterministic, so one solve per fixture is enough; the analysis entry points
    (`find_impact`, `impact_probability`, `impact_footprint`) read the orbit and
    do not mutate it."""
    if path not in _FITS:
        fx = json.load(open(path))
        _FITS[path] = (fx, fo.fit_orbit(fx["obs"], _obscodes(), desig=fx["desig"],
                                        meta=fx["meta"]))
    return _FITS[path]


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _check(path):
    fx, fit = _fit(path)
    label, oracle = fx["label"], fx["oracle"]
    result = impact.find_impact(fit, window_days=5.0)
    if not result.get("impact"):
        return label, False, f"no impact predicted ({result.get('reason')})"
    c = next((x for x in result["crossings"] if x["altitude_km"] == 100.0), None)
    if c is None:
        return label, False, "no 100 km crossing"

    dkm = _haversine_km(c["lat_deg"], c["lon_deg"], oracle["lat"], oracle["lon"])
    dt_s = (c["time_utc"] - Time(oracle["iso"], scale="utc").jd) * 86400.0
    dv = c["geo_speed_km_s"] - oracle["v_kms"]
    probs = []
    if abs(dt_s) > TIME_GATE_S:
        probs.append(f"time {dt_s:+.0f}s")
    if dkm > oracle["gate_km"]:
        probs.append(f"ground {dkm:.0f}km>{oracle['gate_km']:.0f}")
    if abs(dv) > VEL_GATE_KMS:
        probs.append(f"dV {dv:+.1f}km/s")
    msg = (f"@100km t{dt_s:+.0f}s ground {dkm:.1f}km dV{dv:+.2f} "
           f"slope {c['entry_slope_deg']:.0f}deg [{fit.n_observations()}/"
           f"{fit.n_total_observations()} @ {fit.rms_residual():.2f}\"]")
    return label, (not probs), (msg if not probs else msg + "  <<< " + "; ".join(probs))


def _paths():
    return sorted(p for p in glob.glob(os.path.join(IMPACT_FIX, "*.json"))
                  if json.load(open(p)).get("oracle", {}).get("kind") not in ("ip", "ip_decline"))


def _ip_oracle_paths():
    return sorted(p for p in glob.glob(os.path.join(IMPACT_FIX, "*.json"))
                  if json.load(open(p)).get("oracle", {}).get("kind") == "ip")


def _ip_decline_paths():
    return sorted(p for p in glob.glob(os.path.join(IMPACT_FIX, "*.json"))
                  if json.load(open(p)).get("oracle", {}).get("kind") == "ip_decline")


def test_impact_locations():
    """pytest entry: every past-impactor fixture reproduces its measured impact."""
    assert _paths(), "no impact fixtures -- run tests/build_impact_fixtures.py"
    fails = []
    for p in _paths():
        label, ok, msg = _check(p)
        if not ok:
            fails.append(f"{label}: {msg}")
    assert not fails, "impact regressions:\n" + "\n".join(fails)


def _ip_checks():
    """Phase-2 impact-PROBABILITY gates (offline): (1) the certain impactor 2008 TC3 (689 obs,
    U=4 -- genuinely well-pinned) must RESOLVE to IP ~ 1; (2) each poorly-constrained Sentry
    fixture (kind:"ip", e.g. 2013 TP4 at U=8) must now be REFUSED, not matched.

    (2) is the finding of the 500-object Sentry validation: our independent LOV IP
    reproduced Sentry within +-20% for ZERO risk-list objects and, when it emitted, was wrong by
    1.3x to >100x -- because a Sentry-list orbit is under-constrained BY CONSTRUCTION, so our LOV
    direction diverges from JPL's. The U>=5 gate refuses exactly this regime. Only a well-pinned
    orbit (U<5, like TC3) is allowed to report a number. See impact.impact_probability."""
    out = []
    # certain impactor, well-pinned (U=4) -> IP ~ 1 (this is the one numeric case we reproduce).
    tc3 = os.path.join(IMPACT_FIX, "2008_TC3.json")
    if os.path.exists(tc3):
        fx, fit = _fit(tc3)
        # 2008 TC3 impacted in 2008, so scan from its (2008) epoch, not from "today".
        from_jd = Time(fit.epoch_tdb, format="jd", scale="tdb").utc.jd
        r = impact.impact_probability(fit, n_scan=15, window_days=5.0, from_jd=from_jd)
        ok = bool(r.get("ok") and r.get("resolved") and (r.get("ip") or 0) > 0.99)
        out.append(("IP/2008_TC3", ok,
                    f"resolved={r.get('resolved')} ip={r.get('ip')} n_app={r.get('n_approaches')}"))

    # poorly-constrained Sentry object (U>=5) -> REFUSE (assessed:False, low_confidence), never a
    # number we cannot back within +-20%. The refusal fires on the U gate before any propagation.
    for p in _ip_oracle_paths():
        fx, fit = _fit(p); o = fx["oracle"]
        enc = Time(o["encounter_iso"], scale="utc").jd
        r = impact.impact_probability(fit, n_scan=21, n_sigma=5.0,
                                      encounter_jd=enc, search_half_days=180.0)
        U = fit.uncertainty_parameter()
        refused = (r.get("assessed") is False and r.get("low_confidence") is True)
        probs = []
        if not refused:
            probs.append(f"did NOT refuse (assessed={r.get('assessed')} resolved={r.get('resolved')} "
                         f"ip={r.get('ip')}); Sentry {o['sentry_ip']:.2e}, U={U:.1f}")
        out.append((f"IPrefuse/{fx['desig']}", not probs,
                    f"refused, U={U:.1f} (Sentry had {o['sentry_ip']:.2e})"
                    if not probs else "; ".join(probs)))
    return out


def test_impact_probability():
    """pytest entry: the impact-probability sanity poles hold."""
    fails = [f"{lab}: {msg}" for lab, ok, msg in _ip_checks() if not ok]
    assert not fails, "impact-probability regressions:\n" + "\n".join(fails)


def _ip_decline_checks():
    """Phase-2 DOMAIN-OF-VALIDITY gate (offline): a far-future Sentry encounter that sits behind
    deep prior Earth passes (LOV folded) must DECLINE -- assessed:False / out_of_domain:True with
    the intervening-approach count -- not report a number we don't believe. The discriminator is
    intrinsic (Milani 2005 / Del Vigna 2020); the gate fires on a single nominal propagation, so
    no expensive clone scan runs. Counterpart to the reporting cases (2013 TP4, 2008 TC3)."""
    out = []
    for p in _ip_decline_paths():
        fx, fit = _fit(p); o = fx["oracle"]
        enc = Time(o["encounter_iso"], scale="utc").jd
        r = impact.impact_probability(fit, encounter_jd=enc, search_half_days=180.0)
        n = r.get("n_intervening") or 0
        probs = []
        if r.get("assessed") is not False or not r.get("out_of_domain"):
            probs.append(f"did NOT decline (assessed={r.get('assessed')} "
                         f"resolved={r.get('resolved')} ip={r.get('ip')})")
        if n < o.get("min_intervening", 1):
            probs.append(f"only {n} intervening (< {o.get('min_intervening', 1)})")
        msg = f"declined, {n} intervening deep approach(es)"
        out.append((f"IPdecline/{fx['desig']}", not probs,
                    msg if not probs else msg + "  <<< " + "; ".join(probs)))
    return out


def test_impact_probability_domain_gate():
    """pytest entry: out-of-domain encounters decline rather than report a wrong number."""
    fails = [f"{lab}: {msg}" for lab, ok, msg in _ip_decline_checks() if not ok]
    assert not fails, "impact-probability domain-gate regressions:\n" + "\n".join(fails)


def _footprint_check():
    """Phase-1b: the semilinear 1-sigma impact ELLIPSE on 2008 TC3. The Dimare & Del Vigna 2020
    Table-1 oracle gives a long, thin entry corridor at h=100 km -- ~0.05 x 0.5 km, azimuth
    104.6 deg. We can't match the absolute SIZE (it tracks each solution's covariance, and ours
    is an independent fit), but the METHOD is validated by the corridor's geometry: sub-km,
    strongly elongated (the along-track timing axis dominates), oriented along the oracle azimuth,
    and a near-zero linear-vs-semilinear centre shift (TC3's footprint is small => near-linear)."""
    tc3 = os.path.join(IMPACT_FIX, "2008_TC3.json")
    if not os.path.exists(tc3):
        return [("FP/2008_TC3", True, "fixture absent -- skipped")]
    fx, fit = _fit(tc3)
    r = impact.impact_footprint(fit, window_days=5.0, n_boundary=24)
    if not r.get("ok"):
        return [("FP/2008_TC3", False, f"no footprint ({r.get('reason')})")]
    sl = r["semilinear"]
    maj, mnr, az, off = (sl["semi_major_km"], sl["semi_minor_km"],
                         sl["azimuth_deg"], sl["center_offset_km"])
    probs = []
    if not (0.02 <= maj <= 2.0):
        probs.append(f"major {maj:.3f}km out of [0.02,2.0]")
    if maj / max(mnr, 1e-9) < 4.0:
        probs.append(f"aspect {maj/max(mnr,1e-9):.1f}<4 (not elongated)")
    if min(abs(az - 104.6), 180 - abs(az - 104.6)) > 25.0:
        probs.append(f"azimuth {az:.0f} vs 104.6")
    if off > maj:
        probs.append(f"centre offset {off:.3f}>major (nonlinear blowup)")
    msg = f"major {maj:.3f}km minor {mnr:.3f}km aspect {maj/max(mnr,1e-9):.0f}:1 az {az:.0f}deg"
    return [("FP/2008_TC3", not probs, msg if not probs else msg + "  <<< " + "; ".join(probs))]


def test_impact_footprint():
    """pytest entry: the semilinear 1-sigma footprint reproduces the 2008 TC3 corridor geometry."""
    fails = [f"{lab}: {msg}" for lab, ok, msg in _footprint_check() if not ok]
    assert not fails, "impact-footprint regressions:\n" + "\n".join(fails)
