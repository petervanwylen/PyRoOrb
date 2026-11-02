"""Build STATIC astrometry fixtures for the impact-location regression test.

Run ONCE (needs network + the fetch cache) to snapshot the ESA "past impactor"
oracles into tests/impact_fixtures/*.json. Each fixture holds the parsed obs +
meta (what fit_orbit consumes) and the measured-impact ORACLE the test asserts
against. Kept SEPARATE from the main regression fixtures so test_impact runs as its
own offline correctness gate and does not bloat / slow the linkage suite.

    python tests/build_impact_fixtures.py
"""
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import sys, os, json, warnings; warnings.filterwarnings("ignore")
from _fixtures import slim_obs, write_fixture
from pyroorb import obs_sources
IMPACT_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "impact_fixtures")

# label : (designation, oracle).  Oracle fields:
#   iso         measured impact/airburst time, UTC
#   v_kms       atmospheric-entry speed (ESA NEOCC / per-event papers)
#   lat,lon     ground reference point (deg); `precise` flags the km-level oracle
#   region      human label
#   gate_km     allowed ground offset at 100 km (precise: km-level; others: region)
#   precise     True only for the km-level Dimare/JPL 2008 TC3 oracle
OBJECTS = {
    "2008_TC3": ("2008 TC3", {
        "iso": "2008-10-07 02:45:30", "v_kms": 12.4,
        "lat": 21.0871, "lon": 30.5378, "region": "Nubian Desert, Sudan",
        "gate_km": 5.0, "precise": True,
        "source": "Dimare & Del Vigna 2020 Table 1 / JPL Farnocchia 2017, at h=100 km"}),
    "2018_LA": ("2018 LA", {
        "iso": "2018-06-02 16:45:30", "v_kms": 16.98,
        "lat": -22.5, "lon": 25.4, "region": "Central Kalahari, Botswana",
        "gate_km": 250.0, "precise": False,
        "source": "ESA NEOCC past-impactors; strewn field (downrange of 100 km entry)"}),
    "2023_CX1": ("2023 CX1", {
        "iso": "2023-02-13 02:59:22", "v_kms": 14.02,
        "lat": 49.83, "lon": 0.78, "region": "Normandy, France",
        "gate_km": 250.0, "precise": False,
        "source": "ESA NEOCC past-impactors; strewn field (downrange of 100 km entry)"}),
    "2024_BX1": ("2024 BX1", {
        "iso": "2024-01-21 00:32:44", "v_kms": 15.26,
        "lat": 52.62, "lon": 12.76, "region": "Ribbeck, Germany",
        "gate_km": 250.0, "precise": False,
        "source": "ESA NEOCC past-impactors / A&A 686 A67; strewn field"}),
    "2024_XA1": ("2024 XA1", {
        "iso": "2024-12-03 16:14:52", "v_kms": 15.52,
        "lat": 60.6285, "lon": 119.0713, "region": "Sakha Republic, Siberia",
        "gate_km": 8.0, "precise": True,
        "source": "Vokrouhlicky & Farnocchia 2025 (arXiv:2502.09712), at h=100 km "
                  "-- second km-level entry oracle alongside 2008 TC3"}),
}


# Phase-2 impact-PROBABILITY oracles (JPL CNEOS Sentry). `kind:"ip"`; the test compares the
# LOV b-plane IP against Sentry's per-encounter virtual impactor (date, ip, sigma-LOV, v_imp).
IP_OBJECTS = {
    "IP_2013_TP4": ("2013 TP4", {
        "kind": "ip", "encounter_iso": "2026-10-01", "sentry_ip": 3.526e-5,
        "sentry_sigma_lov": 1.457, "sentry_v_imp_kms": 27.64,
        "ip_factor": 4.0, "sigma_tol": 0.15, "v_tol_kms": 1.5,
        "source": "JPL Sentry; well-determined 9.9-day/23-obs arc -> clean partial-IP match"}),
}


# Phase-2 DOMAIN-OF-VALIDITY oracles. `kind:"ip_decline"`: a far-future Sentry encounter that
# sits behind deep prior Earth passes, so our 1-D LOV scan is out of its validated regime and the
# engine MUST decline (assessed:False / out_of_domain:True) rather than report a number. The
# intervening-encounter count is intrinsic (no Sentry comparison) and separates these cleanly from
# the reporting cases above (2013 TP4 / 2008 TC3: 0 intervening). Milani 2005 / Del Vigna 2020.
IP_DECLINE_OBJECTS = {
    "IPDECLINE_2000_SG344": ("2000 SG344", {
        "kind": "ip_decline", "encounter_iso": "2071-09-16", "min_intervening": 3,
        "source": "JPL Sentry 2071 virtual impactor behind ~6 deep Earth passes -- out of "
                  "our 1-D LOV domain; our independent IP came out ~30x below Sentry's"}),
}


def main():
    os.makedirs(IMPACT_FIX, exist_ok=True)
    oc = obs_sources.load_obscodes()
    only = sys.argv[1:]
    for label, (desig, oracle) in {**OBJECTS, **IP_OBJECTS, **IP_DECLINE_OBJECTS}.items():
        if only and not any(o in label or o in desig for o in only):
            continue
        print(f"fetching {desig} ...", flush=True)
        obs, meta = obs_sources.fetch_observations(desig, oc)
        payload = {"label": label, "desig": desig, "obs": slim_obs(obs),
                   "meta": meta, "oracle": oracle}
        path = os.path.join(IMPACT_FIX, label + ".json")
        write_fixture(path, payload)
        print(f"  wrote {path}  ({len(obs)} obs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
