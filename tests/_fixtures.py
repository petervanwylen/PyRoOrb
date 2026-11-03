"""Shared helpers for the fixture tooling -- imported by build_fixtures.py (full
rebuild, needs network), update_goldens.py (offline re-baseline), and
test_regression.py (offline guard). Single source of truth for: where fixtures
live, which obs fields are persisted, how a golden baseline is computed, the
ok/suspect link rule, and the atomic write. Keeping these here means the builder,
the updater, and the test can never drift apart.
"""
import sys, os, glob, json
# make fit_object importable (parent dir) AND this module importable from siblings
import numpy as np

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

# Per-obs fields persisted in each fixture -- exactly what fit_orbit consumes
# (positions already debiased, per-obs sigmas attached).
OBS_KEYS = ["jd_utc", "ra_deg", "dec_deg", "obscode", "packed", "designation",
            "sigma_ra", "sigma_dec", "astcat", "mode", "uncert_src", "mag", "band",
            "mag_sigma", "debiased",
            # Observer position for the codeless stations -- a space-based satellite
            # vector ("sat_pos_au") or a roving observer's Earth-fixed ITRS position
            # ("site_itrs_au"). Without these the re-fit can't place those obs (e.g.
            # P/2023 JN16 has 27 code-247 roving obs), so a fixture that omitted them
            # silently degraded its own baseline. None for ordinary ground stations.
            "sat_pos_au", "site_itrs_au"]


def jsonable(v):
    """Coerce numpy scalars (the ADES path yields float32) to JSON-safe types."""
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def link_status(rms, used, total):
    """The ok/suspect rule -- ONE definition shared by builder, updater, and test."""
    return "ok" if (rms < 0.6 and used >= max(6, total * 0.55)) else "suspect"


def compute_golden(fit, secs):
    """The golden baseline dict recorded for an object (incl. the timing baseline).

    `secs` is CPU seconds (`time.process_time`), NOT wall clock -- see
    test_regression's module docstring for why. Callers: build_fixtures.py (new
    fixture) and update_goldens.py (re-baseline), both via test_regression.refit."""
    rms = fit.rms_residual()
    used, total = fit.n_observations(), fit.n_total_observations()
    return {"used": used, "total": total, "rms": round(rms, 4),
            "n_excluded": fit.n_excluded(), "status": link_status(rms, used, total),
            "secs": round(secs, 1), **nongrav_golden(fit)}


def nongrav_golden(fit):
    """The non-gravitational half of a golden: WHICH model was delivered, and each fitted
    coefficient's SIGN.

    Recorded because nothing else in the golden can see a selection flip. `used`/`rms`/`status`
    move only a little when an object changes force model -- 2003 HM went None -> yarkovsky at
    130 obs / 0.7355" -> 134 / 0.7466", which every existing tolerance absorbs -- so a change to
    the selection rule (replacing the effect-size floors with a BIC bar moved at least
    two fixtures) could not fail this suite. The model name is the guard.

    The SIGN, not the value: a coefficient's magnitude moves with convergence, the mask and the
    epoch, and pinning it would make the suite brittle for no gain -- but the sign is the physical
    claim (A2's sign is the drift direction, and a NEGATIVE A1 is refused outright by
    _asteroid_phys_ok), so a silent sign flip is exactly the regression worth failing on. An exact
    0.0 is recorded as sign 0: this project reads it as a STALL, not a finding, so it must not
    compare equal to either sign. Gravity fits record `nongrav: None` and an empty sign map."""
    import math
    params = dict(getattr(fit, "nongrav_params", None) or {})
    signs = {}
    for k, v in sorted(params.items()):
        v = float(v)
        signs[k] = 0 if (v == 0.0 or math.isnan(v)) else (1 if v > 0 else -1)
    return {"nongrav": getattr(fit, "nongrav_model", None), "nongrav_signs": signs}


def slim_obs(obs):
    """Reduce parsed obs dicts to the JSON-safe subset stored in a fixture."""
    return [{k: jsonable(o.get(k)) for k in OBS_KEYS} for o in obs]


def write_fixture(path, payload, _tries=5):
    """Atomic write (tmp -> flush+fsync -> rename) then read-back verify, with RETRY.
    The sandbox FS can truncate a large write at a 512 KB boundary (or leave null-byte
    holes), so always go through tmp+fsync AND re-write until the read-back parses --
    a single attempt intermittently corrupts the >0.5 MB heavy-object fixtures."""
    blob = json.dumps(payload)
    tmp = path + ".tmp"
    for attempt in range(_tries):
        with open(tmp, "w", encoding="utf-8") as fp:
            fp.write(blob); fp.flush(); os.fsync(fp.fileno())
        with open(tmp, encoding="utf-8") as fp:                   # verify the TMP before it clobbers path
            try:
                json.load(fp)
            except json.JSONDecodeError:
                if attempt == _tries - 1:
                    raise
                continue                        # truncated write -> rewrite
        os.replace(tmp, path)
        return


def load_fixture(path):
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def iter_fixtures():
    return sorted(glob.glob(os.path.join(FIX, "*.json")))


def fixture_obs(name):
    """The (obs, meta) of ONE frozen fixture, by stem -- ``fixture_obs("2023_QJ3")``.

    For tests that need a real, well-determined object but not FRESH astrometry.
    They used to call ``obs_sources.fetch_observations`` instead, which is
    TTL-gated (see ``obs80.cache_fresh``: a cache older than the TTL is re-fetched
    on purpose, so a fit never stays pinned to a stale arc). That makes a "cached
    object" test quietly network-dependent -- with the MPC unreachable the fetch
    returns no rows and the fit raises ``need at least 3 observations (got 0)``,
    which reads like a fit_object regression and is not. The fixture archive is
    the mechanism that already exists for this; it is the same astrometry, frozen.
    """
    fx = load_fixture(os.path.join(FIX, f"{name}.json"))
    return fx["obs"], fx["meta"]


def tracklets(obs, max_min=120.0):
    """Split observations into tracklets: one station, gaps <= `max_min` minutes."""
    o = sorted(obs, key=lambda r: r["jd_utc"])
    groups, cur = [], [o[0]]
    for r in o[1:]:
        if r.get("obscode") != cur[-1].get("obscode") or \
           (r["jd_utc"] - cur[-1]["jd_utc"]) * 1440.0 > max_min:
            groups.append(cur)
            cur = [r]
        else:
            cur.append(r)
    groups.append(cur)
    return groups


def best_tracklet(obs):
    """The richest single tracklet of `obs` -- most observations, ties broken by span."""
    return max(tracklets(obs), key=lambda t: (len(t), t[-1]["jd_utc"] - t[0]["jd_utc"]))
