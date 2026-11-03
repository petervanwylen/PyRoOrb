"""Framework-agnostic JSON bridge between the PyRoOrb engine and the web UI.

This module is deliberately free of any web-framework imports: it turns a
designation (or an uploaded ADES/OBS80 blob) into a single JSON-serialisable
dict that the front end renders. The stdlib ``server.py`` calls it today; a
future FastAPI app (per WEB_UI_PLAN §1 backend sketch) can import the very same
functions unchanged.

It fetches/runs/serialises a fit, then holds the evaluator in a lightweight
session so the UI can toggle observations and re-fit (see ``_store_session`` and
the session helpers below). Serialisation is a pure *read* of the existing engine
outputs (``fit_orbit`` -> ``FittedOrbit.as_dict``/``.residuals``/``.history``),
plus a small Keplerian element->xyz sampler so the browser can draw the orbit in
3D without re-deriving the geometry. Display formatting shared with the CLI report
(the orbital-element table, scalar coercion) lives in ``presentation``.
"""
import contextlib
import os
import re
import sys
import math
import secrets
import threading
import time
import uuid

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
from pyroorb import guards  # noqa: E402
from pyroorb.web import capacity  # noqa: E402
import pyroorb.deadline as _deadline  # noqa: E402  (cooperative stop of a running ITF search)
from pyroorb.itf.obs80_fields import obs_unique_id  # noqa: E402
from pyroorb.itf.verdict import SearchVerdict  # noqa: E402
from pyroorb import obs_sources  # noqa: E402
from pyroorb import datadirs  # noqa: E402  (where download caches live)
import pyroorb.fit_object as fo  # noqa: E402
from pyroorb import arc_extend  # noqa: E402  (teaching-mode snapshot flag lives here)
from pyroorb import ident_check  # noqa: E402  (systematic-misfit flag for the O-C pane)
from pyroorb import constraint  # noqa: E402  (essentially-unconstrained-orbit heads-up banner)
from pyroorb import obs80  # noqa: E402  (synthesize MPC 80-col lines for the astrometry view)
from pyroorb import presentation  # noqa: E402  (element table + scalar coercion shared with the CLI)
from pyroorb.web import fitcache  # noqa: E402  (resolve a pasted fit's obs by its export key, for add-to-fit)
import pyroorb.nongrav as _ngm  # noqa: E402  (SRP1AU / the AMR find_orb reports)
from pyroorb.web import itf_wording  # noqa: E402  (ITF's 1.5/1.75-opp + 0/0.6/1 shorthands -> plain English)

# Scalar coercion shared with the CLI report -- numpy/None/non-finite -> JSON-safe.
_safe = presentation.safe

# The observatory-code table, parsed once and reused on every fit -- re-parsed when the
# server's periodic refresh (server._maybe_start_obscodes_refresh) changed the file.
_OBSCODES = None
_OBSNAMES = None
_OBSCODES_TOK = None
_OBSNAMES_TOK = None


def _obscodes():
    global _OBSCODES, _OBSCODES_TOK
    tok = obs_sources.obscodes_token()
    if _OBSCODES is None or tok != _OBSCODES_TOK:
        _OBSCODES, _OBSCODES_TOK = obs_sources.load_obscodes(), tok
    return _OBSCODES


def _obsnames():
    """Cached MPC obscode -> observatory name map (display-only, for the obs panel)."""
    global _OBSNAMES, _OBSNAMES_TOK
    tok = obs_sources.obscodes_token()
    if _OBSNAMES is None or tok != _OBSNAMES_TOK:
        _OBSNAMES, _OBSNAMES_TOK = obs_sources.load_obscode_names(), tok
    return _OBSNAMES


def _mp_names(designations):
    """Resolve the fitted designations to their MP number + name for the object label.

    The status strip shows a working designation (e.g. '1927 EC'); this looks up the
    published '(279) Thule' form via the MPC orbits API so the header can carry the
    number and any name. Returns a list of dicts (one per DISTINCT resolved body, in
    first-seen order):
        number, name -- as strings, either possibly None
        label        -- '(279) Thule', or '(626135)' when unnamed, or just the name
                        when a body is somehow named but unnumbered

    A single object queried by name/number expands to several alias designations
    (Thule -> ['1927 EC','1954 FF','279','A888 UA','A920 GA']); each lookup returns the
    body's full alias set, so the remaining aliases are skipped rather than re-fetched.
    A genuinely distinct linked body still gets its own entry. Best-effort and
    display-only: any lookup that fails (unnumbered NEOCP object, network hiccup) is
    simply omitted, and the caller falls back to the raw designation."""
    from pyroorb.mpc_orbit import fetch_designation_info
    out, seen, keys = [], set(), set()
    for d in designations:
        if d in seen:
            continue                            # covered by an already-resolved body
        try:
            cache = os.path.join(datadirs.cache_dir(), "mpcorb_%s.json" % obs80.cache_stem(d))
            info = fetch_designation_info(d, cache=cache)
        except Exception:
            info = None
        if not info:
            continue
        seen |= info.get("aliases") or set()
        key = info.get("number") or info.get("name")
        if key in keys:
            continue                            # same body reached via a second alias
        keys.add(key)
        num, name = info.get("number"), info.get("name")
        label = ("(%s) %s" % (num, name)) if (num and name) \
            else ("(%s)" % num) if num else name
        out.append({"number": num, "name": name, "label": label})
    return out


# --------------------------------------------------------------------------
# Environment / data-file status (read-only Settings panel).
#
# A fit silently DEGRADES when an optional runtime data file is absent: no
# asteroid kernel -> gravity-only force model; no EFCC18 table -> undebiased
# astrometry. Both change residuals enough to flip a borderline observation's
# keep/exclude, which is exactly how two hosts running the same code can give
# different orbits. This panel surfaces what actually resolved on THIS host so
# such a divergence is diagnosable at a glance (it mutates nothing).
# --------------------------------------------------------------------------
def _human_bytes(n):
    """Bytes -> a short human string (e.g. '645 MB'). Whole units, terse."""
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return ("%d %s" % (round(f), unit)) if unit in ("B", "KB") \
                else ("%.1f %s" % (f, unit))
        f /= 1024
    return "%d B" % n


def _file_item(name, path, *, critical=False, note=""):
    """Probe one expected file: present? size? -- as a Settings-panel row dict.
    A symlink is resolved (a dev layout may link the big data dirs)."""
    item = {"name": name, "critical": critical, "note": note}
    try:
        p = os.path.realpath(str(path)) if path else ""
        item["path"] = str(path) if path else ""
        if path and os.path.isfile(p):
            item["present"] = True
            item["detail"] = _human_bytes(os.path.getsize(p))
        else:
            item["present"] = False
            item["detail"] = "absent"
    except OSError as ex:
        item["present"] = False
        item["detail"] = "unreadable (%s)" % type(ex).__name__
    return item


def environment_status():
    """Read-only inventory of what this host resolved: platform + every runtime
    data file (present/absent/size) + the physics-affecting env overrides. Pure
    inspection -- never mutates, never triggers a download."""
    import platform as _plat
    out = {"ok": True, "groups": []}

    out["platform"] = {
        "machine": _plat.machine(),
        "system": _plat.system(),
        "release": _plat.release(),
        "python": _plat.python_version(),
        "cwd": os.getcwd(),
    }

    # --- Ephemerides (required -- the integrator cannot run without DE440) ---
    import pyroorb.force_assist as fa  # lazy: pulls rebound/assist; engine already loads it
    ast_path = fa.asteroid_kernel_path()        # resolved kernel, or None -> gravity-only
    eph = [
        _file_item("DE440 planetary ephemeris", fa.PLANETS_PATH, critical=True,
                   note="Sun + 8 planets + Moon background; every fit integrates in it."),
        _file_item("16 massive-asteroid perturbers", ast_path or fa.ASTEROIDS_DEFAULT_PATH,
                   note="Optional sb441-n16 kernel. Absent -> the fit runs gravity-only, "
                        "which can shift residuals enough to flip a borderline observation."),
    ]
    out["groups"].append({"title": "Ephemerides", "items": eph})

    # --- Optional corrections that change residuals/weights ------------------
    from pyroorb import debias
    bias_present = debias.available()
    bias_file = (debias.DEBIAS_DIR / "bias.npy")
    if not bias_file.exists():
        bias_file = debias.DEBIAS_DIR / "bias.dat"
    corr = [
        _file_item("EFCC18 star-catalog debias table", bias_file,
                   note="Absent -> astrometry is left undebiased (bias_for -> 0); "
                        "residuals shift by up to ~tenths of an arcsec."),
    ]
    # available() also accepts the .npy cache; reflect the effective state.
    corr[0]["present"] = bool(bias_present)
    if bias_present and not corr[0].get("detail", "").endswith("B"):
        corr[0]["detail"] = "present"
    out["groups"].append({"title": "Corrections", "items": corr})

    # --- Caches & reference data (auto-fetched; nice-to-have) ----------------
    from pyroorb import mpcorb
    ref = [
        _file_item("MPC observatory codes (ObsCodes.htm)", obs_sources._current_obscodes_path(),
                   note="Station parallax constants; a copy ships with PyRoOrb, and a newer "
                        "one is downloaded when an observation names a station it lacks."),
        _file_item("MPCORB catalogue index", mpcorb.MPCORB_NPZ,
                   note="Local orbit catalogue for sampling / cross-checks."),
    ]
    out["groups"].append({"title": "Caches & reference data", "items": ref})

    # --- Physics-affecting environment overrides -----------------------------
    env_keys = [
        ("PYROORB_ASTEROIDS_PATH", "Whole-fit asteroid-perturber kernel override"),
        ("PYROORB_EPHEM_DIR", "Ephemeris directory override"),
        ("PYROORB_DEBIAS_DIR", "Debias-table directory override"),
        ("PYROORB_DEBIAS_AUTOFETCH", "Auto-download the debias table on first use"),
        ("PYROORB_OBSCODES_PATH", "Observatory-code table override"),
    ]
    env_items = []
    for key, desc in env_keys:
        val = os.environ.get(key)
        env_items.append({"name": key, "note": desc, "present": None,
                          "detail": val if val else "(default)"})
    out["groups"].append({"title": "Environment overrides", "items": env_items})

    return out


# --------------------------------------------------------------------------
# Orbit geometry: classical elements -> heliocentric ecliptic xyz (au).
# Sampled purely for visualisation; the science is untouched.
# --------------------------------------------------------------------------
def _rotate_to_ecliptic(x_orb, y_orb, asc_node, arg_per, incl):
    """Rotate a point in the orbital plane (perifocal x toward perihelion) into
    heliocentric ecliptic coordinates via the standard 3-1-3 (Omega, i, omega)."""
    O = math.radians(asc_node)
    w = math.radians(arg_per)
    i = math.radians(incl)
    cO, sO = math.cos(O), math.sin(O)
    cw, sw = math.cos(w), math.sin(w)
    ci, si = math.cos(i), math.sin(i)
    # Perifocal -> ecliptic rotation matrix columns applied to (x_orb, y_orb, 0).
    px = (cO * cw - sO * sw * ci)
    py = (sO * cw + cO * sw * ci)
    pz = (sw * si)
    qx = (-cO * sw - sO * cw * ci)
    qy = (-sO * sw + cO * cw * ci)
    qz = (cw * si)
    return [px * x_orb + qx * y_orb,
            py * x_orb + qy * y_orb,
            pz * x_orb + qz * y_orb]


def _kepler_E(M_deg, e):
    """Solve Kepler's equation E - e sin E = M for the eccentric anomaly."""
    M = math.radians(M_deg % 360.0)
    E = M if e < 0.8 else math.pi
    for _ in range(60):
        dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= dE
        if abs(dE) < 1e-12:
            break
    return E


SOLAR_RADIUS_AU = 0.00465047     # IAU 2015 nominal solar radius: a perihelion inside it is not an orbit


def _orbit_path(elem, n=720):
    """Sample the closed ellipse plus the orbital angles + epoch the front end
    needs to place the body at ANY date (it advances M by mean motion and draws
    the moving dot itself, so the date slider scrubs without a backend call).
    n was 240: at Orbarium's tight NEO-encounter zoom (span down to ~0.05 au) the
    straight chords between sample points were visibly off the true ellipse, same
    class of bug as PLANET_ORBIT_SEGMENTS in app.js -- 3x here for the same reason."""
    a = float(elem["a"])
    e = float(elem["e"])
    inc = float(elem["i"])
    node = float(elem["asc_node"])
    peri = float(elem["arg_per"])
    M = float(elem["M"])
    # mean motion (deg/day) from the period, for the client-side date advance
    n_deg = None
    try:
        P_years = float(elem.get("P"))
        if P_years > 0:
            n_deg = 360.0 / (P_years * 365.25)
    except (TypeError, ValueError):
        pass
    # `closed` says whether `path` is a full ellipse whose last point joins its first. The
    # client's ecliptic split walks that seam to catch a node landing on it, and walking it
    # on an OPEN curve drew a straight chord from one end of the hyperbola to the other.
    out = {"a": _safe(a), "e": _safe(e), "i": _safe(inc), "node": _safe(node),
           "peri": _safe(peri), "M": _safe(M),
           "epoch_jd": _safe(elem["epoch"]), "n_deg_per_day": _safe(n_deg),
           "q": _safe(elem["q"]), "Q": _safe(elem["Q"]),
           "path": None, "position": None, "closed": True}
    if a > 0 and 0 <= e < 1.0:                  # closed ellipse -> sample it
        b = a * math.sqrt(1.0 - e * e)
        path = []
        for k in range(n + 1):
            E = 2.0 * math.pi * k / n
            path.append(_rotate_to_ecliptic(a * (math.cos(E) - e), b * math.sin(E),
                                           node, peri, inc))
        E_now = _kepler_E(M, e)
        out["path"] = path
        out["position"] = _rotate_to_ecliptic(a * (math.cos(E_now) - e),
                                              b * math.sin(E_now), node, peri, inc)
    elif e > 1.0 and a < 0:
        # Unbound: there is no closed orbit, so draw the branch out to a few times the perihelion
        # distance, and place the body by the hyperbolic Kepler equation (M = e sinh H - H, M in
        # radians and never wrapped). The mean motion is as well defined as for an ellipse
        # (element_vector: n = sqrt(mu / (-a)^3), carried on the element dict as `n` in
        # deg/day); the client advances M by it and solves the HYPERBOLIC equation itself
        # (keplerXYZ branches on e), so the date slider moves the body along the branch.
        out["closed"] = False
        try:
            n_h = float(elem.get("n"))
            out["n_deg_per_day"] = _safe(n_h) if n_h > 0 else None
        except (TypeError, ValueError):
            out["n_deg_per_day"] = None
        s, sq = -a, math.sqrt(e * e - 1.0)
        r_max = max(6.0, 4.0 * a * (1.0 - e))
        hmax = math.acosh(max(1.0, (r_max / s + 1.0) / e))
        path = []
        for k in range(n + 1):
            H = -hmax + 2.0 * hmax * k / n
            path.append(_rotate_to_ecliptic(s * (e - math.cosh(H)), s * sq * math.sinh(H),
                                           node, peri, inc))
        Mr = math.radians(M)
        H_now = math.asinh(Mr / e)
        for _ in range(60):
            dH = (e * math.sinh(H_now) - H_now - Mr) / (e * math.cosh(H_now) - 1.0)
            H_now -= dH
            if abs(dH) < 1e-12:
                break
        out["path"] = path
        out["position"] = _rotate_to_ecliptic(s * (e - math.cosh(H_now)), s * sq * math.sinh(H_now),
                                              node, peri, inc)
    return out


# --------------------------------------------------------------------------
# Serialisation of a completed fit into the front-end payload.
# --------------------------------------------------------------------------
def _stage_snapshots(fit):
    """Per-stage residuals for the teaching replay: for each history entry tagged
    with a state+mask (SNAPSHOTS mode), recompute O-C for ALL obs at that stage's
    orbit. Index-aligned with `residuals` (same obs order). Entries without a
    snapshot (or if the evaluator is unavailable) come back as None."""
    ctx = getattr(fit, "_ctx", None)
    if not ctx or "ev" not in ctx:
        return []
    ev, sra, sdec = ctx["ev"], ctx["sigma_ra"], ctx["sigma_dec"]
    out = []
    for h in (fit.history or []):
        st, mk = h.get("_state"), h.get("_mask")
        if st is None or mk is None:
            out.append(None)
            continue
        try:
            dra, ddec = ev.resid_arcsec(np.asarray(st, dtype=float))
            rs = np.hypot(dra / sra, ddec / sdec)
            # resid_arcsec returns Computed - Observed; report O-C = Observed - Computed
            # (find_orb convention, matching FittedOrbit.residuals()). Negate for display.
            dra, ddec = -dra, -ddec
            out.append({
                "included": [bool(b) for b in mk],
                "d_ra_arcsec": [_safe(x) for x in dra],
                "d_dec_arcsec": [_safe(x) for x in ddec],
                "resid_sigmas": [_safe(x) for x in rs],
            })
        except Exception:
            out.append(None)
    return out


def _planetocentric_block(fit):
    """When the object is a natural satellite (``fit.center`` set -- by ``--center`` or by the
    on-by-default auto-detect), the OSCULATING PLANETOCENTRIC orbit about its planet: element rows
    in the SAME shape as the heliocentric table (so the UI can swap between them), plus the
    satellite class + range. None for an ordinary heliocentric object. See planetocentric.py."""
    center = getattr(fit, "center", None)
    if not center:
        return None
    try:
        import pyroorb.planetocentric as pc
        from pyroorb.orbital_elements import ELEM_NAMES
        elem = pc.planetocentric_elements(fit.state_vector(), fit.epoch_tdb, center)
        sig = pc.planetocentric_sigmas(fit, center)
        as_dict = {}
        for k in ELEM_NAMES:
            as_dict[k] = _safe(elem.get(k))
            as_dict[f"{k} sigma"] = _safe(sig.get(k))
        return {
            "center": center,
            "center_label": center.capitalize(),
            "class": pc.classify(elem),
            "bound": bool(elem["bound"]),
            "range_au": _safe(elem["planetocentric_range_au"]),
            "frame": f"J2000 ecliptic ({center.capitalize()} system barycentre)",
            "validated": pc.is_validated(center),
            "elements": presentation.element_rows(as_dict),
        }
    except Exception:
        return None


def _radar_block(fit):
    """The radar (delay/Doppler) summary for the payload, or None when the object has no
    radar. Mirrors the CLI's radar report (cli.py): the same delay/Doppler rows fetched by
    `obs_sources.fetch_observations` (JPL dump) or carried in a pasted/added file are already
    folded into `fit_orbit`'s final differential correction -- so the ORBIT is radar-refined in
    the web exactly as on the CLI; this just SURFACES that so the UI can show it instead of the
    radar being silently invisible. Station names are resolved like the CLI print."""
    rd = getattr(fit, "radar", None)
    if not rd:
        return None
    import pyroorb.radar as _radar
    rows = [{
        "kind": rr["kind"],                                   # 'delay' | 'doppler'
        "jd_utc": _safe(rr["jd_utc"]),
        "tx": _radar.station_name(rr["tx"]), "rx": _radar.station_name(rr["rx"]),
        # delay O-C in microseconds, Doppler in Hz -- same units the CLI prints
        "resid": _safe(rr["resid"] * (1e6 if rr["kind"] == "delay" else 1.0)),
        "unit": "us" if rr["kind"] == "delay" else "Hz",
        "z": _safe(rr["z"]), "included": bool(rr["included"]),
    } for rr in rd.get("rows", [])]
    return {
        "applied": bool(rd.get("applied")),
        "n": rd.get("n"), "n_used": rd.get("n_used"),
        "post_median_z": _safe(rd.get("post_median_z")),
        "note": rd.get("note"), "rows": rows,
    }


def serialise_fit(fit):
    """Turn a FittedOrbit into the single dict the browser consumes."""
    # The honest profile-likelihood nongrav sigma re-fits the orbit at many trial values -- seconds of
    # ASSIST work, far too slow to block any interactive step. So the browser NEVER auto-computes it:
    # force the fast LINEAR sigma here and let the "±" button (/api/nongrav_sigmas -> nongrav_sigmas)
    # pay for the real interval ON DEMAND. The UI hides the linear sigma until the button runs, so the
    # overconfident value is never presented as final. (CLI / as_dict elsewhere still serve the profile
    # sigma; only the browser display defers it.)
    if getattr(fit, "nongrav_model", None) not in (None, "none"):
        fit._nongrav_profile = False
        fit._rational_sigmas = None
    d = fit.as_dict()

    elements = presentation.element_rows(d)
    planetocentric = _planetocentric_block(fit)

    # Per-obs residuals, enriched with the sky position + magnitude the engine
    # keeps on the parsed-obs records (residuals() iterates fit._obs in order).
    raw = fit.residuals()
    obs_records = fit._obs
    # Per-obs photometric residuals (computed_mag, dmag) are index-aligned with
    # _obs (same order as residuals()) -- see FittedOrbit.magnitude_residuals.
    phot_per = getattr(fit, "_phot", {}).get("per", [])
    residuals = []
    for k, r in enumerate(raw):
        o = obs_records[k] if k < len(obs_records) else {}
        pp = phot_per[k] if k < len(phot_per) else {}
        residuals.append({
            "i": k,
            "jd_utc": _safe(r.get("jd_utc")),
            "obscode": r.get("obscode"),
            "designation": r.get("designation"),
            "ra_deg": _safe(o.get("ra_deg")),
            "dec_deg": _safe(o.get("dec_deg")),
            "mag": _safe(o.get("mag")),
            "band": o.get("band"),
            # Photometric O-C: dmag = obs_mag - computed_mag (mag), and the model's
            # computed V-band-reduced magnitude in the obs's own band. None when the
            # obs carried no magnitude or the geometry was invalid.
            "computed_mag": _safe(pp.get("computed_mag")),
            "dmag": _safe(pp.get("dmag")),
            # Per-obs geometry (au): r = heliocentric (Sun->object) distance,
            # delta = topocentric (observer/Earth->object) distance.
            "r_au": _safe(pp.get("r")),
            "delta_au": _safe(pp.get("delta")),
            "d_ra_arcsec": _safe(r.get("d_ra_arcsec")),
            "d_dec_arcsec": _safe(r.get("d_dec_arcsec")),
            "sigma_ra": _safe(r.get("sigma_ra")),
            "sigma_dec": _safe(r.get("sigma_dec")),
            "resid_sigmas": _safe(r.get("resid_sigmas")),
            "included": bool(r.get("included")),
            # Optional ADES provenance for the astrometry pane's "addl info" columns
            # (all None for OBS80 or an unpopulated submission -> rendered as "—").
            # Display-only extras that never touch the fit.
            "obs_id": o.get("obs_id"),
            "trksub": o.get("trksub"),
            "ades_rms_ra": _safe(o.get("ades_rms_ra")),
            "ades_rms_dec": _safe(o.get("ades_rms_dec")),
            "mag_sigma": _safe(o.get("mag_sigma")),
            "log_snr": _safe(o.get("log_snr")),
            # Canonical MPC 80-col line (col-65 marker cleared) for the astrometry
            # view; the front end re-inserts find_orb's 'x' for excluded obs.
            "obs80": obs80.obs80_base_line({**o, "jd_utc": r.get("jd_utc")}),
        })

    # Per-obscode roll-up (counts + median total residual) for the obs panel.
    by_code = {}
    for r in residuals:
        c = r["obscode"]
        by_code.setdefault(c, {"used": [], "total": 0})
        by_code[c]["total"] += 1
        if r["included"] and r["resid_sigmas"] is not None:
            tot = math.hypot(r["d_ra_arcsec"] or 0.0, r["d_dec_arcsec"] or 0.0)
            by_code[c]["used"].append(tot)
    observatories = []
    names = _obsnames()
    for c, v in sorted(by_code.items()):
        used = sorted(v["used"])
        med = used[len(used) // 2] if used else None
        observatories.append({"obscode": c, "name": names.get(c, ""), "n": v["total"],
                              "n_used": len(used), "median_resid": _safe(med)})

    H, sH = fit.absolute_magnitude()
    # Built here rather than inline in the payload because the unconstrained verdict below
    # edits both: an orbit that is not shown does not get to be shown as a class, a MOID or
    # an ellipse either.
    physical = {
        "class": d["class"], "pha": d["pha"],
        "moid_earth_au": _safe(d["moid_earth_au"]),
        "tisserand_jupiter": _safe(d["tisserand_jupiter"]),
        "U": _safe(d["U"]), "H": _safe(H), "H_sigma": _safe(sH),
        "G": _safe(d["G"]), "n_mags": d["n_mags"], "frame": d["frame"],
    }
    orbit3d = _orbit_path(d)

    # A non-physical solution (undefined elements, or a and e on opposite sides of e = 1) means
    # the fit didn't converge -- still show the residuals + trace, but warn that the orbit is
    # junk. An UNBOUND orbit is not that: e > 1 has a real perihelion time and mean anomaly
    # (orbital_elements.element_vector), and whether this arc determines it is constraint.assess's
    # question, answered below -- 1I/2I/3I used to get this warning for being interstellar.
    # Two more tells that do not depend on e: a perihelion inside the Sun, and residuals the
    # orbit does not fit by the engine's own catastrophe bar. Master only flagged the
    # 280000" C/2013 A1 full-arc fit (q = 2e-5 au) because it happened to be hyperbolic.
    from pyroorb.fit_object import CATASTROPHIC_RMS_N
    a_v, e_v, q_v = _safe(d.get("a")), _safe(d.get("e")), _safe(d.get("q"))
    rms_n = _safe(d.get("rms_normalized"))
    degenerate = (a_v is None or e_v is None or q_v is None
                  or _safe(d.get("M")) is None or (e_v < 1.0) != (a_v > 0)
                  or q_v < SOLAR_RADIUS_AU
                  or (rms_n is not None and rms_n > CATASTROPHIC_RMS_N))
    # Say WHICH test failed, in one sentence, with a next step the UI can actually offer.
    # (It used to list all three causes and end "try force" -- an engine option the UI
    # does not expose, per runFit(): dead advice on the page's most prominent line.)
    warning = None
    if degenerate:
        if rms_n is not None and rms_n > CATASTROPHIC_RMS_N:
            cause = (f"The best solution leaves residuals of {rms_n:.0f}σ, far beyond what "
                     f"any orbit through this astrometry can absorb, which usually means two "
                     f"different objects were fitted together.")
        elif q_v is not None and q_v < SOLAR_RADIUS_AU:
            cause = (f"The solution passes inside the Sun (perihelion {q_v:.4f} au), which no "
                     f"real orbit does; usually a wrong night, or two objects fitted together.")
        else:
            cause = ("The solve ended on undefined elements without settling, which usually "
                     "means the arc is too short or one night is wrong.")
        warning = ("No orbit fits these observations. " + cause + " The residuals and stage "
                   "trace are shown; the elements and the 3D view are not meaningful. Check the "
                   "per-night residuals, or add observations.")

    # Systematic-misfit flag: a within-apparition residual TREND that an otherwise
    # clean overall rms hides -- the tell of a wrong linkage / contaminated
    # apparition. Returns a short description past the operational threshold, else
    # None. The front end paints the O-C pane title red when this is set.
    try:
        systematic = ident_check.systematic_misfit(fit)
    except Exception:
        systematic = None

    # Essentially-unconstrained heads-up: a short arc that fits cleanly but leaves the
    # orbit's scale (a, e, Q, P) almost free -- the linear sigmas are then misleading
    # (huge), so we describe the honest non-linear spread. None for a determined orbit.
    try:
        constraint_warning = constraint.assess(fit)
    except Exception:
        constraint_warning = None

    # By design, when that verdict fires there is NO nominal orbit to show.
    # A value with a linear sigma beside it is indistinguishable, to a reader, from the ordinary
    # determined case -- and it will not reproduce against find_orb or anyone else, because every
    # code lands somewhere different in the same free valley. What replaces it is the interval
    # `constraint` supplies per element. The blanking happens HERE, on the server: /api/fit is a
    # plain JSON endpoint, so a number that must not be presented must not be sent. (The exports
    # refuse for the same reason, in server.py -- they travel further than any banner.)
    # The user may ASK for an orbit anyway: `nominal_optin` is set on the
    # FittedOrbit by `nominal_optin()` below, so it lives exactly as long as this one orbit --
    # any re-solve builds a new FittedOrbit and the page is back to the ranges. The banner
    # stays; the values come back marked as asked-for; the hazard-shaped products stay blank.
    optin = getattr(fit, "nominal_optin", None) if constraint_warning else None
    orbit_suppressed = bool(constraint_warning) and not optin
    if optin:
        rngs = constraint_warning.get("ranges") or {}
        for row in elements:
            row["range"] = rngs.get(row["name"])          # the 5-95% span rides on as a tooltip
        # Class and H come back (with the "~" the page puts on everything here); a PHA badge,
        # a MOID or a Tisserand would answer a HAZARD question from an orbit the arc did not
        # choose -- and on a tracklet one the population prior chose -- so they stay blank.
        physical.update({"pha": None, "moid_earth_au": None, "tisserand_jupiter": None,
                         "U": None})
        warning = None
    if orbit_suppressed:
        # The constraint verdict has already blanked the elements and explains why in its own
        # banner; a second "no orbit" line beside it tells a short-arc user two different stories
        # about one situation. (`degenerate` itself is kept: the products below still read it.)
        warning = None
        rngs = constraint_warning.get("ranges") or {}
        # Everything DERIVED from the nominal goes with it. A "PHA, MOID 0.003 au" badge or a
        # drawn ellipse is the same claim the element table was refused, made in a form that
        # looks even more settled -- and H needs r and Delta, so a free range frees it too.
        # What survives is what the OBSERVATIONS say by themselves: the arc, the residuals, the
        # station list, the magnitudes as reported, the force model and frame labels.
        physical.update({"class": None, "pha": None, "moid_earth_au": None,
                         "tisserand_jupiter": None, "U": None, "H": None, "H_sigma": None})
        orbit3d = None
        for row in elements:
            row["range"] = rngs.get(row["name"])
            row["value"] = None
            row["sigma"] = None
        if planetocentric:            # the same undetermined orbit, expressed about a planet
            for row in planetocentric.get("elements", []):
                row["range"] = None
                row["value"] = None
                row["sigma"] = None

    # Statistical degrees of freedom of the delivered fit: residual ROWS (2 per observation)
    # minus free parameters. Zero or negative means the solution interpolates its astrometry
    # exactly and the reported sigmas carry no measured scale (see FittedOrbit.degrees_of_
    # freedom). Numbers only -- the wording lives in the client (dof.js), like the
    # convergence banner.
    try:
        dof_block = {"dof": fit.degrees_of_freedom(), "n_resids": fit.n_resids(),
                     "n_params": fit.n_free_params(), "n_obs": fit.n_observations()}
    except Exception:
        dof_block = None

    # Unbound orbits (the rules in hyperbolic.py). Informational only -- never
    # `warning`: a genuine interstellar orbit is supposed to arrive with nothing to say. `check`
    # is what _prefer_bound_ecc weighed when the free fit came out e > 1 (unbound.js says why a
    # bound orbit is shown although an unbound one fits); `reminder` is a CLEARLY unbound orbit
    # whose excess energy a planet could have supplied. Skipped on a suppressed or degenerate
    # orbit, which already says the elements are not to be read.
    unbound_block = None
    if not degenerate and not orbit_suppressed:
        from pyroorb import hyperbolic
        check = getattr(fit, "unbound_check", None)
        reminder = None
        if e_v is not None and e_v > 1.0:
            try:
                reminder = hyperbolic.inbound_reminder(
                    fit.elements(), fit.element_sigmas().get("e"), fit.epoch_tdb)
            except Exception:
                reminder = None
        if check or reminder:
            unbound_block = {
                "check": ({k: (_safe(v) if isinstance(v, float) else v) for k, v in check.items()}
                          if check else None),
                "reminder": ({k: (_safe(v) if isinstance(v, float) else v)
                              for k, v in reminder.items()} if reminder else None),
            }

    return {
        "ok": True,
        "degenerate": degenerate,
        "warning": warning,
        "constraint": constraint_warning,
        "unbound": unbound_block,
        "dof": dof_block,
        # False when the delivered LM stopped at its iteration cap rather than at a minimum
        # (FittedOrbit.converged): elements are a mid-iteration state and the sigmas are not
        # formal uncertainties. Drives the convergence banner (convergence.js).
        "converged": bool(d.get("converged", True)),
        "object": d["object"],
        "combined": d["combined"],
        "designations": fit.linked_designations(),
        "mp_names": _mp_names(fit.linked_designations()),
        "other_designations": d.get("other_designations", []),
        "force_model": d["force_model"],
        "iod_method": getattr(fit, "iod_method", "?"),
        "summary": {
            "rms_residual": _safe(d["rms_residual"]),
            "rms_normalized": _safe(d["rms_normalized"]),
            "n_obs": d["n_obs"], "n_obs_total": d["n_obs_total"],
            "n_excluded": d["n_excluded"], "n_oppositions": d["n_oppositions"],
            "arc_days": _safe(d["arc_days"]), "obs_codes": d["obs_codes"],
            "systematic": systematic,
        },
        "epoch": {"jd": _safe(d["epoch"]), "iso": d["epoch_iso"]},
        # Non-gravitational solution: the model name + fitted A1/A2/A3 (au/day^2) and
        # their 1-sigma, or model=None for a pure-gravity fit. Drives the nongrav panel.
        "nongrav": {
            "model": d.get("nongrav_model"),
            "params": {k: _safe(v) for k, v in (d.get("nongrav_params") or {}).items()},
            "sigmas": {k: _safe(v) for k, v in (d.get("nongrav_sigmas") or {}).items()},
            # advisory (non-blocking) note when the fitted coefficients may be non-physical
            # (e.g. A1 implies a sub-rock density); drives the amber warning banner, never a veto.
            "warning": d.get("nongrav_warning"),
            # AREA/MASS RATIO (m^2/kg) implied by A1 on the 1-parameter SRP model, and None on
            # every other model -- find_orb's "Epoch ... ; AMR 0.034 m^2/kg" line, same rule and
            # same arithmetic (nongrav.amr_m2_per_kg). Computed server-side so the UI carries no
            # copy of SRP1AU. The ± arrives later, with the profile sigmas (nongrav_sigmas).
            "amr": _safe(_ngm.amr_m2_per_kg(d.get("nongrav_model"), d.get("nongrav_params"))),
        },
        # Held-fixed orbital elements {element: target}, or {} for an unconstrained fit
        # (find_orb limited_orbit). Drives the constraints panel.
        "constraints": {k: _safe(v) for k, v in (d.get("constraints") or {}).items()},
        # True when the 16 massive-asteroid perturbers are active (precision polish).
        "asteroids": bool(d.get("asteroids")),
        # Radar (delay/Doppler) rows folded into the fit, or None. The orbit is already
        # radar-refined (fit_orbit); this surfaces it for the UI. Mirrors the CLI radar report.
        "radar": _radar_block(fit),
        "physical": physical,
        "elements": elements,
        # True when `constraint` refused this orbit: the element rows carry [lo, hi] ranges
        # instead of values, and the orbit exports refuse. See the blanking above.
        "orbit_suppressed": orbit_suppressed,
        # Planetocentric (natural-satellite) orbit about a giant planet, or None. Present when the
        # object was auto-detected (or --center'd) as a satellite; drives the UI badge + the
        # heliocentric<->planetocentric element toggle.
        "planetocentric": planetocentric,
        "residuals": residuals,
        "observatories": observatories,
        "history": [{
            "stage": h.get("stage"), "n_obs": h.get("n_obs"),
            "rms": _safe(h.get("rms")), "score": _safe(h.get("score")),
            "kept": h.get("kept"),
        } for h in (fit.history or [])],
        "orbit3d": orbit3d,
        "stage_residuals": _stage_snapshots(fit),
        # The orbits the arc ADMITS, drawn faintly by both 3D views whenever the banner is up
        # (orbitfan.js) -- in place of nothing, and behind the asked-for orbit when there is one.
        "orbit_family": _orbit_family_block(fit) if constraint_warning else None,
        "nominal_optin": optin or None,
        # digest2 for every single tracklet, determined or not, and every suppressed banner
        # (the note used to appear only under the banner, so a tracklet whose
        # distance the profile DID pin showed none). Both flavours; the page toggles raw/no-ID.
        "digest2": _digest2_block(fit, constraint_warning),
        # Whether the observations table offers the tracklet-by-tracklet digest2 column
        # (q < 1.8); the scores themselves are fetched on demand (d2_tracklets).
        "d2_tracklets": {"eligible": _d2_tracklets_eligible(fit, constraint_warning)},
    }


D2_TRACKLET_Q_MAX = 1.8        # the per-tracklet column is for q < 1.8
D2_TRACKLETS_MAX = 2000        # tracklets scored per request (a few ms each)


def _digest2_block(fit, constraint_warning=None):
    """d2score.scores for the arc as fitted when it is a single tracklet (arc within
    constraint.SR_SINGLE_TRACKLET_MAX, one opposition) or the orbit is suppressed; else None."""
    if constraint_warning and constraint_warning.get("digest2"):
        return constraint_warning["digest2"]
    try:
        arc = float(fit.arc_length_days() or 0.0)
        single = arc <= constraint.SR_SINGLE_TRACKLET_MAX and (fit.n_oppositions() or 0) <= 1
    except Exception:
        single = False
    if not (single or constraint_warning):
        return None
    try:
        from pyroorb import d2score
        return d2score.scores(fit.included_observations())
    except Exception:
        return None


def _d2_tracklets_eligible(fit, constraint_warning=None):
    """q < D2_TRACKLET_Q_MAX on the nominal orbit -- or, when the orbit is suppressed and its
    nominal means little, anywhere in the 5-95% span of the orbits that fit."""
    try:
        q = _safe(fit.elements().get("q"))
    except Exception:
        q = None
    if q is not None and q < D2_TRACKLET_Q_MAX:
        return True
    rng = ((constraint_warning or {}).get("ranges") or {}).get("q")
    return bool(rng and rng[0] is not None and rng[0] < D2_TRACKLET_Q_MAX)


def d2_tracklets(session=None, fit_key=None):
    """digest2 scores tracklet by tracklet for the displayed fit's observations (all of them,
    included or not -- it is the astrometry as reported that digest2 judges), keyed by the
    residual-table index of each tracklet's first observation. Refused when the orbit is not a
    q < 1.8 one (the column is offered only there)."""
    fit = _fit_for(session, fit_key)
    if fit is None:
        return {"ok": False, "error": "the current fit has expired -- re-run it", "expired": True}
    try:
        cw = constraint.assess(fit)
    except Exception:
        cw = None
    if not _d2_tracklets_eligible(fit, cw):
        return {"ok": False, "error": f"tracklet digest2 scores are offered for q < "
                                      f"{D2_TRACKLET_Q_MAX:g} au only"}
    from pyroorb import d2score
    groups = d2score.tracklet_scores(list(fit._obs))
    truncated = len(groups) > D2_TRACKLETS_MAX
    keys = ("n", "station", "scores", "noid_scores", "shown", "shown_noid", "rms")
    return {"ok": True, "truncated": truncated,
            "tracklets": [{"first": int(g["first"]), **{k: g[k] for k in keys}}
                          for g in groups[:D2_TRACKLETS_MAX]]}


def _orbit_family_block(fit, n_pts=180):
    """constraint.orbit_family as drawable paths: ``{"source", "orbits": [{path, closed,
    weight}]}`` or None. Sampled coarser than the nominal (180 points): faint context lines at
    full Orbarium zoom do not need the 720 the measured orbit gets."""
    try:
        fam = constraint.orbit_family(fit)
    except Exception:
        return None
    if not fam:
        return None
    from pyroorb.orbital_elements import element_vector, ELEM_NAMES
    out = []
    for m in fam["members"]:
        try:
            v = element_vector(np.asarray(m["state"], float), float(fam["epoch_tdb"]))
            el = dict(zip(ELEM_NAMES, (float(x) for x in v)))
            el["epoch"] = float(fam["epoch_tdb"])
            o = _orbit_path(el, n=n_pts)
        except Exception:
            continue
        if o.get("path"):
            out.append({"path": o["path"], "closed": o["closed"], "weight": _safe(m["weight"])})
    return {"source": fam["source"], "orbits": out} if out else None


def nominal_optin(sid, on=True):
    """"Show me an orbit anyway" on an arc that does not determine one.

    On a single tracklet the orbit shown is the MOST PROBABLE one (popprior.most_probable_fit:
    the range profile's own solution at the population-posterior median distance); elsewhere,
    or when no posterior can be formed, the delivered fit. It becomes the session's displayed
    orbit, flagged on the FittedOrbit itself, so the exports that resolve this session publish
    it (all exports, once asked) and the next re-solve -- a new FittedOrbit -- quietly
    returns the page to the ranges. ``on=False`` is the page's "back to the ranges": the orbit
    displayed before the request is restored, unflagged."""
    s = _SESSIONS.get(sid)
    if not s or s.get("fit") is None:
        return {"ok": False, "error": "session expired -- re-fit the object", "expired": True}
    if not on:
        if getattr(s["fit"], "nominal_optin", None) and s.get("fit_before_optin") is not None:
            s["fit"] = s.pop("fit_before_optin")
        return _session_payload(s, sid, s["fit"])
    fit = s["fit"]
    if getattr(fit, "nominal_optin", None):
        return _session_payload(s, sid, fit)              # already showing it: idempotent
    try:
        if not constraint.assess(fit):
            return {"ok": False, "error": "this orbit is already determined -- nothing to ask for"}
    except Exception as ex:
        return {"ok": False, "error": f"could not assess the orbit: {type(ex).__name__}: {ex}"}
    # The delivered orbit is shown through a SHALLOW COPY: the flag must not land on the fit
    # "back to the ranges" restores (they share the ctx, mask and arrays, which is the point).
    import copy as _copy
    chosen, info = _copy.copy(fit), {"source": "fit"}
    try:
        from pyroorb import popprior
        mp = popprior.most_probable_fit(fit)
    except Exception:
        mp = None
    if mp is not None:
        chosen = mp
        info = {"source": "population", **{k: (_safe(v) if isinstance(v, float) else v)
                                            for k, v in mp.most_probable.items()}}
    chosen.nominal_optin = info
    s["fit_before_optin"] = fit
    s["fit"] = chosen
    return _session_payload(s, sid, chosen)


def _session_payload(s, sid, fit):
    """The interactive payload for `fit` on session `sid` (as `_apply_interactive` builds it)."""
    payload = serialise_fit(fit)
    payload["session"] = sid
    payload["meta"] = {"interactive": True}
    payload["nongrav_request"] = s.get("model") or "none"
    payload["constraints_request"] = _clean_constraints(s.get("constraints"))
    pend = _linkage_block(s)
    if pend is not None:
        payload["itf"] = pend
    return payload


# --------------------------------------------------------------------------
# Interactive sessions: hold the evaluator so the UI can toggle obs and re-fit.
# (Toggling is client-side -- the orbit is fixed; only Full step needs a refit.)
# --------------------------------------------------------------------------
import threading  # noqa: E402

_SESSIONS = {}
_SES_ORDER = []
_SES_LOCK = threading.Lock()
_SES_COUNTER = 0
# Keep the last N interactive fits live; evict oldest.  Twelve was right when
# this table served one local user, and wrong the moment a crowd arrived: the
# thirteenth visitor to interact silently evicted the first, who then found
# every control dead with no way back.  Two things changed that make a bigger
# number safe AND an eviction survivable:
#   * the expensive part of a fit is now cached on the fitqueue job, so an
#     evicted session is re-claimed from its `restore` token in milliseconds
#     (session_from_job) instead of costing a whole re-fit;
#   * sessions are only ever created for people who actually interact, because
#     a coalesced fit hands out a restore token rather than a session handle.
# Dialable for a small container, where the bound that matters is memory.
_SES_MAX = int(os.environ.get("PYROORB_SESSION_MAX") or 64)


def _store_session(fit, grav_pure=None):
    """Stash a fit under a fresh session handle and return the handle.

    The handle is UNGUESSABLE by construction.  It used to be the bare counter
    ("s1", "s2", ...), which is fine for the single-user local case this table
    was written for but not for a hosted deploy: the endpoints take the handle
    as their whole authorisation, so a sequential id lets anyone read or mutate
    somebody else's open fit (and, where MPC submission is enabled, submit it)
    just by counting.  The counter stays as the human-readable prefix -- it is
    what makes a handle legible in a log -- with 144 bits of entropy after it.
    """
    global _SES_COUNTER
    with _SES_LOCK:
        _SES_COUNTER += 1
        sid = f"s{_SES_COUNTER}-{secrets.token_urlsafe(18)}"
        _SESSIONS[sid] = {"ctx": fit._ctx, "history": fit.history,
                          "state": np.asarray(fit._res.x, dtype=float),
                          # the ACTIVE force-model baseline (gravity or +16ast) -- a nongrav
                          # refine / constrained fit always re-seeds from it (so switching never
                          # compounds). "model" is the active nongrav model (None = gravity) and
                          # "constraints" the active element constraints {element: target}
                          # (find_orb limited_orbit). These THREE selectors -- force model,
                          # nongrav model, constraints -- are ORTHOGONAL: each endpoint mutates
                          # only its own and _apply_interactive re-applies the others, so they
                          # compose and never clobber one another.
                          "grav_fit": fit, "model": None, "constraints": {},
                          # PURE-gravity baseline for the 16-asteroid A/B toggle: `grav_fit`
                          # may itself be the asteroid-polished orbit, so keep the gravity one
                          # to switch back to. "asteroids" is the current toggle state.
                          "grav_pure": grav_pure if grav_pure is not None else fit,
                          "asteroids": bool(fit.asteroids),
                          # a good asteroid STATE to seed the toggle from -- the asteroid fit on
                          # an obs set that needs the perturbers has a spurious near-gravity local
                          # min, so turning them ON must start from here, not the gravity orbit.
                          "ast_state": (np.asarray(fit._state, dtype=float)
                                        if fit.asteroids else None),
                          # the orbit currently DISPLAYED (gravity or the active nongrav fit)
                          # and its osculating epoch -- re-anchored by set_epoch so a Full
                          # step (and the elements view) run from the user-chosen epoch.
                          "fit": fit, "epoch_tdb": float(fit.epoch_tdb)}
        _SES_ORDER.append(sid)
        while len(_SES_ORDER) > _SES_MAX:
            _SESSIONS.pop(_SES_ORDER.pop(0), None)
        return sid


# Non-gravitational models the UI offers (value -> human label). 'none' is the
# gravity baseline; the rest map to fit_object/nongrav model keys. Labels and order
# mirror find_orb's force-model menu (efindorb.txt text 2060), minus the two entries
# we don't implement: the comet (A1,A2,A3,DT) time-lag and Delta-V.
NONGRAV_MODELS = [
    {"value": "none", "label": "No non-gravs (gravity only)"},
    {"value": "srp", "label": "Solar radiation pressure (radial, 1/r^2)"},
    {"value": "srp2", "label": "1/r² model, A1+A2 (SRP + Yarkovsky-type)"},
    {"value": "srp3", "label": "1/r² model, A1+A2+A3 (SRP + Yarkovsky-type + normal)"},
    {"value": "comet1", "label": "Comet model (A1, radial only)"},
    {"value": "comet2", "label": "Comet model (A1 and A2)"},
    {"value": "comet3", "label": "Comet model (A1, A2, A3)"},
    {"value": "yarkovsky", "label": "Yarkovsky (A2-only)"},
]

# Orbital elements the constraints panel can hold fixed (find_orb limited_orbit), with
# the unit/help shown beside each input. e=1 is a parabolic orbit, e=0 a circular one.
CONSTRAINABLE_ELEMENTS = [
    {"value": "a", "label": "a — semi-major axis", "unit": "au"},
    {"value": "e", "label": "e — eccentricity", "unit": "(0=circular, 1=parabolic)"},
    {"value": "q", "label": "q — perihelion distance", "unit": "au"},
    {"value": "Q", "label": "Q — aphelion distance", "unit": "au"},
    {"value": "i", "label": "i — inclination", "unit": "deg"},
]


def _clean_constraints(constraints):
    """Validate a client constraints dict {element: target} -> floats on known elements."""
    allowed = {c["value"] for c in CONSTRAINABLE_ELEMENTS}
    out = {}
    for k, v in (constraints or {}).items():
        if k in allowed and v is not None and v != "":
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _underdetermined(n_obs, npar=6):
    """``None`` when ``n_obs`` included observations can determine an ``npar``-parameter fit,
    else the refusal payload.

    The floor is stated in the currency the mathematics uses. An optical observation supplies
    TWO residual rows (RA, Dec), so an npar-parameter state is determined at ``2*n >= npar``:
    for the 6-element gravity state that is **3** observations, not 6. Three is also the
    engine's own pipeline-wide floor (``arc_extend.FIT_MIN_OBS``, used by every CLI entry
    point) and find_orb's -- ``full_improvement`` refuses below 3 included obs
    (orb_func.cpp), and its dof is likewise ``n_included*2 - n_params``. The interactive
    endpoints used to hard-code 6, which refused re-fits the CLI and the oracle both perform
    on the user's OWN inclusion mask.

    What lives between 3 and 6 obs is a statistics problem, not a determinacy one: dof is 0 at
    n=3, so the residuals vanish identically and the covariance carries no information about
    the fit's quality. That is REPORTED (``serialise_fit``'s ``dof`` block -> the zero-dof
    banner), not refused -- and the failure that actually matters, a range-degenerate short
    arc, is a property of the arc's GEOMETRY that an observation count cannot see either way
    (six obs inside one tracklet are as degenerate as three; the constraint banner is what
    catches it)."""
    need = max(arc_extend.FIT_MIN_OBS, -(-int(npar) // 2))     # ceil(npar/2), never below 3
    if int(n_obs) >= need:
        return None
    return {"ok": False,
            "error": (f"need at least {need} included observations to fit "
                      f"({npar} parameters need {npar} residual rows; each observation "
                      f"supplies 2)")}


def _client_mask(base, mask_list):
    """The client's inclusion set as a bool array aligned with ``base._obs`` -- or
    ``(None, error)`` when it is not.

    A mask is a POSITIONAL claim about a specific observation list, and the client
    builds it from the residual table of whatever payload it last rendered. If that
    view is stale (a second tab, an add-to-fit in between), the array that arrives
    describes a DIFFERENT arc. Nothing downstream notices: ``np.flatnonzero`` on a
    short mask silently yields only the positions it covers, so the engine solves a
    truncated arc and _build_orbit then dies with an opaque IndexError deep in the
    products; a long one dies indexing the evaluator. Say what is actually wrong
    instead, and refuse before any fitting happens. (The ITF search path has checked
    this since it was written -- see itf_search_web.)"""
    if not mask_list:
        return np.asarray(base._mask, dtype=bool), None
    mask = np.array([bool(b) for b in mask_list], dtype=bool)
    if len(mask) != len(base._obs):
        return None, {"ok": False, "stale": True,
                      "error": (f"the observation set changed ({len(mask)} on screen, "
                                f"{len(base._obs)} in the fit) -- re-fit to refresh the view")}
    return mask, None


def _apply_interactive(s, sid, mask_list, single=False):
    """Re-solve the session honouring ALL THREE orthogonal interactive selectors AT ONCE:
    the force model (``s["asteroids"]`` -- already baked into ``s["grav_fit"]``), the active
    non-gravitational model (``s["model"]``) and the active element constraints
    (``s["constraints"]``). The client's inclusion mask is honoured EXACTLY (no auto-rejection).

    These selectors are INDEPENDENT: each endpoint (run_nongrav / run_constrained /
    set_asteroids / full_step) mutates only its own and then calls this, which re-applies the
    other two -- so toggling one never clobbers the others' state. Compose order mirrors
    fit_object._finalize: the force-model base, then the nongrav A's and/or the constraint
    penalty rows (refine_nongrav now accepts `constraints`, so the two compose). ``none`` model
    with no constraints is a plain gravity fit on the base.

    ``single=True`` is the UI's "Full step" (find_orb's `f`): ONE damped differential-
    correction step instead of the solve-to-convergence "Multi-step". Only the plain-gravity
    branch distinguishes them (Evaluator.step vs the polish solve); the nongrav/constrained
    branches re-solve from the baseline either way, so a single step has no separate meaning
    there and both buttons perform the same (already interactive-tol) re-fit."""
    base = s["grav_fit"]                           # reflects the force-model (asteroid) choice
    ctx = base._ctx
    model = (s.get("model") or "none")
    model = model.strip() if (model and model != "none") else "none"
    constraints = _clean_constraints(s.get("constraints"))
    mask, err = _client_mask(base, mask_list)
    if err is not None:
        return err
    try:
        if model != "none":
            import pyroorb.nongrav as _ng
            # 2n vs npar, not n vs npar: the old form compared an OBSERVATION count to a
            # PARAMETER count, so a 7-parameter comet fit demanded 7 obs (14 rows) where 4
            # suffice. See _underdetermined.
            need = 6 + len(_ng.free_indices(model))            # state + this model's free A's
            err_n = _underdetermined(mask.sum(), need)
            if err_n is not None:
                return err_n
            # DETERMINISM: re-seed from THIS model's own last-converged orbit (cached on the session)
            # rather than the profile-seed grid, so re-selecting a model always REPRODUCES its good fit
            # -- find_orb's behaviour, and immune to whatever the intervening play did to the gravity
            # baseline or shared fit state (the reported "come back to srp and it's worse / loose ±").
            # The seed still gets fully polished on the CURRENT obs mask, so it adapts to obs edits; it
            # just can't fall into a different local minimum. The FIRST fit of a model (no cache yet)
            # uses the profile-seed as before. Guarded on same model + arity + fit epoch.
            #
            # `need` above is the (6+k) state+A's this model's LM vector carries -- reused here so a
            # cached fit from a DIFFERENT model that happens to share this session's cache slot (it
            # can't, in practice, since ng_cache is keyed by model, but the shape check is the actual
            # guard against a stale/foreign state) can't be mistaken for a matching seed.
            seed_state = None
            prev = s.get("ng_cache", {}).get(model)
            if prev is not None:
                try:
                    ps = np.asarray(prev._state, dtype=float)
                    if (getattr(prev, "nongrav_model", None) == model and ps.shape[0] == need
                            and abs(float(prev.epoch_tdb) - float(base.epoch_tdb)) < 1e-6):
                        seed_state = ps
                except Exception:
                    seed_state = None
            # Use the TIGHT delivery tol (refine_nongrav's default); radial_seed_all seeds the joint
            # radial models (srp2/srp3/comet) off the gravity corner when there is NO cache to seed from.
            fit2 = fo.refine_nongrav(base, model, mask=mask, revert_if_worse=False,
                                     asteroids=fo._nongrav_asteroids_gate(base),
                                     constraints=(constraints or None),
                                     radial_seed_all=True, seed_state=seed_state,
                                     profile_sigmas=False)   # linear sigma now; ± button computes the profile
            s.setdefault("ng_cache", {})[model] = fit2       # cache the CANONICAL fit (pre display re-anchor)
        elif constraints:
            err_n = _underdetermined(mask.sum())
            if err_n is not None:
                return err_n
            fit2 = fo.refit_constrained(base, constraints, mask=mask)
        else:
            err_n = _underdetermined(mask.sum())
            if err_n is not None:
                return err_n
            if single:
                # ONE damped differential-correction step (the "Full step" button):
                # honest per-click stepping, find_orb `f` semantics. A converged seed
                # comes back unchanged -- that no-op IS the convergence indicator.
                state, result = ctx["ev"].step(s["state"], mask)
            else:
                # polish=True: a DILIGENT re-seed solve (the "Multi-step" button). The
                # client seeds from the previous converged orbit and then changes the
                # obs mask, so a plain loose LM call would freeze at that (old) minimum
                # (nfev=2, orbit unchanged while only the covariance/H update); the
                # polish loop drives it to the NEW minimum. See Evaluator._polish_solve.
                state, result = ctx["ev"].fit(s["state"], mask, thin=False, polish=True)
            s["state"] = np.asarray(state, dtype=float)
            fit2 = fo._build_orbit(ctx, state, mask, result, s["history"])
    except Exception as ex:
        return {"ok": False, "error": f"fit step failed: {type(ex).__name__}: {ex}"}
    s["fit"] = fit2                               # remember the displayed orbit (for set_epoch)
    if s.get("itf_pending"):
        # The user has RE-SOLVED the pending find (a Full step, a Multi-step, a
        # model or force change -- everything interactive comes through here).
        # A profile-only possible may be offered for submission after that, and
        # only after that: the search's own payload never carries the button,
        # and a hand-crafted POST before any step is refused (_pending_verdict).
        s["itf_pending"]["stepped"] = True
    payload = serialise_fit(fit2)
    payload["session"] = sid
    payload["meta"] = {"interactive": True}
    payload["nongrav_request"] = model            # active model      (drives the UI dropdown)
    payload["constraints_request"] = constraints  # active constraints (drives the panel)
    pend = _linkage_block(s)
    if pend is not None:
        payload["itf"] = pend                     # button persists; follows the mask
    return payload


def base_obs_for(session=None, fit_key=None):
    """Resolve the parsed-obs records (incl. rejected) of the CURRENTLY-displayed fit,
    so a paste can be appended to it. A fetched object carries a live ``session``; a
    pasted object carries only a ``fit_key`` (no interactive session). Returns
    ``(obs_list, error)`` -- exactly one is non-None."""
    if session:
        s = _SESSIONS.get(session)
        if s and s.get("fit") is not None:
            return list(s["fit"]._obs), None
    if fit_key:
        fit = fitcache.get(fit_key)
        if fit is not None:
            return list(fit._obs), None
    return None, "the current fit has expired -- re-run it before adding observations"


def _fit_for(session=None, fit_key=None):
    """The FittedOrbit behind the currently-displayed object, by the same resolution
    :func:`base_obs_for` uses (live session first, then the paste cache), or None. Where
    base_obs_for wants the observations, this wants what the FIT knows about them -- which
    designations the user actually requested, above all."""
    if session:
        s = _SESSIONS.get(session)
        if s and s.get("fit") is not None:
            return s["fit"]
    if fit_key:
        return fitcache.get(fit_key)
    return None


def session_fit(sid):
    """The live FittedOrbit held by interactive session ``sid`` (reflecting the user's
    current included/excluded mask, non-grav model and epoch), or None if it expired."""
    s = _SESSIONS.get(sid)
    return s.get("fit") if s else None


def site_has_horizon(site):
    """True if MPC code ``site`` is a real ground station (so the suppress-unobservables
    rule can bite); False for the geocentre, roving and space-based codes."""
    from pyroorb import ephemerides as ephemeris
    try:
        return ephemeris._earth_location(site, _obscodes()) is not None
    except Exception:
        return False


def ephemeris_rows(fit, jd_start, step_days, n, site="500", faint_limit=None,
                   suppress_unobservable=False):
    """Structured ephemeris for the View-Ephemeris pane: ``n`` rows from ``jd_start``
    (UTC JD) at ``step_days``, seen from MPC observatory ``site``. Wraps
    ``ephemeris.compute_ephemeris`` and returns the JSON-ready columns (sky position,
    geometry, photometry, motion, and the J*Cov*J^T sky-plane uncertainty ellipse).

    ``faint_limit`` (V mag) and ``suppress_unobservable`` are find_orb's two display
    cuts: epochs the site can't reach are dropped from ``rows`` and tallied in
    ``hidden`` -- the step grid is unchanged, so the table simply comes back shorter.
    Suppression needs the topocentric alt/az columns, so it turns on the (slower)
    astropy observer pass; without it the cheap ``full=False`` path is kept."""
    from pyroorb import ephemerides as ephemeris
    n = max(1, min(int(n), 2000))
    step = float(step_days)
    jd_stop = jd_start + step * (n - 1)
    raw = ephemeris.compute_ephemeris(fit, _obscodes(), jd_start, jd_stop, step,
                                      site=site, full=bool(suppress_unobservable))
    raw, hidden = ephemeris.filter_rows(raw, faint_limit=faint_limit,
                                        suppress_unobservable=suppress_unobservable)
    keep = ("jd_utc", "date_iso", "ra_deg", "dec_deg", "ra_hms", "dec_dms",
            "delta_au", "r_au", "elong_deg", "phase_deg", "v_mag",
            "motion_arcsec_min", "motion_pa_deg",
            "unc_smaj_arcsec", "unc_smin_arcsec", "unc_pa_deg",
            "alt_deg", "az_deg", "sun_alt_deg",
            # per-break marks: what the cuts removed immediately before this row (and,
            # on the last row, after it) so the table can show the gap it left
            "hidden_before", "hidden_after")
    rows = [{k: x.get(k) for k in keep} for x in raw]
    # Render each break's reason clause server-side: the wording lives in ephemerides.py
    # only, so the pane and the text table can't drift apart on it.
    for r in rows:
        for k in ("hidden_before", "hidden_after"):
            if r.get(k):
                r[k] = {"n": sum(r[k].values()), "why": ephemeris.hidden_reasons(r[k])}
    return {"ok": True, "designation": fit.designation, "site": site,
            "force_model": fit.force_model, "rows": rows,
            "hidden": hidden, "hidden_note": ephemeris.hidden_note(hidden),
            "site_has_horizon": site_has_horizon(site),
            "faint_limit": (None if faint_limit is None else float(faint_limit)),
            "suppress_unobservable": bool(suppress_unobservable)}


def impact_assessment(fit, window_days=30.0):
    """Phase-1 impact LOCATION for the Impact pane: propagate the fitted orbit forward
    over ``window_days`` and report where/when it reaches the ground, if it does.
    Wraps ``impact.find_impact`` (validated to ~km on 2008 TC3); returns the JSON-ready
    crossings (100 km atmospheric entry + ground point, with entry speed/slope) plus the
    close-approach summary. For a non-impacting object it carries the explicit miss
    distance -- never a fabricated crossing."""
    from pyroorb import impact
    result = impact.find_impact(fit, window_days=float(window_days))
    result["ok"] = True
    result["designation"] = fit.designation
    result["force_model"] = fit.force_model
    result["window_days"] = float(window_days)
    result["epoch_iso"] = fit.epoch_iso()[:19]
    # Phase-1b: the semilinear 1-sigma ground footprint (only meaningful when the nominal
    # orbit actually impacts; a flyby carries no certain-impact ellipse). Best-effort -- a
    # degenerate covariance simply omits it rather than failing the location report.
    if result.get("impact"):
        try:
            fp = impact.impact_footprint(fit, window_days=float(window_days))
            if fp.get("ok"):
                result["footprint"] = fp
        except Exception:
            pass
    return result


def full_step(sid, mask_list, single=False):
    """Differential correction on the client's current included set. ``single=True`` is the
    "Full step" button -- ONE damped correction step, find_orb's `f` (click repeatedly; an
    unchanged orbit means converged). ``single=False`` is the "Multi-step" button -- re-solve
    to convergence. No auto-rejection: the mask is honoured exactly, and the session's CURRENT
    force model + nongrav model + constraints are ALL kept (a step on a
    comet-fit-with-constraints-and-asteroids stays exactly that)."""
    s = _SESSIONS.get(sid)
    if not s:
        return {"ok": False, "error": "session expired -- re-fit the object", "expired": True}
    return _apply_interactive(s, sid, mask_list, single=bool(single))


def reauto_solve(session=None, fit_key=None):
    """Re-run the FULL automatic pipeline on the currently-displayed fit's observation set --
    start orbit, extension/outlier rejection, asteroid polish, auto-nongrav discovery --
    exactly as if the whole set had just been pasted in (the UI's "Re-auto-solve" button,
    offered once observations have been added to a fit). Deliberately DISCARDS the user's
    manual include/exclude edits in favour of the engine's own auto-rejection, and returns a
    FRESH interactive session payload (same shape as run_fit). Resolves the obs like
    ``base_obs_for``: a live ``session`` for a fetched object, ``fit_key`` for a pasted one."""
    obs, err = base_obs_for(session=session, fit_key=fit_key)
    if err:
        return {"ok": False, "error": err, "expired": True}
    present = sorted({o.get("designation") for o in obs if o.get("designation")})
    meta = {
        "designation": present[0] if present else "pasted object",
        "designations": present or ["pasted object"],
        "combined": len(present) > 1,
        "other_designations": present[1:],
    }
    payload = fit_obs_list(obs, meta, _obscodes(),
                           meta_extra={"source": "reauto",
                                       "n_input_designations": len(present)})
    # Keep the export handle working after a re-auto-solve of a pasted/added union (an
    # export by designation would re-fetch from cache and drop the pasted obs) -- same
    # reasoning as loader.add_to_fit.
    if payload.get("ok", True):
        s2 = _SESSIONS.get(payload.get("session") or "")
        if s2 is not None and s2.get("fit") is not None:
            payload["fit_key"] = fitcache.put(s2["fit"])
    return payload


# Live ITF-search trace feeds, keyed by the client-chosen trace_id: the search
# thread appends elapsed-stamped lines while GET /api/itf_trace polls them out
# (the web UI's "see trace" live view). Bounded two ways: at most _TRACES_MAX
# feeds (finished feeds evicted first, so a flood of fresh trace_ids cannot
# kill someone's LIVE feed unless the table is all-live) and at most
# _TRACE_MAX_LINES lines per feed (a runaway search cannot grow server memory
# without bound).
_TRACES = {}
_TRACES_LOCK = threading.Lock()      # guards the dict itself (register/evict)
_TRACES_MAX = 16
_TRACE_MAX_LINES = 20000


def _trace_register(trace_id):
    # "tracer" is filled in by the caller once it has built one: holding it here
    # is what lets the feed poll re-level a search that is already running.
    ent = {"lines": [], "done": False, "tracer": None,
           "lock": threading.Lock()}
    with _TRACES_LOCK:
        while len(_TRACES) >= _TRACES_MAX:
            done_ids = [k for k, e in _TRACES.items() if e["done"]]
            _TRACES.pop(done_ids[0] if done_ids else next(iter(_TRACES)), None)
        _TRACES[trace_id] = ent
    return ent


def _trace_append(ent, line):
    """Append a line to a feed, respecting the per-feed cap (one truncation
    marker, then silence).

    The line is rewritten on the way in (:mod:`itf_wording`): the engine's
    fractional opposition counts and status codes are comparison shorthands,
    not English, and this feed is read by a person."""
    line = itf_wording.humanize(line)
    with ent["lock"]:
        if len(ent["lines"]) < _TRACE_MAX_LINES:
            ent["lines"].append(line)
        elif len(ent["lines"]) == _TRACE_MAX_LINES:
            ent["lines"].append("[trace truncated -- line cap reached]")


def itf_trace_feed(trace_id, start=0, level=None):
    """Poll a live ITF-search trace: lines from ``start`` on, the next cursor,
    and whether the search has finished (GET /api/itf_trace).

    ``level`` re-levels the RUNNING search's tracer from this moment on (the
    viewer's "Detail" control). Verbosity is a threshold the trace reads per
    line, so moving it mid-search is safe -- it changes what is written next,
    never what was already written. A bad value is clamped, never fatal: this
    is a spectating knob and must not be able to disturb a search."""
    with _TRACES_LOCK:
        ent = _TRACES.get(str(trace_id or ""))
    if ent is None:
        # expired tells the client to STOP polling (vs a transient error)
        return {"ok": False, "expired": True, "error": "unknown trace id"}
    try:
        start = max(0, int(start or 0))
    except (TypeError, ValueError):
        start = 0
    tracer = ent.get("tracer")
    if level is not None and tracer is not None:
        from pyroorb.itf.trace import L_HEAD, L_FIREHOSE
        try:
            tracer.level = max(L_HEAD, min(int(level), L_FIREHOSE))
        except (TypeError, ValueError):
            pass
    with ent["lock"]:
        lines = list(ent["lines"][start:])
        done = ent["done"]
    return {"ok": True, "lines": lines, "next": start + len(lines),
            "done": done,
            "level": getattr(tracer, "level", None)}


# --- background ITF searches -------------------------------------------------
# An ITF search runs for tens of seconds to many minutes, and a hosted deploy
# sits behind a reverse proxy that will not hold a single HTTP request open that
# long: on Railway the edge answers the still-running POST with the plain-text
# body "upstream error", which the client's response.json() then died on
# ("Unexpected token 'u'") -- the search itself was fine and often finished.
# So the request no longer waits: POST /api/itf_search starts the search on a
# worker thread and returns a job handle immediately, and the client polls
# GET /api/itf_result (alongside the trace feed it already polls). Every request
# is short, so no proxy timeout can reach the search.
def _rss_mb():
    """This process's resident size in MB, or None off Linux. Cheap enough to call
    on every search start (one small read of /proc/self/status)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


_ITF_JOBS = {}
_ITF_JOBS_LOCK = threading.Lock()
_ITF_JOBS_MAX = 8
# how long a FINISHED job stays collectable. Generous: the user may have the tab
# backgrounded, and the payload is what their search cost minutes to produce.
_ITF_JOB_TTL = 1800.0


# The only thing that has ever bounded concurrent ITF searches: nothing.
# `_ITF_JOBS_MAX` bounds RETAINED FINISHED PAYLOADS -- every eviction branch
# skips a running job, deliberately, since evicting one would not stop its
# thread -- so nine users got nine live sweeps, each allocating on top of the
# resident sessions, and the box died.  This cap is the web UI's alone: the CLI
# is one search per process and rations itself.
#
# DERIVED FROM THIS CONTAINER, in `capacity`, alongside the fit pool -- the two
# caps spend the same memory and the same cores, so neither can be settled
# without the other.  What a search costs is measured (24 live searches):
# the median grows 189 MB and the p90 563 MB over a ~180 MB baseline, and the
# budget those divide into is what the container has left after the resident
# process, a FULL fit pool's transient, and the sweep's own safety floor.
#
# This replaces a hand-set THREE (2026-08-29) whose reasoning was sound and
# whose premise was not: it argued that six concurrent searches at ~3.4 GB was
# "more than the container has", which was true of the ~1 GB box the number was
# reasoned against and false of the 7629 MB one the service actually runs on.
# A constant that must be re-derived by hand whenever the deploy moves is a
# constant that is wrong after the next move, which is why this is computed.
#
# The sweep floor still keeps an over-subscribed box survivable -- past it the
# sweeps truncate instead of the box dying, and truncation is reported to every
# user it happens to (`itf.incomplete`) -- but truncation is a WORSE answer than
# waiting, so the derived cap is one the box can hold, not the largest it can
# start.  An admitted search is separately checked against live headroom by
# `search_capacity`: a free slot is still not free memory.
#
# PYROORB_ITF_MAX_RUNNING still overrides, for a deploy that must differ from
# the arithmetic; it is read once at import, so a change needs a restart.
_ITF_MAX_RUNNING = capacity.itf_max_running()
from pyroorb.itf.memory import CAPACITY_UNUSABLE as _CAPACITY_UNUSABLE  # noqa: E402


def _itf_reap():
    """Answer for any job whose WORKER IS GONE but which never marked itself
    done.  Caller holds the lock.

    A running entry is never evicted -- deliberately, since losing a live
    search's result is the failure this whole table exists to prevent -- and
    nothing else clears one either, so a job stuck at ``done: False`` holds its
    admission slot (`_ITF_MAX_RUNNING`) for the life of the process.  Six of
    those and every later search is refused "server busy" until someone
    restarts the server, which is a worse outage than the concurrency the cap
    was added to prevent.

    ``_run`` marks its own job done on every path it can reach, so the entry
    can only be stranded when the worker never got there: a thread that failed
    to start (handled at the call site) or one killed outside Python's control.
    The exact test for that is the thread object itself -- ``is_alive()``, not
    a wall-clock guess, because a legitimate variant-scan search can run for
    many minutes and a timeout long enough to be safe for it is too long to be
    much use here.  A job with no thread recorded (the blocking path, which
    releases its own slot in a ``finally``, or a hand-built test entry) is
    left alone.

    Stranded jobs are marked DONE with an error rather than deleted, so a
    client polling /api/itf_result gets an answer instead of polling a job
    that will never finish.
    """
    now = time.time()
    for j in _ITF_JOBS.values():
        th = j.get("thread")
        if j["done"] or th is None or th.is_alive():
            continue
        j["payload"] = {"ok": False, "error": (
            "the search worker stopped without returning a result -- the "
            "server may have been out of memory. Please try again.")}
        j["ended"] = now
        j["done"] = True
        j["thread"] = None


def _itf_running():
    """How many searches are in flight right now.  Caller holds the lock.

    Reaps first: a slot held by a worker that is gone is not in use, and this
    is the number admission is decided on (see :func:`_itf_reap`).
    """
    _itf_reap()
    return sum(1 for j in _ITF_JOBS.values() if not j["done"])


def _itf_admit(job_id, now):
    """Decide whether a new ITF search may start, and if so register its job
    entry.  Returns ``(None, refusal_dict)`` when the box is full or too small,
    else ``(job_entry, None)`` with the entry already inserted in ``_ITF_JOBS``.

    THE ADMISSION GATE FOR EVERY SEARCH PATH, not just the async one.  Both the
    backgrounded search (:func:`itf_search_start`) and the blocking legacy
    fallback (:func:`itf_search_sync`, reached when a stale client omits
    ``async``) come through here, so no POST can start a live sweep unseen by
    the concurrency cap (see `_ITF_MAX_RUNNING`).

    Caller holds ``_ITF_JOBS_LOCK``.  Idempotency (a re-POST of a still-running
    job id) is the caller's to check BEFORE calling this, because only the async
    path has a meaningful "same search already running" answer.
    """
    running = _itf_running()
    if running >= _ITF_MAX_RUNNING:
        return None, {"ok": False, "busy": True, "running": running,
                      "error": (f"{running} ITF searches are already running "
                                f"(the limit is {_ITF_MAX_RUNNING}). Please try "
                                f"again in a minute -- your object is not the "
                                f"problem, the server is just full.")}
    # A FREE SLOT IS NOT FREE MEMORY.  The sweep floor is global headroom, so
    # six admitted searches can still leave the seventh -- or the second --
    # with nothing to sweep with.  Starting a search that provably cannot run
    # is worse than refusing it: it burns minutes and answers "no match" from a
    # corridor it never queried, which reads to the user as evidence of absence.
    cap = _itf_capacity_block()
    if cap and cap["verdict"] == _CAPACITY_UNUSABLE:
        return None, {"ok": False, "busy": True, "running": running,
                      "capacity": cap,
                      "error": (f"The server is low on memory right now, so an "
                                f"ITF search would be cut short before it found "
                                f"anything ({cap['reason']}). Please try again "
                                f"in a minute.")}
    entry = {"done": False, "started": now, "ended": 0.0,
             "collected": 0.0, "payload": None,
             # The worker, once it exists -- the async path fills this in right
             # after starting it.  Held so :func:`_itf_reap` can tell "still
             # searching" from "the worker is gone and this slot is stuck"
             # exactly, rather than by a timeout.  None on the blocking path,
             # which runs on the request thread and frees its own slot.
             "thread": None,
             # Set to ask the worker to stop (POST /api/itf_cancel).  Only the
             # async path arms it; a job without one cannot be cancelled.
             "cancel": None,
             "cancel_requested": 0.0,
             # RSS when this search began, so /memory can show how far the
             # process has moved since. A hint, not an attribution -- anything
             # else running moves it too.
             "rss_start_mb": _rss_mb()}
    _ITF_JOBS[job_id] = entry
    return entry, None


@contextlib.contextmanager
def itf_background_slot(label):
    """A search slot for BACKGROUND work (the NEOCP monitor's ITF follow-up),
    yielding True when granted and False when not.

    Registered in the same `_ITF_JOBS` table the user searches are admitted
    against, so the cap counts it -- but granted only while at least ONE slot
    would still be free for a person, and never on an `unusable` box: a
    background search waits (the monitor retries next pass), a user does not.
    """
    job_id = f"bg:{label}:{time.time():.3f}"
    with _ITF_JOBS_LOCK:
        running = _itf_running()
        cap = _itf_capacity_block()
        ok = (running < _ITF_MAX_RUNNING - 1
              and not (cap and cap["verdict"] == _CAPACITY_UNUSABLE))
        if ok:
            entry, _refused = _itf_admit(job_id, time.time())
            ok = entry is not None
            if ok:
                entry["thread"] = threading.current_thread()
    try:
        yield ok
    finally:
        if ok:
            with _ITF_JOBS_LOCK:
                _ITF_JOBS.pop(job_id, None)


def _itf_jobs_evict(now):
    """Drop expired jobs, then (if still over the cap) the most expendable
    FINISHED one: a payload the client already COLLECTED goes before one still
    waiting to be picked up, because dropping the latter loses a search the user
    never saw. A running search is never evicted -- losing its result is the one
    failure this whole mechanism exists to prevent. Caller holds
    _ITF_JOBS_LOCK."""
    # ...and a job whose worker is GONE is not a running search, however it
    # looks in the table: reap first so it can age out like any other finished
    # one instead of sitting here forever (see _itf_reap).
    _itf_reap()
    for k, j in list(_ITF_JOBS.items()):
        if j["done"] and now - j["ended"] > _ITF_JOB_TTL:
            _ITF_JOBS.pop(k, None)
    while len(_ITF_JOBS) >= _ITF_JOBS_MAX:
        # (collected-first, then oldest): sorts every collected job ahead of
        # every uncollected one, and orders within each group by finish time
        done = sorted((not j["collected"], j["ended"], k)
                      for k, j in _ITF_JOBS.items() if j["done"])
        if not done:
            break                        # all running: let the table grow a little
        _ITF_JOBS.pop(done[0][2], None)


def itf_search_start(session, options=None, mask_list=None, fit_key=None,
                     _searcher=None):
    """Start :func:`itf_search_web` on a worker thread and return its handle
    ``{"ok": True, "job": <id>}`` at once (POST /api/itf_search with
    ``async``). Poll :func:`itf_search_status` for the payload.

    The job id is the client's trace_id when it sent one, so the result feed and
    the trace feed share a handle; a search launched without one gets a
    generated id."""
    options = options or {}
    job_id = str(options.get("trace_id") or "")[:64] or f"job{uuid.uuid4().hex[:12]}"
    now = time.time()
    with _ITF_JOBS_LOCK:
        _itf_jobs_evict(now)
        prev = _ITF_JOBS.get(job_id)
        if prev is not None and not prev["done"]:
            return {"ok": True, "job": job_id}     # idempotent: same search
        # Admission, checked INSIDE the lock and immediately before the insert
        # so two simultaneous POSTs cannot both see five running and both start.
        entry, refused = _itf_admit(job_id, now)
        if refused is not None:
            return refused
        cancel = entry["cancel"] = threading.Event()

    def _run():
        try:
            # Armed on THIS thread, so the token binds this search and nothing
            # else in the process (deadline.py: a ContextVar, not a global).
            with _deadline.cancellable(cancel, label="ITF search"):
                payload = itf_search_web(session, options, mask_list=mask_list,
                                         fit_key=fit_key, _searcher=_searcher)
        except _deadline.Cancelled:
            payload = {"ok": False, "cancelled": True,
                       "error": "the search was stopped at your request"}
        except Exception as ex:          # a crashed search must still ANSWER
            payload = {"ok": False,
                       "error": f"{type(ex).__name__}: {ex}"}
        with _ITF_JOBS_LOCK:
            j = _ITF_JOBS.get(job_id)
            if j is not None:
                j["payload"] = payload
                j["ended"] = time.time()
                j["done"] = True
                j["thread"] = None       # nothing left to watch; see _itf_reap
        # A search is the biggest transient this process makes, and glibc keeps a
        # per-thread arena's pages once a few small survivors are scattered through
        # it -- four concurrent searches left 4760 MB "free" but resident against
        # 99.6 MB actually in use, and the ITF gate then refuses every later search
        # for the life of the process. Trimming HERE, on the search's own dying
        # thread, is the moment its arena is emptiest. ~1 ms, and gated so an
        # ordinary run pays nothing; see webmem.trim_allocator.
        try:
            from pyroorb.web import webmem
            webmem.trim_allocator()
        except Exception:
            pass

    th = threading.Thread(target=_run, name=f"itf-search-{job_id}", daemon=True)
    # The start is GUARDED: the entry is already in the table and counting
    # against the concurrency cap, so a thread that cannot be created (an
    # out-of-memory box is exactly when that happens, and exactly when this
    # route matters) must not leave a slot held by a search that never began.
    # Answering here also means the client's first poll gets the error rather
    # than waiting out a job with no worker.
    try:
        th.start()
    except Exception as ex:              # RuntimeError: can't start new thread
        with _ITF_JOBS_LOCK:
            entry["payload"] = {"ok": False,
                                "error": f"could not start the search worker: "
                                         f"{type(ex).__name__}: {ex}"}
            entry["ended"] = time.time()
            entry["done"] = True
        return {"ok": False, "error": f"could not start the search worker: "
                                      f"{type(ex).__name__}: {ex}"}
    # ...and the thread is recorded only AFTER it is running.  A Thread that has
    # not been started yet reports ``is_alive() == False``, exactly as a dead one
    # does, so recording it any earlier lets a concurrent _itf_reap strand a
    # search that is about to run perfectly well.  Until this line the entry has
    # no thread and simply counts as running, which is the safe reading.
    with _ITF_JOBS_LOCK:
        if not entry["done"]:            # unless it already finished, that fast
            entry["thread"] = th
    return {"ok": True, "job": job_id}


def itf_search_sync(session, options=None, mask_list=None, fit_key=None,
                    _searcher=None):
    """The BLOCKING search path -- run the search on the request thread and
    return its payload directly, for a client that POSTs without ``async``.

    This exists only for older cached clients (the current UI always sends
    ``async: true`` and polls).  It goes through the SAME admission gate as the
    async path rather than calling :func:`itf_search_web` straight through,
    which used to make it invisible to the concurrency cap and the
    unusable-box refusal.  The job entry it registers counts toward
    `_itf_running()` for the duration and is marked done (and thus
    TTL-evictable) when the search returns, whether it succeeded, refused, or
    raised.
    """
    options = options or {}
    now = time.time()
    # A fresh internal key, NEVER the client's trace_id: a blocking caller reads
    # its result from the return value, not by polling this id, so the entry is
    # only a concurrency-slot marker -- and reusing trace_id here would clobber
    # the _ITF_JOBS entry of an async search that happened to share it.  (The
    # tracer still wires itself up from options["trace_id"], untouched.)
    job_id = f"sync{uuid.uuid4().hex[:12]}"
    with _ITF_JOBS_LOCK:
        _itf_jobs_evict(now)
        _entry, refused = _itf_admit(job_id, now)
        if refused is not None:
            return refused
    try:
        payload = itf_search_web(session, options, mask_list=mask_list,
                                 fit_key=fit_key, _searcher=_searcher)
    except Exception as ex:              # a crashed search must still ANSWER
        payload = {"ok": False, "error": f"{type(ex).__name__}: {ex}"}
    finally:
        with _ITF_JOBS_LOCK:
            j = _ITF_JOBS.get(job_id)
            if j is not None:
                # collected already: a blocking caller reads the payload from
                # the return value, never from /api/itf_result, so this entry's
                # only remaining job is to hold a concurrency slot until now
                j["payload"] = None
                j["ended"] = time.time()
                j["collected"] = j["ended"]
                j["done"] = True
    return payload


def itf_search_cancel(job_id):
    """Ask a running background ITF search to stop (POST /api/itf_cancel).

    The search is a thread, and Python cannot kill a thread, so this SETS the
    job's cancel token and returns at once; the worker stops at its next
    agreement point (every propagator step, every database batch, every
    between-fit budget check -- well under a second in practice) and answers
    the client's next /api/itf_result poll with ``cancelled: True``.  Its
    admission slot is released only then: until the worker has unwound it is
    still holding the memory the slot accounts for.

    The job id is the handle the client chose when it started the search, so
    only a page that started a search can stop it.  Idempotent: stopping a
    finished, already-stopping or unknown search is harmless and says so.
    """
    with _ITF_JOBS_LOCK:
        _itf_reap()
        j = _ITF_JOBS.get(str(job_id or ""))
        if j is None:
            return {"ok": False, "expired": True,
                    "error": "unknown or expired ITF search job"}
        if j["done"]:
            return {"ok": True, "done": True, "cancelling": False}
        ev = j.get("cancel")
        if ev is None:
            return {"ok": False, "error": "this search cannot be stopped"}
        if not ev.is_set():
            ev.set()
            j["cancel_requested"] = time.time()
        return {"ok": True, "done": False, "cancelling": True}


def itf_search_status(job_id):
    """Poll a background ITF search (GET /api/itf_result?id=). Still running ->
    ``{"ok": True, "done": False, "elapsed": secs}``; finished -> the full
    search payload under ``result``. An unknown id answers ``expired`` so the
    client stops polling instead of spinning forever."""
    with _ITF_JOBS_LOCK:
        # a worker that died without answering must end this client's polling
        # with an error, not leave it asking forever (see _itf_reap)
        _itf_reap()
        j = _ITF_JOBS.get(str(job_id or ""))
        if j is None:
            return {"ok": False, "expired": True,
                    "error": "unknown or expired ITF search job"}
        if not j["done"]:
            return {"ok": True, "done": False,
                    "elapsed": round(time.time() - j["started"], 1),
                    "cancelling": bool(j.get("cancel_requested"))}
        payload = j["payload"]
        # one collection is enough -- but keep it (a duplicate poll in flight, a
        # reload racing the last poll) rather than dropping it here. Marking it
        # collected only makes it the FIRST thing eviction takes when the table
        # fills, ahead of a payload nobody has picked up yet.
        j["collected"] = j["collected"] or time.time()
    return {"ok": True, "done": True,
            "elapsed": round(j["ended"] - j["started"], 1),
            "result": payload}


def _itf_pending_block(s):
    """The ITF block for an interactive payload, recomputed from the session's
    CURRENT observation mask -- or None when this session has no pending
    linkage.

    Two things follow from building it live rather than freezing it at search
    time.  The "Submit linkage" button stops vanishing the moment you take a
    Full step to look at the linkage (the pending linkage is not stale just
    because the orbit was re-solved -- it is the same observations).  And the
    submission FOLLOWS THE SCREEN: switch off every observation of a tracklet
    and that tracklet leaves the document, because what gets submitted is
    whatever is still included, not what the search happened to return.
    """
    pend = s.get("itf_pending")
    fit = s.get("fit")
    if not pend or fit is None:
        return None
    live = _live_lines(s)
    itf_ids = set(pend.get("itf_ids") or ())
    # the linked tracklets still standing, grouped as the search reports them
    tracklets = {}
    n_itf = 0
    for ln in live:
        if obs_unique_id(ln) not in itf_ids:
            continue
        n_itf += 1
        sub = ln[5:12].strip()
        t = tracklets.setdefault(sub, {"trksub": sub, "date": ln[15:25].strip(),
                                       "stn": ln[77:80], "n_obs": 0})
        t["n_obs"] += 1
    doc, _err = _itf_live_linkage(s, live)
    submittable, reason, possible = _pending_verdict(s, pend, fit, live, doc)
    return {
        "found": True,
        "pending": True,                # carried across re-fits, not a fresh search
        "possible": possible,           # the search's 0.6, still unconfirmed on this set
        "outcome": itf_wording.outcome_label(SearchVerdict.LINKED if submittable
                                             else SearchVerdict.POSSIBLE),
        "reason": reason,
        "tracklets": sorted(tracklets.values(), key=lambda t: t["date"]),
        "added_obs": n_itf,
        "linkage": doc,
        "submittable": submittable,
        "direct_submit_allowed": _direct_submit_allowed(),
        "warnings": itf_wording.humanize_all(pend.get("warnings")),
        # already humanized when the search stored them; never a submit gate
        "diagnostics": list(pend.get("diagnostics") or ()),
        "vetting": itf_wording.humanize_vetting(pend.get("vetting")),
    }


def _included_ids(fit):
    """The identities (:func:`obs_unique_id`) of a fit's INCLUDED observations that carry an
    OBS80 line -- the set a submittability verdict was passed on. Empty when there is no fit."""
    out = set()
    if fit is None:
        return out
    try:
        for o, keep in zip(fit._obs, fit.is_included()):
            line = o.get("obs80_line")
            if keep and line:
                out.add(obs_unique_id(line))
    except Exception:
        return set()
    return out


def _seed_gate_inputs(seed_obs80, seed_fit):
    """The seed-side inputs of :func:`itf.search.opp_profile_acceptable`, as the search itself reads
    them off its seed (``ctx.pub_ids``, ``ctx.pub.arclength_days``, ``ctx.initial_opp_profile``), stashed
    with the pending linkage so a later re-judge runs the search's gate and not a copy."""
    from pyroorb.itf.obs80_fields import obs_ids
    from pyroorb.itf.opp_profile import opp_profile
    out = {"pub_ids": sorted(obs_ids(seed_obs80 or "")),
           "initial_opp_profile": opp_profile(seed_obs80 or ""),
           "pub_arclength_days": 0.0}
    try:
        if seed_fit is not None:
            from pyroorb.itf.fitter import trial_from_fit
            out["pub_arclength_days"] = float(trial_from_fit(seed_fit).arclength_days)
    except Exception:
        pass
    return out


def _pending_verdict(s, pend, fit, live, doc):
    """``(submittable, reason, possible)`` for a pending ITF find rebuilt on the live set.

    The live rebuild may NARROW a linkage the search stood behind (switch a tracklet off and the
    document follows) but may not WIDEN the search's refusal on the same evidence: a find the
    search filed as a 0.6 "possible" (``result.linkage`` None -- a lone new night, a fragile 1.75
    that raised a caution, a partial pickup with no submittable remainder) stays unsubmittable
    through every Full step and Multi-step that re-solves the SAME observations. It becomes
    submittable only when the included set has GROWN since the search -- observations pasted in,
    or ones the search had off switched on -- AND that grown set passes THE SAME gate the search
    applied (:func:`itf.search.opp_profile_acceptable`: the opposition profile with the seed-side
    inputs stashed at search time, the multi-opposition seed waiver and the NEO waiver). One gate,
    not a web-side approximation of it. A set that only shrank, or changed without growing,
    keeps the search's refusal and its reason.

    ONE exception: a possible the search refused on OPPOSITION PROFILE ALONE -- the
    recursion-accumulated 1.75, two new oppositions of one night each, fitting cleanly with no
    doubt raised (``vetting["promotable_on_step"]``, set only there) -- is offered for submission
    once the user has RE-SOLVED it (``pend["stepped"]``, set by :func:`_apply_interactive`), at
    their own judgement. Not before: the search's own payload carries no button, and a POST that
    skipped the step is refused here. Every other possible carries a doubt about the fit or the
    astrometry and keeps the growth rule above.
    """
    # ...and a comet says so in its own words: the set IS a linkage, just not one
    # this route can carry, so the generic "not a linkage" below would misstate it.
    from pyroorb.itf.linkage import comet_refusal
    comet_why = comet_refusal("\n".join(live) + "\n", pend.get("primaries") or ())
    if comet_why:
        return False, comet_why, True
    if doc is None:
        return False, "not a linkage on the current observation set", False
    # The MPC's night requirement (tracklets < 12 h apart are one night) on the LIVE set, ahead of
    # every path below that can say yes: switching a tracklet off can break it on a find the search
    # passed, and neither the step-promotion nor the grown-set gate asks it.
    from pyroorb.itf.mpcrules import submittable_nights
    nights_ok, nights_why = submittable_nights("\n".join(live) + "\n")
    if not nights_ok:
        return False, nights_why, True
    if pend.get("submittable", True):
        return True, "pending linkage (follows the current observation set)", False
    refusal = pend.get("reason") or "the search did not confirm this linkage"
    if (pend.get("vetting") or {}).get("promotable_on_step"):
        if pend.get("stepped"):
            return (True,
                    "the search filed this as a possible on its opposition profile alone (two new oppositions "
                    "of one night each); re-solved after a step, it is offered for submission "
                    "at your judgement",
                    False)
        return (False,
                "the search filed this as a possible on its opposition profile alone (two new oppositions of "
                "one night each); a Full step or Multi-step re-solving it will offer it for "
                "submission",
                True)
    judged = set(pend.get("judged_ids") or ())
    included = _included_ids(fit)
    grown = included - judged
    # GROWN means the judged set is still wholly there AND something was added -- not merely
    # that the live set holds some id the search never saw. A SWAP (switch a judged
    # observation off, switch an unjudged one on) satisfies the second alone, and would carry
    # the find to opp_profile_acceptable on evidence the search's candidate ranking and recursion
    # never evaluated -- possibly without the very tracklet the linkage needed. Nothing
    # downstream recovers that: opp_profile_acceptable is deliberately the ctx-free OPPOSITION-PROFILE clause,
    # and the per-tracklet quality gates (_accept_quality_ok) are not reachable from here at
    # all, which is what makes this precondition load-bearing rather than decorative. A fit
    # whose own rejection round dropped a judged observation is a changed set too, and takes
    # the refusal for the same reason: the safe direction.
    if not judged <= included:
        return (False,
                f"the search filed this as a possible ({refusal}); "
                f"{len(judged - included)} observation(s) it judged are no longer in the fit, "
                f"and a set it has not judged is not one it can be overruled on",
                True)
    if not grown:
        return (False,
                f"the search filed this as a possible ({refusal}), and re-solving the same "
                f"observations does not change that",
                True)
    from pyroorb.itf.config import ItfConfig
    from pyroorb.itf.fitter import trial_from_fit
    from pyroorb.itf.search import opp_profile_acceptable
    from pyroorb.itf.opp_profile import OppProfile
    gate = pend.get("seed_gate") or {}
    try:
        trial = trial_from_fit(fit)
        ok, why = opp_profile_acceptable(
            ItfConfig(), "\n".join(live) + "\n", trial,
            pub_arclength_days=float(gate.get("pub_arclength_days") or 0.0),
            pub_ids=set(gate.get("pub_ids") or ()),
            initial_opp_profile=gate.get("initial_opp_profile") or OppProfile(1, 0))
    except Exception as ex:                       # an unjudgeable set is not a submittable one
        ok, why = False, f"could not be judged ({type(ex).__name__}: {ex})"
    if ok:
        return (True,
                f"pending linkage: {len(grown)} observation(s) added or switched on since the "
                f"search, and the grown set passes the search's own submittability gate",
                False)
    return (False,
            f"{len(grown)} observation(s) added or switched on since the search, but the set "
            f"still fails the search's submittability gate: {why}",
            True)


def _ident_linkage_block(s):
    """The linkage block for a fit the USER assembled -- two or more designations typed into the
    search box (or pasted), joined by the engine into one orbit -- or None when this session is not
    that.

    An ITF find is not the only linkage worth submitting.  Type ``J99T43V OOn0024`` and the engine
    fetches both (the second from the NEOCP), fits them as one object and shows you the joint
    orbit; up to now the one thing it would not do is let you TELL anyone, even though the MPC
    identification that fit implies is exactly the document :func:`itf.linkage.build_linkage`
    already builds and :func:`itf_submit_web` already knows how to transmit.

    Offered only when the engine's own identification rule (``ident_check.linkage_verdict`` -- the
    same verdict ``--ident-check`` prints, thresholds and all) says this joint fit is a real
    candidate: a clean identification, or an ambiguous-but-plausible one whose reason rides along
    as a caution.  A 'no' -- one side of the link only surviving by having most of its observations
    thrown away, say -- shows no button, because the orbit being pretty is not evidence that the
    two designations are one object.
    """
    fit = s.get("fit")
    if fit is None or s.get("itf_pending") or s.get("linkage_submitted"):
        return None                       # an ITF find owns the block (see _itf_pending_block);
                                          # an already-sent identification is not offered again
    try:
        desigs = {o.get("designation") for o in fit._obs}
    except Exception:
        return None
    if len(desigs) < 2:
        return None                       # one object: nothing to identify with anything
    if getattr(fit, "n_requested", None) == 1:
        # ONE designation typed in the search box. The others rode in on that single fetch
        # because the MPC has ALREADY acknowledged the identification -- that is why they came
        # back together. Nothing to offer and nothing to say: no block at all, so no "Submit
        # linkage" button and no banner suggesting one. (ident_check.linkage_verdict declines
        # this for the same reason, for the CLI; the early return here keeps the payload clean.)
        return None
    from pyroorb import ident_check
    try:
        # the same builder the submission uses, so what is shown and what is sent cannot differ
        doc, err = _itf_live_linkage(s)
        verdict = ident_check.linkage_verdict(fit)
    except Exception as ex:               # never let a report break the fit payload
        return {"found": False, "kind": "ident",
                "reason": f"linkage check failed: {type(ex).__name__}: {ex}"}
    if not verdict["offer"]:
        return {"found": False, "kind": "ident", "submittable": False,
                "reason": f"not offered as an identification: {verdict['reason']}"}
    from pyroorb.itf.linkage import comet_refusal
    comet_why = comet_refusal("\n".join(_live_lines(s)) + "\n")
    if comet_why:
        return {"found": False, "kind": "ident", "submittable": False,
                "reason": comet_why}
    if doc is None:
        return {"found": False, "kind": "ident", "submittable": False,
                "reason": f"not a linkage: {err}"}
    link = doc.get("links", {}).get("link_0", {})
    n_by_trksub = {}
    for ln in _live_lines(s):
        sub = ln[5:12].strip()
        n_by_trksub[sub] = n_by_trksub.get(sub, 0) + 1
    ids = [{"trksub": t[0], "date": t[1][:10], "stn": t[2],
            "n_obs": n_by_trksub.get(t[0], 0)}
           for t in link.get("trksubs") or []]
    return {
        "found": True,
        "kind": "ident",                  # ...as opposed to an ITF search's find
        "pending": True,
        "outcome": itf_wording.outcome_label(SearchVerdict.LINKED),
        # A 'y2' and an 'unknown' state their own case ("an identification WITH EXCEPTIONS: ...",
        # "NOT submittable -- the engine could not determine..."); only the ambiguous 'y-examine'
        # needs the candidate framing in front of its reason.
        "reason": ("clean identification" if verdict["clean"] else
                   verdict["reason"] if verdict.get("code") in ("y2", "unknown") else
                   f"identification candidate: {verdict['reason']}"),
        "clean": verdict["clean"],
        "designations": list(link.get("designations") or []),
        "tracklets": ids,
        "neocp": link.get("identification_type") == "neocp",
        "added_obs": 0,                   # nothing was ADDED -- the user brought both arcs
        "linkage": doc,
        # ...shown for review, but a 'y2' cannot be TRANSMITTED: the document has no way to carry
        # the exception the identification rests on, so it goes to the MPC as a Jira ticket and
        # the front end raises no button (ident_check.linkage_verdict['submittable']).
        "submittable": bool(verdict.get("submittable", True)),
        "direct_submit_allowed": _direct_submit_allowed(),
        "warnings": itf_wording.humanize_all(verdict["cautions"]),
        "vetting": {"clean": verdict["clean"], "reason": verdict["reason"],
                    "designations": len(link.get("designations") or []),
                    "trksubs": len(link.get("trksubs") or [])},
    }


def _linkage_block(s):
    """The session's submittable linkage, from whichever way it arose: an ITF search's pending find
    or a multi-designation fit the user assembled. One shape, so the front end has one button."""
    return _itf_pending_block(s) or _ident_linkage_block(s)


def _unsubmittable(block):
    """The refusal for a linkage the engine will not stand behind, or None when it will.

    OFFERED is not SUBMITTABLE. A 'y2' (the join rests on excepting whole tracklets, which the
    identification document cannot express), a 'y-systematic' (the kept observations trend) and an
    'unknown' (the exception shape could not be measured) are all shown for review with their
    reasons -- and the front end raises no button for them -- but the verdict that hides the button
    is the same verdict that must refuse the request, or the click-through is the only thing
    standing between a hand-crafted POST and a claim the engine declined to make.

    Both routes a document leaves by ask this, at the same point: AFTER the live rebuild. An ITF
    find's block reports ``submittable`` from that same rebuild, and it is the rebuild that knows
    WHY it is no longer a linkage (every linked tracklet switched off, fewer than two ids left) --
    asked earlier, this generic refusal would swallow that specific one.
    """
    if block.get("submittable", True):
        return None
    return {"ok": False,
            "error": f"this linkage is not submittable: "
                     f"{block.get('reason') or 'the engine declined to confirm it'}"}


def _included_obs80_lines(fit):
    """The OBS80 lines of a fit's INCLUDED observations, synthesised when the
    obs carried no verbatim line (an ADES fetch) exactly as the seed is. One
    implementation, in the module that builds the submission document."""
    from pyroorb.itf.linkage import included_obs80
    return included_obs80(fit)


def _leaveout_rejudge(s):
    """The leave-out-demoted cautions that must come BACK as click-throughs at Submit.

    The search demoted them on the observation set it judged (``judged_ids``).  When
    the included set is unchanged that judgement stands and this returns [].  When the
    user has switched observations on or off since, each demoted caution is re-judged by
    :func:`itf.leaveout.confirm_cautions` on what is being SUBMITTED: one that no longer
    confirms is returned (humanized, the text the click-through acks); one whose flagged
    tracklet is now wholly switched off is moot and dropped; and any failure to re-judge
    returns them all -- the safe direction.  Cached per included set, so the retry after
    the user acknowledges them does not refit again."""
    pend = s.get("itf_pending") or {}
    demoted = list(pend.get("leaveout_demoted") or ())
    fit = s.get("fit")
    if not demoted or fit is None:
        return []
    live = _included_ids(fit)
    if live == set(pend.get("judged_ids") or ()):
        return []
    key = frozenset(live)
    cached = s.get("_leaveout_rejudged")
    if cached and cached[0] == key:
        return list(cached[1])
    from pyroorb.itf import leaveout
    from pyroorb.itf.obsreg import AdesRegistry
    back = demoted
    try:
        itf_ids = set(pend.get("itf_ids") or ())
        lines = _live_lines(s)
        live_itf = [l for l in lines if obs_unique_id(l) in itf_ids]
        all_itf = "\n".join(l for l in pend.get("lines") or () if obs_unique_id(l) in itf_ids) + "\n"
        itf_text = "\n".join(live_itf) + "\n"
        combined = "\n".join(lines) + "\n"
        magt = pend.get("leaveout_magt")
        reg = AdesRegistry()
        reg.add_obs(fit._obs)
        # a caution whose tracklet is switched off has nothing left to vouch for;
        # the rest are judged in ONE call, so a shared tracklet is refitted once
        judge = [w for w in demoted
                 if leaveout.is_own_arc(w) or not leaveout.named_tracklets(w, all_itf, magt)
                 or leaveout.named_tracklets(w, itf_text, magt)]
        back, _d, _r = leaveout.confirm_cautions(judge, [], itf_text, combined,
                                                 obs_sources.load_obscodes(),
                                                 registry=reg, mag_tracklet=magt)
    except Exception:                                    # noqa: BLE001
        back = demoted
    out = itf_wording.humanize_all(back)
    s["_leaveout_rejudged"] = (key, out)
    return out


def _live_lines(s):
    """The pending linkage's OBS80 lines that are INCLUDED right now.

    Filters the astrometry the SEARCH produced by the current mask rather than
    re-rendering it from the fit: those lines are already correct (the search
    built them, and the session's fit was parsed from them), and re-rendering
    risks the byte differences that make an observation's identity ambiguous --
    the same trap that once let a search find its own observations.  Falls back
    to rendering when a session has no stored text.
    """
    pend = s.get("itf_pending") or {}
    fit = s.get("fit")
    stored = [l for l in (pend.get("lines") or []) if l.strip()]
    if fit is None:
        return stored
    known, included = set(), set()
    try:
        for o, keep in zip(fit._obs, fit.is_included()):
            line = o.get("obs80_line")
            if not line:
                continue
            oid = obs_unique_id(line)
            known.add(oid)
            if keep:
                included.add(oid)
    except Exception:
        return stored
    if not stored:
        return _included_obs80_lines(fit)
    # Drop only what the user actually switched OFF.  A line the fit does not
    # carry at all was never switched off by anyone -- it is simply not
    # represented (the payload falls back to the search's internal fit when the
    # re-fit from the combined astrometry fails), and dropping it would silently
    # empty the linkage instead of honouring a decision.
    return [l for l in stored
            if obs_unique_id(l) in included or obs_unique_id(l) not in known]


def _itf_live_linkage(s, live_lines=None):
    """Build the MPC identification document from what is INCLUDED right now.

    ``(doc, error)`` -- doc is None when the current set no longer makes a
    linkage (every linked tracklet switched off, say, leaving one id).  This is
    the same build_linkage the search runs, fed the live astrometry instead of
    the search's frozen output, so the rule "fewer than two distinct ids is not
    a linkage" applies to the screen as it did to the search.
    """
    from pyroorb.itf.config import ItfConfig
    from pyroorb.itf.linkage import build_linkage
    pend = s.get("itf_pending") or {}
    fit = s.get("fit")
    if fit is None:
        return None, "no fit on this session"
    lines = live_lines if live_lines is not None else _live_lines(s)
    if not lines:
        return None, "no included observations"
    itf_ids = set(pend.get("itf_ids") or ())
    banned = set(pend.get("banned_designations") or ())
    if banned:
        # the search's ban travels with the pending linkage: those lines stay
        # on screen and in the fit, and leave only the DOCUMENT (itf/pickup.py)
        from pyroorb.itf.pickup import drop_designations
        lines = [l for l in drop_designations("\n".join(lines) + "\n",
                                              banned).splitlines() if l.strip()]
        if not lines:
            return None, "no submittable observations left"
    combined = "\n".join(lines) + "\n"
    itf_out = "\n".join(l for l in lines if obs_unique_id(l) in itf_ids)
    itf_out += "\n" if itf_out else ""
    primaries = list(pend.get("primaries") or ())
    if not primaries:
        # A fit the user assembled from two or more typed designations names all of them
        # (one per proposed identification), never just the first -- and never the aliases
        # that rode in acknowledged. See itf.linkage.requested_primaries.
        from pyroorb.itf.linkage import requested_primaries
        try:
            present = {o.get("designation")
                       for o, keep in zip(fit._obs, fit.is_included()) if keep}
        except Exception:
            present = None
        primaries = requested_primaries(fit, present)
    if not primaries and getattr(fit, "n_requested", None) == 1:
        # A one-designation fetch is ONE MPC identification family, so its designations are
        # aliases the MPC has already acknowledged: the document must name the family PRIMARY
        # once and never restate the DES=DES, even when the news is a freshly linked tracklet.
        # (The ITF-search path stashes `primaries` for exactly this; a session that never ran a
        # search has none, so derive it the same way -- alphabetically-earliest, the engine's
        # family-primary convention.)
        primaries = sorted({o.get("designation") for o in fit._obs
                            if o.get("designation")})[:1]
    return build_linkage(combined, ItfConfig(),
                         fit=fit,
                         primary_desigs=primaries,
                         itf_obs_out=itf_out,
                         trksubs=pend.get("itf_trksubs"))


def _itf_trksubs(result):
    """``{obs_unique_id: trksub}`` of a search result's ITF lines, from the
    database column (itf.linkage.db_trksubs).  A linked result carries it; a
    possible does not, so its registry (which banked every fitted candidate row)
    answers for the same lines."""
    ids = {obs_unique_id(l) for l in (result.itf_obs80 or "").splitlines() if l.strip()}
    reg = getattr(result, "registry", None)
    out = reg.trksubs(ids) if reg is not None and hasattr(reg, "trksubs") else {}
    out.update(getattr(result, "itf_trksubs", None) or {})
    return out


def _direct_submit_allowed():
    """DIRECT_SUBMIT_ALLOWED is the SERVER-SIDE authorization for sending a linkage
    straight to the MPC from the web, skipping the identification queue (default:
    off).  It only adds that third choice to the submit menu AND is re-checked
    inside :func:`itf_submit_web`, so a hand-crafted POST cannot skip the queue
    either.  The two everyday routes -- submit to the queue, download the file --
    need no switch (SHOW_ITF_SUBMIT_BUTTON is gone).  An exported
    variable wins; the gitignored ``.env`` is a fallback
    (:func:`itf.dbio.flag_enabled`)."""
    from pyroorb.itf.dbio import flag_enabled
    return flag_enabled("DIRECT_SUBMIT_ALLOWED")


def _queue_url():
    """Where "Submit linkage" sends: the identification queue.  EVERY
    server posts there over HTTP -- pyroorb.net included, which simply posts to
    itself -- so there is one route, no database credentials outside the queue
    host, and one place every submission is deduplicated and paced."""
    return (os.environ.get("IDQUEUE_URL") or "https://www.pyroorb.net/identifications/")


def queue_pages():
    """The identification queue's public pages, from ``IDQUEUE_URL`` (one home for
    the address, never a literal in the page): documentation, paste/upload, dashboard."""
    base = _queue_url().rstrip("/")
    return {"ok": True, "policies": base + "/policies", "submit": base + "/submit",
            "dashboard": base + "/dashboard"}


def _post_to_queue(doc):
    """POST ``doc`` to the queue; ``(http_code, body)``."""
    import requests
    from pyroorb.itf.linkage import anonymous_header
    anon = anonymous_header(doc)
    if anon:                    # the last check before anything leaves this server
        return 0, {"ok": False, "error": f"not sent: {anon}"}
    url = _queue_url()
    # Never follow a redirect: requests re-sends a redirected POST as a GET with
    # no body, so a host that redirects (pyroorb.net -> https://www.pyroorb.net,
    # path dropped) answered with the home page and the submission was lost.
    r = requests.post(url, json=doc, timeout=30, allow_redirects=False)
    if 300 <= r.status_code < 400:
        return r.status_code, {"ok": False, "error": (
            f"the queue address {url} redirects to {r.headers.get('Location') or '?'} "
            f"-- nothing was submitted; set IDQUEUE_URL to the queue's own address")}
    try:
        body = r.json()
    except ValueError:
        body = {"ok": False, "error": (r.text or "")[:300]}
    return r.status_code, body


def itf_defaults():
    """The ITF-search options modal's defaults, read from :class:`ItfConfig`.

    The modal used to carry these numbers as literals in index.html, which is
    how the corridor multiplier went stale: ``uncert_mult`` moved 1.1 -> 2.5 in
    the engine and the input kept sending 1.1, so every web search ran a 2.3x
    too-narrow corridor and quietly missed tracklets the CLI found (2026 DR42 /
    P10OAL0).  Serving them from the one home of these numbers keeps the two in
    step by construction."""
    from pyroorb.itf.config import ItfConfig
    from pyroorb.itf.trace import DEFAULT_LEVEL
    cfg = ItfConfig()
    return {"ok": True,
            "max_uncert_arcsec": round(cfg.max_uncert_deg * 3600.0, 3),
            "uncert_mult": cfg.uncert_mult,
            "mag_limit": cfg.mag_limit,
            "mag_limit_deep": cfg.mag_limit_deep,
            "rank_mode": cfg.ranking_mode,
            "candidate_source": cfg.candidate_path,
            "trace_level": DEFAULT_LEVEL,
            # ...and how much room the server has RIGHT NOW, so the modal can
            # warn before the user commits to minutes of waiting rather than
            # explaining a thin result afterwards
            "capacity": itf_capacity_now(),
            "slots": itf_slots_now()}


def _itf_capacity_block():
    """The raw capacity verdict (all three states) or None. `itf_capacity_now`
    is the UI-facing filter over this; admission needs the UNFILTERED answer,
    since CAPACITY_OK must be distinguishable from "cannot measure"."""
    from pyroorb.itf.config import ItfConfig
    from pyroorb.itf.memory import search_capacity
    return search_capacity(ItfConfig().sweep_min_free_mb)


def itf_capacity_now():
    """The server's current ITF headroom, as the UI should present it, or None
    where memory cannot be measured (an uncontained dev box usually).

    Deliberately re-read per call: on a shared server the answer changes with
    whoever else is searching, and a value cached at import would be a
    confident lie by the time anyone read it."""
    from pyroorb.itf.memory import CAPACITY_OK
    cap = _itf_capacity_block()
    if not cap or cap["verdict"] == CAPACITY_OK:
        return None
    return {"verdict": cap["verdict"], "reason": cap["reason"],
            "headroom_mb": cap["headroom_mb"]}


def itf_slots_now():
    """``{"running": n, "limit": n}`` -- what the ITF-search button should say
    about queueing before it is pressed."""
    with _ITF_JOBS_LOCK:
        return {"running": _itf_running(), "limit": _ITF_MAX_RUNNING}


#: How many rivals the web search fits to price a non-exclusive one-night show
#: (0 would restore the engine default of not counting at all).  See
#: ItfConfig.show_n_rivals.
WEB_RIVAL_COUNT_K = 6


def itf_search_web(session, options=None, mask_list=None, fit_key=None,
                   _searcher=None):
    """Search the live ITF for tracklets of the CURRENTLY-displayed object (the
    web UI's "ITF Search" button; see src/pyroorb/itf/).

    Seeds from the session's obs (honouring the user's include/exclude mask when
    given), runs the native corridor search against the SBN database, and on a
    found linkage returns a FRESH full fit payload (same shape as ``run_fit``)
    for the combined seed+ITF orbit, plus an ``itf`` block describing what was
    linked. When nothing is found, returns ``{"ok": True, "itf": {"found":
    False, ...}}`` so the client can show a transient notice. The SEARCH never
    submits (``do_submit=False``); transmitting is a SEPARATE, explicitly-gated
    action -- :func:`itf_submit_web` on the web (to the queue, or the MPC behind
    DIRECT_SUBMIT_ALLOWED,
    re-vetted server-side), or ``--itf-submit`` on the CLI.

    ``options``: max_uncert_arcsec, uncert_mult, rank_mode (ml|certainty|legacy),
    candidate_source (index|scan), jitter, mag_limit (overrides ItfConfig.mag_limit; omit for
    the engine default), trace_id, trace_level. ``_searcher`` is a test seam.
    """
    options = options or {}
    obs, err = base_obs_for(session=session, fit_key=fit_key)
    if err:
        return {"ok": False, "error": err, "expired": True}
    if mask_list is not None and len(mask_list) == len(obs):
        obs = [o for o, keep in zip(obs, mask_list) if keep]

    # Seed lines: the verbatim OBS80 when the obs carried one, else a
    # SYNTHESIZED 80-col line (obs80_base_line -- the same rendering the
    # astrometry view uses). ADES/NEOCP-sourced fits have no obs80_line at
    # all, and without the synthesis every such fit died with "no seed
    # astrometry": ANY fit can launch an ITF search.
    # The rendering is a bookkeeping currency, NOT a data downgrade: the obs
    # dicts behind these lines go to the search as `seed_obs`, so their ADES
    # fields (reported rmsRA/rmsDec -> the weighting sigma, rmsMag, logSNR,
    # obsID, trkSub) survive into every fit the search runs and into the fit
    # handed back below.  See itf/obsreg.py.
    lines, seed_obs = [], []
    for o in sorted(obs, key=lambda o: o.get("jd_utc", 0.0)):
        try:
            parts = obs80.obs80_base_line(o).splitlines()
        except Exception:
            continue                     # one unrenderable obs never kills the seed
        lines.extend(parts)
        seed_obs.append({**o, "obs80_line": parts[0]})
    # ONE designation per identification. A session fetched under a single name is one
    # acknowledged family however many designations came back, so it seeds and submits
    # under the family PRIMARY only (the engine's alphabetically-earliest convention):
    # fetching every alias would pull the same family once per name (a duplicate-poisoned
    # seed), and the document must not restate an existing DESDES identification.
    # But a session the USER assembled -- "K13M06C = K21C05V" typed into the search box --
    # is a proposed identification per typed designation, and dropping all but the first
    # is how the ITF document came back naming one half of a linkage the user had already
    # made (itf.linkage.requested_primaries). Acknowledged aliases riding along with
    # either of them still stay out.
    aliases = sorted({o.get("designation") for o in obs
                      if o.get("designation")})
    from pyroorb.itf.linkage import requested_primaries
    seed_fit = _fit_for(session=session, fit_key=fit_key)
    primaries = requested_primaries(seed_fit, aliases) if seed_fit else []
    primaries = primaries or aliases[:1]
    seed_obs80 = "\n".join(lines) + ("\n" if lines else "")
    if len(lines) < 3 and not primaries:
        return {"ok": False, "error": "the current fit has too few OBS80 "
                                      "observations to seed an ITF search"}

    from pyroorb.itf.config import ItfConfig

    def _f(key, default):
        try:
            return float(options.get(key, default))
        except (TypeError, ValueError):
            return default
    rank_mode = options.get("rank_mode") or "ml"
    if rank_mode not in ("ml", "certainty", "legacy"):
        rank_mode = "ml"
    candidate_source = options.get("candidate_source") or "index"
    if candidate_source not in ("index", "scan"):
        candidate_source = "index"
    cfg = ItfConfig(ranking_mode=rank_mode, candidate_path=candidate_source,
                    jitter=bool(options.get("jitter")),
                    # THE WEB IS WHERE THIS IS WORTH PAYING FOR.  When the one-night show is about to surface
                    # a lone night whose exclusivity it never measured, count
                    # how many rival tracklets link on their own and stay quiet
                    # if too many do (ItfConfig.show_n_rivals carries the
                    # measurement).  It costs a median 12 s on the ~quarter of
                    # searches that reach that path and nothing on the rest --
                    # a price worth paying for a person who might go searching
                    # archival images on the strength of the banner, and NOT
                    # worth paying for `itf batch`, which is why the
                    # engine default is 0 and this is the only surface that
                    # turns it on.
                    show_n_rivals=WEB_RIVAL_COUNT_K)
    # None -> itf_search falls back to the ItfConfig defaults (the single home
    # of these numbers); the client only overrides what the user typed
    max_uncert_arcsec = _f("max_uncert_arcsec", None)
    max_uncert_deg = (None if max_uncert_arcsec is None
                      else max_uncert_arcsec / 3600.0)
    uncert_mult = _f("uncert_mult", None)
    # corridor faint cut.  Unlike the two above it lives ON the config rather
    # than in the itf_search signature, so an untyped field must leave the
    # engine default alone -- _f returns None and we simply don't assign.
    mag_limit = _f("mag_limit", None)
    if mag_limit is not None:
        cfg.set_mag_limit(mag_limit)

    # live trace feed: the client picks a trace_id up front so it can start
    # polling /api/itf_trace while this request is still running
    trace_ent = None
    tracer = None
    trace_id = str(options.get("trace_id") or "")[:64]
    if trace_id:
        from pyroorb.itf.trace import Tracer, DEFAULT_LEVEL, L_HEAD, L_FIREHOSE
        trace_ent = _trace_register(trace_id)
        # verbosity: the client may pick a level; default is the narrative one.
        # Clamped rather than validated -- a bad value should cost readability,
        # never the search.  There is no level 0 on this surface: on the two
        # CLIs 0 means "no trace", but the page always records one for its
        # viewer (no trace_id is how a caller opts out), so a 0 or a missing
        # level takes the default.
        try:
            level = int(options.get("trace_level") or DEFAULT_LEVEL)
        except (TypeError, ValueError):
            level = DEFAULT_LEVEL
        level = max(L_HEAD, min(level, L_FIREHOSE))
        tracer = Tracer(lambda line, _e=trace_ent: _trace_append(_e, line),
                        level)
        trace_ent["tracer"] = tracer     # so the feed poll can re-level it live

    log_lines = []

    def _log(msg):
        log_lines.append(msg)
        if tracer is not None:
            tracer.head(f"* {msg}")       # milestones join the live feed too

    try:
        if _searcher is not None:
            result = _searcher(seed_obs80, primaries, cfg)
        else:
            from pyroorb.itf.search import ItfDbUnreachable, itf_search_live
            seed_ok = len(lines) >= 3
            try:
                result = itf_search_live(
                    None if seed_ok else primaries,
                    seed_obs80=seed_obs80 if seed_ok else None,
                    seed_obs=seed_obs if seed_ok else None,
                    primary_desigs=primaries if seed_ok else None,
                    cfg=cfg, max_uncert_deg=max_uncert_deg,
                    uncert_mult=uncert_mult, do_submit=False,
                    verbose=False, log=_log, trace=tracer)
            except ItfDbUnreachable as ex:
                return {"ok": False,
                        "error": f"ITF database unreachable: {ex}"}
    except _deadline.Cancelled:
        # the user's Stop: say so at the end of the feed, then let the caller
        # that armed the token answer it
        if tracer is not None:
            tracer.head("* search stopped at the user's request")
        raise
    finally:
        if trace_ent is not None:
            with trace_ent["lock"]:
                trace_ent["done"] = True

    # status 0.6 with a fit attached = a POSSIBLE-but-unconfirmed thin-profile
    # linkage (1.5/1.75 opps): shown to the user with its warning, never
    # submittable (result.linkage is None, so nothing is stashed to submit)
    # Part of the corridor never queried -- shown on EVERY outcome, including a
    # find.  A user who is told "no match" from a sweep that stopped early has
    # been told something false in effect, and a user whose linkage came out of
    # a truncated sweep is entitled to know more may be there.  Deliberately a
    # field of its own and NOT one of `warnings`: those gate the submit
    # click-through, and an incomplete sweep is not a doubt about the linkage.
    inc = result.incomplete_sweep
    incomplete = None
    if inc:
        # The headline is the DOMINANT actual cause, not a default: a mixed
        # cancel+memory truncation where neither reaches the total must not read
        # as "a row budget was reached" when the budget dropped nothing (which
        # would point the user at raising a budget that did nothing).  A trailing
        # "(and other limits)" flags the mixed case.
        _causes = [("the server ran low on memory", inc.get("lowmem", 0)),
                   ("the database cancelled queries", inc.get("cancelled", 0)),
                   ("a row budget was reached", inc.get("budget", 0))]
        _causes.sort(key=lambda c: c[1], reverse=True)
        why = _causes[0][0]
        if sum(1 for _label, n in _causes if n) > 1:
            why += " (and other limits)"
        incomplete = {
            "windows": inc["windows"],
            "memory": bool(inc["lowmem"]),
            "why": why,
            "detail": itf_wording.humanize(inc["message"]),
            "verdict_affected": inc["verdict_affected"],
        }

    # Memory pressure DURING the search that did not (yet) truncate it: the
    # search completed, but on this box it was running close to the line. Told
    # separately from `incomplete`, because "we finished, but shallowly" and
    # "we stopped early" are different things to a user deciding whether to
    # re-run later.
    cap = (result.telemetry or {}).get("capacity")
    constrained = None
    if cap and not inc:
        constrained = {"verdict": cap["verdict"], "reason": cap["reason"]}

    possible = (result.verdict is SearchVerdict.POSSIBLE
                and result.fit is not None
                and bool(result.itf_obs80))
    if ((result.verdict is not SearchVerdict.LINKED and not possible)
            or result.fit is None):
        # "Nothing found" and "something linked and we distrust it" are
        # different findings and must not share a headline: the second sets
        # ``suppressed``, and result.reason then carries the clause that
        # refused it in plain English (search._refuse).
        suppressed = bool((result.vetting or {}).get("show_suppressed"))
        return {"ok": True,
                "itf": {"found": False,
                        "suppressed": suppressed,
                        "outcome": itf_wording.outcome_label(result.verdict),
                        "reason": itf_wording.humanize(result.reason),
                        "diagnostics": itf_wording.humanize_all(result.diagnostics),
                        "incomplete": incomplete,
                        "constrained": constrained,
                        "log": itf_wording.humanize_all(log_lines[-6:])}}

    def _found_payload(result):
        """The page's payload for one found result: the paste-style refit of
        seed + linked ITF obs on a new session, with the linkage stashed on it
        and the ``itf`` block the client shows."""
        possible = result.verdict is SearchVerdict.POSSIBLE
        # linked-tracklet summary for the client banner
        tracklets = {}
        for ln in (result.itf_obs80 or "").splitlines():
            if not ln.strip():
                continue
            sub = ln[5:12].strip()
            t = tracklets.setdefault(sub, {"trksub": sub,
                                           "date": ln[15:25].strip(),
                                           "stn": ln[77:80], "n_obs": 0})
            t["n_obs"] += 1

        # Finish the linkage EXACTLY like a paste of the combined astrometry would
        # (fit_obs_list: pure-gravity baseline -> asteroid precision polish ->
        # auto-nongrav discovery -> interactive session). Serialising the search's
        # INTERNAL trial fit instead (batch-tolerance, gravity-only) leaves
        # borderline residual patterns that flag as "systematic" in the O-C pane
        # until the first Full step re-converges them -- a bogus warning.
        #
        # The combined arc is re-parsed from LINES (that is the search's currency),
        # but the observations are then restored to ADES: each seed ob goes back to
        # the very dict this session was already holding -- with its obsID, trkSub,
        # reported rmsRA/rmsDec and the sigma weighting that follows from them --
        # and each linked ITF ob picks up whatever ADES the database had for it.
        # Without this the whole arc, the ORIGINAL object included, came back as
        # bare 80-column data: no obsID/trkSub/sigmas in the "addl info" pane, and a
        # fit weighted by the station model where a reported sigma existed.
        payload = None
        try:
            from pyroorb.itf.fitter import _meta_for
            from pyroorb.itf.obs80_fields import obs_unique_id
            obs2, _fmt2 = obs_sources.read_obs_file(result.combined_obs80,
                                                    _obscodes(), in_format="obs80")
            if obs2:
                by_id = {obs_unique_id(o["obs80_line"]): o for o in seed_obs}
                merged, n_ades = [], 0
                for o in obs2:
                    orig = by_id.get(obs_unique_id(o.get("obs80_line") or ""))
                    if orig is not None:
                        merged.append(dict(orig))     # the session's ADES ob, whole
                        n_ades += 1
                    else:
                        merged.append(o)              # a linked ITF ob
                if result.registry is not None:
                    n_ades += result.registry.enrich(merged)
                obs2 = merged
                obs2.sort(key=lambda o: o["jd_utc"])
                meta2, _prim2 = _meta_for(obs2, fmt="ADES" if n_ades else "OBS80")
                cand = fit_obs_list(obs2, meta2, _obscodes(),
                                    meta_extra={"format": meta2["format"]})
                if cand.get("ok", True) and cand.get("session"):
                    payload = cand
                    s2 = _SESSIONS.get(cand["session"])
                    if s2 is not None and s2.get("fit") is not None:
                        payload["fit_key"] = fitcache.put(s2["fit"])
        except Exception as ex:
            # fall through to the search's internal fit -- but say so: that fit is
            # batch-tolerance and can flag a bogus "systematic" in the O-C pane,
            # so a regression here must not hide as a mystery UI warning
            _log(f"paste-style refit of the linkage failed "
                 f"({type(ex).__name__}: {ex}); showing the search's own fit")
            payload = None
        if payload is None:                   # fall back to the search's own fit
            payload = serialise_fit(result.fit)
            payload["session"] = _store_session(result.fit)
            payload["fit_key"] = fitcache.put(result.fit)
        sid_new = payload["session"]
        # stash the found linkage on the new session so the "Submit Linkage" button
        # can transmit it server-side WITHOUT the client ever holding a submittable
        # document (the dedup + ledger path re-runs at submit time, unchanged)
        # humanized ONCE, here: the same strings gate the submit click-through
        # (the client acks them verbatim), so the stash and the payload must carry
        # the very same text
        # the incomplete-sweep note travels in its own field (above); excluding it
        # here keeps it off the submit click-through, which exists for doubts about
        # the LINKAGE and would otherwise make every user ack a capacity message
        _gate = [w for w in (result.warnings or [])
                 if not (inc and w == inc["message"])]
        warnings = itf_wording.humanize_all(_gate)
        diagnostics = itf_wording.humanize_all(result.diagnostics)
        s_new = _SESSIONS.get(sid_new)
        if s_new is not None:
            from pyroorb.itf.leaveout import original as leaveout_original
            s_new["itf_linkage"] = result.linkage
            s_new["itf_warnings"] = warnings          # gate submission click-through
            # ...and everything needed to REBUILD the document from the live
            # observation set later, so the button survives a re-fit and the
            # submission follows what is switched on (see _itf_pending_block)
            s_new["itf_pending"] = {
                "primaries": list(primaries),
                # The SEARCH's own verdict on this find, and the observation set it
                # judged. A 0.6 "possible" has result.linkage None: the search would
                # not stand behind a document, and the live rebuild below may not
                # overrule it on the SAME evidence -- build_linkage's two-id minimum
                # is all a rebuild knows, and one designation plus one tracklet
                # satisfies it, which is how a Full step used to raise the button
                # on a linkage the search had just refused (2021 PA97, one lone
                # tracklet). Rule: the button may appear after a step only
                # when observations were ADDED or TURNED ON since the search, and
                # only if that grown set then passes the submittability gate
                # (see _itf_pending_block).
                "submittable": result.linkage is not None,
                "reason": itf_wording.humanize(result.reason),
                "judged_ids": sorted(_included_ids(s_new.get("fit"))),
                # ...and the seed-side inputs of the search's OWN opposition-profile gate
                # (itf.search.opp_profile_acceptable), so the re-judge IS that gate
                "seed_gate": _seed_gate_inputs(seed_obs80, seed_fit),
                "itf_ids": sorted({obs_unique_id(l)
                                   for l in (result.itf_obs80 or "").splitlines()
                                   if l.strip()}),
                # the database trksub of each ITF line, which names its tracklet in
                # the rebuilt document instead of the line's columns 6-12
                "itf_trksubs": _itf_trksubs(result),
                "warnings": warnings,
                # Advisory measurements (the residual-trend diagnostic) must survive a re-fit
                # like the warnings do: they are computed by the SEARCH and nothing downstream
                # can recompute them, so a Full step used to make the info banner vanish -- the
                # one thing 27723c5 landed the diagnostic channel to keep visible.
                "diagnostics": diagnostics,
                # the cautions the leave-out orbit vouched for (itf/leaveout.py), RAW, and the
                # photometric tracklet the search named: judged on the search's observation set,
                # so they are re-judged at Submit if that set has changed (_leaveout_rejudge)
                "leaveout_demoted": [o for o in (leaveout_original(d) for d in result.diagnostics or ())
                                     if o],
                "leaveout_magt": (result.vetting or {}).get("mag_tracklet"),
                "vetting": result.vetting or {},
                # designations the partial-pickup check condemned (itf/pickup.py):
                # shown on screen with the rest of the arc, but never claimed by a
                # document -- so the LIVE rebuild has to drop them too, or the
                # submit button would quietly re-add what the search removed
                "banned_designations": list(
                    (result.vetting or {}).get("linkage_dropped_designations")
                    or ()),
                # the search's OWN rendering of the whole accepted arc; the live
                # filter runs over these, never over re-synthesised lines
                "lines": [l for l in (result.combined_obs80 or "").splitlines()
                          if l.strip()],
            }
        # NOTE: ITF quality warnings are deliberately NOT copied into
        # summary["systematic"] -- that field is the O-C pane's residual-trend flag
        # and now reflects only the real systematic_misfit() verdict on the
        # polished orbit; the ITF warnings surface as banners + submit gates.
        n_itf_lines = sum(1 for ln in (result.itf_obs80 or "").splitlines()
                          if ln.strip())
        payload["itf"] = {
            "found": True,
            "possible": possible,           # thin-profile candidate, review only
            "outcome": itf_wording.outcome_label(result.verdict),
            "reason": itf_wording.humanize(result.reason),
            "tracklets": sorted(tracklets.values(), key=lambda t: t["date"]),
            "added_obs": result.telemetry.get("obs_count", 0) or n_itf_lines,
            "itf_obs80": result.itf_obs80,
            "linkage": result.linkage,      # shown for inspection
            "submittable": result.linkage is not None,
            "direct_submit_allowed": _direct_submit_allowed(),
            "warnings": warnings,           # surfaced as caution banners + submit gates
            # Advisory measurements that are NOT doubts: banner only, never a
            # submit gate and never acknowledged.  Same contract as `incomplete`.
            "diagnostics": diagnostics,
            "incomplete": incomplete,       # capacity, not doubt: banner only
            "constrained": constrained,     # tight, but it finished
            "vetting": itf_wording.humanize_vetting(result.vetting),
        }
        return payload

    payload = _found_payload(result)
    # The search can hold two answers (ItfSearchResult.alternative): a linkage
    # it would submit and a fuller unconfirmed 0.6 linking more of the ITF.
    # The page asks the user which to show, so both are built -- each on its
    # own session -- and the other rides on the first; a user who picks it
    # gets a whole fit, not a re-run.
    alt = getattr(result, "alternative", None)
    if (alt is not None and alt.fit is not None and alt.itf_obs80
            and alt.verdict in (SearchVerdict.LINKED, SearchVerdict.POSSIBLE)):
        payload["itf"]["alternative"] = _found_payload(alt)
    return payload


def itf_submit_web(session, acked_warnings=None, _submitter=None, *, name=None, email=None,
                   route="queue", _poster=None):
    """Send the linkage on ``session`` -- the web "Submit linkage" button.

    ``route``:
      * ``"queue"`` (the default) -- POST the document to the
        identification queue (``IDQUEUE_URL``, pyroorb.net's ``/identifications/``;
        a server on pyroorb.net posts to itself), which dedups, checks, paces and
        forwards it.  A linkage that came out of an ITF search here carries
        ``"noextend": true`` so the queue does not run the same search again.
      * ``"mpc"`` -- straight to the MPC, skipping the queue: only where
        ``DIRECT_SUBMIT_ALLOWED`` is set, re-checked here so a hand-crafted POST
        cannot skip the queue either.

    The header is the person's OWN ``name`` / ``email`` (remembered in their
    browser); there is no built-in submitter.  The client never sends the
    document: it is rebuilt from the server-held linkage and the observations
    included NOW.  One-shot: a send that was taken clears the pending linkage.
    """
    name, email = (name or "").strip(), (email or "").strip()
    from pyroorb.itf.linkage import header_problem
    if header_problem({"name": name, "email": email}):
        return {"ok": False, "need_identity": True,
                "error": "a name (first initial + last name) and a valid e-mail address "
                         "are required -- they go in the document header"}
    if route not in ("queue", "mpc"):
        return {"ok": False, "error": f"unknown route {route!r}"}
    if route == "mpc" and not _direct_submit_allowed():
        return {"ok": False,
                "error": "direct submission to the MPC is not allowed on this server "
                         "(DIRECT_SUBMIT_ALLOWED); submit to the PyRoOrb queue instead"}
    s = _SESSIONS.get(session or "")
    # A linkage must EXIST on this session -- an ITF find, or an identification the engine is
    # willing to offer. Whether it is still submittable RIGHT NOW is decided below, by rebuilding
    # the document from the included observations: that path knows why (every linked tracklet
    # switched off, fewer than two ids left) and can say so.
    block = _linkage_block(s) if s else None
    if not s or not block or not block.get("found"):
        return {"ok": False, "expired": True,
                "error": "no linkage on this session -- re-run the ITF search, "
                         "or re-fit the designations together"
                         + (f" ({block.get('reason')})" if block and block.get("reason") else "")}
    # Server-side enforcement of the quality-caution click-throughs: the
    # client must echo back every stored warning it showed the user
    # (``acked_warnings``).  A stale or hand-crafted POST that skipped one
    # is refused -- the browser confirm() loop alone is not a gate.  An ITF
    # find's cautions were stored by the search; a user-assembled
    # identification's are recomputed with the block, so they always describe
    # the fit as it stands right now.
    required = (s.get("itf_warnings") or [] if s.get("itf_pending")
                else block.get("warnings") or [])
    # ...plus any caution the leave-out orbit had vouched for that no longer stands on the
    # observations being submitted (the user changed the set since the search judged it)
    rejudged = _leaveout_rejudge(s) if s.get("itf_pending") else []
    required = list(required) + [w for w in rejudged if w not in required]
    missing = [w for w in required
               if w not in (acked_warnings or [])]
    if missing:
        back = [w for w in missing if w in rejudged]
        return {"ok": False, "warnings": missing, "rejudged": back,
                "error": ("the observations you are submitting no longer confirm "
                          f"{len(back)} caution(s) the search had set aside -- confirm "
                          "each one to submit" if back else
                          "this linkage carries unacknowledged quality "
                          "cautions -- re-run the submission from the UI and "
                          "confirm each caution")}
    # Build from what is INCLUDED NOW, not from the document the search
    # returned.  The user has been looking at this linkage -- stepping the
    # orbit, switching observations off -- and what they submit has to be what
    # they are looking at.  Switch off every observation of a tracklet and it
    # is simply not in the document; switch off enough that fewer than two
    # distinct ids remain and there is no linkage left to send.
    from pyroorb.itf.linkage import comet_refusal
    comet_why = comet_refusal("\n".join(_live_lines(s)) + "\n",
                              (s.get("itf_pending") or {}).get("primaries") or ())
    if comet_why:
        return {"ok": False, "error": comet_why}
    doc, err = _itf_live_linkage(s)
    if doc is None:
        return {"ok": False,
                "error": f"the current observation set is no longer a "
                         f"linkage ({err}) -- switch tracklet observations "
                         f"back on, or re-run the ITF search"}
    refused = _unsubmittable(block)         # a verdict the engine declined to confirm
    if refused:
        return refused
    import copy
    from pyroorb.itf.linkage import AnonymousDocument, signed_document
    try:
        doc = signed_document(copy.deepcopy(doc), name, email)   # the draft, signed
    except AnonymousDocument as ex:
        return {"ok": False, "need_identity": True, "error": str(ex)}
    log_lines = []
    queued = None
    if route == "queue":
        # An ITF find here has had its search: tell the queue not to run it again.
        if s.get("itf_pending"):
            doc["noextend"] = True
        # ...and the person at this page saw and clicked through every caution:
        # the queue must not censor it as bogus or hold it.
        doc["noreject"] = True
        try:
            code, body = (_poster or _post_to_queue)(doc)
        except Exception as ex:
            return {"ok": False, "error": f"could not reach the identification queue "
                                          f"({type(ex).__name__}: {ex})"}
        if code != 200 or not body.get("ok"):
            return {"ok": False, "error": "the identification queue refused it: "
                                          + str(body.get("error") or code)}
        sent, accepted, status = True, True, "queued"
        mpc_msg = body.get("message")
        base = _queue_url().split("/identifications")[0]
        queued = {"document": body.get("document"), "token": body.get("token"),
                  "status_url": base + (body.get("status_url") or ""),
                  "dashboard": base + "/identifications/dashboard"}
    else:
        from pyroorb.itf.config import ItfConfig
        from pyroorb.itf.linkage import submit_linkage
        cfg = ItfConfig(submit=True)            # arm the outward submission
        try:
            if _submitter is not None:
                sent, status, mpc_msg = _submitter(doc, cfg, log_lines.append)
            else:
                sent, status, mpc_msg = submit_linkage(doc, cfg, do_submit=True,
                                                       log=log_lines.append)
        except Exception as ex:
            return {"ok": False, "error": f"{type(ex).__name__}: {ex}",
                    "log": log_lines[-8:]}
        accepted = bool(sent) and "200" in (status or "")
    if accepted:
        s["itf_linkage"] = None             # prevent a double submission
        s["itf_warnings"] = []
        s["itf_pending"] = None
        # A user-assembled identification has no pending record to clear -- the linkage is implied
        # by the fit, which is still on screen. Mark the session so the button does not invite the
        # same submission twice (a re-fit of the same designations is a new session, as it should
        # be: that is a deliberate second look, not a stray double-click).
        s["linkage_submitted"] = True
    return {"ok": True, "submitted": bool(sent), "accepted": accepted, "route": route,
            "queued": queued, "status": status, "message": mpc_msg, "linkage": doc,
            "log": log_lines[-8:]}


def itf_download_web(session, name, email):
    """Build the MPC identification JSON for the linkage stashed on ``session``
    and hand it back for a client-side download (the web "Download linkage
    file" item of the submit menu -- nothing is transmitted).
    The caller-supplied name/email replace the configured submitter in
    the document header; the same pre-submit DES=DES dedup a real submission
    runs is applied best-effort (skipped without DB access), and -- because that
    dedup can collapse two designations into one family primary -- the same
    two-id backstop (:func:`itf.linkage.short_link`) decides whether the file is
    worth writing at all. A hand-submitted one-id document is the resubmitted
    DES=DES the dedup exists to prevent, so this route refuses it exactly as
    ``submit_linkage`` does rather than handing the user a rejection to post.
    NOT one-shot -- the pending linkage stays so the file can be re-downloaded."""
    name = (name or "").strip()
    email = (email or "").strip()
    from pyroorb.itf.linkage import header_problem
    if header_problem({"name": name, "email": email}):
        return {"ok": False,
                "error": "a name (first initial + last name) and a valid "
                         "e-mail address are required"}
    s = _SESSIONS.get(session or "")
    # A linkage must EXIST on this session -- an ITF find, or an identification the engine is
    # willing to offer. Whether it is still submittable RIGHT NOW is decided below, by rebuilding
    # the document from the included observations: that path knows why (every linked tracklet
    # switched off, fewer than two ids left) and can say so.
    block = _linkage_block(s) if s else None
    if not s or not block or not block.get("found"):
        return {"ok": False, "expired": True,
                "error": "no linkage on this session -- re-run the ITF search, "
                         "or re-fit the designations together"
                         + (f" ({block.get('reason')})" if block and block.get("reason") else "")}
    import copy
    # same live-state rule as submission: the file you download is the linkage
    # your current observation set makes, not the search's frozen output
    live, err = _itf_live_linkage(s)
    if live is None:
        return {"ok": False,
                "error": f"the current observation set is no longer a "
                         f"linkage ({err})"}
    # ...and the same verdict the submit route asks, at the same point: the download is the
    # offline alternative to submitting, not a way around it -- the file it writes IS the
    # identification document. Both buttons hide on the same flag.
    refused = _unsubmittable(block)
    if refused:
        return refused
    from pyroorb.itf.linkage import AnonymousDocument, signed_document
    try:
        doc = signed_document(copy.deepcopy(live), name, email)  # the draft, signed
    except AnonymousDocument as ex:
        return {"ok": False, "error": str(ex)}
    note = None
    try:
        from pyroorb.itf.config import ItfConfig
        from pyroorb.itf.dbio import connect_pub
        from pyroorb.itf.linkage import _family_map, dedup_designations, unpack
        link = doc.get("links", {}).get("link_0", {})
        unp = [unpack(pk) or pk for pk in link.get("designations") or []]
        conn = connect_pub(ItfConfig())
        try:
            dedup_designations(doc, _family_map(conn, unp).get,
                               log=lambda *_a: None)
        finally:
            conn.close()
    except Exception as ex:
        note = (f"identification dedup unavailable "
                f"({type(ex).__name__}) -- document written as-is")
    # The dedup ran AFTER build_linkage's own two-id guard and may have dropped
    # an id; downloading is the one route with nothing downstream to catch that.
    from pyroorb.itf.linkage import short_link
    short = short_link(doc)
    if short:
        return {"ok": False, "error": short}
    link = doc.get("links", {}).get("link_0", {})
    first = (link.get("designations") or link.get("trksubs") or ["linkage"])[0]
    prefix = "itf_" if s.get("itf_pending") else "ident_"
    fname = prefix + re.sub(r"[^A-Za-z0-9_-]+", "", str(first)) + ".json"
    return {"ok": True, "filename": fname, "document": doc, "note": note}


def run_nongrav(sid, model, mask_list=None):
    """Switch the session's active non-gravitational model and re-solve (find_orb's
    "turn on A1/A2/..."). Honours the client's current edited obs set and re-seeds from the
    force-model baseline, so switching models never compounds and never discards the user's
    inclusion edits. ``none`` returns to a gravity fit on the same obs set. This touches ONLY
    the nongrav selector -- any active element constraints and the +16ast force model are kept
    and re-applied (they all compose)."""
    s = _SESSIONS.get(sid)
    if not s or "grav_fit" not in s:
        return {"ok": False, "error": "session expired -- re-fit the object", "expired": True}
    model = (model or "none").strip()
    prev = s.get("model")
    s["model"] = None if model == "none" else model           # update ONLY this selector
    payload = _apply_interactive(s, sid, mask_list)
    if not payload.get("ok", True):
        s["model"] = prev                                     # roll back on failure
    return payload


def nongrav_sigmas(sid):
    """On-demand HONEST (profile-likelihood, Delta-chi2=1) 1-sigma for the session's CURRENT
    non-gravitational fit -- backs the "±" button. The interactive model switch serves the fast
    LINEAR sigma, which reports a fake-tight, high-sigma "detection" for a coefficient that is
    degenerate with the state (fits equally well at zero). Computing the real interval re-fits the
    orbit at many trial values (seconds of ASSIST), so it runs ONLY when the user asks. Reuses the
    stored fit (no re-solve). Returns {sigmas: {A?: profile-sigma}, intervals: {A?: [minus, plus]}
    (the honest asymmetric half-widths), unconstrained: [axes the profile could not bound]}."""
    s = _SESSIONS.get(sid)
    if not s or s.get("fit") is None:
        return {"ok": False, "error": "session expired -- re-fit the object", "expired": True}
    fit = s["fit"]
    if getattr(fit, "nongrav_model", None) in (None, "none"):
        return {"ok": True, "sigmas": {}, "intervals": {}, "unconstrained": []}
    fit._nongrav_profile = True          # opt this fit into the profile...
    fit._rational_sigmas = None          # ...and discard any linear fallback so it recomputes
    sig = fit.rational_nongrav_sigmas()
    rint = fit.rational_nongrav_intervals()
    # The delete-1-tracklet JACKKNIFE robustness sigma, computed alongside the profile on this same
    # button (both cost real refits, so both wait for the user to ask). It removes a whole tracklet
    # -- the unit a correlated systematic lives in -- and refits, so a coefficient resting on a few
    # high-leverage tracklets or fitting a systematic reads a LOW jackknife z even when the profile z
    # is high (2008 NA srp2 A1: profile z~200, jackknife z~1). `not_robust` lists the axes whose
    # leave-one-tracklet cloud straddles zero. This is the ONE uncertainty covariance/profile (and
    # find_orb's Monte Carlo, which jitters obs independently) cannot give -- see the method.
    jk = fit.nongrav_jackknife_detail()
    # AMR is a fixed multiple of A1 (1/SRP1AU), so its interval is that of A1 rescaled -- no
    # extra refits. Sent only for the model find_orb reports it for (the 1-parameter SRP).
    amr = _ngm.amr_report(fit.nongrav_model, fit.nongrav_params, rint, sig,
                          fit.nongrav_unconstrained)
    amr_sig = _safe(max(amr["minus"], amr["plus"])) if amr and amr["plus"] is not None else None
    amr_int = ([_safe(amr["minus"]), _safe(amr["plus"])]
               if amr and amr["plus"] is not None else None)
    return {"ok": True,
            "sigmas": {k: _safe(v) for k, v in sig.items()},
            "intervals": {k: [_safe(v[0]), _safe(v[1])] for k, v in rint.items()},
            "unconstrained": list(fit.nongrav_unconstrained),
            "amr_sigma": amr_sig, "amr_interval": amr_int,
            # `sampled` says the delete-1 cloud is a SUBSET of the tracklets (the budget is
            # JACKKNIFE_MAX_REFITS = 16). Without it the client cannot tell a partial sweep
            # from a complete one -- `n` is converged refits, so n < n_tracklets is equally
            # consistent with rejected refits -- and the panel claimed the sign survives
            # dropping "a whole tracklet" when an unsampled one had never been tried.
            # Both branches also carry `attempted` (how many deletions were actually TRIED,
            # vs `n_tracklets` the arc HAS -- the refit budget or an undeletable tracklet can
            # make the two differ) alongside `n` (CONVERGED refits, always <= attempted). The
            # panel used to label `n` "of n_tracklets refit/converged" in both branches, which
            # reads as "we tried every tracklet and most refits failed" even when the budget
            # never reached most of them, or as "N were refit" when N is only the CONVERGED
            # subset of a larger attempted set. NOT-ASSESSED also carries `reason` (too few
            # converged / too many rejected / no spread).
            "jackknife": {k: ({"assessed": True, "se": _safe(d["se"]), "z": _safe(d["z"]),
                                "spans_zero": bool(d["spans_zero"]), "n": d["n"],
                                "attempted": d.get("attempted", d["n"]),
                                "sampled": bool(d.get("sampled")),
                                "n_tracklets": d["n_tracklets"]}
                               if d.get("assessed") else
                               {"assessed": False, "n": d.get("n", 0),
                                "attempted": d.get("attempted", 0),
                                "reason": d.get("reason"),
                                "n_tracklets": d.get("n_tracklets")})
                          for k, d in jk.items()},
            "not_robust": list(fit.nongrav_not_robust)}


def run_constrained(sid, constraints, mask_list=None):
    """Hold one or more orbital ELEMENTS fixed and re-solve (find_orb's limited_orbit /
    evaluate_limited_orbit). `constraints` is {element: target} over a/e/q/Q/i; an empty dict
    releases all constraints. Honours the client's edited obs set. This touches ONLY the
    constraints selector -- any active nongrav model and the +16ast force model are kept and
    re-applied, so a constrained fit no longer drops the comet/Yarkovsky model (they compose)."""
    s = _SESSIONS.get(sid)
    if not s or "grav_fit" not in s:
        return {"ok": False, "error": "session expired -- re-fit the object", "expired": True}
    prev = s.get("constraints")
    s["constraints"] = _clean_constraints(constraints)        # update ONLY this selector
    payload = _apply_interactive(s, sid, mask_list)
    if not payload.get("ok", True):
        s["constraints"] = prev                               # roll back on failure
    return payload


def set_asteroids(sid, on, mask_list=None):
    """Interactive force-model toggle: flip the 16 massive-asteroid perturbers and re-fit the
    SAME observation set -- it does NOT redeem or drop any obs, only the force model changes.
    Honours the client's current inclusion set exactly (`refit_force_model`), seeding from and
    building both force models off the pure-gravity baseline. This touches ONLY the force-model
    selector -- any active nongrav model and element constraints are KEPT and re-applied on the
    new base (they all compose). If no kernel is available, ON gracefully stays gravity (the
    payload reports the true state)."""
    s = _SESSIONS.get(sid)
    if not s or "grav_pure" not in s:
        return {"ok": False, "error": "session expired -- re-fit the object", "expired": True}
    mask, err = _client_mask(s["grav_fit"], mask_list)
    if err is not None:
        return err
    err_n = _underdetermined(mask.sum())
    if err_n is not None:
        return err_n
    # Self-perturber: if the object IS one of the 16 +16ast perturbers, "on" means the other
    # 15 (the fit's backend excludes the object itself -- see refit_force_model). If that
    # exclusion is unavailable the engine stays on gravity; banner it then.
    sp = guards.self_perturber_name(fit=s["grav_pure"])
    seed = None
    if bool(on):
        # seed the asteroid fit from a good asteroid state (avoids the spurious near-gravity
        # local min on perturber-dependent obs sets); compute + cache it once if absent.
        if s.get("ast_state") is None:
            try:
                aseed = fo.refine_asteroids(s["grav_pure"], force=True)
                s["ast_state"] = (np.asarray(aseed._state, dtype=float)
                                  if aseed.asteroids else None)
            except Exception:
                s["ast_state"] = None
        seed = s.get("ast_state")
    try:
        base = fo.refit_force_model(s["grav_pure"], bool(on), mask=mask, seed=seed)
    except Exception as ex:
        return {"ok": False, "error": f"asteroid toggle failed: {type(ex).__name__}: {ex}"}
    s["grav_fit"] = base                     # the new force-model baseline
    s["ctx"] = base._ctx
    s["state"] = np.asarray(base._res.x, dtype=float)
    s["history"] = base.history
    s["epoch_tdb"] = float(base.epoch_tdb)
    s["asteroids"] = bool(base.asteroids)
    # s["model"] and s["constraints"] are PRESERVED -- re-applied on the new base below.
    payload = _apply_interactive(s, sid, mask_list)
    if not payload.get("ok", True):
        return payload
    if sp is not None:                       # object IS a +16ast perturber
        payload["self_perturber"] = sp       # the toggle's "other 15" label on re-render
        if bool(on) and not base.asteroids:  # asked for perturbers, engine could not exclude
            payload["banner"] = self_perturber_banner(sp)
    return payload


def _epoch_to_jd_tdb(epoch):
    """Parse a UI epoch value -- an ISO date/datetime string ("2026-06-19") or a bare
    JD number -- to JD (TDB). A date with no time is taken at 00:00 TDB, matching the
    date-only granularity the elements view shows."""
    from astropy.time import Time
    if isinstance(epoch, (int, float)):
        return float(epoch)
    s = str(epoch).strip()
    if not s:
        raise ValueError("empty epoch")
    try:                                           # a bare JD ("2461000.5")
        return float(s)
    except ValueError:
        pass
    t = s if (" " in s or "T" in s) else s + " 00:00:00"
    return float(Time(t.replace("T", " "), format="iso", scale="tdb").jd)


def set_epoch(sid, epoch, mask_list=None):
    """Re-anchor the session's orbit to a new osculating epoch (find_orb's "set epoch").
    Integrates the orbit to `epoch` and reports the new osculating elements there WITHOUT
    a differential correction -- the residuals are unchanged (it is the same trajectory),
    and a subsequent Full step runs from this new epoch. `epoch` is an ISO date string
    (date picker) or a JD. The gravity baseline, the live evaluator/state (so Full step
    anchors here) and -- if a nongrav model is active -- the displayed out-gassing orbit
    are all re-anchored."""
    s = _SESSIONS.get(sid)
    if not s or "grav_fit" not in s:
        return {"ok": False, "error": "session expired -- re-fit the object", "expired": True}
    try:
        new_jd = _epoch_to_jd_tdb(epoch)
    except Exception as ex:
        return {"ok": False, "error": f"could not parse epoch '{epoch}': {ex}"}
    # honour the user's current inclusion edits (decoupled from the epoch, but they set
    # which obs the elements' covariance is weighted over and carry into the next Full step)
    mask, err = _client_mask(s["grav_fit"], mask_list)
    if err is not None:
        return err
    try:
        grav2 = fo.reanchor_fit_to_epoch(s["grav_fit"], new_jd, mask=mask)
    except Exception as ex:
        return {"ok": False, "error": f"set-epoch failed: {type(ex).__name__}: {ex}"}
    # Re-anchor the live session state so a Full step (and model switches) run from here.
    s["grav_fit"] = grav2
    s["ctx"] = grav2._ctx
    s["state"] = np.asarray(grav2._res.x, dtype=float)
    s["history"] = grav2.history
    s["epoch_tdb"] = float(new_jd)
    model = s.get("model")
    if model:                                      # re-derive the active out-gassing at the new
        try:                                       # epoch (reseeds from the re-anchored gravity
            fit2 = fo.refine_nongrav(grav2, model, mask=mask, revert_if_worse=False,
                                     asteroids=fo._nongrav_asteroids_gate(grav2),
                                     radial_seed_all=True,   # match _apply_interactive: seed the joint
                                     #   radial models + converge to the true minimum (tight default tol)
                                     profile_sigmas=False)   # ± computed on demand by the button
        except Exception as ex:
            return {"ok": False, "error": f"set-epoch (nongrav re-anchor) failed: "
                                          f"{type(ex).__name__}: {ex}"}
    else:
        fit2 = grav2
    s["fit"] = fit2
    payload = serialise_fit(fit2)
    payload["session"] = sid
    payload["meta"] = {"interactive": True, "set_epoch": True}
    payload["nongrav_request"] = model or "none"
    return payload


# --------------------------------------------------------------------------
# Top-level entry points used by the HTTP layer.
# --------------------------------------------------------------------------
MAX_DESIGS = 4   # more than this in the box is a paste mistake, not a real linkage

_YEAR_RE = re.compile(r"^\d{4}$")
# A comet-form provisional carries its half-month letter+number after a space, just like an
# asteroid provisional, but the leading "year" token is prefixed with the orbit-type letter and
# a slash ("C/2023", "P/2010", also D/X/I/A). Without this, the box would split "C/2023 A3" into
# two bogus designations -- and ensure_unpacked would even mangle "A3" -> "103".
_COMET_YEAR_RE = re.compile(rf"^[{obs80._COMET_TYPES}]/\d{{4}}$", re.IGNORECASE)  # types from obs80 (single source)
# A natural-satellite provisional is "S/YYYY L N" -- a satellite-prefixed year, then a single
# PLANET letter, then a bare index number, with a space before BOTH ("S/2003 J 2"). The leading
# "S/2003" token must count as a year (so the suffix merges), and the trailing " 2" must be
# absorbed into the same designation rather than read as a second object to identify.
_SAT_YEAR_RE = re.compile(r"^S/\d{4}$", re.IGNORECASE)


def parse_designations(text):
    """Tokenise the object box into a list of designations to fit as one combined /
    identified object. Any run of whitespace, comma or semicolon separates
    designations -- EXCEPT a space between a year token and a following letter-led
    token, which is part of one provisional designation: a bare 4-digit year
    ("2010 VL65") or a comet-prefixed year ("C/2023 A3", "P/2010 A2").
    All '=' (the identity sign) are stripped first. Packed forms (e.g. "K10V65L")
    have no internal space, so they split naturally and are unpacked downstream.

        "2010 VL65 2021 UA12"      -> ["2010 VL65", "2021 UA12"]
        "2010 VL65 = 2021 UA12"    -> ["2010 VL65", "2021 UA12"]
        "K10V65L   K21U12A"        -> ["K10V65L", "K21U12A"]
        "C/2023 A3"                -> ["C/2023 A3"]
        "S/2003 J 2"               -> ["S/2003 J 2"]       # natural satellite (two internal spaces)
    """
    if not text:
        return []
    raw = [t for t in re.split(r"[\s,;]+", text.replace("=", " ").replace("+", " ").strip()) if t]
    out, i = [], 0
    n = len(raw)
    while i < n:
        t = raw[i]
        is_year = _YEAR_RE.match(t) or _COMET_YEAR_RE.match(t) or _SAT_YEAR_RE.match(t)
        if is_year and i + 1 < n and raw[i + 1][:1].isalpha():
            suffix = raw[i + 1]
            # Natural-satellite form "S/2003 J 2" / "2003 J 2": a SINGLE planet letter followed by a
            # bare index number (1-3 digits, so a following 4-digit YEAR is never swallowed) is all
            # one designation -- absorb the number too.
            if (len(suffix) == 1 and i + 2 < n
                    and raw[i + 2].isdigit() and len(raw[i + 2]) <= 3):
                out.append(f"{t} {suffix} {raw[i + 2]}")
                i += 3
            else:
                out.append(f"{t} {suffix}")     # year + provisional suffix -> one desig
                i += 2
        else:
            out.append(t)
            i += 1
    return out


def pluto_banned_payload():
    """The error payload returned for any banned Pluto fit: ``ok:False`` so the front
    end's failure path fires, plus ``pluto_banned``/``banner`` so it can raise a
    prominent banner with the engine's PLUTO_BAN_MSG (Pluto is a DE440 perturber and
    cannot be fit unperturbed by itself)."""
    return {"ok": False, "error": guards.PLUTO_BAN_MSG,
            "banner": guards.PLUTO_BAN_MSG, "pluto_banned": True}


def self_perturber_banner(name):
    """User-facing message for an object that IS one of the 16 +16ast perturbers when the
    engine could NOT run the other 15 with it excluded (force_assist._masked_ephem failed):
    it stays on gravity. Normally the toggle simply runs +15ast and no banner is shown."""
    return (f"({name}) is one of the 16 massive asteroids in the +16ast perturber set and "
            f"can't be perturbed by itself; excluding it from the perturber field failed on "
            f"this server, so it stays on gravity.")


def run_fit(desig, force=False):
    """Fetch (cache-first) and fit one object, or several designations as one
    combined / identified object. Returns the serialised payload, or an error
    dict the front end can show."""
    desigs = parse_designations(desig)
    if not desigs:
        return {"ok": False, "error": "no designation given"}
    if len(desigs) > MAX_DESIGS:
        return {"ok": False, "error": f"too many designations ({len(desigs)}); the limit "
                f"is {MAX_DESIGS} — a longer list is probably a paste mistake"}
    desigs = [obs_sources.ensure_unpacked(d) for d in desigs]   # accept packed or unpacked
    # Pluto ban (by typed name/number): a DE440 perturber, can't be fit unperturbed by
    # itself. A self-identifying-by-number fetch is caught again in fit_obs_list/fit_orbit.
    if any(guards.is_pluto(d) for d in desigs):
        return pluto_banned_payload()
    obscodes = _obscodes()
    obs_sources.pop_unknown_obscodes()       # reset the collector so we surface only this fetch's
    try:
        obs, meta = obs_sources.fetch_observations(desigs, obscodes)
    except Exception as ex:
        return {"ok": False, "error": f"fetch failed: {ex}",
                "hint": "object may not be cached and network is offline"}
    if not obs:
        # "no observations" and "the MPC did not answer" are different answers and only
        # one of them is about the object.  Saying the first when the second happened
        # tells the user their designation does not exist on the strength of a dropped
        # connection -- and under load (a throttled container on launch day) it says it
        # to everyone at once.  obs_sources records which it was.
        failed = (meta or {}).get("fetch_errors") or {}
        if failed:
            names = ", ".join(sorted(failed))
            return {"ok": False, "retryable": True,
                    "error": f"the MPC did not answer for {names} — this is a problem "
                             f"reaching the observation service, not a statement about "
                             f"the object. Please try again in a moment.",
                    "detail": failed}
        return {"ok": False, "error": f"no observations found for {desigs}"}
    # Rows came back, but not necessarily for everything asked for: a linkage whose
    # second designation failed to fetch would otherwise be fitted on half its evidence
    # and reported as a success.  Refuse rather than answer from a silently short arc.
    failed = (meta or {}).get("fetch_errors") or {}
    if failed:
        names = ", ".join(sorted(failed))
        return {"ok": False, "retryable": True,
                "error": f"the MPC did not answer for {names}, so only part of the "
                         f"requested astrometry arrived — fitting it would answer from "
                         f"a shorter arc than you asked for. Please try again in a moment.",
                "detail": failed}
    # fit_obs_list drains the unknown-obscode record into the payload (covers both the
    # fetch path here and the pasted-blob path).
    return fit_obs_list(obs, meta, obscodes, force=force,
                        meta_extra={"format": meta.get("format"),
                                    "n_input_designations": len(desigs),
                                    # cross-listed copies the fetch collapsed (obs_sources.
                                    # dedup_observations); the UI notes it under the header
                                    "duplicates_dropped": meta.get("duplicates_dropped") or []})


def post_discovery_tail(display, obscodes):
    """fit_object._finalize's POST-DISCOVERY tail, run on the web's discovered orbit:
    the final stale-exclusion reclaim and then the delivered verification. Returns
    ``(display, auto_model)`` -- the orbit to show and the model it actually carries.

    Split out of :func:`fit_obs_list` so it can be tested with stubs, and so the failure
    handling is one thing you can read. ONE try per step, and neither of them silent: a
    single ``except Exception: pass`` around the pair (the first version) put back the
    swallow that ``_delivered_verification`` is emphatic about -- "FAIL LOUD, never
    open" -- one level up, on the surface most people actually look at, and left
    ``auto_model`` holding the PRE-tail model whenever the reclaim raised, so the model
    dropdown pre-selected something the displayed orbit no longer carried. The gate
    records its own internal errors; only what ESCAPES to here can be reported by us, and
    ``nongrav_warning`` is the advisory channel the panel renders. `auto_model` is re-read
    from the delivered orbit unconditionally, so the two can no longer disagree."""
    try:
        display = fo._final_reclaim(display, [], obscodes)
    except Exception as ex:
        fo._ng_warn_on(display, f"the post-delivery reclaim did not complete "
                                f"({type(ex).__name__}: {ex}) -- this is the pre-reclaim orbit")
    try:
        display = fo._delivered_verification(display, [])
    except Exception as ex:
        fo._ng_warn_on(display, f"the delivered-model verification did not complete "
                                f"({type(ex).__name__}: {ex}) -- the model was kept and is "
                                f"UNVERIFIED; treat the coefficients as provisional")
    return display, getattr(display, "nongrav_model", None)


def fit_obs_list(obs, meta, obscodes, force=False, meta_extra=None):
    """Fit a ready parsed-obs list (the shared core of ``run_fit`` and
    ``add_observations``): pure-gravity baseline -> asteroid polish -> auto-nongrav
    discovery -> interactive session. Returns the serialised payload (with a live
    ``session``) or an error dict. ``meta_extra`` is merged into ``payload["meta"]``."""
    # Drain any ground obs dropped during parsing (fetch or paste) for an unrecognised
    # station -> surfaced on the payload so the UI can warn the table may be stale.
    unknown = obs_sources.pop_unknown_obscodes()
    # Pluto ban: catch obs that self-identify as 134340 / Pluto (e.g. a paste, or a
    # fetch that came back under Pluto's number) before any fit work runs.
    if any(guards.is_pluto(c) for c in guards._pluto_candidates(meta=meta, obs=obs)):
        return pluto_banned_payload()
    try:
        arc_extend.SNAPSHOTS = True          # capture per-stage state+mask for the replay
        # auto_nongrav=False here: keep this a PURE gravity fit so it can serve as the
        # baseline the model dropdown / asteroid toggle reseed from. We run discovery
        # explicitly below and DISPLAY its result, but never overwrite the baseline.
        fit = fo.fit_orbit(obs, obscodes, desig=meta["designation"],
                           meta=meta, verbose=False, force=force, auto_nongrav=False,
                           mpc_seed=fo.fetch_mpc_seed(obs, meta))
    except guards.PlutoBanError:
        return pluto_banned_payload()
    except Exception as ex:
        return {"ok": False, "error": f"fit failed: {ex}"}
    finally:
        arc_extend.SNAPSHOTS = False
    # Precision polish (UI default-on): for main-belt, multi-opposition objects refine with
    # the 16 massive-asteroid perturbers -- auto-downloads the kernel on first use, falls back
    # to gravity if unavailable, and is a gated no-op for objects it can't help. Stored as the
    # session baseline, so interactive Full steps run with the perturbers too. Never fatal.
    grav = fit
    try:
        fit = fo.refine_asteroids(fit)
    except Exception:
        fit = grav
    # Automatic non-gravitational discovery (the same two routines the CLI runs by default).
    # `fit` (gravity/asteroid) stays the session baseline the dropdown reseeds from; we DISPLAY
    # the discovered orbit and pre-select its model. COMET out-gassing (discover_comet_nongrav,
    # triggered by a C//P/ designation or a cometary Tisserand) runs FIRST and WINS the overlap:
    # if it applies, the asteroid Yarkovsky/SRP discovery is skipped; if it only triggers-but-
    # declines, the asteroid path still runs (a low-q object can be Yarkovsky-not-outgassing).
    # Mirrors fit_object._finalize so the web result matches the CLI -- INCLUDING the
    # post-discovery tail (stale-exclusion reclaim + the delivered verification), which is
    # where a basin-escape phantom is exposed: 2012 VP6's yarkovsky clears every
    # in-discovery bar and only dies at the final pinned-A=0 refit, which runs AFTER the
    # reclaim's delivery-grade rebuild has walked the co-adapted (state, A) pair into its
    # real basin. Without this tail the web showed VP6 as an auto-yarkovsky the CLI
    # (correctly) refuses -- a web/CLI divergence. Radar is CLI-only here, so the
    # tail runs with no radar rows (the verification's radar suppression never engages).
    auto_model, display = None, fit
    try:
        cg_fit, cg_info = fo.discover_comet_nongrav(fit)
        if cg_info.get("applied"):
            auto_model, display = cg_info["model"], cg_fit
    except Exception:
        pass
    if auto_model is None:
        try:
            ng_fit, ng_info = fo.discover_nongrav(fit)
            if ng_info.get("applied"):
                auto_model, display = ng_info["model"], ng_fit
        except Exception:
            pass
    if auto_model is not None:
        display, auto_model = post_discovery_tail(display, obscodes)
    payload = serialise_fit(display)
    payload["nongrav_auto"] = auto_model
    # True only when the gated polish KEPT the 16 perturbers because they meaningfully
    # lowered the same-obs RMS (refine_asteroids reverts to gravity otherwise). The front
    # end flashes a one-shot banner inviting the user to A/B the force model. A manual
    # +16-ast toggle (force=True) goes through serialise_fit without this flag, so it
    # never re-fires the banner.
    payload["asteroids_auto"] = bool(getattr(fit, "asteroids", False)
                                     and not getattr(grav, "asteroids", False))
    # If THIS object is one of the 16 +16ast perturbers, tell the front end so it can label
    # the +16ast toggle "the other 15" -- the object is excluded from its own field.
    sp = guards.self_perturber_name(fit=grav)
    if sp is not None:
        payload["self_perturber"] = sp
    sid = _store_session(fit, grav_pure=grav)
    if auto_model:                                # display + remember the auto-detected model
        _SESSIONS[sid]["model"] = auto_model      # (a model switch / Full step keeps using it)
        _SESSIONS[sid]["fit"] = display
        # Seed the per-model cache with the auto-detected orbit so re-selecting it reproduces the
        # auto fit exactly (the canonical good minimum), not a fresh profile-seed. See _apply_interactive.
        _SESSIONS[sid]["ng_cache"] = {auto_model: display}
    payload["session"] = sid
    payload["meta"] = dict(meta_extra or {})
    # A multi-designation fit IS a proposed identification -- offer it as one from the first
    # payload, not only after an ITF search (see _ident_linkage_block).
    link = _linkage_block(_SESSIONS[sid])
    if link is not None:
        payload["itf"] = link
    if unknown:
        payload["unknown_obscodes"] = sorted(unknown)
    return payload


# --------------------------------------------------------------------- random object
# The top bar's "Random asteroid" button: pick a real object out of the cached MPCORB
# index and fit it. Two deliberate limits:
#   * at most RANDOM_MAX_OPPOSITIONS oppositions -- a 30-opposition main-belter is a
#     multi-thousand-observation fit, and handing those out at the click of a button is
#     how a public server burns its CPU; 1-7 oppositions is a real object with a real
#     (often interesting) arc that fits quickly;
#   * the front end rate-limits the button (see static/randomobj.js).
# Families are element boxes over the same index `mpcorb.sample_asteroids` filters on
# (a, e, i, q, H, nopp) -- served to the UI by :func:`random_families` so the dropdown
# can never drift out of step with the engine's own definitions (the ITF-modal lesson).
RANDOM_MAX_OPPOSITIONS = 7

RANDOM_FAMILIES = [
    {"key": "any",     "label": "Any asteroid",
     "hint": "anything in MPCORB with a short enough arc"},
    {"key": "neo",     "label": "Near-Earth (NEO)",
     "hint": "perihelion within 1.3 au"},
    {"key": "mba",     "label": "Main belt (MBA)",
     "hint": "a = 2.0–3.3 au, outside the Mars-crossing zone"},
    {"key": "mars",    "label": "Mars crosser",
     "hint": "perihelion 1.3–1.666 au"},
    {"key": "hilda",   "label": "Hilda",
     "hint": "3:2 resonance with Jupiter, a = 3.7–4.2 au"},
    {"key": "trojan",  "label": "Jupiter Trojan",
     "hint": "co-orbital with Jupiter, a = 5.05–5.35 au"},
    {"key": "centaur", "label": "Centaur",
     "hint": "between Jupiter and Neptune, a = 5.5–30 au"},
    {"key": "tno",     "label": "Trans-Neptunian (TNO)",
     "hint": "a beyond Neptune, 30 au and out"},
    {"key": "highinc", "label": "Highly inclined",
     "hint": "inclination above 40°"},
]

# key -> filter kwargs for mpcorb.sample_asteroids; (min, max), either bound optional.
_RANDOM_FILTERS = {
    "any":     {},
    "neo":     {"q": (None, 1.3)},
    "mba":     {"a": (2.0, 3.3), "q": (1.666, None)},
    "mars":    {"q": (1.3, 1.666), "a": (None, 3.2)},
    "hilda":   {"a": (3.7, 4.2), "e": (None, 0.3)},
    "trojan":  {"a": (5.05, 5.35)},
    "centaur": {"a": (5.5, 30.1)},
    "tno":     {"a": (30.1, None)},
    "highinc": {"incl": (40.0, None)},
}


def random_families():
    """The families the random-object dropdown offers, plus the opposition cap that
    applies to all of them. The UI builds its menu from this, so adding a family here
    is the whole change."""
    return {"ok": True, "families": RANDOM_FAMILIES,
            "max_oppositions": RANDOM_MAX_OPPOSITIONS}


def random_object(family="any"):
    """One random designation from `family`, ready to drop into the search box and fit.

    Returns the designation plus the index's own elements for it (a/e/i/H and the
    opposition count) so the front end can say what it just handed you before the fit
    comes back. Cheap: the parsed MPCORB index is a local .npz, no network."""
    from pyroorb import mpcorb
    key = (family or "any").strip().lower()
    if key not in _RANDOM_FILTERS:
        return {"ok": False, "error": f"unknown family {family!r}"}
    filters = dict(_RANDOM_FILTERS[key], nopp=(1, RANDOM_MAX_OPPOSITIONS))
    try:
        mpcorb.build_index(verbose=False)
        z = np.load(mpcorb.MPCORB_NPZ, allow_pickle=False)
    except Exception as ex:                       # no catalogue cached and no network
        return {"ok": False, "error": f"MPCORB index unavailable: {type(ex).__name__}: {ex}"}
    av, ev, iv, Hv = z["a"], z["e"], z["incl"], z["H"]
    npv = z["nopp"].astype(float)
    qv = av * (1.0 - ev)
    mask = (mpcorb._between(av, filters.get("a")) & mpcorb._between(ev, filters.get("e"))
            & mpcorb._between(iv, filters.get("incl")) & mpcorb._between(qv, filters.get("q"))
            & mpcorb._between(Hv, filters.get("H")) & mpcorb._between(npv, filters["nopp"]))
    idx = np.flatnonzero(mask)
    if not idx.size:
        return {"ok": False, "error": f"no {key} objects with "
                f"1–{RANDOM_MAX_OPPOSITIONS} oppositions in the catalogue"}
    i = int(np.random.default_rng().choice(idx))   # unseeded: a fresh draw every click
    lab = next((f["label"] for f in RANDOM_FAMILIES if f["key"] == key), key)
    return {"ok": True, "family": key, "family_label": lab,
            "designation": obs_sources._unpack(str(z["packed"][i]).strip()),
            "oppositions": int(npv[i]), "a": _safe(av[i]), "e": _safe(ev[i]),
            "incl": _safe(iv[i]), "H": _safe(Hv[i]), "n_matching": int(idx.size)}


def published_orbit(desig):
    """The MPC's OWN published orbit for `desig` (GET /api/published_orbit).

    NOT our fit, and the payload says so in every response (`source: "MPC"`), because
    the one thing this must never do is let a visitor mistake the Minor Planet Center's
    orbit for one PyRoOrb computed.  It is here because the two have wildly different
    costs: a well-observed object's published orbit is one cheap request, while fitting
    its whole arc is minutes (433 Eros: 18151 observations, ~10 minutes), so the honest
    thing to show someone who just typed "433" is the known orbit immediately, with our
    fit replacing it when it arrives.

    A missing orbit and an unreachable MPC are answered differently, for the same reason
    the fetch path distinguishes them: one is a fact about the object, the other about
    the network (see obs_sources.fetch_observations)."""
    d = (desig or "").strip()
    if not d:
        return {"ok": False, "error": "no designation given"}
    if guards.is_pluto(d):
        return pluto_banned_payload()
    try:
        d = obs_sources.ensure_unpacked(d)
    except Exception as ex:
        return {"ok": False, "error": f"could not read that designation: {ex}"}
    import urllib.error
    from pyroorb.mpc_orbit import fetch_published_elements
    try:                                    # same cache file the seed fetch already uses
        cache = os.path.join(datadirs.cache_dir(), "mpcorb_%s.json" % obs80.cache_stem(d))
        doc = fetch_published_elements(d, cache=cache)
    # Only a genuine TRANSPORT failure may be reported as the MPC not answering.
    # The first version caught everything, and promptly blamed the MPC for a local
    # FileNotFoundError writing our own cache -- the exact category error this
    # endpoint's error messages exist to avoid, committed in the error message.
    except (urllib.error.URLError, TimeoutError, ConnectionError) as ex:
        return {"ok": False, "retryable": True,
                "error": f"the MPC did not answer for {d} — this is a problem reaching "
                         f"the orbit service, not a statement about the object.",
                "detail": f"{type(ex).__name__}: {ex}"}
    except Exception as ex:                 # ours, not theirs -- say so
        return {"ok": False,
                "error": f"could not read the published orbit for {d}",
                "detail": f"{type(ex).__name__}: {ex}"}
    if not doc:
        return {"ok": False, "error": f"the MPC has no published orbit for {d}"}
    return dict(doc, ok=True)


# ====================================================================== queued fits
# Admission-controlled fitting for the hosted deploy.  `/api/fit` used to run the
# whole fit on the request thread, which fails two ways at once under a crowd: a
# hundred concurrent fits exhaust the container's memory (and Railway stops
# restarting after three OOMs), and a slow fit outlives the proxy's request
# limit.  So a web fit is now submitted to `fitqueue`, which bounds concurrency,
# coalesces identical work, and reports the caller's place in line.  See
# fitqueue's module docstring for why sessions are never shared.
import copy as _copy_mod      # noqa: E402
import functools as _ft       # noqa: E402
from pyroorb.web import fitqueue               # noqa: E402


def _fit_for_queue(desig, force):
    """One fit, as the worker pool runs it.

    Identical to :func:`run_fit` except for what happens to the interactive
    session it creates.  That session is the fit's PRISTINE TEMPLATE: it is
    lifted straight back out of ``_SESSIONS`` (so a hundred cached fits cannot
    evict a hundred real users' open sessions out of a table sized for twelve)
    and handed to fitqueue under ``_retain``, which parks it on the job entry.
    Clients get a ``restore`` token pointing at it and clone a private session
    from it only if they interact.
    """
    payload = run_fit(desig, force=force)
    if isinstance(payload, dict) and payload.get("ok") and payload.get("session"):
        sid = payload.pop("session")
        with _SES_LOCK:
            tmpl = _SESSIONS.pop(sid, None)
            try:
                _SES_ORDER.remove(sid)
            except ValueError:
                pass
        if tmpl is not None:
            payload["_retain"] = tmpl
    return payload


def run_fit_queued(desig, force=False, fresh=False):
    """Submit a fit and return its job handle (POST /api/fit_start).

    Input is validated HERE, before the queue, so a typo or a Pluto request is
    answered instantly instead of occupying a worker slot behind real work.

    ``fresh`` skips the disk cache for this call.  It is what a client sends
    when it wants to INTERACT with an object the disk tier answered: that
    payload carries no template, so only a real fit can produce the session,
    and without the flag the re-fit was handed the same cached answer forever.
    """
    desigs = parse_designations(desig)
    if not desigs:
        return {"ok": False, "error": "no designation given"}
    if len(desigs) > MAX_DESIGS:
        return {"ok": False, "error": f"too many designations ({len(desigs)}); the limit "
                f"is {MAX_DESIGS} — a longer list is probably a paste mistake"}
    try:
        desigs = [obs_sources.ensure_unpacked(d) for d in desigs]
    except Exception as ex:
        return {"ok": False, "error": f"could not read that designation: {ex}"}
    if any(guards.is_pluto(d) for d in desigs):
        return pluto_banned_payload()
    joined = ",".join(desigs)
    return fitqueue.submit(fitqueue.fit_key(desigs, force),
                           _ft.partial(_fit_for_queue, joined, force),
                           desig=joined, bypass_cache=bool(fresh),
                           max_age=obs_sources.fit_reuse_max_age(desigs))


def fit_job_status(job_id):
    """Poll a queued fit (GET /api/fit_result?id=)."""
    return fitqueue.status(job_id)


def session_from_job(job_id):
    """Clone a PRIVATE interactive session from a finished job's template
    (POST /api/session), so a shared fit can be played with by many people at
    once without any of them editing each other's copy.

    Cheap by design.  Nothing is re-fitted: ``_build_evaluator`` rebuilds the
    observer geometry and a fresh rebound base sim from the same observations,
    which is deterministic and costs about a second even on a large arc, while
    the solved state, mask, history and selectors are simply carried across.
    Finding the state is what a fit is expensive for; rebuilding the machinery
    around it is not.

    The template itself is READ-ONLY and stays that way -- every mutable thing
    (the evaluator, the history list, the per-model cache) is copied, and the
    read-mostly displayed fit is replaced the moment the client interacts.
    """
    tmpl = fitqueue.retained(job_id)
    if tmpl is None:
        return {"ok": False, "expired": True,
                "error": "that fit has expired -- re-run it to interact with it"}
    try:
        pure = tmpl["grav_pure"]
        gfit = tmpl["grav_fit"]
        obs = list(pure._obs)
        ctx_old = pure._ctx
        _ev, ctx = fo._build_evaluator(obs, _obscodes(), float(ctx_old["epoch_tdb"]))
        for k in ("desig", "meta", "seed_d", "method"):
            if k in ctx_old:
                ctx[k] = ctx_old[k]
        # Geometry (ra/dec/times/backend) IS a pure function of the obs and is safe to
        # rebuild -- but per-obs SIGMA is not. Arc extension's
        # _relax_overconfident_obscodes (arc_extend.py) permanently widens a whole
        # station's weighting/rejection sigma IN PLACE on the LIVE evaluator when that
        # station's reported precision looks over-confident for this object, and the
        # correction lives ONLY on that evaluator -- never written back to the
        # observation record. A fresh _build_evaluator call re-derives sigma from the
        # raw (still-tight) reported values and silently undoes the relaxation, so a
        # cloned session ends up fitting that station tighter than the original ever
        # did. Found on 2012 VP6's M57 tracklet (reported 0.12"-0.33", relaxed to
        # 0.5") -- invisible on a well-determined fit, but enough to walk a weakly-
        # determined Yarkovsky A2 to a measurably different converged value, so
        # re-selecting "yarkovsky" after a session restore silently stopped
        # reproducing the original auto-detected orbit. Carry the LIVE evaluator's
        # sigma arrays across verbatim instead of re-deriving them (same fix for the
        # phase-bridge link-relaxation state, the analogous in-place correction for a
        # multi-revolution link).
        old_ev = ctx_old["ev"]
        _ev.sig_ra = np.array(old_ev.sig_ra, copy=True)
        _ev.sig_dec = np.array(old_ev.sig_dec, copy=True)
        _ev.fit_sig_ra = np.array(old_ev.fit_sig_ra, copy=True)
        _ev.fit_sig_dec = np.array(old_ev.fit_sig_dec, copy=True)
        _ev._link_saved = (None if old_ev._link_saved is None
                           else tuple(np.array(a, copy=True) for a in old_ev._link_saved))
        _ev._link_relaxed_mask = (None if old_ev._link_relaxed_mask is None
                                  else np.array(old_ev._link_relaxed_mask, copy=True))
        _ev._resid_key = _ev._sub_key = None       # sigma changed -> drop the stale memo
        pure2 = _copy_mod.copy(pure)
        pure2._ctx = ctx
        if gfit is pure:
            gfit2 = pure2
        else:                                   # the asteroid-polished baseline
            ev_a, ctx_a = fo.with_force_model(ctx, True, gfit)
            ev_a.nongrav = None
            gfit2 = _copy_mod.copy(gfit)
            gfit2._ctx = ctx_a
    except Exception as ex:
        return {"ok": False, "error": f"could not restore that fit: "
                                      f"{type(ex).__name__}: {ex}"}
    ast_state = tmpl.get("ast_state")
    with _SES_LOCK:
        global _SES_COUNTER
        _SES_COUNTER += 1
        sid = f"s{_SES_COUNTER}-{secrets.token_urlsafe(18)}"
        _SESSIONS[sid] = {
            "ctx": gfit2._ctx, "history": list(tmpl.get("history") or []),
            "state": np.asarray(tmpl["state"], dtype=float).copy(),
            "grav_fit": gfit2, "grav_pure": pure2,
            "model": tmpl.get("model"), "constraints": dict(tmpl.get("constraints") or {}),
            "asteroids": bool(tmpl.get("asteroids")),
            "ast_state": (None if ast_state is None
                          else np.asarray(ast_state, dtype=float).copy()),
            # Read-mostly carry-overs from the template: the DISPLAYED orbit (which
            # may be an auto-detected nongrav fit) and the per-model cache that
            # makes re-selecting a model reproduce it.  The dict is copied so a
            # later write cannot reach the template; the fits inside it are only
            # ever read, and _apply_interactive replaces the displayed one with a
            # privately-built orbit on the first interaction.
            "fit": tmpl.get("fit"), "ng_cache": dict(tmpl.get("ng_cache") or {}),
            "epoch_tdb": float(tmpl.get("epoch_tdb", ctx["epoch_tdb"])),
        }
        pend = tmpl.get("itf_pending")
        if pend:
            _SESSIONS[sid]["itf_pending"] = pend
        _SES_ORDER.append(sid)
        while len(_SES_ORDER) > _SES_MAX:
            _SESSIONS.pop(_SES_ORDER.pop(0), None)
    return {"ok": True, "session": sid}


def fit_queue_stats():
    """Admission numbers for /healthz and the options UI."""
    return fitqueue.stats()
