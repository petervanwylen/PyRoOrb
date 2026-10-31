"""Paste-an-observation-blob loader for the web UI (WEB_UI_PLAN §2.7).

The read-only viewer fetches obs by designation (cache-first). This adds the other
half of the loader: let a user paste raw observations -- OBS80 (MPC 80-column) or
ADES (PSV/XML) -- and fit them directly, no MPC round-trip. Reuses the engine's
existing parsers (`obs_sources.parse_obs`, `ades.parse_ades_file`) and the same
`fit_orbit` -> `serialise_fit` path the designation fetch uses, so the front end
renders a pasted object identically to a fetched one.
"""
from pyroorb import guards
from pyroorb import obs_sources
from xml.etree import ElementTree as ET
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
import pyroorb.fit_object as fo  # noqa: E402
from pyroorb import ades  # noqa: E402
from pyroorb import arc_extend  # noqa: E402  (teaching-mode snapshot flag)

from pyroorb.web import engine_api  # noqa: E402  (serialise_fit reused verbatim)
from pyroorb.web import fitcache  # noqa: E402


def _looks_like_ades(text):
    """ADES is either XML (<ades.../<obsBlock...) or a pipe-separated-value table
    with a '|'-delimited header. A bare '|' is NOT a discriminator: OBS80's column 14
    (Note1, the program code) legitimately carries one, and a paste whose first line
    had it used to go to the ADES parser and come back empty. So the PSV test is the
    CLI's own (`obs_sources._detect_obs_format`: a '|' line naming an ADES field)."""
    if text.lstrip().startswith("<"):
        return True
    return obs_sources._detect_obs_format(text) == "ades"


def _parse(text, obscodes, radar_out=None):
    """Return (obs, fmt). Picks the ADES or OBS80 parser by sniffing the blob. When
    `radar_out` (a list) is passed, any radar (OBS80 'R'/'r' pairs or ADES radar records) the
    blob carries is collected into it -- the same channel the CLI file path uses
    (obs_sources.read_obs_file) so a PASTED radar object is radar-refined in the web exactly as
    on the CLI. Without it (the old behaviour) radar lines are silently dropped.

    This is the web's PARSING BOUNDARY, so it is where a malformed-input failure gets its
    name: the readers' input errors (bad XML from ElementTree, an unreadable number or
    timestamp from the field converters) are translated to obs_sources.ObsParseError, and
    the handlers below catch only that. Anything else -- a TypeError, an AttributeError --
    is a bug in our own code and propagates to the caller's error boundary instead of
    being dressed up as the user's bad paste."""
    try:
        if _looks_like_ades(text):
            obs = ades.parse_ades_file(text, obscodes, radar_out=radar_out)
            return obs, "ADES"
        obs = obs_sources.parse_obs(text, obscodes, radar_out=radar_out)
    except obs_sources.ObsParseError:
        raise
    except (ValueError, ET.ParseError) as ex:      # UnicodeDecodeError is a ValueError
        raise obs_sources.ObsParseError(f"{type(ex).__name__}: {ex}") from ex
    for o in obs:                                  # mirror fetch_combined's labelling
        o.setdefault("designation", obs_sources._unpack(o.get("packed", "")))
    return obs, "OBS80"


def fit_pasted(text, obscodes, force=False):
    """Parse a pasted obs blob and fit it. Returns the same payload shape as
    ``engine_api.run_fit`` (plus a ``fit_key`` the export endpoints can reuse), or
    an ``{ok: False, error}`` dict the front end shows."""
    if not text or not text.strip():
        return {"ok": False, "error": "no observations pasted"}
    obs_sources.pop_unknown_obscodes()       # reset so fit_obs_list surfaces only this blob's unknowns
    radar_rows = []
    try:
        obs, fmt = _parse(text, obscodes, radar_out=radar_rows)
    except obs_sources.ObsParseError as ex:        # malformed input -> the user's problem
        return {"ok": False, "error": f"could not parse pasted obs: {ex}"}
    if not obs:
        return {"ok": False, "error": "no usable observations found in the pasted text",
                "hint": "expected MPC 80-column (OBS80) or ADES (PSV/XML)"}

    present = sorted({o.get("designation") for o in obs if o.get("designation")})
    # Pluto ban: refuse pasted Pluto astrometry (by per-obs designation/packed number or
    # name). It is a DE440 perturber and can't be fit unperturbed by itself.
    if any(guards.is_pluto(c) for c in guards._pluto_candidates(obs=obs)) or any(guards.is_pluto(d) for d in present):
        return engine_api.pluto_banned_payload()
    meta = {
        "designation": present[0] if present else "pasted object",
        "designations": present or ["pasted object"],
        "combined": len(present) > 1,
        "other_designations": present[1:],
        "format": fmt,
    }
    if radar_rows:                                # blob-carried radar -> fit_orbit folds it in (CLI parity)
        meta["radar"] = radar_rows
    try:
        arc_extend.SNAPSHOTS = True               # so the fit-history replay works on pasted objects too
        fit = fo.fit_orbit(obs, obscodes, desig=meta["designation"], meta=meta,
                           verbose=False, force=force,
                           mpc_seed=fo.fetch_mpc_seed(obs, meta))
    except guards.PlutoBanError:
        return engine_api.pluto_banned_payload()
    except Exception as ex:
        return {"ok": False, "error": f"fit failed: {type(ex).__name__}: {ex}"}
    finally:
        arc_extend.SNAPSHOTS = False

    payload = engine_api.serialise_fit(fit)
    payload["meta"] = {"format": fmt, "n_input_designations": len(present),
                       "source": "pasted"}
    payload["fit_key"] = fitcache.put(fit)
    # Open an interactive session like run_fit does, so a PASTED object supports Full step,
    # nongrav, constrain and the asteroid toggle too (without it the UI had a fit but no
    # session, so those re-fit actions silently did nothing).
    payload["session"] = engine_api._store_session(fit)
    return payload


# De-duplication when merging a paste into an existing arc reuses the fetch path's
# rule (obs_sources.dedup_observations): same site, ~same instant, ~same sky
# position, by tolerance rather than rounded key so a near-coincident pair never
# splits across a bucket boundary.
_dedup_index = obs_sources._dup_index


def _is_duplicate(o, idx):
    return obs_sources.find_duplicate(o, idx) is not None


def add_to_fit(text, obscodes, session=None, fit_key=None, force=False):
    """Append a pasted OBS80/ADES blob to the CURRENTLY-displayed fit and re-fit the
    union. Resolves the open fit's observations (by live ``session`` for a fetched
    object, or ``fit_key`` for a pasted one), parses the new blob, merges + de-dupes,
    and runs the same full fit path ``run_fit`` uses (asteroid polish, auto-nongrav,
    a fresh interactive session). Returns that payload, or an ``{ok: False}`` dict."""
    if not text or not text.strip():
        return {"ok": False, "error": "no observations pasted"}
    base_obs, err = engine_api.base_obs_for(session=session, fit_key=fit_key)
    if err:
        return {"ok": False, "error": err}
    radar_rows = []
    try:
        new_obs, fmt = _parse(text, obscodes, radar_out=radar_rows)
    except obs_sources.ObsParseError as ex:        # malformed input -> the user's problem
        return {"ok": False, "error": f"could not parse pasted obs: {ex}"}
    if not new_obs:
        return {"ok": False, "error": "no usable observations found in the pasted text",
                "hint": "expected MPC 80-column (OBS80) or ADES (PSV/XML)"}

    idx = _dedup_index(base_obs)
    added = []
    for o in new_obs:
        if _is_duplicate(o, idx):                 # already in the fit -> skip
            continue
        added.append(o)
        jd = float(o.get("jd_utc") or 0.0)        # let it dedup later pasted copies too
        idx.setdefault((o.get("obscode"), int(jd)), []).append(o)
    if not added:
        return {"ok": False, "error": "every pasted observation is already in the current fit"}
    # Restore the chronological ordering every other entry path guarantees (parse_obs /
    # parse_ades_file / fetch_combined all sort by jd_utc). The appended `added` obs are
    # each sorted, but base+added is not -- and downstream code (arc_length_days, the obs
    # table, the O-C arc-squeeze) assumes obs are in time order. Sort so an added night
    # that falls inside or before the existing arc doesn't strand the arc display.
    combined = sorted(list(base_obs) + added, key=lambda o: o["jd_utc"])

    present = sorted({o.get("designation") for o in combined if o.get("designation")})
    meta = {
        "designation": present[0] if present else "pasted object",
        "designations": present or ["pasted object"],
        "combined": len(present) > 1,
        "other_designations": present[1:],
        "format": fmt,
    }
    if radar_rows:                                # radar carried by the ADDED blob rides the union fit
        meta["radar"] = radar_rows
    # fit_obs_list manages the per-stage snapshot flag itself (so the fit-history
    # replay works on the combined arc too) -- no SNAPSHOTS wrapper needed here.
    payload = engine_api.fit_obs_list(
        combined, meta, obscodes, force=force,
        meta_extra={"format": fmt, "n_input_designations": len(present),
                    "source": "added", "n_added": len(added)})
    # Export must reproduce the EXACT union (re-fitting by designation would re-fetch
    # from the cache and drop the pasted obs), so cache the combined fit and hand the
    # front end its key -- the same handle a freshly-pasted fit exports through.
    if payload.get("ok", True):
        sid = payload.get("session")
        s = engine_api._SESSIONS.get(sid) if sid else None
        if s is not None and s.get("fit") is not None:
            payload["fit_key"] = fitcache.put(s["fit"])
    return payload
