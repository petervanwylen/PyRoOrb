"""ADES astrometry ingest — the modern replacement for 80-column OBS80.

Two entry points:
  - `fetch_ades(desig)`  : pull ADES from the MPC observations API (`ADES_DF`,
                           structured JSON) — carries per-obs rmsRA/rmsDec, the
                           star catalog (astCat) and observing mode.
  - `parse_ades_file(s)` : parse a user-supplied ADES PSV or XML string natively
                           (no third-party ADES dependency).

Both yield our internal obs dicts, each tagged with a per-observation 1-sigma
uncertainty (`sigma_ra`, `sigma_dec`, arcsec) from ADES rms when present, else the
Veres-2017 station lookup (`weights.sigma_for`). Those sigmas drive the NORMALIZED
residuals the fitter and arc-extension now minimize (find_orb-style).

PROVENANCE: ADES is the IAU astrometry standard; we read it from the MPC API and
parse user files natively (PSV logic ported from adam_core's ADES_string_to_tables,
B612, MIT; no copied source). Designation packing is native (packing.py).
"""
import json, ssl, sys, urllib.request, urllib.parse
from pathlib import Path
from xml.etree import ElementTree as ET
from astropy.time import Time

from pyroorb.weights import sigma_for, ades_rms_usable
from pyroorb.obs80 import (geodetic_to_itrs_au, pack_designation, cache_stem, cache_fresh,
                   astrometry_ttl, NEOCP_CACHE_TTL_S)
from pyroorb import datadirs

OBS_API = "https://data.minorplanetcenter.net/api/get-obs"
# NEOCP (NEO Confirmation Page) tracklets are NOT yet in the published-obs API
# (get-obs returns HTTP 500). Their astrometry lives behind this CGI, which serves
# ADES XML when given xml=y -- the same source find_orb pulls unconfirmed objects
# from. A NEOCP/temporary designation is short (<=7 chars) and space-free.
NEOCP_OBS_CGI = "https://cgi.minorplanetcenter.net/cgi-bin/cgipy/showobsorbs.cgi"
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE     # MPC chain not always present in sandbox

KM_PER_AU = 1.495978707e8            # IAU 2012 astronomical unit (km)


def _sat_geocentric_au(r):
    """Geocentric observer position for a SPACE-BASED ADES record as ``(xyz_au, units)``, or None
    when the record is not a usable satellite obs. ADES gives the spacecraft position as
    sys=ICRF_KM (or ICRF_AU), ctr=399 (Earth geocenter), pos1/2/3; we return [x,y,z] in au. The
    fit then forms the observer position as earth_center + this offset (observer_positions_batch),
    since a spacecraft has no obscode ground position. ``units`` ('km'/'au') is the ADES-native
    unit, kept so obs80.format_obs80 can round-trip the 's' companion line in the same units the
    submitter used (km can't hold a distant L2 spacecraft; au can). Radar/other frames -> None."""
    sysv = (r.get("sys") or "").strip().upper()
    if not sysv.startswith("ICRF"):
        return None
    try:
        if int(float(r.get("ctr"))) != 399:        # only geocentric satellite vectors handled
            return None
        p = [float(r["pos1"]), float(r["pos2"]), float(r["pos3"])]
    except (TypeError, ValueError, KeyError):
        return None
    is_au = sysv.endswith("AU")
    scale = 1.0 if is_au else (1.0 / KM_PER_AU)     # ICRF_KM -> au; ICRF_AU as-is
    return [p[0] * scale, p[1] * scale, p[2] * scale], ("au" if is_au else "km")


def _rov_geodetic_au(r):
    """Geocentric position for a ROVING-observer ADES record as ``(itrs_au, (lon, lat, alt))``,
    or None. ADES gives the mobile site as sys=WGS84, ctr=399, pos1 = E-longitude (deg),
    pos2 = latitude (deg), pos3 = altitude (m) -- the same geodetic triple as the OBS80 'v' line.
    ``itrs_au`` is the Earth-FIXED ITRS vector the fit rotates ITRS->GCRS at the obs time like a
    fixed station (observer_positions_batch site_itrs); the raw (lon, lat, alt) is kept so
    obs80.format_obs80 can write the 'v' companion line verbatim. ADES analogue of
    obs80.mpc80_roving_itrs_au; e.g. 2006 OE10's code-247 observations."""
    if (r.get("sys") or "").strip().upper() != "WGS84":
        return None
    try:
        if int(float(r.get("ctr"))) != 399:
            return None
        lon, lat, alt = float(r["pos1"]), float(r["pos2"]), float(r["pos3"])
    except (TypeError, ValueError, KeyError):
        return None
    return geodetic_to_itrs_au(lon, lat, alt), (lon, lat, alt)


def fetch_ades(desig, cache=None):
    """GET ADES_DF records (list of dicts) for one designation from the MPC API."""
    if cache_fresh(cache, astrometry_ttl(desig)):
        try:
            return json.loads(Path(cache).read_text())
        except (json.JSONDecodeError, ValueError):
            # Corrupt/truncated cache (the sandbox FS can truncate large writes) ->
            # discard it and re-fetch fresh rather than failing on every retry.
            Path(cache).unlink()
    body = json.dumps({"desigs": [desig], "output_format": ["ADES_DF"]}).encode()
    req = urllib.request.Request(OBS_API, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="GET")
    raw = json.loads(urllib.request.urlopen(req, context=_CTX, timeout=90).read())
    recs = raw[0]["ADES_DF"]
    if isinstance(recs, str):
        recs = json.loads(recs)
    if cache:
        datadirs.write_cache(cache, json.dumps(recs))
    return recs


def _designation(rec):
    """Unpacked designation for a record: provisional preferred, else number."""
    d = rec.get("provid") or rec.get("permid") or rec.get("trksub")
    return str(d).strip() if d else None


def _jd_utc(obstime):
    # ADES obsTime is ISO-8601 UTC, e.g. '2014-09-28T08:40:58.656Z'
    return float(Time(obstime.replace("Z", ""), format="isot", scale="utc").jd)


def _reported(v):
    """True when an ADES field carries a value at all.

    Deliberately NOT a truthiness test. `ra` and `dec` arrive from the API as numbers,
    and 0.0 is a perfectly good measurement -- 0 deg RA is the meridian the coordinate
    system starts at, 0 deg Dec is the celestial equator -- so `if not ra` silently
    DISCARDED any observation that landed on either, with no warning and no trace. The
    string paths (PSV/XML) reach here with "" for an absent field, hence the strip."""
    return v is not None and str(v).strip() != ""


def records_to_obs(records, obscodes):
    """Convert ADES_DF records to internal obs dicts with per-obs sigmas.

    Skips radar/satellite/roving and anything without a usable optical RA/Dec at a
    known ground station."""
    obs = []
    for r in records:
        # A deprecated/deleted obs is dropped. The ADES `deprecated` flag is a presence
        # marker: any non-empty value (MPC uses "X"; "1"/"true"/"Y" also seen) means dead.
        # Such a record is often a stub missing its position vector -- a deleted SATELLITE
        # obs (e.g. C51) arrives with sys/pos1-3 null, so it must be skipped HERE or it
        # falls through to the unknown-ground-station path and spuriously warns "C51".
        if str(r.get("deprecated") or "").strip().lower() not in ("", "0", "false", "n", "no"):
            continue                                  # MPC-flagged dead obs
        stn = (r.get("stn") or "").strip()
        ra, dec, ot = r.get("ra"), r.get("dec"), r.get("obstime")
        if not (stn and _reported(ra) and _reported(dec) and _reported(ot)):
            continue                                  # need optical RA/Dec + time
        # Observers with no fixed obscode position: a SPACE-BASED obs (C51/WISE: sys=ICRF, a
        # geocentric satellite vector) or a ROVING observer (247: sys=WGS84, geodetic lon/lat/alt).
        # Accept either via its ADES position; otherwise it must be a known ground station.
        sat = _sat_geocentric_au(r)
        rov = _rov_geodetic_au(r)
        sat_pos = sat[0] if sat else None
        site_itrs = rov[0] if rov else None
        if sat_pos is None and site_itrs is None and stn not in obscodes:
            # Unknown ground station -> ObsCodes.htm may be stale; record so the caller
            # can prompt. Lazy import avoids a fit_object<->ades import cycle.
            from pyroorb import obs_sources
            obs_sources.record_unknown_obscode(stn)
            continue
        desig = _designation(r)
        if not desig:
            continue
        packed = pack_designation(desig)             # comet-aware native packer ('P/...' etc.)
        jd = _jd_utc(ot)                             # parsed ONCE; sigma_for's era floor and the
        sra, sdec = sigma_for(stn, r.get("astcat"),  # row both need it
                              r.get("rmsra"), r.get("rmsdec"), jd_utc=jd)
        def _f(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None
        o = {
            "jd_utc": jd,
            "ra_deg": float(ra), "dec_deg": float(dec),
            "obscode": stn, "packed": packed, "designation": desig,
            # The PERMANENT number, when the object has one. `packed` above follows
            # the engine's provisional-first identity; the 80-column record has a
            # separate 5-column field for the number (cols 1-5), so keep it apart.
            "packed_perm": pack_designation(str(r.get("permid")).strip())
                           if str(r.get("permid") or "").strip() else None,
            "sigma_ra": float(sra), "sigma_dec": float(sdec),
            "astcat": r.get("astcat"), "mode": r.get("mode"),
            # 80-column provenance bytes ADES keeps in named fields: the note /
            # programme code (col 14), the publication reference (cols 73-77) and
            # the REPORTED precision of time/RA/Dec, so a rendered 80-col line
            # carries the submitter's own digits rather than invented ones.
            "notes": r.get("notes"), "prog": r.get("prog"), "ref": r.get("ref"),
            "nucmag": _f(r.get("nucmag")),          # comet: mag is nuclear (flag)
            "prec_time": r.get("prectime"), "prec_ra": r.get("precra"),
            "prec_dec": r.get("precdec"),
            # Label the SOURCE of the sigma above with the same predicate sigma_for used
            # to choose it, so the two can't disagree (a reported rms of 0 is not usable
            # -- see weights.ades_rms_usable -- so such an obs is honestly Veres2017).
            "uncert_src": "ADES" if ades_rms_usable(r.get("rmsra"), r.get("rmsdec")) else "Veres2017",
            "sigma_time_sec": _f(r.get("rmstime")),    # ADES rmsTime (s) for timing-error smear

            # photometry: reported magnitude, filter band, and (ADES) mag sigma
            "mag": _f(r.get("mag")), "band": (r.get("band") or " "),
            "mag_sigma": _f(r.get("rmsmag")),

            # Display-only ADES provenance for the web UI's "addl info" columns.
            # The observation identifier and the tracklet-submission id (trkSub) are
            # otherwise consumed only to derive the designation; the ADES-REPORTED
            # RA/Dec sigmas (as opposed to the possibly-Veres2017 weighting sigmas
            # above) and the log-SNR are carried through verbatim. None when the
            # submitter didn't populate the field; absent entirely for OBS80.
            "obs_id": (str(r.get("obsid")).strip() or None) if r.get("obsid") else None,
            "trksub": (str(r.get("trksub")).strip() or None) if r.get("trksub") else None,
            # ADES discovery flag (disc="*") -> the OBS80 col-13 asterisk when
            # this obs is rendered (format_obs80); the identification pipeline
            # needs it to stamp a NEOCP object by its DISCOVERY tracklet.
            "discovery": str(r.get("disc") or "").strip() == "*" or None,
            "ades_rms_ra": _f(r.get("rmsra")),      # arcsec (already RA*cos(dec))
            "ades_rms_dec": _f(r.get("rmsdec")),    # arcsec
            "log_snr": _f(r.get("logsnr")),
        }
        if sat_pos is not None:
            o["sat_pos_au"] = sat_pos                  # inertial geocentric equatorial au (space)
            o["sat_units"] = sat[1]                    # ADES-native 'km'/'au' for the 's' line
        if site_itrs is not None:
            o["site_itrs_au"] = site_itrs              # Earth-fixed geocentric ITRS au (roving)
            o["site_geodetic"] = rov[1]                # (lon, lat, alt) deg/deg/m for the 'v' line
        obs.append(o)
    obs.sort(key=lambda o: o["jd_utc"])
    # Star-catalog debiasing (EFCC18): subtract per-catalog systematic offsets so
    # all observations are on the Gaia-DR2 frame before fitting. No-op if the bias
    # table isn't present, or per-obs for catalogs already on Gaia-DR2/DR3.
    try:
        from pyroorb import debias
        debias.debias_obs(obs)
    except Exception as ex:
        # A missing bias table is handled INSIDE debias_obs (a clean no-op), so reaching
        # here means the debias step itself broke -- and silently shipping raw positions
        # would quietly reintroduce the catalog bias this call exists to remove.
        print(f"[debias] step failed, positions left raw: {type(ex).__name__}: {ex}",
              file=sys.stderr)
    return obs


def fetch_combined_ades(desigs, obscodes, cache_dir=None):
    """ADES analogue of obs_sources.fetch_combined: fetch + merge one or more
    designations into a single combined object. Returns (obs, meta)."""
    if isinstance(desigs, str):
        desigs = [desigs]
    cache_dir = Path(cache_dir) if cache_dir else None
    records = []
    for d in desigs:
        cache = (cache_dir / f"ades_{cache_stem(d)}.json") if cache_dir else None
        records += fetch_ades(d, cache=cache)
    obs = records_to_obs(records, obscodes)
    present = sorted({o["designation"] for o in obs})
    meta = {"designation": present[0] if present else (desigs[0]),
            "designations": present,
            "combined": len(present) > 1,
            "n_requested": len(desigs),     # how many desigs the CALLER asked for (see fit_object)
            "other_designations": present[1:],
            "format": "ADES"}
    return obs, meta


def is_neocp_desig(desig):
    """Heuristic for a NEOCP / temporary tracklet designation: short and space-free.

    Published objects are numbers ('433'), packed ('K16UL0A') or unpacked provisional
    designations with a space ('2016 UA210'). A NEOCP temp id ('8N62001', 'P21abcd')
    is <=7 characters and contains no space, so it cannot be an unpacked designation;
    that's the same shape find_orb treats as 'fetch from the NEOCP CGI'."""
    s = str(desig).strip()
    return bool(s) and " " not in s and len(s) <= 7


def _neocp_xml_to_records(xml_text):
    """Parse showobsorbs.cgi ADES XML into record dicts keyed like ADES_DF.

    Each <optical> element's children (trkSub/stn/obsTime/ra/dec/rmsRA/...) become a
    dict with lower-cased tag names, which is exactly what records_to_obs() consumes
    (it reads stn/ra/dec/obstime/rmsra/rmsdec/astcat/mode/mag/band/rmsmag and the
    sys/ctr/pos1.. satellite/roving keys). Non-optical (<radar>) records are skipped."""
    root = ET.fromstring(xml_text.strip())
    recs = []
    for opt in root.iter("optical"):
        recs.append({child.tag.lower(): (child.text or "").strip() for child in opt})
    return recs


def fetch_neocp_xml(desig, cache=None):
    """GET the raw ADES XML for one NEOCP/temporary designation (cached as text)."""
    if cache_fresh(cache, NEOCP_CACHE_TTL_S):
        txt = Path(cache).read_text()
        if txt.strip():
            return txt
        Path(cache).unlink()                 # empty/truncated cache -> re-fetch
    url = NEOCP_OBS_CGI + "?" + urllib.parse.urlencode({"Obj": desig, "xml": "y"})
    req = urllib.request.Request(url, method="GET")
    txt = urllib.request.urlopen(req, context=_CTX, timeout=90).read().decode("utf-8", "replace")
    if "<optical" not in txt and "<ades" not in txt:
        raise ValueError(f"NEOCP CGI returned no ADES for {desig!r}")
    if cache:
        datadirs.write_cache(cache, txt)
    return txt


def fetch_combined_neocp(desigs, obscodes, cache_dir=None):
    """NEOCP analogue of fetch_combined_ades: fetch + merge temporary tracklets via
    the NEOCP CGI (ADES XML). Returns (obs, meta).

    The identity of every row is the designation the CALLER asked for, not the
    <trkSub> the row happens to carry. A NEOCP object is a set of tracklets the MPC
    has already decided are one body, and each was submitted under its OWN observer's
    tracklet id: P12pLhW comes back as four rows labelled "P12pLhW" and three
    labelled "X90166". Taking <trkSub> as the designation therefore split one
    unconfirmed object into two, which is wrong twice over -- the fit is presented as
    a combined/linked object (``combined``, ``other_designations``, and the linkage
    surfaces that read them) when nothing was linked, and the 80-column lines we
    render, submit and count nights on carry a name the MPC's own listing for that
    object does not use (showobsorbs.cgi?Obj=P12pLhW&obs=y labels all seven P12pLhW).
    So a row whose designation could only have come from <trkSub> -- no provID and no
    permID; a NEOCP object that has since been designated keeps that published name
    -- is relabelled to the requested id. The submitter's own trkSub is untouched on
    the obs (``o["trksub"]``, the web UI's provenance column): renaming the object is
    not forgetting who reported it."""
    if isinstance(desigs, str):
        desigs = [desigs]
    cache_dir = Path(cache_dir) if cache_dir else None
    records, asked_for = [], {}
    for d in desigs:
        cache = (cache_dir / f"neocp_{cache_stem(d)}.xml") if cache_dir else None
        for r in _neocp_xml_to_records(fetch_neocp_xml(d, cache=cache)):
            trksub = str(r.get("trksub") or "").strip()
            published = str(r.get("provid") or "").strip() or str(r.get("permid") or "").strip()
            if trksub and not published:
                asked_for.setdefault(trksub, d)   # first requester wins; deterministic
            records.append(r)
    obs = records_to_obs(records, obscodes)
    renamed = {}
    for o in obs:
        o["neocp"] = True            # render flag: stamp 'NEOCP' in the 80-col reference field
        # Relabel only where the designation IS the trkSub -- records_to_obs prefers
        # provID/permID, so a designated object is never renamed by this.
        want = asked_for.get(o.get("trksub"))
        if want and o["designation"] == o["trksub"] and o["designation"] != want:
            renamed[o["designation"]] = want
            o["designation"] = want
            o["packed"] = pack_designation(want)
    for was, now in sorted(renamed.items()):
        print(f"(NEOCP: {was} is a submitter tracklet id for {now}; labelled {now})")
    present = sorted({o["designation"] for o in obs})
    meta = {"designation": present[0] if present else (desigs[0]),
            "designations": present,
            "combined": len(present) > 1,
            "n_requested": len(desigs),     # how many desigs the CALLER asked for (see fit_object)
            "other_designations": present[1:],
            "format": "ADES", "source": "NEOCP"}
    return obs, meta


def _psv_to_records(psv_text):
    """Parse an ADES PSV (pipe-separated-value) string into record dicts keyed like
    ADES_DF, mirroring _neocp_xml_to_records' output so both feed records_to_obs.

    The PSV form is a '|'-delimited table: a header row naming the fields (permID,
    provID, trkSub, stn, obsTime, ra, dec, rmsRA, rmsDec, astCat, mode, mag, band,
    rmsTime, ...) followed by data rows, optionally preceded/interleaved by '#'/'!'
    metadata lines (version, observatory context). Field names are lower-cased so the
    record keys match what records_to_obs reads (stn/ra/dec/obstime/rmsra/rmsdec/...
    and the sys/ctr/pos1-3 satellite/roving keys); ADES's rmsRA (which is already
    RA*cos(dec)) lower-cases to 'rmsra', the same key the XML path produces.

    A single blob may carry more than one header block (e.g. concatenated tracklets
    with different column sets); each header line resets the active column layout, so
    rows are always matched against their own header. Logic ported from adam_core's
    observations/ades.py::ADES_string_to_tables (B612, MIT) -- the metadata/obsContext
    sections it also returns are unused here and dropped."""
    headers = []
    recs = []
    for raw in psv_text.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue                              # blank or '#'/'!' metadata line
        # A header line names the designation columns; reset the active layout.
        if "permID" in line or "provID" in line or "trkSub" in line:
            headers = [h.strip().lower() for h in line.split("|")]
            continue
        if not headers:
            continue                              # data before any header -> nothing to key on
        values = [v.strip() for v in line.split("|")]
        recs.append(dict(zip(headers, values)))   # zip drops any unheadered trailing fields
    return recs


def parse_ades_file(text, obscodes, radar_out=None):
    """Parse a user-supplied ADES PSV or XML string, then convert to internal obs.
    Lets us ingest ADES files, not just the MPC API.

    Both flavours are parsed natively here (no third-party ADES dependency): XML via
    _neocp_xml_to_records (adam_core's PSV parser would mangle XML, treating
    '<provID>...</provID>' as a header line) and the '|'-delimited PSV via
    _psv_to_records. Both yield ADES_DF-shaped, lower-cased record dicts that
    records_to_obs converts to internal obs. ADES *radar* records (delay/doppler +
    trx/rcv, no optical RA/Dec) are collected into `radar_out` when given
    (radar.ades_records_to_radar), else skipped as before."""
    low = text.lstrip()[:8000].lower()
    if low.startswith("<?xml") or "<ades" in low or "<optical" in low:
        records = _neocp_xml_to_records(text)
    else:
        records = _psv_to_records(text)
    if radar_out is not None:
        from pyroorb.radar import ades_records_to_radar
        radar_out.extend(ades_records_to_radar(records, obscodes))
    return records_to_obs(records, obscodes)
