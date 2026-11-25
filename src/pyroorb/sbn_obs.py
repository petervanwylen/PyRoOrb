"""Published astrometry from the SBN replica of the MPC database.

The fallback for when the MPC Observation API does not answer: the same rows,
read from ``obs_sbn``, rendered as the API's ``ADES_DF`` records so that
:func:`ades.records_to_obs` treats them exactly as it treats an API response.

WHICH rows make up one object.  A designation names a FAMILY:
``current_identifications`` maps every provisional designation (the primary
listed as its own secondary) to the family's primary, and
``numbered_identifications`` maps a primary to its number.  A numbered object's
observations are filed mostly under ``permid`` with ``provid`` left NULL --
(433) Eros has 18,732 rows by permid against 567 carrying a provisional
designation -- so a lookup by provisional designations alone loses nearly all
of a numbered object.  The family's rows are therefore the union of
``permid = <number>`` and ``provid = ANY(<every provisional designation>)``.

WHICH statuses.  The API serves ``P`` (published) and ``p`` (accepted,
publication pending) and nothing else: for Eros it returns exactly its 18,679
``P`` rows and none of its 53 ``I`` rows.  ``deprecated`` rows are returned
like the API returns them and dropped by ``records_to_obs``.

Measured against 198 cached API responses (10,477 observations): every row
the API served for a family is here, except where the replica is newer than
the cached response (rows filed under the family, or withdrawn, since).

Every lookup goes through the packed-designation, ``permid`` and ``provid``
indexes; the unpacked designation columns of ``current_identifications`` are
not indexed.
"""

import re
from decimal import Decimal

# The API's ADES_DF fields, which are obs_sbn column names.
ADES_COLUMNS = (
    "artsat", "astcat", "band", "com", "ctr", "dec", "decstar", "delay",
    "deltadec", "deltara", "deprecated", "disc", "dist", "doppler", "exp",
    "fltr", "frq", "localuse", "logsnr", "mag", "mode", "notes", "nstars",
    "nucmag", "obscenter", "obsid", "obssubid", "obstime", "pa", "permid",
    "photap", "photcat", "pos1", "pos2", "pos3", "poscov11", "poscov12",
    "poscov13", "poscov22", "poscov23", "poscov33", "precdec", "precra",
    "prectime", "prog", "provid", "ra", "rastar", "rcv", "ref", "remarks",
    "rmscorr", "rmsdec", "rmsdelay", "rmsdist", "rmsdoppler", "rmsfit",
    "rmsmag", "rmspa", "rmsra", "rmstime", "seeing", "shapeocc", "stn", "subfmt",
    "subfrm", "sys", "trkid", "trkmpc", "trksub", "trx", "unctime", "vel1",
    "vel2", "vel3")
PUBLISHED_STATUSES = ("P", "p")
# A number: an asteroid's ("433"), a numbered comet's ("73P"), or a numbered comet's FRAGMENT
# ("73P-C", "73P-BV"), which obs_sbn files under its own permid (73P-C: 3525 rows; the
# parent "73P" holds none of them). A provisional comet's fragment ("C/2025 K1-B") is filed
# under provid and resolves through the provisional branch.
_NUMBER = re.compile(r"\d+(?:[PDI](?:-[A-Z]{1,2})?)?")


def resolve_family(cur, desig):
    """``(primary, permid, provids)`` for a number, IAU name, or provisional
    designation (primary or secondary; packed or not): the family's primary provisional
    designation (None for a number without one), its number (None if
    unnumbered) and every provisional designation filed under it.

    Numbered comets and interstellar objects (``2P``, ``354P``, ``1I``) are
    numbers too: ``numbered_identifications`` carries them, and their
    observations are filed under that ``permid``.  Comet designations do not
    pack, so a primary's number is looked up by its unpacked spelling (also a
    unique index)."""
    from pyroorb.packing import pack, unpack
    s = str(desig).strip()
    s = unpack(s) or s                      # a packed designation or number
    permid = primary = None
    if _NUMBER.fullmatch(s):
        permid = s
    else:
        cur.execute("select permid, unpacked_primary_provisional_designation "
                    "from numbered_identifications where iau_name = %s", (s,))
        row = cur.fetchone()
        if row:
            permid, primary = row
    if permid and not primary:
        cur.execute("select unpacked_primary_provisional_designation "
                    "from numbered_identifications where permid = %s", (permid,))
        row = cur.fetchone()
        primary = row[0] if row else None
    if not permid:
        p = pack(s)
        if p:
            cur.execute("select unpacked_primary_provisional_designation from "
                        "current_identifications where packed_secondary_provisional_designation = %s",
                        (p,))
            row = cur.fetchone()
            primary = row[0] if row else s
        else:
            primary = s
        cur.execute("select permid from numbered_identifications "
                    "where unpacked_primary_provisional_designation = %s", (primary,))
        row = cur.fetchone()
        permid = row[0] if row else None
    provids = [primary] if primary else []
    if primary and pack(primary):
        cur.execute("select unpacked_secondary_provisional_designation from current_identifications "
                    "where packed_primary_provisional_designation = %s", (pack(primary),))
        provids = sorted({primary, *(r[0] for r in cur.fetchall() if r[0])})
    return primary, permid, provids


def _api_value(v):
    """An obs_sbn value as the API serves it: numerics as their decimal text,
    a blank string as null."""
    if isinstance(v, Decimal):
        return str(v)
    return None if isinstance(v, str) and not v.strip() else v


def family_records(cur, desig):
    """The published rows of ``desig``'s family as API ``ADES_DF`` records,
    in observation-time order."""
    _primary, permid, provids = resolve_family(cur, desig)
    if not permid and not provids:
        return []
    cols = ", ".join(c for c in ADES_COLUMNS if c != "obstime")
    cur.execute(f"select {cols}, obstime, obstime_text from obs_sbn "
                "where (provid = any(%s) or permid = %s) and status = any(%s)",
                (list(provids), permid or "", list(PUBLISHED_STATUSES)))
    return _as_records(cur)


def _as_records(cur):
    """The rows of an executed ``select <ADES cols>, obstime, obstime_text`` as API
    ``ADES_DF`` records, in observation-time order."""
    names = [d[0] for d in cur.description]
    out = []
    for row in cur.fetchall():
        r = dict(zip(names, row))
        t, text = r.pop("obstime"), r.pop("obstime_text")
        r = {k: _api_value(v) for k, v in r.items()}
        r["obstime"] = text or (t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                                if t is not None else None)
        r["Obstype"] = "optical"
        out.append(r)
    out.sort(key=lambda r: r["obstime"] or "")
    return out


# --- a submitter's trksub (an ITF tracklet, say) typed in place of a designation ----------
#
# A trksub is the SUBMITTER's label, not the MPC's: it is unique only within one station's
# submissions, and not even reliably there, so one string can name several tracklets
# (``gb00884`` is a designated 2024 HP tracklet from G19 while it is also a live NEOCP id).
# A lookup therefore answers with TRACKLETS -- one per ``trkid`` -- and the caller decides:
# exactly one undesignated tracklet is fetched, several are a choice for the user, and one
# that already carries a designation is that designation (it is no longer a trksub's
# business).  Statuses: ``I`` (isolated -- the ITF) plus the published ``P``/``p``, which
# are how a designated tracklet is recognised; nothing else is served.
TRKSUB_STATUSES = ("I", "P", "p")
# "H005311@568" or "H005311@568/2015-01-26": the trksub narrowed by station (and the UTC
# date of the tracklet's first observation) -- the form the web's tracklet picker sends back.
_QUALIFIED = re.compile(r"^([^@\s]{1,7})@([0-9A-Za-z]{3})(?:/(\d{4})-?(\d\d)-?(\d\d))?$")


def split_trksub(token):
    """``(trksub, station, date)`` of a possibly-qualified trksub token, or None when the
    token is not one (``date`` is ``"YYYY-MM-DD"`` or None, ``station`` None when bare).
    A bare token is any space-free id of 1-7 characters; WHETHER it is a trksub rather
    than a designation is the caller's question (obs_sources.is_published_designation)."""
    s = str(token or "").strip()
    m = _QUALIFIED.match(s)
    if m:
        date = f"{m.group(3)}-{m.group(4)}-{m.group(5)}" if m.group(3) else None
        return m.group(1), m.group(2).upper(), date
    if s and " " not in s and "@" not in s and len(s) <= 7:
        return s, None, None
    return None


def qualified_token(t):
    """The token naming exactly one tracklet of :func:`trksub_tracklets`: trksub@stn/date."""
    return f"{t['trksub']}@{t['stn']}/{t['date']}"


def trksub_tracklets(cur, token):
    """The tracklets a (possibly qualified) trksub token names: one dict per ``trkid``,
    ordered by first observation -- ``trksub, stn, trkid, date`` (UTC date of the first
    observation), ``first``/``last`` (ISO), ``n_obs`` and ``designation`` (the provisional
    designation or number the tracklet is now filed under, None for an isolated one).
    ``[]`` when the token is not a trksub shape or nothing matches."""
    parts = split_trksub(token)
    if not parts:
        return []
    trksub, stn, date = parts
    where, params = ["trksub = %s", "status = any(%s)"], [trksub, list(TRKSUB_STATUSES)]
    if stn:
        where.append("stn = %s")
        params.append(stn)
    cur.execute("select trkid, stn, min(obstime), max(obstime), count(*), "
                "max(provid), max(permid) from obs_sbn where " + " and ".join(where)
                + " group by trkid, stn order by min(obstime)", tuple(params))
    out = []
    for trkid, st, first, last, n, provid, permid in cur.fetchall():
        d = first.strftime("%Y-%m-%d") if first is not None else None
        if date and d != date:
            continue
        out.append({"trksub": trksub, "stn": st, "trkid": trkid, "date": d,
                    "first": first.isoformat() if first is not None else None,
                    "last": last.isoformat() if last is not None else None,
                    "n_obs": int(n),
                    "designation": (str(provid).strip() if provid else None)
                                   or (str(permid).strip() if permid else None)})
    return out


def trkid_records(cur, trkids):
    """The rows of the given tracklets as API ``ADES_DF`` records, in time order -- the
    same shape :func:`family_records` returns, so ``ades.records_to_obs`` reads them
    alike.  An isolated row has no provid/permid, so ``records_to_obs`` files it under
    its trksub, which is the id the linkage document must name it by."""
    cols = ", ".join(c for c in ADES_COLUMNS if c != "obstime")
    cur.execute(f"select {cols}, obstime, obstime_text from obs_sbn "
                "where trkid = any(%s) and status = any(%s)",
                (list(trkids), list(TRKSUB_STATUSES)))
    return _as_records(cur)


def lookup_trksub(token):
    """``(tracklets, records)`` for a trksub token on a fresh replica connection: every
    tracklet it names (:func:`trksub_tracklets`) and, when that is exactly ONE tracklet
    with no designation, its rows; otherwise ``records`` is ``[]`` and the caller tells
    the user why (a choice to make, or the designation it already has)."""
    from pyroorb.itf.config import ItfConfig
    from pyroorb.itf.dbio import connect_pub
    conn = connect_pub(ItfConfig())
    try:
        cur = conn.cursor()
        trk = trksub_tracklets(cur, token)
        if len(trk) == 1 and not trk[0]["designation"]:
            return trk, trkid_records(cur, [trk[0]["trkid"]])
        return trk, []
    finally:
        conn.close()


def fetch_records(desig):
    """:func:`family_records` on a fresh read-only replica connection."""
    from pyroorb.itf.config import ItfConfig
    from pyroorb.itf.dbio import connect_pub
    conn = connect_pub(ItfConfig())
    try:
        return family_records(conn.cursor(), desig)
    finally:
        conn.close()


def available():
    """True when a replica is configured and the fallback is not switched off
    (``PYROORB_SBN_FALLBACK=0``)."""
    import os
    from pyroorb.itf.dbio import load_dotenv
    try:
        load_dotenv()
    except Exception:
        pass
    if os.environ.get("PYROORB_SBN_FALLBACK", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    return bool(os.environ.get("SBN_DB_HOST"))
