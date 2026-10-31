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
# a permanent number: a minor planet's, or a numbered comet's / interstellar's
_NUMBER = re.compile(r"\d+[PDI]?")


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
