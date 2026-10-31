"""MPC identification-submission JSON for a found linkage.

Assembles the ``{"header": ..., "links": {"link_0": ...}}`` document the MPC
identifications endpoint expects, natively (no find_orb ``linkage.json``).
The shape follows find_orb's ``make_linkage_json`` (elem_out.cpp) where it is
authoritative, with the granularity fixes of the proven notebook pipeline:

* **designations**: PACKED designations (the working v23 pipeline submitted
  packed; find_orb emits unpacked-stripped -- both accepted).  Per MPC rules
  the list is NOT every designation present in the astrometry: an
  ALREADY-ACCEPTED DESDES identification must not be resubmitted, so it is the
  designations the identification actually PROPOSES -- the ones the user
  requested (:func:`requested_primaries`), or the family primary when a single
  fetch handed its aliases back -- plus any newly-picked-up designated
  (pseudo-ITF) tracklets.  A REQUESTED id the astrometry files as a trksub (an
  NEOCP temporary id, typed into the search box beside a real designation) is
  not one of them: it is named once, under ``trksubs``.
* **trksubs**: one ``[trksub, "YYYY MM DD.ddddd", station]`` entry per
  (trksub, station, night) -- finer than find_orb, which dedups by packed id
  only and stamps the first obs; the per-night/station granularity is what
  the in-use pipeline submits.  Date is UTC decimal-day, find_orb-formatted
  (taken verbatim from the obs line's date columns).  EXCEPTION: a trksub
  that is ON the NEOCP is submitted as ONE entry stamped with its DISCOVERY
  tracklet (col-13 asterisk / ADES ``disc="*"``), never one per night -- see
  :func:`collect_ids`.  The trksub NAMED is the database's ``trksub`` column
  whenever the search supplies it (:func:`db_trksubs`), not columns 6-12 of the
  line: the 80-column rendering can spell the same tracklet differently
  ('M00000' for the submitter's 'MB00000'), and only the submitter's spelling
  identifies it at the MPC and joins ``obs_sbn``.
* **identification_type: "neocp"** when any obs line's reference columns
  (73-77) read ``NEOCP`` (find_orb checks the same field).
* **orbit** (optional in the MPC format, ALL seven fields required when present):
  the linked fit's osculating elements, heliocentric J2000 ECLIPTIC, angles in
  degrees, ``epoch`` and ``pericenter_time`` as JD (TT) -- exactly what find_orb's
  ``make_linkage_json`` writes (elem_out.cpp).  Built by :func:`orbit_section`
  from the fit the linkage was judged on, and LEFT OUT -- never approximated --
  when that fit has no orbit to publish (see there).
* find_orb's validity guard: fewer than 2 distinct ids, or 50+, is not a
  submittable linkage.  Two is the ONLY shape rule -- any mix satisfies it
  (DES+DES, DES+ITF, DES+DES+ITF, ITF+ITF); a lone designation or a lone
  tracklet claims no identification.  :func:`build_linkage` applies it when the
  document is built and :func:`short_link` re-applies it on the way out, after
  the pre-submit dedup has had its chance to drop an id.
* **no COMET or INTERSTELLAR designation, anywhere in the document.**  The MPC's
  identification pipeline takes minor-planet designations only, so a linkage
  claiming ``C/2023 A3``, ``73P``, ``P/2019 LD2`` or ``1I`` -- packed or
  unpacked, as a requested primary or as a picked-up tracklet -- is REFUSED
  rather than built, and the refusal names the MPC **Jira** route, which is
  where such an identification has to go instead.  Enforced in the same two
  places as the two-id minimum: :func:`build_linkage` at build time and
  :func:`short_link` on the way out.

Submission itself is ALWAYS a dry-run unless explicitly enabled -- and even
then goes through :func:`submit_linkage` so the caller sees exactly what was
sent.  The transport (GET with a JSON body) is what the proven pipeline used
against this endpoint.
"""

import json
import re

from pyroorb.packing import pack, unpack, unpack_comet

from .obs80_fields import obs_unique_id
from .opp_profile import obs_night_ordinal

MAX_LINKAGE_IDS = 50           # find_orb's cap (elem_out.cpp MAX_LINKAGE_IDS)

# The MPC's optional per-link ``orbit`` block: when present every one of these is
# compulsory (identifications/submission-format).  Value -> our element name.
ORBIT_FIELDS = {
    "arg_pericenter": "arg_per",
    "eccentricity": "e",
    "epoch": None,                 # the fit's epoch, not an element
    "inclination": "i",
    "lon_asc_node": "asc_node",
    "pericenter_distance": "q",
    "pericenter_time": "Tp",
}
# Decimals per field: find_orb's printf widths (%f = 6, ecc %.8f, q %.15f).
_ORBIT_DECIMALS = {"arg_pericenter": 6, "eccentricity": 8, "epoch": 6,
                   "inclination": 6, "lon_asc_node": 6,
                   "pericenter_distance": 15, "pericenter_time": 6}

# The packed SURVEY designations (Palomar-Leiden / Trojan), and a numbered COMET
# ('0073P' -- cols 1-4 the periodic number, col 5 the type letter).
_SURVEY_RE = re.compile(r"^(PLS|T[123]S)\d{4}$")
_NUM_COMET_RE = re.compile(r"^\d{4}[A-Z]$")

# The MPC orbit-TYPE letters: C long-period, P periodic, D defunct, X uncertain,
# A reclassified-asteroidal, I interstellar.  The same set as
# ``obs_sources._COMET_TYPE_LETTERS`` / ``obs80._COMET_TYPES``.  Plus S, a NATURAL
# SATELLITE provisional ('S/2021 J 7' packs 'SK21J070', the 'S' in column 5 exactly
# where a comet keeps its type letter) -- no more a minor planet than a comet is, so
# it takes the same refusal.
_COMET_TYPE_LETTERS = "CPDXAIS"
# packed numbered comet, optional fragment slot: '0073P', '0073Pb'
_PACKED_NUM_COMET_RE = re.compile(r"^\d{4}[CPDXAI][a-z]{0,2}$")
# unpacked: '1P', '73P-B', '3I', 'C/2020 F3', 'P/2019 LD2', 'A/2017 U1', 'S/2021 J 7'
_UNPACKED_COMET_RE = re.compile(r"^(\d+[CPDXAI](-[A-Za-z]{1,2})?|[CPDXAIS]/.+)$")


def line_identifier(line):
    """The id an OBS80 line is filed under: provisional/trksub columns 6-12,
    else the number columns 1-5."""
    return line[5:12].strip() or line[0:5].strip()


def is_designation(ident):
    """True when ``ident`` is a real packed MPC designation (numbered or
    provisional); False for ITF-style trksubs.

    STRING-ONLY, so it cannot tell a digits-only trksub from a number -- prefer
    :func:`line_designation`, which knows which OBS80 column the id came from.
    """
    return unpack(ident) is not None


def is_packed_provisional(ident):
    """True for the packed forms that may legitimately occupy OBS80 columns
    6-12: a packed PROVISIONAL designation ('K23R67C', or the '_'-extended
    form) or a survey designation ('PLS2040', 'T1S2040').

    A packed NUMBER is deliberately NOT one of them.  The permanent number
    lives in columns 1-5; columns 6-12 carry the provisional *or* a trksub, and
    the MPC never files a permanent id there.  Without that rule a digits-only
    SBN trksub -- '133651' is a real one, sibling to '10b51e' and '1207c2' --
    round-trips through ``unpack`` as minor planet 133651 and gets submitted as
    a DESIGNATION we never fitted, silently losing its nights from ``trksubs``.

    A packed COMET-form provisional core ('K23A030', fragment slot last) counts
    too -- an OBS80 comet line files its core in 6-12 with the type letter in
    column 5.

    The test IS the strict unpacker: ``packing.unpack`` checks every character
    against the slot it occupies, so only a token ``pack`` could have produced
    reads as a designation.  A trksub that merely LOOKS packed -- 'J053207',
    whose 'half-month letter' is '3' and 'order letter' is '7' -- is refused and
    keeps its nights in ``trksubs``, where they belong.

    The 80-column format cannot fully disambiguate: a 7-character trksub shaped
    exactly like a packed provisional ('J92X01A') is indistinguishable from one.
    That ambiguity is why ADES/SBN carry permID, provID and trkSub separately.
    """
    p = str(ident).strip()
    if len(p) == 7 and (p[0] == "_" or p[0] in "IJK") and (
            unpack(p) is not None or unpack_comet(p) is not None):
        return True
    return bool(_SURVEY_RE.match(p))


def line_designation(line):
    """The packed DESIGNATION an OBS80 line is filed under, or None when the
    line's identity is a trksub.

    Column-aware, unlike :func:`is_designation`: columns 6-12 count only when
    they hold a packed provisional/survey designation, and a permanent number
    is read only from columns 1-5 (where col 5 may be a comet type letter).

    A line carrying BOTH a number in 1-5 and a trksub in 6-12 is a
    contradiction the MPC does not emit -- an observation of a numbered object
    has either its packed provisional or nothing in 6-12 -- so it raises rather
    than guessing which identity to submit.
    """
    prov = line[5:12].strip()
    num = line[0:5].strip()
    if prov:
        if is_packed_provisional(prov):
            return prov
        if num:
            raise ValueError(
                "OBS80 line carries a permanent id in cols 1-5 (%r) alongside a "
                "trksub in cols 6-12 (%r); cannot tell which identity to "
                "submit: %r" % (num, prov, line))
        return None
    if num and (unpack(num) is not None or _NUM_COMET_RE.match(num)):
        return num
    return None


def is_comet_designation(ident):
    """True when ``ident`` names a COMET or INTERSTELLAR object, in any form this
    pipeline can produce -- packed or unpacked.

    The MPC's identification endpoint takes MINOR-PLANET designations only; an
    identification involving a comet is made by hand, through a Jira ticket.  So
    this is the test that keeps one out of a submission document
    (:func:`build_linkage`, :func:`short_link`).

    Four shapes reach us: the unpacked forms ('73P', '73P-B', '3I', 'C/2020 F3',
    'A/2017 U1'), the packed numbered comet ('0073P'), the packed provisional
    core WITH its type letter ('CK23A030', what ``obs80.pack_designation``
    emits), and that core WITHOUT the letter ('K23A030') -- which is how an
    OBS80 line carries it, the type letter living apart in column 5.
    :func:`comet_line_id` is what reads that column.

    A packed extended NUMBER ('~AZAP') can end in an upper-case type letter
    without being a comet, which is why the packed-number form is matched as
    four digits and not as "ends in a letter".
    """
    s = str(ident or "").strip()
    if not s:
        return False
    if _UNPACKED_COMET_RE.match(s) or _PACKED_NUM_COMET_RE.match(s):
        return True
    if len(s) == 8 and s[0] in _COMET_TYPE_LETTERS and unpack_comet(s[1:]) is not None:
        return True
    return unpack_comet(s) is not None


def comet_line_id(line):
    """The comet / interstellar id an OBS80 line is filed under, or None.

    Deliberately NOT built on :func:`line_designation`, which reads the
    provisional in columns 6-12 and the number in 1-5.  On a comet line the
    orbit-TYPE letter lives in **column 5**, apart from the core -- so a comet
    whose core has ASTEROID form ('P/2023 JN16' packs as a bare 'K23J16N', the
    type letter carried separately) is invisible to any test that reads 6-12
    alone: it reads as an ordinary packed provisional and would be claimed as a
    minor planet.  Column 5 is the only place such a line says "comet".

    The two comet layouts, per ``obs_sources.obs80_line_id``: a PROVISIONAL
    comet leaves 1-4 blank and puts the type letter in 5 ('    CK26B030'); a
    NUMBERED comet fills 1-4 with the periodic number ('0073P').  A packed
    extended number ('~AZAP') can put an upper-case letter in column 5 too, so
    neither branch fires unless columns 1-4 are blank or all digits.
    """
    num, prov = line[0:5], line[5:12].strip()
    typ = num[4:5]
    # len() first: ''            is a substring of every string, so a short or
    # truncated line would sail through a bare ``in`` test.
    if len(typ) != 1 or typ not in _COMET_TYPE_LETTERS:
        return None
    if prov and not num[:4].strip():
        return typ + prov                      # '    CK26B030' -> 'CK26B030'
    if num[:4].strip().isdigit():
        return num.strip()                     # '0073P'
    return None


def comet_designations(astrometry, primary_desigs=()):
    """Every comet / interstellar id a document built from this material would
    claim, in first-seen order -- empty when there is none.

    Reads BOTH sides, because they carry the fact differently: the astrometry
    puts the type letter in column 5 (:func:`comet_line_id`), while a requested
    primary arrives as the user typed it and reaches ``designations`` through
    :func:`build_linkage`'s ``pack(d) or d`` passthrough, which has no opinion
    about comets.

    ONE form is genuinely undetectable, and is documented rather than claimed:
    'P/2023 JN16' packs BARE ('K23J16N'), so a comet offered as a designation
    STRING with no astrometry behind it and no 'P/' prefix is indistinguishable
    from the asteroid provisional of the same core.  Every path that carries
    observations reads the column instead, which is why this takes the
    astrometry too.
    """
    out = []
    for line in (astrometry or "").splitlines():
        if not line.strip():
            continue
        cid = comet_line_id(line)
        if cid and cid not in out:
            out.append(cid)
    for d in primary_desigs or ():
        s = str(d).strip()
        if s and is_comet_designation(s) and s not in out:
            out.append(s)
    return out


def readable_comet_ids(ids):
    """``ids`` as a person reads them: 'CK23A030' -> 'C/2023 A3', '0073P' ->
    '73P', 'PK23J16N' -> 'P/2023 JN16'.  For a refusal message.

    The comet subset of ``obs_sources.unpack_designation``, done here from
    ``packing`` alone and deliberately: a REFUSAL must not depend on importing a
    heavier module (obs_sources pulls the observatory tables and the fetch
    layer), and an id nothing can read is worth showing raw rather than failing
    the refusal that carries it.  It also reads the one form obs_sources returns
    unchanged -- the asteroid-FORM comet core 'PK23J16N', whose last character
    is an upper-case order letter rather than a fragment slot.
    """
    out = []
    for i in ids:
        s = str(i)
        if _PACKED_NUM_COMET_RE.match(s):                  # '0073P' / '0073Pb'
            frag = s[5:]
            out.append("%d%s" % (int(s[:4]), s[4])
                       + ("-" + frag.upper() if frag else ""))
            continue
        typed = len(s) == 8 and s[0] in _COMET_TYPE_LETTERS
        core = s[1:] if typed else s
        readable = unpack_comet(core) or unpack(core)
        if readable and typed and s[0] == "S":             # 'S/2021 J 7'
            year, rest = readable.split()
            out.append("S/%s %s %s" % (year, rest[0], rest[1:]))
            continue
        if readable:
            frag = core[6] if len(core) == 7 else ""
            out.append((s[0] + "/" if typed else "") + readable
                       + ("-" + frag.upper() if frag.islower() else ""))
            continue
        out.append(s)                      # already readable ('73P', 'C/2020 F3')
    return ", ".join(out)


COMET_REFUSAL = (
    "the MPC's identification pipeline does not accept comet, interstellar or "
    "natural-satellite designations, and this linkage claims {ids}. An "
    "identification involving one has to go to the MPC as a Jira ticket instead -- the astrometry, the "
    "fit and the residuals here are what it needs")


def comet_refusal(astrometry, primary_desigs=()):
    """The sentence refusing this material because it names a comet, or None.

    The ONE home of that wording.  Callers say it unprefixed: every surface's
    generic framing ("no linkage", "not a linkage", "no longer a linkage") is
    wrong here -- the material IS a linkage, it is simply not one this route can
    carry -- and a refusal that misstates its own cause is how a real warning
    stops being read (cf. :func:`itf.pickup.warning_text`).
    """
    comets = comet_designations(astrometry, primary_desigs)
    if not comets:
        return None
    return COMET_REFUSAL.format(ids=readable_comet_ids(comets))


def db_trksubs(rows):
    """``{obs_unique_id: trksub}`` from database rows carrying ``obs80`` and the
    ``trksub`` column -- the spelling :func:`collect_ids` names a tracklet by.

    Columns 6-12 of the stored 80-column record are a RENDERING of the trkSub,
    and it is not always the submitter's: one tracklet reads 'MB00000' in the
    ``trksub`` column and 'M00000' in its line.  The MPC identifies a tracklet
    by the submitted trkSub, and ``obs_sbn`` is keyed by it, so the column wins.
    Only rows whose column is non-blank are mapped."""
    out = {}
    for r in rows or ():
        trk = str(r.get("trksub") or "").strip()
        line = r.get("obs80") or ""
        if len(line) == 160:            # two-line record: identity is line 1
            line = line[:80]
        if trk and line.strip():
            out[obs_unique_id(line)] = trk
    return out


def collect_ids(astrometry, trksubs=None):
    """Split an OBS80 blob into (designations, trksub_triples, neocp).

    ``designations``: ordered distinct packed ids each line is filed under, as
    judged column-aware by :func:`line_designation` (a digits-only trksub is a
    trksub, not a number).  ``trksub_triples``: ``[trksub, date, station]`` entries,
    one per (trksub, station, night) group of the sorted lines -- a >0.5-day
    jump starts a new night.  ``neocp``: any line referenced NEOCP.

    NEOCP oddity: an object that is ON the NEOCP must be submitted as ONE
    trksub entry stamped with its DISCOVERY tracklet's date -- not one entry
    per night.  A trksub is NEOCP when any of its lines reference NEOCP
    (cols 73-77); its discovery tracklet is the group whose line carries the
    col-13 discovery asterisk (the ADES ``disc="*"`` flag, which
    ``format_obs80`` renders into col 13).  Without an asterisk anywhere the
    first tracklet stands in.  Ordinary ITF trksubs keep the per-night
    granularity the in-use pipeline submits.

    ``trksubs``: ``{obs_unique_id: trksub}`` (:func:`db_trksubs`).  A line that is
    not a designation and appears there is filed under that trksub instead of
    its columns 6-12, so the document names the tracklet as the database does.
    """
    trksubs = trksubs or {}
    # Sort with the col-13 discovery asterisk neutralized: it sits BEFORE the
    # date columns, so a flagged line would otherwise sort after every
    # unflagged line of its trksub and land in the wrong night group.
    lines = sorted((l for l in astrometry.splitlines() if l.strip()),
                   key=lambda l: l[:12] + " " + l[13:])
    designations = []
    triples = []
    neocp = False
    trk_groups = {}                # trksub -> [triple index, ...]
    trk_neocp = set()              # trksubs with any NEOCP-referenced line
    trk_disc = {}                  # trksub -> triple index of the discovery group
    old_trk = old_stn = None
    old_day = None
    for line in lines:
        trk = line_identifier(line)
        if not trk:
            continue
        if line[72:77] == "NEOCP":
            neocp = True
        day = obs_night_ordinal(line)
        stn = line[77:80]
        desig = line_designation(line)
        if desig:
            if desig not in designations:
                designations.append(desig)
        else:
            trk = trksubs.get(obs_unique_id(line)) or trk
            if (trk != old_trk or stn != old_stn
                    or old_day is None or day > old_day + 0.5):
                triples.append([trk, line[15:32].strip(), stn])
                trk_groups.setdefault(trk, []).append(len(triples) - 1)
            if line[72:77] == "NEOCP":
                trk_neocp.add(trk)
            if len(line) > 12 and line[12] == "*":
                trk_disc[trk] = trk_groups[trk][-1]
        old_trk, old_stn, old_day = trk, stn, day
    drop = set()
    for trk in trk_neocp:
        idxs = trk_groups.get(trk) or []
        keep = trk_disc.get(trk, idxs[0] if idxs else None)
        drop.update(i for i in idxs if i != keep)
    if drop:
        triples = [t for i, t in enumerate(triples) if i not in drop]
    return designations, triples, neocp


def orbit_section(fit):
    """``(orbit, None)`` -- the MPC ``orbit`` block for ``fit`` -- or ``(None, why)``.

    Heliocentric J2000 ecliptic osculating elements at the fit's epoch, angles
    in degrees in [0, 360), ``epoch`` / ``pericenter_time`` as JD on the fit's
    dynamical time scale (TDB; TT to within 2 ms, which is what find_orb
    writes).  An UNBOUND orbit is fine: q, e and the pericentre time are all
    defined for it (find_orb writes it the same way).

    The block is optional, so an orbit we would not publish anywhere else is
    LEFT OUT rather than sent: an identification document outlives every banner
    that qualified the orbit, exactly as an MPCORB line does
    (``web/server._orbit_export_refusal``).  So it is omitted when

    * the arc does not determine an orbit (``constraint.assess``, the one
      verdict the elements panel and the exports obey) -- including when the
      user opted into a display orbit anyway: ``nominal_optin`` may pick its
      orbit by the population PRIOR, which is display-only and must never
      feed an MPC product (``popprior``);
    * any field is not a finite number.
    """
    import math
    if fit is None:
        return None, "no fit"
    if getattr(fit, "nominal_optin", None):
        return None, "the shown orbit is a display choice on an undetermined arc"
    try:
        el = fit.elements()
        epoch = float(fit.epoch_jd())
    except Exception as ex:
        return None, f"no elements ({type(ex).__name__})"
    out = {}
    for key, name in ORBIT_FIELDS.items():
        try:
            v = epoch if name is None else float(el[name])
        except (KeyError, TypeError, ValueError):
            return None, f"no {key}"
        if not math.isfinite(v):
            return None, f"{key} is not finite"
        if name in ("arg_per", "asc_node"):
            v %= 360.0
        out[key] = round(v, _ORBIT_DECIMALS[key])
    try:
        from pyroorb import constraint
        verdict = constraint.assess(fit)
    except Exception:
        verdict = None                 # the guard must never cost the linkage
    if verdict:
        return None, ("the arc does not determine an orbit ("
                      + verdict.get("headline", "essentially unconstrained") + ")")
    return out, None


def orbit_section_problem(orbit):
    """Why an ``orbit`` block would be refused by the MPC (a field missing or not a
    finite number), or None -- the send-time check :func:`short_link` runs."""
    import math
    if not isinstance(orbit, dict):
        return "orbit is not an object"
    for key in ORBIT_FIELDS:
        v = orbit.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return f"orbit field {key!r} missing or not a finite number"
    return None


def submission_header(cfg, co_credit=None):
    """The ``header`` of every identification document we write: the submitter
    identity from ``cfg`` (name, optionally ``", co_credit"``), email, comment.
    With no identity configured (there is no built-in one) the name and e-mail
    are left EMPTY, never padded -- :func:`header_problem` then refuses it."""
    name = (cfg.submit_name or "").strip()
    if co_credit and name:
        name = f"{name}, {co_credit}"
    return {"name": name, "email": (cfg.submit_email or "").strip(), "comment": "REPLY_EMAIL"}


_NAME_FORBIDDEN = re.compile(r"[{}\[\]]")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AnonymousDocument(ValueError):
    """An identification document was about to be assembled without a valid
    submitter header -- refused at the source (:func:`submission_document`)."""


def header_problem(header):
    """Why ``header`` cannot head an identification document, or None.

    THE one rule (by design an unsigned document is never generated,
    never sent, never accepted): a non-empty ``name`` without ``{ } [ ]`` (the MPC
    rejects those) and a plausible ``email``, which is where the MPC credits and
    replies.  Applied where a document is ASSEMBLED (:func:`submission_document`),
    before every TRANSMISSION (:func:`submit_linkage`, the web's queue POST) and on
    RECEIPT by the identification queue."""
    if not isinstance(header, dict):
        return "the document has no header"
    name, email = header.get("name"), header.get("email")
    if not isinstance(name, str) or not name.strip():
        return "the header has no submitter name"
    if _NAME_FORBIDDEN.search(name):
        return "the header name contains { } [ or ] (the MPC rejects it)"
    if not isinstance(email, str) or not _EMAIL.match(email.strip()):
        return "the header has no valid submitter e-mail"
    return None


def anonymous_header(doc):
    """Why ``doc`` may not be sent for want of a submitter, or None (see
    :func:`header_problem`).  There is no built-in identity: the CLI tools sign
    with ITF_SUBMIT_NAME / ITF_SUBMIT_EMAIL, the web with the person's own."""
    p = header_problem((doc or {}).get("header"))
    if p:
        return (f"no submitter: {p} -- set ITF_SUBMIT_NAME and ITF_SUBMIT_EMAIL "
                "(the MPC credits and replies to them)")
    return None


def submission_document(header, links):
    """THE one place an identification document is assembled: ``header`` plus
    ``links`` (a list of link bodies, in order) keyed ``link_0 .. link_N``.
    Every link body comes from :func:`build_linkage`; this only wraps them --
    one search, a batch's merge, a bundled submission all go through here.

    REFUSES (:class:`AnonymousDocument`) a header :func:`header_problem` rejects:
    an unsigned document is never generated.  What exists before anyone has
    signed is a :func:`link_draft`, which has no header at all."""
    p = header_problem(header)
    if p:
        raise AnonymousDocument(f"refusing to build an identification document: {p}")
    return {"header": dict(header),
            "links": {f"link_{i}": dict(link) for i, link in enumerate(links)}}


def link_draft(links):
    """The links of a linkage BEFORE anyone has signed it: ``{"links": ...}`` with
    NO header -- not a document, never sent as one.  The web server (which has no
    identity of its own) and a search run without ITF_SUBMIT_NAME produce these;
    whoever submits wraps them with :func:`signed_document`."""
    return {"links": {f"link_{i}": dict(link) for i, link in enumerate(links)}}


def ordered_links(doc):
    """A document's or draft's link bodies in ``link_0 .. link_N`` order."""
    links = (doc or {}).get("links") or {}

    def n(k):
        try:
            return int(str(k).rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 1 << 30
    return [links[k] for k in sorted(links, key=lambda k: (n(k), str(k)))]


def signed_document(doc_or_draft, name, email, comment=None):
    """``doc_or_draft``'s links under the header ``name`` / ``email`` -- the ONLY
    way a draft becomes a document.  Raises :class:`AnonymousDocument` when the
    identity is not one :func:`header_problem` accepts."""
    old = (doc_or_draft or {}).get("header") or {}
    header = {"name": (name or "").strip(), "email": (email or "").strip(),
              "comment": comment if comment is not None else old.get("comment", "REPLY_EMAIL")}
    return submission_document(header, ordered_links(doc_or_draft))


def build_linkage(combined_astrometry, cfg, primary_desigs=(), itf_obs_out="",
                  co_credit=None, fit=None, trksubs=None):
    """The submission document for a found linkage, or ``(None, reason)``.

    ``combined_astrometry``: the full accepted OBS80 blob (published + ITF).
    ``primary_desigs``: the seed object's family PRIMARY designations
    (unpacked) -- when non-empty, the designations array is restricted to
    these plus any designated pickups found in ``itf_obs_out``, per the
    MPC no-resubmitted-DESDES rule.  ``fit``: the orbit the linkage was
    judged on; its elements ride in the link's optional ``orbit`` block
    (:func:`orbit_section`).  ``trksubs``: the database trksub of each ITF line
    (:func:`db_trksubs`), which names its tracklet in place of columns 6-12.
    Returns ``(linkage_dict, None)`` on success.
    """
    designations, triples, neocp = collect_ids(combined_astrometry, trksubs)

    # A comet/interstellar designation is refused OUTRIGHT, ahead of the shape
    # rules: no amount of evidence makes it submittable by this route, so "how
    # many ids does it name" is not the question to answer first.
    comet_why = comet_refusal(combined_astrometry, primary_desigs)
    if comet_why:
        return None, comet_why

    if primary_desigs:
        # An id the ASTROMETRY files under a trksub is not a designation, however
        # the user typed it: a NEOCP temporary id ('A11FwDx') has no packed form,
        # so `pack(d) or d` would pass it through verbatim and name it in BOTH
        # arrays -- claiming a designation the MPC has never issued.  collect_ids
        # has already put it where it belongs, under trksubs.
        trksub_ids = {t[0] for t in triples}
        wanted = []
        for d in primary_desigs:
            p = pack(d) or d
            if p in trksub_ids or str(d).strip() in trksub_ids:
                continue
            if p not in wanted:
                wanted.append(p)
        # designated (pseudo-ITF) pickups among the newly-linked tracklets
        pickup_designations, _t, _n = collect_ids(itf_obs_out, trksubs)
        for p in pickup_designations:
            if p not in wanted:
                wanted.append(p)
        designations = wanted

    n_ids = len(designations) + len(triples)
    if n_ids < 2:
        return None, f"not a linkage: {n_ids} distinct id(s)"
    if n_ids >= MAX_LINKAGE_IDS:
        return None, f"too many ids for one linkage ({n_ids} >= {MAX_LINKAGE_IDS})"

    link = {}
    if designations:
        link["designations"] = designations
    if triples:
        link["trksubs"] = triples
    if neocp:
        link["identification_type"] = "neocp"
    if fit is not None:
        orbit, _why = orbit_section(fit)
        if orbit is not None:
            link["orbit"] = orbit
    header = submission_header(cfg, co_credit)
    if header_problem(header):
        # no identity configured (the web server never has one): a DRAFT, not an
        # unsigned document -- the submitter signs it (signed_document)
        return link_draft([link]), None
    return submission_document(header, [link]), None


def included_obs80(fit):
    """The OBS80 lines of a fit's INCLUDED observations, synthesised for obs that carry no
    verbatim line (an ADES fetch keeps none)."""
    from pyroorb import obs80
    out = []
    try:
        inc = fit.is_included()
    except Exception:
        return out
    for o, keep in zip(fit._obs, inc):
        if not keep:
            continue
        line = o.get("obs80_line")
        if not line:
            try:
                line = obs80.obs80_base_line(o).splitlines()[0]
            except Exception:
                continue
        out.append(line)
    return out


def requested_primaries(fit, present=None):
    """The designations a linkage document must name because the USER asked for them --
    two or more typed into the CLI or the web search box -- or ``[]`` when this fit is
    not that.

    A fetch of ``K13M06C K21C05V`` comes back carrying both designations AND whatever
    aliases the MPC has already acknowledged for either of them.  Those aliases are old
    news (an ALREADY-ACCEPTED DES=DES is what the MPC rejects -- a DES=DES it has not
    accepted is exactly what an identification submission is FOR), but the REQUESTED pair is the
    identification being proposed and every one of them belongs in the document -- so
    ``designations`` is one id per identification: the requested ones, plus (added by
    :func:`build_linkage`) any designated pickup the search newly linked.  Collapsing to
    a single alphabetically-earliest primary is right only for a ONE-designation fetch,
    where the several designations that came back really are one acknowledged family.

    ``present``: the designations the astrometry actually carries (default: the fit's
    obs).  A requested designation whose observations are all gone -- never fetched, or
    switched off in the UI -- is not part of the linkage and drops out here; if that
    leaves fewer than two, there is no user-assembled identification left to name.
    """
    if present is None:
        try:
            present = {o.get("designation") for o in fit._obs}
        except Exception:
            return []
    present = {d for d in present if d}
    out = []
    for d in getattr(fit, "requested_designations", None) or []:
        if d in present and d not in out:
            out.append(d)
    return out if len(out) > 1 else []


def linkage_from_fit(fit, cfg, primary_desigs=()):
    """The MPC identification document for a joint fit the USER assembled -- typing two or more
    designations into the CLI or the web search box -- rather than one the ITF search found.

    Nothing about :func:`build_linkage` was ever ITF-specific: it reads whatever ids the astrometry
    carries, files real designations under ``designations`` and unrecognised ones (ITF trksubs,
    NEOCP temporary ids) under ``trksubs``, and stamps ``identification_type: "neocp"`` when a line
    references the NEOCP.  So a fit of ``1999 TV43`` + the NEOCP object ``OOn0024`` yields exactly
    the NEOCP identification the MPC expects, with no new document code.  What was missing was
    anyone asking for it: the linkage only ever reached the user through the ITF search.

    Built from the INCLUDED observations, so it follows the screen -- switch a designation's obs
    off and it leaves the linkage.  Returns ``(doc, error)``."""
    lines = included_obs80(fit)
    if not lines:
        return None, "no included observations"
    if not primary_desigs:
        # name the designations the USER requested, one per identification -- not every
        # alias the MPC handed back with them (see :func:`requested_primaries`)
        try:
            present = {o.get("designation")
                       for o, keep in zip(fit._obs, fit.is_included()) if keep}
        except Exception:
            present = None
        primary_desigs = requested_primaries(fit, present)
    return build_linkage("\n".join(lines) + "\n", cfg, primary_desigs=primary_desigs,
                         fit=fit)


def all_unpacked_designations(doc):
    """Every designation in ``doc``, UNPACKED, across all of its links -- what
    a family lookup has to cover before :func:`dedup_designations` runs."""
    out = []
    for _key, link in sorted((doc or {}).get("links", {}).items()):
        for pk in link.get("designations") or []:
            unp = unpack(pk) or pk
            if unp not in out:
                out.append(unp)
    return out


def dedup_designations(doc, family_of, log=print, keep_spelling=False):
    """Collapse each link's designations array to ONE representative per MPC
    identification family -- the family PRIMARY, at the position of the
    family's first occurrence.

    The MPC REJECTS a linkage that restates an already-acknowledged DES=DES
    identification -- even when the actual news is the attached ITF tracklets
    -- and the representative it should see is the family's primary
    designation, not whichever alias the search happened to be seeded with.
    ``family_of(unpacked) -> family primary`` (per
    ``current_identifications``, where every member of an n-designation
    family has its own row, the primary self-listed; itself when
    unidentified).  Mutates and returns ``doc``.

    EVERY link is deduped, independently -- families are a property of the
    designations, not of the document, so two links naming aliases of one
    family each keep their own representative.  A document carrying many links
    is what ``itf batch`` and a bundled submission transmit, and one
    unchecked link among them is enough to have the whole submission refused;
    this used to read ``link_0`` alone.  On a single-link document -- the web
    UI, one search, a bundled submission -- the behaviour is identical.

    This is the FINAL catcher, not the primary rule: which designations a
    document may name is settled upstream from what the user REQUESTED (see
    :func:`requested_primaries`), which needs no database.  Collapsing can take
    a link below the two-id minimum, which is why :func:`short_link` re-checks
    the document afterwards.

    ``keep_spelling=True`` (the identification queue, where a link is
    never edited) only DROPS: the first designation of each family stays exactly
    as the submitter wrote it, never swapped for the primary or repacked, so the
    only change is the removal of designations the MPC has already joined.
    """
    for key, link in sorted((doc or {}).get("links", {}).items()):
        packed_list = link.get("designations") or []
        kept, seen, changed = [], set(), []
        for pk in packed_list:
            unp = unpack(pk) or pk
            fam = family_of(unp) or unp
            if fam in seen:
                changed.append(f"dropped {pk} (family {fam})")
                continue
            seen.add(fam)
            if keep_spelling:
                kept.append(pk)
                continue
            rep = pack(fam) or pk        # prefer the PRIMARY, packed
            if rep != pk:
                changed.append(f"{pk} -> primary {rep}")
            kept.append(rep)
        if changed:
            where = f"{key}: " if len(doc.get("links") or {}) > 1 else ""
            log(f"designations vs MPC identifications: {where}"
                + "; ".join(changed))
            link["designations"] = kept
    return doc


def short_link(doc):
    """The reason ``doc`` is not submittable as it stands, or None when every
    link is fine -- the SEND-TIME re-check of the build-time rules.

    The MPC accepts any MIX of ids -- DES+DES, DES+ITF, DES+DES+ITF, ITF+ITF --
    and the only shape rule is that a link must name at least TWO of them: one
    designation, or one tracklet, on its own claims no identification.
    :func:`build_linkage` already refuses to construct such a document, so this
    exists for the one way a valid document can LOSE an id afterwards:
    :func:`dedup_designations` collapsing two designations that turn out to be
    one already-acknowledged family.  A pair the MPC has already merged, with no
    new tracklet attached, is not news -- it is the resubmitted DES=DES the dedup
    exists to prevent, arriving as a one-id document instead of a two-id one.
    """
    for key, link in sorted((doc or {}).get("links", {}).items()):
        # A comet cannot be identified by this route at all.  build_linkage
        # refuses to construct such a document; this is the backstop for a doc
        # that reached here another way (a caller assembling one by hand, a
        # stashed document replayed after the rule changed).
        comets = [d for d in (link.get("designations") or [])
                  if is_comet_designation(d)]
        if comets:
            return "%s: %s" % (key, COMET_REFUSAL.format(
                ids=readable_comet_ids(comets)))
        if "orbit" in link:
            bad = orbit_section_problem(link["orbit"])
            if bad:
                return f"{key}: {bad} -- the MPC requires all seven when an orbit is sent"
        n_ids = len(link.get("designations") or []) + len(link.get("trksubs") or [])
        if n_ids < 2:
            return (f"{key} names {n_ids} id(s) after the identification dedup -- a "
                    f"linkage needs two. Its designations belong to one identification "
                    f"the MPC has already accepted and nothing new is attached to them, "
                    f"so there is nothing to submit")
    return None


def _family_map(conn, unpacked_desigs):
    """{unpacked designation -> family primary} via public.current_identifications
    (a designation absent from the table is its own family)."""
    out = {d: d for d in unpacked_desigs}
    if not unpacked_desigs:
        return out
    with conn.cursor() as cur:
        cur.execute(
            """SELECT unpacked_secondary_provisional_designation,
                      unpacked_primary_provisional_designation
               FROM current_identifications
               WHERE unpacked_secondary_provisional_designation = ANY(%s)""",
            (list(unpacked_desigs),))
        for sec, prim in cur.fetchall():
            out[sec] = prim
    return out


def record_submission(doc, cfg, trkid_map=None, log=print):
    """Write a REAL (transmitted) submission into the SQL ledger
    (``itf.submitted_links`` + ``itf.submitted_link_items``): one link row
    stamped now, one item row per designation (UNPACKED) and per linked
    tracklet (resolved to its joinable ``trkid``).

    ``trkid_map``: ``{(trksub, stn, 'YYYY MM DD'): trkid}`` from the search's
    candidate rows; any triple it can't answer is resolved from ``obs_sbn``.
    Returns the ledger link ids -- ONE PER LINK in the document, in key order
    -- or None (recording failure is LOGGED, never raised: the submission
    already happened).  A multi-link document used to have only its ``link_0``
    recorded, so a bundled submission's other linkages left no trace of having
    been sent.
    """
    from .dbio import connect_index
    trkid_map = trkid_map or {}
    links = [link for _k, link in sorted((doc or {}).get("links", {}).items())]
    try:
        conn = connect_index(cfg)
        try:
            cur = conn.cursor()
            link_ids = []
            for link in links:
                link_ids.append(_record_one(cur, link, trkid_map, log=log))
            conn.commit()
            if len(link_ids) > 1:
                log(f"ledger: {len(link_ids)} submission(s) recorded")
            return link_ids
        finally:
            conn.close()
    except Exception as ex:
        log(f"ledger: RECORDING FAILED ({type(ex).__name__}: {ex}) -- "
            f"the submission itself was already transmitted")
        return None


def _record_one(cur, link, trkid_map, log=print):
    """One ledger link row for one document link: the link row, an item row per
    UNPACKED designation and one per linked tracklet resolved to its trkid."""
    trkids = []
    for trksub, date, stn in link.get("trksubs") or []:
        key = (trksub, stn, str(date)[:10])
        tid = trkid_map.get(key)
        if tid is None:
            cur.execute(
                """SELECT DISTINCT trkid FROM obs_sbn
                   WHERE trksub = %s AND stn = %s
                     AND obstime >= to_timestamp(%s, 'YYYY MM DD')
                     AND obstime < to_timestamp(%s, 'YYYY MM DD')
                                   + interval '1 day'""",
                (trksub, stn, str(date)[:10], str(date)[:10]))
            rows = cur.fetchall()
            tid = rows[0][0] if len(rows) == 1 else None
        if tid is None:
            log(f"ledger: could not resolve trkid for "
                f"{(trksub, stn, date)}; item skipped")
        elif tid not in trkids:
            trkids.append(tid)
    cur.execute("INSERT INTO itf.submitted_links DEFAULT VALUES RETURNING id")
    link_id = cur.fetchone()[0]
    for pk in link.get("designations") or []:
        cur.execute("INSERT INTO itf.submitted_link_items "
                    "(link_id, designation) VALUES (%s, %s)",
                    (link_id, unpack(pk) or pk))
    for tid in trkids:
        cur.execute("INSERT INTO itf.submitted_link_items "
                    "(link_id, trkid) VALUES (%s, %s)", (link_id, tid))
    log(f"ledger: recorded submission as link {link_id} "
        f"({len(link.get('designations') or [])} designation(s), "
        f"{len(trkids)} tracklet(s))")
    return link_id


def submit_linkage(doc, cfg, do_submit=False, log=print, trkid_map=None, dedup=True,
                   keep_spelling=False):
    """Dry-run print (default) or actually POST the linkage to the MPC.

    A REAL transmission (``do_submit`` AND ``cfg.submit`` -- two explicit
    switches) first dedups the designations against the MPC's own
    ``current_identifications`` (see :func:`dedup_designations`: restating an
    ALREADY-ACCEPTED DES=DES pair gets the whole linkage rejected), and
    afterwards records the submission in the SQL ledger
    (:func:`record_submission`).  Returns ``(sent, status_text, mpc_message)``
    -- ``mpc_message`` is the ``message`` field of the MPC's JSON
    acknowledgement (None on dry-run, a refusal, or an unparseable body).

    BACKSTOP: the dedup can drop an id, so the two-id rule is re-checked on the
    document as it would go out (:func:`short_link`) and a collapsed linkage is
    REFUSED rather than transmitted -- ``build_linkage``'s own ``n_ids >= 2``
    guard ran before the dedup and cannot see this.  A refusal comes back as
    ``sent=False`` with the reason as the status text, so the CLI prints it and
    the web endpoint surfaces it instead of reporting a successful send.

    ``dedup=False`` skips that rewrite: the identification queue never edits a
    designations-only link and decides for itself
    (the identification queue's sender).  ``keep_spelling=True`` makes the dedup DROP
    only, never rename (:func:`dedup_designations`) -- the queue's setting.
    """
    really = do_submit and cfg.submit
    if really and dedup:
        try:
            from .dbio import connect_pub
            conn = connect_pub(cfg)
            try:
                unp = all_unpacked_designations(doc)
                fam = _family_map(conn, unp)
                dedup_designations(doc, fam.get, log=log, keep_spelling=keep_spelling)
            finally:
                conn.close()
        except Exception as ex:
            # The dedup is the FINAL catcher, not the primary rule: which
            # designations may be named is decided upstream from what the user
            # REQUESTED (itf.linkage.requested_primaries / the fit's
            # n_requested -- two typed designations are a proposal, aliases
            # that rode in on one fetch are already acknowledged).  Losing the
            # catcher is therefore a degraded check, not a blind submission.
            log(f"pre-submit identification dedup unavailable "
                f"({type(ex).__name__}: {ex}); submitting as-is")
    log(json.dumps(doc, indent=2))
    if not really:
        return False, "dry-run (submission disabled)", None
    short = short_link(doc)
    if short:
        log(f"NOT SUBMITTED: {short}")
        return False, f"not submitted: {short}", None
    anon = anonymous_header(doc)
    if anon:
        log(f"NOT SUBMITTED: {anon}")
        return False, f"not submitted: {anon}", None
    import requests
    # (connect, read) seconds: a hung MPC must not wedge the caller -- the queue's
    # send thread most of all, which sends nothing else while one link is in flight
    resp = requests.get(cfg.submit_url, json=doc, timeout=(15, 120))  # the endpoint's verb
    log(f"MPC submission: {resp.status_code} {resp.reason}")
    log(resp.text)
    # the MPC acknowledges with a JSON body like {"message": "Thank you for
    # submitting your identifications. ..."} -- surface it to the caller
    msg = None
    try:
        body = resp.json()
        if isinstance(body, dict):
            msg = body.get("message") or None
    except Exception:
        pass
    if resp.status_code == 200:
        record_submission(doc, cfg, trkid_map=trkid_map, log=log)
    return True, f"{resp.status_code} {resp.reason}", msg
