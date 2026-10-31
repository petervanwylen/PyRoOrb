"""Partial pickup of a DESIGNATED (pseudo-ITF) object -- and what it implies.

The search links tracklets, but the submission document names DESIGNATIONS.
Those are not the same granularity, and the gap between them is a real defect:

  2024 TF21 (K24T21F) linked one 2017 tracklet that the archive files under
  2017 SS396 (K17Sd6S).  2017 SS396 has THREE tracklets; the other two sit
  1.05 deg and 0.93 deg off the linked orbit, which fits everything it kept at
  0.33".  So only the 2017-10-15 G96 tracklet of 2017 SS396 is 2024 TF21 --
  2017 SS396 is ITSELF a mislinkage, assembled before either object had an
  orbit worth the name.

The identification the search found is real; the document it built was not.
``{"designations": ["K24T21F", "K17Sd6S"]}`` claims that ALL of 2017 SS396 is
2024 TF21, and two thirds of it demonstrably is not.  The MPC identification
pipeline has no way to say "this tracklet of that designation, not the rest" --
splitting an existing designation is a Jira ticket, not a submission.

So this module answers one question about every DESIGNATED tracklet a search
picked up: **did we take the whole designation, or part of it?**  A part-take
whose leftovers do not fit the linked orbit is a mislinked designation
(``compromised``), and the caller's rules are:

  * the designation is banned from the submission document -- never claimed,
    on any surface (the fit and the astrometry keep the tracklet; it is real
    astrometry of the object and the user should see it);
  * if what remains is still a submittable linkage, it is submitted without it;
  * if it is not (the arc drops back to a thin profile, or to one id), no
    document is generated at all and the find is reported for review.

Deliberately NOT flagged: a part-take whose leftovers DO fit the orbit.  That
is the ordinary case where the corridor never swept the other nights (out of
the era, under the faint cut, or simply not retrieved) -- the designation is
whole, claiming it is correct, and warning about it would spend a
click-through on nothing.  It is recorded in the vetting either way.

The leftover test is one ephemeris node per missing tracklet, computed at
that tracklet's OWN station and mid-epoch and compared with what was actually
observed there.  On the case above it reads 0.4" for the
tracklet that belongs and 3779"/3332" for the two that do not, against a
formal sky sigma of 0.08" -- the discriminator is not close to either edge,
which is why the thresholds (``pickup_off_arcsec``, ``pickup_off_sigma``) can
be blunt.
"""

import math

from pyroorb.obs80 import mpc80_to_radec
from pyroorb.packing import unpack

from .linkage import line_designation
from .obs80_fields import obs_unique_id, split_obs80_long_lines
from .opp_profile import obs_night_ordinal


def _obs_time_key(line):
    """(station, observation epoch to ~9 s) -- the granularity at which a linked
    line and an archived observation are "the same observation".

    Not the full observation identity: the two sides come from the same database
    rows, but a re-rendered line (the web UI synthesises OBS80 from ADES at a
    finer position precision) is byte-different, so :func:`obs_unique_id` --
    which folds in the sky position -- misses it.  Identity is still tried first
    by :func:`_is_linked`; this is the fallback that survives a re-render.

    It keys on STATION + EPOCH, deliberately dropping only the position field.
    A coarser key that truncated the epoch to the calendar day (the old
    ``_night_key``) collapsed two tracklets taken at the same station on the
    same UT night into one: if the search linked one of them and the other sat
    off the orbit, the off-orbit tracklet inherited the linked one's day key,
    was filed as linked, and the mislinked designation was claimed whole with
    no partial-pickup record -- exactly the failure this module exists to catch.
    The re-render changes an observation's POSITION, not its TIME, so the epoch
    is the discriminator that both survives the re-render and separates
    same-night tracklets (which are minutes to hours apart).  Rounded to 1e-4
    day (~8.6 s) so a last-digit reformatting of the epoch cannot split one
    physical observation off its own tracklet, while any two genuinely distinct
    tracklets still land on different keys."""
    day = int(obs_night_ordinal(line))          # integer ordinal day
    try:
        frac = float(line[25:32])               # full day fraction (sub-second)
    except ValueError:
        frac = 0.0
    return (line[77:80], day, round(frac, 4))


def _is_linked(lines, linked_ids, linked_times):
    return (any(obs_unique_id(l) in linked_ids for l in lines)
            or any(_obs_time_key(l) in linked_times for l in lines))


def designated_pickups(itf_obs80, own_desigs=()):
    """``{packed designation: [line, ...]}`` for the linked ITF lines that are
    filed under a DESIGNATION rather than a trksub -- the pseudo-ITF pickups,
    which are the only ones a document can name as a designation.

    ``own_desigs`` (both spellings, as :func:`search.own_designations` builds
    it) is excluded: the seed's own family is not a pickup.
    """
    own = set(own_desigs or ())
    out = {}
    for line in (itf_obs80 or "").splitlines():
        if not line.strip():
            continue
        try:
            desig = line_designation(line)
        except ValueError:              # a contradictory line names nothing
            continue
        if not desig or desig in own or (unpack(desig) or "") in own:
            continue
        out.setdefault(desig, []).append(line)
    return out


def _archive_source(source):
    """The first source in the chain that can answer for a designation's whole
    archived tracklet set (a live :class:`candidates.DbSource`), or None -- an
    overlay/harness source wraps one, and a fully offline one answers for
    nothing, in which case there is no check to run."""
    from .candidates import source_chain
    for s in source_chain(source):
        if hasattr(s, "designation_obs"):
            return s
    return None


def _tracklets_of(rows):
    """``{trkid: [line, ...]}`` from database observation rows, with the
    160-character two-line records split exactly as the sweep splits them."""
    out = {}
    for r in rows:
        for line in split_obs80_long_lines(r["obs80"]).splitlines():
            if line.strip():
                out.setdefault(r["trkid"], []).append(line)
    for lines in out.values():
        lines.sort(key=lambda l: l[15:32])
    return out


def _describe(trkid, lines, rows_trksub=None):
    first = lines[0]
    return {"trkid": trkid,
            "trksub": rows_trksub or first[5:12].strip(),
            "stn": first[77:80],
            "date": first[15:32].strip(),
            "n_obs": len(lines)}


def _placeable(stn, obscodes):
    """Can the ephemeris put the OBSERVER where this tracklet was observed?

    Only for the geocentre and for a station with real parallax constants.  A
    SATELLITE (C51/WISE, 250/HST -- not in the ground table at all) or a ROVING
    observer (247, rho == 0) has no fixed position, and its true place comes
    from the `s`/`v` companion line, which a leftover tracklet's OBS80 text
    here does not carry.  Asked because the ephemeris does NOT refuse such a
    station -- ``observer_positions_batch`` silently falls back to a geocentre
    placeholder -- so the comparison would be an orbiting observer's
    astrometry against a geocentric prediction.  That is a parallax of ~12" for
    a main-belt target at low Earth orbit and hundreds for a NEO, i.e. past
    ``pickup_off_arcsec`` on geometry alone, and it would file an innocent
    designation as a mislinkage.  Untestable is the honest answer.
    """
    if stn == "500":
        return True
    lon_rcp_rsp = (obscodes or {}).get(stn)
    return bool(lon_rcp_rsp) and (abs(lon_rcp_rsp[1]) > 1e-9
                                  or abs(lon_rcp_rsp[2]) > 1e-9)


def _offset_arcsec(fit, obscodes, lines):
    """``(offset, sigma)`` in arcsec between the orbit's prediction and what
    was observed, at the tracklet's middle observation and its own station.
    ``(None, None)`` when the tracklet cannot be tested at all."""
    from pyroorb.ephemerides import compute_ephemeris
    line = lines[len(lines) // 2]
    try:
        jd, ra, dec, stn = mpc80_to_radec(line)
        if not _placeable(stn or "500", obscodes):
            return None, None
        row = compute_ephemeris(fit, obscodes, jd, jd, 1.0, site=stn or "500",
                                full=False)[0]
    except Exception:                                    # noqa: BLE001
        return None, None
    cosd = math.cos(math.radians(dec))
    d_ra = ((row["ra_deg"] - ra + 180.0) % 360.0 - 180.0) * cosd * 3600.0
    d_dec = (row["dec_deg"] - dec) * 3600.0
    return math.hypot(d_ra, d_dec), row.get("unc_smaj_arcsec")


def _unchecked(packed, why, log=None):
    """A designated pickup this check COULD NOT MEASURE at all -- distinct from one it
    measured and found wanting (``compromised``), and from one it measured and cleared.

    Silence was the old answer to all three of the live-database ways of getting here,
    and silence is indistinguishable from "checked, nothing to report".  That makes the
    whole guard capable of being inert -- permanently, invisibly -- which is how a
    mislinked designation gets claimed whole in the first place.  The module already
    draws exactly this distinction one level down, where an unplaceable tracklet reads
    "NOT CLAIMED, COULD NOT CHECK" instead of being accused; this is the same rule at
    designation granularity.

    ``compromised`` stays FALSE on purpose, so :func:`banned_designations` does not drop
    it from the document (by design: a failure to retrieve is not evidence
    against the identification, and a linkage must not vanish because a query did).  The
    consequence is carried by the WARNING instead: it rides in ``ItfSearchResult.warnings``,
    which puts it in the batch CSV, in the web banner, and in front of the submit
    click-through -- reviewable everywhere, auto-submittable nowhere.
    """
    desig = unpack(packed) or packed
    rec = {"packed": packed, "desig": desig, "unchecked": True,
           "compromised": False, "why": why,
           "n_archive": None, "linked": [], "missing": [],
           "warning": (
               f"COULD NOT CHECK {desig}: this identification claims the whole "
               f"designation, but there was trouble pulling in {desig}'s other "
               f"astrometry -- {why} -- so nothing here confirms that the REST of "
               f"{desig} is the same object. That check is what catches a designation "
               f"which is itself a mislinkage, and it did not run. The linkage and its "
               f"astrometry are unaffected and {desig} is still named; examine it by "
               f"hand before submitting.")}
    if log:
        log(rec["warning"])
    return rec


def review(itf_obs80, source, *, fit=None, obscodes=None, cfg=None,
           own_desigs=(), log=None):
    """One record per DESIGNATED pickup that this linkage took only PART of.

    Each record: the designation (packed + unpacked), the archive's whole
    tracklet inventory split into ``linked``/``missing``, each missing
    tracklet's offset from the linked orbit, and ``compromised`` -- True when
    at least one missing tracklet provably does not fit it (or could not be
    tested, which is not a licence to claim the designation either).

    Returns ``[]`` -- silently, this is never a reason to fail a search --
    when there are no designated pickups, when no source can answer for the
    archive, or when the database refuses.
    """
    if cfg is not None and not getattr(cfg, "pickup_check", True):
        return []
    picked = designated_pickups(itf_obs80, own_desigs)
    if not picked:
        return []
    db = _archive_source(source)
    if db is None:
        return []
    by_desig = {}
    unreadable = []
    for packed in picked:
        unpacked = unpack(packed)
        if unpacked:
            by_desig[unpacked] = packed
        else:
            unreadable.append(packed)
    if not by_desig:
        return [_unchecked(p, "its packed designation could not be read", log)
                for p in unreadable]
    try:
        rows = db.designation_obs(sorted(by_desig))
    except Exception as ex:                              # noqa: BLE001
        why = f"the archive query failed ({type(ex).__name__}: {ex})"
        return ([_unchecked(p, why, log) for p in sorted(by_desig.values())]
                + [_unchecked(p, "its packed designation could not be read", log)
                   for p in unreadable])
    if not rows:
        # The designations came OUT of archive rows, so an empty answer is not "nothing
        # to report" -- something between the query and the archive disagrees about how
        # they are spelled, and every one of them is now unverified.
        why = "the archive returned no observations under that designation"
        return ([_unchecked(p, why, log) for p in sorted(by_desig.values())]
                + [_unchecked(p, "its packed designation could not be read", log)
                   for p in unreadable])

    off_floor = getattr(cfg, "pickup_off_arcsec", 10.0) if cfg else 10.0
    off_sigma = getattr(cfg, "pickup_off_sigma", 5.0) if cfg else 5.0

    rows_by_desig = {}
    trksub_of = {}
    for r in rows:
        rows_by_desig.setdefault(r["provid"], []).append(r)
        if r.get("trksub"):
            trksub_of[r["trkid"]] = r["trksub"]

    out = []
    for unpacked, packed in sorted(by_desig.items()):
        lines = picked[packed]
        linked_ids = {obs_unique_id(l) for l in lines}
        linked_times = {_obs_time_key(l) for l in lines}
        tracklets = _tracklets_of(rows_by_desig.get(unpacked) or [])
        if not tracklets:
            # Rows came back, but none filed under THIS designation: the archive spells it
            # differently from `packing.unpack`. Nothing was compared, so this designation
            # is unverified -- which is a thing to say, not a thing to pass over.
            out.append(_unchecked(packed, "the archive's observations could not be "
                                          "matched to that designation", log))
            continue
        linked, missing = [], []
        for trkid, tlines in sorted(tracklets.items(),
                                    key=lambda kv: kv[1][0][15:32]):
            entry = _describe(trkid, tlines, trksub_of.get(trkid))
            if _is_linked(tlines, linked_ids, linked_times):
                linked.append(entry)
                continue
            entry["off_arcsec"], entry["sigma_arcsec"] = (
                _offset_arcsec(fit, obscodes, tlines)
                if fit is not None and obscodes is not None else (None, None))
            off, sig = entry["off_arcsec"], entry["sigma_arcsec"]
            if off is None:
                entry["fits"] = None
            else:
                entry["fits"] = off <= max(off_floor,
                                           off_sigma * (sig or 0.0))
            missing.append(entry)
        if not linked or not missing:
            continue
        rec = {"packed": packed, "desig": unpacked,
               "n_archive": len(tracklets),
               "linked": linked, "missing": missing,
               "compromised": any(m["fits"] is not True for m in missing)}
        rec["warning"] = warning_text(rec)
        out.append(rec)
    return out


def _tracklet_label(tracklet):
    off = tracklet.get("off_arcsec")
    where = f"{tracklet['date'][:10]} {tracklet['stn']}"
    if off is None:
        return f"{where} (not testable)"
    if off > 60.0:
        return f"{where} ({off / 3600.0:.2f} deg off)"
    return f"{where} ({off:.1f}\" off)"


def warning_text(rec):
    """The caution a person reads -- what was taken, what was left, what that
    means, and what to do about it."""
    n_link, n_miss = len(rec["linked"]), len(rec["missing"])
    took = ", ".join(f"{t['date'][:10]} {t['stn']}" for t in rec["linked"])
    left = ", ".join(_tracklet_label(t) for t in rec["missing"])
    if not rec["compromised"]:
        return (f"PARTIAL PICKUP: this linkage takes {n_link} of the "
                f"{rec['n_archive']} tracklets filed under {rec['desig']} "
                f"({took}); the other {n_miss} ({left}) fit the linked orbit "
                f"too and were simply never retrieved. The designation is "
                f"claimed whole, which is correct -- but the unretrieved "
                f"tracklets are not in this fit.")
    # A tracklet we could not TEST is not a tracklet that failed the test, and
    # saying so would accuse an innocent designation of being a mislinkage on
    # the strength of a measurement that never happened (the usual cause is a
    # satellite or roving observer, whose position this check cannot
    # reconstruct -- see _placeable).  The designation is still left out of the
    # document, because a claim we cannot check is not one to make; the
    # SENTENCE has to say which of the two it is.
    bad = [t for t in rec["missing"] if t.get("fits") is False]
    if not bad:
        return (f"NOT CLAIMED, COULD NOT CHECK: this linkage takes {n_link} of "
                f"the {rec['n_archive']} tracklets filed under {rec['desig']} "
                f"({took}), and the other {n_miss} ({left}) could not be "
                f"measured against the linked orbit -- an observer this check "
                f"cannot place (a satellite or a roving site), or an ephemeris "
                f"that would not compute. Nothing here says the designation is "
                f"wrong; it says we did not verify it, so it is left out of "
                f"the document rather than claimed unchecked (its observations "
                f"stay in the fit). Check those tracklets by hand.")
    # Deliberately NOT phrased as a refusal.  Whether the identification can
    # still be sent WITHOUT this designation is a separate question, answered
    # after this text is written (search._reduced_submittable), and a caution
    # that says "not submittable" on a linkage that was then submitted without
    # the designation reads as a contradiction -- which is how a real warning
    # stops being read.  This states what is true either way; the refusal, when
    # there is one, gets its own head warning.
    untested = [t for t in rec["missing"] if t.get("fits") is None]
    tail = ("" if not untested else
            f" ({len(untested)} further tracklet(s) could not be tested at "
            f"all: {', '.join(_tracklet_label(t) for t in untested)})")
    which = (f"the other {n_miss}" if len(bad) == n_miss
             else f"{len(bad)} of the other {n_miss}")
    return (f"MISLINKED DESIGNATION, NOT CLAIMED: this linkage takes {n_link} "
            f"of the {rec['n_archive']} tracklets filed under {rec['desig']} "
            f"({took}), and {which} "
            f"({', '.join(_tracklet_label(t) for t in bad)}) do NOT fit the "
            f"linked orbit{tail} -- so only part of {rec['desig']} is this "
            f"object. An "
            f"identification document names a WHOLE designation and cannot say "
            f"that, so {rec['desig']} is left out of the document entirely "
            f"(its observations stay in the fit). The split itself has to go "
            f"to the MPC as a Jira ticket, quoting the tracklets above.")


def banned_designations(records):
    """The packed designations a document must not claim."""
    return {r["packed"] for r in records if r["compromised"]}


def drop_designations(astrometry, banned):
    """``astrometry`` without the lines filed under any of ``banned`` -- the
    submission's view of the arc.  The FIT keeps every line; only the document
    is reduced (deliberately no refit: the orbit does not change materially and
    the user is looking at the whole thing)."""
    if not banned:
        return astrometry
    out = []
    for line in (astrometry or "").splitlines():
        if not line.strip():
            continue
        try:
            desig = line_designation(line)
        except ValueError:
            desig = None
        if desig in banned:
            continue
        out.append(line)
    return "".join(l + "\n" for l in out)
