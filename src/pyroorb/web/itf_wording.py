"""Human wording for the ITF search's internal numbers, for the web UI ONLY.

The engine's TEXT (trace lines, reasons, the stored vetting record) still
speaks in two numeric shorthands that people cannot read:

* **fractional opposition counts** -- an ``OppProfile`` prints 1.5 for two
  oppositions of which only one has 2+ nights, and 1.75 for three oppositions
  of which only one does.  Both mean "more than one apparition, but too thin an
  opposition profile to submit".
* **the legacy status code** -- 1 = submittable linkage, 0.6 = a possible
  linkage that is never submittable, 0 = nothing found.  The engine now
  carries a :class:`~itf.verdict.SearchVerdict`; the number survives in the
  live trace text and in stored records.

Neither is interpretable to somebody reading a banner or the live trace, so
everything the browser is shown passes through here first: the numbers are
replaced by the sentence they stand for.  Nothing in the engine changes --
the CLI keeps its terse output, and this module is a presentation layer over
the strings on their way out of ``engine_api``.
"""

import re

from pyroorb.itf.verdict import SearchVerdict

__all__ = ["outcome_label", "opp_text", "humanize", "humanize_all",
           "humanize_vetting"]


# --- the two shorthands, spelled out ---------------------------------------

#: search verdict -> what it means to a person
_OUTCOMES = {
    SearchVerdict.LINKED: "submittable linkage",
    SearchVerdict.POSSIBLE: "possible linkage (not submittable)",
    SearchVerdict.NONE: "no linkage",
}

#: fractional opposition count -> (whole count, what makes the profile thin)
_OPPS = {
    "1.5": ("2", "1 single-night", "one of them a single night"),
    "1.75": ("3", "2 single-night", "two of them single nights"),
}


def outcome_label(verdict):
    """Readable wording for a search :class:`~itf.verdict.SearchVerdict` (a
    stored legacy 1 / 0.6 / 0 is read as the verdict it stands for)."""
    if not isinstance(verdict, SearchVerdict):
        try:
            verdict = SearchVerdict.from_legacy(verdict)
        except (TypeError, ValueError):
            return "no linkage"
    return _OUTCOMES[verdict]


def opp_text(value, compact=True):
    """Readable opposition count.  Whole counts pass through as plain
    integers; the fractional 1.5 / 1.75 profiles say what they really are."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    key = "1.5" if f == 1.5 else "1.75" if f == 1.75 else None
    if key is None:
        return str(int(f)) if f == int(f) else str(f)
    whole, short_note, long_note = _OPPS[key]
    return ("%s (%s)" % (whole, short_note) if compact
            else "%s oppositions, %s" % (whole, long_note))


_FRACT = r"1\.75|1\.5"


def _parts(num):
    return _OPPS["1.75" if float(num) == 1.75 else "1.5"]


# Rules run in order.  Each one only fires where the number is unambiguously an
# opposition count or a status code -- a bare "1.5" elsewhere in a trace line is
# an rms, a U or a sigma, and must reach the reader exactly as the engine wrote
# it.
_RULES = (
    # the thin-profile upgrade: "1.75 opps -> 3" / "treating 1.75 opps as 3"
    (re.compile(r"\b(%s)\s+opps?\s*(->|as)\s*(\d+)" % _FRACT),
     lambda m: "the thin %s-opposition profile (%s) %s %s full oppositions"
               % (_parts(m.group(1))[0], _parts(m.group(1))[1],
                  "->" if m.group(2) == "->" else "as", m.group(3))),
    # "1.75-opposition profile"
    (re.compile(r"\b(%s)\s*-\s*opposition\b" % _FRACT),
     lambda m: "%s-opposition (%s)" % (_parts(m.group(1))[0],
                                       _parts(m.group(1))[1])),
    # "1.75 opps" / "1.5 oppositions" -- parenthesised note, unless the phrase
    # is ITSELF already inside brackets (no nested parens on a banner)
    (re.compile(r"\b(%s)\s+(opps?|oppositions?)\b" % _FRACT),
     lambda m: ("%s %s, %s" if m.string[max(0, m.start() - 1):m.start()] == "("
                else "%s %s (%s)")
               % (_parts(m.group(1))[0], m.group(2), _parts(m.group(1))[1])),
    # "opps 1 -> 1.75", "opps 1.5, nights 3"
    (re.compile(r"\b(opps?|oppositions?)\s+(\d+(?:\.\d+)?)"
                r"(?:\s*->\s*(\d+(?:\.\d+)?))?"),
     lambda m: "%s %s%s" % (m.group(1), opp_text(m.group(2)),
                            (" -> " + opp_text(m.group(3))) if m.group(3)
                            else "")),
    # "1.75 refused", "the 1.5 profile", "the fragile 1.5 hunch" ("shape" is
    # what the engine wrote before 2026-10-01, and a stored reason still does)
    (re.compile(r"\b(%s)(?=\s+(?:refused|profile|shape|hunch|tier))" % _FRACT),
     lambda m: "%s-opposition (%s)" % (_parts(m.group(1))[0],
                                       _parts(m.group(1))[1])),
    # "status 0.6 -- ...", "status: 1"
    (re.compile(r"\bstatus[:=]?\s+(1(?:\.0)?|0\.6|0)\b"),
     lambda m: "outcome " + outcome_label(m.group(1))),
)


def humanize(text):
    """Rewrite one string so no unexplained 1.5 / 1.75 / 0.6 reaches a reader."""
    if not text or not isinstance(text, str):
        return text
    for rx, repl in _RULES:
        text = rx.sub(repl, text)
    return text


def humanize_all(lines):
    """:func:`humanize` over a sequence, preserving order and length."""
    return [humanize(x) for x in (lines or [])]


#: vetting-dict keys holding a raw opposition count
_OPP_KEYS = ("opps", "opps_pre", "opps_post")


def humanize_vetting(vetting):
    """A search's vetting dict with its opposition counts spelled out and its
    free text rewritten (numbers elsewhere -- rms, U, counts -- untouched)."""
    out = {}
    for key, val in (vetting or {}).items():
        if key in _OPP_KEYS:
            out[key] = opp_text(val)
        elif isinstance(val, str):
            out[key] = humanize(val)
        else:
            out[key] = val
    return out
