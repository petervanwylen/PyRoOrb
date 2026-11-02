"""Native MPC packed-designation pack/unpack for asteroids.

WHY THIS EXISTS — we used to reach sbpy (``sbpy.data.names.Names.to_packed`` /
``from_packed``) to convert designations.  sbpy is correct, but ``import sbpy.data``
costs ~600 ms (cold) and was paid once per fit: ``records_to_obs`` packs every record's
designation, dragging the whole sbpy import into every fit process.  The conversion
itself is a few lines of base-62 arithmetic with no dependencies, so we do it here and
never import sbpy on the hot path.  This module (plus ``pack_comet`` for comet-form
provisionals) now covers every form the engine packs/unpacks, so **sbpy is no longer a
project dependency** — it is used only as an OPTIONAL cross-check inside ``_validate``.

THE SCHEME (https://minorplanetcenter.net/iau/info/PackedDes.html) — three families:

  * a catalogue NUMBER:   <100000 → 5 digits ('00433'); <620000 → base-62 high digit
    + 4 digits ('A0000'=100000); larger → '~' + 4 base-62 digits ('~0000'=620000).
  * a PROVISIONAL designation 'YYYY H1H2n':  century letter (I/J/K = 18/19/20xx) + 2-digit
    year + half-month letter H1 + the cycle number n packed into the order letter H2.  For
    n ≤ 619 the cycle is two chars (base-62 high digit + decimal low) giving the classic
    7-char form 'K04TI6D'; for the busier modern half-months (n ≥ 620) MPC switched to the
    '_'-prefixed extended form, where (n, H2) are folded into one base-62 counter.
  * a SURVEY designation (Palomar–Leiden / Trojan):  '2040 P-L' ↔ 'PLS2040', 'T-1/2/3'.

Both directions return ``None`` for anything they don't recognise, so callers can fall
back to sbpy — this module can only ever AVOID a sbpy call, never answer wrongly for one.

PROVENANCE — this is our own implementation; sbpy's ``names.py`` (BSD-3) ``to_packed`` /
``from_packed`` were read as the reference for the obj_num folding in the extended scheme
and the comet-form provisional packing (``pack_comet``).  Logic ported, no source lines
copied.  Validated byte-for-byte (both directions) against sbpy across
the full MPCORB catalogue (1,548,690 designations, 0 mismatches): run ``python packing.py``
(cross-checks against sbpy when it is installed, else round-trips against itself).
"""
import re

# base-62 digit alphabet (0-9, A-Z, a-z); and the 25-letter order alphabet (A-Z minus 'I')
_B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_VAL = {c: i for i, c in enumerate(_B62)}
_ORDER = "ABCDEFGHJKLMNOPQRSTUVWXYZ"          # half-month order letters, no 'I'
_HALFMONTH = "ABCDEFGHJKLMNOPQRSTUVWXY"       # half-month letters, no 'I', no 'Z'
_CENTURY = "IJK"                              # 1800s / 1900s / 2000s

# The two letter slots are NOT the same alphabet, and writing them as one was a
# real defect: the HALF-MONTH letter runs A-Y omitting 'I' (24 letters for 24
# half-months -- "I is omitted and Z is unused", MPC OldDesDoc), while the ORDER
# letter within a half-month runs A-Z omitting 'I' (25 -- 'Z' is the 25th and is
# perfectly legal, 25770 of them in MPCORB).  Accepting 'Z' in the half-month
# slot let ``pack`` emit a token ``unpack`` refuses to read back, e.g.
# pack("2020 ZA") -> 'K20Z00A' -> unpack -> None: a one-way trip whose output
# reaches OBS80 lines and submission documents.  Comets share the minor planets'
# half-month scheme verbatim (IAU 1994 resolution), so the comet form is bounded
# the same way.  Confirmed over all 1548555 MPCORB entries: zero 'I' and zero 'Z'
# in the half-month slot, packed and readable alike.
_PROV_RE = re.compile(r"^(\d{4}) ([A-HJ-Ya-hj-y])([A-HJ-Za-hj-z])(\d*)$")  # year H1 H2 num
_COMET_PROV_RE = re.compile(r"^(\d{4}) ([A-HJ-Ya-hj-y])(\d+)$")             # year H num (comet form)
_SURVEY_PACKED_RE = re.compile(r"^(PLS|T[123]S)\d{4}$")                    # '2040 P-L' / 'T-1'


def _enc(n, width):
    """`n` as a fixed-width base-62 string (high digit first)."""
    out = []
    for _ in range(width):
        n, r = divmod(n, 62)
        out.append(_B62[r])
    return "".join(reversed(out))


def _dec(s):
    """Base-62 string → int."""
    n = 0
    for c in s:
        n = n * 62 + _VAL[c]
    return n


# --------------------------------------------------------------------------------- pack
def pack(desig):
    """MPC packed form of an asteroid number / provisional / survey designation, else None."""
    s = str(desig).strip()
    if s.isdigit():
        n = int(s)
        if n < 100000:
            return "%05d" % n
        if n < 620000:
            return _B62[n // 10000] + "%04d" % (n % 10000)
        if n < 620000 + 62 ** 4:
            return "~" + _enc(n - 620000, 4)
        return None
    if s.endswith("P-L"):
        return "PLS" + s[:4]
    if s[-3:] in ("T-1", "T-2", "T-3"):
        return "T" + s[-1] + "S" + s[:4]
    m = _PROV_RE.match(s)
    if not m:
        return None
    # half-month letters are upper-case in the packed form; normalising here keeps
    # pack/unpack a true round trip for a lower-case-typed designation
    year, h1, h2 = int(m.group(1)), m.group(2).upper(), m.group(3).upper()
    num = int(m.group(4)) if m.group(4) else 0
    if not 1800 <= year < 2100:
        return None
    if num <= 619:                                       # classic 7-char form
        return "%s%02d%s%s%d%s" % (_CENTURY[year // 100 - 18], year % 100,
                                   h1, _B62[num // 10], num % 10, h2)
    if year < 2000:                                      # extended form is 20xx only
        return None
    counter = num * 25 + _ORDER.index(h2) - 15500        # fold (num, H2) → one counter
    return "_" + _B62[year % 100] + h1 + _enc(counter, 4)


# -------------------------------------------------------------------------- pack (comet)
def pack_comet(core):
    """MPC packed 7-char core of a COMET-form provisional designation 'YYYY Hn' --
    a single half-month letter H immediately followed by the discovery number n,
    e.g. '2023 A3' -> 'K23A030'. (Two-letter asteroid-style cores like '2023 JN16'
    are handled by pack(); the comet form is exactly the shape pack() returns None
    for.) The comet TYPE letter (C/P/D/...) is NOT prepended here -- the caller adds
    it. Returns None for anything that isn't this comet form.

    The number n is packed into 2 chars the same base-62-high way as an asteroid
    cycle count (n//10 -> a base-62 digit, n%10 -> a decimal digit), and the trailing
    fragment slot is the no-fragment marker '0'."""
    m = _COMET_PROV_RE.match(str(core).strip())
    if not m:
        return None
    year, half, num = int(m.group(1)), m.group(2).upper(), int(m.group(3))
    if not 1800 <= year < 2100 or num > 619:             # n>619 overflows the 2-char field
        return None
    return "%s%02d%s%s%d0" % (_CENTURY[year // 100 - 18], year % 100,
                              half, _B62[num // 10], num % 10)


# ------------------------------------------------------------------------------- unpack
def unpack(packed):
    """Readable designation (as str) for a packed asteroid identifier, else None.

    STRICT: every character is checked against the slot it occupies, so the only
    7-character tokens that unpack are ones ``pack`` could have produced.  That
    strictness is the point -- callers (notably ``itf.linkage``) use "does it
    unpack?" as the test that separates a real packed designation from an ITF
    **trksub**, and a lax reader turns a trksub into a designation the pipeline
    then submits.  'J053207' is a live SBN trksub: without the slot checks it
    read as '1905 3720' (half-month letter '3', order letter '7' -- neither is a
    letter at all) and was named as a designation in an identification document.
    """
    p = str(packed)
    if p.isdigit():                                      # '00433'
        return str(int(p))
    if len(p) == 5 and p[0] in _VAL and _VAL[p[0]] >= 10 and p[1:].isdigit():
        return str(_VAL[p[0]] * 10000 + int(p[1:]))      # 'A0000' → 100000
    if p[:1] == "~" and len(p) == 5 and all(c in _VAL for c in p[1:]):
        return str(620000 + _dec(p[1:]))
    if _SURVEY_PACKED_RE.match(p):
        if p[0] == "P":
            return p[3:] + " P-L"
        return p[3:] + " T-" + p[1]
    if len(p) == 7 and p[0] in _CENTURY:
        if not (p[1:3].isdigit() and p[3] in _HALFMONTH and p[4] in _VAL
                and p[5].isdigit() and p[6] in _ORDER):
            return None                                  # trksub / comet core, not a designation
        year = (18 + _CENTURY.index(p[0])) * 100 + int(p[1:3])
        num = _VAL[p[4]] * 10 + int(p[5])
        return "%d %s%s%s" % (year, p[3], p[6], str(num) if num else "")
    if len(p) == 7 and p[0] == "_":                      # extended provisional
        if not (p[1] in _VAL and p[2] in _HALFMONTH
                and all(c in _VAL for c in p[3:7])):
            return None
        counter = 15501 + _dec(p[3:7])
        h2 = _ORDER[(counter - 1) % 25]
        num = (counter - 1) // 25
        return "20%02d %s%s%d" % (_VAL[p[1]], p[2], h2, num)
    return None


def unpack_comet(core):
    """Readable core 'YYYY Hn' of a packed COMET-form provisional core, else None --
    the inverse of :func:`pack_comet`.  The comet TYPE letter (col 5 of an OBS80 line)
    is NOT part of the core; the trailing slot is the fragment ('0' = none, else a
    lowercase letter), which the readable core does not carry.

    A comet core can never be mistaken for an asteroid packing: its last character is
    '0' or lowercase, where an asteroid packing carries an uppercase order letter.
    """
    p = str(core)
    if not (len(p) == 7 and p[0] in _CENTURY and p[1:3].isdigit()
            and p[3] in _HALFMONTH and p[4] in _VAL and p[5].isdigit()
            and (p[6] == "0" or p[6].islower())):
        return None
    year = (18 + _CENTURY.index(p[0])) * 100 + int(p[1:3])
    num = _VAL[p[4]] * 10 + int(p[5])
    if not num:
        return None
    return "%d %s%d" % (year, p[3], num)


# --------------------------------------------------------------------------- validation
def _validate(limit=None):
    """Validate native pack/unpack over the full MPCORB. Run as ``python packing.py``
    (optional row cap).

    sbpy is no longer a project dependency, so this adapts to what's installed:
      * sbpy PRESENT  -> byte-for-byte cross-check against ``Names.from_packed`` /
        ``to_packed`` (the gold standard the native code was ported to match);
      * sbpy ABSENT   -> self-consistency round-trip: ``pack(unpack(packed))`` must
        reproduce every packed designation in the catalogue.
    Both modes must finish with zero mismatches."""
    import gzip
    from pyroorb import mpcorb
    try:
        from sbpy.data import Names
    except ImportError:
        Names = None
    mode = "sbpy cross-check" if Names is not None else "round-trip self-check (sbpy absent)"
    mpcorb.ensure_mpcorb(verbose=False)
    n_ok = n_checked = 0
    mism = []
    with gzip.open(mpcorb.MPCORB_GZ, "rt", errors="replace") as f:
        started = False
        for ln in f:
            if not started:
                started = ln.startswith("----------")
                continue
            packed = ln[0:7].strip()
            if not packed or len(ln) < 100:
                continue
            n_checked += 1
            if limit and n_checked > limit:
                break
            got = unpack(packed)
            if Names is not None:
                try:
                    want = str(Names.from_packed(packed))
                except Exception:
                    want = None
                if got != want:
                    mism.append(("unpack", packed, got, want))
                    continue
                n_ok += 1
                if want is not None:                      # round-trip the readable form
                    try:
                        want_p = str(Names.to_packed(want))
                    except Exception:
                        want_p = None
                    if pack(want) != want_p:
                        mism.append(("pack", want, pack(want), want_p))
            else:
                if got is None or pack(got) != packed:    # native must round-trip itself
                    mism.append(("roundtrip", packed, got, pack(got) if got else None))
                    continue
                n_ok += 1
    print(f"[{mode}] checked={n_checked} native_ok={n_ok} mismatches={len(mism)}")
    for kind, inp, got, want in mism[:25]:
        print(f"  {kind}: {inp!r} -> {got!r} != {want!r}")
    return not mism


if __name__ == "__main__":
    import sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(0 if _validate(lim) else 1)
