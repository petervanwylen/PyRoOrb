"""Fit-refusal guards: two objects we must not fit (or must not perturb) because doing so
would propagate a body in a force field that already contains it -- the self-perturbation
paradox.

  * The Pluto ban -- minor planet (134340) Pluto is a DE440 perturber baked into our force
    model, so it cannot be fit unperturbed by itself. Refused outright, in every form a
    request can arrive (name / number / discovery designation / per-obs self-identification).
  * The self-perturbing asteroid guard -- the 16 massive main-belt asteroids in the sb441-n16
    perturber kernel. Unlike Pluto these are OPT-IN, so such an object is still fit normally
    with gravity; we just refuse to turn +16ast ON for it.

Factored out of fit_object as pure guard logic. Depends only on `re` and obs_sources's
designation unpacker -- never on fit_object -- so it stays a leaf. fit_object re-exports
these names for backward compatibility.
"""
import re

from pyroorb.obs_sources import ensure_unpacked


# ---------------------------------------------------------------------------
# Pluto ban
# ---------------------------------------------------------------------------
# Minor planet (134340) Pluto is one of the masses baked into the DE440 ephemeris
# that drives our force model (ASSIST integrates against it as a perturber). Fitting
# Pluto's own astrometry would therefore propagate it in a field that ALREADY
# contains Pluto -- it would be perturbed by itself, which is meaningless. So we
# refuse it outright, by every form a request can arrive in: its name, its number
# (134340), its discovery designation (1930 BM), and the per-obs designation that
# comes back from a fetch or a paste (in case the astrometry self-identifies as
# Pluto by number even when the box said something else).
PLUTO_BAN_MSG = ("Pluto is part of our core force model / perturbers and thus we are "
                 "unable to fit it unperturbed by itself.")
_PLUTO_NUMBER = 134340


class PlutoBanError(ValueError):
    """Raised when a fit is attempted on minor planet (134340) Pluto -- see is_pluto().

    Pluto is a DE440 perturber in our force model, so it cannot be fit unperturbed by
    itself. Carries the user-facing PLUTO_BAN_MSG as its message; the CLI prints it and
    the web layer turns it into a banner payload."""
    def __init__(self, msg=None):
        super().__init__(msg or PLUTO_BAN_MSG)


def is_pluto(desig):
    """True if `desig` denotes minor planet (134340) Pluto, in ANY form: the name
    'Pluto' (e.g. '134340 Pluto', '(134340) Pluto'), the bare number 134340, the
    OBS80/extended packed numbered form ('D4340', '00134340'), or the discovery
    provisional designation '1930 BM' (packed 'J30B00M'). Unrecognised / unrelated
    input returns False. Used to BAN Pluto from being fit (it is a DE440 perturber)."""
    if desig is None:
        return False
    s = str(desig).strip()
    if not s:
        return False
    # ensure_unpacked turns every packed numbered/provisional form into its readable
    # string ('D4340'->'134340', 'J30B00M'->'1930 BM'); raw names pass through.
    name = str(ensure_unpacked(s)).strip().lower()
    if "pluto" in name or "pluto" in s.lower():
        return True
    if name.replace(" ", "") == "1930bm":          # discovery designation
        return True
    # leading number: '134340', '(134340) ...', '0134340' (zero-padded, same rule as
    # perturber_number below -- a leading-zero form must not slip the ban).
    m = re.match(r"^\(?\s*0*(\d{1,7})\)?(?:\b|$)", name)
    if m and int(m.group(1)) == _PLUTO_NUMBER:
        return True
    return False


def _pluto_candidates(desig=None, meta=None, obs=None):
    """Yield every designation-ish string attached to a fit request, so a single
    is_pluto() sweep can catch Pluto whether it arrived as the typed designation, in
    `meta`, or as a per-obs label/packed code from a fetch or paste."""
    yield desig
    if meta:
        yield meta.get("designation")
        for d in (meta.get("designations") or ()):
            yield d
    for o in (obs or ()):
        yield o.get("designation")
        yield o.get("packed")


def assert_not_pluto(desig=None, meta=None, obs=None):
    """Raise PlutoBanError if any part of this request (typed designation, meta, or
    the fetched/pasted observations) identifies minor planet (134340) Pluto."""
    if any(is_pluto(c) for c in _pluto_candidates(desig, meta, obs)):
        raise PlutoBanError()


# ---------------------------------------------------------------------------
# Self-perturbing asteroid guard  (the "+16ast" analogue of the Pluto ban)
# ---------------------------------------------------------------------------
# The 16 massive main-belt asteroids in the sb441-n16 perturber kernel ARE the bodies
# refine_asteroids / refit_force_model add to the force model. If the object being fit
# is itself one of those 16, turning the perturbers on would propagate it in a field
# that already contains it -- it would be perturbed by itself, exactly the Pluto
# paradox. Unlike Pluto (a DE440 perturber that is ALWAYS present, so it is banned from
# being fit at all), the +16ast perturbers are OPT-IN, so these objects are still fit
# normally with gravity -- we just refuse to turn +16ast ON for them.
#
# Number -> name, verified against ephem/sb441-n16.bsp (the SPK targets are 2000000 +
# MP number); tests/test_asteroid_self_perturb.py re-checks this constant against the
# real kernel when it is present, so a future kernel swap can't silently drift it.
N16_PERTURBERS = {
    1: "Ceres",   2: "Pallas",   3: "Juno",     4: "Vesta",
    7: "Iris",    10: "Hygiea",  15: "Eunomia",  16: "Psyche",
    31: "Euphrosyne", 52: "Europa", 65: "Cybele", 87: "Sylvia",
    88: "Thisbe", 107: "Camilla", 511: "Davida", 704: "Interamnia",
}
_N16_BY_NAME = {v.lower(): k for k, v in N16_PERTURBERS.items()}
_N16_BY_NAME["hygeia"] = 10        # common alternate spelling of 10 Hygiea


def perturber_number(desig):
    """If `desig` denotes one of the 16 sb441-n16 perturber asteroids, return its MP
    number; else None. Matches the bare/parenthesised number ('4', '(4)'), the
    number+name ('4 Vesta', '(4) Vesta'), the name alone ('Vesta'), and the packed
    numbered forms ('00004' -> '4') via ensure_unpacked. The match is EXACT on the
    number (or a known name), so near-miss numbers ('40', '1073') and unrelated
    provisional designations ('2004 AB' -> leading 2004) are spared."""
    if desig is None:
        return None
    s = str(desig).strip()
    if not s:
        return None
    name = str(ensure_unpacked(s)).strip().lower()
    m = re.match(r"^\(?\s*0*(\d{1,7})\)?(?:\b|\s|$)", name)   # leading MP number
    if m:
        n = int(m.group(1))
        if n in N16_PERTURBERS:
            return n
    for token in (name, name.rsplit(" ", 1)[-1]):            # bare or trailing name
        if token in _N16_BY_NAME:
            return _N16_BY_NAME[token]
    return None


def self_perturber_name(desig=None, meta=None, obs=None, fit=None):
    """If this fit request identifies one of the 16 sb441-n16 perturber asteroids -- in
    ANY form it can arrive in (typed designation, meta, a FittedOrbit's designations, or
    a per-obs designation/packed code that self-identifies by number) -- return its
    display name ('4 Vesta'); else None. Used to refuse turning +16ast on for it."""
    cands = list(_pluto_candidates(desig, meta, obs))         # same desig-ish harvest
    if fit is not None:
        cands += list(getattr(fit, "designations", []) or [])
        cands.append(getattr(fit, "designation", None))
        for o in (getattr(fit, "_obs", None) or ()):
            cands.append(o.get("designation"))
            cands.append(o.get("packed"))
    for c in cands:
        n = perturber_number(c)
        if n is not None:
            return f"{n} {N16_PERTURBERS[n]}"
    return None
