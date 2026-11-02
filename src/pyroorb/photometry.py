"""Absolute-magnitude (H) estimation + photometric residuals — ported from find_orb.

⚠️ PROVENANCE: originally ported from Bill Gray's find_orb, read from its source:
  - phase function / magnitude reduction  `calc_obs_magnitude`  (elem_out.cpp:3644)
  - band -> V conversion, the LOOKUP shape `mag_band_shift`       (elem_out.cpp:3733)
  - H fit (weighted mean of implied H)     `_calc_absolute_magnitude_internal` (3772)
GPL-2.0 logic, no source lines copied; this project is GPL-3.0.

The band VALUES are no longer find_orb's. find_orb ports the MPC's static
BandConversion.txt (last touched 2021); we take the MPC's live mag-band service
instead, which disagrees with that file on seven survey bands -- two of them in
sign. See BAND_SHIFT below for the table, the diff, and the reasoning.

Why band conversion matters (the user's point): a reported magnitude is only as
good as the filter/reference it came from. An unfiltered 'C' point, a Pan-STARRS
'w', a Gaia 'G', and an infrared 'K' are NOT the same magnitude — each must be
shifted to a common V band before you can average them into one H. Skipping this
biases H by up to ~1-2 mag for some bands.
"""
import math

# Asteroid HG slope parameter (Bowell et al. 1989); find_orb default.
ASTEROID_G = 0.15
# Default photometric 1-sigma if an obs carries no rmsMag (mag).
DEFAULT_MAG_SIGMA = 0.5

# Band -> additive correction to reach V (V = reported_mag + shift).
#
# SOURCE: the MPC's LIVE magnitude-band service, not the static file find_orb
# ported. Fetched 2026-08-03 from
#   GET https://data.minorplanetcenter.net/api/mag-band
#       Content-Type: application/json, body {"band": "all"}   -> 57 bands
# (documented at https://docs.minorplanetcenter.net/mpc-ops-docs/apis/mag-band/).
# THIS TABLE IS NOT UNDER TEST, AND NOT UNDER THE MPC'S CONTROL. Taking a live
# service as the source raises the obvious question of what happens when the
# service moves, and the answer here is deliberately the passive one: nothing,
# until someone looks. The check is a read-only re-pull of all 57 bands, diffed
# against this table on demand, when you are thinking about it.
#
# What it is NOT: a test that fails on disagreement (that hands the schedule to
# the MPC, so their revision breaks our build on their timetable, and the
# pressure becomes making red go away rather than judging the number), and not a
# script that rewrites the table (a calibration value should change because
# somebody read the disagreement and decided). `tests/test_band_table.py`
# accordingly asserts the LOOKUP only -- full-code-wins, filter fallback, the
# blank/UNK/unrecoverable routes -- and never any value here, so every number
# below can be edited without touching a test.
#
# Update BAND_TABLE_FETCHED when you take a change.
#
# Why we left BandConversion.txt behind. That file (Last-Modified 2021-10-07) is
# what find_orb's mag_band_shift ports, and we matched it exactly. It disagrees
# with the live service on seven bands -- every one of them in the Sloan/survey
# family, with the Johnson-Cousins heritage bands identical in both:
#
#     band   BandConversion.txt   API      delta
#      u          +2.50          -2.50     -5.00   <- sign
#      w          -0.13          +0.16     +0.29   <- sign
#      z          +0.26          +0.37     +0.11
#      r          +0.14          +0.23     +0.09
#      i          +0.32          +0.39     +0.07
#      g          -0.35          -0.28     +0.07
#      y          +0.32          +0.36     +0.04
#
# The two sign flips are why we trust the service over the file rather than
# splitting the difference. The old table contradicts ITSELF: it puts Sloan u
# (355 nm) at +2.50 while putting Johnson U (365 nm) at -1.30 -- 3.8 mag apart
# for filters ten nanometres apart, on opposite sides of V. No asteroid is
# brighter in the near-UV than in V. Likewise its w = -0.13 places Pan-STARRS w
# (effective ~608 nm) blueward of V, though the filter spans 400-820 nm. Sort
# either table by effective wavelength and the correction should rise redward:
# the legacy one breaks that at u->U and V->w, the API one breaks neither. (Both
# break R->o and I->z, harmlessly -- ATLAS o is a wide filter effectively bluer
# than its nominal centre, and asteroid reflectance flattens past ~900 nm.)
# No MPC announcement declares the service authoritative; this is an inference
# from internal consistency.
#
# KEYED BY THE FULL ADES BAND, deliberately. 'Lg' is looked up as 'Lg' -- the
# collapse to a single character is NOT applied on the way in. Today the MPC
# publishes v_conversion=null for all 33 of its two-character codes (every LSST,
# Pan-STARRS, Sloan, ATLAS, Gaia BP/RP, Johnson-Cousins and DECam entry), so
# those still RESOLVE through the filter letter -- see band_shift -- but the day
# the MPC populates 'Lg' it is one table edit and every call site improves, with
# no code change and nothing to un-collapse. A Sloan r, a Pan-STARRS r and an
# LSST r really do differ; the table can now say so.
#
# THE BLANK BAND IS ZERO, and that follows the MPC rather than deviating from it.
# The legacy file carries an explicit "blank -> -0.8" row (no band stated meant a
# photographic plate, i.e. ~B). find_orb replaced it with +0.43 (a modern
# unfiltered CCD reads ~R). The live service does neither: it has NO blank band at
# all. The nearest thing is 'UNK' -- "Unknown" -- with v_conversion null. So the
# MPC did not revise the number, it WITHDREW it: a magnitude whose band nobody
# stated has no conversion, and null is not a value.
#
# MEASURED here, 2026-08-04, and the archive says the
# MPC is right to refuse. Over 29 408 blank-band observations of 376 objects with
# a simultaneous partner in a solved band, the implied blank shift is +0.14
# weighting per observation, -0.56 per object, -0.06 per station; DePhOCUS gets
# -0.037. No route reaches +0.43. And the 0.7 mag swing between those weightings
# IS the finding -- the blank band is not a bandpass, it is a per-observatory zero
# point: 645 Apache Point-SDSS -0.80, 705 Apache Point -0.63, 704 LINEAR -0.26,
# 644 NEAT -0.03, W84 Cerro Tololo-DECam +0.60, station-to-station sd 0.43 over a
# 1.4 mag range. That spread is larger than any correction in this whole table, so
# no constant describes the population and the honest constant is the one that
# claims nothing.
#
# 0.0 also makes the unknown-band cases agree: ' ', 'UNK', 'IR', 'VR' and 'W2'-'W4'
# now all mean "we cannot say what this magnitude is on" and all apply no shift --
# where before, blank and UNK alone were charged an R-band correction. Set
# BLANK_BAND_SHIFT = 0.43 to restore find_orb's behaviour, -0.8 for the legacy
# file's. NOTE our own measurement would say ~-0.5; we do not use it, because it
# is a statement about which observatories dominate our archive, not about physics.
#
# u is where the two tables differ most and the argument above is a priori, so it
# was checked TWO independent ways, and both say the same thing.
#
# MEASURED here, 2026-08-03: Rubin observes an object in Lu and in Lg/Lr/Li on the
# same nights, so its u-to-other colour falls out almost independently of the orbit
# (2025 OC430's Lu and Lg points are TEN MINUTES apart). Over the three designated
# objects with Lu photometry, their own other bands imply -1.50 +/- 0.09 for Lu --
# right sign, and 0.2 mag from Johnson U (-1.30), the adjacent filter, which is
# where Rubin's u (367.1 nm) belongs. It is NOT a measurement of the MPC's bare 'u'
# (SDSS u', 356.2 nm); that 1 mag gap is real, and it is why the LSST bands are
# tabulated in their own right below rather than left to the prefix strip.
#
# PUBLISHED: DePhOCUS (arXiv:2408.07474) measures the generic u at -2.436 +/- 0.045
# over 20 observations of 4 asteroids -- 1.4 sigma from the service's -2.50 and
# ~5 mag from the legacy +2.50. Its Table 1 also shows this whole table IS the one
# NEOCC and JPL already run, agreeing on all 22 shared bands. The legacy file is
# the outlier; the sign question is settled.
#
# THE MAGNITUDE WAS QUESTIONED AND THE PUBLISHED VALUE KEPT (a deliberate
# call). We measured u ourselves and got -1.937 +/- 0.042 over 41 objects
# and 1025 same-epoch pairs -- 0.5 mag off both published values -- and briefly
# baked -1.9 before reverting. The reasoning on both sides is worth keeping,
# because this is the closest the table has come to a deviation on evidence.
#
# FOR OURS: the value is unmoved by widening the pairing window 0.02 -> 0.3 d and
# consistent across every partner band independently (g -1.98, r -1.98, G -1.96,
# w -1.94, R -1.90), and those partners span BOTH photometric systems, which an
# AB/Vega mixup could not survive (0.90 mag for u alone). Detection-limit bias is
# measured and flat: >2 mag above the station's u limit gives -1.966, 0.5-1.25
# above gives -1.982. And the physics: with (u-g)_sun = 1.28 (Willmer 2018, AB)
# and the SDSS Moving Object Catalog's asteroid u reflectance R_u = 0.6-0.95
# (Hasselmann 2017), s(u) is bounded to [-2.07, -1.66]. -2.50 needs R_u = 0.42.
# A solar-colour check shows every OTHER entry in this table is the
# solar colour plus a small reddening term, and u alone is 0.92 mag off it.
#
# AGAINST: two independent sources agree on ~-2.44. DePhOCUS (arXiv:2408.07474)
# Table 2 measures -2.436 +/- 0.045 by a method with no failure mode in common
# with ours, and the service publishes -2.50. Our measurement is one telescope
# (41 of 42 u objects are WFST Lenghu), so it may be that observatory's u zero
# point rather than the u bandpass -- exactly the failure mode the blank band
# demonstrates two paragraphs up. The reflectance bound is the only
# observatory-independent leg and its lower edge is read off a published figure's
# axis, which is not a strong enough foundation to outvote two published values.
#
# So: -2.50 stands, the disagreement is documented rather than baked. What would
# settle it: a second telescope's u (ours has one), a larger u sample than
# DePhOCUS's 4 asteroids / 20 observations (theirs has one), or the SDSS MOC's
# actual R_u distribution rather than a figure's axis limits.
#
# V-equivalent bands (V, N, T, j, v) take 0.
BAND_TABLE_FETCHED = "2026-08-03"
BAND_TABLE_SOURCE = "https://data.minorplanetcenter.net/api/mag-band"
_V_EQUIV = set("VNTjv")
BLANK_BAND_SHIFT = 0.0           # no band stated -> no conversion (find_orb: +0.43)
BAND_SHIFT = {
    " ": BLANK_BAND_SHIFT,
    # --- single-character bands the service publishes a conversion for -------
    "B": -0.8,  "c": -0.05, "C": 0.4,   "g": -0.28, "G": 0.28,
    "H": 1.4,   "i": 0.39,  "I": 0.8,   "J": 1.2,   "K": 1.7,   "L": 0.2,
    "o": 0.33,  "r": 0.23,  "R": 0.4,   "u": -2.5,  "U": -1.3,  "W": 0.4,
    "w": 0.16,  "y": 0.36,  "Y": 0.7,   "z": 0.37,
    "v": 0.0,   "V": 0.0,
    # --- LSST/Rubin, the one system with a PUBLISHED filter-specific table ----
    # Veres & Chesley (2017), "High-fidelity Simulations of the Near-Earth Object
    # Search Performance of the LSST" (arXiv:1706.09398), Table 4: the magnitude
    # transformation from Johnson V to the LSST filter system, computed from the
    # LSST bandpasses with DeMeo et al. (2009) mean reflectance spectra and the
    # Chance & Kurucz (2010) solar spectrum. Their V-x IS our additive shift.
    #
    #             V-u     V-g     V-r    V-i    V-z    V-y
    #   C group  -1.614  -0.302  0.172  0.291  0.298  0.303
    #   S group  -1.927  -0.395  0.255  0.455  0.401  0.406
    #
    # We bake the C/S mean, the same 50:50 split the paper adopts for its own NEA
    # population (Stuart & Binzel 2004), because a single average correction is
    # what a band table can express -- the taxonomy of the object in hand is
    # exactly the thing we do not know.
    #
    # This is why BAND_SHIFT is keyed on the full ADES band. Falling back through
    # the bare letter would charge LSST g the generic g and so on -- within 0.02-
    # 0.07 mag for g/r/i/z/y, but 0.47 mag out for u, whose filter (367.1 nm) is
    # nowhere near the MPC's bare 'u' (SDSS u', 356.2 nm). MEASURED independently
    # on the three designated objects carrying Lu photometry -- Rubin takes Lu and
    # Lg/Lr/Li on the same nights, 2025 OC430's ten minutes apart -- their own
    # other bands imply -1.50 +/- 0.09 for Lu, which sits 0.11 mag from the C-group
    # value. Re-measured 2026-08-04 on 12 objects and 253 pairs: -1.669 +/- 0.101,
    # i.e. within 1 sigma of the tabulated -1.771, so Lu stands as published --
    # unlike the bare 'u' above, Rubin's u needed no revision.
    "Lu": -1.771, "Lg": -0.349, "Lr": 0.214,
    "Li": 0.373,  "Lz": 0.350,  "Ly": 0.355,
}

# Bands whose value is NOT the MPC's -- so the live drift test knows not to expect
# the service to agree. Today: the LSST six above, which the MPC leaves null.
NON_MPC_SOURCED = frozenset({"Lu", "Lg", "Lr", "Li", "Lz", "Ly"})

# Bands where we deliberately differ from a value the service DOES publish, mapped
# to what it published when we chose. Empty today -- `u` was the one candidate and
# the deviation was withdrawn (see above). Kept because the live drift test reads
# it: NON_MPC_SOURCED exempts a band from comparison entirely, so any future
# standing disagreement has to be recorded HERE or it goes unwatched.
MPC_PUBLISHES = {}

# Bands the MPC NAMES but publishes no conversion for (v_conversion=null). Listed
# explicitly so the fallback below is a recorded gap rather than a silent miss,
# and so a refresh that starts populating them shows up as a diff. 'N'/'T' are the
# comet magnitude descriptors (not allowed in new submissions) and are handled by
# _V_EQUIV; 'IR', 'W2'-'W4', 'Lx' and 'UNK' have no filter letter to fall back on
# and end up unadjusted, as they did before.
BANDS_WITHOUT_PUBLISHED_SHIFT = frozenset({
    "Ac", "Ao", "Bj", "Gb", "Gr", "IR", "Ic", "Lx", "N", "Pg", "Pi", "Pr",
    "Pw", "Py", "Pz", "Rc", "Sg", "Si", "Sr", "Sz", "T", "Uj", "UNK", "VR",
    "Vj", "W2", "W3", "W4",
})


# ADES 2-char band codes encode the photometric SYSTEM and the filter, but the
# order depends on the system (per MPC's ADES<->MPC80 band mapping):
#   - survey systems put the SYSTEM first, filter second:
#       S=Sloan, P=Pan-STARRS, L=LSST/Rubin, A=ATLAS, G=Gaia  ->  'Pw'->w, 'Li'->i
#   - Johnson-Cousins put the FILTER first, system second (j=Johnson, c=Cousins):
#       'Rc'->R, 'Vj'->V, 'Bj'->B
# TWO callers only, and neither is "internal processing": obs80.band_to_obs80,
# which fills the one byte MPC 80-column column 71 has, and band_shift's fallback
# for a band the MPC has not given a conversion. Everything else -- the obs dicts,
# the registry, the fit, the residual rows, the ML feature -- carries the ADES
# band whole.
_SURVEY_PREFIX = set("SPLAG")
_JOHNSON_FILTERS = set("UBVRI")

# Stripping the system prefix assumes a survey's filter IS the bare letter. That
# holds to a few hundredths for most of them -- and fails for LSST u, which is
# why the LSST bands are tabulated outright above rather than left to it.


def ades_band_filter(band):
    """The single-character MPC80 filter letter an ADES band names, ``" "`` when
    it explicitly names none, or ``None`` when the code names no filter we can
    recover.

    The three outcomes matter, and conflating them is how this went wrong. The
    old rule ended in "best effort: take the last character", which is fine for
    a structured code and nonsense otherwise -- 'UNK' (the MPC's "the submitter
    declined to say") collapsed to 'K' and drew a +1.7 mag INFRARED correction,
    and 'IR' and 'VR' both landed on 'R' (+0.4) by the same accident. A code we
    cannot parse must say so, not guess a letter out of its spelling."""
    b = (band or "").strip()
    if not b or b.upper() == "UNK":
        return " "                       # no filter stated -> the blank band
    if len(b) == 1:
        return b
    if len(b) == 2:
        first, second = b[0], b[1]
        if first in _SURVEY_PREFIX:                       # Sloan/PS/LSST/ATLAS/Gaia
            return second
        if second in ("j", "c") and first in _JOHNSON_FILTERS:   # Johnson-Cousins
            return first
    return None      # 'IR', 'VR', 'W2'-'W4', a typo: no filter to recover


def ades_band_to_mpc(band):
    """Collapse an ADES band code to the one character MPC 80-column col 71 holds.
    An unrecoverable code renders blank rather than inventing a letter."""
    f = ades_band_filter(band)
    return " " if f is None else f


def band_shift(band):
    """Additive correction (mag) to convert a `band` magnitude to V (V = mag + shift).

    The lookup is on the FULL ADES band: 'Lg' is asked of the table as 'Lg', not
    collapsed to 'g' first. Three outcomes, in order:

      1. the table publishes a value for the band as written -> use it. Every
         single-character band, and any two-character band the MPC later gives a
         conversion for, lands here.
      2. no published value, but the code names a filter we do know (all 33 of
         the MPC's current two-character codes, which carry v_conversion=null) ->
         fall back to that filter's shift, e.g. 'Lg' -> 'g' -> -0.28. This is OUR
         inference, not the MPC's: it assumes an LSST g behaves like a generic g,
         which is right to a few hundredths and wrong in the third decimal.
         'UNK' resolves to the BLANK band here, the same thing col 71 renders it
         as -- it means "no filter stated", so it gets the no-filter shift.
      3. the code names no filter we can recover ('IR', 'VR', 'W2'-'W4', a typo)
         -> 0.0, unadjusted, which is find_orb's behaviour for an unknown band.
         Guessing a letter out of the spelling is what used to charge 'UNK' a
         +1.7 mag K-band correction; see ades_band_filter.

    V-equivalent bands (V, N, T, j, v) are 0 by definition."""
    b = (band or " ").strip() or " "
    if b in _V_EQUIV:
        return 0.0
    if b in BAND_SHIFT:                       # 1. as published, full code
        return BAND_SHIFT[b]
    key = ades_band_filter(b)                 # 2. resolve via the filter letter
    if key is None or key in _V_EQUIV:        # 3. unrecoverable -> unadjusted
        return 0.0
    return BAND_SHIFT.get(key, 0.0)


def phase_reduction(r, delta, earth_sun, G=ASTEROID_G):
    """The distance + phase-angle reduction so that V = H + reduction, for the
    H,G system (Bowell 1989). r=heliocentric, delta=topocentric (observer->object),
    earth_sun=observer heliocentric distance (all au). Returns (reduction, phase_rad)."""
    cos_a = (r*r + delta*delta - earth_sun*earth_sun) / (2.0 * delta * r)
    cos_a = max(-1.0, min(1.0, cos_a))
    alpha = math.acos(cos_a)
    if alpha < 1e-10:
        phi1 = phi2 = 1.0
    else:
        t = math.tan(alpha / 2.0)
        phi1 = math.exp(-3.33 * t**0.63)
        phi2 = math.exp(-1.87 * t**1.22)
    tval = (1.0 - G) * phi1 + G * phi2
    if tval < 1e-50:
        reduction = 9999.0
    else:
        reduction = 5.0 * math.log10(r * delta) - 2.5 * math.log10(tval)
    return reduction, alpha


def fit_absolute_magnitude(records, G=ASTEROID_G):
    """Estimate H from photometry, find_orb-style.

    `records`: list of dicts with keys obs_mag (float or None), band (str),
    mag_sigma (float), r, delta, earth_sun (au), included (bool). Observations
    without a magnitude, excluded, or geometrically invalid are skipped from the H
    fit but still get a computed V (for residual display).

    For each usable obs the implied absolute magnitude is
        H_i = (obs_mag + band_shift(band)) - reduction(r, delta, earth_sun)
    and H is the 1/mag_sigma-weighted mean (find_orb's scheme). Returns a dict with
    H, sigma_H, n_mags, and a per-record list of {computed_mag, dmag, phase_deg,
    used}. The photometric residual dMag = obs_mag - computed_mag = H_i - H."""
    Hs, w = [], []
    per = []
    for rec in records:
        r, delta, es = rec.get("r"), rec.get("delta"), rec.get("earth_sun")
        bsh = band_shift(rec.get("band"))
        if r and delta and es:
            reduction, alpha = phase_reduction(r, delta, es, G)
        else:
            reduction, alpha = None, None
        # H_i is computable for ANY obs with a magnitude and valid geometry, so
        # excluded observations still get a dMag for display. `used` is the
        # stricter gate that feeds the weighted-mean H fit (excludes excluded obs).
        have_mag = (rec.get("obs_mag") is not None
                    and reduction is not None and reduction < 9000.0)
        used = have_mag and rec.get("included", True)
        H_i = (rec["obs_mag"] + bsh - reduction) if have_mag else None
        per.append({"reduction": reduction, "band_shift": bsh, "H_i": H_i,
                    "phase_deg": (math.degrees(alpha) if alpha is not None else None),
                    # Carry the geometry through for display: r = heliocentric
                    # (Sun->object) au, delta = topocentric (observer->object) au.
                    "r": r, "delta": delta,
                    "used": used})
        if used:
            sig = rec.get("mag_sigma") or DEFAULT_MAG_SIGMA
            Hs.append(H_i); w.append(1.0 / sig)

    if not Hs:
        return {"H": None, "sigma_H": None, "n_mags": 0, "per": _finish_per(per, None)}
    wsum = sum(w)
    H = sum(h*wi for h, wi in zip(Hs, w)) / wsum
    # weighted standard error of the mean (guards single-point case)
    if len(Hs) > 1:
        var = sum(wi*(h-H)**2 for h, wi in zip(Hs, w)) / wsum
        sigma_H = math.sqrt(var / len(Hs))
    else:
        sigma_H = DEFAULT_MAG_SIGMA
    return {"H": H, "sigma_H": sigma_H, "n_mags": len(Hs), "per": _finish_per(per, H)}


def _finish_per(per, H):
    """Fill computed_mag (= reduction + H) and dmag (= H_i - H) per record."""
    for p in per:
        if H is not None and p["reduction"] is not None and p["reduction"] < 9000.0:
            p["computed_mag"] = p["reduction"] + H - p["band_shift"]  # in the obs's band
            p["dmag"] = (p["H_i"] - H) if p["H_i"] is not None else None
        else:
            p["computed_mag"] = None
            p["dmag"] = None
    return per
