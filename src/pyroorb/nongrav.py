"""Non-gravitational force models -- find_orb-style selectable nongrav parameters.

A small asteroid or comet feels accelerations beyond Sun+planet gravity:
out-gassing on a comet, the Yarkovsky thermal recoil on an asteroid, solar
radiation pressure. The standard (Marsden, Sekanina & Yeomans 1973) model adds

    a_ng = [ A1 r^ + A2 t^ + A3 n^ ] * g(r)

where r^/t^/n^ are the radial / transverse / normal unit vectors of the
heliocentric orbit and g(r) is a scaling law in heliocentric distance. ASSIST
implements exactly this (forces.c::assist_additional_force_non_gravitational,
particle_params[3j+0..2] = A1,A2,A3; globals alpha/r0/nm/nn/nk define g(r)); we
just choose the constants and which A_i to solve for.

find_orb does the same: it offers a handful of MODELS, each a fixed g(r) plus a
subset of {A1,A2,A3} (and, not yet here, the asymmetric-outgassing time-lag DT)
that the differential correction is allowed to vary. The rest are held at zero.
This module is that menu: a registry the fit reads, nothing object-specific.

PROVENANCE: the g(r) laws and the A1/A2/A3 radial/transverse/normal decomposition
are the public Marsden-Sekanina-Yeomans (1973) standard, also used by JPL's
small-body nongrav solutions and ASSIST; emulated, not copied.
"""

# --- g(r) heliocentric-distance scaling laws -------------------------------------
# ASSIST evaluates  g = alpha * (r/r0)^-nm * (1 + (r/r0)^nn)^-nk .
#
# COMET_GR: the Marsden-Sekanina-Yeomans water-ice sublimation law, normalised so
# g(1 au) = 1. The canonical comet nongrav scaling (alpha 0.1113, r0 2.808 au,
# m 2.15, n 5.093, k 4.6142).
COMET_GR = {"alpha": 0.1112620426, "r0": 2.808, "nm": 2.15, "nn": 5.093, "nk": 4.6142}

# INVSQ_GR: a pure inverse-square law g(r) = (1 au / r)^2 (nk=0 collapses the
# bracket to 1). This is the form used for the asteroid Yarkovsky A2 term and for
# solar-radiation-pressure A1, matching JPL's asteroid nongrav convention.
INVSQ_GR = {"alpha": 1.0, "r0": 1.0, "nm": 2.0, "nn": 1.0, "nk": 0.0}


# --- model registry --------------------------------------------------------------
# Each model: the ordered tuple of FREE parameters (subset/order of A1,A2,A3), the
# g(r) law, and a per-fit `scale` (au/day^2) so the solved-for values land near
# O(1) and the LM stays well conditioned -- comet out-gassing is ~1e-8 au/day^2,
# asteroid Yarkovsky ~1e-14.
A_AXES = ("A1", "A2", "A3")                  # radial, transverse, normal (ASSIST order)

MODELS = {
    # pure gravity -- the default; refine is a no-op
    "none":        {"params": (),                 "gr": None,      "scale": 0.0},
    # comet out-gassing: radial-only (the conservative first step of the comet model sequence) ...
    "comet1":      {"params": ("A1",),            "gr": COMET_GR,  "scale": 1e-8},
    # ... radial+transverse (find_orb's default comet model) ...
    "comet2":      {"params": ("A1", "A2"),       "gr": COMET_GR,  "scale": 1e-8},
    # ... and the full radial+transverse+normal set
    "comet3":      {"params": ("A1", "A2", "A3"), "gr": COMET_GR,  "scale": 1e-8},
    # asteroid Yarkovsky: a single transverse term on an inverse-square law
    "yarkovsky":   {"params": ("A2",),            "gr": INVSQ_GR,  "scale": 1e-14},
    # solar radiation pressure on an inverse-square law -- find_orb's SRP family:
    #   srp  = radial only (1 param, the classic area/mass term)
    #   srp2 = radial + transverse (find_orb SRP_TWO_PARAM)
    #   srp3 = radial + transverse + normal (find_orb SRP_THREE_PARAM)
    "srp":         {"params": ("A1",),            "gr": INVSQ_GR,  "scale": 1e-11},
    "srp2":        {"params": ("A1", "A2"),       "gr": INVSQ_GR,  "scale": 1e-11},
    "srp3":        {"params": ("A1", "A2", "A3"), "gr": INVSQ_GR,  "scale": 1e-11},
}


def model(name):
    """Return the model spec dict for `name`, raising a helpful error otherwise."""
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"unknown nongrav model {name!r}; choose one of "
                         f"{', '.join(MODELS)}")


def free_indices(name):
    """Indices into the (A1,A2,A3) triple that `name` solves for, in fit order."""
    return [A_AXES.index(p) for p in model(name)["params"]]


def full_A(name, free_values):
    """Expand the k free, SCALED parameter values into the length-3 A vector
    [A1,A2,A3] in physical au/day^2 (fixed axes are 0). `free_values` are in units
    of the model's `scale`, i.e. what the LM actually carries."""
    spec = model(name)
    A = [0.0, 0.0, 0.0]
    for idx, v in zip(free_indices(name), free_values):
        A[idx] = float(v) * spec["scale"]
    return A


# --- physical-plausibility constants & helpers -----------------------------------
# Used by the automatic nongrav-discovery routine (fit_object.discover_nongrav) to
# reject a "detection" whose fitted coefficient is physically absurd -- i.e. a free A
# that is really absorbing astrometric systematics (regional star-catalog bias, timing
# error, a missed perturber) rather than a real non-gravitational force.

# SRP1AU: solar-radiation-pressure acceleration at 1 au for a 1 m^2/kg absorber, in
# au/day^2. For the inverse-square (rock-like) model, A1 = SRP1AU * AMR, so the fitted
# radial term implies an area/mass ratio AMR = A1 / SRP1AU (find_orb constant.h; Chesley).
#   SRP1AU = SOLAR_CONSTANT * sec_per_day^2 / (AU_m * c_km/s * m_per_km)
_SOLAR_CONSTANT = 1361.0          # W/m^2 at 1 au
_SEC_PER_DAY    = 86400.0
_AU_IN_METERS   = 1.495978707e11
_C_KM_S         = 299792.458      # find_orb's SPEED_OF_LIGHT is in km/s
_M_PER_KM       = 1000.0
SRP1AU = _SOLAR_CONSTANT * _SEC_PER_DAY * _SEC_PER_DAY / (_AU_IN_METERS * _C_KM_S * _M_PER_KM)
# ~2.265e-7 au/day^2 per (m^2/kg)

# Plausibility bounds (magnitude only; deliberately generous so we only veto the absurd):
#   A2 Yarkovsky: real NEA values run ~1e-15..1e-13 au/day^2 (Bennu ~-4.6e-14). The
#     transverse term scales with SIZE -- the smaller the body, the larger its physical
#     Yarkovsky (A2 ~ 1/(rho*D)), so a single flat ceiling is wrong: 1e-12 fits ~km bodies
#     but vetoes the genuine multi-1e-12 drifts of metre-scale NEAs (e.g. 2011 CL50, H=27.3,
#     JPL A2=+4.4e-12). When the absolute magnitude H is known we use a SIZE-AWARE ceiling
#     -- the maximal physical transverse thrust for a body of that diameter (see a2_ceiling).
#     Validated against the full JPL a<1.6 A2 population (SNR>=3): admits 471/472 real
#     detections, vetoes only the debris-like mini-moon 2006 RH120. The flat A2_MAX_AUD2 is
#     kept as a floor (used when H is unknown, and for ~km+ bodies it dominates anyway).
#   A1 SRP: like A2, the radial solar-radiation-pressure term scales with area/mass, hence with
#     SIZE (A1 = SRP1AU * AMR, AMR = 1.5/(rho*D) for a sphere), so a single flat AMR ceiling is the
#     WRONG test -- it is far too loose. A1 and H together IMPLY a bulk density rho = 1.5/(D*AMR);
#     a credible natural body sits above ~0.1 g/cm^3 (dense rock ~2-3, porous rubble/extinct comet
#     ~0.5-1, e.g. JPL's real SRP object 2009 BD ~0.7 g/cm^3). An A1 that implies a density BELOW
#     that floor is not solar radiation pressure on a rock -- it is either artificial/debris (a
#     rocket body, foil) or, on a multi-apparition NEA, an A1 absorbing astrometric systematics.
#     And SRP pushes OUTWARD, so a real A1 is POSITIVE; a negative fitted A1 is compensating for
#     bad astrometry or an unmodelled perturber. The implied-density test (a1_implied_density) turns
#     A1+H into that density -- e.g. 2026 NA1: H=25/~35 m but A1 => rho~0.006 g/cm^3, hollow. It is
#     ADVISORY: it drives a non-blocking WARNING (nongrav_physical_warnings), NOT a veto. We want to
#     FIT these outliers -- a rocket body or a dark comet genuinely has a high area/mass ratio and a
#     low implied density -- and just flag that the numbers are unusual. The SELECTION guard
#     (a1_is_plausible) stays the flat, loose AMR ceiling so discovery still picks them.
A2_MAX_AUD2 = 1e-12               # flat |A2| floor / fallback when H (size) is unknown
AMR_MAX_M2KG = 0.1                # flat |AMR| ceiling -- the loose SELECTION guard (a1_is_plausible)
A1_MIN_DENSITY_KG_M3 = 100.0      # min implied bulk density (0.1 g/cm^3) for a natural-rock SRP A1;
                                 #   below it the ADVISORY warning fires (2009 BD ~712 kg/m^3 clears it,
                                 #   2026 NA1 ~6 kg/m^3 does not). Not a veto -- see nongrav_physical_warnings.

# Size-aware Yarkovsky ceiling constants. The maximal physical transverse acceleration of a
# body is (Yarkovsky efficiency) x (direct radiation-pressure accel), and the radiation accel
# is SRP1AU x AMR with AMR=1.5/(rho*D) for a sphere. We take a GENEROUS upper bound: a high
# efficiency, a low (rubble-pile) density floor, and a standard albedo to turn H into D -- so
# the ceiling is the most a real thermal recoil could be, and anything above it is absorbing
# astrometric systematics, not physics.
A2_YARK_EFFICIENCY = 0.25        # transverse-thrust fraction of direct SRP (real NEAs <~0.1)
A2_DENSITY_FLOOR   = 1000.0      # kg/m^3 low-density (rubble-pile) floor -> largest AMR
A2_REF_ALBEDO      = 0.14        # standard NEA albedo for the H->diameter conversion


def diameter_km_from_h(H, albedo=A2_REF_ALBEDO):
    """Diameter (km) implied by absolute magnitude H and a visual albedo (standard relation
    D = 1329 / sqrt(p_V) * 10^(-H/5))."""
    return 1329.0 / (albedo ** 0.5) * 10.0 ** (-0.2 * H)


def a2_ceiling(H=None):
    """The maximal physically credible |A2| (au/day^2). With a known absolute magnitude H this
    is SIZE-AWARE -- efficiency * SRP1AU * AMR_max(D) for the body's diameter, floored at the
    flat A2_MAX_AUD2 so it is never tighter than the size-agnostic bound (and so ~km+ bodies
    keep the old ceiling). Without H, returns the flat floor."""
    if H is None:
        return A2_MAX_AUD2
    d_m = diameter_km_from_h(H) * 1000.0                       # diameter in metres
    amr_max = 1.5 / (A2_DENSITY_FLOOR * d_m)                   # m^2/kg for a sphere
    return max(A2_MAX_AUD2, A2_YARK_EFFICIENCY * SRP1AU * amr_max)


def amr_from_a1(a1):
    """Area/mass ratio (m^2/kg) implied by a radial SRP coefficient A1 (au/day^2)."""
    return a1 / SRP1AU


# find_orb reports the area/mass ratio on the elements' Epoch line ("Epoch 1997 Oct 13.0 TT;
# AMR 0.034 m^2/kg", elem_out.cpp) for ONE force model only: FORCE_MODEL_SRP, its
# single-parameter inverse-square radial term -- as `orbit[6] * SOLAR_GM / SRP1AU`, which is
# exactly A1 / SRP1AU once find_orb's internal parameter is scaled to an acceleration at 1 au
# (runge.cpp adds `ival[6]` to the dimensionless solar term before the -SOLAR_GM/r^3 factor).
# It is NOT printed for the two- and three-parameter SRP models, and we follow that: with A2
# (and A3) free the radial coefficient is no longer the whole photon push, so calling it an
# area/mass ratio would over-claim. The number is also a PERFECT-ABSORBER equivalent -- a real
# body reflects and re-radiates, so the true area/mass is smaller by the (unknown) ~1+albedo
# radiation-pressure coefficient -- and find_orb's own note (constant.h) says as much.
AMR_MODEL = "srp"                 # the 1-parameter inverse-square radial model = FORCE_MODEL_SRP


def amr_m2_per_kg(model, params):
    """The AMR (m^2/kg) find_orb prints for its 1-parameter SRP model, or None when the fit is
    any other model (or has no A1). `params` is a {axis: au/day^2} mapping."""
    if model != AMR_MODEL or not params:
        return None
    a1 = params.get("A1")
    if a1 is None:
        return None
    return amr_from_a1(float(a1))


def amr_report(model, params, intervals=None, sigmas=None, unconstrained=()):
    """The whole AMR view-model, or None when this fit is not one find_orb reports AMR for.

    ``{"amr", "minus", "plus", "unconstrained"}`` in m^2/kg. AMR is A1/SRP1AU, a FIXED multiple,
    so A1's own profile-likelihood interval (or its sigma, for a symmetric fallback) rescales
    straight onto it -- no extra refits are needed for the uncertainty. ``minus``/``plus`` are
    None when the sigmas have not been computed yet."""
    amr = amr_m2_per_kg(model, params)
    if amr is None:
        return None
    lo = hi = None
    iv = (intervals or {}).get("A1")
    if iv is not None:
        lo, hi = float(iv[0]) / SRP1AU, float(iv[1]) / SRP1AU
    elif (sigmas or {}).get("A1") is not None:
        lo = hi = float(sigmas["A1"]) / SRP1AU
    return {"amr": amr, "minus": lo, "plus": hi, "unconstrained": "A1" in set(unconstrained)}


def a2_is_plausible(a2, H=None):
    """True if a fitted Yarkovsky transverse term is within the physical range for a body of
    absolute magnitude H (size-aware; see a2_ceiling). H=None falls back to the flat ceiling."""
    return abs(a2) <= a2_ceiling(H)


def a1_implied_density(a1, H, albedo=A2_REF_ALBEDO):
    """Bulk density (kg/m^3) a fitted radial SRP coefficient A1 (au/day^2) implies for a body of
    absolute magnitude H. A1 = SRP1AU * AMR and, for a sphere of diameter D, AMR = 1.5/(rho*D), so
    rho = 1.5 / (D * AMR) = 1.5 * SRP1AU / (D * A1) with D = diameter_km_from_h(H)*1000. A real rock
    lands at ~500-3000; artificial/systematics-absorbing A1's imply an absurdly low (hollow) density.
    Undefined for A1<=0 (SRP pushes outward); returns +inf so the plausibility test fails it there."""
    if a1 <= 0.0:
        return float("inf")
    d_m = diameter_km_from_h(H, albedo) * 1000.0
    return 1.5 * SRP1AU / (d_m * a1)


def a1_is_plausible(a1):
    """True if a fitted radial SRP term implies a physically credible area/mass ratio (the flat,
    H-blind, deliberately-loose ceiling). This is the SELECTION guard: it must stay generous so the
    auto-discovery can still PICK an SRP model for the interesting outliers -- rocket bodies, dark
    comets, and other high-area/mass objects -- rather than veto them. The sharper, H-aware
    IMPLIED-DENSITY test (a1_implied_density) is ADVISORY only: it drives a non-blocking warning on
    the delivered orbit (see fit_object.refine_nongrav / nongrav_physical_warnings), never a veto."""
    return abs(amr_from_a1(a1)) <= AMR_MAX_M2KG


# Out-gassing mass budget (Taylor et al. 2024, dark-comet accelerations): sustaining an
# acceleration |A| by ejecting mass at speed v depletes the body on tau ~ v / |A|. With v ~ 350
# m/s the inner dark comets come out at ~1e4-1e5 yr, which that paper itself calls perplexing.
# A utility only: the A3 advisory does NOT quote it (too speculative for a
# banner), and no A3/A2 ratio is claimed either, since no single mechanism theory bounds one.
OUTGAS_EJECTION_SPEED_MS = 350.0
_AUD2_TO_MS2 = _AU_IN_METERS / (_SEC_PER_DAY * _SEC_PER_DAY)   # 1 au/day^2 in m/s^2 (~20.0)


def outgassing_depletion_years(a_aud2, v_ms=OUTGAS_EJECTION_SPEED_MS):
    """Years to deplete a body that sustains a non-gravitational acceleration |a| (au/day^2)
    by ejecting mass at v (m/s): tau = v / |a|. Independent of size and density."""
    a = abs(float(a_aud2)) * _AUD2_TO_MS2
    return float("inf") if a == 0.0 else v_ms / a / (365.25 * 86400.0)


def nongrav_physical_warnings(params, H=None, with_kinds=False):
    """A human-readable WARNING (or None) when a fitted/applied non-gravitational model carries
    values that may be non-physical -- surfaced as an advisory banner, NOT used to reject the model.
    We deliberately DO fit these (a rocket body has a genuinely high area/mass ratio and shows
    exactly such 'implausible' coefficients); the warning just tells the user the numbers
    are unusual so they interpret them with care. Flags: a radial A1 whose implied bulk density
    (with H) is below the natural-body floor, a NEGATIVE A1 (solar radiation pressure pushes
    outward), and a transverse A2 above the size-aware Yarkovsky ceiling. Returns one combined
    string or None if every term is within physical range. `with_kinds=True` returns
    (text, kinds) where kinds is a set drawn from {"a1_negative", "a1_density", "a2_ceiling"},
    so a caller that must act on ONE kind (discover_nongrav's delivery guard is about the radial
    term's data confidence, never the A2 size prior) does not have to parse the prose.

    A NORMAL (out-of-plane) A3 on the asteroid family is flagged whenever it is present (kind
    "a3_asteroid"): finding a real A3 on an asteroid is rare. The note says what the srp3
    model actually is (a 1/r^2 law on all three axes, so possibly some mix of solar radiation
    pressure and Yarkovsky), that the A3 could be the model absorbing some other force or an
    observational data issue, and suggests the comparisons that can actually say something:
    A2-only, A1+A2, and per-apparition fits (a comet3 comparison was dropped on review -- inside
    ~1.5 au the Marsden g(r) is effectively 1/r^2, so it cannot separate mechanisms). It claims
    NO A3/A2 ratio bound and carries no z (the linear sigma
    known here is not the profile sigma the reports show).
    """
    notes, kinds = [], set()
    a1 = float(params.get("A1", 0.0)) if params else 0.0
    a2 = float(params.get("A2", 0.0)) if params else 0.0
    a3 = float(params.get("A3", 0.0)) if params else 0.0
    if a1 < 0.0:
        kinds.add("a1_negative")
        notes.append("A1 is negative (solar radiation pressure pushes outward)")
    elif a1 > 0.0 and H is not None:
        rho = a1_implied_density(a1, H)
        if rho < A1_MIN_DENSITY_KG_M3:
            kinds.add("a1_density")
            # Order-of-magnitude only: the density scales with diameter (from H via an assumed albedo,
            # D proportional to albedo^-0.5) and with A1, so it is uncertain by a factor of a few --
            # report one significant figure, not the false precision of a 3-decimal value.
            notes.append(f"A1 implies a bulk density of order ~{rho/1000:.1g} g/cm^3 for this size "
                         f"(H={H:.1f}, via an assumed albedo -- uncertain by a factor of a few), below "
                         f"the ~{A1_MIN_DENSITY_KG_M3/1000:.1g} g/cm^3 floor for a natural rock. "
                         f"Worth considering: an unusually high area/mass ratio (an artificial "
                         f"body, or something not yet understood), radial out-gassing, or an A1 "
                         f"absorbing astrometric systematics")
    if a2 and not a2_is_plausible(a2, H=H):
        kinds.add("a2_ceiling")
        # Report the NUMBERS, not just the verdict: this is advisory and the reader is expected to
        # judge it, which they cannot do from a bare "exceeds". The ceiling is a soft size prior
        # (H via an assumed albedo -> diameter, uncertain by a factor of a few), so a value a small
        # multiple over it is a prompt to look, not a fault.
        notes.append(f"A2 = {a2:.2e} au/day^2 exceeds the size-aware Yarkovsky ceiling "
                     f"~{a2_ceiling(H):.1e} for this body"
                     + (f" (H={H:.1f}, via an assumed albedo -- uncertain by a factor of a few)"
                        if H is not None else "")
                     + " -- consistent with a higher-albedo (hence smaller) body than H assumes, a non-thermal "
                       "(outgassing) term, or an A2 absorbing astrometric systematics")
    if a3:
        kinds.add("a3_asteroid")
        notes.append(a3_asteroid_note(a3, a2, a1))
    if not notes:
        return (None, kinds) if with_kinds else None
    text = ("The selected non-gravitational model has values worth a second look: "
            + "; ".join(notes) + ".")
    return (text, kinds) if with_kinds else text


def a3_asteroid_note(a3, a2, a1=0.0):
    """The advisory sentence for a NORMAL-direction A3 delivered on the asteroid (1/r^2) family.
    It reports the numbers (A3, and its size against A2 and A1 where those are fitted) and
    deliberately passes NO physical verdict on the ratio: comets, Yarkovsky and seasonal polar
    out-gassing each bound A3 differently and no single theory gives an A3/A2 limit.
    What it does say: srp3 is a 1/r^2 law on three axes, so what it delivers
    may be some mix of SRP and Yarkovsky; the A3 could be the model absorbing some other force
    or an observational data issue; and the user should compare A2-only, A1+A2 and
    per-apparition fits before reading anything into it.
    The out-gassing depletion timescale (outgassing_depletion_years) was quoted here once and
    dropped as too speculative for a banner -- it stays a utility, not a claim."""
    a3, a2, a1 = float(a3), float(a2 or 0.0), float(a1 or 0.0)
    scale = [f"{abs(a3) / abs(a2):.0f}x |A2|"] if a2 else []
    if a1:
        scale.append(f"{abs(a3) / abs(a1):.2g}x |A1|")
    sized = f" ({', '.join(scale)})" if scale else ""
    return (f"a normal-direction A3 = {a3:.2e} au/day^2{sized} was fitted -- finding real A3 "
            f"values in asteroids is rare. The srp3 model is a 1/r^2 law on all three "
            f"axes, so what it delivers may be some mix of solar radiation pressure (radial) and "
            f"Yarkovsky (transverse) plus an out-of-plane term: it "
            f"could be the model absorbing some other force, or an observational data issue. "
            f"Worth comparing before reading much into it: the A2-only and A1+A2 fits of the "
            f"same arc, and each apparition fitted on its own -- an A3 "
            f"that vanishes with a subset of the arc is a statement about how the apparitions "
            f"join, not about the body")


# Comet out-gassing (Marsden A1/A2/A3 on the water-ice g(r)) runs FAR larger than asteroid
# Yarkovsky/SRP -- ~1e-8 au/day^2 is typical and very active comets reach ~1e-6. A single
# generous magnitude ceiling, applied to every fitted axis, vetoes only a runaway term that is
# really absorbing astrometric systematics; the chi2 effect-size + nested-F-test bars do the
# real discrimination (see fit_object.discover_comet_nongrav). 1e-5 au/day^2 is ~3% of solar
# gravity at 1 au -- already extreme for a comet, so anything above it is non-physical.
COMET_A_MAX_AUD2 = 1e-5           # |A1|,|A2|,|A3| ceiling for a credible comet out-gassing term


def comet_a_is_plausible(a):
    """True if a fitted comet out-gassing coefficient (any of A1/A2/A3, au/day^2) is within the
    generous physical range -- a guard against a runaway term absorbing astrometric systematics."""
    return abs(a) <= COMET_A_MAX_AUD2
