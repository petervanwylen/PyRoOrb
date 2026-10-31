"""Hazard / "what is it" products: MOID, orbit classification, MPC uncertainty U.

Cheap geometry + bookkeeping on an orbit we already have:
  - moid(): minimum orbit intersection distance between two orbits (au) -- the key
    impact-relevance number -- numerical (dense grid + local refinement).
  - classify(): dynamical class (Atira/Aten/Apollo/Amor NEO subclasses, MBA, Hilda,
    Trojan, Centaur, TNO, ...), the PHA flag, and the Tisserand parameter wrt Jupiter.
  - uncertainty_parameter(): the MPC "U" (0-9), ported from find_orb's formula
    (monte0.cpp::dump_monte_data_to_file): U = ln(runoff)/1.49 + 1 with the
    along-orbit longitude runoff built from sigma_Tp and sigma_(1/a).

PROVENANCE: U formula ported from find_orb (read from its source); MOID and the
classification cutoffs are standard definitions, not find_orb code.
"""
import math
import numpy as np

GAUSS_K = 0.01720209895
A_JUPITER = 5.20336301        # au, semimajor axis of Jupiter


def _anomaly_grid(elem, n, r_max=100.0):
    """`n` anomaly samples covering the orbit, and their spacing: eccentric anomaly over the whole
    ellipse, or hyperbolic anomaly over the branch out to `r_max` au -- a hyperbola has no closed
    orbit to sample, and nothing past Neptune matters to a MOID."""
    a, e = elem["a"], elem["e"]
    if e > 1.0 and a < 0:
        hmax = math.acosh(max(1.0, (r_max / -a + 1.0) / e))
        return np.linspace(-hmax, hmax, n), 2.0 * hmax / n
    return np.linspace(0.0, 2*math.pi, n, endpoint=False), 2*math.pi / n


def _points_at(elem, E):
    """Heliocentric ECLIPTIC xyz (n,3 au) at the given anomaly array: eccentric anomaly for an
    ellipse, hyperbolic anomaly for a hyperbola (e > 1, a < 0). The ellipse formula at e > 1 used
    to give a zero minor axis, i.e. a MOID measured to a straight segment."""
    a, e = elem["a"], elem["e"]
    i, Om, w = map(math.radians, (elem["i"], elem["asc_node"], elem["arg_per"]))
    if e > 1.0 and a < 0:
        xp = -a * (e - np.cosh(E))
        yp = -a * math.sqrt(e*e - 1) * np.sinh(E)
    else:
        xp = a * (np.cos(E) - e)
        yp = a * math.sqrt(max(0.0, 1 - e*e)) * np.sin(E)
    cw, sw = math.cos(w), math.sin(w)
    cO, sO = math.cos(Om), math.sin(Om)
    ci, si = math.cos(i), math.sin(i)
    R = np.array([[cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci],
                  [sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci],
                  [sw*si,             cw*si]])
    return (R @ np.vstack([xp, yp])).T          # (n,3)


def moid(elem_a, elem_b, refine=4):
    """Minimum distance (au) between two Keplerian orbits given as element dicts
    with a/e/i/asc_node/arg_per (angles in degrees). Coarse global grid, then a few
    rounds of local densification around the best pair — tracking the actual
    eccentric anomalies (E) and shrinking the search window each round."""
    na = nb = 1440
    Ea = np.linspace(0.0, 2*math.pi, na, endpoint=False)
    Eb = np.linspace(0.0, 2*math.pi, nb, endpoint=False)
    A, B = _points_at(elem_a, Ea), _points_at(elem_b, Eb)
    d2 = np.sum((A[:, None, :] - B[None, :, :])**2, axis=2)
    ia, ib = np.unravel_index(int(np.argmin(d2)), d2.shape)
    best = math.sqrt(d2[ia, ib])
    E0a, E0b = Ea[ia], Eb[ib]
    spanA, spanB = 2*math.pi/na, 2*math.pi/nb
    for _ in range(refine):
        Ea = np.linspace(E0a - spanA, E0a + spanA, 200)
        Eb = np.linspace(E0b - spanB, E0b + spanB, 200)
        A, B = _points_at(elem_a, Ea), _points_at(elem_b, Eb)
        d2 = np.sum((A[:, None, :] - B[None, :, :])**2, axis=2)
        ia, ib = np.unravel_index(int(np.argmin(d2)), d2.shape)
        best = math.sqrt(d2[ia, ib])
        E0a, E0b = Ea[ia], Eb[ib]
        spanA /= 10.0; spanB /= 10.0
    return best


def tisserand_jupiter(a, e, i_deg):
    if a <= 0:
        return float("nan")
    return A_JUPITER/a + 2*math.cos(math.radians(i_deg))*math.sqrt((a/A_JUPITER)*(1 - e*e))


def classify(a, e, i_deg, H=None, earth_moid=None):
    """Dynamical class string + PHA flag. Cutoffs are the standard MPC/JPL ones."""
    q = a*(1 - e); Q = a*(1 + e)
    if a <= 0:
        cls = "hyperbolic/parabolic"
    elif q < 1.3:                                  # near-Earth object
        if a < 1.0 and Q < 0.983:
            cls = "Atira (NEO)"
        elif a < 1.0:
            cls = "Aten (NEO)"
        elif q < 1.017:
            cls = "Apollo (NEO)"
        else:
            cls = "Amor (NEO)"
    elif q < 1.666:
        cls = "Mars-crosser"
    elif a < 2.0:
        cls = "inner main-belt / Hungaria"
    elif a < 3.2:
        cls = "main-belt asteroid"
    elif a < 3.7:
        cls = "outer main-belt / Cybele"
    elif a < 4.1:
        cls = "Hilda"
    elif a < 4.6:
        cls = "outer belt"
    elif 5.05 < a < 5.4:
        cls = "Jupiter Trojan"
    elif a < 5.5:
        cls = "Jupiter-coorbital region"
    elif a < 30.1:
        cls = "Centaur"
    else:
        cls = "trans-Neptunian object"
    is_pha = bool(q < 1.3 and earth_moid is not None and earth_moid <= 0.05
                  and (H is not None and H <= 22.0))
    return {"class": cls, "q": q, "Q": Q, "pha": is_pha,
            "tisserand_jupiter": tisserand_jupiter(a, e, i_deg)}


def uncertainty_parameter(a, e, sigma_a, sigma_Tp):
    """MPC uncertainty parameter U (0-9), find_orb's formula (monte0.cpp): build the
    along-orbit longitude runoff (arcsec) from the period and perihelion-time
    uncertainties, then U = ln(runoff)/1.49 + 1.

    sigma_a in au, sigma_Tp in days. Returns float U (>=0) or None if unusable."""
    if a <= 0 or sigma_a is None or sigma_Tp is None or not np.isfinite(sigma_a):
        return None
    per_yrs = a**1.5                                   # heliocentric (mass = 1)
    per_days = per_yrs * 365.25
    sigma_P_days = 1.5 * per_days * sigma_a / a        # period sigma
    runoff_coeff = 3.0 * 3600.0 * (180.0/math.pi) * GAUSS_K
    runoff = (runoff_coeff / per_yrs) * (sigma_Tp * e + 10.0 * sigma_P_days / per_yrs)
    if runoff <= 0:
        return None
    return max(0.0, math.log(runoff) / 1.49 + 1.0)
