"""Vector math, solar ephemeris, and great-circle fitting.

Ported from ``d2math.c``. These are the pieces of the algorithm that
are *not* the adaptive orbit search itself: turning observation times
into Sun/observer vectors, and fitting/evaluating a great circle
through a tracklet's observations (used both to synthesize a clean
two-point motion vector and to report the RMS residual column).

Three-vectors are plain ``(x, y, z)`` tuples rather than NumPy arrays.
For fixed-size-3 vectors called millions of times across a tracklet
search, per-call NumPy overhead (array allocation, dtype dispatch)
comfortably loses to plain Python float arithmetic -- this is scalar
code by design, not an oversight; see the module docstring in
:mod:`pydigest2.ranging` for where the NumPy/parallelism budget
actually goes instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .constants import TWO_PI

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Vec3, Vec3, Vec3]


def sub3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot3(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross3(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def ec_rotate(c: Vec3, soe: float, coe: float) -> Vec3:
    """Rotate an equatorial vector to ecliptic coordinates (d2math.c: ecRotate)."""
    e1 = c[2] * soe + c[1] * coe
    return (c[0], e1, c[2] * coe - c[1] * soe)


def se2000(mjd: float) -> Tuple[Vec3, float, float]:
    """Approximate solar ephemeris (USNO algorithm, d2math.c: se2000).

    Returns (sun_earth_vector_equatorial, sin_obliquity, cos_obliquity).
    """
    d = mjd - 51544.5
    g = 357.529 + 0.98560028 * d       # mean anomaly of sun, degrees
    q = 280.459 + 0.98564736 * d       # mean longitude of sun, degrees
    g2 = g + g

    l = q + 1.915 * math.sin(math.radians(g)) + 0.020 * math.sin(math.radians(g2))
    r = 1.00014 - 0.01671 * math.cos(math.radians(g)) - 0.00014 * math.cos(math.radians(g2))

    e = 23.439 - 0.00000036 * d
    soe = math.sin(math.radians(e))
    coe = math.cos(math.radians(e))

    x = r * math.cos(math.radians(l))
    y0 = r * math.sin(math.radians(l))
    z = y0 * soe
    y = y0 * coe
    return (x, y, z), soe, coe


def lst(j0: float, longitude: float) -> float:
    """Local sidereal time (d2math.c: lst); longitude in circles."""
    t = (j0 - 15019.5) / 36525.0
    th = (6.6460656 + (2400.051262 + 0.00002581 * t) * t) / 24.0
    ut = math.fmod(1.0, j0 - 0.5)
    return math.fmod(th + ut + longitude, TWO_PI)


# --- great circle fit ---------------------------------------------------

def _sphr_to_cart(ra: float, dec: float) -> Vec3:
    t = math.cos(dec)
    return (t * math.cos(ra), t * math.sin(ra), math.sin(dec))


def _cart_to_sphr(c: Vec3) -> Tuple[float, float]:
    ra = math.fmod(math.atan2(c[1], c[0]) + TWO_PI, TWO_PI)
    dec = math.asin(c[2])
    return ra, dec


def _mat_vec(m: Mat3, v: Vec3) -> Vec3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _transpose3(m: Mat3) -> Mat3:
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


@dataclass
class GreatCircleFit:
    m_rot: Mat3
    ra0: float
    t0: float
    rs: List[Tuple[float, float]]   # rotated+projected (ra, dec), ra normalized near 0
    ntime: List[float]
    r0: float
    rr: float
    d0: float
    dr: float


def gc_fit(mjd: Sequence[float], sphr: Sequence[Tuple[float, float]]) -> GreatCircleFit:
    """Fit a great circle through n >= 2 (ra, dec) observations (d2math.c: gcFit)."""
    n = len(mjd)
    cart = [_sphr_to_cart(ra, dec) for ra, dec in sphr]

    n_vec = cross3(cart[0], cart[-1])
    nmag2 = dot3(n_vec, n_vec)
    nmag = math.sqrt(nmag2)

    nxy = 1.0 / math.sqrt(n_vec[0] * n_vec[0] + n_vec[1] * n_vec[1])
    gcix = n_vec[1] * nxy
    gciy = -n_vec[0] * nxy

    sina = math.sqrt(nmag2 - n_vec[2] * n_vec[2]) / nmag
    cosa = n_vec[2] / nmag

    sinagx = sina * gcix
    sinagy = sina * gciy
    onemcosa = 1 - cosa
    onemcosagx = onemcosa * gcix
    onemcosagxgy = onemcosagx * gciy
    m_rot = (
        (cosa + onemcosagx * gcix, onemcosagxgy, sinagy),
        (onemcosagxgy, cosa + onemcosa * gciy * gciy, -sinagx),
        (-sinagy, sinagx, cosa),
    )

    rotated = [_mat_vec(m_rot, c) for c in cart]
    m_rot_t = _transpose3(m_rot)  # transpose so it de-rotates after the fit

    # invariant: first and last points should lie on the rotated z=0 plane
    from .constants import RTOL
    if abs(rotated[0][2]) > RTOL or abs(rotated[-1][2]) > RTOL:
        raise ArithmeticError("great circle rotation failed (numerically degenerate tracklet)")

    rs_raw = [_cart_to_sphr(c) for c in rotated]

    ra0 = rs_raw[0][0]
    rs = [(math.fmod(ra + 3 * math.pi - ra0, TWO_PI) - math.pi, dec) for ra, dec in rs_raw]

    t0 = mjd[0]
    ntime = [t - t0 for t in mjd]

    if n == 2:
        r0, rr, d0, dr = 0.0, rs[1][0] / ntime[1], 0.0, 0.0
    else:
        sumt = sum(ntime)
        sumra = sum(p[0] for p in rs)
        sumdec = sum(p[1] for p in rs)
        sumt2 = sum(t * t for t in ntime)
        sumtra = sum(t * p[0] for t, p in zip(ntime, rs))
        sumtdec = sum(t * p[1] for t, p in zip(ntime, rs))
        invd = 1.0 / (n * sumt2 - sumt * sumt)
        r0 = invd * (sumra * sumt2 - sumtra * sumt)
        rr = invd * (n * sumtra - sumra * sumt)
        d0 = invd * (sumdec * sumt2 - sumtdec * sumt)
        dr = invd * (n * sumtdec - sumdec * sumt)

    return GreatCircleFit(m_rot=m_rot_t, ra0=ra0, t0=t0, rs=rs, ntime=ntime,
                           r0=r0, rr=rr, d0=d0, dr=dr)


def gc_pos(gcf: GreatCircleFit, t: float) -> Tuple[float, float]:
    """Position on the fitted great circle at time t (d2math.c: gcPos)."""
    nt = t - gcf.t0
    rsc = (gcf.r0 + gcf.rr * nt + gcf.ra0, gcf.d0 + gcf.dr * nt)
    rcc = _sphr_to_cart(*rsc)
    cc = _mat_vec(gcf.m_rot, rcc)
    return _cart_to_sphr(cc)


def gc_rms(gcf: GreatCircleFit) -> float:
    """RMS of the 2D residual between observed and fitted positions
    (d2math.c: gcRms/gcRmsRes/gcRes, folded together since nothing else
    needs the per-point residual vector)."""
    n = len(gcf.ntime)
    rsc = [(gcf.r0 + gcf.rr * t, gcf.d0 + gcf.dr * t) for t in gcf.ntime]

    obs_ra = [ra + gcf.ra0 for ra, _ in gcf.rs]
    comp_ra = [ra + gcf.ra0 for ra, _ in rsc]

    from .constants import ARCSEC_RAD
    s = 0.0
    for i in range(n):
        co = _mat_vec(gcf.m_rot, _sphr_to_cart(obs_ra[i], gcf.rs[i][1]))
        cc = _mat_vec(gcf.m_rot, _sphr_to_cart(comp_ra[i], rsc[i][1]))
        so_ra, so_dec = _cart_to_sphr(co)
        sc_ra, sc_dec = _cart_to_sphr(cc)
        d_dec = (so_dec - sc_dec) / ARCSEC_RAD
        d_ra = (so_ra - sc_ra) * math.cos(sc_dec) / ARCSEC_RAD
        s += d_ra * d_ra + d_dec * d_dec
    return math.sqrt(s / n)
