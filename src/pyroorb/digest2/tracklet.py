"""Motion-vector synthesis: picking/synthesizing the two observations that
define a tracklet's apparent motion, and the great-circle RMS that goes
with them.

Ported from ``d2math.c``: ``oneObs`` and ``twoObs`` (endpoint synthesis)
and ``clipErr`` (turning a computed RMS into a per-observation error
estimate, floored/ceilinged by the configured per-site error).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .constants import ARCSEC_RAD
from .geometry import gc_fit, gc_pos, gc_rms
from .observations import Observation
from .obscodes import Site, SiteTable, UNSET_OBSERR, parse_cod3


def clip_err(computed_rms_arcsec: float, site: Site, default_obserr_rad: float) -> float:
    """Combine a per-site configured error with a computed RMS (d2math.c: clipErr).

    Returns the observational error to use, in radians.
    """
    default_err = site.obs_err if site.obs_err != UNSET_OBSERR else default_obserr_rad
    if default_err == 0:
        return 0.0
    if computed_rms_arcsec == 0:
        return default_err
    computed_err = computed_rms_arcsec * ARCSEC_RAD
    return default_err if default_err > computed_err else computed_err


def _one_obs(olist: Sequence[Observation], o1: int, o2: int, arcs_use_all_obs: bool,
             pt: float) -> tuple:
    """Synthesize (or select) one endpoint observation for the motion
    vector, from the sub-arc olist[o1..o2] (d2math.c: oneObs).

    Returns (Observation, rms_or_0.0).
    """
    result = olist[o1]

    if o1 == o2:
        return result, 0.0

    if o1 == o2 - 1:
        if not arcs_use_all_obs:
            if o1 == 0:
                result = olist[1]
                dt = result.mjd - pt
            else:
                dt = pt - result.mjd
            if dt < 0:
                return result, 0.0
        base = olist[o1]
        fit = gc_fit([olist[o1].mjd, olist[o2].mjd],
                     [(olist[o1].ra, olist[o1].dec), (olist[o2].ra, olist[o2].dec)])
        new_mjd = (olist[o1].mjd + olist[o2].mjd) * 0.5 if arcs_use_all_obs else pt
        ra, dec = gc_pos(fit, new_mjd)
        result = _replace_pos(base, new_mjd, ra, dec)
        return result, 0.0

    # 3+ points
    if arcs_use_all_obs:
        is_ = (o1 + o2) // 2
        tr = olist[is_].mjd
        if is_ + is_ < o1 + o2:
            tr = (tr + olist[is_ + 1].mjd) * 0.5
    else:
        if o1 == 0:
            endpoint = olist[o2]
            dt = endpoint.mjd - pt
        else:
            endpoint = olist[o1]
            dt = pt - endpoint.mjd
        tr = endpoint.mjd if dt < 0 else pt

    span = olist[o1:o2 + 1]
    fit = gc_fit([o.mjd for o in span], [(o.ra, o.dec) for o in span])
    ra, dec = gc_pos(fit, tr)
    result = _replace_pos(olist[o1], tr, ra, dec)
    rms_val = gc_rms(fit)
    return result, rms_val


def _replace_pos(base: Observation, mjd: float, ra: float, dec: float) -> Observation:
    obs = Observation(**vars(base))
    obs.mjd, obs.ra, obs.dec = mjd, ra, dec
    return obs


@dataclass
class MotionVector:
    obs_pair: tuple           # (Observation, Observation)
    obs_err_rms: tuple        # (float, float) -- great-circle rms feeding clip_err, arcsec
    rms: float                # whole-tracklet great-circle RMS, arcsec (0 if nObs == 2)


def _is_spacebased_site(obs: Observation, site_table: SiteTable) -> bool:
    """Whether obs's *site table entry* has zero parallax (d2math.c's
    twoObs checks this directly, independent of the observation's own
    ``spacebased``/earth_observer fields -- a geocenter code like 500,
    or any obscode absent from the table, has rhoCosPhi==rhoSinPhi==0
    by construction and counts as space-based here)."""
    site = site_table[parse_cod3(obs.obscode)]
    return site.rho_cos_phi == 0.0 and site.rho_sin_phi == 0.0


def two_obs(olist: Sequence[Observation], site_table: SiteTable) -> MotionVector:
    """Select or synthesize the two observations defining the tracklet's
    motion vector (d2math.c: twoObs)."""
    n = len(olist)
    pair0, pair1 = olist[0], olist[-1]

    if n == 2:
        return MotionVector((pair0, pair1), (0.0, 0.0), 0.0)

    fit = gc_fit([o.mjd for o in olist], [(o.ra, o.dec) for o in olist])
    tk_rms = gc_rms(fit)

    all_same = all(o.obscode == olist[0].obscode for o in olist)
    space_based = any(_is_spacebased_site(o, site_table) for o in olist)

    whole = math.floor((n - 1) / 6.0)
    fs = (n - 1) / 6.0 - whole
    is_ = int(whole)

    if space_based:
        pair0 = olist[is_]
        pair1 = olist[n - 1 - is_]
        return MotionVector((pair0, pair1), (0.0, 0.0), tk_rms)

    t17 = olist[is_].mjd + (olist[is_ + 1].mjd - olist[is_].mjd) * fs
    is2 = n - 1 - is_
    t83 = olist[is2].mjd - (olist[is2].mjd - olist[is2 - 1].mjd) * fs

    if all_same and (olist[-1].mjd - olist[0].mjd) < 0.125:
        ra0, dec0 = gc_pos(fit, t17)
        ra1, dec1 = gc_pos(fit, t83)
        pair0 = _replace_pos(olist[0], t17, ra0, dec0)
        pair1 = _replace_pos(olist[-1], t83, ra1, dec1)
        return MotionVector((pair0, pair1), (tk_rms, tk_rms), tk_rms)

    # general case: split off same-site, <3hr initial/final sub-arcs
    o1, o2 = 0, n - 1
    code1, code2 = olist[o1].obscode, olist[o2].obscode
    t1, t2 = olist[o1].mjd, olist[o2].mjd

    while o2 > o1 + 1:
        dt1 = olist[o1 + 1].mjd - t1
        if olist[o1 + 1].obscode != code1 or dt1 > 0.125:
            while True:
                o = o2 - 1
                if o == o1 or olist[o].obscode != code2 or t2 - olist[o].mjd > 0.125:
                    break
                o2 = o
            break
        dt2 = t2 - olist[o2 - 1].mjd
        if olist[o2 - 1].obscode != code2 or dt2 > 0.125:
            while True:
                o = o1 + 1
                if o == o2 or olist[o].obscode != code1 or olist[o].mjd - t1 > 0.125:
                    break
                o1 = o
            break
        if dt1 < dt2:
            o1 += 1
        else:
            o2 -= 1

    pair0, rms0 = _one_obs(olist, 0, o1, o2 == o1 + 1, t17)
    pair1, rms1 = _one_obs(olist, o2, n - 1, o2 == o1 + 1, t83)
    return MotionVector((pair0, pair1), (rms0, rms1), tk_rms)
