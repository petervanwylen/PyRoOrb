"""Features of one 3-5 night single-opposition linkage (a set of tracklets).

Three families, kept apart so the model can be ablated by family:

* ``gap_*``  -- the night/gap STRUCTURE of the arc: a 1-1-30 four-nighter and a 2-4-26 four-nighter both span 32
  days, but the first's 2-day lever must reach 30 days out over a large
  uncertainty region, the second's 6-day lever reaches 26.  Rules mostly look
  at arc length; these look at how the arc is built.
* ``meta_*`` / ``phot_*`` / ``mot_*`` -- cheap observational metadata: the
  ``process_astrometry`` arc/gap/V-mag metrics (single
  opposition, so its opposition fields collapse), per-tracklet photometry
  with the MPC band->V table, motion and data quality.
* ``fit_*`` / ``lo_*`` -- one joint ASSIST fit plus ONE leave-out: the
  minority side of the largest gap is left out, the rest refitted, and the
  prediction ellipse at the left-out night measured -- and the number of OTHER
  pool tracklets that fall inside it counted (``lo_n_compet``).  That count is
  the empirical chance of a spurious match the arc structure allows.
"""
from __future__ import annotations

import math

import numpy as np

#: MPC v_conversion band->V shifts (blank band left
#: unadjusted).  Multi-character codes are ADES-only.
V_BAND = {
    'V': 0.0, 'v': 0.0, 'N': 0.0, 'T': 0.0, 'j': 0.0,
    'u': -2.50, 'U': -1.30, 'B': -0.80, 'g': -0.28, 'c': -0.05, 'w': +0.16,
    'r': +0.23, 'G': +0.28, 'o': +0.33, 'y': +0.36, 'z': +0.37, 'i': +0.39,
    'C': +0.40, 'R': +0.40, 'W': +0.40, 'L': +0.20, 'Y': +0.70, 'I': +0.80,
    'J': +1.20, 'H': +1.40, 'K': +1.70, ' ': 0.0, '': 0.0,
    'Lu': -1.771, 'Lg': -0.349, 'Lr': +0.214, 'Li': +0.373, 'Lz': +0.350,
    'Ly': +0.355, 'Ac': -0.05, 'Ao': +0.33, 'Bj': -0.80, 'Gb': +0.28,
    'Gr': +0.28, 'Ic': +0.80, 'Rc': +0.40, 'Uj': -1.30, 'Vj': 0.0,
    'Pg': -0.28, 'Pi': +0.39, 'Pr': +0.23, 'Pw': +0.16, 'Py': +0.36,
    'Pz': +0.37, 'Sg': -0.28, 'Si': +0.39, 'Sr': +0.23, 'Sz': +0.37,
    'UNK': 0.0, 'IR': 0.0, 'VR': 0.0, 'Lx': 0.0, 'W2': 0.0, 'W3': 0.0, 'W4': 0.0,
}

#: Same split as ``process_astrometry``: <0.5 d is the same night.
SAME_NIGHT_DAYS = 0.5

#: A night is "lone" when both its neighbours are at least this far away.
LONE_GAP_DAYS = 3.0

#: Leave-out competitor count: tracklets inside this many sigma of the
#: prediction (the ellipse, floored at the tracklet's own position error).
COMPET_NSIG = 3.0


def _vmag(rec):
    m = rec.get("mag")
    if m is None:
        return None
    return float(m) + V_BAND.get((rec.get("band") or " ").strip() or " ", 0.0)


def nights_of(tracklets):
    """Group tracklets (sorted by time) into nights; returns list of lists."""
    from pyroorb.orbit_common import night_index
    tr = sorted(tracklets, key=lambda t: t["epoch_mid_sec"])
    groups = {}
    for t, n in zip(tr, night_index([t["epoch_mid_sec"] / 86400 for t in tr])):
        groups.setdefault(n, []).append(t)
    return [groups[k] for k in sorted(groups)]


def gap_features(tracklets):
    """Night/gap structure.  Units: days."""
    nights = nights_of(tracklets)
    mid = np.array([np.mean([t["epoch_mid_sec"] for t in n]) for n in nights]) / 86400
    gaps = np.diff(mid)
    arc = float(mid[-1] - mid[0])
    f = {"gap_n_nights": len(nights), "gap_arc": arc}
    gs = sorted(gaps, reverse=True) + [0.0] * 4
    for k in range(4):
        f[f"gap_g{k + 1}"] = float(gs[k])
    f["gap_min"] = float(min(gaps)) if len(gaps) else 0.0
    f["gap_g1_frac"] = gs[0] / arc if arc > 0 else 0.0
    f["gap_g2_frac"] = gs[1] / arc if arc > 0 else 0.0
    j = int(np.argmax(gaps))                       # largest gap after night j
    left, right = mid[: j + 1], mid[j + 1:]
    nl, nr = len(left), len(right)
    sl, sr = float(left[-1] - left[0]), float(right[-1] - right[0])
    major_span = sl if nl > nr or (nl == nr and sl >= sr) else sr
    f.update(gap_n_left=nl, gap_n_right=nr, gap_n_minor=min(nl, nr),
             gap_n_major=max(nl, nr), gap_span_major=major_span,
             gap_span_minor=sr if major_span == sl else sl,
             gap_lever=gs[0] / max(major_span, 0.1),
             gap_lever_log=math.log10(gs[0] / max(major_span, 0.1) + 1e-3),
             gap_g1_pos=(j + 0.5) / max(len(gaps), 1))
    # the two-largest-gap lever: what the densest consecutive run must reach
    lone = 0
    for k in range(len(mid)):
        before = mid[k] - mid[k - 1] if k > 0 else np.inf
        after = mid[k + 1] - mid[k] if k + 1 < len(mid) else np.inf
        lone += before >= LONE_GAP_DAYS and after >= LONE_GAP_DAYS
    f["gap_n_lone"] = int(lone)
    # densest run: longest stretch of consecutive nights with gaps < 3 d
    best, cur = 1, 1
    for g in gaps:
        cur = cur + 1 if g < LONE_GAP_DAYS else 1
        best = max(best, cur)
    f["gap_dense_run"] = best
    return f, nights, j


def meta_features(recs):
    """``process_astrometry`` arc metrics (single opposition).

    Day arcs: runs of consecutive obs < 0.5 d apart (summed spacing).  Gap
    arcs: runs of consecutive >= 0.5 d spacings (summed).  Night count by a
    0.5 d step from each night's first obs.  V mags band-corrected.
    """
    t = sorted(r["_es"] / 86400 for r in recs)
    day_arcs, gap_arcs, day, gap = [], [], 0.0, 0.0
    nights, last_night = 1, t[0]
    for a, b in zip(t, t[1:]):
        g = b - a
        if b - last_night > SAME_NIGHT_DAYS:
            nights += 1
            last_night = b
        if g < SAME_NIGHT_DAYS:
            day += g
            if gap > 0:
                gap_arcs.append(gap)
                gap = 0.0
        else:
            gap += g
            if day > 0:
                day_arcs.append(day)
                day = 0.0
    if day > 0:
        day_arcs.append(day)
    if gap > 0:
        gap_arcs.append(gap)
    v = sorted(x for x in (_vmag(r) for r in recs) if x is not None)
    ga = sorted(gap_arcs, reverse=True) + [0.0, 0.0]
    return {
        "meta_longest_day_arc": max(day_arcs, default=0.0),
        "meta_longest_gap_arc": ga[0], "meta_second_gap_arc": ga[1],
        "meta_shortest_gap_arc": min(gap_arcs, default=0.0),
        "meta_total_arc": t[-1] - t[0], "meta_nights": nights,
        "meta_n_obs": len(recs),
        "meta_v_min": v[0] if v else None,
        "meta_v_2min": v[1] if len(v) > 1 else None,
        "meta_v_max": v[-1] if v else None,
        "meta_v_2max": v[-2] if len(v) > 1 else None,
        "meta_v_avg": float(np.mean(v)) if v else None,
    }


def phot_motion_features(tracklets, recs_by_trk):
    """Per-tracklet photometry and motion, cheap (no orbit)."""
    tm = []
    for t in tracklets:
        vs = [x for x in (_vmag(r) for r in recs_by_trk[t["tracklet_id"]]) if x is not None]
        if vs:
            tm.append(float(np.mean(vs)))
    f = {"phot_n_trk_mag": len(tm)}
    if len(tm) >= 2:
        med = float(np.median(tm))
        f.update(phot_trk_range=max(tm) - min(tm),
                 phot_trk_maxdev=max(abs(x - med) for x in tm),
                 phot_trk_sd=float(np.std(tm)))
    rates = np.array([t["rate_deg_day"] for t in tracklets])
    nobs = np.array([t["n_obs"] for t in tracklets])
    f.update(mot_rate_mean=float(rates.mean()),
             mot_rate_ratio=float(rates.max() / max(rates.min(), 1e-6)),
             mot_min_trk_obs=int(nobs.min()), mot_n_two_obs=int((nobs <= 2).sum()),
             mot_n_trk=len(tracklets),
             mot_n_stn=len({t["stn"] for t in tracklets}),
             mot_max_trk_chi2=max((t.get("fit_chi2_dof") or 0.0) for t in tracklets),
             mot_max_sigpos=max(math.hypot(t.get("sig_pos_e_arcsec") or 0,
                                           t.get("sig_pos_n_arcsec") or 0)
                                for t in tracklets))
    dec = np.radians([t["center_dec"] for t in tracklets])
    ra = np.radians([t["center_ra"] for t in tracklets])
    # ecliptic latitude of the mean direction
    eps = math.radians(23.4392911)
    v = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]).mean(1)
    v /= np.linalg.norm(v)
    f["mot_ecl_lat"] = math.degrees(math.asin(-math.sin(eps) * v[1] + math.cos(eps) * v[2]))
    return f


# --------------------------------------------------------------------- fit side
def engine_obs(recs, oc, label="HYP"):
    """obs_sbn records -> engine obs dicts, all under ONE hypothesis label so
    the engine treats a chimera as one object (the truth stays on ``obj``)."""
    from pyroorb import ades
    obs = ades.records_to_obs(recs, oc)
    # records_to_obs may SKIP records, so match back by obsid, never position
    tid = {r.get("obsid"): r.get("_tid") for r in recs}
    for o in obs:
        o["_tid"] = tid.get(o.get("obs_id"))
        o["designation"] = label
        o["packed"] = label
        o["packed_perm"] = None
    return obs


def fit_features(recs, tracklets, nights, j_gap, pool_index, oc):
    """Joint fit + leave-out features.  Never raises: failures are features."""
    import pyroorb.fit_object as fo
    from pyroorb.ident_check import tracklet_mag_audit
    from pyroorb.ephemerides import compute_ephemeris
    f = {}
    obs = engine_obs(recs, oc)
    if len(obs) != len(recs):
        f["fit_obs_lost"] = len(recs) - len(obs)
    obs.sort(key=lambda o: o["jd_utc"])
    try:
        fit = fo.fit_orbit(obs, oc, desig="HYP", meta={"designation": "HYP"})
    except Exception as e:                                    # noqa: BLE001
        f["fit_ok"] = 0
        f["fit_err"] = type(e).__name__
        return f
    f["fit_ok"] = 1
    f["fit_rms"] = fit.rms_residual()
    f["fit_rms_n"] = fit.rms_normalized()
    f["fit_n_rej"] = fit.n_excluded()
    f["fit_rej_frac"] = fit.n_excluded() / max(fit.n_total_observations(), 1)
    for name, fn in (("a", fit.semimajor_axis), ("e", fit.eccentricity),
                     ("i", fit.inclination), ("q", fit.perihelion_distance)):
        try:
            v, s = fn()
            f[f"fit_{name}"] = float(v)
            f[f"fit_sig_{name}"] = float(s) if s is not None else None
        except Exception:                                     # noqa: BLE001
            pass
    if f.get("fit_sig_a") and f.get("fit_a"):
        f["fit_sig_a_log"] = math.log10(max(f["fit_sig_a"], 1e-9))
    try:
        audit = tracklet_mag_audit(fit) or {}
        trk = audit.get("trk", [])
        dh = [abs(e["dH"]) for e in trk if e.get("dH") is not None]
        ps = [e["p"] for e in trk if e.get("p") is not None]
        f["fit_max_dH"] = max(dh) if dh else None
        f["fit_min_p_mag"] = min(ps) if ps else None
        f["fit_H"] = audit.get("H")
    except Exception:                                         # noqa: BLE001
        trk = []
    f.update(isolated_tracklet_features(tracklets, trk))
    # per-tracklet astrometric evidence, re-using the mislinkage scanner's
    # measured statistics (chi, z, walk, ramp) on this fit
    try:
        from pyroorb.mislink import scan_mislinkage as sm
        mask = np.asarray(fit._mask, dtype=bool)
        groups = sm._tracklet_groups(fit._obs, None)
        trends = sm._trends_for_groups(fit, fit._obs, groups)
        chis, zs, walks, ramps = [], [], [], []
        for i, g in enumerate(groups):
            kept = [k for k in g if mask[k]]
            c = sm._chi(fit, kept)
            z = sm._z_displacement(fit, kept)
            if c is not None:
                chis.append(c)
                zz = z[0] if z else 0.0
                walks.append(math.sqrt(max(c * c - zz * zz / max(len(kept), 1), 0.0)))
            if z:
                zs.append(abs(z[0]))
            r = trends.get(i)
            if r:
                ramps.append(abs(r["swing_loo"]))
        f["fit_max_chi"] = max(chis) if chis else None
        f["fit_max_z"] = max(zs) if zs else None
        f["fit_max_walk"] = max(walks) if walks else None
        f["fit_max_ramp"] = max(ramps) if ramps else None
    except Exception as e:                                    # noqa: BLE001
        f["fit_trk_err"] = type(e).__name__
    f.update(leaveout_features(obs, nights, j_gap, pool_index, oc, fit))
    f.update(chain_features(obs, nights, oc, fit))
    return f


def isolated_tracklet_features(tracklets, audit_trk):
    """The MOST ISOLATED tracklet, judged on its own.

    TG10: three O18 nights agree; a lone N94 tracklet four days later is
    0.78 mag bright (p = 0.06), walks (ramp 2.73) and degrades the rest
    (drop 1.49) -- every arm leaning the same way, none over its flag, and the
    arc-wide maxima cannot say it is the SAME tracklet each time.  These
    features name that tracklet and describe it:

    * ``iso_gap``  -- days to its nearest neighbour;
    * ``iso_ratio`` -- that gap over the median nearest-neighbour gap of the
      others (the scanner's isolation);
    * ``iso_stn_novel`` -- its station appears nowhere else in the arc (a
      second station's tracklet is where a stranger gets in);
    * ``iso_dH`` / ``iso_p`` -- its own photometric discrepancy.
    """
    f = {}
    if len(tracklets) < 3:
        return f
    mid = np.array([t["epoch_mid_sec"] for t in tracklets]) / 86400
    near = np.array([np.min(np.abs(np.delete(mid, i) - mid[i])) for i in range(len(mid))])
    i = int(np.argmax(near))
    others = np.delete(near, i)
    t = tracklets[i]
    f["iso_gap"] = float(near[i])
    f["iso_ratio"] = float(near[i] / max(np.median(others), 0.05))
    f["iso_stn_novel"] = int(sum(u["stn"] == t["stn"] for u in tracklets) == 1)
    jd0 = 2440587.5 + t["epoch_start_sec"] / 86400
    for e in audit_trk or []:
        if e.get("code") == t["stn"] and abs(e.get("jd0", 0) - jd0) < 0.1 \
                and e.get("dH") is not None:
            f["iso_dH"] = abs(float(e["dH"]))
            f["iso_p"] = float(e["p"]) if e.get("p") is not None else None
            break
    return f


def chain_features(obs, nights, oc, fit):
    """Chained reasoning as features.

    Trust grows outward from the best-defined part of the arc: the densest
    consecutive night pair is taken as the core (most tracklets, then most
    stations, then the shortest gap), and nights are added one at a time,
    nearest first.  Before each addition the current set is fitted and every
    tracklet of the incoming night is checked against the prediction:

    * position: Mahalanobis miss in the predicted ellipse (as ``lo_D``);
    * MOTION: the tracklet's measured sky rate against the predicted rate --
      "the right on-sky motion", which a chance tracklet rarely matches;
    * the change in rms_n when the night joins (2026 ST12: 1.07 -> 1.08).

    Features are the WORST step of each, plus the core's strength.  2026 KH
    and ST12 -- clearly real, flagged by the model -- motivated these.
    """
    import pyroorb.fit_object as fo
    from pyroorb.ephemerides import compute_ephemeris
    f = {}
    if len(nights) < 3:
        return f

    def stations(n):
        return {t["stn"] for t in n}

    pairs = [(len(nights[i]) + len(nights[i + 1]), len(stations(nights[i]) | stations(nights[i + 1])),
              -(nights[i + 1][0]["epoch_mid_sec"] - nights[i][-1]["epoch_mid_sec"]), i)
             for i in range(len(nights) - 1)]
    best = max(pairs)[3]
    core = {best, best + 1}
    f["core_n_trk"] = len(nights[best]) + len(nights[best + 1])
    f["core_n_stn"] = len(stations(nights[best]) | stations(nights[best + 1]))
    f["core_span"] = (nights[best + 1][-1]["epoch_mid_sec"] - nights[best][0]["epoch_mid_sec"]) / 86400
    f["core_n_obs"] = sum(t["n_obs"] for k in core for t in nights[k])
    mids = [np.mean([t["epoch_mid_sec"] for t in n]) for n in nights]
    have = set(core)
    max_d, max_rate, max_rate_n, max_drms = 0.0, 0.0, 0.0, 0.0
    prev_rms = None
    steps = 0
    while len(have) < len(nights):
        cur = [o for o in obs if o.get("_tid") in
               {t["tracklet_id"] for k in have for t in nights[k]}]
        rest = [k for k in range(len(nights)) if k not in have]
        nxt = min(rest, key=lambda k: min(abs(mids[k] - mids[h]) for h in have))
        if len(cur) < 4:
            break
        try:
            cfit = fo.fit_orbit(cur, oc, desig="HYP", meta={"designation": "HYP"})
        except Exception:                                     # noqa: BLE001
            f["chain_fail"] = 1
            break
        rms_now = cfit.rms_normalized()
        if prev_rms is not None:
            max_drms = max(max_drms, rms_now - prev_rms)
        prev_rms = rms_now
        for t in nights[nxt]:
            jd = 2440587.5 + t["epoch_mid_sec"] / 86400
            try:
                row = compute_ephemeris(cfit, oc, jd, jd, 1.0, site=t["stn"], full=False)[0]
            except Exception:                                 # noqa: BLE001
                continue
            a = max(float(row["unc_smaj_arcsec"]), 1.0)
            b = max(float(row["unc_smin_arcsec"]), 1.0)
            pa = math.radians(float(row["unc_pa_deg"]))
            cosd = math.cos(math.radians(t["center_dec"]))
            dE = ((t["center_ra"] - row["ra_deg"] + 180) % 360 - 180) * cosd * 3600
            dN = (t["center_dec"] - row["dec_deg"]) * 3600
            max_d = max(max_d, _mahal(dE, dN, a, b, pa))
            # motion, arcsec/min on sky; measured from the tracklet's own fit
            m, pm = float(row["motion_arcsec_min"]), math.radians(float(row["motion_pa_deg"]))
            pvE, pvN = m * math.sin(pm), m * math.cos(pm)
            k = 3600.0 / 1440.0                               # deg/day -> arcsec/min
            tvE = (t.get("dra_cosdec_rate") or 0.0) * k
            tvN = (t.get("ddec_rate") or 0.0) * k
            miss = math.hypot(tvE - pvE, tvN - pvN)
            sig_trk = math.hypot(t.get("sig_vel_e_degday") or 0.0, t.get("sig_vel_n_degday") or 0.0) * k
            # The PREDICTED rate is uncertain too: the position ellipse grows
            # ~linearly away from the fitted nights, so its growth rate (smaj /
            # time to the nearest fitted night) is the predicted-rate sigma.
            # Without it 2026 ST12's near-perfect 10/03 motion read 58.6 sigma
            # (raw miss 0.099"/min against a 1-day core 13 d away).
            dt_near = max(min(abs(t["epoch_mid_sec"] / 86400 - mids[h] / 86400) for h in have), 1.0)
            sig_pred = a / dt_near / 1440.0                   # arcsec/min
            sig = math.hypot(sig_trk, sig_pred)
            max_rate = max(max_rate, miss)
            max_rate_n = max(max_rate_n, miss / max(sig, 1e-4))
        have.add(nxt)
        steps += 1
    if steps:
        try:
            final_rms = fit.rms_normalized()
            if prev_rms is not None:
                max_drms = max(max_drms, final_rms - prev_rms)
        except Exception:                                     # noqa: BLE001
            pass
        f.update(chain_max_D=max_d, chain_max_D_log=math.log10(max_d + 1e-3),
                 chain_max_rate_miss=max_rate, chain_max_rate_n=max_rate_n,
                 chain_max_rate_n_log=math.log10(max_rate_n + 1e-3),
                 chain_max_drms=max_drms, chain_steps=steps)
    return f


def leaveout_features(obs, nights, j_gap, pool_index, oc, fit):
    """Leave out the MINORITY side of the largest gap, refit, and measure how
    big the prediction region is where the left-out night(s) sit -- and how
    many other pool tracklets live inside it."""
    import pyroorb.fit_object as fo
    from pyroorb.ephemerides import compute_ephemeris
    f = {}
    left = [t for n in nights[: j_gap + 1] for t in n]
    right = [t for n in nights[j_gap + 1:] for t in n]
    out_side = right if len(right) < len(left) or (
        len(right) == len(left)) else left
    out_ids = {t["tracklet_id"] for t in out_side}
    keep = [o for o in obs if o.get("_tid") not in out_ids]
    if len(keep) < 6 or len(keep) == len(obs):
        f["lo_ok"] = 0
        return f
    try:
        f2 = fo.fit_orbit(keep, oc, desig="HYP", meta={"designation": "HYP"})
    except Exception as e:                                    # noqa: BLE001
        f["lo_ok"] = 0
        f["lo_err"] = type(e).__name__
        return f
    f["lo_ok"] = 1
    smajs, misses, Ds, competitors = [], [], [], []
    for t in out_side:
        jd = 2440587.5 + t["epoch_mid_sec"] / 86400
        try:
            row = compute_ephemeris(f2, oc, jd, jd, 1.0, site=t["stn"], full=False)[0]
        except Exception:                                     # noqa: BLE001
            continue
        a = max(float(row["unc_smaj_arcsec"]), 1.0)
        b = max(float(row["unc_smin_arcsec"]), 1.0)
        pa = math.radians(float(row["unc_pa_deg"]))
        cosd = math.cos(math.radians(t["center_dec"]))
        dE = ((t["center_ra"] - row["ra_deg"] + 180) % 360 - 180) * cosd * 3600
        dN = (t["center_dec"] - row["dec_deg"]) * 3600
        smajs.append(a)
        misses.append(math.hypot(dE, dN))
        Ds.append(_mahal(dE, dN, a, b, pa))
        competitors.append(pool_index.count_in_ellipse(
            jd, row["ra_deg"], row["dec_deg"], COMPET_NSIG * a, COMPET_NSIG * b, pa,
            exclude=t["tracklet_id"]))
    if smajs:
        f["lo_smaj"] = max(smajs)
        f["lo_smaj_log"] = math.log10(max(smajs))
        f["lo_miss"] = max(misses)
        f["lo_D"] = max(Ds)
        f["lo_n_compet"] = max(competitors)
        f["lo_n_compet_log"] = math.log10(1 + max(competitors))
        f["lo_rms_n"] = f2.rms_normalized()
    return f


def _mahal(dE, dN, a, b, pa):
    """Mahalanobis distance of (dE, dN) in an ellipse with semi-axes a, b
    (arcsec) and position angle pa (radians, east of north) of a."""
    u = dE * math.sin(pa) + dN * math.cos(pa)                 # along major
    v = dE * math.cos(pa) - dN * math.sin(pa)                 # along minor
    return math.hypot(u / a, v / b)


class PoolIndex:
    """Tracklets of one pool, bucketed by night, for ellipse counts."""

    def __init__(self, tracklets):
        self.by_night = {}
        for t in tracklets:
            k = int(t["epoch_mid_sec"] // 86400)
            self.by_night.setdefault(k, []).append(t)
        self.arr = {k: (np.array([x["center_ra"] for x in v]),
                        np.array([x["center_dec"] for x in v]),
                        [x["tracklet_id"] for x in v])
                    for k, v in self.by_night.items()}

    def count_in_ellipse(self, jd, ra, dec, a, b, pa, exclude=None):
        k = int((jd - 2440587.5))
        n = 0
        for kk in (k - 1, k, k + 1):
            if kk not in self.arr:
                continue
            ras, decs, ids = self.arr[kk]
            cosd = math.cos(math.radians(dec))
            dE = ((ras - ra + 180) % 360 - 180) * cosd * 3600
            dN = (decs - dec) * 3600
            u = dE * math.sin(pa) + dN * math.cos(pa)
            v = dE * math.cos(pa) - dN * math.sin(pa)
            inside = (u / a) ** 2 + (v / b) ** 2 <= 1.0
            n += int(inside.sum()) - sum(1 for i, ok in zip(ids, inside) if ok and i == exclude)
        return n

    def density(self, jd, ra, dec, radius_deg=1.0):
        """Tracklets per square degree within radius on that night."""
        k = int(jd - 2440587.5)
        if k not in self.arr:
            return 0.0
        ras, decs, _ = self.arr[k]
        cosd = math.cos(math.radians(dec))
        d2 = (((ras - ra + 180) % 360 - 180) * cosd) ** 2 + (decs - dec) ** 2
        return float((d2 <= radius_deg ** 2).sum() / (math.pi * radius_deg ** 2))


def all_features(hyp_tracklets, recs_by_trk, pool_index, oc, *, with_fit=True):
    """Every feature for one hypothesis (list of tracklet dicts)."""
    recs = [r for t in hyp_tracklets for r in recs_by_trk[t["tracklet_id"]]]
    f, nights, j = gap_features(hyp_tracklets)
    f.update(meta_features(recs))
    f.update(phot_motion_features(hyp_tracklets, recs_by_trk))
    dens = [pool_index.density(2440587.5 + t["epoch_mid_sec"] / 86400,
                               t["center_ra"], t["center_dec"]) for t in hyp_tracklets]
    f["mot_density"] = float(np.mean(dens))
    if with_fit:
        for t in hyp_tracklets:
            for r in recs_by_trk[t["tracklet_id"]]:
                r["_tid"] = t["tracklet_id"]
        f.update(fit_features(recs, hyp_tracklets, nights, j, pool_index, oc))
    return f
