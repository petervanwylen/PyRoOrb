"""Clause statistics for the thin-profile gates, measured whatever the gate did.

:func:`itf.search._show_lone_night` short-circuits: the first clause that
refuses returns, and the rest are never computed.  That is right for a search
(the cheap tests run first so the expensive ones are rarely reached) and wrong
for a study -- you cannot ask "would loosening clause 3 have shown this
correct night?" of a run that stopped at clause 2.

So this module recomputes the WHOLE clause set for one attempt, as pure
measurement: no state is written, no verdict is changed, and the search's own
outcome is recorded alongside so the two can be compared.  It runs only when
:mod:`itf.gatelog` is switched on.

The numbers are deliberately raw -- chi2 and its cap, Delta-chi2 and its
budget, trend swing and the arc's own reference, every ML probability in the
field -- so a re-calibration can sweep thresholds offline without re-running
anything.  A row that carried only "clause 2 failed" would force a re-run per
candidate threshold, which for live searches is days.
"""

__all__ = ["one_night_row", "confirm_row", "count_linking_rivals"]


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as exc:          # instrumentation never fails a search
        return {"error": f"{type(exc).__name__}: {exc}"} if default is None \
            else default


def one_night_row(ctx, trial, obs_out, trial_meta, trial_opp_profile, depth,
                  tracklets_all, shown, reason, run_exhaustion=True):
    """Every statistic the one-night show consults, for one attempt.

    ``shown`` is what the gate actually decided (True = surfaced as a 0.6),
    ``reason`` the sentence it recorded.  Everything else is measurement.
    """
    from scipy.stats import chi2 as _chi2

    from . import residual_trend as _trend
    from .fitter import subset_chi2, subset_rms
    from .obs80_fields import count_obs, obs_ids
    from .opp_profile import count_spans, OppProfileGrade
    from .search import (SeedPreservation, _exhaustion_verdict,
                         _linked_tracklets_as_swept, _seed_pre_rms,
                         seed_preserve_cap)

    cfg = ctx.cfg
    row = {
        "depth": depth,
        # the gate log keeps the legacy codes on disk
        "curr_opps": trial_opp_profile.legacy_code,
        "orig_opps": ctx.initial_opp_profile.legacy_code,
        "shown": bool(shown),
        "reason": reason,
        "new_trksubs": sorted({l[5:12].strip()
                               for l in (obs_out or "").splitlines()
                               if l.strip()}),
        "n_new_obs": count_obs(obs_out) if obs_out else 0,
        "n_new_nights": (count_spans(obs_out) if obs_out          # spans (opp_profile.SPAN_DAYS)
                         and count_obs(obs_out) else 0),
        "n_rung_tracklets": len((trial_meta or {}).get("tracklets") or []),
        "trial_used": getattr(trial, "used", None),
        "trial_count": getattr(trial, "count", None),
        "trial_rms": _safe(lambda: round(float(trial.rms), 4), None),
        # --- the seed, i.e. what the claim rests on ---
        "seed_U": _safe(lambda: round(float(ctx.pub.U), 3), None),
        "seed_arc_days": _safe(
            lambda: round(float(ctx.pub.arclength_days), 3), None),
        "seed_used": getattr(ctx.pub, "used", None),
        "seed_count": getattr(ctx.pub, "count", None),
        "seed_rms": _safe(lambda: round(float(ctx.pub.rms), 4), None),
        "cap_seed_U": cfg.single_night_show_max_U,
    }

    # --- clause 1: opposition profile (the row key stays `shape_ok`) --------
    row["shape_ok"] = bool(
        depth == 0 and trial_opp_profile.grade is OppProfileGrade.TWO_THIN
        and ctx.initial_opp_profile.grade is OppProfileGrade.ONE
        and row["n_new_nights"] == 1
        and (row["seed_U"] is not None
             and row["seed_U"] < cfg.single_night_show_max_U)
        and trial_meta and (trial_meta.get("tracklets") or []))

    # --- the ML field --------------------------------------------------------
    ml_ps = sorted((t["ml_p"] for t in (tracklets_all or [])
                    if t.get("ml_p") is not None), reverse=True)
    row["n_scored"] = len(ml_ps)
    row["best_ml"] = round(ml_ps[0], 5) if ml_ps else None
    row["ml_top10"] = [round(p, 5) for p in ml_ps[:10]]
    row["winner_ml"] = _safe(
        lambda: max(t.get("ml_p") for t in trial_meta["tracklets"]
                    if t.get("ml_p") is not None), None)
    row["cap_ml_abs"] = cfg.show_min_ml_p_abs
    row["cap_ml_multi"] = cfg.show_multi_min_ml_p

    have_fits = ctx.pub.fit is not None and trial.fit is not None
    row["have_fits"] = have_fits
    if not have_fits:
        return row

    # --- clause 2a: does the new night fit its OWN sigma? --------------------
    new_ids = obs_ids(obs_out)
    chi2_new, n_new = _safe(
        lambda: subset_chi2(trial.fit, new_ids, station_floor=True,
                            included_only=False), (None, 0))
    row["n_new_in_fit"] = n_new
    if n_new:
        dof = 2 * n_new
        cap = float(_chi2.ppf(cfg.show_new_tracklet_conf, dof))
        row.update(new_chi2=round(chi2_new, 4), new_dof=dof,
                   new_chi2_cap=round(cap, 3),
                   new_redchi2=round(chi2_new / dof, 4),
                   new_conf=cfg.show_new_tracklet_conf,
                   clause_new_chi2_ok=bool(chi2_new <= cap))
        # ...and each tracklet of the night on its own
        per = []
        for tag, ids in _safe(lambda: _linked_tracklets_as_swept(ctx, obs_out),
                              []) or []:
            c2_t, n_t = _safe(
                lambda ids=ids: subset_chi2(trial.fit, set(ids),
                                            station_floor=True,
                                            included_only=False), (None, 0))
            if not n_t:
                continue
            cap_t = float(_chi2.ppf(cfg.show_new_tracklet_conf, 2 * n_t))
            per.append({"tag": tag, "chi2": round(c2_t, 4), "n": n_t,
                        "cap": round(cap_t, 3),
                        "redchi2": round(c2_t / (2 * n_t), 4)})
        row["per_tracklet"] = per
        row["clause_per_tracklet_ok"] = all(t["chi2"] <= t["cap"] for t in per)

    # --- clause 2b: what it does to the object's OWN astrometry -------------
    def _seed_block():
        seed = SeedPreservation(ctx, trial)
        cap = seed_preserve_cap(cfg.show_seed_preserve_conf,
                                cfg.accept_seed_preserve_frac, seed.pre)
        pre_rms = _seed_pre_rms(ctx)
        post_rms, n_rms = subset_rms(trial.fit, ctx.pub_ids)
        sigma = ((seed.post / (2.0 * seed.n_post)) ** 0.5 if seed.n_post
                 else None)
        warn_lvl = (max(pre_rms * cfg.warn_rms_factor,
                        pre_rms + cfg.veto_rms_floor_arcsec)
                    if pre_rms else None)
        return {
            "seed_missing": seed.missing, "seed_n_ids": len(seed.ids),
            "seed_chi2_pre": round(seed.pre, 4),
            "seed_chi2_post": round(seed.post, 4),
            "seed_dchi2": round(seed.delta, 4),
            "seed_dchi2_cap": round(cap, 4),
            "seed_dchi2_frac": (round(seed.delta / cap, 4) if cap else None),
            # PER-OBSERVATION, deliberately -- the one exception to "reported
            # is per-axis" (fitter's docstring).  These rows are a study feed,
            # not a report: they are appended to a JSONL that earlier runs have
            # already written, and re-basing the currency mid-file would make
            # rows from either side of the change silently incomparable.  A
            # consumer that shows one to a person converts with fitter.per_axis.
            "seed_rms_pre": round(pre_rms, 4) if pre_rms else None,
            "seed_rms_post": round(post_rms, 4) if n_rms else None,
            "seed_rms_ratio": (round(post_rms / pre_rms, 4)
                               if n_rms and pre_rms else None),
            "seed_sigma_post": round(sigma, 4) if sigma is not None else None,
            "seed_moved": bool(n_rms and pre_rms
                               and post_rms >= warn_lvl),
            "clause_seed_ok": bool(not seed.missing and seed.delta <= cap),
        }
    row.update(_safe(_seed_block, {}) or {})

    # --- clause 2c: the residual TREND ---------------------------------------
    def _trend_block():
        ref = _trend.trend_reference(trial.fit, ctx.pub_ids)
        linked = _trend.tracklet_trends(trial.fit, new_ids)
        worst = linked[0] if linked else None
        fires = _trend.measure_trends(trial.fit, new_ids, ctx.pub_ids)
        out = {"ramp_ref": round(ref, 4), "ramp_fires": bool(fires),
               "clause_ramp_ok": not fires, "n_ramp_measured": len(linked)}
        if worst:
            out.update(ramp_swing=round(worst["swing"], 4),
                       ramp_swing_loo=round(worst["swing_loo"], 4),
                       ramp_t=round(worst["t"], 4),
                       ramp_mono=round(worst["mono"], 4),
                       ramp_mono_loo=round(worst["mono_loo"], 4),
                       ramp_loo_tested=bool(worst["loo_tested"]),
                       ramp_n=worst["n"], ramp_axis=worst["axis"],
                       ramp_ratio=(round(abs(worst["swing_loo"]) / ref, 4)
                                   if ref > 0 else None))
        return out
    row.update(_safe(_trend_block, {}) or {})

    # --- clause 3: is it the ONLY one that fits? (spends trial fits) --------
    if run_exhaustion:
        def _exhaust_block():
            sole, ex = _exhaustion_verdict(ctx, trial_meta, tracklets_all, depth)
            return {"sole_standout": bool(sole), "exhaustion": ex,
                    "clause_exclusive_ok": bool(
                        sole or (row["best_ml"] is not None
                                 and row["best_ml"] >= cfg.show_multi_min_ml_p))}
        row.update(_safe(_exhaust_block, {}) or {})

    # HOW MANY fit, not just whether one does -- on the shows that were made
    # WITHOUT exclusivity being measured, which is the only place the question
    # is open.  Costs K trial fits, so it runs only when a study asks for it.
    from . import gatelog
    n_top = gatelog.n_rivals()
    if (n_top and shown and not row.get("sole_standout")
            and trial_meta and (trial_meta.get("tracklets") or [])):
        row["rival_count"] = _safe(
            lambda: count_linking_rivals(ctx, trial_meta, tracklets_all, n_top), {})
    return row


def count_linking_rivals(ctx, trial_meta, tracklets_all, n_top=6):
    """How many of the top ``n_top`` rivals ALSO link?

    Lives in :func:`itf.search.count_linking_rivals` -- it started here as
    study instrumentation and became a gate clause, and one implementation is
    what keeps the measurement and the shipped behaviour from drifting apart
    (an earlier study's separate copy drifted on its tracklet unit).
    """
    from .search import count_linking_rivals as _count
    return _count(ctx, trial_meta, tracklets_all, n_top)


def confirm_row(ctx, trial, combined, trial_meta, depth, same_opp, confirmed):
    """The evidence the three-thin upgrade weighs, and what it decided.

    Every path's inputs are recorded whether or not that path fired, so the
    study can ask which threshold refused a correct pair -- the seed-strength
    arc/U pair, the same-opposition arc/U pair, and the top-pair identity and
    residual test.
    """
    from .fitter import residuals_by_obs_id
    from .search import _linked_ids

    cfg = ctx.cfg
    row = {
        "depth": depth,
        "confirmed": bool(confirmed),
        "seed_U": _safe(lambda: round(float(ctx.pub.U), 3), None),
        "seed_arc_days": _safe(
            lambda: round(float(ctx.pub.arclength_days), 3), None),
        "cap_seed_arc_days": cfg.confirm_seed_arc_days,
        "cap_seed_U": cfg.confirm_seed_U_max,
        "cap_min_ml_p": cfg.confirm_min_ml_p,
        "cap_max_resid": cfg.confirm_max_resid_arcsec,
        "linked_trksubs": sorted({l[5:12].strip()
                                  for l in (combined or "").splitlines()
                                  if l.strip() and l[5:12].strip()}),
        # the seed's own ids, so an analysis can tell the object's published
        # designation apart from a linked tracklet without re-packing anything
        "own_trksubs": sorted({l[5:12].strip()
                               for l in (ctx.pub_obs80 or "").splitlines()
                               if l.strip() and l[5:12].strip()}),
    }
    if same_opp is not None:
        so = same_opp[0]
        row.update(same_opp_arc_days=_safe(
            lambda: round(float(so.arclength_days), 3), None),
            same_opp_U=_safe(lambda: round(float(so.U), 3), None),
            same_opp_rms=_safe(lambda: round(float(so.rms), 4), None),
            same_opp_used=getattr(so, "used", None),
            same_opp_count=getattr(so, "count", None))

    top2 = getattr(ctx, "depth0_top2", None)
    if top2 and len(top2) == 2:
        row["top2_tags"] = [t.get("tag") for t in top2]
        row["top2_ml"] = [t.get("ml_p") for t in top2]
        row["top2_is_linkage"] = _safe(
            lambda: bool(_linked_ids(ctx, combined)
                         == (top2[0]["ids"] | top2[1]["ids"])), None)

        def _worst():
            resid = residuals_by_obs_id(trial.fit)
            ids = top2[0]["ids"] | top2[1]["ids"]
            vals = [resid.get(i) for i in ids if resid.get(i) is not None]
            return round(max(vals), 4) if vals else None
        row["top2_worst_resid"] = _safe(_worst, None)

    if trial_meta:
        row["rung_is_top_pair"] = bool(trial_meta.get("is_top_pair"))
        row["rung_ml_ps"] = trial_meta.get("ml_ps")
    return row
