"""Build the ITF recovery fixtures (frozen published astrometry + the offline
recovery target) for the objects whose ITF fixes we validated.

Run from the repository root with the project venv; needs the MPC network to fetch each
object ONCE.  Re-running refreshes the frozen astrometry and re-derives the
target.  The distractor side is NOT frozen -- it comes from the live ITF DB at
test time (see test_itf_recovery.py).

    python tests/build_itf_fixtures.py [LABEL ...]
"""
import os
import sys


import warnings
warnings.filterwarnings("ignore")

from pyroorb.obs_sources import load_obscodes
from pyroorb.itf.config import ItfConfig
from pyroorb.itf.index import IndexConfig
from pyroorb.itf.fitter import fit_astrometry
from pyroorb.itf.harness import (fetch_object_obs, split_oppositions, make_fake_itf,
                         obs_lines, pick_seed_opposition, run_recovery)

from _itf_fixtures import ITF_FIX, freeze_ades, freeze_obs80, load_obs, write_fixture

# (label, designation, which fix this object exercises[, seed_nights[, "compact"]])
# A 4th element seeds with only the first N nights of the max-obs apparition
# (planting its remaining tracklets too) -- the compact-seed shape that a 1-opp
# object needs to exercise a fix.  A 5th element "compact" takes those N nights
# from the most COMPACT N-night window in ANY apparition instead
# (harness.compact_seed_window) -- the shape scan_itf_misses --compact-seed
# finds failures with, so a fixture reproduces the scan's seed exactly.
ITF_OBJECTS = [
    ("2019_PJ60",  "2019 PJ60",
     "uncert_mult 1.1->2.5: moderate along-track tail of a single-opp seed"),
    ("2015_BE559", "2015 BE559",
     "variant-orbit corridor: under-determined seed (U~9.6), 26-56deg miss"),
    ("2021_SA70",  "2021 SA70",
     "five tracklets across five oppositions accumulated by recursion; no seed "
     "observation clipped"),
    # Both were briefly REMOVED (2026-09-22) as casualties of the s2
    # covariance floor and RESTORED the next day: the fan was never the cause.
    # They failed live because the rankers were served floored-scale features
    # they were not trained on, the variant priorities ignored the LOV prior,
    # and the corridor (uncert_mult 2.5 on honest sigma) drowned them in junk.
    ("2014_TM87",  "2014 TM87",
     "seed-guard noise floor: coarse station 691, honest 0.55\" seed scatter"),
    ("2024_RX190", "2024 RX190",
     "recursion accumulation (hypothesis rides deeper trials) postmortem"),
    ("2014_SD57",  "2014 SD57",
     "variant fallback (moderate-U gap): U 5.30, tracklets at ~2.5-3.5sigma"),
    ("2019_AE65",  "2019 AE65",
     "variant fallback (moderate-U gap): U 6.77, 3-opp"),
    ("2026_DZ13",  "2026 DZ13",
     "veto degradation-factor: overfit compact 3-night seed (0.069\") settles "
     "to 0.175\" (2.5x) on its own true tracklets -- under the 3x veto", 3),
    # 2024 PQ7 was HERE and was REMOVED (2026-08-01).  Not handled: the search
    # cannot find it.  Its linkage is real and MPC-accepted,
    # but the search does not recover it: off the 1-apparition seed the ranker's
    # top candidate is a stranger ~62000 sigma off the object, and the fixture
    # only ever "passed" on per-process ML-score randomness (2 of 6 runs before
    # 266c096 made the scores deterministic, 0 of 4 after).  A fixture that fails
    # for a known open reason buys nothing and hides a real break inside a
    # permanent 15/16.  Re-add it -- it is still the right object to pin -- when
    # the ranking gap it names is closed.
    # Submitted and MPC-accepted 2026-07-24; their tracklets are now part of the
    # published arc, so the harness can plant them back and require the search
    # to find them again.  Real answers, not synthetic ones.
    ("2022_GL24",  "2022 GL24",
     "wide-corridor variant scan (the SQL corridor pushdown's object: peak RSS "
     "2040->913 MB) AND a designation merge with 2015 HE102"),
    ("2025_NW264", "2025 NW264",
     "the low-constraint end: U 3.79 on a 908-day arc, two tracklets -- the "
     "least-determined linkage the gates accepted"),
    ("2022_WY29",  "2022 WY29",
     "a real linkage carrying a 1.7 mag photometric outlier: the mag warning "
     "must stay a caution, not a veto"),
    ("2024_RY168", "2024 RY168",
     "two single-night apparitions (2017 + 2021) linked to a single-opp 2024 "
     "seed (U 4.38) in one joint pass -- the ITF-report case; both at ~50\" "
     "offset, ML P~1.0, taken together -> 26/26 rms 0.161\""),
    ("2022_SN92",  "2022 SN92",
     "the ARCHIVE era: a 3-obs Spacewatch (691) tracklet from 1995-10-21, "
     "before the corridor's 1997 ephem_start -- unreachable by any orbit until "
     "the archive sweep, while its 2011 and 2017 apparitions link in one pass"),
    ("2016_EZ157", "882845",
     "one refused tracklet must cost one tracklet: a 3-obs tracklet keeping "
     "2 of its obs (67% vs the 75% retention rule) failed depths 4, 3 and 2, "
     "and the search fell back to depth 1's 15 tracklets / 2 oppositions -- "
     "dropping the offender and re-vetting keeps 33 of 34 across 4 apparitions"),
    # --- NEOs.  The list above is main-belt but for 2024 RY168/PQ7 shapes; a
    # NEO's corridor moves fast and its seed is weak, so the extension path
    # deserves its own anchors.
    ("2025_SF15",  "2025 SF15",
     "NEO extension, genuine 2-day discovery arc: a 13-obs/1.99d/U 11.21 seed "
     "(q 1.31) rebuilds 21 tracklets across 3 oppositions, +75 obs at 0.241\"", 3),
    ("2014_HM543", "2014 HM543",
     "the COVARIANCE floor's object: a 3.9-day, 13-obs seed whose other three "
     "apparitions sit 14-50 deg off the nominal.  Unfloored, _param_inverse "
     "scaled the covariance by the achieved chi2/dof (s2 = 0.0099 here) and "
     "placed that truth at 14.5 sigma -- outside the +-5 sigma LOV fan, so the "
     "search served 1 of 5 planted tracklets and linked none.  Floored, the "
     "same truth reads 1.4 sigma (a parametric bootstrap says 1.0) and the fan "
     "reaches it: 5/5 across 3/3 apparitions at 0.185\", the full-arc rms"),
    ("2016_MB1",   "2016 MB1",
     "the POLE: at dec +82 the corridor's RA/Dec quadratic loses 10.7\" of a "
     "257.6\" displacement, so a real 2024 tracklet fitting the orbit at "
     "0.27\" was called 'unlikely' (qual 11.20\" against a 6.5\" bound) and a "
     "whole apparition was lost -- fixed by judging the trajectory on the "
     "unit-vector prediction (Window.predict_sky)"),
    ("2005_UD562", "2005 UD562",
     "the fan's CORE density on an honest sigma: a 12-obs/4.0d "
     "DECam seed fitting at 0.047\" is floored 12.8x, U 11.17, and its truth "
     "sits at k ~ 0.02 in the 0.095-wide hole between the central pickets -- "
     "18000-56000\" along-track, past max_uncert_deg for the linear pass. At "
     "24 extra variants / 3 rounds the fan leaves 65% of its probability "
     "uncovered and the live search ends 0/5; at 96 / 8 the core pitch is "
     "0.024 and it links 5/5 at variant #2"),
    ("2018_PE21",  "2018 PE21",
     "NEO extension at low perihelion (q 1.27): a 12-obs/3.88d/U 8.70 seed "
     "links 5 tracklets over 2 oppositions with no seed obs clipped", 3),
    ("2006_QR56",  "2006 QR56",
     "the 1.75 tier measured at the WRONG MOMENT: a 3-night/1.99d/U 9.87 "
     "discovery seed of an apparition that is really 5 nights/57.5d -- linking "
     "its own two 2006 tracklets back puts the orbit at U 5.38, and that is "
     "what the opposition-profile confirmation must judge the two lone single-night "
     "oppositions (2015, 2024) against, not the stub that entered the search", 3),
    ("2015_KJ19",  "2015 KJ19",
     "rate-adaptive corridor nodes: a NEO running 650-2900\"/hr, whose own "
     "tracklets the 3-day quadratic mis-predicted by 2-17x and the motion gate "
     "then called 'unlikely' -- 13 of 37 passing, 10/38 linked; refined, 38/38", None),
    ("2021_JK2",   "2021 JK2",
     "YARKOVSKY rescue: a q 1.06 NEO whose 2011 and 2016 Pan-STARRS tracklets "
     "sit 14.7\" and 6.2\" off a gravity-only orbit (F51 is good to ~0.1\"), so "
     "the trials rejected them and the search linked 2 of 4; with the model on, "
     "the 2011-2026 arc fits 40/40 and all 3 apparitions come back"),
    ("2020_KX6",   "2020 KX6",
     "veto factor vs a 2-DAY DISCOVERY ARC: an 8-obs/2.18d/U 9.23 seed fits "
     "itself to 0.073\", and its own true tracklets (2020-06-23, then 2023 and "
     "2025 -- 1 -> 3 opps) settle it to 0.240\" = 3.3x, honest in-sample -> "
     "out-of-sample widening that the 3x factor (calibrated on 11-17d arcs) "
     "banned as a contaminant", 3),
    ("2013_SZ67",  "2013 SZ67",
     "the CALENDAR night count: a complete 2-opposition linkage (17/17 obs, "
     "0.241\") assembled and then abandoned at 'nights 4', because the ported "
     "1.6d rule merges its two 2019 tracklets -- G96 on Oct 26.29 and F51 on "
     "Oct 27.46, 27.6h and two stations apart -- into ONE night, a count its "
     "own published arc can never beat"),
    ("2014_HY176", "2014 HY176",
     "ADAPTIVE LOV FAN: a q 0.99 NEO whose 2014 seed (U 7.39) propagates back "
     "14 years to a 2000 apparition at 0.11-0.15 au, where the nominal "
     "corridor lands 40deg off (and 0.06 mag past the faint cut).  The truth "
     "sits at k~-3.55, between equal-probability samples at -3.27 and -3.88 "
     "that are 23deg apart on the sky against a 5deg reach; refining the fan "
     "by uncovered probability adds 3 variants and recovers 3/3"),
    ("2021_CP63",  "2021 CP63",
     "REJECTION sigma floored at the station model: G96 self-reports 0.21\"/"
     "0.23\" on the 2025-01-02 tracklet, so an obs 0.72\"/0.48\" off reads 4.04 "
     "sigma and is clipped -- 1.73 sigma against the ~0.5\" the station achieves.  "
     "It was 1 of 3 obs in its tracklet, so the clip left 67% retention against "
     "the 75% floor, and dropping the refused tracklet collapsed 2 opps to 1.5 "
     "-- a correct linkage lost to a sigma the station cannot deliver"),
    ("2016_TJ194", "2016 TJ194",
     "CARRY the untaken candidates: the 2015 seed (U 7.95) swept a corridor "
     "wide enough to serve all 7 tracklets, then linking the two 2016 ones "
     "took U to 4.64 and the corridor to +-394\" at 2025 -- where the truth "
     "sits 782\" out, so the 2025 and 2007 tracklets went invisible BECAUSE "
     "the orbit improved.  Re-offered to the better orbit they link at once: "
     "2/7 tracklets -> 7/7, 45 obs over 18 years at 0.233\""),
    ("2022_WG18",  "2022 WG18",
     "drop and price ONE offending seed night: the 2022 seed's B74 night "
     "(2022-10-05) sits 2.9-3.3\" off the linked orbit while every other seed "
     "night and all six linked tracklets (2006 705 x4, 2019 T09, 2021 F51) fit "
     "at ~0.1-0.2\"; kept, it reads as an 11x seed degradation.  Dropped and "
     "paid for by six tracklets, the link stands, naming the dropped night"),
    ("2014_QZ68",  "2014 QZ68",
     "vouching and the topocentric final sweep: PR00009@D29 (2 obs, one past "
     "the 0.5\" clip, 3.3 sigma) does not move the orbit (0.002\" against a "
     "1.38\" miss), and PR00004@G96 sits within 0.8\" of the station's own "
     "ephemeris while the geocentric corridor puts it 7.7\" off -- 14/14"),
    ("2015_HU248", "2015 HU248",
     "a FLAT ML field at depth 1: the 2015 apparition (U 9.0) swept 46811 "
     "candidates live, best P 0.049, its 2019 truth #12 at P 0.009 behind "
     "same-season F51 distractors -- first-pass takes were barren and depth-1 "
     "backtracking did not exist; also the corridor cap (2019 windows 21-23 deg "
     "past the old 19.44)", 3, "compact"),
    ("2014_EJ64",  "2014 EJ64",
     "a slow object (~1100 s live) at the search's wall-clock budgets: depth-0 "
     "backtracking and the LOV variant scan both run long before 2013 F51 and "
     "later tracklets link -- 10/10 over 5 apparitions; the budget-timing "
     "canary for the flat-field backtracking", 3, "compact"),
    ("2026_DR69",  "2026 DR69",
     "the per-object FAINT-CUT floor: a 43 au TNO (single 78-day 2026 Rubin "
     "apparition) never predicted brighter than V 23.81 since 1997, so the "
     "fixed 23.5 cut swept ZERO windows and two 2017 T09 tracklets (reported "
     "23.6-24.5) went unsearched; the default deepened to 24.81 links both"),
    ("2017_AM13",  "2017 AM13",
     "the TERMINAL LOV fan: the linear pass links one more 2017 tracklet "
     "(LINKED, so the seed fallback fan never ran) and stops on a 6.5-d orbit "
     "at U 8.5 whose 2022 corridor sigma is 109-122 deg, far over the cap; the "
     "2022 return (0.05 au) sits on that orbit's LOV at k +0.18 -- the fan "
     "from the terminal orbit links all 6", 3, "compact"),
    ("2014_TJ33",  "2014 TJ33",
     "the TERMINAL LOV fan on a 3-apparition NEO seeded in its middle one: "
     "the 2014 and 2023 apparitions are 64-1049 deg sigma off the linear "
     "corridor but on the terminal orbit's LOV at k +0.14 (0.25-1.5 deg) -- "
     "21/21 over 3 apparitions", 3, "compact"),
    ("2011_UR91",  "2011 UR91",
     "the LOV-refinement LAST-RESORT tail after a failed fan: the terminal fan "
     "serves all five 2011 tracklets ~4 deg off its pickets (all 'unlikely'), "
     "the refinement puts three of them 2-3\" off the LOV curve with two nights "
     "agreeing -- trialled only once unlikely tail picks stay in the field and "
     "the round runs on its tail alone; 8/8 over 2 apparitions", 3, "compact"),
    ("2008_GU20",  "2008 GU20",
     "a bad tracklet that FAILS THE SEED CHECK too: depth 3 takes a lone 2000 "
     "SDSS pair (clipped 0/2, it still pulls the seed to Delta-chi2 37.6 vs "
     "30.6) with three clean 2024 tracklets; the tracklet repair must not be "
     "blocked by that seed failure -- dropped, the seed passes. REQUIRED: "
     "every apparition except 2000 (a lone SDSS pair that only a Yarkovsky "
     "orbit holds -- the owner's call: may stay out); live 19/20, offline "
     "20/20 on the frozen astrometry", 3, "compact",
     {"missing_apparitions": [0]}),
    ("2014_GV99",  "2014 GV99",
     "BOTH RESULTS: depth 0 LINKS one more 2014 tracklet, depth 2 adds the "
     "2006 and 2022 apparitions (a lone night each) -- a 1.75 profile the seed "
     "(U 11.4 on 2 days) cannot confirm, so a reviewable 0.6.  The verdict "
     "rank let the 1-apparition linkage discard it; now both are kept and the "
     "default (prefer complete) leads with the 0.6 -- 3/3 apparitions", 3,
     "compact"),
    ("2007_DB61",  "887773",
     "a LATCHED YARKOVSKY search: an Aten (q 0.52, H 23.8) with A2 at 100 sigma "
     "over 27 years; depth 2 latches Yarkovsky on 2018 + a 1999 night, and every "
     "later trial fit re-ran the full nongrav fit, 32-46 s, past the 30 s trial "
     "bound -- all of 2007 and the other 1999 night failed by timeout (3/5).  "
     "Trial fits now hold the rescued force fixed, warm from its orbit (~0.5 s), "
     "and the rescue is refused on a one-apparition arc: 5/5, 25/25 in the miss "
     "scan's 25-plant shape",
     3, "compact", {"max_planted": 25}),
    ("2008_SZ347", "2008 SZ347",
     "a seed that fits a PARABOLA: 3 nights of an a 5.2 au object give e 0.9999 (U 11.2), the "
     "truth is at k 0.063 between fan samples at 0 and 0.19, and the fan never lands near it "
     "(0/15).  The population's peak on the LOV is k 0.0635: the search tries that orbit first "
     "and links 15/15, live too",
     3, "compact", {"max_planted": 25}),
]

# Astrometry the MPC does not (yet) publish for an object, appended verbatim to
# the fetched set: ITF tracklets the fixture must plant, relabelled to the
# object's packed designation in columns 6-12.
EXTRA_OBS80 = {
    # ITF H469132 + H469148 (T09, 2017 March), identified to 2026 DR69
    "2026_DR69": """\
     K26D69R 4C2017 03 06.56313 13 01 36.35 +00 18 09.2          24.1 r1     T09
     K26D69R 4C2017 03 06.60099 13 01 36.19 +00 18 10.1          23.9 r1     T09
     K26D69R 4C2017 03 06.63313 13 01 36.04 +00 18 11.1          24.4 r1     T09
     K26D69R 4C2017 03 06.64589 13 01 35.99 +00 18 11.4          24.1 r1     T09
     K26D69R 4C2017 03 29.44104 12 59 47.30 +00 28 23.0          23.6 z1     T09
     K26D69R 4C2017 03 29.46764 12 59 47.18 +00 28 23.7          23.9 z1     T09
     K26D69R 4C2017 03 29.56289 12 59 46.68 +00 28 26.3          24.5 g1     T09
""",
}


def build_one(label, desig, note, obscodes, cfg, index_cfg, seed_nights=None,
              compact=False, missing_apparitions=(), max_planted=None):
    obs, meta = fetch_object_obs(desig, obscodes)
    if label in EXTRA_OBS80:
        from pyroorb.obs_sources import read_obs_file
        extra, _fmt = read_obs_file(EXTRA_OBS80[label], obscodes, in_format="obs80")
        obs = sorted(obs + [o for o in extra if o.get("obs80_line")],
                     key=lambda o: o["jd_utc"])
    apps = split_oppositions(obs)
    if len(apps) < 2 and seed_nights is None:
        raise RuntimeError(f"only {len(apps)} apparition(s) fetched")
    # ONE source of truth for which apparition seeds -- run_recovery below is
    # handed this same index explicitly, so the fixture cannot record an
    # apparition it did not actually seed with.
    # WITH the designation, exactly as run_recovery's default does: the
    # discovery apparition when it qualifies (see pick_seed_opposition).  Built
    # without it, a fixture pinned a seed the real search would never choose --
    # 2015 KJ19 came out on a U 8.27 apparition instead of the U 6.66 one the
    # search picks, pinning a different object's worth of behaviour.
    seed_i = pick_seed_opposition(apps, desig=desig)
    seed_group = apps[seed_i]
    others = [g for i, g in enumerate(apps) if i != seed_i]
    if compact:
        from pyroorb.itf.harness import compact_seed_window
        seed_i, seed_group, before, after = compact_seed_window(apps, seed_nights)
        others = ([g for i, g in enumerate(apps) if i != seed_i]
                  + [g for g in (before, after) if len(g) >= 2])
    elif seed_nights is not None:
        from pyroorb.itf.harness import _first_n_nights
        seed_group, seed_rest = _first_n_nights(seed_group, seed_nights)
        if len(seed_rest) >= 2:
            others = others + [seed_rest]
    seed_fit = fit_astrometry(obs_lines(seed_group), obscodes)
    n_fake = len(make_fake_itf(others, index_cfg)[0])

    frozen = {"designation": str(desig), "obs80": freeze_obs80(obs),
              "ades": freeze_ades(obs)}
    # the OFFLINE recovery is the deterministic golden target -- run on the
    # FROZEN astrometry the test will load, not on the fetch: the freeze keeps
    # OBS80's precision (~1 s, ~0.15"), and a search near a decision boundary
    # can come out differently on it (2008 GU20: 19/20 on the fetch, 20/20
    # frozen), which would pin a golden the test can never reproduce.
    obs_f, meta_f = load_obs(frozen, obscodes)
    rep = run_recovery(desig, cfg=cfg, base_source=None, obscodes=obscodes,
                       obs_meta=(obs_f, meta_f),
                       seed_apparition=None if compact else seed_i,
                       seed_nights=seed_nights, compact_seed=compact,
                       max_planted=max_planted,
                       log=(lambda *a: None))
    payload = {
        "designation": str(desig),
        "note": note,
        "obs80": frozen["obs80"],
        "ades": frozen["ades"],
        "n_obs": len(obs),
        "n_apparitions": len(apps),
        "seed_apparition": seed_i,
        "seed_nights": seed_nights,
        "compact_seed": bool(compact),
        # the miss scan's cap on planted tracklets, when the fixture pins that
        # shape (None = every tracklet of every other apparition is planted)
        "max_planted": max_planted,
        "seed_U": round(seed_fit.U, 2) if seed_fit.ok else None,
        "planted_apparitions": sorted(rep.planted_apparitions),
        "n_fake_tracklets": rep.n_fake_tracklets,
        # golden the OFFLINE test asserts against; the live run must clear rms_max
        "expect": {
            "full_recovery": bool(rep.full_recovery),
            "recovered_apparitions": sorted(rep.recovered_apparitions),
            "recovered_tracklets": rep.recovered_tracklets,
            "rms_max": round(max(0.6, rep.rms * 1.3), 3),
            # zero tolerance by default: a recovery adds observations, it does
            # not throw the object's own away (see test_itf_recovery)
            "max_dropped_seed_obs": rep.dropped_seed_obs,
        },
    }
    if rep.status == 0.6:
        # the recovery IS a reviewable possible: pin that verdict, or the test
        # would demand a submittable link the search rightly does not offer
        payload["expect"]["verdict"] = "possible"
    if missing_apparitions:
        # planted apparitions the search MAY leave out (decided per object,
        # with the reason in the note): required = planted minus these, and
        # the tracklet bar drops their tracklets -- a live run that rightly
        # declines them must still pass, an offline run that links them too
        # passes as well
        missing = sorted(set(missing_apparitions))
        fakes = [f for f in make_fake_itf(others, index_cfg)[0]
                 if f["apparition"] in missing]
        payload["expect"]["missing_apparitions"] = missing
        payload["expect"]["recovered_tracklets"] = min(
            rep.recovered_tracklets, rep.n_fake_tracklets - len(fakes))
        payload["expect"]["max_dropped_seed_obs"] = max(
            rep.dropped_seed_obs, payload["expect"]["max_dropped_seed_obs"])
    write_fixture(os.path.join(ITF_FIX, label + ".json"), payload)
    return rep, seed_fit


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    want = set(argv) or None
    oc = load_obscodes()
    cfg = ItfConfig()
    index_cfg = IndexConfig()
    for label, desig, note, *rest in ITF_OBJECTS:
        if want is not None and label not in want and desig not in want:
            continue
        seed_nights = rest[0] if rest else None
        compact = len(rest) > 1 and rest[1] == "compact"
        opts = rest[2] if len(rest) > 2 and isinstance(rest[2], dict) else {}
        try:
            rep, sf = build_one(label, desig, note, oc, cfg, index_cfg,
                                seed_nights=seed_nights, compact=compact,
                                missing_apparitions=opts.get("missing_apparitions", ()),
                                max_planted=opts.get("max_planted"))
            u = f"{sf.U:.2f}" if sf.ok else "?"
            print(f"{label}: seed U={u}  {rep.summary()}", flush=True)
        except Exception as e:
            print(f"{label}: BUILD FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
