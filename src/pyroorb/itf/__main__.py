"""ITF-search CLI.

    python -m pyroorb.itf search "2017 CL37" [options]     # search the ITF for an object
    python -m pyroorb.itf recovery-suite [--n-objects 10] [--live]          # the recovery test suite
    python -m pyroorb.itf maintain <job>                   # index maintenance (see itf.maintain)

Runs anywhere ``pyroorb`` is importable (an install, or ``src/`` on PYTHONPATH).  The
same search is reachable from the main CLI as
``pyroorb DESIG --itf-search``.
"""

import argparse
import os
import sys

if __name__ == "__main__":
    # Run as a program (`python -m pyroorb.itf`): load the `.env` settings file first.
    from pyroorb.envfile import load_env_file
    load_env_file()

from .config import PREFER_CHOICES  # noqa: E402
from .verdict import SearchVerdict  # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "maintain":
        from .maintain import main as maintain_main
        return maintain_main(argv[1:])

    ap = argparse.ArgumentParser(prog="itf", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("search", help="search the ITF for a known object")
    p.add_argument("designations", nargs="+",
                   help="designation(s) of the seed object")
    p.add_argument("--max-uncert", type=float, default=None,
                   help="max ephemeris uncertainty, arcsec "
                        "(default: ItfConfig.max_uncert_deg)")
    p.add_argument("--uncert-mult", type=float, default=None,
                   help="corridor width multiplier "
                        "(default: ItfConfig.uncert_mult)")
    p.add_argument("--mag-limit", type=float, default=None,
                   help="skip corridor windows where the object is predicted "
                        "fainter than this, applied exactly (default: "
                        "ItfConfig.mag_limit, deepened per object to 1 mag past "
                        "its brightest since 1995)")
    p.add_argument("--thorough", action="store_true",
                   help="sweep to mag_limit_deep (the legacy 24.0) instead of "
                        "the default faint cut -- the thorough, slower search")
    p.add_argument("--mag-fallback", action="store_true",
                   help="EXPERIMENTAL, off by default: when nothing links, "
                        "re-sweep out to mag_limit_deep before giving up "
                        "(see ItfConfig.mag_limit_fallback)")
    p.add_argument("--candidate-source", choices=["index", "scan"], default="index",
                   help="candidate retrieval path (default: index fast path)")
    p.add_argument("--rank-mode", choices=["ml", "certainty", "legacy"],
                   default="ml",
                   help="how candidates are ranked: ml = the trained match-probability "
                        "model (default); certainty = the match chi2 of the "
                        "tracklet against the corridor; legacy = the ported "
                        "distance + rate-residual score")
    p.add_argument("--jitter", action="store_true",
                   help="variant-orbit jitter on the seed astrometry")
    p.add_argument("--submit", action="store_true",
                   help="actually transmit a found linkage to the MPC "
                        "(default: dry-run print)")
    p.add_argument("--prefer", choices=PREFER_CHOICES, default=None,
                   help="when the search finds both a submittable linkage and a "
                        "fuller unconfirmed (0.6) one: which leads -- complete "
                        "(default) or submittable (default with --submit)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--trace-level", type=int, default=None,
                   choices=[0, 1, 2, 3, 4],
                   help="live decision trace: 0 off, 1 headline (rounds, takes, "
                        "verdict), 2 narrative (default), 3 every trial and "
                        "candidate, 4 + per-window and per-DB-query rows "
                        "(--quiet implies 0)")

    p = sub.add_parser("train-model", help="train the xgboost tracklet-match "
                       "classifier on a traindata CSV")
    p.add_argument("csv", help="training CSV from traindata")
    p.add_argument("--model", default=None, help="save the model here (json)")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="train",
                   help="which side of the permanent wall to train on")
    p.add_argument("--no-balance", action="store_true",
                   help="drop scale_pos_weight -- worse ranking, honest "
                        "probabilities (the search reads ml_p as one)")
    p.add_argument("--object-weight", action="store_true",
                   help="give every object equal say in the loss (positives "
                        "are concentrated: one object held 88%% of them)")
    p.add_argument("--pos-weight", type=float, default=None,
                   help="scale_pos_weight override; <1 down-weights positives "
                        "to simulate the serve-time base rate")
    p.add_argument("--neg-per-window", type=int, default=None,
                   help="keep at most N randomly-chosen negatives per window "
                        "(a sample of the corridor, not a filter on it)")
    p.add_argument("--v2-features", action="store_true",
                   help="train on the frozen v2 feature set only, so the "
                        "model can be scored on v2-era rows (the v3-vs-v2 "
                        "decider needs this)")

    p = sub.add_parser("train-audit", help="what a traindata CSV contains, and "
                                          "what the per-window cap did to it")
    p.add_argument("csv")

    p = sub.add_parser("train-compare", help="score two models on the held-out "
                                            "eval split, head to head")
    p.add_argument("csv")
    p.add_argument("--model", action="append", required=True,
                   metavar="NAME=PATH", help="repeatable, e.g. v2=path.json")
    p.add_argument("--split", default="eval")

    p = sub.add_parser("train-eval", help="out-of-fold predictions for a "
                       "dataset under its own recipe (leave-one-object-out) "
                       "-- the honest self-baseline a later model must beat")
    p.add_argument("csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pred-out", default=None,
                   help="write per-row OOF predictions here (CSV)")

    p = sub.add_parser("train-recalibrate", help="what the four ml_p gates "
                       "should be for a new model, holding precision fixed")
    p.add_argument("csv")
    p.add_argument("--model", required=True, help="the NEW model")
    p.add_argument("--ref", required=True, help="the model the current "
                                                "thresholds were chosen for")
    p.add_argument("--split", default="eval")

    p = sub.add_parser("recovery-suite", help="multi-opposition recovery test suite")
    p.add_argument("--n-objects", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--objects", nargs="*", default=None,
                   help="explicit designations (overrides the random sample)")
    p.add_argument("--live", action="store_true",
                   help="merge the live ITF under the planted tracklets "
                        "(default: offline, fake-only)")
    p.add_argument("--seed-nights", type=int, default=None,
                   help="seed each object with only the first N nights of its "
                        "seed opposition, planting the rest (3 = the "
                        "discovery-arc stress shape: a ~2-day, 3-night seed)")
    p.add_argument("--report", default=None, help="write a JSON report here")
    p.add_argument("--verbose", action="store_true")

    args = ap.parse_args(argv)

    from .config import ItfConfig, default_prefer

    if args.cmd == "search":
        from .search import alternative_note, itf_search_live
        from .trace import as_tracer, DEFAULT_LEVEL
        cfg = ItfConfig(candidate_path=args.candidate_source, ranking_mode=args.rank_mode,
                        jitter=args.jitter, submit=args.submit,
                        prefer=args.prefer or default_prefer(args.submit))
        # --thorough starts where the (dormant) fallback would have ended up;
        # an explicit --mag-limit wins over both
        if args.thorough:
            cfg.apply_thorough()
        if args.mag_limit is not None:
            cfg.set_mag_limit(args.mag_limit)
        if args.mag_fallback:
            cfg.mag_limit_fallback = True
        max_unc = (args.max_uncert / 3600.0 if args.max_uncert is not None
                   else None)
        trace = None if (args.quiet or args.trace_level == 0) else print
        level = (args.trace_level if args.trace_level is not None
                 else DEFAULT_LEVEL)
        trace = as_tracer(trace, level) if trace is not None else None
        result = itf_search_live(args.designations, cfg=cfg,
                                 max_uncert_deg=max_unc,
                                 uncert_mult=args.uncert_mult,
                                 do_submit=args.submit,
                                 verbose=not args.quiet,
                                 trace=trace)
        print(f"status={result.verdict.legacy_status} {result.reason}")
        if alternative_note(result):
            print(alternative_note(result))
        for w in result.warnings or []:
            print(f"WARNING: {w}")
        # Advisory measurements that gate nothing (result.diagnostics).  A
        # different word on purpose: WARNING means "this linkage may be wrong
        # and you are being asked to look before submitting"; NOTE means "here
        # is a number you may want to see".  Printing the trend as a WARNING is
        # what made it read as a doubt back when it was one.
        for d in result.diagnostics or []:
            print(f"NOTE: {d}")
        if result.itf_obs80:
            print("linked ITF astrometry:" if result.verdict is SearchVerdict.LINKED else
                  "POSSIBLE (unconfirmed, NOT submitted) ITF astrometry:")
            print(result.itf_obs80)
        return 0 if result.verdict is SearchVerdict.LINKED else 1

    if args.cmd == "train-model":
        from .training import train_model, V2_FEATURE_COLS
        _bst, metrics = train_model(args.csv, model_out=args.model,
                                    test_frac=args.test_frac, seed=args.seed,
                                    balance=not args.no_balance,
                                    split=args.split,
                                    neg_per_window=args.neg_per_window,
                                    object_weight=args.object_weight,
                                    pos_weight=args.pos_weight,
                                    features=(V2_FEATURE_COLS
                                              if args.v2_features else None))
        return 0 if metrics["valid_auc"] > 0.5 else 1

    if args.cmd == "train-audit":
        from .traineval import audit
        audit(args.csv)
        return 0

    if args.cmd == "train-eval":
        from .traineval import oof
        oof(args.csv, seed=args.seed, pred_out=args.pred_out)
        return 0

    if args.cmd == "train-recalibrate":
        from .traineval import recalibrate
        recalibrate(args.csv, args.model, args.ref, split=args.split)
        return 0

    if args.cmd == "train-compare":
        from .traineval import compare
        models = {}
        for spec in args.model:
            name, _, path = spec.partition("=")
            if not path:
                return print(f"--model wants NAME=PATH, got {spec!r}") or 2
            models[name] = path
        compare(args.csv, models, split=args.split)
        return 0

    if args.cmd == "recovery-suite":
        from .harness import run_suite, sample_multi_opposition_objects
        cfg = ItfConfig()
        base = None
        if args.live:
            from .candidates import DbSource
            from .dbio import connect_pub
            base = DbSource(connect_pub(cfg), cfg,
                            reconnect=lambda: connect_pub(cfg))
        objects = args.objects or sample_multi_opposition_objects(args.n_objects, args.seed)
        reports = run_suite(objects, cfg=cfg, base_source=base,
                            verbose=args.verbose, report_path=args.report,
                            seed_nights=args.seed_nights)
        return 0 if all(r.full_recovery for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
