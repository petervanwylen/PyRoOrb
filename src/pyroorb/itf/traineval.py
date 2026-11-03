"""Dataset audit + head-to-head model evaluation on a held-out eval split.

Two jobs, both read the CSVs :mod:`itf.collect` writes:

``audit`` -- what did a collection run actually produce?  Positives by stratum,
variant-corridor coverage, how often the per-window cap bit, and how many
PLANTED positives sat in a window where the cap would have discarded a real
tracklet of the same geometry (``would_be_capped``).  That last number is the
one the pilot exists to measure: it is a live recall risk the recovery harness
cannot otherwise see, because plants are injected past the cap.

``compare`` -- score two models on the SAME eval rows and report where they
differ.  Global AUC is the flattering metric (context features let a model
separate populations rather than candidates); the honest ones are WITHIN-WINDOW
ranking, where candidates compete inside a single corridor so the context
cancels, and CALIBRATION, because the search now reads ``ml_p`` as an absolute
probability at four configured thresholds rather than as a rank.
"""

import math

import numpy as np

# thresholds the search compares ml_p against directly (itf/config.py)
DEPLOYED_THRESHOLDS = {
    "confirm_min_ml_p": 0.2,
    "show_min_ml_p_abs": 0.15,
    "show_likely_min_ml_p": 0.25,
    "show_multi_min_ml_p": 0.7,
}


def load(csv_path, split=None, neg_per_window=None, usecols=None):
    """The dataset, read the memory-bounded way.

    Delegates to :func:`training.read_dataset` rather than calling read_csv
    itself: a finished v3 CSV is 1.5 GB, pandas wants ~1.5x that to hold it,
    and this module's own audit was OOM-killed (exit 137, no output, no
    traceback) reading one.  The chunked reader exists for exactly that and
    there is no reason for two loaders with different memory behavior."""
    from .training import read_dataset
    return read_dataset(csv_path, split=split, neg_per_window=neg_per_window,
                        usecols=usecols, log=lambda *a: None)


def _auc(y, p):
    """ROC AUC via rank statistic; nan when one class is absent."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks over ties, or ties inflate the score
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _grouped_ranking(df, score_col, group_cols):
    """Mean within-group AUC and top-1 accuracy over groups that contain at
    least one positive AND one negative -- the groups where ranking is a
    question with an answer."""
    aucs, top1, n_groups = [], [], 0
    for _key, g in df.groupby(group_cols, sort=False, observed=True):
        y = g["label"].values
        if y.sum() == 0 or y.sum() == len(y):
            continue
        n_groups += 1
        aucs.append(_auc(y, g[score_col].values))
        top1.append(float(y[np.argmax(g[score_col].values)] == 1))
    return {"groups": n_groups,
            "auc": float(np.nanmean(aucs)) if aucs else float("nan"),
            "top1": float(np.mean(top1)) if top1 else float("nan")}


AUDIT_COLS = ["designation", "trkid", "label", "label_source", "split",
              "stratum",
              "is_variant", "truncated", "n_in_corridor", "cap_rank",
              "would_be_capped", "trial_U", "window_t"]


def audit(csv_path, log=print):
    """What a collection run produced, and what the cap did to it."""
    df = load(csv_path, usecols=AUDIT_COLS)
    log(f"=== {csv_path} ===")
    pos = df[df["label"] == 1]
    # Count distinct truths by (designation, trkid), never trkid alone: the
    # harness names planted tracklets ZT00000, ZT00001, ... with a counter that
    # RESTARTS for every object (harness.make_fake_itf), so trkid collides
    # across objects.  Counting it alone reported 236 distinct tracklets for a
    # collection that holds 4,144 -- a 17x undercount that made replication
    # look 332x when it is 19x.
    n_trk = len(pos.groupby(["designation", "trkid"]).size()) if len(pos) else 0
    log(f"rows {len(df)}  objects {df['designation'].nunique()}  "
        f"positives {int(df['label'].sum())}")
    log(f"distinct planted tracklets (designation,trkid): {n_trk}"
        + (f"  -> each appears {len(pos) / n_trk:.0f}x" if n_trk else ""))
    log("\nby split / stratum (objects, rows, positives):")
    for (sp, st), g in df.groupby(["split", "stratum"], sort=True,
                                  observed=True):
        log(f"  {sp:>5} {st:>8}: {g['designation'].nunique():>4} obj  "
            f"{len(g):>8} rows  {int(g['label'].sum()):>5} pos")
    log("\nlabel sources:")
    for src, n in df["label_source"].value_counts().items():
        log(f"  {src:>16}: {n}")

    if "is_variant" in df:
        v = df[df["is_variant"] == 1]
        log(f"\nvariant-corridor rows: {len(v)} "
            f"({len(v) / max(1, len(df)) * 100:.1f}%), "
            f"{int(v['label'].sum())} positives, "
            f"{v['designation'].nunique()} objects")
    if "truncated" in df:
        t = df[df["truncated"] == 1]
        log(f"truncated-search rows: {len(t)} "
            f"({t['designation'].nunique()} objects)")

    # the cap question.  NB: "over the cap" must be read off the corridor
    # POPULATION, not off cap_rank -- a plant is ranked against the real
    # candidates on every window it lands in, so cap_rank >= 0 says nothing
    # about whether that window was saturated.
    if "n_in_corridor" in df:
        from .config import ItfConfig
        cap = ItfConfig().max_candidates_per_window
        over = df[df["n_in_corridor"] > cap]
        log(f"\ncap ({cap}/window): {len(over)} rows came from saturated "
            f"windows ({len(over) / max(1, len(df)) * 100:.1f}%)")
        if len(df):
            log(f"     corridor population: median "
                f"{df['n_in_corridor'].median():.0f}, "
                f"p95 {df['n_in_corridor'].quantile(0.95):.0f}, "
                f"max {df['n_in_corridor'].max():.0f}")
        pos = df[df["label"] == 1]
        lost = pos[pos["would_be_capped"] == 1.0]
        log(f"     PLANTED positives the cap would have discarded: "
            f"{len(lost)} of {len(pos)} "
            f"({len(lost) / max(1, len(pos)) * 100:.2f}%)"
            + (f" across {lost['designation'].nunique()} objects"
               if len(lost) else ""))
        if len(lost):
            log("     -> a real tracklet in those windows IS lost by the live "
                "search; this is a recall bug, not a data artifact")
    return df


def score_with(df, model_path):
    """P(match) for every row under one model artifact."""
    import json

    import pandas as pd
    import xgboost as xgb
    bst = xgb.Booster()
    bst.load_model(model_path)
    with open(model_path + ".meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    from .training import fill_categorical
    x = df.copy()
    for c, levels in meta["categories"].items():
        # meta records category levels for every CAT_COL, but a model trained
        # on a restricted feature list (--v2-features) does not consume them
        # all -- and the rows being scored may not carry the unused ones
        if c not in meta["features"]:
            continue
        x[c] = pd.Categorical(fill_categorical(x[c]).astype(str),
                              categories=levels)
    d = xgb.DMatrix(x[meta["features"]], enable_categorical=True)
    return bst.predict(d)


def recalibrate(csv_path, new_model, ref_model, split="eval", log=print):
    """What the four ``ml_p`` gates should be for ``new_model``.

    The constants in config.py (confirm_min_ml_p and the three show_* gates)
    are ABSOLUTE probabilities, chosen while the deployed model was
    ``ref_model``.  Swapping in a model whose scores are distributed
    differently silently re-points every one of them -- so this reports, per
    gate, the threshold at which the new model is at least as PRECISE as the
    old one is today, and what recall that buys.

    Precision is the quantity to hold fixed rather than recall, because these
    gates guard what gets submitted to the MPC and what gets shown as a claim:
    a false accept costs more than a missed one, and recall is the thing we
    are hoping to gain anyway.
    """
    import numpy as np
    df = load(csv_path, split=split)
    y = df["label"].values
    p_new = score_with(df, new_model)
    p_ref = score_with(df, ref_model)

    def pr(p, thr):
        sel = p >= thr
        tp = int((sel & (y == 1)).sum())
        return (tp / max(1, int((y == 1).sum())),          # recall
                tp / max(1, int(sel.sum())))               # precision

    log(f"=== gate recalibration on {split}: {len(df)} rows, "
        f"{int(y.sum())} positives ===")
    log("  gate                    current   ref P/R        new P/R at same "
        "thr   -> thr for equal precision")
    out = {}
    grid = np.unique(np.concatenate([np.linspace(0, 1, 2001),
                                     np.quantile(p_new, np.linspace(0, 1, 501))]))
    for name, thr in DEPLOYED_THRESHOLDS.items():
        r_ref, p_ref_prec = pr(p_ref, thr)
        r_new, p_new_prec = pr(p_new, thr)
        # lowest threshold whose precision matches the reference's -- lowest,
        # because among equally safe options we want the most recall
        best = None
        for t in grid:
            rr, pp = pr(p_new, t)
            if pp >= p_ref_prec:
                best = (t, rr, pp)
                break
        out[name] = {"current": thr, "ref": (r_ref, p_ref_prec),
                     "new_at_current": (r_new, p_new_prec),
                     "equal_precision": best}
        b = (f"{best[0]:.3f} (recall {best[1]:.3f}, precision {best[2]:.3f})"
             if best else "unreachable")
        log(f"  {name:>22} {thr:<8} R{r_ref:.3f}/P{p_ref_prec:.3f}   "
            f"R{r_new:.3f}/P{p_new_prec:.3f}   -> {b}")
    return out


def metric_block(df, score_col, log=print, label=""):
    """The standard metric set for one score column over labeled rows --
    global/within-window/within-object AUC + top-1, Brier, and recall/
    precision at the deployed thresholds.  One definition, used by
    ``compare`` (model artifacts) and ``oof`` (out-of-fold predictions), so
    a baseline computed here is comparable digit-for-digit with a model
    scored there."""
    y = df["label"].values
    p = df[score_col].values
    m = {"global_auc": _auc(y, p),
         "within_window": _grouped_ranking(df, score_col,
                                           ["designation", "window_t"]),
         "within_object": _grouped_ranking(df, score_col, ["designation"]),
         "brier": float(np.mean((p - y) ** 2)),
         "mean_p_neg": float(p[y == 0].mean()),
         "mean_p_pos": float(p[y == 1].mean()),
         "thresholds": {}}
    for tname, thr in DEPLOYED_THRESHOLDS.items():
        sel = p >= thr
        m["thresholds"][tname] = {
            "recall": float((sel & (y == 1)).sum() / max(1, (y == 1).sum())),
            "precision": float((sel & (y == 1)).sum() / max(1, sel.sum())),
        }
    if label:
        log(f"\n--- {label} ---")
        log(f"  global AUC        {m['global_auc']:.4f}")
        w = m["within_window"]
        log(f"  WITHIN-WINDOW     AUC {w['auc']:.4f}  top-1 {w['top1']:.3f}  "
            f"({w['groups']} contested windows)")
        o = m["within_object"]
        log(f"  within-object     AUC {o['auc']:.4f}  top-1 {o['top1']:.3f}")
        log(f"  Brier {m['brier']:.4f}   mean P: pos {m['mean_p_pos']:.3f} / "
            f"neg {m['mean_p_neg']:.4f}")
        for tname, thr in DEPLOYED_THRESHOLDS.items():
            t = m["thresholds"][tname]
            log(f"    {tname:>22} >= {thr:<5} recall {t['recall']:.3f}  "
                f"precision {t['precision']:.3f}")
    return m


def oof(csv_path, features=None, seed=0, test_frac=0.2, num_rounds=400,
        pred_out=None, log=print):
    """OUT-OF-FOLD predictions for a dataset under its own training recipe:
    leave-one-OBJECT-out, v2 hyperparameters, early stopping on an inner
    object-level split -- so every row is predicted by a model that never saw
    its object.

    This is the honest baseline for "how well does this dataset's model
    predict this dataset": in-sample predictions flatter (the booster has
    memorized its own rows), a single 80/20 split throws away the other 80%
    of eval rows.  The resulting p_oof column is the yardstick a LATER
    dataset's model must beat on these same rows (the v3-vs-v2 decider --
    see training_data/README.md).

    Replicates ``training.train_model``'s v2-era recipe exactly: same params,
    balanced scale_pos_weight computed on each fold's training side, early
    stopping (30 rounds) on an inner 20%-of-objects validation split.
    """
    import pandas as pd
    import xgboost as xgb

    from .training import V2_FEATURE_COLS, fill_categorical

    feats = features or V2_FEATURE_COLS
    cats = [c for c in ("stn", "band", "traj_category") if c in feats]
    need = sorted(set(feats) | {"designation", "label", "window_t", "trkid"})
    df = load(csv_path, usecols=need)
    for c in cats:
        df[c] = fill_categorical(df[c])
    objs = sorted(df["designation"].unique())
    log(f"OOF over {len(objs)} objects / {len(df)} rows / "
        f"{int(df['label'].sum())} pos ({csv_path})")

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": 5,
        "eta": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": seed,
    }
    p_oof = np.full(len(df), np.nan, dtype=float)
    for i, held in enumerate(objs):
        te_mask = (df["designation"] == held).values
        rest = df[~te_mask]
        rest_objs = np.array(sorted(rest["designation"].unique()))
        rng_f = np.random.default_rng(seed * 100003 + i)
        rng_f.shuffle(rest_objs)
        n_va = max(1, int(len(rest_objs) * test_frac))
        va_objs = set(rest_objs[:n_va])
        va = rest[rest["designation"].isin(va_objs)]
        tr = rest[~rest["designation"].isin(va_objs)]
        pos = max(1, int(tr["label"].sum()))
        params["scale_pos_weight"] = (len(tr) - pos) / pos
        dtr = xgb.DMatrix(tr[feats], label=tr["label"],
                          enable_categorical=True)
        dva = xgb.DMatrix(va[feats], label=va["label"],
                          enable_categorical=True)
        bst = xgb.train(params, dtr, num_boost_round=num_rounds,
                        evals=[(dva, "valid")], early_stopping_rounds=30,
                        verbose_eval=False)
        dte = xgb.DMatrix(df.loc[te_mask, feats], enable_categorical=True)
        p_oof[te_mask] = bst.predict(
            dte, iteration_range=(0, bst.best_iteration + 1))
        n_pos = int(df.loc[te_mask, "label"].sum())
        log(f"  [{i + 1}/{len(objs)}] {held}: {int(te_mask.sum())} rows "
            f"({n_pos} pos), best_iter {bst.best_iteration}")
    assert not np.isnan(p_oof).any(), "some rows never received a prediction"
    df["p_oof"] = p_oof
    m = metric_block(df, "p_oof", log=log, label=f"OOF baseline ({csv_path})")
    if pred_out:
        cols = ["designation", "trkid", "window_t", "label", "p_oof"]
        df[cols].to_csv(pred_out, index=False)
        log(f"OOF predictions -> {pred_out}")
    return df, m


def compare(csv_path, models, split="eval", log=print):
    """Head-to-head on the eval split.  ``models`` is {name: path}."""
    df = load(csv_path, split=split)
    log(f"=== eval: {len(df)} rows / {df['designation'].nunique()} objects / "
        f"{int(df['label'].sum())} positives ===")
    out = {}
    for name, path in models.items():
        p = score_with(df, path)
        df[f"p_{name}"] = p
        y = df["label"].values
        m = {"global_auc": _auc(y, p)}
        m["within_window"] = _grouped_ranking(df, f"p_{name}",
                                              ["designation", "window_t"])
        m["within_object"] = _grouped_ranking(df, f"p_{name}", ["designation"])
        # calibration matters now that gates read ml_p as a probability
        m["brier"] = float(np.mean((p - y) ** 2))
        m["mean_p_neg"] = float(p[y == 0].mean())
        m["mean_p_pos"] = float(p[y == 1].mean())
        m["thresholds"] = {}
        for tname, thr in DEPLOYED_THRESHOLDS.items():
            sel = p >= thr
            m["thresholds"][tname] = {
                "recall": float((sel & (y == 1)).sum() / max(1, (y == 1).sum())),
                "precision": float((sel & (y == 1)).sum() / max(1, sel.sum())),
            }
        out[name] = m
        log(f"\n--- {name} ({path}) ---")
        log(f"  global AUC        {m['global_auc']:.4f}")
        w = m["within_window"]
        log(f"  WITHIN-WINDOW     AUC {w['auc']:.4f}  top-1 {w['top1']:.3f}  "
            f"({w['groups']} contested windows)")
        o = m["within_object"]
        log(f"  within-object     AUC {o['auc']:.4f}  top-1 {o['top1']:.3f}")
        log(f"  Brier {m['brier']:.4f}   mean P: pos {m['mean_p_pos']:.3f} / "
            f"neg {m['mean_p_neg']:.4f}")
        log("  at the deployed thresholds:")
        for tname, thr in DEPLOYED_THRESHOLDS.items():
            t = m["thresholds"][tname]
            log(f"    {tname:>22} >= {thr:<5} recall {t['recall']:.3f}  "
                f"precision {t['precision']:.3f}")
    if len(models) == 2:
        (a, ma), (b, mb) = list(out.items())
        log(f"\n=== {b} vs {a} ===")
        log(f"  within-window AUC  {mb['within_window']['auc']:.4f} vs "
            f"{ma['within_window']['auc']:.4f}  "
            f"({mb['within_window']['auc'] - ma['within_window']['auc']:+.4f})")
        log(f"  within-window top1 {mb['within_window']['top1']:.3f} vs "
            f"{ma['within_window']['top1']:.3f}  "
            f"({mb['within_window']['top1'] - ma['within_window']['top1']:+.3f})")
        log(f"  global AUC         {mb['global_auc']:.4f} vs "
            f"{ma['global_auc']:.4f}")
    return out
