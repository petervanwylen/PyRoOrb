"""The v4 ranker: trained on serve-faithful decision rows with REAL ITF truths.

Data: ``python -m pyroorb.itf.exitf run`` collections (training.V4_ALL_COLS rows).
What differs from the v2/v3 line, and why:

* **Truths are real ITF tracklets** (label_source ``exitf``), not the object's
  published astrometry, so "publishedness" is not a shortcut the data can
  teach.  ``planted`` rows (weak-mode cases plant the rest of the as-of
  astrometry as well) are real truths of the WRONG population: by default
  they are dropped from training altogether -- neither positive nor negative.
* **Honest-scale features.**  The rows carry the engine's own covariance scale
  (no ``_pre_floor_scale`` shim); the model is served the same.
* **The decision context the old rows lacked**: whether the corridor is a LOV
  variant's (``in_variant_corridor``, which ``is_variant`` was not -- it stays
  set on a variant scan's deeper, ordinary corridors), the variant's |k| (its
  prior weight), and the continuous LOV offset geometry: the residual off the LOV curve
  at the candidate's own solved k, how far that k is from the serving
  variant's, and how many other nights agree on it.
* **Evaluated in the units the search consumes P in** (the search's own
  decision logs): the truth's rank within its ROUND and within its variant
  SCAN, not a pooled AUC -- pooled AUC saturated at ~0.999 on every v3 model
  while live order in the decisive windows failed.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from .training import CAT_COLS, FEATURE_COLS, fill_categorical

#: v3/v2 features the v4 model does NOT take: ``is_variant`` is replaced by
#: ``in_variant_corridor`` (see the module docstring).
_DROP = {"is_variant"}

V4_BASE = [c for c in FEATURE_COLS if c not in _DROP]
V4_CONTEXT = ["in_variant_corridor", "abs_variant_k",
              "lov_resid", "lov_abs_dk", "lov_nights", "lov_abs_kstar",
              "lov_resid_over_width"]
V4_FEATURES = V4_BASE + V4_CONTEXT

#: the "publishedness" columns the v3b recipe masked -- kept as a named set so
#: an ablation can drop them and measure what they carry on REAL truths
PRIOR_COLS = ["trk_sig_pos_arcsec", "trk_sig_vel_degday", "stn", "band",
              "mag_mean", "mag_rms", "n_mag", "win_vmag", "mag_resid",
              "trk_fit_rms", "trk_chi2_dof", "trial_U"]


def derive(df):
    """The V4_CONTEXT columns from the raw DECISION_COLS (in place).

    Pure per-row arithmetic, so serving computes exactly the same thing from
    the same raw values (:func:`serve_frame`)."""
    iv = df["in_variant_corridor"].fillna(0.0).astype(float)
    df["in_variant_corridor"] = iv
    vk = df["variant_k"].astype(float)
    df["abs_variant_k"] = np.where(iv > 0, vk.abs(), np.nan)
    df["lov_resid"] = df["lov_resid"].astype(float)
    df["lov_abs_dk"] = df["lov_dk"].astype(float).abs()
    df["lov_nights"] = df["lov_nights"].astype(float)
    df["lov_abs_kstar"] = df["lov_kstar"].astype(float).abs()
    w = df["width_dec_arcsec"].astype(float)
    df["lov_resid_over_width"] = df["lov_resid"] / w.where(w > 0)
    return df


def load(paths, truth_sources=("exitf",), log=print, usecols=None):
    """Rows from one or more collection CSVs, labelled for v4 training.

    Positives: label 1 whose source is in ``truth_sources``.  Other label-1
    rows (``planted``) are DROPPED; ``linked_foreign`` rows are dropped too
    (a real ITF tracklet the search linked: possibly a genuine detection of
    the object, so neither label is safe)."""
    import pandas as pd
    fr = [pd.read_csv(p, low_memory=False, usecols=usecols) for p in paths]
    df = pd.concat(fr, ignore_index=True)
    n0 = len(df)
    # THE EVAL WALL.  No row of an object any evaluation has used -- the
    # eval-side collections, every live A/B list, the recovery fixtures --
    # may ever reach training.  The hash wall (training.split_of) already
    # keeps the train collector off the eval side; this is the hard stop in
    # case a list, a resume or a copied file ever says otherwise.
    wall = never_train()
    hit = set(df["designation"].astype(str)) & wall
    if hit or (df.get("split") is not None
               and (df["split"].astype(str) == "eval").any()):
        raise RuntimeError(f"eval-wall violation: {len(hit)} forbidden "
                           f"object(s) in training rows, e.g. "
                           f"{sorted(hit)[:5]}")
    keep = ~(((df["label"] == 1) & ~df["label_source"].isin(truth_sources))
             | (df["label_source"] == "linked_foreign"))
    # a plant the motion pre-filter would have pruned is injected anyway
    # (harness recall telemetry); a real tracklet like it is never SERVED, so
    # it is not a decision the ranker makes
    if "motion_bound_ok" in df.columns:
        keep &= ~(df["motion_bound_ok"] == 0)
    df = df[keep].copy()
    df["label"] = (df["label"] == 1).astype(int)
    df["case"] = df["designation"].astype(str) + "|" + df["stratum"].astype(str)
    derive(df)
    for c in CAT_COLS:
        df[c] = fill_categorical(df[c])
    log(f"v4 rows: {n0} -> {len(df)} ({int(df['label'].sum())} truths, "
        f"{df['designation'].nunique()} objects, {df['case'].nunique()} cases)")
    return df


#: The eval-wall list (one designation per line), named by ``PYROORB_NEVER_TRAIN``.
NEVER_TRAIN_ENV = "PYROORB_NEVER_TRAIN"


def never_train(path=None):
    """Designations that must never be trained on (see :func:`load`).
    Missing file = refuse: an absent wall is not an open one. So is an unnamed one."""
    path = path or os.environ.get(NEVER_TRAIN_ENV)
    if not path:
        raise RuntimeError(f"no eval wall: set {NEVER_TRAIN_ENV} to the list of designations "
                           "that must never be trained on")
    with open(path, encoding="utf-8") as fh:
        return {l.strip() for l in fh if l.strip()}


def case_weights(df):
    """Each CASE sums to the same weight: a crowded variant scan emits
    hundreds of thousands of rows and would otherwise own the loss.  A
    frontier-filtered store (exitf.frontier) carries ``sample_w`` on the
    randomly sampled easy rows; it multiplies in, and the per-case total is
    taken over the WEIGHTED rows so a filtered case and a full one count the
    same."""
    sw = (df["sample_w"].fillna(1.0).astype(float)
          if "sample_w" in df.columns else pd_ones(df.index))
    tot = sw.groupby(df["case"], observed=True).transform("sum")
    return np.asarray((len(df) / df["case"].nunique()) * sw / tot,
                      dtype="float32")


def pd_ones(index):
    import pandas as pd
    return pd.Series(1.0, index=index)


DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2.0,
    "lambda": 1.0,
}


def train(df, features=None, params=None, rounds=500, seed=0, log=print):
    import xgboost as xgb
    feats = list(features or V4_FEATURES)
    p = dict(DEFAULT_PARAMS, **(params or {}), seed=seed)
    d = xgb.DMatrix(df[feats], label=df["label"], weight=case_weights(df),
                    enable_categorical=True)
    bst = xgb.train(p, d, num_boost_round=rounds, verbose_eval=False)
    meta = {"features": feats,
            "categories": {c: [str(v) for v in df[c].cat.categories]
                           for c in CAT_COLS if c in feats},
            "v4": True, "feature_scale": "honest",
            "params": p, "rounds": rounds,
            "rows": int(len(df)), "truths": int(df["label"].sum()),
            "objects": int(df["designation"].nunique())}
    return bst, meta


def predict(bst, meta, df):
    import pandas as pd
    import xgboost as xgb
    x = df[meta["features"]].copy()
    for c, levels in meta["categories"].items():
        x[c] = pd.Categorical(x[c].astype(str), categories=levels)
    return bst.predict(xgb.DMatrix(x, enable_categorical=True))


def save(bst, meta, path):
    bst.save_model(path)
    with open(path + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)


def group_folds(df, k=5, seed=0):
    """Object-level folds (never split one object's rows)."""
    objs = np.array(sorted(df["designation"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(objs)
    fold_of = {o: i % k for i, o in enumerate(objs)}
    return df["designation"].astype(str).map(fold_of).to_numpy()


def oof(df, features=None, params=None, rounds=500, k=5, seed=0,
        log=print):
    """Out-of-fold predictions over ``df`` (object-level folds)."""
    f = group_folds(df, k, seed)
    out = np.full(len(df), np.nan)
    for i in range(k):
        tr = df[f != i]
        bst, meta = train(tr, features, params, rounds, seed, log)
        out[f == i] = predict(bst, meta, df[f == i])
        log(f"  fold {i + 1}/{k} done")
    return out
