"""The numpy tree evaluator (src/pyroorb/mislink/trees.py) against XGBoost's own scores and
TreeSHAP, stored when the model was built (mislink_data/parity.json) -- so the check runs
without XGBoost installed -- and the model/meta consistency check."""
import builtins
import json
import math
import os
import shutil

import numpy as np
import pytest

from pyroorb.mislink import score as S
from pyroorb.mislink.trees import TreeEnsemble

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mislink_data")


@pytest.fixture(scope="module")
def parity():
    return json.load(open(os.path.join(_DATA, "parity.json"), encoding="utf-8"))


def _rows(parity):
    return np.array([[np.nan if v is None else v for v in r] for r in parity["rows"]], float)


def test_parity_fixture_is_of_the_shipped_model(parity):
    import hashlib
    with open(S.MODEL_PATH, "rb") as fh:
        assert parity["model_md5"] == hashlib.md5(fh.read()).hexdigest()
    assert parity["features"] == list(S.FEATURES)


def test_scores_match_xgboost(parity):
    X = _rows(parity)
    assert np.isnan(X).any()                    # the missing-value branch is exercised
    got = S._model().predict(X)
    assert np.abs(got - np.array(parity["score"])).max() < 1e-5


def test_treeshap_matches_xgboost_and_adds_up(parity):
    t = S._model()
    X = _rows(parity)
    for x, ref in zip(X, parity["contributions"]):
        c = t.contributions(x)
        assert np.abs(c - np.array(ref)).max() < 1e-4
        assert abs(c.sum() - t.margin(x)[0]) < 1e-4


def test_scoring_needs_no_xgboost(monkeypatch):
    real_import = builtins.__import__

    def no_xgboost(name, *a, **k):
        if name == "xgboost" or name.startswith("xgboost."):
            raise ImportError("xgboost is not installed")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_xgboost)
    monkeypatch.setattr(S, "_ensemble", None)
    ex = S.explain({"gap_n_nights": 3, "fit_e": 0.45, "fit_a": 2.7, "fit_q": 1.5})
    assert 0.0 <= ex["score"] <= 1.0
    total = ex["base"] + sum(c["shap"] for c in ex["contributions"])
    assert abs(1 / (1 + math.exp(-total)) - ex["score"]) < 2e-3


def test_a_model_its_meta_does_not_describe_is_refused(tmp_path):
    model = tmp_path / "model.json"
    meta = tmp_path / "model.meta.json"
    shutil.copyfile(S.MODEL_PATH, model)
    shutil.copyfile(S.META_PATH, meta)
    assert S.model_problem(model, meta) is None
    m = json.loads(meta.read_text())
    m["features"] = list(reversed(m["features"]))           # a reordered column
    meta.write_text(json.dumps(m))
    assert "other features" in S.model_problem(model, meta)
    m["features"] = list(S.FEATURES)
    meta.write_text(json.dumps(m))
    model.write_bytes(model.read_bytes() + b" ")             # a model retrained, meta stale
    assert "another model file" in S.model_problem(model, meta)
    meta.unlink()
    assert "unreadable" in S.model_problem(model, meta)


def test_an_unsupported_model_is_refused_not_misread(tmp_path):
    m = json.loads(S.MODEL_PATH.read_text(encoding="utf-8"))
    m["learner"]["objective"]["name"] = "reg:squarederror"
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="binary:logistic"):
        TreeEnsemble(p)


def test_a_score_without_its_explanation_is_the_same_score(parity):
    """``explain=False`` (the link scorer, the watch-list rescores) reads the score alone;
    it must be the score the explained path reports."""
    keys = parity["features"]
    for row in parity["rows"][:20]:
        f = {k: v for k, v in zip(keys, row) if v is not None}
        alone = round(float(S._model().predict(S.vector(f))[0]), 4)
        assert alone == S.explain(f)["score"]


def test_a_cached_score_without_explanation_is_not_served_to_one_that_wants_it(monkeypatch):
    calls = []

    def fake_records(rows, desig, obscodes, *, explain=True):
        calls.append(explain)
        out = {"status": "ok", "score": 0.5}
        if explain:
            out["explanation"] = {"base": 0.0, "contributions": []}
        return out
    monkeypatch.setattr(S, "score_records", fake_records)
    monkeypatch.setattr(S, "fetch_rows", lambda conn, d: [])
    monkeypatch.setattr(S, "_cache", {})
    S.score_designation("2020 AB1", obscodes={}, conn=object(), explain=False)
    S.score_designation("2020 AB1", obscodes={}, conn=object(), explain=False)
    assert calls == [False]                                   # reused
    r = S.score_designation("2020 AB1", obscodes={}, conn=object())
    assert calls == [False, True] and "explanation" in r      # recomputed with it
    S.score_designation("2020 AB1", obscodes={}, conn=object(), explain=False)
    assert calls == [False, True]                             # a fuller result serves both
