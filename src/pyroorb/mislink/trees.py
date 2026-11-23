"""A boosted-tree model evaluated in numpy, read from XGBoost's own JSON model file.

The mislinkage model is 400 depth-4 trees: a score is a walk down each tree and a sum of
the leaves it reaches, which needs no model library.  Reading the file XGBoost saves
(``Booster.save_model("model.json")``) directly means a retrained model is used as it is
written, and the scorer runs on every install, XGBoost or not.

Scores match XGBoost's own to float32 rounding (a parity test compares the two on
stored feature rows), which takes the same three rules XGBoost applies:

* features and thresholds compared in float32 (XGBoost stores both that way; compared in
  float64, a value exactly on a threshold can take the other branch);
* a split sends ``x < threshold`` left, and a missing value (NaN) down the branch the
  node learned for it (``default_left``);
* the margin is ``logit(base_score)`` plus the sum of leaves, the score its logistic.

:meth:`TreeEnsemble.contributions` gives the exact TreeSHAP attribution of one row
(Lundberg, Erion & Lee 2018, "Consistent individualized feature attribution for tree
ensembles", Algorithm 2), with each node's cover (``sum_hessian``) as its weight, as
XGBoost's ``pred_contribs`` does: one value per feature plus the expected margin, which
add up to the row's margin.

Only what this model uses is supported: a ``gbtree`` booster, a ``binary:logistic``
objective, numerical splits, one value per leaf.  Anything else is refused on load
rather than evaluated wrongly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _param_float(v):
    """XGBoost writes some parameters as text, a few as a one-element list ('[4.8E-1]')."""
    return float(str(v).strip().strip("[]"))


class TreeEnsemble:
    """The trees of one XGBoost JSON model, flattened into numpy arrays."""

    def __init__(self, path):
        model = json.loads(Path(path).read_text(encoding="utf-8"))
        learner = model["learner"]
        objective = learner["objective"]["name"]
        booster = learner["gradient_booster"]
        if booster.get("name") != "gbtree" or objective != "binary:logistic":
            raise ValueError(f"{Path(path).name}: only a gbtree binary:logistic model is "
                             f"supported, not {booster.get('name')} {objective}")
        param = learner["learner_model_param"]
        if int(param.get("num_class", 0)) > 1 or int(param.get("num_target", 1)) != 1:
            raise ValueError(f"{Path(path).name}: a single-output model is required")
        self.n_features = int(param["num_feature"])
        base = _param_float(param["base_score"])
        self.base_margin = math.log(base / (1.0 - base))
        trees = booster["model"]["trees"]
        for t in trees:
            if any(t.get("split_type") or ()) or t.get("categories"):
                raise ValueError(f"{Path(path).name}: categorical splits are not supported")
            if int(t["tree_param"].get("size_leaf_vector", 1)) > 1:
                raise ValueError(f"{Path(path).name}: vector leaves are not supported")
        sizes = np.array([len(t["left_children"]) for t in trees], dtype=np.int64)
        self.n_trees = len(trees)
        self.roots = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        shift = np.repeat(self.roots, sizes)

        def cat(key, dtype):
            return np.concatenate([np.asarray(t[key], dtype=dtype) for t in trees])

        left, right = cat("left_children", np.int64), cat("right_children", np.int64)
        self.is_leaf = left == -1
        self.left = np.where(self.is_leaf, -1, left + shift)
        self.right = np.where(self.is_leaf, -1, right + shift)
        self.feature = cat("split_indices", np.int64)
        # a leaf's value is stored in its split_conditions slot
        self.threshold = cat("split_conditions", np.float32)
        self.leaf_value = self.threshold.astype(np.float64)
        self.default_left = cat("default_left", np.int64).astype(bool)
        self.cover = cat("sum_hessian", np.float64)
        self.depth = int(max(self._tree_depth(r) for r in self.roots))

    def _tree_depth(self, node):
        if self.is_leaf[node]:
            return 0
        return 1 + max(self._tree_depth(self.left[node]), self._tree_depth(self.right[node]))

    def _rows(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.n_features:
            raise ValueError(f"rows have {X.shape[1]} features, the model {self.n_features}")
        return X

    def _goes_left(self, node, x):
        """Branch of ``node`` for feature value(s) ``x`` (float32, NaN = missing)."""
        return np.where(np.isnan(x), self.default_left[node], x < self.threshold[node])

    def leaves(self, X):
        """(rows, trees) index of the leaf each row reaches in each tree."""
        X = self._rows(X)
        node = np.broadcast_to(self.roots, (X.shape[0], self.n_trees)).copy()
        rows = np.arange(X.shape[0])[:, None]
        for _ in range(self.depth):
            inner = ~self.is_leaf[node]
            if not inner.any():
                break
            x = X[rows, np.where(inner, self.feature[node], 0)]
            nxt = np.where(self._goes_left(node, x), self.left[node], self.right[node])
            node = np.where(inner, nxt, node)
        return node

    def margin(self, X):
        """Log-odds of each row."""
        return self.base_margin + self.leaf_value[self.leaves(X)].sum(axis=1)

    def predict(self, X):
        """Score (probability of the positive class) of each row."""
        return 1.0 / (1.0 + np.exp(-self.margin(X)))

    # ------------------------------------------------------------------ TreeSHAP

    def contributions(self, x):
        """Exact TreeSHAP of ONE row: an array of ``n_features + 1``, the last entry the
        expected margin (base score plus each tree's cover-weighted mean leaf); the
        entries add up to :meth:`margin` of the row."""
        x = self._rows(x)[0]
        phi = np.zeros(self.n_features + 1)
        for root in self.roots:
            phi[-1] += self._expected(root)
            self._recurse(root, x, phi, [], 1.0, 1.0, -1)
        phi[-1] += self.base_margin
        return phi

    def _expected(self, node):
        """Cover-weighted mean leaf value under ``node``."""
        if self.is_leaf[node]:
            return self.leaf_value[node]
        lo, hi = self.left[node], self.right[node]
        return ((self.cover[lo] * self._expected(lo) + self.cover[hi] * self._expected(hi))
                / self.cover[node])

    def _recurse(self, node, x, phi, path, zero_fraction, one_fraction, feature):
        # path: [feature, zero_fraction, one_fraction, weight] per unique feature so far
        path = [list(e) for e in path]
        _extend(path, zero_fraction, one_fraction, feature)
        if self.is_leaf[node]:
            for i in range(1, len(path)):
                w = _unwound_sum(path, i)
                phi[path[i][0]] += w * (path[i][2] - path[i][1]) * self.leaf_value[node]
            return
        split = int(self.feature[node])
        hot, cold = ((self.left[node], self.right[node]) if self._goes_left(node, x[split])
                     else (self.right[node], self.left[node]))
        incoming_zero, incoming_one = 1.0, 1.0
        k = next((i for i, e in enumerate(path) if e[0] == split), None)
        if k is not None:                       # the feature was split on above: undo it
            incoming_zero, incoming_one = path[k][1], path[k][2]
            _unwind(path, k)
        total = self.cover[node]
        self._recurse(hot, x, phi, path, incoming_zero * self.cover[hot] / total,
                      incoming_one, split)
        self._recurse(cold, x, phi, path, incoming_zero * self.cover[cold] / total,
                      0.0, split)


def _extend(path, zero_fraction, one_fraction, feature):
    depth = len(path)
    path.append([feature, zero_fraction, one_fraction, 1.0 if depth == 0 else 0.0])
    for i in range(depth - 1, -1, -1):
        path[i + 1][3] += one_fraction * path[i][3] * (i + 1) / (depth + 1)
        path[i][3] = zero_fraction * path[i][3] * (depth - i) / (depth + 1)


def _unwind(path, k):
    depth = len(path) - 1
    zero_fraction, one_fraction = path[k][1], path[k][2]
    next_one = path[depth][3]
    for i in range(depth - 1, -1, -1):
        if one_fraction != 0.0:
            tmp = path[i][3]
            path[i][3] = next_one * (depth + 1) / ((i + 1) * one_fraction)
            next_one = tmp - path[i][3] * zero_fraction * (depth - i) / (depth + 1)
        else:
            path[i][3] = path[i][3] * (depth + 1) / (zero_fraction * (depth - i))
    weights = [e[3] for e in path[:depth]]
    del path[k]
    for i, w in enumerate(weights):
        path[i][3] = w


def _unwound_sum(path, k):
    depth = len(path) - 1
    zero_fraction, one_fraction = path[k][1], path[k][2]
    next_one = path[depth][3]
    total = 0.0
    for i in range(depth - 1, -1, -1):
        if one_fraction != 0.0:
            tmp = next_one * (depth + 1) / ((i + 1) * one_fraction)
            total += tmp
            next_one = path[i][3] - tmp * zero_fraction * (depth - i) / (depth + 1)
        else:
            total += path[i][3] / zero_fraction / ((depth - i) / (depth + 1))
    return total
