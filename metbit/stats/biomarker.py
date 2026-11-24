"""Biomarker panel: ROC AUC per feature and a cross-validated logistic panel of a few features.

Matches the Biomarker step of the metbit desktop app. Without named features, each CV fold
picks its own features from its training samples only, so held-out samples never inflate the AUC.
"""
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def feature_auc(X, y) -> np.ndarray:
    """AUC of every feature for telling class 1 apart (Mann-Whitney ranks, ties averaged; constant: 0.5)."""
    values, pos = np.asarray(X, dtype=float), np.asarray(y) == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    ranks = rankdata(values, axis=0)
    return (ranks[pos].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def best_features(X, y, ppm, top: int = 5, min_gap: float = 0.04) -> list:
    """Indices of the `top` most separating features, at least `min_gap` ppm apart."""
    auc = feature_auc(X, y)
    picked: list = []
    for j in np.argsort(-np.maximum(auc, 1 - auc), kind="stable"):
        if all(abs(ppm[j] - ppm[k]) >= min_gap for k in picked):
            picked.append(int(j))
        if len(picked) == top:
            break
    return picked


def biomarker_panel(X: pd.DataFrame, labels, group_a, group_b, features: Optional[Sequence[float]] = None,
                    top: int = 5, kfold: int = 5, bootstrap: int = 500, seed: int = 0) -> dict:
    """Single-feature AUCs and a cross-validated logistic-regression panel for group B vs A.

    Returns a dict with 'table' (feature, AUC, direction; best first), 'panel' (ppm of the panel
    features), 'auc' and 'ci' (bootstrap 95% interval of the CV AUC), 'y' and 'probability'
    (cross-validated probability of group B).
    """
    labels = np.asarray(labels).astype(str)
    keep = np.isin(labels, [str(group_a), str(group_b)])
    values = np.asarray(X, dtype=float)[keep]
    y = (labels[keep] == str(group_b)).astype(int)
    ppm = np.asarray(X.columns, dtype=float)
    smallest = int(np.bincount(y, minlength=2).min())
    if kfold > smallest:
        raise ValueError(f"CV folds ({kfold}) exceed the smaller group ({smallest} samples)")
    auc = feature_auc(values, y)
    table = pd.DataFrame({
        "feature": ppm, "AUC": np.maximum(auc, 1 - auc),
        "direction": np.where(auc >= 0.5, f"higher in {group_b}", f"lower in {group_b}"),
    }).sort_values("AUC", ascending=False, kind="stable").reset_index(drop=True)
    named = sorted({int(np.abs(ppm - w).argmin()) for w in (features or [])})
    shown = named or best_features(values, y, ppm, top)
    prob = np.zeros(len(y))
    for train, test in StratifiedKFold(kfold, shuffle=True, random_state=seed).split(values, y):
        cols = named or best_features(values[train], y[train], ppm, top)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(values[train][:, cols], y[train])
        prob[test] = model.predict_proba(values[test][:, cols])[:, 1]
    rng, boots = np.random.default_rng(seed), []
    for _ in range(bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            boots.append(roc_auc_score(y[idx], prob[idx]))
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (np.nan, np.nan)
    return {"table": table, "panel": [float(ppm[c]) for c in shown], "auc": float(roc_auc_score(y, prob)),
            "ci": ci, "y": y, "probability": prob}


__all__ = ["feature_auc", "best_features", "biomarker_panel"]
