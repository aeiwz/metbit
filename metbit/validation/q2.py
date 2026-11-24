"""Leakage-aware validation of OPLS-DA: grouped folds, repeated Q2, nested CV, and a permutation
test on the model's own cross-validated Q2.

Matches the validation settings of the metbit desktop app. With CV groups (such as a subject
column) a group's samples always share a fold, so no subject is both trained and tested on;
repeated measures otherwise make Q2 look better than it is.

Unlike metbit.validation's permutation_test (scikit-learn's default score of the bare PLS step),
q2_permutation_test reruns metbit's own OPLS cross-validation for each shuffle, with the same
scaling and folds, so every permuted Q2 is directly comparable with the model's Q2.
"""
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold

from metbit.models.cross_validation import CrossValidation

SCALING_POWER = {"pareto": 0.5, "uv": 1.0, "mean": 0.0}


def make_folds(y, groups=None, cv: str = "kfold", k: int = 7, repeats: int = 1,
               classes: bool = True, seed: int = 0) -> list:
    """(train, test) index pairs of each repeat; every sample is in one test set per repeat.

    cv: 'kfold' (stratified for classes, reshuffled per repeat) or 'loo' (each sample or group out).
    groups: CV group per sample (such as subject); a group never spans training and test.
    """
    y = np.asarray(y)
    n = len(y)
    if cv == "loo":
        if groups is not None:
            return [list(LeaveOneGroupOut().split(np.zeros(n), y, groups))]
        idx = np.arange(n)
        return [[(np.delete(idx, i), np.array([i])) for i in idx]]
    if cv != "kfold":
        raise ValueError("cv must be 'kfold' or 'loo'")
    if groups is not None and len(set(groups)) < k:
        raise ValueError(f"CV folds ({k}) exceed the number of CV groups ({len(set(groups))})")
    if classes and groups is None:
        smallest = int(pd.Series(y).value_counts().min())
        if k > smallest:
            raise ValueError(f"CV folds ({k}) exceed the smallest group ({smallest} samples)")
    out = []
    for r in range(repeats):
        rs = seed + r
        if groups is not None:
            split = StratifiedGroupKFold(k, shuffle=True, random_state=rs) if classes else GroupKFold(k, shuffle=True, random_state=rs)
            out.append(list(split.split(np.zeros(n), y, groups)))
        else:
            split = StratifiedKFold(k, shuffle=True, random_state=rs) if classes else KFold(k, shuffle=True, random_state=rs)
            out.append(list(split.split(np.zeros(n), y)))
    return out


def label_shuffler(y, groups=None, seed: int = 0) -> Callable[[], np.ndarray]:
    """Label shuffles that respect CV groups. A group whose samples share one label (a subject in
    one study arm) moves as a whole; a label that varies within groups is shuffled within each."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    if groups is None:
        return lambda: y[rng.permutation(len(y))]
    groups = np.asarray(groups)
    frame = pd.DataFrame({"g": groups, "y": y})
    if (frame.groupby("g")["y"].nunique() == 1).all():
        per_group = frame.groupby("g")["y"].first()

        def whole() -> np.ndarray:
            moved = dict(zip(per_group.index, per_group.to_numpy()[rng.permutation(len(per_group))]))
            return np.array([moved[g] for g in groups])
        return whole
    members = [np.flatnonzero(groups == g) for g in pd.unique(groups)]

    def within() -> np.ndarray:
        out = y.copy()
        for m in members:
            out[m] = y[m[rng.permutation(len(m))]]
        return out
    return within


def opls_q2(X, y, splits, kfold: int = 7, scaling: str = "pareto") -> tuple:
    """(Q2, R2Y) of metbit's OPLS cross-validation on the given folds."""
    cv = CrossValidation(kfold=kfold, estimator="opls", scaler=scaling, splits=splits)
    cv.fit(np.asarray(X, dtype=float), np.asarray(y))
    return float(cv.q2), float(cv.R2y)


def repeated_q2(X, y, groups=None, k: int = 7, repeats: int = 10, scaling: str = "pareto", seed: int = 0) -> np.ndarray:
    """Q2 of each reshuffled k-fold repeat: its spread shows how much one split can mislead."""
    return np.array([opls_q2(X, y, f, k, scaling)[0] for f in make_folds(y, groups, "kfold", k, repeats, seed=seed)])


def q2_permutation_test(X, y, groups=None, k: int = 7, n_permutations: int = 200,
                        scaling: str = "pareto", seed: int = 0, n_jobs: int = -1) -> dict:
    """Permutation test on OPLS-DA cross-validated Q2.

    Each permutation shuffles labels (respecting `groups`), remakes the folds for the shuffled
    labels and refits the full cross-validation. p = (b + 1) / (n + 1), where b permutations reach
    the observed Q2. Returns q2, r2y, q2_perm, r2y_perm, p_q2.
    """
    from joblib import Parallel, delayed

    X, y = np.asarray(X, dtype=float), np.asarray(y)

    def score(labels):
        return opls_q2(X, labels, make_folds(labels, groups, "kfold", k, 1, seed=seed)[0], k, scaling)

    q2, r2y = score(y)
    shuffle = label_shuffler(y, groups, seed)
    shuffled = [shuffle() for _ in range(n_permutations)]  # drawn up front: independent of thread timing
    # threads: numpy releases the GIL in the linear algebra, and frozen apps cannot fork safely
    perm = np.array(Parallel(n_jobs=n_jobs, prefer="threads")(delayed(score)(s) for s in shuffled)) if shuffled else np.empty((0, 2))
    q2_perm, r2y_perm = perm[:, 0], perm[:, 1]
    return {"q2": q2, "r2y": r2y, "q2_perm": q2_perm, "r2y_perm": r2y_perm,
            "p_q2": (int((q2_perm >= q2).sum()) + 1) / (len(q2_perm) + 1)}


def nested_opls_cv(X, y, groups=None, k: int = 7, inner_k: int = 7, scaling: str = "pareto", seed: int = 0) -> dict:
    """Nested CV: each outer training set runs its own CV to choose the component count, and the
    outer test set is predicted by that model. Q2 and accuracy that component choice has not seen."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    yb = (y == np.unique(y)[1]).astype(float)  # metbit codes the sorted second class as 1
    power = SCALING_POWER[scaling]
    press = ss = 0.0
    right, components = 0, []
    for tr, te in make_folds(y, groups, "kfold", k, 1, seed=seed)[0]:
        kk = min(inner_k, int(pd.Series(y[tr]).value_counts().min()))
        inner_groups = None if groups is None else np.asarray(groups)[tr]
        if inner_groups is not None:
            kk = min(kk, len(set(inner_groups)))
        cv = CrossValidation(kfold=kk, estimator="opls", scaler=scaling,
                             splits=make_folds(y[tr], inner_groups, "kfold", kk, 1, seed=seed)[0])
        cv.fit(X[tr], y[tr])
        scaled = np.ravel(cv.predict(X[te]))
        centre = yb[tr].mean()
        spread = yb[tr].std() ** power if power else 1.0
        press += float((((scaled * spread + centre) - yb[te]) ** 2).sum())
        ss += float(((yb[te] - centre) ** 2).sum())
        right += int(((scaled > 0) == (yb[te] == 1)).sum())
        components.append(int(cv.optimal_component_num))
    return {"q2": 1 - press / ss, "accuracy": right / len(y), "components": components}


__all__ = ["make_folds", "label_shuffler", "opls_q2", "repeated_q2", "q2_permutation_test", "nested_opls_cv"]
