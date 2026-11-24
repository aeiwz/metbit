"""Time series: ASCA and per-feature repeated-measures tests (linear mixed model, Friedman).

Matches the Time series steps of the metbit desktop app. `time`, `group` and `subject` are one
value per sample (row of X); time points are ordered numerically when they are numbers, else
naturally ('day 2' before 'day 10'), unless `time_order` is given.
"""
import re
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, friedmanchisquare

from metbit.preprocessing.qc import scale_columns
from metbit.stats.features import adjust_pvalues

INTERACTION = "Time x Group"


def _natural_key(text: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def time_positions(time, time_order: Optional[Sequence] = None):
    """(position of each sample's time point, ordered time labels)."""
    time = pd.Series(np.asarray(time).astype(str))
    values = sorted(set(time))
    if time_order is not None:
        order = [str(t) for t in time_order]
        missing = set(values) - set(order)
        if missing:
            raise ValueError(f"time_order is missing: {', '.join(sorted(missing, key=_natural_key))}")
        order = [t for t in order if t in values]
    else:
        num = pd.to_numeric(pd.Series(values), errors="coerce")
        order = [values[i] for i in num.argsort()] if num.notna().all() else sorted(values, key=_natural_key)
    if len(order) < 2:
        raise ValueError("A time series needs at least two time points")
    return time.map({t: i for i, t in enumerate(order)}).to_numpy(), order


def _level_means(X: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Each row replaced by the mean of its level: the effect matrix of one factor."""
    D = np.eye(codes.max() + 1)[codes]
    counts = D.sum(axis=0)[:, None]
    return D @ ((D.T @ X) / np.where(counts > 0, counts, 1))


def _effect_pca(M: np.ndarray, E: np.ndarray) -> dict:
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    V = np.zeros((2, M.shape[1]))
    V[: min(2, len(Vt))] = Vt[:2]
    var = s**2 / (s**2).sum() if (s**2).sum() > 0 else np.zeros_like(s)
    explained = [float(v) * 100 for v in np.pad(var[:2], (0, max(0, 2 - len(var[:2]))))]
    return {"scores": (M + E) @ V.T, "loadings": V, "explained": explained}


def asca(X, time, group=None, subject=None, scaling: str = "pareto", permutations: int = 200,
         time_order: Optional[Sequence] = None, seed: int = 0) -> dict:
    """ANOVA-simultaneous component analysis on level-mean effect matrices.

    Effects: Time, and with two or more groups Group and Time x Group. Permutation p-values:
    Time and Time x Group permute the data within each subject (when `subject` is given), and
    Group moves whole subjects between groups, so a subject's repeated samples stay together.
    Permuting single samples, or residuals of the main-effects model, rejects far too often
    when subjects differ (checked by simulation in tests/test_app_features.py).

    Returns {'table': effect, ssq_percent, p_value, pc1_percent, pc2_percent;
             'effects': name -> scores (effect + residual), loadings, explained}.
    """
    from metbit.validation.q2 import label_shuffler

    # ponytail: level-mean ASCA, exact for balanced designs; ASCA+ (GLM) for unbalanced ones
    Xs = scale_columns(X, scaling)
    t, _ = time_positions(time, time_order)
    names = sorted(set(np.asarray(group).astype(str))) if group is not None else ["all"]
    g = (pd.Series(np.asarray(group).astype(str)).map({n: i for i, n in enumerate(names)}).to_numpy()
         if group is not None else np.zeros(len(Xs), dtype=int))
    k = len(names)
    specs = {"Time": lambda t_, g_: _level_means(Xs, t_)}
    if k > 1:
        specs["Group"] = lambda t_, g_: _level_means(Xs, g_)
        specs[INTERACTION] = lambda t_, g_: _level_means(Xs, t_ * k + g_) - _level_means(Xs, t_) - _level_means(Xs, g_)
    rng = np.random.default_rng(seed)
    n = len(Xs)
    if subject is None:
        within = lambda: rng.permutation(n)  # noqa: E731
        regroup = lambda: g[rng.permutation(n)]  # noqa: E731
    else:
        subj = np.asarray(subject).astype(str)
        blocks = [np.flatnonzero(subj == s) for s in np.unique(subj)]
        regroup = label_shuffler(g, subj, seed)

        def within():
            rows = np.arange(n)
            for b in blocks:
                rows[b] = rng.permutation(b)
            return rows
    total = (Xs**2).sum()
    matrices, rows = {}, []
    for name, effect in specs.items():
        M = effect(t, g)
        matrices[name] = M
        ssq = (M**2).sum()
        pv = np.nan
        if permutations:
            def null():
                if name == "Group":
                    return effect(t, regroup())
                r = within()
                return effect(t[r], g[r])
            hits = sum((null() ** 2).sum() >= ssq for _ in range(permutations))
            pv = (hits + 1) / (permutations + 1)
        rows.append({"effect": name, "ssq_percent": ssq / total * 100, "p_value": pv})
    E = Xs - sum(matrices.values())
    rows.append({"effect": "Residual", "ssq_percent": (E**2).sum() / total * 100, "p_value": np.nan})
    effects = {name: _effect_pca(M, E) for name, M in matrices.items()}
    table = pd.DataFrame(rows)
    table["pc1_percent"] = [effects[e]["explained"][0] if e in effects else np.nan for e in table.effect]
    table["pc2_percent"] = [effects[e]["explained"][1] if e in effects else np.nan for e in table.effect]
    return {"table": table, "effects": effects}


def mixed_model_test(X, time, subject, group=None, time_as: str = "categorical",
                     time_order: Optional[Sequence] = None, correction: Optional[str] = "fdr_bh") -> pd.DataFrame:
    """y ~ time * group + (1 | subject) per feature (REML), with a Wald test of each whole term.

    time_as='numeric' fits a linear trend (time values when they are numbers, else the time order).
    Returns feature, p_<term> and p_<term>_adj for time, group and interaction, ranked.
    """
    import patsy
    from statsmodels.regression.mixed_linear_model import MixedLM

    if time_as not in ("categorical", "numeric"):
        raise ValueError("time_as must be 'categorical' or 'numeric'")
    pos, order = time_positions(time, time_order)
    tlab = pd.Series(np.asarray(time).astype(str))
    if time_as == "numeric":
        values = pd.to_numeric(tlab, errors="coerce")
        tcol = values if values.notna().all() else pd.Series(pos, dtype=float)
    else:
        tcol = pd.Categorical(tlab, categories=order)
    grp = pd.Categorical(np.asarray(group).astype(str)) if group is not None else pd.Categorical(["all"] * len(tlab))
    frame = pd.DataFrame({"time": tcol, "group": grp})
    dm = patsy.dmatrix("time * group" if frame.group.nunique() > 1 else "time", frame, return_type="dataframe")
    terms = {"time": "time", "group": "group", "time:group": "interaction"}
    slices = {terms[k]: v for k, v in dm.design_info.term_name_slices.items() if k in terms}
    Y = np.asarray(X, dtype=float)
    out = {term: np.full(Y.shape[1], np.nan) for term in slices}
    exog, subjects = dm.to_numpy(), np.asarray(subject).astype(str)
    failure = None
    for j in range(Y.shape[1]):
        y = Y[:, j]
        if np.ptp(y) <= 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # statsmodels' lbfgs fallback passes `disp`, which SciPy >= 1.18 rejects
                fit = MixedLM(y, exog, groups=subjects).fit(reml=True, method=["bfgs", "cg", "nm"])
            k = exog.shape[1]
            b, V = np.asarray(fit.fe_params), np.asarray(fit.cov_params())[:k, :k]
            for term, sl in slices.items():
                stat = float(b[sl] @ np.linalg.pinv(V[sl, sl]) @ b[sl])
                out[term][j] = chi2.sf(stat, sl.stop - sl.start)
        except Exception as e:  # one feature that cannot be fitted stays NaN
            failure = failure or e
    if failure is not None and all(np.isnan(v).all() for v in out.values()):
        raise ValueError(f"The mixed model could not be fitted: {failure}")
    return _table(X, out, correction)


def friedman_test(X, time, subject, time_order: Optional[Sequence] = None,
                  correction: Optional[str] = "fdr_bh") -> pd.DataFrame:
    """Friedman rank test across time points within subjects, per feature (complete subjects only)."""
    _, order = time_positions(time, time_order)
    if len(order) < 3:
        raise ValueError("The Friedman test needs at least three time points")
    frame = pd.DataFrame(np.asarray(X, dtype=float))
    means = frame.groupby([np.asarray(subject).astype(str), np.asarray(time).astype(str)]).mean()
    blocks = [means.xs(t, level=1) for t in order]
    subjects = sorted(set.intersection(*(set(b.index) for b in blocks)))
    if len(subjects) < 2:
        raise ValueError("The Friedman test needs at least two subjects measured at every time point")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pv = friedmanchisquare(*[b.loc[subjects].to_numpy() for b in blocks], axis=0)
    return _table(X, {"time": np.asarray(pv, dtype=float)}, correction)


def _table(X, pvals: dict, correction) -> pd.DataFrame:
    cols = X.columns if isinstance(X, pd.DataFrame) else range(np.shape(X)[1])
    table = pd.DataFrame({"feature": list(cols)})
    for term, pv in pvals.items():
        table[f"p_{term}"] = pv
        table[f"p_{term}_adj"] = np.where(np.isfinite(pv), adjust_pvalues(pv, correction), np.nan)
    rank = table[[c for c in table.columns if c.endswith("_adj")]].min(axis=1, skipna=True)
    return table.iloc[np.argsort(rank.fillna(np.inf).to_numpy(), kind="stable")].reset_index(drop=True)


__all__ = ["time_positions", "asca", "mixed_model_test", "friedman_test"]
