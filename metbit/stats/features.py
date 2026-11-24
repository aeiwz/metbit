"""Per-feature univariate statistics, all features at once.

Two-group tests with effect sizes and 95% intervals, a covariate-adjusted linear model,
correlation with a numeric variable, and normality tests. Matches the Statistics steps of
the metbit desktop app. X is a sample-by-feature DataFrame (ppm columns); every function
returns one row per feature, sorted by p-value.
"""
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as st

from metbit.stats.multitest import _apply_correction

TWO_GROUP_TESTS = ("welch", "student", "mannwhitney", "paired_t", "wilcoxon")


def adjust_pvalues(p, method: Optional[str] = "fdr_bh") -> np.ndarray:
    """Multiple-testing correction; a NaN p-value (a constant feature) is never significant."""
    p = np.where(np.isfinite(p), p, 1.0)
    return p if method in (None, "none") else _apply_correction(p, method)


def _ranked(table: pd.DataFrame) -> pd.DataFrame:
    return table.sort_values("p_value", kind="stable", na_position="last").reset_index(drop=True)


def _ppm(X) -> np.ndarray:
    cols = X.columns if isinstance(X, pd.DataFrame) else np.arange(np.shape(X)[1])
    try:
        return np.asarray(cols, dtype=float)
    except (TypeError, ValueError):
        return np.asarray(cols)


def two_group_test(X, labels, group_a, group_b, test: str = "welch", pairs=None,
                   correction: Optional[str] = "fdr_bh") -> pd.DataFrame:
    """Compare group B with reference group A at every feature.

    test: 'welch', 'student', 'mannwhitney', or the paired 'paired_t' / 'wilcoxon'.
    pairs: for paired tests, the pair (subject) of each sample; each pair needs one sample per group.
    Effect size is Hedges' g (paired: d_z) with a 95% interval; log2 FC is mean B / mean A.
    """
    if test not in TWO_GROUP_TESTS:
        raise ValueError(f"test must be one of {TWO_GROUP_TESTS}")
    values = np.asarray(X, dtype=float)
    labels = np.asarray(labels).astype(str)
    group_a, group_b = str(group_a), str(group_b)
    paired = test in ("paired_t", "wilcoxon")
    if paired:
        if pairs is None:
            raise ValueError("Paired tests need `pairs`, the pair of each sample")
        frame = pd.DataFrame({"pair": np.asarray(pairs).astype(str), "group": labels, "row": np.arange(len(labels))})
        frame = frame[frame.group.isin([group_a, group_b])]
        wide = frame.drop_duplicates(["pair", "group"], keep=False).pivot(index="pair", columns="group", values="row").dropna()
        if len(wide) < 3:
            raise ValueError(f"Paired tests need at least 3 pairs with one sample in each group; found {len(wide)}")
        A, B = values[wide[group_a].to_numpy(dtype=int)], values[wide[group_b].to_numpy(dtype=int)]
    else:
        A, B = values[labels == group_a], values[labels == group_b]
        if min(len(A), len(B)) < 2:
            raise ValueError("Each group needs at least 2 samples")
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if test in ("welch", "student"):
            pv = st.ttest_ind(B, A, axis=0, equal_var=test == "student").pvalue
        elif test == "mannwhitney":
            pv = st.mannwhitneyu(B, A, axis=0).pvalue
        elif test == "paired_t":
            pv = st.ttest_rel(B, A, axis=0).pvalue
        else:
            pv = np.array([st.wilcoxon(d).pvalue if np.any(d != 0) else np.nan for d in (B - A).T])
        ma, mb = A.mean(axis=0), B.mean(axis=0)
        if paired:
            d = B - A
            g = d.mean(axis=0) / d.std(axis=0, ddof=1)
            se = np.sqrt(1 / len(d) + g**2 / (2 * len(d)))
        else:
            na, nb = len(A), len(B)
            pooled = np.sqrt(((na - 1) * A.var(axis=0, ddof=1) + (nb - 1) * B.var(axis=0, ddof=1)) / (na + nb - 2))
            g = (mb - ma) / pooled * (1 - 3 / (4 * (na + nb) - 9))
            se = np.sqrt((na + nb) / (na * nb) + g**2 / (2 * (na + nb)))
        fc = np.where((ma > 0) & (mb > 0), np.log2(mb / ma), np.nan)
    return _ranked(pd.DataFrame({
        "feature": _ppm(X), f"mean {group_a}": ma, f"mean {group_b}": mb, "log2 FC": fc,
        "effect size": g, "effect 95% CI low": g - 1.96 * se, "effect 95% CI high": g + 1.96 * se,
        "p_value": pv, "p_adj": adjust_pvalues(pv, correction),
    }))


def _covariates(meta: pd.DataFrame, names: Sequence[str]):
    """Numeric covariates standardized, others as dummy columns; and rows that have them all."""
    parts, ok = [], np.ones(len(meta), dtype=bool)
    for name in names:
        col = meta[name].reset_index(drop=True)
        numeric = pd.to_numeric(col, errors="coerce")
        if numeric.notna().sum() == col.notna().sum():
            ok &= numeric.notna().to_numpy()
            parts.append(pd.DataFrame({name: (numeric - numeric.mean()) / (numeric.std() or 1)}))
        else:
            ok &= col.notna().to_numpy()
            parts.append(pd.get_dummies(col.astype(str), prefix=name, drop_first=True, dtype=float))
    return (pd.concat(parts, axis=1) if parts else pd.DataFrame(index=range(len(meta)))), ok


def linear_model_test(X, labels, covariates: Optional[pd.DataFrame] = None,
                      correction: Optional[str] = "fdr_bh") -> pd.DataFrame:
    """y ~ covariates + group at every feature (ordinary least squares).

    Two groups: t test of the adjusted group difference (last level minus first, sorted).
    More groups: F test of the group term against the covariates-only model.
    covariates: DataFrame aligned with X; numeric columns as they are, others as categories.
    Samples missing a covariate are left out.
    """
    labels = pd.Series(np.asarray(labels).astype(str))
    meta = covariates.reset_index(drop=True) if covariates is not None else pd.DataFrame(index=range(len(labels)))
    covs, ok = _covariates(meta, list(meta.columns))
    levels = sorted(set(labels[ok]))
    if len(levels) < 2:
        raise ValueError("The linear model needs at least two groups with complete covariates")
    dummies = pd.get_dummies(pd.Categorical(labels[ok], categories=levels), drop_first=True, dtype=float).to_numpy()
    base = np.column_stack([np.ones(ok.sum()), covs[ok].to_numpy(dtype=float)])
    full = np.column_stack([base, dummies])
    Y = np.asarray(X, dtype=float)[ok]
    n = len(Y)
    df = n - np.linalg.matrix_rank(full)
    if df < 2:
        raise ValueError(f"Too few samples ({n}) for {full.shape[1] - 1} group and covariate terms")
    beta, *_ = np.linalg.lstsq(full, Y, rcond=None)
    rss = ((Y - full @ beta) ** 2).sum(axis=0)
    with np.errstate(all="ignore"):
        if dummies.shape[1] == 1:
            cov = np.linalg.pinv(full.T @ full)[-1, -1]
            effect = beta[-1]
            stat = effect / np.sqrt(rss / df * cov)
            pv = 2 * st.t.sf(np.abs(stat), df)
            table = pd.DataFrame({"feature": _ppm(X), f"{levels[-1]} - {levels[0]} (adjusted)": effect, "t": stat})
        else:
            b0, *_ = np.linalg.lstsq(base, Y, rcond=None)
            rss0 = ((Y - base @ b0) ** 2).sum(axis=0)
            q = dummies.shape[1]
            stat = ((rss0 - rss) / q) / (rss / df)
            pv = st.f.sf(stat, q, df)
            table = pd.DataFrame({"feature": _ppm(X), "F": stat})
    table["p_value"], table["p_adj"] = pv, adjust_pvalues(pv, correction)
    return _ranked(table)


def correlation_test(X, target, method: str = "spearman", correction: Optional[str] = "fdr_bh") -> pd.DataFrame:
    """Pearson or Spearman r of every feature with a numeric variable; samples without it are left out."""
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    x = pd.to_numeric(pd.Series(np.asarray(target)), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 4:
        raise ValueError("Correlation needs the variable for at least 4 samples")
    Y, xv = np.asarray(X, dtype=float)[ok], x[ok]
    if method == "spearman":
        Y, xv = st.rankdata(Y, axis=0), st.rankdata(xv)
    with np.errstate(all="ignore"):
        yc, xc = Y - Y.mean(axis=0), xv - xv.mean()
        r = (xc @ yc) / np.sqrt((xc**2).sum() * (yc**2).sum(axis=0))
        n = len(xv)
        t = r * np.sqrt((n - 2) / (1 - r**2))
        pv = 2 * st.t.sf(np.abs(t), n - 2)
    return _ranked(pd.DataFrame({"feature": _ppm(X), "r": r, "p_value": pv, "p_adj": adjust_pvalues(pv, correction)}))


def normality_test(X, labels=None, test: str = "shapiro", correction: Optional[str] = "fdr_bh",
                   alpha: float = 0.05) -> pd.DataFrame:
    """Shapiro-Wilk or D'Agostino-Pearson ('dagostino') per feature, within each group when labels are given."""
    if test not in ("shapiro", "dagostino"):
        raise ValueError("test must be 'shapiro' or 'dagostino'")
    values = np.asarray(X, dtype=float)
    names = ["All samples"] if labels is None else list(pd.unique(np.asarray(labels).astype(str)))
    masks = [np.ones(len(values), bool)] if labels is None else [np.asarray(labels).astype(str) == g for g in names]
    minimum = 3 if test == "shapiro" else 8
    fn = st.shapiro if test == "shapiro" else st.normaltest
    rows = []
    for j, feature in enumerate(_ppm(X)):
        for group, mask in zip(names, masks):
            v = values[mask, j]
            v = v[np.isfinite(v)]
            stat = pv = np.nan
            note = ""
            if len(v) < minimum:
                note = f"Insufficient samples (need {minimum})"
            elif np.ptp(v) == 0:
                note = "Constant feature"
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, pv = fn(v)
            rows.append({"feature": feature, "group": group, "n": len(v), "statistic": stat, "p_value": pv, "note": note})
    table = pd.DataFrame(rows)
    p = table.p_value.to_numpy(dtype=float)
    table["p_adj"] = p if correction in (None, "none") else _apply_correction(p, correction)
    table["normal"] = np.where(np.isfinite(table.p_adj), table.p_adj >= alpha, np.nan)
    return _ranked(table)


__all__ = ["adjust_pvalues", "two_group_test", "linear_model_test", "correlation_test", "normality_test"]
