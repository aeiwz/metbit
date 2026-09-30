"""VolcanoPlot computes all features at once; the answers must be the per-feature loop's."""
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from metbit.stats.multitest import VolcanoPlot


def _loop(df, cols, a, b):
    """The per-feature computation VolcanoPlot used before, kept as the reference."""
    rows = []
    for feat in cols:
        av = df.loc[df.Group == a, feat].dropna().values
        bv = df.loc[df.Group == b, feat].dropna().values
        ma = np.mean(av) if len(av) else np.nan
        mb = np.mean(bv) if len(bv) else np.nan
        if ma > 0 and mb > 0:
            fc = np.log2(mb / ma)
        elif ma == 0 and mb == 0:
            fc = 0.0
        else:
            fc = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = stats.ttest_ind(av, bv, equal_var=False)[1] if len(av) >= 2 and len(bv) >= 2 else np.nan
        rows.append((fc, p))
    return np.array(rows)


def _frame(with_gaps: bool) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(3)
    X = np.abs(rng.normal(size=(24, 60))) + 0.5
    X[12:, 5] += 3
    X[:, 7] = 0.0  # both means zero
    X[:, 8] = 2.0  # constant
    if with_gaps:
        X[rng.random(X.shape) < 0.1] = np.nan
        X[:, 9] = np.nan          # all missing
        X[1:12, 10] = np.nan      # one value left in group A
    cols = [f"f{i}" for i in range(60)]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "Group", ["A"] * 12 + ["B"] * 12)
    return df, cols


def test_vectorised_matches_the_per_feature_loop():
    for gaps in (False, True):
        df, cols = _frame(gaps)
        got = VolcanoPlot(df, group_col="Group", value_cols=cols, group_a="A", group_b="B").get_table()
        want = _loop(df, cols, "A", "B")
        assert np.allclose(got["log2FC"], want[:, 0], equal_nan=True), gaps
        assert np.allclose(got["p_value"], want[:, 1], equal_nan=True, rtol=1e-9), gaps


def test_many_features_are_quick():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(np.abs(rng.normal(size=(200, 20_000))) + 1, columns=[f"f{i}" for i in range(20_000)])
    df.insert(0, "Group", ["A", "B"] * 100)
    start = time.perf_counter()
    VolcanoPlot(df, group_col="Group", value_cols=list(df.columns[1:]), group_a="A", group_b="B")
    assert time.perf_counter() - start < 3  # was about 8 s one feature at a time
