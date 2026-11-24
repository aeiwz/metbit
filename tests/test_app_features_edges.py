"""Edge cases and input checks of the desktop-app features (see test_app_features.py)."""
import numpy as np
import pandas as pd
import pytest

import metbit
from metbit.analysis import longitudinal as lg
from metbit.nmr import integrate as ig
from metbit.validation.q2 import label_shuffler
from test_app_features import longitudinal, two_groups


def test_regions_inputs():
    assert metbit.parse_windows(None) == []
    assert metbit.parse_windows([(1, 2)]) == [(2.0, 1.0)]
    X = pd.DataFrame(np.ones((2, 11)), columns=np.round(np.linspace(1, 0, 11), 2))
    assert metbit.select_regions(X).shape == X.shape  # nothing suppressed
    with pytest.raises(ValueError):
        metbit.select_regions(X, suppress_to="mean")
    with pytest.raises(ValueError):
        metbit.select_regions(X, keep="0.1-0.2, 0.5-0.6")
    with pytest.raises(ValueError):
        metbit.select_regions(X, cut="0-1")
    assert metbit.nmr.regions.nearest_column(X, 0.31) == 7


def test_integrate_inputs():
    ppm = np.linspace(1, 0, 101)
    y = np.exp(-((ppm - 0.5) / 0.01) ** 2)
    peaks, regions = ig.pick_peaks(ppm, y, at=[0.5])
    assert np.isclose(peaks[0], 0.5) and regions
    assert ig._walk(np.array([5.0, 4.0, 3.5]), 0.0, 0.0001) == 2  # never reaches the floor
    # two touching lines of one multiplet share a region
    y2 = np.exp(-((ppm - 0.50) / 0.01) ** 2) + np.exp(-((ppm - 0.53) / 0.01) ** 2)
    assert len(ig.pick_peaks(ppm, y2 + 1e-6 * np.sin(np.arange(101)), snr=5)[1]) == 1
    X = pd.DataFrame(np.ones((2, 101)), columns=ppm)
    with pytest.raises(ValueError):
        metbit.integrate_regions(X, windows="0.501-0.502")
    with pytest.raises(ValueError):
        metbit.calibrate_integrals(np.ones((2, 1)), "max")
    with pytest.raises(ValueError):
        metbit.calibrate_integrals(np.zeros((2, 1)), "reference")
    with pytest.raises(ValueError):
        metbit.calibrate_integrals(np.zeros((2, 1)), "sum")
    assert np.allclose(metbit.calibrate_integrals(np.ones((2, 2)), "sum", value=100), 50)
    assert metbit.integral_table(pd.DataFrame([y, 2 * y], columns=ppm), snr=5).shape[0] == 2


def test_pick_peaks_caps_count(monkeypatch):
    monkeypatch.setattr(ig, "MAX_AUTO_PEAKS", 2)
    ppm = np.linspace(1, 0, 1001)
    y = sum(np.exp(-((ppm - c) / 0.003) ** 2) * h for c, h in ((0.2, 1), (0.5, 2), (0.8, 3)))
    assert len(ig.pick_peaks(ppm, y + 1e-6 * np.sin(np.arange(1001)), snr=5)[0]) == 2


def test_normalize_inputs():
    with pytest.raises(ValueError):
        metbit.normalize_spectra(np.ones((3, 1)))
    with pytest.raises(ValueError):
        metbit.normalize_spectra(np.array([[1.0, np.nan], [1, 2]]))
    with pytest.raises(ValueError):
        metbit.normalize_spectra(np.ones((3, 4)), "tsa")
    with pytest.raises(ValueError):
        metbit.normalize_spectra(np.zeros((3, 4)))  # zero reference
    with pytest.raises(ValueError):
        metbit.normalize_spectra(np.ones((3, 4)), "snv")  # flat rows
    with pytest.raises(ValueError):
        metbit.normalize_spectra(np.ones((3, 4)), "msc")  # flat reference
    with pytest.raises(ValueError):
        metbit.normalize_spectra(np.array([[1.0, -1.0], [-1.0, 1.0], [2.0, 3.0]]), "pqn")
    with pytest.raises(ValueError):
        metbit.normalize_spectra(np.array([[1e308, 1e-308], [1e-308, 1e308], [1.0, 1.0]]), "snv")
    assert np.allclose(metbit.preprocessing.qc.scale_columns(np.eye(3), "mean").sum(axis=0), 0)


def test_qc_inputs():
    X = np.ones((8, 3))
    assert np.isnan(metbit.qc_rsd(np.zeros((3, 2))))
    with pytest.raises(ValueError):
        metbit.qc_drift_correct(X, [np.nan] * 8, np.ones(8, bool))
    with pytest.raises(ValueError):
        metbit.qc_drift_correct(X, np.arange(8), np.arange(8) < 3)
    out = metbit.qc_drift_correct(pd.DataFrame(X), np.arange(8.0), np.arange(8) % 2 == 0)
    assert isinstance(out, pd.DataFrame) and np.allclose(out, 1)
    with pytest.raises(ValueError):
        metbit.detect_outliers(np.ones((4, 10)), n_components=3)
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 5)), index=list("abcdefghij"))
    assert list(metbit.detect_outliers(df).index) == list("abcdefghij")


def test_feature_tests_inputs():
    X, y = two_groups()
    with pytest.raises(ValueError):
        metbit.two_group_test(X, y, "A", "B", "anova")
    with pytest.raises(ValueError):
        metbit.two_group_test(X, y, "A", "B", "paired_t", pairs=np.arange(24))  # no pair has both groups
    with pytest.raises(ValueError):
        metbit.two_group_test(X.iloc[:13], y[:13], "A", "B")
    pairs = np.r_[np.arange(12), np.arange(12)]
    w = metbit.two_group_test(X.values, y, "A", "B", "wilcoxon", pairs=pairs, correction="none")
    assert w.feature.dtype.kind in "if"
    mw = metbit.two_group_test(X, y, "A", "B", "mannwhitney")
    assert mw.p_value.iloc[0] < 0.01
    assert metbit.stats.features._ppm(pd.DataFrame(np.ones((2, 2)), columns=["a", "b"])).tolist() == ["a", "b"]


def test_linear_model_inputs():
    X, y = two_groups()
    y3 = np.array(["A", "B", "C"] * 8)
    assert "F" in metbit.linear_model_test(X, y3).columns
    with pytest.raises(ValueError):
        metbit.linear_model_test(X, ["A"] * 24)
    with pytest.raises(ValueError):
        metbit.linear_model_test(X.iloc[:3], ["A", "B", "C"])
    cov = pd.DataFrame({"age": [np.nan] + [40.0] * 23, "site": [None] + ["x", "y"] * 11 + ["x"]})
    assert len(metbit.linear_model_test(X, y, cov)) == X.shape[1]


def test_correlation_and_normality_inputs():
    X, _ = two_groups()
    with pytest.raises(ValueError):
        metbit.correlation_test(X, np.arange(24), "kendall")
    with pytest.raises(ValueError):
        metbit.correlation_test(X, [1, 2, 3] + [np.nan] * 21)
    assert metbit.correlation_test(X, np.arange(24), "pearson").shape[0] == X.shape[1]
    with pytest.raises(ValueError):
        metbit.normality_test(X, test="ks")
    Z = pd.DataFrame({1.0: np.ones(10), 2.0: np.arange(10.0)})
    t = metbit.normality_test(Z, test="dagostino", correction="none").set_index("feature")
    assert t.loc[1.0, "note"] == "Constant feature" and t.loc[2.0, "note"] == ""
    assert metbit.normality_test(Z.iloc[:2]).note.str.startswith("Insufficient").all()


def test_biomarker_inputs():
    X, y = two_groups(n=4)
    with pytest.raises(ValueError):
        metbit.biomarker_panel(X, y, "A", "B", kfold=5)
    X, y = two_groups(n=10)
    r = metbit.biomarker_panel(X, y, "A", "B", features=[X.columns[0]], kfold=3, bootstrap=0)
    assert r["panel"] == [float(X.columns[0])] and np.isnan(r["ci"][0])


def test_time_inputs():
    pos, order = lg.time_positions(["b", "a", "b"], time_order=["b", "a", "c"])
    assert order == ["b", "a"] and list(pos) == [0, 1, 0]
    with pytest.raises(ValueError):
        lg.time_positions(["a", "b"], time_order=["a"])
    with pytest.raises(ValueError):
        lg.time_positions(["a", "a"])
    X, time, grp, subj = longitudinal()
    r = metbit.asca(X, time, permutations=0)  # no group, no subject
    assert list(r["table"].effect) == ["Time", "Residual"] and np.isnan(r["table"].p_value[0])
    assert metbit.asca(X, time, permutations=5)["table"].p_value[0] <= 1


def test_mixed_inputs():
    X, time, grp, subj = longitudinal()
    with pytest.raises(ValueError):
        metbit.mixed_model_test(X, time, subj, grp, time_as="ordinal")
    X2 = X.copy()
    X2[9.0] = 1.0  # a constant feature stays NaN
    num = metbit.mixed_model_test(X2, [t.split()[1] for t in time], subj, time_as="numeric")
    assert num.set_index("feature").loc[9.0].isna().all()
    lab = metbit.mixed_model_test(X, time, subj, time_as="numeric")
    assert "p_time" in lab.columns


def test_mixed_failure(monkeypatch):
    from statsmodels.regression import mixed_linear_model as mlm

    def broken(*a, **k):
        raise np.linalg.LinAlgError("singular")
    monkeypatch.setattr(mlm.MixedLM, "fit", broken)
    X, time, grp, subj = longitudinal()
    with pytest.raises(ValueError, match="could not be fitted"):
        metbit.mixed_model_test(X, time, subj, grp)


def test_friedman_inputs():
    X, time, grp, subj = longitudinal()
    keep = time != "day 10"
    with pytest.raises(ValueError):
        metbit.friedman_test(X[keep], time[keep], subj[keep])
    with pytest.raises(ValueError):
        metbit.friedman_test(X.iloc[:3], time[:3], subj[:3])


def test_validation_inputs():
    y = np.repeat(["A", "B"], 6)
    assert len(metbit.make_folds(y, cv="loo")[0]) == 12
    assert len(metbit.make_folds(y, np.repeat(np.arange(4), 3), cv="loo")[0]) == 4
    with pytest.raises(ValueError):
        metbit.make_folds(y, cv="bootstrap")
    with pytest.raises(ValueError):
        metbit.make_folds(y, np.repeat(np.arange(2), 6), k=3)
    with pytest.raises(ValueError):
        metbit.make_folds(y, k=7)
    assert len(metbit.make_folds(np.arange(12.0), k=3, classes=False)[0]) == 3
    assert len(metbit.make_folds(np.arange(12.0), np.repeat(np.arange(4), 3), k=3, classes=False)[0]) == 3
    groups = np.tile(np.arange(6), 2)  # labels vary within each group
    s = label_shuffler(y, groups)()
    assert sorted(s) == sorted(y)
    assert sorted(label_shuffler(y)()) == sorted(y)


def test_permutation_and_nested_with_groups():
    X, y = two_groups(n=12, p=10, shift=3)
    groups = np.r_[np.repeat(np.arange(6), 2), np.repeat(np.arange(6, 12), 2)]
    r = metbit.q2_permutation_test(X, y, groups, k=3, n_permutations=0)
    assert len(r["q2_perm"]) == 0 and r["p_q2"] == 1.0
    n = metbit.nested_opls_cv(X, y, groups, k=3, inner_k=3)
    assert 0 <= n["accuracy"] <= 1


def test_vip_regions_empty():
    assert metbit.vip_regions(np.linspace(1, 0, 5), np.zeros(5)) == []
