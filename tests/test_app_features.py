"""Features shared with the metbit desktop app: regions, integration, normalization, QC,
univariate tests, biomarker panel, time series and leakage-aware validation."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

import metbit


def lorentz(ppm, centre, width=0.004, height=1.0):
    return height / (1 + ((ppm - centre) / width) ** 2)


@pytest.fixture
def spectra():
    rng = np.random.default_rng(1)
    ppm = np.round(np.linspace(10, 0, 2001), 4)
    base = lorentz(ppm, 1.33) + 0.5 * lorentz(ppm, 3.04) + 0.3 * lorentz(ppm, 4.8)
    X = np.array([base * rng.uniform(0.8, 1.2) + rng.normal(0, 1e-4, len(ppm)) for _ in range(20)])
    return pd.DataFrame(X, columns=ppm)


def two_groups(n=12, p=30, shift=2.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(10, 1, (2 * n, p))
    X[n:, 0] += shift
    y = np.array(["A"] * n + ["B"] * n)
    return pd.DataFrame(X, columns=np.linspace(9, 1, p)), y


# regions and integration ----------------------------------------------------

def test_parse_windows_orders_high_low():
    assert metbit.parse_windows("3.2-2.8, 1.2-1.5") == [(3.2, 2.8), (1.5, 1.2)]
    with pytest.raises(ValueError):
        metbit.parse_windows("abc")


def test_select_regions_keep_cut_suppress(spectra):
    out = metbit.select_regions(spectra, keep="0.5-9.5", cut="4.7-4.9", suppress="3.0-3.1", suppress_to="zero")
    ppm = out.columns.to_numpy(dtype=float)
    assert ppm.max() <= 9.5 and ppm.min() >= 0.5
    assert not ((ppm >= 4.7) & (ppm <= 4.9)).any()
    assert (out.loc[:, (ppm >= 3.0) & (ppm <= 3.1)] == 0).all().all()
    line = metbit.select_regions(spectra, suppress="1.30-1.36")
    seg = line.loc[0, (ppm_all := line.columns.to_numpy(dtype=float)) <= 1.36]
    assert np.all(np.diff(seg[ppm_all[ppm_all <= 1.36] >= 1.30].to_numpy()) ** 2 < 1e-6)  # straight line


def test_bin_spectra_averages(spectra):
    b = metbit.bin_spectra(spectra, 0.5)
    assert b.shape[1] == 21 and np.isclose(b.columns[0], 10.25)  # 10.0 opens the [10, 10.5) bin
    assert metbit.bin_spectra(spectra, 0) is spectra


def test_pick_peaks_and_integrate(spectra):
    med = np.median(spectra.to_numpy(), axis=0)
    peaks, regions = metbit.pick_peaks(spectra.columns.to_numpy(dtype=float), med)
    assert any(abs(p - 1.33) < 0.002 for p in peaks) and any(abs(p - 3.04) < 0.002 for p in peaks)
    table = metbit.integral_table(spectra, regions, scale="reference", reference=1, value=9)
    assert np.allclose(table.iloc[:, 0], 9)
    # trapezoid area of a constant: width * height
    flat = pd.DataFrame(np.ones((2, 101)), columns=np.round(np.linspace(1, 0, 101), 4))
    assert np.allclose(metbit.integrate_regions(flat, windows="0.2-0.6"), 0.4)
    assert np.allclose(metbit.integrate_regions(flat, windows="0.2-0.6", chord=True), 0)


# normalization, QC, outliers ------------------------------------------------

@pytest.mark.parametrize("method", ["pqn", "snv", "snv_pqn", "msc"])
def test_normalize_spectra_affine(spectra, method):
    out, f, o = metbit.normalize_spectra(spectra, method)
    assert np.allclose(out.to_numpy(), (spectra.to_numpy() - o[:, None]) * f[:, None])


def test_pqn_undoes_dilution(spectra):
    shape = spectra.iloc[0].abs() + 1e-3
    diluted = pd.DataFrame(np.outer(np.linspace(0.5, 2, 10), shape), columns=spectra.columns)
    out, factors, _ = metbit.normalize_spectra(diluted, "pqn")
    assert np.allclose(out.to_numpy(), out.to_numpy()[0]) and np.allclose(factors * np.linspace(0.5, 2, 10), factors[0] * 0.5)


def test_qc_drift_correct_removes_trend():
    rng = np.random.default_rng(0)
    order = np.arange(40.0)
    X = np.outer(1 + 0.02 * order, np.ones(5)) * 100 + rng.normal(0, 0.5, (40, 5))
    qc = order % 4 == 0
    out = metbit.qc_drift_correct(X, order, qc, span=0.75)
    assert metbit.qc_rsd(out[qc]) < metbit.qc_rsd(X[qc]) / 5


def test_detect_outliers_flags_extreme_sample():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (30, 40))
    X[5] += 15
    res = metbit.detect_outliers(X, n_components=2, scaling="uv")
    assert res.loc[5, "outlier"] != "" and (res["outlier"] != "").sum() <= 4


# univariate -----------------------------------------------------------------

def test_two_group_test_matches_scipy():
    X, y = two_groups()
    t = metbit.two_group_test(X, y, "A", "B", "welch")
    assert np.isclose(t.feature.iloc[0], X.columns[0])
    ref = st.ttest_ind(X.values[y == "B", 0], X.values[y == "A", 0], equal_var=False).pvalue
    assert np.isclose(t.p_value.iloc[0], ref) and t["effect size"].iloc[0] > 1
    assert t["effect 95% CI low"].iloc[0] < t["effect size"].iloc[0] < t["effect 95% CI high"].iloc[0]


def test_paired_tests_need_pairs():
    X, y = two_groups()
    pairs = np.r_[np.arange(12), np.arange(12)]
    t = metbit.two_group_test(X, y, "A", "B", "paired_t", pairs=pairs)
    assert t.p_value.notna().all()
    with pytest.raises(ValueError):
        metbit.two_group_test(X, y, "A", "B", "wilcoxon")


def test_linear_model_two_groups_equals_student_t():
    X, y = two_groups()
    lm = metbit.linear_model_test(X, y).sort_values("feature").reset_index(drop=True)
    tt = metbit.two_group_test(X, y, "A", "B", "student").sort_values("feature").reset_index(drop=True)
    assert np.allclose(lm.p_value, tt.p_value)
    cov = pd.DataFrame({"age": np.random.default_rng(3).normal(50, 5, len(y)), "sex": ["m", "f"] * 12})
    assert metbit.linear_model_test(X, y, cov).p_value.iloc[0] < 0.01


def test_correlation_matches_scipy():
    X, _ = two_groups()
    target = X.iloc[:, 3] * 2 + np.random.default_rng(2).normal(0, 0.1, len(X))
    c = metbit.correlation_test(X, target, "spearman")
    assert np.isclose(c.feature.iloc[0], X.columns[3])
    assert np.isclose(c.r.iloc[0], st.spearmanr(X.iloc[:, 3], target).statistic)


def test_normality_test_rows():
    X, y = two_groups(p=4)
    t = metbit.normality_test(X, y)
    assert len(t) == 8 and set(t.group) == {"A", "B"}


def test_biomarker_panel_separates():
    X, y = two_groups(n=20, shift=3)
    r = metbit.biomarker_panel(X, y, "A", "B", kfold=5, bootstrap=50)
    assert np.isclose(r["panel"][0], X.columns[0]) and r["auc"] > 0.85
    assert r["ci"][0] <= r["auc"] <= r["ci"][1] + 1e-9


# time series ----------------------------------------------------------------

def longitudinal(seed=0):
    rng = np.random.default_rng(seed)
    subj, time, grp, rows = [], [], [], []
    for s in range(10):
        level = rng.normal(0, 1, 6)
        for t in range(3):
            g = "treated" if s < 5 else "control"
            x = level + rng.normal(0, 0.3, 6)
            if g == "treated":
                x[0] += 2.0 * t
            rows.append(x)
            subj.append(f"s{s}"), time.append(f"day {t * 5}"), grp.append(g)
    return pd.DataFrame(rows, columns=[5.0, 4.0, 3.0, 2.0, 1.5, 1.0]), np.array(time), np.array(grp), np.array(subj)


def test_time_positions_natural_order():
    pos, order = metbit.analysis.longitudinal.time_positions(["day 10", "day 2", "day 2"])
    assert order == ["day 2", "day 10"] and list(pos) == [1, 0, 0]


def test_asca_finds_interaction():
    X, time, grp, subj = longitudinal()
    r = metbit.asca(X, time, grp, subj, permutations=99)
    tab = r["table"].set_index("effect")
    assert np.isclose(tab.ssq_percent.sum(), 100)
    assert tab.loc["Time x Group", "p_value"] < 0.05


def test_mixed_model_and_friedman():
    X, time, grp, subj = longitudinal()
    m = metbit.mixed_model_test(X, time, subj, grp)
    assert m.feature.iloc[0] == 5.0 and m.p_interaction_adj.iloc[0] < 0.05
    f = metbit.friedman_test(X[grp == "treated"], time[grp == "treated"], subj[grp == "treated"])
    assert f.feature.iloc[0] == 5.0


# validation -----------------------------------------------------------------

def test_grouped_folds_keep_subjects_together():
    y = np.repeat(["A", "B"], 20)
    groups = np.repeat(np.arange(10), 4)
    for train, test in metbit.make_folds(y, groups, k=5)[0]:
        assert not set(groups[train]) & set(groups[test])


def test_label_shuffler_moves_whole_subjects():
    from metbit.validation.q2 import label_shuffler
    y = np.repeat(["A", "B"], 20)
    groups = np.repeat(np.arange(10), 4)
    s = label_shuffler(y, groups)()
    assert all(len(set(s[groups == g])) == 1 for g in range(10))


def test_q2_permutation_and_nested():
    X, y = two_groups(n=15, p=20, shift=3)
    r = metbit.q2_permutation_test(X, y, k=5, n_permutations=19, n_jobs=1)
    assert r["q2"] > 0.3 and r["p_q2"] == 0.05 and len(r["q2_perm"]) == 19
    n = metbit.nested_opls_cv(X, y, k=5, inner_k=4)
    assert n["accuracy"] > 0.8 and len(n["components"]) == 5
    assert len(metbit.repeated_q2(X, y, k=5, repeats=3)) == 3


def test_vip_regions_merges_and_ranks():
    ppm = np.linspace(5, 0, 501)
    vip = np.zeros(501)
    vip[100:110] = 3
    vip[111:115] = 2  # one point below threshold: same peak
    vip[400] = 1.5
    regions = metbit.vip_regions(ppm, vip, 1.0)
    assert len(regions) == 2 and regions[0][2] in range(100, 110)


def test_asca_permutations_keep_subjects_together():
    """Under no effect, subjects with their own baselines: single-sample shuffles rejected Group in
    every run and Time x Group in about a quarter; the subject-aware schemes stay near alpha."""
    rejected = {"Group": 0, "Time x Group": 0}
    for i in range(30):
        rng = np.random.default_rng(100 + i)
        offsets = np.repeat(rng.normal(0, 1, (12, 20)), 3, axis=0)
        X = pd.DataFrame(10 + offsets + rng.normal(0, 0.5, (36, 20)))
        time, grp, subj = np.tile(["0", "1", "2"], 12), np.repeat(["a", "b"], 18), np.repeat(np.arange(12), 3)
        tab = metbit.asca(X, time, grp, subj, permutations=49, seed=i)["table"].set_index("effect")
        for k in rejected:
            rejected[k] += tab.loc[k, "p_value"] < 0.05
    assert rejected["Group"] <= 6 and rejected["Time x Group"] <= 6
