import contextlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit.stats.multitest import ANOVAStats
from metbit_server.app import create_app
from metbit_server.stages import STAGES, GroupTestParams, bin_matrix, resolve_schema

BY_ID = {s.id: s for s in STAGES}


def test_full_registry_order():
    assert [s.id for s in STAGES] == [
        "import", "normalize", "align",
        "pca", "opls", "plsr", "lda", "ica", "hca", "stocsy",
        "volcano", "grouptests",
    ]
    assert BY_ID["grouptests"].section == "Statistics"


def test_bin_matrix():
    X = pd.DataFrame(np.arange(20.0).reshape(2, 10), columns=np.linspace(1.95, 0.05, 10))
    assert bin_matrix(X, 0) is X
    b = bin_matrix(X, 1.0)
    assert list(b.columns) == [1.5, 0.5]
    assert b.iloc[0].tolist() == [np.mean(range(0, 5)), np.mean(range(5, 10))]


def test_anova_ranks_the_spiked_feature_first_and_matches_metbit(run_all):
    s = run_all(("grouptests", {"bin_width": 0}))
    r = s.output("grouptests")
    spiked = s.output("import").X.columns[5]
    assert np.isclose(r.table["ppm"].iloc[0], spiked)
    assert r.feature == str(spiked)
    with contextlib.redirect_stdout(io.StringIO()):
        ref = ANOVAStats(r.data, x_col="Group", y_col=r.feature)
        ref.fit()
    assert np.isclose(r.table["p_value"].iloc[0], ref.get_anova_table()["p_value"].iloc[0])
    assert list(BY_ID["grouptests"].table(r).columns) == ["ppm", "statistic", "p_value", "p_adj"]
    assert isinstance(BY_ID["grouptests"].figures["boxplot"](r), go.Figure)


def test_kruskal_survives_constant_features(run_all, csv_path, tmp_path):
    df = pd.read_csv(csv_path)
    df[df.columns[10]] = 1.0  # a flat region, like an excluded water window
    path = tmp_path / "flat.csv"
    df.to_csv(path, index=False)
    r = run_all(("import", {"path": str(path)}), ("grouptests", {"test": "kruskal", "bin_width": 0})).output("grouptests")
    assert r.table["p_value"].isna().sum() == 1
    assert r.table["p_value"].iloc[-1] != r.table["p_value"].iloc[-1]  # NaN sorted last
    assert isinstance(BY_ID["grouptests"].figures["boxplot"](r), go.Figure)


def test_feature_choice_and_no_correction(run_all):
    r = run_all(("grouptests", {"bin_width": 0, "feature_ppm": 2.0, "correction": "none"})).output("grouptests")
    assert np.isclose(float(r.feature), 2.0, atol=0.13)  # nearest of 40 points spaced 0.256 ppm
    assert np.allclose(r.table["p_adj"], r.table["p_value"], equal_nan=True)


def test_blank_feature_means_top_hit():
    assert GroupTestParams(feature_ppm="").feature_ppm is None


def test_needs_two_groups(run_all, csv_path, tmp_path):
    df = pd.read_csv(csv_path)
    df["Group"] = "A"
    path = tmp_path / "one.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="at least two groups"):
        run_all(("import", {"path": str(path)}), ("grouptests", {}))


def test_schema_and_http_run(run_all, csv_path):
    prop = resolve_schema(BY_ID["grouptests"], run_all())["properties"]["feature_ppm"]
    assert prop["type"] == "number" and prop["pick_ppm"] is True
    c = TestClient(create_app(TOKEN))
    for sid, params in (("import", {"path": csv_path}), ("grouptests", {"feature_ppm": "", "bin_width": 0.5})):
        job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
        assert "event: done" in c.get(f"/jobs/{job}/events", headers=AUTH).text
    page = c.get("/stages/grouptests/table", headers=AUTH).json()
    assert page["columns"] == ["ppm", "statistic", "p_value", "p_adj"]
