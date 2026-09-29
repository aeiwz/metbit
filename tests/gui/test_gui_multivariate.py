import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pydantic import ValidationError

from metbit_server.stages import STAGES, HCAParams, resolve_schema

BY_ID = {s.id: s for s in STAGES}


@pytest.fixture
def csv3(tmp_path, csv_path):
    """Three groups of four plus a numeric Age column."""
    df = pd.read_csv(csv_path)
    df["Group"] = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    df.insert(2, "Age", np.linspace(20, 70, 12))
    path = tmp_path / "three.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_analysis_rail_order():
    assert [s.id for s in STAGES if s.section == "Analysis"] == ["pca", "opls", "plsr", "lda", "ica", "hca", "stocsy"]
    for sid in ("plsr", "lda", "ica", "hca"):
        assert BY_ID[sid].requires == "import"
        assert BY_ID[sid].after == ["import", "process", "normalize", "align"]


def test_plsr_target_defaults_to_numeric_column_and_runs(run_all, csv3):
    s = run_all(("import", {"path": csv3}))
    assert resolve_schema(BY_ID["plsr"], s)["properties"]["target"]["default"] == "Age"
    s = run_all(("import", {"path": csv3}), ("plsr", {"target": "Age"}))
    model = s.output("plsr")
    assert list(BY_ID["plsr"].table(model).columns) == ["R2", "Q2", "RMSE"]
    for view in ("predicted", "scores"):
        assert isinstance(BY_ID["plsr"].figures[view](model), go.Figure)


@pytest.mark.parametrize("target,msg", [("Sample", "must be numeric"), ("Weight", "not found")])
def test_plsr_rejects_bad_targets(run_all, csv3, target, msg):
    with pytest.raises(ValueError, match=msg):
        run_all(("import", {"path": csv3}), ("plsr", {"target": target}))


def test_lda_three_groups(run_all, csv3):
    r = run_all(("import", {"path": csv3}), ("lda", {})).output("lda")
    assert r.lds == ["LD1", "LD2"]
    for view in ("scores", "loadings"):
        assert isinstance(BY_ID["lda"].figures[view](r), go.Figure)
    assert BY_ID["lda"].ppm_views == {"loadings": "x"}


def test_lda_two_groups_uses_a_strip_plot(run_all):
    r = run_all(("lda", {})).output("lda")
    assert r.lds == ["LD1"]
    assert isinstance(BY_ID["lda"].figures["scores"](r), go.Figure)


def test_lda_rejects_too_many_components(run_all, csv3):
    with pytest.raises(ValueError, match="at most 2 components for 3 groups"):
        run_all(("import", {"path": csv3}), ("lda", {"n_components": 5}))


def test_ica(run_all):
    r = run_all(("ica", {"n_components": 3})).output("ica")
    assert BY_ID["ica"].table(r).shape == (12, 3)
    for view in ("components", "mixing"):
        assert isinstance(BY_ID["ica"].figures[view](r), go.Figure)


def test_hca_table_and_views(run_all, csv3):
    r = run_all(("import", {"path": csv3}), ("hca", {"n_clusters": 3})).output("hca")
    table = BY_ID["hca"].table(r)
    assert list(table.columns) == ["Sample", "Group", "Age", "Cluster"]
    assert table["Cluster"].nunique() <= 3
    for view in ("dendrogram", "heatmap"):
        assert isinstance(BY_ID["hca"].figures[view](r), go.Figure)


def test_hca_validation(run_all):
    with pytest.raises(ValidationError, match="Ward linkage requires the euclidean metric"):
        HCAParams(method="ward", metric="correlation")
    assert HCAParams(method="average", metric="correlation").metric == "correlation"
    with pytest.raises(ValueError, match="exceeds the number of samples"):
        run_all(("hca", {"n_clusters": 20}))


def test_hca_heatmap_labels_sit_left_of_the_cluster_strip(run_all):
    r = run_all(("hca", {"n_clusters": 2})).output("hca")
    assert BY_ID["hca"].figures["heatmap"](r).layout.yaxis.anchor == "x2"
