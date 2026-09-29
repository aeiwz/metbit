import plotly.graph_objects as go
import pytest
from pydantic import ValidationError

from metbit_server.session import Session
from metbit_server.stages import STAGES, ImportParams, PCAParams

BY_ID = {s.id: s for s in STAGES}


def _noop(_pct):
    pass


def test_import_and_pca_registration():
    assert STAGES[0].id == "import" and STAGES[0].section == "Data"
    assert BY_ID["pca"].section == "Analysis"
    assert BY_ID["pca"].requires == "import"
    assert BY_ID["pca"].after == ["import", "process", "normalize", "align"]
    assert BY_ID["pca"].ppm_views == {"loadings": "x"}
    assert ImportParams.model_json_schema()["properties"]["path"]["format"] == "file-path"


def test_import_splits_features_metadata_and_labels(csv_path):
    data = BY_ID["import"].run(ImportParams(path=csv_path), Session(["import"]), _noop)
    assert data.X.shape == (12, 40)
    assert data.X.columns.dtype == float
    assert list(data.meta.columns) == ["Sample", "Group"]
    assert list(data.labels) == ["A"] * 6 + ["B"] * 6
    assert list(BY_ID["import"].table(data).columns) == ["Sample", "Group"]
    assert isinstance(BY_ID["import"].figures["spectra"](data), go.Figure)


def test_import_rejects_missing_group_column(csv_path):
    with pytest.raises(ValueError, match="Group column 'Class' not found"):
        BY_ID["import"].run(ImportParams(path=csv_path, group_column="Class"), Session(["import"]), _noop)


def test_import_rejects_csv_without_ppm_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Sample,Group\ns1,A\n")
    with pytest.raises(ValueError, match="No spectral columns"):
        BY_ID["import"].run(ImportParams(path=str(bad)), Session(["import"]), _noop)


def test_pca_run_figures_and_table(run_all):
    s = run_all()
    seen = []
    result = BY_ID["pca"].run(PCAParams(n_components=3), s, seen.append)
    assert seen[-1] == 1.0
    assert result.pcs == ["PC1", "PC2"]
    for view in ("scores", "loadings", "variance"):
        assert isinstance(BY_ID["pca"].figures[view](result), go.Figure)
    assert "Explained variance" in BY_ID["pca"].table(result).columns


def test_pca_params_reject_axis_beyond_components():
    with pytest.raises(ValidationError, match="exceeds"):
        PCAParams(n_components=2, pc_y=3)
    with pytest.raises(ValidationError, match="differ"):
        PCAParams(pc_x=1, pc_y=1)


def test_session_staleness_follows_dependencies():
    after = {"normalize": ["import"], "pca": ["import", "normalize"], "opls": ["import", "normalize"]}
    s = Session(["import", "normalize", "pca", "opls"], after)
    for sid in ("import", "normalize", "pca", "opls"):
        s.complete(sid, {}, sid)
    s.complete("pca", {}, "again")
    assert s.status["opls"] == "done"  # siblings never stale each other
    s.complete("import", {}, "new")
    assert s.status == {"import": "done", "normalize": "stale", "pca": "stale", "opls": "stale"}
    assert s.dependents("import") == ["normalize", "pca", "opls"]
    with pytest.raises(KeyError):
        Session(["import"]).output("import")


def test_working_matrix_and_group_options(run_all):
    from metbit_server.stages import resolve_schema
    from metbit_server.stages.base import working

    s = run_all()
    assert working(s) is s.output("import")
    empty = Session([st.id for st in STAGES])
    with pytest.raises(ValueError, match="Import data first"):
        working(empty)
    assert resolve_schema(BY_ID["pca"], s) == PCAParams.model_json_schema()  # no tagged fields
