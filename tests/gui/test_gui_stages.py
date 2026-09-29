import plotly.graph_objects as go
import pytest
from pydantic import ValidationError

from metbit_server.session import Session
from metbit_server.stages import STAGES, ImportParams, PCAParams

BY_ID = {s.id: s for s in STAGES}


def _noop(_pct):
    pass


def _session_with_import(csv_path):
    s = Session([st.id for st in STAGES])
    p = ImportParams(path=csv_path)
    s.complete("import", p.model_dump(), BY_ID["import"].run(p, s, _noop))
    return s


def test_registry_order_and_requirements():
    assert [s.id for s in STAGES] == ["import", "pca"]
    assert BY_ID["pca"].requires == "import"
    assert ImportParams.model_json_schema()["properties"]["path"]["format"] == "file-path"


def test_import_splits_features_and_metadata(csv_path):
    s = Session(["import", "pca"])
    data = BY_ID["import"].run(ImportParams(path=csv_path), s, _noop)
    assert data.X.shape == (12, 40)
    assert list(data.meta.columns) == ["Sample", "Group"]
    assert list(BY_ID["import"].table(data).columns) == ["Sample", "Group"]


def test_import_rejects_missing_group_column(csv_path):
    with pytest.raises(ValueError, match="Group column 'Class' not found"):
        BY_ID["import"].run(ImportParams(path=csv_path, group_column="Class"), Session(["import"]), _noop)


def test_import_rejects_csv_without_ppm_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Sample,Group\ns1,A\n")
    with pytest.raises(ValueError, match="No spectral columns"):
        BY_ID["import"].run(ImportParams(path=str(bad)), Session(["import"]), _noop)


def test_pca_run_figures_and_table(csv_path):
    s = _session_with_import(csv_path)
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


def test_session_complete_marks_later_done_stages_stale():
    s = Session(["import", "pca"])
    s.complete("import", {}, 1)
    s.complete("pca", {}, 2)
    s.complete("import", {}, 3)
    assert s.status == {"import": "done", "pca": "stale"}
    with pytest.raises(KeyError):
        Session(["import"]).output("import")
