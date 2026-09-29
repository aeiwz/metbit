from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from metbit_server.stages import STAGES, ImportParams
from metbit_server.stages.base import working
from metbit_server.stages.data import BrukerSource, find_experiments

BY_ID = {s.id: s for s in STAGES}


def make_tree(root: Path, names, depth=1):
    for n in names:
        exp = (root / "batch" / n) if depth == 2 else (root / n)
        exp.mkdir(parents=True, exist_ok=True)
        (exp / "fid").write_bytes(b"")
    return root


@pytest.fixture
def bruker(tmp_path):
    folder = make_tree(tmp_path / "exps", ["61030601", "61020601", "61040601"])
    meta = pd.DataFrame({
        "fid": [61020601, 61030601, 61040601, 99],
        "Group": ["sham", "nx", "nx", "x"],
        "Week": [6, 6, 6, 1],
    })
    meta_path = tmp_path / "map.csv"
    meta.to_csv(meta_path, index=False)
    return {"source": "bruker", "folder": str(folder), "metadata_path": str(meta_path), "sample_column": "fid"}


def test_find_experiments_sorted_at_depth_one_and_two(tmp_path):
    make_tree(tmp_path / "one", ["b", "a"])
    assert [n for n, _ in find_experiments(str(tmp_path / "one"))] == ["a", "b"]
    make_tree(tmp_path / "two", ["y", "x"], depth=2)
    assert [n for n, _ in find_experiments(str(tmp_path / "two"))] == ["x", "y"]
    assert find_experiments(str(tmp_path / "one" / "a")) == [("a", str(tmp_path / "one" / "a"))]
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No Bruker experiments"):
        find_experiments(str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="Folder not found"):
        find_experiments(str(tmp_path / "missing"))


def test_bruker_import_joins_metadata_in_experiment_order(run_all, bruker):
    src = run_all(("import", bruker)).output("import")
    assert isinstance(src, BrukerSource)
    assert [n for n, _ in src.experiments] == ["61020601", "61030601", "61040601"]
    assert list(src.meta["fid"]) == [61020601, 61030601, 61040601]
    assert list(src.labels) == ["sham", "nx", "nx"]
    assert list(BY_ID["import"].table(src).columns) == ["fid", "Group", "Week"]
    assert "Run Process" in BY_ID["import"].figures["spectra"](src).layout.annotations[0].text


def test_bruker_import_reports_bad_metadata(run_all, bruker, tmp_path):
    pd.DataFrame({"fid": [61020601], "Group": ["sham"]}).to_csv(tmp_path / "short.csv", index=False)
    with pytest.raises(ValueError, match=r"2 experiment\(s\) have no metadata row"):
        run_all(("import", {**bruker, "metadata_path": str(tmp_path / "short.csv")}))
    with pytest.raises(ValueError, match="Metadata column 'Sample' not found"):
        run_all(("import", {**bruker, "sample_column": "Sample"}))


def test_import_params_require_the_right_paths():
    with pytest.raises(ValidationError, match="Choose a CSV file"):
        ImportParams()
    with pytest.raises(ValidationError, match="Bruker folder and a metadata CSV"):
        ImportParams(source="bruker", folder="/x")
    props = ImportParams.model_json_schema()["properties"]
    assert props["path"]["format"] == "file-path"
    assert props["folder"]["format"] == "dir-path"
    assert props["folder"]["show_if"] == {"source": "bruker"}


def test_analyses_wait_for_process_after_bruker_import(run_all, bruker):
    s = run_all(("import", bruker))
    with pytest.raises(ValueError, match="Run Process first"):
        working(s)
