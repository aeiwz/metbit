from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app
from metbit_server.stages import STAGES, ProcessParams
from metbit_server.stages import data as data_mod
from metbit_server.stages.base import working

BY_ID = {s.id: s for s in STAGES}
HEART = Path(__file__).resolve().parents[2] / "Benchmark" / "data" / "MTBLS2052" / "proc_subset_heart_wk6"
SAMPLE_MAP = HEART.parent / "noesy_sample_map.csv"
needs_data = pytest.mark.skipif(not HEART.is_dir(), reason="MTBLS2052 subset is local-only benchmark data")


@pytest.fixture
def bruker(tmp_path):
    for n in ["61030601", "61020601", "61040601"]:
        (tmp_path / "exps" / n).mkdir(parents=True)
        (tmp_path / "exps" / n / "fid").write_bytes(b"")
    meta = pd.DataFrame({"fid": [61020601, 61030601, 61040601], "Group": ["sham", "nx", "nx"]})
    meta.to_csv(tmp_path / "map.csv", index=False)
    return {"source": "bruker", "folder": str(tmp_path / "exps"), "metadata_path": str(tmp_path / "map.csv"), "sample_column": "fid"}


@pytest.fixture
def heart_subset(tmp_path):
    """First six heart experiments plus their metadata rows."""
    names = sorted(d.name for d in HEART.iterdir() if (d / "fid").exists())[:6]
    subset = tmp_path / "heart"
    subset.mkdir()
    for n in names:
        (subset / n).symlink_to(HEART / n)
    m = pd.read_csv(SAMPLE_MAP)
    meta = tmp_path / "heart_meta.csv"
    m[m["fid"].astype(str).isin(names)].to_csv(meta, index=False)
    return {
        "source": "bruker",
        "folder": str(subset),
        "metadata_path": str(meta),
        "sample_column": "fid",
        "group_column": "Factor Value[nephrectomy]",
    }


def test_registry_and_dependencies():
    assert [s.id for s in STAGES][:4] == ["import", "process", "normalize", "align"]
    assert BY_ID["process"].after == ["import"]
    assert BY_ID["normalize"].after == ["import", "process"]
    assert BY_ID["align"].after == ["import", "process", "normalize"]
    assert BY_ID["pca"].after == ["import", "process", "normalize", "align"]


def test_process_stacks_experiments_with_per_sample_progress(run_all, bruker, monkeypatch):
    calls = []

    def one(path, p):
        calls.append(Path(path).name)
        return np.full(50, float(len(calls))), np.linspace(10, -0.5, 50)

    monkeypatch.setattr(data_mod, "_process_one", one)
    s = run_all(("import", bruker))
    seen = []
    out = BY_ID["process"].run(ProcessParams(calibration="none"), s, seen.append)
    assert calls == ["61020601", "61030601", "61040601"]
    assert out.X.shape == (3, 50) and list(out.X.index) == calls
    assert out.X.iloc[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert out.X.columns.dtype == float
    assert len(seen) == 4 and seen == sorted(seen) and seen[-1] == 1.0
    assert list(out.labels) == ["sham", "nx", "nx"]
    s.complete("process", {}, out)
    assert working(s) is out


def test_process_rejects_mismatched_sizes(run_all, bruker, monkeypatch):
    sizes = iter([50, 50, 40])

    def one(path, p):
        n = next(sizes)
        return np.zeros(n), np.linspace(10, 0, n)

    monkeypatch.setattr(data_mod, "_process_one", one)
    with pytest.raises(ValueError, match="same sweep width"):
        run_all(("import", bruker), ("process", {"calibration": "none"}))


def test_process_is_blocked_after_csv_import(run_all, csv_path):
    assert "Bruker" in BY_ID["process"].blocked(run_all())
    c = TestClient(create_app(TOKEN))
    job = c.post("/stages/import/run", json={"path": csv_path}, headers=AUTH).json()["job_id"]
    c.get(f"/jobs/{job}/events", headers=AUTH)
    stages = {x["id"]: x for x in c.get("/stages", headers=AUTH).json()}
    assert "Bruker" in stages["process"]["blocked"]
    assert stages["pca"]["blocked"] is None
    r = c.post("/stages/process/run", json={}, headers=AUTH)
    assert r.status_code == 409 and "Bruker" in r.json()["detail"]


def test_process_params_schema():
    props = ProcessParams.model_json_schema()["properties"]
    assert props["auto_phasing"]["type"] == "boolean"
    assert props["baseline"]["default"] == "none"


@needs_data
def test_process_matches_metbit_on_real_bruker_data(run_all, heart_subset):
    from metbit.nmr.preprocess import nmr_preprocessing

    out = run_all(("import", heart_subset), ("process", {})).output("process")
    ref = nmr_preprocessing(heart_subset["folder"]).get_data()
    assert out.X.shape == ref.shape
    assert np.allclose(out.X.to_numpy(), ref.to_numpy())
    assert np.allclose(out.X.columns.to_numpy(float), ref.columns.to_numpy(float))
    assert list(out.X.index) == [str(i) for i in ref.index]


@needs_data
def test_real_bruker_pipeline_over_http(heart_subset):
    c = TestClient(create_app(TOKEN))

    def run(sid, params):
        job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
        return c.get(f"/jobs/{job}/events", headers=AUTH).text

    assert "event: done" in run("import", heart_subset)
    events = run("process", {})
    assert events.count("event: progress") == 7 and "event: done" in events
    assert "event: done" in run("pca", {})
    assert c.get("/stages/process/figure?view=spectra", headers=AUTH).status_code == 200
