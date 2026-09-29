import pandas as pd
from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app


def client_after(csv_path, *runs):
    c = TestClient(create_app(TOKEN))
    for sid, params in (("import", {"path": csv_path}), *runs):
        job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
        assert "event: done" in c.get(f"/jobs/{job}/events", headers=AUTH).text
    return c


def test_normalized_matrix_exports_as_a_reimportable_csv(csv_path, tmp_path):
    c = client_after(csv_path, ("normalize", {"method": "pqn"}))
    out = tmp_path / "normalized.csv"
    assert c.post("/stages/normalize/export", json={"path": str(out)}, headers=AUTH).json() == {"path": str(out)}
    df = pd.read_csv(out)
    assert list(df.columns[:2]) == ["Sample", "Group"]
    assert df.shape == (12, 42)
    client_after(str(out))  # the export imports cleanly as a new dataset


def test_analysis_exports_its_table(csv_path, tmp_path):
    c = client_after(csv_path, ("opls", {"group_a": "A", "group_b": "B"}))
    c.post("/stages/opls/export", json={"path": str(tmp_path / "vip.csv")}, headers=AUTH)
    assert "VIP" in pd.read_csv(tmp_path / "vip.csv").columns


def test_export_errors(csv_path, tmp_path):
    c = client_after(csv_path)
    assert c.post("/stages/pca/export", json={"path": str(tmp_path / "x.csv")}, headers=AUTH).status_code == 409
    assert c.post("/stages/import/export", json={"path": "relative.csv"}, headers=AUTH).status_code == 422
    missing_dir = tmp_path / "nope" / "x.csv"
    assert c.post("/stages/import/export", json={"path": str(missing_dir)}, headers=AUTH).status_code == 422
    c = client_after(csv_path, ("stocsy", {"anchor_ppm": 5.0}))
    assert c.post("/stages/stocsy/export", json={"path": str(tmp_path / "s.csv")}, headers=AUTH).status_code == 404


def test_listing_flags_exportable_stages():
    stages = {s["id"]: s for s in TestClient(create_app(TOKEN)).get("/stages", headers=AUTH).json()}
    assert stages["normalize"]["exportable"] is True
    assert stages["pca"]["exportable"] is True
    assert stages["stocsy"]["exportable"] is False
