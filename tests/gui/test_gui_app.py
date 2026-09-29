import threading

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from conftest import AUTH, TOKEN
from metbit_server.app import create_app
from metbit_server.stages import Stage


@pytest.fixture
def client():
    return TestClient(create_app(TOKEN))


def run_stage(client, sid, params):
    r = client.post(f"/stages/{sid}/run", json=params, headers=AUTH)
    assert r.status_code == 200, r.text
    body = client.get(f"/jobs/{r.json()['job_id']}/events", headers=AUTH).text
    return [ln.split(": ", 1)[1] for ln in body.splitlines() if ln.startswith("event: ")]


def statuses(client):
    return {s["id"]: s["status"] for s in client.get("/stages", headers=AUTH).json()}


def test_token_required(client):
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/health", headers=AUTH).json() == {"ok": True}


def test_cors_preflight_needs_no_token(client):
    r = client.options("/health", headers={"Origin": "null", "Access-Control-Request-Method": "GET",
                                           "Access-Control-Request-Headers": "authorization"})
    assert r.status_code == 200


def test_stages_listing(client):
    stages = {s["id"]: s for s in client.get("/stages", headers=AUTH).json()}
    assert next(iter(stages)) == "import"
    assert stages["import"]["status"] == "empty"
    assert stages["import"]["section"] == "Data"
    assert stages["import"]["has_table"] is True
    assert stages["import"]["schema"]["properties"]["path"]["format"] == "file-path"
    assert stages["pca"]["views"] == ["scores", "loadings", "variance"]
    assert stages["pca"]["ppm_views"] == {"loadings": "x"}


def test_import_then_pca_then_reimport_marks_stale(client, csv_path):
    assert run_stage(client, "import", {"path": csv_path})[-1] == "done"
    assert run_stage(client, "pca", {"n_components": 2})[-1] == "done"
    assert (statuses(client)["import"], statuses(client)["pca"]) == ("done", "done")
    run_stage(client, "import", {"path": csv_path})
    assert (statuses(client)["import"], statuses(client)["pca"]) == ("done", "stale")
    assert client.get("/stages", headers=AUTH).json()[0]["params"]["path"] == csv_path


def test_pca_before_import_is_409(client):
    r = client.post("/stages/pca/run", json={}, headers=AUTH)
    assert r.status_code == 409
    assert "import" in r.json()["detail"]


def test_invalid_params_are_422_with_field_loc(client):
    r = client.post("/stages/import/run", json={"path": "x", "group_column": 5}, headers=AUTH)
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["group_column"]


def test_unknown_stage_is_404(client):
    assert client.post("/stages/nope/run", json={}, headers=AUTH).status_code == 404


def test_failed_job_reports_error_and_status(client, tmp_path):
    events = run_stage(client, "import", {"path": str(tmp_path / "missing.csv")})
    assert events[-1] == "error"
    assert statuses(client)["import"] == "failed"


def test_second_run_while_busy_is_409():
    gate = threading.Event()

    class P(BaseModel):
        pass

    slow = Stage(id="slow", title="Slow", params=P, run=lambda p, s, prog: gate.wait(5), table=lambda o: None)
    client = TestClient(create_app(TOKEN, stages=[slow]))
    first = client.post("/stages/slow/run", json={}, headers=AUTH)
    assert first.status_code == 200
    assert client.post("/stages/slow/run", json={}, headers=AUTH).status_code == 409
    gate.set()
    body = client.get(f"/jobs/{first.json()['job_id']}/events", headers=AUTH).text
    assert "event: done" in body
