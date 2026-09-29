import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app

SERVER_DIR = Path(__file__).resolve().parents[2] / "GUI" / "server"


def run(c, sid, params):
    job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
    return c.get(f"/jobs/{job}/events", headers=AUTH).text


def statuses(c):
    return {s["id"]: s["status"] for s in c.get("/stages", headers=AUTH).json()}


def test_autosave_after_each_job_then_restore(csv_path, tmp_path):
    auto = tmp_path / "autosave.metbit"
    c = TestClient(create_app(TOKEN, autosave=str(auto)))
    run(c, "import", {"path": csv_path})
    assert auto.is_file()
    run(c, "pca", {})
    restarted = TestClient(create_app(TOKEN, autosave=str(auto)))
    assert restarted.post("/session/restore", headers=AUTH).json() == {"restored": True}
    st = statuses(restarted)
    assert (st["import"], st["pca"]) == ("done", "done")


def test_restore_without_an_autosave(tmp_path):
    assert TestClient(create_app(TOKEN)).post("/session/restore", headers=AUTH).json() == {"restored": False}
    c = TestClient(create_app(TOKEN, autosave=str(tmp_path / "none.metbit")))
    assert c.post("/session/restore", headers=AUTH).json() == {"restored": False}


def test_failed_job_does_not_autosave(tmp_path):
    auto = tmp_path / "autosave.metbit"
    c = TestClient(create_app(TOKEN, autosave=str(auto)))
    assert "event: error" in run(c, "import", {"path": str(tmp_path / "missing.csv")})
    assert not auto.exists()


def test_save_and_load_routes(csv_path, tmp_path):
    c = TestClient(create_app(TOKEN))
    run(c, "import", {"path": csv_path})
    target = tmp_path / "study.metbit"
    assert c.post("/session/save", json={"path": str(target)}, headers=AUTH).json() == {"path": str(target)}
    other = TestClient(create_app(TOKEN))
    assert other.post("/session/load", json={"path": str(target)}, headers=AUTH).status_code == 200
    assert statuses(other)["import"] == "done"
    (tmp_path / "junk.metbit").write_text("junk")
    r = other.post("/session/load", json={"path": str(tmp_path / "junk.metbit")}, headers=AUTH)
    assert r.status_code == 422 and "Not a metbit project" in r.json()["detail"]
    assert c.post("/session/save", json={"path": "relative.metbit"}, headers=AUTH).status_code == 422


def test_cli_accepts_autosave():
    env = {**os.environ, "PYTHONPATH": str(SERVER_DIR.parents[1])}  # repo root, so `import metbit` works
    out = subprocess.run(
        [sys.executable, "-m", "metbit_server", "--help"], cwd=SERVER_DIR, env=env, capture_output=True, text=True, check=True
    )
    assert "--autosave" in out.stdout
