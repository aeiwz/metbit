import plotly.graph_objects as go
from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app
from metbit_server.stages import STAGES

BY_ID = {s.id: s for s in STAGES}


def test_registry_order_and_sections():
    assert [s.id for s in STAGES][:3] == ["import", "normalize", "align"]
    assert all(s.section == "Data" for s in STAGES[:3])
    assert BY_ID["volcano"].section == "Statistics"


def test_volcano_table_and_figure(run_all):
    out = run_all(("volcano", {"group_a": "A", "group_b": "B"})).output("volcano")
    table = BY_ID["volcano"].table(out)
    assert {"feature", "log2FC", "p_value", "p_adj", "label"} <= set(table.columns)
    assert len(table) == 40
    assert isinstance(BY_ID["volcano"].figures["volcano"](out), go.Figure)


def test_volcano_without_correction(run_all):
    out = run_all(("volcano", {"group_a": "B", "group_b": "A", "correction": "none"})).output("volcano")
    assert "label" in BY_ID["volcano"].table(out).columns


def _run(c, sid, params):
    job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
    assert "event: done" in c.get(f"/jobs/{job}/events", headers=AUTH).text


def _statuses(c):
    return {s["id"]: s["status"] for s in c.get("/stages", headers=AUTH).json()}


def test_http_reimport_stales_analyses_but_siblings_stay(csv_path):
    c = TestClient(create_app(TOKEN))
    _run(c, "import", {"path": csv_path})
    _run(c, "pca", {})
    _run(c, "volcano", {"group_a": "A", "group_b": "B"})
    _run(c, "pca", {})
    st = _statuses(c)
    assert (st["pca"], st["volcano"]) == ("done", "done")
    _run(c, "normalize", {"method": "pqn"})
    st = _statuses(c)
    assert (st["import"], st["normalize"], st["pca"], st["volcano"]) == ("done", "done", "stale", "stale")


def test_http_listing_injects_groups_and_flags_tables(csv_path):
    c = TestClient(create_app(TOKEN))
    assert c.get("/stages/stocsy/table", headers=AUTH).status_code == 404
    _run(c, "import", {"path": csv_path})
    stages = {s["id"]: s for s in c.get("/stages", headers=AUTH).json()}
    assert stages["volcano"]["schema"]["properties"]["group_b"]["enum"] == ["A", "B"]
    assert stages["volcano"]["schema"]["properties"]["group_b"]["default"] == "B"
    assert stages["stocsy"]["has_table"] is False
    assert stages["volcano"]["section"] == "Statistics"
