from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app
from metbit_server.project import load_project, save_project
from metbit_server.session import Session
from metbit_server.stages import STAGES
from metbit_server.stages.style import PALETTE


def run(c, sid, params):
    job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
    assert "event: done" in c.get(f"/jobs/{job}/events", headers=AUTH).text


def group_traces(fig: dict, group: str) -> list[dict]:
    """Traces for a group; metbit names them 'A' or 'A, A' when color and symbol are both set."""
    return [
        t for t in fig["data"]
        if str(t.get("name", "")).split(",")[0].strip() == group and "markers" in (t.get("mode") or "markers")
    ]


def client_with(csv_path, *runs):
    c = TestClient(create_app(TOKEN))
    for sid, params in (("import", {"path": csv_path}), *runs):
        run(c, sid, params)
    return c


def test_global_style_fills_groups_after_import(csv_path):
    c = TestClient(create_app(TOKEN))
    assert c.get("/style", headers=AUTH).json()["groups"] == {}
    run(c, "import", {"path": csv_path})
    assert c.get("/style", headers=AUTH).json()["groups"] == {
        "A": {"color": PALETTE[0], "symbol": "circle"},
        "B": {"color": PALETTE[1], "symbol": "circle"},
    }


def test_put_style_validates_and_recolors_figures(csv_path):
    c = client_with(csv_path, ("pca", {}))
    style = c.get("/style", headers=AUTH).json()
    style["groups"]["A"]["color"] = "#ff0000"
    assert c.put("/style", json={**style, "marker_size": 99}, headers=AUTH).status_code == 422
    assert c.put("/style", json=style, headers=AUTH).json()["groups"]["A"]["color"] == "#ff0000"
    fig = c.get("/stages/pca/figure?view=scores", headers=AUTH).json()
    assert group_traces(fig, "A")[0]["marker"]["color"] == "#ff0000"


def test_stage_override_only_affects_its_stage(csv_path):
    c = client_with(csv_path, ("pca", {}), ("opls", {"group_a": "A", "group_b": "B"}))
    body = c.put("/stages/pca/style", json={"groups": {"B": {"color": "#00ff00"}}, "font_size": 20}, headers=AUTH).json()
    assert body["override"] == {"groups": {"B": {"color": "#00ff00"}}, "font_size": 20}
    assert body["effective"]["groups"]["B"]["color"] == "#00ff00"
    assert body["effective"]["marker_size"] == 12
    pca = c.get("/stages/pca/figure?view=scores", headers=AUTH).json()
    opls = c.get("/stages/opls/figure?view=scores", headers=AUTH).json()
    assert group_traces(pca, "B")[0]["marker"]["color"] == "#00ff00"
    assert group_traces(opls, "B")[0]["marker"]["color"] == PALETTE[1]
    assert c.put("/stages/pca/style", json={"font_size": 99}, headers=AUTH).status_code == 422
    assert c.put("/stages/pca/style", json={}, headers=AUTH).json()["override"] == {}
    assert c.get("/stages/nope/style", headers=AUTH).status_code == 404


def test_style_survives_a_project_roundtrip(run_all, tmp_path):
    s = run_all()
    s.style = {"font_size": 18}
    s.stage_styles = {"pca": {"marker_size": 30}}
    save_project(s, str(tmp_path / "p.metbit"))
    t = Session([st.id for st in STAGES], {st.id: st.after for st in STAGES})
    load_project(t, STAGES, str(tmp_path / "p.metbit"))
    assert t.style == {"font_size": 18}
    assert t.stage_styles == {"pca": {"marker_size": 30}}
