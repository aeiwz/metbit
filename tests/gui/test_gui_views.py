import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app


@pytest.fixture
def client(csv_path):
    c = TestClient(create_app(TOKEN))
    for sid, params in (("import", {"path": csv_path}), ("pca", {"n_components": 3})):
        job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
        assert "event: done" in c.get(f"/jobs/{job}/events", headers=AUTH).text
    return c


def test_figure_is_plotly_json_without_fixed_size(client):
    fig = client.get("/stages/pca/figure?view=scores", headers=AUTH).json()
    assert fig["data"]
    assert "width" not in fig["layout"] and "height" not in fig["layout"]
    assert fig["layout"]["autosize"] is True


def test_unknown_view_is_404(client):
    assert client.get("/stages/pca/figure?view=nope", headers=AUTH).status_code == 404


def test_no_result_is_409():
    c = TestClient(create_app(TOKEN))
    assert c.get("/stages/pca/figure?view=scores", headers=AUTH).status_code == 409
    assert c.get("/stages/import/table", headers=AUTH).status_code == 409


def test_table_pagination(client):
    page = client.get("/stages/import/table?offset=10&limit=5", headers=AUTH).json()
    assert page["columns"] == ["Sample", "Group"]
    assert page["rows"] == [["s10", "B"], ["s11", "B"]]
    assert page["total"] == 12
    assert client.get("/stages/import/table?limit=5000", headers=AUTH).status_code == 422


def test_table_keeps_tiny_p_values():
    import pandas as pd
    from pydantic import BaseModel

    from metbit_server.stages import Stage

    class P(BaseModel):
        pass

    tiny = pd.DataFrame({"p_value": [3.2e-11, 7.5e-300, 0.5]})
    stage = Stage(id="t", title="T", params=P, run=lambda p, s, prog: tiny, table=lambda df: df)
    c = TestClient(create_app(TOKEN, stages=[stage]))
    job = c.post("/stages/t/run", json={}, headers=AUTH).json()["job_id"]
    c.get(f"/jobs/{job}/events", headers=AUTH)
    rows = c.get("/stages/t/table", headers=AUTH).json()["rows"]
    assert [r[0] for r in rows] == [3.2e-11, 7.5e-300, 0.5]
