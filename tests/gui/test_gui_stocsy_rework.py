from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app
from metbit_server.stages import STAGES, STOCSYParams
from metbit_server.stages.analysis import sample_indices

BY_ID = {s.id: s for s in STAGES}


def run(c, sid, params):
    job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
    assert "event: done" in c.get(f"/jobs/{job}/events", headers=AUTH).text


def test_sample_indices_follow_the_dash_app():
    assert sample_indices(3) == [0, 1, 2]
    assert len(sample_indices(50)) == 30
    assert len(sample_indices(200)) == 40
    assert sample_indices(50) == sample_indices(50)  # seeded
    assert len(set(sample_indices(50))) == 30


def test_spectra_pre_view_renders_before_any_run(csv_path):
    c = TestClient(create_app(TOKEN))
    assert c.get("/stages/stocsy/figure?view=spectra", headers=AUTH).status_code == 409
    run(c, "import", {"path": csv_path})
    stocsy = {s["id"]: s for s in c.get("/stages", headers=AUTH).json()}["stocsy"]
    assert stocsy["pre_views"] == ["spectra"] and stocsy["status"] == "empty"
    assert stocsy["ppm_views"] == {"spectra": "x", "correlation": "x"}
    fig = c.get("/stages/stocsy/figure?view=spectra", headers=AUTH).json()
    assert len(fig["data"]) == 12
    assert c.get("/stages/stocsy/figure?view=correlation", headers=AUTH).status_code == 409


def test_results_are_cached_per_anchor_and_threshold(run_all):
    s = run_all(("stocsy", {"anchor_ppm": 5.0}))
    first = s.output("stocsy")
    again = BY_ID["stocsy"].run(STOCSYParams(anchor_ppm=5.0), s, lambda _p: None)
    other_p = BY_ID["stocsy"].run(STOCSYParams(anchor_ppm=5.0, p_threshold=0.01), s, lambda _p: None)
    assert again is first
    assert other_p is not first


def test_new_data_starts_a_fresh_cache(run_all):
    s = run_all(("stocsy", {"anchor_ppm": 5.0}), ("normalize", {"method": "pqn"}))
    first = s.output("stocsy")
    assert BY_ID["stocsy"].run(STOCSYParams(anchor_ppm=5.0), s, lambda _p: None) is not first


def test_correlation_figure_is_not_mutated_by_repeated_fetches(csv_path):
    c = TestClient(create_app(TOKEN))
    run(c, "import", {"path": csv_path})
    run(c, "stocsy", {"anchor_ppm": 5.0})
    c.post("/annotations", json={"ppm": 5.0, "label": "anchor"}, headers=AUTH)
    for _ in range(3):
        fig = c.get("/stages/stocsy/figure?view=correlation", headers=AUTH).json()
    assert [n.get("text") for n in fig["layout"]["annotations"]].count("anchor") == 1
