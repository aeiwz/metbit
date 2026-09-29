import numpy as np
import plotly.graph_objects as go
from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app
from metbit_server.stages import STAGES
from metbit_server.stages.style import PlotStyle, render

BY_ID = {s.id: s for s in STAGES}


def text(title) -> str:
    """metbit bolds plot titles with <b> tags."""
    return title.text.replace("<b>", "").replace("</b>", "")


def run(c, sid, params):
    job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
    assert "event: done" in c.get(f"/jobs/{job}/events", headers=AUTH).text


def test_render_forwards_only_declared_keywords():
    seen = {}
    render(lambda r: go.Figure(), None, PlotStyle(), ppm=1.0)  # no ppm parameter: not passed
    render(lambda r, style=None, ppm=None: seen.update(ppm=ppm) or go.Figure(), None, PlotStyle(), ppm=2.5)
    assert seen == {"ppm": 2.5}


def test_boxplot_follows_a_requested_ppm_and_reports_its_statistics(run_all):
    s = run_all(("grouptests", {"bin_width": 0}))
    r = s.output("grouptests")
    target = s.output("import").X.columns[20]
    fig = BY_ID["grouptests"].figures["boxplot"](r, None, ppm=float(target) + 0.01)
    row = r.table.loc[np.isclose(r.table["ppm"], target)].iloc[0]
    assert text(fig.layout.title) == f"{target:.4f} ppm · ANOVA p = {row['p_value']:.2g} · FDR {row['p_adj']:.2g}"


def test_default_boxplot_is_the_top_feature_and_correction_can_be_off(run_all):
    r = run_all(("grouptests", {"bin_width": 0, "test": "kruskal", "correction": "none"})).output("grouptests")
    title = text(BY_ID["grouptests"].figures["boxplot"](r).layout.title)
    assert title.startswith(f"{r.table['ppm'].iloc[0]:.4f} ppm · Kruskal-Wallis p = ")
    assert "FDR" not in title and "Bonferroni" not in title


def test_companion_spectra_mark_the_selected_ppm(run_all):
    s = run_all(("grouptests", {"bin_width": 0}))
    r = s.output("grouptests")
    fig = BY_ID["grouptests"].figures["spectra"](r, None, ppm=5.0)
    assert [t.name for t in fig.data] == ["A", "B"]
    ppm = s.output("import").X.columns.to_numpy(dtype=float)
    nearest = ppm[np.abs(ppm - 5.0).argmin()]
    assert abs(fig.layout.shapes[0].x0 - nearest) < 1e-9
    assert BY_ID["grouptests"].companions == {"boxplot": "spectra"}
    assert BY_ID["grouptests"].ppm_views == {"spectra": "x"}


def test_http_ppm_query_and_listing(csv_path):
    c = TestClient(create_app(TOKEN))
    run(c, "import", {"path": csv_path})
    run(c, "grouptests", {"bin_width": 0})
    stage = {s["id"]: s for s in c.get("/stages", headers=AUTH).json()}["grouptests"]
    assert stage["companions"] == {"boxplot": "spectra"}
    box = c.get("/stages/grouptests/figure?view=boxplot&ppm=5.0", headers=AUTH).json()
    assert " ppm · ANOVA p = " in box["layout"]["title"]["text"]
    spectra = c.get("/stages/grouptests/figure?view=spectra&ppm=5.0", headers=AUTH).json()
    assert abs(spectra["layout"]["shapes"][0]["x0"] - 5.0) < 0.2
