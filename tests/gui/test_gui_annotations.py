import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.annotations import AnnotationIn, csv_safe, parse_annotation_list, sanitize_label
from metbit_server.app import create_app
from metbit_server.project import load_project, save_project
from metbit_server.session import Session
from metbit_server.stages import STAGES

BY_ID = {s.id: s for s in STAGES}
SPIKE_PPM = 10 - 5 * 10 / 39  # column 5 of the csv_path fixture, where group B has an extra peak


def run(c, sid, params):
    job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
    assert "event: done" in c.get(f"/jobs/{job}/events", headers=AUTH).text


def test_labels_are_sanitized_like_metbit():
    assert sanitize_label("  <b>Lac</b> ") == "&lt;b&gt;Lac&lt;/b&gt;"
    assert sanitize_label(sanitize_label("a & b")) == "a &amp; b"  # idempotent
    assert len(sanitize_label("x" * 400)) == 256
    with pytest.raises(ValueError, match="empty"):
        AnnotationIn(ppm=1.0, label="   ")
    assert (csv_safe("=cmd"), csv_safe("lactate")) == ("'=cmd", "lactate")


def test_parse_accepts_metbit_annotator_json_and_skips_junk():
    items = parse_annotation_list(
        [{"x": 1.33, "text": "lactate", "textangle": 30}, {"ppm": 3.03, "label": "creatine"}, {"x": 2.0}, "junk"]
    )
    assert [(a["ppm"], a["label"], a["angle"]) for a in items] == [(1.33, "lactate", 30.0), (3.03, "creatine", 0.0)]
    assert len({a["id"] for a in items}) == 2


def test_add_delete_and_validate_over_http():
    c = TestClient(create_app(TOKEN))
    a = c.post("/annotations", json={"ppm": 1.33, "label": "lactate"}, headers=AUTH).json()
    assert (a["ppm"], a["label"], a["angle"]) == (1.33, "lactate", 0.0)
    assert c.get("/annotations", headers=AUTH).json() == [a]
    bad = c.post("/annotations", json={"ppm": 1.0, "label": " "}, headers=AUTH)
    assert bad.status_code == 422 and bad.json()["detail"][0]["loc"] == ["label"]
    assert c.delete(f"/annotations/{a['id']}", headers=AUTH).status_code == 200
    assert c.get("/annotations", headers=AUTH).json() == []
    assert c.delete("/annotations/nope", headers=AUTH).status_code == 404


def test_import_and_export_json_files(tmp_path):
    c = TestClient(create_app(TOKEN))
    (tmp_path / "in.json").write_text(json.dumps([{"x": 1.33, "text": "lactate"}]))
    imported = c.post("/annotations/import", json={"path": str(tmp_path / "in.json")}, headers=AUTH).json()
    assert [a["label"] for a in imported] == ["lactate"]
    c.post("/annotations/export", json={"path": str(tmp_path / "out.json")}, headers=AUTH)
    assert json.loads((tmp_path / "out.json").read_text())[0]["label"] == "lactate"
    (tmp_path / "bad.json").write_text("{not json")
    assert c.post("/annotations/import", json={"path": str(tmp_path / "bad.json")}, headers=AUTH).status_code == 422


def test_labels_overlay_spectra_but_not_scores(csv_path):
    c = TestClient(create_app(TOKEN))
    run(c, "import", {"path": csv_path})
    run(c, "pca", {})
    c.post("/annotations", json={"ppm": SPIKE_PPM, "label": "spike"}, headers=AUTH)
    notes = c.get("/stages/import/figure?view=spectra", headers=AUTH).json()["layout"]["annotations"]
    spike = next(n for n in notes if n["text"] == "spike")
    assert abs(spike["x"] - SPIKE_PPM) < 1e-9
    assert spike["yref"] == "y" and spike["y"] > 3  # sits on the tallest curve at that ppm
    scores = c.get("/stages/pca/figure?view=scores", headers=AUTH).json()
    assert all(n.get("text") != "spike" for n in scores["layout"].get("annotations", []))


def test_annotate_stage_single_and_median_views(run_all):
    single = BY_ID["annotate"].figures["spectra"](run_all(("annotate", {})).output("annotate"))
    assert len(single.data) == 12
    median = BY_ID["annotate"].figures["spectra"](run_all(("annotate", {"mode": "median"})).output("annotate"))
    assert [t.name for t in median.data] == ["A", "B"]
    assert BY_ID["annotate"].section == "Statistics"
    assert BY_ID["annotate"].ppm_views == {"spectra": "x"}


def test_intensity_table_reads_nearest_point_live_and_guards_csv(run_all):
    s = run_all(("annotate", {}))
    r = s.output("annotate")
    assert list(BY_ID["annotate"].table(r).columns) == ["Sample", "Group"]
    s.annotations.append({"id": "1", "ppm": 8.72, "label": "=spike", "angle": 0.0})
    table = BY_ID["annotate"].table(r)  # no re-run: the result shares the session's list
    X = s.output("import").X
    name = f"'=spike_{X.columns[5]:.3f}"
    assert list(table.columns) == ["Sample", "Group", name]
    assert np.allclose(table[name], X.iloc[:, 5])


def test_annotations_survive_a_project_roundtrip(run_all, tmp_path):
    s = run_all(("annotate", {}))
    s.annotations.append({"id": "x", "ppm": 1.33, "label": "lactate", "angle": 15.0})
    save_project(s, str(tmp_path / "p.metbit"))
    t = Session([st.id for st in STAGES], {st.id: st.after for st in STAGES})
    load_project(t, STAGES, str(tmp_path / "p.metbit"))
    assert [(a["ppm"], a["label"], a["angle"]) for a in t.annotations] == [(1.33, "lactate", 15.0)]
    t.annotations.append({"id": "y", "ppm": 8.72, "label": "late", "angle": 0.0})
    assert any(c.startswith("late_") for c in BY_ID["annotate"].table(t.output("annotate")).columns)
