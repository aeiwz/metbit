from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app
from metbit_server.project import load_project, save_project
from metbit_server.session import Session
from metbit_server.stages import STAGES

BY_ID = {s.id: s for s in STAGES}


def run(c, sid, params):
    job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
    assert "event: done" in c.get(f"/jobs/{job}/events", headers=AUTH).text


def annotated(csv_path):
    c = TestClient(create_app(TOKEN))
    run(c, "import", {"path": csv_path})
    run(c, "annotate", {})
    return c


def spectra(c, query=""):
    return c.get(f"/stages/annotate/figure?view=spectra{query}", headers=AUTH).json()


def notes(fig, label):
    return [n for n in fig["layout"].get("annotations", []) if n.get("text") == label]


def test_labels_carry_their_offset_and_can_be_dragged(csv_path):
    c = annotated(csv_path)
    a = c.post("/annotations", json={"ppm": 5.0, "label": "lac"}, headers=AUTH).json()
    assert (a["ax"], a["ay"]) == (0.0, -40.0)
    moved = c.patch(f"/annotations/{a['id']}", json={"ax": 25, "ay": -80}, headers=AUTH).json()
    assert (moved["ax"], moved["ay"]) == (25.0, -80.0)
    note = notes(spectra(c), "lac")[0]
    assert (note["ax"], note["ay"], note["name"]) == (25, -80, a["id"])
    assert c.patch(f"/annotations/{a['id']}", json={"ay": 9999}, headers=AUTH).status_code == 422
    assert c.patch("/annotations/nope", json={"ax": 1}, headers=AUTH).status_code == 404


def test_mode_switches_without_a_rerun(csv_path):
    c = annotated(csv_path)
    single, median = spectra(c), spectra(c, "&mode=median")
    assert [t["name"] for t in single["data"]][:2] == ["s0", "s1"]
    assert {t.get("legendgroup") for t in single["data"]} == {"A", "B"}
    assert [t["name"] for t in median["data"]] == ["A", "B"]
    assert [t["name"] for t in spectra(c, "&mode=bogus")["data"]][:1] == ["s0"]  # unknown mode: stage default


def test_label_font_size_and_line_styles(csv_path):
    c = annotated(csv_path)
    c.post("/annotations", json={"ppm": 5.0, "label": "lac"}, headers=AUTH)
    body = c.put(
        "/annotations/settings",
        json={"font_size": 22, "line_styles": {"A": {"color": "#112233", "dash": "dot"}, "s1": {"width": 3}}},
        headers=AUTH,
    ).json()
    assert body["font_size"] == 22
    assert body["targets"] == {"groups": ["A", "B"], "samples": [f"s{i}" for i in range(12)]}
    fig = spectra(c)
    s0, s1, s6 = fig["data"][0], fig["data"][1], fig["data"][6]
    assert (s0["line"]["color"], s0["line"]["dash"]) == ("#112233", "dot")
    assert (s1["line"]["width"], s1["line"]["color"]) == (3, "#112233")  # sample style on top of its group's
    assert s6["line"].get("dash") is None  # group B untouched
    assert notes(fig, "lac")[0]["font"]["size"] == 22
    bad = c.put("/annotations/settings", json={"line_styles": {"A": {"dash": "zigzag"}}}, headers=AUTH)
    assert bad.status_code == 422


def test_html_export_is_self_contained(csv_path, tmp_path):
    c = annotated(csv_path)
    c.post("/annotations", json={"ppm": 5.0, "label": "lac"}, headers=AUTH)
    target = tmp_path / "annotated.html"
    r = c.post("/stages/annotate/html", json={"path": str(target), "view": "spectra", "mode": "median"}, headers=AUTH)
    assert r.status_code == 200
    html = target.read_text()
    assert "lac" in html and target.stat().st_size > 1_000_000  # plotly.js is embedded


def test_settings_survive_a_project_roundtrip(run_all, tmp_path):
    s = run_all(("annotate", {}))
    s.annotation_settings.update({"font_size": 18, "line_styles": {"A": {"dash": "dash"}}})
    save_project(s, str(tmp_path / "p.metbit"))
    t = Session([st.id for st in STAGES], {st.id: st.after for st in STAGES})
    load_project(t, STAGES, str(tmp_path / "p.metbit"))
    assert t.annotation_settings == {"font_size": 18, "line_styles": {"A": {"dash": "dash"}}}
    assert BY_ID["annotate"].figures["spectra"](t.output("annotate")).data[0].line.dash == "dash"


def test_label_font_family_and_style(csv_path):
    c = annotated(csv_path)
    c.post("/annotations", json={"ppm": 5.0, "label": "lac"}, headers=AUTH)
    body = c.put(
        "/annotations/settings",
        json={"font_family": "Times New Roman", "font_weight": "bold", "font_style": "italic"},
        headers=AUTH,
    ).json()
    assert (body["font_family"], body["font_weight"], body["font_style"]) == ("Times New Roman", "bold", "italic")
    note = notes(spectra(c), "lac")[0]
    assert note["font"]["family"] == "Times New Roman"
    assert (note["font"]["weight"], note["font"]["style"]) == ("bold", "italic")
    assert note["captureevents"] is True  # a click on the label opens it for editing
    bad = c.put("/annotations/settings", json={"font_weight": "heavy"}, headers=AUTH)
    assert bad.status_code == 422


def test_labels_can_be_edited(csv_path):
    c = annotated(csv_path)
    a = c.post("/annotations", json={"ppm": 5.0, "label": "lac"}, headers=AUTH).json()
    edited = c.patch(f"/annotations/{a['id']}", json={"label": "lactate", "angle": -45}, headers=AUTH).json()
    assert (edited["label"], edited["angle"]) == ("lactate", -45)
    assert notes(spectra(c), "lactate")[0]["textangle"] == -45
    assert c.patch(f"/annotations/{a['id']}", json={"label": "  "}, headers=AUTH).status_code == 422
