# metbit GUI Phase F (Peak Annotation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring metbit's peak annotator into the GUI: label peaks by clicking a spectrum, see the labels on every spectrum view and in exports, manage them in a list with JSON import / export, and export an intensity table.

**Architecture:** `metbit_server/annotations.py` holds label sanitizing (same rules as `metbit.apps.annotate`, reimplemented because that module imports Dash), metbit-compatible JSON parsing, the spectrum overlay and the intensity table. Annotations are a global list on the session, edited through `/annotations` routes (no jobs) and saved in projects. A new `annotate` stage shows single or median spectra; its result shares the session's list, so the intensity table follows label changes without re-running. The renderer adds a click popover, a label list and arrow-color theming. Spec: `GUI/docs/specs/2026-09-29-metbit-gui-style-annotation-stocsy-design.md` (section 2).

**Tech Stack:** FastAPI, Pydantic, Plotly, pandas; Preact, Vitest, Playwright.

## Global Constraints

- All GUI code lives under `GUI/`; Python GUI tests under `tests/gui/`.
- Never use the em dash character or the section sign anywhere.
- Commit messages carry no `Co-Authored-By` trailer.
- Keep every source file under 500 lines.
- Label rules (as metbit): trimmed, at most 256 characters, HTML-escaped (idempotent); empty labels rejected (422).
- CSV guard (as metbit): text starting with `=`, `+`, `-`, `@` or TAB gets a leading `'` in the intensity table.
- Annotation shape: `{"id": str, "ppm": float, "label": str, "angle": float (-180..180, default 0)}`; import also accepts metbit's `x` / `text` / `textangle`.
- Labels overlay the views named `spectra`, `before` and `correlation` only.
- Annotation edits never re-run or stale a stage; the session list is mutated in place (the Annotate result holds a reference).
- Python GUI tests: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`. Renderer: `cd GUI && npm run typecheck && npm test`. E2E: `cd GUI && npm run e2e`.

## File Map

```
GUI/server/metbit_server/annotations.py      NEW
GUI/server/metbit_server/stages/annotate.py  NEW: Annotate stage
GUI/server/metbit_server/stages/__init__.py  MODIFY: register Annotate
GUI/server/metbit_server/session.py          MODIFY: annotations list
GUI/server/metbit_server/project.py          MODIFY: persist annotations
GUI/server/metbit_server/app.py              MODIFY: routes, overlay
tests/gui/test_gui_annotations.py            NEW
tests/gui/test_gui_group_tests.py            MODIFY: registry order
GUI/renderer/api.ts                          MODIFY
GUI/renderer/components/AnnotationPopover.tsx (+ test)  NEW
GUI/renderer/components/AnnotationList.tsx (+ test)     NEW
GUI/renderer/components/PlotView.tsx         MODIFY: click coordinates, arrow color
GUI/renderer/App.tsx                         MODIFY
GUI/renderer/theme.css                       MODIFY
GUI/e2e/annotate.spec.ts                     NEW
```

---

### Task 1: Annotation model, routes, overlay, Annotate stage and persistence

**Files:**
- Create: `GUI/server/metbit_server/annotations.py`, `GUI/server/metbit_server/stages/annotate.py`
- Modify: `GUI/server/metbit_server/stages/__init__.py`, `session.py`, `project.py`, `app.py`, `tests/gui/test_gui_group_tests.py`
- Test: `tests/gui/test_gui_annotations.py`

**Interfaces:**
- Consumes: `Dataset`, `working`, `spectra_figure`, `MAX_PLOT_POINTS`, `DATA_ORDER`, `Stage` (base); `colors` (style); `PathBody`, `effective_style` (app).
- Produces:
  - `annotations.sanitize_label(value) -> str`, `csv_safe(value) -> str`, `AnnotationIn(ppm, label, angle=0)`, `new_annotation(a) -> dict`, `parse_annotation_list(items: list) -> list[dict]`, `parse_annotation_file(text: str) -> list[dict]`, `overlay(fig, annotations, font_size) -> go.Figure`, `intensity_table(X, meta, annotations) -> pd.DataFrame`, `ANNOTATED_VIEWS = {"spectra", "before", "correlation"}`.
  - `Session.annotations: list[dict]`.
  - Stage `annotate` / `Annotate`, section `Statistics`, params `AnnotateParams(mode="single" | "median")`, view `spectra`, table = intensity table, `ppm_views={"spectra": "x"}`.
  - HTTP: `GET /annotations`; `POST /annotations` `{ppm, label, angle?}` -> created item (422 on bad label); `DELETE /annotations/{id}` (404 unknown); `POST /annotations/import` `{path}` -> list (422 unreadable); `POST /annotations/export` `{path}`.
  - Projects save `annotations` and restore them (re-sanitized, new ids) before analyses re-run.

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_annotations.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_annotations.py --no-cov -q -p no:warnings`
Expected: FAIL with `ModuleNotFoundError: No module named 'metbit_server.annotations'`.

- [ ] **Step 3: Annotation module**

Create `GUI/server/metbit_server/annotations.py`:

```python
"""Peak annotations: sanitizing, overlaying on spectra and the intensity table.

Label and CSV rules follow metbit.apps.annotate; they are reimplemented here because
that module imports Dash, which the sidecar does not need.
"""
import html
import json
import uuid
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field, ValidationError, field_validator

MAX_LABEL = 256
ANNOTATED_VIEWS = {"spectra", "before", "correlation"}


def sanitize_label(value: Any) -> str:
    """Trimmed, at most 256 characters, HTML-escaped. Unescaping first makes it idempotent."""
    text = html.unescape(str(value or "")).strip()[:MAX_LABEL].strip()
    return html.escape(text, quote=True)


def csv_safe(value: Any) -> str:
    """Neutralize spreadsheet formula injection, as metbit's intensity export does."""
    text = str(value if value is not None else "")
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t") else text


class AnnotationIn(BaseModel):
    ppm: float
    label: str
    angle: float = Field(0, ge=-180, le=180)

    @field_validator("label")
    @classmethod
    def _clean(cls, v: str) -> str:
        clean = sanitize_label(v)
        if not clean:
            raise ValueError("Label is empty")
        return clean


def new_annotation(a: AnnotationIn) -> dict:
    return {"id": uuid.uuid4().hex[:8], **a.model_dump()}


def parse_annotation_list(items: list) -> list[dict]:
    """Our shape or metbit's annotator JSON (x, text, textangle); invalid entries are skipped."""
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw = {
            "ppm": item.get("ppm", item.get("x")),
            "label": item.get("label", item.get("text")),
            "angle": item.get("angle", item.get("textangle")) or 0,
        }
        try:
            out.append(new_annotation(AnnotationIn.model_validate(raw)))
        except ValidationError:
            continue
    return out


def parse_annotation_file(text: str) -> list[dict]:
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of annotations")
    return parse_annotation_list(data)


def _peak_height(fig: go.Figure, ppm: float) -> Optional[float]:
    """Tallest y among the figure's curves at the point nearest `ppm`."""
    best = None
    for trace in fig.data:
        x, y = getattr(trace, "x", None), getattr(trace, "y", None)
        if x is None or y is None or len(x) == 0:
            continue
        try:
            xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            continue
        value = ys[int(np.nanargmin(np.abs(xs - ppm)))]
        if np.isfinite(value):
            best = value if best is None else max(best, value)
    return best


def overlay(fig: go.Figure, annotations: list[dict], font_size: int) -> go.Figure:
    for a in annotations:
        y = _peak_height(fig, a["ppm"])
        fig.add_annotation(
            x=a["ppm"],
            y=y if y is not None else 1,
            xref="x",
            yref="y" if y is not None else "paper",
            text=a["label"],
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            textangle=a["angle"],
            font={"size": font_size},
        )
    return fig


def intensity_table(X: pd.DataFrame, meta: pd.DataFrame, annotations: list[dict]) -> pd.DataFrame:
    """Metadata, then one column per label with each sample's value at the nearest ppm (metbit's export)."""
    ppm = X.columns.to_numpy(dtype=float)
    columns = {}
    for a in annotations:
        j = int(np.argmin(np.abs(ppm - a["ppm"])))
        columns[f"{csv_safe(html.unescape(a['label']))}_{ppm[j]:.3f}"] = X.iloc[:, j].to_numpy()
    safe_meta = meta.reset_index(drop=True).apply(lambda col: col.map(csv_safe) if col.dtype == object else col)
    return pd.concat([safe_meta, pd.DataFrame(columns)], axis=1)
```

- [ ] **Step 4: Annotate stage and registry**

Create `GUI/server/metbit_server/stages/annotate.py`:

```python
"""Annotate stage: spectra to label peaks on, and the intensity table of the labelled peaks."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from pydantic import BaseModel, Field

from ..annotations import intensity_table
from ..session import Session
from .base import DATA_ORDER, MAX_PLOT_POINTS, Dataset, Progress, Stage, spectra_figure, working
from .style import colors


class AnnotateParams(BaseModel):
    mode: Literal["single", "median"] = Field(
        "single", title="Spectra", description="Single draws every spectrum; median draws one line per group."
    )


@dataclass
class AnnotateResult:
    data: Dataset
    mode: str
    annotations: list  # the session's live list: new labels show up without re-running


def run_annotate(p: AnnotateParams, session: Session, progress: Progress) -> AnnotateResult:
    progress(1.0)
    return AnnotateResult(data=working(session), mode=p.mode, annotations=session.annotations)


def annotate_spectra(r: AnnotateResult, style=None) -> go.Figure:
    if r.mode == "single":
        return spectra_figure(r.data.X, r.data.labels, "Annotated spectra", style)
    X, labels = r.data.X, r.data.labels
    step = max(1, X.shape[1] // MAX_PLOT_POINTS)  # display-only decimation, as other spectrum views
    ppm = X.columns.to_numpy(dtype=float)[::step]
    palette = colors(style) if style else {}
    fig = go.Figure()
    for group in sorted(set(labels)):
        median = np.median(X[(labels == group).to_numpy()].to_numpy(), axis=0)[::step]
        line = {"color": palette.get(group), "width": style.line_width if style else 1.2}
        fig.add_trace(go.Scatter(x=ppm, y=median, name=group, mode="lines", line=line))
    fig.update_layout(
        title="Median spectra",
        template="plotly_white",
        xaxis={"autorange": "reversed", "title": "δ¹H (ppm)"},
        yaxis={"title": "Intensity"},
    )
    return fig


ANNOTATE_STAGES: list[Stage] = [
    Stage(
        id="annotate",
        title="Annotate",
        section="Statistics",
        params=AnnotateParams,
        run=run_annotate,
        table=lambda r: intensity_table(r.data.X, r.data.meta, r.annotations),
        figures={"spectra": annotate_spectra},
        requires="import",
        after=list(DATA_ORDER),
        ppm_views={"spectra": "x"},
    ),
]
```

Then run from the repo root:

```bash
python3 - <<'EOF'
import pathlib


def rep(path, pairs):
    p = pathlib.Path(path)
    s = p.read_text()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:70])
        s = s.replace(a, b)
    p.write_text(s)


G = "GUI/server/metbit_server/"
rep(G + "stages/__init__.py", [
    ("from .analysis import", "from .annotate import ANNOTATE_STAGES, AnnotateParams\nfrom .analysis import"),
    ("    *STATS_STAGES,\n", "    *STATS_STAGES,\n    *ANNOTATE_STAGES,\n"),
])
rep("tests/gui/test_gui_group_tests.py", [('"volcano", "grouptests",', '"volcano", "grouptests", "annotate",')])
rep(G + "session.py", [
    ("        self.stage_styles: dict[str, dict] = {}  # per-stage StyleOverride dicts\n",
     "        self.stage_styles: dict[str, dict] = {}  # per-stage StyleOverride dicts\n"
     "        self.annotations: list[dict] = []        # peak labels; mutate in place (Annotate holds a reference)\n"),
])
rep(G + "project.py", [
    ('        "stage_styles": session.stage_styles,\n',
     '        "stage_styles": session.stage_styles,\n        "annotations": session.annotations,\n'),
    ("    fresh = Session(session.order, session.after)  # swapped in only after everything loads\n",
     "    fresh = Session(session.order, session.after)  # swapped in only after everything loads\n"
     "    # before analyses re-run, so the Annotate result shares this list; re-sanitized like any import\n"
     "    fresh.annotations = parse_annotation_list(manifest.get(\"annotations\") or [])\n"),
    ('    session.style = manifest.get("style") or {}\n',
     '    session.style = manifest.get("style") or {}\n    session.annotations = fresh.annotations\n'),
    ("from .session import Session\n", "from .annotations import parse_annotation_list\nfrom .session import Session\n"),
])
rep(G + "app.py", [
    ("from .jobs import Busy, Job, start_job\n",
     "from .annotations import ANNOTATED_VIEWS, AnnotationIn, new_annotation, overlay, parse_annotation_file\n"
     "from .jobs import Busy, Job, start_job\n"),
    ("        fig = render(make, out, effective_style(sid))\n",
     "        style = effective_style(sid)\n"
     "        fig = render(make, out, style)\n"
     "        if view in ANNOTATED_VIEWS and session.annotations:\n"
     "            overlay(fig, session.annotations, style.font_size)\n"),
    ("    def autosave_now() -> None:",
     '''    @app.get("/annotations")
    def list_annotations():
        return session.annotations

    @app.post("/annotations")
    def add_annotation(body: dict = Body(...)):
        try:
            item = new_annotation(AnnotationIn.model_validate(body))
        except ValidationError as e:
            raise HTTPException(422, json.loads(e.json(include_url=False)))
        session.annotations.append(item)
        return item

    @app.delete("/annotations/{aid}")
    def delete_annotation(aid: str):
        for i, a in enumerate(session.annotations):
            if a["id"] == aid:
                del session.annotations[i]
                return {"deleted": aid}
        raise HTTPException(404, "Unknown annotation")

    @app.post("/annotations/import")
    def import_annotations(body: PathBody):
        try:
            items = parse_annotation_file(Path(body.path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise HTTPException(422, f"Cannot read annotations: {e}")
        session.annotations[:] = items  # in place: the Annotate result shares this list
        return session.annotations

    @app.post("/annotations/export")
    def export_annotations(body: PathBody):
        Path(body.path).write_text(json.dumps(session.annotations, indent=2), encoding="utf-8")
        return {"path": body.path}

    def autosave_now() -> None:'''),
])
print("annotations wired")
EOF
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`
Expected: `113 passed` (111 passed and 2 skipped without MTBLS2052).

- [ ] **Step 6: Commit**

```bash
git add GUI/server tests/gui
git commit -m "feat(gui): peak annotations with overlay, Annotate stage and intensity table"
```

---

### Task 2: Popover, label list and app wiring

**Files:**
- Create: `GUI/renderer/components/AnnotationPopover.tsx`, `AnnotationPopover.test.tsx`, `AnnotationList.tsx`, `AnnotationList.test.tsx`
- Modify: `GUI/renderer/api.ts`, `GUI/renderer/components/PlotView.tsx`, `GUI/renderer/App.tsx`, `GUI/renderer/theme.css`

**Interfaces:**
- Consumes: Task 1 routes.
- Produces:
  - `api.ts`: `Annotation { id; ppm; label; angle }`; `Api.listAnnotations()`, `addAnnotation(ppm, label, angle)`, `deleteAnnotation(id)`, `importAnnotations(path)`, `exportAnnotations(path)`.
  - `AnnotationPopover({ ppm, x, y, onAdd(label, angle), onCancel })`: fixed-position form at the click; inputs labelled `Label` and `Angle`; Enter adds, Escape cancels, Add disabled while the label is blank.
  - `AnnotationList({ items, onDelete(id), onImport, onExport })`: decoded labels with ppm, `Delete` buttons, `Import labels…` / `Export labels…` (export disabled when empty).
  - `PickPoint` gains `clientX?` / `clientY?`; annotation arrows follow the theme text color.

- [ ] **Step 1: Write the failing tests**

Create `GUI/renderer/components/AnnotationPopover.test.tsx`:

```tsx
import { render } from "preact";
import { expect, it, vi } from "vitest";
import { AnnotationPopover } from "./AnnotationPopover";

const tick = () => new Promise((r) => setTimeout(r));

function mount() {
  const root = document.createElement("div");
  const onAdd = vi.fn();
  const onCancel = vi.fn();
  render(<AnnotationPopover ppm={1.3312} x={100} y={200} onAdd={onAdd} onCancel={onCancel} />, root);
  return { root, onAdd, onCancel };
}

const type = (el: HTMLInputElement, value: string) => {
  el.value = value;
  el.dispatchEvent(new Event("input"));
};

it("adds the typed label and angle", async () => {
  const { root, onAdd } = mount();
  expect(root.textContent).toContain("1.331 ppm");
  type(root.querySelector<HTMLInputElement>('[aria-label="Label"]')!, "  lactate ");
  type(root.querySelector<HTMLInputElement>('[aria-label="Angle"]')!, "15");
  await tick();
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  expect(onAdd).toHaveBeenCalledWith("lactate", 15);
});

it("disables Add while the label is blank", () => {
  const { root } = mount();
  expect([...root.querySelectorAll("button")].find((b) => b.textContent === "Add")!.disabled).toBe(true);
});

it("cancels on Escape", () => {
  const { root, onCancel } = mount();
  root.querySelector("form")!.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  expect(onCancel).toHaveBeenCalledOnce();
});
```

Create `GUI/renderer/components/AnnotationList.test.tsx`:

```tsx
import { render } from "preact";
import { expect, it, vi } from "vitest";
import type { Annotation } from "../api";
import { AnnotationList } from "./AnnotationList";

function mount(items: Annotation[]) {
  const root = document.createElement("div");
  const handlers = { onDelete: vi.fn(), onImport: vi.fn(), onExport: vi.fn() };
  render(<AnnotationList items={items} {...handlers} />, root);
  return { root, ...handlers };
}

const button = (root: HTMLElement, text: string) =>
  [...root.querySelectorAll("button")].find((b) => b.textContent === text) as HTMLButtonElement;

it("lists decoded labels with their ppm and deletes by id", () => {
  const { root, onDelete } = mount([{ id: "a1", ppm: 1.3312, label: "&lt;b&gt;Lac", angle: 0 }]);
  expect(root.querySelector(".label-list li")!.textContent).toContain("<b>Lac");
  expect(root.textContent).toContain("1.331 ppm");
  button(root, "Delete").click();
  expect(onDelete).toHaveBeenCalledWith("a1");
});

it("explains how to add labels when empty and disables export", () => {
  const { root } = mount([]);
  expect(root.textContent).toContain("Click a peak on the spectra to add a label.");
  expect(button(root, "Export labels…").disabled).toBe(true);
});

it("offers import and export", () => {
  const { root, onImport, onExport } = mount([{ id: "a1", ppm: 1, label: "x", angle: 0 }]);
  button(root, "Import labels…").click();
  button(root, "Export labels…").click();
  expect(onImport).toHaveBeenCalledOnce();
  expect(onExport).toHaveBeenCalledOnce();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd GUI && npx vitest run renderer/components/AnnotationPopover.test.tsx renderer/components/AnnotationList.test.tsx`
Expected: FAIL, cannot resolve the components.

- [ ] **Step 3: Components**

Create `GUI/renderer/components/AnnotationPopover.tsx`:

```tsx
import { useState } from "preact/hooks";

interface Props {
  ppm: number;
  x: number; // viewport coordinates of the click
  y: number;
  onAdd: (label: string, angle: number) => void;
  onCancel: () => void;
}

export function AnnotationPopover({ ppm, x, y, onAdd, onCancel }: Props) {
  const [label, setLabel] = useState("");
  const [angle, setAngle] = useState(0);
  return (
    <form
      class="popover"
      style={{ left: `${x}px`, top: `${y}px` }}
      onSubmit={(e) => {
        e.preventDefault();
        if (label.trim()) onAdd(label.trim(), angle);
      }}
      onKeyDown={(e) => {
        if (e.key === "Escape") onCancel();
      }}
    >
      <span class="popover-title">Label at {ppm.toFixed(3)} ppm</span>
      <input aria-label="Label" autoFocus placeholder="e.g. lactate" value={label} onInput={(e) => setLabel(e.currentTarget.value)} />
      <input
        aria-label="Angle"
        type="number"
        min={-180}
        max={180}
        step={5}
        value={angle}
        onInput={(e) => setAngle(Number(e.currentTarget.value) || 0)}
      />
      <div class="row">
        <button class="btn primary" type="submit" disabled={!label.trim()}>
          Add
        </button>
        <button class="btn" type="button" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}
```

Create `GUI/renderer/components/AnnotationList.tsx`:

```tsx
import type { Annotation } from "../api";

// labels are stored HTML-escaped (for Plotly); show the plain text
const decode = (s: string) => new DOMParser().parseFromString(s, "text/html").documentElement.textContent ?? s;

interface Props {
  items: Annotation[];
  onDelete: (id: string) => void;
  onImport: () => void;
  onExport: () => void;
}

export function AnnotationList({ items, onDelete, onImport, onExport }: Props) {
  return (
    <div class="annotations">
      <h3>Labels</h3>
      {items.length === 0 ? (
        <p class="hint">Click a peak on the spectra to add a label.</p>
      ) : (
        <ul class="label-list">
          {items.map((a) => (
            <li key={a.id}>
              <span>{decode(a.label)}</span>
              <span class="ppm-tag">{a.ppm.toFixed(3)} ppm</span>
              <button type="button" class="link" onClick={() => onDelete(a.id)}>
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
      <div class="row">
        <button type="button" class="link" onClick={onImport}>
          Import labels…
        </button>
        <button type="button" class="link" onClick={onExport} disabled={items.length === 0}>
          Export labels…
        </button>
      </div>
    </div>
  );
}
```

Append to `GUI/renderer/theme.css`:

```css
.popover {
  position: fixed;
  z-index: 10;
  transform: translate(-50%, calc(-100% - 14px));
  width: 220px;
  display: flex;
  flex-direction: column;
  gap: var(--s1);
  padding: 12px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
}
.popover-title { font-size: 12px; color: var(--muted); }
.popover input { font: inherit; color: inherit; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 6px var(--s1); }
.popover input:focus { outline: none; border-color: var(--accent); }
.annotations { border-top: 1px solid var(--border); padding-top: var(--s2); display: flex; flex-direction: column; gap: var(--s1); }
.annotations h3 { margin: 0; font-size: 12px; font-weight: 600; }
.label-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px; }
.label-list li { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: var(--s1); align-items: center; }
.label-list li span:first-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ppm-tag { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.link:disabled { opacity: 0.4; cursor: default; }
```

Run: `cd GUI && npx vitest run renderer/components/AnnotationPopover.test.tsx renderer/components/AnnotationList.test.tsx`
Expected: `6 passed`.

- [ ] **Step 4: API, PlotView and App wiring**

Run from the repo root:

```bash
python3 - <<'EOF'
import pathlib


def rep(path, pairs):
    p = pathlib.Path(path)
    s = p.read_text()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:70])
        s = s.replace(a, b)
    p.write_text(s)


R = "GUI/renderer/"
rep(R + "api.ts", [
    ("export interface StageStyle {",
     "export interface Annotation {\n  id: string;\n  ppm: number;\n  label: string;\n  angle: number;\n}\n\n"
     "export interface StageStyle {"),
    ("  getStyle(): Promise<PlotStyle> {",
     '''  listAnnotations(): Promise<Annotation[]> {
    return this.req("/annotations").then((r) => r.json());
  }

  addAnnotation(ppm: number, label: string, angle: number): Promise<Annotation> {
    return this.post("/annotations", { ppm, label, angle });
  }

  deleteAnnotation(id: string): Promise<unknown> {
    return this.req(`/annotations/${id}`, { method: "DELETE" }).then((r) => r.json());
  }

  importAnnotations(path: string): Promise<Annotation[]> {
    return this.post("/annotations/import", { path });
  }

  exportAnnotations(path: string): Promise<unknown> {
    return this.post("/annotations/export", { path });
  }

  getStyle(): Promise<PlotStyle> {'''),
])
rep(R + "components/PlotView.tsx", [
    ("export interface PickPoint {\n  x: unknown;\n  customdata: unknown;\n}",
     "export interface PickPoint {\n  x: unknown;\n  customdata: unknown;\n  clientX?: number; // viewport position of the click, for popovers\n  clientY?: number;\n}"),
    ("  if (layout.annotations) out.annotations = layout.annotations.map((a: any) => ({ ...a, font: { ...a.font, color: text } }));",
     "  if (layout.annotations) {\n"
     "    out.annotations = layout.annotations.map((a: any) => ({ ...a, arrowcolor: text, font: { ...a.font, color: text } }));\n"
     "  }"),
    ('node.on("plotly_click", (e: { points?: PickPoint[] }) => {',
     'node.on("plotly_click", (e: { points?: PickPoint[]; event?: MouseEvent }) => {'),
    ("if (p) pick.current?.({ x: p.x, customdata: p.customdata });",
     "if (p) pick.current?.({ x: p.x, customdata: p.customdata, clientX: e.event?.clientX, clientY: e.event?.clientY });"),
])
rep(R + "App.tsx", [
    ("type PlotStyle,", "type Annotation, type PlotStyle,"),
    ('import { Rail } from "./components/Rail";\n',
     'import { AnnotationList } from "./components/AnnotationList";\n'
     'import { AnnotationPopover } from "./components/AnnotationPopover";\n'
     'import { Rail } from "./components/Rail";\n'),
    ("  const [busy, setBusy] = useState<string | null>(null); // caption while a project opens or a session restores\n",
     "  const [busy, setBusy] = useState<string | null>(null); // caption while a project opens or a session restores\n"
     "  const [annotations, setAnnotations] = useState<Annotation[]>([]);\n"
     "  const [pending, setPending] = useState<{ ppm: number; x: number; y: number } | null>(null);\n"),
    ('''        if (Number.isFinite(v)) setPicked(v);
      }''',
     '''        if (!Number.isFinite(v)) return;
        setPicked(v);
        // on the Annotate spectra a click opens the label popover at the click
        if (current === "annotate" && tab === "spectra" && pt.clientX !== undefined && pt.clientY !== undefined) {
          setPending({ ppm: v, x: pt.clientX, y: pt.clientY });
        }
      }'''),
    ("  useEffect(() => {\n    api?.stages().then(setStages);\n  }, [api]);\n",
     "  useEffect(() => {\n    api?.stages().then(setStages);\n    api?.listAnnotations().then(setAnnotations);\n  }, [api]);\n"),
    ("    setStageStyle(null); // the next stage's style loads fresh\n",
     "    setStageStyle(null); // the next stage's style loads fresh\n    setPending(null);\n"),
    ("  const openStyle = () =>",
     '''  async function annotationsChanged(client: Api) {
    setAnnotations(await client.listAnnotations());
    setVersion((v) => v + 1); // labels are drawn server-side and feed the intensity table
  }

  const addAnnotation = (label: string, angle: number) =>
    guarded(async () => {
      if (!api || !pending) return;
      await api.addAnnotation(pending.ppm, label, angle);
      setPending(null);
      await annotationsChanged(api);
    });

  const deleteAnnotation = (id: string) =>
    guarded(async () => {
      if (!api) return;
      await api.deleteAnnotation(id);
      await annotationsChanged(api);
    });

  const importAnnotations = () =>
    guarded(async () => {
      const path = await window.metbit.openFile([{ name: "JSON", extensions: ["json"] }]);
      if (!path || !api) return;
      await api.importAnnotations(path);
      await annotationsChanged(api);
      flash(`Imported ${basename(path)}`);
    });

  const exportAnnotations = () =>
    guarded(async () => {
      const path = await window.metbit.saveDialog("annotations.json", [{ name: "JSON", extensions: ["json"] }]);
      if (!path || !api) return;
      await api.exportAnnotations(path);
      flash(`Saved ${basename(path)}`);
    });

  const openStyle = () =>'''),
    ("        {stageStyle && (",
     '''        {current === "annotate" && (
          <AnnotationList
            items={annotations}
            onDelete={deleteAnnotation}
            onImport={importAnnotations}
            onExport={exportAnnotations}
          />
        )}
        {stageStyle && ('''),
    ("      {backendDown && (",
     '''      {pending && (
        <AnnotationPopover ppm={pending.ppm} x={pending.x} y={pending.y} onAdd={addAnnotation} onCancel={() => setPending(null)} />
      )}
      {backendDown && ('''),
]
)
print("renderer wired")
EOF
```

- [ ] **Step 5: Verify**

Run: `cd GUI && npm run typecheck && npm test && npm run build`
Expected: typecheck exits 0; `40 passed`; build succeeds.

- [ ] **Step 6: Commit**

```bash
git add GUI/renderer
git commit -m "feat(gui): label peaks with a click popover and manage labels in the Annotate stage"
```

---

### Task 3: End-to-end annotation

**Files:**
- Create: `GUI/e2e/annotate.spec.ts`

**Interfaces:**
- Consumes: rail titles `Annotate`, `Normalize`; popover inputs `Label`; `.label-list`; tab `Table`; `gd.layout.annotations` on `.plot`.
- Produces: `npm run e2e` runs 9 tests.

- [ ] **Step 1: Write the test**

Create `GUI/e2e/annotate.spec.ts`:

```ts
import { _electron as electron, expect, test, type Page } from "@playwright/test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function makeCsv(dir: string): string {
  const ppm = Array.from({ length: 40 }, (_, i) => (10 - i * 0.25).toFixed(4));
  const rows = Array.from({ length: 12 }, (_, i) => {
    const group = i < 6 ? "A" : "B";
    const values = ppm.map((_, j) => (Math.abs(Math.sin(i * 7 + j)) + 1 + (group === "B" && j === 5 ? 3 : 0)).toFixed(5));
    return [`s${i}`, group, ...values].join(",");
  });
  const file = join(dir, "data.csv");
  writeFileSync(file, [["Sample", "Group", ...ppm].join(","), ...rows].join("\n"));
  return file;
}

const plotLabels = (win: Page) =>
  win.evaluate(() => ((document.querySelector(".plot") as any)?.layout?.annotations ?? []).map((a: any) => a.text));

test("label a peak, see it on other spectra and in the intensity table", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const env = { ...process.env, METBIT_USER_DATA: join(dir, "user") } as Record<string, string>;
  const app = await electron.launch({ args: ["."], env });
  const win = await app.firstWindow();
  const run = () => win.getByRole("button", { name: "Run", exact: true }).click();
  const plot = () => expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

  await win.fill("#f-path", makeCsv(dir));
  await run();
  await expect(win.locator("table.data")).toContainText("s11");

  await win.locator(".rail button", { hasText: "Annotate" }).click();
  await run();
  await plot();
  const box = (await win.locator(".plot .nsewdrag").first().boundingBox())!;
  await win.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.3);
  await win.mouse.click(box.x + box.width * 0.2, box.y + box.height * 0.3);
  await win.getByLabel("Label", { exact: true }).fill("lactate");
  await win.getByLabel("Label", { exact: true }).press("Enter");

  await expect(win.locator(".label-list")).toContainText("lactate");
  await expect.poll(() => plotLabels(win)).toContain("lactate");
  await win.getByRole("tab", { name: "Table" }).click();
  await expect(win.locator("table.data thead")).toContainText("lactate_");

  await win.locator(".rail button", { hasText: "Normalize" }).click();
  await run();
  await plot();
  await expect.poll(() => plotLabels(win)).toContain("lactate");

  await app.close();
});
```

- [ ] **Step 2: Run everything**

Run from the repo root:
```bash
python3 -m pytest -q -p no:cacheprovider && (cd GUI && npm run typecheck && npm test && npm run e2e)
```
Expected: repo suite passes with coverage at or above 99%; typecheck exits 0; `40 passed`; `9 passed`.

- [ ] **Step 3: Commit**

```bash
git add GUI/e2e/annotate.spec.ts
git commit -m "test(gui): cover peak annotation end to end"
```
