# metbit GUI Phase I (Annotator Dashboard UX) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Annotate stage work like metbit's `annotate_peak` Dash dashboard: a toolbar (Single / Median switch, label font size, current peak), collapsible panels (Add annotation, Style line, Current labels), exports (intensity CSV, labels JSON, self-contained HTML), and draggable labels.

**Architecture:** Server: annotations carry `ax` / `ay` offsets and are patchable; a live `annotation_settings` dict (label font size, per-line styles keyed by group or sample) is shared by the session and the Annotate result; figure requests accept `mode` (forwarded like `ppm`); a `build_figure` helper serves both the JSON route and a new HTML export route. Single mode draws one trace per sample (named by sample, grouped by group) so lines can be styled individually. Renderer: `AnnotateToolbar` above the plot, `AnnotatePanels` in the right column replacing the click popover, and `PlotView` label dragging via Plotly `edits.annotationTail`. Spec: `docs/superpowers/specs/2026-09-29-metbit-gui-style-annotation-stocsy-design.md` (section 7).

**Tech Stack:** FastAPI, Pydantic, Plotly; Preact, Vitest, Playwright.

## Global Constraints

- All GUI code lives under `GUI/`; Python GUI tests under `tests/gui/`.
- Never use the em dash character or the section sign anywhere.
- Commit messages carry no `Co-Authored-By` trailer.
- Keep every source file under 500 lines.
- Label offsets: `ax` default 0, `ay` default -40 (pixels, as metbit), each within -600..600.
- Label font size 6..48 (default: the plot style font size). Line style: `color` `#rrggbb`, `dash` one of `solid`, `dash`, `dot`, `dashdot`, `width` 0.1..10; a sample's style is applied on top of its group's.
- Mode switch and styling never re-run the stage; settings are mutated in place (the Annotate result holds a reference).
- HTML export embeds plotly.js (works offline).
- Python GUI tests: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`. Renderer: `cd GUI && npm run typecheck && npm test`. E2E: `cd GUI && npm run e2e`.

---

### Task 1: Server: offsets, settings, per-sample traces, mode, HTML export

**Files:**
- Modify: `GUI/server/metbit_server/annotations.py`, `session.py`, `project.py`, `app.py`
- Modify (replace): `GUI/server/metbit_server/stages/annotate.py`
- Test: `tests/gui/test_gui_annotator.py`

**Interfaces:**
- Produces:
  - Annotation dicts gain `ax`, `ay`; `AnnotationPatch(label?, angle?, ax?, ay?)`; `overlay()` uses `ax` / `ay` and sets each label's Plotly `name` to its id.
  - `LineStyle`, `AnnotatorSettings(font_size?, line_styles)`, `settings_dict(s) -> dict`, `apply_line_styles(fig, line_styles)`.
  - `Session.annotation_settings: dict`.
  - `annotate.sample_names(d) -> list[str]`, `annotate.sample_spectra_figure(d, title, style=None)`, `annotate_spectra(r, style=None, mode=None)`; `AnnotateResult.settings`.
  - HTTP: `PATCH /annotations/{id}`; `GET /annotations/settings` -> settings plus `targets: {groups, samples}`; `PUT /annotations/settings`; `GET /stages/{id}/figure?...&mode=`; `POST /stages/{id}/html` `{path, view, ppm?, mode?}`.
  - Projects save and restore `annotation_settings`.

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_annotator.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_annotator.py --no-cov -q -p no:warnings`
Expected: 5 failed.

- [ ] **Step 3: Annotate stage**

Replace `GUI/server/metbit_server/stages/annotate.py` with:

```python
"""Annotate stage: spectra to label peaks on, and the intensity table of the labelled peaks."""
from dataclasses import dataclass, field
from typing import Literal, Optional

import plotly.graph_objects as go
from pydantic import BaseModel, Field

from ..annotations import apply_line_styles, intensity_table
from ..session import Session
from .base import DATA_ORDER, MAX_PLOT_POINTS, Dataset, Progress, Stage, median_spectra_figure, working
from .style import colors

MODES = ("single", "median")


class AnnotateParams(BaseModel):
    mode: Literal["single", "median"] = Field(
        "single", title="Spectra", description="Starting view; the toolbar switches between single and median instantly."
    )


@dataclass
class AnnotateResult:
    data: Dataset
    mode: str
    annotations: list  # the session's live list: new labels show up without re-running
    settings: dict = field(default_factory=dict)  # the session's live annotator settings (line styles)


def run_annotate(p: AnnotateParams, session: Session, progress: Progress) -> AnnotateResult:
    progress(1.0)
    return AnnotateResult(
        data=working(session), mode=p.mode, annotations=session.annotations, settings=session.annotation_settings
    )


def sample_names(d) -> list[str]:
    """First metadata column (sample IDs such as Sample or fid) when there is one, else the row index."""
    if d.meta.shape[1]:
        return [str(v) for v in d.meta.iloc[:, 0]]
    return [str(v) for v in d.X.index]


def sample_spectra_figure(d: Dataset, title: str, style=None) -> go.Figure:
    """One trace per sample, named by sample and grouped by group in the legend, as metbit's annotator draws."""
    X = d.X
    step = max(1, X.shape[1] // MAX_PLOT_POINTS)  # display-only decimation, as other spectrum views
    ppm = X.columns.to_numpy(dtype=float)[::step]
    palette = colors(style) if style else {}
    width = style.line_width if style else 1.2
    fig = go.Figure()
    titled: set = set()
    for i, (name, group) in enumerate(zip(sample_names(d), d.labels)):
        fig.add_trace(
            go.Scatter(
                x=ppm,
                y=X.iloc[i, ::step].to_numpy(),
                name=name,
                mode="lines",
                legendgroup=group,
                legendgrouptitle_text=None if group in titled else group,
                line={"color": palette.get(group), "width": width},
            )
        )
        titled.add(group)
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis={"autorange": "reversed", "title": "δ¹H (ppm)"},
        yaxis={"title": "Intensity"},
    )
    return fig


def annotate_spectra(r: AnnotateResult, style=None, mode: Optional[str] = None) -> go.Figure:
    mode = mode if mode in MODES else r.mode
    if mode == "median":
        fig = median_spectra_figure(r.data.X, r.data.labels, "Median spectra", style)
    else:
        fig = sample_spectra_figure(r.data, "Spectra", style)
    return apply_line_styles(fig, r.settings.get("line_styles", {}))


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

- [ ] **Step 4: Annotations, session, project and routes**

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


S = "GUI/server/metbit_server/"
rep(S + "annotations.py", [
    ("from typing import Any, Optional\n", "from typing import Any, Literal, Optional\n"),
    ('''    angle: float = Field(0, ge=-180, le=180)

    @field_validator("label")''',
     '''    angle: float = Field(0, ge=-180, le=180)
    ax: float = Field(0, ge=-600, le=600)  # label offset from its peak, in pixels (metbit's default)
    ay: float = Field(-40, ge=-600, le=600)

    @field_validator("label")'''),
    ('''            "angle": item.get("angle", item.get("textangle")) or 0,
''',
     '''            "angle": item.get("angle", item.get("textangle")) or 0,
            "ax": item.get("ax", 0),
            "ay": item.get("ay", -40),
'''),
    ('''            ax=0,
            ay=-40,
            textangle=a["angle"],''',
     '''            ax=a.get("ax", 0),
            ay=a.get("ay", -40),
            name=a["id"],  # lets the renderer map a dragged label back to its annotation
            textangle=a["angle"],'''),
])
p = pathlib.Path(S + "annotations.py")
p.write_text(p.read_text().rstrip("\n") + '''


Dash = Literal["solid", "dash", "dot", "dashdot"]


class AnnotationPatch(BaseModel):
    label: Optional[str] = None
    angle: Optional[float] = Field(None, ge=-180, le=180)
    ax: Optional[float] = Field(None, ge=-600, le=600)
    ay: Optional[float] = Field(None, ge=-600, le=600)

    @field_validator("label")
    @classmethod
    def _clean(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        clean = sanitize_label(v)
        if not clean:
            raise ValueError("Label is empty")
        return clean


class LineStyle(BaseModel):
    color: Optional[str] = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")
    dash: Optional[Dash] = None
    width: Optional[float] = Field(None, ge=0.1, le=10)


class AnnotatorSettings(BaseModel):
    font_size: Optional[int] = Field(None, ge=6, le=48)
    line_styles: dict[str, LineStyle] = {}


def settings_dict(s: AnnotatorSettings) -> dict:
    d = s.model_dump(exclude_none=True)
    d["line_styles"] = {k: v for k, v in d.get("line_styles", {}).items() if v}
    return d


def apply_line_styles(fig: go.Figure, line_styles: dict) -> go.Figure:
    """Per-line overrides like metbit's annotator: a sample's own style goes on top of its group's."""
    for trace in fig.data:
        if trace.type not in ("scatter", "scattergl"):
            continue
        style = {**line_styles.get(str(trace.legendgroup or ""), {}), **line_styles.get(str(trace.name), {})}
        if style.get("color"):
            trace.line.color = style["color"]
        if style.get("dash"):
            trace.line.dash = style["dash"]
        if style.get("width"):
            trace.line.width = style["width"]
    return fig
''')
rep(S + "session.py", [
    ("        self.annotations: list[dict] = []        # peak labels; mutate in place (Annotate holds a reference)\n",
     "        self.annotations: list[dict] = []        # peak labels; mutate in place (Annotate holds a reference)\n"
     "        self.annotation_settings: dict = {}      # label font size and line styles; mutate in place too\n"),
])
rep(S + "project.py", [
    ("from .annotations import parse_annotation_list\n",
     "from pydantic import ValidationError\n\nfrom .annotations import AnnotatorSettings, parse_annotation_list, settings_dict\n"),
    ('        "annotations": session.annotations,\n',
     '        "annotations": session.annotations,\n        "annotation_settings": session.annotation_settings,\n'),
    ('    fresh.annotations = parse_annotation_list(manifest.get("annotations") or [])\n',
     '    fresh.annotations = parse_annotation_list(manifest.get("annotations") or [])\n'
     '    try:\n'
     '        fresh.annotation_settings = settings_dict(AnnotatorSettings.model_validate(manifest.get("annotation_settings") or {}))\n'
     '    except ValidationError:\n'
     '        fresh.annotation_settings = {}\n'),
    ("    session.annotations = fresh.annotations\n",
     "    session.annotations = fresh.annotations\n    session.annotation_settings = fresh.annotation_settings\n"),
])
app = pathlib.Path(S + "app.py")
s = app.read_text()
a = "from .annotations import ANNOTATED_VIEWS, AnnotationIn, new_annotation, overlay, parse_annotation_file\n"
b = ("from .annotations import (\n    ANNOTATED_VIEWS,\n    AnnotationIn,\n    AnnotationPatch,\n    AnnotatorSettings,\n"
     "    new_annotation,\n    overlay,\n    parse_annotation_file,\n    settings_dict,\n)\n"
     "from .stages.annotate import sample_names\n")
assert s.count(a) == 1; s = s.replace(a, b)
a = "def create_app("
b = '''class HtmlBody(PathBody):
    view: str
    ppm: Optional[float] = None
    mode: Optional[str] = None


def create_app('''
assert s.count(a) == 1; s = s.replace(a, b)
start = s.index('    @app.get("/stages/{sid}/figure")')
end = s.index('        return Response(fig.to_json(), media_type="application/json")\n', start) + len('        return Response(fig.to_json(), media_type="application/json")\n')
s = s[:start] + '''    def build_figure(sid: str, view: str, ppm: Optional[float] = None, mode: Optional[str] = None):
        stage = stage_or_404(sid)
        style = effective_style(sid)
        if view in stage.pre_views:
            try:
                data = working(session)  # pre-run views draw from the working matrix, no result needed
            except ValueError as e:
                raise HTTPException(409, str(e))
            fig = render(stage.pre_views[view], data, style, ppm=ppm, mode=mode)
        else:
            out = result_or_409(sid)
            make = stage.figures.get(view)
            if make is None:
                raise HTTPException(404, f"Unknown view '{view}'")
            fig = render(make, out, style, ppm=ppm, mode=mode)
        if view in ANNOTATED_VIEWS and session.annotations:
            overlay(fig, session.annotations, session.annotation_settings.get("font_size") or style.font_size)
        fig.update_layout(width=None, height=None, autosize=True)  # renderer sizes the plot
        return fig

    @app.get("/stages/{sid}/figure")
    def figure(sid: str, view: str, ppm: Optional[float] = None, mode: Optional[str] = None):
        return Response(build_figure(sid, view, ppm, mode).to_json(), media_type="application/json")

    @app.post("/stages/{sid}/html")
    def export_html(sid: str, body: HtmlBody):
        build_figure(sid, body.view, body.ppm, body.mode).write_html(body.path, include_plotlyjs=True, full_html=True)
        return {"path": body.path}  # plotly.js embedded: the file works offline
''' + s[end:]
a = "    def autosave_now() -> None:"
b = '''    @app.patch("/annotations/{aid}")
    def patch_annotation(aid: str, body: dict = Body(...)):
        try:
            patch = AnnotationPatch.model_validate(body)
        except ValidationError as e:
            raise HTTPException(422, json.loads(e.json(include_url=False)))
        for a in session.annotations:
            if a["id"] == aid:
                a.update(patch.model_dump(exclude_none=True))
                return a
        raise HTTPException(404, "Unknown annotation")

    def annotation_targets() -> dict:
        try:
            data = working(session)  # a Bruker import has no matrix until Process runs
        except ValueError:
            return {"groups": [], "samples": []}
        return {"groups": sorted(set(data.labels)), "samples": sample_names(data)}

    @app.get("/annotations/settings")
    def get_annotation_settings():
        return {**session.annotation_settings, "targets": annotation_targets()}

    @app.put("/annotations/settings")
    def put_annotation_settings(body: dict = Body(...)):
        try:
            settings = AnnotatorSettings.model_validate({k: v for k, v in body.items() if k != "targets"})
        except ValidationError as e:
            raise HTTPException(422, json.loads(e.json(include_url=False)))
        session.annotation_settings.clear()  # in place: the Annotate result shares this dict
        session.annotation_settings.update(settings_dict(settings))
        return get_annotation_settings()

    def autosave_now() -> None:'''
assert s.count(a) == 1; s = s.replace(a, b)
app.write_text(s)
print("server wired")
EOF
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`
Expected: `134 passed` (132 passed and 2 skipped without MTBLS2052).

- [ ] **Step 6: Commit**

```bash
git add GUI/server tests/gui/test_gui_annotator.py
git commit -m "feat(gui): draggable label offsets, annotator settings, per-sample spectra, mode switch and HTML export"
```

---

### Task 2: Toolbar and dashboard panels

**Files:**
- Create: `GUI/renderer/components/AnnotateToolbar.tsx`, `AnnotateToolbar.test.tsx`, `AnnotatePanels.tsx`, `AnnotatePanels.test.tsx`
- Modify: `GUI/renderer/components/AnnotationList.tsx` (no own heading; it sits inside a panel)
- Delete: `GUI/renderer/components/AnnotationPopover.tsx`, `AnnotationPopover.test.tsx`

**Interfaces:**
- Produces:
  - `AnnotateToolbar({ mode: "single" | "median", onMode, fontSize, onFontSize, peak: number | null })`: `role="toolbar"`; buttons `Single spectra` / `Median spectra` with `aria-pressed`; range input labelled `Font size` (6..48, reports on release); `.peak-pill` text `Current peak: <ppm to 3 dp> ppm` or `Current peak: none`.
  - `LineStyleValue { color?: string; dash?: string; width?: number }`.
  - `AnnotatePanels({ peak, onAdd(label, angle), targets: {groups, samples}, lineStyles, onApplyStyle(target, style), onResetStyle(target), exports, children })`: `details` panels `📍 Add annotation` (label input `#annotate-label`, angle, `Add Label` disabled until a peak and a label), `🎨 Style line` (`Line to style` select with Groups / Samples, `Line color`, `Line dash`, `Line width`, `Apply Style`, `Reset` when styled), `📌 Current labels` (children), then the `exports` row.

- [ ] **Step 1: Write the failing tests**

Create `GUI/renderer/components/AnnotateToolbar.test.tsx`:

```tsx
import { render } from "preact";
import { expect, it, vi } from "vitest";
import { AnnotateToolbar } from "./AnnotateToolbar";

function mount(peak: number | null = 3.0251) {
  const root = document.createElement("div");
  const onMode = vi.fn();
  const onFontSize = vi.fn();
  render(<AnnotateToolbar mode="single" onMode={onMode} fontSize={14} onFontSize={onFontSize} peak={peak} />, root);
  return { root, onMode, onFontSize };
}

it("switches mode and shows the current peak", () => {
  const { root, onMode } = mount();
  const button = (t: string) => [...root.querySelectorAll("button")].find((b) => b.textContent === t)!;
  expect(button("Single spectra").getAttribute("aria-pressed")).toBe("true");
  button("Median spectra").click();
  expect(onMode).toHaveBeenCalledWith("median");
  expect(root.querySelector(".peak-pill")!.textContent).toBe("Current peak: 3.025 ppm");
  expect(mount(null).root.querySelector(".peak-pill")!.textContent).toBe("Current peak: none");
});

it("reports the label font size when the slider is released", () => {
  const { root, onFontSize } = mount();
  const slider = root.querySelector<HTMLInputElement>('input[type="range"]')!;
  slider.value = "22";
  slider.dispatchEvent(new Event("change"));
  expect(onFontSize).toHaveBeenCalledWith(22);
});
```

Create `GUI/renderer/components/AnnotatePanels.test.tsx`:

```tsx
import { render } from "preact";
import { expect, it, vi } from "vitest";
import { AnnotatePanels } from "./AnnotatePanels";

const tick = () => new Promise((r) => setTimeout(r));

function mount(peak: number | null, lineStyles = {}) {
  const root = document.createElement("div");
  const handlers = { onAdd: vi.fn(), onApplyStyle: vi.fn(), onResetStyle: vi.fn() };
  render(
    <AnnotatePanels
      peak={peak}
      targets={{ groups: ["A", "B"], samples: ["s0", "s1"] }}
      lineStyles={lineStyles}
      exports={<button>Download HTML</button>}
      {...handlers}
    >
      <ul class="label-list" />
    </AnnotatePanels>,
    root,
  );
  return { root, ...handlers };
}

const type = (el: HTMLInputElement, value: string) => {
  el.value = value;
  el.dispatchEvent(new Event("input"));
};
const button = (root: HTMLElement, t: string) => [...root.querySelectorAll("button")].find((b) => b.textContent === t) as HTMLButtonElement;

it("adds a label at the current peak", async () => {
  const { root, onAdd } = mount(3.03);
  expect(button(root, "Add Label").disabled).toBe(true);
  type(root.querySelector<HTMLInputElement>("#annotate-label")!, "lactate");
  await tick();
  expect(button(root, "Add Label").disabled).toBe(false);
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  expect(onAdd).toHaveBeenCalledWith("lactate", 0);
});

it("needs a clicked peak before adding", async () => {
  const { root } = mount(null);
  type(root.querySelector<HTMLInputElement>("#annotate-label")!, "lactate");
  await tick();
  expect(button(root, "Add Label").disabled).toBe(true);
  expect(root.textContent).toContain("Click a peak on the spectra.");
});

it("styles a group or a sample line and can reset it", async () => {
  const { root, onApplyStyle, onResetStyle } = mount(null, { B: { dash: "dot" } });
  const pick = root.querySelector<HTMLSelectElement>('[aria-label="Line to style"]')!;
  expect([...pick.querySelectorAll("optgroup")].map((g) => g.label)).toEqual(["Groups", "Samples"]);
  pick.value = "B";
  pick.dispatchEvent(new Event("change"));
  const dash = root.querySelector<HTMLSelectElement>('[aria-label="Line dash"]')!;
  dash.value = "dot";
  dash.dispatchEvent(new Event("change"));
  await tick();
  button(root, "Apply Style").click();
  expect(onApplyStyle).toHaveBeenCalledWith("B", expect.objectContaining({ dash: "dot" }));
  button(root, "Reset").click();
  expect(onResetStyle).toHaveBeenCalledWith("B");
  expect(root.textContent).toContain("Download HTML");
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd GUI && npx vitest run renderer/components/AnnotateToolbar.test.tsx renderer/components/AnnotatePanels.test.tsx`
Expected: FAIL, cannot resolve the components.

- [ ] **Step 3: Components**

Create `GUI/renderer/components/AnnotateToolbar.tsx`:

```tsx
import { useEffect, useState } from "preact/hooks";

type Mode = "single" | "median";

interface Props {
  mode: Mode;
  onMode: (mode: Mode) => void;
  fontSize: number;
  onFontSize: (size: number) => void;
  peak: number | null;
}

// the toolbar of metbit's annotate_peak dashboard: spectra mode, label font size, current peak
export function AnnotateToolbar({ mode, onMode, fontSize, onFontSize, peak }: Props) {
  const [size, setSize] = useState(fontSize); // follows the slider while dragging; saved on release
  useEffect(() => setSize(fontSize), [fontSize]);
  return (
    <div class="toolbar" role="toolbar" aria-label="Annotation tools">
      <div class="segmented" role="group" aria-label="Spectra">
        {(["single", "median"] as const).map((m) => (
          <button key={m} type="button" aria-pressed={mode === m} onClick={() => onMode(m)}>
            {m === "single" ? "Single spectra" : "Median spectra"}
          </button>
        ))}
      </div>
      <label class="toolbar-slider">
        Font size
        <input
          type="range"
          min={6}
          max={48}
          step={1}
          value={size}
          onInput={(e) => setSize(Number(e.currentTarget.value))}
          onChange={(e) => onFontSize(Number(e.currentTarget.value))}
        />
        <span class="toolbar-value">{size} px</span>
      </label>
      <span class="peak-pill">Current peak: {peak === null ? "none" : `${peak.toFixed(3)} ppm`}</span>
    </div>
  );
}
```

Create `GUI/renderer/components/AnnotatePanels.tsx`:

```tsx
import type { ComponentChildren } from "preact";
import { useState } from "preact/hooks";

export interface LineStyleValue {
  color?: string;
  dash?: string;
  width?: number;
}

interface Props {
  peak: number | null;
  onAdd: (label: string, angle: number) => void;
  targets: { groups: string[]; samples: string[] };
  lineStyles: Record<string, LineStyleValue>;
  onApplyStyle: (target: string, style: LineStyleValue) => void;
  onResetStyle: (target: string) => void;
  exports: ComponentChildren;
  children: ComponentChildren; // the current labels list
}

const DASHES = ["solid", "dash", "dot", "dashdot"];

// the collapsible panels of metbit's annotate_peak dashboard
export function AnnotatePanels({ peak, onAdd, targets, lineStyles, onApplyStyle, onResetStyle, exports, children }: Props) {
  const [label, setLabel] = useState("");
  const [angle, setAngle] = useState(0);
  const [target, setTarget] = useState("");
  const [color, setColor] = useState("#2563eb");
  const [dash, setDash] = useState("solid");
  const [width, setWidth] = useState(1.5);
  const chosen = target || targets.groups[0] || "";
  const canAdd = peak !== null && label.trim() !== "";
  return (
    <div class="annotate-panels">
      <details class="dash-panel" open>
        <summary>📍 Add annotation</summary>
        <form
          class="dash-panel-body"
          onSubmit={(e) => {
            e.preventDefault();
            if (!canAdd) return;
            onAdd(label.trim(), angle);
            setLabel("");
          }}
        >
          <span class="hint">{peak === null ? "Click a peak on the spectra." : `Current peak: ${peak.toFixed(3)} ppm`}</span>
          <label class="field-inline">
            Label
            <input id="annotate-label" placeholder="Enter peak label" value={label} onInput={(e) => setLabel(e.currentTarget.value)} />
          </label>
          <label class="field-inline">
            Angle (°)
            <input
              type="number"
              min={-180}
              max={180}
              step={5}
              value={angle}
              onInput={(e) => setAngle(Number(e.currentTarget.value) || 0)}
            />
          </label>
          <button class="btn primary" type="submit" disabled={!canAdd}>
            Add Label
          </button>
        </form>
      </details>
      <details class="dash-panel">
        <summary>🎨 Style line</summary>
        <div class="dash-panel-body">
          <select aria-label="Line to style" value={chosen} onChange={(e) => setTarget(e.currentTarget.value)}>
            <optgroup label="Groups">
              {targets.groups.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </optgroup>
            <optgroup label="Samples">
              {targets.samples.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </optgroup>
          </select>
          <div class="row">
            <input type="color" aria-label="Line color" value={color} onChange={(e) => setColor(e.currentTarget.value)} />
            <select aria-label="Line dash" value={dash} onChange={(e) => setDash(e.currentTarget.value)}>
              {DASHES.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
            <input
              type="number"
              aria-label="Line width"
              min={0.1}
              max={10}
              step={0.1}
              value={width}
              onChange={(e) => setWidth(Number(e.currentTarget.value) || 1.5)}
            />
          </div>
          <div class="row">
            <button class="btn" type="button" disabled={!chosen} onClick={() => onApplyStyle(chosen, { color, dash, width })}>
              Apply Style
            </button>
            {lineStyles[chosen] && (
              <button class="link" type="button" onClick={() => onResetStyle(chosen)}>
                Reset
              </button>
            )}
          </div>
        </div>
      </details>
      <details class="dash-panel" open>
        <summary>📌 Current labels</summary>
        <div class="dash-panel-body">{children}</div>
      </details>
      <div class="row dash-exports">{exports}</div>
    </div>
  );
}
```

In `GUI/renderer/components/AnnotationList.tsx`, replace `    <div class="annotations">\n      <h3>Labels</h3>\n` with `    <div class="labels">\n`.

Run: `git rm -q GUI/renderer/components/AnnotationPopover.tsx GUI/renderer/components/AnnotationPopover.test.tsx`

Append to `GUI/renderer/theme.css`:

```css
.toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2); padding: var(--s1) var(--s2); border-bottom: 1px solid var(--border); }
.segmented { display: inline-flex; border: 1px solid var(--border); border-radius: 999px; overflow: hidden; }
.segmented button { all: unset; cursor: pointer; padding: 4px 12px; font-size: 12px; color: var(--muted); }
.segmented button[aria-pressed="true"] { background: var(--accent); color: #fff; }
.segmented button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.toolbar-slider { display: flex; align-items: center; gap: var(--s1); font-size: 12px; color: var(--muted); }
.toolbar-slider input { width: 140px; accent-color: var(--accent); }
.toolbar-value { min-width: 38px; color: var(--text); font-variant-numeric: tabular-nums; }
.peak-pill {
  margin-left: auto;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  color: var(--accent);
  background: color-mix(in srgb, var(--accent) 12%, transparent);
}
.annotate-panels { display: flex; flex-direction: column; gap: var(--s1); padding-top: var(--s2); border-top: 1px solid var(--border); }
.dash-panel { border: 1px solid var(--border); border-radius: 8px; }
.dash-panel summary { cursor: pointer; padding: 8px 12px; font-size: 12px; font-weight: 600; }
.dash-panel[open] summary { border-bottom: 1px solid var(--border); }
.dash-panel-body { display: flex; flex-direction: column; gap: var(--s1); padding: 10px 12px; }
.field-inline { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--muted); }
.field-inline input, .dash-panel select, .dash-panel input[type="number"] {
  font: inherit; color: var(--text); background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 6px var(--s1); min-width: 0;
}
.dash-panel input[type="color"] { flex: none; width: 32px; height: 30px; padding: 0; border: 1px solid var(--border); border-radius: 6px; background: none; }
.dash-exports { flex-wrap: wrap; gap: 12px; }
```

Run: `cd GUI && npx vitest run renderer/components`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -A GUI/renderer/components GUI/renderer/theme.css
git commit -m "feat(gui): annotator toolbar and dashboard panels"
```

---

### Task 3: Wire the dashboard, draggable labels and end-to-end coverage

**Files:**
- Modify: `GUI/renderer/api.ts`, `GUI/renderer/components/PlotView.tsx`, `GUI/renderer/App.tsx`
- Modify (replace): `GUI/e2e/annotate.spec.ts`
- Modify: `GUI/e2e/layout.spec.ts` (labels are added through the panel now)

**Interfaces:**
- Consumes: Task 1 routes, Task 2 components.
- Produces: `Api.figure(id, view, opts?: { ppm?: number; mode?: string })`, `patchAnnotation`, `getAnnotationSettings`, `putAnnotationSettings`, `exportHtml`; `AnnotationSettings` type; `PlotView` prop `onLabelMove(id, ax, ay)`.

- [ ] **Step 1: API, PlotView and App**

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
    ("export interface Annotation {\n  id: string;\n  ppm: number;\n  label: string;\n  angle: number;\n}\n",
     "export interface Annotation {\n  id: string;\n  ppm: number;\n  label: string;\n  angle: number;\n  ax?: number;\n  ay?: number;\n}\n\n"
     "export interface AnnotationSettings {\n  font_size?: number;\n"
     "  line_styles: Record<string, { color?: string; dash?: string; width?: number }>;\n"
     "  targets: { groups: string[]; samples: string[] };\n}\n"),
    ('''  figure(id: string, view: string, ppm?: number): Promise<any> {
    const at = ppm === undefined ? "" : `&ppm=${ppm}`;
    return this.req(`/stages/${id}/figure?view=${encodeURIComponent(view)}${at}`).then((r) => r.json());''',
     '''  figure(id: string, view: string, opts: { ppm?: number; mode?: string } = {}): Promise<any> {
    const at = opts.ppm === undefined ? "" : `&ppm=${opts.ppm}`;
    const mode = opts.mode === undefined ? "" : `&mode=${encodeURIComponent(opts.mode)}`;
    return this.req(`/stages/${id}/figure?view=${encodeURIComponent(view)}${at}${mode}`).then((r) => r.json());'''),
    ("  exportAnnotations(path: string): Promise<unknown> {",
     '''  patchAnnotation(id: string, patch: Partial<Annotation>): Promise<Annotation> {
    return this.req(`/annotations/${id}`, { method: "PATCH", body: JSON.stringify(patch) }).then((r) => r.json());
  }

  getAnnotationSettings(): Promise<AnnotationSettings> {
    return this.req("/annotations/settings").then((r) => r.json());
  }

  putAnnotationSettings(settings: AnnotationSettings): Promise<AnnotationSettings> {
    return this.put("/annotations/settings", settings);
  }

  exportHtml(id: string, path: string, view: string, opts: { ppm?: number; mode?: string } = {}): Promise<unknown> {
    return this.post(`/stages/${id}/html`, { path, view, ...opts });
  }

  exportAnnotations(path: string): Promise<unknown> {'''),
])
rep(R + "components/PlotView.tsx", [
    ("export function PlotView({ figure, onPick }: { figure: Figure; onPick?: (point: PickPoint) => void }) {",
     "export function PlotView({\n  figure,\n  onPick,\n  onLabelMove,\n}: {\n  figure: Figure;\n  onPick?: (point: PickPoint) => void;\n"
     "  onLabelMove?: (id: string, ax: number, ay: number) => void;\n}) {"),
    ("  const pickable = onPick !== undefined;\n",
     "  const pickable = onPick !== undefined;\n  const move = useRef(onLabelMove);\n  move.current = onLabelMove;\n"
     "  const draggable = onLabelMove !== undefined;\n"),
    ('    const config = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["toImage"] };',
     '    const config = {\n      responsive: true,\n      displaylogo: false,\n      modeBarButtonsToRemove: ["toImage"],\n'
     '      // drag a label off a crowded peak, as metbit\'s annotator (Plotly editable annotations)\n'
     '      ...(draggable ? { edits: { annotationTail: true } } : {}),\n    };'),
    ('''        if (p) pick.current?.({ x: p.x, customdata: p.customdata, clientX: e.event?.clientX, clientY: e.event?.clientY });
      });''',
     '''        if (p) pick.current?.({ x: p.x, customdata: p.customdata, clientX: e.event?.clientX, clientY: e.event?.clientY });
      });
      node.removeAllListeners?.("plotly_relayout");
      node.on("plotly_relayout", (e: Record<string, unknown>) => {
        // a dragged label reports annotations[i].ax / .ay; our labels carry their id as the annotation name
        for (const key of Object.keys(e)) {
          const m = /^annotations\\[(\\d+)\\]\\.ax$/.exec(key);
          const a = m ? node.layout.annotations?.[Number(m[1])] : undefined;
          if (a?.name) move.current?.(a.name, Math.round(a.ax), Math.round(a.ay));
        }
      });'''),
    ("  }, [figure, pickable]);", "  }, [figure, pickable, draggable]);"),
])
rep(R + "App.tsx", [
    ("type Annotation, type PlotStyle,", "type Annotation, type AnnotationSettings, type PlotStyle,"),
    ('import { AnnotationPopover } from "./components/AnnotationPopover";\n',
     'import { AnnotatePanels } from "./components/AnnotatePanels";\nimport { AnnotateToolbar } from "./components/AnnotateToolbar";\n'),
    ("  const [pending, setPending] = useState<{ ppm: number; x: number; y: number } | null>(null);\n",
     '  const [peak, setPeak] = useState<number | null>(null); // Annotate: the clicked peak\n'
     '  const [annotateMode, setAnnotateMode] = useState<"single" | "median" | null>(null);\n'
     "  const [annotSettings, setAnnotSettings] = useState<AnnotationSettings | null>(null);\n"),
    ('''        // on the Annotate spectra a click opens the label popover at the click
        if (current === "annotate" && tab === "spectra" && pt.clientX !== undefined && pt.clientY !== undefined) {
          setPending({ ppm: v, x: pt.clientX, y: pt.clientY });
        }''',
     '''        // on the Annotate spectra a click sets the current peak and focuses the label, as metbit's annotator
        if (current === "annotate" && tab === "spectra") {
          setPeak(v);
          document.getElementById("annotate-label")?.focus();
        }'''),
    ("    setStageStyle(null); // the next stage's style loads fresh\n    setPending(null);\n",
     "    setStageStyle(null); // the next stage's style loads fresh\n    setPeak(null);\n"),
    ('''      if (!api || !pending) return;
      await api.addAnnotation(pending.ppm, label, angle);
      setPending(null);
      await annotationsChanged(api);''',
     '''      if (!api || peak === null) return;
      await api.addAnnotation(peak, label, angle);
      await annotationsChanged(api);'''),
    ('api.figure(current, tab, at).then((f) => live && setFigure(f));',
     'api.figure(current, tab, { ppm: at, mode: current === "annotate" ? annotateMode ?? undefined : undefined }).then((f) => live && setFigure(f));'),
    ("    if (companion && hasResult) api.figure(current, companion, at).then((f) => live && setCompanionFigure(f));",
     "    if (companion && hasResult) api.figure(current, companion, { ppm: at }).then((f) => live && setCompanionFigure(f));"),
    ("  }, [api, current, tab, hasResult, version, focusPpm]);",
     "  }, [api, current, tab, hasResult, version, focusPpm, annotateMode]);\n\n"
     "  useEffect(() => {\n    if (api && current === \"annotate\") api.getAnnotationSettings().then(setAnnotSettings);\n  }, [api, current, version]);"),
    ("  const openStyle = () =>",
     '''  const annotateModeNow = annotateMode ?? ((stage?.params?.mode as "single" | "median" | undefined) ?? "single");

  const saveAnnotSettings = (next: AnnotationSettings) =>
    guarded(async () => {
      if (!api) return;
      setAnnotSettings(await api.putAnnotationSettings(next));
      setVersion((v) => v + 1);
    });

  const setLabelSize = (font_size: number) => annotSettings && saveAnnotSettings({ ...annotSettings, font_size });

  const applyLineStyle = (target: string, style: { color?: string; dash?: string; width?: number }) =>
    annotSettings && saveAnnotSettings({ ...annotSettings, line_styles: { ...annotSettings.line_styles, [target]: style } });

  const resetLineStyle = (target: string) => {
    if (!annotSettings) return;
    const { [target]: _removed, ...rest } = annotSettings.line_styles;
    saveAnnotSettings({ ...annotSettings, line_styles: rest });
  };

  const moveLabel = (id: string, ax: number, ay: number) =>
    guarded(async () => {
      if (!api) return;
      await api.patchAnnotation(id, { ax, ay }); // the plot already shows the new spot; just remember it
      setAnnotations(await api.listAnnotations());
    });

  const exportHtml = () =>
    guarded(async () => {
      const path = await window.metbit.saveDialog("annotated-spectra.html", [{ name: "HTML", extensions: ["html"] }]);
      if (!path || !api) return;
      await api.exportHtml(current, path, "spectra", { mode: annotateModeNow });
      flash(`Saved ${basename(path)}`);
    });

  const openStyle = () =>'''),
    ('''        {current === "annotate" && (
          <AnnotationList
            items={annotations}
            onDelete={deleteAnnotation}
            onImport={importAnnotations}
            onExport={exportAnnotations}
          />
        )}''',
     '''        {current === "annotate" && annotSettings && (
          <AnnotatePanels
            peak={peak}
            onAdd={addAnnotation}
            targets={annotSettings.targets}
            lineStyles={annotSettings.line_styles ?? {}}
            onApplyStyle={applyLineStyle}
            onResetStyle={resetLineStyle}
            exports={
              <>
                <button type="button" class="link" disabled={!hasResult} onClick={exportCsv}>
                  Intensity table (CSV)
                </button>
                <button type="button" class="link" disabled={!hasResult} onClick={exportHtml}>
                  Download HTML
                </button>
              </>
            }
          >
            <AnnotationList items={annotations} onDelete={deleteAnnotation} onImport={importAnnotations} onExport={exportAnnotations} />
          </AnnotatePanels>
        )}'''),
    ('''      {pending && (
        <AnnotationPopover ppm={pending.ppm} x={pending.x} y={pending.y} onAdd={addAnnotation} onCancel={() => setPending(null)} />
      )}
''', ""),
    ('        <div class="view">{body}</div>',
     '''        {current === "annotate" && hasResult && tab === "spectra" && (
          <AnnotateToolbar
            mode={annotateModeNow}
            onMode={setAnnotateMode}
            fontSize={annotSettings?.font_size ?? 14}
            onFontSize={setLabelSize}
            peak={peak}
          />
        )}
        <div class="view">{body}</div>'''),
]
)
p = pathlib.Path(R + "App.tsx")
s = p.read_text()
old = "<PlotView figure={figure} onPick={onPick} />"
assert s.count(old) == 3, s.count(old)
p.write_text(s.replace(old, "<PlotView figure={figure} onPick={onPick} onLabelMove={moveLabel} />"))
rep("GUI/e2e/layout.spec.ts", [
    ('''    await win.getByLabel("Label", { exact: true }).fill(label);
    await win.getByLabel("Label", { exact: true }).press("Enter");''',
     '''    await win.fill("#annotate-label", label);
    await win.locator("#annotate-label").press("Enter");'''),
])
print("renderer wired")
EOF
```

- [ ] **Step 2: End-to-end test**

Replace `GUI/e2e/annotate.spec.ts` with:

```ts
import { _electron as electron, expect, test, type Page } from "@playwright/test";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
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
const labelOffset = (win: Page, text: string) =>
  win.evaluate((t) => ((document.querySelector(".plot") as any)?.layout?.annotations ?? []).find((a: any) => a.text === t)?.ay, text);
const traces = (win: Page) => win.evaluate(() => ((document.querySelector(".plot") as any)?.data ?? []).map((t: any) => t.name));

test("annotator dashboard: add, switch, restyle, drag and export", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const env = { ...process.env, METBIT_E2E_DIR: dir, METBIT_USER_DATA: join(dir, "user") } as Record<string, string>;
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

  // click a peak: current peak pill + focused label field, then Add Label
  const box = (await win.locator(".plot .nsewdrag").first().boundingBox())!;
  await win.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.3);
  await win.mouse.click(box.x + box.width * 0.2, box.y + box.height * 0.3);
  await expect(win.locator(".peak-pill")).toContainText(" ppm");
  await expect(win.locator("#annotate-label")).toBeFocused();
  await win.fill("#annotate-label", "lactate");
  await win.locator("#annotate-label").press("Enter");
  await expect(win.locator(".label-list")).toContainText("lactate");
  await expect.poll(() => plotLabels(win)).toContain("lactate");

  // median switch: no re-run
  await win.getByRole("button", { name: "Median spectra" }).click();
  await expect.poll(() => traces(win)).toEqual(["A", "B"]);

  // style group B dotted
  await win.locator("summary", { hasText: "Style line" }).click();
  await win.selectOption('[aria-label="Line to style"]', "B");
  await win.selectOption('[aria-label="Line dash"]', "dot");
  await win.getByRole("button", { name: "Apply Style" }).click();
  await expect
    .poll(() => win.evaluate(() => (document.querySelector(".plot") as any).data.find((t: any) => t.name === "B")?.line?.dash))
    .toBe("dot");

  // drag the label; the new offset comes back from the server after a refetch
  const label = (await win.locator(".plot g.annotation", { hasText: "lactate" }).first().boundingBox())!;
  const cx = label.x + label.width / 2;
  const cy = label.y + label.height / 2;
  await win.mouse.move(cx, cy);
  await win.mouse.down();
  await win.mouse.move(cx + 30, cy - 40, { steps: 8 });
  await win.mouse.up();
  await win.getByRole("button", { name: "Single spectra" }).click();
  await expect.poll(() => labelOffset(win, "lactate")).not.toBe(-40);

  // self-contained HTML export
  await win.getByRole("button", { name: "Download HTML" }).click();
  const html = join(dir, "annotated-spectra.html");
  await expect.poll(() => existsSync(html)).toBe(true);
  expect(readFileSync(html, "utf8")).toContain("lactate");

  // labels show on other spectra and in the intensity table
  await win.getByRole("tab", { name: "Table" }).click();
  await expect(win.locator("table.data thead")).toContainText("lactate_");
  await win.locator(".rail button", { hasText: "Normalize" }).click();
  await run();
  await plot();
  await expect.poll(() => plotLabels(win)).toContain("lactate");

  await app.close();
});
```

- [ ] **Step 3: Verify**

Run from the repo root:
```bash
python3 -m pytest -q -p no:cacheprovider && (cd GUI && npm run typecheck && npm test && npm run e2e)
```
Expected: repo suite passes (coverage at or above 99%); typecheck exits 0; `48 passed`; `12 passed`.

- [ ] **Step 4: Commit**

```bash
git add GUI/renderer GUI/e2e
git commit -m "feat(gui): annotator dashboard with toolbar, panels, draggable labels and HTML export"
```
