# metbit GUI Phase E (Plot Styling) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users style every plot (per-group color and symbol, marker size, opacity, font size, line width, confidence ellipses) with a global style and optional per-stage overrides, applied at render time without re-running analyses.

**Architecture:** `stages/style.py` holds the `PlotStyle` / `StyleOverride` models, palette assignment, merging, and `render()`. Figure functions may take a second `style` argument; those get metbit's native arguments (`color_dict`, `symbol_dict`, `marker_size`, ...), the rest get a generic fallback that recolors traces named after a group. The session stores the global style and per-stage overrides as plain dicts (saved in projects); `GET/PUT /style` and `GET/PUT /stages/{id}/style` edit them. The renderer adds a `StyleForm`, a "Plot style" panel (global) and a collapsed "Style" section under each stage's parameters (override). Spec: `GUI/docs/specs/2026-09-29-metbit-gui-style-annotation-stocsy-design.md` (section 1).

**Tech Stack:** Pydantic v2, Plotly, metbit plot functions; Preact, Vitest, Playwright.

## Global Constraints

- All GUI code lives under `GUI/`; Python GUI tests under `tests/gui/`.
- Never use the em dash character or the section sign anywhere.
- Commit messages carry no `Co-Authored-By` trailer.
- Keep every source file under 500 lines.
- `PlotStyle` defaults and bounds (exact): `marker_size` 12 (4..60), `marker_opacity` 0.8 (0.1..1), `font_size` 14 (8..32), `line_width` 1.2 (0.5..5), `show_ellipses` true; colors `#rrggbb`; symbols `circle`, `square`, `diamond`, `triangle-up`, `triangle-down`, `cross`, `x`, `star`.
- Default group colors: Okabe-Ito `#0072B2`, `#D55E00`, `#009E73`, `#CC79A7`, `#E69F00`, `#56B4E9`, `#F0E442`, `#000000`, assigned in sorted group order, skipping colors already used.
- A style change never re-runs an analysis. Stage overrides are stored with `PUT /stages/{id}/style` (not in run params); effective style = global merged with the override, override winning.
- Natively styled figures are not recolored by the fallback (it would flatten the spectra overlay's transparency).
- Python GUI tests: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`. Renderer: `cd GUI && npm run typecheck && npm test`. E2E: `cd GUI && npm run e2e`.

## File Map

```
GUI/server/metbit_server/
  stages/style.py            NEW: models, palette, merge, apply_style, render
  stages/base.py             MODIFY: spectra_figure(style)
  stages/data.py             MODIFY: spectra lambdas take style
  stages/analysis.py         MODIFY: PCA / OPLS-DA scores and VIP native style; results keep labels
  stages/multivariate.py     MODIFY: LDA / ICA native style
  stages/stats.py            MODIFY: boxplot native style
  session.py                 MODIFY: style, stage_styles
  project.py                 MODIFY: persist style
  app.py                     MODIFY: style routes; figures rendered with effective style
tests/gui/test_gui_style.py         NEW
tests/gui/test_gui_style_api.py     NEW
tests/gui/test_gui_native_style.py  NEW
GUI/renderer/api.ts                          MODIFY: style types and calls
GUI/renderer/components/StyleForm.tsx (+ test) NEW
GUI/renderer/components/ParamPanel.tsx        MODIFY: children below the form
GUI/renderer/components/Rail.tsx              MODIFY: Plot style footer button
GUI/renderer/components/PlotView.tsx          MODIFY: keep the server's font size
GUI/renderer/App.tsx                          MODIFY: style panel, stage style section
GUI/renderer/theme.css                        MODIFY
GUI/e2e/style.spec.ts                         NEW
```

---

### Task 1: Style model, merging and rendering

**Files:**
- Create: `GUI/server/metbit_server/stages/style.py`
- Test: `tests/gui/test_gui_style.py`

**Interfaces:**
- Produces:
  - `SYMBOLS: list[str]`, `PALETTE: list[str]`.
  - `GroupStyle(color: str, symbol: str = "circle")`, `PlotStyle(groups: dict[str, GroupStyle] = {}, marker_size=12, marker_opacity=0.8, font_size=14, line_width=1.2, show_ellipses=True)`.
  - `GroupOverride(color: str | None, symbol: str | None)`, `StyleOverride(groups: dict[str, GroupOverride] = {}, marker_size | marker_opacity | font_size | line_width | show_ellipses: Optional)`.
  - `with_groups(style: PlotStyle, names: Iterable[str]) -> PlotStyle`.
  - `merge(base: PlotStyle, override: StyleOverride) -> PlotStyle` (unknown groups ignored).
  - `override_dict(o: StyleOverride) -> dict` (no `None`, no empty `groups`).
  - `colors(style) -> dict[str, str]`, `symbols(style) -> dict[str, str]`.
  - `apply_style(fig, style, recolor: bool = True) -> go.Figure` (sets layout font size; when `recolor`, colors traces whose `name` or `legendgroup` is a group).
  - `render(make, result, style) -> go.Figure`: calls `make(result, style)` when `make` takes two or more parameters, else `make(result)`; then `apply_style(fig, style, recolor=not native)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_style.py`:

```python
import plotly.graph_objects as go
import pytest
from pydantic import ValidationError

from metbit_server.stages.style import (
    PALETTE,
    PlotStyle,
    StyleOverride,
    apply_style,
    merge,
    override_dict,
    render,
    with_groups,
)


def test_defaults_and_validation():
    s = PlotStyle()
    assert (s.marker_size, s.marker_opacity, s.font_size, s.line_width, s.show_ellipses) == (12, 0.8, 14, 1.2, True)
    with pytest.raises(ValidationError):
        PlotStyle(groups={"A": {"color": "red"}})
    with pytest.raises(ValidationError):
        PlotStyle(marker_size=100)
    with pytest.raises(ValidationError):
        PlotStyle(groups={"A": {"color": "#112233", "symbol": "hexagon-open-dot"}})


def test_with_groups_assigns_palette_in_sorted_order_and_keeps_choices():
    s = with_groups(PlotStyle(groups={"B": {"color": "#0072B2"}}), ["C", "B", "A"])
    assert s.groups["B"].color == "#0072B2"
    assert s.groups["A"].color == PALETTE[1]  # PALETTE[0] is already B's
    assert s.groups["C"].color == PALETTE[2]
    assert s.groups["A"].symbol == "circle"


def test_merge_prefers_override_and_ignores_unknown_groups():
    base = with_groups(PlotStyle(), ["A", "B"])
    o = StyleOverride(marker_size=30, groups={"A": {"symbol": "square"}, "Z": {"color": "#000000"}})
    eff = merge(base, o)
    assert (eff.marker_size, eff.font_size) == (30, 14)
    assert (eff.groups["A"].symbol, eff.groups["A"].color) == ("square", base.groups["A"].color)
    assert "Z" not in eff.groups
    assert override_dict(StyleOverride()) == {}
    assert override_dict(o) == {"marker_size": 30, "groups": {"A": {"symbol": "square"}, "Z": {"color": "#000000"}}}


def test_apply_style_colors_traces_named_after_groups():
    style = PlotStyle(groups={"A": {"color": "#112233", "symbol": "diamond"}}, font_size=18)
    fig = go.Figure([
        go.Scatter(name="A", mode="markers", x=[1], y=[1]),
        go.Scatter(name="A", mode="lines", x=[1], y=[1]),
        go.Box(name="A", y=[1, 2]),
        go.Scatter(name="other", mode="markers", x=[1], y=[1], marker_color="#999999"),
    ])
    apply_style(fig, style)
    assert (fig.data[0].marker.color, fig.data[0].marker.symbol) == ("#112233", "diamond")
    assert fig.data[1].line.color == "#112233"
    assert fig.data[2].marker.color == "#112233"
    assert fig.data[3].marker.color == "#999999"
    assert fig.layout.font.size == 18


def test_render_passes_style_to_native_figures_and_skips_their_recolor():
    style = PlotStyle(groups={"A": {"color": "#112233"}}, font_size=20)
    seen = []
    plain = render(lambda r: go.Figure([go.Scatter(name="A", mode="lines")]), "result", style)
    assert plain.data[0].line.color == "#112233"
    native = render(
        lambda r, st=None: seen.append(st) or go.Figure([go.Scatter(name="A", mode="lines", line_color="rgba(1,2,3,0.5)")]),
        "result",
        style,
    )
    assert seen == [style]
    assert native.data[0].line.color == "rgba(1,2,3,0.5)"
    assert native.layout.font.size == 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_style.py --no-cov -q -p no:warnings`
Expected: FAIL with `ModuleNotFoundError: No module named 'metbit_server.stages.style'`.

- [ ] **Step 3: Implement**

Create `GUI/server/metbit_server/stages/style.py`:

```python
"""Plot styling: a global PlotStyle, per-stage overrides, and applying them to figures."""
import inspect
from typing import Any, Callable, Iterable, Literal, Optional

import plotly.graph_objects as go
from pydantic import BaseModel, Field

Symbol = Literal["circle", "square", "diamond", "triangle-up", "triangle-down", "cross", "x", "star"]
SYMBOLS: list[str] = list(Symbol.__args__)
# Okabe-Ito (colorblind-safe); black last because it disappears on dark backgrounds
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
HEX = r"^#[0-9a-fA-F]{6}$"


class GroupStyle(BaseModel):
    color: str = Field(pattern=HEX)
    symbol: Symbol = "circle"


class PlotStyle(BaseModel):
    groups: dict[str, GroupStyle] = {}
    marker_size: int = Field(12, ge=4, le=60)
    marker_opacity: float = Field(0.8, ge=0.1, le=1)
    font_size: int = Field(14, ge=8, le=32)
    line_width: float = Field(1.2, ge=0.5, le=5)
    show_ellipses: bool = True


class GroupOverride(BaseModel):
    color: Optional[str] = Field(None, pattern=HEX)
    symbol: Optional[Symbol] = None


class StyleOverride(BaseModel):
    groups: dict[str, GroupOverride] = {}
    marker_size: Optional[int] = Field(None, ge=4, le=60)
    marker_opacity: Optional[float] = Field(None, ge=0.1, le=1)
    font_size: Optional[int] = Field(None, ge=8, le=32)
    line_width: Optional[float] = Field(None, ge=0.5, le=5)
    show_ellipses: Optional[bool] = None


def with_groups(style: PlotStyle, names: Iterable[str]) -> PlotStyle:
    """Give every group a style; new groups get unused palette colors in sorted order."""
    groups = dict(style.groups)
    used = {g.color.lower() for g in groups.values()}
    free = [c for c in PALETTE if c.lower() not in used] or PALETTE
    new = [n for n in sorted(set(names)) if n not in groups]
    for i, name in enumerate(new):
        groups[name] = GroupStyle(color=free[i % len(free)])
    return style.model_copy(update={"groups": groups})


def merge(base: PlotStyle, override: StyleOverride) -> PlotStyle:
    data = base.model_dump()
    for key, value in override.model_dump(exclude={"groups"}).items():
        if value is not None:
            data[key] = value
    for name, g in override.groups.items():
        if name in data["groups"]:
            data["groups"][name].update({k: v for k, v in g.model_dump().items() if v is not None})
    return PlotStyle.model_validate(data)


def override_dict(o: StyleOverride) -> dict:
    d = o.model_dump(exclude_none=True)
    if not d.get("groups"):
        d.pop("groups", None)
    return d


def colors(style: PlotStyle) -> dict[str, str]:
    return {name: g.color for name, g in style.groups.items()}


def symbols(style: PlotStyle) -> dict[str, str]:
    return {name: g.symbol for name, g in style.groups.items()}


def apply_style(fig: go.Figure, style: PlotStyle, recolor: bool = True) -> go.Figure:
    fig.update_layout(font_size=style.font_size)
    if not recolor:
        return fig
    for trace in fig.data:
        g = style.groups.get(str(trace.name)) or style.groups.get(str(getattr(trace, "legendgroup", "") or ""))
        if g is None:
            continue
        if trace.type in ("scatter", "scattergl"):
            mode = trace.mode or "markers"
            if "markers" in mode:
                trace.marker.color = g.color
                trace.marker.symbol = g.symbol
            if "lines" in mode:
                trace.line.color = g.color
        elif trace.type in ("box", "violin", "bar", "histogram"):
            trace.marker.color = g.color
            if trace.type in ("box", "violin"):
                trace.line.color = g.color
    return fig


def _takes_style(make: Callable) -> bool:
    try:
        return len(inspect.signature(make).parameters) >= 2
    except (TypeError, ValueError):
        return False


def render(make: Callable, result: Any, style: PlotStyle) -> go.Figure:
    """Build a figure with the effective style: native metbit arguments when supported, else the fallback."""
    native = _takes_style(make)
    fig = make(result, style) if native else make(result)
    return apply_style(fig, style, recolor=not native)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`
Expected: `96 passed` (91 + 5; 94 passed and 2 skipped without MTBLS2052).

- [ ] **Step 5: Commit**

```bash
git add GUI/server/metbit_server/stages/style.py tests/gui/test_gui_style.py
git commit -m "feat(gui): add plot style model with palette, overrides and rendering"
```

---

### Task 2: Style storage, routes and projects

**Files:**
- Modify: `GUI/server/metbit_server/session.py`, `GUI/server/metbit_server/project.py`, `GUI/server/metbit_server/app.py`
- Test: `tests/gui/test_gui_style_api.py`

**Interfaces:**
- Consumes: Task 1 `PlotStyle`, `StyleOverride`, `with_groups`, `merge`, `override_dict`, `render`; `base.option_values`.
- Produces:
  - `Session.style: dict` (global, as stored) and `Session.stage_styles: dict[str, dict]`.
  - `GET /style` -> `PlotStyle` dict with every current group filled in; `PUT /style` (full `PlotStyle`) -> same (422 when invalid).
  - `GET /stages/{id}/style` -> `{"override": dict, "effective": PlotStyle dict}`; `PUT /stages/{id}/style` (`StyleOverride`) -> same (422 when invalid, 404 unknown stage). `{}` clears the override.
  - `GET /stages/{id}/figure` renders with the stage's effective style.
  - Projects save `style` and `stage_styles` at the top level of `params.json` and restore them (missing keys load as `{}`).

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_style_api.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_style_api.py --no-cov -q -p no:warnings`
Expected: 4 failed (`/style` 404, `Session` has no `style`).

- [ ] **Step 3: Session and project**

In `GUI/server/metbit_server/session.py`, after `self.outputs: dict[str, Any] = {}` add:

```python
        self.style: dict = {}                    # global PlotStyle as stored (validated on use)
        self.stage_styles: dict[str, dict] = {}  # per-stage StyleOverride dicts
```

In `GUI/server/metbit_server/project.py`:

- In `save_project`, change `manifest: dict = {"format": FORMAT, "version": VERSION, "stages": {}}` to:

```python
    manifest: dict = {
        "format": FORMAT,
        "version": VERSION,
        "stages": {},
        "style": session.style,
        "stage_styles": session.stage_styles,
    }
```

- In `load_project`, change the final line
  `session.status, session.params, session.outputs = fresh.status, fresh.params, fresh.outputs` to:

```python
    session.status, session.params, session.outputs = fresh.status, fresh.params, fresh.outputs
    session.style = manifest.get("style") or {}
    session.stage_styles = manifest.get("stage_styles") or {}
```

- [ ] **Step 4: Routes**

In `GUI/server/metbit_server/app.py`:

Add after the `from .stages import ...` line:

```python
from .stages.base import option_values
from .stages.style import PlotStyle, StyleOverride, merge, override_dict, render, with_groups
```

Add these helpers right after the `exclusive()` context manager:

```python
    def global_style() -> PlotStyle:
        return with_groups(PlotStyle.model_validate(session.style or {}), option_values(session, "groups") or [])

    def effective_style(sid: str) -> PlotStyle:
        return merge(global_style(), StyleOverride.model_validate(session.stage_styles.get(sid, {})))

    def stage_style_body(sid: str) -> dict:
        return {"override": session.stage_styles.get(sid, {}), "effective": effective_style(sid).model_dump()}
```

In the `figure` route replace `fig = make(out)` with:

```python
        fig = render(make, out, effective_style(sid))
```

Add these routes after the `figure` route:

```python
    @app.get("/style")
    def get_style():
        return global_style().model_dump()

    @app.put("/style")
    def put_style(body: dict = Body(...)):
        try:
            style = PlotStyle.model_validate(body)
        except ValidationError as e:
            raise HTTPException(422, json.loads(e.json(include_url=False)))
        session.style = style.model_dump()
        return global_style().model_dump()

    @app.get("/stages/{sid}/style")
    def get_stage_style(sid: str):
        stage_or_404(sid)
        return stage_style_body(sid)

    @app.put("/stages/{sid}/style")
    def put_stage_style(sid: str, body: dict = Body(...)):
        stage_or_404(sid)
        try:
            override = StyleOverride.model_validate(body)
        except ValidationError as e:
            raise HTTPException(422, json.loads(e.json(include_url=False)))
        session.stage_styles[sid] = override_dict(override)
        if not session.stage_styles[sid]:
            del session.stage_styles[sid]
        return stage_style_body(sid)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`
Expected: `100 passed` (98 passed and 2 skipped without MTBLS2052).

- [ ] **Step 6: Commit**

```bash
git add GUI/server tests/gui/test_gui_style_api.py
git commit -m "feat(gui): store global and per-stage plot styles with routes and project persistence"
```

---

### Task 3: Native metbit styling for spectra, scores, VIP, LDA, ICA and boxplots

**Files:**
- Modify: `GUI/server/metbit_server/stages/base.py`, `stages/data.py`, `stages/analysis.py`, `stages/multivariate.py`, `stages/stats.py`
- Test: `tests/gui/test_gui_native_style.py`

**Interfaces:**
- Consumes: `PlotStyle`, `colors`, `symbols` from Task 1.
- Produces:
  - `base.spectra_figure(X, labels, title, style: PlotStyle | None = None)`.
  - `PCAResult.labels` and `OPLSResult.labels` (the sample group labels the model was fitted with; default `None`).
  - Figure callables with signature `(result, style=None)`: import / process / normalize / align spectra, PCA and OPLS-DA scores, VIP, LDA scores, ICA components, Group tests boxplot. All other figures keep `(result)` and get the fallback.

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_native_style.py`:

```python
import pandas as pd

from metbit_server.stages import STAGES
from metbit_server.stages.style import PlotStyle, with_groups

BY_ID = {s.id: s for s in STAGES}
STYLE = with_groups(
    PlotStyle(
        groups={"A": {"color": "#112233", "symbol": "square"}},
        marker_size=20,
        marker_opacity=0.5,
        line_width=2,
        show_ellipses=False,
    ),
    ["A", "B"],
)


def traces(fig, group):
    """Traces for a group, marker traces first (ellipses are line traces with the same name)."""
    found = [t for t in fig.data if str(t.name).split(",")[0].strip() == group]
    return sorted(found, key=lambda t: "markers" not in (getattr(t, "mode", None) or "markers"))


def paint(t) -> set:
    return {getattr(t.marker, "color", None), getattr(t.line, "color", None), getattr(t, "fillcolor", None)}


def test_scores_use_native_color_symbol_size_and_ellipse_toggle(run_all):
    s = run_all(("pca", {}), ("opls", {"group_a": "A", "group_b": "B"}))
    for sid in ("pca", "opls"):
        styled = BY_ID[sid].figures["scores"](s.output(sid), STYLE)
        a = traces(styled, "A")[0]
        assert (a.marker.color, a.marker.symbol, a.marker.size, a.marker.opacity) == ("#112233", "square", 20, 0.5)
        with_ellipses = BY_ID[sid].figures["scores"](s.output(sid), STYLE.model_copy(update={"show_ellipses": True}))
        count = lambda f: len(f.data) + len(f.layout.shapes or ())
        assert count(styled) < count(with_ellipses)


def test_vip_uses_marker_size(run_all):
    r = run_all(("opls", {"group_a": "A", "group_b": "B"})).output("opls")
    fig = BY_ID["opls"].figures["vip"](r, STYLE)
    assert any(t.marker.size == 20 for t in fig.data)


def test_spectra_keep_transparency_and_take_group_color_and_width(run_all):
    fig = BY_ID["import"].figures["spectra"](run_all().output("import"), STYLE)
    a = traces(fig, "A")[0]
    assert a.line.color.startswith("rgba(17,34,51")
    assert a.line.width == 2


def test_lda_ica_and_boxplot_use_group_colors(run_all, csv_path, tmp_path):
    s = run_all(("lda", {}), ("ica", {"n_components": 3}), ("grouptests", {"bin_width": 0}))
    lda = BY_ID["lda"].figures["scores"](s.output("lda"), STYLE)
    ica = BY_ID["ica"].figures["components"](s.output("ica"), STYLE)
    box = BY_ID["grouptests"].figures["boxplot"](s.output("grouptests"), STYLE)
    for fig in (lda, ica, box):
        assert any("#112233" in paint(t) for t in traces(fig, "A")), fig


def test_figures_still_work_without_a_style(run_all):
    s = run_all(("pca", {}))
    assert BY_ID["pca"].figures["scores"](s.output("pca")) is not None
    assert BY_ID["import"].figures["spectra"](s.output("import")) is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_native_style.py --no-cov -q -p no:warnings`
Expected: FAIL (`TypeError: <lambda>() takes 1 positional argument but 2 were given`).

- [ ] **Step 3: Implement**

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


S = "GUI/server/metbit_server/stages/"
rep(S + "base.py", [
    ("from ..session import Session\n",
     "from ..session import Session\nfrom .style import PlotStyle, colors\n"),
    ('''def spectra_figure(X: pd.DataFrame, labels: pd.Series, title: str) -> go.Figure:
    # ponytail: display-only decimation keeps figure JSON small; full data stays in the session
    step = max(1, X.shape[1] // MAX_PLOT_POINTS)
    return SpectraPlot(X.iloc[:, ::step], label=list(labels)).overlay(title=title)''',
     '''def spectra_figure(X: pd.DataFrame, labels: pd.Series, title: str, style: Optional[PlotStyle] = None) -> go.Figure:
    # ponytail: display-only decimation keeps figure JSON small; full data stays in the session
    step = max(1, X.shape[1] // MAX_PLOT_POINTS)
    plot = SpectraPlot(X.iloc[:, ::step], label=list(labels), color_dict=colors(style) if style else None)
    if style is None:
        return plot.overlay(title=title)
    return plot.overlay(title=title, linewidth=style.line_width, font_size=style.font_size)'''),
])
rep(S + "data.py", [
    ('def _import_spectra(d: Union[Dataset, BrukerSource]):',
     'def _import_spectra(d: Union[Dataset, BrukerSource], style=None):'),
    ('    return spectra_figure(d.X, d.labels, "Spectra")',
     '    return spectra_figure(d.X, d.labels, "Spectra", style)'),
    ('figures={"spectra": lambda d: spectra_figure(d.X, d.labels, "Processed spectra")},',
     'figures={"spectra": lambda d, st=None: spectra_figure(d.X, d.labels, "Processed spectra", st)},'),
    ('        "spectra": lambda d: spectra_figure(d.X, d.labels, title),\n'
     '        "before": lambda d: spectra_figure(d.before, d.labels, "Before"),',
     '        "spectra": lambda d, st=None: spectra_figure(d.X, d.labels, title, st),\n'
     '        "before": lambda d, st=None: spectra_figure(d.before, d.labels, "Before", st),'),
])
rep(S + "analysis.py", [
    ("from .base import DATA_ORDER,",
     "from .style import PlotStyle, colors, symbols\nfrom .base import DATA_ORDER,"),
    ("class PCAResult:\n    model: Any\n    pcs: list[str]\n",
     "class PCAResult:\n    model: Any\n    pcs: list[str]\n    labels: Any = None  # group per sample, for styling\n"),
    ('    return PCAResult(model=model, pcs=[f"PC{p.pc_x}", f"PC{p.pc_y}"])',
     '    return PCAResult(model=model, pcs=[f"PC{p.pc_x}", f"PC{p.pc_y}"], labels=groups(session))'),
    ("    vip_threshold: float\n    permuted: bool\n",
     "    vip_threshold: float\n    permuted: bool\n    labels: Any = None  # group per sample in the two-group subset\n"),
    ("    return OPLSResult(model=model, vip_threshold=p.vip_threshold, permuted=p.permutations > 0)",
     "    return OPLSResult(model=model, vip_threshold=p.vip_threshold, permuted=p.permutations > 0, labels=y)"),
    ('            "scores": lambda r: r.model.plot_pca_scores(pc=r.pcs),',
     '            "scores": lambda r, st=None: r.model.plot_pca_scores(pc=r.pcs, **scores_style(st, r.labels)),'),
    ('            "scores": lambda r: r.model.plot_oplsda_scores(),',
     '            "scores": lambda r, st=None: r.model.plot_oplsda_scores(**scores_style(st, r.labels)),'),
    ('            "vip": lambda r: r.model.vip_plot(threshold=r.vip_threshold),',
     '            "vip": lambda r, st=None: r.model.vip_plot(\n'
     '                threshold=r.vip_threshold, **({"marker_size": st.marker_size, "font_size": st.font_size} if st else {})\n'
     '            ),'),
    ("# PCA ----",
     '''def scores_style(style: Optional[PlotStyle], labels: Any) -> dict:
    """metbit's own scores-plot styling arguments (PCA and OPLS-DA share them)."""
    if style is None or labels is None:
        return {}
    return {
        "color_dict": colors(style),
        "symbol_": labels,
        "symbol_dict": symbols(style),
        "marker_size": style.marker_size,
        "marker_opacity": style.marker_opacity,
        "font_size": style.font_size,
        "individual_ellipse": style.show_ellipses,
    }


# PCA ----'''),
    ("from typing import Any, Literal\n", "from typing import Any, Literal, Optional\n"),
])
rep(S + "multivariate.py", [
    ("from .base import DATA_ORDER,", "from .style import colors\nfrom .base import DATA_ORDER,"),
    ('''def _lda_scores(r: LDAResult) -> go.Figure:
    if len(r.lds) >= 2:
        return r.model.plot_lda_scores(ld=r.lds)
    # two groups give a single discriminant: metbit's 2D plot needs LD2
    return px.strip(r.model.get_scores(), x="Group", y="LD1", color="Group", title="LDA scores (LD1)")''',
     '''def _lda_scores(r: LDAResult, style=None) -> go.Figure:
    palette = colors(style) if style else None
    if len(r.lds) >= 2:
        extra = {"color_dict": palette, "marker_size": style.marker_size, "font_size": style.font_size} if style else {}
        return r.model.plot_lda_scores(ld=r.lds, **extra)
    # two groups give a single discriminant: metbit's 2D plot needs LD2
    return px.strip(
        r.model.get_scores(), x="Group", y="LD1", color="Group", title="LDA scores (LD1)", color_discrete_map=palette or {}
    )'''),
    ('            "components": lambda r: r.model.plot_components(ic=["IC1", "IC2"], color_=list(r.labels)),',
     '            "components": lambda r, st=None: r.model.plot_components(\n'
     '                ic=["IC1", "IC2"], color_=list(r.labels), **({"color_dict": colors(st), "font_size": st.font_size} if st else {})\n'
     '            ),'),
])
rep(S + "stats.py", [
    ("from .base import DATA_ORDER,", "from .style import colors\nfrom .base import DATA_ORDER,"),
    ('''def _boxplot(r: GroupTestResult) -> go.Figure:
    cls = ANOVAStats if r.test == "anova" else KruskalStats
    stats = cls(r.data, x_col="Group", y_col=r.feature)
    stats.fit()
    return stats.plot(title=f"{float(r.feature):.4f} ppm")''',
     '''def _boxplot(r: GroupTestResult, style=None) -> go.Figure:
    cls = ANOVAStats if r.test == "anova" else KruskalStats
    stats = cls(r.data, x_col="Group", y_col=r.feature)
    stats.fit()
    extra = {"custom_colors": colors(style), "font_size": style.font_size} if style else {}
    return stats.plot(title=f"{float(r.feature):.4f} ppm", **extra)'''),
])
print("native styling wired")
EOF
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`
Expected: `105 passed` (103 passed and 2 skipped without MTBLS2052).

- [ ] **Step 5: Commit**

```bash
git add GUI/server tests/gui/test_gui_native_style.py
git commit -m "feat(gui): pass plot style to metbit's native plotting arguments"
```

---

### Task 4: Style form, Plot style panel and per-stage override

**Files:**
- Modify: `GUI/renderer/api.ts`, `GUI/renderer/components/ParamPanel.tsx`, `GUI/renderer/components/Rail.tsx`, `GUI/renderer/components/PlotView.tsx`, `GUI/renderer/App.tsx`, `GUI/renderer/theme.css`
- Create: `GUI/renderer/components/StyleForm.tsx`, `GUI/renderer/components/StyleForm.test.tsx`

**Interfaces:**
- Consumes: routes from Task 2.
- Produces:
  - `api.ts`: `GroupStyle`, `PlotStyle`, `StyleOverride`, `StageStyle { override: StyleOverride; effective: PlotStyle }`; `Api.getStyle()`, `Api.putStyle(style)`, `Api.getStageStyle(id)`, `Api.putStageStyle(id, override)`.
  - `StyleForm({ effective, override?, onChange(patch), onReset? })`; group rows `.style-group` with `aria-label="<group> color"` / `"<group> symbol"`; number inputs `#style-marker_size`, `#style-marker_opacity`, `#style-font_size`, `#style-line_width` (clamped to bounds); checkbox "Confidence ellipses"; `.overridden` on fields present in `override`; "Use global style" link calls `onReset`.
  - `applyPatch(base, patch)`: shallow merge with per-group deep merge.
  - `ParamPanel` renders `children` below its form (root `div.panel` containing `form.panel-form`).
  - `Rail` optional `onStyle` adds a `Plot style…` footer button.
  - `PlotView` no longer forces font size 12.

- [ ] **Step 1: Write the failing tests**

Create `GUI/renderer/components/StyleForm.test.tsx`:

```tsx
import { render } from "preact";
import { expect, it, vi } from "vitest";
import type { PlotStyle } from "../api";
import { StyleForm, applyPatch } from "./StyleForm";

const STYLE: PlotStyle = {
  groups: { A: { color: "#112233", symbol: "circle" }, B: { color: "#445566", symbol: "square" } },
  marker_size: 12,
  marker_opacity: 0.8,
  font_size: 14,
  line_width: 1.2,
  show_ellipses: true,
};

function mount(props: Partial<Parameters<typeof StyleForm>[0]> = {}) {
  const root = document.createElement("div");
  const onChange = vi.fn();
  render(<StyleForm effective={STYLE} onChange={onChange} {...props} />, root);
  return { root, onChange };
}

const change = (el: HTMLInputElement | HTMLSelectElement, value: string) => {
  el.value = value;
  el.dispatchEvent(new Event("change"));
};

it("renders a row per group and reports color and symbol changes", () => {
  const { root, onChange } = mount();
  expect(root.querySelectorAll(".style-group")).toHaveLength(2);
  expect(root.querySelector<HTMLInputElement>('[aria-label="A color"]')!.value).toBe("#112233");
  change(root.querySelector<HTMLSelectElement>('[aria-label="B symbol"]')!, "diamond");
  expect(onChange).toHaveBeenLastCalledWith({ groups: { B: { symbol: "diamond" } } });
  change(root.querySelector<HTMLInputElement>('[aria-label="A color"]')!, "#abcdef");
  expect(onChange).toHaveBeenLastCalledWith({ groups: { A: { color: "#abcdef" } } });
});

it("clamps numbers to their range", () => {
  const { root, onChange } = mount();
  change(root.querySelector<HTMLInputElement>("#style-marker_size")!, "100");
  expect(onChange).toHaveBeenLastCalledWith({ marker_size: 60 });
});

it("toggles ellipses", () => {
  const { root, onChange } = mount();
  root.querySelector<HTMLInputElement>('input[type="checkbox"]')!.click();
  expect(onChange).toHaveBeenLastCalledWith({ show_ellipses: false });
});

it("marks overridden fields and offers a reset", () => {
  const onReset = vi.fn();
  const { root } = mount({ override: { marker_size: 30, groups: { A: { color: "#000000" } } }, onReset });
  expect(root.querySelectorAll(".style-group.overridden")).toHaveLength(1);
  expect(root.querySelector("#style-marker_size")!.closest(".field")!.classList.contains("overridden")).toBe(true);
  [...root.querySelectorAll("button")].find((b) => b.textContent === "Use global style")!.click();
  expect(onReset).toHaveBeenCalledOnce();
});

it("asks for data when there are no groups", () => {
  const { root } = mount({ effective: { ...STYLE, groups: {} } });
  expect(root.textContent).toContain("Import data to style its groups.");
});

it("applyPatch merges groups deeply", () => {
  const next = applyPatch(STYLE, { groups: { A: { symbol: "x" } }, font_size: 20 });
  expect(next.groups.A).toEqual({ color: "#112233", symbol: "x" });
  expect(next.groups.B).toEqual(STYLE.groups.B);
  expect(next.font_size).toBe(20);
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd GUI && npx vitest run renderer/components/StyleForm.test.tsx`
Expected: FAIL, cannot resolve `./StyleForm`.

- [ ] **Step 3: API types and calls**

In `GUI/renderer/api.ts`, add after the `TablePage` interface:

```ts
export interface GroupStyle {
  color: string;
  symbol: string;
}

export interface PlotStyle {
  groups: Record<string, GroupStyle>;
  marker_size: number;
  marker_opacity: number;
  font_size: number;
  line_width: number;
  show_ellipses: boolean;
}

export type StyleOverride = Partial<Omit<PlotStyle, "groups">> & { groups?: Record<string, Partial<GroupStyle>> };

export interface StageStyle {
  override: StyleOverride;
  effective: PlotStyle;
}
```

and add these methods to `class Api` after `restore()`:

```ts
  private put(path: string, body: unknown): Promise<any> {
    return this.req(path, { method: "PUT", body: JSON.stringify(body) }).then((r) => r.json());
  }

  getStyle(): Promise<PlotStyle> {
    return this.req("/style").then((r) => r.json());
  }

  putStyle(style: PlotStyle): Promise<PlotStyle> {
    return this.put("/style", style);
  }

  getStageStyle(id: string): Promise<StageStyle> {
    return this.req(`/stages/${id}/style`).then((r) => r.json());
  }

  putStageStyle(id: string, override: StyleOverride): Promise<StageStyle> {
    return this.put(`/stages/${id}/style`, override);
  }
```

- [ ] **Step 4: StyleForm**

Create `GUI/renderer/components/StyleForm.tsx`:

```tsx
import type { PlotStyle, StyleOverride } from "../api";

export const SYMBOLS = ["circle", "square", "diamond", "triangle-up", "triangle-down", "cross", "x", "star"];

type NumberKey = "marker_size" | "marker_opacity" | "font_size" | "line_width";
const NUMBERS: { key: NumberKey; label: string; min: number; max: number; step: number }[] = [
  { key: "marker_size", label: "Marker size", min: 4, max: 60, step: 1 },
  { key: "marker_opacity", label: "Marker opacity", min: 0.1, max: 1, step: 0.05 },
  { key: "font_size", label: "Font size", min: 8, max: 32, step: 1 },
  { key: "line_width", label: "Line width", min: 0.5, max: 5, step: 0.1 },
];

const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v));

export function applyPatch<T extends StyleOverride>(base: T, patch: StyleOverride): T {
  const groups: Record<string, any> = { ...(base.groups ?? {}) };
  for (const [name, g] of Object.entries(patch.groups ?? {})) groups[name] = { ...(groups[name] ?? {}), ...g };
  return { ...base, ...patch, groups };
}

interface Props {
  effective: PlotStyle;
  override?: StyleOverride;
  onChange: (patch: StyleOverride) => void;
  onReset?: () => void;
}

export function StyleForm({ effective, override, onChange, onReset }: Props) {
  const mark = (on: boolean) => (override && on ? " overridden" : "");
  const names = Object.keys(effective.groups);
  const customized =
    override && Object.entries(override).some(([k, v]) => (k === "groups" ? Object.keys(v ?? {}).length > 0 : v !== undefined));
  return (
    <div class="style-form">
      {names.length === 0 ? (
        <p class="hint">Import data to style its groups.</p>
      ) : (
        <div class="style-groups">
          {names.map((name) => {
            const g = effective.groups[name];
            return (
              <div key={name} class={`style-group${mark(!!override?.groups?.[name])}`}>
                <input
                  type="color"
                  aria-label={`${name} color`}
                  value={g.color}
                  onChange={(e) => onChange({ groups: { [name]: { color: e.currentTarget.value } } })}
                />
                <span class="style-name" title={name}>
                  {name}
                </span>
                <select
                  aria-label={`${name} symbol`}
                  value={g.symbol}
                  onChange={(e) => onChange({ groups: { [name]: { symbol: e.currentTarget.value } } })}
                >
                  {SYMBOLS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
            );
          })}
        </div>
      )}
      {NUMBERS.map((f) => (
        <div key={f.key} class={`field${mark(override?.[f.key] !== undefined)}`}>
          <label for={`style-${f.key}`}>{f.label}</label>
          <input
            id={`style-${f.key}`}
            type="number"
            min={f.min}
            max={f.max}
            step={f.step}
            value={effective[f.key]}
            onChange={(e) => {
              const v = Number(e.currentTarget.value);
              if (e.currentTarget.value !== "" && Number.isFinite(v)) onChange({ [f.key]: clamp(v, f.min, f.max) });
            }}
          />
        </div>
      ))}
      <label class={`check${mark(override?.show_ellipses !== undefined)}`}>
        <input
          type="checkbox"
          checked={effective.show_ellipses}
          onClick={(e) => onChange({ show_ellipses: e.currentTarget.checked })}
        />
        Confidence ellipses
      </label>
      {onReset && customized && (
        <button type="button" class="link" onClick={onReset}>
          Use global style
        </button>
      )}
    </div>
  );
}
```

Run: `cd GUI && npx vitest run renderer/components/StyleForm.test.tsx`
Expected: `6 passed`.

- [ ] **Step 5: ParamPanel children, Rail button, PlotView font**

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


R = "GUI/renderer/components/"
rep(R + "ParamPanel.tsx", [
    ('import type { FieldSchema, JsonSchema } from "../api";',
     'import type { ComponentChildren } from "preact";\nimport type { FieldSchema, JsonSchema } from "../api";'),
    ("  onValuesChange?: (values: Values) => void;\n}",
     "  onValuesChange?: (values: Values) => void;\n  children?: ComponentChildren;\n}"),
    ("picked = null, onValuesChange }: Props) {", "picked = null, onValuesChange, children }: Props) {"),
    ('''    <form
      class="panel"
      onSubmit={(e) => {''',
     '''    <div class="panel">
    <form
      class="panel-form"
      onSubmit={(e) => {'''),
    ('''        {busy ? "Running…" : "Run"}
      </button>
    </form>''',
     '''        {busy ? "Running…" : "Run"}
      </button>
    </form>
    {children}
    </div>'''),
])
rep(R + "Rail.tsx", [
    ("  onSave?: () => void;\n}", "  onSave?: () => void;\n  onStyle?: () => void;\n}"),
    ("export function Rail({ stages, current, onSelect, onOpen, onSave }: Props) {",
     "export function Rail({ stages, current, onSelect, onOpen, onSave, onStyle }: Props) {"),
    ("      {(onOpen || onSave) && (\n        <div class=\"rail-footer\">\n",
     "      {(onOpen || onSave || onStyle) && (\n        <div class=\"rail-footer\">\n"
     "          {onStyle && <button onClick={onStyle}>Plot style…</button>}\n"),
])
rep(R + "PlotView.tsx", [
    ('font: { ...layout.font, family: css.getPropertyValue("--font"), size: 12, color: text },',
     '// size comes from the plot style on the server\n    font: { ...layout.font, family: css.getPropertyValue("--font"), color: text },'),
])
print("components updated")
EOF
```

Append to `GUI/renderer/theme.css`:

```css
.panel-form { display: flex; flex-direction: column; gap: var(--s2); }
.style-form { display: flex; flex-direction: column; gap: var(--s2); }
.style-groups { display: flex; flex-direction: column; gap: 6px; }
.style-group { display: grid; grid-template-columns: 28px minmax(0, 1fr) 120px; align-items: center; gap: var(--s1); }
.style-group input[type="color"] { width: 28px; height: 28px; padding: 0; border: 1px solid var(--border); border-radius: 6px; background: none; cursor: pointer; }
.style-group select { font: inherit; color: inherit; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 4px 6px; }
.style-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.field.overridden > label, .style-group.overridden .style-name, .check.overridden { color: var(--accent); }
.check { display: flex; align-items: center; gap: var(--s1); color: var(--muted); font-size: 12px; }
.check input { accent-color: var(--accent); }
.style-override { border-top: 1px solid var(--border); padding-top: var(--s2); }
.style-override summary { cursor: pointer; color: var(--muted); font-size: 12px; margin-bottom: var(--s1); }
```

- [ ] **Step 6: App wiring**

Run from the repo root:

```bash
python3 - <<'EOF'
import pathlib

p = pathlib.Path("GUI/renderer/App.tsx")
s = p.read_text()
pairs = [
    ('import { Api, ApiError, type StageInfo, type TablePage } from "./api";',
     'import { Api, ApiError, type PlotStyle, type StageInfo, type StageStyle, type StyleOverride, type TablePage } from "./api";'),
    ('import { Rail } from "./components/Rail";\n',
     'import { Rail } from "./components/Rail";\nimport { StyleForm, applyPatch } from "./components/StyleForm";\n'),
    ("  const [showGuide, setShowGuide] = useState(false);\n",
     "  const [showGuide, setShowGuide] = useState(false);\n"
     '  const [sideMode, setSideMode] = useState<"params" | "style">("params");\n'
     "  const [globalStyle, setGlobalStyle] = useState<PlotStyle | null>(null);\n"
     "  const [stageStyle, setStageStyle] = useState<StageStyle | null>(null);\n"),
    ("  function select(id: string) {\n    setShowGuide(false);\n",
     '  function select(id: string) {\n    setShowGuide(false);\n    setSideMode("params");\n'),
    ("  function flash(message: string) {",
     '''  useEffect(() => {
    setStageStyle(null);
    if (!api || !stages.find((s) => s.id === current)?.views.length) return;
    let live = true;
    api.getStageStyle(current).then((st) => live && setStageStyle(st));
    return () => {
      live = false;
    };
  }, [api, current, version, globalStyle, stages.length]);

  function flash(message: string) {'''),
    ("  const openProject = () =>",
     '''  const openStyle = () =>
    guarded(async () => {
      if (!api) return;
      setGlobalStyle(await api.getStyle());
      setSideMode("style");
    });

  const patchGlobal = (patch: StyleOverride) =>
    guarded(async () => {
      if (!api || !globalStyle) return;
      setGlobalStyle(await api.putStyle(applyPatch(globalStyle, patch)));
      setVersion((v) => v + 1);
    });

  const patchStage = (patch: StyleOverride) =>
    guarded(async () => {
      if (!api || !stageStyle) return;
      setStageStyle(await api.putStageStyle(current, applyPatch(stageStyle.override, patch)));
      setVersion((v) => v + 1);
    });

  const resetStage = () =>
    guarded(async () => {
      if (!api) return;
      setStageStyle(await api.putStageStyle(current, {}));
      setVersion((v) => v + 1);
    });

  const openProject = () =>'''),
    ("onOpen={openProject} onSave={saveProject} />", "onOpen={openProject} onSave={saveProject} onStyle={openStyle} />"),
]
for a, b in pairs:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)

start = s.index("      <ParamPanel\n")
end = s.index("/>", start) + 2
panel = s[start:end]
opening = panel[: -2].rstrip() + "\n      >"
styled = f'''      {{sideMode === "style" && globalStyle ? (
        <div class="panel">
          <h2>Plot style</h2>
          <p class="hint">Applies to every plot and export. Stages can override it in their Style section.</p>
          <StyleForm effective={{globalStyle}} onChange={{patchGlobal}} />
          <button type="button" class="btn" onClick={{() => setSideMode("params")}}>
            Done
          </button>
        </div>
      ) : (
{opening}
        {{stageStyle && (
          <details class="style-override">
            <summary>Style{{Object.keys(stageStyle.override).length ? " (custom)" : ""}}</summary>
            <StyleForm
              effective={{stageStyle.effective}}
              override={{stageStyle.override}}
              onChange={{patchStage}}
              onReset={{resetStage}}
            />
          </details>
        )}}
      </ParamPanel>
      )}}'''
s = s[:start] + styled + s[end:]
p.write_text(s)
print("App wired")
EOF
```

- [ ] **Step 7: Verify**

Run: `cd GUI && npm run typecheck && npm test && npm run build`
Expected: typecheck exits 0; `32 passed`; build succeeds.

- [ ] **Step 8: Commit**

```bash
git add GUI/renderer
git commit -m "feat(gui): plot style panel and per-stage style overrides"
```

---

### Task 5: End-to-end styling

**Files:**
- Create: `GUI/e2e/style.spec.ts`

**Interfaces:**
- Consumes: `Plot style…` button, `B color` input, `Done`, `details.style-override`, `#style-marker_size`; Plotly's `gd.data` on `.plot`.
- Produces: `npm run e2e` runs 8 tests.

- [ ] **Step 1: Write the test**

Create `GUI/e2e/style.spec.ts`:

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

// read the rendered Plotly trace for a group straight from the graph div
const groupMarker = (win: Page, group: string, prop: "color" | "size") =>
  win.evaluate(
    ([g, p]) => {
      const gd = document.querySelector(".plot") as any;
      const trace = gd?.data?.find(
        (t: any) => String(t.name).split(",")[0].trim() === g && String(t.mode ?? "markers").includes("markers"),
      );
      return trace?.marker?.[p];
    },
    [group, prop] as const,
  );

test("global style recolors a group; a stage override changes marker size", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const env = { ...process.env, METBIT_USER_DATA: join(dir, "user") } as Record<string, string>;
  const app = await electron.launch({ args: ["."], env });
  const win = await app.firstWindow();
  const run = () => win.getByRole("button", { name: "Run", exact: true }).click();

  await win.fill("#f-path", makeCsv(dir));
  await run();
  await expect(win.locator("table.data")).toContainText("s11");
  await win.locator(".rail button", { hasText: "PCA" }).click();
  await run();
  await expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

  await win.getByRole("button", { name: "Plot style…" }).click();
  await win.getByLabel("B color").fill("#ff0000");
  await expect.poll(() => groupMarker(win, "B", "color")).toBe("#ff0000");
  await win.getByRole("button", { name: "Done" }).click();

  await win.locator("details.style-override summary").click();
  await win.fill("#style-marker_size", "30");
  await win.locator("#style-marker_size").press("Tab"); // number inputs report on change (blur)
  await expect.poll(() => groupMarker(win, "B", "size")).toBe(30);
  await expect(win.locator("details.style-override summary")).toContainText("(custom)");

  await app.close();
});
```

- [ ] **Step 2: Run everything**

Run from the repo root:
```bash
python3 -m pytest -q -p no:cacheprovider && (cd GUI && npm run typecheck && npm test && npm run e2e)
```
Expected: repo suite passes with coverage at or above 99%; typecheck exits 0; `32 passed`; `8 passed`.

- [ ] **Step 3: Commit**

```bash
git add GUI/e2e/style.spec.ts
git commit -m "test(gui): cover global and per-stage plot styling end to end"
```
