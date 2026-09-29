# metbit GUI Phase H (Linked Group-Test Boxplot) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the group median spectra under the Group tests boxplot; clicking a peak redraws the boxplot at that ppm (no re-run) with its statistics in the title.

**Architecture:** Figure requests accept an optional `ppm` query, forwarded by `render()` to figure functions that declare a `ppm` parameter. Stages may declare `companions` (main view -> companion view); the renderer hides companion views from the tab bar and draws them under their main view in a split layout, and a click on the companion sets `ppm` for both. Group tests gains a `spectra` companion built from the working matrix. Spec: `docs/superpowers/specs/2026-09-29-metbit-gui-style-annotation-stocsy-design.md` (section 6).

**Tech Stack:** FastAPI, Plotly, numpy; Preact, Playwright.

## Global Constraints

- All GUI code lives under `GUI/`; Python GUI tests under `tests/gui/`.
- Never use the em dash character or the section sign anywhere.
- Commit messages carry no `Co-Authored-By` trailer.
- Keep every source file under 500 lines.
- Boxplot title format (exact): `<ppm, 4 dp> ppm · <ANOVA | Kruskal-Wallis> p = <p, 2 significant digits>` plus ` · <FDR | Bonferroni> <p_adj, 2 significant digits>` when a correction is used; `n/a` for a missing p.
- The selected ppm snaps to the nearest tested (binned) feature; the companion marks that same ppm with a dotted vertical line.
- Split layout: main view 55%, companion 45% of the view height.
- Python GUI tests: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`. Renderer: `cd GUI && npm run typecheck && npm test`. E2E: `cd GUI && npm run e2e`.

---

### Task 1: `ppm` forwarding, companions, and the linked Group tests figures

**Files:**
- Modify: `GUI/server/metbit_server/stages/style.py`, `stages/base.py`, `stages/stats.py`, `stages/annotate.py`, `GUI/server/metbit_server/app.py`
- Test: `tests/gui/test_gui_group_linked.py`

**Interfaces:**
- Consumes: `render`, `spectra_figure`, `working`, `GroupTestResult`.
- Produces:
  - `render(make, result, style, **extra)`: passes each non-`None` extra only to figure functions whose signature names it.
  - `Stage.companions: dict[str, str]` (default `{}`); `/stages` items gain `companions`.
  - `base.median_spectra_figure(X, labels, title, style=None) -> go.Figure` (one line per group, reversed ppm axis); Annotate's median mode uses it.
  - `GroupTestResult` gains `correction: str = "none"` and `spectra: Any = None` (the working `Dataset`).
  - Group tests figures: `boxplot(r, style=None, ppm=None)`, `spectra(r, style=None, ppm=None)`; `companions={"boxplot": "spectra"}`, `ppm_views={"spectra": "x"}`.
  - `GET /stages/{id}/figure?view=&ppm=<float>`.

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_group_linked.py`:

```python
import numpy as np
import plotly.graph_objects as go
from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app
from metbit_server.stages import STAGES
from metbit_server.stages.style import PlotStyle, render

BY_ID = {s.id: s for s in STAGES}


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
    assert fig.layout.title.text == f"{target:.4f} ppm · ANOVA p = {row['p_value']:.2g} · FDR {row['p_adj']:.2g}"


def test_default_boxplot_is_the_top_feature_and_correction_can_be_off(run_all):
    r = run_all(("grouptests", {"bin_width": 0, "test": "kruskal", "correction": "none"})).output("grouptests")
    title = BY_ID["grouptests"].figures["boxplot"](r).layout.title.text
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_group_linked.py --no-cov -q -p no:warnings`
Expected: FAIL (`render()` rejects `ppm`; no `spectra` figure).

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


S = "GUI/server/metbit_server/"
rep(S + "stages/style.py", [
    ('''def render(make: Callable, result: Any, style: PlotStyle) -> go.Figure:
    """Build a figure with the effective style: native metbit arguments when supported, else the fallback."""
    native = _takes_style(make)
    fig = make(result, style) if native else make(result)''',
     '''def _declared(make: Callable, extra: dict) -> dict:
    """The extras (such as ppm) this figure function names in its signature; None means not given."""
    try:
        names = inspect.signature(make).parameters
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in extra.items() if v is not None and k in names}


def render(make: Callable, result: Any, style: PlotStyle, **extra: Any) -> go.Figure:
    """Build a figure with the effective style: native metbit arguments when supported, else the fallback."""
    native = _takes_style(make)
    kwargs = _declared(make, extra)
    fig = make(result, style, **kwargs) if native else make(result, **kwargs)'''),
])
rep(S + "stages/base.py", [
    ("import pandas as pd\n", "import numpy as np\nimport pandas as pd\n"),
    ("    pre_views: dict[str, Callable[..., go.Figure]] = field(default_factory=dict)\n",
     "    pre_views: dict[str, Callable[..., go.Figure]] = field(default_factory=dict)\n"
     "    # main view -> companion view drawn under it (hidden from the tab bar); a click on it sets ppm for both\n"
     "    companions: dict[str, str] = field(default_factory=dict)\n"),
    ("def _numeric_columns(",
     '''def median_spectra_figure(X: pd.DataFrame, labels: pd.Series, title: str, style: Optional[PlotStyle] = None) -> go.Figure:
    """One median line per group, display-decimated like the other spectrum views."""
    step = max(1, X.shape[1] // MAX_PLOT_POINTS)
    ppm = X.columns.to_numpy(dtype=float)[::step]
    palette = colors(style) if style else {}
    fig = go.Figure()
    for group in sorted(set(labels)):
        median = np.median(X[(labels == group).to_numpy()].to_numpy(), axis=0)[::step]
        line = {"color": palette.get(group), "width": style.line_width if style else 1.2}
        fig.add_trace(go.Scatter(x=ppm, y=median, name=group, mode="lines", line=line))
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis={"autorange": "reversed", "title": "δ¹H (ppm)"},
        yaxis={"title": "Intensity"},
    )
    return fig


def _numeric_columns('''),
])
rep(S + "stages/annotate.py", [
    ('''    X, labels = r.data.X, r.data.labels
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
    return fig''',
     '''    return median_spectra_figure(r.data.X, r.data.labels, "Median spectra", style)'''),
    ("from .base import DATA_ORDER, MAX_PLOT_POINTS, Dataset, Progress, Stage, spectra_figure, working",
     "from .base import DATA_ORDER, Dataset, Progress, Stage, median_spectra_figure, spectra_figure, working"),
])
rep(S + "stages/stats.py", [
    ("from .base import DATA_ORDER,", "from .base import median_spectra_figure\nfrom .base import DATA_ORDER,"),
    ('''    test: str
    feature: str         # column of `data` shown in the boxplot
''',
     '''    test: str
    feature: str         # column of `data` shown in the boxplot by default
    correction: str = "none"
    spectra: Any = None  # working Dataset at full resolution, for the companion spectrum
'''),
    ("    return GroupTestResult(table=table, data=data, test=p.test, feature=str(target))",
     "    return GroupTestResult(\n"
     "        table=table, data=data, test=p.test, feature=str(target), correction=p.correction, spectra=working(session)\n"
     "    )"),
    ('''def _boxplot(r: GroupTestResult, style=None) -> go.Figure:
    cls = ANOVAStats if r.test == "anova" else KruskalStats
    stats = cls(r.data, x_col="Group", y_col=r.feature)
    stats.fit()
    extra = {"custom_colors": colors(style), "font_size": style.font_size} if style else {}
    return stats.plot(title=f"{float(r.feature):.4f} ppm", **extra)''',
     '''CORRECTION_NAMES = {"fdr_bh": "FDR", "bonferroni": "Bonferroni"}


def _feature_for(r: GroupTestResult, ppm: Optional[float]) -> str:
    """Column of the tested (binned) matrix nearest `ppm`; the default feature when none is given."""
    if ppm is None:
        return r.feature
    names = [c for c in r.data.columns if c != "Group"]
    values = np.array([float(c) for c in names])
    return names[int(np.abs(values - ppm).argmin())]


def _stat_title(r: GroupTestResult, feature: str) -> str:
    fmt = lambda v: "n/a" if v != v else f"{v:.2g}"  # NaN != NaN
    row = r.table.loc[np.isclose(r.table["ppm"], float(feature))].iloc[0]
    title = f"{float(feature):.4f} ppm · {'ANOVA' if r.test == 'anova' else 'Kruskal-Wallis'} p = {fmt(row['p_value'])}"
    if r.correction in CORRECTION_NAMES:
        title += f" · {CORRECTION_NAMES[r.correction]} {fmt(row['p_adj'])}"
    return title


def _boxplot(r: GroupTestResult, style=None, ppm: Optional[float] = None) -> go.Figure:
    feature = _feature_for(r, ppm)
    cls = ANOVAStats if r.test == "anova" else KruskalStats
    stats = cls(r.data, x_col="Group", y_col=feature)
    stats.fit()
    extra = {"custom_colors": colors(style), "font_size": style.font_size} if style else {}
    return stats.plot(title=_stat_title(r, feature), **extra)


def _group_spectra(r: GroupTestResult, style=None, ppm: Optional[float] = None) -> go.Figure:
    fig = median_spectra_figure(r.spectra.X, r.spectra.labels, "Click a peak to show its boxplot", style)
    fig.add_vline(x=float(_feature_for(r, ppm)), line_dash="dot", line_color="#888888")
    return fig'''),
    ('''        figures={"boxplot": _boxplot},
        requires="import",''',
     '''        figures={"boxplot": _boxplot, "spectra": _group_spectra},
        companions={"boxplot": "spectra"},
        ppm_views={"spectra": "x"},
        requires="import",'''),
    ("from typing import Literal, Optional\n", "from typing import Any, Literal, Optional\n"),
])
rep(S + "app.py", [
    ('                "pre_views": list(s.pre_views),\n',
     '                "pre_views": list(s.pre_views),\n                "companions": s.companions,\n'),
    ("    def figure(sid: str, view: str):", "    def figure(sid: str, view: str, ppm: Optional[float] = None):"),
    ("            fig = render(stage.pre_views[view], data, style)", "            fig = render(stage.pre_views[view], data, style, ppm=ppm)"),
    ("            fig = render(make, out, style)", "            fig = render(make, out, style, ppm=ppm)"),
])
print("server wired")
EOF
```

Note: `base.py` must import `PlotStyle` and `colors` for `median_spectra_figure`; it already imports them from `.style` (Phase E).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`
Expected: `129 passed` (127 passed and 2 skipped without MTBLS2052).

- [ ] **Step 5: Commit**

```bash
git add GUI/server tests/gui/test_gui_group_linked.py
git commit -m "feat(gui): group-test boxplot follows a ppm, with a companion spectrum and stats in the title"
```

---

### Task 2: Split view in the renderer

**Files:**
- Modify: `GUI/renderer/api.ts`, `GUI/renderer/App.tsx`, `GUI/renderer/theme.css`, `GUI/renderer/components/Rail.test.tsx`
- Create: `GUI/e2e/group-linked.spec.ts`

**Interfaces:**
- Consumes: `/stages` `companions`; figure `ppm` query.
- Produces: `StageInfo.companions`; `Api.figure(id, view, ppm?)`; App state `focusPpm`, `companionFigure`; `.split`, `.split-main`, `.split-companion` layout.

- [ ] **Step 1: Wire it**

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
    ("  pre_views: string[];\n", "  pre_views: string[];\n  companions: Record<string, string>;\n"),
    ('''  figure(id: string, view: string): Promise<any> {
    return this.req(`/stages/${id}/figure?view=${encodeURIComponent(view)}`).then((r) => r.json());''',
     '''  figure(id: string, view: string, ppm?: number): Promise<any> {
    const at = ppm === undefined ? "" : `&ppm=${ppm}`;
    return this.req(`/stages/${id}/figure?view=${encodeURIComponent(view)}${at}`).then((r) => r.json());'''),
])
rep(R + "components/Rail.test.tsx", [("  pre_views: [],\n", "  pre_views: [],\n  companions: {},\n")])
rep(R + "App.tsx", [
    ("  const [figure, setFigure] = useState<any>(null);\n",
     "  const [figure, setFigure] = useState<any>(null);\n"
     "  const [companionFigure, setCompanionFigure] = useState<any>(null);\n"
     "  const [focusPpm, setFocusPpm] = useState<number | null>(null); // set by clicking a companion view\n"),
    ('  const tabs = stage ? [...preViews, ...(hasResult ? [...stage.views, ...(stage.has_table ? ["table"] : [])] : [])] : [];',
     '  const companion = stage?.companions?.[tab];\n'
     '  const companionViews = Object.values(stage?.companions ?? {});\n'
     '  const resultTabs = stage ? [...stage.views.filter((v) => !companionViews.includes(v)), ...(stage.has_table ? ["table"] : [])] : [];\n'
     '  const tabs = stage ? [...preViews, ...(hasResult ? resultTabs : [])] : [];'),
    ("    setFigure(null);\n", "    setFigure(null);\n    setCompanionFigure(null);\n"),
    ('    const load = tab === "table" ? api.table(current).then((p) => live && setPage(p)) : api.figure(current, tab).then((f) => live && setFigure(f));',
     '    const at = focusPpm ?? undefined;\n'
     '    const load = tab === "table" ? api.table(current).then((p) => live && setPage(p)) : api.figure(current, tab, at).then((f) => live && setFigure(f));\n'
     '    if (companion && hasResult) api.figure(current, companion, at).then((f) => live && setCompanionFigure(f));'),
    ("  }, [api, current, tab, hasResult, version]);", "  }, [api, current, tab, hasResult, version, focusPpm]);"),
    ("    setInjected(null);\n", "    setInjected(null);\n    setFocusPpm(null);\n"),
    ("      if (ran.pre_views.includes(tab) && ran.views[0]) setTab(ran.views[0]);",
     "      if (ran.pre_views.includes(tab) && ran.views[0]) setTab(ran.views[0]);\n      setFocusPpm(null); // a new result starts from its own default feature"),
    ('''  } else {
    body = figure && <PlotView figure={figure} onPick={onPick} />;
  }''',
     '''  } else if (companion) {
    const pickCompanion = (pt: PickPoint) => {
      const v = Number(pt.x);
      if (Number.isFinite(v)) setFocusPpm(v);
    };
    body = (
      <div class="split">
        <div class="split-main">{figure && <PlotView figure={figure} onPick={onPick} />}</div>
        <div class="split-companion">{companionFigure && <PlotView figure={companionFigure} onPick={pickCompanion} />}</div>
      </div>
    );
  } else {
    body = figure && <PlotView figure={figure} onPick={onPick} />;
  }'''),
])
print("renderer wired")
EOF
```

Append to `GUI/renderer/theme.css`:

```css
.split { position: absolute; inset: 0; display: grid; grid-template-rows: 55fr 45fr; }
.split > div { position: relative; min-height: 0; }
.split-companion { border-top: 1px solid var(--border); }
```

- [ ] **Step 2: E2E test**

Create `GUI/e2e/group-linked.spec.ts`:

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

const boxTitle = (win: Page) =>
  win.evaluate(() => (document.querySelector(".split-main .plot") as any)?.layout?.title?.text ?? "");

test("click the spectrum under the boxplot to test another ppm", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const env = { ...process.env, METBIT_USER_DATA: join(dir, "user") } as Record<string, string>;
  const app = await electron.launch({ args: ["."], env });
  const win = await app.firstWindow();
  const run = () => win.getByRole("button", { name: "Run", exact: true }).click();

  await win.fill("#f-path", makeCsv(dir));
  await run();
  await expect(win.locator("table.data")).toContainText("s11");
  await win.locator(".rail button", { hasText: "Group tests" }).click();
  await win.fill("#f-bin_width", "0");
  await run();

  await expect(win.locator(".split-companion .plot .main-svg").first()).toBeVisible({ timeout: 30_000 });
  await expect(win.getByRole("tab", { name: "Spectra" })).toHaveCount(0); // companion is not a tab
  await expect.poll(() => boxTitle(win)).toContain("8.7500 ppm · ANOVA p = "); // top feature: the spiked peak
  const box = (await win.locator(".split-companion .nsewdrag").first().boundingBox())!;
  await win.mouse.move(box.x + box.width * 0.75, box.y + box.height * 0.5);
  await win.mouse.click(box.x + box.width * 0.75, box.y + box.height * 0.5);
  await expect.poll(() => boxTitle(win)).not.toContain("8.7500 ppm");
  await expect.poll(() => boxTitle(win)).toContain(" ppm · ANOVA p = ");

  await app.close();
});
```

- [ ] **Step 3: Verify**

Run from the repo root:
```bash
python3 -m pytest -q -p no:cacheprovider && (cd GUI && npm run typecheck && npm test && npm run e2e)
```
Expected: repo suite passes (coverage at or above 99%); typecheck exits 0; `46 passed`; `12 passed`.

- [ ] **Step 4: Commit**

```bash
git add GUI/renderer GUI/e2e/group-linked.spec.ts
git commit -m "feat(gui): split Boxplot view with a clickable spectrum that picks the tested ppm"
```
