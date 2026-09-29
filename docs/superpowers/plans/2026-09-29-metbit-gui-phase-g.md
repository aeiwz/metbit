# metbit GUI Phase G (STOCSY Rework) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the STOCSY stage work like metbit's Dash app: spectra visible before any run, click a peak to set the anchor, Run, cached results per anchor and threshold, a Recent anchors list and a Clear action.

**Architecture:** Stages gain `pre_views`, figures rendered from the working matrix without a result; STOCSY declares a seeded `spectra` sample like the Dash app. Results are cached on the `Dataset` object itself, so new data automatically starts an empty cache. The renderer shows pre-views before a run, injects the clicked ppm into the form (`ParamPanel.injected`), switches to Correlation after a successful run, and keeps a Recent anchors list. The correlation figure is now copied per request, fixing labels piling up on the stored figure. Spec: `docs/superpowers/specs/2026-09-29-metbit-gui-style-annotation-stocsy-design.md` (section 3).

**Tech Stack:** FastAPI, numpy, Plotly; Preact, Vitest, Playwright.

## Global Constraints

- All GUI code lives under `GUI/`; Python GUI tests under `tests/gui/`.
- Never use the em dash character or the section sign anywhere.
- Commit messages carry no `Co-Authored-By` trailer.
- Keep every source file under 500 lines.
- Spectra sample (as metbit's Dash app): all spectra when 5 or fewer; 20% when 100 or more; otherwise up to 30; seeded (`numpy.random.default_rng(0)`) so the view does not jump between loads.
- Cache key: `(anchor_ppm, p_threshold)`, stored on the working `Dataset` (`Dataset.cache["stocsy"]`); a data stage re-run produces a new `Dataset` and so an empty cache.
- Stored results are never mutated by figure rendering (figure functions return copies of stored figures).
- Recent anchors: newest first, unique by `(anchor, p)`, at most 8, renderer-only.
- Python GUI tests: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`. Renderer: `cd GUI && npm run typecheck && npm test`. E2E: `cd GUI && npm run e2e`.

---

### Task 1: Pre-views, STOCSY spectra sample, cache and copy-on-render

**Files:**
- Modify: `GUI/server/metbit_server/stages/base.py`, `GUI/server/metbit_server/stages/analysis.py`, `GUI/server/metbit_server/app.py`
- Test: `tests/gui/test_gui_stocsy_rework.py`

**Interfaces:**
- Consumes: `working`, `spectra_figure`, `Dataset`, `render`, `overlay`.
- Produces:
  - `Stage.pre_views: dict[str, Callable[[Dataset, style], go.Figure]]` (default `{}`).
  - `Dataset.cache: dict` (not compared, not shown in repr).
  - `analysis.sample_indices(n: int) -> list[int]`, `analysis.stocsy_spectra(d: Dataset, style=None) -> go.Figure`.
  - STOCSY stage: `pre_views={"spectra": stocsy_spectra}`, `ppm_views={"spectra": "x", "correlation": "x"}`, `figures={"correlation": <copy of stored figure>}`.
  - `/stages` items gain `pre_views: list[str]`. `GET /stages/{id}/figure?view=<pre view>` renders from the working matrix (409 with the working-matrix message when there is none).

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_stocsy_rework.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_stocsy_rework.py --no-cov -q -p no:warnings`
Expected: FAIL with `ImportError: cannot import name 'sample_indices'`.

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
rep(S + "stages/base.py", [
    ("    export: Optional[Callable[[Any], pd.DataFrame]] = None\n",
     "    export: Optional[Callable[[Any], pd.DataFrame]] = None\n"
     "    # views drawn from the working matrix before the stage has a result (called with the Dataset)\n"
     "    pre_views: dict[str, Callable[..., go.Figure]] = field(default_factory=dict)\n"),
    ("    before: Optional[pd.DataFrame] = None  # input matrix, for before / after views\n",
     "    before: Optional[pd.DataFrame] = None  # input matrix, for before / after views\n"
     "    # per-matrix result cache (STOCSY): new data is a new Dataset, so nothing stale survives\n"
     "    cache: dict = field(default_factory=dict, repr=False, compare=False)\n"),
])
rep(S + "stages/analysis.py", [
    ("import plotly.graph_objects as go\n", "import numpy as np\nimport plotly.graph_objects as go\n"),
    ("from .base import DATA_ORDER, Progress, Stage, groups, not_run_figure, two_groups, working",
     "from .base import DATA_ORDER, Dataset, Progress, Stage, groups, not_run_figure, spectra_figure, two_groups, working"),
    ('''def run_stocsy(p: STOCSYParams, session: Session, progress: Progress) -> go.Figure:
    X = working(session).X
    lo, hi = float(X.columns.min()), float(X.columns.max())
    if not lo <= p.anchor_ppm <= hi:
        raise ValueError(f"Anchor {p.anchor_ppm} ppm is outside the spectrum ({lo:.2f} to {hi:.2f} ppm)")
    fig = STOCSY(X, p.anchor_ppm, p.p_threshold)
    progress(1.0)
    return fig''',
     '''def run_stocsy(p: STOCSYParams, session: Session, progress: Progress) -> go.Figure:
    data = working(session)
    X = data.X
    lo, hi = float(X.columns.min()), float(X.columns.max())
    if not lo <= p.anchor_ppm <= hi:
        raise ValueError(f"Anchor {p.anchor_ppm} ppm is outside the spectrum ({lo:.2f} to {hi:.2f} ppm)")
    cache = data.cache.setdefault("stocsy", {})  # like the Dash app's lru_cache, scoped to this matrix
    key = (p.anchor_ppm, p.p_threshold)
    if key not in cache:
        cache[key] = STOCSY(X, p.anchor_ppm, p.p_threshold)
    progress(1.0)
    return cache[key]


def sample_indices(n: int) -> list[int]:
    """Spectra to draw, as metbit's STOCSY app: all up to 5, 20% from 100, else up to 30; seeded."""
    if n <= 5:
        return list(range(n))
    k = int(0.2 * n) if n >= 100 else min(30, n)
    return sorted(np.random.default_rng(0).choice(n, size=k, replace=False).tolist())


def stocsy_spectra(d: Dataset, style=None) -> go.Figure:
    idx = sample_indices(len(d.X))
    return spectra_figure(d.X.iloc[idx], d.labels.iloc[idx], "Click a peak to set the anchor", style)'''),
    ('''        figures={"correlation": lambda fig: fig},
        requires="import",
        after=list(DATA_ORDER),
        ppm_views={"correlation": "x"},''',
     '''        # a copy: rendering restyles and labels the figure, and the stored one is also the cache entry
        figures={"correlation": lambda fig: go.Figure(fig)},
        pre_views={"spectra": stocsy_spectra},
        requires="import",
        after=list(DATA_ORDER),
        ppm_views={"spectra": "x", "correlation": "x"},'''),
])
rep(S + "app.py", [
    ("from .stages.base import option_values\n", "from .stages.base import option_values, working\n"),
    ('                "views": list(s.figures),\n',
     '                "views": list(s.figures),\n                "pre_views": list(s.pre_views),\n'),
    ('''        out = result_or_409(sid)
        make = by_id[sid].figures.get(view)
        if make is None:
            raise HTTPException(404, f"Unknown view '{view}'")
        style = effective_style(sid)
        fig = render(make, out, style)''',
     '''        stage = stage_or_404(sid)
        style = effective_style(sid)
        if view in stage.pre_views:
            try:
                data = working(session)  # pre-run views draw from the working matrix, no result needed
            except ValueError as e:
                raise HTTPException(409, str(e))
            fig = render(stage.pre_views[view], data, style)
        else:
            out = result_or_409(sid)
            make = stage.figures.get(view)
            if make is None:
                raise HTTPException(404, f"Unknown view '{view}'")
            fig = render(make, out, style)'''),
])
print("server wired")
EOF
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`
Expected: `118 passed` (116 passed and 2 skipped without MTBLS2052).

- [ ] **Step 5: Commit**

```bash
git add GUI/server tests/gui/test_gui_stocsy_rework.py
git commit -m "feat(gui): STOCSY pre-run spectra, per-matrix result cache, and copy-on-render"
```

---

### Task 2: Click-to-anchor, Clear, Recent anchors

**Files:**
- Create: `GUI/renderer/components/StocsyTools.tsx`, `GUI/renderer/components/StocsyTools.test.tsx`
- Modify: `GUI/renderer/api.ts`, `GUI/renderer/components/ParamPanel.tsx`, `GUI/renderer/components/ParamPanel.test.tsx`, `GUI/renderer/components/Rail.test.tsx`, `GUI/renderer/App.tsx`

**Interfaces:**
- Consumes: `/stages` field `pre_views`; STOCSY `spectra` pre-view.
- Produces:
  - `StageInfo.pre_views: string[]`.
  - `ParamPanel` prop `injected?: { values: Values; n: number } | null`: merged into the form whenever `n` changes, keeping other edits.
  - `StocsyTools({ recent, onUse(anchor), onClear })`, root `.stocsy-tools`; buttons `<ppm to 3 dp> ppm` and `Clear anchor`.
  - App: pre-view tabs before a run; click on STOCSY spectra fills Anchor; after a successful run the view moves from a pre-view to the first result view; Recent anchors re-run instantly (server cache).

- [ ] **Step 1: Write the failing tests**

Create `GUI/renderer/components/StocsyTools.test.tsx`:

```tsx
import { render } from "preact";
import { expect, it, vi } from "vitest";
import { StocsyTools } from "./StocsyTools";

function mount(recent: { anchor_ppm: number; p_threshold: number }[]) {
  const root = document.createElement("div");
  const onUse = vi.fn();
  const onClear = vi.fn();
  render(<StocsyTools recent={recent} onUse={onUse} onClear={onClear} />, root);
  return { root, onUse, onClear };
}

const button = (root: HTMLElement, text: string) =>
  [...root.querySelectorAll("button")].find((b) => b.textContent === text) as HTMLButtonElement;

it("lists recent anchors and reuses one", () => {
  const { root, onUse } = mount([{ anchor_ppm: 3.03, p_threshold: 0.0001 }]);
  button(root, "3.030 ppm").click();
  expect(onUse).toHaveBeenCalledWith({ anchor_ppm: 3.03, p_threshold: 0.0001 });
});

it("clears the anchor", () => {
  const { root, onClear } = mount([]);
  button(root, "Clear anchor").click();
  expect(onClear).toHaveBeenCalledOnce();
});

it("explains the workflow when nothing has run yet", () => {
  expect(mount([]).root.textContent).toContain("Click a peak on the spectra, then Run.");
});
```

Append to `GUI/renderer/components/ParamPanel.test.tsx`:

```tsx
it("merges injected values without losing other edits", async () => {
  const stocsy: JsonSchema = {
    properties: {
      anchor_ppm: { type: "number", title: "Anchor (ppm)", pick_ppm: true },
      p_threshold: { type: "number", title: "p", default: 0.0001 },
    },
  };
  const root = document.createElement("div");
  const onRun = vi.fn();
  const view = (injected: { values: Record<string, unknown>; n: number } | null) =>
    render(<ParamPanel title="STOCSY" schema={stocsy} initial={null} errors={{}} busy={false} onRun={onRun} injected={injected} />, root);
  view(null);
  const p = root.querySelector<HTMLInputElement>("#f-p_threshold")!;
  p.value = "0.01";
  p.dispatchEvent(new Event("input"));
  await tick();
  view({ values: { anchor_ppm: 3.03 }, n: 1 });
  await vi.waitFor(() => expect(root.querySelector<HTMLInputElement>("#f-anchor_ppm")!.value).toBe("3.03"));
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  expect(onRun).toHaveBeenCalledWith({ anchor_ppm: 3.03, p_threshold: 0.01 });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd GUI && npx vitest run renderer/components/StocsyTools.test.tsx renderer/components/ParamPanel.test.tsx`
Expected: FAIL (`./StocsyTools` missing; injected values ignored).

- [ ] **Step 3: StocsyTools and ParamPanel injection**

Create `GUI/renderer/components/StocsyTools.tsx`:

```tsx
export interface Anchor {
  anchor_ppm: number;
  p_threshold: number;
}

interface Props {
  recent: Anchor[];
  onUse: (anchor: Anchor) => void;
  onClear: () => void;
}

export function StocsyTools({ recent, onUse, onClear }: Props) {
  return (
    <div class="annotations stocsy-tools">
      <div class="row">
        <button type="button" class="link" onClick={onClear}>
          Clear anchor
        </button>
      </div>
      <h3>Recent anchors</h3>
      {recent.length === 0 ? (
        <p class="hint">Click a peak on the spectra, then Run.</p>
      ) : (
        <ul class="label-list">
          {recent.map((a) => (
            <li key={`${a.anchor_ppm}:${a.p_threshold}`}>
              <button type="button" class="link" onClick={() => onUse(a)}>
                {a.anchor_ppm.toFixed(3)} ppm
              </button>
              <span class="ppm-tag">p {a.p_threshold}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

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
rep(R + "components/ParamPanel.tsx", [
    ("  onValuesChange?: (values: Values) => void;\n",
     "  onValuesChange?: (values: Values) => void;\n"
     "  injected?: { values: Values; n: number } | null; // merged in when n changes (a click set a field)\n"),
    ("picked = null, onValuesChange, children }: Props) {", "picked = null, onValuesChange, children, injected }: Props) {"),
    ("  useEffect(() => onValuesChange?.(values), [values]);\n",
     "  useEffect(() => onValuesChange?.(values), [values]);\n"
     "  useEffect(() => {\n    if (injected) setValues((s) => ({ ...s, ...injected.values }));\n  }, [injected?.n]);\n"),
])
rep(R + "api.ts", [("  views: string[];\n", "  views: string[];\n  pre_views: string[];\n")])
rep(R + "components/Rail.test.tsx", [("  views: [],\n", "  views: [],\n  pre_views: [],\n")])
print("panel and types updated")
EOF
```

Run: `cd GUI && npx vitest run renderer/components/StocsyTools.test.tsx renderer/components/ParamPanel.test.tsx`
Expected: all pass.

- [ ] **Step 4: App wiring**

Run from the repo root:

```bash
python3 - <<'EOF'
import pathlib

p = pathlib.Path("GUI/renderer/App.tsx")
s = p.read_text()
pairs = [
    ('import { Rail } from "./components/Rail";\n',
     'import { Rail } from "./components/Rail";\nimport { StocsyTools, type Anchor } from "./components/StocsyTools";\n'),
    ("  const [pending, setPending] = useState<{ ppm: number; x: number; y: number } | null>(null);\n",
     "  const [pending, setPending] = useState<{ ppm: number; x: number; y: number } | null>(null);\n"
     "  const [injected, setInjected] = useState<{ values: Values; n: number } | null>(null);\n"
     "  const [recent, setRecent] = useState<Anchor[]>([]); // STOCSY anchors run this session, newest first\n"),
    ('  const tabs = stage ? [...stage.views, ...(stage.has_table ? ["table"] : [])] : [];',
     '  const preViews = stage?.pre_views ?? [];\n'
     '  const tabs = stage ? [...preViews, ...(hasResult ? [...stage.views, ...(stage.has_table ? ["table"] : [])] : [])] : [];'),
    ('''          setPending({ ppm: v, x: pt.clientX, y: pt.clientY });
        }''',
     '''          setPending({ ppm: v, x: pt.clientX, y: pt.clientY });
        }
        // on the STOCSY spectra a click sets the anchor, as in metbit's STOCSY app
        if (current === "stocsy" && tab === "spectra") setInjected({ values: { anchor_ppm: Number(v.toFixed(4)) }, n: Date.now() });'''),
    ("    if (!api || !hasResult) return;", "    if (!api || (!hasResult && !preViews.includes(tab))) return;"),
    ("    setStageStyle(null); // the next stage's style loads fresh\n    setPending(null);\n",
     "    setStageStyle(null); // the next stage's style loads fresh\n    setPending(null);\n    setInjected(null);\n"),
    ('    setTab(stages.find((s) => s.id === id)?.views[0] ?? "table");',
     '    const next = stages.find((s) => s.id === id);\n    setTab(next?.pre_views[0] ?? next?.views[0] ?? "table");'),
    ('''    setPct(null);
    setStages(await api.stages());
    setVersion((v) => v + 1);
  }''',
     '''    setPct(null);
    const fresh = await api.stages();
    setStages(fresh);
    setVersion((v) => v + 1);
    const ran = fresh.find((s) => s.id === current);
    if (ran?.status === "done") {
      // leave a pre-run view for the result it prepared (STOCSY: spectra to correlation)
      if (ran.pre_views.includes(tab) && ran.views[0]) setTab(ran.views[0]);
      if (current === "stocsy") {
        const a = { anchor_ppm: Number(values.anchor_ppm), p_threshold: Number(values.p_threshold) };
        setRecent((r) => [a, ...r.filter((x) => x.anchor_ppm !== a.anchor_ppm || x.p_threshold !== a.p_threshold)].slice(0, 8));
      }
    }
  }'''),
    ("  } else if (!hasResult) {",
     '''  } else if (stage.status !== "running" && preViews.includes(tab)) {
    body = figure && <PlotView figure={figure} onPick={onPick} />;
  } else if (!hasResult) {'''),
    ("          {hasResult &&\n", "          {tabs.length > 0 &&\n"),
    ('        onValuesChange={current === "import" ? setImportValues : undefined}\n',
     '        onValuesChange={current === "import" ? setImportValues : undefined}\n        injected={injected}\n'),
    ('        {current === "annotate" && (',
     '''        {current === "stocsy" && (
          <StocsyTools
            recent={recent}
            onUse={(a) => {
              setInjected({ values: { ...a }, n: Date.now() });
              run({ ...a });
            }}
            onClear={() => setInjected({ values: { anchor_ppm: "" }, n: Date.now() })}
          />
        )}
        {current === "annotate" && ('''),
]
for a, b in pairs:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
p.write_text(s)
print("App wired")
EOF
```

- [ ] **Step 5: Verify**

Run: `cd GUI && npm run typecheck && npm test && npm run build`
Expected: typecheck exits 0; `44 passed`; build succeeds.

- [ ] **Step 6: Commit**

```bash
git add GUI/renderer
git commit -m "feat(gui): click a spectrum to set the STOCSY anchor, with Clear and Recent anchors"
```

---

### Task 3: End-to-end STOCSY

**Files:**
- Create: `GUI/e2e/stocsy.spec.ts`

**Interfaces:**
- Consumes: rail `STOCSY`; tabs `Spectra`, `Correlation`; `#f-anchor_ppm`; `.stocsy-tools`; button `Clear anchor`.
- Produces: `npm run e2e` runs 10 tests.

- [ ] **Step 1: Write the test**

Create `GUI/e2e/stocsy.spec.ts`:

```ts
import { _electron as electron, expect, test } from "@playwright/test";
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

test("STOCSY: click the spectra to set the anchor, run, reuse and clear", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const env = { ...process.env, METBIT_USER_DATA: join(dir, "user") } as Record<string, string>;
  const app = await electron.launch({ args: ["."], env });
  const win = await app.firstWindow();
  const run = () => win.getByRole("button", { name: "Run", exact: true }).click();
  const plot = () => expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

  await win.fill("#f-path", makeCsv(dir));
  await run();
  await expect(win.locator("table.data")).toContainText("s11");

  await win.locator(".rail button", { hasText: "STOCSY" }).click();
  await expect(win.getByRole("tab", { name: "Spectra" })).toBeVisible();
  await plot(); // spectra before any run

  const box = (await win.locator(".plot .nsewdrag").first().boundingBox())!;
  await win.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.3);
  await win.mouse.click(box.x + box.width * 0.3, box.y + box.height * 0.3);
  await expect.poll(() => win.inputValue("#f-anchor_ppm")).not.toBe("");

  await run();
  await expect(win.getByRole("tab", { name: "Correlation" })).toHaveAttribute("aria-selected", "true");
  await plot();
  await expect(win.locator(".stocsy-tools .label-list")).toContainText("ppm");

  await win.getByRole("button", { name: "Clear anchor" }).click();
  await expect(win.locator("#f-anchor_ppm")).toHaveValue("");

  await app.close();
});
```

- [ ] **Step 2: Run everything**

Run from the repo root:
```bash
python3 -m pytest -q -p no:cacheprovider && (cd GUI && npm run typecheck && npm test && npm run e2e)
```
Expected: repo suite passes with coverage at or above 99%; typecheck exits 0; `44 passed`; `10 passed`.

- [ ] **Step 3: Commit**

```bash
git add GUI/e2e/stocsy.spec.ts
git commit -m "test(gui): cover the STOCSY click-to-anchor workflow end to end"
```
