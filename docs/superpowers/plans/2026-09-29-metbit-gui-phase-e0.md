# metbit GUI Phase E0 (Import Format Guide) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show users, inside the Import stage, mock-up visuals of compliant input files and how to prepare them (processed data table, raw Bruker + metadata), plus a one-click synthetic example CSV.

**Architecture:** Renderer-only. `exampleData.ts` generates a deterministic example CSV. `FormatGuide.tsx` renders static mock-ups (an HTML table for the CSV; an aligned folder-tree / metadata grid for Bruker) and numbered checklists. `ParamPanel` reports live form values through `onValuesChange`, so `App` shows the guide matching the chosen Source. The example is saved through the existing `saveImage` data-URL writer and then prefilled into the form. Spec: `docs/superpowers/specs/2026-09-29-metbit-gui-style-annotation-stocsy-design.md` (section 4).

**Tech Stack:** Preact, TypeScript, plain CSS, Vitest, Playwright.

## Global Constraints

- All GUI code lives under `GUI/`.
- Never use the em dash character or the section sign anywhere.
- Commit messages carry no `Co-Authored-By` trailer.
- Keep every source file under 500 lines.
- Visual language: tokens from `theme.css` only (`--bg`, `--surface`, `--border`, `--text`, `--muted`, `--accent`), hairline borders, no shadows, works in light and dark.
- The example CSV must pass the importer: `Sample`, `Group` (and `Age`) metadata headers, every other header numeric, no missing values.
- Renderer: `cd GUI && npm run typecheck && npm test`. E2E: `cd GUI && npm run e2e`.

## File Map

```
GUI/renderer/exampleData.ts                 NEW: makeExampleCsv()
GUI/renderer/exampleData.test.ts            NEW
GUI/renderer/components/FormatGuide.tsx     NEW: CSV and Bruker mock-ups + checklists
GUI/renderer/components/FormatGuide.test.tsx NEW
GUI/renderer/components/ParamPanel.tsx      MODIFY: onValuesChange
GUI/renderer/components/ParamPanel.test.tsx MODIFY: one test
GUI/renderer/App.tsx                        MODIFY: guide in Import view, example save, File format link
GUI/renderer/theme.css                      MODIFY: guide styles
GUI/electron/main.ts                        MODIFY: CSV filter name for data-URL saves
GUI/e2e/guide.spec.ts                       NEW
```

---

### Task 1: Deterministic example CSV

**Files:**
- Create: `GUI/renderer/exampleData.ts`
- Test: `GUI/renderer/exampleData.test.ts`

**Interfaces:**
- Produces: `makeExampleCsv(samples = 12, points = 400): string`, header `Sample,Group,Age,<ppm...>` with ppm from 10 to -0.5 formatted to 4 dp; half `Control`, half `Treated`; ends with a newline; identical output on every call.

- [ ] **Step 1: Write the failing test**

Create `GUI/renderer/exampleData.test.ts`:

```ts
import { expect, it } from "vitest";
import { makeExampleCsv } from "./exampleData";

it("writes a table the importer accepts", () => {
  const lines = makeExampleCsv().trim().split("\n");
  expect(lines).toHaveLength(13);
  const header = lines[0].split(",");
  expect(header.slice(0, 3)).toEqual(["Sample", "Group", "Age"]);
  const ppm = header.slice(3);
  expect(ppm).toHaveLength(400);
  expect(ppm.every((h) => Number.isFinite(Number(h)))).toBe(true);
  expect(new Set(ppm).size).toBe(400);
  const rows = lines.slice(1).map((l) => l.split(","));
  expect(rows.map((r) => r[1])).toEqual([...Array(6).fill("Control"), ...Array(6).fill("Treated")]);
  expect(rows.every((r) => r.length === 403 && r.slice(2).every((v) => Number.isFinite(Number(v))))).toBe(true);
});

it("is the same every time", () => {
  expect(makeExampleCsv()).toBe(makeExampleCsv());
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd GUI && npx vitest run renderer/exampleData.test.ts`
Expected: FAIL, cannot resolve `./exampleData`.

- [ ] **Step 3: Implement**

Create `GUI/renderer/exampleData.ts`:

```ts
// Seeded PRNG (mulberry32): the example file is identical every time it is saved.
function seeded(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Gaussian peaks at familiar metabolite shifts; "treated" raises two of them.
const PEAKS = [
  { ppm: 0.0, height: 2.0, width: 0.03 }, // reference (TSP / DSS)
  { ppm: 1.33, height: 1.4, width: 0.04, treated: 2.1 }, // lactate
  { ppm: 3.03, height: 1.0, width: 0.04 }, // creatine
  { ppm: 3.25, height: 0.8, width: 0.05 }, // choline
  { ppm: 4.05, height: 0.6, width: 0.04 },
  { ppm: 7.35, height: 0.3, width: 0.06, treated: 0.55 }, // aromatic
];

export function makeExampleCsv(samples = 12, points = 400): string {
  const random = seeded(42);
  const ppm = Array.from({ length: points }, (_, i) => 10 - (10.5 * i) / (points - 1));
  const header = ["Sample", "Group", "Age", ...ppm.map((p) => p.toFixed(4))];
  const rows = Array.from({ length: samples }, (_, s) => {
    const treated = s >= samples / 2;
    const values = ppm.map((p) => {
      let y = 0.002 + random() * 0.001;
      for (const peak of PEAKS) {
        const height = treated && peak.treated ? peak.treated : peak.height;
        y += height * Math.exp(-(((p - peak.ppm) / peak.width) ** 2) / 2) * (0.9 + random() * 0.2);
      }
      return y.toFixed(6);
    });
    const id = `s${String(s + 1).padStart(2, "0")}`;
    return [id, treated ? "Treated" : "Control", String(25 + Math.round(random() * 40)), ...values];
  });
  return [header, ...rows].map((r) => r.join(",")).join("\n") + "\n";
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd GUI && npx vitest run renderer/exampleData.test.ts`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add GUI/renderer/exampleData.ts GUI/renderer/exampleData.test.ts
git commit -m "feat(gui): generate a deterministic example CSV"
```

---

### Task 2: Format guide component

**Files:**
- Create: `GUI/renderer/components/FormatGuide.tsx`, `GUI/renderer/components/FormatGuide.test.tsx`
- Modify: `GUI/renderer/theme.css` (append)

**Interfaces:**
- Produces: `FormatGuide({ source: string; onSaveExample: () => void })`; renders `section.guide`. Source `"bruker"` shows the Bruker guide; anything else shows the CSV guide with a `Save example CSV` button. Experiment folder names in the Bruker tree and their metadata cells carry class `link`.

- [ ] **Step 1: Write the failing test**

Create `GUI/renderer/components/FormatGuide.test.tsx`:

```tsx
import { render } from "preact";
import { expect, it, vi } from "vitest";
import { FormatGuide } from "./FormatGuide";

function mount(source: string) {
  const root = document.createElement("div");
  const onSaveExample = vi.fn();
  render(<FormatGuide source={source} onSaveExample={onSaveExample} />, root);
  return { root, onSaveExample };
}

it("shows the processed table mock-up and checklist for CSV", () => {
  const { root, onSaveExample } = mount("csv");
  expect(root.textContent).toContain("Processed data table");
  const headers = [...root.querySelectorAll(".mock-table thead tr:last-child th")].map((th) => th.textContent);
  expect(headers.slice(0, 4)).toEqual(["Sample", "Group", "Age", "10.000"]);
  expect(root.querySelectorAll(".steps li").length).toBe(6);
  [...root.querySelectorAll("button")].find((b) => b.textContent === "Save example CSV")!.click();
  expect(onSaveExample).toHaveBeenCalledOnce();
});

it("shows the Bruker folder, metadata link and checklist for Bruker", () => {
  const { root } = mount("bruker");
  expect(root.textContent).toContain("Raw Bruker + metadata");
  expect(root.textContent).toContain("pdata/1/procs");
  const linked = [...root.querySelectorAll(".link")].map((e) => e.textContent);
  expect(linked.filter((t) => t?.startsWith("61020601"))).toHaveLength(2); // folder and metadata cell
  expect(root.querySelectorAll(".steps li").length).toBe(5);
  expect(root.textContent).not.toContain("Save example CSV");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd GUI && npx vitest run renderer/components/FormatGuide.test.tsx`
Expected: FAIL, cannot resolve `./FormatGuide`.

- [ ] **Step 3: Implement**

Create `GUI/renderer/components/FormatGuide.tsx`:

```tsx
import { Fragment, type ComponentChildren } from "preact";

interface Props {
  source: string;
  onSaveExample: () => void;
}

export function FormatGuide({ source, onSaveExample }: Props) {
  return source === "bruker" ? <BrukerGuide /> : <CsvGuide onSaveExample={onSaveExample} />;
}

function Steps({ items }: { items: ComponentChildren[] }) {
  return (
    <ol class="steps">
      {items.map((item, i) => (
        <li key={i}>{item}</li>
      ))}
    </ol>
  );
}

// Processed data table -------------------------------------------------------

const META = ["Sample", "Group", "Age"];
const SPEC = ["10.000", "9.997", "9.994", "…", "0.000"];
const ROWS = [
  ["s01", "Control", "34", "0.0123", "0.0119", "0.0121", "…", "0.0045"],
  ["s02", "Treated", "51", "0.0131", "0.0127", "0.0125", "…", "0.0051"],
  ["s03", "Control", "47", "0.0118", "0.0122", "0.0117", "…", "0.0043"],
];

const CSV_STEPS = [
  <>
    One row per sample, one column per chemical shift. Spectral headers are the ppm values as plain numbers (for
    example <code>9.997</code>); any header that reads as a number is treated as a spectral column.
  </>,
  <>Metadata columns (sample ID, group, other factors) have text headers and can sit anywhere, usually first.</>,
  <>
    Include a group column and enter its name in <b>Group column</b>. Numeric metadata such as age or dose can later
    be a PLS-R target.
  </>,
  <>Spectra must already be processed: phased, baseline corrected and calibrated (TSP / DSS at 0 ppm). Binned data is fine.</>,
  <>No missing values. Remove unwanted regions (for example water) by deleting those columns.</>,
  <>
    Save as a comma-separated UTF-8 CSV. From Python:{" "}
    <code>pd.concat([meta, spectra], axis=1).to_csv("data.csv", index=False)</code>. Export CSV on any data stage in
    this app writes the same format.
  </>,
];

function CsvGuide({ onSaveExample }: { onSaveExample: () => void }) {
  return (
    <section class="guide" aria-label="File format">
      <h3>Processed data table (CSV)</h3>
      <p class="lede">Samples in rows. Metadata columns, then spectral columns with ppm values as headers.</p>
      <div class="mock-scroll">
        <table class="mock-table">
          <thead>
            <tr class="bands">
              <th colSpan={META.length} class="meta">
                metadata
              </th>
              <th colSpan={SPEC.length} class="spec">
                spectral columns (ppm)
              </th>
            </tr>
            <tr>
              {META.map((h) => (
                <th key={h} class="meta">
                  {h}
                </th>
              ))}
              {SPEC.map((h) => (
                <th key={h} class="spec">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => (
              <tr key={row[0]}>
                {row.map((cell, i) => (
                  <td key={i} class={i < META.length ? "meta" : "spec"}>
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h4>How to prepare</h4>
      <Steps items={CSV_STEPS} />
      <div class="mock-actions">
        <button type="button" class="btn" onClick={onSaveExample}>
          Save example CSV
        </button>
        <span class="lede">12 samples, 2 groups, 400 ppm points. Saved and filled in for you.</span>
      </div>
    </section>
  );
}

// Raw Bruker + metadata --------------------------------------------------------

const TREE: { depth: number; name: string; exp?: string }[] = [
  { depth: 0, name: "study/" },
  { depth: 1, name: "61020601/", exp: "61020601" },
  { depth: 2, name: "fid" },
  { depth: 2, name: "acqus" },
  { depth: 2, name: "pdata/1/procs" },
  { depth: 1, name: "61030601/", exp: "61030601" },
  { depth: 2, name: "fid, acqus, pdata/…" },
];

const META_ROWS: Record<string, string[]> = { "61020601": ["sham", "6"], "61030601": ["nx", "6"] };

const BRUKER_STEPS = [
  <>
    Export each experiment from TopSpin as its own folder containing <code>fid</code>, <code>acqus</code> and{" "}
    <code>pdata/1/procs</code>. Process reads the stored phase (PHC0 / PHC1) and the spectrum offset from{" "}
    <code>procs</code>.
  </>,
  <>
    Put all experiments of one study under a single folder, either directly (<code>study/&lt;exp&gt;/fid</code>) or
    one level deeper (<code>study/&lt;batch&gt;/&lt;exp&gt;/fid</code>).
  </>,
  <>Acquire every experiment with the same sweep width and settings: spectra are stacked by position, so point counts must match.</>,
  <>Keep an internal reference (TSP or DSS) in each sample for automatic calibration to 0 ppm.</>,
  <>
    Make a metadata CSV with one row per experiment: a column holding the exact folder names (enter its name in{" "}
    <b>Sample column</b>, for example <code>fid</code>), a group column, and any other factors. Every experiment needs
    a row; extra rows are ignored.
  </>,
];

function BrukerGuide() {
  return (
    <section class="guide" aria-label="File format">
      <h3>Raw Bruker + metadata</h3>
      <p class="lede">A folder of experiments, plus a CSV that names each experiment and its group.</p>
      <div class="mock-files">
        <span class="caption">Bruker folder</span>
        <span />
        <span class="caption">metadata.csv</span>
        <span />
        <span />
        <span class="cells head">
          <span>fid</span>
          <span>Group</span>
          <span>Week</span>
        </span>
        {TREE.map((row) => (
          <Fragment key={row.name}>
            <span style={{ paddingLeft: `${row.depth * 16}px` }} class={row.exp ? "link" : undefined}>
              {row.name}
            </span>
            <span class="arrow">{row.exp ? "→" : ""}</span>
            {row.exp ? (
              <span class="cells">
                <span class="link">{row.exp}</span>
                {META_ROWS[row.exp].map((c, i) => (
                  <span key={i}>{c}</span>
                ))}
              </span>
            ) : (
              <span />
            )}
          </Fragment>
        ))}
      </div>
      <h4>How to prepare</h4>
      <Steps items={BRUKER_STEPS} />
    </section>
  );
}
```

Append to `GUI/renderer/theme.css`:

```css
.guide { max-width: 880px; margin: 0 auto; padding: var(--s3); display: flex; flex-direction: column; gap: var(--s2); }
.guide h3 { margin: 0; font-size: 15px; font-weight: 600; }
.guide h4 { margin: var(--s1) 0 0; font-size: 13px; font-weight: 600; }
.guide .lede { margin: 0; color: var(--muted); }
.guide code { font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--surface); padding: 1px 4px; border-radius: 4px; }
.mock-scroll { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; }
.mock-table { border-collapse: collapse; width: 100%; font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }
.mock-table th, .mock-table td { padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }
.mock-table .meta { background: var(--surface); }
.mock-table .spec { background: color-mix(in srgb, var(--accent) 8%, transparent); }
.mock-table .bands th { font: 500 11px var(--font); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
.steps { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 6px; }
.mock-files {
  display: grid;
  grid-template-columns: max-content 40px max-content;
  row-gap: 4px;
  padding: var(--s2);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow-x: auto;
  font: 12px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace;
}
.mock-files .caption { font: 500 11px var(--font); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
.mock-files .arrow { color: var(--accent); text-align: center; }
.mock-files .cells { display: grid; grid-template-columns: 88px 64px 48px; }
.mock-files .cells.head { color: var(--muted); border-bottom: 1px solid var(--border); }
.mock-files .link { color: var(--accent); font-weight: 600; }
.mock-actions { display: flex; align-items: center; gap: var(--s2); }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd GUI && npx vitest run renderer/components/FormatGuide.test.tsx && npm run typecheck`
Expected: `2 passed`; typecheck exits 0.

- [ ] **Step 5: Commit**

```bash
git add GUI/renderer/components/FormatGuide.tsx GUI/renderer/components/FormatGuide.test.tsx GUI/renderer/theme.css
git commit -m "feat(gui): add import format guide with file mock-ups and preparation steps"
```

---

### Task 3: Wire the guide into the Import stage

**Files:**
- Modify: `GUI/renderer/components/ParamPanel.tsx`, `GUI/renderer/components/ParamPanel.test.tsx`, `GUI/renderer/App.tsx`, `GUI/electron/main.ts`

**Interfaces:**
- Consumes: `FormatGuide`, `makeExampleCsv`, `window.metbit.saveImage(defaultName, dataUrl)`.
- Produces:
  - `ParamPanel` optional prop `onValuesChange?: (values: Values) => void`, called with the current values after every change (and once on mount).
  - Import view: guide shown when Import has no result and is not running, or when toggled with the `File format` / `Hide format` tab-bar link.
  - `Save example CSV` writes `metbit-example.csv`, then fills the Import form with `source: "csv"` and that path.

- [ ] **Step 1: Write the failing test**

Append to `GUI/renderer/components/ParamPanel.test.tsx`:

```tsx
it("reports live values to onValuesChange", async () => {
  const seen: unknown[] = [];
  const { root } = mount({ schema: importSchema, onValuesChange: (v) => seen.push(v.source) });
  await tick();
  const source = root.querySelector<HTMLSelectElement>("#f-source")!;
  source.value = "bruker";
  source.dispatchEvent(new Event("change"));
  await tick();
  expect(seen[0]).toBe("csv");
  expect(seen[seen.length - 1]).toBe("bruker");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd GUI && npx vitest run renderer/components/ParamPanel.test.tsx`
Expected: 1 failed (`seen` stays empty).

- [ ] **Step 3: ParamPanel callback**

In `GUI/renderer/components/ParamPanel.tsx`:

Change `import { useState } from "preact/hooks";` to `import { useEffect, useState } from "preact/hooks";`.

In `interface Props`, after `picked?: number | null;` add:

```tsx
  onValuesChange?: (values: Values) => void;
```

Change the signature line to:

```tsx
export function ParamPanel({ title, schema, initial, errors, busy, onRun, picked = null, onValuesChange }: Props) {
```

and after the `const set = ...` line add:

```tsx
  useEffect(() => onValuesChange?.(values), [values]);
```

Run: `cd GUI && npx vitest run renderer/components/ParamPanel.test.tsx`
Expected: all pass.

- [ ] **Step 4: CSV filter name for data-URL saves**

In `GUI/electron/main.ts`, replace:

```ts
  const path = await chooseSavePath(defaultName, [{ name: "Image", extensions: [extname(defaultName).slice(1)] }]);
```

with:

```ts
  // saveImage writes any data URL: images from plot export, the example CSV from the format guide
  const ext = extname(defaultName).slice(1);
  const path = await chooseSavePath(defaultName, [{ name: ext === "csv" ? "CSV" : "Image", extensions: [ext] }]);
```

- [ ] **Step 5: App wiring**

In `GUI/renderer/App.tsx`:

Add imports after the `DataTable` import:

```tsx
import { FormatGuide } from "./components/FormatGuide";
import { makeExampleCsv } from "./exampleData";
```

After `const [notice, setNotice] = useState<string | null>(null);` add:

```tsx
  const [importValues, setImportValues] = useState<Values>({});
  const [showGuide, setShowGuide] = useState(false);
```

In `select(id)`, add `setShowGuide(false);` as its first line.

After the `saveProject` definition add:

```tsx
  const saveExample = () =>
    guarded(async () => {
      const url = `data:text/csv;charset=utf-8,${encodeURIComponent(makeExampleCsv())}`;
      const path = await window.metbit.saveImage("metbit-example.csv", url);
      if (!path) return;
      // prefill the Import form so the next step is just Run
      setStages((ss) => ss.map((s) => (s.id === "import" ? { ...s, params: { ...(s.params ?? {}), source: "csv", path } } : s)));
      setFormKey((k) => k + 1);
      flash(`Saved ${basename(path)}`);
    });
  const guideVisible = current === "import" && !failure && (showGuide || (!hasResult && stage?.status !== "running"));
```

Replace the first branch of the body chain

```tsx
  let body;
  if (failure) {
```

with:

```tsx
  let body;
  if (guideVisible) {
    body = <FormatGuide source={String(importValues.source ?? "csv")} onSaveExample={saveExample} />;
  } else if (failure) {
```

Inside `<div class="tabs-right">`, before the `Export CSV` button, add:

```tsx
            {current === "import" && hasResult && (
              <button class="link" onClick={() => setShowGuide((v) => !v)}>
                {showGuide ? "Hide format" : "File format"}
              </button>
            )}
```

In the tab buttons `onClick={() => setTab(t)}`, change to `onClick={() => { setShowGuide(false); setTab(t); }}`.

On `<ParamPanel`, add the prop:

```tsx
        onValuesChange={current === "import" ? setImportValues : undefined}
```

- [ ] **Step 6: Verify**

Run: `cd GUI && npm run typecheck && npm test && npm run build`
Expected: typecheck exits 0; `26 passed`; build succeeds.

- [ ] **Step 7: Commit**

```bash
git add GUI/renderer GUI/electron/main.ts
git commit -m "feat(gui): show the format guide in Import and save a ready-to-run example CSV"
```

---

### Task 4: End-to-end

**Files:**
- Create: `GUI/e2e/guide.spec.ts`

**Interfaces:**
- Consumes: `METBIT_E2E_DIR`, `METBIT_USER_DATA` seams; `.guide`, `#f-source`, `#f-path`, buttons `Save example CSV`, `Run`, `File format`.
- Produces: `npm run e2e` runs 7 tests.

- [ ] **Step 1: Write the test**

Create `GUI/e2e/guide.spec.ts`:

```ts
import { _electron as electron, expect, test } from "@playwright/test";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

test("format guide follows Source and its example CSV imports", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const env = { ...process.env, METBIT_E2E_DIR: dir, METBIT_USER_DATA: join(dir, "user") } as Record<string, string>;
  const app = await electron.launch({ args: ["."], env });
  const win = await app.firstWindow();
  const guide = win.locator(".guide");

  await expect(guide).toContainText("Processed data table");
  await win.selectOption("#f-source", "bruker");
  await expect(guide).toContainText("pdata/1/procs");
  await win.selectOption("#f-source", "csv");
  await expect(guide).toContainText("Processed data table");

  await win.getByRole("button", { name: "Save example CSV" }).click();
  const example = join(dir, "metbit-example.csv");
  await expect.poll(() => existsSync(example)).toBe(true);
  await expect(win.locator("#f-path")).toHaveValue(example);

  await win.getByRole("button", { name: "Run", exact: true }).click();
  await expect(win.locator("table.data")).toContainText("s12");
  await expect(guide).toHaveCount(0);
  await win.getByRole("button", { name: "File format" }).click();
  await expect(guide).toBeVisible();

  await app.close();
});
```

- [ ] **Step 2: Run everything**

Run: `cd GUI && npm run typecheck && npm test && npm run e2e`
Expected: typecheck exits 0; `26 passed`; `7 passed` (6 plus 1 skipped without MTBLS2052).

- [ ] **Step 3: Commit**

```bash
git add GUI/e2e/guide.spec.ts
git commit -m "test(gui): cover the import format guide and example CSV end to end"
```
